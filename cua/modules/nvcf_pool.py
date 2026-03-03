"""NVCF Function Pool: manages a warm pool of pre-deployed NVCF functions for parallel data collection."""

import logging
import math
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

from openhands.core.logger import openhands_logger
from openhands.nvidia.os_world.nvcf import (
    OSWorldDeployer,
    OSWorldDeploymentConfig,
    OSWorldFunctionConfig,
)

logger = openhands_logger.getChild('nvcf_pool')
logger.setLevel(logging.INFO)


class NVCFPool:
    """Thread-safe pool of pre-deployed NVCF functions.

    Deploys NVCF functions at startup and provides acquire/release semantics
    so workers can check out a warm VM, use it for a trajectory, and return it.

    When num_vms_per_instance > 1, fewer functions are deployed, each with
    multiple VM instances on the same machine, reducing resource overhead.
    """

    def __init__(
        self,
        pool_size: int,
        num_vms_per_instance: int = 1,
        nvcf_api_key: Optional[str] = None,
        nvcf_org: Optional[str] = None,
    ):
        self.pool_size = pool_size
        self.num_vms_per_instance = num_vms_per_instance
        self._deployer = OSWorldDeployer(api_key=nvcf_api_key, org_name=nvcf_org)
        self._nvcf_api_key = nvcf_api_key
        self._nvcf_org = nvcf_org

        # Number of NVCF functions to deploy
        self._num_functions = math.ceil(pool_size / num_vms_per_instance)

        # Each entry is (function_id, version_id)
        self._all_functions: List[Tuple[str, str]] = []
        self._available: queue.Queue[Tuple[str, str]] = queue.Queue()
        self._lock = threading.Lock()

    def _deploy_one(self, index: int) -> Tuple[str, str]:
        """Deploy a single NVCF function and wait for it to become ACTIVE."""
        func_config = OSWorldFunctionConfig(
            name=f"nvcf-pool-{index}",
            description=f"Warm pool function {index} ({self.num_vms_per_instance} VMs)",
        )
        deploy_config = OSWorldDeploymentConfig(
            gpu="L40S",
            min_instances=self.num_vms_per_instance,
            max_instances=self.num_vms_per_instance,
        )

        logger.info(f"[pool-{index}] Creating function ({self.num_vms_per_instance} VMs)...")
        result = self._deployer.create_function(func_config)
        function = result.get("function", {})
        function_id = function.get("id")
        version_id = function.get("versionId")
        if not function_id or not version_id:
            raise RuntimeError(f"[pool-{index}] create_function failed: {result}")

        logger.info(f"[pool-{index}] Deploying {function_id}...")
        self._deployer.deploy(function_id, version_id, deploy_config)

        logger.info(f"[pool-{index}] Waiting for ACTIVE...")
        self._deployer.wait_for_active(
            function_id, version_id, timeout=1800, poll_interval=30
        )
        logger.info(f"[pool-{index}] ACTIVE: {function_id} with {self.num_vms_per_instance} instances")
        return function_id, version_id

    def _undeploy_one(self, function_id: str, version_id: str) -> None:
        """Undeploy and delete a single NVCF function."""
        try:
            logger.info(f"Undeploying {function_id}...")
            self._deployer.undeploy(function_id, version_id, graceful=True)
        except Exception as e:
            logger.warning(f"Failed to undeploy {function_id}: {e}")
        try:
            self._deployer.delete_function(function_id, version_id)
        except Exception as e:
            logger.warning(f"Failed to delete function {function_id}: {e}")

    def deploy_all(self, max_workers: int = 8) -> None:
        """Deploy NVCF functions in parallel and wait for all to become ACTIVE.

        With num_vms_per_instance > 1, deploys fewer functions (each with
        multiple instances) to reach the desired pool_size.
        """
        logger.info(
            f"Deploying {self._num_functions} NVCF function(s) "
            f"x {self.num_vms_per_instance} VMs each = {self.pool_size} total slots..."
        )

        workers = min(max_workers, self._num_functions)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._deploy_one, i) for i in range(self._num_functions)]
            for future in futures:
                fn_id, ver_id = future.result()  # raises if deploy failed
                self._all_functions.append((fn_id, ver_id))
                # Add one entry per VM instance so acquire/release works correctly
                for _ in range(self.num_vms_per_instance):
                    self._available.put((fn_id, ver_id))

        logger.info(
            f"All {self._num_functions} function(s) deployed. "
            f"{self._available.qsize()} VM slots ready."
        )

    def deploy_all_from_ids(self, function_ids: List[Tuple[str, str]], vms_per_function: int = 1) -> None:
        """Use pre-existing function IDs instead of deploying new ones."""
        for fn_id, ver_id in function_ids:
            self._all_functions.append((fn_id, ver_id))
            for _ in range(vms_per_function):
                self._available.put((fn_id, ver_id))
        self.pool_size = len(function_ids) * vms_per_function
        self.num_vms_per_instance = vms_per_function
        logger.info(f"Pool initialized with {len(function_ids)} pre-existing function(s), {self.pool_size} total slots.")

    def acquire(self, timeout: Optional[float] = None) -> Tuple[str, str]:
        """Acquire a function from the pool. Blocks until one is available.

        Returns:
            (function_id, version_id) tuple
        """
        try:
            return self._available.get(block=True, timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"No NVCF function available within {timeout}s")

    def release(self, function_id: str, version_id: str) -> None:
        """Return a function to the pool for reuse."""
        self._available.put((function_id, version_id))

    def health_check(self, function_id: str) -> bool:
        """Check if an NVCF function is still healthy by pinging /platform."""
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self._nvcf_api_key}",
                "Function-ID": function_id,
            }
            r = requests.get(
                "https://grpc.nvcf.nvidia.com/api/platform",
                headers=headers,
                timeout=10.0,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _is_function_gone(self, function_id: str, version_id: str) -> bool:
        """Check if a function has been completely deleted/evicted (404)."""
        try:
            self._deployer.get_function_info(function_id, version_id)
            return False
        except Exception as e:
            if '404' in str(e) or 'Not found' in str(e):
                return True
            return False

    def release_or_replace(self, function_id: str, version_id: str) -> None:
        """Release a function back to pool, replacing it if unhealthy.

        For multi-instance functions, individual instance failures are handled
        by NVCF internally (it maintains min_instances). We only deploy a
        full replacement if the entire function is gone (404).
        """
        if self.health_check(function_id):
            self._available.put((function_id, version_id))
            return

        # For multi-instance functions: check if the function itself is gone
        # vs just a transient instance failure that NVCF will self-heal.
        if self.num_vms_per_instance > 1 and not self._is_function_gone(function_id, version_id):
            logger.warning(
                f"Function {function_id} health check failed but function still exists. "
                f"NVCF should self-heal the instance. Releasing slot back to pool."
            )
            self._available.put((function_id, version_id))
            return

        logger.warning(f"Function {function_id} is gone (404), deploying replacement...")
        # Undeploy broken function in background (best-effort cleanup)
        threading.Thread(
            target=self._undeploy_one, args=(function_id, version_id), daemon=True
        ).start()
        # Deploy replacement
        try:
            new_fn_id, new_ver_id = self._deploy_one(len(self._all_functions))
            with self._lock:
                self._all_functions.append((new_fn_id, new_ver_id))
            # Add slots for all VMs on the replacement function
            for _ in range(self.num_vms_per_instance):
                self._available.put((new_fn_id, new_ver_id))
            logger.info(f"Replacement function {new_fn_id} deployed with {self.num_vms_per_instance} VM slots.")
        except Exception as e:
            logger.error(f"Failed to deploy replacement: {e}. Pool size reduced.")

    def undeploy_all(self) -> None:
        """Undeploy and delete all functions in the pool."""
        logger.info(f"Undeploying {len(self._all_functions)} NVCF function(s)...")
        for fn_id, ver_id in self._all_functions:
            self._undeploy_one(fn_id, ver_id)
        self._all_functions.clear()
        # Drain the queue
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except queue.Empty:
                break
        logger.info("All NVCF functions undeployed.")

    @property
    def nvcf_api_key(self) -> Optional[str]:
        return self._nvcf_api_key

    @property
    def nvcf_org(self) -> Optional[str]:
        return self._nvcf_org
