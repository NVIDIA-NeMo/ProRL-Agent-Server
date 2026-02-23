"""NVCF Function Pool: manages a warm pool of pre-deployed NVCF functions for parallel data collection."""

import logging
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

    Deploys N functions at startup and provides acquire/release semantics
    so workers can check out a warm VM, use it for a trajectory, and return it.
    """

    def __init__(
        self,
        pool_size: int,
        nvcf_api_key: Optional[str] = None,
        nvcf_org: Optional[str] = None,
    ):
        self.pool_size = pool_size
        self._deployer = OSWorldDeployer(api_key=nvcf_api_key, org_name=nvcf_org)
        self._nvcf_api_key = nvcf_api_key
        self._nvcf_org = nvcf_org

        # Each entry is (function_id, version_id)
        self._all_functions: List[Tuple[str, str]] = []
        self._available: queue.Queue[Tuple[str, str]] = queue.Queue()
        self._lock = threading.Lock()

    def _deploy_one(self, index: int) -> Tuple[str, str]:
        """Deploy a single NVCF function and wait for it to become ACTIVE."""
        func_config = OSWorldFunctionConfig(
            name=f"nvcf-pool-{index}",
            description=f"Warm pool function {index}",
        )
        deploy_config = OSWorldDeploymentConfig(
            gpu="L40S",
            min_instances=1,
            max_instances=1,
        )

        logger.info(f"[pool-{index}] Creating function...")
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
        logger.info(f"[pool-{index}] ACTIVE: {function_id}")
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
        """Deploy pool_size NVCF functions in parallel and wait for all to become ACTIVE."""
        logger.info(f"Deploying {self.pool_size} NVCF functions...")

        # Deploy in parallel (bounded by max_workers to avoid API throttling)
        workers = min(max_workers, self.pool_size)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._deploy_one, i) for i in range(self.pool_size)]
            for future in futures:
                fn_id, ver_id = future.result()  # raises if deploy failed
                self._all_functions.append((fn_id, ver_id))
                self._available.put((fn_id, ver_id))

        logger.info(f"All {self.pool_size} functions deployed and ready.")

    def deploy_all_from_ids(self, function_ids: List[Tuple[str, str]]) -> None:
        """Use pre-existing function IDs instead of deploying new ones."""
        for fn_id, ver_id in function_ids:
            self._all_functions.append((fn_id, ver_id))
            self._available.put((fn_id, ver_id))
        self.pool_size = len(function_ids)
        logger.info(f"Pool initialized with {self.pool_size} pre-existing functions.")

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

    def release_or_replace(self, function_id: str, version_id: str) -> None:
        """Release a function back to pool, replacing it if unhealthy."""
        if self.health_check(function_id):
            self._available.put((function_id, version_id))
            return

        logger.warning(f"Function {function_id} is unhealthy, deploying replacement...")
        # Undeploy broken function in background
        threading.Thread(
            target=self._undeploy_one, args=(function_id, version_id), daemon=True
        ).start()
        # Deploy replacement
        try:
            new_fn_id, new_ver_id = self._deploy_one(len(self._all_functions))
            with self._lock:
                self._all_functions.append((new_fn_id, new_ver_id))
            self._available.put((new_fn_id, new_ver_id))
            logger.info(f"Replacement function {new_fn_id} deployed and added to pool.")
        except Exception as e:
            logger.error(f"Failed to deploy replacement: {e}. Pool size reduced.")

    def undeploy_all(self) -> None:
        """Undeploy and delete all functions in the pool."""
        logger.info(f"Undeploying {len(self._all_functions)} NVCF functions...")
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
