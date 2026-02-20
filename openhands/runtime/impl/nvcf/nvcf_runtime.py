"""Base NVCF runtime: extends ActionExecutionClient with deploy-on-connect and undeploy-on-close.

This runtime does not run a local action execution server; connect() deploys an NVCF
function (or attaches to an existing one), and close() undeploys if we deployed in
this session. Subclasses (e.g. OSWorldNVCFRuntime) implement run_action by calling
the NVCF API.
"""

import os
from typing import Any

from openhands.core.config import OpenHandsConfig
from openhands.core.exceptions import AgentRuntimeDisconnectedError
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.events.observation import ErrorObservation, Observation
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE
from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)
from openhands.runtime.plugins import PluginRequirement


class NVCFRuntime(ActionExecutionClient):
    """Base runtime for NVIDIA Cloud Functions (NVCF).

    - connect(): Deploys an NVCF function if nvcf_function_id is not set (uses
      openhands.nvidia.os_world.nvcf), then marks runtime as initialized.
    - close(): Undeploys the function if we deployed it in this session and
      undeploy_on_close is True, then closes the session.
    - check_if_alive(): Raises if runtime is not connected (no action server to ping).
    - run_action(): Returns ErrorObservation (no local action server); subclasses
      override to dispatch to NVCF API (e.g. OSWorld).
    """

    def __init__(
        self,
        config: OpenHandsConfig,
        event_stream: EventStream,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Any | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
        user_id: str | None = None,
        git_provider_tokens: PROVIDER_TOKEN_TYPE | None = None,
        nvcf_function_id: str | None = None,
        nvcf_version_id: str | None = None,
        nvcf_api_key: str | None = None,
        nvcf_org: str | None = None,
        nvcf_function_config: Any = None,
        nvcf_deployment_config: Any = None,
        undeploy_on_close: bool = True,
    ):
        self._nvcf_api_key = nvcf_api_key or os.environ.get("NGC_API_KEY")
        self._nvcf_org = nvcf_org or os.environ.get("NGC_ORG")
        _fid = nvcf_function_id or os.environ.get("NVCF_FUNCTION_ID")
        self._nvcf_function_id = (
            _fid.strip() if isinstance(_fid, str) and _fid else (_fid or None)
        )
        _vid = nvcf_version_id or os.environ.get("NVCF_VERSION_ID")
        self._nvcf_version_id = (
            _vid.strip() if isinstance(_vid, str) and _vid else (_vid or None)
        )
        self._nvcf_function_config = nvcf_function_config
        self._nvcf_deployment_config = nvcf_deployment_config
        self._undeploy_on_close = undeploy_on_close
        self._we_deployed_this = False
        self._deployer = None

        if not self._nvcf_api_key:
            raise ValueError(
                "NGC API key required for NVCF. Provide nvcf_api_key or set NGC_API_KEY."
            )
        if not self._nvcf_function_id and not self._nvcf_org:
            raise ValueError(
                "When deploying on connect, NGC org required. "
                "Provide nvcf_org or set NGC_ORG."
            )

        super().__init__(
            config=config,
            event_stream=event_stream,
            sid=sid,
            plugins=plugins,
            env_vars=env_vars,
            status_callback=status_callback,
            attach_to_existing=attach_to_existing,
            headless_mode=headless_mode,
            user_id=user_id,
            git_provider_tokens=git_provider_tokens,
        )

    def _deploy_nvcf(self) -> tuple[str, str]:
        """Create and deploy an NVCF function; return (function_id, version_id)."""
        from openhands.nvidia.os_world.nvcf import (
            OSWorldDeployer,
            OSWorldFunctionConfig,
            OSWorldDeploymentConfig,
        )
        func_config = self._nvcf_function_config
        if func_config is None:
            func_config = OSWorldFunctionConfig(
                name=f"nvcf-runtime-{self.sid}",
                description="NVCF runtime deploy-on-connect",
            )
        deploy_config = self._nvcf_deployment_config
        if deploy_config is None:
            deploy_config = OSWorldDeploymentConfig(
                gpu="L40S",
                min_instances=1,
                max_instances=1,
            )
        self.log("info", "Deploying NVCF function...")
        self.send_status_message("STATUS$PREPARING_CONTAINER")
        self._deployer = OSWorldDeployer(
            api_key=self._nvcf_api_key,
            org_name=self._nvcf_org,
        )
        result = self._deployer.create_function(func_config)
        function = result.get("function", {})
        function_id = function.get("id")
        version_id = function.get("versionId")
        if not function_id or not version_id:
            raise RuntimeError(f"NVCF create_function did not return ids: {result}")
        self.log("info", f"Created function {function_id}; deploying...")
        self._deployer.deploy(function_id, version_id, deploy_config)
        self.log("info", "Waiting for NVCF function to become ACTIVE...")
        self._deployer.wait_for_active(
            function_id, version_id, timeout=1800, poll_interval=30
        )
        self.log("info", "NVCF function is ACTIVE")
        return function_id, version_id

    async def connect(self) -> None:
        """Deploy NVCF function if needed, then mark runtime as connected."""
        self.send_status_message("STATUS$STARTING_RUNTIME")
        from openhands.utils.async_utils import call_sync_from_async
        if self._nvcf_function_id is None:
            self._nvcf_function_id, self._nvcf_version_id = await call_sync_from_async(
                self._deploy_nvcf
            )
            self._we_deployed_this = True
        self._runtime_initialized = True
        self.log("info", f"NVCF runtime connected: {self._nvcf_function_id}")

    def check_if_alive(self) -> None:
        """NVCF runtime has no local action server; require subclass to implement."""
        if not getattr(self, "_runtime_initialized", False):
            raise AgentRuntimeDisconnectedError("NVCF runtime is not connected.")

    def run_action(self, action) -> Observation:
        """Base NVCF runtime does not execute actions; subclasses override for NVCF API."""
        return ErrorObservation(
            "This runtime does not support the requested action. "
            "Use OSWorldNVCFRuntime for OSWorld actions."
        )

    def close(self, rm_all_containers: bool | None = None) -> None:
        """Undeploy NVCF function if we deployed in this session, then close session."""
        if self._we_deployed_this and self._undeploy_on_close and self._deployer:
            if self._nvcf_function_id and self._nvcf_version_id:
                try:
                    self.log("info", "Undeploying NVCF function...")
                    self._deployer.undeploy(
                        self._nvcf_function_id,
                        self._nvcf_version_id,
                        graceful=True,
                    )
                    self.log("info", "NVCF function undeployed")
                except Exception as e:
                    logger.warning(f"Failed to undeploy NVCF function: {e}")
            self._deployer = None
        super().close()
