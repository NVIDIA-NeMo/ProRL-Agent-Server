"""OSWorld deployer using the NGC SDK."""

import os
import time
from typing import Any, Dict, Iterator, List, Optional

from openhands.nvidia.os_world.nvcf.config import (
    OSWorldDeploymentConfig,
    OSWorldFunctionConfig,
)

# NGC SDK imports (optional dependency: pip install ngcsdk)
try:
    from ngcsdk import Client
    from nvcf.api.deployment_spec import TargetedDeploymentSpecification
except ImportError as e:
    Client = None  # type: ignore[misc, assignment]
    TargetedDeploymentSpecification = None  # type: ignore[misc, assignment]
    _NGCSDK_IMPORT_ERROR = e
else:
    _NGCSDK_IMPORT_ERROR = None


class OSWorldDeployer:
    """Manages OSWorld deployment to NVCF using the NGC SDK.

    This class provides a simplified interface for deploying OSWorld
    containers to NVIDIA Cloud Functions.

    Example:
        >>> deployer = OSWorldDeployer(
        ...     api_key="nvapi-xxx",
        ...     org_name="my-org",
        ... )
        >>> result = deployer.create_function()
        >>> func_id = result["function"]["id"]
        >>> ver_id = result["function"]["versionId"]
        >>> deployer.deploy(func_id, ver_id)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        org_name: Optional[str] = None,
        team_name: str = "no-team",
    ):
        """Initialize the OSWorld deployer.

        Args:
            api_key: NGC API key. If not provided, reads from NGC_API_KEY
                environment variable.
            org_name: NGC organization name. If not provided, reads from
                NGC_ORG environment variable.
            team_name: NGC team name. Defaults to "no-team".

        Raises:
            ValueError: If api_key or org_name is not provided and not
                found in environment variables.
            ImportError: If ngcsdk is not installed.
        """
        if _NGCSDK_IMPORT_ERROR is not None:
            raise ImportError(
                "ngcsdk is required for OSWorldDeployer. Install with: pip install ngcsdk"
            ) from _NGCSDK_IMPORT_ERROR

        api_key = api_key or os.environ.get("NGC_API_KEY")
        org_name = org_name or os.environ.get("NGC_ORG")

        if not api_key:
            raise ValueError(
                "NGC API key required. Provide api_key or set NGC_API_KEY env var."
            )
        if not org_name:
            raise ValueError(
                "NGC org name required. Provide org_name or set NGC_ORG env var."
            )

        self._api_key = api_key
        self._org_name = org_name
        self._client = Client()
        self._client.configure(api_key, org_name=org_name, team_name=team_name)

    @property
    def client(self) -> "Client":
        """Access the underlying NGC SDK client."""
        return self._client

    def create_function(
        self,
        config: Optional[OSWorldFunctionConfig] = None,
        function_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an OSWorld function in NVCF."""
        if config is None:
            config = OSWorldFunctionConfig()

        kwargs: Dict[str, Any] = {
            "name": config.name,
            "inference_url": config.inference_url,
            "container_image": config.container_image,
            "inference_port": config.inference_port,
            "health_uri": config.health_uri,
            "description": config.description,
        }

        if config.container_args:
            kwargs["container_args"] = config.container_args
        if config.container_environment_variables:
            kwargs["container_environment_variables"] = (
                config.container_environment_variables
            )
        if config.tags:
            kwargs["tags"] = config.tags
        if function_id:
            kwargs["function_id"] = function_id

        return self._client.cloud_function.functions.create(**kwargs)

    def deploy(
        self,
        function_id: str,
        function_version_id: str,
        config: Optional[OSWorldDeploymentConfig] = None,
    ) -> Dict[str, Any]:
        """Deploy an OSWorld function with specified GPU configuration."""
        if config is None:
            config = OSWorldDeploymentConfig()

        spec = TargetedDeploymentSpecification(
            gpu=config.gpu,
            instance_type=config.get_instance_type(),
            min_instances=config.min_instances,
            max_instances=config.max_instances,
            max_request_concurrency=config.max_request_concurrency,
            regions=config.regions,
            clusters=config.clusters,
            attributes=config.attributes,
            configuration=config.configuration,
        )

        return self._client.cloud_function.functions.deployments.create(
            function_id=function_id,
            function_version_id=function_version_id,
            targeted_deployment_specifications=[spec],
        )

    def update_deployment(
        self,
        function_id: str,
        function_version_id: str,
        config: OSWorldDeploymentConfig,
    ) -> Dict[str, Any]:
        """Update an existing deployment configuration."""
        spec = TargetedDeploymentSpecification(
            gpu=config.gpu,
            instance_type=config.get_instance_type(),
            min_instances=config.min_instances,
            max_instances=config.max_instances,
            max_request_concurrency=config.max_request_concurrency,
            regions=config.regions,
            clusters=config.clusters,
            attributes=config.attributes,
            configuration=config.configuration,
        )

        return self._client.cloud_function.functions.deployments.update(
            function_id=function_id,
            function_version_id=function_version_id,
            targeted_deployment_specifications=[spec],
        )

    def get_function_info(
        self,
        function_id: str,
        function_version_id: str,
    ) -> Dict[str, Any]:
        """Get information about a function version."""
        return self._client.cloud_function.functions.info(
            function_id=function_id,
            function_version_id=function_version_id,
        )

    def get_deployment_info(
        self,
        function_id: str,
        function_version_id: str,
    ) -> Dict[str, Any]:
        """Get information about a function's deployment."""
        return self._client.cloud_function.functions.deployments.info(
            function_id=function_id,
            function_version_id=function_version_id,
        )

    def get_status(
        self,
        function_id: str,
        function_version_id: str,
    ) -> str:
        """Get the current status of a function."""
        info = self.get_function_info(function_id, function_version_id)
        return info.get("function", {}).get("status", "UNKNOWN")

    def wait_for_active(
        self,
        function_id: str,
        function_version_id: str,
        timeout: int = 1800,
        poll_interval: int = 30,
    ) -> str:
        """Wait for a function to become ACTIVE."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status(function_id, function_version_id)
            if status == "ACTIVE":
                return status
            if status == "ERROR":
                raise RuntimeError("Function entered ERROR state")
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Function did not become ACTIVE within {timeout} seconds. "
            f"Last status: {status}"
        )

    def list_functions(
        self,
        name_pattern: Optional[str] = None,
        access_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """List functions available to the organization."""
        return self._client.cloud_function.functions.list(
            name_pattern=name_pattern,
            access_filter=access_filter or ["private"],
        )

    def list_available_gpus(self) -> Dict[str, Any]:
        """List available GPU types for deployment."""
        return self._client.cloud_function.gpus.list()

    def get_gpu_info(self, gpu_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific GPU type."""
        return self._client.cloud_function.gpus.info(gpu_name)

    def invoke(
        self,
        function_id: str,
        payload: Dict[str, Any],
        api_key: Optional[str] = None,
        function_version_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Invoke an OSWorld function."""
        return self._client.cloud_function.functions.invoke(
            function_id=function_id,
            payload=payload,
            starfleet_api_key=api_key or self._api_key,
            function_version_id=function_version_id,
        )

    def invoke_stream(
        self,
        function_id: str,
        payload: Dict[str, Any],
        api_key: Optional[str] = None,
        function_version_id: Optional[str] = None,
    ) -> Iterator[bytes]:
        """Invoke an OSWorld function with streaming response."""
        return self._client.cloud_function.functions.invoke_stream(
            function_id=function_id,
            payload=payload,
            starfleet_api_key=api_key or self._api_key,
            function_version_id=function_version_id,
        )

    def undeploy(
        self,
        function_id: str,
        function_version_id: str,
        graceful: bool = True,
    ) -> None:
        """Remove a deployment (undeploy a function)."""
        self._client.cloud_function.functions.deployments.delete(
            function_id=function_id,
            function_version_id=function_version_id,
            graceful=graceful,
        )

    def delete_function(
        self,
        function_id: str,
        function_version_id: str,
    ) -> None:
        """Delete a function version. Must be undeployed first."""
        self._client.cloud_function.functions.delete(
            function_id=function_id,
            function_version_id=function_version_id,
        )

    def query_deployment_logs(
        self,
        function_id: str,
        function_version_id: str,
        duration: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Query deployment logs for a function."""
        from datetime import timedelta

        td = None
        if duration:
            unit = duration[-1].upper()
            value = int(duration[:-1])
            if unit == "H":
                td = timedelta(hours=value)
            elif unit == "M":
                td = timedelta(minutes=value)
            elif unit == "D":
                td = timedelta(days=value)
            elif unit == "S":
                td = timedelta(seconds=value)

        return self._client.cloud_function.functions.deployments.query_logs(
            function_id=function_id,
            function_version_id=function_version_id,
            duration=td,
        )
