"""OSWorld NVCF: deploy and interact with OSWorld on NVIDIA Cloud Functions (NVCF)."""

from openhands.nvidia.os_world.nvcf.config import (
    DEFAULT_CONTAINER_IMAGE,
    OSWorldDeploymentConfig,
    OSWorldFunctionConfig,
)
from openhands.nvidia.os_world.nvcf.deployer import OSWorldDeployer

__all__ = [
    "OSWorldDeployer",
    "OSWorldFunctionConfig",
    "OSWorldDeploymentConfig",
    "DEFAULT_CONTAINER_IMAGE",
]
