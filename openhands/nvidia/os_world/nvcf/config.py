"""Configuration dataclasses for OSWorld NVCF deployment."""

import os
import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from typing import List, Optional

# NGC_ORG = os.environ.get("NGC_ORG", "nvidian")
# DEFAULT_CONTAINER_IMAGE = f"nvcr.io/{NGC_ORG}/nemo:osworld-linux-2"

DEFAULT_CONTAINER_IMAGE = "nvcr.io/i01fc6pe8nwm/nemo:osworld-linux-2-debug"


@dataclass
class OSWorldFunctionConfig:
    """Configuration for creating an OSWorld NVCF function.

    Attributes:
        name: Display name of the function in NVCF.
        container_image: NGC container image path. Defaults to DEFAULT_CONTAINER_IMAGE.
        inference_url: Endpoint path for inference requests.
        inference_port: Port the container exposes for inference.
        health_uri: Endpoint path for health checks.
        description: Optional description of the function.
        container_args: Optional arguments to pass to the container.
        container_environment_variables: Optional environment variables
            in the format ["KEY1:value1", "KEY2:value2"].
        tags: Optional list of tags for the function.
    """

    name: str = "osworld-linux"
    container_image: Optional[str] = None  # Set via __post_init__
    inference_url: str = "/api"
    inference_port: int = 8000
    health_uri: str = "/api/version"
    description: str = "OSWorld Linux environment for AI agent evaluation"
    container_args: Optional[str] = None
    container_environment_variables: Optional[List[str]] = None
    tags: Optional[List[str]] = field(default_factory=lambda: ["osworld"])

    def __post_init__(self):
        """Set default container image if not provided."""
        if self.container_image is None:
            self.container_image = DEFAULT_CONTAINER_IMAGE
        logger.info(f"Using container image: {self.container_image}")


@dataclass
class OSWorldDeploymentConfig:
    """Configuration for deploying an OSWorld function.

    Attributes:
        gpu: GPU type to use (e.g., "L40", "H100", "A100").
        instance_type: Specific instance type (e.g., "GFN.GPU.L40_1x").
            If not provided, will use a default based on GPU type.
        min_instances: Minimum number of instances (0 allows scale-to-zero).
        max_instances: Maximum number of instances for autoscaling.
        max_request_concurrency: Maximum concurrent requests per instance.
        backend: Cluster backend (e.g., "GFN", "AZURE", "GCP", "OCI").
        regions: Optional list of regions (e.g., ["us-west-2", "us-east-1"]).
        clusters: Optional list of specific clusters.
        attributes: Optional list of cluster attributes (e.g., ["HIPAA", "SOC2"]).
        configuration: Optional dict of helm chart value overrides.
    """

    gpu: str = "L40S"
    instance_type: Optional[str] = None
    min_instances: int = 1
    max_instances: int = 1
    max_request_concurrency: int = 1
    backend: str = "GFN"
    regions: Optional[List[str]] = None
    clusters: Optional[List[str]] = None
    attributes: Optional[List[str]] = None
    configuration: Optional[dict] = None

    def get_instance_type(self) -> str:
        """Get the instance type, using a default if not explicitly set."""
        if self.instance_type:
            return self.instance_type

        # Default instance types for common GPU/backend combinations
        # (Experiment.md: use exact type from --list-gpus when auto-detection fails)
        defaults = {
            ("GFN", "L40"): "gl40_1.br20_2xlarge",
            ("GFN", "L40G"): "gl40g_1.br25_2xlarge",
            ("GFN", "L40S"): "gl40s_4.br25_small",
            ("GFN", "T10"): "g6.full",
            ("AZURE", "H100"): "AZURE.GPU.H100_1x",
            ("GCP", "H100"): "a3-highgpu-8g_1x",
        }
        return defaults.get((self.backend, self.gpu), f"{self.backend}.GPU.{self.gpu}_1x")
