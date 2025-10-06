# Singularity Runtime Builder

The `SingularityRuntimeBuilder` class provides a unified interface for building container images using Singularity/Apptainer, following the same design patterns as the Docker builder but adapted for Singularity-specific functionality.

## Features

- **Unified Interface**: Implements the same `RuntimeBuilder` interface as Docker, ensuring consistency across different container runtimes
- **Singularity/Apptainer Support**: Works with both `singularity` and `apptainer` commands
- **Multiple Tags**: Supports building images with multiple tags using hard links
- **Docker Image Pulling**: Can pull Docker images and convert them to SIF format
- **Error Handling**: Comprehensive error handling with specific exception types
- **Logging**: Integrated logging with rolling logger support

## Basic Usage

```python
from openhands.runtime.builder import SingularityRuntimeBuilder

# Initialize the builder (defaults to 'apptainer')
builder = SingularityRuntimeBuilder()

# Or specify the command explicitly
builder = SingularityRuntimeBuilder('singularity')

# Build an image
result = builder.build(
    path='/path/to/build/directory',  # Directory containing singularity.def
    tags=['my_image.sif', 'my_image_v2.sif'],
    extra_build_args=['--fakeroot']
)

# Check if an image exists
exists = builder.image_exists('my_image.sif')

# Pull a Docker image and convert to SIF
docker_exists = builder.image_exists('ubuntu:22.04', pull_from_repo=True)
```

## Build Directory Structure

The build directory should contain a `singularity.def` file:

```
build_directory/
├── singularity.def
└── other_files...
```

Example `singularity.def`:
```
Bootstrap: docker
From: ubuntu:22.04

%post
    apt-get update
    apt-get install -y python3 python3-pip

%runscript
    exec "$@"
```

## Integration with OpenHands

The `SingularityRuntimeBuilder` integrates seamlessly with the existing OpenHands runtime infrastructure:

```python
from openhands.runtime.utils.singularity_runtime_build import build_runtime_image
from openhands.runtime.builder import SingularityRuntimeBuilder

# Build a runtime image
builder = SingularityRuntimeBuilder()
image_path = build_runtime_image(
    base_image='ubuntu:22.04',
    runtime_builder=builder,
    extra_deps='numpy pandas',
    force_rebuild=False
)
```

## Error Handling

The builder raises `AgentRuntimeBuildError` exceptions for various failure conditions:

- Missing Singularity/Apptainer installation
- Missing definition file
- Build process failures
- Permission errors

```python
from openhands.core.exceptions import AgentRuntimeBuildError

try:
    result = builder.build(path='/invalid/path', tags=['test.sif'])
except AgentRuntimeBuildError as e:
    print(f"Build failed: {e}")
```

## Comparison with Docker Builder

| Feature | Docker Builder | Singularity Builder |
|---------|----------------|---------------------|
| Base Command | `docker buildx` | `singularity build` or `apptainer build` |
| Definition File | `Dockerfile` | `singularity.def` |
| Image Format | Docker images | SIF files |
| Multiple Tags | Registry tags | Hard links |
| Cache Support | BuildKit cache | Singularity native cache |
| Platform Support | Multi-platform | Single platform |

## Requirements

- Singularity/Apptainer installed on the system
- Appropriate permissions for building images
- Valid definition file in the build directory

## Notes

- The builder defaults to using 'apptainer' but can be configured to use 'singularity'
- Multiple tags are created using hard links to save disk space
- Docker images can be automatically pulled and converted to SIF format
- The builder follows the same interface as `DockerRuntimeBuilder` for consistency
