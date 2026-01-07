#!/bin/bash
# Build NeAR runtime container for ProRL-Agent-Server

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMAGE_NAME="${DOCKER_IMAGE_PREFIX:-nvidia/}near-runtime:latest"

echo "Building NeAR runtime container: $IMAGE_NAME"

# Build Docker image
docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"

echo "Build complete: $IMAGE_NAME"
echo ""
echo "To test the container, run:"
echo "  docker run --rm -it $IMAGE_NAME bash"
echo ""
echo "To verify tools are installed:"
echo "  docker run --rm $IMAGE_NAME python -c 'import playwright; import IPython; print(\"Tools OK\")'"
