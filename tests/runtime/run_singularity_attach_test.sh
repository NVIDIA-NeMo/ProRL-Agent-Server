#!/bin/bash

# Script to run the SingularityRuntime attach test
# This script sets up the proper environment and runs the test

set -e

echo "=== SingularityRuntime Attach Test Runner ==="
echo ""

# Check if Singularity/Apptainer is available
if ! command -v singularity &> /dev/null && ! command -v apptainer &> /dev/null; then
    echo "ERROR: Neither 'singularity' nor 'apptainer' command found."
    echo "Please install Singularity or Apptainer to run this test."
    exit 1
fi

# Check which command is available
if command -v singularity &> /dev/null; then
    SINGULARITY_CMD="singularity"
elif command -v apptainer &> /dev/null; then
    SINGULARITY_CMD="apptainer"
fi

echo "Found $SINGULARITY_CMD command"
$SINGULARITY_CMD --version
echo ""

# Set environment variables
export TEST_RUNTIME=singularity
export PYTHONPATH=${PYTHONPATH}:$(pwd)

# Change to the project root if we're in the tests/runtime directory
if [[ $(basename $(pwd)) == "runtime" ]]; then
    cd ../..
fi

echo "Running SingularityRuntime attach test..."
echo "TEST_RUNTIME=$TEST_RUNTIME"
echo "Working directory: $(pwd)"
echo ""

# Run the test with verbose output
pytest tests/runtime/test_singularity_attach.py -v -s

echo ""
echo "=== Test completed ==="
