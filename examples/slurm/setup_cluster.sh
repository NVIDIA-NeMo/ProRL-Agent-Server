#!/bin/bash
# setup_cluster.sh — One-time cluster setup for Polar on SLURM.
#
# Run via:
#   polar cluster setup -c my-cluster.yaml
#
# Or manually on the cluster:
#   export POLAR_WORKSPACE=/path/to/workspace
#   bash examples/slurm/setup_cluster.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source env.sh from the cluster templates
source "${REPO_ROOT}/src/polar/cluster/templates/env.sh"

echo "==============================================================="
echo "Polar Cluster Setup"
echo "==============================================================="

# ── 1. Create directory structure ──────────────────────────────────────────────
echo "[setup] Creating directories..."
mkdir -p "${POLAR_ROOT}"
mkdir -p "${POLAR_SIFS}"
mkdir -p "${POLAR_RESULTS}"
mkdir -p "${APPTAINER_CACHEDIR}"
echo "  POLAR_ROOT:    ${POLAR_ROOT}"
echo "  POLAR_SIFS:    ${POLAR_SIFS}"
echo "  POLAR_RESULTS: ${POLAR_RESULTS}"
echo "  APPTAINER_CACHEDIR: ${APPTAINER_CACHEDIR}"

# ── 2. Check Apptainer ────────────────────────────────────────────────────────
echo ""
echo "[setup] Checking Apptainer..."
if command -v apptainer &>/dev/null; then
    echo "  Apptainer: $(which apptainer)"
    apptainer --version
else
    echo "  WARNING: Apptainer not found on PATH."
    echo "  Set paths.apptainer_bin_dir in your cluster.yaml."
fi

# ── 3. Create/update Python venv ──────────────────────────────────────────────
echo ""
echo "[setup] Setting up Python venv at ${POLAR_VENV}..."
if [ ! -d "${POLAR_VENV}" ]; then
    echo "  Creating new venv..."
    python3 -m venv "${POLAR_VENV}"
fi
source "${POLAR_VENV}/bin/activate"

echo "  Python: $(which python) ($(python --version))"

# Install polar in editable mode
echo "  Installing polar..."
pip install --upgrade pip
pip install -e "${POLAR_CODE}" 2>&1 | tail -5

# Check if vLLM is installed
if python -c "import vllm" 2>/dev/null; then
    echo "  vLLM: $(python -c 'import vllm; print(vllm.__version__)')"
else
    echo ""
    echo "  WARNING: vLLM is not installed in this venv."
    echo "  To install: pip install vllm"
fi

# ── 4. Verify polar CLI ───────────────────────────────────────────────────────
echo ""
echo "[setup] Verifying polar CLI..."
polar --help > /dev/null 2>&1 && echo "  polar CLI: OK" || echo "  ERROR: polar CLI not working"

# ── 5. Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "==============================================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Build SIF images:"
echo "     polar cluster build-sif -c cluster.yaml --example calculator --harness opencode"
echo ""
echo "  2. Submit a job:"
echo "     polar cluster launch -c cluster.yaml"
echo "==============================================================="
