#!/usr/bin/env bash
# Prepare reusable SWE-Gym assets inside the flappydora/Pyxis training image:
#   1. Qwen3.5-4B HF -> Megatron torch_dist checkpoint
#   2. missing per-instance SWE-Gym Apptainer SIFs
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=./env.cwdfw.sh
source "${SCRIPT_DIR}/env.cwdfw.sh"

PYTHON_BIN="${PYTHON_BIN:-${POLR_TRAIN_VENV}/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-${POLR_TRAIN_VENV}}"
export PYTHONNOUSERSITE=1

export POLAR_JOB_CACHE_ROOT="${POLAR_JOB_CACHE_ROOT:-/tmp/polar-assets-${SLURM_JOB_ID:-manual}}"
rm -rf "${POLAR_JOB_CACHE_ROOT}"
mkdir -p \
    "${POLAR_JOB_CACHE_ROOT}/home" \
    "${POLAR_JOB_CACHE_ROOT}/apptainer-cache" \
    "${POLAR_JOB_CACHE_ROOT}/apptainer-tmp" \
    "${POLAR_JOB_CACHE_ROOT}/triton" \
    "${POLAR_JOB_CACHE_ROOT}/torchinductor" \
    "${POLAR_JOB_CACHE_ROOT}/xdg" \
    "${POLAR_JOB_CACHE_ROOT}/xdg-config" \
    "${POLAR_JOB_CACHE_ROOT}/xdg-runtime"
chmod 700 "${POLAR_JOB_CACHE_ROOT}/xdg-runtime"

export HOME="${POLAR_JOB_CACHE_ROOT}/home"
export APPTAINER_CACHEDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-cache"
export APPTAINER_TMPDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-tmp"
export TRITON_CACHE_DIR="${POLAR_JOB_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${POLAR_JOB_CACHE_ROOT}/torchinductor"
export XDG_CACHE_HOME="${POLAR_JOB_CACHE_ROOT}/xdg"
export XDG_CONFIG_HOME="${POLAR_JOB_CACHE_ROOT}/xdg-config"
export XDG_RUNTIME_DIR="${POLAR_JOB_CACHE_ROOT}/xdg-runtime"

"${PYTHON_BIN}" -c "import torch, slime, polar, slime_bridge, mbridge"

if [ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]; then
    echo "[prepare-assets] converting weights: ${HF_CHECKPOINT} -> ${TORCH_DIST_DIR}"
    if [ -d "${TORCH_DIST_DIR}" ] && [ ! -f "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]; then
        case "${TORCH_DIST_DIR}" in
            "${POLAR_DATA_ROOT}/checkpoints/"*)
                echo "[prepare-assets] removing incomplete checkpoint directory: ${TORCH_DIST_DIR}"
                rm -rf "${TORCH_DIST_DIR}"
                ;;
            *)
                echo "[prepare-assets] refusing to remove unexpected checkpoint directory: ${TORCH_DIST_DIR}" >&2
                exit 1
                ;;
        esac
    fi
    bash "${SCRIPT_DIR}/convert_weights.sh"
else
    echo "[prepare-assets] checkpoint exists: ${REF_LOAD}"
fi

echo "[prepare-assets] building missing SIFs into ${APPTAINER_IMAGE_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_sifs.py" \
    --jobs "${SIF_BUILD_JOBS:-4}" \
    --cache-root "${POLAR_JOB_CACHE_ROOT}"

echo "[prepare-assets] done"
