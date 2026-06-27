#!/usr/bin/env bash
# Node entrypoint used by Pyxis jobs. Keep the Slurm command line short.
set -euo pipefail

if [ -n "${POLAR_TRAIN_ENV_FILE:-}" ]; then
    # shellcheck disable=SC1090
    source "${POLAR_TRAIN_ENV_FILE}"
fi

PROJECT_ROOT="${POLAR_TRAIN_PROJECT_ROOT:?set POLAR_TRAIN_PROJECT_ROOT}"
TRAIN_RUN_SCRIPT="${POLAR_TRAIN_RUN_SCRIPT:?set POLAR_TRAIN_RUN_SCRIPT}"
TRAIN_VENV="${POLR_TRAIN_VENV:-/opt/polr_venv}"
DATA_ROOT="${POLAR_DATA_ROOT:-${PROJECT_ROOT}/tmp}"

export PATH="${TRAIN_VENV}/bin:${PATH}"
export PYTHON_BIN="${TRAIN_VENV}/bin/python3"
export VIRTUAL_ENV="${TRAIN_VENV}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-${SLURM_GPUS:-8}}"

export POLAR_JOB_CACHE_ROOT="/tmp/polar-cache-${SLURM_JOB_ID:-manual}"
rm -rf "${POLAR_JOB_CACHE_ROOT}"
mkdir -p \
    "${POLAR_JOB_CACHE_ROOT}/home" \
    "${POLAR_JOB_CACHE_ROOT}/apptainer-cache" \
    "${POLAR_JOB_CACHE_ROOT}/apptainer-tmp" \
    "${POLAR_JOB_CACHE_ROOT}/apptainer-work" \
    "${POLAR_JOB_CACHE_ROOT}/triton" \
    "${POLAR_JOB_CACHE_ROOT}/torchinductor" \
    "${POLAR_JOB_CACHE_ROOT}/xdg" \
    "${POLAR_JOB_CACHE_ROOT}/xdg-config" \
    "${POLAR_JOB_CACHE_ROOT}/xdg-runtime" \
    "${POLAR_JOB_CACHE_ROOT}/numba"
chmod 700 "${POLAR_JOB_CACHE_ROOT}/xdg-runtime"

export HOME="${POLAR_JOB_CACHE_ROOT}/home"
export APPTAINER_CACHEDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-cache"
export APPTAINER_TMPDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-tmp"
export APPTAINER_WORKDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-work"
export TRITON_CACHE_DIR="${POLAR_JOB_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${POLAR_JOB_CACHE_ROOT}/torchinductor"
export NUMBA_CACHE_DIR="${POLAR_JOB_CACHE_ROOT}/numba"
export XDG_CACHE_HOME="${POLAR_JOB_CACHE_ROOT}/xdg"
export XDG_CONFIG_HOME="${POLAR_JOB_CACHE_ROOT}/xdg-config"
export XDG_RUNTIME_DIR="${POLAR_JOB_CACHE_ROOT}/xdg-runtime"
export RUN_DIR="${RUN_DIR:-${DATA_ROOT}/runs/${RUN_ID}/job-${SLURM_JOB_ID:-manual}}"
export SAVE_DIR="${SAVE_DIR:-${DATA_ROOT}/ckpt/${RUN_ID}}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "FATAL: ${PYTHON_BIN} not found in the training container" >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -c "import torch, slime, polar, slime_bridge" >/dev/null 2>&1; then
    echo "FATAL: ${TRAIN_VENV} cannot import torch, slime, polar, and slime_bridge" >&2
    "${PYTHON_BIN}" -c "import torch, slime, polar, slime_bridge" || true
    exit 1
fi

cd "${DATA_ROOT}"
exec bash "${TRAIN_RUN_SCRIPT}"
