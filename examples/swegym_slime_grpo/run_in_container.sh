#!/usr/bin/env bash
# Node entrypoint used by Pyxis jobs. Keep the Slurm command line short.
set -euo pipefail
_SLIME_CONTAINER_ENTRY_UNIX_NS="$(date +%s%N)"
_SLIME_SLURM_BATCH_START_UNIX_NS="${SLIME_SLURM_BATCH_START_UNIX_NS:-}"

if [ -n "${POLAR_TRAIN_ENV_FILE:-}" ]; then
    # shellcheck disable=SC1090
    source "${POLAR_TRAIN_ENV_FILE}"
fi
export SLIME_CONTAINER_ENTRY_UNIX_NS="${_SLIME_CONTAINER_ENTRY_UNIX_NS}"
unset _SLIME_CONTAINER_ENTRY_UNIX_NS
if [ -n "${_SLIME_SLURM_BATCH_START_UNIX_NS}" ]; then
    export SLIME_SLURM_BATCH_START_UNIX_NS="${_SLIME_SLURM_BATCH_START_UNIX_NS}"
else
    unset SLIME_SLURM_BATCH_START_UNIX_NS
fi
unset _SLIME_SLURM_BATCH_START_UNIX_NS
# Runtime markers describe this allocation only. A submit shell can itself be
# inside an older allocation, so never inherit downstream phase timestamps.
unset \
    SLIME_JOB_SCRIPT_START_UNIX_NS \
    SLIME_RAY_READY_UNIX_NS \
    SLIME_POLAR_ROLLOUT_START_UNIX_NS \
    SLIME_POLAR_ROLLOUT_READY_UNIX_NS \
    SLIME_POLAR_GATEWAY_START_UNIX_NS \
    SLIME_POLAR_GATEWAY_READY_UNIX_NS \
    SLIME_POLAR_UDS_START_UNIX_NS \
    SLIME_POLAR_UDS_READY_UNIX_NS \
    SLIME_POLAR_READY_UNIX_NS \
    SLIME_RAY_JOB_SUBMIT_UNIX_NS

PROJECT_ROOT="${POLAR_TRAIN_PROJECT_ROOT:?set POLAR_TRAIN_PROJECT_ROOT}"
TRAIN_RUN_SCRIPT="${POLAR_TRAIN_RUN_SCRIPT:?set POLAR_TRAIN_RUN_SCRIPT}"
TRAIN_VENV="${POLR_TRAIN_VENV:-/opt/polr_venv}"
TRAIN_PYTHON_OVERLAY="${POLR_TRAIN_PYTHON_OVERLAY:-}"
DATA_ROOT="${POLAR_DATA_ROOT:-${PROJECT_ROOT}/tmp}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
# shellcheck source=./launcher_utils.sh
source "${PROJECT_ROOT}/examples/swegym_slime_grpo/launcher_utils.sh"

# Derive this only after Slurm has assigned the allocation.  Persisting the
# port with RUN_ID made a restarted run reuse ports left by the prior job.
polar_configure_rollout_base_port
polar_configure_sglang_router_port

export PATH="${TRAIN_VENV}/bin:${PATH}"
export PYTHON_BIN="${TRAIN_VENV}/bin/python3"
export VIRTUAL_ENV="${TRAIN_VENV}"
export PYTHONNOUSERSITE=1
# Prefer the copied workspace over the source owner's editable-install .pth.
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
if [ -n "${TRAIN_PYTHON_OVERLAY}" ]; then
    case "${TRAIN_PYTHON_OVERLAY}" in
        /*) ;;
        *)
            echo "FATAL: POLR_TRAIN_PYTHON_OVERLAY must be absolute: ${TRAIN_PYTHON_OVERLAY}" >&2
            exit 1
            ;;
    esac
    if [ ! -d "${TRAIN_PYTHON_OVERLAY}/transformer_engine" ]; then
        echo "FATAL: invalid training Python overlay: ${TRAIN_PYTHON_OVERLAY}" >&2
        exit 1
    fi
    export PYTHONPATH="${TRAIN_PYTHON_OVERLAY}:${PYTHONPATH}"
    export LD_LIBRARY_PATH="${TRAIN_PYTHON_OVERLAY}/transformer_engine/wheel_lib:${LD_LIBRARY_PATH}"
fi
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
# Shared training venvs may contain owner-only __pycache__ entries. Point
# importlib at our overlay cache so readable sources remain usable. Keeping
# this cache across allocations also avoids repeatedly compiling the large
# Torch/FLA import graph on Lustre.
if [ -n "${TRAIN_PYTHON_OVERLAY}" ]; then
    export PYTHONPYCACHEPREFIX="${TRAIN_PYTHON_OVERLAY}/python-pycache"
else
    export PYTHONPYCACHEPREFIX="${POLAR_JOB_CACHE_ROOT}/pycache"
fi
mkdir -p "${PYTHONPYCACHEPREFIX}"
chmod 700 "${PYTHONPYCACHEPREFIX}"
export APPTAINER_CACHEDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-cache"
export APPTAINER_TMPDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-tmp"
export APPTAINER_WORKDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-work"
export TRITON_CACHE_DIR="${POLAR_JOB_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${POLAR_JOB_CACHE_ROOT}/torchinductor"
export NUMBA_CACHE_DIR="${POLAR_JOB_CACHE_ROOT}/numba"
export XDG_CACHE_HOME="${POLAR_JOB_CACHE_ROOT}/xdg"
export XDG_CONFIG_HOME="${POLAR_JOB_CACHE_ROOT}/xdg-config"
export XDG_RUNTIME_DIR="${POLAR_JOB_CACHE_ROOT}/xdg-runtime"
if [ -n "${FLASHINFER_WORKSPACE_BASE:-}" ]; then
    case "${FLASHINFER_WORKSPACE_BASE}" in
        /*) ;;
        *)
            echo "FATAL: FLASHINFER_WORKSPACE_BASE must be absolute: ${FLASHINFER_WORKSPACE_BASE}" >&2
            exit 1
            ;;
    esac
    # All node entrypoints may race here. mkdir/chmod are idempotent and the
    # FlashInfer JIT itself serializes each shared artifact with flock.
    mkdir -p "${FLASHINFER_WORKSPACE_BASE}"
    chmod 700 "${FLASHINFER_WORKSPACE_BASE}"
fi
export RUN_DIR="${RUN_DIR:-${DATA_ROOT}/runs/${RUN_ID}/job-${SLURM_JOB_ID:-manual}}"
export SAVE_DIR="${SAVE_DIR:-${DATA_ROOT}/ckpt/${RUN_ID}}"
export POLAR_ROLLOUT_SAVE_DIR="${POLAR_ROLLOUT_SAVE_DIR:-${RUN_DIR}/rollout_results}"
case "${POLAR_ROLLOUT_SAVE_DIR}" in
    /*) ;;
    *)
        echo "FATAL: POLAR_ROLLOUT_SAVE_DIR must be absolute: ${POLAR_ROLLOUT_SAVE_DIR}" >&2
        exit 1
        ;;
esac

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "FATAL: ${PYTHON_BIN} not found in the training container" >&2
    exit 1
fi
TRAIN_ABI_PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PYTHONPATH}"
if ! PYTHONPATH="${TRAIN_ABI_PYTHONPATH}" "${PYTHON_BIN}" -c \
    "import torch, transformer_engine.pytorch, megatron.core.tensor_parallel, slime, polar, slime_bridge" \
    >/dev/null 2>&1; then
    echo "FATAL: ${TRAIN_VENV} failed the torch/Transformer-Engine/Megatron ABI preflight" >&2
    PYTHONPATH="${TRAIN_ABI_PYTHONPATH}" "${PYTHON_BIN}" -c \
        "import torch, transformer_engine.pytorch, megatron.core.tensor_parallel, slime, polar, slime_bridge" || true
    exit 1
fi
unset TRAIN_ABI_PYTHONPATH

# Nested Apptainer automatically bind-mounts the launcher's current directory.
# Keep that implicit bind on node-local, job-private storage so an untrusted
# task cannot see or mutate the shared dataset / SIF / agent-runtime root.
cd "${HOME}"
exec bash "${TRAIN_RUN_SCRIPT}"
