#!/bin/bash
# Polar SLURM environment configuration.
# Source this file in all SLURM job scripts and helpers.
#
# Required: POLAR_WORKSPACE must be set before sourcing.
# The easiest way is via 'polar cluster launch -c cluster.yaml'.

# ── Cluster paths ──────────────────────────────────────────────────────────────
if [ -z "${POLAR_WORKSPACE:-}" ]; then
    echo "ERROR: POLAR_WORKSPACE must be set before sourcing env.sh" >&2
    echo "  Use: polar cluster launch -c cluster.yaml" >&2
    return 1 2>/dev/null || exit 1
fi

export POLAR_ROOT="${POLAR_ROOT:-${POLAR_WORKSPACE}/polar}"
export POLAR_CODE="${POLAR_CODE:-${POLAR_ROOT}/ProRL-Agent-Server}"
export POLAR_SIFS="${POLAR_SIFS:-${POLAR_ROOT}/sif_images}"
export POLAR_RESULTS="${POLAR_RESULTS:-${POLAR_ROOT}/results}"
export POLAR_VENV="${POLAR_VENV:-${POLAR_ROOT}/.venv}"

# ── Apptainer (optional — skip if system-installed) ───────────────────────────
if [ -n "${APPTAINER_BIN_DIR:-}" ]; then
    export PATH="${APPTAINER_BIN_DIR}:${PATH}"
fi
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${POLAR_ROOT}/apptainer_cache}"

# ── CUDA (optional — for FlashInfer GDN kernel JIT compilation) ───────────────
if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

# ── Caches (redirect to shared storage to avoid home directory quota) ─────────
export HF_HOME="${HF_HOME:-${POLAR_WORKSPACE}/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-${POLAR_WORKSPACE}/torch_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${POLAR_WORKSPACE}/pip_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${POLAR_WORKSPACE}/xdg_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${POLAR_WORKSPACE}/triton_cache}"

# ── Python ─────────────────────────────────────────────────────────────────────
if [ -d "${POLAR_VENV}" ]; then
    source "${POLAR_VENV}/bin/activate"
fi

# Ensure polar source tree is importable even without pip install -e
export PYTHONPATH="${POLAR_CODE}/src${PYTHONPATH:+:${PYTHONPATH}}"

# ── Service ports ──────────────────────────────────────────────────────────────
export VLLM_PORT="${VLLM_PORT:-18000}"
export ROLLOUT_PORT="${ROLLOUT_PORT:-18080}"
export GATEWAY_BASE_PORT="${GATEWAY_BASE_PORT:-18100}"

# ── vLLM defaults ──────────────────────────────────────────────────────────────
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-27B}"
export MODEL_PATH="${MODEL_PATH:-${MODEL_NAME}}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"

# ── Gateway defaults ──────────────────────────────────────────────────────────
export MAX_INIT_WORKERS="${MAX_INIT_WORKERS:-8}"
export MAX_RUN_WORKERS="${MAX_RUN_WORKERS:-4}"
export MAX_POSTRUN_WORKERS="${MAX_POSTRUN_WORKERS:-4}"
export READY_BUFFER_TARGET="${READY_BUFFER_TARGET:-4}"

# ── Training defaults (used by polar_slurm_train.sbatch) ─────────────────────
export SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-9000}"
export RAY_PORT="${RAY_PORT:-6379}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
