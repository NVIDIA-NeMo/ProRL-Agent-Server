#!/usr/bin/env bash
# cw-dfw defaults for TMax Slime-GRPO on eight 8xH100 nodes.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
USER_ROOT="$(dirname "${SPILOT_ROOT}")"

export POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${SPILOT_ROOT}/data}"
export TMAX_DATASET_DIR="${TMAX_DATASET_DIR:-${POLAR_DATA_ROOT}/tmax-15k}"
export APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${POLAR_DATA_ROOT}/tmax-15k-sif}"
export AGENT_CLI_DIR="${AGENT_CLI_DIR:-${POLAR_DATA_ROOT}/agent_cli/opt_node}"
export TMAX_SIF_PYTHON_BIN="${TMAX_SIF_PYTHON_BIN:-${USER_ROOT}/.python/polar/bin/python}"
export MINI_SWE_AGENT_RUNTIME_DIR="${MINI_SWE_AGENT_RUNTIME_DIR:-${POLAR_DATA_ROOT}/mini_swe_agent_runtime}"
export MINI_SWE_AGENT_CONTAINER_DIR="${MINI_SWE_AGENT_CONTAINER_DIR:-/opt/polar-mini-swe-agent}"
export MINI_SWE_AGENT_PYTHON_ROOT="${MINI_SWE_AGENT_PYTHON_ROOT:-${USER_ROOT}/tb_runs/pyportable/cpython-3.12.13-linux-x86_64-gnu}"
export MINI_SWE_AGENT_SPEC="${MINI_SWE_AGENT_SPEC:-mini-swe-agent==2.4.2}"
export MINI_SWE_AGENT_BIN="${MINI_SWE_AGENT_BIN:-${MINI_SWE_AGENT_RUNTIME_DIR}/bin/mini-swe-agent}"

export ACCOUNT="${ACCOUNT:-nvr_lpr_llm}"
export PARTITION="${PARTITION:-batch}"
export SLURM_CONSTRAINT="${SLURM_CONSTRAINT:-H100}"
export SLURM_GPUS="${SLURM_GPUS:-8}"
export NUM_NODES="${NUM_NODES:-8}"
export WALL_TIME="${WALL_TIME:-4:00:00}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
export SLURM_STEP_CPUS_PER_TASK="${SLURM_STEP_CPUS_PER_TASK:-96}"
export SUBMIT_BACKEND="${SUBMIT_BACKEND:-sbatch}"

DEFAULT_TRAIN_SQSH="${POLAR_DATA_ROOT}/container/flappydora-ubuntu22.04-cuda13.3.sqsh"
if [ ! -f "${DEFAULT_TRAIN_SQSH}" ] && [ -f "${USER_ROOT}/spilot-router/container/polar_train.sqsh" ]; then
    DEFAULT_TRAIN_SQSH="${USER_ROOT}/spilot-router/container/polar_train.sqsh"
fi
export POLR_TRAIN_SQSH="${POLR_TRAIN_SQSH:-${DEFAULT_TRAIN_SQSH}}"
export POLR_TRAIN_VENV="${POLR_TRAIN_VENV:-${USER_ROOT}/.python/polar}"
export TRAIN_CONTAINER_MOUNTS="${TRAIN_CONTAINER_MOUNTS:-/lustre/fsw:/lustre/fsw}"
export POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-/usr/bin/apptainer}"
export POLAR_APPTAINER_NO_INSTANCE="${POLAR_APPTAINER_NO_INSTANCE:-1}"
export POLAR_APPTAINER_NO_MOUNT_HOSTFS="${POLAR_APPTAINER_NO_MOUNT_HOSTFS:-1}"
export POLAR_APPTAINER_DIRECT_EXEC_RETRIES="${POLAR_APPTAINER_DIRECT_EXEC_RETRIES:-3}"

export HF_HOME="${HF_HOME:-${USER_ROOT}/.cache/huggingface}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3.5-4B}"
DEFAULT_REF_LOAD="${POLAR_DATA_ROOT}/checkpoints/Qwen3.5-4B_torch_dist"
if [ ! -f "${DEFAULT_REF_LOAD}/latest_checkpointed_iteration.txt" ] && \
   [ -f "${USER_ROOT}/spilot-router/data/checkpoints/Qwen3.5-4B_torch_dist/latest_checkpointed_iteration.txt" ]; then
    DEFAULT_REF_LOAD="${USER_ROOT}/spilot-router/data/checkpoints/Qwen3.5-4B_torch_dist"
fi
export REF_LOAD="${REF_LOAD:-${DEFAULT_REF_LOAD}}"
export TORCH_DIST_DIR="${TORCH_DIST_DIR:-${REF_LOAD}}"
export SLIME_DIR="${SLIME_DIR:-${SPILOT_ROOT}/src/slime}"
DEFAULT_MEGATRON_DIR="${POLAR_DATA_ROOT}/Megatron-LM-slime-v0.3.0"
ROUTER_MEGATRON_DIR="${USER_ROOT}/spilot-router/data/Megatron-LM-slime-v0.3.0"
if [ ! -f "${DEFAULT_MEGATRON_DIR}/megatron/training/tokenizer/tokenizer.py" ] && \
   [ -f "${ROUTER_MEGATRON_DIR}/megatron/training/tokenizer/tokenizer.py" ]; then
    DEFAULT_MEGATRON_DIR="${ROUTER_MEGATRON_DIR}"
fi
export MEGATRON_DIR="${MEGATRON_DIR:-${DEFAULT_MEGATRON_DIR}}"

# Five actor nodes give TP=2 / DP=20. Three rollout nodes host 24 independent
# SGLang engines; 40 trajectories per update is divisible by actor DP.
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-5}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-24}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
export RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-5}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-16000}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-20000}"
export SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-50000}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-1}"
export DIST_CKPT_STRICTNESS="${DIST_CKPT_STRICTNESS:-log_all}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export NUM_EPOCH="${NUM_EPOCH:-1}"
export TMAX_GRACEFUL_EXIT_BUFFER_SECONDS="${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS:-900}"
export TMAX_ENABLE_GRACEFUL_EXIT="${TMAX_ENABLE_GRACEFUL_EXIT:-1}"
export POLAR_FULLY_ASYNC="${POLAR_FULLY_ASYNC:-true}"
export POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-2}"
export POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-1200}"
export POLAR_TASK_TIMEOUT_SECONDS="${POLAR_TASK_TIMEOUT_SECONDS:-840}"
export POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.5}"
export POLAR_MAX_INIT_WORKERS="${POLAR_MAX_INIT_WORKERS:-96}"
export POLAR_MAX_RUN_WORKERS="${POLAR_MAX_RUN_WORKERS:-96}"
export POLAR_MAX_POSTRUN_WORKERS="${POLAR_MAX_POSTRUN_WORKERS:-96}"

export TMAX_AGENT_HARNESS="${TMAX_AGENT_HARNESS:-${POLAR_AGENT_HARNESS:-mini_swe_agent}}"
export POLAR_AGENT_HARNESS="${TMAX_AGENT_HARNESS}"
export POLAR_AGENT_MODEL_NAME="${POLAR_AGENT_MODEL_NAME:-Qwen/Qwen3.5-4B}"
export POLAR_AGENT_STEP_LIMIT="${POLAR_AGENT_STEP_LIMIT:-30}"
export POLAR_AGENT_COST_LIMIT="${POLAR_AGENT_COST_LIMIT:-0}"
case "${TMAX_AGENT_HARNESS}" in
    mini_swe_agent)
        export POLAR_AGENT_PATH="${MINI_SWE_AGENT_CONTAINER_DIR}/bin:/opt/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        printf -v POLAR_AGENT_RUNTIME_VOLUME '        - %s:%s:ro' \
            "${MINI_SWE_AGENT_RUNTIME_DIR}" "${MINI_SWE_AGENT_CONTAINER_DIR}"
        export POLAR_AGENT_RUNTIME_VOLUME
        ;;
    codex)
        export POLAR_AGENT_PATH="/opt/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        export POLAR_AGENT_RUNTIME_VOLUME=""
        ;;
    *)
        echo "ERROR: unsupported TMAX_AGENT_HARNESS=${TMAX_AGENT_HARNESS}" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

export TMAX_ONLY_READY="${TMAX_ONLY_READY:-1}"
export TMAX_MAX_TASKS="${TMAX_MAX_TASKS:-80}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-tmax-mini-swe-qwen35-4b-8n-full-async}"
export RUN_ID="${RUN_ID:-${EXPERIMENT_NAME}-$(date -u +%Y%m%dT%H%M%SZ)}"
export TMAX_TRAIN_DATA="${TMAX_TRAIN_DATA:-${POLAR_DATA_ROOT}/runs/${RUN_ID}/tmax-train.jsonl}"
if [ -z "${SLIME_ROLLOUT_BASE_PORT:-}" ]; then
    _tmax_run_cksum="$(printf '%s' "${RUN_ID}" | cksum | awk '{print $1}')"
    export SLIME_ROLLOUT_BASE_PORT="$((20000 + (_tmax_run_cksum % 64) * 160))"
    unset _tmax_run_cksum
fi
export SAVE_DIR="${SAVE_DIR:-${POLAR_DATA_ROOT}/ckpt/${RUN_ID}}"
export TRAINING_COMPLETE_MARKER="${TRAINING_COMPLETE_MARKER:-${SAVE_DIR}/TRAINING_COMPLETE}"
export TMAX_RUN_STATE_FILE="${TMAX_RUN_STATE_FILE:-${POLAR_DATA_ROOT}/runs/tmax_slime_grpo/current_run.env}"
export WANDB_PROJECT="${WANDB_PROJECT:-polar-tmax-grpo}"
export WANDB_GROUP="${WANDB_GROUP:-tmax-mini-swe-qwen35-4b-full-async-5a3r}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_ALWAYS_USE_TRAIN_STEP="${WANDB_ALWAYS_USE_TRAIN_STEP:-1}"
export TMAX_REQUIRE_WANDB="${TMAX_REQUIRE_WANDB:-1}"
export GPU_MONITOR_ENABLED="${GPU_MONITOR_ENABLED:-1}"
export GPU_MONITOR_NODE_ROLE="${GPU_MONITOR_NODE_ROLE:-rank}"

_polar_load_export_from_zshrc() {
    local name="$1"
    local line value
    if [ -n "${!name:-}" ] || [ ! -f "$HOME/.zshrc" ]; then
        return
    fi
    line="$(grep -E "^export ${name}=" "$HOME/.zshrc" 2>/dev/null | tail -n 1 || true)"
    [ -n "$line" ] || return
    value="${line#export ${name}=}"
    eval "export ${name}=${value}"
}

_polar_load_export_from_zshrc WANDB_API_KEY
_polar_load_export_from_zshrc HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# The converted Megatron checkpoint contains trusted non-tensor metadata. PyTorch
# 2.6+ otherwise changes legacy torch.load() call sites to weights-only loading.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export https_proxy="${https_proxy:-http://cw-dfw-cs-001-container-cache:3128}"
export http_proxy="${http_proxy:-${https_proxy}}"

mkdir -p \
    "${POLAR_DATA_ROOT}/agent_cli" \
    "${POLAR_DATA_ROOT}/checkpoints" \
    "$(dirname "${MINI_SWE_AGENT_RUNTIME_DIR}")" \
    "${POLAR_DATA_ROOT}/runs" \
    "${POLAR_DATA_ROOT}/ckpt"

echo "[tmax env] nodes=${NUM_NODES} gpus/node=${SLURM_GPUS} partition=${PARTITION} no_instance=${POLAR_APPTAINER_NO_INSTANCE}"
echo "[tmax env] dataset=${TMAX_DATASET_DIR} sif_dir=${APPTAINER_IMAGE_DIR} train_data=${TMAX_TRAIN_DATA}"
echo "[tmax env] actor_nodes=${ACTOR_NUM_NODES} rollout_gpus=${ROLLOUT_NUM_GPUS} batch=${ROLLOUT_BATCH_SIZE}x${N_SAMPLES_PER_PROMPT} fully_async=${POLAR_FULLY_ASYNC}/${POLAR_MAX_ASYNC_LEVEL}"
echo "[tmax env] harness=${TMAX_AGENT_HARNESS} runtime=${MINI_SWE_AGENT_RUNTIME_DIR}"
echo "[tmax env] train_sqsh=${POLR_TRAIN_SQSH} slime=${SLIME_DIR} ref_load=${REF_LOAD} run_id=${RUN_ID} sglang_base_port=${SLIME_ROLLOUT_BASE_PORT}"
