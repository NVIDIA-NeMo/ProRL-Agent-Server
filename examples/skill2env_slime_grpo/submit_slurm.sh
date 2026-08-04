#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "${PRM_KEY:-}" ]; then
    # The user's credential is intentionally loaded only into the submission
    # environment; it is never interpolated into YAML or persisted run state.
    # shellcheck source=/dev/null
    source "${HOME}/.bashrc" >/dev/null 2>&1 || true
fi
if [ -z "${PRM_KEY:-}" ]; then
    echo "ERROR: PRM_KEY is not set after sourcing ~/.bashrc" >&2
    exit 1
fi
export PRM_KEY
export PRM_BASE_URL="${PRM_BASE_URL:-https://inference-api.nvidia.com/v1/responses}"
export PRM_MODEL="${PRM_MODEL:-azure/openai/gpt-5.3-codex}"
export PRM_RUBRIC_COEFFICIENT="${PRM_RUBRIC_COEFFICIENT:-0.2}"
export PRM_TIMEOUT_SECONDS="${PRM_TIMEOUT_SECONDS:-180}"
export PRM_INCLUDE_TOOL_OUTPUTS="${PRM_INCLUDE_TOOL_OUTPUTS:-true}"
export PRM_TOOL_OUTPUT_MAX_CHARS="${PRM_TOOL_OUTPUT_MAX_CHARS:-12000}"
export PRM_MAX_TRACES_PER_CALL="${PRM_MAX_TRACES_PER_CALL:-32}"
export POLAR_TRAJECTORY_BUILDER="${POLAR_TRAJECTORY_BUILDER:-per_request}"
case "${POLAR_TRAJECTORY_BUILDER}" in
    per_request|prefix_merging) ;;
    *)
        echo "ERROR: POLAR_TRAJECTORY_BUILDER must be per_request or prefix_merging" >&2
        exit 1
        ;;
esac

export POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${PROJECT_ROOT}/local_data}"
export POLR_TRAIN_VENV="${POLR_TRAIN_VENV:-${POLAR_DATA_ROOT}/train_runtime_venv}"
export POLR_TRAIN_PYTHON_OVERLAY="${POLR_TRAIN_PYTHON_OVERLAY:-${POLAR_DATA_ROOT}/train_python_overlay}"
export SLIME_DIR="${SLIME_DIR:-${POLAR_DATA_ROOT}/slime}"
export MINI_SWE_AGENT_RUNTIME_DIR="${MINI_SWE_AGENT_RUNTIME_DIR:-${POLAR_DATA_ROOT}/mini_swe_agent_runtime}"
export MINI_SWE_AGENT_BIN="${MINI_SWE_AGENT_BIN:-${MINI_SWE_AGENT_RUNTIME_DIR}/bin/mini-swe-agent}"
export SKILL2ENV_TRAIN_DATA="${SKILL2ENV_TRAIN_DATA:-${POLAR_DATA_ROOT}/seeds/skill2env-batch5-ready.jsonl}"
export SKILL2ENV_PREPARE_DATA="${SKILL2ENV_PREPARE_DATA:-1}"
if [ "${SKILL2ENV_PREPARE_DATA}" = "1" ]; then
    bash "${SCRIPT_DIR}/prepare_data.sh" >/dev/null
elif [ "${SKILL2ENV_PREPARE_DATA}" != "0" ]; then
    echo "ERROR: SKILL2ENV_PREPARE_DATA must be 0 or 1" >&2
    exit 1
elif [ ! -s "${SKILL2ENV_TRAIN_DATA}" ]; then
    echo "ERROR: SKILL2ENV_PREPARE_DATA=0 but data is missing or empty: ${SKILL2ENV_TRAIN_DATA}" >&2
    exit 1
fi
task_count="$(awk 'NF { count += 1 } END { print count + 0 }' "${SKILL2ENV_TRAIN_DATA}")"
if [ "${task_count}" -lt 1 ]; then
    echo "ERROR: no Skill2Env tasks with matching SIFs were prepared" >&2
    exit 1
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-skill2env-qwen35-4b-rubric-prm}"
export RUN_ID="${RUN_ID:-${EXPERIMENT_NAME}-${stamp}}"
export JOB_NAME="${JOB_NAME:-polar-${RUN_ID}}"
export SAVE_DIR="${SAVE_DIR:-${POLAR_DATA_ROOT}/ckpt/${RUN_ID}}"
export TMAX_RUN_STATE_FILE="${TMAX_RUN_STATE_FILE:-${POLAR_DATA_ROOT}/runs/skill2env_slime_grpo/current_run.env}"

export TMAX_TRAIN_DATA="${SKILL2ENV_TRAIN_DATA}"
export TMAX_MAX_TASKS="${task_count}"
export TMAX_TOTAL_TASKS="${task_count}"
export TMAX_TRAIN_START_INDEX=0
export TMAX_REQUIRE_EXACT_TOTAL_TASKS=0
export TMAX_ONLY_READY=0
export TMAX_PREPARE_DATA=0
export TMAX_VALIDATE_EXISTING_ASSETS=0
export TMAX_EVAL_ENABLED=0
export TMAX_TRAINING_EVAL_ENABLED=0
export TMAX_EXTERNAL_EVAL_ENABLED=0
export TMAX_PERSIST_RUN_STATE=1

# Default smoke topology is two 8xH100 nodes. Full launchers use either one
# learner plus three rollout nodes, or two learners plus six rollout nodes.
export ACCOUNT="${ACCOUNT:-nvr_lpr_llm}"
export NUM_NODES="${NUM_NODES:-2}"
if [ -z "${PARTITION:-}" ]; then
    if [ "${NUM_NODES}" -eq 8 ]; then
        export PARTITION="batch"
    else
        export PARTITION="interactive"
    fi
fi
if [ "${PARTITION}" = "backfill" ]; then
    echo "ERROR: backfill is disabled for Skill2Env experiments; use batch" >&2
    exit 1
fi
export SLURM_GPUS="${SLURM_GPUS:-8}"
export WALL_TIME="${WALL_TIME:-4:00:00}"
export TMAX_MIN_WALL_TIME="${TMAX_MIN_WALL_TIME:-4:00:00}"
if [ "${NUM_NODES}" -eq 8 ]; then
    export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"
    export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-48}"
    export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
    export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
    export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
    export EVAL_GLOBAL_BATCH_SIZE="${EVAL_GLOBAL_BATCH_SIZE:-128}"
    export POLAR_MULTI_GATEWAY="${POLAR_MULTI_GATEWAY:-1}"
elif [ "${NUM_NODES}" -eq 4 ]; then
    export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
    export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-24}"
    export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
    export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
    export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
    export EVAL_GLOBAL_BATCH_SIZE="${EVAL_GLOBAL_BATCH_SIZE:-64}"
    export POLAR_MULTI_GATEWAY="${POLAR_MULTI_GATEWAY:-1}"
elif [ "${NUM_NODES}" -eq 2 ]; then
    export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
    export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-8}"
    export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
    export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
    export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
    export EVAL_GLOBAL_BATCH_SIZE="${EVAL_GLOBAL_BATCH_SIZE:-32}"
    export POLAR_MULTI_GATEWAY="${POLAR_MULTI_GATEWAY:-0}"
else
    echo "ERROR: Skill2Env supports NUM_NODES=2 (smoke), 4, or 8 (full), got ${NUM_NODES}" >&2
    exit 1
fi
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ACTOR_TENSOR_MODEL_PARALLEL_SIZE=4
export ACTOR_PIPELINE_MODEL_PARALLEL_SIZE=1
export CONTEXT_PARALLEL_SIZE=1
export ROLLOUT_NUM_GPUS_PER_ENGINE=1
export RAY_NUM_GPUS_PER_NODE=8
export TMAX_REQUIRE_FULL_GPU_ALLOCATION=1

# Smoke: 4 prompts x 8 trajectories = 32. Four-node full: 8 x 8 = 64.
# Eight-node full: 16 x 8 = 128.
export NUM_STEPS_PER_ROLLOUT=1
export NUM_EPOCH=1
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export TMAX_MIN_ASYNC_LEVEL=3
export POLAR_MAX_ASYNC_LEVEL=3
export TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU=8
export POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0
export POLAR_EARLY_STOP_GRACE_SESSIONS=0

# Qwen3.5-4B release checkpoint and its matching Megatron architecture.
export HF_CHECKPOINT="${HF_CHECKPOINT:-${POLAR_DATA_ROOT}/checkpoints/Qwen3.5-4B}"
export REF_LOAD="${REF_LOAD:-/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot-router/data/checkpoints/Qwen3.5-4B_torch_dist}"
export TORCH_DIST_DIR="${REF_LOAD}"
export MODEL_ARGS_FILE="${PROJECT_ROOT}/examples/swegym_slime_grpo/model_args.sh"
export POLAR_AGENT_MODEL_NAME="Qwen/Qwen3.5-4B"
export TMAX_ENABLE_FP32_LM_HEAD=0
export SGLANG_ENABLE_FP32_LM_HEAD=0
export TRAIN_LR="${TRAIN_LR:-5e-7}"
export MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF="${MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF:-0.5}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-45056}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-32}"

export APPTAINER_IMAGE_DIR="${SKILL2ENV_IMAGE_DIR:-/lustre/fsw/portfolios/llmservice/users/haozh/data/skill2env/skill2env_batch_5_sif}"
export POLAR_CONFIG_TEMPLATE="${SCRIPT_DIR}/polar_config.yaml"
export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS=600
export POLAR_TASK_TIMEOUT_FLOOR_SECONDS=1800
export TMAX_AGENT_HARNESS="${TMAX_AGENT_HARNESS:-mini_swe_agent}"
export POLAR_AGENT_HARNESS="${TMAX_AGENT_HARNESS}"

export WANDB_PROJECT="${WANDB_PROJECT:-polar-skill2env-grpo}"
export WANDB_GROUP="${WANDB_GROUP:-qwen35-4b-rubric-prm}"
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
export TMAX_REQUIRE_WANDB=0
export TMAX_TRAIN_ABI_PREFLIGHT="${TMAX_TRAIN_ABI_PREFLIGHT:-0}"
export SUBMIT_BACKEND="${SUBMIT_BACKEND:-sbatch}"

exec bash "${PROJECT_ROOT}/examples/tmax_slime_grpo/submit_slurm.sh"
