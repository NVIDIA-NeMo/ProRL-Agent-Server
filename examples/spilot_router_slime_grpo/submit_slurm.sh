#!/usr/bin/env bash
# Submit the fixed-pool, 8-node, 200-step SPilot Router experiment.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
export POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${SPILOT_ROOT}/data}"
export TMAX_RUN_STATE_FILE="${TMAX_RUN_STATE_FILE:-${POLAR_DATA_ROOT}/runs/spilot_router_slime_grpo/current_run.env}"

# One 8-GPU learner node plus 56 independent TP1 Router rollout engines.
export NUM_NODES="${NUM_NODES:-8}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-56}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"

# The remote pool is the bottleneck, not local Router decoding. Keep one
# synchronous 8-prompt x 8-sample GRPO batch in flight (64 episodes) so the
# provider-side 32-request-per-model gates do not accumulate 1,000+ queued
# agents under TMax's default fully-async prefetch.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export POLAR_FULLY_ASYNC="${POLAR_FULLY_ASYNC:-false}"
export POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-1}"
export POLAR_MAX_INIT_WORKERS="${POLAR_MAX_INIT_WORKERS:-64}"
export POLAR_MAX_RUN_WORKERS="${POLAR_MAX_RUN_WORKERS:-64}"
export POLAR_MAX_POSTRUN_WORKERS="${POLAR_MAX_POSTRUN_WORKERS:-64}"

# Slime's --num-rollout is an exclusive boundary. This produces optimizer
# iterations 0..199, i.e. 200 training steps.
export TMAX_NUM_ROLLOUT="${TMAX_NUM_ROLLOUT:-200}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export TMAX_AGENT_HARNESS="${TMAX_AGENT_HARNESS:-spilot_router}"
export POLAR_AGENT_HARNESS="${TMAX_AGENT_HARNESS}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-spilot-router-qwen35-9b-8n-200step}"

# A two-call episode needs a larger active-agent envelope than the single-agent
# TMax default. The Harbor verifier still has a separate 600-second reserve.
export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS="${TMAX_TRAIN_AGENT_TIMEOUT_SECONDS:-3300}"
export POLAR_TASK_TIMEOUT_FLOOR_SECONDS="${POLAR_TASK_TIMEOUT_FLOOR_SECONDS:-4500}"
export POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-5100}"
export TMAX_GRACEFUL_EXIT_BUFFER_SECONDS="${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS:-4500}"

# Keep baseline/final evaluation small enough to complete inside one allocation
# while still measuring held-out TMax routing quality. Terminal-Bench is left
# for a separate post-training evaluation in this first systems experiment.
export TMAX_EVAL_MAX_TASKS="${TMAX_EVAL_MAX_TASKS:-32}"
export TMAX_EXTERNAL_EVAL_ENABLED="${TMAX_EXTERNAL_EVAL_ENABLED:-0}"
# The shared corpus currently has a tiny number of unfinished SIF builds in
# the 14,501-task train window. Select the deterministic ready subset once;
# TMax pins that exact JSONL across all watcher relaunches.
export TMAX_ONLY_READY="${TMAX_ONLY_READY:-1}"
export TMAX_REQUIRE_EXACT_TOTAL_TASKS="${TMAX_REQUIRE_EXACT_TOTAL_TASKS:-0}"

# The shared submitter serializes POLAR_* variables into a private mode-0600
# job environment. Copy the credential under that namespace without writing it
# into YAML, command lines, logs, or the repository.
if [ -z "${POLAR_NVIDIA_API_KEY:-${NVIDIA_API_KEY:-}}" ]; then
    echo "ERROR: NVIDIA_API_KEY is not set; load the credential before submission" >&2
    exit 1
fi
export POLAR_NVIDIA_API_KEY="${POLAR_NVIDIA_API_KEY:-${NVIDIA_API_KEY:-}}"
export POLAR_MODEL_POOL_BASE_URL="${POLAR_MODEL_POOL_BASE_URL:-${NVIDIA_BASE_URL:-https://integrate.api.nvidia.com/v1}}"

# Authenticate the trusted Slime -> rollout -> gateway control path separately
# from public task/session identifiers. The shared submitter copies POLAR_*
# values only into its mode-0600 allocation environment; this value is never
# rendered into YAML, run state, command lines, or agent runtimes.
if [ -z "${POLAR_CONTROL_PLANE_TOKEN:-}" ]; then
    command -v od >/dev/null || { echo "ERROR: od is required" >&2; exit 1; }
    command -v tr >/dev/null || { echo "ERROR: tr is required" >&2; exit 1; }
    POLAR_CONTROL_PLANE_TOKEN="$(od -An -N32 -tx1 /dev/urandom | tr -d '[:space:]')"
    export POLAR_CONTROL_PLANE_TOKEN
fi
if ! [[ "${POLAR_CONTROL_PLANE_TOKEN}" =~ ^[0-9A-Za-z_-]{32,128}$ ]]; then
    echo "ERROR: POLAR_CONTROL_PLANE_TOKEN must be a 32-128 character opaque token" >&2
    exit 1
fi

export POLAR_TRAIN_RUN_SCRIPT="${SCRIPT_DIR}/run.sh"
export POLAR_CONFIG_TEMPLATE="${SCRIPT_DIR}/polar_config.yaml"
export TOPOLOGY_TEMPLATE="${SCRIPT_DIR}/topology.yaml"

exec bash "${SCRIPT_DIR}/../tmax_slime_grpo/submit_slurm.sh" "$@"
