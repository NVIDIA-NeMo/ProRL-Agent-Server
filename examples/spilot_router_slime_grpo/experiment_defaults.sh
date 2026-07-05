#!/usr/bin/env bash
# Immutable-by-default contract shared by first submission and watcher bootstrap.

_SPILOT_ROUTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
_SPILOT_PROJECT_ROOT="$(cd -- "${_SPILOT_ROUTER_DIR}/../.." && pwd)"
_SPILOT_ROOT="$(cd -- "${_SPILOT_PROJECT_ROOT}/../.." && pwd)"

export POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${_SPILOT_ROOT}/data}"
export TMAX_RUN_STATE_FILE="${TMAX_RUN_STATE_FILE:-${POLAR_DATA_ROOT}/runs/spilot_router_slime_grpo/current_run.env}"

# One 8-GPU learner node plus 56 independent TP1 Router rollout engines.
export NUM_NODES="${NUM_NODES:-8}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-56}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"

# One synchronous 8-prompt x 8-sample GRPO batch: 64 episodes/step.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export POLAR_FULLY_ASYNC="${POLAR_FULLY_ASYNC:-false}"
export POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-1}"
export POLAR_MAX_INIT_WORKERS="${POLAR_MAX_INIT_WORKERS:-64}"
export POLAR_MAX_RUN_WORKERS="${POLAR_MAX_RUN_WORKERS:-64}"
export POLAR_MAX_POSTRUN_WORKERS="${POLAR_MAX_POSTRUN_WORKERS:-64}"

# Slime's boundary is exclusive: iterations 0..199 are 200 optimizer steps.
export TMAX_NUM_ROLLOUT="${TMAX_NUM_ROLLOUT:-200}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export TMAX_AGENT_HARNESS="${TMAX_AGENT_HARNESS:-spilot_router}"
export POLAR_AGENT_HARNESS="${TMAX_AGENT_HARNESS}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-spilot-router-qwen35-9b-8n-200step}"

export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS="${TMAX_TRAIN_AGENT_TIMEOUT_SECONDS:-3300}"
export POLAR_TASK_TIMEOUT_FLOOR_SECONDS="${POLAR_TASK_TIMEOUT_FLOOR_SECONDS:-4500}"
export POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-5100}"
export TMAX_GRACEFUL_EXIT_BUFFER_SECONDS="${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS:-4500}"

export TMAX_EVAL_MAX_TASKS="${TMAX_EVAL_MAX_TASKS:-32}"
export TMAX_EXTERNAL_EVAL_ENABLED="${TMAX_EXTERNAL_EVAL_ENABLED:-0}"
# Pin the deterministic ready subset; three train-window SIFs are currently
# unfinished, and TMax persists the resulting JSONL across watcher relaunches.
export TMAX_ONLY_READY="${TMAX_ONLY_READY:-1}"
export TMAX_REQUIRE_EXACT_TOTAL_TASKS="${TMAX_REQUIRE_EXACT_TOTAL_TASKS:-0}"

export POLAR_TRAIN_RUN_SCRIPT="${POLAR_TRAIN_RUN_SCRIPT:-${_SPILOT_ROUTER_DIR}/run.sh}"
export POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${_SPILOT_ROUTER_DIR}/polar_config.yaml}"
export TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${_SPILOT_ROUTER_DIR}/topology.yaml}"

unset _SPILOT_ROUTER_DIR _SPILOT_PROJECT_ROOT _SPILOT_ROOT
