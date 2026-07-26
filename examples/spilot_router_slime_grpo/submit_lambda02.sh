#!/usr/bin/env bash
# T2: cost-aware SPilot Router training (lambda=0.2), the second arm of the
# lambda sweep. Identical to the lambda=0 arm (same seed weights, data, order,
# batch geometry, optimizer, 4-node topology) except the three cost knobs.
# Cost calibration from the 2026-07-12 A3 accounting: real per-episode price
# ratio gpt-5.5:qwen3.6-27b is 8-23x (central 15x); lambda*C/N puts the cost
# term at 10-20% of the [0,1] reward scale for gpt-routed episodes.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"

export POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${SPILOT_ROOT}/data}"
export TMAX_RUN_STATE_FILE="${TMAX_RUN_STATE_FILE:-${POLAR_DATA_ROOT}/runs/spilot_router_slime_grpo/lambda02_current_run.env}"

# Per-model cost configuration (rendered into polar_config_cost.yaml).
export SPILOT_QWEN_COST_WEIGHT="${SPILOT_QWEN_COST_WEIGHT:-1.0}"
export SPILOT_GPT_COST_WEIGHT="${SPILOT_GPT_COST_WEIGHT:-15.0}"
export SPILOT_COST_PENALTY_LAMBDA="${SPILOT_COST_PENALTY_LAMBDA:-0.2}"
export SPILOT_COST_NORMALIZER="${SPILOT_COST_NORMALIZER:-30.0}"


# Validated 4-node topology from the lambda=0 arm (TOPOLOGY_CHANGE_20260710):
# learner 2x8 TP4/DP4 unchanged, 16 TP1 rollout engines, aggregate provider
# concurrency preserved at 4x64=256.
export NUM_NODES="${NUM_NODES:-4}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-16}"
export SPILOT_EPISODE_ADMISSION_ENABLED="${SPILOT_EPISODE_ADMISSION_ENABLED:-false}"
export SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY="${SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY:-64}"
export SPILOT_GPT_GATEWAY_MAX_CONCURRENCY="${SPILOT_GPT_GATEWAY_MAX_CONCURRENCY:-64}"
export POLAR_MAX_INIT_WORKERS="${POLAR_MAX_INIT_WORKERS:-48}"
export POLAR_MAX_RUN_WORKERS="${POLAR_MAX_RUN_WORKERS:-288}"
export POLAR_MAX_POSTRUN_WORKERS="${POLAR_MAX_POSTRUN_WORKERS:-192}"

# Behavior converged well before step 100 in the lambda=0 arm; 150 steps
# leaves margin for the cost-shaped objective to settle.
export TMAX_NUM_ROLLOUT="${TMAX_NUM_ROLLOUT:-150}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-spilot-router-qwen35-9b-4n32-l02-150step}"
export WANDB_GROUP="${WANDB_GROUP:-spilot-router-qwen35-9b-lambda-sweep}"

exec bash "${SCRIPT_DIR}/submit_slurm.sh" "$@"
