#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-skill2env-qwen35-4b-rubric-prm-smoke}"
export TMAX_NUM_ROLLOUT=1
export TMAX_TARGET_ITER=0
export SAVE_INTERVAL=1
# Do not let a one-update connectivity run resample indefinitely if its first
# four prompt groups happen to have constant aggregate rewards.
export TMAX_DYNAMIC_SAMPLING_FILTER_PATH=""
export WANDB_MODE="${WANDB_MODE:-offline}"
exec bash "${SCRIPT_DIR}/submit_slurm.sh"
