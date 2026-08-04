#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-skill2env-qwen35-4b-gpt53-rubric-prm-prefix-merging-4n-full}"
export TMAX_RUN_STATE_FILE="${TMAX_RUN_STATE_FILE:-${PROJECT_ROOT}/local_data/runs/skill2env_slime_grpo/current_4node_prefix_merging_run.env}"
export POLAR_TRAJECTORY_BUILDER=prefix_merging

exec bash "${SCRIPT_DIR}/run_4node_full.sh"
