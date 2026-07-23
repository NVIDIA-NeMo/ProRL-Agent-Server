#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-skill2env-qwen35-4b-rubric-prm-tool-output-4n-full}"
export TMAX_RUN_STATE_FILE="${TMAX_RUN_STATE_FILE:-${PROJECT_ROOT}/local_data/runs/skill2env_slime_grpo/current_4node_run.env}"
export NUM_NODES=4
export PARTITION="batch"
export WALL_TIME="${WALL_TIME:-4:00:00}"
export TMAX_MIN_WALL_TIME="${TMAX_MIN_WALL_TIME:-4:00:00}"
export ACTOR_NUM_NODES=1
export ACTOR_NUM_GPUS_PER_NODE=8
export ROLLOUT_NUM_GPUS=24
export ROLLOUT_BATCH_SIZE=8
export N_SAMPLES_PER_PROMPT=8
export GLOBAL_BATCH_SIZE=64
export EVAL_GLOBAL_BATCH_SIZE=64
export POLAR_MULTI_GATEWAY=1
export PRM_INCLUDE_TOOL_OUTPUTS=true
export PRM_TOOL_OUTPUT_MAX_CHARS="${PRM_TOOL_OUTPUT_MAX_CHARS:-12000}"
export PRM_MAX_TRACES_PER_CALL="${PRM_MAX_TRACES_PER_CALL:-32}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"

# No TMAX_NUM_ROLLOUT override: submit_slurm.sh derives the complete one-epoch
# schedule from all prepared Skill2Env tasks.
unset TMAX_NUM_ROLLOUT TMAX_TARGET_ITER

exec bash "${SCRIPT_DIR}/submit_slurm.sh"
