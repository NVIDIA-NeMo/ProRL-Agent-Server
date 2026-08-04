#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-skill2env-qwen35-4b-gpt53-rubric-prm-prefix-merging-4n-2a2r-100steps}"
export TMAX_RUN_STATE_FILE="${TMAX_RUN_STATE_FILE:-${PROJECT_ROOT}/local_data/runs/skill2env_slime_grpo/current_4node_2actor_2rollout_100steps.env}"

export PRM_MODEL="azure/openai/gpt-5.3-codex"
export POLAR_TRAJECTORY_BUILDER=prefix_merging

export ACTOR_NUM_NODES=2
export ACTOR_NUM_GPUS_PER_NODE=8
export ROLLOUT_NUM_GPUS=16
export ROLLOUT_NUM_GPUS_PER_ENGINE=1

export ROLLOUT_BATCH_SIZE=8
export N_SAMPLES_PER_PROMPT=8
export NUM_STEPS_PER_ROLLOUT=1
export GLOBAL_BATCH_SIZE=64
export EVAL_GLOBAL_BATCH_SIZE=64

# Persist every consumed trajectory locally for the corrected continuation so
# verifier and judge rewards can be audited against the exact optimizer step.
# Keep the large trajectory tables off W&B; scalar training metrics remain on.
export POLAR_ROLLOUT_EXAMPLE_INTERVAL=1
export POLAR_ROLLOUT_EXAMPLE_COUNT=64
export POLAR_ROLLOUT_EXAMPLES_WANDB=0

# Slime treats --num-rollout as an exclusive count and numbers checkpoints from
# zero, so 100 optimizer steps have final rollout/checkpoint index 99.
export SKILL2ENV_PRESERVE_NUM_ROLLOUT=1
export TMAX_NUM_ROLLOUT=100
export TMAX_TARGET_ITER=99

# batch has a four-hour hard limit. Leave the minimum supported 30-minute
# checkpoint margin; the watcher can resume atomically if 100 steps need more
# than one allocation.
export WALL_TIME=4:00:00
export TMAX_MIN_WALL_TIME=4:00:00
export TMAX_GRACEFUL_EXIT_BUFFER_SECONDS=1800
export TMAX_MIN_GRACEFUL_EXIT_BUFFER_SECONDS=1800

exec bash "${SCRIPT_DIR}/run_4node_full.sh"
