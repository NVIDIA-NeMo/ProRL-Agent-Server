#!/usr/bin/env bash
# One prompt, two trajectories, and one learner update on two interactive nodes.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

export NUM_NODES=2
export ACTOR_NUM_NODES=1
export PARTITION=interactive
export WALL_TIME=04:00:00
export TMAX_MIN_WALL_TIME=04:00:00
export POLAR_GATEWAY_COUNT_OVERRIDE=2
export ROLLOUT_BATCH_SIZE=1
export N_SAMPLES_PER_PROMPT=2
export GLOBAL_BATCH_SIZE=2
export TMAX_NUM_ROLLOUT=1
export TMAX_TARGET_ITER=0
export TMAX_OPEN_INSTRUCT_MAX_ROWS=1
export TMAX_TRAIN_DATA=/home/junlongl/nvr/data/training_data/tmax/tmax-15k-open-instruct/polar-controller-v3-smoke.jsonl
export TMAX_MIN_ASYNC_LEVEL=1
export POLAR_MAX_ASYNC_LEVEL=1
export TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU=1
export TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU=1
export TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU=1
export POLAR_MAX_INIT_WORKERS=2
export POLAR_MAX_RUN_WORKERS=2
export POLAR_MAX_POSTRUN_WORKERS=2
export TMAX_MIN_TRAIN_ROLLOUTS_PER_DP=1
export TMAX_DYNAMIC_SAMPLING_FILTER_PATH=
export TMAX_EVAL_ENABLED=0
export TMAX_REQUIRE_WANDB=1
export WANDB_MODE=online
export SAVE_INTERVAL=9999
export EXPERIMENT_NAME=controller-v3-smoke-2n
export WANDB_GROUP=controller-v3-smoke-2n

exec bash "${SCRIPT_DIR}/submit_slurm.sh" "$@"
