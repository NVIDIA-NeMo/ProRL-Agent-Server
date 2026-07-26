#!/usr/bin/env bash
# One-node reproduction of the formal 200-step trainer configuration.
#
# The passing integration smoke (sync, level 1, 24576 tokens, CPU-offload
# optimizer, no dynamic filter) structurally cannot catch the logprob-guard
# failure both formal attempts died on. This wrapper keeps the smoke's small
# scale but flips every trainer-side knob to the formal values so one cheap
# job discriminates the failure mechanism (see logprob_guard diagnostics).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"

export POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${SPILOT_ROOT}/data}"
export TMAX_RUN_STATE_FILE="${TMAX_RUN_STATE_FILE:-${POLAR_DATA_ROOT}/runs/spilot_router_slime_grpo/repro_async_run.env}"

export NUM_NODES="${NUM_NODES:-1}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
export ACTOR_TENSOR_MODEL_PARALLEL_SIZE="${ACTOR_TENSOR_MODEL_PARALLEL_SIZE:-4}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"

# Formal trainer knobs (the exact deltas from the passing smoke).
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-67584}"
export TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP="${TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP:-0}"
export TMAX_OPTIMIZER_CPU_OFFLOAD="${TMAX_OPTIMIZER_CPU_OFFLOAD:-0}"
export POLAR_FULLY_ASYNC="${POLAR_FULLY_ASYNC:-true}"
export POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-3}"
export TMAX_MIN_ASYNC_LEVEL="${TMAX_MIN_ASYNC_LEVEL:-3}"
export POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.5}"
export POLAR_EARLY_STOP_GRACE_SESSIONS="${POLAR_EARLY_STOP_GRACE_SESSIONS:-16}"
# Keep the formal dynamic sampling filter (do NOT blank it like the smoke).

# Small scale: 2 prompts x 32 samples, three optimizer steps so async
# prefetch (level 3) genuinely overlaps generation with training.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-32}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export EVAL_GLOBAL_BATCH_SIZE="${EVAL_GLOBAL_BATCH_SIZE:-64}"
export TMAX_NUM_ROLLOUT="${TMAX_NUM_ROLLOUT:-3}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"

export POLAR_MAX_INIT_WORKERS="${POLAR_MAX_INIT_WORKERS:-12}"
export POLAR_MAX_RUN_WORKERS="${POLAR_MAX_RUN_WORKERS:-72}"
export POLAR_MAX_POSTRUN_WORKERS="${POLAR_MAX_POSTRUN_WORKERS:-48}"

# Match the formal run's provider transport: admission disabled, 32/32.
export SPILOT_EPISODE_ADMISSION_ENABLED="${SPILOT_EPISODE_ADMISSION_ENABLED:-false}"

export TMAX_EVAL_ENABLED="${TMAX_EVAL_ENABLED:-0}"
export TMAX_TRAINING_EVAL_ENABLED="${TMAX_TRAINING_EVAL_ENABLED:-0}"
export TMAX_EXTERNAL_EVAL_ENABLED="${TMAX_EXTERNAL_EVAL_ENABLED:-0}"
export TMAX_MAX_TASKS="${TMAX_MAX_TASKS:-8}"
export TMAX_TOTAL_TASKS="${TMAX_TOTAL_TASKS:-8}"
export TMAX_REQUIRE_EXACT_TOTAL_TASKS="${TMAX_REQUIRE_EXACT_TOTAL_TASKS:-0}"
export TMAX_EXCLUDE_DATA=""

# 4-hour interactive/batch chunks (no backfill); the watcher resumes long
# runs from checkpoints, so short walls only bound one chunk.
export PARTITION="${PARTITION:-interactive}"
export WALL_TIME="${WALL_TIME:-04:00:00}"
export TMAX_MIN_WALL_TIME="${TMAX_MIN_WALL_TIME:-04:00:00}"
export TMAX_GRACEFUL_EXIT_BUFFER_SECONDS="${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS:-1800}"
export TMAX_MIN_GRACEFUL_EXIT_BUFFER_SECONDS="${TMAX_MIN_GRACEFUL_EXIT_BUFFER_SECONDS:-1800}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-spilot-router-repro-async-formal-knobs}"

exec bash "${SCRIPT_DIR}/submit_slurm.sh" "$@"
