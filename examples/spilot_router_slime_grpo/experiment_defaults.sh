#!/usr/bin/env bash
# Immutable-by-default contract shared by first submission and watcher bootstrap.

_SPILOT_ROUTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
_SPILOT_PROJECT_ROOT="$(cd -- "${_SPILOT_ROUTER_DIR}/../.." && pwd)"
_SPILOT_ROOT="$(cd -- "${_SPILOT_PROJECT_ROOT}/../.." && pwd)"

export POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${_SPILOT_ROOT}/data}"
export TMAX_RUN_STATE_FILE="${TMAX_RUN_STATE_FILE:-${POLAR_DATA_ROOT}/runs/spilot_router_slime_grpo/current_run.env}"
_SPILOT_REFERENCE_DATA_DIR="${POLAR_DATA_ROOT}/runs/tmax-14598r-14498t100h-20260701T011143Z"

# Controlled-comparison contract: keep the learner, rollout topology, and
# effective GRPO batch identical to the validated Qwen3.5-9B TMax reference
# run (tmax-8n64-qwen35-9b-lr1e6-b8n32-noeval-fresh-20260702T074934Z).
# Only the agent harness/builder/evaluator and remote candidate-model calls are
# Router-specific.  This prevents a topology or batch-size change from being
# mistaken for a routing gain.
export NUM_NODES="${NUM_NODES:-8}"
export PARTITION="${PARTITION:-backfill,batch}"
export SLURM_GPUS="${SLURM_GPUS:-8}"
export RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-8}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
export ACTOR_TENSOR_MODEL_PARALLEL_SIZE="${ACTOR_TENSOR_MODEL_PARALLEL_SIZE:-4}"
export TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL="${TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL:-0}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-48}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
export TMAX_REQUIRE_FULL_GPU_ALLOCATION="${TMAX_REQUIRE_FULL_GPU_ALLOCATION:-1}"

# Match the reference allocation rather than the earlier 1-GPU/node Router
# bring-up configuration.
export CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
export SLURM_STEP_CPUS_PER_TASK="${SLURM_STEP_CPUS_PER_TASK:-120}"

# Reference batch/admission contract: 8 prompts x 32 sessions = 256 episodes
# per optimizer step, with up to three policy versions in the async pipeline.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-32}"
export POLAR_FULLY_ASYNC="${POLAR_FULLY_ASYNC:-true}"
export TMAX_MIN_ASYNC_LEVEL="${TMAX_MIN_ASYNC_LEVEL:-3}"
export POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-3}"
export TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU="${TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU:-16}"
export TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU="${TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU:-12}"
export TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU="${TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU:-8}"
export POLAR_MAX_INIT_WORKERS="${POLAR_MAX_INIT_WORKERS:-96}"
export POLAR_MAX_RUN_WORKERS="${POLAR_MAX_RUN_WORKERS:-576}"
export POLAR_MAX_POSTRUN_WORKERS="${POLAR_MAX_POSTRUN_WORKERS:-384}"
export POLAR_APPTAINER_BROKER_START_CONCURRENCY="${POLAR_APPTAINER_BROKER_START_CONCURRENCY:-8}"

# Match the reference trainer's token capacity and loss reduction.  Router
# actions are shorter, but changing either value would change optimizer-scale
# semantics independently of the action space.
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-67584}"
export TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP="${TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP:-0}"
export CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"

# Also match the validated Qwen3.5-9B TMax recipe's vocabulary chunk size.
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-64}"

# The 16-GPU learner matches the reference run and fits full-precision Adam on
# GPU, so do not retain the CPU-offload workaround from the 4-GPU bring-up.
export TMAX_OPTIMIZER_CPU_OFFLOAD="${TMAX_OPTIMIZER_CPU_OFFLOAD:-0}"

# Slime's boundary is exclusive: iterations 0..199 are 200 optimizer steps.
export TMAX_NUM_ROLLOUT="${TMAX_NUM_ROLLOUT:-200}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL:-}"
export TMAX_AGENT_HARNESS="${TMAX_AGENT_HARNESS:-spilot_router}"
export POLAR_AGENT_HARNESS="${TMAX_AGENT_HARNESS}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-spilot-router-qwen35-9b-8n64-200step}"
export WANDB_GROUP="${WANDB_GROUP:-spilot-router-qwen35-9b-8n64}"

export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS="${TMAX_TRAIN_AGENT_TIMEOUT_SECONDS:-3300}"
export POLAR_TASK_TIMEOUT_FLOOR_SECONDS="${POLAR_TASK_TIMEOUT_FLOOR_SECONDS:-4500}"
export POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-5100}"
export TMAX_GRACEFUL_EXIT_BUFFER_SECONDS="${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS:-4500}"

# Reuse the exact reference train ordering and held-out task set.  Router cards
# and action instructions are injected by the harness, so the underlying TMax
# JSONL must not be regenerated or shuffled for this comparison.
export TMAX_TRAIN_DATA="${TMAX_TRAIN_DATA:-${_SPILOT_REFERENCE_DATA_DIR}/tmax-train.jsonl}"
export TMAX_TRAIN_START_INDEX="${TMAX_TRAIN_START_INDEX:-0}"
export TMAX_MAX_TASKS="${TMAX_MAX_TASKS:--1}"
export TMAX_TOTAL_TASKS="${TMAX_TOTAL_TASKS:-14601}"
# An explicitly empty value is reserved for targeted smoke runs that select a
# slice by index; an unset value keeps the formal reference exclusion set.
export TMAX_EXCLUDE_DATA="${TMAX_EXCLUDE_DATA-${_SPILOT_REFERENCE_DATA_DIR}/tmax_holdout-eval.jsonl}"
export TMAX_PREPARE_DATA="${TMAX_PREPARE_DATA:-0}"
export TMAX_VALIDATE_EXISTING_ASSETS="${TMAX_VALIDATE_EXISTING_ASSETS:-0}"
export TMAX_TRAIN_DATA_SHA256="${TMAX_TRAIN_DATA_SHA256:-96a1c5929de64516eecc8a7b7ae012ccb888a2d575f15e28b8806ae6804826c8}"

# Match the reference's 100-task holdout contract.  Baseline/final Router
# evaluation is launched outside optimizer training, so it cannot perturb the
# reference-compatible train schedule.
export TMAX_EVAL_ENABLED="${TMAX_EVAL_ENABLED:-1}"
export TMAX_TRAINING_EVAL_ENABLED="${TMAX_TRAINING_EVAL_ENABLED:-0}"
export TMAX_EVAL_SOURCE="${TMAX_EVAL_SOURCE:-tmax}"
export TMAX_EVAL_DATA="${TMAX_EVAL_DATA:-${_SPILOT_REFERENCE_DATA_DIR}/tmax_holdout-eval.jsonl}"
export TMAX_EVAL_START_INDEX="${TMAX_EVAL_START_INDEX:-900}"
export TMAX_EVAL_MAX_TASKS="${TMAX_EVAL_MAX_TASKS:-100}"
export TMAX_EVAL_DATASET_NAME="${TMAX_EVAL_DATASET_NAME:-tmax_holdout}"
export TMAX_EVAL_INTERVAL="${TMAX_EVAL_INTERVAL:-1000000}"
export TMAX_EVAL_SAMPLES_PER_PROMPT="${TMAX_EVAL_SAMPLES_PER_PROMPT:-1}"
export TMAX_EVAL_MIN_VALID_SAMPLES="${TMAX_EVAL_MIN_VALID_SAMPLES:-100}"
export TMAX_EVAL_TEMPERATURE="${TMAX_EVAL_TEMPERATURE:-0.2}"
export TMAX_EVAL_TOP_P="${TMAX_EVAL_TOP_P:-1.0}"
export TMAX_EVAL_MAX_RESPONSE_LEN="${TMAX_EVAL_MAX_RESPONSE_LEN:-16384}"
export TMAX_EVAL_DATA_SHA256="${TMAX_EVAL_DATA_SHA256:-b1fe3e3311c66370c62f73272198c557f6afd3774a456dd1008e35d0153cbec3}"
export TMAX_EVAL_BUNDLE_SHA256="${TMAX_EVAL_BUNDLE_SHA256:-b1fe3e3311c66370c62f73272198c557f6afd3774a456dd1008e35d0153cbec3}"
export TMAX_PREPARE_EVAL_DATA="${TMAX_PREPARE_EVAL_DATA:-0}"
export TMAX_EXTERNAL_EVAL_ENABLED="${TMAX_EXTERNAL_EVAL_ENABLED:-0}"
export TMAX_CONCURRENT_PRETRAIN_EVAL="${TMAX_CONCURRENT_PRETRAIN_EVAL:-0}"
export TMAX_ONLY_READY="${TMAX_ONLY_READY:-1}"
export TMAX_REQUIRE_EXACT_TOTAL_TASKS="${TMAX_REQUIRE_EXACT_TOTAL_TASKS:-1}"

export POLAR_TRAIN_RUN_SCRIPT="${POLAR_TRAIN_RUN_SCRIPT:-${_SPILOT_ROUTER_DIR}/run.sh}"
export POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${_SPILOT_ROUTER_DIR}/polar_config.yaml}"
export TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${_SPILOT_ROUTER_DIR}/topology.yaml}"

unset _SPILOT_ROUTER_DIR _SPILOT_PROJECT_ROOT _SPILOT_ROOT _SPILOT_REFERENCE_DATA_DIR
