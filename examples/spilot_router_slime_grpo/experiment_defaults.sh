#!/usr/bin/env bash
# Immutable-by-default contract shared by first submission and watcher bootstrap.

_SPILOT_ROUTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
_SPILOT_PROJECT_ROOT="$(cd -- "${_SPILOT_ROUTER_DIR}/../.." && pwd)"
_SPILOT_ROOT="$(cd -- "${_SPILOT_PROJECT_ROOT}/../.." && pwd)"

export POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${_SPILOT_ROOT}/data}"
export TMAX_RUN_STATE_FILE="${TMAX_RUN_STATE_FILE:-${POLAR_DATA_ROOT}/runs/spilot_router_slime_grpo/current_run.env}"
_SPILOT_REFERENCE_DATA_DIR="${POLAR_DATA_ROOT}/runs/tmax-14598r-14498t100h-20260701T011143Z"

# Controlled-comparison contract: keep the learner, GPU allocation, gateway
# worker pools, and effective GRPO batch identical to the validated Qwen3.5-9B
# TMax reference run
# (tmax-8n64-qwen35-9b-lr1e6-b8n32-noeval-fresh-20260702T074934Z).
# Candidate-provider pressure is bounded separately by full-episode admission;
# changing the gateway pools would otherwise confound Router quality with a
# rollout-throughput change.
export NUM_NODES="${NUM_NODES:-8}"
# Operator policy: no backfill. Run 4-hour batch chunks and let the
# checkpoint-aware watcher resume; the graceful-exit buffer below must stay
# well under one wall chunk.
export PARTITION="${PARTITION:-batch}"
export WALL_TIME="${WALL_TIME:-04:00:00}"
export TMAX_MIN_WALL_TIME="${TMAX_MIN_WALL_TIME:-04:00:00}"
# Router rollouts hold GPUs at ~0% SM utilization while the frozen pool
# executes on the remote endpoint and mini-SWE runs on CPU, which the
# OccupiedIdleGPUsJobReaper otherwise kills mid-rollout (it cancelled chunks
# 13663902/13671216). Declare the sanctioned exemption for the wall window.
# The reaper accepts only whitelisted reason values; free-form reasons are
# rejected as "Invalid format" and the job stays flagged. Use the sanctioned
# interactive reason verbatim (same as the operator's igpu helper).
if [ -z "${TMAX_SBATCH_COMMENT:-}" ]; then
    TMAX_SBATCH_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"interactive","description":"Interactive and debugging sessions"}}'
fi
export TMAX_SBATCH_COMMENT
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
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export EVAL_GLOBAL_BATCH_SIZE="${EVAL_GLOBAL_BATCH_SIZE:-256}"
export TMAX_OVERRIDE_OPT_PARAM_SCHEDULER="${TMAX_OVERRIDE_OPT_PARAM_SCHEDULER:-1}"
export POLAR_FULLY_ASYNC="${POLAR_FULLY_ASYNC:-true}"
export POLAR_MULTI_GATEWAY="${POLAR_MULTI_GATEWAY:-1}"
export TMAX_MIN_ASYNC_LEVEL="${TMAX_MIN_ASYNC_LEVEL:-3}"
export POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-3}"
export POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.5}"
export POLAR_EARLY_STOP_GRACE_SESSIONS="${POLAR_EARLY_STOP_GRACE_SESSIONS:-16}"
# This is a provider-availability gate, not a model-quality filter.  A frozen
# candidate that produces no usable completion in a sufficiently large cohort
# must stop the optimizer rather than teach the router that a transient backend
# outage is a genuine task outcome.
export POLAR_CANDIDATE_POOL_HEALTH_GATE_ENABLED="${POLAR_CANDIDATE_POOL_HEALTH_GATE_ENABLED:-true}"
export POLAR_CANDIDATE_POOL_HEALTH_MIN_OBSERVED_SESSIONS="${POLAR_CANDIDATE_POOL_HEALTH_MIN_OBSERVED_SESSIONS:-16}"
export POLAR_CANDIDATE_POOL_HEALTH_MIN_COMPLETION_FRACTION="${POLAR_CANDIDATE_POOL_HEALTH_MIN_COMPLETION_FRACTION:-0.1}"
export TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU="${TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU:-16}"
export TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU="${TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU:-12}"
export TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU="${TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU:-8}"
export POLAR_MAX_INIT_WORKERS="${POLAR_MAX_INIT_WORKERS:-96}"
export POLAR_MAX_RUN_WORKERS="${POLAR_MAX_RUN_WORKERS:-576}"
export POLAR_MAX_POSTRUN_WORKERS="${POLAR_MAX_POSTRUN_WORKERS:-384}"
export POLAR_APPTAINER_BROKER_START_CONCURRENCY="${POLAR_APPTAINER_BROKER_START_CONCURRENCY:-8}"

# The Router runner receives short-lived model-pool capabilities through
# ``exec_protected``.  That operation is intentionally unavailable in the
# legacy one-Apptainer-process-per-command backend used by ordinary direct
# Qwen training.  Keep this opt-in here, rather than changing the shared TMax
# default, so direct Qwen3.5 experiments retain their existing runtime mode.
export POLAR_APPTAINER_PERSISTENT_BROKER="${POLAR_APPTAINER_PERSISTENT_BROKER:-1}"
if [ "${POLAR_APPTAINER_PERSISTENT_BROKER}" != 1 ]; then
    echo "ERROR: SPilot Router requires POLAR_APPTAINER_PERSISTENT_BROKER=1" >&2
    return 1 2>/dev/null || exit 1
fi

# The protected broker and host-supplied read-only mini-SWE runtime are only a
# security boundary when task images cannot inherit host mounts/environment or
# share the host PID/IPC namespaces.  Default missing values, but never repair
# an explicit/inherited opt-out: a formal SPilot run must fail closed instead.
for _spilot_isolation_name in \
    POLAR_APPTAINER_NO_MOUNT_HOSTFS \
    POLAR_APPTAINER_NO_MOUNT_TMP \
    POLAR_APPTAINER_ISOLATE_PID \
    POLAR_APPTAINER_ISOLATE_IPC \
    POLAR_APPTAINER_CLEANENV; do
    _spilot_isolation_value="${!_spilot_isolation_name:-1}"
    export "${_spilot_isolation_name}=${_spilot_isolation_value}"
    if [ "${_spilot_isolation_value}" != 1 ]; then
        echo "ERROR: formal SPilot requires ${_spilot_isolation_name}=1 (got ${_spilot_isolation_value})" >&2
        return 1 2>/dev/null || exit 1
    fi
done
unset _spilot_isolation_name _spilot_isolation_value

# Full candidate episodes, rather than individual HTTP requests, are the scarce
# resource.  These totals are aggregate across the eight gateway processes and
# are divided exactly below.  Persist both configured totals and effective
# local/effective totals so a resumed run cannot silently change its provider
# pressure.
export SPILOT_EPISODE_ADMISSION_ENABLED="${SPILOT_EPISODE_ADMISSION_ENABLED:-true}"
export TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT="${TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT:-0}"
if ! [[ "${TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT must be a non-negative integer" >&2
    return 1 2>/dev/null || exit 1
fi
export SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT="${SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT:-${NUM_NODES}}"
if ! [[ "${SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT must be a positive integer" >&2
    return 1 2>/dev/null || exit 1
fi
case "${SPILOT_EPISODE_ADMISSION_ENABLED}" in
    true)
        export SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS="${SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS:-14400}"
        export SPILOT_QWEN_MAX_ACTIVE_EPISODES="${SPILOT_QWEN_MAX_ACTIVE_EPISODES:-8}"
        export SPILOT_GPT_MAX_ACTIVE_EPISODES="${SPILOT_GPT_MAX_ACTIVE_EPISODES:-32}"
        if ! [[ "${SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS}" =~ ^[1-9][0-9]*$ ]] || \
           [ "${SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS}" -gt 86400 ]; then
            echo "ERROR: SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS must be in [1, 86400]" >&2
            return 1 2>/dev/null || exit 1
        fi
        for _spilot_total_name in \
            SPILOT_QWEN_MAX_ACTIVE_EPISODES SPILOT_GPT_MAX_ACTIVE_EPISODES; do
            _spilot_total_value="${!_spilot_total_name}"
            if ! [[ "${_spilot_total_value}" =~ ^[1-9][0-9]*$ ]] || \
               [ $((_spilot_total_value % SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT)) -ne 0 ]; then
                echo "ERROR: ${_spilot_total_name} must be positive and divisible by SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT=${SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT}" >&2
                return 1 2>/dev/null || exit 1
            fi
        done
        _spilot_qwen_local="$((SPILOT_QWEN_MAX_ACTIVE_EPISODES / SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT))"
        _spilot_gpt_local="$((SPILOT_GPT_MAX_ACTIVE_EPISODES / SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT))"
        export SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES="${SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES:-${_spilot_qwen_local}}"
        export SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES="${SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES:-${_spilot_gpt_local}}"
        export SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY="${SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY:-${_spilot_qwen_local}}"
        export SPILOT_GPT_GATEWAY_MAX_CONCURRENCY="${SPILOT_GPT_GATEWAY_MAX_CONCURRENCY:-${_spilot_gpt_local}}"
        export SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES="${SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES:-$((_spilot_qwen_local * SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT))}"
        export SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES="${SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES:-$((_spilot_gpt_local * SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT))}"
        ;;
    false)
        export SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS="${SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS:-0}"
        export SPILOT_QWEN_MAX_ACTIVE_EPISODES="${SPILOT_QWEN_MAX_ACTIVE_EPISODES:-0}"
        export SPILOT_GPT_MAX_ACTIVE_EPISODES="${SPILOT_GPT_MAX_ACTIVE_EPISODES:-0}"
        export SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES="${SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES:-null}"
        export SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES="${SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES:-null}"
        export SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY="${SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY:-32}"
        export SPILOT_GPT_GATEWAY_MAX_CONCURRENCY="${SPILOT_GPT_GATEWAY_MAX_CONCURRENCY:-32}"
        export SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES="${SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES:-0}"
        export SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES="${SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES:-0}"
        ;;
    *)
        echo "ERROR: SPILOT_EPISODE_ADMISSION_ENABLED must be true or false" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
unset _spilot_total_name _spilot_total_value _spilot_qwen_local _spilot_gpt_local

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
# Preserve the baseline's every-five-step synchronous recovery cadence, but
# retain only the latest periodic checkpoint by default.  Slime passes the
# zero-based completed iteration to Megatron, so this cadence writes 4, 9, 14,
# ..., 199; each previous non-multiple of five is pruned on the next save.  The
# current user quota has less free space than the reference run's 9.2-TiB
# checkpoint set; retention changes storage history only, not rollout or
# optimizer semantics.
export SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL:-5}"
export TMAX_AGENT_HARNESS="${TMAX_AGENT_HARNESS:-spilot_router}"
export POLAR_AGENT_HARNESS="${TMAX_AGENT_HARNESS}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-spilot-router-qwen35-9b-8n64-200step}"
export WANDB_GROUP="${WANDB_GROUP:-spilot-router-qwen35-9b-8n64}"

if [ "${SPILOT_EPISODE_ADMISSION_ENABLED}" = "true" ]; then
    # Q=14,400 seconds is shared by both candidate calls. The portable runner
    # keeps total_timeout_seconds=3,000 and credits only measured admission
    # wait. The outer agent envelope is base 3,300 + Q. Task/request retain a
    # second full agent envelope as infrastructure headroom. These bounds are
    # deliberately independent of gateway worker counts and provider caps:
    #   agent   = 3,300 + Q                 = 17,700
    #   task    = 4,500 + Q + agent envelope = 36,600
    #   request = 5,100 + Q + agent envelope = 37,200
    _spilot_agent_timeout="$((3300 + SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS))"
    _spilot_task_timeout="$((4500 + SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS + _spilot_agent_timeout))"
    _spilot_request_timeout="$((5100 + SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS + _spilot_agent_timeout))"
else
    _spilot_agent_timeout=3300
    _spilot_task_timeout=4500
    _spilot_request_timeout=5100
fi
export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS="${TMAX_TRAIN_AGENT_TIMEOUT_SECONDS:-${_spilot_agent_timeout}}"
export POLAR_TASK_TIMEOUT_FLOOR_SECONDS="${POLAR_TASK_TIMEOUT_FLOOR_SECONDS:-${_spilot_task_timeout}}"
export POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-${_spilot_request_timeout}}"
export TMAX_GRACEFUL_EXIT_BUFFER_SECONDS="${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS:-1800}"
export TMAX_MIN_GRACEFUL_EXIT_BUFFER_SECONDS="${TMAX_MIN_GRACEFUL_EXIT_BUFFER_SECONDS:-1800}"
unset _spilot_agent_timeout _spilot_task_timeout _spilot_request_timeout

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

# Keep holdout metadata available for separate baseline/final evaluation, but
# match the reference run's no-eval training lifecycle exactly.  Candidate
# strength and final Router quality are measured by the guarded forced-route
# evaluator outside optimizer training.
export TMAX_EVAL_ENABLED="${TMAX_EVAL_ENABLED:-1}"
export TMAX_TRAINING_EVAL_ENABLED="${TMAX_TRAINING_EVAL_ENABLED:-0}"
export TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN="${TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN:-0}"
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
export TMAX_SUBMIT_SCRIPT="${TMAX_SUBMIT_SCRIPT:-${_SPILOT_ROUTER_DIR}/submit_slurm.sh}"
export POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${_SPILOT_ROUTER_DIR}/polar_config.yaml}"
export TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${_SPILOT_ROUTER_DIR}/topology.yaml}"

unset _SPILOT_ROUTER_DIR _SPILOT_PROJECT_ROOT _SPILOT_ROOT _SPILOT_REFERENCE_DATA_DIR

# Per-model cost contract rendered into polar_config.yaml. Defaults preserve
# the lambda=0 arm exactly; submit_lambda02.sh overrides for the cost-aware
# arm and run state serializes whatever a logical run was started with.
export SPILOT_QWEN_COST_WEIGHT="${SPILOT_QWEN_COST_WEIGHT:-1.0}"
export SPILOT_GPT_COST_WEIGHT="${SPILOT_GPT_COST_WEIGHT:-1.0}"
export SPILOT_COST_PENALTY_LAMBDA="${SPILOT_COST_PENALTY_LAMBDA:-0.0}"
export SPILOT_COST_NORMALIZER="${SPILOT_COST_NORMALIZER:-1.0}"
# Latency shaping (paper lambda_ell term). Disabled by default; normalizer in
# seconds (1800 s = a 30-minute session takes the full unit penalty at lambda=1).
export SPILOT_LATENCY_PENALTY_LAMBDA="${SPILOT_LATENCY_PENALTY_LAMBDA:-0.0}"
export SPILOT_LATENCY_NORMALIZER="${SPILOT_LATENCY_NORMALIZER:-1800}"

# Candidate labelling: "real_names" (default since 2026-07-21) presents
# candidates under their actual pool model names; "anonymous" is the
# historical M0/M1 protocol (the TB2.1 decision probes proved the trained
# policy anchors on the literal "M0" token instead of task semantics) and is
# only for resuming pre-switch lineages.
export SPILOT_SLOT_LABEL_MODE="${SPILOT_SLOT_LABEL_MODE:-real_names}"

# Routing granularity: "task_level" = one ROUTE assigns the whole attempt plus
# an optional final VERIFY (the historical protocol, bounded by
# SPILOT_MAX_POOL_CALLS); "turn_level" = per-STEP routing: one turn is one
# step() — a single pool-model completion plus the execution of the one bash
# action it emitted, inside a shared Vanillux2 conversation — and the router
# re-decides (ROUTE any candidate or SUBMIT) after every step, bounded by the
# pool_step_limit (64). Budget exhaustion always auto-submits the workspace.
#
# COST CALIBRATION WARNING for turn_level lanes: total_cost sums the routed
# candidate's cost_weight PER STEP, so an all-gpt 64-step episode costs up to
# 64 x SPILOT_GPT_COST_WEIGHT (=960 at the lambda02 weights) instead of the
# task_level maximum of 2 calls (=30). Re-derive SPILOT_COST_NORMALIZER for
# turn_level (e.g. scale by the expected step count) or the cost penalty
# saturates to the reward floor on every gpt-heavy success.
export SPILOT_ROUTING_MODE="${SPILOT_ROUTING_MODE:-task_level}"
export SPILOT_MAX_POOL_CALLS="${SPILOT_MAX_POOL_CALLS:-2}"
# turn_level only: how the shared conversation is presented to a newly routed
# candidate.  "shared" (default) keeps the historical raw transcript;
# "switch_notice" appends one attribution notice per model switch;
# "model_tagged" prefixes every assistant turn with the producing slot label;
# "reset_context" collapses history into a bounded executed-step digest at
# each switch.  Non-shared values are rejected under task_level.
export SPILOT_CONTEXT_HANDOFF="${SPILOT_CONTEXT_HANDOFF:-shared}"
# Per-step digest budget for the Router's own trajectory under turn_level
# (chars of executed-step output shown to the Router between decisions; the
# executing pool models still see full Vanillux2 observations).
export SPILOT_ROUTER_OBS_MAX_CHARS="${SPILOT_ROUTER_OBS_MAX_CHARS:-1500}"
