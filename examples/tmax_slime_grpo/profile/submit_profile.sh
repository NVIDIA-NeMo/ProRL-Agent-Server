#!/usr/bin/env bash
# Submit short, isolated TMax mini-swe-agent system-profiling arms.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TMAX_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd -- "${TMAX_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
SUBMIT_SCRIPT="${TMAX_DIR}/submit_slurm.sh"

DEFAULT_ARMS=(
    async-16t16r-l4
    async-8t24r-l4
    async-8t32r-l4
    collocate-32shared
)

usage() {
    cat <<'EOF'
Usage:
  submit_profile.sh list
  submit_profile.sh [plan] [ARM ...]
  submit_profile.sh submit ARM [ARM ...]

Arms:
  async-16t16r-l4    4x8 GPUs; 16 train + 16 rollout; async depth 4
  async-8t24r-l4     4x8 GPUs;  8 train + 24 rollout; async depth 4
  async-8t32r-l4     5x8 GPUs;  8 train + 32 rollout; async depth 4
  async-4t28r-l4     4x8 GPUs;  4 train + 28 rollout; async depth 4
  collocate-32shared 4x8 GPUs; 32 train / 32 rollout on shared GPUs

Useful environment overrides:
  PROFILE_STEPS=3                 optimizer steps per arm
  PROFILE_REPEATS=1               repetitions per arm
  PROFILE_ID=<UTC stamp>          common comparison batch id
  PROFILE_LOAD_DIR=/abs/ckpt      identical release or numbered seed
  PROFILE_TRAIN_DATA=/abs/file    identical prompt JSONL for every arm
  PROFILE_HF_CHECKPOINT=/abs/hf   Qwen3.5-9B HF assets
  PROFILE_REF_LOAD=/abs/ckpt      Qwen3.5-9B reference release
  PROFILE_TORCH_DIST_DIR=/abs/ckpt
  PROFILE_MODEL_ARGS_FILE=/abs/model_args.sh
  PROFILE_AGENT_MODEL_NAME=Qwen/Qwen3.5-9B
  PROFILE_PARTITION=backfill,batch
  PROFILE_WALL_TIME=04:00:00
  PROFILE_AFTER_JOB_ID=12345      chain first arm after another suite
  PROFILE_DEPENDENCY_KIND=afterany

The profile uses the TMax mini-swe-agent harness, disables evaluation,
graceful resume, and checkpoint writes, and keeps GPU telemetry enabled.
EOF
}

configure_arm() {
    local arm="$1"
    MODE=fully_async
    FULLY_ASYNC=true
    NUM_NODES=4
    GPUS_PER_NODE=8
    TP=4
    ROLLOUT_TP=1
    ASYNC_LEVEL=4
    case "${arm}" in
        async-16t16r-l4)
            ACTOR_NODES=2
            ACTOR_GPUS_PER_NODE=8
            ROLLOUT_GPUS=16
            ;;
        async-8t24r-l4)
            ACTOR_NODES=1
            ACTOR_GPUS_PER_NODE=8
            ROLLOUT_GPUS=24
            ;;
        async-8t32r-l4)
            NUM_NODES=5
            ACTOR_NODES=1
            ACTOR_GPUS_PER_NODE=8
            ROLLOUT_GPUS=32
            ;;
        async-4t28r-l4)
            ACTOR_NODES=1
            ACTOR_GPUS_PER_NODE=4
            ROLLOUT_GPUS=28
            ;;
        collocate-32shared|collocate-32)
            MODE=colocate
            FULLY_ASYNC=false
            ASYNC_LEVEL=1
            ACTOR_NODES=4
            ACTOR_GPUS_PER_NODE=8
            ROLLOUT_GPUS=32
            ;;
        *)
            echo "ERROR: unknown profile arm: ${arm}" >&2
            usage >&2
            return 1
            ;;
    esac
    ACTOR_GPUS=$((ACTOR_NODES * ACTOR_GPUS_PER_NODE))
    ALLOCATED_GPUS=$((NUM_NODES * GPUS_PER_NODE))
}

canonical_arm_name() {
    case "$1" in
        collocate-32) printf '%s\n' collocate-32shared ;;
        *) printf '%s\n' "$1" ;;
    esac
}

ACTION="${1:-plan}"
case "${ACTION}" in
    list)
        usage
        exit 0
        ;;
    plan|submit)
        shift || true
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        ACTION=plan
        ;;
esac

if [ "$#" -gt 0 ]; then
    ARMS=("$@")
else
    ARMS=("${DEFAULT_ARMS[@]}")
fi

PROFILE_STEPS="${PROFILE_STEPS:-3}"
PROFILE_REPEATS="${PROFILE_REPEATS:-1}"
PROFILE_ID="${PROFILE_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE_PARTITION="${PROFILE_PARTITION:-backfill,batch}"
PROFILE_WALL_TIME="${PROFILE_WALL_TIME:-04:00:00}"
PROFILE_DEPENDENCY_KIND="${PROFILE_DEPENDENCY_KIND:-afterany}"
POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${WORKSPACE_ROOT}/data}"
export POLAR_DATA_ROOT

# Resolve the model contract once, outside the arm loop.  In particular, do
# not let a caller's TMax/Qwen4 environment silently change one arm in this
# controlled Qwen3.5-9B comparison.  PROFILE_* overrides remain available for
# intentionally profiling another, internally consistent 9B asset location.
PROFILE_HF_CHECKPOINT="${PROFILE_HF_CHECKPOINT:-${POLAR_DATA_ROOT}/checkpoints/Qwen3.5-9B}"
PROFILE_REF_LOAD="${PROFILE_REF_LOAD:-${POLAR_DATA_ROOT}/checkpoints/Qwen3.5-9B_torch_dist}"
PROFILE_TORCH_DIST_DIR="${PROFILE_TORCH_DIST_DIR:-${PROFILE_REF_LOAD}}"
PROFILE_MODEL_ARGS_FILE="${PROFILE_MODEL_ARGS_FILE:-${TMAX_DIR}/model_args.sh}"
PROFILE_AGENT_MODEL_NAME="${PROFILE_AGENT_MODEL_NAME:-Qwen/Qwen3.5-9B}"

for value_name in PROFILE_STEPS PROFILE_REPEATS; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${value_name} must be a positive integer, got ${value}" >&2
        exit 1
    fi
done
if ! [[ "${PROFILE_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: PROFILE_ID may contain only letters, digits, dot, underscore, and dash" >&2
    exit 1
fi
case "${PROFILE_DEPENDENCY_KIND}" in
    afterok|afterany) ;;
    *)
        echo "ERROR: PROFILE_DEPENDENCY_KIND must be afterok or afterany" >&2
        exit 1
        ;;
esac
if [ -n "${PROFILE_AFTER_JOB_ID:-}" ] && \
   ! [[ "${PROFILE_AFTER_JOB_ID}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: PROFILE_AFTER_JOB_ID must be a numeric Slurm job id" >&2
    exit 1
fi
if [ "${ACTION}" = submit ]; then
    if [ -z "${PROFILE_LOAD_DIR:-}" ]; then
        echo "ERROR: submit requires PROFILE_LOAD_DIR as an existing absolute checkpoint" >&2
        exit 1
    fi
    if [ -z "${PROFILE_TRAIN_DATA:-}" ]; then
        echo "ERROR: submit requires PROFILE_TRAIN_DATA as an existing non-empty absolute file" >&2
        exit 1
    fi
fi
if [ -n "${PROFILE_TRAIN_DATA:-}" ]; then
    if [[ "${PROFILE_TRAIN_DATA}" != /* ]] || \
       [ ! -f "${PROFILE_TRAIN_DATA}" ] || [ ! -s "${PROFILE_TRAIN_DATA}" ]; then
        echo "ERROR: PROFILE_TRAIN_DATA must be an existing non-empty absolute file" >&2
        exit 1
    fi
fi

# A numbered Megatron checkpoint resumes at N+1; --num-rollout is exclusive.
PROFILE_NUM_ROLLOUT="${PROFILE_STEPS}"
if [ -n "${PROFILE_LOAD_DIR:-}" ]; then
    if [[ "${PROFILE_LOAD_DIR}" != /* ]] || [ ! -d "${PROFILE_LOAD_DIR}" ] || \
       [ ! -s "${PROFILE_LOAD_DIR}/latest_checkpointed_iteration.txt" ]; then
        echo "ERROR: PROFILE_LOAD_DIR must be an absolute checkpoint with a tracker" >&2
        exit 1
    fi
    seed_tracker="$(tr -d '[:space:]' <"${PROFILE_LOAD_DIR}/latest_checkpointed_iteration.txt")"
    if [[ "${seed_tracker}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        if [ -z "${PROFILE_TRAIN_DATA:-}" ]; then
            echo "ERROR: numbered PROFILE_LOAD_DIR requires the exact PROFILE_TRAIN_DATA" >&2
            exit 1
        fi
        PROFILE_NUM_ROLLOUT="$((seed_tracker + 1 + PROFILE_STEPS))"
    elif [ "${seed_tracker}" != release ]; then
        echo "ERROR: unsupported checkpoint tracker ${seed_tracker@Q}" >&2
        exit 1
    fi
fi

declare -A seen_arms=()
for arm in "${ARMS[@]}"; do
    configure_arm "${arm}"
    canonical_arm="$(canonical_arm_name "${arm}")"
    if [ -n "${seen_arms[${canonical_arm}]:-}" ]; then
        echo "ERROR: duplicate profile arm ${canonical_arm}; use PROFILE_REPEATS" >&2
        exit 1
    fi
    seen_arms["${canonical_arm}"]=1
done
unset seen_arms

printf 'action=%s profile_id=%s steps=%s num_rollout=%s repeats=%s dependency=%s harness=mini_swe_agent\n' \
    "${ACTION}" "${PROFILE_ID}" "${PROFILE_STEPS}" "${PROFILE_NUM_ROLLOUT}" \
    "${PROFILE_REPEATS}" "${PROFILE_DEPENDENCY_KIND}"
printf '%-23s %-12s %7s %7s %5s %8s %8s %s\n' \
    ARM MODE TRAIN ROLLOUT DEPTH NODESxGPU TOTAL_GPU RUN_ID

previous_job_id="${PROFILE_AFTER_JOB_ID:-}"
manifest_dir="${POLAR_DATA_ROOT}/runs/tmax_slime_grpo/profile/${PROFILE_ID}"
manifest_file="${manifest_dir}/manifest.tsv"
if [ "${ACTION}" = submit ]; then
    if [ -e "${manifest_file}" ]; then
        echo "ERROR: refusing to reuse profile batch; choose a new PROFILE_ID: ${manifest_file}" >&2
        exit 1
    fi
    for repeat in $(seq 1 "${PROFILE_REPEATS}"); do
        for arm in "${ARMS[@]}"; do
            canonical_arm="$(canonical_arm_name "${arm}")"
            state_file="${POLAR_DATA_ROOT}/runs/tmax-prof-${canonical_arm}-r${repeat}-${PROFILE_ID}/run_state.env"
            if [ -e "${state_file}" ]; then
                echo "ERROR: refusing to reuse profile state: ${state_file}" >&2
                exit 1
            fi
        done
    done
    mkdir -p "${manifest_dir}"
    printf 'run_id\tarm\trepeat\tmode\tharness\tactor_gpus\trollout_gpus\tasync_level\tallocated_gpus\tjob_id\n' >"${manifest_file}"
fi

for repeat in $(seq 1 "${PROFILE_REPEATS}"); do
    for arm in "${ARMS[@]}"; do
        configure_arm "${arm}"
        canonical_arm="$(canonical_arm_name "${arm}")"
        run_id="tmax-prof-${canonical_arm}-r${repeat}-${PROFILE_ID}"
        receipt_file="${POLAR_DATA_ROOT}/runs/${run_id}/submit/last_submission.env"
        state_file="${POLAR_DATA_ROOT}/runs/${run_id}/run_state.env"
        printf '%-23s %-12s %7d %7d %5d %4dx%-3d %8d %s\n' \
            "${canonical_arm}" "${MODE}" "${ACTOR_GPUS}" "${ROLLOUT_GPUS}" \
            "${ASYNC_LEVEL}" "${NUM_NODES}" "${GPUS_PER_NODE}" \
            "${ALLOCATED_GPUS}" "${run_id}"

        if [ "${ACTION}" != submit ]; then
            continue
        fi
        dependency=""
        if [ -n "${previous_job_id}" ]; then
            dependency="${PROFILE_DEPENDENCY_KIND}:${previous_job_id}"
        fi
        (
            unset RUN_ID JOB_NAME EXPERIMENT_NAME SAVE_DIR RUN_DIR LOAD_DIR
            unset HF_CHECKPOINT REF_LOAD TORCH_DIST_DIR MODEL_ARGS_FILE
            unset POLAR_AGENT_MODEL_NAME TMAX_TRAIN_DATA PROMPT_DATA
            unset WANDB_RUN_ID TMAX_TARGET_ITER TMAX_LAST_JOB_ID
            unset TMAX_RUN_STATE_FILE TMAX_SUBMIT_RECEIPT_FILE SBATCH_DEPENDENCY
            unset POLAR_MAX_INIT_WORKERS POLAR_MAX_RUN_WORKERS POLAR_MAX_POSTRUN_WORKERS
            unset TMAX_TRAIN_DATA_SHA256 TMAX_EVAL_DATA_SHA256
            unset TMAX_EXTERNAL_EVAL_DATA_SHA256 TMAX_EVAL_BUNDLE_SHA256

            export RUN_ID="${run_id}"
            export JOB_NAME="tm-prof-${canonical_arm}-r${repeat}"
            export EXPERIMENT_NAME="tmax-profile-${canonical_arm}"
            export WANDB_RUN_ID="${run_id}"
            export WANDB_GROUP="tmax-system-profile-${PROFILE_ID}"
            export WANDB_MODE="${PROFILE_WANDB_MODE:-offline}"
            export TMAX_REQUIRE_WANDB="${PROFILE_REQUIRE_WANDB:-0}"
            export TMAX_RUN_STATE_FILE="${state_file}"
            export TMAX_SUBMIT_RECEIPT_FILE="${receipt_file}"
            export TMAX_PROFILE_BATCH_ID="${PROFILE_ID}"
            export TMAX_PROFILE_ARM="${canonical_arm}"
            export TMAX_AGENT_HARNESS=mini_swe_agent
            export POLAR_AGENT_HARNESS=mini_swe_agent
            export TMAX_TRAIN_MODE="${MODE}"
            export TMAX_PROFILE_DISABLE_CHECKPOINT=1
            export TMAX_PERSIST_RUN_STATE=1
            export SUBMIT_BACKEND=sbatch
            export SUBMIT_DRY_RUN=0
            export TMAX_ENABLE_GRACEFUL_EXIT=0
            export TMAX_NUM_ROLLOUT="${PROFILE_NUM_ROLLOUT}"

            # Always replace inherited model/data lineage with the immutable
            # profile contract resolved above.  These assignments are
            # deliberately unconditional for every arm and repeat.
            export HF_CHECKPOINT="${PROFILE_HF_CHECKPOINT}"
            export REF_LOAD="${PROFILE_REF_LOAD}"
            export TORCH_DIST_DIR="${PROFILE_TORCH_DIST_DIR}"
            export MODEL_ARGS_FILE="${PROFILE_MODEL_ARGS_FILE}"
            export POLAR_AGENT_MODEL_NAME="${PROFILE_AGENT_MODEL_NAME}"
            export LOAD_DIR="${PROFILE_LOAD_DIR}"
            export TMAX_TRAIN_DATA="${PROFILE_TRAIN_DATA}"
            export PROMPT_DATA="${PROFILE_TRAIN_DATA}"

            export NUM_NODES="${NUM_NODES}"
            export SLURM_GPUS="${GPUS_PER_NODE}"
            export RAY_NUM_GPUS_PER_NODE="${GPUS_PER_NODE}"
            export ACTOR_NUM_NODES="${ACTOR_NODES}"
            export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE}"
            export ACTOR_TENSOR_MODEL_PARALLEL_SIZE="${TP}"
            export CONTEXT_PARALLEL_SIZE=1
            export ROLLOUT_NUM_GPUS="${ROLLOUT_GPUS}"
            export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_TP}"
            export TMAX_REQUIRE_FULL_GPU_ALLOCATION=1

            export POLAR_FULLY_ASYNC="${FULLY_ASYNC}"
            export TMAX_MIN_ASYNC_LEVEL="${ASYNC_LEVEL}"
            export POLAR_MAX_ASYNC_LEVEL="${ASYNC_LEVEL}"
            export ROLLOUT_BATCH_SIZE=8
            export N_SAMPLES_PER_PROMPT=32
            export NUM_STEPS_PER_ROLLOUT=1
            export GLOBAL_BATCH_SIZE=256
            export EVAL_GLOBAL_BATCH_SIZE=256

            # Preserve the historical TMax single-gateway execution path and
            # scale CPU worker capacity with the rollout GPU count.
            export POLAR_MULTI_GATEWAY=0
            export POLAR_GATEWAY_COUNT_OVERRIDE=1
            export TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU=16
            export TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU=16
            export TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU=8
            export POLAR_MAX_INIT_WORKERS="$((ROLLOUT_GPUS * 2))"
            export POLAR_MAX_RUN_WORKERS="$((ROLLOUT_GPUS * 16))"
            export POLAR_MAX_POSTRUN_WORKERS="$((ROLLOUT_GPUS * 8))"
            export POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0
            export POLAR_EARLY_STOP_GRACE_SESSIONS=0

            export TMAX_EVAL_ENABLED=0
            export TMAX_TRAINING_EVAL_ENABLED=0
            export TMAX_EXTERNAL_EVAL_ENABLED=0
            export TMAX_CONCURRENT_PRETRAIN_EVAL=0
            export TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN=0
            export TMAX_VALIDATE_EXISTING_ASSETS="${PROFILE_VALIDATE_ASSETS:-0}"
            export TMAX_PREPARE_DATA=0
            export SAVE_HF_ENABLED=0
            export SAVE_MEGATRON=1
            export POLAR_ROLLOUT_EXAMPLE_INTERVAL=1000000
            export POLAR_ROLLOUT_EXAMPLE_COUNT=1
            export POLAR_ROLLOUT_EXAMPLES_WANDB=0
            export GPU_MONITOR_ENABLED=1

            export PARTITION="${PROFILE_PARTITION}"
            export WALL_TIME="${PROFILE_WALL_TIME}"
            export TMAX_MIN_WALL_TIME="${PROFILE_WALL_TIME}"
            export CPUS_PER_TASK=128
            export SLURM_STEP_CPUS_PER_TASK=120
            if [ -n "${dependency}" ]; then
                export SBATCH_DEPENDENCY="${dependency}"
            fi
            bash "${SUBMIT_SCRIPT}"
        )

        if [ ! -s "${receipt_file}" ]; then
            echo "ERROR: submission produced no receipt: ${receipt_file}" >&2
            exit 1
        fi
        unset POLAR_SUBMITTED_JOB_ID POLAR_SUBMITTED_AT_UNIX
        # shellcheck source=/dev/null
        source "${receipt_file}"
        if ! [[ "${POLAR_SUBMITTED_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: invalid job id in ${receipt_file}" >&2
            exit 1
        fi
        previous_job_id="${POLAR_SUBMITTED_JOB_ID}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${run_id}" "${canonical_arm}" "${repeat}" "${MODE}" mini_swe_agent \
            "${ACTOR_GPUS}" "${ROLLOUT_GPUS}" "${ASYNC_LEVEL}" \
            "${ALLOCATED_GPUS}" "${previous_job_id}" >>"${manifest_file}"
    done
done

if [ "${ACTION}" = plan ]; then
    echo
    echo "Read-only plan. Submit the chained suite with:"
    echo "  bash ${SCRIPT_DIR}/submit_profile.sh submit ${ARMS[*]}"
else
    echo
    echo "Submitted TMax profile batch ${PROFILE_ID}; manifest: ${manifest_file}"
    echo "Arms are serialized; final Slurm job id: ${previous_job_id}"
fi
