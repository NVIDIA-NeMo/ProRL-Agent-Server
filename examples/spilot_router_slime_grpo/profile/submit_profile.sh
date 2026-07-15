#!/usr/bin/env bash
# Submit short, isolated SPilot system-profiling arms.
#
# Planning is read-only and is the default.  ``submit`` chains arms with a
# Slurm dependency so they never contend for the same remote model-pool quota.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SPILOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd -- "${SPILOT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
SUBMIT_SCRIPT="${SPILOT_DIR}/submit_slurm.sh"

DEFAULT_ARMS=(
    async-16t16r-l1
    async-16t16r-l3
    async-8t24r-l3
    async-8t8r-l3
    collocate-16shared
)

usage() {
    cat <<'EOF'
Usage:
  submit_profile.sh list
  submit_profile.sh [plan] [ARM ...]
  submit_profile.sh submit ARM [ARM ...]

Arms:
  async-16t16r-l1    4x8 GPUs; 16 train + 16 rollout; async depth 1
  async-16t16r-l3    4x8 GPUs; 16 train + 16 rollout; async depth 3
  async-8t32r-l3     5x8 GPUs;  8 train + 32 rollout; async depth 3
  async-8t24r-l3     4x8 GPUs;  8 train + 24 rollout; async depth 3
  async-8t8r-l3      4x4 GPUs;  8 train +  8 rollout; async depth 3
  async-4t12r-l3     4x4 GPUs;  4 train + 12 rollout; async depth 3
  collocate-16shared 4x4 GPUs; 16 train / 16 rollout on shared GPUs
  collocate-32shared 4x8 GPUs; 32 train / 32 rollout on shared GPUs

Useful environment overrides:
  PROFILE_STEPS=3                 optimizer steps per arm (1 warmup + 2 steady)
  PROFILE_REPEATS=1               repetitions per arm
  PROFILE_ID=<UTC stamp>          common comparison batch id
  PROFILE_LOAD_DIR=/abs/ckpt      identical release or numbered seed
  PROFILE_TRAIN_DATA=/abs/file    required with a numbered seed
  PROFILE_HF_CHECKPOINT=/abs/hf   Qwen3.5-9B HF assets
  PROFILE_REF_LOAD=/abs/ckpt      Qwen3.5-9B reference release
  PROFILE_AGENT_MODEL_NAME=Qwen/Qwen3.5-9B
  PROFILE_MAX_TOKENS_PER_GPU=32768
  PROFILE_MIN_COMPLETE_ACCEPT_FRACTION=0.5
  PROFILE_EARLY_STOP_GRACE_SESSIONS=2
  PROFILE_PARTITION=batch         Slurm partition
  PROFILE_WALL_TIME=04:00:00      wall time for each arm
  PROFILE_AFTER_JOB_ID=12345      chain the first arm after this job
  PROFILE_DEPENDENCY_KIND=afterok afterok or afterany

The profile disables training eval, graceful resume, and checkpoint writes.
Every arm has a fresh RUN_ID/state/SAVE_DIR.  Do not use the training watcher.
EOF
}

configure_arm() {
    local arm="$1"
    MODE=fully_async
    FULLY_ASYNC=true
    NUM_NODES=4
    GATEWAY_COUNT=4
    TP=4
    ROLLOUT_TP=1
    case "${arm}" in
        async-16t16r-l1)
            GPUS_PER_NODE=8
            ACTOR_NODES=2
            ACTOR_GPUS_PER_NODE=8
            ROLLOUT_GPUS=16
            ASYNC_LEVEL=1
            ;;
        async-16t16r-l3)
            GPUS_PER_NODE=8
            ACTOR_NODES=2
            ACTOR_GPUS_PER_NODE=8
            ROLLOUT_GPUS=16
            ASYNC_LEVEL=3
            ;;
        async-8t32r-l3)
            NUM_NODES=5
            GPUS_PER_NODE=8
            ACTOR_NODES=1
            ACTOR_GPUS_PER_NODE=8
            ROLLOUT_GPUS=32
            ASYNC_LEVEL=3
            ;;
        async-8t24r-l3)
            GPUS_PER_NODE=8
            ACTOR_NODES=1
            ACTOR_GPUS_PER_NODE=8
            ROLLOUT_GPUS=24
            ASYNC_LEVEL=3
            ;;
        async-8t8r-l3)
            GPUS_PER_NODE=4
            ACTOR_NODES=2
            ACTOR_GPUS_PER_NODE=4
            ROLLOUT_GPUS=8
            ASYNC_LEVEL=3
            ;;
        async-4t12r-l3)
            GPUS_PER_NODE=4
            ACTOR_NODES=1
            ACTOR_GPUS_PER_NODE=4
            ROLLOUT_GPUS=12
            ASYNC_LEVEL=3
            ;;
        collocate-16shared|collocate-16)
            MODE=colocate
            FULLY_ASYNC=false
            GPUS_PER_NODE=4
            ACTOR_NODES=4
            ACTOR_GPUS_PER_NODE=4
            ROLLOUT_GPUS=16
            ASYNC_LEVEL=1
            ;;
        collocate-32shared|collocate-32)
            MODE=colocate
            FULLY_ASYNC=false
            GPUS_PER_NODE=8
            ACTOR_NODES=4
            ACTOR_GPUS_PER_NODE=8
            ROLLOUT_GPUS=32
            ASYNC_LEVEL=1
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

ACTION="${1:-plan}"
case "${ACTION}" in
    list)
        usage
        exit 0
        ;;
    plan|submit)
        if [ "$#" -gt 0 ]; then
            shift
        fi
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        # A bare arm is a plan, never an implicit submission.
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
PROFILE_PARTITION="${PROFILE_PARTITION:-batch}"
PROFILE_WALL_TIME="${PROFILE_WALL_TIME:-04:00:00}"
PROFILE_DEPENDENCY_KIND="${PROFILE_DEPENDENCY_KIND:-afterok}"
PROFILE_MAX_TOKENS_PER_GPU="${PROFILE_MAX_TOKENS_PER_GPU:-32768}"
PROFILE_MIN_COMPLETE_ACCEPT_FRACTION="${PROFILE_MIN_COMPLETE_ACCEPT_FRACTION:-0.5}"
PROFILE_EARLY_STOP_GRACE_SESSIONS="${PROFILE_EARLY_STOP_GRACE_SESSIONS:-2}"
POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${WORKSPACE_ROOT}/data}"
export POLAR_DATA_ROOT

# Resolve the Qwen3.5-9B model lineage outside the arm loop.  Every default
# profile starts from the model-only release checkpoint at rollout zero; an
# inherited Qwen4/numeric training shell must not silently change that seed.
PROFILE_HF_CHECKPOINT="${PROFILE_HF_CHECKPOINT:-${POLAR_DATA_ROOT}/checkpoints/Qwen3.5-9B}"
PROFILE_REF_LOAD="${PROFILE_REF_LOAD:-${POLAR_DATA_ROOT}/checkpoints/Qwen3.5-9B_torch_dist}"
PROFILE_TORCH_DIST_DIR="${PROFILE_TORCH_DIST_DIR:-${PROFILE_REF_LOAD}}"
PROFILE_MODEL_ARGS_FILE="${PROFILE_MODEL_ARGS_FILE:-${PROJECT_ROOT}/examples/tmax_slime_grpo/model_args.sh}"
PROFILE_AGENT_MODEL_NAME="${PROFILE_AGENT_MODEL_NAME:-Qwen/Qwen3.5-9B}"

for value_name in PROFILE_STEPS PROFILE_REPEATS; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${value_name} must be a positive integer, got ${value}" >&2
        exit 1
    fi
done
if ! [[ "${PROFILE_MAX_TOKENS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: PROFILE_MAX_TOKENS_PER_GPU must be a positive integer" >&2
    exit 1
fi
if ! [[ "${PROFILE_MIN_COMPLETE_ACCEPT_FRACTION}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
    echo "ERROR: PROFILE_MIN_COMPLETE_ACCEPT_FRACTION must be between 0 and 1" >&2
    exit 1
fi
if ! [[ "${PROFILE_EARLY_STOP_GRACE_SESSIONS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: PROFILE_EARLY_STOP_GRACE_SESSIONS must be a non-negative integer" >&2
    exit 1
fi
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
if [ -n "${PROFILE_TRAIN_DATA:-}" ]; then
    if [[ "${PROFILE_TRAIN_DATA}" != /* ]]; then
        echo "ERROR: PROFILE_TRAIN_DATA must be absolute" >&2
        exit 1
    fi
    if [ ! -s "${PROFILE_TRAIN_DATA}" ]; then
        echo "ERROR: PROFILE_TRAIN_DATA is missing or empty: ${PROFILE_TRAIN_DATA}" >&2
        exit 1
    fi
fi

# A numbered Megatron checkpoint resumes at N+1; --num-rollout is exclusive.
# With no explicit override, pin LOAD_DIR to the Qwen3.5-9B release rather than
# merely leaving it unset and trusting whatever REF_LOAD the caller inherited.
PROFILE_NUM_ROLLOUT="${PROFILE_STEPS}"
PROFILE_SEED_DIR="${PROFILE_REF_LOAD}"
PROFILE_SEED_KIND=release
if [ -n "${PROFILE_LOAD_DIR:-}" ]; then
    if [[ "${PROFILE_LOAD_DIR}" != /* ]]; then
        echo "ERROR: PROFILE_LOAD_DIR must be absolute" >&2
        exit 1
    fi
    if [ ! -s "${PROFILE_LOAD_DIR}/latest_checkpointed_iteration.txt" ]; then
        echo "ERROR: PROFILE_LOAD_DIR has no checkpoint tracker: ${PROFILE_LOAD_DIR}" >&2
        exit 1
    fi
    seed_tracker="$(tr -d '[:space:]' <"${PROFILE_LOAD_DIR}/latest_checkpointed_iteration.txt")"
    PROFILE_SEED_DIR="${PROFILE_LOAD_DIR}"
    PROFILE_SEED_KIND="${seed_tracker}"
    if [[ "${seed_tracker}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        if [ -z "${PROFILE_TRAIN_DATA:-}" ]; then
            echo "ERROR: numbered PROFILE_LOAD_DIR requires the exact PROFILE_TRAIN_DATA used by that checkpoint" >&2
            exit 1
        fi
        PROFILE_NUM_ROLLOUT="$((seed_tracker + 1 + PROFILE_STEPS))"
    elif [ "${seed_tracker}" != "release" ]; then
        echo "ERROR: unsupported checkpoint tracker ${seed_tracker@Q}" >&2
        exit 1
    fi
fi

if [ "${ACTION}" = submit ]; then
    for path_name in \
        PROFILE_HF_CHECKPOINT PROFILE_REF_LOAD PROFILE_TORCH_DIST_DIR \
        PROFILE_MODEL_ARGS_FILE; do
        if [[ "${!path_name}" != /* ]]; then
            echo "ERROR: ${path_name} must be absolute" >&2
            exit 1
        fi
    done
    if [ ! -s "${PROFILE_HF_CHECKPOINT}/config.json" ]; then
        echo "ERROR: Qwen3.5-9B HF config is missing: ${PROFILE_HF_CHECKPOINT}/config.json" >&2
        exit 1
    fi
    if [ ! -s "${PROFILE_REF_LOAD}/latest_checkpointed_iteration.txt" ] || \
       [ "$(tr -d '[:space:]' <"${PROFILE_REF_LOAD}/latest_checkpointed_iteration.txt")" != release ]; then
        echo "ERROR: PROFILE_REF_LOAD must be a release checkpoint: ${PROFILE_REF_LOAD}" >&2
        exit 1
    fi
    if [ ! -s "${PROFILE_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ] || \
       [ "$(tr -d '[:space:]' <"${PROFILE_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt")" != release ]; then
        echo "ERROR: PROFILE_TORCH_DIST_DIR must be a release checkpoint: ${PROFILE_TORCH_DIST_DIR}" >&2
        exit 1
    fi
    if [ ! -r "${PROFILE_MODEL_ARGS_FILE}" ]; then
        echo "ERROR: PROFILE_MODEL_ARGS_FILE is not readable: ${PROFILE_MODEL_ARGS_FILE}" >&2
        exit 1
    fi
fi

# Validate the complete suite before creating a manifest or submitting the
# first arm. Duplicate arms would otherwise collide on RUN_ID midway through a
# batch; PROFILE_REPEATS is the explicit mechanism for repeated measurements.
declare -A seen_arms=()
for arm in "${ARMS[@]}"; do
    configure_arm "${arm}"
    canonical_arm="${arm}"
    case "${canonical_arm}" in
        collocate-16) canonical_arm=collocate-16shared ;;
        collocate-32) canonical_arm=collocate-32shared ;;
    esac
    if [ -n "${seen_arms[${canonical_arm}]:-}" ]; then
        echo "ERROR: duplicate profile arm ${canonical_arm}; use PROFILE_REPEATS instead" >&2
        exit 1
    fi
    seen_arms["${canonical_arm}"]=1
done
unset seen_arms

printf 'action=%s profile_id=%s steps=%s num_rollout=%s repeats=%s dependency=%s seed=%s model=%s max_tokens_per_gpu=%s early_stop=%s+%s\n' \
    "${ACTION}" "${PROFILE_ID}" "${PROFILE_STEPS}" "${PROFILE_NUM_ROLLOUT}" \
    "${PROFILE_REPEATS}" "${PROFILE_DEPENDENCY_KIND}" "${PROFILE_SEED_KIND}" \
    "${PROFILE_AGENT_MODEL_NAME}" "${PROFILE_MAX_TOKENS_PER_GPU}" \
    "${PROFILE_MIN_COMPLETE_ACCEPT_FRACTION}" "${PROFILE_EARLY_STOP_GRACE_SESSIONS}"
printf '%-23s %-12s %7s %7s %5s %8s %8s %s\n' \
    ARM MODE TRAIN ROLLOUT DEPTH NODESxGPU TOTAL_GPU RUN_ID

previous_job_id="${PROFILE_AFTER_JOB_ID:-}"
manifest_dir="${POLAR_DATA_ROOT}/runs/spilot_router_slime_grpo/profile/${PROFILE_ID}"
manifest_file="${manifest_dir}/manifest.tsv"
if [ "${ACTION}" = "submit" ]; then
    if [ -e "${manifest_file}" ]; then
        echo "ERROR: refusing to reuse profile batch; choose a new PROFILE_ID: ${manifest_file}" >&2
        exit 1
    fi
    for repeat in $(seq 1 "${PROFILE_REPEATS}"); do
        for arm in "${ARMS[@]}"; do
            canonical_arm="${arm}"
            case "${canonical_arm}" in
                collocate-16) canonical_arm=collocate-16shared ;;
                collocate-32) canonical_arm=collocate-32shared ;;
            esac
            state_file="${POLAR_DATA_ROOT}/runs/spilot-prof-${canonical_arm}-r${repeat}-${PROFILE_ID}/run_state.env"
            if [ -e "${state_file}" ]; then
                echo "ERROR: refusing to reuse profile state; choose a new PROFILE_ID: ${state_file}" >&2
                exit 1
            fi
        done
    done
    mkdir -p "${manifest_dir}"
    printf 'run_id\tarm\trepeat\tmode\tactor_gpus\trollout_gpus\tasync_level\tallocated_gpus\tjob_id\n' >"${manifest_file}"
fi

for repeat in $(seq 1 "${PROFILE_REPEATS}"); do
    for arm in "${ARMS[@]}"; do
        configure_arm "${arm}"
        canonical_arm="${arm}"
        case "${canonical_arm}" in
            collocate-16) canonical_arm=collocate-16shared ;;
            collocate-32) canonical_arm=collocate-32shared ;;
        esac
        run_id="spilot-prof-${canonical_arm}-r${repeat}-${PROFILE_ID}"
        receipt_file="${POLAR_DATA_ROOT}/runs/${run_id}/submit/last_submission.env"
        state_file="${POLAR_DATA_ROOT}/runs/${run_id}/run_state.env"
        printf '%-23s %-12s %7d %7d %5d %4dx%-3d %8d %s\n' \
            "${canonical_arm}" "${MODE}" "${ACTOR_GPUS}" "${ROLLOUT_GPUS}" \
            "${ASYNC_LEVEL}" "${NUM_NODES}" "${GPUS_PER_NODE}" \
            "${ALLOCATED_GPUS}" "${run_id}"

        if [ "${ACTION}" != "submit" ]; then
            continue
        fi
        dependency=""
        if [ -n "${previous_job_id}" ]; then
            dependency="${PROFILE_DEPENDENCY_KIND}:${previous_job_id}"
        fi
        (
            # Never inherit another logical run's identity or stopping target.
            unset RUN_ID JOB_NAME EXPERIMENT_NAME SAVE_DIR RUN_DIR LOAD_DIR
            unset HF_CHECKPOINT REF_LOAD TORCH_DIST_DIR MODEL_ARGS_FILE
            unset POLAR_AGENT_MODEL_NAME PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
            unset TMAX_PYTORCH_ALLOC_CONF
            unset WANDB_RUN_ID TMAX_TARGET_ITER TMAX_LAST_JOB_ID
            unset TMAX_RUN_STATE_FILE TMAX_SUBMIT_RECEIPT_FILE SBATCH_DEPENDENCY
            unset SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES
            unset SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES
            unset SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY
            unset SPILOT_GPT_GATEWAY_MAX_CONCURRENCY
            unset SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES
            unset SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES

            export RUN_ID="${run_id}"
            export JOB_NAME="sp-prof-${canonical_arm}-r${repeat}"
            export EXPERIMENT_NAME="spilot-profile-${canonical_arm}"
            export WANDB_RUN_ID="${run_id}"
            export WANDB_GROUP="spilot-system-profile-${PROFILE_ID}"
            export WANDB_MODE="${PROFILE_WANDB_MODE:-offline}"
            export TMAX_REQUIRE_WANDB="${PROFILE_REQUIRE_WANDB:-0}"
            export TMAX_RUN_STATE_FILE="${state_file}"
            export TMAX_SUBMIT_RECEIPT_FILE="${receipt_file}"
            export TMAX_PROFILE_BATCH_ID="${PROFILE_ID}"
            export TMAX_PROFILE_ARM="${canonical_arm}"
            export TMAX_TRAIN_MODE="${MODE}"
            if [ "${MODE}" = colocate ]; then
                export TMAX_PYTORCH_ALLOC_CONF="max_split_size_mb:2048"
            else
                export TMAX_PYTORCH_ALLOC_CONF="max_split_size_mb:2048,expandable_segments:True"
            fi
            export TMAX_PROFILE_DISABLE_CHECKPOINT=1
            export TMAX_PERSIST_RUN_STATE=1
            export SUBMIT_BACKEND=sbatch
            export SUBMIT_DRY_RUN=0
            export TMAX_ENABLE_GRACEFUL_EXIT=0
            export TMAX_NUM_ROLLOUT="${PROFILE_NUM_ROLLOUT}"

            # Always replace inherited model lineage.  The default seed is the
            # Qwen3.5-9B release and therefore starts at rollout zero; only an
            # explicit PROFILE_LOAD_DIR opts into a numbered continuation.
            export HF_CHECKPOINT="${PROFILE_HF_CHECKPOINT}"
            export REF_LOAD="${PROFILE_REF_LOAD}"
            export TORCH_DIST_DIR="${PROFILE_TORCH_DIST_DIR}"
            export MODEL_ARGS_FILE="${PROFILE_MODEL_ARGS_FILE}"
            export POLAR_AGENT_MODEL_NAME="${PROFILE_AGENT_MODEL_NAME}"
            export LOAD_DIR="${PROFILE_SEED_DIR}"

            export NUM_NODES="${NUM_NODES}"
            export SLURM_GPUS="${GPUS_PER_NODE}"
            export RAY_NUM_GPUS_PER_NODE="${GPUS_PER_NODE}"
            export ACTOR_NUM_NODES="${ACTOR_NODES}"
            export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE}"
            export ACTOR_TENSOR_MODEL_PARALLEL_SIZE="${TP}"
            export CONTEXT_PARALLEL_SIZE=1
            # Bound a dynamic microbatch to 32K tokens per GPU for both 8-GPU
            # and 16-GPU learners.  Preserve the released sequence/pack shape:
            # an individually oversize trajectory is admitted alone rather
            # than rejected by the topology preflight.
            export MAX_TOKENS_PER_GPU="${PROFILE_MAX_TOKENS_PER_GPU}"
            export TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP=1
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

            # Keep external-provider and gateway pressure identical across arms.
            export POLAR_MULTI_GATEWAY=1
            export POLAR_GATEWAY_COUNT_OVERRIDE="${GATEWAY_COUNT}"
            export SPILOT_EPISODE_ADMISSION_ENABLED=true
            export SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT="${GATEWAY_COUNT}"
            export SPILOT_QWEN_MAX_ACTIVE_EPISODES=8
            export SPILOT_GPT_MAX_ACTIVE_EPISODES=32
            export POLAR_MAX_INIT_WORKERS=96
            export POLAR_MAX_RUN_WORKERS=576
            export POLAR_MAX_POSTRUN_WORKERS=384
            export TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU=16
            export TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU=12
            export TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU=8
            # 32 samples with the defaults stop after 16 accepted + 2 grace,
            # instead of the previous accidental 32/32 long-tail barrier.
            export POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${PROFILE_MIN_COMPLETE_ACCEPT_FRACTION}"
            export POLAR_EARLY_STOP_GRACE_SESSIONS="${PROFILE_EARLY_STOP_GRACE_SESSIONS}"
            export POLAR_CANDIDATE_POOL_HEALTH_GATE_ENABLED=true
            export POLAR_CANDIDATE_POOL_HEALTH_MIN_OBSERVED_SESSIONS=16
            export POLAR_CANDIDATE_POOL_HEALTH_MIN_COMPLETION_FRACTION=0.1

            export SPILOT_QWEN_COST_WEIGHT="${PROFILE_QWEN_COST_WEIGHT:-1.0}"
            export SPILOT_GPT_COST_WEIGHT="${PROFILE_GPT_COST_WEIGHT:-1.0}"
            export SPILOT_COST_PENALTY_LAMBDA="${PROFILE_COST_PENALTY_LAMBDA:-0.0}"
            export SPILOT_COST_NORMALIZER="${PROFILE_COST_NORMALIZER:-1.0}"

            # Preserve the formal train/holdout exclusion contract while
            # disabling all evaluation work inside the short training job.
            export TMAX_EVAL_ENABLED=1
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
            if [ -n "${PROFILE_TRAIN_DATA:-}" ]; then
                export TMAX_TRAIN_DATA="${PROFILE_TRAIN_DATA}"
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
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${run_id}" "${canonical_arm}" "${repeat}" "${MODE}" \
            "${ACTOR_GPUS}" "${ROLLOUT_GPUS}" "${ASYNC_LEVEL}" \
            "${ALLOCATED_GPUS}" "${previous_job_id}" >>"${manifest_file}"
    done
done

if [ "${ACTION}" = "plan" ]; then
    echo
    echo "Read-only plan. Submit one arm or the chained suite with:"
    echo "  bash ${SCRIPT_DIR}/submit_profile.sh submit ${ARMS[*]}"
else
    echo
    echo "Submitted profile batch ${PROFILE_ID}; manifest: ${manifest_file}"
    echo "Arms are serialized; final Slurm job id: ${previous_job_id}"
fi
