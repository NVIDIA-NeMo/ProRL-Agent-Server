#!/usr/bin/env bash
# Build or explicitly submit the TMax comparison matrix. Most historical arms
# remain fixed at four nodes; the explicitly named 8n arm weak-scales the 4B
# fidelity run to two trainer nodes and six rollout nodes.
#
# This script is intentionally inert by default:
#
#   bash submit_matrix.sh
#   bash submit_matrix.sh plan
#
# Both commands only validate assets and print the planned runs. To submit,
# name either `all` or one or more exact settings and provide the confirmation
# token. Every invocation gets fresh RUN_ID/SAVE_DIR paths unless the caller pins
# TMAX_MATRIX_STAMP to a reviewed plan stamp.
#
#   TMAX_MATRIX_STAMP=20260630T140000Z \
#   TMAX_MATRIX_DEPENDENCY=afterok:12345678 \
#   TMAX_MATRIX_CONFIRM=SUBMIT_TMAX_MATRIX \
#     bash submit_matrix.sh submit all
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./lifecycle.sh
source "${SCRIPT_DIR}/lifecycle.sh"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
USER_ROOT="$(dirname "${SPILOT_ROOT}")"

readonly SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_slurm.sh"
readonly DYNAMIC_FILTER="slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
readonly CONFIRM_TOKEN="SUBMIT_TMAX_MATRIX"
readonly -a MATRIX_SETTINGS=(
    qwen35-4b-fidelity
    qwen35-4b-fidelity-8n
    qwen35-4b-fidelity-8n-b16n8-traj
    qwen35-9b-baseline-a2-full65k
    qwen35-9b-b16n16-a2-full65k
    qwen35-9b-async4-full65k
    qwen35-9b-lr5e7-a2-full65k
    qwen35-9b-lr2e6-a2-full65k
)

usage() {
    cat <<'EOF'
Usage:
  submit_matrix.sh [plan [all|SETTING...]]
  submit_matrix.sh submit <all|SETTING...>

Settings:
  qwen35-4b-fidelity
  qwen35-4b-fidelity-8n
  qwen35-4b-fidelity-8n-b16n8-traj
  qwen35-9b-baseline-a2-full65k
  qwen35-9b-b16n16-a2-full65k
  qwen35-9b-async4-full65k
  qwen35-9b-lr5e7-a2-full65k
  qwen35-9b-lr2e6-a2-full65k

`plan` is read-only and is the default. `submit` calls sbatch only when
TMAX_MATRIX_CONFIRM=SUBMIT_TMAX_MATRIX is present. Pin TMAX_MATRIX_STAMP to
submit the exact fresh RUN_ID values shown by an earlier plan. Optionally set
TMAX_MATRIX_DEPENDENCY=afterok:JOBID (or a colon-separated list of numeric job
ids) to queue every arm behind successful completion of existing jobs. Set
TMAX_MATRIX_NUM_ROLLOUT=N to bound only the b16n8 trajectory diagnostic arm.
EOF
}

is_known_setting() {
    local candidate="$1" setting
    for setting in "${MATRIX_SETTINGS[@]}"; do
        [ "${candidate}" = "${setting}" ] && return 0
    done
    return 1
}

select_settings() {
    local output_name="$1"
    shift
    local -n output_ref="${output_name}"
    local candidate
    output_ref=()
    if [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && [ "$1" = "all" ]; }; then
        output_ref=("${MATRIX_SETTINGS[@]}")
        return
    fi
    for candidate in "$@"; do
        if ! is_known_setting "${candidate}"; then
            echo "ERROR: unknown matrix setting: ${candidate}" >&2
            usage >&2
            return 1
        fi
        output_ref+=("${candidate}")
    done
}

require_file() {
    local label="$1" path="$2"
    if [ ! -s "${path}" ]; then
        echo "ERROR: ${label} is missing or empty: ${path}" >&2
        return 1
    fi
}

verify_jsonl() {
    local label="$1" path="$2" expected_rows="$3" expected_sha256="$4"
    require_file "${label}" "${path}"
    local rows digest
    rows="$(awk 'NF { count += 1 } END { print count + 0 }' "${path}")"
    if [ "${rows}" -ne "${expected_rows}" ]; then
        echo "ERROR: ${label} has ${rows} rows, expected ${expected_rows}: ${path}" >&2
        return 1
    fi
    digest="$(sha256sum "${path}" | awk '{print $1}')"
    if [ "${digest}" != "${expected_sha256}" ]; then
        echo "ERROR: ${label} SHA-256 changed: ${digest} != ${expected_sha256}" >&2
        return 1
    fi
}

reset_matrix_overrides() {
    # Prevent a sourced run-state or an interactive smoke environment from
    # silently changing one arm of the comparison.
    unset \
        LOAD_DIR SBATCH_DEPENDENCY SUBMIT_DRY_RUN \
        TMAX_NUM_ROLLOUT TMAX_TARGET_ITER \
        POLAR_GATEWAY_MAX_INIT_WORKERS \
        POLAR_GATEWAY_MAX_RUN_WORKERS \
        POLAR_GATEWAY_MAX_POSTRUN_WORKERS \
        POLAR_GATEWAY_COMPLETION_QUEUE_SIZE \
        POLAR_GATEWAY_COMPLETION_WRITE_WORKERS
}

configure_common() {
    local setting="$1" topology_tag=4n32
    reset_matrix_overrides
    if [ -n "${MATRIX_DEPENDENCY}" ]; then
        export SBATCH_DEPENDENCY="${MATRIX_DEPENDENCY}"
    fi

    case "${setting}" in
        qwen35-4b-fidelity-8n|qwen35-4b-fidelity-8n-b16n8-traj)
            topology_tag=8n64
            ;;
    esac
    export RUN_ID="tmax-${topology_tag}-${setting}-${MATRIX_STAMP}"
    export EXPERIMENT_NAME="tmax-${topology_tag}-${setting}"
    export JOB_NAME="polar-${RUN_ID}"
    export POLAR_DATA_ROOT="${MATRIX_DATA_ROOT}"
    export SAVE_DIR="${POLAR_DATA_ROOT}/ckpt/${RUN_ID}"
    export TRAINING_COMPLETE_MARKER="${SAVE_DIR}/TRAINING_COMPLETE"
    export FINAL_EVAL_COMPLETE_MARKER="${SAVE_DIR}/FINAL_EVAL_COMPLETE"
    export TMAX_RUN_STATE_FILE="${POLAR_DATA_ROOT}/runs/${RUN_ID}/run_state.env"
    export TMAX_SUBMIT_RECEIPT_FILE="${POLAR_DATA_ROOT}/runs/${RUN_ID}/submit/last_submission.env"
    export TMAX_DATA_INTEGRITY_MANIFEST="${POLAR_DATA_ROOT}/runs/${RUN_ID}/tmax-data-integrity.json"
    export TMAX_EVAL_CONFIG_PATH="${POLAR_DATA_ROOT}/runs/${RUN_ID}/tmax-eval-config.json"

    # Train on every currently usable TMax task except the exact pinned
    # 100-row holdout. External benchmarks are evaluated offline.
    export TMAX_TRAIN_DATA="${MATRIX_SOURCE_RUN}/tmax-train.jsonl"
    export TMAX_EVAL_DATA="${MATRIX_SOURCE_RUN}/tmax_holdout-eval.jsonl"
    export TMAX_EXCLUDE_DATA="${TMAX_EVAL_DATA}"
    export TMAX_TRAIN_DATA_SHA256="${MATRIX_TRAIN_SHA256}"
    export TMAX_EVAL_DATA_SHA256="${MATRIX_HOLDOUT_SHA256}"
    unset TMAX_EXTERNAL_EVAL_DATA_SHA256
    export TMAX_EVAL_BUNDLE_SHA256="${MATRIX_EVAL_BUNDLE_SHA256}"
    export TMAX_PREPARE_DATA=0
    export TMAX_PREPARE_EVAL_DATA=0
    # These matrix datasets and images are pinned and were already deeply
    # audited when the source run was built. Keep only fast hash/path checks on
    # each submission; set this to 1 after changing any JSONL or image bundle.
    export TMAX_VALIDATE_EXISTING_ASSETS=0
    export TMAX_TRAIN_START_INDEX=0
    export TMAX_MAX_TASKS=-1
    export TMAX_EVAL_START_INDEX=900
    export TMAX_EVAL_MAX_TASKS=100
    # The source tree still contains 14,601 tasks. Three missing SIFs are
    # deliberately ignored, leaving 14,598 usable tasks and 14,498 train rows
    # after excluding the fixed holdout.
    export TMAX_TOTAL_TASKS=14601
    export TMAX_REQUIRE_EXACT_TOTAL_TASKS=1
    export TMAX_ONLY_READY=1

    export TMAX_EVAL_ENABLED=1
    export TMAX_EVAL_SOURCE=tmax
    export TMAX_EVAL_DATASET_NAME=tmax_holdout
    export TMAX_EVAL_SAMPLES_PER_PROMPT=1
    export TMAX_EVAL_MIN_VALID_SAMPLES=100
    export TMAX_EVAL_TEMPERATURE=0.2
    export TMAX_EVAL_TOP_P=1.0
    export TMAX_EVAL_MAX_RESPONSE_LEN=16384
    export TMAX_EXTERNAL_EVAL_ENABLED=0
    export TMAX_EXTERNAL_EVAL_SOURCE=harbor
    export TMAX_EXTERNAL_EVAL_DATASET_NAME=terminal_bench_2_0
    export TMAX_EXTERNAL_EVAL_MAX_TASKS=89
    export TMAX_EXTERNAL_EVAL_SAMPLES_PER_PROMPT=1
    export TMAX_EXTERNAL_EVAL_MIN_VALID_SAMPLES=89
    export TMAX_EXTERNAL_EVAL_TEMPERATURE=0.7
    export TMAX_EXTERNAL_EVAL_TOP_P=0.95
    export TMAX_EXTERNAL_EVAL_MAX_RESPONSE_LEN=16384
    # Keep the pinned Terminal-Bench 2.0 pass@1 protocol immutable.  The
    # audited 89-row bundle was generated with the benchmark's 50-step cap;
    # changing this value would make scores incomparable and correctly fails
    # prepare_harbor_eval.py's byte-for-byte validation.
    export TMAX_EXTERNAL_EVAL_AGENT_STEP_LIMIT=50
    export TMAX_EVAL_INTERVAL=20
    export TMAX_OVERRIDE_OPT_PARAM_SCHEDULER=0
    # Individual arms may require a strict baseline barrier before rollout 0.
    # Keep the historical concurrent behavior by default for the 9B sweep.
    export TMAX_CONCURRENT_PRETRAIN_EVAL=1
    export TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN=0

    export NUM_NODES=4
    export SLURM_GPUS=8
    export RAY_NUM_GPUS_PER_NODE=8
    export TMAX_REQUIRE_FULL_GPU_ALLOCATION=1
    # Preserve the original single-gateway, fresh direct-exec runtime for the
    # model comparison.  Every command enters a fresh Apptainer process; no
    # persistent broker or instance state is shared between commands.
    export POLAR_MULTI_GATEWAY=0
    export ACCOUNT=nvr_lpr_llm
    export PARTITION="${TMAX_MATRIX_PARTITIONS:-backfill,batch}"
    export SLURM_CONSTRAINT=H100
    export SLURM_EXCLUDE="${TMAX_MATRIX_EXCLUDE_NODES:-}"
    export WALL_TIME=4:00:00
    export TMAX_MIN_WALL_TIME=4:00:00
    export CPUS_PER_TASK=128
    export SLURM_STEP_CPUS_PER_TASK=120
    export SUBMIT_BACKEND=sbatch

    export ROLLOUT_BATCH_SIZE=8
    export N_SAMPLES_PER_PROMPT=32
    export NUM_STEPS_PER_ROLLOUT=1
    export GLOBAL_BATCH_SIZE=256
    export EVAL_GLOBAL_BATCH_SIZE=256
    export NUM_EPOCH=1
    export SAVE_INTERVAL=10
    # Four-node arms reserve one complete trainer node (TP4 x DP2) and use all
    # remaining 24 GPUs as TP1 rollout engines. The explicit 8n arm overrides
    # this below to two trainer nodes (TP4 x DP4) plus 48 rollout engines.
    export ACTOR_NUM_NODES=1
    export ACTOR_NUM_GPUS_PER_NODE=8
    export ACTOR_TENSOR_MODEL_PARALLEL_SIZE=4
    export CONTEXT_PARALLEL_SIZE=1
    export SEQUENCE_PARALLEL=1
    export ROLLOUT_NUM_GPUS=24
    export ROLLOUT_NUM_GPUS_PER_ENGINE=1
    export DIST_CKPT_STRICTNESS=log_all
    export ATTENTION_BACKEND=flash
    # ROLLOUT_MAX_RESPONSE_LEN is a single model turn.  The independent total
    # trajectory response budget plus the prompt forms the trainer pack.
    export TMAX_MAX_TOTAL_RESPONSE_LEN=65536
    export TMAX_TRAIN_PACK_LENGTH=67584
    export SEQ_LENGTH=67584
    export ROLLOUT_MAX_PROMPT_LEN=2048
    export ROLLOUT_MAX_RESPONSE_LEN=16384
    # Slime multiplies this cap by CP only. CP=1 therefore needs the complete
    # 67,584-token pack even though TP4 sequence-parallels the actual compute.
    export MAX_TOKENS_PER_GPU=67584
    # Keep full-trajectory fidelity while bounding per-chunk vocabulary
    # temporaries.  The prior 256-token chunks contributed directly to the
    # first-backward OOM on a worst-case 65k trajectory.
    export LOG_PROBS_CHUNK_SIZE=64
    export TMAX_MODEL_MAX_CONTEXT_LENGTH=262144
    export SGLANG_CONTEXT_LENGTH=262144
    export SGLANG_MEM_FRACTION_STATIC=0.7
    export SGLANG_REASONING_PARSER=qwen3
    export TMAX_ENABLE_FP32_LM_HEAD=1
    export SGLANG_ENABLE_FP32_LM_HEAD=1
    export CALCULATE_PER_TOKEN_LOSS=1

    export POLICY_LOSS_TYPE=dppo
    export DPPO_DIVERGENCE_TYPE=tv
    export DPPO_DIVERGENCE_THRESHOLD=0.1
    export USE_TIS=0
    export KL_LOSS_COEF=0
    export GRPO_STD_NORMALIZATION=0
    export POLAR_FULLY_ASYNC=true
    # Model-specific arms may override these.  The full-fidelity 9B arms keep
    # complete groups; the 4B reproduction restores the proven early-stop
    # policy from job 13241355 so one long-tail agent cannot starve training.
    export POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0
    export POLAR_EARLY_STOP_GRACE_SESSIONS=0
    export TMAX_DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_FILTER}"
    export TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU=16
    # A single gateway runs on 32 host CPUs. 384 simultaneous agent processes
    # (16/GPU) can starve its asyncio control plane when the 189-sample
    # baseline eval overlaps the 576-session async training window, producing
    # connection resets and zero-trace samples. Twelve run slots per rollout
    # GPU still supplies ample decode concurrency while keeping the aggregate
    # gateway pool below the observed failure regime.
    export TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU=12
    export TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU=8
    export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS=1200
    export TMAX_TRAIN_TASK_TIMEOUT_RESERVE_SECONDS=600
    export POLAR_REQUEST_TIMEOUT=3600
    export POLAR_TASK_TIMEOUT_FLOOR_SECONDS=1800

    export TMAX_AGENT_HARNESS=vanillux2
    export POLAR_AGENT_HARNESS=vanillux2
    export POLAR_AGENT_STEP_LIMIT=64
    export POLAR_AGENT_COST_LIMIT=0
    export POLAR_AGENT_TEMPERATURE=1.0
    export POLAR_AGENT_TOP_P=1.0
    export POLAR_AGENT_MAX_TOKENS=16384
    export POLAR_AGENT_ENABLE_THINKING=true
    # Save two complete representative message trajectories every ten train
    # steps (one high-reward and one low-reward when both exist).
    export POLAR_ROLLOUT_EXAMPLE_INTERVAL=10
    export POLAR_ROLLOUT_EXAMPLE_COUNT=2
    export POLAR_ROLLOUT_EXAMPLES_WANDB=1

    # Single-gateway queue and callback capacities.
    export POLAR_COMPLETION_QUEUE_SIZE=32768
    export POLAR_COMPLETION_WRITE_WORKERS=16
    export POLAR_COMPLETION_BATCH_SIZE=16
    export POLAR_COMPLETION_WRITE_MAX_ATTEMPTS=3
    export POLAR_COMPLETION_RETRY_BACKOFF_SECONDS=0.1
    # Use fresh Apptainer exec commands for this comparison matrix.
    export POLAR_APPTAINER_PERSISTENT_BROKER=0
    export POLAR_APPTAINER_NO_INSTANCE=1
    # These gates are inert on the direct path but remain bounded if an invalid
    # downstream override attempts to start a broker.
    export POLAR_APPTAINER_BROKER_START_CONCURRENCY=8
    # The per-image gate is acquired first, so duplicate starts cannot consume
    # aggregate permits needed by other task images.
    export POLAR_APPTAINER_BROKER_START_PER_IMAGE_CONCURRENCY=2
    export POLAR_APPTAINER_BROKER_START_TIMEOUT_SEC=120
    export POLAR_APPTAINER_DIRECT_EXEC_RETRIES=3
    export POLAR_APPTAINER_BROKER_RETRY_BACKOFF_SEC=5
    export POLAR_APPTAINER_BROKER_RETRY_BACKOFF_MAX_SEC=30
    export POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY=2

    export WANDB_PROJECT="${TMAX_MATRIX_WANDB_PROJECT:-polar-tmax-grpo}"
    export WANDB_GROUP="${MATRIX_WANDB_GROUP}"
    export WANDB_RESUME=allow
    export WANDB_ALWAYS_USE_TRAIN_STEP=1
    export TMAX_REQUIRE_WANDB=1
    export GPU_MONITOR_ENABLED=1
    export GPU_MONITOR_PREFIX=polar_tmax_system
    export GPU_MONITOR_NODE_ROLE=rank
    export TMAX_TRAIN_ABI_PREFLIGHT=1
    export TMAX_PERSIST_RUN_STATE=1
}

configure_setting() {
    local setting="$1"
    configure_common "${setting}"

    case "${setting}" in
        qwen35-4b-fidelity)
            export HF_CHECKPOINT=Qwen/Qwen3.5-4B
            export REF_LOAD="${QWEN4_REF_LOAD}"
            export TORCH_DIST_DIR="${REF_LOAD}"
            export MODEL_ARGS_FILE="${PROJECT_ROOT}/examples/swegym_slime_grpo/model_args.sh"
            export POLAR_AGENT_MODEL_NAME=Qwen/Qwen3.5-4B
            # Qwen3.5-4B ties the embedding and output projection. A standalone
            # FP32 LM head would mutate that shared Parameter and is therefore
            # rejected by both the trainer's correctness guard and the model
            # architecture. Keep native projection precision on both sides.
            export TMAX_ENABLE_FP32_LM_HEAD=0
            export SGLANG_ENABLE_FP32_LM_HEAD=0
            # Keep the 8-trainer/24-rollout topology, but close each rollout
            # after eight groups so the trainer is exercised comfortably before
            # cw-dfw's 30-minute idle-GPU reaper. async=3 still keeps a bounded
            # 24-group/192-session terminal window in flight. Run the complete
            # fixed baseline synchronously before rollout 0; periodic evals keep
            # the same strict barrier semantics.
            export ROLLOUT_BATCH_SIZE=8
            export N_SAMPLES_PER_PROMPT=8
            export NUM_STEPS_PER_ROLLOUT=1
            export GLOBAL_BATCH_SIZE=64
            export EVAL_GLOBAL_BATCH_SIZE=64
            export SAVE_INTERVAL=5
            export TRAIN_LR=5e-7
            export TMAX_MIN_ASYNC_LEVEL=3
            export POLAR_MAX_ASYNC_LEVEL=3
            export TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU=8
            export POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0.5
            export POLAR_EARLY_STOP_GRACE_SESSIONS=2
            export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS=600
            export TMAX_CONCURRENT_PRETRAIN_EVAL=0
            export MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF=0.5
            ;;
        qwen35-4b-fidelity-8n)
            export HF_CHECKPOINT=Qwen/Qwen3.5-4B
            export REF_LOAD="${QWEN4_REF_LOAD}"
            export TORCH_DIST_DIR="${REF_LOAD}"
            export MODEL_ARGS_FILE="${PROJECT_ROOT}/examples/swegym_slime_grpo/model_args.sh"
            export POLAR_AGENT_MODEL_NAME=Qwen/Qwen3.5-4B
            export TMAX_ENABLE_FP32_LM_HEAD=0
            export SGLANG_ENABLE_FP32_LM_HEAD=0
            # Weak-scale the proven 4B arm: TP4/CP1 grows from DP2 to DP4,
            # while 48 TP1 rollout engines double the group size without
            # changing prompts per step or sessions per rollout GPU. Use one
            # local gateway per node so 384 active sessions do not bottleneck
            # on the trainer head node.
            export NUM_NODES=8
            export ACTOR_NUM_NODES=2
            export ROLLOUT_NUM_GPUS=48
            export POLAR_MULTI_GATEWAY=1
            export ROLLOUT_BATCH_SIZE=8
            export N_SAMPLES_PER_PROMPT=16
            export NUM_STEPS_PER_ROLLOUT=1
            export GLOBAL_BATCH_SIZE=128
            export EVAL_GLOBAL_BATCH_SIZE=128
            export TMAX_EVAL_INTERVAL=10
            if [ -n "${MATRIX_QWEN4_LOAD_DIR}" ]; then
                export LOAD_DIR="${MATRIX_QWEN4_LOAD_DIR}"
                export TMAX_OVERRIDE_OPT_PARAM_SCHEDULER=1
            fi
            export SAVE_INTERVAL=5
            export TRAIN_LR=5e-7
            export TMAX_MIN_ASYNC_LEVEL=3
            export POLAR_MAX_ASYNC_LEVEL=3
            export TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU=8
            export POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0.5
            # Preserve the current 75% early-stop target: 6/8 becomes 12/16.
            export POLAR_EARLY_STOP_GRACE_SESSIONS=4
            export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS=600
            export TMAX_CONCURRENT_PRETRAIN_EVAL=0
            export MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF=0.5
            ;;
        qwen35-4b-fidelity-8n-b16n8-traj)
            export HF_CHECKPOINT=Qwen/Qwen3.5-4B
            export REF_LOAD="${QWEN4_REF_LOAD}"
            export TORCH_DIST_DIR="${REF_LOAD}"
            export MODEL_ARGS_FILE="${PROJECT_ROOT}/examples/swegym_slime_grpo/model_args.sh"
            export POLAR_AGENT_MODEL_NAME=Qwen/Qwen3.5-4B
            export TMAX_ENABLE_FP32_LM_HEAD=0
            export SGLANG_ENABLE_FP32_LM_HEAD=0
            # Keep the existing 8-node GPU topology and total 128 rollout
            # slots, but double independent prompts and make each trajectory
            # one exchangeable loss unit. With group8, grace4 deliberately
            # requires all 8 sessions so this diagnostic arm has no completion-
            # speed selection bias.
            export NUM_NODES=8
            export ACTOR_NUM_NODES=2
            export ROLLOUT_NUM_GPUS=48
            export POLAR_MULTI_GATEWAY=1
            export ROLLOUT_BATCH_SIZE=16
            export N_SAMPLES_PER_PROMPT=8
            export NUM_STEPS_PER_ROLLOUT=1
            export GLOBAL_BATCH_SIZE=128
            export EVAL_GLOBAL_BATCH_SIZE=128
            export CALCULATE_PER_TOKEN_LOSS=0
            export GRPO_STD_NORMALIZATION=0
            export TMAX_EVAL_INTERVAL=10
            if [ -n "${MATRIX_NUM_ROLLOUT}" ]; then
                export TMAX_NUM_ROLLOUT="${MATRIX_NUM_ROLLOUT}"
                export TMAX_TARGET_ITER="$((TMAX_NUM_ROLLOUT - 1))"
            fi
            if [ -n "${MATRIX_QWEN4_LOAD_DIR}" ]; then
                export LOAD_DIR="${MATRIX_QWEN4_LOAD_DIR}"
                export TMAX_OVERRIDE_OPT_PARAM_SCHEDULER=1
            fi
            export SAVE_INTERVAL=5
            export TRAIN_LR=5e-7
            export TMAX_MIN_ASYNC_LEVEL=3
            export POLAR_MAX_ASYNC_LEVEL=3
            export TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU=8
            export POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0.5
            export POLAR_EARLY_STOP_GRACE_SESSIONS=4
            export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS=600
            export TMAX_CONCURRENT_PRETRAIN_EVAL=0
            export TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN=1
            export MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF=0.5
            ;;
        qwen35-9b-baseline-a2-full65k)
            configure_qwen9b 1e-6 2
            ;;
        qwen35-9b-b16n16-a2-full65k)
            configure_qwen9b 1e-6 2
            export ROLLOUT_BATCH_SIZE=16
            export N_SAMPLES_PER_PROMPT=16
            export GLOBAL_BATCH_SIZE=256
            ;;
        qwen35-9b-async4-full65k)
            configure_qwen9b 1e-6 4
            ;;
        qwen35-9b-lr5e7-a2-full65k)
            configure_qwen9b 5e-7 2
            ;;
        qwen35-9b-lr2e6-a2-full65k)
            configure_qwen9b 2e-6 2
            ;;
        *)
            echo "ERROR: internal unknown setting: ${setting}" >&2
            return 1
            ;;
    esac

    export POLAR_MAX_INIT_WORKERS="$((ROLLOUT_NUM_GPUS * 2))"
    export POLAR_MAX_RUN_WORKERS="$((ROLLOUT_NUM_GPUS * TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU))"
    export POLAR_MAX_POSTRUN_WORKERS="$((ROLLOUT_NUM_GPUS * TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU))"
}

configure_qwen9b() {
    local learning_rate="$1" async_level="$2"
    export HF_CHECKPOINT="${QWEN9_HF_CHECKPOINT}"
    export REF_LOAD="${QWEN9_REF_LOAD}"
    export TORCH_DIST_DIR="${REF_LOAD}"
    export MODEL_ARGS_FILE="${SCRIPT_DIR}/model_args.sh"
    export POLAR_AGENT_MODEL_NAME=Qwen/Qwen3.5-9B
    export TRAIN_LR="${learning_rate}"
    export TMAX_MIN_ASYNC_LEVEL="${async_level}"
    export POLAR_MAX_ASYNC_LEVEL="${async_level}"
    export MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF=1.0
}

validate_configured_setting() {
    local setting="${1:?missing matrix setting}"
    local actor_gpus total_gpus active_sessions minimum_sessions expected_postrun_workers
    local actor_dp pack_tokens trainer_token_capacity expected_train_agent_timeout expected_scheduler_override
    actor_gpus="$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))"
    total_gpus="$((NUM_NODES * SLURM_GPUS))"
    active_sessions="$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT * POLAR_MAX_ASYNC_LEVEL))"
    minimum_sessions="$((ROLLOUT_NUM_GPUS * TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU))"
    expected_postrun_workers="$((ROLLOUT_NUM_GPUS * TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU))"
    if [ "$((actor_gpus + ROLLOUT_NUM_GPUS))" -ne "${total_gpus}" ]; then
        echo "ERROR: ${RUN_ID} uses actor=${actor_gpus} + rollout=${ROLLOUT_NUM_GPUS}, expected ${total_gpus}" >&2
        return 1
    fi
    if [ "${TMAX_MAX_TASKS}" != "-1" ] || \
       [ "${TMAX_TOTAL_TASKS}" -ne 14601 ] || \
       [ "${TMAX_ONLY_READY}" -ne 1 ] || \
       [ "${TMAX_REQUIRE_EXACT_TOTAL_TASKS}" -ne 1 ] || \
       [ "${TMAX_EVAL_MAX_TASKS}" -ne 100 ] || \
       [ "${TMAX_EXTERNAL_EVAL_ENABLED}" -ne 0 ] || \
       [ "${TMAX_EXCLUDE_DATA}" != "${TMAX_EVAL_DATA}" ]; then
        echo "ERROR: ${RUN_ID} must use the 14,498-train/fixed-100-holdout complement with offline external eval" >&2
        return 1
    fi
    if [ "$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))" -ne \
         "$((GLOBAL_BATCH_SIZE * NUM_STEPS_PER_ROLLOUT))" ]; then
        echo "ERROR: ${RUN_ID} rollout samples must exactly feed its configured optimizer steps" >&2
        return 1
    fi
    actor_dp="$((actor_gpus / (ACTOR_TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE)))"
    pack_tokens="$((ROLLOUT_MAX_PROMPT_LEN + TMAX_MAX_TOTAL_RESPONSE_LEN))"
    trainer_token_capacity="$((MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE))"
    expected_scheduler_override=0
    if [ "${NUM_NODES}" -eq 8 ] && [ -n "${LOAD_DIR:-}" ]; then
        expected_scheduler_override=1
    fi
    if [ "${TMAX_OVERRIDE_OPT_PARAM_SCHEDULER}" -ne "${expected_scheduler_override}" ]; then
        echo "ERROR: ${RUN_ID} scheduler override must be enabled only for an 8n continuation" >&2
        return 1
    fi
    case "${NUM_NODES}" in
        4)
            if [ "${actor_gpus}" -ne 8 ] || \
               [ "${ACTOR_TENSOR_MODEL_PARALLEL_SIZE}" -ne 4 ] || \
               [ "${CONTEXT_PARALLEL_SIZE}" -ne 1 ] || \
               [ "${actor_dp}" -ne 2 ] || \
               [ "${ROLLOUT_NUM_GPUS}" -ne 24 ] || \
               [ "${ROLLOUT_NUM_GPUS_PER_ENGINE}" -ne 1 ]; then
                echo "ERROR: ${RUN_ID} 4-node topology must be 8 trainer TP4/CP1/DP2 + 24 TP1 rollout GPUs" >&2
                return 1
            fi
            ;;
        8)
            if [[ "${POLAR_AGENT_MODEL_NAME}" != *4B ]] || \
               [ "${actor_gpus}" -ne 16 ] || \
               [ "${ACTOR_NUM_NODES}" -ne 2 ] || \
               [ "${ACTOR_TENSOR_MODEL_PARALLEL_SIZE}" -ne 4 ] || \
               [ "${CONTEXT_PARALLEL_SIZE}" -ne 1 ] || \
               [ "${actor_dp}" -ne 4 ] || \
               [ "${ROLLOUT_NUM_GPUS}" -ne 48 ] || \
               [ "${ROLLOUT_NUM_GPUS_PER_ENGINE}" -ne 1 ]; then
                echo "ERROR: ${RUN_ID} 8-node 4B topology must be 16 trainer TP4/CP1/DP4 + 48 TP1 rollout GPUs" >&2
                return 1
            fi
            ;;
        *)
            echo "ERROR: ${RUN_ID} matrix NUM_NODES must be 4 or 8" >&2
            return 1
            ;;
    esac
    if [ "${SEQUENCE_PARALLEL}" != 1 ]; then
        echo "ERROR: ${RUN_ID} full65k trainer requires Megatron sequence parallelism" >&2
        return 1
    fi
    if [ "${ROLLOUT_MAX_PROMPT_LEN}" -ne 2048 ] || \
       [ "${ROLLOUT_MAX_RESPONSE_LEN}" -ne 16384 ] || \
       [ "${TMAX_MAX_TOTAL_RESPONSE_LEN}" -ne 65536 ] || \
       [ "${pack_tokens}" -ne 67584 ] || \
       [ "${TMAX_TRAIN_PACK_LENGTH}" -ne "${pack_tokens}" ] || \
       [ "${SEQ_LENGTH}" -ne "${TMAX_TRAIN_PACK_LENGTH}" ]; then
        echo "ERROR: ${RUN_ID} token budget must be per-turn 16384, total response 65536, prompt 2048, pack 67584" >&2
        return 1
    fi
    if [ "${MAX_TOKENS_PER_GPU}" -ne 67584 ] || \
       [ "${trainer_token_capacity}" -lt "${pack_tokens}" ]; then
        echo "ERROR: ${RUN_ID} dynamic trainer cap must preserve the full 67584-token pack" >&2
        return 1
    fi
    if [ "${LOG_PROBS_CHUNK_SIZE}" -ne 64 ]; then
        echo "ERROR: ${RUN_ID} full65k trainer requires 64-token log-prob chunks" >&2
        return 1
    fi
    case "${TMAX_CONCURRENT_PRETRAIN_EVAL}" in
        0|1) ;;
        *)
            echo "ERROR: ${RUN_ID} TMAX_CONCURRENT_PRETRAIN_EVAL must be 0 or 1" >&2
            return 1
            ;;
    esac
    case "${TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN}" in
        0|1) ;;
        *)
            echo "ERROR: ${RUN_ID} TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN must be 0 or 1" >&2
            return 1
            ;;
    esac
    case "${POLAR_AGENT_MODEL_NAME}" in
        *4B)
            if [ "${TMAX_ENABLE_FP32_LM_HEAD}" -ne 0 ] || \
               [ "${SGLANG_ENABLE_FP32_LM_HEAD}" -ne 0 ]; then
                echo "ERROR: ${RUN_ID} tied 4B model must keep native LM-head precision" >&2
                return 1
            fi
            if [ "${NUM_STEPS_PER_ROLLOUT}" -ne 1 ] || \
               [ "${SAVE_INTERVAL}" -ne 5 ] || \
               [ "${TMAX_MIN_ASYNC_LEVEL}" -ne 3 ] || \
               [ "${POLAR_MAX_ASYNC_LEVEL}" -ne 3 ] || \
               [ "${TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU}" -ne 8 ] || \
               [ "${POLAR_MIN_COMPLETE_ACCEPT_FRACTION}" != 0.5 ] || \
               [ "${TMAX_TRAIN_AGENT_TIMEOUT_SECONDS}" -ne 600 ] || \
               [ "${TMAX_CONCURRENT_PRETRAIN_EVAL}" -ne 0 ] || \
               [ "${TRAIN_LR}" != 5e-7 ]; then
                echo "ERROR: ${RUN_ID} 4B arm must preserve the weak-scaling optimizer and async settings" >&2
                return 1
            fi
            case "${setting}" in
                qwen35-4b-fidelity)
                    if [ "${NUM_NODES}" -ne 4 ] || \
                       [ "${ROLLOUT_BATCH_SIZE}" -ne 8 ] || \
                       [ "${N_SAMPLES_PER_PROMPT}" -ne 8 ] || \
                       [ "${GLOBAL_BATCH_SIZE}" -ne 64 ] || \
                       [ "${EVAL_GLOBAL_BATCH_SIZE}" -ne 64 ] || \
                       [ "${CALCULATE_PER_TOKEN_LOSS}" -ne 1 ] || \
                       [ "${TMAX_EVAL_INTERVAL}" -ne 20 ] || \
                       [ "${POLAR_EARLY_STOP_GRACE_SESSIONS}" -ne 2 ] || \
                       [ "${TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN}" -ne 0 ] || \
                       [ "${POLAR_MULTI_GATEWAY}" -ne 0 ]; then
                        echo "ERROR: ${RUN_ID} 4-node 4B arm must use group8/global64/eval20/grace2/single-gateway" >&2
                        return 1
                    fi
                    ;;
                qwen35-4b-fidelity-8n)
                    if [ "${NUM_NODES}" -ne 8 ] || \
                       [ "${ROLLOUT_BATCH_SIZE}" -ne 8 ] || \
                       [ "${N_SAMPLES_PER_PROMPT}" -ne 16 ] || \
                       [ "${GLOBAL_BATCH_SIZE}" -ne 128 ] || \
                       [ "${EVAL_GLOBAL_BATCH_SIZE}" -ne 128 ] || \
                       [ "${CALCULATE_PER_TOKEN_LOSS}" -ne 1 ] || \
                       [ "${TMAX_EVAL_INTERVAL}" -ne 10 ] || \
                       [ "${POLAR_EARLY_STOP_GRACE_SESSIONS}" -ne 4 ] || \
                       [ "${TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN}" -ne 0 ] || \
                       [ "${POLAR_MULTI_GATEWAY}" -ne 1 ]; then
                        echo "ERROR: ${RUN_ID} existing 8-node 4B arm must use 8 prompts x group16/global128/eval10/grace4/token loss/multi-gateway" >&2
                        return 1
                    fi
                    ;;
                qwen35-4b-fidelity-8n-b16n8-traj)
                    if [ "${NUM_NODES}" -ne 8 ] || \
                       [ "${ROLLOUT_BATCH_SIZE}" -ne 16 ] || \
                       [ "${N_SAMPLES_PER_PROMPT}" -ne 8 ] || \
                       [ "${GLOBAL_BATCH_SIZE}" -ne 128 ] || \
                       [ "${EVAL_GLOBAL_BATCH_SIZE}" -ne 128 ] || \
                       [ "${CALCULATE_PER_TOKEN_LOSS}" -ne 0 ] || \
                       [ "${GRPO_STD_NORMALIZATION}" -ne 0 ] || \
                       [ "${TMAX_EVAL_INTERVAL}" -ne 10 ] || \
                       [ "${POLAR_EARLY_STOP_GRACE_SESSIONS}" -ne 4 ] || \
                       [ "${TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN}" -ne 1 ] || \
                       [ "${POLAR_MULTI_GATEWAY}" -ne 1 ]; then
                        echo "ERROR: ${RUN_ID} trajectory arm must use 16 prompts x group8/global128/eval10/grace4/trajectory loss/multi-gateway" >&2
                        return 1
                    fi
                    ;;
                *)
                    echo "ERROR: ${RUN_ID} unknown 4B matrix setting ${setting}" >&2
                    return 1
                    ;;
            esac
            ;;
        *)
            if [ "${TMAX_ENABLE_FP32_LM_HEAD}" -ne 1 ] || \
               [ "${SGLANG_ENABLE_FP32_LM_HEAD}" -ne 1 ] || \
               [ "${CALCULATE_PER_TOKEN_LOSS}" -ne 1 ] || \
               [ "${TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN}" -ne 0 ] || \
               [ "${POLAR_MULTI_GATEWAY}" -ne 0 ]; then
                echo "ERROR: ${RUN_ID} untied 9B arm requires matching FP32 LM heads, token loss, and single gateway" >&2
                return 1
            fi
            ;;
    esac
    if [ "${setting}" = qwen35-4b-fidelity-8n-b16n8-traj ]; then
        if [ -n "${MATRIX_NUM_ROLLOUT}" ]; then
            if [ "${TMAX_NUM_ROLLOUT:-}" != "${MATRIX_NUM_ROLLOUT}" ] || \
               [ "${TMAX_TARGET_ITER:-}" != "$((MATRIX_NUM_ROLLOUT - 1))" ]; then
                echo "ERROR: ${RUN_ID} trajectory bound must map TMAX_MATRIX_NUM_ROLLOUT to trainer and watcher state" >&2
                return 1
            fi
        elif [ -n "${TMAX_NUM_ROLLOUT:-}" ] || [ -n "${TMAX_TARGET_ITER:-}" ]; then
            echo "ERROR: ${RUN_ID} has an unexpected trajectory bound" >&2
            return 1
        fi
    elif [ -n "${TMAX_NUM_ROLLOUT:-}" ] || [ -n "${TMAX_TARGET_ITER:-}" ]; then
        echo "ERROR: ${RUN_ID} inherited the trajectory arm's rollout bound" >&2
        return 1
    fi
    if [ "${POLAR_APPTAINER_PERSISTENT_BROKER}" -ne 0 ] || \
       [ "${POLAR_APPTAINER_NO_INSTANCE}" -ne 1 ] || \
       [ "${POLAR_APPTAINER_DIRECT_EXEC_RETRIES}" -ne 3 ]; then
        echo "ERROR: ${RUN_ID} must use fresh direct Apptainer exec with three retries" >&2
        return 1
    fi
    if [ "${active_sessions}" -lt "${minimum_sessions}" ]; then
        echo "ERROR: ${RUN_ID} has only ${active_sessions}/${minimum_sessions} required active sessions" >&2
        return 1
    fi
    if [ "${POLAR_MAX_POSTRUN_WORKERS}" -ne "${expected_postrun_workers}" ]; then
        echo "ERROR: ${RUN_ID} postrun workers ${POLAR_MAX_POSTRUN_WORKERS} != required ${expected_postrun_workers}" >&2
        return 1
    fi
    case "${POLAR_AGENT_MODEL_NAME}" in
        *4B) expected_train_agent_timeout=600 ;;
        *) expected_train_agent_timeout=1200 ;;
    esac
    if [ "${TMAX_TRAIN_AGENT_TIMEOUT_SECONDS}" -ne "${expected_train_agent_timeout}" ]; then
        echo "ERROR: ${RUN_ID} training agent timeout must be ${expected_train_agent_timeout}s" >&2
        return 1
    fi
    if [ "${POLAR_TASK_TIMEOUT_FLOOR_SECONDS}" -lt \
         "$((TMAX_TRAIN_AGENT_TIMEOUT_SECONDS + TMAX_TRAIN_TASK_TIMEOUT_RESERVE_SECONDS))" ]; then
        echo "ERROR: ${RUN_ID} task timeout floor must cover the training agent timeout and reserve" >&2
        return 1
    fi
    if [ "${POLAR_REQUEST_TIMEOUT}" -lt "${POLAR_TASK_TIMEOUT_FLOOR_SECONDS}" ]; then
        echo "ERROR: ${RUN_ID} request timeout must cover the task timeout floor" >&2
        return 1
    fi
    if [ "${POLAR_MULTI_GATEWAY}" -eq 1 ] && { \
       [ "$((POLAR_MAX_INIT_WORKERS % NUM_NODES))" -ne 0 ] || \
       [ "$((POLAR_MAX_RUN_WORKERS % NUM_NODES))" -ne 0 ] || \
       [ "$((POLAR_MAX_POSTRUN_WORKERS % NUM_NODES))" -ne 0 ] || \
       [ "$((POLAR_COMPLETION_QUEUE_SIZE % NUM_NODES))" -ne 0 ] || \
       [ "$((POLAR_COMPLETION_WRITE_WORKERS % NUM_NODES))" -ne 0 ]; \
    }; then
        echo "ERROR: ${RUN_ID} fleet capacities do not split exactly across ${NUM_NODES} gateways" >&2
        return 1
    fi
}

print_plan_header() {
    echo "TMax mixed 4/8-node x 8-GPU matrix (read-only plan)"
    echo "  stamp:       ${MATRIX_STAMP}"
    echo "  W&B:         ${TMAX_MATRIX_WANDB_PROJECT:-polar-tmax-grpo} / ${MATRIX_WANDB_GROUP}"
    echo "  dependency:  ${MATRIX_DEPENDENCY:-none}"
    echo "  exclude:     ${TMAX_MATRIX_EXCLUDE_NODES:-none}"
    echo "  train/eval:  14,498 train (14,598 ready minus fixed 100), TMax holdout pass@1 only; 4n eval every 20 steps, 8n every 10"
    echo "  data source: ${MATRIX_SOURCE_RUN}"
    echo "  filtering:   trajectory-level reward variance; standard 4B early-stop=50%; trajectory arm requires full group8"
    echo "  timeouts:    4B train agent 600s; 9B 1200s; eval per-row caps unchanged; task floor 1800s; HTTP envelope 3600s"
    echo "  runtime:     4n single gateway; 8n one gateway per node; fresh direct Apptainer exec; no broker/instance; retries=3"
    printf '\n%-31s %-4s %-8s %-5s %-10s %-7s %-9s %-10s %-17s %-17s\n' \
        setting model lr async batch max_tok trainer rollout fleet_workers gateway_workers
}

print_setting_plan() {
    local setting="$1" model actor_gpus engines per_init per_run per_post
    configure_setting "${setting}"
    validate_configured_setting "${setting}"
    actor_gpus="$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))"
    engines="$((ROLLOUT_NUM_GPUS / ROLLOUT_NUM_GPUS_PER_ENGINE))"
    if [ "${POLAR_MULTI_GATEWAY}" -eq 1 ]; then
        per_init="$((POLAR_MAX_INIT_WORKERS / NUM_NODES))"
        per_run="$((POLAR_MAX_RUN_WORKERS / NUM_NODES))"
        per_post="$((POLAR_MAX_POSTRUN_WORKERS / NUM_NODES))"
    else
        per_init="${POLAR_MAX_INIT_WORKERS}"
        per_run="${POLAR_MAX_RUN_WORKERS}"
        per_post="${POLAR_MAX_POSTRUN_WORKERS}"
    fi
    case "${POLAR_AGENT_MODEL_NAME}" in
        *4B) model=4B ;;
        *) model=9B ;;
    esac
    printf '%-31s %-4s %-8s %-5s %-10s %-7s %2sGPU/tp%-2s %2sGPU/%2seng %-17s %-17s\n' \
        "${setting}" "${model}" "${TRAIN_LR}" "${POLAR_MAX_ASYNC_LEVEL}" \
        "${ROLLOUT_BATCH_SIZE}x${N_SAMPLES_PER_PROMPT}/${NUM_STEPS_PER_ROLLOUT}=${GLOBAL_BATCH_SIZE}" \
        "${MAX_TOKENS_PER_GPU}" "${actor_gpus}" "${ACTOR_TENSOR_MODEL_PARALLEL_SIZE}" \
        "${ROLLOUT_NUM_GPUS}" "${engines}" \
        "${POLAR_MAX_INIT_WORKERS}/${POLAR_MAX_RUN_WORKERS}/${POLAR_MAX_POSTRUN_WORKERS}" \
        "${per_init}/${per_run}/${per_post}"
    echo "  tokens=turn${ROLLOUT_MAX_RESPONSE_LEN}/total_response${TMAX_MAX_TOTAL_RESPONSE_LEN}+prompt${ROLLOUT_MAX_PROMPT_LEN}=pack${SEQ_LENGTH} trainer=tp${ACTOR_TENSOR_MODEL_PARALLEL_SIZE}/cp${CONTEXT_PARALLEL_SIZE}/sp${SEQUENCE_PARALLEL} cap=$((MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE))"
    echo "  eval_interval=${TMAX_EVAL_INTERVAL} resume_seed_eval=${TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN} scheduler_override=${TMAX_OVERRIDE_OPT_PARAM_SCHEDULER} load_dir=${LOAD_DIR:-release} num_rollout=${TMAX_NUM_ROLLOUT:-epoch-derived} target_iter=${TMAX_TARGET_ITER:-epoch-derived}"
    echo "  RUN_ID=${RUN_ID}"
}

ensure_fresh_run() {
    local run_dir="${POLAR_DATA_ROOT}/runs/${RUN_ID}"
    if [ -e "${run_dir}" ] || [ -e "${SAVE_DIR}" ]; then
        echo "ERROR: refusing to reuse non-fresh run paths for ${RUN_ID}" >&2
        echo "  run_dir=${run_dir}" >&2
        echo "  save_dir=${SAVE_DIR}" >&2
        return 1
    fi
}

submit_setting() {
    local setting="$1"
    (
        configure_setting "${setting}"
        validate_configured_setting "${setting}"
        ensure_fresh_run
        echo "[matrix] submitting ${setting}: ${RUN_ID}"
        bash "${SUBMIT_SCRIPT}"
    )
}

ACTION="${1:-plan}"
if [ "$#" -gt 0 ]; then
    shift
fi
case "${ACTION}" in
    -h|--help|help)
        usage
        exit 0
        ;;
    plan|submit) ;;
    *)
        echo "ERROR: action must be plan or submit, got ${ACTION}" >&2
        usage >&2
        exit 2
        ;;
esac

MATRIX_STAMP="${TMAX_MATRIX_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
if ! [[ "${MATRIX_STAMP}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "ERROR: unsafe TMAX_MATRIX_STAMP=${MATRIX_STAMP}" >&2
    exit 2
fi
MATRIX_DATA_ROOT="${POLAR_DATA_ROOT:-${SPILOT_ROOT}/data}"
MATRIX_SOURCE_RUN="${TMAX_MATRIX_SOURCE_RUN:-${MATRIX_DATA_ROOT}/runs/tmax-14598r-14498t100h-20260701T011143Z}"
MATRIX_WANDB_GROUP="${TMAX_MATRIX_WANDB_GROUP:-tmax-fidelity-matrix-${MATRIX_STAMP}}"
MATRIX_DEPENDENCY="${TMAX_MATRIX_DEPENDENCY:-}"
MATRIX_QWEN4_LOAD_DIR="${TMAX_MATRIX_QWEN4_LOAD_DIR:-}"
MATRIX_NUM_ROLLOUT="${TMAX_MATRIX_NUM_ROLLOUT:-}"
if [ -n "${MATRIX_DEPENDENCY}" ] && \
   ! [[ "${MATRIX_DEPENDENCY}" =~ ^afterok:[0-9]+(:[0-9]+)*$ ]]; then
    echo "ERROR: TMAX_MATRIX_DEPENDENCY must match afterok:<jobid>[:<jobid>...], got ${MATRIX_DEPENDENCY}" >&2
    exit 2
fi
if [ -n "${MATRIX_NUM_ROLLOUT}" ] && \
   ! [[ "${MATRIX_NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TMAX_MATRIX_NUM_ROLLOUT must be a positive integer, got ${MATRIX_NUM_ROLLOUT}" >&2
    exit 2
fi
MATRIX_TRAIN_SHA256="${TMAX_MATRIX_TRAIN_SHA256:-96a1c5929de64516eecc8a7b7ae012ccb888a2d575f15e28b8806ae6804826c8}"
MATRIX_HOLDOUT_SHA256="${TMAX_MATRIX_HOLDOUT_SHA256:-b1fe3e3311c66370c62f73272198c557f6afd3774a456dd1008e35d0153cbec3}"
MATRIX_EVAL_BUNDLE_SHA256="${TMAX_MATRIX_EVAL_BUNDLE_SHA256:-b1fe3e3311c66370c62f73272198c557f6afd3774a456dd1008e35d0153cbec3}"
QWEN4_REF_LOAD="${TMAX_MATRIX_QWEN4_REF_LOAD:-${USER_ROOT}/spilot-router/data/checkpoints/Qwen3.5-4B_torch_dist}"
QWEN9_HF_CHECKPOINT="${TMAX_MATRIX_QWEN9_HF_CHECKPOINT:-${MATRIX_DATA_ROOT}/checkpoints/Qwen3.5-9B}"
QWEN9_REF_LOAD="${TMAX_MATRIX_QWEN9_REF_LOAD:-${MATRIX_DATA_ROOT}/checkpoints/Qwen3.5-9B_torch_dist}"

verify_jsonl train "${MATRIX_SOURCE_RUN}/tmax-train.jsonl" 14498 "${MATRIX_TRAIN_SHA256}"
verify_jsonl tmax_holdout "${MATRIX_SOURCE_RUN}/tmax_holdout-eval.jsonl" 100 "${MATRIX_HOLDOUT_SHA256}"
require_file "Qwen3.5-4B release marker" "${QWEN4_REF_LOAD}/latest_checkpointed_iteration.txt"
if [ -n "${MATRIX_QWEN4_LOAD_DIR}" ]; then
    if ! _matrix_seed_iter="$(tmax_validate_numbered_checkpoint \
        "${MATRIX_QWEN4_LOAD_DIR}" "ERROR: Qwen3.5-4B continuation")"; then
        exit 1
    fi
    unset _matrix_seed_iter
fi
require_file "Qwen3.5-9B config" "${QWEN9_HF_CHECKPOINT}/config.json"
require_file "Qwen3.5-9B release marker" "${QWEN9_REF_LOAD}/latest_checkpointed_iteration.txt"
require_file "Qwen3.5-4B model args" "${PROJECT_ROOT}/examples/swegym_slime_grpo/model_args.sh"
require_file "Qwen3.5-9B model args" "${SCRIPT_DIR}/model_args.sh"

declare -a SELECTED_SETTINGS
select_settings SELECTED_SETTINGS "$@"

if [ "${ACTION}" = plan ]; then
    print_plan_header
    for setting in "${SELECTED_SETTINGS[@]}"; do
        print_setting_plan "${setting}"
    done
    echo
    echo "No jobs were submitted. To submit this exact plan, reuse:"
    printf '  TMAX_MATRIX_STAMP=%q' "${MATRIX_STAMP}"
    if [ -n "${MATRIX_DEPENDENCY}" ]; then
        printf ' TMAX_MATRIX_DEPENDENCY=%q' "${MATRIX_DEPENDENCY}"
    fi
    if [ -n "${MATRIX_QWEN4_LOAD_DIR}" ]; then
        printf ' TMAX_MATRIX_QWEN4_LOAD_DIR=%q' "${MATRIX_QWEN4_LOAD_DIR}"
    fi
    if [ -n "${MATRIX_NUM_ROLLOUT}" ]; then
        printf ' TMAX_MATRIX_NUM_ROLLOUT=%q' "${MATRIX_NUM_ROLLOUT}"
    fi
    printf ' TMAX_MATRIX_CONFIRM=%q bash %q submit' "${CONFIRM_TOKEN}" "$0"
    printf ' %q' "${SELECTED_SETTINGS[@]}"
    printf '\n'
    exit 0
fi

if [ "$#" -eq 0 ]; then
    echo "ERROR: submit requires explicit 'all' or at least one setting" >&2
    exit 2
fi
if [ "${TMAX_MATRIX_CONFIRM:-}" != "${CONFIRM_TOKEN}" ]; then
    echo "ERROR: submission is locked; set TMAX_MATRIX_CONFIRM=${CONFIRM_TOKEN}" >&2
    exit 2
fi
for setting in "${SELECTED_SETTINGS[@]}"; do
    submit_setting "${setting}"
done
