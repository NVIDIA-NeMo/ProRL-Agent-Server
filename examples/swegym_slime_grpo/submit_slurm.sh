#!/usr/bin/env bash
# Route A: submit SWE-Gym Slime-GRPO (Polar drives agents) as a Pyxis/Slurm job.
#
# Wraps stable's run.sh inside Pyxis training containers (your train.sqsh), one
# task per Slurm node. run.sh starts Polar + Ray head on rank 0, Ray workers on
# other ranks, then submits slime/train_async.py from rank 0;
# the Polar gateway spawns per-task apptainer SIF sandboxes with
# POLAR_APPTAINER_NO_INSTANCE=1 (ephemeral `apptainer exec --overlay`, the only
# mode that works nested in Pyxis).
#
#   source examples/swegym_slime_grpo/env.cwdfw.sh
#   bash   examples/swegym_slime_grpo/submit_slurm.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../path_safety.sh
source "${SCRIPT_DIR}/../path_safety.sh"
TRAIN_RUN_SCRIPT="${POLAR_TRAIN_RUN_SCRIPT:-${SCRIPT_DIR}/run.sh}"
if [ ! -f "${TRAIN_RUN_SCRIPT}" ]; then
    echo "ERROR: training run script not found: ${TRAIN_RUN_SCRIPT}" >&2
    exit 1
fi

# ── Slurm ────────────────────────────────────────────────────────────
ACCOUNT="${ACCOUNT:-nvr_lpr_llm}"
PARTITION="${PARTITION:-interactive}"
WALL_TIME="${WALL_TIME:-4:00:00}"
NUM_NODES="${NUM_NODES:-4}"
JOB_NAME="${JOB_NAME:-polar-swegym-grpo}"
SLURM_GPUS="${SLURM_GPUS:-8}"
SLURM_CONSTRAINT="${SLURM_CONSTRAINT:-}"   # e.g. H100/H200/B200
SLURM_EXCLUDE="${SLURM_EXCLUDE:-}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"
POLAR_SLURM_MEM_PER_NODE="${POLAR_SLURM_MEM_PER_NODE:-0}"
if ! [[ "${POLAR_SLURM_MEM_PER_NODE}" =~ ^(0|[1-9][0-9]*[KMGTP]?)$ ]]; then
    echo "ERROR: POLAR_SLURM_MEM_PER_NODE must be 0 or a positive Slurm memory value" >&2
    exit 1
fi

# ── Training container ───────────────────────────────────────────────
TRAIN_SQSH="${POLR_TRAIN_SQSH:?set POLR_TRAIN_SQSH (your train.sqsh; build via build_training_sqsh.sh)}"
TRAIN_VENV="${POLR_TRAIN_VENV:-/opt/polr_venv}"
TRAIN_MOUNTS="${TRAIN_CONTAINER_MOUNTS:-/lustre/fsw:/lustre/fsw}"
APPT_BIN="${POLAR_APPTAINER_BIN:-/usr/bin/apptainer}"
NO_INSTANCE="${POLAR_APPTAINER_NO_INSTANCE:-1}"

# ── Route-A assets (run.sh reads these for envsubst of topology/polar_config) ──
APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:?set APPTAINER_IMAGE_DIR (dir of <instance_id>.sif from build_sifs.py)}"
AGENT_CLI_DIR="${AGENT_CLI_DIR:?set AGENT_CLI_DIR (Node22+agent-CLI tree -> /opt/node)}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
HF_HOME="${HF_HOME:-}"
HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3.5-4B}"
DATA_ROOT="${POLAR_DATA_ROOT:-${PROJECT_ROOT}/tmp}"   # runs/ckpt land here (set POLAR_DATA_ROOT via env.cwdfw.sh)
EXPERIMENT_NAME="${EXPERIMENT_NAME:-swegym-slime-grpo-qwen35-4b-4n8h100}"
RUN_ID="${RUN_ID:-${EXPERIMENT_NAME}}"
SAVE_DIR="${SAVE_DIR:-${DATA_ROOT}/ckpt/${RUN_ID}}"
RUN_DIR="${RUN_DIR:-}"
POLAR_ROLLOUT_SAVE_DIR="${POLAR_ROLLOUT_SAVE_DIR:-}"

if [ ! -f "${MEGATRON_DIR}/megatron/training/tokenizer/tokenizer.py" ]; then
    echo "ERROR: slime requires a Megatron checkout with megatron.training.tokenizer: ${MEGATRON_DIR}" >&2
    echo "  Use the 26.04-alpha-compatible checkout prepared by launch_e2e.sh." >&2
    exit 1
fi

# Accept EITHER a local .sqsh path OR a docker/registry ref (pyxis enroot-imports
# the latter — e.g. flappydora/ubuntu22.04-cuda13.3:latest, what `igpu` uses).
case "$TRAIN_SQSH" in
    /*|./*|../*)
        if [ ! -f "$TRAIN_SQSH" ]; then
            echo "ERROR: container image file not found: ${TRAIN_SQSH}" >&2
            echo "  use a docker ref (e.g. flappydora/ubuntu22.04-cuda13.3:latest) or build a .sqsh via build_training_sqsh.sh" >&2
            exit 1
        fi ;;
    *) : ;;  # docker:// or registry ref — leave it to pyxis/enroot to resolve
esac
LOG_DIR="${POLAR_SLURM_LOG_DIR:-${DATA_ROOT}/logs/slurm}"
polar_require_absolute_path POLAR_SLURM_LOG_DIR "${LOG_DIR}"
mkdir -p "${LOG_DIR}"
POLAR_LAUNCHER_LABEL="${POLAR_LAUNCHER_LABEL:-Polar SWE-Gym Slime-GRPO (Route A)}"

echo "============================================="
echo "${POLAR_LAUNCHER_LABEL} — SLURM submission"
echo "  Train sqsh: ${TRAIN_SQSH}"
echo "  SIF dir:    ${APPTAINER_IMAGE_DIR}   (<instance_id>.sif)"
echo "  apptainer:  ${APPT_BIN}  NO_INSTANCE=${NO_INSTANCE}"
echo "  slime:      ${SLIME_DIR}"
echo "  account/part/-C/nodes/gpus: ${ACCOUNT}/${PARTITION}/${SLURM_CONSTRAINT:-none}/${NUM_NODES}/${SLURM_GPUS}"
echo "  dependency: ${SBATCH_DEPENDENCY:-none}"
echo "  run/save:  ${RUN_ID} / ${SAVE_DIR}"
echo "============================================="

# Export the runtime contract once; the static node entrypoint keeps the Slurm
# step command short enough for multi-node launch RPCs.
export POLAR_TRAIN_PROJECT_ROOT="${PROJECT_ROOT}"
export POLAR_TRAIN_RUN_SCRIPT="${TRAIN_RUN_SCRIPT}"
export POLR_TRAIN_VENV="${TRAIN_VENV}"
export POLAR_DATA_ROOT="${DATA_ROOT}"
export POLAR_APPTAINER_BIN="${APPT_BIN}"
export POLAR_APPTAINER_NO_INSTANCE="${NO_INSTANCE}"
export POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-2}"
export POLAR_FULLY_ASYNC="${POLAR_FULLY_ASYNC:-false}"
export POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-2400}"
export POLAR_TASK_TIMEOUT_SECONDS="${POLAR_TASK_TIMEOUT_SECONDS:-2400}"
export POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.6}"
export POLAR_APPTAINER_DIRECT_EXEC_RETRIES="${POLAR_APPTAINER_DIRECT_EXEC_RETRIES:-3}"
export APPTAINER_IMAGE_DIR AGENT_CLI_DIR SLIME_DIR MEGATRON_DIR HF_HOME HF_CHECKPOINT
export POLAR_AGENT_HARNESS="${POLAR_AGENT_HARNESS:-codex}"
export POLAR_AGENT_MODEL_NAME="${POLAR_AGENT_MODEL_NAME:-gpt-5.4}"
export POLAR_AGENT_PATH="${POLAR_AGENT_PATH:-/opt/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export POLAR_AGENT_RUNTIME_VOLUME="${POLAR_AGENT_RUNTIME_VOLUME:-}"
export POLAR_AGENT_STEP_LIMIT="${POLAR_AGENT_STEP_LIMIT:-30}"
export POLAR_AGENT_COST_LIMIT="${POLAR_AGENT_COST_LIMIT:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-polar-swegym-grpo}"
export WANDB_GROUP="${WANDB_GROUP:-swegym-qwen35-4b-async-grpo}"
export EXPERIMENT_NAME RUN_ID SAVE_DIR RUN_DIR SLURM_GPUS
export POLAR_ROLLOUT_SAVE_DIR
export RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-${SLURM_GPUS}}"
CONTAINER_ENTRYPOINT="${SCRIPT_DIR}/run_in_container.sh"
if [ ! -f "${CONTAINER_ENTRYPOINT}" ]; then
    echo "ERROR: container entrypoint not found: ${CONTAINER_ENTRYPOINT}" >&2
    exit 1
fi

# Do not propagate the submit shell's SLURM/PMIx state into a new allocation.
# Persist only the training contract and pass one short env-file pointer through
# sbatch; the file is private because it can contain W&B/HF credentials.
export SLIME_SUBMIT_UNIX_NS="$(date +%s%N)"
TRAIN_ENV_DIR="${DATA_ROOT}/runs/${RUN_ID}/submit"
mkdir -p "${TRAIN_ENV_DIR}"
POLAR_TRAIN_ENV_FILE="${TRAIN_ENV_DIR}/env-$(date -u +%Y%m%dT%H%M%SZ)-$$.sh"
TRAIN_ENV_TMP="${POLAR_TRAIN_ENV_FILE}.tmp"
umask 077
{
    while IFS= read -r name; do
        case "$name" in
            POLAR_*|POLR_*|TMAX_*|MINI_SWE_*|WANDB_*|HF_*|HUGGINGFACE_*|\
            APPTAINER_ENV|\
            SPILOT_EPISODE_ADMISSION_ENABLED|SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS|\
            SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT|\
            SPILOT_QWEN_COST_WEIGHT|SPILOT_GPT_COST_WEIGHT|\
            SPILOT_COST_PENALTY_LAMBDA|SPILOT_COST_NORMALIZER|\
            SPILOT_QWEN_MAX_ACTIVE_EPISODES|SPILOT_GPT_MAX_ACTIVE_EPISODES|\
            SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES|SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES|\
            SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY|SPILOT_GPT_GATEWAY_MAX_CONCURRENCY|\
            SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES|SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES|\
            ACTOR_*|EXPERT_*|ROLLOUT_*|RAY_NUM_*|RAY_LAST_NODE_NUM_GPUS|GPU_MONITOR_*|SGLANG_*|FLASHINFER_*|\
            APPTAINER_IMAGE_DIR|AGENT_CLI_DIR|TRAIN_CONTAINER_MOUNTS|\
            SLIME_DIR|SLIME_ROLLOUT_BASE_PORT|SLIME_ROLLOUT_BASE_PORT_FALLBACK|\
            SLIME_EPHEMERAL_PORT_LOWER_BOUND|SLIME_IP_LOCAL_PORT_RANGE_PATH|\
            SLIME_GRACEFUL_EXIT_AT_UNIX_TIME|SLIME_PROFILE_CUDA_PHASES|\
            SLIME_SUBMIT_UNIX_NS|MEGATRON_DIR|REF_LOAD|TORCH_DIST_DIR|MODEL_ARGS_FILE|\
            ACCOUNT|PARTITION|NUM_NODES|WALL_TIME|CPUS_PER_TASK|SLURM_GPUS|\
            N_SAMPLES_PER_PROMPT|NUM_STEPS_PER_ROLLOUT|NUM_EPOCH|\
            SEQ_LENGTH|MAX_TOKENS_PER_GPU|LOG_PROBS_CHUNK_SIZE|SAVE_INTERVAL|SAVE_RETAIN_INTERVAL|\
            SAVE_HF_ENABLED|SAVE_MEGATRON|SAVE_HF_TEMPLATE|SEQUENCE_PARALLEL|CONTEXT_PARALLEL_SIZE|\
            TRAIN_LR|KL_LOSS_COEF|POLICY_LOSS_TYPE|USE_TIS|GRPO_STD_NORMALIZATION|\
            DPPO_DIVERGENCE_TYPE|DPPO_DIVERGENCE_THRESHOLD|\
            MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF|\
            CALCULATE_PER_TOKEN_LOSS|\
            DIST_CKPT_STRICTNESS|ATTENTION_BACKEND|\
            GLOBAL_BATCH_SIZE|EVAL_GLOBAL_BATCH_SIZE|EXPERIMENT_NAME|RUN_ID|\
            SAVE_DIR|LOAD_DIR|RUN_DIR|PROMPT_DATA|REQUIRE_SWEGYM_HARNESS|\
            OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS|\
            NUMEXPR_NUM_THREADS|TOKENIZERS_PARALLELISM|\
            TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD|\
            http_proxy|https_proxy|HTTP_PROXY|HTTPS_PROXY|no_proxy|NO_PROXY)
                printf 'export %s=%q\n' "$name" "${!name}"
                ;;
        esac
    done < <(compgen -e | LC_ALL=C sort)
} >"${TRAIN_ENV_TMP}"
mv "${TRAIN_ENV_TMP}" "${POLAR_TRAIN_ENV_FILE}"
chmod 600 "${POLAR_TRAIN_ENV_FILE}"
export POLAR_TRAIN_ENV_FILE

printf -v SQSH_Q  '%q' "$TRAIN_SQSH"
printf -v MNT_Q   '%q' "$TRAIN_MOUNTS"
printf -v PR_Q    '%q' "$PROJECT_ROOT"
printf -v ENTRY_Q '%q' "$CONTAINER_ENTRYPOINT"
printf -v LOG_Q   '%q' "$LOG_DIR"
SRUN_BIN="$(command -v srun)"
printf -v SRUN_Q '%q' "$SRUN_BIN"
if [ "${POLAR_APPTAINER_JOB_SESSION_MOUNT:-0}" = "1" ]; then
    APPTAINER_SESSIONDIR="${POLAR_APPTAINER_SESSIONDIR:?set POLAR_APPTAINER_SESSIONDIR}"
    printf -v APPTAINER_SESSIONDIR_Q '%q' "${APPTAINER_SESSIONDIR}"
    APPTAINER_SESSION_SETUP="APPTAINER_SESSION_HOST=/tmp/polar-apptainer-session-\${SLURM_JOB_ID}-\${UID}; ${SRUN_Q} --overlap --nodes=${NUM_NODES} --ntasks=${NUM_NODES} --ntasks-per-node=1 --cpus-per-task=1 --cpu-bind=none bash -c 'set -euo pipefail; root=\"\$1\"; if [ -e \"\${root}\" ] || [ -L \"\${root}\" ]; then echo \"ERROR: refusing existing Apptainer session root: \${root}\" >&2; exit 1; fi; install -d -m 700 -- \"\${root}\"' bash \"\${APPTAINER_SESSION_HOST}\"; MNT_Q_WITH_SESSION=${MNT_Q},\${APPTAINER_SESSION_HOST}:${APPTAINER_SESSIONDIR_Q}:rw"
    MNT_ARG='${MNT_Q_WITH_SESSION}'
else
    APPTAINER_SESSION_SETUP=:
    MNT_ARG="${MNT_Q}"
fi
CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
SLURM_STEP_CPUS_PER_TASK="${SLURM_STEP_CPUS_PER_TASK:-96}"
# The job allocation already owns all requested CPU/GPU/memory TRES. Repeating
# the entire 128-CPU allocation on an overlapping step makes the NVIDIA select
# plugin reject step creation. Leave CPU headroom for the batch shell and let
# the step inherit job-level GPU and memory TRES.
WRAP_CMD="umask 077; chmod 600 ${LOG_Q}/\"\${SLURM_JOB_NAME}-\${SLURM_JOB_ID}.out\" ${LOG_Q}/\"\${SLURM_JOB_NAME}-\${SLURM_JOB_ID}.err\" 2>/dev/null || true; export SLIME_SLURM_BATCH_START_UNIX_NS=\$(date +%s%N); ${APPTAINER_SESSION_SETUP}; ${SRUN_Q} --overlap --nodes=${NUM_NODES} --ntasks=${NUM_NODES} --ntasks-per-node=1 --cpus-per-task=${SLURM_STEP_CPUS_PER_TASK} --cpu-bind=none --kill-on-bad-exit=1 --container-image=${SQSH_Q} --container-mounts=${MNT_ARG} --container-workdir=${PR_Q} --container-writable --no-container-mount-home bash ${ENTRY_Q}"

SBATCH_CONSTRAINT_ARG=()
if [ -n "${SLURM_CONSTRAINT}" ]; then
    SBATCH_CONSTRAINT_ARG=(--constraint="${SLURM_CONSTRAINT}")
fi
SBATCH_DEPENDENCY_ARG=()
if [ -n "${SBATCH_DEPENDENCY}" ]; then
    SBATCH_DEPENDENCY_ARG=(--dependency="${SBATCH_DEPENDENCY}")
fi
SBATCH_EXCLUDE_ARG=()
if [ -n "${SLURM_EXCLUDE}" ]; then
    SBATCH_EXCLUDE_ARG=(--exclude="${SLURM_EXCLUDE}")
fi
# Cluster policy hooks (e.g. the OccupiedIdleGPUsJobReaper exemption JSON)
# ride on the job comment. Rollout phases of pool-routed RL runs hold GPUs
# at low utilization while remote model calls execute, so those runs must
# declare an exemption window instead of being reaped mid-rollout.
# Default to the idle-GPU-reaper exemption when the caller did not set
# TMAX_SBATCH_COMMENT at all; an explicitly EMPTY value opts out (no comment).
# ${VAR-default} (no colon) keeps the set-but-empty case distinct from unset.
_TMAX_DEFAULT_REAPER_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"interactive","description":"Interactive and debugging sessions"}}'
TMAX_SBATCH_COMMENT="${TMAX_SBATCH_COMMENT-${_TMAX_DEFAULT_REAPER_COMMENT}}"
unset _TMAX_DEFAULT_REAPER_COMMENT
SBATCH_COMMENT_ARG=()
if [ -n "${TMAX_SBATCH_COMMENT:-}" ]; then
    SBATCH_COMMENT_ARG=(--comment="${TMAX_SBATCH_COMMENT}")
fi

SUBMIT_BACKEND="${SUBMIT_BACKEND:-srun}"
if [ -n "${POLAR_SUBMIT_RECEIPT_FILE:-}" ] && [ "${SUBMIT_BACKEND}" != "sbatch" ]; then
    echo "ERROR: POLAR_SUBMIT_RECEIPT_FILE requires SUBMIT_BACKEND=sbatch" >&2
    exit 1
fi
if [ -n "${POLAR_SUBMIT_RECEIPT_FILE:-}" ] && [ "${SUBMIT_DRY_RUN:-0}" != "1" ]; then
    # Check writability before submitting; after sbatch succeeds, failure to
    # record its id would make an automated watcher unable to avoid duplicates.
    RECEIPT_DIR="$(dirname "${POLAR_SUBMIT_RECEIPT_FILE}")"
    RECEIPT_PROBE="${POLAR_SUBMIT_RECEIPT_FILE}.probe.$$"
    mkdir -p "${RECEIPT_DIR}"
    (umask 077; : >"${RECEIPT_PROBE}")
    rm -f "${RECEIPT_PROBE}"
fi
if [ "${SUBMIT_BACKEND}" = "srun" ]; then
    if [ -n "${SBATCH_DEPENDENCY}" ]; then
        echo "ERROR: SUBMIT_BACKEND=srun does not support SBATCH_DEPENDENCY" >&2
        exit 1
    fi
    TOTAL_GPUS=$((NUM_NODES * SLURM_GPUS))
    SRUN_LOG_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    SRUN_OUT="${SRUN_OUT:-${LOG_DIR}/${JOB_NAME}-srun-${SRUN_LOG_STAMP}.out}"
    SRUN_ERR="${SRUN_ERR:-${LOG_DIR}/${JOB_NAME}-srun-${SRUN_LOG_STAMP}.err}"
    SRUN_CMD=(
        "${SRUN_BIN}"
        "${SBATCH_COMMENT_ARG[@]}"
        --account="${ACCOUNT}"
        --job-name="${JOB_NAME}"
        --partition="${PARTITION}"
        --nodes="${NUM_NODES}"
        --ntasks="${NUM_NODES}"
        --ntasks-per-node=1
        --cpus-per-task="${CPUS_PER_TASK}"
        --gpus="${TOTAL_GPUS}"
        --mem="${POLAR_SLURM_MEM_PER_NODE}"
        --exclusive
        --time="${WALL_TIME}"
        --kill-on-bad-exit=1
        --container-image="${TRAIN_SQSH}"
        --container-mounts="${TRAIN_MOUNTS}"
        --container-workdir="${PROJECT_ROOT}"
        --container-writable
        --no-container-mount-home
    )
    if [ -n "${SLURM_CONSTRAINT}" ]; then
        SRUN_CMD+=(--constraint="${SLURM_CONSTRAINT}")
    fi
    if [ -n "${SLURM_EXCLUDE}" ]; then
        SRUN_CMD+=(--exclude="${SLURM_EXCLUDE}")
    fi
    SRUN_CMD+=(bash "${CONTAINER_ENTRYPOINT}")
    # A submit shell may itself live inside an interactive/CPU allocation.
    # Clear parent step identity so this srun requests a fresh GPU allocation.
    SRUN_LAUNCH=(
        env
        -u SLURM_JOB_ID
        -u SLURM_JOBID
        -u SLURM_STEP_ID
        -u SLURM_STEPID
        -u SLURM_PROCID
        -u SLURM_LOCALID
        -u SLURM_NODEID
        -u SLURM_NTASKS
        -u SLURM_NNODES
        -u SLURM_JOB_NUM_NODES
        -u SLURM_JOB_NODELIST
        -u SLURM_NODELIST
        "${SRUN_CMD[@]}"
    )

    if [ "${SUBMIT_DRY_RUN:-0}" = "1" ]; then
        echo "Dry run only; top-level command:"
        printf ' %q' "${SRUN_LAUNCH[@]}"
        printf '\n'
        exit 0
    fi

    export SLIME_SLURM_BATCH_START_UNIX_NS="$(date +%s%N)"
    nohup "${SRUN_LAUNCH[@]}" >"${SRUN_OUT}" 2>"${SRUN_ERR}" </dev/null &
    SRUN_PID=$!
    echo ""
    echo "Submitted top-level SRUN pid: ${SRUN_PID}"
    echo "Monitor stdout: ${SRUN_OUT}"
    echo "Monitor stderr: ${SRUN_ERR}"
    exit 0
fi

if [ "${SUBMIT_DRY_RUN:-0}" = "1" ]; then
    echo "Dry run only; sbatch wrapper:"
    printf '%s\n' "${WRAP_CMD}"
    exit 0
fi

# Slurm propagates SLURM_* variables even with a restrictive --export list.
# When a CPU watcher submits this GPU allocation, an inherited
# SLURM_MEM_PER_NODE (for example 2 GiB) is otherwise consumed by the nested
# srun below and shrinks that step to the watcher's memory limit even though
# the new job has its own explicit POLAR_SLURM_MEM_PER_NODE allocation.
JOB_ID=$(env \
    -u SLURM_JOB_ID \
    -u SLURM_JOBID \
    -u SLURM_STEP_ID \
    -u SLURM_STEPID \
    -u SLURM_PROCID \
    -u SLURM_LOCALID \
    -u SLURM_NODEID \
    -u SLURM_NTASKS \
    -u SLURM_NNODES \
    -u SLURM_JOB_NUM_NODES \
    -u SLURM_JOB_NODELIST \
    -u SLURM_NODELIST \
    -u SLURM_GPUS \
    -u SLURM_CONSTRAINT \
    -u SLURM_STEP_CPUS_PER_TASK \
    -u SLURM_MEM_PER_NODE \
    -u SLURM_MEM_PER_CPU \
    -u SLURM_MEM_PER_GPU \
    -u SBATCH_MEM_PER_NODE \
    -u SBATCH_MEM_PER_CPU \
    -u SBATCH_MEM_PER_GPU \
    sbatch \
    --nodes="${NUM_NODES}" \
    --ntasks="${NUM_NODES}" \
    --ntasks-per-node=1 \
    --account="${ACCOUNT}" \
    --job-name="${JOB_NAME}" \
    --partition="${PARTITION}" \
    --time="${WALL_TIME}" \
    --gres="gpu:${SLURM_GPUS}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${POLAR_SLURM_MEM_PER_NODE}" \
    "${SBATCH_CONSTRAINT_ARG[@]}" \
    "${SBATCH_DEPENDENCY_ARG[@]}" \
    "${SBATCH_EXCLUDE_ARG[@]}" \
    "${SBATCH_COMMENT_ARG[@]}" \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" \
    --export="POLAR_TRAIN_ENV_FILE=${POLAR_TRAIN_ENV_FILE}" \
    --parsable \
    --wrap="${WRAP_CMD}")
JOB_ID="${JOB_ID%%;*}"

if [ -n "${POLAR_SUBMIT_RECEIPT_FILE:-}" ]; then
    RECEIPT_TMP="${POLAR_SUBMIT_RECEIPT_FILE}.tmp.$$"
    SUBMITTED_AT_UNIX="$(date +%s)"
    umask 077
    {
        printf 'export POLAR_SUBMITTED_JOB_ID=%q\n' "${JOB_ID}"
        printf 'export POLAR_SUBMITTED_AT_UNIX=%q\n' "${SUBMITTED_AT_UNIX}"
    } >"${RECEIPT_TMP}"
    chmod 600 "${RECEIPT_TMP}"
    mv "${RECEIPT_TMP}" "${POLAR_SUBMIT_RECEIPT_FILE}"
fi

echo ""
echo "Submitted SLURM job: ${JOB_ID}"
echo "Monitor: tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out"
