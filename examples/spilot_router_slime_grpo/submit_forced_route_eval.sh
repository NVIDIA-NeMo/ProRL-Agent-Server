#!/usr/bin/env bash
# Submit a fresh, CPU-only paired forced-route benchmark allocation.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_forced_route_eval.sh"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_forced_route_eval.sh"

if [ -z "${POLAR_NVIDIA_API_KEY:-${NVIDIA_API_KEY:-}}" ] && \
   [ "${SPILOT_FORCED_EVAL_SUBMIT_RC_LOADED:-0}" != "1" ]; then
    ZSH_BIN="${ZSH_BIN:-/home/jiaruiy/local/bin/zsh}"
    if [ ! -x "${ZSH_BIN}" ]; then
        echo "ERROR: NVIDIA endpoint key is unset and zsh is unavailable at ${ZSH_BIN}" >&2
        exit 2
    fi
    export SPILOT_FORCED_EVAL_SUBMIT_RC_LOADED=1
    exec "${ZSH_BIN}" -lc \
        'source "$HOME/.zshrc" >/dev/null 2>&1; export POLAR_NVIDIA_API_KEY NVIDIA_API_KEY NVIDIA_BASE_URL POLAR_MODEL_POOL_BASE_URL http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; exec bash "$@"' \
        spilot-forced-eval-submit "${SUBMIT_SCRIPT}" "$@"
fi

POLAR_NVIDIA_API_KEY="${POLAR_NVIDIA_API_KEY:-${NVIDIA_API_KEY:-}}"
if [ -z "${POLAR_NVIDIA_API_KEY}" ]; then
    echo "ERROR: POLAR_NVIDIA_API_KEY or NVIDIA_API_KEY is required" >&2
    exit 2
fi
POLAR_MODEL_POOL_BASE_URL="${POLAR_MODEL_POOL_BASE_URL:-${NVIDIA_BASE_URL:-https://integrate.api.nvidia.com/v1}}"
POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${SPILOT_ROOT}/data}"
SRUN_BIN="${SRUN_BIN:-$(command -v srun || true)}"
if [[ "${SRUN_BIN}" != /* ]] || [ ! -x "${SRUN_BIN}" ]; then
    echo "ERROR: cannot resolve an absolute executable srun path" >&2
    exit 2
fi
export POLAR_NVIDIA_API_KEY POLAR_MODEL_POOL_BASE_URL POLAR_DATA_ROOT SRUN_BIN
unset POLAR_CONTROL_PLANE_TOKEN || true

OUTPUT_DIR=""
SERVICE_DIR=""
DATA_PATH=""
RUN_ID=""
MAX_TASKS=""
START_INDEX=0
RESUME=0
INCLUDE_QWEN35_BASELINE=0
REPLICATES=1
MAX_CONCURRENCY=4
MAX_PAID_ATTEMPTS_PER_WORK=1
AGENT_TIMEOUT_SECONDS=3300
ARGS=("$@")
index=0
while [ "${index}" -lt "${#ARGS[@]}" ]; do
    argument="${ARGS[${index}]}"
    case "${argument}" in
        --output-dir|--service-dir|--data|--run-id|--start-index|--max-tasks|--replicates|--max-concurrency|--max-paid-attempts-per-work|--agent-timeout-seconds)
            next=$((index + 1))
            if [ "${next}" -ge "${#ARGS[@]}" ]; then
                echo "ERROR: ${argument} requires a value" >&2
                exit 2
            fi
            value="${ARGS[${next}]}"
            case "${argument}" in
                --output-dir) OUTPUT_DIR="${value}" ;;
                --service-dir) SERVICE_DIR="${value}" ;;
                --data) DATA_PATH="${value}" ;;
                --run-id) RUN_ID="${value}" ;;
                --start-index) START_INDEX="${value}" ;;
                --max-tasks) MAX_TASKS="${value}" ;;
                --replicates) REPLICATES="${value}" ;;
                --max-concurrency) MAX_CONCURRENCY="${value}" ;;
                --max-paid-attempts-per-work) MAX_PAID_ATTEMPTS_PER_WORK="${value}" ;;
                --agent-timeout-seconds) AGENT_TIMEOUT_SECONDS="${value}" ;;
            esac
            index=$((index + 2))
            ;;
        --output-dir=*) OUTPUT_DIR="${argument#*=}"; index=$((index + 1)) ;;
        --service-dir=*) SERVICE_DIR="${argument#*=}"; index=$((index + 1)) ;;
        --data=*) DATA_PATH="${argument#*=}"; index=$((index + 1)) ;;
        --run-id=*) RUN_ID="${argument#*=}"; index=$((index + 1)) ;;
        --start-index=*) START_INDEX="${argument#*=}"; index=$((index + 1)) ;;
        --max-tasks=*) MAX_TASKS="${argument#*=}"; index=$((index + 1)) ;;
        --replicates=*) REPLICATES="${argument#*=}"; index=$((index + 1)) ;;
        --max-concurrency=*) MAX_CONCURRENCY="${argument#*=}"; index=$((index + 1)) ;;
        --max-paid-attempts-per-work=*) MAX_PAID_ATTEMPTS_PER_WORK="${argument#*=}"; index=$((index + 1)) ;;
        --agent-timeout-seconds=*) AGENT_TIMEOUT_SECONDS="${argument#*=}"; index=$((index + 1)) ;;
        --resume) RESUME=1; index=$((index + 1)) ;;
        --include-qwen35-baseline) INCLUDE_QWEN35_BASELINE=1; index=$((index + 1)) ;;
        *) index=$((index + 1)) ;;
    esac
done

if [ -z "${OUTPUT_DIR}" ] || [ -z "${DATA_PATH}" ] || [ -z "${RUN_ID}" ] || \
   [ -z "${MAX_TASKS}" ]; then
    echo "ERROR: --output-dir, --data, --run-id, and --max-tasks are required" >&2
    exit 2
fi
case "${OUTPUT_DIR}" in /*) ;; *) echo "ERROR: --output-dir must be absolute" >&2; exit 2 ;; esac
case "${DATA_PATH}" in /*) ;; *) echo "ERROR: --data must be absolute" >&2; exit 2 ;; esac
if [ -n "${SERVICE_DIR}" ]; then
    case "${SERVICE_DIR}" in /*) ;; *) echo "ERROR: --service-dir must be absolute" >&2; exit 2 ;; esac
else
    if [ "${RESUME}" = "1" ]; then
        SERVICE_DIR="${OUTPUT_DIR}.service.resume.$(date -u +%Y%m%dT%H%M%SZ).$$"
        ARGS+=(--service-dir "${SERVICE_DIR}")
    else
        SERVICE_DIR="${OUTPUT_DIR}.service"
    fi
fi
if ! [[ "${RUN_ID}" =~ ^[0-9A-Za-z][0-9A-Za-z._-]{0,79}$ ]]; then
    echo "ERROR: --run-id must match [A-Za-z0-9][A-Za-z0-9._-]{0,79}" >&2
    exit 2
fi
if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: evaluation data does not exist: ${DATA_PATH}" >&2
    exit 2
fi

for assignment in \
    "max-tasks:${MAX_TASKS}" \
    "replicates:${REPLICATES}" \
    "max-concurrency:${MAX_CONCURRENCY}" \
    "max-paid-attempts-per-work:${MAX_PAID_ATTEMPTS_PER_WORK}" \
    "agent-timeout-seconds:${AGENT_TIMEOUT_SECONDS}"; do
    name="${assignment%%:*}"
    value="${assignment#*:}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: --${name} must be a positive integer" >&2
        exit 2
    fi
done
if ! [[ "${START_INDEX}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "ERROR: --start-index must be a non-negative integer" >&2
    exit 2
fi
FORCED_EVAL_SLURM_MARGIN_SECONDS="${FORCED_EVAL_SLURM_MARGIN_SECONDS:-1800}"
if ! [[ "${FORCED_EVAL_SLURM_MARGIN_SECONDS}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "ERROR: FORCED_EVAL_SLURM_MARGIN_SECONDS must be a non-negative integer" >&2
    exit 2
fi

PREFLIGHT_PYTHON="${FORCED_EVAL_PREFLIGHT_PYTHON:-$(command -v python3 || true)}"
if [[ "${PREFLIGHT_PYTHON}" != /* ]] || [ ! -x "${PREFLIGHT_PYTHON}" ]; then
    echo "ERROR: an absolute executable Python 3 is required for dataset timeout preflight" >&2
    exit 2
fi
PREFLIGHT_CODE=$'import json, math, sys\npath, start, count, agent = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])\ntimeouts = []\nverifiers = []\nwith open(path, encoding="utf-8") as stream:\n    for index, line in enumerate(stream):\n        if index < start:\n            continue\n        if len(timeouts) >= count:\n            break\n        row = json.loads(line)\n        metadata = row.get("metadata") if isinstance(row, dict) else None\n        if not isinstance(metadata, dict):\n            raise ValueError(f"dataset row {index} metadata must be an object")\n        values = []\n        for field in ("timeout_seconds", "verifier_timeout"):\n            value = metadata.get(field)\n            if isinstance(value, bool):\n                raise ValueError(f"dataset row {index} metadata.{field} must be positive and finite")\n            value = float(value)\n            if not math.isfinite(value) or value <= 0:\n                raise ValueError(f"dataset row {index} metadata.{field} must be positive and finite")\n            values.append(value)\n        timeouts.append(values[0])\n        verifiers.append(values[1])\nif len(timeouts) != count:\n    raise ValueError(f"requested {count} tasks at start index {start}, found {len(timeouts)}")\nmax_task = math.ceil(max(timeouts))\nmax_verifier = math.ceil(max(verifiers))\nouter = max(max_task, agent + max_verifier)\nprint(max_task, max_verifier, outer)'
if ! PREFLIGHT_OUTPUT="$(
    "${PREFLIGHT_PYTHON}" -c "${PREFLIGHT_CODE}" \
        "${DATA_PATH}" "${START_INDEX}" "${MAX_TASKS}" "${AGENT_TIMEOUT_SECONDS}"
)"; then
    echo "ERROR: selected dataset timeout preflight failed" >&2
    exit 2
fi
read -r MAX_DATASET_TASK_TIMEOUT_SECONDS MAX_VERIFIER_TIMEOUT_SECONDS \
    OUTER_TASK_ENVELOPE_SECONDS <<<"${PREFLIGHT_OUTPUT}"
if ! [[ "${MAX_DATASET_TASK_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${MAX_VERIFIER_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${OUTER_TASK_ENVELOPE_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: dataset timeout preflight returned invalid bounds" >&2
    exit 2
fi

FORCED_EVAL_CANDIDATE_COUNT=$((2 + INCLUDE_QWEN35_BASELINE))
FORCED_EVAL_WORK_ITEMS=$((MAX_TASKS * FORCED_EVAL_CANDIDATE_COUNT * REPLICATES))
FORCED_EVAL_MAX_PAID_ATTEMPTS=$((FORCED_EVAL_WORK_ITEMS * MAX_PAID_ATTEMPTS_PER_WORK))
FORCED_EVAL_WAVES=$(((FORCED_EVAL_MAX_PAID_ATTEMPTS + MAX_CONCURRENCY - 1) / MAX_CONCURRENCY))
REQUIRED_WALL_TIME_SECONDS=$((
    FORCED_EVAL_WAVES * OUTER_TASK_ENVELOPE_SECONDS + FORCED_EVAL_SLURM_MARGIN_SECONDS
))

seconds_to_slurm_time() {
    local total="$1"
    local hours=$((total / 3600))
    local minutes=$(((total % 3600) / 60))
    local seconds=$((total % 60))
    printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
}

slurm_time_to_seconds() {
    local value="$1"
    local days=0 hours=0 minutes=0 seconds=0
    if [[ "${value}" =~ ^([0-9]+)-([0-9]{1,2}):([0-9]{1,2}):([0-9]{1,2})$ ]]; then
        days=$((10#${BASH_REMATCH[1]}))
        hours=$((10#${BASH_REMATCH[2]}))
        minutes=$((10#${BASH_REMATCH[3]}))
        seconds=$((10#${BASH_REMATCH[4]}))
        if ((hours > 23 || minutes > 59 || seconds > 59)); then return 1; fi
    elif [[ "${value}" =~ ^([0-9]+)-([0-9]{1,2}):([0-9]{1,2})$ ]]; then
        days=$((10#${BASH_REMATCH[1]}))
        hours=$((10#${BASH_REMATCH[2]}))
        minutes=$((10#${BASH_REMATCH[3]}))
        if ((hours > 23 || minutes > 59)); then return 1; fi
    elif [[ "${value}" =~ ^([0-9]+)-([0-9]{1,2})$ ]]; then
        days=$((10#${BASH_REMATCH[1]}))
        hours=$((10#${BASH_REMATCH[2]}))
        if ((hours > 23)); then return 1; fi
    elif [[ "${value}" =~ ^([0-9]+):([0-9]{1,2}):([0-9]{1,2})$ ]]; then
        hours=$((10#${BASH_REMATCH[1]}))
        minutes=$((10#${BASH_REMATCH[2]}))
        seconds=$((10#${BASH_REMATCH[3]}))
        if ((minutes > 59 || seconds > 59)); then return 1; fi
    elif [[ "${value}" =~ ^([0-9]+):([0-9]{1,2})$ ]]; then
        minutes=$((10#${BASH_REMATCH[1]}))
        seconds=$((10#${BASH_REMATCH[2]}))
        if ((seconds > 59)); then return 1; fi
    elif [[ "${value}" =~ ^[0-9]+$ ]]; then
        minutes=$((10#${value}))
    else
        return 1
    fi
    printf '%s\n' "$((days * 86400 + hours * 3600 + minutes * 60 + seconds))"
}

if [ -z "${WALL_TIME:-}" ]; then
    WALL_TIME="$(seconds_to_slurm_time "${REQUIRED_WALL_TIME_SECONDS}")"
fi
if ! WALL_TIME_SECONDS="$(slurm_time_to_seconds "${WALL_TIME}")"; then
    echo "ERROR: invalid Slurm WALL_TIME value: ${WALL_TIME}" >&2
    exit 2
fi
if [ "${WALL_TIME_SECONDS}" -lt "${REQUIRED_WALL_TIME_SECONDS}" ]; then
    printf '%s\n' \
        "ERROR: WALL_TIME=${WALL_TIME} provides ${WALL_TIME_SECONDS}s, but forced eval requires at least ${REQUIRED_WALL_TIME_SECONDS}s: ${FORCED_EVAL_WORK_ITEMS} work items * ${MAX_PAID_ATTEMPTS_PER_WORK} max paid attempts / ${MAX_CONCURRENCY} concurrency = ${FORCED_EVAL_WAVES} waves * ${OUTER_TASK_ENVELOPE_SECONDS}s outer task envelope (dataset task max ${MAX_DATASET_TASK_TIMEOUT_SECONDS}s, configured agent ${AGENT_TIMEOUT_SECONDS}s + verifier max ${MAX_VERIFIER_TIMEOUT_SECONDS}s) + ${FORCED_EVAL_SLURM_MARGIN_SECONDS}s margin" >&2
    exit 2
fi

if [ "${RESUME}" = "1" ]; then
    if [ ! -d "${OUTPUT_DIR}" ] || [ ! -f "${OUTPUT_DIR}/manifest.json" ]; then
        echo "ERROR: --resume requires an existing output manifest" >&2
        exit 2
    fi
    SUBMIT_DIR="${SERVICE_DIR}.submit"
    for path in "${SERVICE_DIR}" "${SUBMIT_DIR}"; do
        if [ -e "${path}" ]; then
            echo "ERROR: resume service path must be fresh: ${path}" >&2
            exit 2
        fi
    done
else
    SUBMIT_DIR="${OUTPUT_DIR}.submit"
    for path in "${OUTPUT_DIR}" "${SERVICE_DIR}" "${SUBMIT_DIR}"; do
        if [ -e "${path}" ]; then
            echo "ERROR: fresh forced-eval path already exists: ${path}" >&2
            exit 2
        fi
    done
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")"
mkdir -m 700 "${SUBMIT_DIR}"
CREDENTIAL_FILE="${SUBMIT_DIR}/credentials.env"
CREDENTIAL_TMP="${CREDENTIAL_FILE}.tmp.$$"
SUBMITTED=0

cleanup_submit() {
    set +e
    rm -f -- "${CREDENTIAL_TMP}"
    if [ "${SUBMITTED}" != "1" ]; then
        rm -f -- "${CREDENTIAL_FILE}"
        rmdir "${SUBMIT_DIR}" 2>/dev/null || true
    fi
}
trap cleanup_submit EXIT

{
    printf 'export POLAR_NVIDIA_API_KEY=%q\n' "${POLAR_NVIDIA_API_KEY}"
    printf 'export POLAR_MODEL_POOL_BASE_URL=%q\n' "${POLAR_MODEL_POOL_BASE_URL}"
    printf 'export POLAR_DATA_ROOT=%q\n' "${POLAR_DATA_ROOT}"
    # Slurm executes a copied batch script from its spool directory.  Pass the
    # canonical checkout explicitly instead of asking the allocated copy to
    # infer the repository from BASH_SOURCE[0].
    printf 'export SPILOT_FORCED_EVAL_PROJECT_ROOT=%q\n' "${PROJECT_ROOT}"
    printf 'export SRUN_BIN=%q\n' "${SRUN_BIN}"
    printf 'export http_proxy=%q\n' "${http_proxy:-${HTTP_PROXY:-http://cw-dfw-cs-001-container-cache:3128}}"
    printf 'export https_proxy=%q\n' "${https_proxy:-${HTTPS_PROXY:-${http_proxy:-${HTTP_PROXY:-http://cw-dfw-cs-001-container-cache:3128}}}}"
    printf 'export HTTP_PROXY=%q\n' "${HTTP_PROXY:-${http_proxy:-http://cw-dfw-cs-001-container-cache:3128}}"
    printf 'export HTTPS_PROXY=%q\n' "${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-http://cw-dfw-cs-001-container-cache:3128}}}}"
} >"${CREDENTIAL_TMP}"
chmod 600 "${CREDENTIAL_TMP}"
mv "${CREDENTIAL_TMP}" "${CREDENTIAL_FILE}"

ACCOUNT="${ACCOUNT:-nvr_lpr_llm}"
PARTITION="${PARTITION:-interactive}"
SLURM_CONSTRAINT="${SLURM_CONSTRAINT:-H100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
POLAR_SLURM_MEM_PER_NODE="${POLAR_SLURM_MEM_PER_NODE:-128G}"
JOB_NAME="${JOB_NAME:-spilot-forced-eval}"
FORCED_EVAL_GPUS="${FORCED_EVAL_GPUS:-0}"
if ! [[ "${FORCED_EVAL_GPUS}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "ERROR: FORCED_EVAL_GPUS must be a non-negative integer" >&2
    exit 2
fi

SBATCH_ARGS=(
    --parsable
    --nodes=1
    --ntasks=1
    --ntasks-per-node=1
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --time="${WALL_TIME}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${POLAR_SLURM_MEM_PER_NODE}"
    --job-name="${JOB_NAME}"
    --chdir="${PROJECT_ROOT}"
    --output="${SUBMIT_DIR}/%x-%j.out"
    --error="${SUBMIT_DIR}/%x-%j.err"
    --export="POLAR_FORCED_EVAL_ENV_FILE=${CREDENTIAL_FILE}"
)
if [ -n "${SLURM_CONSTRAINT}" ]; then
    SBATCH_ARGS+=(--constraint="${SLURM_CONSTRAINT}")
fi
if [ "${FORCED_EVAL_GPUS}" != "0" ]; then
    SBATCH_ARGS+=(--gres="gpu:${FORCED_EVAL_GPUS}")
fi
SBATCH_ARGS+=(
    "${RUN_SCRIPT}"
    "${ARGS[@]}"
    --slurm-walltime-seconds "${WALL_TIME_SECONDS}"
    --slurm-margin-seconds "${FORCED_EVAL_SLURM_MARGIN_SECONDS}"
)

if [ "${SUBMIT_DRY_RUN:-0}" = "1" ]; then
    printf 'sbatch'
    printf ' %q' "${SBATCH_ARGS[@]}"
    printf '\n'
    exit 0
fi

SBATCH_BIN="${SBATCH_BIN:-$(command -v sbatch || true)}"
if [ -z "${SBATCH_BIN}" ]; then
    echo "ERROR: sbatch is unavailable" >&2
    exit 2
fi
JOB_ID="$(env \
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
    -u SLURM_MEM_PER_NODE \
    -u SLURM_MEM_PER_CPU \
    "${SBATCH_BIN}" "${SBATCH_ARGS[@]}")"
JOB_ID="${JOB_ID%%;*}"
if ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: sbatch returned an invalid job id: ${JOB_ID}" >&2
    exit 2
fi
SUBMITTED=1
printf '%s\n' "${JOB_ID}" >"${SUBMIT_DIR}/job_id"
echo "Submitted services-only forced eval job ${JOB_ID}"
echo "  output: ${OUTPUT_DIR}"
echo "  logs:   ${SUBMIT_DIR}"
