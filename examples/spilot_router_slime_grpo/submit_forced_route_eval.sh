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
ARGS=("$@")
index=0
while [ "${index}" -lt "${#ARGS[@]}" ]; do
    argument="${ARGS[${index}]}"
    case "${argument}" in
        --output-dir|--service-dir|--data|--run-id)
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
            esac
            index=$((index + 2))
            ;;
        --output-dir=*) OUTPUT_DIR="${argument#*=}"; index=$((index + 1)) ;;
        --service-dir=*) SERVICE_DIR="${argument#*=}"; index=$((index + 1)) ;;
        --data=*) DATA_PATH="${argument#*=}"; index=$((index + 1)) ;;
        --run-id=*) RUN_ID="${argument#*=}"; index=$((index + 1)) ;;
        *) index=$((index + 1)) ;;
    esac
done

if [ -z "${OUTPUT_DIR}" ] || [ -z "${DATA_PATH}" ] || [ -z "${RUN_ID}" ]; then
    echo "ERROR: --output-dir, --data, and --run-id are required" >&2
    exit 2
fi
case "${OUTPUT_DIR}" in /*) ;; *) echo "ERROR: --output-dir must be absolute" >&2; exit 2 ;; esac
case "${DATA_PATH}" in /*) ;; *) echo "ERROR: --data must be absolute" >&2; exit 2 ;; esac
if [ -n "${SERVICE_DIR}" ]; then
    case "${SERVICE_DIR}" in /*) ;; *) echo "ERROR: --service-dir must be absolute" >&2; exit 2 ;; esac
else
    SERVICE_DIR="${OUTPUT_DIR}.service"
fi
if ! [[ "${RUN_ID}" =~ ^[0-9A-Za-z][0-9A-Za-z._-]{0,79}$ ]]; then
    echo "ERROR: --run-id must match [A-Za-z0-9][A-Za-z0-9._-]{0,79}" >&2
    exit 2
fi
if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: evaluation data does not exist: ${DATA_PATH}" >&2
    exit 2
fi

SUBMIT_DIR="${OUTPUT_DIR}.submit"
for path in "${OUTPUT_DIR}" "${SERVICE_DIR}" "${SUBMIT_DIR}"; do
    if [ -e "${path}" ]; then
        echo "ERROR: fresh forced-eval path already exists: ${path}" >&2
        exit 2
    fi
done
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
WALL_TIME="${WALL_TIME:-02:00:00}"
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
SBATCH_ARGS+=("${RUN_SCRIPT}" "${ARGS[@]}")

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
