#!/usr/bin/env bash
# One-node, zero-GPU Slurm entrypoint for the paired forced-route benchmark.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
set -euo pipefail
umask 077

SPOOLED_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
FALLBACK_PROJECT_ROOT="$(cd -- "${SPOOLED_SCRIPT_DIR}/../.." && pwd)"
CHILD_PID=""
CREDENTIAL_PATH="${POLAR_FORCED_EVAL_ENV_FILE:-}"
POLAR_JOB_CACHE_ROOT=""

cleanup() {
    set +e
    if [ -n "${CHILD_PID}" ] && kill -0 "${CHILD_PID}" 2>/dev/null; then
        kill -TERM "${CHILD_PID}" 2>/dev/null || true
        wait "${CHILD_PID}" 2>/dev/null || true
    fi
    if [ -n "${CREDENTIAL_PATH}" ]; then
        rm -f -- "${CREDENTIAL_PATH}"
    fi
    if [ -n "${POLAR_JOB_CACHE_ROOT}" ] && \
       [[ "${POLAR_JOB_CACHE_ROOT}" == "/tmp/polar-forced-eval-${SLURM_JOB_ID:-}" ]]; then
        rm -rf -- "${POLAR_JOB_CACHE_ROOT}"
    fi
}

on_signal() {
    exit 130
}

run_child() {
    "$@" &
    CHILD_PID=$!
    set +e
    wait "${CHILD_PID}"
    local rc=$?
    set -e
    CHILD_PID=""
    return "${rc}"
}

trap cleanup EXIT
trap on_signal INT TERM

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: submit this entrypoint with sbatch or run it inside an allocation" >&2
    exit 2
fi
if [ "${SLURM_JOB_NUM_NODES:-1}" != "1" ]; then
    echo "ERROR: forced-route eval requires exactly one node" >&2
    exit 2
fi

# Source the private submission envelope exactly once, then unlink it before
# launching Pyxis. The endpoint key never appears in sbatch argv or YAML.
if [ "${SPILOT_FORCED_EVAL_IN_CONTAINER:-0}" != "1" ]; then
    if [ -z "${CREDENTIAL_PATH}" ] || [ ! -f "${CREDENTIAL_PATH}" ]; then
        echo "ERROR: POLAR_FORCED_EVAL_ENV_FILE must name the private credential file" >&2
        exit 2
    fi
    if [ "$(stat -c '%a' "${CREDENTIAL_PATH}")" != "600" ] || \
       [ "$(stat -c '%u' "${CREDENTIAL_PATH}")" != "$(id -u)" ]; then
        echo "ERROR: credential file must be owned by this user with mode 0600" >&2
        exit 2
    fi
    # shellcheck disable=SC1090
    source "${CREDENTIAL_PATH}"
    rm -f -- "${CREDENTIAL_PATH}"
    unset POLAR_FORCED_EVAL_ENV_FILE
    # The batch job was intentionally submitted with a list-only --export.
    # Reset step export so the just-sourced key/proxy reach the Pyxis process.
    export SLURM_EXPORT_ENV=ALL
fi

if [ -z "${POLAR_NVIDIA_API_KEY:-}" ]; then
    echo "ERROR: private credential file did not provide POLAR_NVIDIA_API_KEY" >&2
    exit 2
fi
if [ -n "${POLAR_CONTROL_PLANE_TOKEN:-}" ]; then
    echo "ERROR: control-plane token must be generated in memory by the allocation launcher" >&2
    exit 2
fi
export POLAR_MODEL_POOL_BASE_URL="${POLAR_MODEL_POOL_BASE_URL:-https://integrate.api.nvidia.com/v1}"
export POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${SPILOT_ROOT}/data}"
export http_proxy="${http_proxy:-${HTTP_PROXY:-http://cw-dfw-cs-001-container-cache:3128}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY:-${http_proxy}}}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"

# A script submitted directly with sbatch is copied below Slurm's spool tree,
# so BASH_SOURCE[0] is not a stable way to locate this checkout on the worker.
# The submitter records the canonical path in the private envelope; direct
# allocation invocations retain the source-relative fallback.
PROJECT_ROOT="${SPILOT_FORCED_EVAL_PROJECT_ROOT:-${FALLBACK_PROJECT_ROOT}}"
case "${PROJECT_ROOT}" in /*) ;; *) echo "ERROR: project root must be absolute" >&2; exit 2 ;; esac
if [ ! -d "${PROJECT_ROOT}/src/polar" ] || \
   [ ! -f "${PROJECT_ROOT}/examples/spilot_router_slime_grpo/run_forced_route_eval.py" ]; then
    echo "ERROR: invalid SPilot project root: ${PROJECT_ROOT}" >&2
    exit 2
fi
SCRIPT_DIR="${PROJECT_ROOT}/examples/spilot_router_slime_grpo"
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
USER_ROOT="$(dirname "${SPILOT_ROOT}")"

TRAIN_SQSH="${POLR_TRAIN_SQSH:-${POLAR_DATA_ROOT}/container/flappydora-ubuntu22.04-cuda13.3.sqsh}"
TRAIN_MOUNTS="${TRAIN_CONTAINER_MOUNTS:-/lustre/fsw:/lustre/fsw}"
PYTHON_BIN="${TMAX_SIF_PYTHON_BIN:-${USER_ROOT}/.python/polar/bin/python}"
if [ ! -f "${TRAIN_SQSH}" ]; then
    echo "ERROR: Pyxis image does not exist: ${TRAIN_SQSH}" >&2
    exit 2
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: Polar Python does not exist: ${PYTHON_BIN}" >&2
    exit 2
fi

if [ "${SPILOT_FORCED_EVAL_IN_CONTAINER:-0}" != "1" ]; then
    if [[ "${SRUN_BIN:-}" != /* ]] || [ ! -x "${SRUN_BIN}" ]; then
        echo "ERROR: private submission envelope did not provide an executable SRUN_BIN" >&2
        exit 2
    fi
    CPUS="${SLURM_CPUS_PER_TASK:-32}"
    run_child "${SRUN_BIN}" \
        --overlap \
        --nodes=1 \
        --ntasks=1 \
        --ntasks-per-node=1 \
        --cpus-per-task="${CPUS}" \
        --cpu-bind=none \
        --kill-on-bad-exit=1 \
        --container-image="${TRAIN_SQSH}" \
        --container-mounts="${TRAIN_MOUNTS}" \
        --container-workdir="${PROJECT_ROOT}" \
        --container-writable \
        --no-container-mount-home \
        env SPILOT_FORCED_EVAL_IN_CONTAINER=1 bash "${SCRIPT_DIR}/run_forced_route_eval.sh" "$@"
    exit $?
fi

# The nested Apptainer runtime needs a private writable home/cache inside the
# Pyxis container. None of these paths or files contain endpoint credentials.
POLAR_JOB_CACHE_ROOT="/tmp/polar-forced-eval-${SLURM_JOB_ID}"
export POLAR_JOB_CACHE_ROOT
rm -rf "${POLAR_JOB_CACHE_ROOT}"
mkdir -p \
    "${POLAR_JOB_CACHE_ROOT}/home" \
    "${POLAR_JOB_CACHE_ROOT}/apptainer-cache" \
    "${POLAR_JOB_CACHE_ROOT}/apptainer-tmp" \
    "${POLAR_JOB_CACHE_ROOT}/apptainer-work" \
    "${POLAR_JOB_CACHE_ROOT}/xdg" \
    "${POLAR_JOB_CACHE_ROOT}/xdg-config" \
    "${POLAR_JOB_CACHE_ROOT}/xdg-runtime"
chmod 700 "${POLAR_JOB_CACHE_ROOT}" "${POLAR_JOB_CACHE_ROOT}/xdg-runtime"
export HOME="${POLAR_JOB_CACHE_ROOT}/home"
export APPTAINER_CACHEDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-cache"
export APPTAINER_TMPDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-tmp"
export APPTAINER_WORKDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-work"
export XDG_CACHE_HOME="${POLAR_JOB_CACHE_ROOT}/xdg"
export XDG_CONFIG_HOME="${POLAR_JOB_CACHE_ROOT}/xdg-config"
export XDG_RUNTIME_DIR="${POLAR_JOB_CACHE_ROOT}/xdg-runtime"
export POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-/usr/bin/apptainer}"
export POLAR_APPTAINER_NO_INSTANCE="${POLAR_APPTAINER_NO_INSTANCE:-1}"
export POLAR_APPTAINER_PERSISTENT_BROKER="${POLAR_APPTAINER_PERSISTENT_BROKER:-0}"
export POLAR_APPTAINER_NO_MOUNT_HOSTFS="${POLAR_APPTAINER_NO_MOUNT_HOSTFS:-1}"
export POLAR_APPTAINER_NO_MOUNT_TMP="${POLAR_APPTAINER_NO_MOUNT_TMP:-1}"
export POLAR_APPTAINER_ISOLATE_PID="${POLAR_APPTAINER_ISOLATE_PID:-1}"
export POLAR_APPTAINER_ISOLATE_IPC="${POLAR_APPTAINER_ISOLATE_IPC:-1}"
export POLAR_APPTAINER_CLEANENV="${POLAR_APPTAINER_CLEANENV:-1}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

run_child "${PYTHON_BIN}" "${SCRIPT_DIR}/run_forced_route_eval.py" "$@"
