#!/usr/bin/env bash
# Build one deterministic shard of SWE-Gym SIF images.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=./env.cwdfw.sh
source "${SCRIPT_DIR}/env.cwdfw.sh"
# shellcheck source=../path_safety.sh
source "${SCRIPT_DIR}/../path_safety.sh"

PYTHON_BIN="${PYTHON_BIN:-${POLR_TRAIN_VENV}/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-${POLR_TRAIN_VENV}}"
export PYTHONNOUSERSITE=1

SIF_NUM_SHARDS="${SIF_NUM_SHARDS:-1}"
SIF_SHARD_INDEX="${SIF_SHARD_INDEX:-0}"
SIF_BUILD_JOBS="${SIF_BUILD_JOBS:-8}"

export POLAR_JOB_CACHE_ROOT="${POLAR_JOB_CACHE_ROOT:-/tmp/polar-sifbuild-${SLURM_JOB_ID:-manual}-${SIF_SHARD_INDEX}}"
polar_safe_remove_tree POLAR_JOB_CACHE_ROOT "${POLAR_JOB_CACHE_ROOT}" /tmp polar-
mkdir -p \
    "${POLAR_JOB_CACHE_ROOT}/home" \
    "${POLAR_JOB_CACHE_ROOT}/apptainer-cache" \
    "${POLAR_JOB_CACHE_ROOT}/apptainer-tmp" \
    "${POLAR_JOB_CACHE_ROOT}/xdg" \
    "${POLAR_JOB_CACHE_ROOT}/xdg-config" \
    "${POLAR_JOB_CACHE_ROOT}/xdg-runtime"
chmod 700 "${POLAR_JOB_CACHE_ROOT}/xdg-runtime"

export HOME="${POLAR_JOB_CACHE_ROOT}/home"
export APPTAINER_CACHEDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-cache"
export APPTAINER_TMPDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-tmp"
export XDG_CACHE_HOME="${POLAR_JOB_CACHE_ROOT}/xdg"
export XDG_CONFIG_HOME="${POLAR_JOB_CACHE_ROOT}/xdg-config"
export XDG_RUNTIME_DIR="${POLAR_JOB_CACHE_ROOT}/xdg-runtime"

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_sifs.py" \
    --jobs "${SIF_BUILD_JOBS}" \
    --num-shards "${SIF_NUM_SHARDS}" \
    --shard-index "${SIF_SHARD_INDEX}" \
    --cache-root "${POLAR_JOB_CACHE_ROOT}"
