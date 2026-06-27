#!/usr/bin/env bash
# Build one deterministic shard of TMax-15K SIF images.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

TMAX_DATA_ROOT="${TMAX_DATA_ROOT:-/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/data}"
TMAX_DATASET_DIR="${TMAX_DATASET_DIR:-${TMAX_DATA_ROOT}/tmax-15k}"
APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${TMAX_DATA_ROOT}/tmax-15k-sif}"
if [ -z "${POLAR_APPTAINER_BIN:-}" ]; then
    POLAR_APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
    POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-/usr/bin/apptainer}"
fi
TMAX_SIF_BUILDER="${TMAX_SIF_BUILDER:-direct-apptainer}"
TMAX_SIF_BASE_SIF="${TMAX_SIF_BASE_SIF:-}"
if [ "${TMAX_SIF_BUILDER}" = "docker-daemon" ] && [ -z "${POLAR_DOCKER_BIN:-}" ]; then
    POLAR_DOCKER_BIN="$(command -v docker || true)"
fi
export TMAX_DATASET_DIR APPTAINER_IMAGE_DIR POLAR_APPTAINER_BIN POLAR_DOCKER_BIN TMAX_SIF_BUILDER TMAX_SIF_BASE_SIF

DEFAULT_POLAR_PYTHON="/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/.python/polar/bin/python"
PYTHON_BIN="${TMAX_SIF_PYTHON_BIN:-${PYTHON_BIN:-}}"
if [ -z "${PYTHON_BIN}" ]; then
    if [ -x "${DEFAULT_POLAR_PYTHON}" ]; then
        PYTHON_BIN="${DEFAULT_POLAR_PYTHON}"
    elif [ -x "${PROJECT_ROOT}/.venv/bin/python3" ]; then
        PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python3"
    else
        PYTHON_BIN="$(command -v python3 || command -v python)"
    fi
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

TMAX_SIF_NUM_SHARDS="${TMAX_SIF_NUM_SHARDS:-1}"
TMAX_SIF_SHARD_INDEX="${TMAX_SIF_SHARD_INDEX:-0}"
TMAX_SIF_BUILD_JOBS="${TMAX_SIF_BUILD_JOBS:-1}"
TMAX_SIF_MAX_TASKS="${TMAX_SIF_MAX_TASKS:--1}"
TMAX_SIF_APPTAINER_FAKEROOT="${TMAX_SIF_APPTAINER_FAKEROOT:-0}"
if [ "${TMAX_SIF_MKSQUASHFS_ARGS+x}" != "x" ]; then
    TMAX_SIF_MKSQUASHFS_ARGS="${POLAR_MKSQUASHFS_ARGS--processors 1 -mem 1024M}"
fi

export POLAR_JOB_CACHE_ROOT="${POLAR_JOB_CACHE_ROOT:-/tmp/polar-tmax-sifbuild-${SLURM_JOB_ID:-manual}-${TMAX_SIF_SHARD_INDEX}}"
rm -rf "${POLAR_JOB_CACHE_ROOT}"
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
export SINGULARITY_CACHEDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-cache"
export APPTAINER_TMPDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-tmp"
export SINGULARITY_TMPDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-tmp"
export TMPDIR="${POLAR_JOB_CACHE_ROOT}/apptainer-tmp"
export XDG_CACHE_HOME="${POLAR_JOB_CACHE_ROOT}/xdg"
export XDG_CONFIG_HOME="${POLAR_JOB_CACHE_ROOT}/xdg-config"
export XDG_RUNTIME_DIR="${POLAR_JOB_CACHE_ROOT}/xdg-runtime"

echo "[tmax-build-sifs] dataset=${TMAX_DATASET_DIR}"
echo "[tmax-build-sifs] image_dir=${APPTAINER_IMAGE_DIR}"
echo "[tmax-build-sifs] python=${PYTHON_BIN}"
echo "[tmax-build-sifs] apptainer_bin=${POLAR_APPTAINER_BIN}"
echo "[tmax-build-sifs] builder=${TMAX_SIF_BUILDER}"
echo "[tmax-build-sifs] base_sif=${TMAX_SIF_BASE_SIF:-docker://ubuntu:22.04}"
if [ "${TMAX_SIF_BUILDER}" = "docker-daemon" ]; then
    echo "[tmax-build-sifs] docker_bin=${POLAR_DOCKER_BIN:-auto-detect failed}"
fi

FORCE_ARGS=()
if [ "${TMAX_SIF_BUILD_FORCE:-0}" = "1" ]; then
    FORCE_ARGS+=(--force)
fi
if [ "${TMAX_SIF_BUILD_FORCE_DOCKER:-0}" = "1" ]; then
    FORCE_ARGS+=(--force-docker)
fi
if [ "${TMAX_SIF_BUILD_SKIP_DOCKER:-0}" = "1" ]; then
    FORCE_ARGS+=(--skip-docker-build)
fi
if [ "${TMAX_SIF_APPTAINER_FAKEROOT}" = "1" ]; then
    FORCE_ARGS+=(--apptainer-fakeroot)
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_sifs.py" \
    --dataset-dir "${TMAX_DATASET_DIR}" \
    --image-dir "${APPTAINER_IMAGE_DIR}" \
    --builder "${TMAX_SIF_BUILDER}" \
    --base-sif "${TMAX_SIF_BASE_SIF}" \
    --max-tasks "${TMAX_SIF_MAX_TASKS}" \
    --jobs "${TMAX_SIF_BUILD_JOBS}" \
    --num-shards "${TMAX_SIF_NUM_SHARDS}" \
    --shard-index "${TMAX_SIF_SHARD_INDEX}" \
    --cache-root "${POLAR_JOB_CACHE_ROOT}" \
    --mksquashfs-args "${TMAX_SIF_MKSQUASHFS_ARGS}" \
    "${FORCE_ARGS[@]}"
