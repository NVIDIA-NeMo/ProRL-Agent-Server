#!/usr/bin/env bash
# Submit CPU-only Slurm array jobs to build TMax-15K Apptainer SIFs.
#
# Defaults:
#   TMAX_DATASET_DIR      $TMAX_DATA_ROOT/tmax-15k
#   APPTAINER_IMAGE_DIR   $TMAX_DATA_ROOT/tmax-15k-sif
#
# Defaults are tuned for max-concurrency builds on the cpu_short partition
# (QOS caps users at node=10, not jobs; nodes are shared, so pack small jobs):
#   ARRAY_PARALLEL=200  MEM=12G  CPUS_PER_TASK=4  JOBS_PER_TASK=4
#   -> ~20 jobs/node x 10 nodes ~= 200 concurrent. MEM<10G gives no extra
#      concurrency (CPU-bound at 96/4=24 jobs/node) and 4G OOM-kills builds.
#   BASE_SIF defaults to the local ubuntu-22.04-base.sif if present (faster
#      than docker://ubuntu:22.04). Builder is idempotent: re-run to recover.
#
# Example (defaults already give a full 1000-shard concurrent build):
#   bash examples/tmax-15k/submit_build_sifs_slurm.sh
#   TMAX_SIF_BUILD_DRY_RUN=1 bash examples/tmax-15k/submit_build_sifs_slurm.sh  # preview only
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../path_safety.sh
source "${SCRIPT_DIR}/../path_safety.sh"

TMAX_DATA_ROOT="${TMAX_DATA_ROOT:-/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/data}"
TMAX_DATASET_DIR="${TMAX_DATASET_DIR:-${TMAX_DATA_ROOT}/tmax-15k}"
APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${TMAX_DATA_ROOT}/tmax-15k-sif}"
if [ -z "${POLAR_APPTAINER_BIN:-}" ]; then
    POLAR_APPTAINER_BIN="$(command -v apptainer || command -v singularity || true)"
    POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-/usr/bin/apptainer}"
fi
export TMAX_DATASET_DIR APPTAINER_IMAGE_DIR POLAR_APPTAINER_BIN

ACCOUNT="${ACCOUNT:-${SBATCH_ACCOUNT:-nvr_lpr_llm}}"
PARTITION="${TMAX_SIF_BUILD_PARTITION:-${SIF_BUILD_PARTITION:-cpu_short}}"
JOB_NAME="${TMAX_SIF_BUILD_JOB_NAME:-tmax-build-sifs}"
WALL_TIME="${TMAX_SIF_BUILD_TIME:-${WALL_TIME:-4:00:00}}"
CONSTRAINT="${TMAX_SIF_BUILD_CONSTRAINT:-}"
LAUNCHER="${TMAX_SIF_BUILD_LAUNCHER:-sbatch-container}"
BUILD_CONTAINER_IMAGE="${TMAX_SIF_BUILD_CONTAINER_IMAGE:-flappydora/ubuntu22.04-cuda13.3:latest}"
BUILD_CONTAINER_MOUNTS="${TMAX_SIF_BUILD_CONTAINER_MOUNTS:-/lustre/fsw:/lustre/fsw}"
LOG_DIR="${TMAX_SIF_BUILD_LOG_DIR:-${TMAX_DATA_ROOT}/logs/slurm}"
polar_require_absolute_path TMAX_SIF_BUILD_LOG_DIR "${LOG_DIR}"
PYTHON_BIN="${TMAX_SIF_PYTHON_BIN:-${PYTHON_BIN:-/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/.python/polar/bin/python}}"

SHARDS="${TMAX_SIF_BUILD_SHARDS:-1000}"
ARRAY_PARALLEL="${TMAX_SIF_BUILD_ARRAY_PARALLEL:-200}"
JOBS_PER_TASK="${TMAX_SIF_BUILD_JOBS_PER_TASK:-4}"
CPUS_PER_TASK="${TMAX_SIF_BUILD_CPUS_PER_TASK:-4}"
BATCH_CPUS_PER_TASK="${TMAX_SIF_BUILD_BATCH_CPUS_PER_TASK:-${CPUS_PER_TASK}}"
MEM="${TMAX_SIF_BUILD_MEM:-12G}"
MAX_TASKS="${TMAX_SIF_MAX_TASKS:--1}"
BUILDER="${TMAX_SIF_BUILDER:-direct-apptainer}"
APPTAINER_FAKEROOT="${TMAX_SIF_APPTAINER_FAKEROOT:-0}"
DEFAULT_BASE_SIF="${TMAX_DATA_ROOT}/container/ubuntu-22.04-base.sif"
if [ -n "${TMAX_SIF_BASE_SIF:-}" ]; then
    BASE_SIF="${TMAX_SIF_BASE_SIF}"
elif [ -f "${DEFAULT_BASE_SIF}" ]; then
    BASE_SIF="${DEFAULT_BASE_SIF}"
else
    BASE_SIF=""
fi
if [ "${TMAX_SIF_MKSQUASHFS_ARGS+x}" = "x" ]; then
    MKSQUASHFS_ARGS="${TMAX_SIF_MKSQUASHFS_ARGS}"
else
    MKSQUASHFS_ARGS="${POLAR_MKSQUASHFS_ARGS--processors 1 -mem 1024M}"
fi

FORCE_FLAG=""
if [ "${TMAX_SIF_BUILD_FORCE:-0}" = "1" ]; then
    FORCE_FLAG="--force"
fi
FORCE_DOCKER_FLAG=""
if [ "${TMAX_SIF_BUILD_FORCE_DOCKER:-0}" = "1" ]; then
    FORCE_DOCKER_FLAG="--force-docker"
fi
SKIP_DOCKER_FLAG=""
if [ "${TMAX_SIF_BUILD_SKIP_DOCKER:-0}" = "1" ]; then
    SKIP_DOCKER_FLAG="--skip-docker-build"
fi
FAKEROOT_FLAG=""
if [ "${APPTAINER_FAKEROOT}" = "1" ]; then
    FAKEROOT_FLAG="--apptainer-fakeroot"
fi

if [ "${SHARDS}" -lt 1 ]; then
    echo "ERROR: TMAX_SIF_BUILD_SHARDS must be >= 1" >&2
    exit 1
fi
ARRAY_LAST=$((SHARDS - 1))
mkdir -p "${LOG_DIR}"

printf -v PR_Q '%q' "${PROJECT_ROOT}"
printf -v DATA_Q '%q' "${TMAX_DATASET_DIR}"
printf -v SIF_Q '%q' "${APPTAINER_IMAGE_DIR}"
printf -v BUILDER_Q '%q' "${BUILDER}"
printf -v BASE_SIF_Q '%q' "${BASE_SIF}"
printf -v MKS_Q '%q' "${MKSQUASHFS_ARGS}"
printf -v PY_Q '%q' "${PYTHON_BIN}"

echo "Submitting TMax SIF build jobs: launcher=${LAUNCHER} partition=${PARTITION:-default} constraint=${CONSTRAINT:-none} shards=${SHARDS} parallel=${ARRAY_PARALLEL} jobs/task=${JOBS_PER_TASK}"
echo "  dataset: ${TMAX_DATASET_DIR}"
echo "  image dir: ${APPTAINER_IMAGE_DIR}"
echo "  python: ${PYTHON_BIN}"
echo "  builder: ${BUILDER}"
echo "  apptainer fakeroot: ${APPTAINER_FAKEROOT}"
echo "  base sif: ${BASE_SIF:-docker://ubuntu:22.04}"
echo "  apptainer bin: ${POLAR_APPTAINER_BIN}"
if [ "${BUILDER}" = "docker-daemon" ]; then
    echo "  docker bin: ${POLAR_DOCKER_BIN:-auto-detect in job}"
fi
if [ "${LAUNCHER}" = "sbatch-container" ]; then
    echo "  container image: ${BUILD_CONTAINER_IMAGE}"
    echo "  container mounts: ${BUILD_CONTAINER_MOUNTS}"
fi

if [ "${LAUNCHER}" != "sbatch-host" ] && [ "${LAUNCHER}" != "sbatch-container" ]; then
    echo "ERROR: TMAX_SIF_BUILD_LAUNCHER must be 'sbatch-host' or 'sbatch-container'." >&2
    exit 1
fi

SBATCH_SCRIPT="${LOG_DIR}/${JOB_NAME}-$(date -u +%Y%m%dT%H%M%SZ)-$$.sbatch"
{
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --account=${ACCOUNT}"
    echo "#SBATCH --job-name=${JOB_NAME}"
    if [ -n "${PARTITION}" ]; then
        echo "#SBATCH --partition=${PARTITION}"
    fi
    echo "#SBATCH --time=${WALL_TIME}"
    echo "#SBATCH --array=0-${ARRAY_LAST}%${ARRAY_PARALLEL}"
    echo "#SBATCH --ntasks=1"
    echo "#SBATCH --cpus-per-task=${BATCH_CPUS_PER_TASK}"
    echo "#SBATCH --mem=${MEM}"
    if [ -n "${CONSTRAINT}" ]; then
        echo "#SBATCH --constraint=${CONSTRAINT}"
    fi
    echo "#SBATCH --output=${LOG_DIR}/%x-%A_%a.out"
    echo "#SBATCH --error=${LOG_DIR}/%x-%A_%a.err"
    echo "#SBATCH --export=ALL"

    if [ "${LAUNCHER}" = "sbatch-container" ]; then
        printf -v IMAGE_Q '%q' "${BUILD_CONTAINER_IMAGE}"
        printf -v MOUNTS_Q '%q' "${BUILD_CONTAINER_MOUNTS}"
        cat <<EOF

set -euo pipefail
echo "[tmax-build-sifs] allocation started on \$(hostname) task=\${SLURM_ARRAY_TASK_ID:-0} at \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CONTAINER_IMAGE=${IMAGE_Q}
CONTAINER_MOUNTS=${MOUNTS_Q}
srun \\
  --overlap \\
  --cpu-bind=none \\
  --nodes=1 \\
  --ntasks=1 \\
  --cpus-per-task="${CPUS_PER_TASK}" \\
  --mem="${MEM}" \\
  --container-image="\${CONTAINER_IMAGE}" \\
  --container-mounts="\${CONTAINER_MOUNTS}" \\
  --container-workdir=${PR_Q} \\
  --container-writable \\
  --no-container-mount-home \\
bash -lc "set -euo pipefail
cd ${PR_Q}
if [ -z \"\\\${POLAR_APPTAINER_BIN:-}\" ]; then export POLAR_APPTAINER_BIN=\\\$(command -v apptainer || command -v singularity || true); fi
if [ ${BUILDER_Q} = docker-daemon ] && [ -z \"\\\${POLAR_DOCKER_BIN:-}\" ]; then export POLAR_DOCKER_BIN=\\\$(command -v docker || true); fi
echo \"[tmax-build-sifs] builder=${BUILDER}\"
echo \"[tmax-build-sifs] apptainer_bin=\\\${POLAR_APPTAINER_BIN:-auto-detect failed}\"
if [ ${BUILDER_Q} = docker-daemon ]; then echo \"[tmax-build-sifs] docker_bin=\\\${POLAR_DOCKER_BIN:-auto-detect failed}\"; fi
export POLAR_JOB_CACHE_ROOT=\"/tmp/polar-tmax-sifbuild-\${SLURM_ARRAY_JOB_ID:-\${SLURM_JOB_ID}}-\${SLURM_ARRAY_TASK_ID:-0}\"
rm -rf \"\\\${POLAR_JOB_CACHE_ROOT}\"
mkdir -p \"\\\${POLAR_JOB_CACHE_ROOT}\"
PY=${PY_Q}
if [ ! -x \"\\\${PY}\" ]; then echo \"ERROR: python not executable: \\\${PY}\" >&2; exit 1; fi
echo \"[tmax-build-sifs] python=\\\${PY}\"
\"\\\${PY}\" examples/tmax-15k/build_sifs.py \\
  --dataset-dir ${DATA_Q} \\
  --image-dir ${SIF_Q} \\
  --builder ${BUILDER_Q} \\
  --base-sif ${BASE_SIF_Q} \\
  --max-tasks \"${MAX_TASKS}\" \\
  --num-shards \"${SHARDS}\" \\
  --shard-index \"\${SLURM_ARRAY_TASK_ID:-0}\" \\
  --jobs \"${JOBS_PER_TASK}\" \\
  --cache-root \"\\\${POLAR_JOB_CACHE_ROOT}\" \\
  --mksquashfs-args ${MKS_Q} ${FORCE_FLAG} ${FORCE_DOCKER_FLAG} ${SKIP_DOCKER_FLAG} ${FAKEROOT_FLAG}"
EOF
    else
        cat <<EOF

set -euo pipefail
echo "[tmax-build-sifs] host allocation started on \$(hostname) task=\${SLURM_ARRAY_TASK_ID:-0} at \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd ${PR_Q}
if [ -z "\${POLAR_APPTAINER_BIN:-}" ]; then export POLAR_APPTAINER_BIN=\$(command -v apptainer || command -v singularity || true); fi
if [ ${BUILDER_Q} = docker-daemon ] && [ -z "\${POLAR_DOCKER_BIN:-}" ]; then export POLAR_DOCKER_BIN=\$(command -v docker || true); fi
echo "[tmax-build-sifs] builder=${BUILDER}"
echo "[tmax-build-sifs] apptainer_bin=\${POLAR_APPTAINER_BIN:-auto-detect failed}"
if [ ${BUILDER_Q} = docker-daemon ]; then echo "[tmax-build-sifs] docker_bin=\${POLAR_DOCKER_BIN:-auto-detect failed}"; fi
export POLAR_JOB_CACHE_ROOT="/tmp/polar-tmax-sifbuild-\${SLURM_ARRAY_JOB_ID:-\${SLURM_JOB_ID}}-\${SLURM_ARRAY_TASK_ID:-0}"
rm -rf "\${POLAR_JOB_CACHE_ROOT}"
mkdir -p "\${POLAR_JOB_CACHE_ROOT}"
PY=${PY_Q}
if [ ! -x "\${PY}" ]; then echo "ERROR: python not executable: \${PY}" >&2; exit 1; fi
echo "[tmax-build-sifs] python=\${PY}"
"\${PY}" examples/tmax-15k/build_sifs.py \\
  --dataset-dir ${DATA_Q} \\
  --image-dir ${SIF_Q} \\
  --builder ${BUILDER_Q} \\
  --base-sif ${BASE_SIF_Q} \\
  --max-tasks "${MAX_TASKS}" \\
  --num-shards "${SHARDS}" \\
  --shard-index "\${SLURM_ARRAY_TASK_ID:-0}" \\
  --jobs "${JOBS_PER_TASK}" \\
  --cache-root "\${POLAR_JOB_CACHE_ROOT}" \\
  --mksquashfs-args ${MKS_Q} ${FORCE_FLAG} ${FORCE_DOCKER_FLAG} ${SKIP_DOCKER_FLAG} ${FAKEROOT_FLAG}
EOF
    fi
} > "${SBATCH_SCRIPT}"

if [ "${TMAX_SIF_BUILD_DRY_RUN:-0}" = "1" ]; then
    echo "Dry run only; generated sbatch script: ${SBATCH_SCRIPT}"
    bash -n "${SBATCH_SCRIPT}"
    sed -n '1,140p' "${SBATCH_SCRIPT}"
    exit 0
fi

JOB_ID=$(sbatch --parsable "${SBATCH_SCRIPT}")

echo "Submitted TMax SIF build array job: ${JOB_ID}"
echo "  shards: ${SHARDS}; array parallelism: ${ARRAY_PARALLEL}; jobs per task: ${JOBS_PER_TASK}"
echo "  sbatch script: ${SBATCH_SCRIPT}"
echo "  logs: tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}_0.out"
