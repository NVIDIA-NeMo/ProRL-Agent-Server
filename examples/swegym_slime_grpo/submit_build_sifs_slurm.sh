#!/usr/bin/env bash
# Submit CPU-only Slurm array jobs to build SWE-Gym Apptainer SIFs.
#
# Usage:
#   source examples/swegym_slime_grpo/env.cwdfw.sh
#   bash examples/swegym_slime_grpo/submit_build_sifs_slurm.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

ACCOUNT="${ACCOUNT:-nvr_lpr_llm}"
PARTITION="${SIF_BUILD_PARTITION:-cpu_short}"
JOB_NAME="${JOB_NAME:-swegym-build-sifs}"
WALL_TIME="${SIF_BUILD_TIME:-${WALL_TIME:-4:00:00}}"
CONSTRAINT="${SIF_BUILD_CONSTRAINT:-}"
LAUNCHER="${SIF_BUILD_LAUNCHER:-sbatch-container}"
BUILD_CONTAINER_IMAGE="${SIF_BUILD_CONTAINER_IMAGE:-${POLR_TRAIN_SQSH:?set POLR_TRAIN_SQSH or SIF_BUILD_CONTAINER_IMAGE}}"
BUILD_CONTAINER_MOUNTS="${SIF_BUILD_CONTAINER_MOUNTS:-${TRAIN_CONTAINER_MOUNTS:-/lustre/fsw:/lustre/fsw}}"

APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:?set APPTAINER_IMAGE_DIR}"
POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${PROJECT_ROOT}/tmp}"
LOG_DIR="${PROJECT_ROOT}/logs/slurm"
mkdir -p "${LOG_DIR}" "${POLAR_DATA_ROOT}/runs"

SHARDS="${SIF_BUILD_SHARDS:-64}"
ARRAY_PARALLEL="${SIF_BUILD_ARRAY_PARALLEL:-64}"
JOBS_PER_TASK="${SIF_BUILD_JOBS_PER_TASK:-1}"
CPUS_PER_TASK="${SIF_BUILD_CPUS_PER_TASK:-4}"
BATCH_CPUS_PER_TASK="${SIF_BUILD_BATCH_CPUS_PER_TASK:-${CPUS_PER_TASK}}"
MEM="${SIF_BUILD_MEM:-16G}"
FORCE_FLAG=""
if [ "${SIF_BUILD_FORCE:-0}" = "1" ]; then
    FORCE_FLAG="--force"
fi

if [ "${SHARDS}" -lt 1 ]; then
    echo "ERROR: SIF_BUILD_SHARDS must be >= 1" >&2
    exit 1
fi

ARRAY_LAST=$((SHARDS - 1))

printf -v PR_Q '%q' "${PROJECT_ROOT}"
printf -v IMAGE_Q '%q' "${BUILD_CONTAINER_IMAGE}"
printf -v MOUNTS_Q '%q' "${BUILD_CONTAINER_MOUNTS}"

echo "Submitting SIF build jobs: launcher=${LAUNCHER} partition=${PARTITION:-default} constraint=${CONSTRAINT:-none} shards=${SHARDS} parallel=${ARRAY_PARALLEL} jobs/task=${JOBS_PER_TASK}"

if [ "${LAUNCHER}" = "srun" ]; then
    RUN_DIR="${LOG_DIR}/${JOB_NAME}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mkdir -p "${RUN_DIR}"
    echo "  run dir: ${RUN_DIR}"
    if [ "${SIF_BUILD_DRY_RUN:-0}" = "1" ]; then
        echo "Dry run only; direct srun logs would be written under: ${RUN_DIR}"
        echo "Example shard 0 command:"
        echo "srun --nodes=1 --ntasks=1 --cpus-per-task=${CPUS_PER_TASK} --mem=${MEM} --account=${ACCOUNT} --job-name=${JOB_NAME}-0 ${PARTITION:+--partition=${PARTITION}} ${CONSTRAINT:+--constraint=${CONSTRAINT}} --time=${WALL_TIME} --container-image=${BUILD_CONTAINER_IMAGE} --container-mounts=${BUILD_CONTAINER_MOUNTS} --container-workdir=${PROJECT_ROOT} --container-writable --no-container-mount-home bash -lc '... build_sifs.py --num-shards ${SHARDS} --shard-index 0 --jobs ${JOBS_PER_TASK} ...'"
        exit 0
    fi

    active=0
    for shard in $(seq 0 "${ARRAY_LAST}"); do
        out="${RUN_DIR}/${JOB_NAME}_${shard}.out"
        err="${RUN_DIR}/${JOB_NAME}_${shard}.err"
        (
            printf '[build-sifs] submitting direct srun shard=%s at %s\n' "${shard}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            srun \
                --nodes=1 \
                --ntasks=1 \
                --cpus-per-task="${CPUS_PER_TASK}" \
                --mem="${MEM}" \
                --account="${ACCOUNT}" \
                --job-name="${JOB_NAME}-${shard}" \
                ${PARTITION:+--partition="${PARTITION}"} \
                ${CONSTRAINT:+--constraint="${CONSTRAINT}"} \
                --time="${WALL_TIME}" \
                --container-image="${BUILD_CONTAINER_IMAGE}" \
                --container-mounts="${BUILD_CONTAINER_MOUNTS}" \
                --container-workdir="${PROJECT_ROOT}" \
                --container-writable \
                --no-container-mount-home \
                bash -lc "set -euo pipefail
cd ${PR_Q}
source examples/swegym_slime_grpo/env.cwdfw.sh
export POLAR_JOB_CACHE_ROOT=\"/tmp/polar-sifbuild-\${SLURM_JOB_ID:-manual}-${shard}\"
rm -rf \"\${POLAR_JOB_CACHE_ROOT}\"
mkdir -p \"\${POLAR_JOB_CACHE_ROOT}\"
python examples/swegym_slime_grpo/build_sifs.py --num-shards \"${SHARDS}\" --shard-index \"${shard}\" --jobs \"${JOBS_PER_TASK}\" ${FORCE_FLAG}"
        ) >"${out}" 2>"${err}" &
        active=$((active + 1))
        if [ "${active}" -ge "${ARRAY_PARALLEL}" ]; then
            wait -n || true
            active=$((active - 1))
        fi
    done
    wait || true
    echo "Direct srun launch complete. Logs: ${RUN_DIR}"
    exit 0
fi

if [ "${LAUNCHER}" != "sbatch-container" ] && [ "${LAUNCHER}" != "sbatch-host" ]; then
    echo "ERROR: SIF_BUILD_LAUNCHER must be 'srun', 'sbatch-container', or 'sbatch-host'." >&2
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
    echo "#SBATCH --export=NONE"
    if [ "${LAUNCHER}" = "sbatch-container" ]; then
        cat <<EOF

set -euo pipefail
echo "[build-sifs] allocation started on \$(hostname) task=\${SLURM_ARRAY_TASK_ID:-0} at \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CONTAINER_IMAGE=${IMAGE_Q}
CONTAINER_MOUNTS=${MOUNTS_Q}
srun \\
  --overlap \\
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
source examples/swegym_slime_grpo/env.cwdfw.sh
export POLAR_JOB_CACHE_ROOT=\"/tmp/polar-sifbuild-\${SLURM_ARRAY_JOB_ID:-\${SLURM_JOB_ID}}-\${SLURM_ARRAY_TASK_ID:-0}\"
rm -rf \"\\\${POLAR_JOB_CACHE_ROOT}\"
mkdir -p \"\\\${POLAR_JOB_CACHE_ROOT}\"
PY=\\\$(command -v python3 || command -v python)
\"\\\${PY}\" examples/swegym_slime_grpo/build_sifs.py \\
  --num-shards \"${SHARDS}\" \\
  --shard-index \"\${SLURM_ARRAY_TASK_ID:-0}\" \\
  --jobs \"${JOBS_PER_TASK}\" ${FORCE_FLAG}"
EOF
    else
        cat <<EOF

set -euo pipefail
echo "[build-sifs] host allocation started on \$(hostname) task=\${SLURM_ARRAY_TASK_ID:-0} at \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd ${PR_Q}
source examples/swegym_slime_grpo/env.cwdfw.sh
export POLAR_JOB_CACHE_ROOT="/tmp/polar-sifbuild-\${SLURM_ARRAY_JOB_ID:-\${SLURM_JOB_ID}}-\${SLURM_ARRAY_TASK_ID:-0}"
rm -rf "\${POLAR_JOB_CACHE_ROOT}"
mkdir -p "\${POLAR_JOB_CACHE_ROOT}"
python3 examples/swegym_slime_grpo/build_sifs.py \\
  --num-shards "${SHARDS}" \\
  --shard-index "\${SLURM_ARRAY_TASK_ID:-0}" \\
  --jobs "${JOBS_PER_TASK}" ${FORCE_FLAG}
EOF
    fi
} > "${SBATCH_SCRIPT}"

if [ "${SIF_BUILD_DRY_RUN:-0}" = "1" ]; then
    echo "Dry run only; generated sbatch script: ${SBATCH_SCRIPT}"
    bash -n "${SBATCH_SCRIPT}"
    sed -n '1,120p' "${SBATCH_SCRIPT}"
    exit 0
fi

JOB_ID=$(sbatch --parsable "${SBATCH_SCRIPT}")

echo "Submitted SIF build array job: ${JOB_ID}"
echo "  shards: ${SHARDS}; array parallelism: ${ARRAY_PARALLEL}; jobs per task: ${JOBS_PER_TASK}"
echo "  sbatch script: ${SBATCH_SCRIPT}"
echo "  logs: tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}_0.out"
