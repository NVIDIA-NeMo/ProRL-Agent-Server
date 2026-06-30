#!/usr/bin/env bash
# Convert the TMax model into a release torch_dist checkpoint atomically.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=./env.cwdfw.sh
source "${SCRIPT_DIR}/env.cwdfw.sh"

FINAL_DIR="${REF_LOAD}"
if [ -s "${FINAL_DIR}/latest_checkpointed_iteration.txt" ] && \
   [ "$(tr -d '[:space:]' <"${FINAL_DIR}/latest_checkpointed_iteration.txt")" = release ] && \
   find "${FINAL_DIR}/release" -maxdepth 1 -type f -name '*.distcp' -size +0c \
       -print -quit | grep -q .; then
    echo "[tmax convert] checkpoint already ready: ${FINAL_DIR}"
    exit 0
fi
if [ -e "${FINAL_DIR}" ]; then
    echo "ERROR: refusing to replace incomplete checkpoint path: ${FINAL_DIR}" >&2
    exit 1
fi

case "${FINAL_DIR}" in
    "${POLAR_DATA_ROOT}/checkpoints/Qwen3.5-9B_torch_dist") ;;
    *)
        echo "ERROR: unexpected TMax conversion destination: ${FINAL_DIR}" >&2
        exit 1
        ;;
esac

STAGING_DIR="${FINAL_DIR}.tmp-${SLURM_JOB_ID:-$$}"
case "${STAGING_DIR}" in
    "${FINAL_DIR}.tmp-"*) rm -rf -- "${STAGING_DIR}" ;;
    *)
        echo "ERROR: unsafe conversion staging path: ${STAGING_DIR}" >&2
        exit 1
        ;;
esac

export PYTHON_BIN="${PYTHON_BIN:-${POLR_TRAIN_VENV}/bin/python3}"
export TORCH_DIST_DIR="${STAGING_DIR}"
bash "${PROJECT_ROOT}/examples/swegym_slime_grpo/convert_weights.sh"

if [ "$(tr -d '[:space:]' <"${STAGING_DIR}/latest_checkpointed_iteration.txt")" != release ] || \
   ! find "${STAGING_DIR}/release" -maxdepth 1 -type f -name '*.distcp' -size +0c \
       -print -quit | grep -q .; then
    echo "ERROR: converted checkpoint failed release validation: ${STAGING_DIR}" >&2
    exit 1
fi
mv -- "${STAGING_DIR}" "${FINAL_DIR}"
echo "[tmax convert] ready: ${FINAL_DIR}"
