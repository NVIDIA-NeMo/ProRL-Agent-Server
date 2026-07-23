#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${POLAR_DATA_ROOT:-${PROJECT_ROOT}/local_data}"
PYTHON_BIN="${TMAX_SIF_PYTHON_BIN:-${DATA_ROOT}/train_runtime_venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

export SKILL2ENV_TASKS_DIR="${SKILL2ENV_TASKS_DIR:-/lustre/fsw/portfolios/llmservice/users/haozh/data/skill2env/skill2env_batch_5}"
export SKILL2ENV_IMAGE_DIR="${SKILL2ENV_IMAGE_DIR:-/lustre/fsw/portfolios/llmservice/users/haozh/data/skill2env/skill2env_batch_5_sif}"
export SKILL2ENV_TRAIN_DATA="${SKILL2ENV_TRAIN_DATA:-${DATA_ROOT}/seeds/skill2env-batch5-ready.jsonl}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/examples/tmax_slime_grpo/prepare_harbor_eval.py" \
    --tasks-dir "${SKILL2ENV_TASKS_DIR}" \
    --image-dir "${SKILL2ENV_IMAGE_DIR}" \
    --output "${SKILL2ENV_TRAIN_DATA}" \
    --dataset-name skill2env_batch_5 \
    --dataset-revision skill2env_batch_5_local_2026_07 \
    --max-tasks -1 \
    --only-ready \
    --skip-unsupported \
    --exclude-task-id task_api-designer_577cac0e96a34da0aadf338522460cd7 \
    --agent-timeout-cap 600 \
    --verifier-timeout-cap 600 \
    --timeout-overhead 300 \
    --agent-step-limit 64

printf '%s\n' "${SKILL2ENV_TRAIN_DATA}"
