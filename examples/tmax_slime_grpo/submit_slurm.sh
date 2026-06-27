#!/usr/bin/env bash
# Prepare a deterministic TMax prompt set, then launch multi-node Slime GRPO.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=./env.cwdfw.sh
source "${SCRIPT_DIR}/env.cwdfw.sh"
# shellcheck source=./run_state.sh
source "${SCRIPT_DIR}/run_state.sh"

if [ ! -x "${TMAX_SIF_PYTHON_BIN}" ]; then
    echo "ERROR: Python not executable: ${TMAX_SIF_PYTHON_BIN}" >&2
    exit 1
fi
case "${TMAX_AGENT_HARNESS}" in
    mini_swe_agent)
        if [ ! -x "${MINI_SWE_AGENT_BIN}" ]; then
            echo "ERROR: shared mini-swe-agent runtime not found at ${MINI_SWE_AGENT_BIN}" >&2
            echo "  Run: bash ${SCRIPT_DIR}/prepare_mini_swe_agent.sh" >&2
            exit 1
        fi
        ;;
    codex)
        if [ ! -x "${AGENT_CLI_DIR}/bin/codex" ]; then
            echo "ERROR: shared Codex CLI not found at ${AGENT_CLI_DIR}/bin/codex" >&2
            echo "  Run: bash ${SCRIPT_DIR}/prepare_agent_cli.sh" >&2
            exit 1
        fi
        ;;
esac
if [ "${TMAX_REQUIRE_WANDB}" = "1" ] && { [ -z "${WANDB_API_KEY:-}" ] || [ "${WANDB_MODE}" != "online" ]; }; then
    echo "ERROR: TMAX_REQUIRE_WANDB=1 requires WANDB_API_KEY and WANDB_MODE=online" >&2
    exit 1
fi

if [ "${TMAX_PREPARE_DATA:-1}" = "1" ] || [ ! -s "${TMAX_TRAIN_DATA}" ]; then
    PREPARE_ARGS=(
        --dataset-dir "${TMAX_DATASET_DIR}"
        --image-dir "${APPTAINER_IMAGE_DIR}"
        --output "${TMAX_TRAIN_DATA}"
        --max-tasks "${TMAX_MAX_TASKS}"
    )
    if [ "${TMAX_ONLY_READY}" = "1" ]; then
        PREPARE_ARGS+=(--only-ready)
    fi
    "${TMAX_SIF_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data.py" "${PREPARE_ARGS[@]}"
fi

export PROMPT_DATA="${TMAX_TRAIN_DATA}"
export TMAX_PREPARE_DATA=0
export POLAR_TRAIN_RUN_SCRIPT="${SCRIPT_DIR}/run.sh"
export JOB_NAME="${JOB_NAME:-polar-tmax-grpo}"
export WANDB_PROJECT WANDB_GROUP RUN_ID SAVE_DIR

if [ "${TMAX_PERSIST_RUN_STATE:-1}" = "1" ] && [ "${SUBMIT_DRY_RUN:-0}" != "1" ]; then
    tmax_write_run_state "${TMAX_RUN_STATE_FILE}"
    echo "[tmax submit] run state: ${TMAX_RUN_STATE_FILE}"
fi

exec bash "${PROJECT_ROOT}/examples/swegym_slime_grpo/submit_slurm.sh"
