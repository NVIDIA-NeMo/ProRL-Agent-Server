#!/usr/bin/env bash
# Monitor and safely resume the SPilot Router logical run.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
export POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${SPILOT_ROOT}/data}"
export TMAX_RUN_STATE_FILE="${TMAX_RUN_STATE_FILE:-${POLAR_DATA_ROOT}/runs/spilot_router_slime_grpo/current_run.env}"
# Existing state is authoritative and must be restored before current defaults
# validate admission, gateway fan-out, and timeout settings.
# shellcheck source=../tmax_slime_grpo/run_state.sh
source "${SCRIPT_DIR}/../tmax_slime_grpo/run_state.sh"
_SPILOT_REQUESTED_RUN_ID="${RUN_ID:-}"
if [ -s "${TMAX_RUN_STATE_FILE}" ]; then
    tmax_load_selected_run_state \
        "${TMAX_RUN_STATE_FILE}" "${_SPILOT_REQUESTED_RUN_ID}"
    tmax_restore_spilot_admission_resume_contract \
        "${TMAX_RUN_STATE_FILE}" "[spilot watch]"
fi
unset _SPILOT_REQUESTED_RUN_ID SPILOT_ROOT PROJECT_ROOT
# A watcher with no run state must bootstrap the exact same contract as the
# first submitter, rather than inheriting the shared TMax example defaults.
# shellcheck source=./experiment_defaults.sh
source "${SCRIPT_DIR}/experiment_defaults.sh"
export TMAX_SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_slurm.sh"
export POLAR_TRAIN_RUN_SCRIPT="${SCRIPT_DIR}/run.sh"
export POLAR_CONFIG_TEMPLATE="${SCRIPT_DIR}/polar_config.yaml"
export TOPOLOGY_TEMPLATE="${SCRIPT_DIR}/topology.yaml"

# Credentials are intentionally excluded from TMax run state. Every watcher
# process must receive them afresh and keeps them only in its environment; each
# relaunched job then gets the key through the mode-0600 submit environment.
if [ -z "${POLAR_NVIDIA_API_KEY:-${NVIDIA_API_KEY:-}}" ]; then
    echo "ERROR: NVIDIA_API_KEY is not set; load the credential before watching" >&2
    exit 1
fi
export POLAR_NVIDIA_API_KEY="${POLAR_NVIDIA_API_KEY:-${NVIDIA_API_KEY:-}}"
export POLAR_MODEL_POOL_BASE_URL="${POLAR_MODEL_POOL_BASE_URL:-${NVIDIA_BASE_URL:-https://integrate.api.nvidia.com/v1}}"

exec bash "${SCRIPT_DIR}/../tmax_slime_grpo/watch_training.sh" "$@"
