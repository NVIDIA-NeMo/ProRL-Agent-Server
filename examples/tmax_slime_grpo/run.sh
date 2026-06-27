#!/usr/bin/env bash
# Run TMax GRPO through the shared, Slime-0.3.0-compatible training launcher.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=./env.cwdfw.sh
source "${SCRIPT_DIR}/env.cwdfw.sh"

slurm_duration_seconds() {
    local spec="$1" rest days=0 has_days=0
    local -a fields
    if [[ "$spec" == *-* ]]; then
        days="${spec%%-*}"
        rest="${spec#*-}"
        has_days=1
    else
        rest="$spec"
    fi
    IFS=: read -r -a fields <<<"$rest"
    if ! [[ "$days" =~ ^[0-9]+$ ]] || [ "${#fields[@]}" -lt 1 ] || [ "${#fields[@]}" -gt 3 ]; then
        return 1
    fi
    local value
    for value in "${fields[@]}"; do
        [[ "$value" =~ ^[0-9]+$ ]] || return 1
    done

    local hours=0 minutes=0 seconds=0
    if [ "$has_days" = "1" ]; then
        case "${#fields[@]}" in
            1) hours=$((10#${fields[0]})) ;;
            2) hours=$((10#${fields[0]})); minutes=$((10#${fields[1]})) ;;
            3) hours=$((10#${fields[0]})); minutes=$((10#${fields[1]})); seconds=$((10#${fields[2]})) ;;
        esac
    else
        case "${#fields[@]}" in
            1) minutes=$((10#${fields[0]})) ;;
            2) minutes=$((10#${fields[0]})); seconds=$((10#${fields[1]})) ;;
            3) hours=$((10#${fields[0]})); minutes=$((10#${fields[1]})); seconds=$((10#${fields[2]})) ;;
        esac
    fi
    printf '%s\n' "$((10#$days * 86400 + hours * 3600 + minutes * 60 + seconds))"
}

if [ "${TMAX_ENABLE_GRACEFUL_EXIT}" = "1" ] && [ -z "${SLIME_GRACEFUL_EXIT_AT_UNIX_TIME:-}" ]; then
    if ! WALL_TIME_SECONDS="$(slurm_duration_seconds "${WALL_TIME}")"; then
        echo "ERROR: unsupported Slurm WALL_TIME format: ${WALL_TIME}" >&2
        exit 1
    fi
    if ! [[ "${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS}" =~ ^[0-9]+$ ]] || \
       [ "${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS}" -ge "${WALL_TIME_SECONDS}" ]; then
        echo "ERROR: TMAX_GRACEFUL_EXIT_BUFFER_SECONDS must be a non-negative integer smaller than WALL_TIME" >&2
        exit 1
    fi
    export SLIME_GRACEFUL_EXIT_AT_UNIX_TIME="$(( $(date +%s) + WALL_TIME_SECONDS - TMAX_GRACEFUL_EXIT_BUFFER_SECONDS ))"
fi
if [ "${SLURM_PROCID:-0}" = "0" ] && [ -n "${SLIME_GRACEFUL_EXIT_AT_UNIX_TIME:-}" ]; then
    echo "[tmax run] graceful checkpoint deadline: $(date -u -d "@${SLIME_GRACEFUL_EXIT_AT_UNIX_TIME}" '+%Y-%m-%dT%H:%M:%SZ')"
fi

if [ -n "${PROMPT_DATA:-}" ]; then
    PREPARE_DATA_DEFAULT=0
else
    export PROMPT_DATA="${TMAX_TRAIN_DATA}"
    PREPARE_DATA_DEFAULT=1
fi
TMAX_PREPARE_DATA="${TMAX_PREPARE_DATA:-${PREPARE_DATA_DEFAULT}}"

if [ "${TMAX_PREPARE_DATA}" = "1" ] || [ ! -f "${PROMPT_DATA}" ]; then
    PREPARE_ARGS=(
        --dataset-dir "${TMAX_DATASET_DIR}"
        --image-dir "${APPTAINER_IMAGE_DIR}"
        --output "${PROMPT_DATA}"
        --max-tasks "${TMAX_MAX_TASKS}"
    )
    if [ "${TMAX_ONLY_READY}" = "1" ]; then
        PREPARE_ARGS+=(--only-ready)
    fi
    "${TMAX_SIF_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data.py" "${PREPARE_ARGS[@]}"
fi

export REQUIRE_SWEGYM_HARNESS=0
export TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${SCRIPT_DIR}/topology.yaml}"
export POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${SCRIPT_DIR}/polar_config.yaml}"
export RUN_DIR="${RUN_DIR:-${POLAR_DATA_ROOT}/runs/${RUN_ID}/job-${SLURM_JOB_ID:-manual}}"
export TOPOLOGY_PATH="${TOPOLOGY_PATH:-${RUN_DIR}/topology.yaml}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${RUN_DIR}/polar_config.yaml}"
export GPU_MONITOR_PREFIX="${GPU_MONITOR_PREFIX:-polar_tmax_system}"

exec bash "${PROJECT_ROOT}/examples/swegym_slime_grpo/run.sh"
