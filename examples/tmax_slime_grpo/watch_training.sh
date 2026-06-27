#!/usr/bin/env bash
# Monitor one logical TMax run and relaunch it from its latest checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"

RELAUNCH=false
LOOP=false
SLEEP_SECONDS="${TMAX_WATCH_SLEEP_SECONDS:-600}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --relaunch) RELAUNCH=true ;;
        --loop) LOOP=true ;;
        --sleep-seconds) shift; SLEEP_SECONDS="${1:?missing seconds}" ;;
        *) echo "Usage: $0 [--relaunch] [--loop] [--sleep-seconds N]" >&2; exit 2 ;;
    esac
    shift
done
[[ "$SLEEP_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: sleep seconds must be positive" >&2; exit 2; }

export POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${SPILOT_ROOT}/data}"
export TMAX_RUN_STATE_FILE="${TMAX_RUN_STATE_FILE:-${POLAR_DATA_ROOT}/runs/tmax_slime_grpo/current_run.env}"
# shellcheck source=./run_state.sh
source "${SCRIPT_DIR}/run_state.sh"

command -v flock >/dev/null || { echo "ERROR: flock is required" >&2; exit 1; }
command -v squeue >/dev/null || { echo "ERROR: squeue is required" >&2; exit 1; }
mkdir -p "$(dirname "${TMAX_RUN_STATE_FILE}")"
exec 9>"${TMAX_RUN_STATE_FILE}.lock"
if ! flock -n 9; then
    echo "[tmax watch] another watcher owns ${TMAX_RUN_STATE_FILE}"
    exit 0
fi

# An explicit RUN_ID starts or selects that run. Otherwise continue the run
# recorded by submit_slurm.sh or an earlier watcher invocation.
LOADED_RUN_STATE=false
if [ -z "${RUN_ID:-}" ] && [ -s "${TMAX_RUN_STATE_FILE}" ]; then
    tmax_load_run_state "${TMAX_RUN_STATE_FILE}"
    LOADED_RUN_STATE=true
fi
# shellcheck source=./env.cwdfw.sh
source "${SCRIPT_DIR}/env.cwdfw.sh" >/dev/null
if [ -n "${TMAX_TARGET_ITER:-}" ] && ! [[ "${TMAX_TARGET_ITER}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: TMAX_TARGET_ITER must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "${ROLLOUT_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${NUM_EPOCH}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ROLLOUT_BATCH_SIZE and NUM_EPOCH must be positive integers" >&2
    exit 2
fi
if [ -z "${TMAX_PREPARE_DATA+x}" ]; then
    if [ "$LOADED_RUN_STATE" = true ] || [ -s "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
        export TMAX_PREPARE_DATA=0
    else
        export TMAX_PREPARE_DATA=1
    fi
fi
tmax_write_run_state "${TMAX_RUN_STATE_FILE}"

WATCH_COMPLETE=false

latest_iter() {
    local pointer="${SAVE_DIR}/latest_checkpointed_iteration.txt"
    local value
    if [ ! -s "$pointer" ]; then
        printf '%s\n' -1
        return
    fi
    value="$(tr -d '[:space:]' <"$pointer")"
    if [[ "$value" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$value"
    else
        printf '%s\n' -1
    fi
}

target_iter() {
    if [ -n "${TMAX_TARGET_ITER:-}" ]; then
        printf '%s\n' "${TMAX_TARGET_ITER}"
        return
    fi
    [ -s "${TMAX_TRAIN_DATA}" ] || return 1
    local samples rollouts
    samples="$(awk 'NF { count += 1 } END { print count + 0 }' "${TMAX_TRAIN_DATA}")"
    [ "$samples" -gt 0 ] || return 1
    rollouts="$(( (samples + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE * NUM_EPOCH ))"
    printf '%s\n' "$((rollouts - 1))"
}

check_once() {
    local iter target="" active owner
    iter="$(latest_iter)"
    target="$(target_iter || true)"

    if [ -f "${TRAINING_COMPLETE_MARKER}" ]; then
        echo "[tmax watch] training complete marker found: ${TRAINING_COMPLETE_MARKER}"
        WATCH_COMPLETE=true
        return
    fi
    if [ -n "$target" ] && [ "$iter" -ge "$target" ]; then
        printf '[tmax watch] target reached: latest_iter=%s target=%s\n' "$iter" "$target"
        WATCH_COMPLETE=true
        return
    fi

    owner="${SLURM_USER:-${USER:-$(id -un)}}"
    active="$(squeue -h -u "$owner" -n "${JOB_NAME:-polar-tmax-grpo}" -t PD,R,CF,CG -o '%i %t %j %R' || true)"
    if [ -n "$active" ]; then
        printf '[tmax watch] job active; run_id=%s latest_iter=%s target=%s\n%s\n' \
            "$RUN_ID" "$iter" "${target:-unknown}" "$active"
        return
    fi

    printf '[tmax watch] no active job; run_id=%s latest_iter=%s target=%s save_dir=%s\n' \
        "$RUN_ID" "$iter" "${target:-unknown}" "$SAVE_DIR"
    if [ "$RELAUNCH" = true ]; then
        if SUBMIT_BACKEND=sbatch bash "${SCRIPT_DIR}/submit_slurm.sh"; then
            export TMAX_PREPARE_DATA=0
            tmax_write_run_state "${TMAX_RUN_STATE_FILE}"
            echo "[tmax watch] submission succeeded"
        else
            echo "[tmax watch] submission failed; will retry after ${SLEEP_SECONDS}s" >&2
        fi
    fi
}

echo "[tmax watch] state=${TMAX_RUN_STATE_FILE} run_id=${RUN_ID} save_dir=${SAVE_DIR}"
if [ "$LOOP" = true ]; then
    while true; do
        date -u '+[tmax watch] %Y-%m-%dT%H:%M:%SZ'
        check_once
        if [ "$WATCH_COMPLETE" = true ]; then
            exit 0
        fi
        sleep "$SLEEP_SECONDS"
    done
else
    check_once
fi
