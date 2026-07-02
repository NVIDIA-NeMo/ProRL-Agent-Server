#!/usr/bin/env bash
# Keep every arm of a submitted mixed-node TMax matrix progressing across the
# cluster's four-hour GPU allocation limit.  Each logical run has an
# independent run-state lock, so the selected watchers can safely share one
# small CPU allocation. By default this discovers the settings actually
# submitted under the stamp, which keeps older five-arm stamps compatible after
# new matrix settings are added. Set TMAX_MATRIX_WATCH_SETTINGS to a
# space-separated list (or "all") to require an explicit subset.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Slurm copies a script submitted directly with ``sbatch path/to/script`` into
# its private spool directory, so BASH_SOURCE no longer points at the checkout.
# Recover the real source directory from the submit cwd in that mode.  A
# ``sbatch --wrap='bash /absolute/path/watch_matrix.sh'`` launch already
# has the correct BASH_SOURCE and simply skips this fallback.
if [ ! -s "${SCRIPT_DIR}/run_every_10_minutes.sh" ] && \
   [ -n "${SLURM_SUBMIT_DIR:-}" ] && \
   [ -s "${SLURM_SUBMIT_DIR}/examples/tmax_slime_grpo/run_every_10_minutes.sh" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/tmax_slime_grpo"
fi
if [ ! -s "${SCRIPT_DIR}/run_every_10_minutes.sh" ]; then
    echo "ERROR: cannot locate run_every_10_minutes.sh from ${SCRIPT_DIR}" >&2
    exit 1
fi
# shellcheck source=./matrix_settings.sh
source "${SCRIPT_DIR}/matrix_settings.sh"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"

MATRIX_STAMP="${TMAX_MATRIX_STAMP:?set TMAX_MATRIX_STAMP to the submitted matrix stamp}"
POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${SPILOT_ROOT}/data}"
SLEEP_SECONDS="${TMAX_MATRIX_WATCH_SLEEP_SECONDS:-60}"
if ! [[ "${SLEEP_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TMAX_MATRIX_WATCH_SLEEP_SECONDS must be a positive integer" >&2
    exit 2
fi

MATRIX_TOPOLOGY_SCOPE="${TMAX_MATRIX_TOPOLOGY_SCOPE:-all}"
if ! tmax_matrix_validate_topology_scope "${MATRIX_TOPOLOGY_SCOPE}"; then
    exit 2
fi

matrix_run_id() {
    local setting="$1" topology_tag
    topology_tag="$(tmax_matrix_setting_topology "${setting}")" || return
    printf "tmax-%s-%s-%s\n" "${topology_tag}" "${setting}" "${MATRIX_STAMP}"
}

declare -a SETTINGS=()
if [ "${TMAX_MATRIX_WATCH_SETTINGS:-}" = all ]; then
    if ! tmax_matrix_settings_for_scope SETTINGS "${MATRIX_TOPOLOGY_SCOPE}"; then
        exit 2
    fi
elif [ -n "${TMAX_MATRIX_WATCH_SETTINGS:-}" ]; then
    declare -a REQUESTED_SETTINGS=()
    read -r -a REQUESTED_SETTINGS <<<"${TMAX_MATRIX_WATCH_SETTINGS}"
    if ! tmax_matrix_select_settings SETTINGS "${MATRIX_TOPOLOGY_SCOPE}" "${REQUESTED_SETTINGS[@]}"; then
        exit 2
    fi
else
    declare -a SCOPED_SETTINGS=()
    if ! tmax_matrix_settings_for_scope SCOPED_SETTINGS "${MATRIX_TOPOLOGY_SCOPE}"; then
        exit 2
    fi
    for setting in "${SCOPED_SETTINGS[@]}"; do
        run_id="$(matrix_run_id "${setting}")"
        state_file="${POLAR_DATA_ROOT}/runs/${run_id}/run_state.env"
        if [ -s "${state_file}" ]; then
            SETTINGS+=("${setting}")
        fi
    done
    if [ "${#SETTINGS[@]}" -eq 0 ]; then
        echo "ERROR: no submitted matrix run states found for stamp ${MATRIX_STAMP}" >&2
        exit 1
    fi
fi

declare -a PIDS=()
terminate_children() {
    if [ "${#PIDS[@]}" -gt 0 ]; then
        kill -TERM "${PIDS[@]}" 2>/dev/null || true
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
}
trap terminate_children TERM INT

for setting in "${SETTINGS[@]}"; do
    run_id="$(matrix_run_id "${setting}")"
    state_file="${POLAR_DATA_ROOT}/runs/${run_id}/run_state.env"
    if [ ! -s "${state_file}" ]; then
        echo "ERROR: matrix run state is missing or empty: ${state_file}" >&2
        terminate_children
        exit 1
    fi
    echo "[matrix watch] starting run_id=${run_id} state=${state_file}"
    (
        # watch_training.sh loads the selected state only when RUN_ID is not
        # inherited from the submitting shell.
        unset RUN_ID
        export POLAR_DATA_ROOT
        export TMAX_RUN_STATE_FILE="${state_file}"
        export TMAX_WATCH_SLEEP_SECONDS="${SLEEP_SECONDS}"
        exec bash "${SCRIPT_DIR}/run_every_10_minutes.sh"
    ) &
    PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || status=1
done
exit "${status}"
