#!/usr/bin/env bash
# Shared TMax comparison-matrix settings and topology selection.

readonly -a TMAX_MATRIX_SETTINGS=(
    qwen35-4b-fidelity
    qwen35-4b-fidelity-8n
    qwen35-4b-fidelity-8n-b16n8-traj
    qwen35-9b-baseline-a2-full65k
    qwen35-9b-b16n16-a2-full65k
    qwen35-9b-async4-full65k
    qwen35-9b-lr5e7-a2-full65k
    qwen35-9b-lr2e6-a2-full65k
)

tmax_matrix_validate_topology_scope() {
    local scope="${1:?missing matrix topology scope}"
    case "${scope}" in
        all|4n32|8n64) ;;
        *)
            echo "ERROR: matrix topology scope must be all, 4n32, or 8n64; got ${scope}" >&2
            return 2
            ;;
    esac
}

tmax_matrix_is_known_setting() {
    local candidate="${1:?missing matrix setting}" setting
    for setting in "${TMAX_MATRIX_SETTINGS[@]}"; do
        [ "${candidate}" = "${setting}" ] && return 0
    done
    return 1
}

tmax_matrix_setting_topology() {
    local setting="${1:?missing matrix setting}"
    if ! tmax_matrix_is_known_setting "${setting}"; then
        echo "ERROR: unknown matrix setting: ${setting}" >&2
        return 2
    fi
    case "${setting}" in
        qwen35-4b-fidelity-8n|qwen35-4b-fidelity-8n-b16n8-traj)
            printf '%s\n' 8n64
            ;;
        *)
            printf '%s\n' 4n32
            ;;
    esac
}

tmax_matrix_settings_for_scope() {
    local output_name="${1:?missing output array name}"
    local scope="${2:?missing matrix topology scope}"
    local -n output_ref="${output_name}"
    local setting topology
    tmax_matrix_validate_topology_scope "${scope}" || return
    output_ref=()
    for setting in "${TMAX_MATRIX_SETTINGS[@]}"; do
        topology="$(tmax_matrix_setting_topology "${setting}")" || return
        if [ "${scope}" = all ] || [ "${topology}" = "${scope}" ]; then
            output_ref+=("${setting}")
        fi
    done
}

tmax_matrix_select_settings() {
    local output_name="${1:?missing output array name}"
    local scope="${2:?missing matrix topology scope}"
    shift 2
    local -n output_ref="${output_name}"
    local candidate topology
    tmax_matrix_validate_topology_scope "${scope}" || return
    output_ref=()

    if [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && [ "$1" = all ]; }; then
        tmax_matrix_settings_for_scope "${output_name}" "${scope}"
        return
    fi

    for candidate in "$@"; do
        if ! tmax_matrix_is_known_setting "${candidate}"; then
            echo "ERROR: unknown matrix setting: ${candidate}" >&2
            return 2
        fi
        topology="$(tmax_matrix_setting_topology "${candidate}")" || return
        if [ "${scope}" != all ] && [ "${topology}" != "${scope}" ]; then
            echo "ERROR: matrix setting ${candidate} uses topology ${topology}, outside ${scope} scope" >&2
            return 2
        fi
        output_ref+=("${candidate}")
    done
}
