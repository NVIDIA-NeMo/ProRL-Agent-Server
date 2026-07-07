#!/usr/bin/env bash

TMAX_LIFECYCLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

tmax_require_spilot_credentials() {
    local stage="${1:-operation}"
    if [ "${TMAX_AGENT_HARNESS:-}" != "spilot_router" ]; then
        return 0
    fi
    if [ -z "${POLAR_CONTROL_PLANE_TOKEN:-}" ]; then
        echo "ERROR: ${stage}: SPilot Router requires POLAR_CONTROL_PLANE_TOKEN; use the SPilot submit wrapper" >&2
        return 1
    fi
    if ! [[ "${POLAR_CONTROL_PLANE_TOKEN}" =~ ^[0-9A-Za-z_-]{32,128}$ ]]; then
        echo "ERROR: ${stage}: POLAR_CONTROL_PLANE_TOKEN must be a 32-128 character opaque token" >&2
        return 1
    fi
    if [ -z "${POLAR_NVIDIA_API_KEY:-}" ]; then
        echo "ERROR: ${stage}: SPilot Router requires POLAR_NVIDIA_API_KEY; use the SPilot submit wrapper" >&2
        return 1
    fi
    if [ -z "${POLAR_MODEL_POOL_BASE_URL:-}" ]; then
        echo "ERROR: ${stage}: SPilot Router requires POLAR_MODEL_POOL_BASE_URL; use the SPilot submit wrapper" >&2
        return 1
    fi
}

tmax_require_spilot_entrypoints() {
    local stage="${1:-operation}"
    local spilot_dir entrypoint name file expected
    if [ "${TMAX_AGENT_HARNESS:-}" != "spilot_router" ]; then
        return 0
    fi
    spilot_dir="$(cd -- "${TMAX_LIFECYCLE_DIR}/../spilot_router_slime_grpo" &>/dev/null && pwd)"
    for entrypoint in \
        "TMAX_SUBMIT_SCRIPT:submit_slurm.sh" \
        "POLAR_TRAIN_RUN_SCRIPT:run.sh" \
        "POLAR_CONFIG_TEMPLATE:polar_config.yaml" \
        "TOPOLOGY_TEMPLATE:topology.yaml"; do
        name="${entrypoint%%:*}"
        file="${entrypoint#*:}"
        expected="${spilot_dir}/${file}"
        if [ "${!name:-}" != "${expected}" ]; then
            echo "ERROR: ${stage}: SPilot Router requires ${name}=${expected}; got ${!name:-unset}" >&2
            return 1
        fi
    done
}

tmax_slurm_duration_seconds() {
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

tmax_configure_graceful_deadline() {
    local wall_time_seconds now_unix
    if [ "${TMAX_ENABLE_GRACEFUL_EXIT:-1}" != "1" ] || \
       [ -n "${SLIME_GRACEFUL_EXIT_AT_UNIX_TIME:-}" ]; then
        return
    fi
    if ! wall_time_seconds="$(tmax_slurm_duration_seconds "${WALL_TIME}")"; then
        echo "ERROR: unsupported Slurm WALL_TIME format: ${WALL_TIME}" >&2
        return 1
    fi
    if ! [[ "${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS}" =~ ^[0-9]+$ ]] || \
       [ "${TMAX_GRACEFUL_EXIT_BUFFER_SECONDS}" -ge "$wall_time_seconds" ]; then
        echo "ERROR: TMAX_GRACEFUL_EXIT_BUFFER_SECONDS must be a non-negative integer smaller than WALL_TIME" >&2
        return 1
    fi

    now_unix="${TMAX_NOW_UNIX:-$(date +%s)}"
    if ! [[ "$now_unix" =~ ^[0-9]+$ ]]; then
        echo "ERROR: current Unix time must be numeric, got ${now_unix}" >&2
        return 1
    fi
    if [[ "${SLURM_JOB_END_TIME:-}" =~ ^[0-9]+$ ]]; then
        # The allocation clock starts before Pyxis, Ray, model loading, and
        # SGLang initialization, so this is authoritative inside Slurm.
        export SLIME_GRACEFUL_EXIT_AT_UNIX_TIME="$((SLURM_JOB_END_TIME - TMAX_GRACEFUL_EXIT_BUFFER_SECONDS))"
        export TMAX_GRACEFUL_DEADLINE_SOURCE="SLURM_JOB_END_TIME"
    else
        export SLIME_GRACEFUL_EXIT_AT_UNIX_TIME="$((now_unix + wall_time_seconds - TMAX_GRACEFUL_EXIT_BUFFER_SECONDS))"
        export TMAX_GRACEFUL_DEADLINE_SOURCE="WALL_TIME fallback"
    fi
    if [ "$SLIME_GRACEFUL_EXIT_AT_UNIX_TIME" -le "$now_unix" ]; then
        echo "ERROR: graceful checkpoint deadline has already passed; reduce TMAX_GRACEFUL_EXIT_BUFFER_SECONDS or startup time" >&2
        return 1
    fi
}

tmax_validate_numbered_checkpoint() {
    local root="${1:?missing checkpoint root}"
    local context="${2:-ERROR: checkpoint}"
    local pointer value iteration_dir model_dir path state_path validation_error

    pointer="${root}/latest_checkpointed_iteration.txt"
    if [ ! -f "${pointer}" ] || [ ! -s "${pointer}" ]; then
        echo "${context}: checkpoint pointer is missing or empty: ${pointer}" >&2
        return 1
    fi
    value="$(tr -d '[:space:]' <"${pointer}")"
    if ! [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "${context}: invalid checkpoint iteration in ${pointer}: ${value}" >&2
        return 1
    fi

    printf -v iteration_dir 'iter_%07s' "${value}"
    iteration_dir="${iteration_dir// /0}"
    model_dir="${root}/${iteration_dir}"
    if [ ! -d "${model_dir}" ]; then
        echo "${context}: tracker ${value} has no matching model checkpoint directory" >&2
        echo "  Missing: ${model_dir}" >&2
        return 1
    fi
    for path in "${model_dir}/common.pt" "${model_dir}/.metadata"; do
        if [ ! -f "${path}" ] || [ ! -s "${path}" ]; then
            echo "${context}: model checkpoint ${value} is incomplete" >&2
            echo "  Missing or empty regular file: ${path}" >&2
            return 1
        fi
    done
    if ! validation_error="$(python3 "${TMAX_LIFECYCLE_DIR}/validate_torch_dist_checkpoint.py" "${model_dir}" 2>&1)"; then
        echo "${context}: model checkpoint ${value} has inconsistent distcp metadata" >&2
        if [ -n "${validation_error}" ]; then
            printf '  %s\n' "${validation_error}" >&2
        fi
        return 1
    fi

    state_path="${root}/rollout/global_dataset_state_dict_${value}.pt"
    if [ ! -f "${state_path}" ] || [ ! -s "${state_path}" ]; then
        echo "${context}: checkpoint ${value} has no matching rollout state" >&2
        echo "  Missing or empty regular file: ${state_path}" >&2
        return 1
    fi
    printf '%s\n' "${value}"
}
