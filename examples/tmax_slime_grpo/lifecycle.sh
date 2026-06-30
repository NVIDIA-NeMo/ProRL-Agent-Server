#!/usr/bin/env bash

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
