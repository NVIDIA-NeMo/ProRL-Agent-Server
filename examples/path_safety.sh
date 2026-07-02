#!/usr/bin/env bash

# Side-effect-free path validation shared by example launchers. Callers must
# provide a narrow allowed root before recursively removing generated assets.

polar_normalize_absolute_path() {
    local label="${1:?missing path label}"
    local path="${2:-}"
    local normalized

    if [ -z "${path}" ] || [[ "${path}" != /* ]]; then
        echo "ERROR: ${label} must be a non-empty absolute path, got ${path:-<empty>}" >&2
        return 1
    fi
    if ! normalized="$(realpath -m -s -- "${path}")"; then
        echo "ERROR: could not normalize ${label}: ${path}" >&2
        return 1
    fi
    printf '%s\n' "${normalized}"
}

polar_require_absolute_path() {
    local label="${1:?missing path label}"
    local normalized
    normalized="$(polar_normalize_absolute_path "${label}" "${2:-}")" || return
    if [ "${normalized}" = / ]; then
        echo "ERROR: ${label} must not be the filesystem root" >&2
        return 1
    fi
}

polar_validate_removal_path() {
    local label="${1:?missing path label}"
    local path="${2:-}"
    local allowed_root="${3:-}"
    local required_prefix="${4:-}"
    local normalized_path normalized_root resolved_path resolved_root relative

    normalized_path="$(polar_normalize_absolute_path "${label}" "${path}")" || return
    normalized_root="$(polar_normalize_absolute_path "${label} allowed root" "${allowed_root}")" || return
    if [ "${normalized_root}" = / ]; then
        echo "ERROR: refusing to use / as the allowed removal root for ${label}" >&2
        return 1
    fi
    case "${normalized_path}" in
        "${normalized_root}"/*) ;;
        *)
            echo "ERROR: refusing to remove ${label} outside ${normalized_root}: ${normalized_path}" >&2
            return 1
            ;;
    esac

    if ! resolved_path="$(realpath -m -- "${path}")" || \
       ! resolved_root="$(realpath -m -- "${allowed_root}")"; then
        echo "ERROR: could not resolve ${label} against its allowed root" >&2
        return 1
    fi
    if [ "${resolved_root}" = / ]; then
        echo "ERROR: refusing to use a removal root that resolves to / for ${label}" >&2
        return 1
    fi
    case "${resolved_path}" in
        "${resolved_root}"|"${resolved_root}"/*) ;;
        *)
            echo "ERROR: refusing to remove ${label} through a symlink outside ${resolved_root}: ${resolved_path}" >&2
            return 1
            ;;
    esac

    relative="${normalized_path#"${normalized_root}"/}"
    if [ -n "${required_prefix}" ] && \
       [ "${relative#"${required_prefix}"}" = "${relative}" ]; then
        echo "ERROR: ${label} must begin with ${normalized_root}/${required_prefix}, got ${normalized_path}" >&2
        return 1
    fi
    printf '%s\n' "${normalized_path}"
}

polar_safe_remove_tree() {
    local normalized_path
    normalized_path="$(polar_validate_removal_path "$@")" || return
    rm -rf -- "${normalized_path}"
}
