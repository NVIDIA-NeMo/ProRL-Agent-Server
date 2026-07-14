#!/usr/bin/env bash

# Shared shell helpers for the Slime/Slurm launchers.  This file is sourced by
# both run_in_container.sh and run.sh, so keep it free of side effects.

# Eight TP=1 rollout engines need 264 ports per node: two fixed ports per
# engine plus a 31-port SGLang distributed-init range.  Reserve a 320-port
# block so find_available_port can skip a few long-lived services without
# crossing into Linux's ephemeral client-port range.
_POLAR_ROLLOUT_BASE_PORT_FLOOR=2048
_POLAR_ROLLOUT_PORT_BLOCK_SIZE=320
_POLAR_ROLLOUT_PORT_STRIDE=320
_POLAR_EPHEMERAL_PORT_LOWER_FALLBACK=32768
_POLAR_SGLANG_ROUTER_PORT_DEFAULT=8680
_POLAR_UNPRIVILEGED_PORT_FLOOR=1024

polar_ephemeral_port_lower_bound() {
    local range_file lower _upper

    if [ -n "${SLIME_EPHEMERAL_PORT_LOWER_BOUND:-}" ]; then
        lower="${SLIME_EPHEMERAL_PORT_LOWER_BOUND}"
    else
        range_file="${SLIME_IP_LOCAL_PORT_RANGE_PATH:-/proc/sys/net/ipv4/ip_local_port_range}"
        if ! read -r lower _upper < "$range_file" 2>/dev/null; then
            lower="${_POLAR_EPHEMERAL_PORT_LOWER_FALLBACK}"
        fi
    fi

    if ! [[ "$lower" =~ ^[0-9]+$ ]] || [ "$lower" -lt 1024 ] || [ "$lower" -gt 65535 ]; then
        echo "ERROR: invalid ephemeral port lower bound ${lower:-<empty>}" >&2
        return 1
    fi
    printf '%s\n' "$lower"
}

polar_max_safe_rollout_base_port() {
    local ephemeral_lower
    ephemeral_lower="$(polar_ephemeral_port_lower_bound)" || return 1
    if [ "$ephemeral_lower" -lt "$((_POLAR_ROLLOUT_BASE_PORT_FLOOR + _POLAR_ROLLOUT_PORT_BLOCK_SIZE))" ]; then
        echo "ERROR: ephemeral port range begins at ${ephemeral_lower}; no ${_POLAR_ROLLOUT_PORT_BLOCK_SIZE}-port rollout block fits above ${_POLAR_ROLLOUT_BASE_PORT_FLOOR}" >&2
        return 1
    fi
    printf '%s\n' "$((ephemeral_lower - _POLAR_ROLLOUT_PORT_BLOCK_SIZE))"
}

polar_validate_rollout_base_port() {
    local port="${1:-}"
    local ephemeral_lower max_safe
    if ! [[ "$port" =~ ^[0-9]+$ ]]; then
        echo "ERROR: SLIME_ROLLOUT_BASE_PORT must be an integer, got ${port:-<empty>}" >&2
        return 1
    fi
    ephemeral_lower="$(polar_ephemeral_port_lower_bound)" || return 1
    max_safe="$(polar_max_safe_rollout_base_port)" || return 1
    if [ "$port" -lt "$_POLAR_ROLLOUT_BASE_PORT_FLOOR" ] || [ "$port" -gt "$max_safe" ]; then
        echo "ERROR: SLIME_ROLLOUT_BASE_PORT=${port} is unsafe; its ${_POLAR_ROLLOUT_PORT_BLOCK_SIZE}-port block must fit between ${_POLAR_ROLLOUT_BASE_PORT_FLOOR} and $((ephemeral_lower - 1)) (base <= ${max_safe})" >&2
        return 1
    fi
}

polar_rollout_base_port_for_allocation() {
    local allocation_id="${1:?missing allocation id}"
    local numeric_id checksum slot slot_count max_safe

    # Preserve the familiar mapping for normal decimal job ids. Hash the full
    # identity for arrays/heterogeneous allocations so task suffixes do not all
    # collide on the same per-node port block.
    if [[ "$allocation_id" =~ ^[0-9]+$ ]]; then
        numeric_id="$allocation_id"
    else
        checksum="$(printf '%s' "$allocation_id" | cksum)"
        numeric_id="${checksum%% *}"
    fi

    # Keep the complete block below the *actual* kernel ephemeral lower bound.
    # This avoids a TOCTOU race where an outbound connection takes a port after
    # Slime checks it but before SGLang binds roughly a minute later.
    max_safe="$(polar_max_safe_rollout_base_port)" || return 1
    slot_count="$(((max_safe - _POLAR_ROLLOUT_BASE_PORT_FLOOR) / _POLAR_ROLLOUT_PORT_STRIDE + 1))"
    if [ "$slot_count" -gt 20 ]; then
        slot_count=20
    fi
    slot="$((10#${numeric_id} % slot_count))"
    printf '%s\n' "$((_POLAR_ROLLOUT_BASE_PORT_FLOOR + slot * _POLAR_ROLLOUT_PORT_STRIDE))"
}

polar_configure_rollout_base_port() {
    local origin allocation_id

    if [ -n "${SLIME_ROLLOUT_BASE_PORT:-}" ]; then
        origin="explicit override"
    elif [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        allocation_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        SLIME_ROLLOUT_BASE_PORT="$(polar_rollout_base_port_for_allocation "$allocation_id")"
        export SLIME_ROLLOUT_BASE_PORT
        origin="Slurm array allocation ${allocation_id}"
    elif [ -n "${SLURM_JOB_ID:-}" ]; then
        allocation_id="${SLURM_JOB_ID}"
        SLIME_ROLLOUT_BASE_PORT="$(polar_rollout_base_port_for_allocation "$allocation_id")"
        export SLIME_ROLLOUT_BASE_PORT
        origin="Slurm allocation ${allocation_id}"
    elif [ -n "${SLIME_ROLLOUT_BASE_PORT_FALLBACK:-}" ]; then
        SLIME_ROLLOUT_BASE_PORT="${SLIME_ROLLOUT_BASE_PORT_FALLBACK}"
        export SLIME_ROLLOUT_BASE_PORT
        origin="explicit non-Slurm fallback"
    else
        SLIME_ROLLOUT_BASE_PORT="$(polar_rollout_base_port_for_allocation "non-slurm-${HOSTNAME:-localhost}")"
        export SLIME_ROLLOUT_BASE_PORT
        origin="ephemeral-safe non-Slurm fallback"
    fi

    polar_validate_rollout_base_port "$SLIME_ROLLOUT_BASE_PORT" || return 1
    echo "[launcher] SGLang rollout base port=${SLIME_ROLLOUT_BASE_PORT} (${origin})"
}

polar_validate_sglang_router_port() {
    local port="${1:-}"
    local ephemeral_lower rollout_base rollout_end
    if ! [[ "$port" =~ ^[0-9]+$ ]]; then
        echo "ERROR: SGLANG_ROUTER_PORT must be an integer, got ${port:-<empty>}" >&2
        return 1
    fi

    ephemeral_lower="$(polar_ephemeral_port_lower_bound)" || return 1
    if [ "$port" -lt "$_POLAR_UNPRIVILEGED_PORT_FLOOR" ] || \
       [ "$port" -ge "$ephemeral_lower" ]; then
        echo "ERROR: SGLANG_ROUTER_PORT=${port} is unsafe; it must be an unprivileged non-ephemeral port between ${_POLAR_UNPRIVILEGED_PORT_FLOOR} and $((ephemeral_lower - 1))" >&2
        return 1
    fi

    rollout_base="${SLIME_ROLLOUT_BASE_PORT:-${_POLAR_ROLLOUT_BASE_PORT_FLOOR}}"
    polar_validate_rollout_base_port "$rollout_base" || return 1
    rollout_end="$((rollout_base + _POLAR_ROLLOUT_PORT_BLOCK_SIZE - 1))"
    if [ "$port" -ge "$rollout_base" ] && [ "$port" -le "$rollout_end" ]; then
        echo "ERROR: SGLANG_ROUTER_PORT=${port} overlaps the reserved rollout-engine port block ${rollout_base}-${rollout_end}" >&2
        return 1
    fi

    # These are fixed by run.sh. Ray's remaining sockets are kernel-assigned
    # from the ephemeral range, which this function rejects above.
    case "$port" in
        6379|8265)
            echo "ERROR: SGLANG_ROUTER_PORT=${port} conflicts with a fixed Ray control-plane port" >&2
            return 1
            ;;
    esac
}

polar_configure_sglang_router_port() {
    local origin
    if [ -n "${SGLANG_ROUTER_PORT:-}" ]; then
        origin="explicit override"
    else
        SGLANG_ROUTER_PORT="${_POLAR_SGLANG_ROUTER_PORT_DEFAULT}"
        export SGLANG_ROUTER_PORT
        origin="safe default"
    fi

    polar_validate_sglang_router_port "$SGLANG_ROUTER_PORT" || return 1
    echo "[launcher] SGLang router port=${SGLANG_ROUTER_PORT} (${origin})"
}

polar_select_load_dir() {
    local save_dir="${1:?missing save dir}"
    local ref_load="${2:?missing reference checkpoint}"
    local requested_load_dir="${3:-}"

    # LOAD_DIR is an initial seed for a new run.  Once this run has produced a
    # checkpoint, its own SAVE_DIR must win on every subsequent allocation.
    if [ -s "${save_dir}/latest_checkpointed_iteration.txt" ]; then
        printf '%s\n' "$save_dir"
    elif [ -n "$requested_load_dir" ]; then
        printf '%s\n' "$requested_load_dir"
    else
        printf '%s\n' "$ref_load"
    fi
}

polar_spilot_admission_fatal_marker_path() {
    local state_file="${TMAX_RUN_STATE_FILE:-}"
    local job_id="${SLURM_JOB_ID:-}"
    local rank="${RAY_NODE_RANK:-}"
    if [ -z "${state_file}" ] || ! [[ "${job_id}" =~ ^[0-9]+$ ]] || \
       ! [[ "${rank}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: SPilot admission health monitor requires TMAX_RUN_STATE_FILE, numeric SLURM_JOB_ID, and numeric RAY_NODE_RANK" >&2
        return 1
    fi
    printf '%s.job-%s.admission-fatal.rank-%s.json\n' \
        "${state_file}" "${job_id}" "${rank}"
}

polar_check_spilot_admission_health() {
    # A fatal runtime-containment failure means a candidate process may still
    # be alive, regardless of whether new episode admission is enabled. Only
    # the gateway can make that determination, so this monitor reacts to its
    # explicit structured 503 rather than treating transient HTTP errors or
    # unrelated unhealthy upstreams as retained-process events.
    [ "${TMAX_AGENT_HARNESS:-}" = "spilot_router" ] || return 0

    local health_url="${POLAR_GATEWAY_LOCAL_URL:-}/health"
    local health_file marker_path http_code parser_output parser_status
    if [ -z "${POLAR_GATEWAY_LOCAL_URL:-}" ] || [ -z "${RUN_DIR:-}" ] || \
       [ -z "${PYTHON_BIN:-}" ]; then
        echo "ERROR: SPilot admission health monitor is missing gateway, run-dir, or Python configuration" >&2
        return 1
    fi
    marker_path="$(polar_spilot_admission_fatal_marker_path)" || return 1
    health_file="${RUN_DIR}/.spilot-admission-health.rank-${RAY_NODE_RANK}.$$"
    http_code="$(
        curl --noproxy '*' -sS --max-time 5 \
            -o "${health_file}" -w '%{http_code}' "${health_url}" 2>/dev/null || true
    )"
    if [ "${http_code}" != "503" ]; then
        rm -f "${health_file}"
        return 0
    fi

    parser_status=0
    parser_output="$(
        "${PYTHON_BIN}" - "${health_file}" "${marker_path}" \
            "${SLURM_JOB_ID}" "${RAY_NODE_RANK}" <<'PY'
import datetime
import json
import os
import pathlib
import sys

health_path, marker_path, job_id, rank_text = sys.argv[1:]
try:
    payload = json.loads(pathlib.Path(health_path).read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError):
    raise SystemExit(10)
if not isinstance(payload, dict):
    raise SystemExit(10)
health = payload.get("model_pool_episode_admission_health")
if not isinstance(health, dict) or health.get("fatal_retained") is not True:
    raise SystemExit(10)
retained = health.get("retained_session_ids")
if not isinstance(retained, list) or not retained or not all(
    isinstance(value, str) and value for value in retained
):
    raise SystemExit(10)
node_id = payload.get("node_id")
if not isinstance(node_id, str) or not node_id:
    node_id = f"slurm-rank-{rank_text}"
marker = {
    "schema_version": 1,
    "kind": "model_pool_episode_admission_fatal_retained",
    "job_id": job_id,
    "rank": int(rank_text),
    "node_id": node_id,
    "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "retained_count": len(retained),
}
path = pathlib.Path(marker_path)
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
try:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(marker, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)
except OSError as exc:
    print(f"could not write SPilot admission fatal marker: {exc}", file=sys.stderr)
    raise SystemExit(20)
finally:
    temporary.unlink(missing_ok=True)
print(len(retained))
PY
    )" || parser_status=$?
    rm -f "${health_file}"
    if [ "${parser_status}" -eq 0 ]; then
        echo "[spilot admission fatal] job=${SLURM_JOB_ID} rank=${RAY_NODE_RANK} retained_count=${parser_output} action=fail-allocation marker=${marker_path}" >&2
        return 70
    fi
    if [ "${parser_status}" -eq 20 ]; then
        echo "[spilot admission fatal] job=${SLURM_JOB_ID} rank=${RAY_NODE_RANK} retained_count=unknown action=fail-allocation marker_write=failed" >&2
        return 70
    fi
    # Generic/malformed 503s are handled by the normal gateway heartbeat and
    # PID policy; they are not evidence that a process-retaining lease exists.
    return 0
}

polar_checkpoint_is_release_seed() {
    local load_dir="${1:?missing load dir}"
    local tracker="${load_dir}/latest_checkpointed_iteration.txt"
    local pointer

    [ -s "$tracker" ] || return 1
    IFS= read -r pointer < "$tracker" || true
    [ "$pointer" = "release" ]
}

# Validate model-architecture invariants before starting Python/Ray. Qwen3.5
# stores RMSNorm weights as zero-centred deltas, so Megatron must evaluate
# ``1 + weight``. Without this flag the checkpoint still loads without missing
# keys, but its logits are unrelated to the Hugging Face/SGLang policy.
polar_validate_model_args() {
    local arg
    local selects_qwen35=0
    local applies_layernorm_1p=0

    for arg in "$@"; do
        case "$arg" in
            slime_plugins.models.qwen3_5|--spec=slime_plugins.models.qwen3_5)
                selects_qwen35=1
                ;;
            --apply-layernorm-1p)
                applies_layernorm_1p=1
                ;;
        esac
    done

    if [ "$selects_qwen35" -eq 1 ] && [ "$applies_layernorm_1p" -ne 1 ]; then
        echo "ERROR: MODEL_ARGS selecting slime_plugins.models.qwen3_5 must include --apply-layernorm-1p" >&2
        return 1
    fi
    return 0
}
