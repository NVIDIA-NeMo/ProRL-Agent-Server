#!/usr/bin/env bash
# Monitor one logical TMax run and relaunch it from its latest checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./lifecycle.sh
source "${SCRIPT_DIR}/lifecycle.sh"
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

for command in flock git squeue sacct python3; do
    command -v "$command" >/dev/null || { echo "ERROR: ${command} is required" >&2; exit 1; }
done
mkdir -p "$(dirname "${TMAX_RUN_STATE_FILE}")"
exec 9>"${TMAX_RUN_STATE_FILE}.lock"
if ! flock -n 9; then
    echo "[tmax watch] another watcher or submitter owns ${TMAX_RUN_STATE_FILE}"
    exit 0
fi
export TMAX_RUN_STATE_LOCK_HELD=1

# Existing state is authoritative even when RUN_ID is explicit. This prevents
# a watcher from silently repinning or overwriting the immutable contract of a
# prior logical run. A new explicit run must use a new state-file path.
LOADED_RUN_STATE=false
_TMAX_REQUESTED_RUN_ID="${RUN_ID:-}"
if [ -s "${TMAX_RUN_STATE_FILE}" ]; then
    tmax_load_selected_run_state "${TMAX_RUN_STATE_FILE}" "${_TMAX_REQUESTED_RUN_ID}"
    LOADED_RUN_STATE=true
fi
unset _TMAX_REQUESTED_RUN_ID
# Run-state files written before the fixed-holdout feature must resume with
# their original prompt/checkpoint semantics. New submissions persist an
# explicit TMAX_EVAL_ENABLED value, so only legacy state reaches this branch.
if [ "$LOADED_RUN_STATE" = true ] && \
   ! tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" TMAX_EVAL_ENABLED; then
    export TMAX_EVAL_ENABLED=0
fi
# Runs created before external Harbor eval existed used the numeric TMax window.
# Preserve that immutable resume contract instead of silently swapping their
# eval JSONL and completion-marker semantics to the new default.
if [ "$LOADED_RUN_STATE" = true ] && \
   ! tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" TMAX_EVAL_SOURCE; then
    export TMAX_EVAL_SOURCE=tmax
fi
# The second eval dataset is opt-in for old run states. Otherwise sourcing new
# defaults would silently alter their eval marker hash and resume contract.
if [ "$LOADED_RUN_STATE" = true ] && \
   ! tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" TMAX_EXTERNAL_EVAL_ENABLED; then
    export TMAX_EXTERNAL_EVAL_ENABLED=0
fi
# Dynamic sampling changes which prompt groups reach the trainer. Preserve the
# behavior of logical runs created before that contract was persisted; new run
# states record the paper-faithful filter path explicitly.
if [ "$LOADED_RUN_STATE" = true ] && \
   ! tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" TMAX_DYNAMIC_SAMPLING_FILTER_PATH; then
    export TMAX_DYNAMIC_SAMPLING_FILTER_PATH=""
fi
# Episode admission changes both provider pressure and timeout semantics. A run
# state that contains none of the contract fields predates admission and must
# resume with the old disabled behavior and 3,300/4,500/5,100 envelopes. A
# partially written contract is ambiguous and therefore rejected.
if [ "$LOADED_RUN_STATE" = true ] && \
   [ "${TMAX_AGENT_HARNESS:-}" = "spilot_router" ]; then
    tmax_restore_spilot_admission_resume_contract \
        "${TMAX_RUN_STATE_FILE}" "[tmax watch]"
fi
# Model identity is part of checkpoint compatibility. Run states created before
# it was persisted belong to the historical Qwen3.5-4B launcher; preserve that
# lineage instead of combining a 4B SAVE_DIR with the new 9B architecture.
if [ "$LOADED_RUN_STATE" = true ] && \
   ! tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" HF_CHECKPOINT; then
    _legacy_user_root="$(dirname "${SPILOT_ROOT}")"
    _legacy_ref_load="${POLAR_DATA_ROOT}/checkpoints/Qwen3.5-4B_torch_dist"
    if [ ! -s "${_legacy_ref_load}/latest_checkpointed_iteration.txt" ] && \
       [ -s "${_legacy_user_root}/spilot-router/data/checkpoints/Qwen3.5-4B_torch_dist/latest_checkpointed_iteration.txt" ]; then
        _legacy_ref_load="${_legacy_user_root}/spilot-router/data/checkpoints/Qwen3.5-4B_torch_dist"
    fi
    export HF_CHECKPOINT=Qwen/Qwen3.5-4B
    export REF_LOAD="${_legacy_ref_load}"
    export TORCH_DIST_DIR="${REF_LOAD}"
    export MODEL_ARGS_FILE="${PROJECT_ROOT}/examples/swegym_slime_grpo/model_args.sh"
    export ACTOR_TENSOR_MODEL_PARALLEL_SIZE="${ACTOR_TENSOR_MODEL_PARALLEL_SIZE:-2}"
    export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.8}"
    unset _legacy_user_root _legacy_ref_load
    echo "[tmax watch] legacy run state: preserving Qwen3.5-4B model lineage" >&2
fi
# shellcheck source=./env.cwdfw.sh
source "${SCRIPT_DIR}/env.cwdfw.sh" >/dev/null
if tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" TMAX_PRORL_GIT_COMMIT ||
   tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" TMAX_SLIME_GIT_COMMIT ||
   tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" TMAX_MEGATRON_GIT_COMMIT; then
    if ! tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" TMAX_PRORL_GIT_COMMIT ||
       ! tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" TMAX_SLIME_GIT_COMMIT ||
       ! tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" TMAX_MEGATRON_GIT_COMMIT; then
        echo "ERROR: run state has a partial source revision lock; refusing an ambiguous resume" >&2
        exit 1
    fi
    tmax_verify_source_revisions "${PROJECT_ROOT}" "${SLIME_DIR}" "${MEGATRON_DIR}"
elif [ "$LOADED_RUN_STATE" = true ]; then
    echo "[tmax watch] legacy run state has no source revision lock" >&2
fi

export TMAX_WATCH_MAX_QUICK_FAILURES="${TMAX_WATCH_MAX_QUICK_FAILURES:-3}"
export TMAX_WATCH_QUICK_FAILURE_SECONDS="${TMAX_WATCH_QUICK_FAILURE_SECONDS:-900}"
export TMAX_WATCH_FAILURE_COUNT="${TMAX_WATCH_FAILURE_COUNT:-0}"
export TMAX_WATCH_FAILURE_SIGNATURE="${TMAX_WATCH_FAILURE_SIGNATURE:-}"
export TMAX_WATCH_LAST_ACCOUNTED_JOB_ID="${TMAX_WATCH_LAST_ACCOUNTED_JOB_ID:-}"
export TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT="${TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT:-0}"
TMAX_SUBMIT_SCRIPT="${TMAX_SUBMIT_SCRIPT:-${SCRIPT_DIR}/submit_slurm.sh}"

# SPilot credentials are deliberately absent from run state, so its watcher
# must relaunch through the wrapper that regenerates the control token and
# normalizes the NVIDIA key.  Refuse a silent fallback to this generic TMax
# submitter: that exact fallback previously produced an all-503 resume job.
if [ "${TMAX_AGENT_HARNESS:-}" = "spilot_router" ]; then
    tmax_require_spilot_entrypoints "watcher preflight"
    if [ -z "${POLAR_NVIDIA_API_KEY:-${NVIDIA_API_KEY:-}}" ]; then
        echo "ERROR: SPilot Router watcher requires NVIDIA_API_KEY before relaunch" >&2
        exit 1
    fi
fi

if [ -n "${TMAX_TARGET_ITER:-}" ] && ! [[ "${TMAX_TARGET_ITER}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: TMAX_TARGET_ITER must be a non-negative integer" >&2
    exit 2
fi
if [ -n "${TMAX_NUM_ROLLOUT:-}" ]; then
    if ! [[ "${TMAX_NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: TMAX_NUM_ROLLOUT must be a positive integer" >&2
        exit 2
    fi
    _tmax_expected_target="$((TMAX_NUM_ROLLOUT - 1))"
    if [ "${TMAX_TARGET_ITER:-}" != "${_tmax_expected_target}" ]; then
        echo "ERROR: watcher target ${TMAX_TARGET_ITER:-unset} must equal TMAX_NUM_ROLLOUT-1=${_tmax_expected_target}" >&2
        exit 2
    fi
    unset _tmax_expected_target
fi
if ! [[ "${ROLLOUT_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${NUM_EPOCH}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ROLLOUT_BATCH_SIZE and NUM_EPOCH must be positive integers" >&2
    exit 2
fi
if ! [[ "${TMAX_WATCH_MAX_QUICK_FAILURES}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${TMAX_WATCH_QUICK_FAILURE_SECONDS}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${TMAX_WATCH_FAILURE_COUNT}" =~ ^[0-9]+$ ]] || \
   ! [[ "${TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid watcher failure-policy setting" >&2
    exit 2
fi
if [ -z "${TMAX_PREPARE_DATA+x}" ]; then
    if [ "$LOADED_RUN_STATE" = true ] || [ -s "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
        export TMAX_PREPARE_DATA=0
    else
        export TMAX_PREPARE_DATA=1
    fi
fi

WATCH_COMPLETE=false
WATCH_ABORT=false

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

validate_checkpoint_pair() {
    local pointer="${SAVE_DIR}/latest_checkpointed_iteration.txt"
    [ -e "${pointer}" ] || return 0
    if ! tmax_validate_numbered_checkpoint \
        "${SAVE_DIR}" "[tmax watch] ERROR" >/dev/null; then
        echo "[tmax watch] refusing to resume from a non-atomic checkpoint" >&2
        return 1
    fi
}

target_iter() {
    if [ -n "${TMAX_NUM_ROLLOUT:-}" ]; then
        printf '%s\n' "$((TMAX_NUM_ROLLOUT - 1))"
        return
    fi
    if [ -n "${TMAX_TARGET_ITER:-}" ]; then
        printf '%s\n' "${TMAX_TARGET_ITER}"
        return
    fi
    [ -s "${TMAX_TRAIN_DATA}" ] || return 1
    local samples rollouts
    samples="$(awk 'NF { count += 1 } END { print count + 0 }' "${TMAX_TRAIN_DATA}")"
    [ "$samples" -gt 0 ] || return 1
    rollouts="$(( (samples + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE * NUM_EPOCH ))"
    # The checkpoint restores the rollout data-source cursor as well as model
    # state. With the same prompt data, NUM_EPOCH still ends at rollouts - 1;
    # adding the seed iteration would overshoot the configured epoch.
    printf '%s\n' "$((rollouts - 1))"
}

current_eval_data_sha256() {
    [ -s "${TMAX_EVAL_DATA}" ] || return 1
    if [ "${TMAX_EXTERNAL_EVAL_ENABLED}" = "1" ]; then
        [ -s "${TMAX_EXTERNAL_EVAL_DATA}" ] || return 1
        "${TMAX_SIF_PYTHON_BIN}" "${SCRIPT_DIR}/validate_data_integrity.py" \
            --bundle-sha256 "${TMAX_EVAL_DATASET_NAME}" "${TMAX_EVAL_DATA}" \
            --bundle-sha256 "${TMAX_EXTERNAL_EVAL_DATASET_NAME}" "${TMAX_EXTERNAL_EVAL_DATA}"
        return
    fi
    python3 - "${TMAX_EVAL_DATA}" <<'PY'
import hashlib
import pathlib
import sys

print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
}

final_eval_marker_matches_target() {
    local target="$1" eval_data_sha256="$2"
    [ -s "${FINAL_EVAL_COMPLETE_MARKER}" ] || return 1
    python3 - "${FINAL_EVAL_COMPLETE_MARKER}" "$target" "$eval_data_sha256" <<'PY'
import json
import sys

path, target_text, eval_data_sha256 = sys.argv[1:]
target = int(target_text)
try:
    with open(path, encoding="utf-8") as marker:
        payload = json.load(marker)
except (OSError, UnicodeError, json.JSONDecodeError):
    raise SystemExit(1)
if not isinstance(payload, dict):
    raise SystemExit(1)

expected = {
    "final_rollout_id": target,
    "model_iteration": target,
    "num_rollout": target + 1,
    "eval_data_sha256": eval_data_sha256,
}
if any(type(payload.get(key)) is not type(value) or payload[key] != value for key, value in expected.items()):
    raise SystemExit(1)
PY
}

import_submission_receipt() {
    local previous_job="${TMAX_LAST_JOB_ID:-}"
    local receipt_job receipt_time
    [ -s "${TMAX_SUBMIT_RECEIPT_FILE}" ] || return 1
    unset POLAR_SUBMITTED_JOB_ID POLAR_SUBMITTED_AT_UNIX
    # shellcheck source=/dev/null
    source "${TMAX_SUBMIT_RECEIPT_FILE}"
    receipt_job="${POLAR_SUBMITTED_JOB_ID:-}"
    receipt_time="${POLAR_SUBMITTED_AT_UNIX:-}"
    [[ "$receipt_job" =~ ^[0-9]+$ ]] && [[ "$receipt_time" =~ ^[0-9]+$ ]] || return 1
    if [ -n "${TMAX_LAST_JOB_SUBMITTED_AT:-}" ] && \
       [[ "${TMAX_LAST_JOB_SUBMITTED_AT}" =~ ^[0-9]+$ ]] && \
       [ "$receipt_time" -lt "${TMAX_LAST_JOB_SUBMITTED_AT}" ]; then
        return 0
    fi
    export TMAX_LAST_JOB_ID="$receipt_job"
    export TMAX_LAST_JOB_SUBMITTED_AT="$receipt_time"
    if [ "$receipt_job" != "$previous_job" ]; then
        export TMAX_LAST_JOB_CHECKPOINT_ITER="$(latest_iter)"
    fi
}

active_job() {
    local owner id active named named_count discovered_id
    ACTIVE_JOB_OUTPUT=""
    owner="${SLURM_USER:-${USER:-$(id -un)}}"
    id="${TMAX_LAST_JOB_ID:-}"
    if [ -n "$id" ]; then
        active="$(squeue -h -j "$id" -t PD,R,CF,CG -o '%i %t %j %R' 2>/dev/null || true)"
    else
        active=""
    fi

    # Always reconcile against the exact RUN_ID-specific job name, even when a
    # stale tracked id exists. This closes the sbatch-success/receipt-write
    # crash window without risking a duplicate allocation.
    named="$(squeue -h -u "$owner" -n "${JOB_NAME}" -t PD,R,CF,CG -o '%i %t %j %R' 2>/dev/null || true)"
    named_count="$(awk 'NF { seen[$1] = 1 } END { for (id in seen) n += 1; print n + 0 }' <<<"$named")"
    if [ "$named_count" -gt 1 ]; then
        echo "[tmax watch] ERROR: multiple active jobs match ${JOB_NAME}; refusing to submit or adopt one:" >&2
        printf '%s\n' "$named" >&2
        WATCH_ABORT=true
        return
    fi
    if [ -z "$active" ] && [ "$named_count" -eq 1 ]; then
        active="$named"
        discovered_id="$(awk 'NF { print $1; exit }' <<<"$named")"
        if [[ "$discovered_id" =~ ^[0-9]+$ ]]; then
            export TMAX_LAST_JOB_ID="$discovered_id"
            export TMAX_LAST_JOB_SUBMITTED_AT="$(date +%s)"
            export TMAX_LAST_JOB_CHECKPOINT_ITER="$(latest_iter)"
            tmax_write_run_state "${TMAX_RUN_STATE_FILE}"
            echo "[tmax watch] adopted active job ${discovered_id} by exact job name after receipt/state reconciliation"
        fi
    fi
    ACTIVE_JOB_OUTPUT="$active"
}

slurm_job_record() {
    local id="${1%%;*}"
    sacct -X -n -P -j "$id" --format=JobIDRaw,State,ElapsedRaw,ExitCode 2>/dev/null |
        awk -F '|' -v wanted="$id" '$1 == wanted { print $2 "|" $3 "|" $4; exit }'
}

spilot_admission_fatal_marker_status() {
    local id="$1"
    python3 - "${TMAX_RUN_STATE_FILE}" "${id}" <<'PY'
import json
import pathlib
import re
import sys

state_path = pathlib.Path(sys.argv[1])
job_id = sys.argv[2]
prefix = f"{state_path.name}.job-{job_id}.admission-fatal.rank-"
paths = sorted(state_path.parent.glob(f"{prefix}*.json"))
if not paths:
    raise SystemExit(1)
retained_total = 0
for path in paths:
    match = re.fullmatch(re.escape(prefix) + r"([0-9]+)\.json", path.name)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SystemExit(2)
    if (
        match is None
        or not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("kind") != "model_pool_episode_admission_fatal_retained"
        or payload.get("job_id") != job_id
        or type(payload.get("rank")) is not int
        or payload["rank"] != int(match.group(1))
        or not isinstance(payload.get("node_id"), str)
        or not payload["node_id"]
        or not isinstance(payload.get("observed_at"), str)
        or not payload["observed_at"]
        or type(payload.get("retained_count")) is not int
        or payload["retained_count"] <= 0
    ):
        raise SystemExit(2)
    retained_total += payload["retained_count"]
print(f"{len(paths)}|{retained_total}")
PY
}

spilot_candidate_pool_health_incident_status() {
    local id="$1"
    python3 - "${SAVE_DIR}" "${TMAX_RUN_STATE_FILE}" "${id}" <<'PY'
import json
import pathlib
import re
import sys

save_dir = pathlib.Path(sys.argv[1])
state_file = pathlib.Path(sys.argv[2])
job_id = sys.argv[3]
incident_dirs = (
    save_dir / "rollout" / "candidate_pool_health_incidents",
    pathlib.Path(f"{state_file}.candidate_pool_health_incidents"),
)
paths = sorted(
    {
        path
        for incident_dir in incident_dirs
        for path in incident_dir.glob("rollout_*.json")
    }
)
matches = []
for path in paths:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        # A file that cannot identify its producing allocation cannot safely
        # be attributed to this terminal job. Valid incident files are written
        # atomically, so leave unrelated/corrupt historical files alone.
        continue
    if not isinstance(payload, dict) or payload.get("slurm_job_id") != job_id:
        continue
    filename_match = re.fullmatch(r"rollout_([0-9]{7})\.json", path.name)
    reasons = payload.get("trigger_reasons")
    rollout_id = payload.get("rollout_id")
    if (
        filename_match is None
        or payload.get("schema_version") != 1
        or payload.get("triggered") is not True
        or type(rollout_id) is not int
        or rollout_id < 0
        or rollout_id != int(filename_match.group(1))
        or not isinstance(reasons, list)
        or not reasons
        or any(not isinstance(reason, str) or not reason for reason in reasons)
    ):
        raise SystemExit(2)
    matches.append((rollout_id, reasons))

if not matches:
    raise SystemExit(1)
latest_rollout_id, latest_reasons = max(matches, key=lambda item: item[0])
print(f"{len(matches)}|{latest_rollout_id}|{','.join(latest_reasons)}")
PY
}

normalize_slurm_state() {
    local state="${1%% *}"
    printf '%s\n' "${state%+}"
}

is_terminal_slurm_state() {
    case "$1" in
        BOOT_FAIL|CANCELLED|COMPLETED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|SPECIAL_EXIT|TIMEOUT)
            return 0 ;;
        *) return 1 ;;
    esac
}

is_fail_closed_quick_failure_state() {
    # These outcomes normally describe a deterministic launcher/runtime or
    # resource-sizing problem. Repeating the same allocation immediately is
    # both expensive and unlikely to make progress. Slurm infrastructure
    # outcomes such as PREEMPTED, NODE_FAIL, and REVOKED intentionally remain
    # on the bounded automatic-recovery path below.
    case "$1" in
        FAILED|OUT_OF_MEMORY|TIMEOUT)
            return 0 ;;
        *) return 1 ;;
    esac
}

is_fail_closed_failure_signature() {
    case "$1" in
        quick-fail-closed/*) return 0 ;;
        *) return 1 ;;
    esac
}

reset_failure_streak() {
    export TMAX_WATCH_FAILURE_COUNT=0
    export TMAX_WATCH_FAILURE_SIGNATURE=""
}

record_failure_signature() {
    local signature="$1"
    # Count consecutive no-progress outcomes independently of their Slurm
    # spelling (FAILED/CANCELLED, 9/137, etc.); retain the latest signature only
    # for diagnostics.
    export TMAX_WATCH_FAILURE_SIGNATURE="$signature"
    export TMAX_WATCH_FAILURE_COUNT="$((TMAX_WATCH_FAILURE_COUNT + 1))"
}

publish_spilot_static_metrics() {
    local wandb_label="${1:?missing W&B publisher label}"
    shift
    local publish_python publish_script publish_timeout wandb_dir static_metric
    local -a publish_args
    [ -n "${WANDB_API_KEY:-}" ] || {
        echo "[tmax watch] W&B ${wandb_label} metric publish skipped: WANDB_API_KEY is unavailable" >&2
        return 0
    }
    case "${WANDB_MODE:-offline}" in
        online|shared) ;;
        *)
            echo "[tmax watch] W&B ${wandb_label} metric publish skipped: WANDB_MODE=${WANDB_MODE:-offline}" >&2
            return 0
            ;;
    esac
    command -v timeout >/dev/null || {
        echo "[tmax watch] WARNING: cannot publish W&B ${wandb_label} metric without timeout(1)" >&2
        return 0
    }
    publish_python="${TMAX_SPILOT_FATAL_WANDB_PYTHON_BIN:-${POLR_TRAIN_VENV}/bin/python3}"
    publish_script="${PROJECT_ROOT}/scripts/monitor_wandb_gpu.py"
    publish_timeout="${TMAX_SPILOT_FATAL_WANDB_PUBLISH_TIMEOUT_SECONDS:-45}"
    if ! [[ "$publish_timeout" =~ ^[1-9][0-9]*$ ]]; then
        echo "[tmax watch] WARNING: invalid TMAX_SPILOT_FATAL_WANDB_PUBLISH_TIMEOUT_SECONDS=${publish_timeout}" >&2
        return 0
    fi
    if [ ! -x "$publish_python" ] || [ ! -f "$publish_script" ]; then
        echo "[tmax watch] WARNING: final W&B fatal metric publisher is unavailable" >&2
        return 0
    fi
    wandb_dir="${POLAR_DATA_ROOT}/runs/${RUN_ID}/watcher-wandb"
    mkdir -p "$wandb_dir"
    publish_args=(
        "$publish_python" "$publish_script"
        --one-shot-static
        --metric-prefix "${GPU_MONITOR_PREFIX:-polar_tmax_system}"
        --train-progress-file "${SAVE_DIR}/train_progress.step"
        --wandb-run-id "${WANDB_RUN_ID:-${RUN_ID}}"
        --wandb-project "${WANDB_PROJECT:-polar-tmax-grpo}"
        --wandb-group "${WANDB_GROUP:-spilot-router-qwen35-9b-8n64}"
        --wandb-dir "$wandb_dir"
        --wandb-label "${wandb_label}"
        --wandb-mode shared
        --wandb-finish-timeout-s 15
    )
    for static_metric in "$@"; do
        publish_args+=(--static-metric "${static_metric}")
    done
    if [ -n "${WANDB_ENTITY:-}" ]; then
        publish_args+=(--wandb-entity "${WANDB_ENTITY}")
    fi
    if timeout --signal=TERM --kill-after=5 "$publish_timeout" "${publish_args[@]}"; then
        if [ "${wandb_label}" = "spilot-fatal-watcher" ]; then
            echo "[tmax watch] published final SPilot admission fatal count=${TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT} to W&B" >&2
        else
            echo "[tmax watch] published SPilot ${wandb_label} metric(s) to W&B" >&2
        fi
    else
        echo "[tmax watch] WARNING: bounded W&B ${wandb_label} metric publish failed; run state/incident remains authoritative" >&2
    fi
}

publish_spilot_admission_fatal_metric() {
    publish_spilot_static_metrics \
        spilot-fatal-watcher \
        "polar/spilot_router/admission_fatal_job_count_total=${TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT}"
}

publish_spilot_candidate_pool_health_metric() {
    publish_spilot_static_metrics \
        spilot-candidate-health-watcher \
        "polar/candidate_pool_health/gate_triggered=1"
}

account_terminal_job() {
    local id="$1" state="$2" elapsed="$3" exit_code="$4" iter="$5"
    local submitted_iter="${TMAX_LAST_JOB_CHECKPOINT_ITER:--1}"
    local progressed=false fail_closed_quick=false signature
    local admission_fatal=false marker_status=0 marker_summary=""
    local candidate_health_incident=false health_status=0 health_summary=""

    if [ "${TMAX_WATCH_LAST_ACCOUNTED_JOB_ID}" = "$id" ]; then
        if is_fail_closed_failure_signature "${TMAX_WATCH_FAILURE_SIGNATURE}"; then
            echo "[tmax watch] fail-closed quick-failure latch remains set: ${TMAX_WATCH_FAILURE_SIGNATURE}" >&2
            echo "[tmax watch] fix the cause, then restart with TMAX_WATCH_RESET_FAILURES=1" >&2
            WATCH_ABORT=true
        elif [ "$TMAX_WATCH_FAILURE_COUNT" -ge "$TMAX_WATCH_MAX_QUICK_FAILURES" ]; then
            WATCH_ABORT=true
        fi
        return
    fi
    if ! [[ "$submitted_iter" =~ ^(-1|[0-9]+)$ ]]; then
        submitted_iter=-1
    fi
    if [ "$iter" -gt "$submitted_iter" ]; then
        progressed=true
    fi

    # Runtime-containment fatal markers remain authoritative after admission
    # has been disabled to quiesce a draining or incident-affected allocation.
    if [ "${TMAX_AGENT_HARNESS:-}" = "spilot_router" ]; then
        marker_summary="$(spilot_admission_fatal_marker_status "$id")" || marker_status=$?
        health_summary="$(spilot_candidate_pool_health_incident_status "$id")" || health_status=$?
    else
        marker_status=1
        health_status=1
    fi
    if [ "$marker_status" -eq 0 ]; then
        admission_fatal=true
        export TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT="$((TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT + 1))"
        echo "[tmax watch] SPilot admission fatal job metric: count_total=${TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT} job=${id} markers_retained=${marker_summary}" >&2
    elif [ "$marker_status" -eq 2 ]; then
        echo "[tmax watch] malformed SPilot admission fatal marker for job=${id}; retaining normal fail-closed policy" >&2
    fi
    if [ "$health_status" -eq 0 ]; then
        candidate_health_incident=true
    elif [ "$health_status" -eq 2 ]; then
        # The payload explicitly names this allocation, so an invalid schema
        # must not turn a provider outage into an automatic retry loop.
        candidate_health_incident=true
        health_summary="malformed"
        echo "[tmax watch] malformed candidate-pool health incident for job=${id}; failing closed" >&2
    fi

    if [ "$candidate_health_incident" = true ]; then
        signature="quick-fail-closed/candidate-pool-health/job=${id}/incident=${health_summary}/${state}/exit=${exit_code}/checkpoint=${iter}"
        record_failure_signature "$signature"
        export TMAX_WATCH_LAST_ACCOUNTED_JOB_ID="$id"
        tmax_write_run_state "${TMAX_RUN_STATE_FILE}"
        if [ "$admission_fatal" = true ]; then
            publish_spilot_admission_fatal_metric
        fi
        publish_spilot_candidate_pool_health_metric
        echo "[tmax watch] candidate-pool health incident: job=${id} incident=${health_summary}; refusing automatic resubmission" >&2
        if [[ "${health_summary}" == *partial_wal_quarantine_failed* ]]; then
            echo "[tmax watch] partial WAL quarantine failed; do not reset this SAVE_DIR for ordinary resume" >&2
        else
            echo "[tmax watch] verify provider health, then restart with TMAX_WATCH_RESET_FAILURES=1" >&2
        fi
        WATCH_ABORT=true
        return
    fi

    # Completion without either the final marker (handled before this function)
    # or a newer atomic checkpoint is still a failed continuation, regardless
    # of elapsed time or Slurm's terminal-state spelling.
    if [ "$progressed" = false ]; then
        signature="${state}/exit=${exit_code}/checkpoint=${iter}"
        if [ "$admission_fatal" = true ]; then
            signature="recoverable/model-pool-episode-admission-retained/${signature}"
        elif [[ "$elapsed" =~ ^[0-9]+$ ]] && \
           [ "$elapsed" -lt "$TMAX_WATCH_QUICK_FAILURE_SECONDS" ] && \
           is_fail_closed_quick_failure_state "$state"; then
            fail_closed_quick=true
            signature="quick-fail-closed/${signature}"
        fi
        record_failure_signature "$signature"
        echo "[tmax watch] no-progress failure ${TMAX_WATCH_FAILURE_COUNT}/${TMAX_WATCH_MAX_QUICK_FAILURES}: job=${id} state=${state} elapsed=${elapsed}s exit=${exit_code} checkpoint=${iter}" >&2
    else
        reset_failure_streak
    fi
    export TMAX_WATCH_LAST_ACCOUNTED_JOB_ID="$id"
    tmax_write_run_state "${TMAX_RUN_STATE_FILE}"
    if [ "$admission_fatal" = true ]; then
        # The allocation-side GPU monitor exited before this watcher could
        # increment the run-level count. Publish this event now so the terminal
        # third fatal is not lost when no subsequent allocation is launched.
        publish_spilot_admission_fatal_metric
    fi

    if [ "$fail_closed_quick" = true ]; then
        echo "[tmax watch] fail-closed quick failure: job=${id} state=${state} elapsed=${elapsed}s is below ${TMAX_WATCH_QUICK_FAILURE_SECONDS}s with no checkpoint progress; refusing automatic resubmission" >&2
        echo "[tmax watch] fix the cause, then restart with TMAX_WATCH_RESET_FAILURES=1" >&2
        WATCH_ABORT=true
        return
    fi

    if [ "$TMAX_WATCH_FAILURE_COUNT" -ge "$TMAX_WATCH_MAX_QUICK_FAILURES" ]; then
        echo "[tmax watch] refusing another automatic submission after ${TMAX_WATCH_FAILURE_COUNT} consecutive no-progress failures; last=${TMAX_WATCH_FAILURE_SIGNATURE}" >&2
        echo "[tmax watch] inspect the job, then restart with TMAX_WATCH_RESET_FAILURES=1 after fixing the cause" >&2
        WATCH_ABORT=true
    fi
}

record_submission_failure() {
    local status="$1" iter="$2"
    record_failure_signature "submission/exit=${status}/checkpoint=${iter}"
    tmax_write_run_state "${TMAX_RUN_STATE_FILE}"
    echo "[tmax watch] submission failure ${TMAX_WATCH_FAILURE_COUNT}/${TMAX_WATCH_MAX_QUICK_FAILURES}; retry interval=${SLEEP_SECONDS}s" >&2
    if [ "$TMAX_WATCH_FAILURE_COUNT" -ge "$TMAX_WATCH_MAX_QUICK_FAILURES" ]; then
        WATCH_ABORT=true
    fi
}

import_submission_source_lock() {
    if tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" TMAX_PRORL_GIT_COMMIT ||
       tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" TMAX_SLIME_GIT_COMMIT ||
       tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" TMAX_MEGATRON_GIT_COMMIT; then
        tmax_import_source_revision_lock "${TMAX_RUN_STATE_FILE}"
    fi
}

submit_training() {
    local iter="$1" status
    rm -f "${TMAX_SUBMIT_RECEIPT_FILE}"
    if SUBMIT_BACKEND=sbatch bash "${TMAX_SUBMIT_SCRIPT}"; then
        if ! import_submission_receipt; then
            # The job may exist but its id is unknown. Retrying here could
            # duplicate it, so stop instead of treating this as a normal error.
            echo "[tmax watch] ERROR: submission returned success without a valid fresh receipt; refusing a duplicate submission" >&2
            WATCH_ABORT=true
            return
        fi
        if ! import_submission_source_lock; then
            echo "[tmax watch] ERROR: submitted job but could not retain its source revision lock; refusing another submission" >&2
            WATCH_ABORT=true
            return
        fi
        export TMAX_PREPARE_DATA=0
        export TMAX_LAST_JOB_CHECKPOINT_ITER="$iter"
        tmax_write_run_state "${TMAX_RUN_STATE_FILE}"
        echo "[tmax watch] submission succeeded: job=${TMAX_LAST_JOB_ID} baseline_iter=${iter}"
    else
        status=$?
        if import_submission_receipt; then
            # sbatch succeeded and the wrapper failed in later bookkeeping.
            # The receipt proves a live job may exist, so track it and never
            # issue a second submission for this polling cycle.
            if ! import_submission_source_lock; then
                echo "[tmax watch] ERROR: submission receipt exists but its source revision lock is invalid; refusing another submission" >&2
                WATCH_ABORT=true
                return
            fi
            export TMAX_PREPARE_DATA=0
            export TMAX_LAST_JOB_CHECKPOINT_ITER="$iter"
            tmax_write_run_state "${TMAX_RUN_STATE_FILE}"
            echo "[tmax watch] submission wrapper exited ${status}, but receipt confirms job=${TMAX_LAST_JOB_ID}; tracking that job" >&2
        else
            if ! import_submission_source_lock; then
                echo "[tmax watch] ERROR: submission failed and its source revision lock is invalid; refusing another submission" >&2
                WATCH_ABORT=true
                return
            fi
            record_submission_failure "$status" "$iter"
        fi
    fi
}

if [ "${TMAX_WATCH_RESET_FAILURES:-0}" = "1" ]; then
    reset_failure_streak
    # The operator is explicitly authorizing a retry after fixing the cause.
    # Keep the old terminal job accounted; clearing this id would immediately
    # classify that same historical quick failure again and re-latch before a
    # corrected submission could be made.
    export TMAX_WATCH_LAST_ACCOUNTED_JOB_ID="${TMAX_LAST_JOB_ID:-}"
fi
# A receipt is the authoritative bridge across the short interval between
# sbatch returning and submit_slurm.sh updating current_run.env.
import_submission_receipt || true
tmax_write_run_state "${TMAX_RUN_STATE_FILE}"

check_once() {
    local iter target="" active record state elapsed exit_code id eval_data_sha256=""
    iter="$(latest_iter)"
    target="$(target_iter || true)"

    active_job
    if [ "$WATCH_ABORT" = true ]; then
        return
    fi
    active="$ACTIVE_JOB_OUTPUT"
    if [ -n "$active" ]; then
        printf '[tmax watch] job active; run_id=%s job_id=%s latest_iter=%s target=%s\n%s\n' \
            "$RUN_ID" "${TMAX_LAST_JOB_ID}" "$iter" "${target:-unknown}" "$active"
        return
    fi

    if ! validate_checkpoint_pair; then
        WATCH_ABORT=true
        return
    fi
    if [ "${TMAX_TRAINING_EVAL_ENABLED:-${TMAX_EVAL_ENABLED}}" = "1" ]; then
        if { [ -s "${FINAL_EVAL_COMPLETE_MARKER}" ] || \
             { [ -n "$target" ] && [ "$iter" -ge "$target" ]; }; }; then
            eval_data_sha256="$(current_eval_data_sha256 2>/dev/null || true)"
        fi
        if [ -n "$target" ] && [ "$iter" -eq "$target" ] && \
           [ -n "$eval_data_sha256" ] && \
           final_eval_marker_matches_target "$target" "$eval_data_sha256"; then
            echo "[tmax watch] final eval complete marker found: ${FINAL_EVAL_COMPLETE_MARKER}"
            WATCH_COMPLETE=true
            return
        fi
        if [ -s "${FINAL_EVAL_COMPLETE_MARKER}" ]; then
            echo "[tmax watch] final eval marker is stale or malformed for target ${target:-unknown}: ${FINAL_EVAL_COMPLETE_MARKER}" >&2
        fi
        if [ -n "$target" ] && [ "$iter" -ge "$target" ]; then
            printf '[tmax watch] final checkpoint reached (latest_iter=%s target=%s), but final eval is incomplete; the next resume will run eval only\n' \
                "$iter" "$target"
        fi
    else
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
    fi

    id="${TMAX_LAST_JOB_ID:-}"
    if [ -n "$id" ]; then
        record="$(slurm_job_record "$id")"
        if [ -z "$record" ]; then
            echo "[tmax watch] job ${id} left squeue but is not in sacct yet; waiting before any relaunch"
            return
        fi
        IFS='|' read -r state elapsed exit_code <<<"$record"
        state="$(normalize_slurm_state "$state")"
        if ! is_terminal_slurm_state "$state"; then
            echo "[tmax watch] job ${id} accounting state=${state}; waiting"
            return
        fi
        account_terminal_job "$id" "$state" "$elapsed" "$exit_code" "$iter"
        if [ "$WATCH_ABORT" = true ]; then
            return
        fi
    fi

    printf '[tmax watch] no active job; run_id=%s last_job=%s latest_iter=%s target=%s save_dir=%s\n' \
        "$RUN_ID" "${TMAX_LAST_JOB_ID:-none}" "$iter" "${target:-unknown}" "$SAVE_DIR"
    if [ "$RELAUNCH" = true ]; then
        submit_training "$iter"
    fi
}

echo "[tmax watch] state=${TMAX_RUN_STATE_FILE} run_id=${RUN_ID} save_dir=${SAVE_DIR} job_name=${JOB_NAME}"
if [ "$LOOP" = true ]; then
    while true; do
        date -u '+[tmax watch] %Y-%m-%dT%H:%M:%SZ'
        check_once
        if [ "$WATCH_COMPLETE" = true ]; then
            exit 0
        fi
        if [ "$WATCH_ABORT" = true ]; then
            exit 1
        fi
        sleep "$SLEEP_SECONDS"
    done
else
    check_once
    if [ "$WATCH_ABORT" = true ]; then
        exit 1
    fi
fi
