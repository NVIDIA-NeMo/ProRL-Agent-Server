#!/usr/bin/env bash
# Run TMax GRPO through the shared, Slime-0.3.0-compatible training launcher.
set -euo pipefail
export SLIME_JOB_SCRIPT_START_UNIX_NS="${SLIME_JOB_SCRIPT_START_UNIX_NS:-$(date +%s%N)}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=./env.cwdfw.sh
source "${SCRIPT_DIR}/env.cwdfw.sh"
# shellcheck source=./lifecycle.sh
source "${SCRIPT_DIR}/lifecycle.sh"
# shellcheck source=./run_state.sh
source "${SCRIPT_DIR}/run_state.sh"

# Repeat the submit-side guard inside the allocation.  This protects direct or
# stale environment-file launches that bypassed the current submitter.
tmax_require_spilot_entrypoints "allocation startup"
tmax_require_spilot_credentials "allocation startup"

_TMAX_SOURCE_LOCK_COUNT=0
if [ -n "${TMAX_PRORL_GIT_COMMIT:-}" ]; then
    _TMAX_SOURCE_LOCK_COUNT=$((_TMAX_SOURCE_LOCK_COUNT + 1))
fi
if [ -n "${TMAX_SLIME_GIT_COMMIT:-}" ]; then
    _TMAX_SOURCE_LOCK_COUNT=$((_TMAX_SOURCE_LOCK_COUNT + 1))
fi
if [ -n "${TMAX_MEGATRON_GIT_COMMIT:-}" ]; then
    _TMAX_SOURCE_LOCK_COUNT=$((_TMAX_SOURCE_LOCK_COUNT + 1))
fi
case "$_TMAX_SOURCE_LOCK_COUNT" in
    0)
        if [ "${SLURM_PROCID:-0}" = "0" ]; then
            echo "[tmax run] WARNING: no source revision lock; continuing a legacy/manual run" >&2
        fi
        ;;
    3)
        tmax_verify_source_revisions "${PROJECT_ROOT}" "${SLIME_DIR}" "${MEGATRON_DIR}"
        ;;
    *)
        echo "ERROR: partial source revision lock at allocation startup" >&2
        exit 1
        ;;
esac
unset _TMAX_SOURCE_LOCK_COUNT

tmax_configure_graceful_deadline
if [ "${SLURM_PROCID:-0}" = "0" ] && [ -n "${SLIME_GRACEFUL_EXIT_AT_UNIX_TIME:-}" ]; then
    echo "[tmax run] graceful checkpoint deadline: $(date -u -d "@${SLIME_GRACEFUL_EXIT_AT_UNIX_TIME}" '+%Y-%m-%dT%H:%M:%SZ') (${TMAX_GRACEFUL_DEADLINE_SOURCE:-explicit override})"
fi

if [ -n "${PROMPT_DATA:-}" ]; then
    PREPARE_DATA_DEFAULT=0
else
    export PROMPT_DATA="${TMAX_TRAIN_DATA}"
    PREPARE_DATA_DEFAULT=1
fi
TMAX_PREPARE_DATA="${TMAX_PREPARE_DATA:-${PREPARE_DATA_DEFAULT}}"
export TMAX_VALIDATE_EXISTING_ASSETS="${TMAX_VALIDATE_EXISTING_ASSETS:-1}"
case "${TMAX_VALIDATE_EXISTING_ASSETS}" in
    0|1) ;;
    *)
        echo "ERROR: TMAX_VALIDATE_EXISTING_ASSETS must be 0 or 1" >&2
        exit 1
        ;;
esac

export RUN_DIR="${RUN_DIR:-${POLAR_DATA_ROOT}/runs/${RUN_ID}/job-${SLURM_JOB_ID:-manual}}"
mkdir -p "${RUN_DIR}"
TMAX_DATA_READY_FILE="${RUN_DIR}/tmax-data-integrity.ready"
TMAX_DATA_INTEGRITY_SCRIPT="${SCRIPT_DIR}/validate_data_integrity.py"
TMAX_DATA_VALIDATION_WAIT_SECONDS="${TMAX_DATA_VALIDATION_WAIT_SECONDS:-600}"
if ! [[ "${TMAX_DATA_VALIDATION_WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TMAX_DATA_VALIDATION_WAIT_SECONDS must be a positive integer" >&2
    exit 1
fi
PREPARE_ARGS=(
    --dataset-dir "${TMAX_DATASET_DIR}"
    --image-dir "${APPTAINER_IMAGE_DIR}"
    --output "${PROMPT_DATA}"
    --start-index "${TMAX_TRAIN_START_INDEX}"
    --max-tasks "${TMAX_MAX_TASKS}"
)
if [ -n "${TMAX_EXCLUDE_DATA}" ]; then
    PREPARE_ARGS+=(--exclude-data "${TMAX_EXCLUDE_DATA}")
fi
if [ "${TMAX_REQUIRE_EXACT_TOTAL_TASKS}" = "1" ]; then
    PREPARE_ARGS+=(--expected-total-tasks "${TMAX_TOTAL_TASKS}")
fi
if [ "${TMAX_ONLY_READY}" = "1" ]; then
    PREPARE_ARGS+=(--only-ready)
fi
if [ "${TMAX_EVAL_ENABLED}" = "1" ]; then
    if [ "${TMAX_EVAL_SOURCE}" = "harbor" ]; then
        EVAL_PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_harbor_eval.py"
        EVAL_PREPARE_ARGS=(
            --tasks-dir "${TMAX_HARBOR_EVAL_TASKS_DIR}"
            --image-dir "${TMAX_HARBOR_EVAL_IMAGE_DIR}"
            --output "${TMAX_EVAL_DATA}"
            --dataset-name "${TMAX_EVAL_DATASET_NAME}"
            --dataset-revision "${TMAX_HARBOR_EVAL_REVISION}"
            --max-tasks "${TMAX_EVAL_MAX_TASKS}"
            --agent-timeout-cap "${TMAX_HARBOR_EVAL_AGENT_TIMEOUT_CAP}"
            --verifier-timeout-cap "${TMAX_HARBOR_EVAL_VERIFIER_TIMEOUT_CAP}"
            --timeout-overhead "${TMAX_HARBOR_EVAL_TIMEOUT_OVERHEAD}"
            --agent-step-limit "${TMAX_HARBOR_EVAL_AGENT_STEP_LIMIT}"
        )
    else
        EVAL_PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_data.py"
        EVAL_PREPARE_ARGS=(
            --dataset-dir "${TMAX_DATASET_DIR}"
            --image-dir "${APPTAINER_IMAGE_DIR}"
            --output "${TMAX_EVAL_DATA}"
            --start-index "${TMAX_EVAL_START_INDEX}"
            --max-tasks "${TMAX_EVAL_MAX_TASKS}"
        )
    fi
    INTEGRITY_EVAL_ARGS=(
        --eval-data "${TMAX_EVAL_DATA}"
        --eval-name "${TMAX_EVAL_DATASET_NAME}"
    )
    if [ "${TMAX_EXTERNAL_EVAL_ENABLED}" = "1" ]; then
        EXTERNAL_PREPARE_ARGS=(
            --tasks-dir "${TMAX_HARBOR_EVAL_TASKS_DIR}"
            --image-dir "${TMAX_HARBOR_EVAL_IMAGE_DIR}"
            --output "${TMAX_EXTERNAL_EVAL_DATA}"
            --dataset-name "${TMAX_EXTERNAL_EVAL_DATASET_NAME}"
            --dataset-revision "${TMAX_HARBOR_EVAL_REVISION}"
            --max-tasks "${TMAX_EXTERNAL_EVAL_MAX_TASKS}"
            --agent-timeout-cap "${TMAX_HARBOR_EVAL_AGENT_TIMEOUT_CAP}"
            --verifier-timeout-cap "${TMAX_HARBOR_EVAL_VERIFIER_TIMEOUT_CAP}"
            --timeout-overhead "${TMAX_HARBOR_EVAL_TIMEOUT_OVERHEAD}"
            --agent-step-limit "${TMAX_EXTERNAL_EVAL_AGENT_STEP_LIMIT}"
        )
        INTEGRITY_EVAL_ARGS+=(
            --eval-dataset "${TMAX_EXTERNAL_EVAL_DATASET_NAME}" "${TMAX_EXTERNAL_EVAL_DATA}"
        )
    fi
fi

if [ "${SLURM_PROCID:-0}" = "0" ]; then
    rm -f "${TMAX_DATA_READY_FILE}"
    if [ "${TMAX_EVAL_ENABLED}" = "1" ]; then
        _TMAX_EXPECTED_TRAIN_DATA_SHA256="${TMAX_TRAIN_DATA_SHA256:-}"
        _TMAX_EXPECTED_EVAL_DATA_SHA256="${TMAX_EVAL_DATA_SHA256:-}"
        _TMAX_EXPECTED_EXTERNAL_DATA_SHA256="${TMAX_EXTERNAL_EVAL_DATA_SHA256:-}"
        _TMAX_EXPECTED_EVAL_BUNDLE_SHA256="${TMAX_EVAL_BUNDLE_SHA256:-}"
        if { [ -z "${_TMAX_EXPECTED_TRAIN_DATA_SHA256}" ] || \
             [ -z "${_TMAX_EXPECTED_EVAL_DATA_SHA256}" ]; } && \
           [ -s "${TMAX_DATA_INTEGRITY_MANIFEST}" ]; then
            _TMAX_PINNED_DATA_SHA256="$(
                "${TMAX_SIF_PYTHON_BIN}" -c \
                    'import json,pathlib,sys; m=json.loads(pathlib.Path(sys.argv[1]).read_text()); d={x["name"]:x["sha256"] for x in m["datasets"]}; print(m["train"]["sha256"],d.get(sys.argv[2],""),d.get(sys.argv[3],""))' \
                    "${TMAX_DATA_INTEGRITY_MANIFEST}" "${TMAX_EVAL_DATASET_NAME}" "${TMAX_EXTERNAL_EVAL_DATASET_NAME}"
            )"
            read -r _TMAX_MANIFEST_TRAIN_SHA256 _TMAX_MANIFEST_EVAL_SHA256 \
                _TMAX_MANIFEST_EXTERNAL_SHA256 \
                <<<"${_TMAX_PINNED_DATA_SHA256}"
            _TMAX_EXPECTED_TRAIN_DATA_SHA256="${_TMAX_EXPECTED_TRAIN_DATA_SHA256:-${_TMAX_MANIFEST_TRAIN_SHA256}}"
            _TMAX_EXPECTED_EVAL_DATA_SHA256="${_TMAX_EXPECTED_EVAL_DATA_SHA256:-${_TMAX_MANIFEST_EVAL_SHA256}}"
            _TMAX_EXPECTED_EXTERNAL_DATA_SHA256="${_TMAX_EXPECTED_EXTERNAL_DATA_SHA256:-${_TMAX_MANIFEST_EXTERNAL_SHA256}}"
            unset _TMAX_PINNED_DATA_SHA256 _TMAX_MANIFEST_TRAIN_SHA256 \
                _TMAX_MANIFEST_EVAL_SHA256 _TMAX_MANIFEST_EXTERNAL_SHA256
        fi
        if [ -n "${_TMAX_EXPECTED_TRAIN_DATA_SHA256}" ] && \
           ! [[ "${_TMAX_EXPECTED_TRAIN_DATA_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
            echo "ERROR: pinned TMAX_TRAIN_DATA_SHA256 is invalid" >&2
            exit 1
        fi
        if [ -n "${_TMAX_EXPECTED_EVAL_DATA_SHA256}" ] && \
           ! [[ "${_TMAX_EXPECTED_EVAL_DATA_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
            echo "ERROR: pinned TMAX_EVAL_DATA_SHA256 is invalid" >&2
            exit 1
        fi
        if [ -n "${_TMAX_EXPECTED_EXTERNAL_DATA_SHA256}" ] && \
           ! [[ "${_TMAX_EXPECTED_EXTERNAL_DATA_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
            echo "ERROR: pinned TMAX_EXTERNAL_EVAL_DATA_SHA256 is invalid" >&2
            exit 1
        fi
        # Reject aliases before either preparation step can overwrite the
        # other dataset through a symlink or a differently-spelled path.
        "${TMAX_SIF_PYTHON_BIN}" "${TMAX_DATA_INTEGRITY_SCRIPT}" \
            --train-data "${TMAX_TRAIN_DATA}" \
            "${INTEGRITY_EVAL_ARGS[@]}" \
            --check-paths-only
        if [ "${PROMPT_DATA}" != "${TMAX_TRAIN_DATA}" ]; then
            "${TMAX_SIF_PYTHON_BIN}" "${TMAX_DATA_INTEGRITY_SCRIPT}" \
                --train-data "${PROMPT_DATA}" \
                "${INTEGRITY_EVAL_ARGS[@]}" \
                --check-paths-only
        fi
    fi

    if [ "${TMAX_PREPARE_DATA}" = "1" ] || [ ! -s "${PROMPT_DATA}" ]; then
        "${TMAX_SIF_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data.py" "${PREPARE_ARGS[@]}"
    fi
    # Submission-time validation is not a runtime trust boundary: queued jobs
    # can start much later and the shared JSONL/SIF files remain mutable. The
    # pinned matrix opts into the fast path explicitly after its source bundle
    # has been deeply audited; other launchers retain runtime revalidation.
    if [ "${TMAX_VALIDATE_EXISTING_ASSETS}" = "1" ]; then
        "${TMAX_SIF_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data.py" \
            "${PREPARE_ARGS[@]}" --validate-existing
    else
        echo "[tmax run] skipping deep validation of existing training assets"
    fi

    if [ "${TMAX_EVAL_ENABLED}" = "1" ]; then
        if [ "${TMAX_PREPARE_EVAL_DATA}" = "1" ] || [ ! -s "${TMAX_EVAL_DATA}" ]; then
            "${TMAX_SIF_PYTHON_BIN}" "${EVAL_PREPARE_SCRIPT}" "${EVAL_PREPARE_ARGS[@]}"
        fi
        if [ "${TMAX_VALIDATE_EXISTING_ASSETS}" = "1" ]; then
            "${TMAX_SIF_PYTHON_BIN}" "${EVAL_PREPARE_SCRIPT}" \
                "${EVAL_PREPARE_ARGS[@]}" --validate-existing
        else
            echo "[tmax run] skipping deep validation of existing primary eval assets"
        fi
        if [ "${TMAX_EXTERNAL_EVAL_ENABLED}" = "1" ]; then
            if [ "${TMAX_PREPARE_EVAL_DATA}" = "1" ] || [ ! -s "${TMAX_EXTERNAL_EVAL_DATA}" ]; then
                "${TMAX_SIF_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_harbor_eval.py" "${EXTERNAL_PREPARE_ARGS[@]}"
            fi
            if [ "${TMAX_VALIDATE_EXISTING_ASSETS}" = "1" ]; then
                "${TMAX_SIF_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_harbor_eval.py" \
                    "${EXTERNAL_PREPARE_ARGS[@]}" --validate-existing
            else
                echo "[tmax run] skipping deep validation of existing external eval assets"
            fi
        fi
        _TMAX_DATA_INTEGRITY_CANDIDATE="${TMAX_DATA_INTEGRITY_MANIFEST}.candidate.$$"
        rm -f "${_TMAX_DATA_INTEGRITY_CANDIDATE}"
        POLAR_EVAL_DATA_INTEGRITY_B64="$(
            "${TMAX_SIF_PYTHON_BIN}" "${TMAX_DATA_INTEGRITY_SCRIPT}" \
                --train-data "${PROMPT_DATA}" \
                "${INTEGRITY_EVAL_ARGS[@]}" \
                --manifest "${_TMAX_DATA_INTEGRITY_CANDIDATE}" \
                --print-base64
        )"
        export POLAR_EVAL_DATA_INTEGRITY_B64
        _TMAX_CANDIDATE_DATA_SHA256="$(
            "${TMAX_SIF_PYTHON_BIN}" -c \
                'import base64,json,sys; m=json.loads(base64.b64decode(sys.argv[1])); d={x["name"]:x["sha256"] for x in m["datasets"]}; print(m["train"]["sha256"],d.get(sys.argv[2],""),d.get(sys.argv[3],""))' \
                "${POLAR_EVAL_DATA_INTEGRITY_B64}" "${TMAX_EVAL_DATASET_NAME}" "${TMAX_EXTERNAL_EVAL_DATASET_NAME}"
        )"
        read -r TMAX_TRAIN_DATA_SHA256 TMAX_EVAL_DATA_SHA256 \
            TMAX_EXTERNAL_EVAL_DATA_SHA256 \
            <<<"${_TMAX_CANDIDATE_DATA_SHA256}"
        if [ "${TMAX_EXTERNAL_EVAL_ENABLED}" = "1" ]; then
            TMAX_EVAL_BUNDLE_SHA256="$(
                "${TMAX_SIF_PYTHON_BIN}" "${TMAX_DATA_INTEGRITY_SCRIPT}" \
                    --bundle-sha256 "${TMAX_EVAL_DATASET_NAME}" "${TMAX_EVAL_DATA}" \
                    --bundle-sha256 "${TMAX_EXTERNAL_EVAL_DATASET_NAME}" "${TMAX_EXTERNAL_EVAL_DATA}"
            )"
        else
            TMAX_EVAL_BUNDLE_SHA256="${TMAX_EVAL_DATA_SHA256}"
            unset TMAX_EXTERNAL_EVAL_DATA_SHA256
        fi
        export TMAX_TRAIN_DATA_SHA256 TMAX_EVAL_DATA_SHA256 TMAX_EVAL_BUNDLE_SHA256
        export TMAX_EXTERNAL_EVAL_DATA_SHA256
        unset _TMAX_CANDIDATE_DATA_SHA256
        if [ -n "${_TMAX_EXPECTED_TRAIN_DATA_SHA256}" ] && \
           [ "${TMAX_TRAIN_DATA_SHA256}" != "${_TMAX_EXPECTED_TRAIN_DATA_SHA256}" ]; then
            rm -f "${_TMAX_DATA_INTEGRITY_CANDIDATE}"
            echo "ERROR: training data changed from pinned sha256=${_TMAX_EXPECTED_TRAIN_DATA_SHA256} to ${TMAX_TRAIN_DATA_SHA256}" >&2
            exit 1
        fi
        if [ -n "${_TMAX_EXPECTED_EVAL_DATA_SHA256}" ] && \
           [ "${TMAX_EVAL_DATA_SHA256}" != "${_TMAX_EXPECTED_EVAL_DATA_SHA256}" ]; then
            rm -f "${_TMAX_DATA_INTEGRITY_CANDIDATE}"
            echo "ERROR: fixed eval data changed from pinned sha256=${_TMAX_EXPECTED_EVAL_DATA_SHA256} to ${TMAX_EVAL_DATA_SHA256}" >&2
            exit 1
        fi
        if [ -n "${_TMAX_EXPECTED_EXTERNAL_DATA_SHA256}" ] && \
           [ "${TMAX_EXTERNAL_EVAL_DATA_SHA256:-}" != "${_TMAX_EXPECTED_EXTERNAL_DATA_SHA256}" ]; then
            rm -f "${_TMAX_DATA_INTEGRITY_CANDIDATE}"
            echo "ERROR: fixed external eval data changed from pinned sha256=${_TMAX_EXPECTED_EXTERNAL_DATA_SHA256} to ${TMAX_EXTERNAL_EVAL_DATA_SHA256:-<unset>}" >&2
            exit 1
        fi
        if [ -n "${_TMAX_EXPECTED_EVAL_BUNDLE_SHA256}" ] && \
           [ "${TMAX_EVAL_BUNDLE_SHA256}" != "${_TMAX_EXPECTED_EVAL_BUNDLE_SHA256}" ]; then
            rm -f "${_TMAX_DATA_INTEGRITY_CANDIDATE}"
            echo "ERROR: eval bundle changed from pinned sha256=${_TMAX_EXPECTED_EVAL_BUNDLE_SHA256} to ${TMAX_EVAL_BUNDLE_SHA256}" >&2
            exit 1
        fi
        mv -f -- "${_TMAX_DATA_INTEGRITY_CANDIDATE}" "${TMAX_DATA_INTEGRITY_MANIFEST}"
        unset _TMAX_DATA_INTEGRITY_CANDIDATE _TMAX_EXPECTED_TRAIN_DATA_SHA256 \
            _TMAX_EXPECTED_EVAL_DATA_SHA256 _TMAX_EXPECTED_EXTERNAL_DATA_SHA256 \
            _TMAX_EXPECTED_EVAL_BUNDLE_SHA256
        echo "[tmax run] data integrity manifest: ${TMAX_DATA_INTEGRITY_MANIFEST}"
    else
        unset POLAR_EVAL_DATA_INTEGRITY_B64
        unset TMAX_TRAIN_DATA_SHA256
        unset TMAX_EVAL_DATA_SHA256
        unset TMAX_EXTERNAL_EVAL_DATA_SHA256
        unset TMAX_EVAL_BUNDLE_SHA256
    fi

    printf 'ok\n' >"${TMAX_DATA_READY_FILE}.tmp.$$"
    mv -f -- "${TMAX_DATA_READY_FILE}.tmp.$$" "${TMAX_DATA_READY_FILE}"
else
    _tmax_data_wait_deadline=$((SECONDS + TMAX_DATA_VALIDATION_WAIT_SECONDS))
    while [ ! -s "${TMAX_DATA_READY_FILE}" ] && [ "${SECONDS}" -lt "${_tmax_data_wait_deadline}" ]; do
        sleep 0.2
    done
    if [ ! -s "${TMAX_DATA_READY_FILE}" ]; then
        echo "ERROR: timed out waiting for rank 0 TMax data validation: ${TMAX_DATA_READY_FILE}" >&2
        exit 1
    fi
    unset _tmax_data_wait_deadline
fi

export REQUIRE_SWEGYM_HARNESS=0
export TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${SCRIPT_DIR}/topology.yaml}"
export POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${SCRIPT_DIR}/polar_config.yaml}"
export POLAR_ROLLOUT_SAVE_DIR="${POLAR_ROLLOUT_SAVE_DIR:-${RUN_DIR}/rollout_results}"
export TOPOLOGY_PATH="${TOPOLOGY_PATH:-${RUN_DIR}/topology.yaml}"
export CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${RUN_DIR}/polar_config.yaml}"
export GPU_MONITOR_PREFIX="${GPU_MONITOR_PREFIX:-polar_tmax_system}"

# Each Slurm rank gets an immutable launcher snapshot.  The repository is bind
# mounted into long-running allocations, and changing the shared launcher while
# bash is still reading it can otherwise leave that allocation parsing a mixed
# old/new file.
SHARED_RUN_SOURCE="${PROJECT_ROOT}/examples/swegym_slime_grpo/run.sh"
SHARED_RUN_SNAPSHOT="${RUN_DIR}/shared_run.sh"
mkdir -p "${RUN_DIR}"
if [ "${SLURM_PROCID:-0}" = "0" ]; then
    cp -- "${SHARED_RUN_SOURCE}" "${SHARED_RUN_SNAPSHOT}.tmp.$$"
    chmod 0755 "${SHARED_RUN_SNAPSHOT}.tmp.$$"
    mv -f -- "${SHARED_RUN_SNAPSHOT}.tmp.$$" "${SHARED_RUN_SNAPSHOT}"
else
    for _ in $(seq 1 300); do
        [ -s "${SHARED_RUN_SNAPSHOT}" ] && break
        sleep 0.1
    done
    if [ ! -s "${SHARED_RUN_SNAPSHOT}" ]; then
        echo "ERROR: timed out waiting for rank 0 launcher snapshot: ${SHARED_RUN_SNAPSHOT}" >&2
        exit 1
    fi
fi
export POLAR_SHARED_SCRIPT_DIR="$(dirname -- "${SHARED_RUN_SOURCE}")"
export POLAR_TRAIN_PROJECT_ROOT="${PROJECT_ROOT}"

exec bash "${SHARED_RUN_SNAPSHOT}"
