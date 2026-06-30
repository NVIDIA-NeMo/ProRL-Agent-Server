#!/usr/bin/env bash
# Prepare a deterministic TMax prompt set, then launch multi-node Slime GRPO.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../swegym_slime_grpo/launcher_utils.sh
source "${PROJECT_ROOT}/examples/swegym_slime_grpo/launcher_utils.sh"
if [ -n "${TMAX_TRAIN_DATA+x}" ]; then
    _TMAX_TRAIN_DATA_WAS_EXPLICIT=1
else
    _TMAX_TRAIN_DATA_WAS_EXPLICIT=0
fi
if [ -n "${TMAX_PREPARE_DATA+x}" ]; then
    _TMAX_PREPARE_DATA_WAS_EXPLICIT=1
else
    _TMAX_PREPARE_DATA_WAS_EXPLICIT=0
fi
# shellcheck source=./env.cwdfw.sh
source "${SCRIPT_DIR}/env.cwdfw.sh"
# shellcheck source=./run_state.sh
source "${SCRIPT_DIR}/run_state.sh"

if [ "${TMAX_PERSIST_RUN_STATE:-1}" = "1" ] && \
   [ "${SUBMIT_DRY_RUN:-0}" != "1" ] && \
   [ "${TMAX_RUN_STATE_LOCK_HELD:-0}" != "1" ]; then
    command -v flock >/dev/null || { echo "ERROR: flock is required" >&2; exit 1; }
    mkdir -p "$(dirname "${TMAX_RUN_STATE_FILE}")"
    exec 8>"${TMAX_RUN_STATE_FILE}.lock"
    if ! flock -n 8; then
        echo "ERROR: a TMax watcher or submitter already owns ${TMAX_RUN_STATE_FILE}" >&2
        exit 1
    fi
fi

if [ ! -x "${TMAX_SIF_PYTHON_BIN}" ]; then
    echo "ERROR: Python not executable: ${TMAX_SIF_PYTHON_BIN}" >&2
    exit 1
fi

# Fail before requesting GPUs when the shared training venv no longer matches
# its CUDA/Transformer-Engine ABI.  This caught a real failure where a package
# operation replaced cuBLAS 13.3 with 13.1: importing torch alone still worked,
# but Megatron's Transformer Engine import failed only after the allocation had
# started.  The node entrypoint repeats the same check inside the Pyxis image.
TMAX_TRAIN_ABI_PREFLIGHT="${TMAX_TRAIN_ABI_PREFLIGHT:-1}"
case "${TMAX_TRAIN_ABI_PREFLIGHT}" in
    0|1) ;;
    *)
        echo "ERROR: TMAX_TRAIN_ABI_PREFLIGHT must be 0 or 1" >&2
        exit 1
        ;;
esac
if [ "${TMAX_TRAIN_ABI_PREFLIGHT}" = "1" ]; then
    TMAX_TRAIN_PYTHON_BIN="${TMAX_TRAIN_PYTHON_BIN:-${POLR_TRAIN_VENV}/bin/python3}"
    if [ ! -x "${TMAX_TRAIN_PYTHON_BIN}" ]; then
        echo "ERROR: training Python not executable: ${TMAX_TRAIN_PYTHON_BIN}" >&2
        exit 1
    fi
    if ! PYTHONNOUSERSITE=1 \
         PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src" \
         "${TMAX_TRAIN_PYTHON_BIN}" - <<'PY'
from importlib.metadata import version

import torch
import transformer_engine.pytorch  # noqa: F401
import megatron.core.tensor_parallel  # noqa: F401
import polar  # noqa: F401
import slime  # noqa: F401
import slime_bridge  # noqa: F401

print(
    "[tmax ABI] "
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"transformer-engine={version('transformer-engine')} "
    f"nvidia-cublas={version('nvidia-cublas')}"
)
PY
    then
        echo "ERROR: training ABI preflight failed before Slurm submission" >&2
        echo "  Verify torch, transformer-engine, nvidia-cublas, and the Megatron checkout in ${POLR_TRAIN_VENV}." >&2
        exit 1
    fi
fi
TMAX_DATA_INTEGRITY_SCRIPT="${SCRIPT_DIR}/validate_data_integrity.py"
if [ "${TMAX_EVAL_ENABLED}" = "1" ]; then
    INTEGRITY_EVAL_ARGS=(
        --eval-data "${TMAX_EVAL_DATA}"
        --eval-name "${TMAX_EVAL_DATASET_NAME}"
    )
    if [ "${TMAX_EXTERNAL_EVAL_ENABLED}" = "1" ]; then
        INTEGRITY_EVAL_ARGS+=(
            --eval-dataset "${TMAX_EXTERNAL_EVAL_DATASET_NAME}" "${TMAX_EXTERNAL_EVAL_DATA}"
        )
    fi
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
    if [ -n "${_TMAX_EXPECTED_EVAL_BUNDLE_SHA256}" ] && \
       ! [[ "${_TMAX_EXPECTED_EVAL_BUNDLE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "ERROR: pinned TMAX_EVAL_BUNDLE_SHA256 is invalid" >&2
        exit 1
    fi
    "${TMAX_SIF_PYTHON_BIN}" "${TMAX_DATA_INTEGRITY_SCRIPT}" \
        --train-data "${TMAX_TRAIN_DATA}" \
        "${INTEGRITY_EVAL_ARGS[@]}" \
        --check-paths-only
fi
case "${TMAX_AGENT_HARNESS}" in
    mini_swe_agent|vanillux2)
        _mini_swe_python="${MINI_SWE_AGENT_RUNTIME_DIR}/venv/bin/python"
        _mini_swe_timing_source="${PROJECT_ROOT}/src/polar/agent/presets/mini_swe_timing.py"
        _mini_swe_runner_source="${PROJECT_ROOT}/src/polar/agent/presets/mini_swe_runner.py"
        _mini_swe_vanillux_source="${PROJECT_ROOT}/src/polar/agent/presets/mini_swe_vanillux.py"
        _mini_swe_vanillux_config_source="${PROJECT_ROOT}/src/polar/agent/presets/vanillux2.yaml"
        _mini_swe_vanillux_config_installed="${MINI_SWE_AGENT_RUNTIME_DIR}/config/vanillux2.yaml"
        _mini_swe_timing_installed=""
        _mini_swe_runner_installed=""
        _mini_swe_vanillux_installed=""
        if [ -x "${_mini_swe_python}" ]; then
            if ! _mini_swe_timing_installed="$(
                "${_mini_swe_python}" -c \
                  'import polar_mini_swe_timing as m; assert m.TIMING_SCHEMA_VERSION == 1; print(m.__file__)' \
                  2>/dev/null | tail -n 1
            )"; then
                _mini_swe_timing_installed=""
            fi
            if ! _mini_swe_runner_installed="$(
                "${_mini_swe_python}" -c \
                  'import polar_mini_swe_runner as m; print(m.__file__)' \
                  2>/dev/null | tail -n 1
            )"; then
                _mini_swe_runner_installed=""
            fi
            if ! _mini_swe_vanillux_installed="$(
                "${_mini_swe_python}" -c \
                  'import polar_mini_swe_vanillux as m; print(m.__file__)' \
                  2>/dev/null | tail -n 1
            )"; then
                _mini_swe_vanillux_installed=""
            fi
        fi
        if [ ! -x "${MINI_SWE_AGENT_BIN}" ] || \
           [ ! -x "${_mini_swe_python}" ] || \
           ! grep -q 'POLAR_TASK_PYTHONPATH' "${MINI_SWE_AGENT_BIN}" || \
           grep -Eq '^[[:space:]]*export[[:space:]]+PYTHONPATH=' "${MINI_SWE_AGENT_BIN}" || \
           [ ! -f "${_mini_swe_timing_source}" ] || \
           [ ! -f "${_mini_swe_runner_source}" ] || \
           [ ! -f "${_mini_swe_vanillux_source}" ] || \
           [ ! -f "${_mini_swe_vanillux_config_source}" ] || \
           [ -z "${_mini_swe_timing_installed}" ] || \
           [ ! -f "${_mini_swe_timing_installed}" ] || \
           [ -z "${_mini_swe_runner_installed}" ] || \
           [ ! -f "${_mini_swe_runner_installed}" ] || \
           [ -z "${_mini_swe_vanillux_installed}" ] || \
           [ ! -f "${_mini_swe_vanillux_installed}" ] || \
           [ ! -f "${_mini_swe_vanillux_config_installed}" ] || \
           ! cmp -s "${_mini_swe_timing_source}" "${_mini_swe_timing_installed}" || \
           ! cmp -s "${_mini_swe_runner_source}" "${_mini_swe_runner_installed}" || \
           ! cmp -s "${_mini_swe_vanillux_source}" "${_mini_swe_vanillux_installed}" || \
           ! cmp -s "${_mini_swe_vanillux_config_source}" "${_mini_swe_vanillux_config_installed}"; then
            echo "ERROR: shared mini-swe-agent runtime is missing or stale at ${MINI_SWE_AGENT_RUNTIME_DIR}" >&2
            echo "  The injected timing and isolated-network modules must match this checkout." >&2
            echo "  Run: bash ${SCRIPT_DIR}/prepare_mini_swe_agent.sh" >&2
            exit 1
        fi
        unset _mini_swe_python _mini_swe_timing_source _mini_swe_timing_installed \
            _mini_swe_runner_source _mini_swe_runner_installed \
            _mini_swe_vanillux_source _mini_swe_vanillux_installed \
            _mini_swe_vanillux_config_source _mini_swe_vanillux_config_installed
        ;;
    codex)
        if [ ! -x "${AGENT_CLI_DIR}/bin/codex" ]; then
            echo "ERROR: shared Codex CLI not found at ${AGENT_CLI_DIR}/bin/codex" >&2
            echo "  Run: bash ${SCRIPT_DIR}/prepare_agent_cli.sh" >&2
            exit 1
        fi
        ;;
esac
if [ "${TMAX_REQUIRE_WANDB}" = "1" ] && { [ -z "${WANDB_API_KEY:-}" ] || [ "${WANDB_MODE}" != "online" ]; }; then
    echo "ERROR: TMAX_REQUIRE_WANDB=1 requires WANDB_API_KEY and WANDB_MODE=online" >&2
    exit 1
fi

if [ -n "${LOAD_DIR:-}" ] && \
   [ ! -s "${SAVE_DIR}/latest_checkpointed_iteration.txt" ] && \
   ! polar_checkpoint_is_release_seed "${LOAD_DIR}"; then
    # A numbered training checkpoint also owns the rollout data-source cursor,
    # so it must be paired with the exact prompt JSONL. A release checkpoint is
    # model-only and intentionally starts a fresh cursor at rollout zero.
    if [ "${_TMAX_TRAIN_DATA_WAS_EXPLICIT}" != "1" ] || \
       [ "${_TMAX_PREPARE_DATA_WAS_EXPLICIT}" != "1" ] || \
       [ "${TMAX_PREPARE_DATA}" != "0" ]; then
        echo "ERROR: seeding a new SAVE_DIR from LOAD_DIR requires explicit TMAX_TRAIN_DATA and TMAX_PREPARE_DATA=0" >&2
        echo "  Reuse the exact prompt JSONL whose data-source state is stored in the checkpoint." >&2
        exit 1
    fi
    if [ ! -s "${LOAD_DIR}/latest_checkpointed_iteration.txt" ] || [ ! -s "${TMAX_TRAIN_DATA}" ]; then
        echo "ERROR: LOAD_DIR checkpoint or TMAX_TRAIN_DATA is missing" >&2
        exit 1
    fi
    _tmax_seed_iter="$(tr -d '[:space:]' <"${LOAD_DIR}/latest_checkpointed_iteration.txt")"
    if ! [[ "${_tmax_seed_iter}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: LOAD_DIR must point to a numbered training checkpoint" >&2
        exit 1
    fi
    _tmax_seed_samples="$(awk 'NF { count += 1 } END { print count + 0 }' "${TMAX_TRAIN_DATA}")"
    _tmax_seed_target="$(( (_tmax_seed_samples + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE * NUM_EPOCH - 1 ))"
    if [ "${_tmax_seed_iter}" -ge "${_tmax_seed_target}" ]; then
        echo "ERROR: seed iteration ${_tmax_seed_iter} has already reached target ${_tmax_seed_target} for ${_tmax_seed_samples} prompts" >&2
        echo "  This usually means TMAX_TRAIN_DATA does not match the checkpoint's rollout data-source state." >&2
        exit 1
    fi
    unset _tmax_seed_iter _tmax_seed_samples _tmax_seed_target
fi

PREPARE_ARGS=(
    --dataset-dir "${TMAX_DATASET_DIR}"
    --image-dir "${APPTAINER_IMAGE_DIR}"
    --output "${TMAX_TRAIN_DATA}"
    --start-index "${TMAX_TRAIN_START_INDEX}"
    --max-tasks "${TMAX_MAX_TASKS}"
)
if [ "${TMAX_EXTERNAL_EVAL_ENABLED}" = "1" ] && \
   [ "${TMAX_REQUIRE_EXACT_TOTAL_TASKS}" = "1" ]; then
    PREPARE_ARGS+=(--expected-total-tasks "${TMAX_TOTAL_TASKS}")
fi
if [ "${TMAX_ONLY_READY}" = "1" ]; then
    PREPARE_ARGS+=(--only-ready)
fi
if [ "${TMAX_PREPARE_DATA:-1}" = "1" ] || [ ! -s "${TMAX_TRAIN_DATA}" ]; then
    "${TMAX_SIF_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data.py" "${PREPARE_ARGS[@]}"
else
    "${TMAX_SIF_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data.py" \
        "${PREPARE_ARGS[@]}" --validate-existing
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
    fi
    if [ "${TMAX_PREPARE_EVAL_DATA}" = "1" ] || [ ! -s "${TMAX_EVAL_DATA}" ]; then
        "${TMAX_SIF_PYTHON_BIN}" "${EVAL_PREPARE_SCRIPT}" "${EVAL_PREPARE_ARGS[@]}"
    else
        "${TMAX_SIF_PYTHON_BIN}" "${EVAL_PREPARE_SCRIPT}" \
            "${EVAL_PREPARE_ARGS[@]}" --validate-existing
    fi
    if [ "${TMAX_EXTERNAL_EVAL_ENABLED}" = "1" ]; then
        if [ "${TMAX_PREPARE_EVAL_DATA}" = "1" ] || [ ! -s "${TMAX_EXTERNAL_EVAL_DATA}" ]; then
            "${TMAX_SIF_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_harbor_eval.py" "${EXTERNAL_PREPARE_ARGS[@]}"
        else
            "${TMAX_SIF_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_harbor_eval.py" \
                "${EXTERNAL_PREPARE_ARGS[@]}" --validate-existing
        fi
    fi
    _TMAX_DATA_INTEGRITY_CANDIDATE="${TMAX_DATA_INTEGRITY_MANIFEST}.candidate.$$"
    rm -f "${_TMAX_DATA_INTEGRITY_CANDIDATE}"
    POLAR_EVAL_DATA_INTEGRITY_B64="$(
        "${TMAX_SIF_PYTHON_BIN}" "${TMAX_DATA_INTEGRITY_SCRIPT}" \
            --train-data "${TMAX_TRAIN_DATA}" \
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
    echo "[tmax submit] data integrity manifest: ${TMAX_DATA_INTEGRITY_MANIFEST}"
else
    unset POLAR_EVAL_DATA_INTEGRITY_B64
    unset TMAX_TRAIN_DATA_SHA256
    unset TMAX_EVAL_DATA_SHA256
    unset TMAX_EXTERNAL_EVAL_DATA_SHA256
    unset TMAX_EVAL_BUNDLE_SHA256
fi

export PROMPT_DATA="${TMAX_TRAIN_DATA}"
export TMAX_PREPARE_DATA=0
export TMAX_PREPARE_EVAL_DATA=0
export POLAR_TRAIN_RUN_SCRIPT="${SCRIPT_DIR}/run.sh"
export POLAR_LAUNCHER_LABEL="Polar TMax Slime-GRPO"
export JOB_NAME="${JOB_NAME:-polar-tmax-${RUN_ID}}"
export POLAR_SUBMIT_RECEIPT_FILE="${TMAX_SUBMIT_RECEIPT_FILE}"
export WANDB_PROJECT WANDB_GROUP RUN_ID SAVE_DIR LOAD_DIR

if [ "${TMAX_PERSIST_RUN_STATE:-1}" = "1" ] && [ "${SUBMIT_DRY_RUN:-0}" != "1" ]; then
    tmax_write_run_state "${TMAX_RUN_STATE_FILE}"
    echo "[tmax submit] run state: ${TMAX_RUN_STATE_FILE}"
fi

if [ "${SUBMIT_DRY_RUN:-0}" != "1" ]; then
    rm -f "${TMAX_SUBMIT_RECEIPT_FILE}"
fi
bash "${PROJECT_ROOT}/examples/swegym_slime_grpo/submit_slurm.sh"

if [ "${SUBMIT_DRY_RUN:-0}" = "1" ]; then
    exit 0
fi
if [ ! -s "${TMAX_SUBMIT_RECEIPT_FILE}" ]; then
    echo "ERROR: sbatch succeeded without a submission receipt: ${TMAX_SUBMIT_RECEIPT_FILE}" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${TMAX_SUBMIT_RECEIPT_FILE}"
if ! [[ "${POLAR_SUBMITTED_JOB_ID:-}" =~ ^[0-9]+$ ]] || \
   ! [[ "${POLAR_SUBMITTED_AT_UNIX:-}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid submission receipt: ${TMAX_SUBMIT_RECEIPT_FILE}" >&2
    exit 1
fi
export TMAX_LAST_JOB_ID="${POLAR_SUBMITTED_JOB_ID}"
export TMAX_LAST_JOB_SUBMITTED_AT="${POLAR_SUBMITTED_AT_UNIX}"
export TMAX_LAST_JOB_CHECKPOINT_ITER=-1
if [ -s "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
    _tmax_latest_iter="$(tr -d '[:space:]' <"${SAVE_DIR}/latest_checkpointed_iteration.txt")"
    if [[ "${_tmax_latest_iter}" =~ ^[0-9]+$ ]]; then
        export TMAX_LAST_JOB_CHECKPOINT_ITER="${_tmax_latest_iter}"
    fi
    unset _tmax_latest_iter
fi
if [ "${TMAX_PERSIST_RUN_STATE:-1}" = "1" ]; then
    tmax_write_run_state "${TMAX_RUN_STATE_FILE}"
fi
