#!/usr/bin/env bash
# Submit Controller V3 through Jiarui's TMax launcher; --dry-run is side-effect free.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
# shellcheck source=./profile.sh
source "${SCRIPT_DIR}/profile.sh"

dry_run="${DRY_RUN:-0}"
if [ "${1:-}" = "--dry-run" ]; then
    dry_run=1
    shift
fi
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd -P)"
POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${SPILOT_ROOT}/data}"
if [ "$#" -ne 0 ]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

for path in     "${HF_CHECKPOINT}"     "${REF_LOAD}"     "${MINI_SWE_AGENT_RUNTIME_DIR}"     "${MODEL_ARGS_FILE}"     "${POLAR_CONFIG_TEMPLATE}"     "${TOPOLOGY_TEMPLATE}"; do
    if [ ! -e "${path}" ]; then
        echo "ERROR: required Controller V3 path is missing: ${path}" >&2
        exit 1
    fi
done

if [ ! -x "${POLAR_APPTAINER_BIN}" ]; then
    echo "ERROR: project-local Apptainer is not executable: ${POLAR_APPTAINER_BIN}" >&2
    exit 1
fi
actual_session_dir="$("${POLAR_APPTAINER_BIN}" buildcfg | sed -n 's/^SESSIONDIR=//p')"
if [ "${actual_session_dir}" != "${POLAR_APPTAINER_SESSIONDIR}" ]; then
    echo "ERROR: Apptainer SESSIONDIR mismatch: ${actual_session_dir}" >&2
    exit 1
fi
unset actual_session_dir

for path in "${TMAX_DATASET_DIR}" "${TMAX_OPEN_INSTRUCT_DIR}" "${APPTAINER_IMAGE_DIR}"; do
    if [ ! -e "${path}" ]; then
        echo "ERROR: required TMax path is missing: ${path}" >&2
        exit 1
    fi
done

"${TMAX_SIF_PYTHON_BIN:-python3}" - "${MINI_SWE_AGENT_RUNTIME_DIR}/.polar-mini-runtime-manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["python_version"] == "3.10.20", manifest["python_version"]
assert manifest["mini_swe_agent_version"] == "2.4.0", manifest["mini_swe_agent_version"]
PY

PREPARE_ARGS=(
    --open-instruct-dir "${TMAX_OPEN_INSTRUCT_DIR}"
    --output "${TMAX_TRAIN_DATA}"
    --tasks-dir "${TMAX_OPEN_INSTRUCT_TASKS_DIR}"
)
if [ -n "${TMAX_OPEN_INSTRUCT_MAX_ROWS:-}" ]; then
    PREPARE_ARGS+=(--max-rows "${TMAX_OPEN_INSTRUCT_MAX_ROWS}")
fi
if [ "${dry_run}" = "1" ]; then
    PREPARE_ARGS+=(--check-only)
fi
"${TMAX_SIF_PYTHON_BIN}" "${SCRIPT_DIR}/prepare_open_instruct_data.py" "${PREPARE_ARGS[@]}"

if [ "${dry_run}" = "1" ]; then
    printf '%s\n' "Controller V3 Slurm dry-run (no command executed)"
    printf '%s\n' "  nodes=${NUM_NODES} gpus_per_node=8 total_gpus=$((NUM_NODES * 8)) partition=${PARTITION}"
    printf '%s\n' "  actor=${ACTOR_NUM_NODES}x8 controller_rollout=1x2 frozen_qwen=3x2"
    printf '%s\n' "  gateways=${POLAR_GATEWAY_COUNT_OVERRIDE}"
    printf '%s\n' "  controller=Qwen3.6-35B-A3B"
    printf '%s\n' "  small=pool/qwen3.6-35b-a3b local_replicas=3 tp=2"
    printf '%s\n' "  large=pool/gpt-5.6-luna upstream=openai/openai/gpt-5.6-luna api=responses reasoning=max"
    printf '%s\n' "  request_caps_per_gateway=qwen:1,gpt:4 episode_admission=false"
    printf '%s\n' "  hf_checkpoint=${HF_CHECKPOINT}"
    printf '%s\n' "  ref_checkpoint=${REF_LOAD}"
    printf '%s\n' "  runtime=${MINI_SWE_AGENT_RUNTIME_DIR} (Python 3.10.20, Mini-SWE-Agent 2.4.0)"
    printf '%s\n' "  tmax_dataset=${TMAX_DATASET_DIR:-${POLAR_DATA_ROOT}/tmax-15k}"
    printf '%s\n' "  tmax_sif_dir=${APPTAINER_IMAGE_DIR:-${POLAR_DATA_ROOT}/tmax-15k-sif}"
    printf '%s\n' "  tmax_open_instruct=${TMAX_OPEN_INSTRUCT_DIR}"
    printf '%s\n' "  tmax_train_data=${TMAX_TRAIN_DATA}"
    printf '  command='
    printf '%q ' sbatch         --nodes="${NUM_NODES}"         --ntasks="${NUM_NODES}"         --ntasks-per-node=1         --gres=gpu:8         --partition="${PARTITION}"         --time="${WALL_TIME}"         --wrap="srun --nodes=${NUM_NODES} --ntasks=${NUM_NODES} bash ${SCRIPT_DIR}/run.sh"
    printf '\n'
    exit 0
fi

if [ "${WANDB_MODE:-offline}" = "online" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    if [ "$(stat -c '%U:%a' "${CONTROLLER_V3_WANDB_NETRC}")" != "$(id -un):600" ]; then
        echo "ERROR: W&B netrc must be owned by the submitter with mode 600" >&2
        exit 1
    fi
    WANDB_API_KEY="$("${TMAX_SIF_PYTHON_BIN}" - "${CONTROLLER_V3_WANDB_NETRC}" <<'PY'
import netrc
import sys

credentials = netrc.netrc(sys.argv[1]).authenticators("api.wandb.ai")
if credentials is None:
    raise SystemExit("W&B credential for api.wandb.ai is missing")
print(credentials[2])
PY
)"
    export WANDB_API_KEY
fi

if [ -z "${POLAR_NVIDIA_API_KEY:-${NVIDIA_API_KEY:-${NVIDIA_INFERENCE_API_KEY:-}}}" ] && [ -f "${CONTROLLER_V3_NVIDIA_CREDENTIALS_FILE}" ]; then
    if [ "$(stat -c '%U:%a' "${CONTROLLER_V3_NVIDIA_CREDENTIALS_FILE}")" != "$(id -un):600" ]; then
        echo "ERROR: NVIDIA credentials must be owned by the submitter with mode 600" >&2
        exit 1
    fi
    set -a
    # shellcheck disable=SC1090
    source "${CONTROLLER_V3_NVIDIA_CREDENTIALS_FILE}"
    set +a
fi
export POLAR_NVIDIA_API_KEY="${POLAR_NVIDIA_API_KEY:-${NVIDIA_API_KEY:-${NVIDIA_INFERENCE_API_KEY:-}}}"
if [ -z "${POLAR_NVIDIA_API_KEY}" ]; then
    echo "ERROR: NVIDIA credential is required for GPT-5.6 Luna" >&2
    exit 1
fi
if [ -z "${POLAR_CONTROL_PLANE_TOKEN:-}" ]; then
    POLAR_CONTROL_PLANE_TOKEN="$(od -An -N32 -tx1 /dev/urandom | tr -d '[:space:]')"
    export POLAR_CONTROL_PLANE_TOKEN
fi

export TMAX_SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_slurm.sh"
export POLAR_TRAIN_RUN_SCRIPT="${SCRIPT_DIR}/run.sh"
export POLAR_CONFIG_TEMPLATE="${SCRIPT_DIR}/polar_config.yaml"
export TOPOLOGY_TEMPLATE="${SCRIPT_DIR}/topology.yaml"
exec bash "${SCRIPT_DIR}/../tmax_slime_grpo/submit_slurm.sh"
