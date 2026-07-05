#!/usr/bin/env bash
# Submit the fixed-pool, 8-node, 200-step SPilot Router experiment.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./experiment_defaults.sh
source "${SCRIPT_DIR}/experiment_defaults.sh"

# The shared submitter serializes POLAR_* variables into a private mode-0600
# job environment. Copy the credential under that namespace without writing it
# into YAML, command lines, logs, or the repository.
if [ -z "${POLAR_NVIDIA_API_KEY:-${NVIDIA_API_KEY:-}}" ]; then
    echo "ERROR: NVIDIA_API_KEY is not set; load the credential before submission" >&2
    exit 1
fi
export POLAR_NVIDIA_API_KEY="${POLAR_NVIDIA_API_KEY:-${NVIDIA_API_KEY:-}}"
export POLAR_MODEL_POOL_BASE_URL="${POLAR_MODEL_POOL_BASE_URL:-${NVIDIA_BASE_URL:-https://integrate.api.nvidia.com/v1}}"

# Authenticate the trusted Slime -> rollout -> gateway control path separately
# from public task/session identifiers. The shared submitter copies POLAR_*
# values only into its mode-0600 allocation environment; this value is never
# rendered into YAML, run state, command lines, or agent runtimes.
if [ -z "${POLAR_CONTROL_PLANE_TOKEN:-}" ]; then
    command -v od >/dev/null || { echo "ERROR: od is required" >&2; exit 1; }
    command -v tr >/dev/null || { echo "ERROR: tr is required" >&2; exit 1; }
    POLAR_CONTROL_PLANE_TOKEN="$(od -An -N32 -tx1 /dev/urandom | tr -d '[:space:]')"
    export POLAR_CONTROL_PLANE_TOKEN
fi
if ! [[ "${POLAR_CONTROL_PLANE_TOKEN}" =~ ^[0-9A-Za-z_-]{32,128}$ ]]; then
    echo "ERROR: POLAR_CONTROL_PLANE_TOKEN must be a 32-128 character opaque token" >&2
    exit 1
fi

exec bash "${SCRIPT_DIR}/../tmax_slime_grpo/submit_slurm.sh" "$@"
