#!/usr/bin/env bash
# Run SPilot Router through the battle-tested TMax Slime-GRPO launcher.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${SCRIPT_DIR}/topology.yaml}"
export POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${SCRIPT_DIR}/polar_config.yaml}"

exec bash "${SCRIPT_DIR}/../tmax_slime_grpo/run.sh" "$@"
