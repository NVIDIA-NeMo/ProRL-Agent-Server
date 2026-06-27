#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=./env.cwdfw.sh
source "${SCRIPT_DIR}/env.cwdfw.sh"

export AGENT_CLI_DIR
exec bash "${PROJECT_ROOT}/examples/swegym_slime_grpo/prepare_agent_cli.sh"
