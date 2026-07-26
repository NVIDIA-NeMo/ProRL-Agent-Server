#!/usr/bin/env bash
# Backward-compatible entry point; use watch_matrix.sh for new invocations.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Preserve the legacy entry point's original resource boundary even when its
# stamp also contains runs from the generic matrix's 8-node settings.
export TMAX_MATRIX_TOPOLOGY_SCOPE=4n32
exec bash "${SCRIPT_DIR}/watch_matrix.sh" "$@"
