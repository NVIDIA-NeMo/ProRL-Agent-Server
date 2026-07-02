#!/usr/bin/env bash
# Backward-compatible entry point; use submit_matrix.sh for new invocations.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Preserve the legacy entry point's original resource boundary even as the
# generic matrix gains settings with larger topologies.
export TMAX_MATRIX_TOPOLOGY_SCOPE=4n32
if [ "${TMAX_MATRIX_CONFIRM:-}" = "SUBMIT_4NODE_MATRIX" ]; then
    export TMAX_MATRIX_CONFIRM=SUBMIT_TMAX_MATRIX
fi
exec bash "${SCRIPT_DIR}/submit_matrix.sh" "$@"
