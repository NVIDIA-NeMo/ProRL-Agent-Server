#!/usr/bin/env bash
# Backward-compatible entry point; use watch_matrix.sh for new invocations.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "${SCRIPT_DIR}/watch_matrix.sh" "$@"
