#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "${SCRIPT_DIR}/watch_training.sh" --relaunch --loop --sleep-seconds "${TMAX_WATCH_SLEEP_SECONDS:-600}"
