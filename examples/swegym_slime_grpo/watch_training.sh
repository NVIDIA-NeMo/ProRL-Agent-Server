#!/usr/bin/env bash
# Periodically relaunch the 4-node SWE-Gym GRPO training job after assets exist.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

RELAUNCH=false
LOOP=false
SLEEP_SECONDS="${SWEGYM_WATCH_SLEEP_SECONDS:-600}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --relaunch) RELAUNCH=true ;;
        --loop) LOOP=true ;;
        --sleep-seconds) shift; SLEEP_SECONDS="${1:?missing seconds}" ;;
        *) echo "Usage: $0 [--relaunch] [--loop] [--sleep-seconds N]" >&2; exit 2 ;;
    esac
    shift
done

# shellcheck source=./env.cwdfw.sh
source "${SCRIPT_DIR}/env.cwdfw.sh" >/dev/null

count_sifs() {
    python3 - <<'PY'
from pathlib import Path
import json
import os
base=Path(os.environ["SCRIPT_DIR"])
ids=set()
for name in ["swegym_train_293.jsonl", "swegym_eval_23.jsonl"]:
    with (base/name).open() as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["metadata"]["instance_id"])
sif=Path(os.environ["APPTAINER_IMAGE_DIR"])
missing=sorted(i for i in ids if not (sif/f"{i}.sif").exists())
print(len(ids)-len(missing), len(ids), len(missing))
if missing:
    print(" ".join(missing[:20]))
PY
}

latest_iter() {
    local f="${SAVE_DIR}/latest_checkpointed_iteration.txt"
    if [ -f "$f" ]; then
        tr -dc '0-9' < "$f"
    else
        printf '0'
    fi
}

check_once() {
    export SCRIPT_DIR APPTAINER_IMAGE_DIR
    read -r have total missing < <(count_sifs | head -n 1)
    printf '[watch] sif_status=%s/%s missing=%s\n' "$have" "$total" "$missing"
    if [ "$missing" != "0" ]; then
        count_sifs | sed -n '2p' | sed 's/^/[watch] missing_sample=/'
        return 0
    fi

    local marker="${REF_LOAD}/.swegym_reconverted_ok"
    if [ ! -f "$marker" ]; then
        printf '[watch] waiting for reconverted checkpoint marker: %s\n' "$marker"
        return 0
    fi

    local target="${SWEGYM_TARGET_ITER:-}"
    local iter
    iter="$(latest_iter)"
    if [ -n "$target" ] && [ "$iter" -ge "$target" ]; then
        printf '[watch] target reached: latest_iter=%s target=%s\n' "$iter" "$target"
        return 0
    fi

    local active
    active="$(squeue -h -u "${USER}" -n "${JOB_NAME:-polar-swegym-grpo}" -t PD,R,CF,CG -o '%i %t %j %R' || true)"
    if [ -n "$active" ]; then
        printf '[watch] training job already active:\n%s\n' "$active"
        return 0
    fi

    printf '[watch] no active training job; latest_iter=%s save_dir=%s\n' "$iter" "$SAVE_DIR"
    if [ "$RELAUNCH" = true ]; then
        SUBMIT_BACKEND="${SUBMIT_BACKEND:-srun}" bash "${SCRIPT_DIR}/submit_slurm.sh"
    fi
}

if [ "$LOOP" = true ]; then
    mkdir -p "${POLAR_DATA_ROOT}/runs/${RUN_ID}"
    exec 9>"${POLAR_DATA_ROOT}/runs/${RUN_ID}/watch_training.lock"
    if ! flock -n 9; then
        echo "[watch] another watcher is already running for ${RUN_ID}"
        exit 0
    fi
    while true; do
        date -u '+[watch] %Y-%m-%dT%H:%M:%SZ'
        check_once
        sleep "$SLEEP_SECONDS"
    done
else
    check_once
fi
