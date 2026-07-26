#!/usr/bin/env bash
# Rebuild the Qwen3.5 Megatron torch_dist checkpoint from HF weights.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ ! -f "${SCRIPT_DIR}/env.cwdfw.sh" ] && [ -f "${PWD}/examples/swegym_slime_grpo/env.cwdfw.sh" ]; then
    SCRIPT_DIR="${PWD}/examples/swegym_slime_grpo"
fi
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=./env.cwdfw.sh
source "${SCRIPT_DIR}/env.cwdfw.sh"
# shellcheck source=../path_safety.sh
source "${SCRIPT_DIR}/../path_safety.sh"

export PYTHON_BIN="${PYTHON_BIN:-${POLR_TRAIN_VENV}/bin/python3}"
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-${POLR_TRAIN_VENV}}"
export PYTHONNOUSERSITE=1

export POLAR_JOB_CACHE_ROOT="${POLAR_JOB_CACHE_ROOT:-/tmp/polar-convert-${SLURM_JOB_ID:-manual}}"
polar_safe_remove_tree POLAR_JOB_CACHE_ROOT "${POLAR_JOB_CACHE_ROOT}" /tmp polar-
mkdir -p \
    "${POLAR_JOB_CACHE_ROOT}/home" \
    "${POLAR_JOB_CACHE_ROOT}/triton" \
    "${POLAR_JOB_CACHE_ROOT}/torchinductor" \
    "${POLAR_JOB_CACHE_ROOT}/xdg" \
    "${POLAR_JOB_CACHE_ROOT}/xdg-config" \
    "${POLAR_JOB_CACHE_ROOT}/xdg-runtime"
chmod 700 "${POLAR_JOB_CACHE_ROOT}/xdg-runtime"

export HOME="${POLAR_JOB_CACHE_ROOT}/home"
export TRITON_CACHE_DIR="${POLAR_JOB_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${POLAR_JOB_CACHE_ROOT}/torchinductor"
export XDG_CACHE_HOME="${POLAR_JOB_CACHE_ROOT}/xdg"
export XDG_CONFIG_HOME="${POLAR_JOB_CACHE_ROOT}/xdg-config"
export XDG_RUNTIME_DIR="${POLAR_JOB_CACHE_ROOT}/xdg-runtime"
export SCRIPT_DIR APPTAINER_IMAGE_DIR

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import json
import os

base = Path(os.environ["SCRIPT_DIR"])
sif_dir = Path(os.environ["APPTAINER_IMAGE_DIR"])
ids = set()
for name in ("swegym_train_293.jsonl", "swegym_eval_23.jsonl"):
    with (base / name).open() as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["metadata"]["instance_id"])
missing = sorted(i for i in ids if not (sif_dir / f"{i}.sif").is_file())
print(f"[reconvert] SIF status: {len(ids) - len(missing)}/{len(ids)}")
if missing:
    print("[reconvert] Missing SIF sample:", ", ".join(missing[:20]))
    raise SystemExit(1)
PY

polar_safe_remove_tree \
    TORCH_DIST_DIR "${TORCH_DIST_DIR}" "${POLAR_DATA_ROOT}/checkpoints"

bash "${SCRIPT_DIR}/convert_weights.sh"

date -u +%Y-%m-%dT%H:%M:%SZ > "${TORCH_DIST_DIR}/.swegym_reconverted_ok"
