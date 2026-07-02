#!/usr/bin/env bash
# Snapshot the existing training venv into a Pyxis/Enroot sqsh image.
# This copies the environment exactly; it does not install or resolve packages.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
USER_ROOT="$(dirname "${SPILOT_ROOT}")"

ACCOUNT="${ACCOUNT:-nvr_lpr_llm}"
PARTITION="${PARTITION:-interactive}"
WALL_TIME="${WALL_TIME:-1:00:00}"
GPUS="${GPUS:-1}"
BASE_SQSH="${BASE_SQSH:-flappydora/ubuntu22.04-cuda13.3:latest}"
OUT_SQSH="${POLR_TRAIN_SQSH:-${SPILOT_ROOT}/container/polar_train.sqsh}"
HOST_VENV="${HOST_VENV:-${USER_ROOT}/.python/polar}"
HOST_PYTHON="$(readlink -f "${HOST_VENV}/bin/python")"
HOST_CPYTHON="${HOST_CPYTHON:-$(cd -- "$(dirname -- "${HOST_PYTHON}")/.." && pwd)}"

if [ ! -x "${HOST_VENV}/bin/python" ]; then
    echo "ERROR: source venv not found: ${HOST_VENV}" >&2
    exit 1
fi
if [ ! -x "${HOST_CPYTHON}/bin/python3" ] && [ ! -x "${HOST_CPYTHON}/bin/python3.12" ]; then
    echo "ERROR: source CPython not found: ${HOST_CPYTHON}" >&2
    exit 1
fi
case "${BASE_SQSH}" in
    /*|./*|../*)
        [ -f "${BASE_SQSH}" ] || { echo "ERROR: base sqsh not found: ${BASE_SQSH}" >&2; exit 1; }
        ;;
esac

mkdir -p "$(dirname "${OUT_SQSH}")"
rm -f "${OUT_SQSH}"

BUILD_CMD="$(cat <<'EOF'
set -euo pipefail
rm -rf /opt/polar_venv /opt/polar_cpython
cp -a /mnt/polar_venv /opt/polar_venv
cp -a /mnt/polar_cpython /opt/polar_cpython

PYTHON_NAME="$(basename "$(readlink -f /opt/polar_cpython/bin/python3 2>/dev/null || find /opt/polar_cpython/bin -maxdepth 1 -name 'python3.*' -type f | head -1)")"
rm -f /opt/polar_venv/bin/python /opt/polar_venv/bin/python3 /opt/polar_venv/bin/python3.12
ln -s "/opt/polar_cpython/bin/${PYTHON_NAME}" /opt/polar_venv/bin/python
ln -s python /opt/polar_venv/bin/python3
ln -s python /opt/polar_venv/bin/python3.12

PYTHON_VERSION="$(/opt/polar_venv/bin/python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
cat >/opt/polar_venv/pyvenv.cfg <<CFG
home = /opt/polar_cpython/bin
implementation = CPython
version_info = ${PYTHON_VERSION}
include-system-site-packages = false
CFG

/opt/polar_venv/bin/python - <<'PY'
from pathlib import Path

for path in (Path("/opt/polar_venv") / "bin").iterdir():
    if not path.is_file() or path.is_symlink():
        continue
    try:
        data = path.read_bytes()
        first, separator, rest = data.partition(b"\n")
    except OSError:
        continue
    if first.startswith(b"#!") and b"/bin/python" in first:
        path.write_bytes(b"#!/opt/polar_venv/bin/python" + separator + rest)
PY

/opt/polar_venv/bin/python - <<'PY'
import polar
import ray
import sglang
import slime
import slime_bridge
import torch

print("python environment copied without dependency resolution")
print("torch", torch.__version__, torch.version.cuda)
print("ray", ray.__version__)
print("sglang", getattr(sglang, "__version__", "unknown"))
print("polar", polar.__file__)
print("slime", slime.__file__)
PY
EOF
)"

srun \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --gres="gpu:${GPUS}" \
    --time="${WALL_TIME}" \
    --mem=0 \
    --container-image="${BASE_SQSH}" \
    --container-save="${OUT_SQSH}" \
    --container-writable \
    --no-container-mount-home \
    --container-mounts="${HOST_VENV}:/mnt/polar_venv:ro,${HOST_CPYTHON}:/mnt/polar_cpython:ro" \
    --container-workdir=/ \
    bash -lc "${BUILD_CMD}"

echo "Built training sqsh: ${OUT_SQSH}"
