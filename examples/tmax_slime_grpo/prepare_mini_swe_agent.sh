#!/usr/bin/env bash
# Build a self-contained mini-swe-agent runtime without modifying the training venv.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./env.cwdfw.sh
source "${SCRIPT_DIR}/env.cwdfw.sh"

UV_BIN="${UV_BIN:-${POLR_TRAIN_VENV}/bin/uv}"
if [ ! -x "${UV_BIN}" ]; then
    echo "ERROR: uv not executable: ${UV_BIN}" >&2
    exit 1
fi
if [ ! -x "${MINI_SWE_AGENT_PYTHON_ROOT}/bin/python3" ]; then
    echo "ERROR: portable Python not found: ${MINI_SWE_AGENT_PYTHON_ROOT}" >&2
    exit 1
fi

if [ "${FORCE_REBUILD:-0}" != "1" ] && [ -x "${MINI_SWE_AGENT_BIN}" ]; then
    "${MINI_SWE_AGENT_BIN}" --help >/dev/null
    echo "mini-swe-agent runtime already ready: ${MINI_SWE_AGENT_RUNTIME_DIR}"
    exit 0
fi

runtime_parent="$(dirname "${MINI_SWE_AGENT_RUNTIME_DIR}")"
mkdir -p "${runtime_parent}"
staging="$(mktemp -d "${runtime_parent}/.mini-swe-agent-runtime.XXXXXX")"
backup=""
runtime_activated=0
runtime_ready=0

cleanup() {
    local status=$?
    if [ -n "${staging}" ] && [ -d "${staging}" ]; then
        rm -rf "${staging}"
    fi
    if [ "${status}" -ne 0 ] && [ "${runtime_activated}" = "1" ] && [ "${runtime_ready}" != "1" ]; then
        rm -rf "${MINI_SWE_AGENT_RUNTIME_DIR}"
        if [ -n "${backup}" ] && [ -e "${backup}" ]; then
            mv "${backup}" "${MINI_SWE_AGENT_RUNTIME_DIR}"
        fi
    fi
    return "${status}"
}
trap cleanup EXIT

mkdir -p "${staging}/python"
cp -a "${MINI_SWE_AGENT_PYTHON_ROOT}/." "${staging}/python/"
if [ -e "${MINI_SWE_AGENT_RUNTIME_DIR}" ]; then
    backup="${MINI_SWE_AGENT_RUNTIME_DIR}.backup.$(date -u +%Y%m%dT%H%M%SZ)"
    mv "${MINI_SWE_AGENT_RUNTIME_DIR}" "${backup}"
fi
mv "${staging}" "${MINI_SWE_AGENT_RUNTIME_DIR}"
staging=""
runtime_activated=1

"${UV_BIN}" venv \
    --python "${MINI_SWE_AGENT_RUNTIME_DIR}/python/bin/python3" \
    "${MINI_SWE_AGENT_RUNTIME_DIR}/venv"
"${UV_BIN}" pip install \
    --python "${MINI_SWE_AGENT_RUNTIME_DIR}/venv/bin/python" \
    --link-mode copy \
    "${MINI_SWE_AGENT_SPEC}"

mkdir -p "${MINI_SWE_AGENT_RUNTIME_DIR}/bin"
cat >"${MINI_SWE_AGENT_BIN}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$(dirname "${script_dir}")"
site_packages=""
for candidate in "${runtime_dir}"/venv/lib/python*/site-packages; do
    if [ -d "${candidate}" ]; then
        site_packages="${candidate}"
        break
    fi
done
if [ -z "${site_packages}" ]; then
    echo "ERROR: mini-swe-agent site-packages not found under ${runtime_dir}/venv" >&2
    exit 1
fi
export PYTHONPATH="${site_packages}"
exec "${runtime_dir}/python/bin/python3" -m minisweagent.run.mini "$@"
SH
chmod 755 "${MINI_SWE_AGENT_BIN}"

if ! "${MINI_SWE_AGENT_BIN}" --help >/dev/null; then
    echo "ERROR: installed mini-swe-agent runtime failed validation" >&2
    exit 1
fi
runtime_ready=1
if [ -n "${backup}" ]; then
    rm -rf "${backup}"
fi

echo "mini-swe-agent runtime ready: ${MINI_SWE_AGENT_RUNTIME_DIR}"
