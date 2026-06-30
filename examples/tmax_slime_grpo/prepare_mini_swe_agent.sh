#!/usr/bin/env bash
# Build a self-contained mini-swe-agent runtime without modifying the training venv.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./env.cwdfw.sh
source "${SCRIPT_DIR}/env.cwdfw.sh"

UV_BIN="${UV_BIN:-${POLR_TRAIN_VENV}/bin/uv}"
TIMING_MODULE_SOURCE="${PROJECT_ROOT}/src/polar/agent/presets/mini_swe_timing.py"
RUNNER_MODULE_SOURCE="${PROJECT_ROOT}/src/polar/agent/presets/mini_swe_runner.py"
VANILLUX_MODULE_SOURCE="${PROJECT_ROOT}/src/polar/agent/presets/mini_swe_vanillux.py"
VANILLUX_CONFIG_SOURCE="${PROJECT_ROOT}/src/polar/agent/presets/vanillux2.yaml"
if [ ! -x "${UV_BIN}" ]; then
    echo "ERROR: uv not executable: ${UV_BIN}" >&2
    exit 1
fi
if [ ! -x "${MINI_SWE_AGENT_PYTHON_ROOT}/bin/python3" ]; then
    echo "ERROR: portable Python not found: ${MINI_SWE_AGENT_PYTHON_ROOT}" >&2
    exit 1
fi
if [ ! -f "${TIMING_MODULE_SOURCE}" ]; then
    echo "ERROR: mini-swe timing module not found: ${TIMING_MODULE_SOURCE}" >&2
    exit 1
fi
if [ ! -f "${RUNNER_MODULE_SOURCE}" ]; then
    echo "ERROR: mini-swe isolated runner module not found: ${RUNNER_MODULE_SOURCE}" >&2
    exit 1
fi
if [ ! -f "${VANILLUX_MODULE_SOURCE}" ]; then
    echo "ERROR: Vanillux2 model adapter not found: ${VANILLUX_MODULE_SOURCE}" >&2
    exit 1
fi
if [ ! -f "${VANILLUX_CONFIG_SOURCE}" ]; then
    echo "ERROR: Vanillux2 config not found: ${VANILLUX_CONFIG_SOURCE}" >&2
    exit 1
fi

runtime_layout_is_current() {
    local venv_python="${MINI_SWE_AGENT_RUNTIME_DIR}/venv/bin/python"
    local installed_timing_module installed_runner_module installed_vanillux_module
    local installed_vanillux_config="${MINI_SWE_AGENT_RUNTIME_DIR}/config/vanillux2.yaml"
    [ -x "${MINI_SWE_AGENT_BIN}" ] || return 1
    grep -q 'POLAR_TASK_PYTHONPATH' "${MINI_SWE_AGENT_BIN}" || return 1
    [ "$(readlink "${venv_python}" 2>/dev/null || true)" = "../../python/bin/python3" ] || return 1
    ! grep -Eq '^[[:space:]]*export[[:space:]]+PYTHONPATH=' "${MINI_SWE_AGENT_BIN}" || return 1
    "${venv_python}" -c \
        'import polar_mini_swe_timing as m; assert m.TIMING_SCHEMA_VERSION == 1' \
        >/dev/null 2>&1 || return 1
    installed_timing_module="$("${venv_python}" -c \
        'import polar_mini_swe_timing as m; print(m.__file__)' 2>/dev/null | tail -n 1)" || return 1
    [ -n "${installed_timing_module}" ] || return 1
    installed_runner_module="$("${venv_python}" -c \
        'import polar_mini_swe_runner as m; print(m.__file__)' 2>/dev/null | tail -n 1)" || return 1
    [ -n "${installed_runner_module}" ] || return 1
    installed_vanillux_module="$("${venv_python}" -c \
        'import polar_mini_swe_vanillux as m; print(m.__file__)' 2>/dev/null | tail -n 1)" || return 1
    [ -n "${installed_vanillux_module}" ] || return 1
    cmp -s "${TIMING_MODULE_SOURCE}" "${installed_timing_module}" && \
        cmp -s "${RUNNER_MODULE_SOURCE}" "${installed_runner_module}" && \
        cmp -s "${VANILLUX_MODULE_SOURCE}" "${installed_vanillux_module}" && \
        cmp -s "${VANILLUX_CONFIG_SOURCE}" "${installed_vanillux_config}"
}

if [ "${FORCE_REBUILD:-0}" != "1" ] && runtime_layout_is_current; then
    "${MINI_SWE_AGENT_BIN}" --help >/dev/null
    echo "mini-swe-agent runtime already ready: ${MINI_SWE_AGENT_RUNTIME_DIR}"
    exit 0
fi
if [ "${FORCE_REBUILD:-0}" != "1" ] && [ -e "${MINI_SWE_AGENT_RUNTIME_DIR}" ]; then
    echo "rebuilding legacy mini-swe-agent runtime layout: ${MINI_SWE_AGENT_RUNTIME_DIR}"
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
# Build and validate entirely in the sibling staging directory. Direct
# Apptainer exec resolves the bind source on every command, so publishing a
# half-built target here would break sessions in an active allocation.
"${UV_BIN}" venv \
    --relocatable \
    --python "${staging}/python/bin/python3" \
    "${staging}/venv"
"${UV_BIN}" pip install \
    --python "${staging}/venv/bin/python" \
    --link-mode copy \
    "${MINI_SWE_AGENT_SPEC}"
# uv records that the venv is relocatable, but its interpreter symlink still
# points at the build-time absolute path.  Keep the copied interpreter and venv
# as siblings so the complete runtime can be mounted at a different path.
ln -sfn ../../python/bin/python3 "${staging}/venv/bin/python"
site_packages="$("${staging}/venv/bin/python" -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
install -m 0644 "${TIMING_MODULE_SOURCE}" "${site_packages}/polar_mini_swe_timing.py"
install -m 0644 "${RUNNER_MODULE_SOURCE}" "${site_packages}/polar_mini_swe_runner.py"
install -m 0644 "${VANILLUX_MODULE_SOURCE}" "${site_packages}/polar_mini_swe_vanillux.py"
mkdir -p "${staging}/config"
install -m 0644 "${VANILLUX_CONFIG_SOURCE}" "${staging}/config/vanillux2.yaml"

mkdir -p "${staging}/bin"
staging_bin="${staging}/bin/mini-swe-agent"
cat >"${staging_bin}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="$(dirname "${script_dir}")"
venv_python="${runtime_dir}/venv/bin/python"
if [ ! -x "${venv_python}" ]; then
    echo "ERROR: mini-swe-agent interpreter not found under ${runtime_dir}/venv" >&2
    exit 1
fi
# The portable interpreter must not import task modules while bootstrapping the
# runner. Preserve an image-defined PYTHONPATH under a private marker, clear it
# for interpreter startup, then polar_mini_swe_runner restores it only for the
# agent's action subprocesses.
if [ "${PYTHONPATH+x}" = x ]; then
    export POLAR_TASK_PYTHONPATH="${PYTHONPATH}"
fi
unset PYTHONPATH
exec "${venv_python}" -m polar_mini_swe_runner "$@"
SH
chmod 755 "${staging_bin}"

if ! "${staging_bin}" --help >/dev/null; then
    echo "ERROR: installed mini-swe-agent runtime failed validation" >&2
    exit 1
fi
if ! "${staging}/venv/bin/python" -c \
    'import polar_mini_swe_runner, polar_mini_swe_vanillux, polar_mini_swe_timing as m; assert m.TIMING_SCHEMA_VERSION == 1' || \
   [ ! -f "${staging}/config/vanillux2.yaml" ]; then
    echo "ERROR: mini-swe-agent injected environment failed validation" >&2
    exit 1
fi

# The only visibility gap is the two same-filesystem renames below. If final
# validation somehow fails, the EXIT trap restores the previous complete tree.
if [ -e "${MINI_SWE_AGENT_RUNTIME_DIR}" ]; then
    backup="${MINI_SWE_AGENT_RUNTIME_DIR}.backup.$(date -u +%Y%m%dT%H%M%SZ)"
    mv "${MINI_SWE_AGENT_RUNTIME_DIR}" "${backup}"
fi
mv "${staging}" "${MINI_SWE_AGENT_RUNTIME_DIR}"
staging=""
runtime_activated=1
if ! "${MINI_SWE_AGENT_BIN}" --help >/dev/null || \
   ! "${MINI_SWE_AGENT_RUNTIME_DIR}/venv/bin/python" -c \
     'import polar_mini_swe_runner, polar_mini_swe_vanillux, polar_mini_swe_timing as m; assert m.TIMING_SCHEMA_VERSION == 1' || \
   [ ! -f "${MINI_SWE_AGENT_RUNTIME_DIR}/config/vanillux2.yaml" ]; then
    echo "ERROR: published mini-swe-agent runtime failed final validation" >&2
    exit 1
fi
runtime_ready=1
if [ -n "${backup}" ]; then
    rm -rf "${backup}"
fi

echo "mini-swe-agent runtime ready: ${MINI_SWE_AGENT_RUNTIME_DIR}"
