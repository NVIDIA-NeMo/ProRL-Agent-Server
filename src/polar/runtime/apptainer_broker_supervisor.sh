#!/usr/bin/env bash
# Keep the Apptainer namespace alive if an untrusted task kills Python
# processes.  The command broker and its proxy are infrastructure, but they
# share the task's PID namespace so broad process cleanup can reach them.

set -u

runtime_dir="${POLAR_APPTAINER_BROKER_RUNTIME_DIR:-/polar/session/.apptainer-broker}"
broker_script="${POLAR_APPTAINER_BROKER_SCRIPT:-${runtime_dir}/apptainer_broker.py}"
interpreter_file="${POLAR_APPTAINER_BROKER_INTERPRETER_FILE:-${runtime_dir}/interpreter.path}"
broker_process_name="${POLAR_APPTAINER_BROKER_PROCESS_NAME:-polar-broker}"
max_restarts="${POLAR_APPTAINER_BROKER_MAX_RESTARTS:-3}"
socket_path="${runtime_dir}/control.sock"
protected_socket_name="${POLAR_APPTAINER_PROTECTED_SOCKET_NAME:-}"
case "${protected_socket_name}" in
    p-[a-f0-9]*.sock) ;;
    *) printf 'broker supervisor: invalid protected socket name\n' >&2; exit 64 ;;
esac
protected_socket_path="${runtime_dir}/${protected_socket_name}"
result_dir="${runtime_dir}/results"
active_dir="${runtime_dir}/active"
broker_pid_file="${runtime_dir}/broker.pid"
proxy_pid_file="${runtime_dir}/proxy.pid"
generation_file="${runtime_dir}/broker.generation"
restart_count_file="${runtime_dir}/broker.restart_count"
broker_pid=""
generation=0
restart_count=0

case "${max_restarts}" in
    ''|*[!0-9]*) max_restarts=3 ;;
    *)
        if [ "${max_restarts}" -gt 10 ]; then
            max_restarts=3
        fi
        ;;
esac

broker_python=""
if [ -r "${interpreter_file}" ]; then
    IFS= read -r broker_python <"${interpreter_file}" || true
fi
if [ -z "${broker_python}" ]; then
    # Backward-compatible fallback for hand-run supervisors.  Production writes
    # the interpreter path into the session bind so it never appears in the
    # long-lived Apptainer/supervisor argv inspected by agent cleanup commands.
    broker_python="${POLAR_APPTAINER_BROKER_PYTHON:-python3}"
fi
broker_launcher="${runtime_dir}/${broker_process_name}"

mkdir -p "${result_dir}" "${active_dir}"
printf '0\n' >"${generation_file}"
printf '0\n' >"${restart_count_file}"

kill_pid_file() {
    local pid_file="$1"
    local as_group="${2:-0}"
    local pid=""
    if [ -r "${pid_file}" ]; then
        IFS= read -r pid <"${pid_file}" || true
    fi
    case "${pid}" in
        ''|*[!0-9]*) ;;
        *)
            if [ "${as_group}" = "1" ]; then
                kill -KILL -- "-${pid}" 2>/dev/null || true
            else
                kill -KILL "${pid}" 2>/dev/null || true
            fi
            ;;
    esac
    rm -f "${pid_file}"
}

cleanup_crashed_broker() {
    local marker=""
    # Each foreground RPC command starts a new process group.  If its broker
    # dies, terminate only that in-flight group; previously launched background
    # services remain available in the persistent namespace.
    for marker in "${active_dir}"/*.pid; do
        [ -e "${marker}" ] || continue
        kill_pid_file "${marker}" 1
    done
    kill_pid_file "${proxy_pid_file}" 0
    rm -f "${socket_path}"
    rm -f "${protected_socket_path}"
}

terminate_supervisor() {
    if [ -n "${broker_pid}" ]; then
        kill -TERM "${broker_pid}" 2>/dev/null || true
    fi
    cleanup_crashed_broker
    exit 143
}

trap terminate_supervisor TERM INT HUP

while :; do
    generation="$((generation + 1))"
    printf '%s\n' "${generation}" >"${generation_file}"
    # Launch through a basename-only symlink so CPython can still resolve its
    # standard library under ``-I`` while neither argv[0] nor PR_SET_NAME
    # matches broad task cleanup such as ``pkill -f python``. A protected
    # caller pins the first broker PID/starttime, so a later supervisor restart
    # is never trusted with secrets even if this task-visible symlink changes.
    ln -sfn -- "${broker_python}" "${broker_launcher}"
    (
        PATH="${runtime_dir}:${PATH}" exec "${broker_process_name}" -I "${broker_script}" \
            --socket "${socket_path}" \
            --protected-socket "${protected_socket_path}" \
            --result-dir "${result_dir}"
    ) &
    broker_pid="$!"
    printf '%s\n' "${broker_pid}" >"${broker_pid_file}"
    wait "${broker_pid}"
    status="$?"
    broker_pid=""
    rm -f "${broker_pid_file}"

    # A clean exit is the explicit shutdown RPC.  Signal/non-zero exits are
    # control-plane crashes, usually caused by an agent cleanup command.
    if [ "${status}" -eq 0 ]; then
        # Normal broker teardown removes these markers.  If a proxy resisted
        # TERM/KILL long enough for close() to return, its PID marker is kept
        # deliberately so the supervisor can make this final cleanup attempt.
        cleanup_crashed_broker
        exit 0
    fi
    restart_count="$((restart_count + 1))"
    printf '%s\n' "${restart_count}" >"${restart_count_file}"
    if [ "${restart_count}" -gt "${max_restarts}" ]; then
        printf 'broker supervisor: broker exited with status %s; restart budget exhausted (%s)\n' \
            "${status}" "${max_restarts}" >&2
        cleanup_crashed_broker
        exit "${status}"
    fi
    printf 'broker supervisor: broker exited with status %s; restarting (%s/%s)\n' \
        "${status}" "${restart_count}" "${max_restarts}" >&2
    cleanup_crashed_broker
    sleep 0.1
done
