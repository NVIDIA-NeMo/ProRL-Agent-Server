"""Long-lived command broker for nested Apptainer direct-exec runtimes.

The module intentionally uses only the Python standard library.  It is copied
into the per-session bind mount and executed by a portable Python interpreter
inside one long-lived ``apptainer exec``.  Subsequent commands arrive over a
filesystem Unix socket, so agent actions and the verifier share the same PID,
IPC, and network namespaces.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from typing import Any, Mapping


_PROXY_SOCKET_ENV = "POLAR_HTTP_PROXY_UDS"
_PROXY_PORT_ENV = "POLAR_HTTP_PROXY_PORT"
_PROXY_READY_ENV = "POLAR_HTTP_PROXY_BROKER_READY"
_ALLOW_INTERNET_ENV = "POLAR_ALLOW_INTERNET"
_DEFAULT_PROXY_PORT = 28100
_PROXY_ENV_NAMES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "ftp_proxy",
    "FTP_PROXY",
)
_NO_PROXY_ENV_NAMES = ("no_proxy", "NO_PROXY")
_LOOPBACK_NO_PROXY = ("localhost", "127.0.0.1", "::1")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_BROKER_PROCESS_NAME = "polar-broker"
_PROXY_PROCESS_NAME = "polar-proxy"
_PR_SET_NAME = 15


def _current_python_executable() -> str:
    """Return the running interpreter even when ``exec -a`` hid argv[0]."""

    if sys.executable:
        return sys.executable
    if sys.platform.startswith("linux") and Path("/proc/self/exe").exists():
        # ``exec -a`` intentionally gives the broker a non-Python argv[0],
        # which makes CPython leave ``sys.executable`` empty.  The procfs link
        # still names the actual interpreter and is not exposed in the child
        # command line matched by broad agent cleanup commands.
        return "/proc/self/exe"
    raise RuntimeError("unable to locate the running Python interpreter")


def _set_process_name(name: str) -> None:
    """Set Linux ``comm`` without adding a non-stdlib dependency.

    The supervisor supplies a non-Python argv0 with ``exec -a``.  Linux keeps
    ``comm`` separately, however, and ``pkill python`` matches that field.  A
    best-effort no-op on non-Linux platforms keeps the standalone broker easy
    to unit test while Apptainer production (Linux) fails loudly if prctl is
    unexpectedly unavailable.
    """

    if not sys.platform.startswith("linux"):
        return
    encoded = name.encode("ascii")
    if not encoded or len(encoded) > 15:
        raise ValueError("Linux process names must contain 1..15 ASCII bytes")
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.restype = ctypes.c_int
    result = prctl(_PR_SET_NAME, ctypes.c_char_p(encoded), 0, 0, 0)
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def merge_loopback_no_proxy(*values: str | None) -> str:
    """Return the union of existing NO_PROXY entries and loopback hosts."""

    entries: list[str] = []
    seen: set[str] = set()
    for value in (*values, *_LOOPBACK_NO_PROXY):
        if value is None:
            continue
        for raw_entry in value.split(","):
            entry = raw_entry.strip()
            normalized = entry.lower()
            if entry and normalized not in seen:
                entries.append(entry)
                seen.add(normalized)
    return ",".join(entries)


def _relay_bidirectional(left: socket.socket, right: socket.socket) -> None:
    peers = {left: right, right: left}
    while peers:
        readable, _, _ = select.select(list(peers), [], [])
        for source in readable:
            destination = peers.get(source)
            if destination is None:
                continue
            try:
                chunk = source.recv(64 * 1024)
            except (ConnectionError, OSError):
                peers.clear()
                break
            if not chunk:
                peers.clear()
                break
            try:
                destination.sendall(chunk)
            except (ConnectionError, OSError):
                peers.clear()
                break


class _UnixForwardHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            upstream.connect(str(getattr(self.server, "unix_path")))
            _relay_bidirectional(self.request, upstream)
        finally:
            upstream.close()


class _LoopbackProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 256

    def __init__(self, port: int, unix_path: str) -> None:
        self.unix_path = unix_path
        super().__init__(("127.0.0.1", port), _UnixForwardHandler)


class _LoopbackProxy:
    """Supervise the TCP-to-UDS proxy outside the control broker process.

    Agents often inspect a conflicting localhost port with ``ss -p`` and kill
    the reported PID.  Running the listener as a thread made that PID the
    command broker itself, so an innocent port cleanup destroyed the complete
    runtime.  A separate, monitored process makes the listener disposable.
    """

    def __init__(self, unix_path: str, port: int, *, pid_path: Path | None = None) -> None:
        self.unix_path = unix_path
        self.port = port
        self.pid_path = pid_path
        self.process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._monitor: threading.Thread | None = None

    def _start_process_locked(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            return
        if self.pid_path is not None:
            self.pid_path.unlink(missing_ok=True)
        process = subprocess.Popen(
            [
                _PROXY_PROCESS_NAME,
                "-I",
                str(Path(__file__)),
                "--forward-proxy-socket",
                self.unix_path,
                "--forward-proxy-port",
                str(self.port),
            ],
            executable=_current_python_executable(),
        )
        self.process = process
        if self.pid_path is not None:
            self.pid_path.write_text(f"{process.pid}\n")
        # Binding happens before serve_forever.  Give immediate bind/import
        # failures a chance to surface without probing (and consuming) a real
        # upstream proxy connection.
        time.sleep(0.05)
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"loopback proxy exited during startup with code {return_code}"
            )

    def start(self) -> None:
        with self._lock:
            self._start_process_locked()
            if self._monitor is None:
                self._monitor = threading.Thread(
                    target=self._monitor_process,
                    name="polar-apptainer-http-proxy-monitor",
                    daemon=True,
                )
                self._monitor.start()

    def _monitor_process(self) -> None:
        while not self._stopping.wait(0.1):
            with self._lock:
                process = self.process
                if process is None or process.poll() is not None:
                    try:
                        self._start_process_locked()
                    except Exception as exc:
                        print(
                            f"polar apptainer broker: proxy restart failed: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )

    def ensure_running(self) -> None:
        with self._lock:
            self._start_process_locked()

    def close(self) -> None:
        self._stopping.set()
        with self._lock:
            process = self.process
            self.process = None
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if self.pid_path is not None:
                self.pid_path.unlink(missing_ok=True)
        if self._monitor is not None:
            self._monitor.join(timeout=5.0)


def _proxy_port(environment: Mapping[str, str]) -> int:
    raw = environment.get(_PROXY_PORT_ENV, str(_DEFAULT_PROXY_PORT))
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_PROXY_PORT_ENV} must be an integer, got {raw!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{_PROXY_PORT_ENV} must be between 1 and 65535, got {port}")
    return port


def configure_proxy_environment(
    environment: dict[str, str],
    *,
    proxy_pid_path: Path | None = None,
) -> tuple[_LoopbackProxy | None, str | None, bool]:
    """Start the namespace-local proxy and make its policy authoritative."""

    allow_internet = _truthy(environment.get(_ALLOW_INTERNET_ENV))
    if not allow_internet:
        for name in (*_PROXY_ENV_NAMES, *_NO_PROXY_ENV_NAMES):
            environment.pop(name, None)
        environment.pop(_PROXY_SOCKET_ENV, None)
        environment.pop(_PROXY_PORT_ENV, None)
        environment.pop(_PROXY_READY_ENV, None)
        return None, None, False

    no_proxy = merge_loopback_no_proxy(
        environment.get("no_proxy"), environment.get("NO_PROXY")
    )
    environment["no_proxy"] = no_proxy
    environment["NO_PROXY"] = no_proxy

    socket_path = environment.pop(_PROXY_SOCKET_ENV, "").strip()
    if not socket_path:
        return None, None, True

    proxy = _LoopbackProxy(
        socket_path,
        _proxy_port(environment),
        pid_path=proxy_pid_path,
    )
    proxy.start()
    port = int(proxy.port)
    proxy_url = f"http://127.0.0.1:{port}"
    for name in _PROXY_ENV_NAMES:
        environment[name] = proxy_url
    environment[_PROXY_READY_ENV] = "1"
    return proxy, proxy_url, True


def child_environment(
    base_environment: Mapping[str, str],
    requested_environment: Mapping[str, object],
    *,
    proxy_url: str | None,
    allow_internet: bool,
) -> dict[str, str]:
    """Merge a command environment without allowing it to bypass policy."""

    environment = dict(base_environment)
    environment.update({str(key): str(value) for key, value in requested_environment.items()})
    environment[_ALLOW_INTERNET_ENV] = "true" if allow_internet else "false"
    environment.pop(_PROXY_SOCKET_ENV, None)

    if not allow_internet:
        environment.pop(_PROXY_PORT_ENV, None)
        environment.pop(_PROXY_READY_ENV, None)
        for name in (*_PROXY_ENV_NAMES, *_NO_PROXY_ENV_NAMES):
            environment.pop(name, None)
        return environment

    no_proxy = merge_loopback_no_proxy(
        environment.get("no_proxy"), environment.get("NO_PROXY")
    )
    environment["no_proxy"] = no_proxy
    environment["NO_PROXY"] = no_proxy
    if proxy_url is not None:
        for name in _PROXY_ENV_NAMES:
            environment[name] = proxy_url
        environment[_PROXY_READY_ENV] = "1"
    return environment


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class _BrokerServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(
        self,
        socket_path: str,
        result_dir: Path,
        *,
        base_environment: Mapping[str, str],
        proxy: _LoopbackProxy | None,
        proxy_url: str | None,
        allow_internet: bool,
    ) -> None:
        self.result_dir = result_dir
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.base_environment = dict(base_environment)
        self.proxy = proxy
        self.proxy_url = proxy_url
        self.allow_internet = allow_internet
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._cancelled_before_start: set[str] = set()
        self._process_lock = threading.Lock()
        self.active_dir = result_dir.parent / "active"
        self.active_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(socket_path, _BrokerHandler)

    def _result_path(self, request_id: str, suffix: str) -> Path:
        if not _REQUEST_ID_RE.fullmatch(request_id):
            raise ValueError(f"invalid request id: {request_id!r}")
        return self.result_dir / f"{request_id}.{suffix}"

    def execute(self, request: Mapping[str, Any]) -> dict[str, object]:
        request_id = str(request.get("id", ""))
        command = request.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("exec request command must be a non-empty string")
        stdout_path = self._result_path(request_id, "stdout")
        stderr_path = self._result_path(request_id, "stderr")
        raw_env = request.get("env", {})
        if not isinstance(raw_env, dict):
            raise ValueError("exec request env must be an object")
        cwd = request.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ValueError("exec request cwd must be a string or null")
        raw_timeout = request.get("timeout_sec")
        timeout = None if raw_timeout is None else float(raw_timeout)
        if timeout is not None and timeout <= 0:
            raise ValueError("exec request timeout_sec must be positive")

        environment = child_environment(
            self.base_environment,
            raw_env,
            proxy_url=self.proxy_url,
            allow_internet=self.allow_internet,
        )
        process: subprocess.Popen[bytes] | None = None
        active_marker = self.active_dir / f"{request_id}.pid"
        return_code = 127
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            try:
                if self.proxy is not None:
                    self.proxy.ensure_running()
                process = subprocess.Popen(
                    ["bash", "-lc", command],
                    cwd=cwd,
                    env=environment,
                    start_new_session=True,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
                active_marker.write_text(f"{process.pid}\n")
                with self._process_lock:
                    self._processes[request_id] = process
                    cancelled = request_id in self._cancelled_before_start
                    self._cancelled_before_start.discard(request_id)
                if cancelled:
                    _kill_process_group(process)
                try:
                    return_code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    _kill_process_group(process)
                    process.wait()
                    return_code = -1
            except Exception as exc:  # report spawn/cwd failures like a shell command
                stderr_file.write(f"polar apptainer broker: {exc}\n".encode(errors="replace"))
                return_code = 127
            finally:
                if process is not None:
                    with self._process_lock:
                        self._processes.pop(request_id, None)
                active_marker.unlink(missing_ok=True)
        return {"return_code": return_code}

    def cancel_command(self, request_id: str) -> bool:
        if not _REQUEST_ID_RE.fullmatch(request_id):
            raise ValueError(f"invalid request id: {request_id!r}")
        with self._process_lock:
            process = self._processes.get(request_id)
            if process is None:
                self._cancelled_before_start.add(request_id)
                return False
        _kill_process_group(process)
        return True

    def kill_active_commands(self) -> None:
        with self._process_lock:
            processes = tuple(self._processes.values())
        for process in processes:
            _kill_process_group(process)


class _BrokerHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        response: dict[str, object]
        shutdown = False
        try:
            raw_request = self.rfile.readline()
            if not raw_request:
                return
            request = json.loads(raw_request)
            if not isinstance(request, dict):
                raise ValueError("broker request must be a JSON object")
            operation = request.get("operation")
            server = self.server
            assert isinstance(server, _BrokerServer)
            if operation == "ping":
                response = {"ok": True}
            elif operation == "exec":
                response = {"ok": True, **server.execute(request)}
            elif operation == "cancel":
                response = {
                    "ok": True,
                    "cancelled": server.cancel_command(str(request.get("id", ""))),
                }
            elif operation == "shutdown":
                server.kill_active_commands()
                response = {"ok": True}
                shutdown = True
            else:
                raise ValueError(f"unknown broker operation: {operation!r}")
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        self.wfile.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")
        self.wfile.flush()
        if shutdown:
            # socketserver.shutdown must be called from a thread other than the
            # serve_forever thread.  A handler already runs in its own thread,
            # but another daemon keeps response delivery independent of it.
            threading.Thread(target=self.server.shutdown, daemon=True).start()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket")
    parser.add_argument("--result-dir")
    parser.add_argument("--forward-proxy-socket")
    parser.add_argument("--forward-proxy-port", type=int)
    args = parser.parse_args()
    if args.forward_proxy_socket is not None:
        if args.forward_proxy_port is None:
            parser.error("--forward-proxy-port is required in proxy mode")
    elif args.socket is None or args.result_dir is None:
        parser.error("--socket and --result-dir are required in broker mode")
    return args


def _run_forward_proxy(unix_path: str, port: int) -> int:
    server = _LoopbackProxyServer(port, unix_path)

    def request_shutdown(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
    return 0


def main() -> int:
    args = _parse_args()
    if args.forward_proxy_socket is not None:
        _set_process_name(_PROXY_PROCESS_NAME)
        assert args.forward_proxy_port is not None
        return _run_forward_proxy(args.forward_proxy_socket, args.forward_proxy_port)

    _set_process_name(_BROKER_PROCESS_NAME)
    assert args.socket is not None
    assert args.result_dir is not None
    socket_path = Path(args.socket)
    result_dir = Path(args.result_dir)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)

    environment = dict(os.environ)
    proxy, proxy_url, allow_internet = configure_proxy_environment(
        environment,
        proxy_pid_path=socket_path.parent / "proxy.pid",
    )
    server = _BrokerServer(
        str(socket_path),
        result_dir,
        base_environment=environment,
        proxy=proxy,
        proxy_url=proxy_url,
        allow_internet=allow_internet,
    )
    socket_path.chmod(0o600)

    def request_shutdown(_signum: int, _frame: object) -> None:
        server.kill_active_commands()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        socket_path.unlink(missing_ok=True)
        if proxy is not None:
            proxy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
