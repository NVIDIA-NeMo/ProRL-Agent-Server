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
from dataclasses import dataclass
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import socketserver
import stat
import struct
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
_PR_SET_DUMPABLE = 4
_PROTECTED_READY_FD_ENV = "POLAR_PROTECTED_EXEC_READY_FD"
_PROTECTED_FD_SUFFIX = "_FD"
_PROTECTED_SOCKET_ENV = "POLAR_PROTECTED_EXEC_SOCKET"
_PROTECTED_BROKER_PID_ENV = "POLAR_PROTECTED_EXEC_BROKER_PID"
_PROTECTED_REQUEST_ID_ENV = "POLAR_PROTECTED_EXEC_REQUEST_ID"
_PROTECTED_READY_TIMEOUT_SECONDS = 30.0
_PROTECTED_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_SHA256_RE = re.compile(r"^[A-Fa-f0-9]{64}$")
_PROTECTED_PYTHON = "/opt/polar-mini-swe-agent/venv/bin/python"
_SPILOT_RUNNER = "/polar/session/spilot_router_runner.py"
_SPILOT_FORCED_RUNNER = "/polar/session/spilot_forced_route_eval_runner.py"
_PROTECTED_FILE_MAX_BYTES = 8 * 1024 * 1024
_MFD_CLOEXEC = 0x0001
_MFD_ALLOW_SEALING = 0x0002
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_F_SEAL_SEAL = 0x0001
_F_SEAL_SHRINK = 0x0002
_F_SEAL_GROW = 0x0004
_F_SEAL_WRITE = 0x0008
_ALL_FILE_SEALS = _F_SEAL_SEAL | _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_WRITE
_FORCED_RUNNER_BOOTSTRAP = (
    "import importlib.machinery,importlib.util,runpy,sys;"
    "p='/proc/self/fd/'+sys.argv[2];"
    "l=importlib.machinery.SourceFileLoader('spilot_router_runner',p);"
    "s=importlib.util.spec_from_loader(l.name,l);"
    "m=importlib.util.module_from_spec(s);"
    "sys.modules[l.name]=m;"
    "l.exec_module(m);"
    "runpy.run_path('/proc/self/fd/'+sys.argv[3],run_name='__main__')"
)
_PROTECTED_ENV_DENYLIST = frozenset(
    {
        "BASH_ENV",
        "ENV",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
    }
)
_PROXY_START_TIMEOUT_SECONDS = 5.0
_PROXY_START_POLL_SECONDS = 0.01


def _memfd_create(name: str) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "memfd_create", None)
    if function is None:
        raise RuntimeError("protected exec requires Linux memfd_create")
    function.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    descriptor = function(name.encode("ascii"), _MFD_CLOEXEC | _MFD_ALLOW_SEALING)
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return descriptor


def _sealed_verified_file(path: str, expected_sha256: str) -> int:
    """Copy one allowlisted file into an immutable, digest-pinned memfd."""

    if not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("protected file digest is invalid")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise RuntimeError("protected exec requires O_NOFOLLOW")
    source = os.open(path, os.O_RDONLY | os.O_CLOEXEC | nofollow)
    try:
        metadata = os.fstat(source)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("protected runner must be a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(source, 128 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _PROTECTED_FILE_MAX_BYTES:
                raise ValueError("protected runner exceeds the size limit")
            chunks.append(chunk)
    finally:
        os.close(source)
    payload = b"".join(chunks)
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, expected_sha256.lower()):
        raise PermissionError("protected runner digest mismatch")

    sealed = _memfd_create("polar-protected-runner")
    try:
        view = memoryview(payload)
        while view:
            written = os.write(sealed, view)
            view = view[written:]
        os.lseek(sealed, 0, os.SEEK_SET)
        fcntl.fcntl(sealed, _F_ADD_SEALS, _ALL_FILE_SEALS)
        if fcntl.fcntl(sealed, _F_GET_SEALS) & _ALL_FILE_SEALS != _ALL_FILE_SEALS:
            raise RuntimeError("protected runner memfd sealing failed")
        return sealed
    except BaseException:
        os.close(sealed)
        raise


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


def _set_process_dumpable(enabled: bool) -> None:
    """Prevent same-UID task processes from inspecting broker memory."""

    if not sys.platform.startswith("linux"):
        raise RuntimeError("protected broker execution requires Linux prctl")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(_PR_SET_DUMPABLE, int(enabled), 0, 0, 0)
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _peer_credentials(connection: socket.socket) -> tuple[int, int, int]:
    if not hasattr(socket, "SO_PEERCRED"):
        raise RuntimeError("protected broker execution requires SO_PEERCRED")
    payload = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    return struct.unpack("3i", payload)


def _process_start_time(pid: int) -> int:
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close_paren = stat.rfind(")")
    fields = stat[close_paren + 1 :].strip().split()
    if close_paren <= 0 or len(fields) <= 19:
        raise RuntimeError("could not identify protected process")
    return int(fields[19])


@dataclass(slots=True)
class _PendingProtectedDelivery:
    pid: int
    start_time: int
    values: dict[str, str]
    delivered: threading.Event


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
        self._process_ready = False
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._monitor: threading.Thread | None = None

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes], *, terminate_timeout: float) -> bool:
        """Request termination without ever waiting indefinitely."""

        if process.poll() is not None:
            return True
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=terminate_timeout)
            return True
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        try:
            process.wait(timeout=1.0)
            return True
        except subprocess.TimeoutExpired:
            return False

    def _start_process_locked(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            if self._process_ready:
                return
            raise RuntimeError("previous loopback proxy process is still exiting")
        if self.pid_path is not None:
            self.pid_path.unlink(missing_ok=True)
        argv = [
            _PROXY_PROCESS_NAME,
            "-I",
            str(Path(__file__)),
            "--forward-proxy-socket",
            self.unix_path,
            "--forward-proxy-port",
            str(self.port),
        ]
        executable: str | None = _current_python_executable()
        child_env: dict[str, str] | None = None
        if self.pid_path is not None:
            # CPython 3.13 cannot find its stdlib under ``-I`` when argv[0] is
            # an exec-a style synthetic name. Resolve that basename through a
            # local symlink on PATH instead; the child keeps the pkill-safe
            # ``polar-proxy`` argv while isolated startup remains functional.
            launcher = self.pid_path.parent / _PROXY_PROCESS_NAME
            launcher.unlink(missing_ok=True)
            launcher.symlink_to(executable)
            child_env = dict(os.environ)
            child_env["PATH"] = f"{launcher.parent}:{child_env.get('PATH', '')}"
            executable = None
        process = subprocess.Popen(
            argv,
            executable=executable,
            env=child_env,
        )
        self.process = process
        self._process_ready = False
        # Publish immediately so the outer supervisor can kill the child even
        # if the broker itself dies during cold imports below.
        if self.pid_path is not None:
            self.pid_path.write_text(f"{process.pid}\n")

        # ``/proc/self/exe`` initially gives the child a generic ``exe`` comm
        # name. Cold Lustre imports can exceed the old fixed 50 ms delay, so
        # wait until the child reaches main() and applies PR_SET_NAME. This
        # avoids returning while broad ``pkill python`` rules can still match
        # the proxy during startup.
        if sys.platform.startswith("linux"):
            deadline = time.monotonic() + _PROXY_START_TIMEOUT_SECONDS
            last_process_name = "<unavailable>"
            while process.poll() is None:
                try:
                    last_process_name = (
                        Path(f"/proc/{process.pid}/comm").read_text().strip()
                    )
                except OSError:
                    last_process_name = "<unavailable>"
                if last_process_name == _PROXY_PROCESS_NAME:
                    break
                if time.monotonic() >= deadline:
                    stopped = self._stop_process(process, terminate_timeout=1.0)
                    if stopped and self.pid_path is not None:
                        self.pid_path.unlink(missing_ok=True)
                    cleanup = "" if stopped else "; process cleanup is still pending"
                    raise RuntimeError(
                        "loopback proxy did not finish startup within "
                        f"{_PROXY_START_TIMEOUT_SECONDS:.0f}s; process name={last_process_name!r}"
                        f"{cleanup}"
                    )
                time.sleep(_PROXY_START_POLL_SECONDS)

        # PR_SET_NAME happens immediately before the bind. Give bind failures
        # a short chance to surface without consuming a real proxy connection.
        time.sleep(0.05)
        return_code = process.poll()
        if return_code is not None:
            if self.pid_path is not None:
                self.pid_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"loopback proxy exited during startup with code {return_code}"
            )

        self._process_ready = True

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
                if process is None or process.poll() is not None or not self._process_ready:
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
            self._process_ready = False
            stopped = process is None or self._stop_process(
                process,
                terminate_timeout=5.0,
            )
            # Keep the PID visible to the outer supervisor if SIGKILL is still
            # pending on an uninterruptible child.
            if stopped and self.pid_path is not None:
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
        protected_only: bool = False,
        runtime_socket_path: str | None = None,
    ) -> None:
        self.result_dir = result_dir
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.base_environment = dict(base_environment)
        self.proxy = proxy
        self.proxy_url = proxy_url
        self.allow_internet = allow_internet
        self.protected_only = protected_only
        self.runtime_socket_path = runtime_socket_path or socket_path
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._cancelled_before_start: set[str] = set()
        self._process_lock = threading.Lock()
        self._trusted_control_peer: tuple[int, int, int] | None = None
        self._protected_pending: dict[str, _PendingProtectedDelivery] = {}
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

    def execute_protected(self, request: Mapping[str, Any]) -> dict[str, object]:
        """Spawn an isolated runner; it fetches secrets only after dumpable=0."""

        if not self.protected_only:
            raise RuntimeError("protected exec is unavailable on the ordinary broker socket")
        request_id = str(request.get("id", ""))
        raw_argv = request.get("argv")
        if (
            not isinstance(raw_argv, list)
            or len(raw_argv) != 2
            or not all(
                isinstance(item, str) and item and "\x00" not in item
                for item in raw_argv
            )
        ):
            raise ValueError("protected exec requires an exact two-element argv")
        argv = list(raw_argv)
        if argv[0] != _PROTECTED_PYTHON or argv[1] not in {
            _SPILOT_RUNNER,
            _SPILOT_FORCED_RUNNER,
        }:
            raise ValueError("protected exec runner argv is not allowlisted")

        raw_file_digests = request.get("file_digests", {})
        if not isinstance(raw_file_digests, dict) or not all(
            isinstance(path, str) and isinstance(digest, str)
            for path, digest in raw_file_digests.items()
        ):
            raise ValueError("protected exec file digests must be an object")
        expected_paths = {argv[1]}
        if argv[1] == _SPILOT_FORCED_RUNNER:
            expected_paths.add(_SPILOT_RUNNER)
        if set(raw_file_digests) != expected_paths:
            raise ValueError("protected exec file digest set is incomplete")

        raw_env = request.get("env", {})
        raw_protected = request.get("protected_env", {})
        if not isinstance(raw_env, dict) or not isinstance(raw_protected, dict):
            raise ValueError("protected exec environments must be objects")
        protected_values: dict[str, str] = {}
        for raw_key, raw_value in raw_protected.items():
            key = str(raw_key)
            if not _PROTECTED_KEY_RE.fullmatch(key):
                raise ValueError("protected exec contains an invalid environment key")
            if not isinstance(raw_value, str) or not raw_value or "\x00" in raw_value:
                raise ValueError("protected exec contains an invalid secret value")
            if len(raw_value.encode("utf-8")) > 16_384:
                raise ValueError("protected exec secret exceeds the size limit")
            protected_values[key] = raw_value
        if not protected_values:
            raise ValueError("protected exec requires at least one protected value")
        raw_protected.clear()

        cwd = request.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ValueError("protected exec cwd must be a string or null")
        raw_timeout = request.get("timeout_sec")
        timeout = None if raw_timeout is None else float(raw_timeout)
        if timeout is not None and timeout <= 0:
            raise ValueError("protected exec timeout_sec must be positive")
        deadline = None if timeout is None else time.monotonic() + timeout

        environment = child_environment(
            self.base_environment,
            raw_env,
            proxy_url=self.proxy_url,
            allow_internet=self.allow_internet,
        )
        task_pythonpath = environment.get("PYTHONPATH")
        for name in _PROTECTED_ENV_DENYLIST:
            environment.pop(name, None)
        for key in protected_values:
            environment.pop(key, None)
            environment.pop(f"{key}{_PROTECTED_FD_SUFFIX}", None)
        if task_pythonpath is not None:
            environment["POLAR_TASK_PYTHONPATH"] = task_pythonpath
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONSAFEPATH"] = "1"
        environment[_PROTECTED_SOCKET_ENV] = self.runtime_socket_path
        environment[_PROTECTED_BROKER_PID_ENV] = str(os.getpid())
        environment[_PROTECTED_REQUEST_ID_ENV] = request_id
        base_url = environment.get("OPENAI_BASE_URL")
        if base_url:
            environment["OPENAI_API_BASE"] = base_url

        process: subprocess.Popen[bytes] | None = None
        stdout_path = self._result_path(request_id, "stdout")
        stderr_path = self._result_path(request_id, "stderr")
        active_marker = self.active_dir / f"{request_id}.pid"
        return_code = 127
        pending: _PendingProtectedDelivery | None = None
        pinned_files: dict[str, int] = {}
        ready_read = -1
        ready_write = -1
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            try:
                if self.proxy is not None:
                    self.proxy.ensure_running()
                for path in sorted(expected_paths):
                    pinned_files[path] = _sealed_verified_file(
                        path,
                        raw_file_digests[path],
                    )
                entry_fd = pinned_files[argv[1]]
                if argv[1] == _SPILOT_FORCED_RUNNER:
                    core_fd = pinned_files[_SPILOT_RUNNER]
                    trusted_argv = [
                        argv[0],
                        "-I",
                        "-c",
                        _FORCED_RUNNER_BOOTSTRAP,
                        "polar-forced-route-eval",
                        str(core_fd),
                        str(entry_fd),
                    ]
                else:
                    trusted_argv = [argv[0], "-I", f"/proc/self/fd/{entry_fd}"]
                ready_read, ready_write = os.pipe()
                environment[_PROTECTED_READY_FD_ENV] = str(ready_read)
                process = subprocess.Popen(
                    trusted_argv,
                    cwd=cwd,
                    env=environment,
                    start_new_session=True,
                    close_fds=True,
                    pass_fds=(*pinned_files.values(), ready_read),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
                os.close(ready_read)
                ready_read = -1
                pending = _PendingProtectedDelivery(
                    pid=process.pid,
                    start_time=_process_start_time(process.pid),
                    values=protected_values,
                    delivered=threading.Event(),
                )
                with self._process_lock:
                    self._processes[request_id] = process
                    self._protected_pending[request_id] = pending
                    cancelled = request_id in self._cancelled_before_start
                    self._cancelled_before_start.discard(request_id)
                # The child cannot connect until its PID/start-time ownership
                # record is visible to the protected socket handler.
                os.write(ready_write, b"1")
                os.close(ready_write)
                ready_write = -1
                if cancelled:
                    _kill_process_group(process)

                ready_budget = _PROTECTED_READY_TIMEOUT_SECONDS
                if deadline is not None:
                    ready_budget = min(
                        ready_budget,
                        max(0.0, deadline - time.monotonic()),
                    )
                if not pending.delivered.wait(timeout=ready_budget):
                    raise RuntimeError("protected runner did not establish its secure channel")
                wait_timeout = None
                if deadline is not None:
                    wait_timeout = max(0.0, deadline - time.monotonic())
                try:
                    return_code = process.wait(timeout=wait_timeout)
                except subprocess.TimeoutExpired:
                    _kill_process_group(process)
                    process.wait()
                    return_code = -1
            except Exception as exc:
                if process is not None and process.poll() is None:
                    _kill_process_group(process)
                    process.wait()
                stderr_file.write(
                    f"polar apptainer broker: protected exec failed: {type(exc).__name__}\n".encode()
                )
                return_code = 127
            finally:
                for descriptor in (ready_read, ready_write):
                    if descriptor >= 0:
                        try:
                            os.close(descriptor)
                        except OSError:
                            pass
                for descriptor in pinned_files.values():
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                pinned_files.clear()
                with self._process_lock:
                    self._processes.pop(request_id, None)
                    self._protected_pending.pop(request_id, None)
                protected_values.clear()
                if pending is not None:
                    pending.values.clear()
                active_marker.unlink(missing_ok=True)
        return {"return_code": return_code}

    def pin_secure_control(self, peer: tuple[int, int, int]) -> dict[str, object]:
        if not self.protected_only:
            raise RuntimeError("secure control pinning requires the protected socket")
        with self._process_lock:
            if self._trusted_control_peer is None:
                self._trusted_control_peer = peer
            elif self._trusted_control_peer != peer:
                raise PermissionError("protected broker control peer changed")
        return {
            "broker_pid": os.getpid(),
            "broker_start_time": _process_start_time(os.getpid()),
        }

    def authorize_secure_control(self, peer: tuple[int, int, int]) -> None:
        with self._process_lock:
            if self._trusted_control_peer is None or peer != self._trusted_control_peer:
                raise PermissionError("protected broker control peer is not pinned")

    def deliver_protected(
        self,
        request: Mapping[str, Any],
        peer: tuple[int, int, int],
    ) -> dict[str, object]:
        request_id = str(request.get("id", ""))
        with self._process_lock:
            pending = self._protected_pending.get(request_id)
            if pending is None:
                raise PermissionError("protected runner request is unknown")
            if peer[0] != pending.pid:
                raise PermissionError("protected runner peer identity is invalid")
            if _process_start_time(peer[0]) != pending.start_time:
                raise PermissionError("protected runner process identity changed")
            values = dict(pending.values)
            pending.values.clear()
            pending.delivered.set()
        return {"protected_env": values}

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
            peer = _peer_credentials(self.request)
            if operation == "ping":
                response = {"ok": True}
            elif operation == "secure_pin":
                response = {"ok": True, **server.pin_secure_control(peer)}
            elif operation == "exec":
                if server.protected_only:
                    raise PermissionError("ordinary exec is disabled on protected socket")
                response = {"ok": True, **server.execute(request)}
            elif operation == "exec_protected":
                server.authorize_secure_control(peer)
                response = {"ok": True, **server.execute_protected(request)}
            elif operation == "protected_child_ready":
                response = {"ok": True, **server.deliver_protected(request, peer)}
            elif operation == "cancel":
                if server.protected_only:
                    server.authorize_secure_control(peer)
                response = {
                    "ok": True,
                    "cancelled": server.cancel_command(str(request.get("id", ""))),
                }
            elif operation == "shutdown":
                if server.protected_only:
                    server.authorize_secure_control(peer)
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
    parser.add_argument("--protected-socket")
    parser.add_argument("--forward-proxy-socket")
    parser.add_argument("--forward-proxy-port", type=int)
    args = parser.parse_args()
    if args.forward_proxy_socket is not None:
        if args.forward_proxy_port is None:
            parser.error("--forward-proxy-port is required in proxy mode")
    elif args.socket is None or args.result_dir is None or args.protected_socket is None:
        parser.error(
            "--socket, --protected-socket and --result-dir are required in broker mode"
        )
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
    # Protected-exec secrets arrive in RPC bodies, never argv/environment.
    # Make the long-lived broker non-dumpable before it accepts any client so
    # arbitrary same-UID task processes cannot inspect handler memory.
    _set_process_dumpable(False)
    assert args.socket is not None
    assert args.protected_socket is not None
    assert args.result_dir is not None
    socket_path = Path(args.socket)
    protected_socket_path = Path(args.protected_socket)
    result_dir = Path(args.result_dir)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    protected_socket_path.unlink(missing_ok=True)

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
    protected_server = _BrokerServer(
        str(protected_socket_path),
        result_dir,
        base_environment=environment,
        proxy=proxy,
        proxy_url=proxy_url,
        allow_internet=allow_internet,
        protected_only=True,
        runtime_socket_path=str(protected_socket_path),
    )
    socket_path.chmod(0o600)
    protected_socket_path.chmod(0o600)
    protected_thread = threading.Thread(
        target=protected_server.serve_forever,
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    protected_thread.start()

    def request_shutdown(_signum: int, _frame: object) -> None:
        server.kill_active_commands()
        protected_server.kill_active_commands()
        threading.Thread(target=protected_server.shutdown, daemon=True).start()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        protected_server.shutdown()
        protected_server.server_close()
        protected_thread.join(timeout=1.0)
        socket_path.unlink(missing_ok=True)
        protected_socket_path.unlink(missing_ok=True)
        if proxy is not None:
            proxy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
