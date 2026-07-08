"""Apptainer-backed rollout runtime.

Three execution modes (selected once at construction):

* instance mode (default): ``apptainer instance start`` once, then every
  command runs as ``apptainer exec instance://<name>``. This is the original
  behavior and is efficient on bare metal.
* direct-exec mode (``POLAR_APPTAINER_NO_INSTANCE`` in {1,true,yes,on}): no
  daemon instance is started.  One long-lived ``apptainer exec`` hosts a small
  Unix-socket command broker, and all agent/evaluator commands run through it.
  This is required when running nested inside Pyxis/enroot, where ``instance
  start`` + ``exec instance://`` cannot re-enter the nested user namespace.
  Keeping one direct exec alive also preserves localhost services and process
  state between the agent and verifier while retaining PID/network isolation.
* legacy direct-exec mode (also set ``POLAR_APPTAINER_PERSISTENT_BROKER=0``):
  every command is a fresh ``apptainer exec`` that reuses the session overlay.
  This is a compatibility and performance-comparison path for nested runtimes;
  the persistent broker remains the default.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import signal
import shlex
import shutil
import socket
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any
import uuid

from polar.runtime.base import (
    BaseRuntime,
    RuntimeContainmentError,
    RuntimeDestroyedError,
)
from polar.runtime.models import ExecResult, RuntimeSpec

logger = logging.getLogger(__name__)

_LOCAL_LAUNCHER_CWD = "/"
_BROKER_DIR_NAME = ".apptainer-broker"
_BROKER_SOCKET_NAME = "control.sock"
_BROKER_RESULTS_DIR_NAME = "results"
_BROKER_TRANSFERS_DIR_NAME = "transfers"
_BROKER_LOG_NAME = "broker.log"
_BROKER_SCRIPT_NAME = "apptainer_broker.py"
_BROKER_SUPERVISOR_NAME = "apptainer_broker_supervisor.sh"
_BROKER_INTERPRETER_NAME = "interpreter.path"
_BROKER_GENERATION_NAME = "broker.generation"
_BROKER_RESTART_COUNT_NAME = "broker.restart_count"
_BROKER_RUNTIME_DIR = f"/polar/session/{_BROKER_DIR_NAME}"
_BROKER_RUNTIME_SOCKET = f"{_BROKER_RUNTIME_DIR}/{_BROKER_SOCKET_NAME}"
_BROKER_RUNTIME_RESULTS_DIR = f"{_BROKER_RUNTIME_DIR}/{_BROKER_RESULTS_DIR_NAME}"
_BROKER_RUNTIME_TRANSFERS_DIR = f"{_BROKER_RUNTIME_DIR}/{_BROKER_TRANSFERS_DIR_NAME}"
_BROKER_TRUSTED_RUNTIME_DIR = "/polar/runtime-control"
_BROKER_TRUSTED_RUNTIME_SCRIPT = (
    f"{_BROKER_TRUSTED_RUNTIME_DIR}/{_BROKER_SCRIPT_NAME}"
)
_BROKER_TRUSTED_RUNTIME_SUPERVISOR = (
    f"{_BROKER_TRUSTED_RUNTIME_DIR}/{_BROKER_SUPERVISOR_NAME}"
)
_BROKER_START_CONCURRENCY_ENV = "POLAR_APPTAINER_BROKER_START_CONCURRENCY"
_PERSISTENT_BROKER_ENV = "POLAR_APPTAINER_PERSISTENT_BROKER"
_DEFAULT_BROKER_START_CONCURRENCY = 2
_BROKER_START_GATE_ATTR = "_polar_apptainer_broker_start_gate"
_BROKER_START_GATE_SIZE_ATTR = "_polar_apptainer_broker_start_gate_size"
_BROKER_START_PER_IMAGE_CONCURRENCY_ENV = "POLAR_APPTAINER_BROKER_START_PER_IMAGE_CONCURRENCY"
_DEFAULT_BROKER_START_PER_IMAGE_CONCURRENCY = 2
_BROKER_START_PER_IMAGE_GATES_ATTR = "_polar_apptainer_broker_start_per_image_gates"
_BROKER_START_PER_IMAGE_GATE_SIZE_ATTR = "_polar_apptainer_broker_start_per_image_gate_size"
_BROKER_START_TIMEOUT_ENV = "POLAR_APPTAINER_BROKER_START_TIMEOUT_SEC"
_DEFAULT_BROKER_START_TIMEOUT_SEC = 120.0
_BROKER_RETRY_BACKOFF_ENV = "POLAR_APPTAINER_BROKER_RETRY_BACKOFF_SEC"
_DEFAULT_BROKER_RETRY_BACKOFF_SEC = 5.0
_BROKER_RETRY_BACKOFF_MAX_ENV = "POLAR_APPTAINER_BROKER_RETRY_BACKOFF_MAX_SEC"
_DEFAULT_BROKER_RETRY_BACKOFF_MAX_SEC = 30.0
_BROKER_RECOVERY_TIMEOUT_ENV = "POLAR_APPTAINER_BROKER_RECOVERY_TIMEOUT_SEC"
_DEFAULT_BROKER_RECOVERY_TIMEOUT_SEC = 30.0
_BROKER_MAX_RESPONSE_BYTES = 1024 * 1024
_BROKER_READ_CHUNK_BYTES = 64 * 1024
_PROC_ROOT = Path("/proc")
_DIRECT_BROKER_TERM_GRACE_SEC = 0.25
_DIRECT_BROKER_KILL_TIMEOUT_SEC = 2.0
_DIRECT_BROKER_CLEANUP_POLL_SEC = 0.05
_DIRECT_BROKER_STABLE_EMPTY_PROBES = 2


class _BrokerDisconnectedError(RuntimeError):
    """The broker process disappeared after accepting an RPC connection."""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_bool(name: str, *, default: bool) -> bool:
    """Read a conventional boolean environment value with a safe default."""

    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning("Ignoring invalid boolean env %s=%r; using %s", name, value, default)
    return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer env %s=%r; using %d", name, value, default)
        return default
    return max(1, parsed)


def _bounded_env_int(name: str, default: int, *, maximum: int) -> int:
    """Read a bounded positive integer without allowing an unsafe fan-out."""

    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        parsed = 0
    if not 1 <= parsed <= maximum:
        logger.warning(
            "Ignoring invalid integer env %s=%r; expected 1..%d, using %d",
            name,
            value,
            maximum,
            default,
        )
        return default
    return parsed


def _bounded_env_float(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """Read a finite bounded float, falling back safely on bad operator input."""

    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        parsed = float("nan")
    if not minimum <= parsed <= maximum:
        logger.warning(
            "Ignoring invalid numeric env %s=%r; expected %.3g..%.3g, using %.3g",
            name,
            value,
            minimum,
            maximum,
            default,
        )
        return default
    return parsed


def _broker_start_gate() -> asyncio.Semaphore:
    """Return the event-loop-wide gate for complete broker startup attempts.

    The generic subprocess gate only covers ``Popen`` itself.  An Apptainer
    process does most of its expensive image/overlay mount work afterwards, so
    releasing admission at ``Popen`` allowed dozens of mounts to stampede the
    node and then retry in synchronized waves.  This separate gate is held
    until the broker socket answers a ping (or the failed process is reaped).
    """

    loop = asyncio.get_running_loop()
    concurrency = _bounded_env_int(
        _BROKER_START_CONCURRENCY_ENV,
        _DEFAULT_BROKER_START_CONCURRENCY,
        maximum=64,
    )
    gate = getattr(loop, _BROKER_START_GATE_ATTR, None)
    configured_size = getattr(loop, _BROKER_START_GATE_SIZE_ATTR, None)
    if gate is None:
        gate = asyncio.Semaphore(concurrency)
        setattr(loop, _BROKER_START_GATE_ATTR, gate)
        setattr(loop, _BROKER_START_GATE_SIZE_ATTR, concurrency)
    elif configured_size != concurrency:
        raise RuntimeError(
            f"{_BROKER_START_CONCURRENCY_ENV} changed from "
            f"{configured_size} to {concurrency} after this event loop started"
        )
    return gate


def _canonical_image_path(image: str) -> str:
    """Return one stable key for equivalent local Apptainer image paths."""

    return os.fspath(Path(image).expanduser().resolve(strict=False))


def _broker_start_per_image_gate(image: str) -> asyncio.Semaphore:
    """Return the event-loop-local startup gate for one canonical image.

    Many sessions for a prompt group use the same SIF.  Their concurrent
    loop/FUSE setup can overload one image even when the node-wide aggregate
    startup fan-out is otherwise safe.  Keep a separate semaphore for each
    canonical image path so unrelated images can still fill the aggregate
    gate.
    """

    loop = asyncio.get_running_loop()
    concurrency = _bounded_env_int(
        _BROKER_START_PER_IMAGE_CONCURRENCY_ENV,
        _DEFAULT_BROKER_START_PER_IMAGE_CONCURRENCY,
        maximum=64,
    )
    gates = getattr(loop, _BROKER_START_PER_IMAGE_GATES_ATTR, None)
    configured_size = getattr(loop, _BROKER_START_PER_IMAGE_GATE_SIZE_ATTR, None)
    if gates is None:
        gates = {}
        setattr(loop, _BROKER_START_PER_IMAGE_GATES_ATTR, gates)
        setattr(loop, _BROKER_START_PER_IMAGE_GATE_SIZE_ATTR, concurrency)
    elif configured_size != concurrency:
        raise RuntimeError(
            f"{_BROKER_START_PER_IMAGE_CONCURRENCY_ENV} changed from "
            f"{configured_size} to {concurrency} after this event loop started"
        )

    image_key = _canonical_image_path(image)
    gate = gates.get(image_key)
    if gate is None:
        gate = asyncio.Semaphore(concurrency)
        gates[image_key] = gate
    return gate


def _read_proc_table(proc_root: Path) -> dict[int, tuple[int, int, int]]:
    """Return pid -> (ppid, session id, start time) for a procfs snapshot."""

    processes: dict[int, tuple[int, int, int]] = {}
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeContainmentError("could not read procfs for containment proof") from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            # comm may contain spaces or parentheses. Everything after its
            # final ')' starts at field 3 (state); ppid/session/starttime are
            # fields 4/6/22 respectively.
            remainder = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            processes[pid] = (int(remainder[1]), int(remainder[3]), int(remainder[19]))
        except (FileNotFoundError, ProcessLookupError):
            # Processes can disappear at every point in a procfs walk.
            continue
        except (IndexError, OSError, ValueError) as exc:
            raise RuntimeContainmentError(
                f"could not read process identity for containment proof: {entry.name}"
            ) from exc
    return processes


def _engine_config_references_session(
    pid: int,
    session_dir: Path,
    *,
    proc_root: Path,
) -> bool:
    """Whether an Apptainer runtime parent's ENGINE_CONFIG owns session_dir."""

    try:
        entries = (proc_root / str(pid) / "environ").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError as exc:
        raise RuntimeContainmentError(
            f"could not read process environment for containment proof: {pid}"
        ) from exc
    chunks: list[tuple[int, bytes]] = []
    for entry in entries:
        key, separator, value = entry.partition(b"=")
        if not separator or not key.startswith(b"ENGINE_CONFIG"):
            continue
        suffix = key.removeprefix(b"ENGINE_CONFIG")
        # ENGINE_CONFIG_CHUNKS is metadata, not part of the JSON payload.
        if suffix == b"_CHUNKS":
            continue
        index = int(suffix) if suffix.isdigit() else 0
        chunks.append((index, value))
    if not chunks:
        return False
    payload = b"".join(value for _, value in sorted(chunks))
    # Match the complete JSON string, not a bare path substring: session
    # names commonly share prefixes and killing a sibling runtime is worse
    # than leaving one failed mount behind.
    encoded_session = json.dumps(os.fspath(session_dir)).encode()
    return encoded_session in payload


def _direct_broker_process_snapshot(
    session_dir: Path,
    *,
    proc_root: Path = _PROC_ROOT,
    known_sessions: set[int] | None = None,
) -> tuple[set[int], dict[int, int]]:
    """Find every process belonging to this direct-exec Apptainer runtime.

    Apptainer's runtime parent calls setsid/clone-parent, so it escapes the
    launcher's process group and is reparented directly to the gateway. Its
    ENGINE_CONFIG still contains the unique per-session bind source. Once the
    runtime SID is identified, include all members of that SID plus recursive
    descendants that created another session/process group.
    """

    table = _read_proc_table(proc_root)
    runtime_sessions = set(known_sessions or ())
    # ENGINE_CONFIG can be large. Discover ownership once, then subsequent
    # TERM/KILL polling can identify survivors solely by the stable SID.
    if not runtime_sessions:
        for pid, (_, session_id, _) in sorted(table.items()):
            try:
                if (proc_root / str(pid)).stat().st_uid != os.getuid():
                    continue
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError as exc:
                raise RuntimeContainmentError(
                    f"could not identify process owner for containment proof: {pid}"
                ) from exc
            if _engine_config_references_session(
                pid,
                session_dir,
                proc_root=proc_root,
            ):
                runtime_sessions.add(session_id)
                break

    targets = {pid for pid, (_, session_id, _) in table.items() if session_id in runtime_sessions}
    # Catch commands that called setsid after entering the runtime namespace.
    while True:
        descendants = {pid for pid, (ppid, _, _) in table.items() if ppid in targets}
        expanded = targets | descendants
        if expanded == targets:
            break
        targets = expanded
    return runtime_sessions, {pid: table[pid][2] for pid in targets}


def _signal_proc_snapshot(
    processes: dict[int, int],
    signum: int,
    *,
    proc_root: Path,
) -> None:
    """Signal only PIDs that still have the snapshotted start time."""

    for pid, start_time in processes.items():
        if pid == os.getpid():
            continue
        current = _read_proc_table_for_pid(pid, proc_root=proc_root)
        if current is None or current[2] != start_time:
            continue
        try:
            os.kill(pid, signum)
        except (PermissionError, ProcessLookupError):
            continue


def _read_proc_table_for_pid(
    pid: int,
    *,
    proc_root: Path,
) -> tuple[int, int, int] | None:
    """Read one proc identity without rescanning all of procfs."""

    try:
        remainder = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return int(remainder[1]), int(remainder[3]), int(remainder[19])
    except (IndexError, OSError, ValueError):
        return None


def _reap_direct_children(processes: dict[int, int]) -> None:
    for pid in processes:
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, ProcessLookupError):
            continue


def _exec_failure_message(operation: str, result: ExecResult) -> str:
    """Keep the command's real return code and diagnostic in wrapper errors."""

    diagnostic = (result.stderr or result.stdout or "").strip()
    suffix = f": {diagnostic}" if diagnostic else ""
    return f"{operation} failed with exit code {result.return_code}{suffix}"


class ApptainerRuntime(BaseRuntime):
    """Apptainer runtime used across rollout stages (instance or direct-exec)."""

    def __init__(self, spec: RuntimeSpec, session_id: str, session_dir: Path) -> None:
        super().__init__(spec, session_id, session_dir)
        # Use a hash suffix to guarantee uniqueness even when session IDs
        # share a long prefix (e.g. "sk-polar-...-eval" vs "sk-polar-...").
        short_hash = hashlib.sha256(session_id.encode()).hexdigest()[:8]
        safe_name = session_id.replace("/", "-")[:30]
        self._instance_name = f"polar-{safe_name}-{short_hash}"
        self._binary = self._resolve_binary()
        # Read once at construction; changing the env later has no effect.
        self._use_instance = not _env_truthy("POLAR_APPTAINER_NO_INSTANCE")
        self._use_persistent_broker = not self._use_instance and _env_bool(
            _PERSISTENT_BROKER_ENV,
            default=True,
        )
        self._overlay_dir: Path | None = None
        self._broker_dir = self.session_dir / _BROKER_DIR_NAME
        self._broker_socket = self._broker_dir / _BROKER_SOCKET_NAME
        # Keep enough randomness to avoid collisions without pushing local
        # test/session roots over Linux's 108-byte AF_UNIX pathname limit.
        self._protected_broker_socket_name = f"p-{uuid.uuid4().hex[:8]}.sock"
        self._protected_broker_socket = (
            self._broker_dir / self._protected_broker_socket_name
        )
        self._protected_broker_identity: tuple[int, int] | None = None
        self._known_runtime_sessions: set[int] = set()
        self._broker_log = self._broker_dir / _BROKER_LOG_NAME
        self._broker_interpreter_file = self._broker_dir / _BROKER_INTERPRETER_NAME
        self._broker_generation_file = self._broker_dir / _BROKER_GENERATION_NAME
        self._broker_restart_count_file = self._broker_dir / _BROKER_RESTART_COUNT_NAME
        self._broker_results_dir = self._broker_dir / _BROKER_RESULTS_DIR_NAME
        self._broker_transfers_dir = self._broker_dir / _BROKER_TRANSFERS_DIR_NAME
        self._broker_task: asyncio.Task[tuple[int, str | None, str | None]] | None = None
        self._broker_was_started = False
        self._broker_recovery_lock = asyncio.Lock()
        self._teardown_lock = asyncio.Lock()
        self._teardown_started = False
        self._broker_observed_generation = 0
        self._broker_recovery_count = 0
        self._broker_recovery_failure_count = 0
        self._broker_recovery_ms = 0.0
        self._broker_preflight_failure_count = 0
        self._broker_disconnect_count = 0

    @property
    def runtime_id(self) -> str:
        return self._instance_name

    @property
    def supports_gpus(self) -> bool:
        return True

    @property
    def can_disable_internet(self) -> bool:
        return True

    def _runtime_options(self) -> list[str]:
        """The apptainer flags shared by `instance start` and direct `exec`."""
        if self._overlay_dir is None:
            raise RuntimeError("apptainer runtime not started (overlay dir unset)")
        # Host-backed overlay (NOT --writable-tmpfs: its 64 MB tmpfs is too small).
        options = ["--overlay", str(self._overlay_dir)]
        # Never expose the launcher's HOME or CWD to untrusted task commands.
        # Apptainer binds both automatically, independently of ``hostfs``. In
        # the Slurm launcher the inherited CWD is the shared data root, so a
        # task such as ``rm -rf /*`` would otherwise delete shared assets.
        # ``bind-paths`` also suppresses administrator-configured static bind
        # paths. Explicit per-session ``--bind`` mounts added below are still
        # honored, so the sandbox sees only the assets declared by RuntimeSpec.
        disabled_mounts: list[str] = ["cwd", "home", "bind-paths"]
        # The gateway keeps each bind source under the outer container's
        # /tmp. If an inner task sees that same /tmp, a broad cleanup such as
        # ``rm -rf /tmp/*`` can delete every concurrently active session.
        if _env_truthy("POLAR_APPTAINER_NO_MOUNT_HOSTFS"):
            disabled_mounts.append("hostfs")
        if _env_truthy("POLAR_APPTAINER_NO_MOUNT_TMP"):
            disabled_mounts.append("tmp")
        if disabled_mounts:
            options.extend(["--no-mount", ",".join(disabled_mounts)])
        # Coding tasks are untrusted with respect to broad process-management
        # commands (pkill, kill -1, test-suite cleanup helpers). Keep them out
        # of the launcher's PID/IPC namespaces so a sandbox cannot terminate
        # the gateway, Ray worker, or a sibling sandbox in the same Slurm step.
        if _env_truthy("POLAR_APPTAINER_CLEANENV"):
            # Do not leak the outer Slurm/PMI/PMIx contract into arbitrary
            # task commands. An inner mpi4py import can otherwise join the
            # launcher's PMIx step and tear down rank 0 when it times out.
            # Runtime/task variables are supplied explicitly by ``exec`` via
            # the inner ``env KEY=value ...`` command below.
            options.append("--cleanenv")
        if _env_truthy("POLAR_APPTAINER_ISOLATE_PID"):
            options.append("--pid")
        if _env_truthy("POLAR_APPTAINER_ISOLATE_IPC"):
            options.append("--ipc")
        if self.spec.gpus > 0:
            options.append("--nv")
        if not self.spec.allow_internet:
            network_name: str | None = "none"
        else:
            network_name = self.spec.network
        if network_name and network_name != "host":
            options.extend(["--net", "--network", network_name])
        options.extend(["--bind", f"{self.session_dir}:{self.runtime_session_dir}"])
        if not self._use_instance and self._use_persistent_broker:
            # The session bind is intentionally writable so task commands can
            # exchange files and the broker can publish sockets/results.  The
            # broker implementation and its supervisor are part of the trusted
            # control plane, however, and must never be copied into that bind:
            # an already-running untrusted task could replace either file
            # before a supervisor restart.  Mount the installed runtime source
            # directory at a separate read-only path instead.
            trusted_source_dir = Path(__file__).resolve().parent
            broker_source = trusted_source_dir / _BROKER_SCRIPT_NAME
            supervisor_source = trusted_source_dir / _BROKER_SUPERVISOR_NAME
            if not broker_source.is_file() or not supervisor_source.is_file():
                raise RuntimeError("trusted Apptainer broker sources are unavailable")
            options.extend(
                [
                    "--bind",
                    f"{trusted_source_dir}:{_BROKER_TRUSTED_RUNTIME_DIR}:ro",
                ]
            )
        # Match DockerRuntime's kwargs.volumes contract (src[:dst[:opts]]).
        for volume in self.spec.kwargs.get("volumes", []):
            options.extend(["--bind", str(volume)])
        if self.spec.allow_internet:
            for volume in self.spec.internet_volumes:
                options.extend(["--bind", volume])
        return options

    def _exec_base_args(self) -> list[str]:
        """Command prefix for every exec/upload/download."""
        if self._use_instance:
            return [self._binary, "exec", f"instance://{self._instance_name}"]
        # Direct mode: the overlay/binds/image are re-specified on every command
        # (no persistent instance carries them).
        return [self._binary, "exec", *self._runtime_options(), self.spec.image]

    @staticmethod
    def _shell_join(args: list[str]) -> str:
        return " ".join(shlex.quote(a) for a in args)

    def _broker_connect_path(
        self,
        socket_name: str = _BROKER_SOCKET_NAME,
    ) -> tuple[str, int | None]:
        """Return a short host-visible path for a possibly long session UDS.

        Linux limits AF_UNIX pathnames to roughly 108 bytes.  Session roots on
        Lustre can be much longer, even though the broker binds the short
        in-container path under ``/polar/session``.  Resolving the directory
        through an open ``/proc/self/fd`` keeps the client pathname short.
        """

        directory_fd: int | None = None
        try:
            directory_fd = os.open(self._broker_dir, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            return str(self._broker_dir / socket_name), None
        return f"/proc/self/fd/{directory_fd}/{socket_name}", directory_fd

    @staticmethod
    def _host_process_start_time(pid: int) -> int:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close_paren = stat.rfind(")")
        fields = stat[close_paren + 1 :].strip().split()
        if close_paren <= 0 or len(fields) <= 19:
            raise RuntimeError("protected broker peer has an invalid proc record")
        return int(fields[19])

    @staticmethod
    def _host_process_session_id(pid: int) -> int:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close_paren = stat_text.rfind(")")
        fields = stat_text[close_paren + 1 :].strip().split()
        if close_paren <= 0 or len(fields) <= 3:
            raise RuntimeError("protected broker peer has an invalid proc record")
        return int(fields[3])

    def _validate_protected_broker_peer(self, peer: tuple[int, int, int]) -> tuple[int, int]:
        pid, uid, _gid = peer
        if pid <= 0 or uid != os.getuid():
            raise RuntimeError("protected broker peer credentials are invalid")
        command_line = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if not command_line or command_line[0] != b"polar-broker":
            raise RuntimeError("protected broker peer process profile is invalid")
        if not any(item.endswith(b"apptainer_broker.py") for item in command_line):
            raise RuntimeError("protected broker peer script profile is invalid")
        return pid, self._host_process_start_time(pid)

    async def _protected_broker_rpc(
        self,
        request: dict[str, object],
        *,
        socket_timeout: float | None,
        pin: bool = False,
    ) -> dict[str, Any]:
        """Send secrets only after SO_PEERCRED matches the pinned broker."""

        connect_path, directory_fd = self._broker_connect_path(
            self._protected_broker_socket_name
        )
        writer: asyncio.StreamWriter | None = None
        try:
            connection = asyncio.open_unix_connection(path=connect_path)
            if socket_timeout is None:
                reader, writer = await connection
            else:
                reader, writer = await asyncio.wait_for(connection, timeout=socket_timeout)
            transport_socket = writer.get_extra_info("socket")
            if transport_socket is None or not hasattr(socket, "SO_PEERCRED"):
                raise RuntimeError("protected broker peer credentials are unavailable")
            import struct

            peer = struct.unpack(
                "3i",
                transport_socket.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12),
            )
            identity = self._validate_protected_broker_peer(peer)
            if pin:
                if self._protected_broker_identity not in (None, identity):
                    raise RuntimeError("protected broker identity changed during pinning")
                self._protected_broker_identity = identity
                self._known_runtime_sessions.add(
                    self._host_process_session_id(identity[0])
                )
            elif identity != self._protected_broker_identity:
                raise RuntimeError("protected broker identity changed")

            writer.write(json.dumps(request, separators=(",", ":")).encode() + b"\n")
            drain = writer.drain()
            if socket_timeout is None:
                await drain
                response_payload = await reader.readline()
            else:
                await asyncio.wait_for(drain, timeout=socket_timeout)
                response_payload = await asyncio.wait_for(
                    reader.readline(), timeout=socket_timeout
                )
            if not response_payload or len(response_payload) > _BROKER_MAX_RESPONSE_BYTES:
                raise RuntimeError("protected broker returned an invalid response")
            response = json.loads(response_payload)
            if not isinstance(response, dict) or response.get("ok") is not True:
                detail = response.get("error") if isinstance(response, dict) else None
                raise RuntimeError(f"protected broker request failed: {detail or 'unknown error'}")
            return response
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionError, OSError):
                    pass
            if directory_fd is not None:
                os.close(directory_fd)

    async def _broker_rpc(
        self,
        request: dict[str, object],
        *,
        socket_timeout: float | None,
    ) -> dict[str, Any]:
        """Exchange one newline-delimited RPC without occupying a worker thread.

        ``socket_timeout`` retains socket-style semantics: connect, drain, and
        every receive operation each get the complete timeout budget.  The
        directory descriptor must remain open until the AF_UNIX connection is
        established because long Lustre paths are reached through
        ``/proc/self/fd/<fd>``.
        """

        connect_path, directory_fd = self._broker_connect_path()
        writer: asyncio.StreamWriter | None = None
        try:
            connection = asyncio.open_unix_connection(path=connect_path)
            if socket_timeout is None:
                reader, writer = await connection
            else:
                reader, writer = await asyncio.wait_for(
                    connection,
                    timeout=socket_timeout,
                )

            request_payload = json.dumps(request, separators=(",", ":")).encode() + b"\n"
            writer.write(request_payload)
            if socket_timeout is None:
                await writer.drain()
            else:
                await asyncio.wait_for(writer.drain(), timeout=socket_timeout)

            response_payload = bytearray()
            response_started = False
            while True:
                receive = reader.read(_BROKER_READ_CHUNK_BYTES)
                if socket_timeout is None:
                    chunk = await receive
                else:
                    chunk = await asyncio.wait_for(
                        receive,
                        timeout=socket_timeout,
                    )
                if not chunk:
                    break
                response_started = True
                newline = chunk.find(b"\n")
                if newline >= 0:
                    response_payload.extend(chunk[:newline])
                else:
                    response_payload.extend(chunk)
                if len(response_payload) > _BROKER_MAX_RESPONSE_BYTES:
                    raise RuntimeError("apptainer broker response exceeded 1 MiB")
                if newline >= 0:
                    break

            if not response_started:
                raise _BrokerDisconnectedError(
                    "apptainer broker closed the connection without a response"
                )
            response = json.loads(response_payload)
            if not isinstance(response, dict):
                raise RuntimeError("apptainer broker returned a non-object response")
            if not response.get("ok"):
                raise RuntimeError(
                    f"apptainer broker request failed: {response.get('error', 'unknown error')}"
                )
            return response
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionError, OSError):
                    pass
            if directory_fd is not None:
                os.close(directory_fd)

    @staticmethod
    def _read_broker_counter(path: Path) -> int:
        try:
            value = int(path.read_text().strip())
        except (OSError, ValueError):
            return 0
        return max(0, value)

    def _assert_direct_broker_owner_live(self) -> None:
        """Reject recovery when the namespace owner is gone or teardown began."""

        if self._destroyed or self._teardown_started:
            raise RuntimeDestroyedError(
                f"apptainer runtime {self.runtime_id} was already destroyed"
            )
        task = self._broker_task
        if task is None:
            raise RuntimeError(
                f"apptainer runtime owner for {self.runtime_id} is not running; "
                "refusing to start a replacement namespace"
            )
        if not task.done():
            return
        if task.cancelled():
            diagnostic = "launcher task was cancelled"
        else:
            try:
                return_code, stdout, stderr = task.result()
                detail = (stderr or stdout or "").strip()
                diagnostic = f"launcher exited with code {return_code}"
                if detail:
                    diagnostic = f"{diagnostic}: {detail}"
            except BaseException as exc:
                diagnostic = f"launcher failed: {exc}"
        raise RuntimeError(
            f"apptainer runtime owner for {self.runtime_id} is not running "
            f"({diagnostic}); refusing to start a replacement namespace"
        )

    def _record_broker_recovery(self, *, started_at: float, trigger: str) -> None:
        generation = self._read_broker_counter(self._broker_generation_file)
        self._broker_observed_generation = max(
            self._broker_observed_generation,
            generation,
        )
        elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        self._broker_recovery_count += 1
        self._broker_recovery_ms += elapsed_ms
        logger.warning(
            "Recovered in-runtime Apptainer broker for %s "
            "(trigger=%s generation=%d supervisor_restarts=%d elapsed_ms=%.1f)",
            self.runtime_id,
            trigger,
            generation,
            self._read_broker_counter(self._broker_restart_count_file),
            elapsed_ms,
        )

    async def _recover_live_broker(self, *, trigger: str, error: BaseException) -> None:
        """Wait for the existing supervisor to restart its broker child once.

        This deliberately never calls ``_start_direct_broker``: doing so would
        create a new Apptainer namespace and could silently lose services and
        state owned by the failed runtime.  The shell supervisor has a bounded
        restart budget; this host-side lock only coalesces concurrent waiters.
        """

        if self._protected_broker_identity is not None:
            # Once the secret-bearing control plane is pinned, accepting a
            # replacement same-UID process would make its identity ambiguous.
            # The session must be torn down instead of recovering or replaying.
            self._broker_recovery_failure_count += 1
            raise RuntimeError(
                "protected Apptainer broker identity changed; recovery is forbidden"
            ) from error

        recovery_count_before = self._broker_recovery_count
        async with self._broker_recovery_lock:
            self._assert_direct_broker_owner_live()
            started_at = time.perf_counter()
            generation_before = self._broker_observed_generation

            # Another exec may have completed recovery while this caller was
            # waiting for the lock.  Count the new supervisor generation once.
            try:
                await self._broker_rpc({"operation": "ping"}, socket_timeout=0.5)
            except (OSError, RuntimeError, ValueError):
                pass
            else:
                generation = self._read_broker_counter(self._broker_generation_file)
                if generation > generation_before:
                    self._record_broker_recovery(started_at=started_at, trigger=trigger)
                return

            logger.warning(
                "Waiting for in-runtime Apptainer broker recovery for %s "
                "(trigger=%s generation=%d): %s",
                self.runtime_id,
                trigger,
                generation_before,
                error,
            )
            try:
                await self._wait_for_broker_ready(
                    _bounded_env_float(
                        _BROKER_RECOVERY_TIMEOUT_ENV,
                        _DEFAULT_BROKER_RECOVERY_TIMEOUT_SEC,
                        minimum=1.0,
                        maximum=300.0,
                    )
                )
                self._assert_direct_broker_owner_live()
            except Exception:
                elapsed_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
                self._broker_recovery_failure_count += 1
                self._broker_recovery_ms += elapsed_ms
                logger.exception(
                    "In-runtime Apptainer broker recovery failed for %s "
                    "(trigger=%s elapsed_ms=%.1f)",
                    self.runtime_id,
                    trigger,
                    elapsed_ms,
                )
                raise

            # The generation file is authoritative in production.  The count
            # fallback keeps injected/unit-test supervisors observable too.
            generation = self._read_broker_counter(self._broker_generation_file)
            if (
                generation > self._broker_observed_generation
                or self._broker_recovery_count == recovery_count_before
            ):
                self._record_broker_recovery(started_at=started_at, trigger=trigger)

    async def _ensure_direct_broker_ready_for_exec(self) -> None:
        """Health-check before an exec so a between-command crash is retry-safe."""

        self._assert_direct_broker_owner_live()
        try:
            await self._broker_rpc({"operation": "ping"}, socket_timeout=0.5)
        except (OSError, RuntimeError, ValueError) as exc:
            self._broker_preflight_failure_count += 1
            await self._recover_live_broker(trigger="pre_exec", error=exc)
        else:
            generation = self._read_broker_counter(self._broker_generation_file)
            if generation > self._broker_observed_generation:
                self._record_broker_recovery(
                    started_at=time.perf_counter(),
                    trigger="pre_exec_observed",
                )
        self._assert_direct_broker_owner_live()

    def _broker_launch_args(self) -> list[str]:
        broker_python = self.spec.env.get("POLAR_APPTAINER_BROKER_PYTHON", "python3").strip()
        if not broker_python:
            broker_python = "python3"
        if any(character in broker_python for character in ("\0", "\n", "\r")):
            raise ValueError("POLAR_APPTAINER_BROKER_PYTHON must be a single path")
        # Do not pass the interpreter as an ``env KEY=...python`` argv element.
        # Agent cleanup commands inspect full process command lines, and in a
        # shared PID namespace that made even the non-Python namespace owner a
        # target for ``pkill -f python``.  The session bind is already private
        # to this runtime and is a safer control channel.
        self._broker_interpreter_file.write_text(f"{broker_python}\n")
        self._broker_interpreter_file.chmod(0o600)
        # Keep a non-Python namespace owner around the control broker.  Coding
        # agents commonly use broad cleanup commands such as ``pkill python``;
        # without the supervisor that also killed the runtime control plane and
        # turned an ordinary command failure into an infrastructure error.
        broker_command = self._shell_join(
            ["bash", _BROKER_TRUSTED_RUNTIME_SUPERVISOR]
        )
        broker_command = f"exec {broker_command}"
        # The broker and its protected control socket must exist before any
        # task-controlled initializer can spawn a same-UID scanner.  The init
        # hook is executed through the pinned broker only after readiness.
        # The owner process lives for the complete rollout. Redirect its own
        # output (including initializer-started services) to a regular file so
        # the launcher does not retain an ever-growing capture buffer.
        broker_log = f"{_BROKER_RUNTIME_DIR}/{_BROKER_LOG_NAME}"
        broker_command = f"exec >>{shlex.quote(broker_log)} 2>&1; {broker_command}"

        environment = {
            **self.spec.env,
            "POLAR_ALLOW_INTERNET": "true" if self.spec.allow_internet else "false",
            # These values select trusted control-plane code and paths.  Do not
            # let a RuntimeSpec environment redirect the supervisor to a
            # task-writable replacement.
            "POLAR_APPTAINER_BROKER_RUNTIME_DIR": _BROKER_RUNTIME_DIR,
            "POLAR_APPTAINER_BROKER_SCRIPT": _BROKER_TRUSTED_RUNTIME_SCRIPT,
            "POLAR_APPTAINER_BROKER_INTERPRETER_FILE": (
                f"{_BROKER_RUNTIME_DIR}/{_BROKER_INTERPRETER_NAME}"
            ),
            "POLAR_APPTAINER_BROKER_PROCESS_NAME": "polar-broker",
            "POLAR_APPTAINER_PROTECTED_SOCKET_NAME": (
                self._protected_broker_socket_name
            ),
        }
        environment.pop("POLAR_APPTAINER_BROKER_PYTHON", None)
        shell_exports = [
            f"export {key}={shlex.quote(str(environment[key]))};"
            for key in ("HOME", "PATH")
            if key in environment
        ]
        if shell_exports:
            broker_command = " ".join(shell_exports + [broker_command])
        args = [self._binary, "exec", *self._runtime_options(), self.spec.image]
        if environment:
            args.append("env")
            args.extend(f"{key}={value}" for key, value in environment.items())
        args.extend(["bash", "-lc", broker_command])
        return args

    async def _wait_for_broker_ready(self, timeout: float) -> None:
        # Capture the task once.  Concurrent early-stop cancellation clears
        # ``self._broker_task`` while this loop is polling; dereferencing the
        # mutable attribute on every iteration caused the observed
        # ``NoneType.done`` startup race.
        broker_task = self._broker_task
        if broker_task is None:
            raise RuntimeError("apptainer broker launch task is missing")
        deadline = asyncio.get_running_loop().time() + timeout
        last_error = "socket was not created"
        while asyncio.get_running_loop().time() < deadline:
            if broker_task.done():
                rc, stdout, stderr = broker_task.result()
                diagnostic = (
                    self._read_optional_text(self._broker_log) or stderr or stdout or ""
                ).strip()
                suffix = f": {diagnostic}" if diagnostic else ""
                raise RuntimeError(
                    f"apptainer broker exited during startup with exit code {rc}{suffix}"
                )
            if self._broker_socket.exists() and self._protected_broker_socket.exists():
                try:
                    await self._broker_rpc({"operation": "ping"}, socket_timeout=0.5)
                    await self._protected_broker_rpc(
                        {"operation": "secure_pin"},
                        socket_timeout=0.5,
                        pin=True,
                    )
                    return
                except (OSError, RuntimeError, ValueError) as exc:
                    last_error = str(exc)
            await asyncio.sleep(0.05)
        raise TimeoutError(
            f"apptainer broker did not become ready within {timeout:.0f}s: {last_error}"
        )

    async def _force_stop_broker(self) -> None:
        # Capture the escaped SID before canceling the launcher. The runtime
        # parent can exit during cancellation while its FUSE helpers remain
        # reparented to PID 1; after that, no surviving process is guaranteed
        # to retain ENGINE_CONFIG for ownership discovery.
        task = self._broker_task
        if task is None and not self._broker_was_started and not self._known_runtime_sessions:
            self._broker_socket.unlink(missing_ok=True)
            self._protected_broker_socket.unlink(missing_ok=True)
            return
        runtime_sessions, _ = await asyncio.to_thread(
            _direct_broker_process_snapshot,
            self.session_dir,
            known_sessions=self._known_runtime_sessions,
        )
        if task is not None and not task.done() and not runtime_sessions:
            raise RuntimeContainmentError(
                "could not identify the live Apptainer runtime session before teardown"
            )
        self._known_runtime_sessions.update(runtime_sessions)
        self._broker_task = None
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        # Apptainer's runtime parent deliberately escapes the launcher's
        # process group (setsid + clone-parent). Killing/reaping the Popen is
        # therefore insufficient: its squashfuse/fuse-overlayfs helpers can
        # stay alive under PID 1 and eventually make every later broker mount
        # time out. Resolve the escaped runtime by its unique ENGINE_CONFIG
        # session bind, then tear down every process in that SID/subtree.
        await asyncio.to_thread(
            self._cleanup_direct_broker_processes,
            known_sessions=self._known_runtime_sessions,
        )
        self._broker_socket.unlink(missing_ok=True)
        self._protected_broker_socket.unlink(missing_ok=True)
        self._broker_was_started = False

    def _cleanup_direct_broker_processes(
        self,
        *,
        proc_root: Path = _PROC_ROOT,
        term_grace_seconds: float = _DIRECT_BROKER_TERM_GRACE_SEC,
        kill_timeout_seconds: float = _DIRECT_BROKER_KILL_TIMEOUT_SEC,
        known_sessions: set[int] | None = None,
    ) -> None:
        runtime_sessions, processes = _direct_broker_process_snapshot(
            self.session_dir,
            proc_root=proc_root,
            known_sessions=known_sessions,
        )
        if processes:
            logger.warning(
                "Cleaning %d escaped Apptainer process(es) for %s (runtime SID(s): %s)",
                len(processes),
                self.session_id,
                ",".join(str(value) for value in sorted(runtime_sessions)),
            )
            _signal_proc_snapshot(processes, signal.SIGTERM, proc_root=proc_root)

        term_deadline = time.monotonic() + max(0.0, term_grace_seconds)
        remaining = processes
        while remaining and time.monotonic() < term_deadline:
            _reap_direct_children(remaining)
            runtime_sessions, remaining = _direct_broker_process_snapshot(
                self.session_dir,
                proc_root=proc_root,
                known_sessions=runtime_sessions,
            )
            time.sleep(_DIRECT_BROKER_CLEANUP_POLL_SEC)

        runtime_sessions, remaining = _direct_broker_process_snapshot(
            self.session_dir,
            proc_root=proc_root,
            known_sessions=runtime_sessions,
        )
        if remaining:
            _signal_proc_snapshot(remaining, signal.SIGKILL, proc_root=proc_root)

        kill_deadline = time.monotonic() + max(0.0, kill_timeout_seconds)
        while remaining and time.monotonic() < kill_deadline:
            _reap_direct_children(remaining)
            time.sleep(_DIRECT_BROKER_CLEANUP_POLL_SEC)
            runtime_sessions, remaining = _direct_broker_process_snapshot(
                self.session_dir,
                proc_root=proc_root,
                known_sessions=runtime_sessions,
            )
            if remaining:
                _signal_proc_snapshot(remaining, signal.SIGKILL, proc_root=proc_root)

        _reap_direct_children(remaining)
        stable_empty = 0
        proof_deadline = time.monotonic() + max(
            _DIRECT_BROKER_CLEANUP_POLL_SEC,
            kill_timeout_seconds,
        )
        while stable_empty < _DIRECT_BROKER_STABLE_EMPTY_PROBES:
            runtime_sessions, residual = _direct_broker_process_snapshot(
                self.session_dir,
                proc_root=proc_root,
                known_sessions=runtime_sessions,
            )
            if residual:
                stable_empty = 0
                _signal_proc_snapshot(residual, signal.SIGKILL, proc_root=proc_root)
                _reap_direct_children(residual)
            else:
                stable_empty += 1
            if stable_empty >= _DIRECT_BROKER_STABLE_EMPTY_PROBES:
                return
            if time.monotonic() >= proof_deadline:
                break
            time.sleep(_DIRECT_BROKER_CLEANUP_POLL_SEC)
        raise RuntimeContainmentError(
            "failed to prove escaped Apptainer processes were reaped for runtime "
            f"{self.session_id}"
        )

    async def _start_direct_broker_once(self) -> None:
        """Launch one broker and wait until its control socket is responsive."""

        configured_home = self.spec.env.get("HOME")
        if configured_home:
            host_home = self.resolve_host_path(configured_home)
            if host_home is not None:
                host_home.mkdir(parents=True, exist_ok=True)
        self._broker_dir.mkdir(parents=True, exist_ok=True)
        self._broker_results_dir.mkdir(parents=True, exist_ok=True)
        self._broker_transfers_dir.mkdir(parents=True, exist_ok=True)
        self._broker_socket.unlink(missing_ok=True)
        self._broker_log.unlink(missing_ok=True)
        args = self._broker_launch_args()
        self._broker_was_started = True
        self._broker_task = asyncio.create_task(
            self._run_local_command(
                *args,
                capture=True,
                cwd=_LOCAL_LAUNCHER_CWD,
            )
        )
        await self._wait_for_broker_ready(
            _bounded_env_float(
                _BROKER_START_TIMEOUT_ENV,
                _DEFAULT_BROKER_START_TIMEOUT_SEC,
                minimum=1.0,
                maximum=900.0,
            )
        )
        self._broker_observed_generation = self._read_broker_counter(self._broker_generation_file)

    async def _start_direct_broker(self) -> None:
        """Start one broker under mount-pressure admission control.

        Keep the permit through the socket ping and, on failure, through full
        process cleanup.  Releasing it after ``Popen`` (or before cleanup)
        merely moves the mount storm into Apptainer's asynchronous startup.
        """

        # Take the image-local permit first. Otherwise several starts waiting
        # on one busy SIF can occupy every aggregate permit and prevent starts
        # for independent images from making progress.
        async with _broker_start_per_image_gate(self.spec.image):
            async with _broker_start_gate():
                try:
                    await self._start_direct_broker_once()
                except BaseException:
                    await self._force_stop_broker()
                    raise

    def _broker_retry_backoff(self, attempt: int) -> float:
        """Return capped exponential backoff with per-session deterministic jitter."""

        base = _bounded_env_float(
            _BROKER_RETRY_BACKOFF_ENV,
            _DEFAULT_BROKER_RETRY_BACKOFF_SEC,
            minimum=0.0,
            maximum=300.0,
        )
        maximum = _bounded_env_float(
            _BROKER_RETRY_BACKOFF_MAX_ENV,
            _DEFAULT_BROKER_RETRY_BACKOFF_MAX_SEC,
            minimum=0.0,
            maximum=900.0,
        )
        if base == 0.0 or maximum == 0.0:
            return 0.0
        delay = min(base * (2 ** max(0, attempt - 1)), maximum)
        # Session UUIDs naturally spread retry admission.  Stable jitter also
        # keeps tests/replays deterministic and avoids a process-global RNG.
        digest = hashlib.sha256(f"{self.session_id}:{attempt}".encode()).digest()
        jitter = int.from_bytes(digest[:4], "big") / (2**32 - 1)
        return min(delay * (1.0 + 0.25 * jitter), maximum)

    async def start(self) -> None:
        if self._destroyed or self._teardown_started:
            raise RuntimeError("apptainer runtime was already destroyed")
        self._overlay_dir = self.session_dir / "overlay"
        self._overlay_dir.mkdir(parents=True, exist_ok=True)

        if not self._use_instance:
            if not self._use_persistent_broker:
                await self._start_legacy_direct_exec()
                return
            # Nested instance mode is unavailable, so retain one direct exec as
            # a namespace owner and route every command through its UDS broker.
            logger.info("Using brokered direct apptainer runtime for %s", self._instance_name)
            attempts = _env_int("POLAR_APPTAINER_DIRECT_EXEC_RETRIES", 2)
            last_error = ""
            for attempt in range(1, attempts + 1):
                try:
                    await self._start_direct_broker()
                    if self.spec.direct_exec_init_command:
                        result = await self.exec(
                            self.spec.direct_exec_init_command,
                            cwd=self.spec.workdir or self.runtime_session_dir,
                        )
                        if result.return_code != 0:
                            raise RuntimeError(
                                "persistent broker init command failed with exit code "
                                f"{result.return_code}: {result.stderr or ''}"
                            )
                    return
                except Exception as exc:
                    last_error = str(exc)
                if attempt < attempts:
                    retry_delay = self._broker_retry_backoff(attempt)
                    logger.warning(
                        "%s direct broker startup failed for %s image=%s "
                        "(attempt %d/%d; retrying in %.2fs): %s",
                        self._binary,
                        self._instance_name,
                        self.spec.image,
                        attempt,
                        attempts,
                        retry_delay,
                        last_error,
                    )
                    await asyncio.sleep(retry_delay)
            raise RuntimeError(
                f"{self._binary} direct broker startup failed for {self._instance_name} "
                f"image={self.spec.image} after {attempts} attempts: {last_error}"
            )

        args = [self._binary, "instance", "start", *self._runtime_options()]
        args.extend([self.spec.image, self._instance_name])
        rc, _, _ = await self._run_local_command(*args, cwd=_LOCAL_LAUNCHER_CWD)
        if rc != 0:
            raise RuntimeError(f"{self._binary} instance start failed with exit code {rc}")

    async def _start_legacy_direct_exec(self) -> None:
        """Validate the original fresh-``apptainer exec`` compatibility path."""

        logger.info("Using legacy direct apptainer exec runtime for %s", self._instance_name)
        args = [self._binary, "exec", *self._runtime_options(), self.spec.image, "true"]
        attempts = _env_int("POLAR_APPTAINER_DIRECT_EXEC_RETRIES", 3)
        last_rc = 0
        last_stderr = ""
        for attempt in range(1, attempts + 1):
            last_rc, _, stderr = await self._run_local_command(
                *args,
                capture=True,
                cwd=_LOCAL_LAUNCHER_CWD,
            )
            last_stderr = stderr or ""
            if last_rc == 0:
                return
            if attempt < attempts:
                logger.warning(
                    "%s legacy direct exec validation failed for %s image=%s "
                    "(attempt %d/%d, rc=%s): %s",
                    self._binary,
                    self._instance_name,
                    self.spec.image,
                    attempt,
                    attempts,
                    last_rc,
                    last_stderr.strip(),
                )
                await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 4.0))
        raise RuntimeError(
            f"{self._binary} legacy direct exec validation failed for "
            f"{self._instance_name} image={self.spec.image} after {attempts} "
            f"attempts with exit code {last_rc}: {last_stderr}"
        )

    _STOP_TIMEOUT = 30.0

    async def stop(self) -> None:
        async with self._teardown_lock:
            if self._destroyed:
                return
            self._teardown_started = True
            if not self._use_instance:
                if not self._use_persistent_broker:
                    # No namespace owner exists. The shared overlay remains under
                    # session_dir and is removed by normal session cleanup.
                    self._destroyed = True
                    return
                task = self._broker_task
                try:
                    if task is not None and not task.done():
                        await self._broker_rpc(
                            {"operation": "shutdown"}, socket_timeout=5.0
                        )
                    if task is not None:
                        rc, stdout, stderr = await asyncio.wait_for(
                            asyncio.shield(task), timeout=self._STOP_TIMEOUT
                        )
                        if rc != 0:
                            diagnostic = (
                                self._read_optional_text(self._broker_log)
                                or stderr
                                or stdout
                                or ""
                            ).strip()
                            logger.warning(
                                "%s direct broker exited for %s (rc=%s): %s",
                                self._binary,
                                self._instance_name,
                                rc,
                                diagnostic,
                            )
                except (OSError, RuntimeError, TimeoutError) as exc:
                    logger.warning(
                        "%s direct broker graceful stop failed for %s: %s",
                        self._binary,
                        self._instance_name,
                        exc,
                    )
                # This scan is the destruction proof. Run it even when an
                # earlier cancellation already cleared `_broker_task`; a prior
                # cleanup failure must remain retryable and observable.
                await self._force_stop_broker()
                self._destroyed = True
                return
            rc, _, stderr = await self._run_local_command(
                self._binary,
                "instance",
                "stop",
                self._instance_name,
                timeout=self._STOP_TIMEOUT,
                capture=True,
                cwd=_LOCAL_LAUNCHER_CWD,
            )
            if rc != 0:
                raise RuntimeContainmentError(
                    f"{self._binary} instance stop failed for {self._instance_name} "
                    f"with exit code {rc}: {stderr or ''}"
                )
            self._destroyed = True

    async def cancel(self) -> None:
        if self._use_instance or not self._use_persistent_broker:
            await super().cancel()
            return
        async with self._teardown_lock:
            if self._destroyed:
                return
            self._teardown_started = True
            await self._force_stop_broker()
            self._destroyed = True

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        if self._destroyed or self._teardown_started:
            raise RuntimeDestroyedError(
                f"apptainer runtime {self.runtime_id} was already destroyed"
            )
        started_at = time.perf_counter()
        return_code: int | None = None
        raised_exception = False
        cancelled = False
        effective_env = {**self.spec.env, **(env or {})}
        effective_workdir = cwd or self.spec.workdir or self.runtime_session_dir
        wrapped_command = command
        if (self._use_instance or not self._use_persistent_broker) and effective_workdir:
            wrapped_command = f"cd {shlex.quote(effective_workdir)} && {wrapped_command}"
        shell_exports = []
        for key in ("HOME", "PATH"):
            if key in effective_env:
                shell_exports.append(f"export {key}={shlex.quote(str(effective_env[key]))};")
        if shell_exports:
            wrapped_command = " ".join(shell_exports + [wrapped_command])
        if not self._use_instance and not self._use_persistent_broker:
            # A fresh exec has no owner shell in which to initialize services.
            # Re-run the same hook inside each workload shell so any background
            # helpers live for the complete command that needs them.
            if self.spec.direct_exec_init_command:
                wrapped_command = (
                    f"{{ {self.spec.direct_exec_init_command}; }} && {wrapped_command}"
                )
        try:
            if self._use_instance or not self._use_persistent_broker:
                args = list(self._exec_base_args())
                if effective_env:
                    args.append("env")
                    args.extend(f"{key}={value}" for key, value in effective_env.items())
                args.extend(["bash", "-lc", wrapped_command])
                rc, stdout, stderr = await self._run_local_command(
                    *args,
                    timeout=timeout_sec,
                    capture=True,
                    cwd=_LOCAL_LAUNCHER_CWD,
                )
            else:
                rc, stdout, stderr = await self._broker_exec(
                    wrapped_command,
                    cwd=effective_workdir,
                    env=effective_env,
                    timeout_sec=timeout_sec,
                )
            return_code = rc
            return ExecResult(stdout=stdout, stderr=stderr, return_code=rc)
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            raised_exception = True
            raise
        finally:
            self._record_exec_timing(
                command,
                started_at,
                return_code,
                raised_exception=raised_exception,
                cancelled=cancelled,
            )

    async def exec_protected(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        protected_env: dict[str, str] | None = None,
        protected_file_digests: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        if self._destroyed or self._teardown_started:
            raise RuntimeDestroyedError(
                f"apptainer runtime {self.runtime_id} was already destroyed"
            )
        if self._use_instance or not self._use_persistent_broker:
            raise RuntimeError(
                "protected exec requires the persistent Apptainer broker backend"
            )
        if not protected_env:
            raise RuntimeError("protected exec requires protected environment values")
        if not protected_file_digests:
            raise RuntimeError("protected exec requires pinned file digests")
        effective_env = {**self.spec.env, **(env or {})}
        for key in protected_env:
            effective_env.pop(key, None)
        effective_workdir = cwd or self.spec.workdir or self.runtime_session_dir
        started_at = time.perf_counter()
        return_code: int | None = None
        raised_exception = False
        cancelled = False
        try:
            rc, stdout, stderr = await self._broker_protected_exec(
                argv,
                cwd=effective_workdir,
                env=effective_env,
                protected_env=protected_env,
                protected_file_digests=protected_file_digests,
                timeout_sec=timeout_sec,
            )
            return_code = rc
            return ExecResult(stdout=stdout, stderr=stderr, return_code=rc)
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            raised_exception = True
            raise
        finally:
            self._record_exec_timing(
                "protected_exec",
                started_at,
                return_code,
                raised_exception=raised_exception,
                cancelled=cancelled,
            )

    async def _broker_exec(
        self,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str],
        timeout_sec: float | None,
    ) -> tuple[int, str | None, str | None]:
        await self._ensure_direct_broker_ready_for_exec()
        request_id = uuid.uuid4().hex
        request: dict[str, object] = {
            "operation": "exec",
            "id": request_id,
            "command": command,
            "cwd": cwd,
            "env": env,
            "timeout_sec": timeout_sec,
        }
        return await self._broker_exec_request(
            request_id,
            request,
            timeout_sec=timeout_sec,
            protected=False,
        )

    async def _broker_protected_exec(
        self,
        argv: list[str],
        *,
        cwd: str | None,
        env: dict[str, str],
        protected_env: dict[str, str],
        protected_file_digests: dict[str, str],
        timeout_sec: float | None,
    ) -> tuple[int, str | None, str | None]:
        self._assert_direct_broker_owner_live()
        request_id = uuid.uuid4().hex
        request: dict[str, object] = {
            "operation": "exec_protected",
            "id": request_id,
            "argv": argv,
            "cwd": cwd,
            "env": env,
            "protected_env": dict(protected_env),
            "file_digests": dict(protected_file_digests),
            "timeout_sec": timeout_sec,
        }
        try:
            return await self._broker_exec_request(
                request_id,
                request,
                timeout_sec=timeout_sec,
                protected=True,
            )
        finally:
            protected_values = request.get("protected_env")
            if isinstance(protected_values, dict):
                protected_values.clear()

    async def _broker_exec_request(
        self,
        request_id: str,
        request: dict[str, object],
        *,
        timeout_sec: float | None,
        protected: bool,
    ) -> tuple[int, str | None, str | None]:
        stdout_path = self._broker_results_dir / f"{request_id}.stdout"
        stderr_path = self._broker_results_dir / f"{request_id}.stderr"
        socket_timeout = None if timeout_sec is None else timeout_sec + 30.0
        rpc = self._protected_broker_rpc if protected else self._broker_rpc
        rpc_task = asyncio.create_task(rpc(request, socket_timeout=socket_timeout))
        try:
            response = await asyncio.shield(rpc_task)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(
                    rpc(
                        {"operation": "cancel", "id": request_id},
                        socket_timeout=5.0,
                    )
                )
            except (OSError, RuntimeError, TimeoutError):
                pass
            stdout_path.unlink(missing_ok=True)
            stderr_path.unlink(missing_ok=True)

            def consume_rpc_result(task: asyncio.Task[dict[str, Any]]) -> None:
                try:
                    task.result()
                except BaseException:
                    pass
                finally:
                    stdout_path.unlink(missing_ok=True)
                    stderr_path.unlink(missing_ok=True)

            rpc_task.add_done_callback(consume_rpc_result)
            raise
        except (_BrokerDisconnectedError, OSError) as exc:
            if protected:
                stdout_path.unlink(missing_ok=True)
                stderr_path.unlink(missing_ok=True)
                raise RuntimeError("protected broker channel disconnected") from exc
            self._broker_disconnect_count += 1
            try:
                await self._recover_live_broker(trigger="mid_exec_disconnect", error=exc)
                stdout = await asyncio.to_thread(self._read_optional_text, stdout_path)
                stderr = await asyncio.to_thread(self._read_optional_text, stderr_path)
                diagnostic = (
                    "polar apptainer broker recovered after the command terminated "
                    f"its control process: {exc}"
                )
                stderr = (
                    f"{stderr.rstrip()}\n{diagnostic}\n"
                    if stderr
                    else f"{diagnostic}\n"
                )
                return 125, stdout, stderr
            finally:
                stdout_path.unlink(missing_ok=True)
                stderr_path.unlink(missing_ok=True)
        except Exception:
            stdout_path.unlink(missing_ok=True)
            stderr_path.unlink(missing_ok=True)
            raise
        try:
            return_code = int(response["return_code"])
            stdout = await asyncio.to_thread(self._read_optional_text, stdout_path)
            stderr = await asyncio.to_thread(self._read_optional_text, stderr_path)
            return return_code, stdout, stderr
        finally:
            stdout_path.unlink(missing_ok=True)
            stderr_path.unlink(missing_ok=True)

    @staticmethod
    def _read_optional_text(path: Path) -> str | None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        return data.decode(errors="replace") if data else None

    def exec_timing_summary(self) -> dict[str, object]:
        """Include bounded broker recovery work in runtime telemetry."""

        summary = super().exec_timing_summary()
        summary.update(
            {
                "broker_recovery_ms": self._broker_recovery_ms,
                "broker_recovery_count": self._broker_recovery_count,
                "broker_recovery_failure_count": self._broker_recovery_failure_count,
                "broker_preflight_failure_count": self._broker_preflight_failure_count,
                "broker_disconnect_count": self._broker_disconnect_count,
                "broker_supervisor_restart_count": self._read_broker_counter(
                    self._broker_restart_count_file
                ),
                "broker_generation": self._read_broker_counter(self._broker_generation_file),
            }
        )
        return summary

    def _new_transfer_archive(self) -> tuple[Path, str]:
        self._broker_transfers_dir.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}.tar"
        return (
            self._broker_transfers_dir / name,
            f"{_BROKER_RUNTIME_TRANSFERS_DIR}/{name}",
        )

    @staticmethod
    def _archive_upload_source(
        source: Path,
        archive_path: Path,
        *,
        destination_name: str | None,
    ) -> None:
        with tarfile.open(archive_path, mode="w", dereference=False) as archive:
            if source.is_dir() and destination_name is None:
                for entry in source.iterdir():
                    archive.add(entry, arcname=entry.name, recursive=True)
            else:
                archive.add(
                    source,
                    arcname=destination_name or source.name,
                    recursive=True,
                )

    @staticmethod
    def _extract_download_archive(archive_path: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, mode="r") as archive:
            # The archive is produced from an untrusted task filesystem.  The
            # data filter blocks absolute paths, traversal, devices, and links
            # that escape the requested host destination.
            archive.extractall(destination, filter="data")

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        # Bind-mounted transfers can recursively copy large verifier trees on
        # Lustre.  Keep that blocking filesystem work off the gateway's only
        # asyncio event loop so health/cancel/callback traffic stays responsive
        # while many sessions enter post-run together.
        if await asyncio.to_thread(self._copy_to_bind_mount, local_path, remote_path):
            return
        parent = str(Path(remote_path).parent)
        destination_name = Path(remote_path).name
        result = await self.exec(f"mkdir -p {shlex.quote(parent)}")
        if result.return_code != 0:
            raise RuntimeError(
                _exec_failure_message(
                    f"failed to create directory {parent} in runtime",
                    result,
                )
            )
        archive_path, runtime_archive = self._new_transfer_archive()
        try:
            await asyncio.to_thread(
                self._archive_upload_source,
                Path(local_path),
                archive_path,
                destination_name=destination_name,
            )
            result = await self.exec(
                f"tar -xf {shlex.quote(runtime_archive)} -C {shlex.quote(parent)}"
            )
            if result.return_code != 0:
                raise RuntimeError(_exec_failure_message("apptainer upload_file", result))
        finally:
            archive_path.unlink(missing_ok=True)

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        if await asyncio.to_thread(self._copy_to_bind_mount, local_path, remote_path):
            return
        result = await self.exec(f"mkdir -p {shlex.quote(remote_path)}")
        if result.return_code != 0:
            raise RuntimeError(
                _exec_failure_message(
                    f"failed to create directory {remote_path} in runtime",
                    result,
                )
            )
        archive_path, runtime_archive = self._new_transfer_archive()
        try:
            await asyncio.to_thread(
                self._archive_upload_source,
                Path(local_path),
                archive_path,
                destination_name=None,
            )
            result = await self.exec(
                f"tar -xf {shlex.quote(runtime_archive)} -C {shlex.quote(remote_path)}"
            )
            if result.return_code != 0:
                raise RuntimeError(_exec_failure_message("apptainer upload_dir", result))
        finally:
            archive_path.unlink(missing_ok=True)

    async def download_file(self, remote_path: str, local_path: str) -> None:
        if await asyncio.to_thread(self._copy_from_bind_mount, remote_path, Path(local_path)):
            return
        parent = str(Path(remote_path).parent)
        filename = Path(remote_path).name
        archive_path, runtime_archive = self._new_transfer_archive()
        extraction_dir = Path(tempfile.mkdtemp(prefix="download-", dir=self._broker_transfers_dir))
        try:
            result = await self.exec(
                f"tar -cf {shlex.quote(runtime_archive)} "
                f"-C {shlex.quote(parent)} {shlex.quote(f'./{filename}')}"
            )
            if result.return_code != 0:
                raise RuntimeError(_exec_failure_message("apptainer download_file", result))
            await asyncio.to_thread(self._extract_download_archive, archive_path, extraction_dir)
            extracted = extraction_dir / filename
            if not extracted.exists() and not extracted.is_symlink():
                raise RuntimeError(f"apptainer download_file archive did not contain {filename!r}")
            destination = Path(local_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copy2, extracted, destination)
        finally:
            archive_path.unlink(missing_ok=True)
            await asyncio.to_thread(shutil.rmtree, extraction_dir, True)

    async def download_dir(self, remote_path: str, local_path: str) -> None:
        if await asyncio.to_thread(self._copy_from_bind_mount, remote_path, Path(local_path)):
            return
        Path(local_path).mkdir(parents=True, exist_ok=True)
        archive_path, runtime_archive = self._new_transfer_archive()
        try:
            result = await self.exec(
                f"tar -cf {shlex.quote(runtime_archive)} -C {shlex.quote(remote_path)} ."
            )
            if result.return_code != 0:
                raise RuntimeError(_exec_failure_message("apptainer download_dir", result))
            await asyncio.to_thread(
                self._extract_download_archive,
                archive_path,
                Path(local_path),
            )
        finally:
            archive_path.unlink(missing_ok=True)

    @staticmethod
    def _resolve_binary() -> str:
        override = os.environ.get("POLAR_APPTAINER_BIN")
        if override:
            return override
        for candidate in ("/usr/bin/apptainer", "/bin/apptainer"):
            if Path(candidate).is_file():
                return candidate
        resolved = shutil.which("apptainer")
        if resolved:
            return resolved
        return "apptainer"
