"""Runtime abstraction for container-backed rollout execution."""

from __future__ import annotations

import asyncio
import math
import os
import signal
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final, TypeVar

from polar.runtime.models import ExecResult, RuntimeSpec
from polar.runtime.command_timing import (
    RUNTIME_EXEC_CATEGORIES,
    classify_runtime_command,
)

RUNTIME_SESSION_DIR: Final[str] = "/polar/session"
RUNTIME_ARTIFACTS_DIR: Final[str] = f"{RUNTIME_SESSION_DIR}/artifacts"
RUNTIME_LOGS_DIR: Final[str] = f"{RUNTIME_SESSION_DIR}/logs"
RUNTIME_AGENT_LOG_DIR: Final[str] = f"{RUNTIME_LOGS_DIR}/agent"
RUNTIME_EVAL_LOG_DIR: Final[str] = f"{RUNTIME_LOGS_DIR}/eval"
RUNTIME_EVAL_ARTIFACT_DIR: Final[str] = f"{RUNTIME_SESSION_DIR}/eval_artifacts"

_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY_ENV: Final[str] = "POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY"
_DEFAULT_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY: Final[int] = 4
_LOCAL_SUBPROCESS_SPAWN_GATE_ATTR: Final[str] = "_polar_local_subprocess_spawn_gate"
_LOCAL_SUBPROCESS_SPAWN_GATE_SIZE_ATTR: Final[str] = "_polar_local_subprocess_spawn_gate_size"
# The portable Python used on cluster nodes may not expose os.pidfd_open.  A
# short fallback interval created thousands of wait4(WNOHANG) callbacks per
# second with hundreds of live runtimes, so trade sub-second reap detection for
# a responsive gateway control plane.
_LOCAL_SUBPROCESS_POLL_INTERVAL_SEC: Final[float] = 0.25

_T = TypeVar("_T")


class RuntimeDestroyedError(RuntimeError):
    """Raised when work is submitted after a runtime has been torn down."""


def _local_subprocess_spawn_concurrency() -> int:
    raw_value = os.environ.get(_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY_ENV)
    if raw_value is None:
        return _DEFAULT_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY_ENV} must be a positive "
            f"integer, got {raw_value!r}"
        ) from exc
    if value <= 0 or value > 1024:
        raise ValueError(
            f"{_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY_ENV} must be between 1 and 1024, got {value}"
        )
    return value


def _local_subprocess_spawn_gate() -> asyncio.Semaphore:
    """Return the spawn admission gate shared by every runtime on this loop."""

    loop = asyncio.get_running_loop()
    concurrency = _local_subprocess_spawn_concurrency()
    gate = getattr(loop, _LOCAL_SUBPROCESS_SPAWN_GATE_ATTR, None)
    configured_size = getattr(loop, _LOCAL_SUBPROCESS_SPAWN_GATE_SIZE_ATTR, None)
    if gate is None:
        gate = asyncio.Semaphore(concurrency)
        setattr(loop, _LOCAL_SUBPROCESS_SPAWN_GATE_ATTR, gate)
        setattr(loop, _LOCAL_SUBPROCESS_SPAWN_GATE_SIZE_ATTR, concurrency)
    elif configured_size != concurrency:
        raise RuntimeError(
            f"{_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY_ENV} changed from "
            f"{configured_size} to {concurrency} after this event loop started"
        )
    return gate


async def _await_task_despite_cancellation(task: asyncio.Future[_T]) -> _T:
    """Finish a cleanup task even if its caller is cancelled again."""

    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


class BaseRuntime(ABC):
    """Base class for long-lived per-session execution runtimes."""

    def __init__(self, spec: RuntimeSpec, session_id: str, session_dir: Path) -> None:
        self.spec = spec
        self.session_id = session_id
        self.session_dir = session_dir
        self.artifacts_dir = session_dir / "artifacts"
        self.runtime_session_dir = RUNTIME_SESSION_DIR
        self.runtime_artifacts_dir = RUNTIME_ARTIFACTS_DIR
        self.runtime_logs_dir = RUNTIME_LOGS_DIR
        self.runtime_agent_log_dir = RUNTIME_AGENT_LOG_DIR
        self._active_process: subprocess.Popen[bytes] | None = None
        self._active_spawn_task: asyncio.Task[subprocess.Popen[bytes]] | None = None
        self._destroyed = False
        self._exec_timing_ms = {category: 0.0 for category in RUNTIME_EXEC_CATEGORIES}
        self._exec_timing_count = {category: 0 for category in RUNTIME_EXEC_CATEGORIES}
        self._exec_timeout_count = 0
        self._exec_failure_count = 0
        self._exec_exception_count = 0
        self._exec_cancelled_count = 0

    @property
    def destroyed(self) -> bool:
        """Whether teardown has made this runtime unavailable for new work."""

        return self._destroyed

    @property
    @abstractmethod
    def runtime_id(self) -> str:
        """Identifier for the live runtime instance."""

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        return False

    @property
    def supports_cpu_limits(self) -> bool:
        return False

    @property
    def supports_memory_limits(self) -> bool:
        return False

    @property
    def supports_storage_limits(self) -> bool:
        return False

    @abstractmethod
    async def start(self) -> None:
        """Create and start the runtime instance."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop and remove the runtime instance."""

    async def cancel(self) -> None:
        """Stop any in-flight command and tear the runtime down."""
        process = self._active_process
        spawn_task = self._active_spawn_task
        if process is None and spawn_task is not None:
            # Popen runs in a worker thread and cannot itself be cancelled. If
            # cancellation races spawn, wait for its handle so the child can
            # still be killed and reaped instead of becoming an orphan.
            try:
                process = await asyncio.shield(spawn_task)
            except Exception:
                process = None
        if process is not None and process.returncode is None:
            await self._kill_and_wait(process)
        await self.stop()

    @abstractmethod
    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        """Execute one command inside the runtime and return captured output."""

    @abstractmethod
    async def upload_file(self, local_path: str, remote_path: str) -> None:
        """Copy a single file from the host into the runtime."""

    @abstractmethod
    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        """Copy a directory tree from the host into the runtime."""

    @abstractmethod
    async def download_file(self, remote_path: str, local_path: str) -> None:
        """Copy a single file from inside the runtime to the host."""

    @abstractmethod
    async def download_dir(self, remote_path: str, local_path: str) -> None:
        """Copy a directory tree from inside the runtime to the host."""

    def resolve_host_path(self, runtime_path: str) -> Path | None:
        """Map a runtime path back to a host path via the session bind mount."""
        normalized = Path(runtime_path)
        runtime_root = Path(RUNTIME_SESSION_DIR)
        try:
            relative = normalized.relative_to(runtime_root)
        except ValueError:
            return None
        return self.session_dir / relative

    def _record_exec_timing(
        self,
        command: str,
        started_at: float,
        return_code: int | None,
        *,
        raised_exception: bool = False,
        cancelled: bool = False,
    ) -> None:
        """Record one runtime exec without retaining its command text."""

        duration_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        if not math.isfinite(duration_ms):
            return
        category = classify_runtime_command(command)
        self._exec_timing_ms[category] += duration_ms
        self._exec_timing_count[category] += 1
        if cancelled:
            self._exec_cancelled_count += 1
        elif raised_exception:
            self._exec_exception_count += 1
            self._exec_failure_count += 1
        elif return_code == -1:
            self._exec_timeout_count += 1
        elif return_code not in (None, 0):
            self._exec_failure_count += 1

    def exec_timing_summary(self) -> dict[str, object]:
        """Return a fixed-schema aggregate suitable for session telemetry."""

        return {
            "total_ms": sum(self._exec_timing_ms.values()),
            "count": sum(self._exec_timing_count.values()),
            "timeout_count": self._exec_timeout_count,
            "failure_count": self._exec_failure_count,
            "exception_count": self._exec_exception_count,
            "cancelled_count": self._exec_cancelled_count,
            "ms_by_category": dict(self._exec_timing_ms),
            "count_by_category": dict(self._exec_timing_count),
        }

    def _copy_from_bind_mount(self, runtime_path: str, local_path: Path) -> bool:
        host_path = self.resolve_host_path(runtime_path)
        if host_path is None or not host_path.exists():
            return False
        if host_path.is_dir():
            if local_path.exists():
                shutil.rmtree(local_path)
            shutil.copytree(host_path, local_path)
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(host_path, local_path)
        return True

    def _copy_to_bind_mount(self, local_path: str, runtime_path: str) -> bool:
        host_path = self.resolve_host_path(runtime_path)
        if host_path is None:
            return False
        source = Path(local_path)
        if not source.exists():
            raise FileNotFoundError(f"source path does not exist: {local_path}")
        if source.is_dir():
            if host_path.exists():
                shutil.rmtree(host_path)
            shutil.copytree(source, host_path)
        else:
            host_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, host_path)
        return True

    async def _run_local_command(
        self,
        *args: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        capture: bool = False,
        cwd: str | Path | None = None,
    ) -> tuple[int, str | None, str | None]:
        """Run a local subprocess, optionally capturing stdout/stderr."""
        process_env = None if env is None else {**os.environ, **env}
        if capture:
            stdout_target = subprocess.PIPE
            stderr_target = subprocess.PIPE
        else:
            stdout_target = subprocess.DEVNULL
            stderr_target = subprocess.DEVNULL

        process: subprocess.Popen[bytes] | None = None
        try:
            # asyncio.create_subprocess_exec performs clone/pipe/child-watcher
            # setup synchronously on the event-loop thread. Under hundreds of
            # simultaneous Apptainer commands that can starve health, callback,
            # and cancellation traffic for seconds. Popen is admitted through
            # the shared gate but the blocking spawn itself runs in a worker.
            async with _local_subprocess_spawn_gate():
                spawn_task = asyncio.create_task(
                    asyncio.to_thread(
                        subprocess.Popen,
                        args,
                        env=process_env,
                        cwd=cwd,
                        start_new_session=True,
                        stdout=stdout_target,
                        stderr=stderr_target,
                        bufsize=0,
                    )
                )
                self._active_spawn_task = spawn_task
                try:
                    process = await asyncio.shield(spawn_task)
                except asyncio.CancelledError as cancellation:
                    # Cancelling to_thread does not stop Popen. Recover the
                    # eventual process handle before propagating cancellation.
                    try:
                        process = await _await_task_despite_cancellation(spawn_task)
                    except Exception:
                        raise cancellation
                    self._active_process = process
                    raise cancellation
                finally:
                    if self._active_spawn_task is spawn_task:
                        self._active_spawn_task = None
                self._active_process = process

            assert process is not None
            try:
                if timeout is None:
                    stdout_bytes, stderr_bytes = await self._communicate(process)
                else:
                    async with asyncio.timeout(timeout):
                        stdout_bytes, stderr_bytes = await self._communicate(process)
            except TimeoutError:
                await self._kill_and_wait(process)
                self._close_process_pipes(process)
                return -1, None, None
            except Exception:
                await self._kill_and_wait(process)
                self._close_process_pipes(process)
                raise
            self._close_process_pipes(process)
        except asyncio.CancelledError:
            # Do not clear _active_process until the command has actually been
            # killed and reaped. Early-stop cancellation otherwise loses the
            # only handle while an agent/container command keeps consuming
            # resources in the background.
            if process is not None:
                cleanup_task = asyncio.create_task(self._kill_and_wait(process))
                await _await_task_despite_cancellation(cleanup_task)
                self._close_process_pipes(process)
            raise
        finally:
            if self._active_process is process:
                self._active_process = None

        assert process is not None
        rc = process.returncode or 0
        stdout_str = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr_str = stderr_bytes.decode(errors="replace") if stderr_bytes else None
        return rc, stdout_str, stderr_str

    @staticmethod
    async def _communicate(
        process: subprocess.Popen[bytes],
    ) -> tuple[bytes | None, bytes | None]:
        """Capture pipes and reap a Popen without blocking an executor thread."""

        tasks: list[asyncio.Task[object]] = [
            asyncio.create_task(BaseRuntime._wait_for_process(process))
        ]
        stdout_index: int | None = None
        stderr_index: int | None = None
        if process.stdout is not None:
            stdout_index = len(tasks)
            tasks.append(asyncio.create_task(BaseRuntime._read_pipe(process.stdout)))
        if process.stderr is not None:
            stderr_index = len(tasks)
            tasks.append(asyncio.create_task(BaseRuntime._read_pipe(process.stderr)))
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            cleanup = asyncio.gather(*tasks, return_exceptions=True)
            await _await_task_despite_cancellation(cleanup)
            raise

        stdout = results[stdout_index] if stdout_index is not None else None
        stderr = results[stderr_index] if stderr_index is not None else None
        assert stdout is None or isinstance(stdout, bytes)
        assert stderr is None or isinstance(stderr, bytes)
        return stdout, stderr

    @staticmethod
    async def _read_pipe(pipe: object) -> bytes:
        """Read one Popen pipe with event-loop readiness notifications."""

        fileno = getattr(pipe, "fileno")
        fd = fileno()
        os.set_blocking(fd, False)
        loop = asyncio.get_running_loop()
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(fd, 64 * 1024)
            except BlockingIOError:
                readable = loop.create_future()

                def mark_readable() -> None:
                    loop.remove_reader(fd)
                    if not readable.done():
                        readable.set_result(None)

                loop.add_reader(fd, mark_readable)
                try:
                    await readable
                finally:
                    loop.remove_reader(fd)
                continue
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            await asyncio.sleep(0)

    @staticmethod
    async def _wait_for_process(process: subprocess.Popen[bytes]) -> int:
        """Wait for Popen exit without dedicating one worker per live command."""

        if process.returncode is not None:
            return process.returncode
        loop = asyncio.get_running_loop()
        pidfd: int | None = None
        try:
            pid = getattr(process, "pid", None)
            pidfd_open = getattr(os, "pidfd_open", None)
            if pidfd_open is not None and isinstance(pid, int) and pid > 0:
                try:
                    pidfd = pidfd_open(pid)
                except OSError:
                    pidfd = None
            if pidfd is not None:
                exited = loop.create_future()

                def mark_exited() -> None:
                    assert pidfd is not None
                    loop.remove_reader(pidfd)
                    if not exited.done():
                        exited.set_result(None)

                loop.add_reader(pidfd, mark_exited)
                try:
                    await exited
                finally:
                    loop.remove_reader(pidfd)
                return process.wait()

            while process.poll() is None:
                await asyncio.sleep(_LOCAL_SUBPROCESS_POLL_INTERVAL_SEC)
            assert process.returncode is not None
            return process.returncode
        finally:
            if pidfd is not None:
                os.close(pidfd)

    @staticmethod
    async def _kill_and_wait(process: subprocess.Popen[bytes]) -> None:
        """Best-effort kill/reap when process exit races timeout or cancel."""
        try:
            pid = getattr(process, "pid", None)
            if isinstance(pid, int) and pid > 0:
                os.killpg(pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            # The child exited between the returncode check/timeout and kill.
            pass
        try:
            await BaseRuntime._wait_for_process(process)
        except ProcessLookupError:
            pass

    @staticmethod
    def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
