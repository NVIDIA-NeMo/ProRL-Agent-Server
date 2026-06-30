"""Docker-backed rollout runtime."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult, RuntimeSpec

logger = logging.getLogger(__name__)


class DockerRuntime(BaseRuntime):
    """Long-lived Docker container used across init, run, and post-run."""

    def __init__(self, spec: RuntimeSpec, session_id: str, session_dir: Path) -> None:
        super().__init__(spec, session_id, session_dir)
        # Use enough of the session_id to preserve the "-eval" suffix used by
        # fresh evaluator runtimes, avoiding collisions with the agent runtime.
        safe_name = session_id.replace("/", "-")[:55]
        self._container_name = f"polar-{safe_name}"
        self._chmod_needed: bool | None = None

    @property
    def runtime_id(self) -> str:
        return self._container_name

    @property
    def supports_gpus(self) -> bool:
        return True

    @property
    def can_disable_internet(self) -> bool:
        return True

    @property
    def supports_cpu_limits(self) -> bool:
        return True

    @property
    def supports_memory_limits(self) -> bool:
        return True

    async def start(self) -> None:
        if self._destroyed:
            raise RuntimeError("docker runtime was already destroyed")
        create_args = ["docker", "create", "--name", self._container_name]
        if not self.spec.allow_internet:
            create_args.extend(["--network", "none"])
        elif self.spec.network:
            create_args.extend(["--network", self.spec.network])
        if self.spec.gpus > 0:
            create_args.extend(["--gpus", str(self.spec.gpus)])
        if self.spec.cpus is not None:
            create_args.extend(["--cpus", str(self.spec.cpus)])
        if self.spec.memory_mb is not None:
            create_args.extend(["--memory", f"{self.spec.memory_mb}m"])
        create_args.extend(["-v", f"{self.session_dir}:{self.runtime_session_dir}"])
        # Additional volumes from kwargs (e.g., Docker socket for agents that need DinD)
        for vol in self.spec.kwargs.get("volumes", []):
            create_args.extend(["-v", vol])
        if self.spec.allow_internet:
            for vol in self.spec.internet_volumes:
                create_args.extend(["-v", vol])
        create_args.extend([self.spec.image, "sleep", "infinity"])
        rc, _, stderr = await self._run_local_command(
            *create_args, capture=True, timeout=self._START_TIMEOUT,
        )
        if rc != 0:
            raise RuntimeError(f"docker create failed with exit code {rc}: {stderr}")
        rc, _, stderr = await self._run_local_command(
            "docker", "start", self._container_name,
            capture=True, timeout=self._START_TIMEOUT,
        )
        if rc != 0:
            await self.stop()
            raise RuntimeError(f"docker start failed with exit code {rc}: {stderr}")
        # Skip the chmod when container and host UIDs match — recursive chmod
        # over a large session dir can be expensive and is only needed when the
        # container user can't write to host-owned bind-mounted files.
        self._chmod_needed = await self._detect_chmod_needed()
        if self._chmod_needed:
            await self._run_local_command(
                "docker", "exec", "--user", "root",
                self._container_name, "chmod", "-R", "a+rwX", self.runtime_session_dir,
                timeout=self._STOP_TIMEOUT,
            )

    _START_TIMEOUT = 600.0  # seconds for docker create / start under high rollout load
    _STOP_TIMEOUT = 30.0  # seconds per cleanup command

    async def stop(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        # chmod is best-effort so the host can reclaim bind-mounted files.
        # Skip when UIDs match (no permission mismatch to resolve).
        if self._chmod_needed is not False:
            try:
                await self._run_local_command(
                    "docker", "exec", "--user", "root",
                    self._container_name, "chmod", "-R", "a+rwX",
                    self.runtime_session_dir,
                    timeout=self._STOP_TIMEOUT,
                )
            except Exception:
                logger.warning("chmod cleanup failed for %s", self._container_name)
        # kill first (instant SIGKILL), then rm to remove metadata.
        await self._run_local_command(
            "docker", "kill", self._container_name,
            timeout=self._STOP_TIMEOUT,
        )
        rc, _, stderr = await self._run_local_command(
            "docker", "rm", "-f", self._container_name,
            timeout=self._STOP_TIMEOUT, capture=True,
        )
        if rc != 0:
            logger.warning(
                "docker rm -f failed for %s (rc=%s): %s",
                self._container_name, rc, stderr,
            )

    async def _detect_chmod_needed(self) -> bool:
        """True unless the container's effective UID matches the host's."""
        rc, stdout, _ = await self._run_local_command(
            "docker", "exec", self._container_name, "id", "-u",
            capture=True, timeout=self._STOP_TIMEOUT,
        )
        if rc != 0:
            return True
        try:
            return int(stdout.strip()) != os.getuid()
        except ValueError:
            return True

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        started_at = time.perf_counter()
        return_code: int | None = None
        raised_exception = False
        cancelled = False
        args = ["docker", "exec"]
        effective_workdir = cwd or self.spec.workdir or self.runtime_session_dir
        if effective_workdir:
            args.extend(["-w", effective_workdir])
        for key, value in (env or {}).items():
            args.extend(["-e", f"{key}={value}"])
        args.extend([self._container_name, "bash", "-lc", command])
        try:
            rc, stdout, stderr = await self._run_local_command(
                *args, timeout=timeout_sec, capture=True
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

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        try:
            if await asyncio.to_thread(
                self._copy_to_bind_mount, local_path, remote_path
            ):
                await self._make_runtime_path_writable(remote_path, recursive=False)
                return
        except PermissionError:
            pass
        parent = str(Path(remote_path).parent)
        await self._run_local_command(
            "docker", "exec", self._container_name, "mkdir", "-p", parent
        )
        rc, _, _ = await self._run_local_command(
            "docker", "cp", local_path, f"{self._container_name}:{remote_path}"
        )
        if rc != 0:
            raise RuntimeError(f"docker cp upload_file failed with exit code {rc}")
        await self._make_runtime_path_writable(remote_path, recursive=False)

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        try:
            if await asyncio.to_thread(
                self._copy_to_bind_mount, local_path, remote_path
            ):
                await self._make_runtime_path_writable(remote_path, recursive=True)
                return
        except PermissionError:
            pass
        await self._run_local_command(
            "docker", "exec", self._container_name, "mkdir", "-p", remote_path
        )
        rc, _, _ = await self._run_local_command(
            "docker", "cp", f"{local_path}/.", f"{self._container_name}:{remote_path}"
        )
        if rc != 0:
            raise RuntimeError(f"docker cp upload_dir failed with exit code {rc}")
        await self._make_runtime_path_writable(remote_path, recursive=True)

    async def _make_runtime_path_writable(
        self, remote_path: str, *, recursive: bool
    ) -> None:
        if self._chmod_needed is False:
            return
        chmod_args = ["chmod"]
        if recursive:
            chmod_args.append("-R")
        chmod_args.extend(["a+rwX", remote_path])
        rc, _, stderr = await self._run_local_command(
            "docker", "exec", "--user", "root",
            self._container_name, *chmod_args,
            capture=True, timeout=self._STOP_TIMEOUT,
        )
        if rc != 0:
            raise RuntimeError(
                f"docker chmod failed for {remote_path} with exit code {rc}: {stderr}"
            )

    async def download_file(self, remote_path: str, local_path: str) -> None:
        try:
            if await asyncio.to_thread(
                self._copy_from_bind_mount, remote_path, Path(local_path)
            ):
                return
        except PermissionError:
            pass
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        rc, _, _ = await self._run_local_command(
            "docker", "cp", f"{self._container_name}:{remote_path}", local_path
        )
        if rc != 0:
            raise RuntimeError(f"docker cp download_file failed with exit code {rc}")

    async def download_dir(self, remote_path: str, local_path: str) -> None:
        try:
            if await asyncio.to_thread(
                self._copy_from_bind_mount, remote_path, Path(local_path)
            ):
                return
        except PermissionError:
            pass
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        rc, _, _ = await self._run_local_command(
            "docker", "cp", f"{self._container_name}:{remote_path}", local_path
        )
        if rc != 0:
            raise RuntimeError(f"docker cp download_dir failed with exit code {rc}")
