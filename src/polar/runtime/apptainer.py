"""Apptainer-backed rollout runtime.

Two execution modes (selected once at construction via the env var
``POLAR_APPTAINER_NO_INSTANCE``):

* instance mode (default): ``apptainer instance start`` once, then every
  command runs as ``apptainer exec instance://<name>``. This is the original
  behavior and is efficient on bare metal.
* direct-exec mode (``POLAR_APPTAINER_NO_INSTANCE`` in {1,true,yes,on}): no
  daemon instance is started; every command is a fresh ``apptainer exec
  --overlay <dir> ... <image> ...``. Required when running NESTED inside a
  Pyxis/enroot container, where ``apptainer instance start`` + ``exec
  instance://`` fails with "Failed to enter in user namespace: Invalid
  argument" (setns into the nested instance userns). A host-backed ``--overlay``
  directory makes writes persist across the otherwise-independent exec calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shlex
import shutil
from pathlib import Path

from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult, RuntimeSpec

logger = logging.getLogger(__name__)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


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
        self._overlay_dir: Path | None = None

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
        if _env_truthy("POLAR_APPTAINER_NO_MOUNT_HOSTFS"):
            options.extend(["--no-mount", "hostfs"])
        if self.spec.gpus > 0:
            options.append("--nv")
        if not self.spec.allow_internet:
            network_name: str | None = "none"
        else:
            network_name = self.spec.network
        if network_name and network_name != "host":
            options.extend(["--net", "--network", network_name])
        options.extend(["--bind", f"{self.session_dir}:{self.runtime_session_dir}"])
        # Match DockerRuntime's kwargs.volumes contract (src[:dst[:opts]]).
        for volume in self.spec.kwargs.get("volumes", []):
            options.extend(["--bind", str(volume)])
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

    async def start(self) -> None:
        if self._destroyed:
            raise RuntimeError("apptainer runtime was already destroyed")
        self._overlay_dir = self.session_dir / "overlay"
        self._overlay_dir.mkdir(parents=True, exist_ok=True)

        if not self._use_instance:
            # Direct-exec mode: no daemon instance. Validate the sandbox can run.
            logger.info("Using direct apptainer exec runtime for %s", self._instance_name)
            args = [self._binary, "exec", *self._runtime_options(), self.spec.image, "true"]
            attempts = _env_int("POLAR_APPTAINER_DIRECT_EXEC_RETRIES", 3)
            last_rc = 0
            last_stderr = ""
            for attempt in range(1, attempts + 1):
                last_rc, _, stderr = await self._run_local_command(*args, capture=True)
                last_stderr = stderr or ""
                if last_rc == 0:
                    return
                if attempt < attempts:
                    logger.warning(
                        "%s direct exec validation failed for %s image=%s "
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
                f"{self._binary} direct exec validation failed for {self._instance_name} "
                f"image={self.spec.image} after {attempts} attempts with exit code "
                f"{last_rc}: {last_stderr}"
            )

        args = [self._binary, "instance", "start", *self._runtime_options()]
        args.extend([self.spec.image, self._instance_name])
        rc, _, _ = await self._run_local_command(*args)
        if rc != 0:
            raise RuntimeError(
                f"{self._binary} instance start failed with exit code {rc}"
            )

    _STOP_TIMEOUT = 30.0

    async def stop(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        if not self._use_instance:
            # Nothing to tear down: no instance; overlay lives under session_dir
            # which session cleanup removes.
            return
        rc, _, stderr = await self._run_local_command(
            self._binary, "instance", "stop", self._instance_name,
            timeout=self._STOP_TIMEOUT, capture=True,
        )
        if rc != 0:
            logger.warning(
                "%s instance stop failed for %s (rc=%s): %s",
                self._binary, self._instance_name, rc, stderr,
            )

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        effective_env = {**self.spec.env, **(env or {})}
        effective_workdir = cwd or self.spec.workdir or self.runtime_session_dir
        wrapped_command = command
        if effective_workdir:
            wrapped_command = f"cd {shlex.quote(effective_workdir)} && {command}"
        shell_exports = []
        for key in ("HOME", "PATH"):
            if key in effective_env:
                shell_exports.append(f"export {key}={shlex.quote(str(effective_env[key]))};")
        if shell_exports:
            wrapped_command = " ".join(shell_exports + [wrapped_command])
        args = list(self._exec_base_args())
        if effective_env:
            args.append("env")
            args.extend(f"{key}={value}" for key, value in effective_env.items())
        args.extend(["bash", "-lc", wrapped_command])
        rc, stdout, stderr = await self._run_local_command(
            *args, timeout=timeout_sec, capture=True
        )
        return ExecResult(stdout=stdout, stderr=stderr, return_code=rc)

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        if self._copy_to_bind_mount(local_path, remote_path):
            return
        parent = str(Path(remote_path).parent)
        filename = Path(local_path).name
        source_dir = str(Path(local_path).parent)
        result = await self.exec(f"mkdir -p {shlex.quote(parent)}")
        if result.return_code != 0:
            raise RuntimeError(f"failed to create directory {parent} in runtime")
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"tar -cf - -C {shlex.quote(source_dir)} {shlex.quote(filename)} | "
            f"{self._shell_join(self._exec_base_args())} "
            f"tar -xf - -C {shlex.quote(parent)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(f"apptainer upload_file failed with exit code {rc}")

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        if self._copy_to_bind_mount(local_path, remote_path):
            return
        result = await self.exec(f"mkdir -p {shlex.quote(remote_path)}")
        if result.return_code != 0:
            raise RuntimeError(
                f"failed to create directory {remote_path} in runtime"
            )
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"tar -cf - -C {shlex.quote(local_path)} . | "
            f"{self._shell_join(self._exec_base_args())} "
            f"tar -xf - -C {shlex.quote(remote_path)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(f"apptainer upload_dir failed with exit code {rc}")

    async def download_file(self, remote_path: str, local_path: str) -> None:
        if self._copy_from_bind_mount(remote_path, Path(local_path)):
            return
        parent = str(Path(remote_path).parent)
        filename = Path(remote_path).name
        local_dir = str(Path(local_path).parent)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"{self._shell_join(self._exec_base_args())} "
            f"tar -cf - -C {shlex.quote(parent)} {shlex.quote(filename)} | "
            f"tar -xf - -C {shlex.quote(local_dir)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(
                f"apptainer download_file failed with exit code {rc}"
            )

    async def download_dir(self, remote_path: str, local_path: str) -> None:
        if self._copy_from_bind_mount(remote_path, Path(local_path)):
            return
        Path(local_path).mkdir(parents=True, exist_ok=True)
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"{self._shell_join(self._exec_base_args())} "
            f"tar -cf - -C {shlex.quote(remote_path)} . | "
            f"tar -xf - -C {shlex.quote(local_path)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(
                f"apptainer download_dir failed with exit code {rc}"
            )

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
