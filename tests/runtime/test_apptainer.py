from __future__ import annotations

import asyncio
from pathlib import Path

from polar.runtime.apptainer import ApptainerRuntime
from polar.runtime.models import RuntimeSpec


def _runtime(monkeypatch, tmp_path: Path, *, direct: bool) -> ApptainerRuntime:
    if direct:
        monkeypatch.setenv("POLAR_APPTAINER_NO_INSTANCE", "1")
    else:
        monkeypatch.delenv("POLAR_APPTAINER_NO_INSTANCE", raising=False)
    monkeypatch.setattr(
        ApptainerRuntime,
        "_resolve_binary",
        staticmethod(lambda: "/usr/bin/apptainer"),
    )
    return ApptainerRuntime(
        RuntimeSpec(
            backend="apptainer",
            image="/images/task.sif",
            workdir="/work",
            kwargs={"volumes": ["/host/cli:/opt/node:ro"]},
        ),
        "session-1",
        tmp_path / "session",
    )


def test_instance_mode_remains_the_default(monkeypatch, tmp_path: Path) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=False)
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, **_kwargs):
        calls.append(args)
        return 0, None, None

    runtime._run_local_command = fake_run  # type: ignore[method-assign]
    asyncio.run(runtime.start())
    asyncio.run(runtime.exec("echo ok"))
    asyncio.run(runtime.stop())

    assert calls[0][1:3] == ("instance", "start")
    assert "--overlay" in calls[0]
    assert "/images/task.sif" in calls[0]
    assert calls[1][1:3] == ("exec", f"instance://{runtime.runtime_id}")
    assert calls[2][1:3] == ("instance", "stop")


def test_direct_exec_reuses_overlay_and_skips_instance_lifecycle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, **_kwargs):
        calls.append(args)
        return 0, "", ""

    runtime._run_local_command = fake_run  # type: ignore[method-assign]
    asyncio.run(runtime.start())
    asyncio.run(runtime.exec("echo ok", env={"HOME": "/root"}))
    asyncio.run(runtime.stop())

    assert len(calls) == 2
    validation, command = calls
    for call in (validation, command):
        assert call[:2] == ("/usr/bin/apptainer", "exec")
        assert "instance" not in call
        assert "--overlay" in call
        assert str(tmp_path / "session" / "overlay") in call
        assert "/images/task.sif" in call
        assert "/host/cli:/opt/node:ro" in call
    assert validation[-1] == "true"
    assert command[-3:] == (
        "bash",
        "-lc",
        "export HOME=/root; cd /work && echo ok",
    )


def test_runtime_can_disable_automatic_hostfs_mounts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("POLAR_APPTAINER_NO_MOUNT_HOSTFS", "1")
    runtime = _runtime(monkeypatch, tmp_path, direct=True)
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, **_kwargs):
        calls.append(args)
        return 0, "", ""

    runtime._run_local_command = fake_run  # type: ignore[method-assign]
    asyncio.run(runtime.start())

    options = calls[0]
    no_mount_index = options.index("--no-mount")
    assert options[no_mount_index + 1] == "hostfs"
