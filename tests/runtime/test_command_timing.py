from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from polar.runtime.apptainer import ApptainerRuntime
from polar.runtime.command_timing import classify_runtime_command
from polar.runtime.docker import DockerRuntime
from polar.runtime.models import RuntimeSpec


def test_runtime_command_classifier_has_stable_git_diff_category() -> None:
    assert classify_runtime_command("cd /repo && git diff --stat") == "git_diff"
    assert classify_runtime_command("git -C /repo status --short") == "git_status"
    assert classify_runtime_command("python -m pytest -q") == "test"
    assert classify_runtime_command("bash /tests/test.sh") == "verifier"


def test_apptainer_exec_records_timeout_without_command_text(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("POLAR_APPTAINER_NO_INSTANCE", "1")
    monkeypatch.setattr(
        ApptainerRuntime,
        "_resolve_binary",
        staticmethod(lambda: "/usr/bin/apptainer"),
    )
    runtime = ApptainerRuntime(
        RuntimeSpec(backend="apptainer", image="/image.sif"),
        "session",
        tmp_path,
    )

    async def fake_broker_exec(*_args, **_kwargs):
        return -1, None, None

    runtime._broker_exec = fake_broker_exec  # type: ignore[method-assign]
    asyncio.run(runtime.exec("git diff -- token-that-must-not-be-a-metric"))

    summary = runtime.exec_timing_summary()
    assert summary["count"] == 1
    assert summary["timeout_count"] == 1
    assert summary["count_by_category"]["git_diff"] == 1  # type: ignore[index]
    assert "token-that-must-not-be-a-metric" not in repr(summary)


def test_docker_exec_records_failed_command(monkeypatch, tmp_path: Path) -> None:
    runtime = DockerRuntime(
        RuntimeSpec(backend="docker", image="image"),
        "session",
        tmp_path,
    )

    async def fake_run(*_args, **_kwargs):
        return 17, "", "failed"

    runtime._run_local_command = fake_run  # type: ignore[method-assign]
    asyncio.run(runtime.exec("python -m pytest tests/unit"))

    summary = runtime.exec_timing_summary()
    assert summary["failure_count"] == 1
    assert summary["count_by_category"]["test"] == 1  # type: ignore[index]


def test_docker_exec_records_spawn_exception(monkeypatch, tmp_path: Path) -> None:
    runtime = DockerRuntime(
        RuntimeSpec(backend="docker", image="image"),
        "session",
        tmp_path,
    )

    async def fake_run(*_args, **_kwargs):
        raise OSError("docker binary disappeared")

    runtime._run_local_command = fake_run  # type: ignore[method-assign]

    with pytest.raises(OSError, match="docker binary disappeared"):
        asyncio.run(runtime.exec("git status --short"))

    summary = runtime.exec_timing_summary()
    assert summary["count"] == 1
    assert summary["failure_count"] == 1
    assert summary["exception_count"] == 1
    assert summary["timeout_count"] == 0


@pytest.mark.parametrize("allow_internet", [True, False])
def test_docker_mounts_internet_volumes_only_for_online_runtime(
    tmp_path: Path,
    allow_internet: bool,
) -> None:
    proxy_bind = "/host/proxy:/polar/proxy:ro"
    runtime = DockerRuntime(
        RuntimeSpec(
            backend="docker",
            image="image",
            allow_internet=allow_internet,
            internet_volumes=[proxy_bind],
        ),
        "session",
        tmp_path,
    )
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, **_kwargs):
        calls.append(args)
        if args[:2] == ("docker", "exec") and args[-2:] == ("id", "-u"):
            return 0, str(os.getuid()), ""
        return 0, "", ""

    runtime._run_local_command = fake_run  # type: ignore[method-assign]
    asyncio.run(runtime.start())

    create = calls[0]
    assert create[:2] == ("docker", "create")
    assert (proxy_bind in create) is allow_internet
