from __future__ import annotations

import importlib
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "tmax-15k"


def _build_sifs_module(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in ("build_sifs", "build_images", "dataset"):
        sys.modules.pop(name, None)
    return importlib.import_module("build_sifs")


def _task(module, root: Path, name: str):
    task_dir = root / name
    environment_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    environment_dir.mkdir(parents=True)
    tests_dir.mkdir()
    (environment_dir / "post_install.sh").write_text("#!/bin/sh\ntrue\n")
    (environment_dir / "Dockerfile").write_text(
        "FROM ubuntu:22.04\n"
        "COPY post_install.sh /tmp/post_install.sh\n"
        "RUN /tmp/post_install.sh && rm /tmp/post_install.sh\n"
    )
    return module.TmaxTask(
        name=name,
        task_dir=task_dir,
        instruction="",
        environment_dir=environment_dir,
        tests_dir=tests_dir,
        agent_timeout=600.0,
        verifier_timeout=120.0,
        cpus=None,
        memory_mb=None,
        allow_internet=True,
        workdir=None,
    )


def test_concurrent_direct_builds_merge_private_tmp_before_squashing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    build_sifs = _build_sifs_module(monkeypatch)
    tasks = [_task(build_sifs, tmp_path / "dataset", f"task_{index}") for index in range(2)]
    image_dir = tmp_path / "images"
    definition_dir = tmp_path / "definitions"
    shared_tmp = tmp_path / "shared-apptainer-tmp"
    shared_tmp.mkdir()
    base_env = {
        "APPTAINER_TMPDIR": str(shared_tmp),
        "SINGULARITY_TMPDIR": str(shared_tmp),
        "TMPDIR": str(shared_tmp),
    }
    barrier = threading.Barrier(2)
    observations: list[tuple[dict[str, str], Path, Path]] = []

    def fake_run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
        assert env is not None
        workspace = Path(env["APPTAINER_TMPDIR"]).parent
        assert workspace.is_dir()
        if command[0] == "cp":
            source = Path(command[-2].removesuffix("/."))
            destination = Path(command[-1])
            (destination / "persisted.txt").write_text(
                (source / "persisted.txt").read_text()
            )
            return
        if "--sandbox" in command:
            bind = command[command.index("--bind") + 1]
            container_tmp = Path(bind.removesuffix(":/tmp"))
            sandbox = Path(command[-2])
            assert container_tmp.parent == workspace
            sandbox.mkdir()
            (sandbox / "tmp").mkdir()
            (container_tmp / "persisted.txt").write_text("task-local")
            observations.append((env, container_tmp, workspace))
            barrier.wait(timeout=5)
            return

        assert "--bind" not in command
        sandbox = Path(command[-1])
        assert (sandbox / "tmp" / "persisted.txt").read_text() == "task-local"
        Path(command[-2]).write_bytes(b"sif")

    monkeypatch.setattr(build_sifs, "run_command", fake_run_command)

    def build(task):
        return build_sifs.build_one(
            task,
            image_dir=image_dir,
            binary="apptainer",
            builder="direct-apptainer",
            definition_dir=definition_dir,
            base_sif="",
            docker_bin=None,
            env=base_env,
            force=False,
            force_docker=False,
            skip_docker_build=False,
            mksquashfs_args="",
            apptainer_fakeroot=False,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(build, tasks))

    assert [result[0] for result in results] == ["built", "built"]
    assert len(observations) == 2
    first_env, first_container_tmp, first_workspace = observations[0]
    second_env, second_container_tmp, second_workspace = observations[1]
    assert first_workspace != second_workspace
    assert first_container_tmp != second_container_tmp
    assert first_env["APPTAINER_TMPDIR"] != second_env["APPTAINER_TMPDIR"]
    assert first_env["SINGULARITY_TMPDIR"] != second_env["SINGULARITY_TMPDIR"]
    assert first_env["TMPDIR"] != second_env["TMPDIR"]
    assert Path(first_env["APPTAINER_TMPDIR"]).parent == first_workspace
    assert Path(first_env["SINGULARITY_TMPDIR"]).parent == first_workspace
    assert Path(first_env["TMPDIR"]).parent == first_workspace
    assert not first_workspace.exists()
    assert not second_workspace.exists()
    assert all((image_dir / build_sifs.sif_filename_for(task.name)).is_file() for task in tasks)


def _build_one(build_sifs, task, tmp_path: Path, *, force: bool = False):
    return build_sifs.build_one(
        task,
        image_dir=tmp_path / "images",
        binary="apptainer",
        builder="direct-apptainer",
        definition_dir=tmp_path / "definitions",
        base_sif="",
        docker_bin=None,
        env={"TMPDIR": str(tmp_path / "scratch")},
        force=force,
        force_docker=False,
        skip_docker_build=False,
        mksquashfs_args="",
        apptainer_fakeroot=False,
    )


def _fake_direct_build(command: list[str], *, env: dict[str, str] | None) -> bool:
    assert env is not None
    if command[0] == "cp":
        return True
    if "--sandbox" in command:
        sandbox = Path(command[-2])
        sandbox.mkdir()
        (sandbox / "tmp").mkdir()
        return True
    return False


def test_final_pack_retries_without_rebuilding_sandbox(monkeypatch, tmp_path: Path):
    build_sifs = _build_sifs_module(monkeypatch)
    task = _task(build_sifs, tmp_path / "dataset", "task_retry")
    calls = {"sandbox": 0, "copy": 0, "pack": 0}
    pack_commands: list[list[str]] = []
    sleeps: list[int] = []

    def fake_run(command: list[str], *, env: dict[str, str] | None = None) -> None:
        if command[0] == "cp":
            calls["copy"] += 1
            return
        if "--sandbox" in command:
            calls["sandbox"] += 1
            assert _fake_direct_build(command, env=env)
            return
        calls["pack"] += 1
        pack_commands.append(command)
        output = Path(command[-2])
        if calls["pack"] < 3:
            output.write_bytes(b"partial")
            raise subprocess.CalledProcessError(255, command)
        output.write_bytes(b"complete-sif")

    monkeypatch.setattr(build_sifs, "run_command", fake_run)
    monkeypatch.setattr(build_sifs.time, "sleep", sleeps.append)

    status, _, target = _build_one(build_sifs, task, tmp_path)

    assert status == "built"
    assert Path(target).read_bytes() == b"complete-sif"
    assert calls == {"sandbox": 1, "copy": 1, "pack": 3}
    assert all("-no-duplicates" not in command for command in pack_commands[:2])
    assert pack_commands[2][pack_commands[2].index("--mksquashfs-args") + 1] == "-no-duplicates"
    assert sleeps == [1, 2]
    assert list((tmp_path / "images").glob(".*.tmp-*")) == []


def test_final_pack_failure_preserves_existing_target(monkeypatch, tmp_path: Path):
    build_sifs = _build_sifs_module(monkeypatch)
    task = _task(build_sifs, tmp_path / "dataset", "task_existing")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    target = image_dir / build_sifs.sif_filename_for(task.name)
    target.write_bytes(b"trusted-old-sif")
    pack_calls = 0

    def fake_run(command: list[str], *, env: dict[str, str] | None = None) -> None:
        nonlocal pack_calls
        if _fake_direct_build(command, env=env):
            return
        pack_calls += 1
        if command[0] == "mksquashfs":
            raise subprocess.CalledProcessError(1, command)
        Path(command[-2]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(255, command)

    monkeypatch.setattr(build_sifs, "run_command", fake_run)
    monkeypatch.setattr(build_sifs.time, "sleep", lambda _seconds: None)

    with pytest.raises(subprocess.CalledProcessError):
        _build_one(build_sifs, task, tmp_path, force=True)

    assert pack_calls == 4
    assert target.read_bytes() == b"trusted-old-sif"
    assert list(image_dir.glob(".*.tmp-*")) == []
    assert list((tmp_path / "scratch").iterdir()) == []


@pytest.mark.parametrize("payload", [None, b""])
def test_final_pack_success_without_nonempty_regular_file_is_rejected(
    monkeypatch,
    tmp_path: Path,
    payload: bytes | None,
):
    build_sifs = _build_sifs_module(monkeypatch)
    task = _task(build_sifs, tmp_path / "dataset", "task_empty")

    def fake_run(command: list[str], *, env: dict[str, str] | None = None) -> None:
        if _fake_direct_build(command, env=env):
            return
        if payload is not None:
            Path(command[-2]).write_bytes(payload)

    monkeypatch.setattr(build_sifs, "run_command", fake_run)

    with pytest.raises(RuntimeError, match="without a non-empty regular file"):
        _build_one(build_sifs, task, tmp_path)

    assert not (tmp_path / "images" / build_sifs.sif_filename_for(task.name)).exists()
    assert list((tmp_path / "images").glob(".*.tmp-*")) == []


def test_cleanup_warning_does_not_replace_pack_error(monkeypatch, tmp_path: Path, capsys):
    build_sifs = _build_sifs_module(monkeypatch)
    task = _task(build_sifs, tmp_path / "dataset", "task_cleanup")

    def fake_run(command: list[str], *, env: dict[str, str] | None = None) -> None:
        if _fake_direct_build(command, env=env):
            return
        raise subprocess.CalledProcessError(255, command)

    monkeypatch.setattr(build_sifs, "run_command", fake_run)
    monkeypatch.setattr(build_sifs.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        build_sifs.shutil,
        "rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("cleanup denied")),
    )

    with pytest.raises(subprocess.CalledProcessError):
        _build_one(build_sifs, task, tmp_path)

    assert "WARNING: failed to remove isolated build workspace" in capsys.readouterr().err


def test_merge_private_tmp_rejects_sandbox_tmp_symlink(monkeypatch, tmp_path: Path):
    build_sifs = _build_sifs_module(monkeypatch)
    container_tmp = tmp_path / "private"
    sandbox = tmp_path / "sandbox"
    outside = tmp_path / "outside"
    container_tmp.mkdir()
    sandbox.mkdir()
    outside.mkdir()
    (sandbox / "tmp").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="non-directory sandbox path"):
        build_sifs.merge_private_tmp(container_tmp, sandbox, env={})

    assert list(outside.iterdir()) == []


def test_duplicate_fallback_preserves_existing_mksquashfs_args(monkeypatch):
    build_sifs = _build_sifs_module(monkeypatch)
    command = [
        "apptainer",
        "build",
        "--force",
        "--mksquashfs-args",
        "-processors 1 -mem 1024M",
        "output.sif",
        "sandbox",
    ]

    fallback = build_sifs._disable_mksquashfs_duplicates(command)

    assert command[4] == "-processors 1 -mem 1024M"
    assert fallback[4] == "-processors 1 -mem 1024M -no-duplicates"


def test_direct_squashfs_fallback_creates_system_partition(monkeypatch, tmp_path: Path):
    build_sifs = _build_sifs_module(monkeypatch)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    output = tmp_path / "image.sif"
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, env: dict[str, str] | None = None) -> None:
        calls.append(command)
        if command[0] == "mksquashfs":
            Path(command[2]).write_bytes(b"squashfs")
        elif command[1:3] == ["sif", "new"]:
            Path(command[3]).write_bytes(b"sif-header")
        elif command[1:3] == ["sif", "add"]:
            Path(command[3]).write_bytes(b"complete-sif")

    monkeypatch.setattr(build_sifs, "run_command", fake_run)
    build_sifs.pack_sandbox_sif_without_reimport(
        "apptainer",
        sandbox,
        output,
        env={},
        mksquashfs_args="-processors 1 -mem 1024M",
    )

    assert output.read_bytes() == b"complete-sif"
    assert calls[0] == [
        "mksquashfs",
        str(sandbox),
        f"{output}.rootfs.squashfs",
        "-noappend",
        "-processors",
        "1",
        "-mem",
        "1024M",
    ]
    assert calls[1] == ["apptainer", "sif", "new", str(output)]
    assert calls[2][-10:] == [
        "--groupid",
        "1",
        "--datatype",
        "4",
        "--parttype",
        "2",
        "--partfs",
        "1",
        "--partarch",
        "2",
    ]
    assert not Path(f"{output}.rootfs.squashfs").exists()
