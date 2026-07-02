from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "tmax_slime_grpo" / "prepare_harbor_eval.py"


def _module():
    spec = importlib.util.spec_from_file_location("prepare_harbor_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task(root: Path, images: Path, name: str = "task-a") -> None:
    (root / "dataset.toml").parent.mkdir(parents=True, exist_ok=True)
    (root / "dataset.toml").write_text('[dataset]\nname = "example/eval"\n')
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "instruction.md").write_text("Complete the terminal task\n")
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /workspace\n")
    (task / "task.toml").write_text(
        """
[agent]
timeout_sec = 1200
[verifier]
timeout_sec = 900
[environment]
docker_image = "example/task-a:rev1"
cpus = 2
memory_mb = 4096
storage_mb = 8192
allow_internet = true
""".strip()
        + "\n"
    )
    images.mkdir(parents=True, exist_ok=True)
    (images / f"{name}+rev1.sqsh").write_bytes(b"sqsh")


def _args(tasks: Path, images: Path, output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        tasks_dir=str(tasks),
        image_dir=str(images),
        output=str(output),
        dataset_name="terminal_bench_2_1",
        dataset_revision="rev6",
        max_tasks=1,
        agent_timeout_cap=900.0,
        verifier_timeout_cap=600.0,
        timeout_overhead=120.0,
        agent_step_limit=50,
        validate_existing=False,
    )


def test_harbor_eval_rows_pin_official_assets_and_budget(tmp_path: Path) -> None:
    module = _module()
    tasks = tmp_path / "tasks"
    images = tmp_path / "images"
    output = tmp_path / "eval.jsonl"
    _task(tasks, images)

    rows = module._task_rows(_args(tasks, images, output))

    assert len(rows) == 1
    metadata = rows[0]["metadata"]
    assert metadata["task_name"] == "terminal_bench_2_1/task-a"
    assert metadata["source_dataset_revision"] == "rev6"
    assert metadata["sif_path"].endswith("task-a+rev1.sqsh")
    assert metadata["agent_timeout"] == 900.0
    assert metadata["verifier_timeout"] == 600.0
    assert metadata["source_agent_timeout"] == 1200.0
    assert metadata["source_verifier_timeout"] == 900.0
    assert metadata["timeout_seconds"] == 1620.0
    assert metadata["agent_step_limit"] == 50
    assert metadata["workdir"] == "/workspace"
    assert metadata["cpus"] == 2
    assert metadata["memory_mb"] == 4096
    assert metadata["storage_mb"] == 8192
    assert metadata["source_dataset_manifest_sha256"]
    assert metadata["instruction_sha256"]
    assert metadata["task_toml_sha256"]
    assert metadata["tests_tree_sha256"]
    assert metadata["runtime_semantics_version"] == 1


def test_harbor_eval_validation_detects_changed_task_content(tmp_path: Path) -> None:
    module = _module()
    tasks = tmp_path / "tasks"
    images = tmp_path / "images"
    output = tmp_path / "eval.jsonl"
    _task(tasks, images)
    args = _args(tasks, images, output)
    rows = module._task_rows(args)
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (tasks / "task-a" / "tests" / "test.sh").write_text("#!/bin/sh\nexit 1\n")

    with pytest.raises(SystemExit, match="does not match"):
        module.validate_existing_output(output, module._task_rows(args))


def test_harbor_eval_rejects_ambiguous_images(tmp_path: Path) -> None:
    module = _module()
    tasks = tmp_path / "tasks"
    images = tmp_path / "images"
    output = tmp_path / "eval.jsonl"
    _task(tasks, images)
    (images / "task-a+other.sqsh").write_bytes(b"other")

    with pytest.raises(SystemExit, match="exactly one ready image"):
        module._task_rows(_args(tasks, images, output))


def test_harbor_eval_requires_requested_task_count(tmp_path: Path) -> None:
    module = _module()
    tasks = tmp_path / "tasks"
    images = tmp_path / "images"
    output = tmp_path / "eval.jsonl"
    _task(tasks, images)
    args = _args(tasks, images, output)
    args.max_tasks = 2

    with pytest.raises(SystemExit, match="Requested 2 Harbor eval tasks"):
        module._task_rows(args)


def test_harbor_eval_requires_concrete_revision_and_valid_resources(
    tmp_path: Path,
) -> None:
    module = _module()
    tasks = tmp_path / "tasks"
    images = tmp_path / "images"
    output = tmp_path / "eval.jsonl"
    _task(tasks, images)
    args = _args(tasks, images, output)
    args.dataset_revision = "unknown"
    with pytest.raises(SystemExit, match="concrete Harbor dataset revision"):
        module._task_rows(args)

    args.dataset_revision = "rev6"
    task_toml = tasks / "task-a" / "task.toml"
    task_toml.write_text(task_toml.read_text().replace("cpus = 2", "cpus = -1"))
    with pytest.raises(SystemExit, match="environment.cpus"):
        module._task_rows(args)


def test_harbor_eval_restores_docker_env_workdir_and_cmd(tmp_path: Path) -> None:
    module = _module()
    tasks = tmp_path / "tasks"
    images = tmp_path / "images"
    output = tmp_path / "eval.jsonl"
    _task(tasks, images)
    dockerfile = tasks / "task-a" / "environment" / "Dockerfile"
    dockerfile.write_text(
        "FROM ubuntu:24.04\n"
        "ENV USER=root PASSWORD=password1\n"
        "ENV PYTHONPATH=/app:$PYTHONPATH\n"
        "WORKDIR /app\n"
        'CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]\n'
    )

    metadata = module._task_rows(_args(tasks, images, output))[0]["metadata"]

    assert metadata["workdir"] == "/app"
    assert metadata["runtime_env"] == {
        "USER": "root",
        "PASSWORD": "password1",
        "PYTHONPATH": "/app:",
    }
    assert metadata["docker_cmd"] == {
        "form": "exec",
        "argv": ["supervisord", "-c", "/etc/supervisor/supervisord.conf"],
    }
    assert metadata["docker_entrypoint"] is None
    assert metadata["runtime_init_command"] == (
        "mkdir -p /polar/session/logs && "
        "(supervisord -c /etc/supervisor/supervisord.conf "
        ">>/polar/session/logs/container-init.log 2>&1 &)"
    )
    assert metadata["dockerfile_sha256"]


def test_harbor_eval_uses_exported_sqsh_environment_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(
        "FROM python:3.11-slim\nENV PYTHONPATH=/app:$PYTHONPATH\n"
    )
    image = tmp_path / "task.sqsh"
    image.write_bytes(b"hsqs-image")
    docker_metadata = module._docker_runtime_metadata(task)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/unsquashfs")
    commands: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(args)
        return SimpleNamespace(
            returncode=0,
            stdout="PATH=/usr/bin:/bin\nPYTHONPATH=/app:/base/python\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    runtime_env = module._runtime_environment(
        docker_metadata=docker_metadata,
        task_environment={},
        image=image,
        task_dir=task,
    )

    assert runtime_env == {"PYTHONPATH": "/app:/base/python"}
    assert commands == [
        [
            "/usr/bin/unsquashfs",
            "-processors",
            "1",
            "-cat",
            str(image),
            "etc/environment",
        ]
    ]


@pytest.mark.parametrize(
    ("directive", "error"),
    [
        ("USER nobody", "Unsupported non-root Docker USER"),
        ("VOLUME /data", "Unsupported Docker VOLUME"),
    ],
)
def test_harbor_eval_fails_closed_on_unmodeled_runtime_semantics(
    tmp_path: Path,
    directive: str,
    error: str,
) -> None:
    module = _module()
    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(
        f"FROM ubuntu:24.04\n{directive}\n"
    )

    with pytest.raises(SystemExit, match=error):
        module._docker_runtime_metadata(task)
