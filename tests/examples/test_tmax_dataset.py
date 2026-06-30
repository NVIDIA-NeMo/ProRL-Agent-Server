from __future__ import annotations

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "tmax-15k"
TMAX_GRPO_DIR = Path(__file__).resolve().parents[2] / "examples" / "tmax_slime_grpo"


def _dataset_module(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("dataset", None)
    return importlib.import_module("dataset")


def _prepare_data_module(monkeypatch):
    monkeypatch.syspath_prepend(str(TMAX_GRPO_DIR))
    sys.modules.pop("prepare_data", None)
    return importlib.import_module("prepare_data")


def _write_task(root: Path, name: str, *, task_toml: str = "") -> None:
    task_dir = root / name
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "task.toml").write_text(task_toml)
    (task_dir / "instruction.md").write_text(f"instruction for {name}")
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\n")


def test_load_tasks_stops_parsing_after_requested_prefix(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset_module(monkeypatch)
    _write_task(tmp_path, "task_a")
    # This later task is intentionally malformed. A max_tasks=1 caller must
    # not parse it after the requested valid prefix has already been loaded.
    _write_task(tmp_path, "task_b", task_toml="not valid = [toml")

    tasks = dataset.load_tasks(tmp_path, max_tasks=1)

    assert [task.name for task in tasks] == ["task_a"]


def test_load_tasks_rejects_incomplete_task_in_selected_prefix(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = _dataset_module(monkeypatch)
    _write_task(tmp_path, "task_a")
    (tmp_path / "task_a" / "environment" / "Dockerfile").unlink()

    with pytest.raises(
        SystemExit,
        match=r"Invalid TMax task directory .*missing required file\(s\): environment/Dockerfile",
    ):
        dataset.load_tasks(tmp_path, max_tasks=1)


def _selection_args(dataset_dir: Path, image_dir: Path, *, only_ready: bool) -> Namespace:
    return Namespace(
        dataset_dir=str(dataset_dir),
        image_dir=str(image_dir),
        task=[],
        only_ready=only_ready,
        max_tasks=-1,
    )


def test_prepare_data_only_ready_skips_zero_byte_sif(monkeypatch, tmp_path: Path) -> None:
    prepare_data = _prepare_data_module(monkeypatch)
    dataset_dir = tmp_path / "dataset"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_task(dataset_dir, "task_a")
    _write_task(dataset_dir, "task_b")
    (image_dir / "task_a.sif").touch()
    (image_dir / "task_b.sif").write_bytes(b"sif")

    selected, missing_count = prepare_data.select_tasks(
        _selection_args(dataset_dir, image_dir, only_ready=True)
    )

    assert [task.name for task in selected] == ["task_b"]
    assert missing_count == 1


def test_prepare_data_full_mode_rejects_zero_byte_sif(monkeypatch, tmp_path: Path) -> None:
    prepare_data = _prepare_data_module(monkeypatch)
    dataset_dir = tmp_path / "dataset"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_task(dataset_dir, "task_a")
    (image_dir / "task_a.sif").touch()

    with pytest.raises(SystemExit, match="Missing 1/1 selected SIF"):
        prepare_data.select_tasks(_selection_args(dataset_dir, image_dir, only_ready=False))


def test_prepare_data_only_ready_never_backfills_selected_prefix(
    monkeypatch, tmp_path: Path
) -> None:
    prepare_data = _prepare_data_module(monkeypatch)
    dataset_dir = tmp_path / "dataset"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("task_a", "task_b", "task_c"):
        _write_task(dataset_dir, name)
    (image_dir / "task_b.sif").write_bytes(b"sif")
    (image_dir / "task_c.sif").write_bytes(b"sif")
    args = _selection_args(dataset_dir, image_dir, only_ready=True)
    args.max_tasks = 2

    selected, missing_count = prepare_data.select_tasks(args)

    assert [task.name for task in selected] == ["task_b"]
    assert missing_count == 1


def test_prepare_data_selects_stable_window_after_start_index(monkeypatch, tmp_path: Path) -> None:
    prepare_data = _prepare_data_module(monkeypatch)
    dataset_dir = tmp_path / "dataset"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("task_a", "task_b", "task_c", "task_d"):
        _write_task(dataset_dir, name)
        (image_dir / f"{name}.sif").write_bytes(b"sif")
    args = _selection_args(dataset_dir, image_dir, only_ready=False)
    args.start_index = 1
    args.max_tasks = 2

    selected, missing_count = prepare_data.select_tasks(args)

    assert [task.name for task in selected] == ["task_b", "task_c"]
    assert missing_count == 0


def test_prepare_data_full_mode_requires_requested_task_count(monkeypatch, tmp_path: Path) -> None:
    prepare_data = _prepare_data_module(monkeypatch)
    dataset_dir = tmp_path / "dataset"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_task(dataset_dir, "task_a")
    (image_dir / "task_a.sif").write_bytes(b"sif")
    args = _selection_args(dataset_dir, image_dir, only_ready=False)
    args.max_tasks = 2

    with pytest.raises(SystemExit, match="first 2 TMax task.*only 1 valid"):
        prepare_data.select_tasks(args)


def _write_prompt_row(path: Path, task_name: str, sif_path: Path) -> None:
    row = {
        "prompt": [{"role": "user", "content": "fix it"}],
        "metadata": {"task_name": task_name, "sif_path": str(sif_path)},
    }
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def _validate_existing(prepare_data, args: Namespace, output: Path) -> int:
    expected_tasks = prepare_data.load_task_prefix(args)
    if not args.only_ready:
        prepare_data.require_complete_prefix(args, expected_tasks)
    return prepare_data.validate_existing_output(
        output,
        expected_tasks=expected_tasks,
        image_dir=Path(args.image_dir).resolve(),
        allow_partial=args.only_ready,
    )


def test_validate_existing_output_requires_exact_prefix_and_nonempty_sifs(
    monkeypatch, tmp_path: Path
) -> None:
    prepare_data = _prepare_data_module(monkeypatch)
    dataset_dir = tmp_path / "dataset"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_task(dataset_dir, "task_a")
    _write_task(dataset_dir, "task_b")
    args = _selection_args(dataset_dir, image_dir, only_ready=False)
    args.max_tasks = 2
    output = tmp_path / "train.jsonl"
    task_a_sif = image_dir / "task_a.sif"
    task_b_sif = image_dir / "task_b.sif"
    task_a_sif.write_bytes(b"sif")
    task_b_sif.touch()
    _write_prompt_row(output, "task_a", task_a_sif)

    with pytest.raises(SystemExit, match="must match.*prefix exactly.*expected_rows=2"):
        _validate_existing(prepare_data, args, output)

    output.unlink()
    _write_prompt_row(output, "task_b", task_b_sif)
    _write_prompt_row(output, "task_a", task_a_sif)
    with pytest.raises(SystemExit, match="row 1: expected task_a, found task_b"):
        _validate_existing(prepare_data, args, output)

    output.unlink()
    _write_prompt_row(output, "task_a", task_a_sif)
    _write_prompt_row(output, "task_b", task_b_sif)
    with pytest.raises(SystemExit, match="Missing or empty SIF.*1/2"):
        _validate_existing(prepare_data, args, output)

    task_b_sif.write_bytes(b"sif")
    assert _validate_existing(prepare_data, args, output) == 2


def test_validate_existing_partial_output_must_be_ordered_prefix_subsequence(
    monkeypatch, tmp_path: Path
) -> None:
    prepare_data = _prepare_data_module(monkeypatch)
    dataset_dir = tmp_path / "dataset"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("task_a", "task_b", "task_c"):
        _write_task(dataset_dir, name)
        (image_dir / f"{name}.sif").write_bytes(b"sif")
    args = _selection_args(dataset_dir, image_dir, only_ready=True)
    args.max_tasks = 2
    output = tmp_path / "train.jsonl"

    _write_prompt_row(output, "task_c", image_dir / "task_c.sif")
    with pytest.raises(SystemExit, match="ordered subsequence.*unexpected=.*task_c"):
        _validate_existing(prepare_data, args, output)

    output.unlink()
    _write_prompt_row(output, "task_b", image_dir / "task_b.sif")
    _write_prompt_row(output, "task_a", image_dir / "task_a.sif")
    with pytest.raises(SystemExit, match="ordered subsequence"):
        _validate_existing(prepare_data, args, output)

    output.unlink()
    _write_prompt_row(output, "task_b", image_dir / "task_b.sif")
    assert _validate_existing(prepare_data, args, output) == 1
