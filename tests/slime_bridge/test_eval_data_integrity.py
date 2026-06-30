from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime_bridge.rollout import (
    PolarEvalDataIntegrityError,
    _load_eval_sample_groups,
)


def _row(task_name: str) -> dict[str, object]:
    return {
        "prompt": [{"role": "user", "content": f"fix {task_name}"}],
        "label": "",
        "metadata": {"task_name": task_name},
    }


def _write_jsonl(path: Path, *task_names: str) -> None:
    path.write_text("".join(json.dumps(_row(name)) + "\n" for name in task_names))


def _manifest_b64(path: Path, *, name: str = "holdout") -> str:
    payload = path.read_bytes()
    manifest = {
        "schema_version": 1,
        "algorithm": "sha256",
        "datasets": [
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "row_count": len(payload.splitlines()),
            }
        ],
    }
    return base64.b64encode(json.dumps(manifest).encode()).decode()


def _args_and_dataset(path: Path):
    args = SimpleNamespace(
        input_key="prompt",
        label_key="label",
        metadata_key="metadata",
        tool_key=None,
        n_samples_per_eval_prompt=1,
    )
    dataset = SimpleNamespace(
        name="holdout",
        path=str(path),
        input_key=None,
        label_key=None,
        metadata_key=None,
        tool_key=None,
        n_samples_per_eval_prompt=2,
        custom_generate_function_path=None,
        inject_metadata=lambda metadata: dict(metadata or {}),
    )
    return args, dataset


def test_eval_data_manifest_accepts_exact_file_then_detects_mutation(
    monkeypatch, tmp_path: Path
) -> None:
    eval_data = tmp_path / "eval.jsonl"
    _write_jsonl(eval_data, "task_a", "task_b")
    monkeypatch.setenv("POLAR_EVAL_DATA_INTEGRITY_B64", _manifest_b64(eval_data))
    args, dataset = _args_and_dataset(eval_data)

    groups = _load_eval_sample_groups(args, dataset)

    assert len(groups) == 2
    assert [len(group) for group in groups] == [2, 2]
    assert groups[0][0].metadata["task_name"] == "task_a"

    _write_jsonl(eval_data, "task_a", "task_changed")
    with pytest.raises(PolarEvalDataIntegrityError, match="changed after launcher validation"):
        _load_eval_sample_groups(args, dataset)


def test_eval_data_manifest_rejects_unprotected_dataset(monkeypatch, tmp_path: Path) -> None:
    protected = tmp_path / "protected.jsonl"
    unprotected = tmp_path / "unprotected.jsonl"
    _write_jsonl(protected, "task_a")
    _write_jsonl(unprotected, "task_b")
    monkeypatch.setenv("POLAR_EVAL_DATA_INTEGRITY_B64", _manifest_b64(protected))
    args, dataset = _args_and_dataset(unprotected)

    with pytest.raises(PolarEvalDataIntegrityError, match="absent from.*integrity manifest"):
        _load_eval_sample_groups(args, dataset)


def test_eval_data_loading_remains_backward_compatible_without_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    eval_data = tmp_path / "eval.jsonl"
    _write_jsonl(eval_data, "task_a")
    monkeypatch.delenv("POLAR_EVAL_DATA_INTEGRITY_B64", raising=False)
    args, dataset = _args_and_dataset(eval_data)

    groups = _load_eval_sample_groups(args, dataset)

    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_eval_data_manifest_rejects_invalid_environment(monkeypatch, tmp_path: Path) -> None:
    eval_data = tmp_path / "eval.jsonl"
    _write_jsonl(eval_data, "task_a")
    monkeypatch.setenv("POLAR_EVAL_DATA_INTEGRITY_B64", "not-base64!")
    args, dataset = _args_and_dataset(eval_data)

    with pytest.raises(PolarEvalDataIntegrityError, match="Invalid POLAR_EVAL_DATA"):
        _load_eval_sample_groups(args, dataset)
