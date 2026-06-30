from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "tmax_slime_grpo" / "validate_data_integrity.py"


def _module():
    spec = importlib.util.spec_from_file_location("tmax_validate_data_integrity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_rows(path: Path, *task_names: str) -> None:
    with path.open("w") as stream:
        for task_name in task_names:
            stream.write(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": f"prompt for {task_name}"}],
                        "metadata": {"task_name": task_name},
                    }
                )
                + "\n"
            )


def test_integrity_rejects_canonical_path_alias(tmp_path: Path) -> None:
    integrity = _module()
    data = tmp_path / "data.jsonl"
    data.write_text("placeholder\n")
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(data)

    with pytest.raises(SystemExit, match="must resolve to different files"):
        integrity.require_distinct_paths(data, alias)


def test_integrity_rejects_train_eval_task_overlap(tmp_path: Path) -> None:
    integrity = _module()
    train = tmp_path / "train.jsonl"
    eval_data = tmp_path / "eval.jsonl"
    _write_rows(train, "train_a", "shared")
    _write_rows(eval_data, "eval_a", "shared")

    with pytest.raises(SystemExit, match=r"overlap \(1 task\(s\)\): shared"):
        integrity.build_manifest(
            train_data=train,
            eval_data=eval_data,
            eval_name="holdout",
        )


def test_integrity_rejects_normalized_prompt_overlap(tmp_path: Path) -> None:
    integrity = _module()
    train = tmp_path / "train.jsonl"
    eval_data = tmp_path / "eval.jsonl"
    train.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "Fix  the BUG\nnow"}],
                "metadata": {"task_name": "train_a"},
            }
        )
        + "\n"
    )
    eval_data.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": " fix the bug now "}],
                "metadata": {"task_name": "eval_a"},
            }
        )
        + "\n"
    )

    with pytest.raises(SystemExit, match="normalized-prompt overlap"):
        integrity.build_manifest(
            train_data=train,
            eval_data=eval_data,
            eval_name="holdout",
        )


def test_integrity_manifest_records_exact_eval_sha_and_environment_copy(
    tmp_path: Path,
) -> None:
    integrity = _module()
    train = tmp_path / "train.jsonl"
    eval_data = tmp_path / "eval.jsonl"
    output = tmp_path / "integrity.json"
    _write_rows(train, "train_a", "train_b")
    _write_rows(eval_data, "eval_a")

    manifest = integrity.build_manifest(
        train_data=train,
        eval_data=eval_data,
        eval_name="holdout",
    )
    integrity.write_manifest(output, manifest)
    decoded = json.loads(base64.b64decode(integrity.encode_manifest(manifest)))

    expected_sha = hashlib.sha256(eval_data.read_bytes()).hexdigest()
    assert manifest["train"]["sha256"] == hashlib.sha256(train.read_bytes()).hexdigest()
    assert manifest["datasets"] == [
        {
            "name": "holdout",
            "path": str(eval_data.resolve()),
            "sha256": expected_sha,
            "size_bytes": eval_data.stat().st_size,
            "row_count": 1,
        }
    ]
    assert json.loads(output.read_text()) == manifest
    assert decoded == manifest
