"""Tests for the filesystem index of rollout_results/."""

from __future__ import annotations

import json
from pathlib import Path

from polar.platform.fs_index import FsIndex


def _write_session(
    root: Path,
    *,
    task_id: str,
    session_id: str,
    status: str,
    reward: float | None = 1.0,
    harness_in_task_id: bool = True,
) -> Path:
    task_dir = root / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "task_id": task_id,
        "status": status,
        "node_id": "node-a",
        "timing": {"init_ms": 100, "run_ms": 200, "postrun_ms": 50, "register_to_init_queue_ms": 1.0},
        "trajectory": {
            "metadata": {
                "api_type": "openai",
                "model_used": "Qwen/Qwen3.5-4B",
            },
            "traces": [{"reward": reward}],
        },
    }
    path = task_dir / f"ses_{session_id}.json"
    path.write_text(json.dumps(payload))
    return path


def test_fs_index_lists_tasks(tmp_path: Path) -> None:
    _write_session(tmp_path, task_id="calculator-claude_code-AAA", session_id="s1", status="COMPLETED")
    _write_session(tmp_path, task_id="calculator-codex-BBB", session_id="s2", status="ERROR", reward=0.0)
    idx = FsIndex(tmp_path)
    idx.scan()
    tasks = idx.list_tasks()
    assert len(tasks) == 2
    by_id = {t.task_id: t for t in tasks}
    assert by_id["calculator-claude_code-AAA"].harness == "claude_code"
    assert by_id["calculator-claude_code-AAA"].mean_reward == 1.0
    assert by_id["calculator-codex-BBB"].errored_sessions == 1


def test_fs_index_session_lookup(tmp_path: Path) -> None:
    _write_session(tmp_path, task_id="calculator-claude_code-AAA", session_id="abc", status="COMPLETED")
    idx = FsIndex(tmp_path)
    idx.scan()
    path = idx.session_file_for("abc")
    assert path is not None
    assert path.name == "ses_abc.json"


def test_fs_index_status_filter(tmp_path: Path) -> None:
    _write_session(tmp_path, task_id="t-codex-1", session_id="x", status="COMPLETED")
    idx = FsIndex(tmp_path)
    idx.scan()
    completed = idx.list_tasks(status="completed")
    assert len(completed) == 1
    running = idx.list_tasks(status="running")
    assert running == []
