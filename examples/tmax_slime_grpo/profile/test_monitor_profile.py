from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import monitor_profile as monitor


def test_parse_job_rejects_ambiguous_values() -> None:
    assert monitor.parse_job("first=123") == ("first", "123")
    with pytest.raises(Exception):
        monitor.parse_job("first")
    with pytest.raises(Exception):
        monitor.parse_job("first=not-a-job")


def test_snapshot_uses_squeue_then_sacct_and_classifies_terminal_state() -> None:
    queue = {
        "11": {
            "state": "RUNNING",
            "elapsed": "1:00",
            "reason": "node-1",
            "exit_code": "",
            "source": "squeue",
        }
    }
    account = {
        "12": {
            "state": "COMPLETED",
            "elapsed": "2:00",
            "reason": "",
            "exit_code": "0:0",
            "source": "sacct",
        }
    }
    with patch.object(monitor, "query_squeue", return_value=(queue, "")), patch.object(
        monitor,
        "query_sacct",
        return_value=(account, ""),
    ):
        report = monitor.snapshot(
            [("active", "11"), ("done", "12")],
            squeue="squeue",
            sacct="sacct",
        )

    assert report["jobs"]["11"]["terminal"] is False
    assert report["jobs"]["12"]["terminal"] is True
    assert report["jobs"]["12"]["success"] is True
    assert report["all_terminal"] is False
    assert report["any_failed"] is False


def test_snapshot_preserves_last_state_during_transient_scheduler_failure() -> None:
    previous = {
        "13": {
            "state": "PENDING",
            "elapsed": "0:00",
            "reason": "Dependency",
            "exit_code": "",
            "source": "squeue",
        }
    }
    with patch.object(monitor, "query_squeue", return_value=({}, "offline")), patch.object(
        monitor,
        "query_sacct",
        return_value=({}, "offline"),
    ):
        report = monitor.snapshot(
            [("waiting", "13")],
            squeue="squeue",
            sacct="sacct",
            previous=previous,
        )

    assert report["jobs"]["13"]["state"] == "PENDING"
    assert report["scheduler_errors"] == {"squeue": "offline", "sacct": "offline"}


def test_atomic_json_replaces_complete_document(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "status.json"
    monitor.atomic_json(output, {"state": "RUNNING"})
    assert json.loads(output.read_text()) == {"state": "RUNNING"}
    monitor.atomic_json(output, {"state": "COMPLETED"})
    assert json.loads(output.read_text()) == {"state": "COMPLETED"}


@pytest.mark.parametrize(
    ("handoff", "expected_return_code", "expected_reason"),
    [
        (False, 2, "time_budget_exhausted"),
        (True, 0, "relay_handoff"),
    ],
)
def test_timeout_can_be_an_expected_relay_handoff(
    tmp_path: Path,
    handoff: bool,
    expected_return_code: int,
    expected_reason: str,
) -> None:
    status_json = tmp_path / "status.json"
    report = {
        "generated_at": "2026-07-15T00:00:00+00:00",
        "jobs": {
            "11": {
                "label": "active",
                "state": "RUNNING",
                "terminal": False,
                "success": False,
            }
        },
        "all_terminal": False,
        "any_failed": False,
        "scheduler_errors": {"squeue": "", "sacct": ""},
    }
    argv = [
        "--job",
        "active=11",
        "--status-json",
        str(status_json),
        "--poll-seconds",
        "1",
        "--heartbeat-seconds",
        "1",
        "--max-seconds",
        "0.5",
    ]
    if handoff:
        argv.append("--handoff-on-timeout")

    with patch.object(monitor, "snapshot", return_value=report), patch.object(
        monitor.time,
        "monotonic",
        side_effect=[0.0, 1.0],
    ):
        return_code = monitor.main(argv)

    saved = json.loads(status_json.read_text())
    assert return_code == expected_return_code
    assert saved["monitor_handoff"] is handoff
    assert saved["monitor_exit_reason"] == expected_reason
