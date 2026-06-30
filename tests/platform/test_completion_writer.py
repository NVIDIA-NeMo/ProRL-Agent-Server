"""Tests for the gateway CompletionWriter."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from polar.gateway.completion_writer import CompletionWriter, _truncate_value


def test_truncate_value_string() -> None:
    long = "a" * 100
    truncated = _truncate_value(long, max_bytes=20)
    assert isinstance(truncated, str)
    assert len(truncated.encode("utf-8")) <= 24  # plus ellipsis


def test_truncate_value_under_budget() -> None:
    short = {"foo": "bar"}
    assert _truncate_value(short, max_bytes=1024) == short


@pytest.mark.asyncio
async def test_writer_persists_records(tmp_path: Path) -> None:
    writer = CompletionWriter(save_dir=tmp_path, queue_size=8)
    await writer.start()
    for i in range(3):
        assert writer.enqueue(
            task_id="t1",
            session_id="sess1",
            completion_id=f"id{i}",
            record={"completion_id": f"id{i}", "payload": {"i": i}},
        )
    # close() is the flush barrier; callers do not need an arbitrary sleep.
    await writer.close()

    out_dir = tmp_path / "task_t1" / "sessions" / "sess1" / "completions"
    files = sorted(out_dir.glob("*.json"))
    assert len(files) == 3
    first = json.loads(files[0].read_text())
    assert first["payload"]["i"] == 0
    assert writer.stats()["persisted"] == 3
    assert writer.stats()["dropped"] == 0


@pytest.mark.asyncio
async def test_writer_truncates_records_on_writer_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = CompletionWriter(
        save_dir=tmp_path,
        max_field_bytes=32,
        write_workers=1,
    )
    enqueue_thread = threading.get_ident()
    truncate_threads: list[int] = []
    original_truncate = writer._truncate_record

    def tracked_truncate(record: dict) -> dict:
        truncate_threads.append(threading.get_ident())
        return original_truncate(record)

    monkeypatch.setattr(writer, "_truncate_record", tracked_truncate)
    await writer.start()
    assert writer.enqueue(
        task_id="threaded",
        session_id="session",
        completion_id="completion",
        record={"large": "x" * 1024},
    )
    await writer.close()

    assert truncate_threads
    assert all(thread_id != enqueue_thread for thread_id in truncate_threads)
    output = next(
        (tmp_path / "task_threaded" / "sessions" / "session" / "completions").glob("*.json")
    )
    assert json.loads(output.read_text())["large"].endswith("…")


@pytest.mark.asyncio
async def test_writer_disabled_when_no_save_dir() -> None:
    writer = CompletionWriter(save_dir=None, enabled=True)
    await writer.start()  # no-op
    ok = writer.enqueue(task_id="t", session_id="s", completion_id="c", record={})
    assert ok is False
    await writer.close()


@pytest.mark.asyncio
async def test_writer_requires_task_id(tmp_path: Path) -> None:
    writer = CompletionWriter(save_dir=tmp_path)
    await writer.start()
    ok = writer.enqueue(task_id=None, session_id="s", completion_id="c", record={})
    assert ok is False
    await writer.close()


@pytest.mark.asyncio
async def test_writer_backpressures_without_dropping_queue_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = CompletionWriter(
        save_dir=tmp_path,
        queue_size=1,
        write_workers=1,
        batch_size=1,
    )
    original_write = writer._write_to_disk

    def slow_write(item) -> None:
        time.sleep(0.01)
        original_write(item)

    monkeypatch.setattr(writer, "_write_to_disk", slow_write)
    await writer.start()
    for index in range(12):
        assert writer.enqueue(
            task_id="pressure",
            session_id="session",
            completion_id=f"completion-{index}",
            record={"index": index},
        )
    await writer.close()

    files = list(
        (tmp_path / "task_pressure" / "sessions" / "session" / "completions").glob("*.json")
    )
    stats = writer.stats()
    assert len(files) == 12
    assert stats["accepted"] == 12
    assert stats["persisted"] == 12
    assert stats["failed"] == 0
    assert stats["dropped"] == 0
    assert stats["queue_full_events"] > 0
    assert stats["backpressure_ms_total"] > 0


@pytest.mark.asyncio
async def test_writer_retries_transient_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = CompletionWriter(
        save_dir=tmp_path,
        write_max_attempts=3,
        retry_backoff_seconds=0,
    )
    original_write = writer._write_to_disk
    attempts = 0

    def flaky_write(item) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("transient storage failure")
        original_write(item)

    monkeypatch.setattr(writer, "_write_to_disk", flaky_write)
    await writer.start()
    assert writer.enqueue(
        task_id="retry",
        session_id="session",
        completion_id="completion",
        record={"value": 1},
    )
    await writer.close()

    assert attempts == 3
    assert writer.stats()["persisted"] == 1
    assert writer.stats()["failed"] == 0
