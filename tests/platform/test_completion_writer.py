"""Tests for the gateway CompletionWriter."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

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


def test_truncate_value_dict_omitted_keys_include_triggering_key() -> None:
    value = {"a": "x" * 10, "b": "y" * 10, "c": "z" * 10}
    truncated = _truncate_value(value, max_bytes=15)
    assert truncated["a"] == "x" * 10
    assert truncated["__truncated"] is True
    # "b" triggered truncation, so it is omitted and must be listed.
    assert truncated["_truncated_keys_omitted"] == ["b", "c"]


def test_truncate_value_dict_omitted_keys_when_first_key_overflows() -> None:
    value = {"a": "x" * 10, "b": "y" * 10}
    truncated = _truncate_value(value, max_bytes=5)
    assert truncated["__truncated"] is True
    assert truncated["_truncated_keys_omitted"] == ["a", "b"]


@pytest.mark.asyncio
async def test_writer_persists_records(tmp_path: Path) -> None:
    writer = CompletionWriter(save_dir=tmp_path, queue_size=8)
    await writer.start()
    for i in range(3):
        writer.enqueue(
            task_id="t1",
            session_id="sess1",
            completion_id=f"id{i}",
            record={"completion_id": f"id{i}", "payload": {"i": i}},
        )
    # Give the drain loop a moment to flush.
    await asyncio.sleep(0.2)
    await writer.close()

    out_dir = tmp_path / "task_t1" / "sessions" / "sess1" / "completions"
    files = sorted(out_dir.glob("*.json"))
    assert len(files) == 3
    first = json.loads(files[0].read_text())
    assert first["payload"]["i"] == 0


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
