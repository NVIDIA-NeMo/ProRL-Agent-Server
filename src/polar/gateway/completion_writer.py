"""Asynchronous on-disk persistence for gateway CompletionRecords.

This runs as a single background asyncio task per gateway process. It drains a
bounded queue and writes one JSON file per completion under:

    <save_dir>/task_<task_id>/sessions/<session_id>/completions/<NNNN>-<id>.json

If the queue is full, completions are dropped (with a warning) so the proxy hot
path never blocks. The in-memory copy in `SessionStore` remains authoritative.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRUNCATED_MARKER = "__truncated"


@dataclass(slots=True)
class _WriteItem:
    task_id: str
    session_id: str
    sequence: int
    completion_id: str
    payload: dict[str, Any]


def _approx_byte_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8", errors="ignore"))
    if isinstance(value, (int, float, bool)):
        return 16
    if isinstance(value, dict):
        return sum(_approx_byte_size(k) + _approx_byte_size(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_approx_byte_size(v) for v in value)
    try:
        return len(json.dumps(value, default=str).encode("utf-8", errors="ignore"))
    except Exception:
        return 0


def _truncate_value(value: Any, max_bytes: int) -> Any:
    size = _approx_byte_size(value)
    if size <= max_bytes:
        return value
    if isinstance(value, str):
        # Keep the first part within budget
        encoded = value.encode("utf-8", errors="ignore")[:max_bytes]
        return encoded.decode("utf-8", errors="ignore") + "…"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        running = 0
        for key, item in value.items():
            piece_size = _approx_byte_size(item)
            if running + piece_size > max_bytes:
                out[_TRUNCATED_MARKER] = True
                out["_truncated_keys_omitted"] = list(value.keys())[len(out):]
                break
            out[key] = item
            running += piece_size
        return out
    if isinstance(value, list):
        out_list: list[Any] = []
        running = 0
        for item in value:
            piece_size = _approx_byte_size(item)
            if running + piece_size > max_bytes:
                out_list.append({_TRUNCATED_MARKER: True})
                break
            out_list.append(item)
            running += piece_size
        return out_list
    return value


class CompletionWriter:
    """Drains queued completion records to disk on a background asyncio task."""

    def __init__(
        self,
        save_dir: Path | None,
        *,
        max_field_bytes: int = 1 * 1024 * 1024,
        queue_size: int = 1024,
        enabled: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.max_field_bytes = max_field_bytes
        self.queue_size = queue_size
        self.enabled = enabled and self.save_dir is not None
        self._queue: asyncio.Queue[_WriteItem | None] | None = None
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sequences: dict[str, int] = {}
        self._seq_lock = threading.Lock()
        self._drop_count = 0

    async def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self.queue_size)
        self._task = self._loop.create_task(self._drain())

    async def close(self) -> None:
        if self._task is None or self._queue is None:
            return
        await self._queue.put(None)
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            self._task.cancel()
        self._task = None
        self._queue = None
        self._loop = None

    def enqueue(
        self,
        *,
        task_id: str | None,
        session_id: str,
        completion_id: str,
        record: dict[str, Any],
    ) -> bool:
        """Non-blocking enqueue; safe to call from any thread."""
        if not self.enabled:
            return False
        if not task_id:
            return False
        if self._queue is None or self._loop is None:
            return False
        with self._seq_lock:
            self._sequences[session_id] = self._sequences.get(session_id, 0) + 1
            sequence = self._sequences[session_id]
        truncated = self._truncate_record(record)
        item = _WriteItem(
            task_id=task_id,
            session_id=session_id,
            sequence=sequence,
            completion_id=completion_id,
            payload=truncated,
        )
        try:
            # _loop.call_soon_threadsafe lets sync `save_message` callers enqueue safely.
            self._loop.call_soon_threadsafe(self._put_nowait_or_drop, item)
        except RuntimeError:
            return False
        return True

    def _put_nowait_or_drop(self, item: _WriteItem) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._drop_count += 1
            if self._drop_count == 1 or self._drop_count % 100 == 0:
                logger.warning(
                    "CompletionWriter queue full (%d drops); persistence dropped for session %s",
                    self._drop_count,
                    item.session_id,
                )

    def _truncate_record(self, record: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in record.items():
            out[key] = _truncate_value(value, self.max_field_bytes)
        return out

    def _path_for(self, item: _WriteItem) -> Path | None:
        if self.save_dir is None:
            return None
        return (
            self.save_dir
            / f"task_{item.task_id}"
            / "sessions"
            / item.session_id
            / "completions"
            / f"{item.sequence:04d}-{item.completion_id}.json"
        )

    async def _drain(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            if item is None:
                return
            try:
                await asyncio.to_thread(self._write_to_disk, item)
            except Exception:
                logger.exception(
                    "CompletionWriter failed to write %s/%s",
                    item.session_id,
                    item.completion_id,
                )

    def _write_to_disk(self, item: _WriteItem) -> None:
        path = self._path_for(item)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(item.payload)
        payload.setdefault("__written_at", datetime.now(timezone.utc).isoformat())
        path.write_text(json.dumps(payload, default=str, ensure_ascii=False))


__all__ = ["CompletionWriter"]
