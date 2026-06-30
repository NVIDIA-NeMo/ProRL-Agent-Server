"""Backpressured on-disk persistence for gateway completion records.

Each gateway owns a bounded, thread-safe queue and a configurable pool of
writer threads.  A completion is never discarded merely because the queue is
full: after a fast ``put_nowait`` attempt, the producer waits for a writer to
free a slot.  Writer threads do not depend on the asyncio event loop, so this
backpressure cannot deadlock the gateway and only reaches the request path
during sustained storage overload.

Records are written one file per completion under::

    <save_dir>/task_<task_id>/sessions/<session_id>/completions/<NNNN>-<id>.json

The large default queue absorbs normal rollout bursts; bounded backpressure
keeps memory use finite and preserves the per-completion audit trail.  Runtime
queue pressure and write failures are available through :meth:`stats`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import json
import logging
from pathlib import Path
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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


_STOP = object()


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
                out["_truncated_keys_omitted"] = list(value.keys())[len(out) :]
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
    """Drain completion records to disk without dropping queue overflow."""

    def __init__(
        self,
        save_dir: Path | None,
        *,
        max_field_bytes: int = 1 * 1024 * 1024,
        queue_size: int = 16_384,
        write_workers: int = 8,
        batch_size: int = 16,
        write_max_attempts: int = 3,
        retry_backoff_seconds: float = 0.1,
        enabled: bool = True,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if write_workers <= 0:
            raise ValueError("write_workers must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if write_max_attempts <= 0:
            raise ValueError("write_max_attempts must be positive")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")

        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.max_field_bytes = max_field_bytes
        self.queue_size = queue_size
        self.write_workers = write_workers
        self.batch_size = batch_size
        self.write_max_attempts = write_max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.enabled = enabled and self.save_dir is not None

        self._queue: queue.Queue[_WriteItem | object] | None = None
        self._threads: list[threading.Thread] = []
        self._accepting = False
        self._lifecycle_lock = threading.Lock()
        self._sequences: dict[str, int] = {}
        self._seq_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._accepted_count = 0
        self._persisted_count = 0
        self._failed_count = 0
        self._rejected_count = 0
        self._queue_full_count = 0
        self._backpressure_seconds = 0.0
        self._max_backpressure_seconds = 0.0

    async def start(self) -> None:
        if not self.enabled:
            return
        with self._lifecycle_lock:
            if self._queue is not None:
                return
            work_queue: queue.Queue[_WriteItem | object] = queue.Queue(maxsize=self.queue_size)
            self._queue = work_queue
            self._accepting = True
            self._threads = [
                threading.Thread(
                    target=self._worker_loop,
                    args=(work_queue,),
                    name=f"polar-completion-writer-{index}",
                    daemon=True,
                )
                for index in range(self.write_workers)
            ]
            for thread in self._threads:
                thread.start()

    async def close(self) -> None:
        with self._lifecycle_lock:
            work_queue = self._queue
            if work_queue is None:
                return
            # Holding this lock through the accepting transition guarantees
            # that no producer can enqueue after the drain barrier begins.
            self._accepting = False
            threads = list(self._threads)

        # queue.join() is a blocking primitive.  Keep it off the event loop and
        # wait without a short timeout: returning early would silently lose the
        # exact audit records this writer exists to preserve.
        await asyncio.to_thread(work_queue.join)
        for _ in threads:
            work_queue.put(_STOP)
        await asyncio.gather(*(asyncio.to_thread(thread.join) for thread in threads))

        with self._lifecycle_lock:
            if self._queue is work_queue:
                self._queue = None
                self._threads = []

    def enqueue(
        self,
        *,
        task_id: str | None,
        session_id: str,
        completion_id: str,
        record: dict[str, Any],
    ) -> bool:
        """Enqueue one audit record, applying bounded backpressure when full.

        This method is safe from any thread.  It is non-blocking in the normal
        case.  When storage falls behind enough to fill the bounded queue, it
        waits for a writer-thread slot instead of discarding the completion.
        """
        if not self.enabled or not task_id:
            return False

        with self._seq_lock:
            self._sequences[session_id] = self._sequences.get(session_id, 0) + 1
            sequence = self._sequences[session_id]
        item = _WriteItem(
            task_id=task_id,
            session_id=session_id,
            sequence=sequence,
            completion_id=completion_id,
            # Only take the cheap top-level snapshot on the request/event-loop
            # path.  Recursive sizing/truncation can walk hundreds of thousands
            # of token/logprob values and belongs on the writer worker.
            payload=dict(record),
        )

        # Serialize the accepting check and put against close().  Worker
        # threads drain independently, so a full queue can still make progress
        # while this lock is held.
        with self._lifecycle_lock:
            work_queue = self._queue
            if work_queue is None or not self._accepting:
                self._increment_stat("rejected")
                return False
            try:
                work_queue.put_nowait(item)
            except queue.Full:
                started = time.monotonic()
                with self._stats_lock:
                    self._queue_full_count += 1
                    queue_full_count = self._queue_full_count
                if queue_full_count == 1 or queue_full_count % 100 == 0:
                    logger.warning(
                        "CompletionWriter queue full (%d pressure events); "
                        "backpressuring session %s (no audit record dropped)",
                        queue_full_count,
                        session_id,
                    )
                work_queue.put(item)
                elapsed = time.monotonic() - started
                with self._stats_lock:
                    self._backpressure_seconds += elapsed
                    self._max_backpressure_seconds = max(
                        self._max_backpressure_seconds,
                        elapsed,
                    )
            self._increment_stat("accepted")
        return True

    def stats(self) -> dict[str, Any]:
        """Return a thread-safe persistence snapshot for health telemetry."""
        with self._lifecycle_lock:
            work_queue = self._queue
            accepting = self._accepting
        with self._stats_lock:
            return {
                "enabled": self.enabled,
                "accepting": accepting,
                "queue_depth": work_queue.qsize() if work_queue is not None else 0,
                "queue_capacity": self.queue_size,
                "write_workers": self.write_workers,
                "batch_size": self.batch_size,
                "accepted": self._accepted_count,
                "persisted": self._persisted_count,
                "failed": self._failed_count,
                "rejected": self._rejected_count,
                # Overflow never drops records; retain an explicit metric so a
                # dashboard/alert can distinguish fixed behavior from absence.
                "dropped": 0,
                "queue_full_events": self._queue_full_count,
                "backpressure_ms_total": self._backpressure_seconds * 1000.0,
                "backpressure_ms_max": self._max_backpressure_seconds * 1000.0,
            }

    def _increment_stat(self, name: str) -> None:
        with self._stats_lock:
            if name == "accepted":
                self._accepted_count += 1
            elif name == "persisted":
                self._persisted_count += 1
            elif name == "failed":
                self._failed_count += 1
            elif name == "rejected":
                self._rejected_count += 1
            else:  # pragma: no cover - private programming error guard
                raise ValueError(f"unknown CompletionWriter statistic: {name}")

    def _worker_loop(self, work_queue: queue.Queue[_WriteItem | object]) -> None:
        while True:
            first = work_queue.get()
            if first is _STOP:
                work_queue.task_done()
                return

            batch = [first]
            for _ in range(self.batch_size - 1):
                try:
                    next_item = work_queue.get_nowait()
                except queue.Empty:
                    break
                # Stop sentinels are only inserted after all records drain, so
                # one cannot normally occur here.  Preserve it defensively.
                if next_item is _STOP:
                    work_queue.task_done()
                    break
                batch.append(next_item)
            self._write_batch(batch, work_queue)

    def _write_batch(
        self,
        batch: Sequence[_WriteItem | object],
        work_queue: queue.Queue[_WriteItem | object],
    ) -> None:
        for value in batch:
            assert isinstance(value, _WriteItem)
            try:
                self._write_with_retries(value)
            finally:
                work_queue.task_done()

    def _write_with_retries(self, item: _WriteItem) -> None:
        try:
            # Do the CPU-heavy recursive sizing on this writer thread rather
            # than in enqueue(), which is called by the gateway event loop.
            item.payload = self._truncate_record(item.payload)
        except Exception:
            self._increment_stat("failed")
            logger.exception(
                "CompletionWriter failed to prepare %s/%s for persistence",
                item.session_id,
                item.completion_id,
            )
            return

        for attempt in range(1, self.write_max_attempts + 1):
            try:
                self._write_to_disk(item)
            except Exception:
                if attempt >= self.write_max_attempts:
                    self._increment_stat("failed")
                    logger.exception(
                        "CompletionWriter permanently failed to write %s/%s after %d attempts",
                        item.session_id,
                        item.completion_id,
                        attempt,
                    )
                    return
                logger.warning(
                    "CompletionWriter write attempt %d/%d failed for %s/%s; retrying",
                    attempt,
                    self.write_max_attempts,
                    item.session_id,
                    item.completion_id,
                    exc_info=True,
                )
                if self.retry_backoff_seconds:
                    time.sleep(self.retry_backoff_seconds * attempt)
            else:
                self._increment_stat("persisted")
                return

    def _truncate_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {key: _truncate_value(value, self.max_field_bytes) for key, value in record.items()}

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

    def _write_to_disk(self, item: _WriteItem) -> None:
        path = self._path_for(item)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(item.payload)
        payload.setdefault("__written_at", datetime.now(timezone.utc).isoformat())
        path.write_text(json.dumps(payload, default=str))


__all__ = ["CompletionWriter"]
