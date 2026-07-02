"""Filesystem index over `<save_dir>/task_*/`.

Scans rollout result directories so the platform service can list and detail
historical tasks even when the rollout server has been restarted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TaskSummary:
    task_id: str
    status: str
    harness: str | None = None
    model: str | None = None
    num_samples: int = 0
    completed_sessions: int = 0
    errored_sessions: int = 0
    mean_reward: float | None = None
    mean_traces: float | None = None
    mean_completions: float | None = None
    created_at: float | None = None
    updated_at: float | None = None
    save_dir_path: str = ""
    session_files: list[str] = field(default_factory=list)
    source: str = "filesystem"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "harness": self.harness,
            "model": self.model,
            "num_samples": self.num_samples,
            "completed_sessions": self.completed_sessions,
            "errored_sessions": self.errored_sessions,
            "mean_reward": self.mean_reward,
            "mean_traces": self.mean_traces,
            "mean_completions": self.mean_completions,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "save_dir_path": self.save_dir_path,
            "session_files": list(self.session_files),
            "source": self.source,
        }


@dataclass(slots=True)
class SessionSummary:
    session_id: str
    task_id: str
    status: str
    node_id: str | None = None
    reward: float | None = None
    timing: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    file_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "status": self.status,
            "node_id": self.node_id,
            "reward": self.reward,
            "timing": dict(self.timing),
            "error": self.error,
            "file_path": self.file_path,
        }


def _trace_reward(session_json: dict[str, Any]) -> float | None:
    trajectory = session_json.get("trajectory") or {}
    traces = trajectory.get("traces") or []
    if traces:
        last = traces[-1]
        if isinstance(last, dict):
            reward = last.get("reward")
            if isinstance(reward, (int, float)):
                return float(reward)
    # Fall back to evaluation.outcome_reward in trajectory.metadata.
    metadata = trajectory.get("metadata") or {}
    evaluation = metadata.get("evaluation") or {}
    outcome = evaluation.get("outcome_reward")
    if isinstance(outcome, (int, float)):
        return float(outcome)
    return None


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not parse %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _extract_session_summary(path: Path) -> SessionSummary | None:
    data = _safe_load_json(path)
    if data is None:
        return None
    session_id = data.get("session_id")
    task_id = data.get("task_id")
    if not session_id or not task_id:
        return None
    return SessionSummary(
        session_id=str(session_id),
        task_id=str(task_id),
        status=str(data.get("status") or "UNKNOWN"),
        node_id=data.get("node_id"),
        reward=_trace_reward(data),
        timing=dict(data.get("timing") or {}),
        error=data.get("error"),
        file_path=str(path),
    )


def _harness_and_model_from(session_data: dict[str, Any]) -> tuple[str | None, str | None]:
    trajectory = session_data.get("trajectory") or {}
    metadata = trajectory.get("metadata") or {}
    api_type = metadata.get("api_type")
    model_used = metadata.get("model_used") or metadata.get("model_requested")
    # Harness is not directly on the session result, but task_id usually encodes it.
    task_id = session_data.get("task_id") or ""
    harness = None
    for known in (
        "claude_code",
        "codex",
        "gemini_cli",
        "opencode",
        "qwen_code",
        "swe_agent",
        "pi",
    ):
        if known in task_id:
            harness = known
            break
    if harness is None and api_type:
        harness = str(api_type)
    return harness, model_used


def _read_index_file(task_dir: Path) -> dict[str, Any] | None:
    return _safe_load_json(task_dir / "_index.json")


def _scan_task_dir(task_dir: Path) -> TaskSummary | None:
    if not task_dir.is_dir():
        return None
    task_id = task_dir.name
    if task_id.startswith("task_"):
        task_id = task_id[len("task_"):]
    session_files = sorted(task_dir.glob("ses_*.json"))
    summary = TaskSummary(
        task_id=task_id,
        status="unknown",
        save_dir_path=str(task_dir),
        session_files=[str(p) for p in session_files],
    )
    try:
        stat = task_dir.stat()
        summary.created_at = stat.st_ctime
        summary.updated_at = stat.st_mtime
    except OSError:
        pass

    index = _read_index_file(task_dir)
    if index is not None:
        summary.status = str(index.get("status") or "completed")
        summary.harness = index.get("harness")
        summary.model = index.get("model")
        summary.num_samples = int(index.get("num_samples") or 0)
        summary.completed_sessions = int(index.get("completed_sessions") or 0)
        summary.mean_reward = index.get("mean_reward")
        summary.mean_traces = index.get("mean_traces")
        summary.mean_completions = index.get("mean_completions")
        if index.get("created_at") is not None:
            try:
                summary.created_at = float(index["created_at"])
            except (TypeError, ValueError):
                pass
        if index.get("updated_at") is not None:
            try:
                summary.updated_at = float(index["updated_at"])
            except (TypeError, ValueError):
                pass
        return summary

    # No index file — fall back to scanning each session file.
    rewards: list[float] = []
    trace_counts: list[int] = []
    completion_counts: list[int] = []
    completed = 0
    errored = 0
    earliest = summary.created_at
    latest = summary.updated_at
    harness: str | None = None
    model: str | None = None
    for session_path in session_files:
        data = _safe_load_json(session_path)
        if not isinstance(data, dict):
            continue
        status = str(data.get("status") or "")
        if status == "COMPLETED":
            completed += 1
        elif status in {"ERROR", "TIMEOUT"}:
            errored += 1
        reward = _trace_reward(data)
        if reward is not None:
            rewards.append(reward)
        traj = data.get("trajectory") or {}
        traces = traj.get("traces") or []
        trace_counts.append(len(traces))
        record_count = (traj.get("metadata") or {}).get("record_count")
        completion_counts.append(int(record_count) if isinstance(record_count, int) else len(traces))
        try:
            mtime = session_path.stat().st_mtime
            ctime = session_path.stat().st_ctime
        except OSError:
            mtime = None
            ctime = None
        if ctime is not None:
            earliest = ctime if earliest is None else min(earliest, ctime)
        if mtime is not None:
            latest = mtime if latest is None else max(latest, mtime)
        if harness is None or model is None:
            h, m = _harness_and_model_from(data)
            harness = harness or h
            model = model or m
    summary.harness = harness
    summary.model = model
    summary.num_samples = len(session_files)
    summary.completed_sessions = completed
    summary.errored_sessions = errored
    summary.mean_reward = sum(rewards) / len(rewards) if rewards else None
    summary.mean_traces = sum(trace_counts) / len(trace_counts) if trace_counts else None
    summary.mean_completions = (
        sum(completion_counts) / len(completion_counts) if completion_counts else None
    )
    summary.created_at = earliest
    summary.updated_at = latest
    if completed + errored == len(session_files) and session_files:
        summary.status = "completed"
    elif session_files:
        summary.status = "running"
    return summary


class FsIndex:
    """Maintains an in-memory summary of `<save_dir>/task_*/` directories."""

    def __init__(self, save_dir: Path) -> None:
        self.save_dir = Path(save_dir)
        self._lock = threading.RLock()
        self._tasks: dict[str, TaskSummary] = {}
        self._last_scan: float = 0.0
        self._stop = asyncio.Event()
        self._poll_task: asyncio.Task | None = None
        self.poll_interval_seconds: float = 2.0
        self._listeners: list[asyncio.AbstractEventLoop] = []
        self._on_change_callbacks: list = []

    def add_listener(self, callback) -> None:
        """Register a `callback(task_summary)` invoked on every detected change."""
        self._on_change_callbacks.append(callback)

    def scan(self) -> None:
        """Rebuild the index by walking save_dir. Safe to call from any thread."""
        if not self.save_dir.exists():
            return
        new_tasks: dict[str, TaskSummary] = {}
        for entry in sorted(self.save_dir.iterdir()):
            if not entry.is_dir():
                continue
            if not entry.name.startswith("task_"):
                continue
            summary = _scan_task_dir(entry)
            if summary is None:
                continue
            new_tasks[summary.task_id] = summary
        with self._lock:
            previous = self._tasks
            self._tasks = new_tasks
            self._last_scan = time.time()
        # Notify changed tasks (added or updated_at changed).
        for tid, summary in new_tasks.items():
            prior = previous.get(tid)
            if prior is None or prior.updated_at != summary.updated_at:
                for cb in self._on_change_callbacks:
                    try:
                        cb(summary)
                    except Exception:
                        logger.exception("FsIndex listener failed")

    def get_task(self, task_id: str) -> TaskSummary | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(
        self,
        *,
        status: str | None = None,
        harness: str | None = None,
        since: float | None = None,
        limit: int = 200,
    ) -> list[TaskSummary]:
        with self._lock:
            tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        if harness:
            tasks = [t for t in tasks if t.harness == harness]
        if since is not None:
            tasks = [t for t in tasks if (t.updated_at or 0) >= since]
        tasks.sort(key=lambda t: (t.updated_at or 0), reverse=True)
        return tasks[:limit]

    def list_sessions_for(self, task_id: str) -> list[SessionSummary]:
        summary = self.get_task(task_id)
        if summary is None:
            return []
        out: list[SessionSummary] = []
        for path_str in summary.session_files:
            session_summary = _extract_session_summary(Path(path_str))
            if session_summary is not None:
                out.append(session_summary)
        return out

    def session_file_for(self, session_id: str) -> Path | None:
        with self._lock:
            for summary in self._tasks.values():
                for path_str in summary.session_files:
                    if Path(path_str).stem == f"ses_{session_id}":
                        return Path(path_str)
        return None

    def task_dir_for(self, task_id: str) -> Path | None:
        candidate = self.save_dir / f"task_{task_id}"
        if candidate.exists():
            return candidate
        return None

    async def start_polling(self) -> None:
        if self._poll_task is not None:
            return
        loop = asyncio.get_running_loop()
        self._poll_task = loop.create_task(self._poll_loop())

    async def stop_polling(self) -> None:
        self._stop.set()
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
            self._poll_task = None
        self._stop = asyncio.Event()

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self.scan)
            except Exception:
                logger.exception("FsIndex scan failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                continue


__all__ = [
    "FsIndex",
    "SessionSummary",
    "TaskSummary",
]
