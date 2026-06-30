from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from polar.agent.models import AgentSpec
from polar.rollout.balancer import NodeScheduler
from polar.rollout.models import (
    SessionContext,
    SessionDispatchRequest,
    SessionResult,
    SessionStatus,
    TaskRequest,
)
from polar.rollout.pipeline import Pipeline
from polar.trajectory.models import Trace, Trajectory


def _sessions(*, count: int, threshold: int) -> list[SessionContext]:
    request = TaskRequest(
        task_id="task-early-stop",
        instruction="test early stopping",
        num_samples=count,
        early_stop_min_usable_sessions=threshold,
        agent=AgentSpec(harness="codex"),
    )
    return [
        SessionContext(
            session_id=f"session-{index}",
            task_id=request.task_id,
            request=request,
        )
        for index in range(count)
    ]


def _result(
    session: SessionContext,
    *,
    status: SessionStatus = SessionStatus.COMPLETED,
    usable: bool = True,
) -> SessionResult:
    traces = []
    if usable:
        traces = [
            Trace(
                prompt_ids=[1],
                response_ids=[2],
                loss_mask=[1],
                reward=1.0,
            )
        ]
    return SessionResult(
        session_id=session.session_id,
        task_id=session.task_id,
        status=status,
        trajectory=Trajectory(status=status, traces=traces),
    )


def _pipeline(tmp_path: Path | None = None) -> Pipeline:
    pipeline = Pipeline(
        callback_url="http://rollout/callback",
        save_dir=None if tmp_path is None else str(tmp_path),
        scheduler=NodeScheduler(),
    )
    response = SimpleNamespace(status_code=200, raise_for_status=lambda: None)
    pipeline._client = SimpleNamespace(delete=AsyncMock(return_value=response))
    pipeline._started = True
    return pipeline


def _install_waitable_dispatch(monkeypatch, pipeline: Pipeline) -> None:
    async def dispatch(session: SessionContext) -> SessionDispatchRequest:
        session.node_id = "node-1"
        session.gateway_url = "http://gateway"
        return SessionDispatchRequest(
            session_id=session.session_id,
            task_id=session.task_id,
            instruction=session.request.instruction,
            remaining_timeout_seconds=60.0,
            agent=session.request.agent,
            callback_url=pipeline.callback_url,
        )

    async def wait_for_result(
        _session: SessionContext,
        _request: SessionDispatchRequest,
        future: asyncio.Future[SessionResult],
    ) -> SessionResult:
        return await future

    monkeypatch.setattr(pipeline, "_dispatch_session", dispatch)
    monkeypatch.setattr(pipeline, "_wait_for_result", wait_for_result)


async def _wait_for_pending(pipeline: Pipeline, expected: int) -> None:
    async with asyncio.timeout(1.0):
        while pipeline.status()["pending_sessions"] != expected:
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_early_stop_preserves_order_and_finalizes_every_session_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sessions = _sessions(count=3, threshold=2)
    pipeline = _pipeline(tmp_path)
    _install_waitable_dispatch(monkeypatch, pipeline)
    callback_ids: list[str] = []
    persisted_ids: list[str] = []
    persist_result = pipeline._persist_result

    def record_persist(result: SessionResult) -> None:
        persisted_ids.append(result.session_id)
        persist_result(result)

    monkeypatch.setattr(pipeline, "_persist_result", record_persist)

    async def on_result(result: SessionResult) -> None:
        callback_ids.append(result.session_id)

    batch = asyncio.create_task(pipeline.run_batch(sessions, on_result=on_result))
    await _wait_for_pending(pipeline, 3)

    # Complete out of order. The third session is still running when the
    # second usable result reaches the batch threshold.
    assert await pipeline.accept_callback_result(_result(sessions[1]))
    assert await pipeline.accept_callback_result(_result(sessions[0]))
    results = await asyncio.wait_for(batch, timeout=1.0)

    assert [result.session_id for result in results] == [
        "session-0",
        "session-1",
        "session-2",
    ]
    cancelled = results[2]
    assert cancelled.status == SessionStatus.ERROR
    assert cancelled.trajectory.traces == []
    assert cancelled.metadata["early_stop_cancelled"] is True
    assert cancelled.metadata["fully_masked"] is True
    assert cancelled.trajectory.metadata["early_stop_cancelled"] is True
    assert cancelled.metadata["early_stop_min_usable_sessions"] == 2
    assert cancelled.metadata["early_stop_usable_sessions"] == 2

    assert Counter(callback_ids) == Counter(session.session_id for session in sessions)
    assert Counter(persisted_ids) == Counter(session.session_id for session in sessions)
    persisted = sorted(tmp_path.glob("task_task-early-stop/ses_*.json"))
    assert len(persisted) == 3
    assert {path.stem for path in persisted} == {
        "ses_session-0",
        "ses_session-1",
        "ses_session-2",
    }
    assert pipeline._client.delete.await_count == 3
    assert pipeline.status()["pending_sessions"] == 0


@pytest.mark.asyncio
async def test_early_stop_cleans_all_stragglers_concurrently(monkeypatch) -> None:
    sessions = _sessions(count=4, threshold=1)
    pipeline = _pipeline()
    _install_waitable_dispatch(monkeypatch, pipeline)
    cleanup_release = asyncio.Event()
    all_stragglers_cleaning = asyncio.Event()
    active_straggler_cleanups = 0
    peak_straggler_cleanups = 0

    async def cleanup(session: SessionContext) -> None:
        nonlocal active_straggler_cleanups, peak_straggler_cleanups
        if not session.early_stop_requested:
            return
        active_straggler_cleanups += 1
        peak_straggler_cleanups = max(
            peak_straggler_cleanups,
            active_straggler_cleanups,
        )
        if active_straggler_cleanups == 3:
            all_stragglers_cleaning.set()
        try:
            await cleanup_release.wait()
        finally:
            active_straggler_cleanups -= 1

    monkeypatch.setattr(pipeline, "_cleanup_session", cleanup)
    batch = asyncio.create_task(pipeline.run_batch(sessions))
    await _wait_for_pending(pipeline, 4)

    assert await pipeline.accept_callback_result(_result(sessions[0]))
    await asyncio.wait_for(all_stragglers_cleaning.wait(), timeout=1.0)
    assert peak_straggler_cleanups == 3

    cleanup_release.set()
    results = await asyncio.wait_for(batch, timeout=1.0)
    assert sum(result.metadata.get("early_stop_cancelled") is True for result in results) == 3
    assert pipeline.status()["pending_sessions"] == 0


@pytest.mark.asyncio
async def test_no_early_stop_when_usable_threshold_is_not_reached(monkeypatch) -> None:
    sessions = _sessions(count=3, threshold=2)
    pipeline = _pipeline()
    _install_waitable_dispatch(monkeypatch, pipeline)
    batch = asyncio.create_task(pipeline.run_batch(sessions))
    await _wait_for_pending(pipeline, 3)

    assert await pipeline.accept_callback_result(_result(sessions[0]))
    assert await pipeline.accept_callback_result(_result(sessions[1], usable=False))
    assert await pipeline.accept_callback_result(
        _result(sessions[2], status=SessionStatus.ERROR, usable=False)
    )
    results = await asyncio.wait_for(batch, timeout=1.0)

    assert [result.session_id for result in results] == [
        "session-0",
        "session-1",
        "session-2",
    ]
    assert not any(result.metadata.get("early_stop_cancelled") for result in results)
    assert not any(session.early_stop_requested for session in sessions)
    assert pipeline.status()["pending_sessions"] == 0


@pytest.mark.asyncio
async def test_external_batch_cancellation_does_not_emit_early_stop_results(
    monkeypatch,
    tmp_path: Path,
) -> None:
    sessions = _sessions(count=2, threshold=1)
    pipeline = _pipeline(tmp_path)
    _install_waitable_dispatch(monkeypatch, pipeline)
    callback_ids: list[str] = []
    batch = asyncio.create_task(
        pipeline.run_batch(
            sessions,
            on_result=lambda result: callback_ids.append(result.session_id),
        )
    )
    await _wait_for_pending(pipeline, 2)

    batch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await batch

    assert callback_ids == []
    assert list(tmp_path.rglob("ses_*.json")) == []
    assert not any(session.early_stop_requested for session in sessions)
    assert pipeline._client.delete.await_count == 2
    assert pipeline.status()["pending_sessions"] == 0


def test_early_stop_threshold_cannot_exceed_session_count() -> None:
    with pytest.raises(ValidationError, match="cannot exceed num_samples"):
        TaskRequest(
            task_id="invalid-threshold",
            instruction="test",
            num_samples=2,
            early_stop_min_usable_sessions=3,
            agent=AgentSpec(harness="codex"),
        )
