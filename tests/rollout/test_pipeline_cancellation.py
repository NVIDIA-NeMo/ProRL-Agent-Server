from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from polar.agent.models import AgentSpec
from polar.rollout.balancer import NodeScheduler
from polar.rollout.models import SessionContext, SessionDispatchRequest, TaskRequest
from polar.rollout.pipeline import Pipeline


def _session() -> SessionContext:
    request = TaskRequest(
        task_id="task-1",
        instruction="test cancellation",
        agent=AgentSpec(harness="codex"),
    )
    return SessionContext(
        session_id="session-1",
        task_id=request.task_id,
        request=request,
        deadline_monotonic=time.monotonic() + 60.0,
    )


@pytest.mark.asyncio
async def test_cancelling_pipeline_session_deletes_gateway_session_and_clears_pending(
    monkeypatch,
) -> None:
    pipeline = Pipeline(
        callback_url="http://rollout/callback",
        save_dir=None,
        scheduler=NodeScheduler(),
    )
    response = SimpleNamespace(status_code=200, raise_for_status=lambda: None)
    client = SimpleNamespace(delete=AsyncMock(return_value=response))
    pipeline._client = client
    pipeline._started = True
    waiting_for_result = asyncio.Event()

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

    async def wait_forever(*_args) -> None:
        waiting_for_result.set()
        await asyncio.Future()

    monkeypatch.setattr(pipeline, "_dispatch_session", dispatch)
    monkeypatch.setattr(pipeline, "_wait_for_result", wait_forever)
    session = _session()
    task = asyncio.create_task(pipeline._dispatch_and_collect(session, None))
    await asyncio.wait_for(waiting_for_result.wait(), timeout=1.0)
    assert list(pipeline._pending) == [session.session_id]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    client.delete.assert_awaited_once_with(
        "http://gateway/sessions/session-1",
        timeout=10.0,
    )
    assert pipeline._pending == {}


@pytest.mark.asyncio
async def test_multi_gateway_cleanup_returns_each_session_to_its_assigned_node() -> None:
    pipeline = Pipeline(
        callback_url="http://rollout/callback",
        save_dir=None,
        scheduler=NodeScheduler(),
    )
    response = SimpleNamespace(status_code=200, raise_for_status=lambda: None)
    client = SimpleNamespace(delete=AsyncMock(return_value=response))
    pipeline._client = client
    pipeline._started = True
    first = _session()
    first.gateway_url = "http://gateway-a"
    second = _session()
    second.session_id = "session-2"
    second.gateway_url = "http://gateway-b"

    await asyncio.gather(
        pipeline._cleanup_session(first),
        pipeline._cleanup_session(second),
    )

    assert {call.args[0] for call in client.delete.await_args_list} == {
        "http://gateway-a/sessions/session-1",
        "http://gateway-b/sessions/session-2",
    }
