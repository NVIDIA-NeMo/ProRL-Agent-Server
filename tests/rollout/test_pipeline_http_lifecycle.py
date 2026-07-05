from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from polar.agent.models import AgentSpec
from polar.rollout.balancer import NodeScheduler
from polar.rollout.models import SessionContext, TaskRequest
from polar.rollout.pipeline import Pipeline


def _pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        callback_url="http://rollout/callback",
        save_dir=None,
        scheduler=NodeScheduler(),
        **kwargs,
    )


def _session(index: int = 0) -> SessionContext:
    request = TaskRequest(
        task_id="task-http",
        instruction="test HTTP cleanup",
        agent=AgentSpec(harness="codex"),
    )
    return SessionContext(
        session_id=f"session-{index}",
        task_id=request.task_id,
        request=request,
        gateway_url="http://gateway",
        deadline_monotonic=time.monotonic() + 60.0,
    )


@pytest.mark.asyncio
async def test_start_sizes_http_pool_for_fully_async_sessions(monkeypatch) -> None:
    client = SimpleNamespace(aclose=AsyncMock())
    cleanup_client = SimpleNamespace(aclose=AsyncMock())
    constructor = Mock(side_effect=[client, cleanup_client])
    monkeypatch.setattr(httpx, "AsyncClient", constructor)
    monkeypatch.setenv("POLAR_CONTROL_PLANE_TOKEN", "trusted-control-token")
    pipeline = _pipeline(
        http_max_connections=640,
        http_max_keepalive_connections=192,
        cleanup_max_concurrency=64,
    )

    await pipeline.start()
    primary_limits = constructor.call_args_list[0].kwargs["limits"]
    assert primary_limits.max_connections == 640
    assert primary_limits.max_keepalive_connections == 192
    cleanup_limits = constructor.call_args_list[1].kwargs["limits"]
    assert cleanup_limits.max_connections == 64
    assert cleanup_limits.max_keepalive_connections == 64
    expected_headers = {"X-Polar-Control-Token": "trusted-control-token"}
    assert constructor.call_args_list[0].kwargs["headers"] == expected_headers
    assert constructor.call_args_list[1].kwargs["headers"] == expected_headers

    await pipeline.close()
    client.aclose.assert_awaited_once()
    cleanup_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_uses_dedicated_client_when_primary_pool_is_saturated() -> None:
    response = SimpleNamespace(status_code=200)
    primary_client = SimpleNamespace(
        delete=AsyncMock(side_effect=httpx.PoolTimeout("primary pool busy"))
    )
    cleanup_client = SimpleNamespace(delete=AsyncMock(return_value=response))
    pipeline = _pipeline()
    pipeline._client = primary_client
    pipeline._cleanup_client = cleanup_client
    pipeline._started = True

    await pipeline._cleanup_session(_session())

    primary_client.delete.assert_not_awaited()
    cleanup_client.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_retries_pool_timeout_then_accepts_not_found() -> None:
    request = httpx.Request("DELETE", "http://gateway/sessions/session-0")
    not_found = httpx.Response(404, request=request)
    client = SimpleNamespace(
        delete=AsyncMock(side_effect=[httpx.PoolTimeout("pool busy"), not_found])
    )
    pipeline = _pipeline(
        cleanup_max_attempts=3,
        cleanup_retry_backoff_seconds=0.0,
    )
    pipeline._client = client
    pipeline._started = True

    await pipeline._cleanup_session(_session())

    assert client.delete.await_count == 2


@pytest.mark.asyncio
async def test_cleanup_does_not_retry_non_transient_http_error() -> None:
    request = httpx.Request("DELETE", "http://gateway/sessions/session-0")
    bad_request = httpx.Response(400, request=request)
    client = SimpleNamespace(delete=AsyncMock(return_value=bad_request))
    pipeline = _pipeline(
        cleanup_max_attempts=3,
        cleanup_retry_backoff_seconds=0.0,
    )
    pipeline._client = client
    pipeline._started = True

    await pipeline._cleanup_session(_session())

    client.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_fanout_is_limited() -> None:
    active = 0
    peak = 0
    two_active = asyncio.Event()
    release = asyncio.Event()

    async def delete(_url: str, *, timeout: float):
        nonlocal active, peak
        assert timeout == 10.0
        active += 1
        peak = max(peak, active)
        if active == 2:
            two_active.set()
        try:
            await release.wait()
        finally:
            active -= 1
        return SimpleNamespace(status_code=200)

    pipeline = _pipeline(cleanup_max_concurrency=2)
    pipeline._client = SimpleNamespace(delete=delete)
    pipeline._started = True
    cleanups = [
        asyncio.create_task(pipeline._cleanup_session(_session(index))) for index in range(6)
    ]

    await asyncio.wait_for(two_active.wait(), timeout=1.0)
    await asyncio.sleep(0)
    assert peak == 2

    release.set()
    await asyncio.gather(*cleanups)
    assert peak == 2


@pytest.mark.asyncio
async def test_close_drains_shielded_cleanup_before_client_close() -> None:
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    closed = False

    async def delete(_url: str, *, timeout: float):
        assert timeout == 10.0
        cleanup_started.set()
        await cleanup_release.wait()
        assert not closed
        return SimpleNamespace(status_code=200)

    async def close_client() -> None:
        nonlocal closed
        closed = True

    pipeline = _pipeline()
    pipeline._client = SimpleNamespace(delete=delete, aclose=close_client)
    pipeline._started = True
    finalize = asyncio.create_task(pipeline._finalize_session_cleanup(_session()))
    await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)

    finalize.cancel()
    with pytest.raises(asyncio.CancelledError):
        await finalize
    close = asyncio.create_task(pipeline.close())
    await asyncio.sleep(0)
    assert not close.done()
    assert not closed

    cleanup_release.set()
    await close
    assert closed
