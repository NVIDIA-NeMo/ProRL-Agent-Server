from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from polar.agent.models import AgentSpec
from polar.rollout.models import SessionContext, TaskRequest
from polar.rollout.pipeline import Pipeline


def _session() -> SessionContext:
    request = TaskRequest(
        task_id="eval-task",
        instruction="evaluate this task",
        dispatch_priority=100,
        timeout_seconds=60.0,
        agent=AgentSpec(harness="codex"),
    )
    return SessionContext(
        session_id="eval-session",
        task_id=request.task_id,
        request=request,
        deadline_monotonic=time.monotonic() + 60.0,
    )


@pytest.mark.asyncio
async def test_task_priority_is_forwarded_to_gateway_dispatch() -> None:
    node = SimpleNamespace(node_id="node-1", gateway_url="http://gateway")
    scheduler = SimpleNamespace(
        acquire_node=Mock(return_value=node),
        release_reservation=Mock(),
        mark_unhealthy=Mock(),
    )
    response = SimpleNamespace(raise_for_status=Mock())
    client = SimpleNamespace(post=AsyncMock(return_value=response))
    pipeline = Pipeline(
        callback_url="http://rollout/callback",
        save_dir=None,
        scheduler=scheduler,
    )
    pipeline._client = client
    session = _session()

    dispatch = await pipeline._dispatch_session(session)

    assert dispatch.dispatch_priority == 100
    assert client.post.await_args.kwargs["json"]["dispatch_priority"] == 100


@pytest.mark.parametrize(
    "error_type",
    [
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.ReadError,
        httpx.WriteError,
        httpx.RemoteProtocolError,
    ],
)
@pytest.mark.asyncio
async def test_ambiguous_dispatch_is_not_reassigned_to_another_gateway(error_type) -> None:
    node_a = SimpleNamespace(node_id="node-a", gateway_url="http://gateway-a")
    node_b = SimpleNamespace(node_id="node-b", gateway_url="http://gateway-b")
    scheduler = SimpleNamespace(
        acquire_node=Mock(side_effect=[node_a, node_b]),
        release_reservation=Mock(),
        mark_unhealthy=Mock(),
    )
    post_request = httpx.Request("POST", "http://gateway-a/sessions")
    get_request = httpx.Request("GET", "http://gateway-a/sessions/eval-session")
    client = SimpleNamespace(
        post=AsyncMock(side_effect=error_type("ACK was lost", request=post_request)),
        get=AsyncMock(side_effect=httpx.ConnectError("gateway unavailable", request=get_request)),
    )
    pipeline = Pipeline(
        callback_url="http://rollout/callback",
        save_dir=None,
        scheduler=scheduler,
        dispatch_poll_interval_seconds=0.001,
    )
    pipeline._client = client
    session = _session()

    with pytest.raises(RuntimeError, match="refusing cross-gateway retry"):
        await pipeline._dispatch_session(session)

    scheduler.acquire_node.assert_called_once()
    scheduler.release_reservation.assert_not_called()
    scheduler.mark_unhealthy.assert_called_once_with("node-a")
    client.post.assert_awaited_once()
    assert client.get.await_count == 5
    assert session.gateway_url == "http://gateway-a"


@pytest.mark.parametrize(
    "error_type",
    [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout],
)
@pytest.mark.asyncio
async def test_definitely_not_connected_can_reassign_to_another_gateway(error_type) -> None:
    node_a = SimpleNamespace(node_id="node-a", gateway_url="http://gateway-a")
    node_b = SimpleNamespace(node_id="node-b", gateway_url="http://gateway-b")
    scheduler = SimpleNamespace(
        acquire_node=Mock(side_effect=[node_a, node_b]),
        release_reservation=Mock(),
        mark_unhealthy=Mock(),
    )
    post_request = httpx.Request("POST", "http://gateway-a/sessions")
    get_request = httpx.Request("GET", "http://gateway-a/sessions/eval-session")
    accepted = SimpleNamespace(raise_for_status=Mock())
    client = SimpleNamespace(
        post=AsyncMock(
            side_effect=[
                error_type("not connected", request=post_request),
                accepted,
            ]
        ),
        get=AsyncMock(side_effect=httpx.ConnectError("not connected", request=get_request)),
    )
    pipeline = Pipeline(
        callback_url="http://rollout/callback",
        save_dir=None,
        scheduler=scheduler,
        dispatch_poll_interval_seconds=0.001,
    )
    pipeline._client = client
    session = _session()

    dispatch = await pipeline._dispatch_session(session)

    assert dispatch.session_id == session.session_id
    assert scheduler.acquire_node.call_count == 2
    scheduler.release_reservation.assert_called_once_with("node-a")
    scheduler.mark_unhealthy.assert_called_once_with("node-a")
    assert client.post.await_count == 2
    assert session.gateway_url == "http://gateway-b"


@pytest.mark.asyncio
async def test_lost_dispatch_ack_stays_on_gateway_when_get_confirms_acceptance() -> None:
    node_a = SimpleNamespace(node_id="node-a", gateway_url="http://gateway-a")
    scheduler = SimpleNamespace(
        acquire_node=Mock(return_value=node_a),
        release_reservation=Mock(),
        mark_unhealthy=Mock(),
    )
    post_request = httpx.Request("POST", "http://gateway-a/sessions")
    confirmed = SimpleNamespace(
        raise_for_status=Mock(),
        json=Mock(return_value={"task_id": "eval-task"}),
    )
    client = SimpleNamespace(
        post=AsyncMock(side_effect=httpx.ReadTimeout("ACK was lost", request=post_request)),
        get=AsyncMock(return_value=confirmed),
    )
    pipeline = Pipeline(
        callback_url="http://rollout/callback",
        save_dir=None,
        scheduler=scheduler,
    )
    pipeline._client = client
    session = _session()

    dispatch = await pipeline._dispatch_session(session)

    assert dispatch.session_id == session.session_id
    scheduler.acquire_node.assert_called_once()
    scheduler.release_reservation.assert_not_called()
    scheduler.mark_unhealthy.assert_not_called()
    assert client.post.await_count == 1
    assert session.gateway_url == "http://gateway-a"


@pytest.mark.asyncio
async def test_lost_dispatch_ack_retries_confirmation_on_same_gateway() -> None:
    node_a = SimpleNamespace(node_id="node-a", gateway_url="http://gateway-a")
    scheduler = SimpleNamespace(
        acquire_node=Mock(return_value=node_a),
        release_reservation=Mock(),
        mark_unhealthy=Mock(),
    )
    post_request = httpx.Request("POST", "http://gateway-a/sessions")
    get_request = httpx.Request("GET", "http://gateway-a/sessions/eval-session")
    confirmed = SimpleNamespace(
        raise_for_status=Mock(),
        json=Mock(return_value={"task_id": "eval-task"}),
    )
    client = SimpleNamespace(
        post=AsyncMock(side_effect=httpx.ReadError("ACK was lost", request=post_request)),
        get=AsyncMock(
            side_effect=[
                httpx.ReadError("gateway busy", request=get_request),
                httpx.ReadError("gateway busy", request=get_request),
                confirmed,
            ]
        ),
    )
    pipeline = Pipeline(
        callback_url="http://rollout/callback",
        save_dir=None,
        scheduler=scheduler,
        dispatch_poll_interval_seconds=0.001,
    )
    pipeline._client = client
    session = _session()

    dispatch = await pipeline._dispatch_session(session)

    assert dispatch.session_id == session.session_id
    scheduler.acquire_node.assert_called_once()
    scheduler.release_reservation.assert_not_called()
    scheduler.mark_unhealthy.assert_not_called()
    client.post.assert_awaited_once()
    assert client.get.await_count == 3
    assert session.gateway_url == "http://gateway-a"
