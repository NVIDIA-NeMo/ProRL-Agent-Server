from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from slime_bridge import rollout as rollout_module
from slime_bridge.rollout import AsyncPolarRolloutWorker


def _worker() -> AsyncPolarRolloutWorker:
    args = SimpleNamespace(
        polar_rollout_url="http://rollout:8080",
        polar_task_template={"agent": {"harness": "codex"}},
        polar_max_async_level=2,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        update_weights_interval=1,
        polar_callback_host="127.0.0.1",
    )
    return AsyncPolarRolloutWorker(args, data_source=SimpleNamespace())


@pytest.mark.asyncio
async def test_submit_cancellation_deletes_remote_task_with_tombstone() -> None:
    worker = _worker()
    post_started = asyncio.Event()

    async def post_forever(*_args, **_kwargs) -> None:
        post_started.set()
        await asyncio.Future()

    response = SimpleNamespace(raise_for_status=lambda: None)
    client = SimpleNamespace(
        post=AsyncMock(side_effect=post_forever),
        delete=AsyncMock(return_value=response),
    )
    payload = {"task_id": "task-cancelled"}
    task = asyncio.create_task(worker._submit_with_callback(client, payload))
    await asyncio.wait_for(post_started.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    client.delete.assert_awaited_once_with(
        "http://rollout:8080/rollout/task/task-cancelled",
        params={"register_if_missing": "true"},
        timeout=10.0,
    )
    assert "task-cancelled" not in worker._task_events
    assert "task-cancelled" not in worker._task_results


def _terminal_status_response(task_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "task_id": task_id,
            "status": "completed",
            "total_sessions": 0,
            "completed_sessions": 0,
            "results": [],
            "result_paths": [],
        },
    )


@pytest.mark.parametrize(
    "error_type",
    [httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError],
)
@pytest.mark.asyncio
async def test_callback_fallback_retries_transient_status_get(monkeypatch, error_type) -> None:
    worker = _worker()
    task_id = "task-transient-get"
    request = httpx.Request("GET", f"http://rollout:8080/rollout/task/{task_id}")
    submit_response = SimpleNamespace(raise_for_status=lambda: None)
    client = SimpleNamespace(
        post=AsyncMock(return_value=submit_response),
        get=AsyncMock(
            side_effect=[
                error_type("transient status connection failure", request=request),
                _terminal_status_response(task_id),
            ]
        ),
    )
    monkeypatch.setattr(rollout_module, "_CALLBACK_FALLBACK_POLL_SECONDS", 0.0)
    monkeypatch.setattr(rollout_module, "_TASK_STATUS_GET_RETRY_BACKOFF_SECONDS", 0.0)

    result = await worker._submit_with_callback(client, {"task_id": task_id})

    assert result.task_id == task_id
    assert result.status == "completed"
    assert client.post.await_count == 1
    assert client.get.await_count == 2


@pytest.mark.asyncio
async def test_callback_fallback_status_get_retry_is_bounded(monkeypatch) -> None:
    worker = _worker()
    task_id = "task-exhausted-get"
    request = httpx.Request("GET", f"http://rollout:8080/rollout/task/{task_id}")
    submit_response = SimpleNamespace(raise_for_status=lambda: None)
    client = SimpleNamespace(
        post=AsyncMock(return_value=submit_response),
        get=AsyncMock(
            side_effect=[
                httpx.RemoteProtocolError("persistent failure", request=request)
                for _ in range(rollout_module._TASK_STATUS_GET_MAX_ATTEMPTS)
            ]
        ),
    )
    monkeypatch.setattr(rollout_module, "_CALLBACK_FALLBACK_POLL_SECONDS", 0.0)
    monkeypatch.setattr(rollout_module, "_TASK_STATUS_GET_RETRY_BACKOFF_SECONDS", 0.0)

    with pytest.raises(httpx.RemoteProtocolError, match="persistent failure"):
        await worker._submit_with_callback(client, {"task_id": task_id})

    assert client.post.await_count == 1
    assert client.get.await_count == rollout_module._TASK_STATUS_GET_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_callback_fallback_does_not_retry_nonretryable_http_status(monkeypatch) -> None:
    worker = _worker()
    task_id = "task-not-found"
    request = httpx.Request("GET", f"http://rollout:8080/rollout/task/{task_id}")
    not_found = httpx.Response(404, request=request)
    submit_response = SimpleNamespace(raise_for_status=lambda: None)
    client = SimpleNamespace(
        post=AsyncMock(return_value=submit_response),
        get=AsyncMock(
            side_effect=httpx.HTTPStatusError("not found", request=request, response=not_found)
        ),
    )
    monkeypatch.setattr(rollout_module, "_CALLBACK_FALLBACK_POLL_SECONDS", 0.0)

    with pytest.raises(httpx.HTTPStatusError, match="not found"):
        await worker._submit_with_callback(client, {"task_id": task_id})

    assert client.post.await_count == 1
    assert client.get.await_count == 1
