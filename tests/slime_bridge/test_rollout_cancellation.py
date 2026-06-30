from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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
