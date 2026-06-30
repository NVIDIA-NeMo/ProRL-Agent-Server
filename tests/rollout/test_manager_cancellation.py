from __future__ import annotations

import asyncio

import pytest

from polar.agent.models import AgentSpec
from polar.rollout.balancer import NodeScheduler
from polar.rollout.manager import RolloutManager
from polar.rollout.models import TaskRequest


class _BlockingPipeline:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.run_calls = 0

    async def run_batch(self, _sessions, *, on_result=None):
        del on_result
        self.run_calls += 1
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.cancelled.set()

    @staticmethod
    def result_path_for(_task_id: str, _session_id: str) -> None:
        return None

    @staticmethod
    def status() -> dict[str, int]:
        return {"pending_sessions": 0}


class _SilentEventBus:
    @staticmethod
    def publish_threadsafe(_loop, _event_type, _payload) -> None:
        pass


def _request(task_id: str) -> TaskRequest:
    return TaskRequest(
        task_id=task_id,
        instruction="test cancellation",
        agent=AgentSpec(harness="codex"),
    )


@pytest.mark.asyncio
async def test_cancel_running_task_is_idempotent() -> None:
    pipeline = _BlockingPipeline()
    manager = RolloutManager(
        pipeline=pipeline,
        scheduler=NodeScheduler(),
        event_bus=_SilentEventBus(),
    )
    await manager.submit_task(_request("running-task"))
    await asyncio.wait_for(pipeline.started.wait(), timeout=1.0)

    first = await manager.cancel_task("running-task")
    second = await manager.cancel_task("running-task")

    assert first is not None
    assert first.status == "cancelled"
    assert second is not None
    assert second.status == "cancelled"
    assert pipeline.cancelled.is_set()
    assert pipeline.run_calls == 1


@pytest.mark.asyncio
async def test_cancel_tombstone_blocks_late_submit() -> None:
    pipeline = _BlockingPipeline()
    manager = RolloutManager(
        pipeline=pipeline,
        scheduler=NodeScheduler(),
        event_bus=_SilentEventBus(),
    )

    cancelled = await manager.cancel_task("late-task", register_if_missing=True)

    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.total_sessions == 0
    with pytest.raises(ValueError, match="cancelled before submission"):
        await manager.submit_task(_request("late-task"))
    assert pipeline.run_calls == 0
