from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from polar.agent.models import AgentSpec
from polar.gateway.dispatcher import ManagedSession, SessionDispatcher
from polar.rollout.models import SessionDispatchRequest
from polar.rollout.timer import StageTimer


def _managed(
    session_id: str,
    *,
    priority: int,
    tmp_path: Path,
) -> ManagedSession:
    session_dir = tmp_path / session_id
    artifacts_dir = session_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    return ManagedSession(
        request=SessionDispatchRequest(
            session_id=session_id,
            task_id=f"task-{session_id}",
            instruction="fix the bug",
            dispatch_priority=priority,
            remaining_timeout_seconds=60.0,
            agent=AgentSpec(harness="codex"),
        ),
        timer=StageTimer(),
        session_dir=session_dir,
        artifacts_dir=artifacts_dir,
    )


@pytest.mark.asyncio
async def test_high_priority_init_overtakes_queued_work_without_preemption(
    tmp_path: Path,
) -> None:
    dispatcher = SessionDispatcher(
        max_init_workers=1,
        max_run_workers=3,
        max_postrun_workers=1,
    )
    init_order: list[str] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def on_init(managed: ManagedSession) -> None:
        init_order.append(managed.session_id)
        if managed.session_id == "train-inflight":
            first_started.set()
            await release_first.wait()

    async def noop(_managed: ManagedSession) -> None:
        return None

    dispatcher.on_init = on_init
    dispatcher.on_run = noop
    dispatcher.on_postrun = noop
    await dispatcher.start()

    inflight = _managed("train-inflight", priority=0, tmp_path=tmp_path)
    queued = _managed("train-queued", priority=0, tmp_path=tmp_path)
    evaluation = _managed("eval-priority-1", priority=100, tmp_path=tmp_path)
    evaluation_second = _managed("eval-priority-2", priority=100, tmp_path=tmp_path)
    await dispatcher.enqueue(inflight)
    await asyncio.wait_for(first_started.wait(), timeout=1.0)
    await dispatcher.enqueue(queued)
    await dispatcher.enqueue(evaluation)
    await dispatcher.enqueue(evaluation_second)

    # Priority is non-preemptive: the already-running INIT completes first,
    # then eval overtakes only the lower-priority session still in the queue.
    release_first.set()
    await asyncio.wait_for(
        asyncio.gather(
            inflight.done_event.wait(),
            queued.done_event.wait(),
            evaluation.done_event.wait(),
            evaluation_second.done_event.wait(),
        ),
        timeout=1.0,
    )
    assert init_order == [
        "train-inflight",
        "eval-priority-1",
        "eval-priority-2",
        "train-queued",
    ]
    await dispatcher.stop()
