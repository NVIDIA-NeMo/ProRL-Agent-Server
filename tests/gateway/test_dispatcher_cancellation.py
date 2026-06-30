from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from polar.agent.models import AgentSpec
from polar.gateway.dispatcher import ManagedSession, SessionDispatcher, SessionStage
from polar.gateway.node import GatewayNodeManager
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.rollout.models import SessionDispatchRequest, SessionStatus
from polar.rollout.timer import StageTimer


def _managed(index: int, runtime, tmp_path: Path) -> ManagedSession:
    session_id = f"session-{index}"
    request = SessionDispatchRequest(
        session_id=session_id,
        task_id="task-cancel",
        instruction="cancel me",
        remaining_timeout_seconds=60.0,
        callback_url="http://rollout/callback",
        agent=AgentSpec(harness="codex"),
    )
    session_dir = tmp_path / session_id
    artifacts_dir = session_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    return ManagedSession(
        request=request,
        timer=StageTimer(),
        session_dir=session_dir,
        artifacts_dir=artifacts_dir,
        runtime=runtime,
        stage=SessionStage.RUNNING,
        inflight=True,
    )


@pytest.mark.asyncio
async def test_mass_cancel_ack_is_fast_and_shutdown_drains_runtime_kills(tmp_path) -> None:
    count = 64
    cancel_started = 0
    all_started = asyncio.Event()
    release = asyncio.Event()

    async def cancel_runtime() -> None:
        nonlocal cancel_started
        cancel_started += 1
        if cancel_started == count:
            all_started.set()
        await release.wait()

    dispatcher = SessionDispatcher(
        max_init_workers=1,
        max_run_workers=1,
        max_postrun_workers=1,
    )
    await dispatcher.start()
    sessions = [
        _managed(index, SimpleNamespace(cancel=cancel_runtime), tmp_path)
        for index in range(count)
    ]
    async with dispatcher._lock:
        dispatcher._sessions.update(
            {managed.session_id: managed for managed in sessions}
        )

    done_events = await asyncio.wait_for(
        asyncio.gather(*(dispatcher.cancel(managed.session_id) for managed in sessions)),
        timeout=0.25,
    )
    assert all(done_event is managed.done_event for done_event, managed in zip(done_events, sessions))
    await asyncio.wait_for(all_started.wait(), timeout=1.0)
    assert len(dispatcher._runtime_cancel_tasks) == count

    stop = asyncio.create_task(dispatcher.stop())
    await asyncio.sleep(0)
    assert not stop.done()
    release.set()
    await asyncio.wait_for(stop, timeout=1.0)

    assert dispatcher._runtime_cancel_tasks == set()
    assert all(managed.done_event.is_set() for managed in sessions)


@pytest.mark.asyncio
async def test_repeated_cancel_reuses_kill_and_postrun_waits_for_reap(tmp_path) -> None:
    cancel_started = asyncio.Event()
    release = asyncio.Event()
    cancel_calls = 0
    postrun_called = asyncio.Event()

    async def cancel_runtime() -> None:
        nonlocal cancel_calls
        cancel_calls += 1
        cancel_started.set()
        await release.wait()

    async def postrun(_managed: ManagedSession) -> None:
        assert release.is_set()
        postrun_called.set()

    dispatcher = SessionDispatcher(
        max_init_workers=1,
        max_run_workers=1,
        max_postrun_workers=1,
    )
    dispatcher.on_postrun = postrun
    await dispatcher.start()
    managed = _managed(0, SimpleNamespace(cancel=cancel_runtime), tmp_path)
    managed.stage = SessionStage.READY
    managed.inflight = False
    # A real READY session owns one run slot until it moves to postrun.
    await dispatcher._ready_slots.acquire()
    async with dispatcher._lock:
        dispatcher._sessions[managed.session_id] = managed

    first = await dispatcher.cancel(managed.session_id)
    second = await dispatcher.cancel(managed.session_id)
    assert first is managed.done_event
    assert second is first
    await asyncio.wait_for(cancel_started.wait(), timeout=1.0)
    await asyncio.sleep(0)
    assert not postrun_called.is_set()

    release.set()
    await asyncio.wait_for(managed.done_event.wait(), timeout=1.0)
    assert postrun_called.is_set()
    assert cancel_calls == 1
    await dispatcher.stop()


@pytest.mark.asyncio
async def test_node_cancel_removes_registry_only_after_dispatcher_finishes() -> None:
    session_id = "session-finalize"
    done_event = asyncio.Event()
    storage = SessionStore()
    registry = SessionRegistry()
    info = registry.register(
        session_id,
        task_id="task-cancel",
        registered=True,
        status=SessionStatus.RUNNING,
    )
    storage.ensure_session(
        session_id,
        model_requested=None,
        model_used=None,
        api_type=None,
        task_id=info.task_id,
    )
    manager = object.__new__(GatewayNodeManager)
    manager.storage = storage
    manager.session_registry = registry
    manager._cancel_lock = asyncio.Lock()
    manager._cancel_finalizers = {}
    manager._dispatcher = SimpleNamespace(cancel=AsyncMock(return_value=done_event))

    assert await manager.cancel(session_id)
    assert await manager.cancel(session_id)
    manager._dispatcher.cancel.assert_awaited_once_with(session_id)
    assert registry.get(session_id) is not None
    assert storage.get_session_metadata(session_id) is not None

    done_event.set()
    async with asyncio.timeout(1.0):
        while registry.get(session_id) is not None:
            await asyncio.sleep(0)
    assert storage.get_session_metadata(session_id) is None
