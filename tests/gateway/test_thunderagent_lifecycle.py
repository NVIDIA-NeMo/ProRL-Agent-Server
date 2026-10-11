from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from polar.agent.models import AgentSpec
from polar.gateway.dispatcher import ManagedSession
from polar.gateway.node import GatewayNodeManager
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.rollout.models import SessionDispatchRequest, SessionResult, SessionStatus
from polar.rollout.timer import StageTimer
from polar.trajectory.models import Trajectory
from polar.trajectory.registry import StrategyRegistry


def _manager(
    program_releaser: Callable[[str], Awaitable[bool]],
) -> tuple[GatewayNodeManager, SessionRegistry, SessionStore]:
    registry = SessionRegistry()
    storage = SessionStore()
    manager = GatewayNodeManager(
        node_id="node-1",
        gateway_url="http://gateway.test",
        max_init_workers=1,
        max_run_workers=1,
        max_postrun_workers=1,
        storage=storage,
        session_registry=registry,
        builders=StrategyRegistry(object),
        evaluators=StrategyRegistry(object),
        program_releaser=program_releaser,
    )
    return manager, registry, storage


def _managed_session(
    tmp_path: Path,
    *,
    session_id: str,
    status: SessionStatus | None,
) -> ManagedSession:
    request = SessionDispatchRequest(
        session_id=session_id,
        task_id="task-1",
        instruction="test",
        remaining_timeout_seconds=10,
        agent=AgentSpec(harness="codex"),
    )
    session_dir = tmp_path / session_id
    artifacts_dir = session_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    result = None
    if status is not None:
        error = None if status == SessionStatus.COMPLETED else "terminal error"
        result = SessionResult(
            session_id=session_id,
            task_id=request.task_id,
            status=status,
            trajectory=Trajectory(status=status.value, error=error),
            error=error,
        )
    return ManagedSession(
        request=request,
        timer=StageTimer(),
        session_dir=session_dir,
        artifacts_dir=artifacts_dir,
        final_result=result,
        cancel_requested=status is None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_status"),
    [
        (SessionStatus.COMPLETED, SessionStatus.COMPLETED),
        (SessionStatus.ERROR, SessionStatus.ERROR),
        (SessionStatus.TIMEOUT, SessionStatus.TIMEOUT),
        (None, SessionStatus.ERROR),
    ],
    ids=["completed", "error", "timeout", "cancelled"],
)
async def test_postrun_releases_program_for_every_terminal_state(
    tmp_path: Path,
    status: SessionStatus | None,
    expected_status: SessionStatus,
) -> None:
    releaser = AsyncMock(return_value=True)
    manager, registry, storage = _manager(releaser)
    managed = _managed_session(tmp_path, session_id=f"session-{expected_status}", status=status)
    registry.register(managed.session_id, task_id=managed.request.task_id)
    storage.ensure_session(managed.session_id, None, None, None)

    try:
        await manager._handle_postrun(managed)

        info = registry.get(managed.session_id)
        assert info is not None
        assert info.status == expected_status
        releaser.assert_awaited_once_with(managed.session_id)
        assert storage.get_session_metadata(managed.session_id) is None
        assert not managed.session_dir.exists()
    finally:
        await manager.close()
        storage.close()


@pytest.mark.asyncio
async def test_releaser_failure_preserves_terminal_result_and_cleanup(tmp_path: Path) -> None:
    releaser = AsyncMock(side_effect=RuntimeError("release failed"))
    manager, registry, storage = _manager(releaser)
    managed = _managed_session(
        tmp_path,
        session_id="session-release-failure",
        status=SessionStatus.TIMEOUT,
    )
    registry.register(managed.session_id, task_id=managed.request.task_id)
    storage.ensure_session(managed.session_id, None, None, None)

    try:
        await manager._handle_postrun(managed)

        info = registry.get(managed.session_id)
        assert info is not None
        assert info.result is not None
        assert info.result.status == SessionStatus.TIMEOUT
        releaser.assert_awaited_once_with(managed.session_id)
        assert storage.get_session_metadata(managed.session_id) is None
        assert not managed.session_dir.exists()
    finally:
        await manager.close()
        storage.close()
