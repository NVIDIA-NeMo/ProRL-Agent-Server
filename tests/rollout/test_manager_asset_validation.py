from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from polar.agent.models import AgentSpec
from polar.rollout.balancer import NodeScheduler
from polar.rollout.manager import RolloutManager
from polar.rollout.models import NodeRegistrationRequest, TaskRequest
from polar.rollout import server as rollout_server
from polar.runtime.assets import RuntimeAssetUnavailableError
from polar.runtime.models import RuntimeSpec


class _PipelineThatMustNotRun:
    def __init__(self) -> None:
        self.run_calls = 0

    async def run_batch(self, _sessions, *, on_result=None):
        del on_result
        self.run_calls += 1
        raise AssertionError("invalid assets must fail before session fan-out")

    @staticmethod
    def status() -> dict[str, int]:
        return {"pending_sessions": 0}


class _SilentEventBus:
    @staticmethod
    def publish_threadsafe(_loop, _event_type, _payload) -> None:
        pass


@pytest.mark.asyncio
async def test_submit_rejects_missing_assets_before_creating_task(
    tmp_path: Path,
) -> None:
    pipeline = _PipelineThatMustNotRun()
    manager = RolloutManager(
        pipeline=pipeline,
        scheduler=NodeScheduler(),
        event_bus=_SilentEventBus(),
    )
    request = TaskRequest(
        task_id="missing-assets",
        instruction="test",
        num_samples=8,
        runtime=RuntimeSpec(
            backend="apptainer",
            image=str(tmp_path / "missing.sif"),
        ),
        agent=AgentSpec(harness="mini_swe_agent"),
    )

    with pytest.raises(RuntimeAssetUnavailableError, match="before session dispatch"):
        await manager.submit_task(request)

    assert pipeline.run_calls == 0
    assert manager.get_task(request.task_id) is None


@pytest.mark.asyncio
async def test_submit_endpoint_reports_asset_outage_as_service_unavailable(
    monkeypatch,
) -> None:
    class _RejectingManager:
        @staticmethod
        async def submit_task(_request: TaskRequest) -> str:
            raise RuntimeAssetUnavailableError("shared runtime bundle disappeared")

    class _State:
        manager = _RejectingManager()

    monkeypatch.setattr(rollout_server, "get_state", lambda: _State())
    request = TaskRequest(
        task_id="missing-assets",
        instruction="test",
        agent=AgentSpec(harness="mini_swe_agent"),
    )

    with pytest.raises(HTTPException) as error:
        await rollout_server.submit_task_async(SimpleNamespace(headers={}), request)

    assert error.value.status_code == 503
    assert "runtime bundle disappeared" in str(error.value.detail)


@pytest.mark.asyncio
async def test_spilot_submit_requires_control_plane_token(monkeypatch) -> None:
    class _Manager:
        calls = 0

        @classmethod
        async def submit_task(cls, request: TaskRequest) -> str:
            cls.calls += 1
            return request.task_id

    class _State:
        manager = _Manager()

    monkeypatch.setenv("POLAR_CONTROL_PLANE_TOKEN", "trusted-control-token")
    monkeypatch.setattr(rollout_server, "get_state", lambda: _State())
    request = TaskRequest(
        task_id="spilot-task",
        instruction="test",
        agent=AgentSpec(harness="spilot_router"),
    )

    with pytest.raises(HTTPException) as error:
        await rollout_server.submit_task_async(SimpleNamespace(headers={}), request)
    assert error.value.status_code == 401
    assert _Manager.calls == 0

    response = await rollout_server.submit_task_async(
        SimpleNamespace(headers={"x-polar-control-token": "trusted-control-token"}),
        request,
    )
    assert response == {"task_id": "spilot-task", "status": "running"}
    assert _Manager.calls == 1


@pytest.mark.asyncio
async def test_fake_node_registration_requires_control_plane_token(monkeypatch) -> None:
    class _Scheduler:
        calls = 0

        @classmethod
        def register_node(cls, request: NodeRegistrationRequest):
            cls.calls += 1
            return request

    class _State:
        scheduler = _Scheduler()

    monkeypatch.setenv("POLAR_CONTROL_PLANE_TOKEN", "trusted-control-token")
    monkeypatch.setattr(rollout_server, "get_state", lambda: _State())
    registration = NodeRegistrationRequest(
        node_id="attacker-node",
        gateway_url="http://attacker.invalid",
        max_init_workers=1000,
        max_run_workers=1000,
        max_postrun_workers=1000,
    )

    with pytest.raises(HTTPException) as error:
        await rollout_server.register_node(SimpleNamespace(headers={}), registration)
    assert error.value.status_code == 401
    assert _Scheduler.calls == 0
