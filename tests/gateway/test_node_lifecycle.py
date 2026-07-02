from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from polar.agent.models import AgentRunResult, AgentSpec
from polar.gateway.dispatcher import ManagedSession, SessionStage
from polar.gateway.node import GatewayExecutionTimeout, GatewayNodeManager
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.rollout.models import SessionDispatchRequest, SessionStatus
from polar.rollout.timer import StageTimer
from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult, PrepareAction, RuntimeSpec
from polar.trajectory.models import EvalResult, EvaluatorSpec, Trace, Trajectory


class _PostrunRuntime(BaseRuntime):
    @property
    def runtime_id(self) -> str:
        return "postrun-runtime"

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self._destroyed = True

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        return ExecResult(return_code=0)

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        pass

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        pass

    async def download_file(self, remote_path: str, local_path: str) -> None:
        pass

    async def download_dir(self, remote_path: str, local_path: str) -> None:
        pass


class _BlockingEvaluator:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def evaluate(self, trajectory: Trajectory, **_runtime) -> EvalResult:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def test_exec_log_write_recreates_directory_after_concurrent_cleanup(tmp_path) -> None:
    session_dir = tmp_path / "session"
    log_dir = session_dir / "logs" / "agent"
    log_dir.mkdir(parents=True)
    shutil.rmtree(session_dir)

    GatewayNodeManager._write_exec_log(log_dir, "step.00", "stdout", "stderr")

    assert (log_dir / "step.00.stdout.log").read_text() == "stdout"
    assert (log_dir / "step.00.stderr.log").read_text() == "stderr"


def test_runtime_prepare_retries_configured_transient_exec_failure(tmp_path) -> None:
    manager = object.__new__(GatewayNodeManager)
    manager._runtime_env = lambda *_args, **_kwargs: {}  # type: ignore[method-assign]
    manager._remaining_budget = lambda _managed: 60.0  # type: ignore[method-assign]

    results = [
        ExecResult(return_code=255, stderr="transient apptainer failure"),
        ExecResult(return_code=0),
    ]

    class Runtime:
        runtime_session_dir = "/polar/session"

        async def exec(self, *_args, **_kwargs) -> ExecResult:
            return results.pop(0)

    spec = RuntimeSpec(
        image="task.sif",
        prepare=[
            PrepareAction(
                type="exec",
                command="git config --global core.pager ''",
                max_attempts=3,
            )
        ],
    )
    managed = SimpleNamespace(cancel_requested=False, session_dir=tmp_path)

    asyncio.run(
        manager._run_runtime_prepare(
            Runtime(),  # type: ignore[arg-type]
            spec,
            SimpleNamespace(),  # type: ignore[arg-type]
            managed,  # type: ignore[arg-type]
        )
    )

    assert not results
    assert (tmp_path / "logs" / "prepare.00.attempt-01.stderr.log").exists()


@pytest.mark.asyncio
async def test_timeout_before_postprocess_does_not_create_coroutine(tmp_path: Path) -> None:
    manager = object.__new__(GatewayNodeManager)
    manager._start_eval_prewarm = lambda _managed: None  # type: ignore[method-assign]
    manager._runtime_env = lambda *_args, **_kwargs: {}  # type: ignore[method-assign]

    async def setup(_runtime: BaseRuntime) -> None:
        pass

    postprocess = Mock()
    harness = SimpleNamespace(
        setup=setup,
        run_steps=Mock(return_value=[]),
        postprocess=postprocess,
        postrun_steps=Mock(return_value=[]),
    )
    manager._resolve_agent_harness = Mock(return_value=harness)  # type: ignore[method-assign]

    request = SessionDispatchRequest(
        session_id="session-timeout-before-postprocess",
        task_id="task-timeout-before-postprocess",
        instruction="test",
        remaining_timeout_seconds=60,
        runtime=RuntimeSpec(image="task.sif"),
        agent=AgentSpec(harness="codex"),
    )
    runtime = _PostrunRuntime(request.runtime, request.session_id, tmp_path)
    managed = ManagedSession(
        request=request,
        timer=StageTimer(),
        session_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        runtime=runtime,
        execution_deadline=asyncio.get_running_loop().time() + 60,
        stage=SessionStage.RUNNING,
    )

    async def finish_agent_then_expire_budget(*_args) -> AgentRunResult:
        managed.execution_deadline = asyncio.get_running_loop().time() - 1
        return AgentRunResult(status="completed", return_code=0)

    manager._run_exec_inputs = finish_agent_then_expire_budget  # type: ignore[method-assign]

    await manager._handle_run(managed)

    postprocess.assert_not_called()
    assert managed.agent_result is not None
    assert managed.agent_result.status == "timeout"
    assert managed.agent_result.metadata["timeout_source"] == "session"
    assert managed.agent_result.metadata["timeout_stage"] == "exec"
    assert "agent_postprocess_started" not in managed.timer._marks
    assert "agent_postprocess_finished" not in managed.timer._marks


@pytest.mark.asyncio
async def test_agent_budget_exhausted_before_postprocess_remains_exec_timeout(
    tmp_path: Path,
) -> None:
    manager = object.__new__(GatewayNodeManager)
    manager._start_eval_prewarm = lambda _managed: None  # type: ignore[method-assign]
    manager._runtime_env = lambda *_args, **_kwargs: {}  # type: ignore[method-assign]

    async def setup(_runtime: BaseRuntime) -> None:
        pass

    postprocess = Mock()
    harness = SimpleNamespace(
        setup=setup,
        run_steps=Mock(return_value=[]),
        postprocess=postprocess,
        postrun_steps=Mock(return_value=[]),
    )
    manager._resolve_agent_harness = Mock(return_value=harness)  # type: ignore[method-assign]

    request = SessionDispatchRequest(
        session_id="session-agent-timeout-before-postprocess",
        task_id="task-agent-timeout-before-postprocess",
        instruction="test",
        remaining_timeout_seconds=60,
        runtime=RuntimeSpec(image="task.sif"),
        agent=AgentSpec(harness="codex"),
        metadata={"agent_timeout": 60.0},
    )
    runtime = _PostrunRuntime(request.runtime, request.session_id, tmp_path)
    managed = ManagedSession(
        request=request,
        timer=StageTimer(),
        session_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        runtime=runtime,
        execution_deadline=asyncio.get_running_loop().time() + 60,
        stage=SessionStage.RUNNING,
    )

    async def finish_agent_then_expire_budget(*_args) -> AgentRunResult:
        managed.agent_deadline = asyncio.get_running_loop().time() - 1
        return AgentRunResult(status="completed", return_code=0)

    manager._run_exec_inputs = finish_agent_then_expire_budget  # type: ignore[method-assign]

    await manager._handle_run(managed)

    postprocess.assert_not_called()
    assert managed.agent_result is not None
    assert managed.agent_result.status == "timeout"
    assert managed.agent_result.error == "agent execution timeout"
    assert managed.agent_result.metadata["timeout_source"] == "agent"
    assert managed.agent_result.metadata["timeout_stage"] == "exec"
    assert "agent_postprocess_started" not in managed.timer._marks
    assert "agent_postprocess_finished" not in managed.timer._marks


@pytest.mark.asyncio
async def test_timeout_after_postprocess_starts_is_classified_as_postprocess(
    tmp_path: Path,
) -> None:
    manager = object.__new__(GatewayNodeManager)
    manager._start_eval_prewarm = lambda _managed: None  # type: ignore[method-assign]
    manager._runtime_env = lambda *_args, **_kwargs: {}  # type: ignore[method-assign]

    async def setup(_runtime: BaseRuntime) -> None:
        pass

    postprocess_started = asyncio.Event()

    async def slow_postprocess(*_args) -> None:
        postprocess_started.set()
        await asyncio.sleep(1)

    harness = SimpleNamespace(
        setup=setup,
        run_steps=Mock(return_value=[]),
        postprocess=slow_postprocess,
        postrun_steps=Mock(return_value=[]),
    )
    manager._resolve_agent_harness = Mock(return_value=harness)  # type: ignore[method-assign]

    request = SessionDispatchRequest(
        session_id="session-timeout-in-postprocess",
        task_id="task-timeout-in-postprocess",
        instruction="test",
        remaining_timeout_seconds=60,
        runtime=RuntimeSpec(image="task.sif"),
        agent=AgentSpec(harness="codex"),
        metadata={"agent_timeout": 0.1},
    )
    managed = ManagedSession(
        request=request,
        timer=StageTimer(),
        session_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        runtime=_PostrunRuntime(request.runtime, request.session_id, tmp_path),
        execution_deadline=asyncio.get_running_loop().time() + 60,
        stage=SessionStage.RUNNING,
    )

    await asyncio.wait_for(manager._handle_run(managed), timeout=1.0)

    assert postprocess_started.is_set()
    assert managed.agent_result is not None
    assert managed.agent_result.status == "timeout"
    assert managed.agent_result.error == "agent execution timeout"
    assert managed.agent_result.metadata["timeout_source"] == "agent"
    assert managed.agent_result.metadata["timeout_stage"] == "postprocess"
    assert "agent_postprocess_started" in managed.timer._marks
    assert "agent_postprocess_finished" in managed.timer._marks


@pytest.mark.asyncio
async def test_agent_timeout_caps_active_run_stage(tmp_path: Path) -> None:
    manager = object.__new__(GatewayNodeManager)
    manager._start_eval_prewarm = lambda _managed: None  # type: ignore[method-assign]

    async def slow_setup(_runtime: BaseRuntime) -> None:
        await asyncio.sleep(1)

    harness = SimpleNamespace(
        setup=slow_setup,
        run_steps=Mock(return_value=[]),
        postprocess=AsyncMock(),
        postrun_steps=Mock(return_value=[]),
    )
    manager._resolve_agent_harness = Mock(return_value=harness)  # type: ignore[method-assign]
    request = SessionDispatchRequest(
        session_id="session-agent-cap",
        task_id="task-agent-cap",
        instruction="test",
        remaining_timeout_seconds=60,
        runtime=RuntimeSpec(image="task.sif"),
        agent=AgentSpec(harness="codex"),
        metadata={"agent_timeout": 0.01},
    )
    runtime = _PostrunRuntime(request.runtime, request.session_id, tmp_path)
    managed = ManagedSession(
        request=request,
        timer=StageTimer(),
        session_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        runtime=runtime,
        execution_deadline=asyncio.get_running_loop().time() + 60,
        stage=SessionStage.RUNNING,
    )

    await asyncio.wait_for(manager._handle_run(managed), timeout=0.5)

    assert managed.agent_result is not None
    assert managed.agent_result.status == "timeout"
    assert managed.agent_result.error == "agent execution timeout"
    assert managed.agent_result.metadata["timeout_source"] == "agent"
    assert managed.agent_result.metadata["timeout_stage"] == "setup"
    harness.run_steps.assert_not_called()
    harness.postprocess.assert_not_awaited()


@pytest.mark.asyncio
async def test_ready_queue_delay_does_not_spend_agent_timeout(tmp_path: Path) -> None:
    request = SessionDispatchRequest(
        session_id="session-agent-queue-delay",
        task_id="task-agent-queue-delay",
        instruction="test",
        remaining_timeout_seconds=60,
        runtime=RuntimeSpec(image="task.sif"),
        agent=AgentSpec(harness="codex"),
        metadata={"agent_timeout": 0.2},
    )
    managed = ManagedSession(
        request=request,
        timer=StageTimer(),
        session_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        execution_deadline=asyncio.get_running_loop().time() + 60,
        stage=SessionStage.READY,
    )

    await asyncio.sleep(0.03)
    assert managed.agent_deadline is None
    GatewayNodeManager._start_agent_deadline(managed)

    assert managed.agent_deadline is not None
    assert managed.agent_deadline - asyncio.get_running_loop().time() > 0.15


@pytest.mark.asyncio
async def test_total_session_timeout_wins_over_larger_agent_cap(tmp_path: Path) -> None:
    manager = object.__new__(GatewayNodeManager)
    manager._start_eval_prewarm = lambda _managed: None  # type: ignore[method-assign]

    async def slow_setup(_runtime: BaseRuntime) -> None:
        await asyncio.sleep(1)

    harness = SimpleNamespace(
        setup=slow_setup,
        run_steps=Mock(return_value=[]),
        postprocess=AsyncMock(),
        postrun_steps=Mock(return_value=[]),
    )
    manager._resolve_agent_harness = Mock(return_value=harness)  # type: ignore[method-assign]
    request = SessionDispatchRequest(
        session_id="session-total-cap",
        task_id="task-total-cap",
        instruction="test",
        remaining_timeout_seconds=60,
        runtime=RuntimeSpec(image="task.sif"),
        agent=AgentSpec(harness="codex"),
        metadata={"agent_timeout": 10.0},
    )
    runtime = _PostrunRuntime(request.runtime, request.session_id, tmp_path)
    managed = ManagedSession(
        request=request,
        timer=StageTimer(),
        session_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        runtime=runtime,
        execution_deadline=asyncio.get_running_loop().time() + 0.01,
        stage=SessionStage.RUNNING,
    )

    await asyncio.wait_for(manager._handle_run(managed), timeout=0.5)

    assert managed.agent_result is not None
    assert managed.agent_result.status == "timeout"
    assert managed.agent_result.error == "session execution timeout"
    assert managed.agent_result.metadata["timeout_source"] == "session"
    assert managed.agent_result.metadata["timeout_stage"] == "setup"


@pytest.mark.asyncio
async def test_timeout_before_trajectory_build_does_not_create_to_thread(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manager = object.__new__(GatewayNodeManager)
    manager.session_registry = SimpleNamespace(set_status=Mock())
    request = SessionDispatchRequest(
        session_id="session-timeout-before-build",
        task_id="task-timeout-before-build",
        instruction="test",
        remaining_timeout_seconds=60,
        runtime=RuntimeSpec(image="task.sif"),
        agent=AgentSpec(harness="codex"),
    )
    managed = ManagedSession(
        request=request,
        timer=StageTimer(),
        session_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        agent_result=AgentRunResult(status="timeout", return_code=-1),
        execution_deadline=asyncio.get_running_loop().time() - 1,
        stage=SessionStage.POSTRUN,
    )
    to_thread = Mock()
    monkeypatch.setattr("polar.gateway.node.asyncio.to_thread", to_thread)

    with pytest.raises(GatewayExecutionTimeout, match="session execution timeout"):
        await manager._build_session_result(managed)

    to_thread.assert_not_called()


@pytest.mark.asyncio
async def test_eval_budget_timeout_masks_preexisting_agent_timeout(tmp_path: Path) -> None:
    manager = object.__new__(GatewayNodeManager)
    manager.node_id = "node-test"
    manager.session_registry = SimpleNamespace(set_status=Mock())
    manager._build_trajectory = Mock(
        return_value=Trajectory(
            status="COMPLETED",
            traces=[
                Trace(
                    prompt_ids=[1],
                    response_ids=[2, 3],
                    loss_mask=[1, 1],
                    response_logprobs=[-0.1, -0.2],
                    reward=1.0,
                )
            ],
        )
    )
    manager._run_eval = AsyncMock(side_effect=GatewayExecutionTimeout("session execution timeout"))
    request = SessionDispatchRequest(
        session_id="session-agent-then-eval-timeout",
        task_id="task-agent-then-eval-timeout",
        instruction="test",
        remaining_timeout_seconds=60,
        runtime=RuntimeSpec(image="task.sif"),
        agent=AgentSpec(harness="codex"),
        evaluator=EvaluatorSpec(strategy="blocking"),
    )
    managed = ManagedSession(
        request=request,
        timer=StageTimer(),
        session_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        runtime=_PostrunRuntime(request.runtime, request.session_id, tmp_path),
        agent_result=AgentRunResult(
            status="timeout",
            return_code=-1,
            error="agent execution timeout",
            metadata={"timeout_source": "agent", "timeout_stage": "exec"},
        ),
        execution_deadline=asyncio.get_running_loop().time() + 60,
        stage=SessionStage.POSTRUN,
    )

    result = await manager._build_session_result(managed)

    assert result.status == "TIMEOUT"
    evaluation = result.trajectory.metadata["evaluation"]
    assert evaluation["verifier_timeout"] is True
    assert evaluation["verifier_timeout_error"] == "session execution timeout"
    assert evaluation["outcome_reward"] == 0.0
    assert evaluation["reward_discard_reason"] == "session_timeout"
    trace = result.trajectory.traces[0]
    assert trace.reward == 0.0
    assert trace.loss_mask == [0, 0]
    assert trace.metadata["training_filter"]["reason"] == "session_timeout"
    assert trace.metadata["training_filter"]["masked"] is True


@pytest.mark.asyncio
async def test_callback_retries_dropped_ack_idempotently(monkeypatch) -> None:
    manager = object.__new__(GatewayNodeManager)
    request = httpx.Request("POST", "http://rollout/callbacks/session_result")
    response = Mock()
    response.raise_for_status.return_value = None
    manager._client = SimpleNamespace(
        post=AsyncMock(
            side_effect=[
                httpx.ReadError("callback acknowledgement dropped", request=request),
                response,
            ]
        )
    )
    monkeypatch.setattr("polar.gateway.node._CALLBACK_RETRY_BACKOFF_SECONDS", 0.0)
    result = SimpleNamespace(
        session_id="session-callback-retry",
        model_dump=Mock(return_value={"session_id": "session-callback-retry"}),
    )

    delivered = await manager._push_result(
        "http://rollout/callbacks/session_result",
        result,
    )

    assert delivered is True
    assert manager._client.post.await_count == 2
    assert manager._client.post.await_args_list[0].kwargs["timeout"] == 5.0
    result.model_dump.assert_called_once_with(mode="json")


@pytest.mark.parametrize(("allow_internet", "expected"), [(True, "true"), (False, "false")])
def test_runtime_env_injects_authoritative_string_internet_policy(
    tmp_path: Path,
    allow_internet: bool,
    expected: str,
) -> None:
    manager = object.__new__(GatewayNodeManager)
    manager.gateway_url = "http://gateway.test"
    runtime = _PostrunRuntime(
        RuntimeSpec(
            image="task.sif",
            allow_internet=allow_internet,
            env={"POLAR_ALLOW_INTERNET": "runtime-override"},
        ),
        "session-policy",
        tmp_path,
    )
    request = SessionDispatchRequest(
        session_id="session-policy",
        task_id="task-policy",
        instruction="test",
        remaining_timeout_seconds=60,
        runtime=runtime.spec,
        agent=AgentSpec(
            harness="codex",
            env={"POLAR_ALLOW_INTERNET": "agent-override"},
        ),
    )
    managed = SimpleNamespace(
        runtime=runtime,
        session_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
    )

    environment = manager._runtime_env(
        request,
        managed,  # type: ignore[arg-type]
        include_agent_env=True,
    )

    assert environment["POLAR_ALLOW_INTERNET"] == expected


@pytest.mark.asyncio
async def test_cancel_during_postrun_eval_returns_masked_cancel_result(
    tmp_path: Path,
) -> None:
    session_id = "session-cancel-during-eval"
    session_dir = tmp_path / session_id
    artifacts_dir = session_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    runtime_spec = RuntimeSpec(image="task.sif")
    runtime = _PostrunRuntime(runtime_spec, session_id, session_dir)
    evaluator = _BlockingEvaluator()
    registry = SessionRegistry()
    storage = SessionStore()
    manager = GatewayNodeManager(
        node_id="node-test",
        gateway_url="http://gateway.test",
        max_init_workers=1,
        max_run_workers=1,
        max_postrun_workers=1,
        storage=storage,
        session_registry=registry,
        builders=SimpleNamespace(),  # type: ignore[arg-type]
        evaluators=SimpleNamespace(create=lambda _spec: evaluator),  # type: ignore[arg-type]
        default_runtime=runtime_spec,
    )
    request = SessionDispatchRequest(
        session_id=session_id,
        task_id="task-cancel-during-eval",
        instruction="test",
        remaining_timeout_seconds=60,
        runtime=runtime_spec,
        agent=AgentSpec(harness="codex"),
        evaluator=EvaluatorSpec(strategy="blocking"),
    )
    timer = StageTimer()
    timer.mark("dispatch", "started")
    managed = ManagedSession(
        request=request,
        timer=timer,
        session_dir=session_dir,
        artifacts_dir=artifacts_dir,
        runtime=runtime,
        agent_result=AgentRunResult(status="completed", return_code=0),
        execution_deadline=asyncio.get_running_loop().time() + 60,
        stage=SessionStage.POSTRUN,
    )
    registry.register(
        session_id,
        task_id=request.task_id,
        registered=True,
        status=SessionStatus.POST_RUN,
    )
    storage.ensure_session(
        session_id,
        model_requested=None,
        model_used=None,
        api_type=None,
        task_id=request.task_id,
    )
    manager._build_trajectory = lambda _request: Trajectory(  # type: ignore[method-assign]
        status="COMPLETED"
    )

    await manager.start()
    try:
        async with manager._dispatcher._lock:
            manager._dispatcher._sessions[session_id] = managed
        await manager._dispatcher._postrun_queue.put(session_id)
        await asyncio.wait_for(evaluator.started.wait(), timeout=1)

        done_event = await manager._dispatcher.cancel(session_id)
        assert done_event is managed.done_event
        await asyncio.wait_for(managed.done_event.wait(), timeout=1)

        assert evaluator.cancelled.is_set()
        info = registry.get(session_id)
        assert info is not None
        assert info.result is not None
        assert info.result.status == SessionStatus.ERROR
        assert info.result.error == "session cancelled"
        assert info.result.trajectory.status == "ERROR"
        assert info.result.trajectory.traces == []
        assert not session_dir.exists()
    finally:
        await manager.close()
