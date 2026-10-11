"""Gateway-node execution lifecycle for dispatched rollout sessions."""

from __future__ import annotations

import asyncio
import logging
import shutil
from contextlib import suppress
from pathlib import Path
from tempfile import mkdtemp
from typing import Awaitable, Callable

import httpx

from polar.gateway.dispatcher import (
    DispatcherSnapshot,
    ManagedSession,
    SessionDispatcher,
    SessionStage,
)
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.agent.base import BaseHarness
from polar.agent.factory import create_harness
from polar.agent.models import AgentRunResult
from polar.rollout.models import (
    NodeHeartbeatRequest,
    NodeRegistrationRequest,
    NodeStageMetrics,
    SessionDispatchRequest,
    SessionResult,
    SessionStatus,
)
from polar.rollout.timer import StageTimer
from polar.runtime.base import BaseRuntime
from polar.runtime.factory import create_runtime
from polar.runtime.models import ExecInput, RuntimeSpec
from polar.trajectory.models import EvalResult, EvaluatorSpec, StrategySpec, Trajectory
from polar.trajectory.registry import StrategyRegistry

logger = logging.getLogger(__name__)


class GatewayExecutionTimeout(TimeoutError):
    """Raised when a session exhausts its shared gateway execution budget."""


class GatewayNodeManager:
    """Run the INIT/READY/RUN/POST_RUN lifecycle on one gateway node."""

    def __init__(
        self,
        *,
        node_id: str,
        gateway_url: str,
        max_init_workers: int,
        max_run_workers: int,
        max_postrun_workers: int,
        storage: SessionStore,
        session_registry: SessionRegistry,
        builders: StrategyRegistry,
        evaluators: StrategyRegistry,
        default_runtime: RuntimeSpec | None = None,
        session_base_dir: str | None = None,
        rollout_server_url: str | None = None,
        heartbeat_interval_seconds: int = 30,
        program_releaser: Callable[[str], Awaitable[bool]] | None = None,
    ) -> None:
        self.node_id = node_id
        self.gateway_url = gateway_url.rstrip("/")
        self.max_init_workers = max_init_workers
        self.max_run_workers = max_run_workers
        self.max_postrun_workers = max_postrun_workers
        self.storage = storage
        self.session_registry = session_registry
        self.builders = builders
        self.evaluators = evaluators
        self.default_runtime = default_runtime
        self._session_base_dir = session_base_dir
        self._client = httpx.AsyncClient(timeout=30.0)
        self._dispatcher = SessionDispatcher(
            max_init_workers=max_init_workers,
            max_run_workers=max_run_workers,
            max_postrun_workers=max_postrun_workers,
        )
        self._dispatcher.on_init = self._handle_init
        self._dispatcher.on_run = self._handle_run
        self._dispatcher.on_postrun = self._handle_postrun
        self._dispatcher.on_stage_change = self._handle_dispatcher_stage_change

        self._rollout_server_url = rollout_server_url.rstrip("/") if rollout_server_url else None
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._control_client: httpx.AsyncClient | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._program_releaser = program_releaser

    async def start(self) -> None:
        await self._dispatcher.start()
        if self._rollout_server_url is not None:
            self._control_client = httpx.AsyncClient(
                base_url=self._rollout_server_url, timeout=15.0
            )
            await self._register_with_rollout_server()
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def close(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            self._heartbeat_task = None
        if self._control_client is not None:
            await self._control_client.aclose()
            self._control_client = None
        await self._dispatcher.stop()
        await self._client.aclose()

    async def _register_with_rollout_server(self) -> None:
        if self._control_client is None:
            return
        try:
            response = await self._control_client.post(
                "/nodes/register",
                json=NodeRegistrationRequest(
                    node_id=self.node_id,
                    gateway_url=self.gateway_url,
                    max_init_workers=self.max_init_workers,
                    max_run_workers=self.max_run_workers,
                    max_postrun_workers=self.max_postrun_workers,
                    heartbeat_interval_seconds=self._heartbeat_interval_seconds,
                ).model_dump(mode="json"),
            )
            response.raise_for_status()
        except Exception:
            logger.warning("Node registration failed", exc_info=True)

    async def _heartbeat_loop(self) -> None:
        assert self._control_client is not None
        while True:
            await asyncio.sleep(self._heartbeat_interval_seconds)
            try:
                metrics = await self.stage_metrics()
                response = await self._control_client.post(
                    f"/nodes/{self.node_id}/heartbeat",
                    json=NodeHeartbeatRequest(metrics=metrics).model_dump(mode="json"),
                )
                if response.status_code == 404:
                    await self._register_with_rollout_server()
                    continue
                response.raise_for_status()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Node heartbeat failed", exc_info=True)

    async def dispatch(self, request: SessionDispatchRequest) -> None:
        session_id = request.session_id
        if self.session_registry.get(session_id) is not None:
            raise ValueError(
                f"session {session_id} already exists; rollout session IDs are single-use"
            )

        session_dir: Path | None = None
        try:
            info = self.session_registry.register(
                session_id,
                task_id=request.task_id,
                registered=True,
                status=SessionStatus.REGISTERED,
                metadata=dict(request.metadata),
            )
            self.storage.ensure_session(
                info.session_id,
                model_requested=None,
                model_used=None,
                api_type=None,
                task_id=info.task_id,
                created_at=info.created_at.isoformat(),
                metadata=dict(request.metadata),
            )

            timer = StageTimer()
            timer.mark("dispatch", "started")
            session_dir = Path(
                mkdtemp(prefix=f"session-{session_id[:8]}-", dir=self._session_base_dir)
            )
            artifacts_dir = session_dir / "artifacts"
            artifacts_dir.mkdir()
            (session_dir / "logs" / "agent").mkdir(parents=True, exist_ok=True)
            await self._dispatcher.enqueue(
                ManagedSession(
                    request=request,
                    timer=timer,
                    session_dir=session_dir,
                    artifacts_dir=artifacts_dir,
                )
            )
        except Exception:
            self.storage.delete_session(session_id)
            self.session_registry.remove(session_id)
            if session_dir is not None:
                await self._remove_session_dir_best_effort(session_dir, session_id)
            raise

    async def cancel(self, session_id: str) -> bool:
        return await self._dispatcher.cancel(session_id)

    async def release_program(self, session_id: str) -> None:
        """Best-effort release of external scheduler state for one session."""
        if self._program_releaser is None:
            return
        try:
            await self._program_releaser(session_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Failed to release scheduler program for session %s",
                session_id,
                exc_info=True,
            )

    async def active_sessions(self) -> int:
        return await self._dispatcher.active_count()

    async def stage_metrics(self) -> NodeStageMetrics:
        snapshot = await self._dispatcher.snapshot()
        return self._snapshot_to_metrics(snapshot)

    def _handle_dispatcher_stage_change(self, managed: ManagedSession) -> None:
        status = {
            SessionStage.INIT: SessionStatus.INITIALIZING,
            SessionStage.READY: SessionStatus.READY,
            SessionStage.RUNNING: SessionStatus.RUNNING,
            SessionStage.POSTRUN: SessionStatus.POST_RUN,
        }.get(managed.stage)
        if status is not None:
            self.session_registry.set_status(managed.request.session_id, status)

    # ------------------------------------------------------------------
    # INIT stage
    # ------------------------------------------------------------------

    async def _handle_init(self, managed: ManagedSession) -> None:
        request = managed.request
        self._start_execution_deadline(managed)
        managed.timer.mark("init", "started")
        try:
            runtime_spec = self._resolve_runtime_spec(request)
            runtime = create_runtime(runtime_spec, request.session_id, managed.session_dir)
            managed.runtime = runtime
            await self._await_with_budget(runtime.start(), managed)
            # Run ordered prepare actions
            await self._run_runtime_prepare(runtime, runtime_spec, request, managed)
        except GatewayExecutionTimeout as exc:
            managed.final_result = self._timeout_result(request, managed.timer, str(exc))
        except Exception as exc:
            if managed.cancel_requested:
                logger.info("Initialization cancelled for session %s", request.session_id)
            else:
                logger.exception("Initialization failed for session %s", request.session_id)
                managed.final_result = self._error_result(
                    request,
                    managed.timer,
                    f"runtime initialization failed: {exc}",
                )
        finally:
            managed.timer.mark("init", "finished")

    def _resolve_runtime_spec(self, request: SessionDispatchRequest) -> RuntimeSpec:
        spec = request.runtime or self.default_runtime
        if spec is None:
            raise RuntimeError(
                "no runtime configured: request has no runtime and gateway "
                "node has no default_runtime"
            )
        return spec

    async def _run_runtime_prepare(
        self,
        runtime: BaseRuntime,
        spec: RuntimeSpec,
        request: SessionDispatchRequest,
        managed: ManagedSession,
        *,
        actions: list | None = None,
        log_prefix: str = "prepare",
    ) -> None:
        """Execute an ordered prepare action list (``spec.prepare`` by default)."""
        steps = actions if actions is not None else spec.prepare
        base_env = self._runtime_env(request, managed, runtime_override=runtime)
        for i, action in enumerate(steps):
            if managed.cancel_requested:
                return
            if action.type == "upload_file":
                await runtime.upload_file(action.source, action.target)
            elif action.type == "upload_dir":
                await runtime.upload_dir(action.source, action.target)
            elif action.type == "exec":
                merged_env = {**base_env, **(action.env or {})}
                effective_cwd = action.cwd or runtime.runtime_session_dir
                result = await runtime.exec(
                    action.command,
                    cwd=effective_cwd,
                    env=merged_env,
                    timeout_sec=self._remaining_budget(managed),
                )
                log_dir = managed.session_dir / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                self._write_exec_log(
                    log_dir, f"{log_prefix}.{i:02d}", result.stdout, result.stderr
                )
                if result.return_code == -1:
                    raise RuntimeError(f"{log_prefix} action {i} timed out")
                if result.return_code != 0:
                    raise RuntimeError(
                        f"{log_prefix} action {i} failed with exit code {result.return_code}"
                    )

    # ------------------------------------------------------------------
    # RUN stage
    # ------------------------------------------------------------------

    async def _handle_run(self, managed: ManagedSession) -> None:
        request = managed.request
        if managed.final_result is not None or managed.cancel_requested:
            return
        managed.timer.mark("run", "started")

        harness: BaseHarness | None = None
        try:
            runtime = managed.runtime
            if runtime is None:
                raise RuntimeError("runtime is required for execution")

            self._start_eval_prewarm(managed)
            harness = self._resolve_agent_harness(request)

            # Setup
            await self._await_with_budget(harness.setup(runtime), managed)

            # Run
            steps = harness.run_steps(request.instruction)
            env = self._runtime_env(request, managed, include_agent_env=True)
            agent_result = await self._run_exec_inputs(runtime, steps, env, managed)

            # Postprocess always runs so harnesses can collect artifacts from
            # failed or timed-out agent runs before post-run evaluation.
            await self._await_with_budget(harness.postprocess(runtime, agent_result), managed)
            managed.agent_result = agent_result

        except GatewayExecutionTimeout as exc:
            # Don't set final_result — let _handle_postrun build a partial
            # trajectory from the completions captured so far.
            managed.agent_result = AgentRunResult(
                status="timeout", return_code=-1, error=str(exc),
            )
        except Exception as exc:
            if managed.cancel_requested:
                logger.info("Agent execution cancelled for session %s", request.session_id)
            else:
                logger.exception("Agent execution failed for session %s", request.session_id)
                managed.final_result = self._error_result(
                    request,
                    managed.timer,
                    f"agent execution failed: {exc}",
                )
        finally:
            if harness is not None:
                managed.postrun_steps = harness.postrun_steps()
            managed.timer.mark("run", "finished")

    def _resolve_agent_harness(self, request: SessionDispatchRequest) -> BaseHarness:
        return create_harness(request.agent)

    async def _run_exec_inputs(
        self,
        runtime: BaseRuntime,
        steps: list[ExecInput],
        env: dict[str, str],
        managed: ManagedSession,
    ) -> AgentRunResult:
        """Execute a list of ExecInput steps and return an AgentRunResult."""
        log_dir = managed.session_dir / "logs" / "agent"
        log_dir.mkdir(parents=True, exist_ok=True)

        for i, step in enumerate(steps):
            if managed.cancel_requested:
                return AgentRunResult(
                    status="failed", return_code=-1, error="cancelled"
                )
            merged_env = {**env, **(step.env or {})}
            result = await runtime.exec(
                step.command,
                cwd=step.cwd,
                env=merged_env,
                timeout_sec=self._remaining_budget(managed),
            )
            self._write_exec_log(
                log_dir, f"step.{i:02d}", result.stdout, result.stderr
            )
            if result.return_code == -1:
                return AgentRunResult(
                    status="timeout",
                    return_code=-1,
                    error=f"step {i} timed out",
                    metadata=self._step_metadata(log_dir, i, managed),
                )
            if result.return_code != 0:
                return AgentRunResult(
                    status="failed",
                    return_code=result.return_code,
                    error=f"step {i} exited with code {result.return_code}",
                    metadata=self._step_metadata(log_dir, i, managed),
                )

        return AgentRunResult(
            status="completed",
            return_code=0,
            metadata=self._step_metadata(log_dir, len(steps) - 1, managed),
        )

    # ------------------------------------------------------------------
    # Evaluator runtime prewarm
    # ------------------------------------------------------------------

    def _start_eval_prewarm(self, managed: ManagedSession) -> None:
        """Spawn a background task to prewarm a fresh evaluator runtime."""
        request = managed.request
        if request.evaluator is None or not request.evaluator.refresh_runtime:
            return
        if managed.eval_prewarm_task is not None:
            return
        managed.eval_prewarm_task = asyncio.create_task(
            self._prepare_eval_runtime(managed)
        )

    async def _prepare_eval_runtime(
        self, managed: ManagedSession
    ) -> BaseRuntime | None:
        """Create and prepare a fresh runtime for the evaluator. Returns None on failure."""
        request = managed.request
        runtime_spec = self._resolve_runtime_spec(request)
        eval_session_dir = managed.session_dir / "eval_runtime"
        eval_artifacts_dir = eval_session_dir / "artifacts"
        eval_artifacts_dir.mkdir(parents=True, exist_ok=True)

        eval_runtime = create_runtime(
            runtime_spec, f"{request.session_id}-eval", eval_session_dir
        )
        try:
            await self._await_with_budget(eval_runtime.start(), managed)
            eval_actions = (
                runtime_spec.eval_prepare
                if runtime_spec.eval_prepare is not None
                else runtime_spec.prepare
            )
            await self._run_runtime_prepare(
                eval_runtime,
                runtime_spec,
                request,
                managed,
                actions=eval_actions,
                log_prefix="eval_prepare",
            )
            return eval_runtime
        except asyncio.CancelledError:
            with suppress(Exception):
                await eval_runtime.stop()
            raise
        except Exception as exc:
            logger.warning(
                "Eval runtime prewarm failed for session %s: %s",
                request.session_id,
                exc,
            )
            with suppress(Exception):
                await eval_runtime.stop()
            return None

    async def _acquire_prepared_eval_runtime(
        self, managed: ManagedSession
    ) -> BaseRuntime | None:
        """Await the prewarm task and return its runtime, if any."""
        task = managed.eval_prewarm_task
        if task is None:
            return None
        try:
            return await asyncio.wait_for(
                asyncio.shield(task), timeout=self._remaining_budget(managed)
            )
        except asyncio.TimeoutError as exc:
            raise GatewayExecutionTimeout(
                "timed out waiting for a fresh evaluator runtime"
            ) from exc

    async def _drain_eval_prewarm_task(
        self, managed: ManagedSession
    ) -> BaseRuntime | None:
        """Resolve the prewarm task during teardown. Cancel if still running."""
        task = managed.eval_prewarm_task
        if task is None:
            return None
        if not task.done():
            task.cancel()
        try:
            return await task
        except (asyncio.CancelledError, Exception):
            return None

    # ------------------------------------------------------------------
    # POSTRUN stage
    # ------------------------------------------------------------------

    async def _handle_postrun(self, managed: ManagedSession) -> None:
        request = managed.request
        result: SessionResult | None = managed.final_result
        managed.timer.mark("postrun", "started")
        try:
            if result is None:
                if managed.cancel_requested:
                    result = self._cancelled_result(request, managed.timer)
                else:
                    result = await self._build_session_result(managed)
        except GatewayExecutionTimeout as exc:
            result = self._timeout_result(request, managed.timer, str(exc))
        except Exception as exc:
            logger.exception("Post-run handling failed for session %s", request.session_id)
            result = self._error_result(request, managed.timer, f"post-run failed: {exc}")
        finally:
            managed.timer.mark("postrun", "finished")
            managed.timer.mark("teardown", "started")
            await self._run_postrun_steps(managed)
            stop_tasks = []
            eval_runtime = await self._drain_eval_prewarm_task(managed)
            if eval_runtime is not None:
                stop_tasks.append(
                    self._stop_runtime_best_effort(
                        eval_runtime, request.session_id, "eval runtime"
                    )
                )
            if managed.runtime is not None:
                stop_tasks.append(
                    self._stop_runtime_best_effort(
                        managed.runtime, request.session_id, "runtime"
                    )
                )
            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)
            managed.timer.mark("teardown", "finished")
            managed.timer.mark("return", "finished")

        if result is None:
            result = self._error_result(
                request,
                managed.timer,
                "post-run finished without producing a session result",
            )
        try:
            normalized = result.model_copy(
                update={
                    "timing": managed.timer.to_session_timing(),
                    "node_id": self.node_id,
                    "error": result.error or result.trajectory.error,
                }
            )
            self.session_registry.set_result(request.session_id, normalized)
            self.storage.delete_session(request.session_id)
            if await self._push_result(request.callback_url, normalized):
                # Rollout server has acked; free the heavy payload but keep
                # status/task_id visible for debugging via the polling endpoint.
                self.session_registry.clear_result_payload(request.session_id)
        finally:
            try:
                await self.release_program(request.session_id)
            finally:
                await self._remove_session_dir_best_effort(
                    managed.session_dir, request.session_id
                )

    async def _build_session_result(self, managed: ManagedSession) -> SessionResult:
        request = managed.request
        agent_result = managed.agent_result
        if agent_result is None:
            return self._error_result(
                request,
                managed.timer,
                "session did not produce an agent result",
            )

        self.session_registry.set_status(request.session_id, SessionStatus.BUILDING)
        managed.timer.mark("build", "started")
        try:
            trajectory = await self._await_with_budget(
                asyncio.to_thread(self._build_trajectory, request),
                managed,
            )
        finally:
            managed.timer.mark("build", "finished")

        error = trajectory.error
        if agent_result.status == "timeout":
            trajectory = trajectory.model_copy(
                update={"status": "TIMEOUT", "error": agent_result.error or error}
            )
        elif agent_result.status == "failed":
            trajectory = trajectory.model_copy(
                update={"status": "ERROR", "error": agent_result.error or error}
            )

        managed.timer.mark("eval", "started")
        try:
            if request.evaluator is not None:
                self.session_registry.set_status(request.session_id, SessionStatus.EVALUATING)
                trajectory = await self._run_eval(
                    request,
                    trajectory,
                    agent_result=agent_result,
                    managed=managed,
                )
        except GatewayExecutionTimeout as exc:
            # Preserve the built trajectory even when eval times out.
            logger.warning("Eval timed out for session %s: %s", request.session_id, exc)
            if trajectory.status not in ("TIMEOUT", "ERROR"):
                trajectory = trajectory.model_copy(
                    update={"status": "TIMEOUT", "error": f"eval timed out: {exc}"}
                )
        except Exception as exc:
            logger.exception("Eval failed for session %s", request.session_id)
            trajectory = trajectory.model_copy(
                update={"status": "ERROR", "error": f"evaluator failed: {exc}"}
            )
        finally:
            managed.timer.mark("eval", "finished")

        error = trajectory.error or error
        return SessionResult(
            session_id=request.session_id,
            task_id=request.task_id,
            status=trajectory.status,
            trajectory=trajectory,
            timing=managed.timer.to_session_timing(),
            node_id=self.node_id,
            error=error,
            metadata=dict(request.metadata),
        )

    def _build_trajectory(self, request: SessionDispatchRequest) -> Trajectory:
        completion_session = self.storage.load_completion_session(request.session_id)
        builder = self.builders.create(request.builder)
        result = builder.build(completion_session)
        if asyncio.iscoroutine(result):
            trajectory = asyncio.run(result)
        else:
            trajectory = result
        return Trajectory.model_validate(trajectory)

    async def _run_eval(
        self,
        request: SessionDispatchRequest,
        trajectory: Trajectory,
        *,
        agent_result: AgentRunResult,
        managed: ManagedSession,
    ) -> Trajectory:
        evaluator_spec = request.evaluator
        if evaluator_spec is None:
            return trajectory

        live_runtime = managed.runtime
        if live_runtime is None:
            raise RuntimeError("runtime is required for evaluation")

        fresh_eval_runtime: BaseRuntime | None = None
        if evaluator_spec.refresh_runtime:
            fresh_eval_runtime = await self._acquire_prepared_eval_runtime(managed)
            if fresh_eval_runtime is None:
                return trajectory.model_copy(
                    update={
                        "status": "ERROR",
                        "error": "refresh_runtime=true requires a fresh runtime: eval runtime prewarm did not produce a usable runtime",
                    }
                )

        # Convert EvaluatorSpec to StrategySpec for registry
        strategy_spec = StrategySpec(
            strategy=evaluator_spec.strategy,
            config=evaluator_spec.config,
        )

        try:
            evaluator = self.evaluators.create(strategy_spec)
            eval_result = await self._await_with_budget(
                evaluator.evaluate(
                    trajectory,
                    session_id=request.session_id,
                    task_id=request.task_id,
                    session_dir=managed.session_dir,
                    artifacts_dir=managed.artifacts_dir,
                    agent_result=agent_result,
                    env=dict(evaluator_spec.env),
                    timeout_seconds=self._remaining_budget(managed),
                    runtime=live_runtime,
                    fresh_eval_runtime=fresh_eval_runtime,
                    runtime_spec=request.runtime or self.default_runtime,
                    refresh_runtime=evaluator_spec.refresh_runtime,
                ),
                managed,
            )
        except Exception as exc:
            logger.exception(
                "Evaluator %s failed for session %s",
                evaluator_spec.strategy,
                request.session_id,
            )
            return trajectory.model_copy(
                update={"status": "ERROR", "error": f"evaluator failed: {exc}"}
            )

        return self._merge_eval_result(trajectory, eval_result, evaluator_spec)

    @staticmethod
    def _merge_eval_result(
        trajectory: Trajectory,
        eval_result: EvalResult,
        evaluator_spec: EvaluatorSpec,
    ) -> Trajectory:
        """Apply rewards from EvalResult to trajectory traces."""
        traces = list(trajectory.traces)

        if eval_result.trace_rewards is not None:
            if len(eval_result.trace_rewards) != len(traces):
                return trajectory.model_copy(
                    update={
                        "status": "ERROR",
                        "error": (
                            f"evaluator returned {len(eval_result.trace_rewards)} "
                            f"trace_rewards but trajectory has {len(traces)} traces"
                        ),
                    }
                )
            traces = [
                trace.model_copy(update={"reward": reward})
                for trace, reward in zip(traces, eval_result.trace_rewards)
            ]
        elif eval_result.outcome_reward is not None and traces:
            # Broadcast trajectory-level reward 
            traces = [
                trace.model_copy(update={"reward": eval_result.outcome_reward})
                for trace in traces
            ]

        eval_metadata = {
            "strategy": evaluator_spec.strategy,
            "outcome_reward": eval_result.outcome_reward,
            "trace_rewards": eval_result.trace_rewards,
            **eval_result.metadata,
        }
        metadata = {**trajectory.metadata, "evaluation": eval_metadata}
        return trajectory.model_copy(update={"traces": traces, "metadata": metadata})

    # ------------------------------------------------------------------
    # Environment and helpers
    # ------------------------------------------------------------------

    def _runtime_env(
        self,
        request: SessionDispatchRequest,
        managed: ManagedSession,
        *,
        include_agent_env: bool = False,
        runtime_override: BaseRuntime | None = None,
    ) -> dict[str, str]:
        runtime = runtime_override or managed.runtime
        if runtime is None:
            session_dir = str(managed.session_dir)
            artifacts_dir = str(managed.artifacts_dir)
            logs_dir = str(managed.session_dir / "logs")
            agent_log_dir = str(managed.session_dir / "logs" / "agent")
            runtime_env: dict[str, str] = {}
        else:
            session_dir = runtime.runtime_session_dir
            artifacts_dir = runtime.runtime_artifacts_dir
            logs_dir = runtime.runtime_logs_dir
            agent_log_dir = runtime.runtime_agent_log_dir
            runtime_env = dict(runtime.spec.env)
        agent_env = dict(request.agent.env) if include_agent_env else {}
        return {
            "ANTHROPIC_BASE_URL": self.gateway_url,
            "ANTHROPIC_API_KEY": request.session_id,
            "OPENAI_BASE_URL": f"{self.gateway_url.rstrip('/')}/v1",
            "OPENAI_API_KEY": request.session_id,
            "GOOGLE_API_URL": self.gateway_url,
            "GOOGLE_API_KEY": request.session_id,
            "SESSION_ID": request.session_id,
            "TASK_ID": request.task_id,
            "SESSION_DIR": session_dir,
            "ARTIFACTS_DIR": artifacts_dir,
            "LOGS_DIR": logs_dir,
            "AGENT_LOG_DIR": agent_log_dir,
            **{key: str(value) for key, value in runtime_env.items()},
            **{key: str(value) for key, value in agent_env.items()},
        }

    @staticmethod
    def _write_exec_log(
        log_dir: Path, prefix: str, stdout: str | None, stderr: str | None
    ) -> None:
        if stdout:
            (log_dir / f"{prefix}.stdout.log").write_text(stdout)
        if stderr:
            (log_dir / f"{prefix}.stderr.log").write_text(stderr)

    @staticmethod
    def _step_metadata(log_dir: Path, step_index: int, managed: ManagedSession) -> dict:
        return {
            "log_dir": str(log_dir),
            "last_step": step_index,
            "cwd": str(managed.session_dir),
        }

    def _error_result(
        self,
        request: SessionDispatchRequest,
        timer: StageTimer,
        error: str,
    ) -> SessionResult:
        return SessionResult(
            session_id=request.session_id,
            task_id=request.task_id,
            status="ERROR",
            trajectory=Trajectory(
                status="ERROR",
                metadata={
                    "builder": request.builder.strategy,
                    "record_count": 0,
                    "task_metadata": dict(request.metadata),
                },
                traces=[],
                error=error,
            ),
            timing=timer.to_session_timing(),
            node_id=self.node_id,
            error=error,
            metadata=dict(request.metadata),
        )

    def _timeout_result(
        self,
        request: SessionDispatchRequest,
        timer: StageTimer,
        error: str,
    ) -> SessionResult:
        return SessionResult(
            session_id=request.session_id,
            task_id=request.task_id,
            status="TIMEOUT",
            trajectory=Trajectory(
                status="TIMEOUT",
                metadata={
                    "builder": request.builder.strategy,
                    "record_count": 0,
                    "task_metadata": dict(request.metadata),
                },
                traces=[],
                error=error,
            ),
            timing=timer.to_session_timing(),
            node_id=self.node_id,
            error=error,
            metadata=dict(request.metadata),
        )

    def _cancelled_result(self, request: SessionDispatchRequest, timer: StageTimer) -> SessionResult:
        return self._error_result(request, timer, "session cancelled")

    async def _push_result(self, callback_url: str | None, result: SessionResult) -> bool:
        """POST the terminal result to the rollout server. Return True on success."""
        if not callback_url:
            return False
        try:
            response = await self._client.post(callback_url, json=result.model_dump(mode="json"))
            response.raise_for_status()
            return True
        except Exception:
            logger.warning(
                "Failed to deliver callback for session %s to %s",
                result.session_id,
                callback_url,
                exc_info=True,
            )
            return False

    @staticmethod
    def _snapshot_to_metrics(snapshot: DispatcherSnapshot) -> NodeStageMetrics:
        return NodeStageMetrics(
            init_queue_depth=snapshot.init_queue_depth,
            init_inflight=snapshot.init_inflight,
            ready_depth=snapshot.ready_depth,
            run_inflight=snapshot.run_inflight,
            postrun_queue_depth=snapshot.postrun_queue_depth,
            postrun_inflight=snapshot.postrun_inflight,
        )

    def _remaining_budget(self, managed: ManagedSession) -> float:
        deadline = managed.execution_deadline
        if deadline is None:
            raise RuntimeError("session execution deadline was not initialized")
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise GatewayExecutionTimeout("session execution timeout")
        return remaining

    async def _await_with_budget(
        self,
        awaitable,
        managed: ManagedSession,
    ):
        try:
            return await asyncio.wait_for(
                awaitable,
                timeout=self._remaining_budget(managed),
            )
        except asyncio.TimeoutError as exc:
            raise GatewayExecutionTimeout("session execution timeout") from exc

    @staticmethod
    def _start_execution_deadline(managed: ManagedSession) -> None:
        if managed.execution_deadline is not None:
            return
        managed.execution_deadline = (
            asyncio.get_running_loop().time()
            + managed.request.remaining_timeout_seconds
        )

    async def _run_postrun_steps(self, managed: ManagedSession) -> None:
        if not managed.postrun_steps or managed.runtime is None:
            return
        log_dir = managed.session_dir / "logs" / "teardown"
        log_dir.mkdir(parents=True, exist_ok=True)
        env = self._runtime_env(managed.request, managed, include_agent_env=True)
        for i, step in enumerate(managed.postrun_steps):
            try:
                merged_env = {**env, **(step.env or {})}
                result = await managed.runtime.exec(
                    step.command,
                    cwd=step.cwd,
                    env=merged_env,
                    timeout_sec=self._remaining_budget(managed),
                )
                self._write_exec_log(
                    log_dir,
                    f"step.{i:02d}",
                    result.stdout,
                    result.stderr,
                )
            except Exception:
                logger.debug(
                    "Teardown step failed for session %s",
                    managed.request.session_id,
                    exc_info=True,
                )

    async def _stop_runtime_best_effort(
        self,
        runtime: BaseRuntime,
        session_id: str,
        label: str,
    ) -> None:
        try:
            await runtime.stop()
        except Exception:
            logger.warning(
                "Failed to stop %s for session %s",
                label,
                session_id,
                exc_info=True,
            )

    async def _remove_session_dir_best_effort(
        self,
        session_dir: Path,
        session_id: str,
    ) -> None:
        try:
            await asyncio.to_thread(shutil.rmtree, session_dir)
        except FileNotFoundError:
            return
        except Exception:
            logger.warning(
                "Failed to remove session directory for session %s",
                session_id,
                exc_info=True,
            )
