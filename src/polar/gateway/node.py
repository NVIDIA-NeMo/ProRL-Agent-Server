"""Gateway-node execution lifecycle for dispatched rollout sessions."""

from __future__ import annotations

import asyncio
import logging
import shutil
from contextlib import suppress
from pathlib import Path
from tempfile import mkdtemp

import httpx

from polar.gateway.dispatcher import (
    DispatcherSnapshot,
    ManagedSession,
    PreparedRuntimeLease,
    SessionDispatcher,
    SessionStage,
)
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.agent.base import BaseHarness
from polar.agent.factory import create_harness
from polar.agent.models import AgentRunResult
from polar.rollout.models import NodeStageMetrics, SessionDispatchRequest, SessionResult
from polar.rollout.timer import StageTimer
from polar.runtime.base import BaseRuntime
from polar.runtime.factory import create_runtime
from polar.runtime.models import ExecInput, RuntimeSpec
from polar.trajectory.models import CompletionSession, EvalResult, EvaluatorSpec, StrategySpec, Trajectory
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
        ready_buffer_target: int,
        storage: SessionStore,
        session_registry: SessionRegistry,
        builders: StrategyRegistry,
        evaluators: StrategyRegistry,
        default_runtime: RuntimeSpec | None = None,
        session_base_dir: str | None = None,
    ) -> None:
        self.node_id = node_id
        self.gateway_url = gateway_url.rstrip("/")
        self.storage = storage
        self.session_registry = session_registry
        self.builders = builders
        self.evaluators = evaluators
        self.default_runtime = default_runtime
        self._session_base_dir = session_base_dir
        self._client = httpx.AsyncClient(timeout=30.0)
        self._dispatched_session_ids: set[str] = set()
        self._dispatcher = SessionDispatcher(
            max_init_workers=max_init_workers,
            max_run_workers=max_run_workers,
            max_postrun_workers=max_postrun_workers,
            ready_buffer_target=ready_buffer_target,
        )
        self._dispatcher.on_init = self._handle_init
        self._dispatcher.on_eval_prewarm = self._handle_eval_prewarm
        self._dispatcher.on_run = self._handle_run
        self._dispatcher.on_postrun = self._handle_postrun
        self._dispatcher.on_stage_change = self._handle_dispatcher_stage_change

    async def start(self) -> None:
        await self._dispatcher.start()

    async def close(self) -> None:
        await self._dispatcher.stop()
        await self._client.aclose()

    async def dispatch(self, request: SessionDispatchRequest) -> None:
        session_id = request.session_id
        if session_id in self._dispatched_session_ids:
            raise ValueError(
                f"session {session_id} has already been used; rollout session IDs are single-use"
            )
        existing = self.session_registry.get(session_id)
        if existing is not None:
            raise ValueError(
                f"session {session_id} already exists; rollout session IDs are single-use"
            )

        session_dir: Path | None = None
        self._dispatched_session_ids.add(session_id)
        try:
            info = self.session_registry.register(
                session_id,
                task_id=request.task_id,
                registered=True,
                status="REGISTERED",
            )
            self.storage.ensure_session(
                info.session_id,
                model_requested=None,
                model_used=None,
                api_type=None,
                task_id=info.task_id,
                created_at=info.created_at.isoformat(),
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
                    execution_deadline=(
                        asyncio.get_running_loop().time()
                        + request.remaining_timeout_seconds
                    ),
                )
            )
        except Exception:
            self.storage.delete_session(session_id)
            self.session_registry.remove(session_id)
            self._dispatched_session_ids.discard(session_id)
            if session_dir is not None:
                await self._remove_session_dir_best_effort(session_dir, session_id)
            raise

    async def cancel(self, session_id: str) -> bool:
        return await self._dispatcher.cancel(session_id)

    async def active_sessions(self) -> int:
        return await self._dispatcher.active_count()

    async def stage_metrics(self) -> NodeStageMetrics:
        snapshot = await self._dispatcher.snapshot()
        return self._snapshot_to_metrics(snapshot)

    def _handle_dispatcher_stage_change(
        self,
        managed: ManagedSession,
        stage: SessionStage,
    ) -> None:
        status = {
            SessionStage.INITIALIZING: "INITIALIZING",
            SessionStage.READY: "READY",
            SessionStage.RUNNING: "RUNNING",
            SessionStage.POSTRUN_PENDING: "POST_RUN",
            SessionStage.POSTRUNNING: "POST_RUN",
        }.get(stage)
        if status is not None:
            self.session_registry.set_status(managed.request.session_id, status)

    # ------------------------------------------------------------------
    # INIT stage
    # ------------------------------------------------------------------

    async def _handle_init(self, managed: ManagedSession) -> None:
        request = managed.request
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
    ) -> None:
        """Execute the ordered prepare action list."""
        base_env = self._runtime_env(request, managed, runtime_override=runtime)
        for i, action in enumerate(spec.prepare):
            if managed.cancel_requested:
                return
            if action.type == "upload_file":
                await runtime.upload_file(action.source, action.target)
            elif action.type == "upload_dir":
                await runtime.upload_dir(action.source, action.target)
            elif action.type == "exec":
                merged_env = {**base_env, **(action.env or {})}
                # Use action.cwd, falling back to runtime session dir
                # (not spec.workdir which may not exist during prepare)
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
                    log_dir, f"prepare.{i:02d}", result.stdout, result.stderr
                )
                if result.return_code == -1:
                    raise RuntimeError(f"prepare action {i} timed out")
                if result.return_code != 0:
                    raise RuntimeError(
                        f"prepare action {i} failed with exit code {result.return_code}"
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

            await self._maybe_start_eval_runtime_prewarm(managed)
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

        last_stdout: str | None = None
        last_stderr: str | None = None
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
            last_stdout = result.stdout
            last_stderr = result.stderr
            self._write_exec_log(
                log_dir, f"step.{i:02d}", result.stdout, result.stderr
            )
            logger.info(
                "Step %d for session %s: rc=%s stdout_tail=%s",
                i,
                managed.request.session_id,
                result.return_code,
                (result.stdout or "")[-500:],
            )
            if result.return_code == -1:
                return AgentRunResult(
                    status="timeout",
                    return_code=-1,
                    error=f"step {i} timed out",
                    metadata=self._step_metadata(
                        log_dir, i, managed, last_stdout, last_stderr
                    ),
                )
            if result.return_code != 0:
                return AgentRunResult(
                    status="failed",
                    return_code=result.return_code,
                    error=f"step {i} exited with code {result.return_code}",
                    metadata=self._step_metadata(
                        log_dir, i, managed, last_stdout, last_stderr
                    ),
                )

        return AgentRunResult(
            status="completed",
            return_code=0,
            metadata=self._step_metadata(
                log_dir, len(steps) - 1, managed, last_stdout, last_stderr
            ),
        )

    # ------------------------------------------------------------------
    # Evaluator runtime prewarm
    # ------------------------------------------------------------------

    async def _maybe_start_eval_runtime_prewarm(self, managed: ManagedSession) -> None:
        request = managed.request
        if request.evaluator is None or not request.evaluator.refresh_runtime:
            return
        if managed.eval_runtime_lease is None:
            managed.eval_runtime_lease = PreparedRuntimeLease(
                owner_session_id=request.session_id,
                purpose="evaluator_refresh",
            )
        await self._dispatcher.request_eval_prewarm(request.session_id)

    async def _handle_eval_prewarm(self, managed: ManagedSession) -> None:
        """Prepare a fresh runtime for evaluator use."""
        request = managed.request
        lease = managed.eval_runtime_lease
        if lease is None:
            lease = PreparedRuntimeLease(
                owner_session_id=request.session_id,
                purpose="evaluator_refresh",
            )
            managed.eval_runtime_lease = lease
        eval_runtime: BaseRuntime | None = None
        try:
            acquired = await self._dispatcher.acquire_eval_prewarm_slot(
                request.session_id
            )
            if not acquired or lease.cancelled:
                return
            lease.slot_held = True

            runtime_spec = self._resolve_runtime_spec(request)
            eval_session_dir = managed.session_dir / "eval_runtime"
            eval_artifacts_dir = eval_session_dir / "artifacts"
            eval_artifacts_dir.mkdir(parents=True, exist_ok=True)

            eval_runtime = create_runtime(
                runtime_spec, f"{request.session_id}-eval", eval_session_dir
            )
            await self._await_with_budget(eval_runtime.start(), managed)
            await self._run_runtime_prepare(
                eval_runtime, runtime_spec, request, managed
            )

            lease.runtime = eval_runtime
            lease.session_dir = eval_session_dir
            lease.artifacts_dir = eval_artifacts_dir
            lease.error = None
        except GatewayExecutionTimeout as exc:
            lease.error = str(exc)
            logger.warning(
                "Eval runtime prewarm timed out for session %s: %s",
                request.session_id,
                exc,
            )
        except Exception as exc:
            lease.error = str(exc)
            logger.warning(
                "Eval runtime prewarm failed for session %s: %s",
                request.session_id,
                exc,
            )
        finally:
            if lease.error is not None and eval_runtime is not None:
                with suppress(Exception):
                    await eval_runtime.stop()
            if lease.error is not None and lease.slot_held:
                await self._dispatcher.consume_eval_prewarm(request.session_id)
            lease.ready.set()

    async def _acquire_prepared_eval_runtime(
        self, managed: ManagedSession
    ) -> BaseRuntime | None:
        """Wait for and return the prewarmed evaluator runtime."""
        lease = managed.eval_runtime_lease
        if lease is None:
            return None
        try:
            await asyncio.wait_for(
                lease.ready.wait(),
                timeout=self._remaining_budget(managed),
            )
        except asyncio.TimeoutError as exc:
            raise GatewayExecutionTimeout(
                "timed out waiting for a fresh evaluator runtime"
            ) from exc
        lease = await self._dispatcher.consume_eval_prewarm(managed.request.session_id) or lease
        if lease.error or lease.cancelled or lease.runtime is None:
            return None
        return lease.runtime

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
            if managed.eval_runtime_lease is not None:
                managed.eval_runtime_lease.cancelled = True
                if managed.eval_runtime_lease.slot_held:
                    await self._dispatcher.consume_eval_prewarm(request.session_id)
                eval_rt = managed.eval_runtime_lease.runtime
                if eval_rt is not None:
                    stop_tasks.append(
                        self._stop_runtime_best_effort(
                            eval_rt, request.session_id, "eval runtime"
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
            await self._push_result(request.callback_url, normalized)
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

        self.session_registry.set_status(request.session_id, "BUILDING")
        await self.storage.drain_pending_saves(request.session_id)
        managed.timer.mark("build", "started")
        try:
            trajectory, completion_session = await self._await_with_budget(
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
                self.session_registry.set_status(request.session_id, "EVALUATING")
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
            completion_session=completion_session,
            timing=managed.timer.to_session_timing(),
            node_id=self.node_id,
            error=error,
        )

    def _build_trajectory(
        self, request: SessionDispatchRequest
    ) -> tuple[Trajectory, CompletionSession]:
        completion_session = self.storage.load_completion_session(request.session_id)
        builder = self.builders.create(request.builder)
        result = builder.build(completion_session)
        if asyncio.iscoroutine(result):
            trajectory = asyncio.run(result)
        else:
            trajectory = result
        return Trajectory.model_validate(trajectory), completion_session

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
                lease = managed.eval_runtime_lease
                failure = (
                    lease.error
                    if lease is not None and lease.error
                    else "fresh evaluator runtime was unavailable"
                )
                return trajectory.model_copy(
                    update={
                        "status": "ERROR",
                        "error": f"refresh_runtime=true requires a fresh runtime: {failure}",
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
            last = traces[-1].model_copy(update={"reward": eval_result.outcome_reward})
            traces = traces[:-1] + [last]

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
    def _step_metadata(
        log_dir: Path,
        step_index: int,
        managed: ManagedSession,
        last_stdout: str | None = None,
        last_stderr: str | None = None,
    ) -> dict:
        meta: dict = {
            "log_dir": str(log_dir),
            "last_step": step_index,
            "cwd": str(managed.session_dir),
        }
        # Include truncated output tails so they survive session dir cleanup
        if last_stdout:
            meta["stdout_tail"] = last_stdout[-4000:]
        if last_stderr:
            meta["stderr_tail"] = last_stderr[-4000:]
        return meta

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
                metadata={"builder": request.builder.strategy, "record_count": 0},
                traces=[],
                error=error,
            ),
            timing=timer.to_session_timing(),
            node_id=self.node_id,
            error=error,
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
                metadata={"builder": request.builder.strategy, "record_count": 0},
                traces=[],
                error=error,
            ),
            timing=timer.to_session_timing(),
            node_id=self.node_id,
            error=error,
        )

    def _cancelled_result(self, request: SessionDispatchRequest, timer: StageTimer) -> SessionResult:
        return self._error_result(request, timer, "session cancelled")

    async def _push_result(self, callback_url: str | None, result: SessionResult) -> None:
        if not callback_url:
            return
        try:
            response = await self._client.post(callback_url, json=result.model_dump(mode="json"))
            response.raise_for_status()
        except Exception:
            logger.warning(
                "Failed to deliver callback for session %s to %s",
                result.session_id,
                callback_url,
                exc_info=True,
            )

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
