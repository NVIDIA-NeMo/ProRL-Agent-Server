"""Gateway-node execution lifecycle for dispatched rollout sessions."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import httpx

from polar.gateway.dispatcher import (
    DispatcherSnapshot,
    ManagedSession,
    SessionDispatcher,
    SessionStage,
)
from polar.gateway.session import (
    MODEL_POOL_ADMISSION_CAPABILITY_ENV,
    MODEL_POOL_ADMISSION_CAPABILITY_SCOPE,
    MODEL_POOL_CAPABILITY_ENV,
    ROUTER_CAPABILITY_ENV,
    ROUTER_CAPABILITY_SCOPE,
    SessionRegistry,
)
from polar.gateway.episode_admission import ModelPoolEpisodeAdmission
from polar.gateway.storage import SessionStore
from polar.agent.base import BaseHarness
from polar.agent.factory import create_harness
from polar.agent.models import AgentRunResult
from polar.agent.presets.mini_swe_agent import load_mini_swe_command_timing
from polar.rollout.models import (
    NodeHeartbeatRequest,
    NodeRegistrationRequest,
    NodeStageMetrics,
    SessionDispatchRequest,
    SessionResult,
    SessionStatus,
)
from polar.rollout.timer import StageTimer
from polar.runtime.base import BaseRuntime, RuntimeContainmentError
from polar.runtime.factory import create_runtime
from polar.runtime.models import ExecInput, RuntimeSpec
from polar.trajectory.models import EvalResult, EvaluatorSpec, StrategySpec, Trace, Trajectory
from polar.trajectory.registry import StrategyRegistry
from polar.trajectory.training_filter import zero_reward_parser_invalid_tool_call_trace

logger = logging.getLogger(__name__)

_CALLBACK_MAX_ATTEMPTS = 3
_CALLBACK_RETRY_BACKOFF_SECONDS = 0.1
_CALLBACK_REQUEST_TIMEOUT_SECONDS = 5.0
_CALLBACK_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_AGENT_RESULT_METADATA_KEY = "agent_result"
_TRAINABLE_AGENT_TIMEOUT_REASON = "agent_timeout"


class GatewayExecutionTimeout(TimeoutError):
    """Raised when a session exhausts its shared gateway execution budget."""


class GatewayExecutionCancelled(RuntimeError):
    """Raised when cancellation wins a race with post-run evaluation."""


class GatewayNodeUnhealthyError(RuntimeError):
    """The node retained an unsafe episode lease and cannot accept work."""


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
        episode_admission: ModelPoolEpisodeAdmission | None = None,
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
        self.episode_admission = episode_admission or ModelPoolEpisodeAdmission({})
        self._fatal_retained_episode_sessions: set[str] = set()
        self._session_base_dir = session_base_dir
        control_token = os.environ.get("POLAR_CONTROL_PLANE_TOKEN", "").strip()
        control_headers = (
            {"X-Polar-Control-Token": control_token} if control_token else None
        )
        self._control_headers = control_headers
        self._client = httpx.AsyncClient(timeout=30.0, headers=control_headers)
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
        self._cancel_finalizers: dict[str, asyncio.Task[None]] = {}
        self._cancel_lock = asyncio.Lock()

    async def start(self) -> None:
        await self._dispatcher.start()
        if self._rollout_server_url is not None:
            self._control_client = httpx.AsyncClient(
                base_url=self._rollout_server_url,
                timeout=15.0,
                headers=self._control_headers,
            )
            await self._register_with_rollout_server()
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def close(self) -> None:
        await self.episode_admission.poison("gateway node is shutting down")
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            self._heartbeat_task = None
        if self._control_client is not None:
            await self._control_client.aclose()
            self._control_client = None
        close_errors: list[BaseException] = []
        try:
            await self._dispatcher.stop()
        except BaseException as exc:
            close_errors.append(exc)
        try:
            await self.episode_admission.close()
        except BaseException as exc:
            close_errors.append(exc)
        if self._cancel_finalizers:
            await asyncio.gather(
                *tuple(self._cancel_finalizers.values()),
                return_exceptions=True,
            )
        try:
            await self._client.aclose()
        except BaseException as exc:
            close_errors.append(exc)
        if close_errors:
            raise close_errors[0]

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
            await self._send_heartbeat_once()

    async def _send_heartbeat_once(self) -> bool:
        """Send one heartbeat, or return false for a retained-lease fatal node."""

        assert self._control_client is not None
        # A retained active episode lease means the candidate runner could not
        # prove its subreaper-owned candidate process scope disappeared. Stop
        # advertising this
        # node as schedulable; the launcher observes the structured local 503
        # and fails the allocation while rollout-side staleness remains a
        # second line of defense.
        if self._fatal_retained_episode_sessions:
            return False
        try:
            metrics = await self.stage_metrics()
            response = await self._control_client.post(
                f"/nodes/{self.node_id}/heartbeat",
                json=NodeHeartbeatRequest(metrics=metrics).model_dump(mode="json"),
            )
            if response.status_code == 404:
                await self._register_with_rollout_server()
                return True
            response.raise_for_status()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Node heartbeat failed", exc_info=True)
        return True

    async def dispatch(self, request: SessionDispatchRequest) -> None:
        if self._fatal_retained_episode_sessions:
            retained = ", ".join(sorted(self._fatal_retained_episode_sessions))
            raise GatewayNodeUnhealthyError(
                "gateway node is unhealthy after a fatal runtime-containment "
                f"failure for session(s): {retained}"
            )
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
                metadata={
                    **dict(request.metadata),
                    "_polar_agent_harness": request.agent.harness,
                },
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
            router_capability: str | None = None
            model_pool_capability: str | None = None
            model_pool_admission_capability: str | None = None
            if request.agent.harness == "spilot_router":
                router_capability = self.session_registry.issue_capability(
                    session_id,
                    scope=ROUTER_CAPABILITY_SCOPE,
                )
                model_pool_admission_capability = self.session_registry.issue_capability(
                    session_id,
                    scope=MODEL_POOL_ADMISSION_CAPABILITY_SCOPE,
                )
            await self._dispatcher.enqueue(
                ManagedSession(
                    request=request,
                    timer=timer,
                    session_dir=session_dir,
                    artifacts_dir=artifacts_dir,
                    router_capability=router_capability,
                    model_pool_capability=model_pool_capability,
                    model_pool_admission_capability=model_pool_admission_capability,
                )
            )
        except Exception:
            active_retained = await self.episode_admission.cleanup_session_records(
                session_id
            )
            if active_retained:
                await self._mark_fatal_retained_episode(session_id)
            self.storage.delete_session(session_id)
            self.session_registry.remove(session_id)
            if session_dir is not None:
                await self._remove_session_dir_best_effort(session_dir, session_id)
            raise

    async def cancel(self, session_id: str) -> bool:
        """Accept cancellation quickly and finalize registry cleanup in background."""
        episode_admission = getattr(self, "episode_admission", None)
        if episode_admission is not None:
            await episode_admission.cancel_waiters(session_id)
        async with self._cancel_lock:
            existing = self._cancel_finalizers.get(session_id)
            if existing is not None:
                return True
            done_event = await self._dispatcher.cancel(session_id)
            if done_event is None:
                return False
            task = asyncio.create_task(
                self._finalize_cancelled_session(session_id, done_event),
                name=f"polar-session-cancel-finalize-{session_id}",
            )
            self._cancel_finalizers[session_id] = task
            task.add_done_callback(
                lambda completed, sid=session_id: self._forget_cancel_finalizer(sid, completed)
            )
            return True

    async def _finalize_cancelled_session(
        self,
        session_id: str,
        done_event: asyncio.Event,
    ) -> None:
        await done_event.wait()
        self.storage.delete_session(session_id)
        self.session_registry.remove(session_id)

    def _forget_cancel_finalizer(
        self,
        session_id: str,
        completed: asyncio.Task[None],
    ) -> None:
        if self._cancel_finalizers.get(session_id) is completed:
            self._cancel_finalizers.pop(session_id, None)

    async def active_sessions(self) -> int:
        return await self._dispatcher.active_count()

    async def stage_metrics(self) -> NodeStageMetrics:
        snapshot = await self._dispatcher.snapshot()
        return self._snapshot_to_metrics(snapshot)

    def episode_admission_health(self) -> dict[str, object]:
        retained = sorted(self._fatal_retained_episode_sessions)
        return {
            "healthy": not retained,
            "fatal_retained": bool(retained),
            "retained_session_ids": retained,
        }

    async def _mark_fatal_retained_episode(
        self,
        session_id: str,
        *,
        reason: str = "active model-pool episode lease was retained",
    ) -> None:
        self._fatal_retained_episode_sessions.add(session_id)
        await self.episode_admission.poison(
            f"fatal runtime containment failure for session {session_id}: {reason}"
        )
        logger.critical(
            "Gateway node %s is unhealthy after an unproven runtime teardown "
            "for session %s (%s)",
            self.node_id,
            session_id,
            reason,
        )

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
            managed.timer.mark("runtime_validation", "started")
            try:
                await self._await_with_budget(runtime.start, managed)
            finally:
                managed.timer.mark("runtime_validation", "finished")
            # Run ordered prepare actions
            managed.timer.mark("prepare", "started")
            try:
                await self._run_runtime_prepare(runtime, runtime_spec, request, managed)
            finally:
                managed.timer.mark("prepare", "finished")
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
                result = None
                for attempt in range(1, action.max_attempts + 1):
                    result = await runtime.exec(
                        action.command,
                        cwd=effective_cwd,
                        env=merged_env,
                        timeout_sec=self._remaining_budget(managed),
                    )
                    log_dir = managed.session_dir / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    attempt_suffix = "" if action.max_attempts == 1 else f".attempt-{attempt:02d}"
                    self._write_exec_log(
                        log_dir,
                        f"{log_prefix}.{i:02d}{attempt_suffix}",
                        result.stdout,
                        result.stderr,
                    )
                    if result.return_code in (0, -1):
                        break
                    if attempt < action.max_attempts:
                        logger.warning(
                            "%s action %d failed on attempt %d/%d with exit code %d; retrying",
                            log_prefix,
                            i,
                            attempt,
                            action.max_attempts,
                            result.return_code,
                        )
                        delay = min(
                            action.retry_backoff_seconds * (2 ** (attempt - 1)),
                            5.0,
                        )
                        if delay:
                            await asyncio.sleep(delay)
                assert result is not None
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
        self._start_agent_deadline(managed)

        harness: BaseHarness | None = None
        timeout_stage = "setup"
        try:
            runtime = managed.runtime
            if runtime is None:
                raise RuntimeError("runtime is required for execution")

            self._start_eval_prewarm(managed)
            harness = self._resolve_agent_harness(request)

            # Setup
            managed.timer.mark("agent_setup", "started")
            try:
                await self._await_with_agent_budget(lambda: harness.setup(runtime), managed)
            finally:
                managed.timer.mark("agent_setup", "finished")

            # Run
            timeout_stage = "exec"
            steps = harness.run_steps(request.instruction)
            env = self._runtime_env(request, managed, include_agent_env=True)
            managed.timer.mark("agent_exec", "started")
            try:
                agent_result = await self._run_exec_inputs(runtime, steps, env, managed)
            finally:
                managed.timer.mark("agent_exec", "finished")
            managed.agent_result = agent_result
            if managed.cancel_requested:
                return

            # Attempt postprocess while budget remains so harnesses can collect
            # artifacts from failed agent runs before post-run evaluation.
            postprocess_started = False

            def start_postprocess():
                # `_await_with_agent_budget` validates the remaining budget
                # before it invokes this factory. Keep the stage as `exec`
                # when the agent consumed its budget before postprocess could
                # start; that is an aligned policy timeout, not a postprocess
                # failure. Once invoked, a later timeout is a genuine
                # postprocess timeout and remains fail-closed.
                nonlocal postprocess_started, timeout_stage
                timeout_stage = "postprocess"
                postprocess_started = True
                managed.timer.mark("agent_postprocess", "started")
                return harness.postprocess(runtime, agent_result)

            try:
                await self._await_with_agent_budget(start_postprocess, managed)
            finally:
                if postprocess_started:
                    managed.timer.mark("agent_postprocess", "finished")
        except GatewayExecutionTimeout as exc:
            # Don't set final_result — let _handle_postrun build a partial
            # trajectory from the completions captured so far.
            managed.agent_result = AgentRunResult(
                status="timeout",
                return_code=-1,
                error=str(exc),
                metadata={
                    "timeout_source": self._timeout_source_from_error(exc),
                    "timeout_stage": timeout_stage,
                },
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
                return AgentRunResult(status="failed", return_code=-1, error="cancelled")
            merged_env = {**env, **(step.env or {})}
            if step.protected_argv is not None:
                protected_env: dict[str, str] = {}
                for key in step.protected_env_keys:
                    value = merged_env.get(key, "")
                    if not value:
                        raise RuntimeError(
                            f"protected runtime credential {key} is unavailable"
                        )
                    protected_env[key] = value
                # Never let a privileged SPilot capability fall back into the
                # ordinary child environment.  The per-lease pool-call token
                # is created later by admission and is not session-scoped.
                for key in (
                    ROUTER_CAPABILITY_ENV,
                    MODEL_POOL_CAPABILITY_ENV,
                    MODEL_POOL_ADMISSION_CAPABILITY_ENV,
                ):
                    merged_env.pop(key, None)
                result = await runtime.exec_protected(
                    step.protected_argv,
                    cwd=step.cwd,
                    env=merged_env,
                    protected_env=protected_env,
                    protected_file_digests=step.protected_file_digests,
                    timeout_sec=self._remaining_agent_budget(managed),
                )
            else:
                assert step.command is not None
                result = await runtime.exec(
                    step.command,
                    cwd=step.cwd,
                    env=merged_env,
                    timeout_sec=self._remaining_agent_budget(managed),
                )
            self._write_exec_log(log_dir, f"step.{i:02d}", result.stdout, result.stderr)
            if result.return_code == -1:
                metadata = self._step_metadata(log_dir, i, managed)
                metadata["timeout_source"] = self._active_timeout_source(managed)
                metadata["timeout_stage"] = "exec"
                return AgentRunResult(
                    status="timeout",
                    return_code=-1,
                    error=f"step {i} timed out",
                    metadata=metadata,
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
        managed.eval_prewarm_task = asyncio.create_task(self._prepare_eval_runtime(managed))

    async def _prepare_eval_runtime(self, managed: ManagedSession) -> BaseRuntime | None:
        """Create and prepare a fresh runtime for the evaluator. Returns None on failure."""
        request = managed.request
        runtime_spec = self._resolve_runtime_spec(request)
        eval_session_dir = managed.session_dir / "eval_runtime"
        eval_artifacts_dir = eval_session_dir / "artifacts"
        eval_artifacts_dir.mkdir(parents=True, exist_ok=True)

        eval_runtime = create_runtime(runtime_spec, f"{request.session_id}-eval", eval_session_dir)
        # Publish ownership before the first await. Dispatcher shutdown can
        # now always discover and retry teardown, including when prewarm is
        # cancelled during start() or completes before RUN consumes its task.
        managed.eval_runtime = eval_runtime
        try:
            managed.timer.mark("eval_runtime_validation", "started")
            try:
                await self._await_with_budget(eval_runtime.start, managed)
            finally:
                managed.timer.mark("eval_runtime_validation", "finished")
            eval_actions = (
                runtime_spec.eval_prepare
                if runtime_spec.eval_prepare is not None
                else runtime_spec.prepare
            )
            managed.timer.mark("eval_prepare", "started")
            try:
                await self._run_runtime_prepare(
                    eval_runtime,
                    runtime_spec,
                    request,
                    managed,
                    actions=eval_actions,
                    log_prefix="eval_prepare",
                )
            finally:
                managed.timer.mark("eval_prepare", "finished")
            return eval_runtime
        except asyncio.CancelledError:
            try:
                await asyncio.shield(eval_runtime.stop())
                if not eval_runtime.destroyed:
                    raise RuntimeContainmentError(
                        "eval runtime stop returned without destruction proof"
                    )
            except BaseException as stop_exc:
                managed.runtime_cancel_error = stop_exc
            managed.timer.add_runtime_exec_summary(eval_runtime.exec_timing_summary())
            raise
        except Exception as exc:
            logger.warning(
                "Eval runtime prewarm failed for session %s: %s",
                request.session_id,
                exc,
            )
            try:
                await asyncio.shield(eval_runtime.stop())
                if not eval_runtime.destroyed:
                    raise RuntimeContainmentError(
                        "eval runtime stop returned without destruction proof"
                    )
            except BaseException as stop_exc:
                managed.runtime_cancel_error = stop_exc
            managed.timer.add_runtime_exec_summary(eval_runtime.exec_timing_summary())
            return None

    async def _acquire_prepared_eval_runtime(self, managed: ManagedSession) -> BaseRuntime | None:
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

    async def _drain_eval_prewarm_task(self, managed: ManagedSession) -> BaseRuntime | None:
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
        fatal_retained_episode = False
        managed.timer.mark("postrun", "started")
        try:
            if result is None:
                if managed.cancel_requested:
                    result = self._cancelled_result(request, managed.timer)
                else:
                    result = await self._build_session_result(managed)
        except GatewayExecutionTimeout as exc:
            result = self._timeout_result(request, managed.timer, str(exc))
        except GatewayExecutionCancelled:
            result = self._cancelled_result(request, managed.timer)
        except Exception as exc:
            logger.exception("Post-run handling failed for session %s", request.session_id)
            result = self._error_result(request, managed.timer, f"post-run failed: {exc}")
        finally:
            managed.timer.mark("postrun", "finished")
            if managed.runtime is not None:
                # Read the bind-mounted JSONL directly. This deliberately does
                # not consume the exhausted execution budget, so failed and
                # timed-out agent actions retain their most useful telemetry.
                managed.timer.add_mini_swe_command_summary(
                    load_mini_swe_command_timing(managed.runtime)
                )
            managed.timer.mark("teardown", "started")
            managed.timer.mark("postrun_exec", "started")
            try:
                if not managed.cancel_requested:
                    await self._run_postrun_steps(managed)
            finally:
                managed.timer.mark("postrun_exec", "finished")

            runtimes: list[BaseRuntime] = []
            agent_runtime_destroyed = managed.runtime is None
            all_runtimes_destroyed = True
            managed.timer.mark("runtime_stop", "started")
            try:
                stop_tasks = []
                eval_runtime = await self._drain_eval_prewarm_task(managed)
                if eval_runtime is None:
                    # A cancelled/failed prewarm may have retained a runtime
                    # whose first cleanup attempt failed. Its published owner
                    # remains authoritative for this final retry.
                    eval_runtime = managed.eval_runtime
                if eval_runtime is not None:
                    runtimes.append(eval_runtime)
                    stop_tasks.append(
                        self._stop_runtime_best_effort(
                            eval_runtime, request.session_id, "eval runtime"
                        )
                    )
                if managed.runtime is not None:
                    runtimes.append(managed.runtime)
                    stop_tasks.append(
                        self._stop_runtime_best_effort(
                            managed.runtime, request.session_id, "runtime"
                        )
                    )
                if stop_tasks:
                    stop_results = await asyncio.gather(
                        *stop_tasks, return_exceptions=True
                    )
                    if not all(result is True for result in stop_results):
                        all_runtimes_destroyed = False
                        logger.error(
                            "One or more runtimes did not report a clean stop for "
                            "session %s",
                            request.session_id,
                        )
                    if managed.runtime is not None:
                        agent_runtime_destroyed = stop_results[-1] is True
            finally:
                managed.timer.mark("runtime_stop", "finished")
                for runtime in runtimes:
                    managed.timer.add_runtime_exec_summary(runtime.exec_timing_summary())
                # Runtime.stop() only tears down the outer sandbox.  It is not
                # proof that every descendant in the candidate process scope
                # disappeared, so postrun must never reclaim an active lease.
                # A normal runner explicitly releases only after its Linux
                # child-subreaper/procfs proof shows the scope is empty.
                # Anything still active here is a
                # fatal retained lease and makes this node unschedulable until
                # gateway shutdown.
                episode_admission = getattr(self, "episode_admission", None)
                if episode_admission is not None:
                    if (
                        request.agent.harness == "spilot_router"
                        and not all_runtimes_destroyed
                    ):
                        await self._mark_fatal_retained_episode(
                            request.session_id,
                            reason="runtime destruction could not be proven",
                        )
                        fatal_retained_episode = True
                    active_retained = await episode_admission.cleanup_session_records(
                        request.session_id
                    )
                    if active_retained and agent_runtime_destroyed:
                        try:
                            await asyncio.wait_for(
                                episode_admission.release_after_runtime_destroyed(
                                    request.session_id,
                                    runtime_destroyed=True,
                                ),
                                timeout=30.0,
                            )
                            await episode_admission.cleanup_session_records(
                                request.session_id
                            )
                            active_retained = False
                        except Exception:
                            logger.exception(
                                "Failed to drain retained episode lease after proven "
                                "runtime destruction for session %s",
                                request.session_id,
                            )
                    if active_retained:
                        await self._mark_fatal_retained_episode(request.session_id)
                        fatal_retained_episode = True
            managed.timer.mark("teardown", "finished")
            managed.timer.mark("return", "finished")

        # DELETE can arrive after evaluator completion but while teardown is
        # awaiting runtime cleanup. The scheduler has already discarded that
        # session, so never publish the previously built trainable payload.
        if managed.cancel_event.is_set():
            result = self._cancelled_result(request, managed.timer)
        if result is None:
            result = self._error_result(
                request,
                managed.timer,
                "post-run finished without producing a session result",
            )
        if fatal_retained_episode:
            result = self._attach_fatal_retained_episode_metadata(result)
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
            if fatal_retained_episode:
                logger.critical(
                    "Preserving session directory after fatal containment failure: %s",
                    managed.session_dir,
                )
            else:
                await self._remove_session_dir_best_effort(
                    managed.session_dir, request.session_id
                )

    @staticmethod
    def _attach_fatal_retained_episode_metadata(result: SessionResult) -> SessionResult:
        """Expose fatal telemetry and make the result impossible to train."""

        trajectory_metadata = dict(result.trajectory.metadata)
        evaluation = trajectory_metadata.get("evaluation")
        evaluation = dict(evaluation) if isinstance(evaluation, dict) else {}
        router_metadata = evaluation.get("spilot_router")
        router_metadata = (
            dict(router_metadata) if isinstance(router_metadata, dict) else {}
        )
        router_metadata.update(
            {
                "admission_enabled": True,
                "admission_fatal_retained": True,
                "admission_node_healthy": False,
                "runtime_containment_proven": False,
            }
        )
        evaluation["spilot_router"] = router_metadata
        trajectory_metadata["evaluation"] = evaluation
        filtered_traces: list[Trace] = []
        for trace in result.trajectory.traces:
            trace_metadata = dict(trace.metadata)
            current_filter = trace_metadata.get("training_filter")
            training_filter = (
                dict(current_filter) if isinstance(current_filter, dict) else {}
            )
            training_filter.update(
                {
                    "masked": True,
                    "trainable": False,
                    "reason": "runtime_containment_unproven",
                    "detail": "runtime destruction could not be proven",
                }
            )
            training_filter.setdefault("original_reward", trace.reward)
            trace_metadata["training_filter"] = training_filter
            filtered_traces.append(
                trace.model_copy(
                    update={
                        "reward": 0.0,
                        "loss_mask": [0] * len(trace.response_ids),
                        "metadata": trace_metadata,
                    }
                )
            )
        trajectory = result.trajectory.model_copy(
            update={
                "status": "ERROR",
                "error": "runtime containment could not be proven",
                "traces": filtered_traces,
                "metadata": trajectory_metadata,
            }
        )
        result_metadata = dict(result.metadata)
        result_metadata["model_pool_episode_admission_fatal_retained"] = True
        result_metadata["runtime_containment_proven"] = False
        return result.model_copy(
            update={
                "status": "ERROR",
                "error": "runtime containment could not be proven",
                "trajectory": trajectory,
                "metadata": result_metadata,
            }
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
                lambda: asyncio.to_thread(self._build_trajectory, request),
                managed,
            )
        finally:
            managed.timer.mark("build", "finished")

        trajectory = self._attach_agent_result_metadata(trajectory, agent_result)
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
        except GatewayExecutionCancelled:
            raise
        except GatewayExecutionTimeout as exc:
            # Preserve the built trajectory even when eval times out.
            logger.warning("Eval timed out for session %s: %s", request.session_id, exc)
            evaluation = trajectory.metadata.get("evaluation")
            evaluation_metadata = dict(evaluation) if isinstance(evaluation, dict) else {}
            evaluation_metadata.update(
                {
                    "verifier_timeout": True,
                    "verifier_timeout_error": str(exc),
                }
            )
            update: dict[str, Any] = {
                "metadata": {
                    **trajectory.metadata,
                    "evaluation": evaluation_metadata,
                }
            }
            if trajectory.status not in ("TIMEOUT", "ERROR"):
                update.update({"status": "TIMEOUT", "error": f"eval timed out: {exc}"})
            trajectory = trajectory.model_copy(update=update)
        except Exception as exc:
            logger.exception("Eval failed for session %s", request.session_id)
            trajectory = trajectory.model_copy(
                update={"status": "ERROR", "error": f"evaluator failed: {exc}"}
            )
        finally:
            managed.timer.mark("eval", "finished")

        # Evaluation is deliberately allowed to run after a failed agent so
        # its verifier output remains available for diagnosis. Agent-budget
        # timeouts are aligned sampled policy actions and remain trainable as
        # explicit zero-reward negatives; evaluator, infrastructure, and ERROR
        # outcomes remain diagnostic-only. Enforce the distinction at the
        # persisted gateway boundary and again in the trainer adapter.
        trajectory = self._mask_noncompleted_trajectory(trajectory)
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
        if managed.cancel_event.is_set():
            raise GatewayExecutionCancelled("session cancelled during evaluation")

        live_runtime = managed.runtime
        if live_runtime is None:
            raise RuntimeError("runtime is required for evaluation")

        fresh_eval_runtime: BaseRuntime | None = None
        if evaluator_spec.refresh_runtime:
            fresh_eval_runtime = await self._await_eval_with_budget_or_cancel(
                self._acquire_prepared_eval_runtime(managed),
                managed,
            )
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
            eval_result = await self._await_eval_with_budget_or_cancel(
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
        except GatewayExecutionCancelled:
            raise
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
            # Broadcast trajectory-level reward.
            traces = [
                trace.model_copy(update={"reward": eval_result.outcome_reward}) for trace in traces
            ]

        # A successful session can contain abandoned retry chains. Never let a
        # terminal outcome positively reinforce a chain whose attempted tool
        # call could not be parsed. This is still a sampled policy action: if
        # token/log-prob alignment is intact, keep its loss mask so reward=0
        # gives it a negative centered advantage when sibling rollouts solve.
        filtered_traces = []
        parser_invalid_traces_zero_rewarded = 0
        for trace in traces:
            filtered_trace, invalid_reason = zero_reward_parser_invalid_tool_call_trace(trace)
            filtered_traces.append(filtered_trace)
            if invalid_reason is not None:
                parser_invalid_traces_zero_rewarded += 1
        traces = filtered_traces

        eval_metadata = {
            "strategy": evaluator_spec.strategy,
            "outcome_reward": eval_result.outcome_reward,
            "trace_rewards": eval_result.trace_rewards,
            **eval_result.metadata,
        }
        if parser_invalid_traces_zero_rewarded:
            eval_metadata["parser_invalid_traces_zero_rewarded"] = (
                parser_invalid_traces_zero_rewarded
            )
        metadata = {**trajectory.metadata, "evaluation": eval_metadata}
        merged = trajectory.model_copy(update={"traces": traces, "metadata": metadata})
        return GatewayNodeManager._mask_noncompleted_trajectory(merged)

    @staticmethod
    def _mask_noncompleted_trajectory(trajectory: Trajectory) -> Trajectory:
        """Fail-close non-completed trajectories at the gateway boundary.

        Evaluators may still inspect a failed agent's final container state.
        A timeout caused specifically by the model agent exhausting its own
        budget is a valid sampled policy failure: preserve aligned masks and
        old-policy logprobs but force its effective reward to zero. All other
        TIMEOUT/ERROR outcomes remain diagnostic-only and fully masked.
        """

        if trajectory.status == "COMPLETED":
            return trajectory

        trainable_agent_timeout = GatewayNodeManager._is_trainable_agent_timeout(trajectory)
        filter_reason = (
            _TRAINABLE_AGENT_TIMEOUT_REASON
            if trainable_agent_timeout
            else "session_timeout"
            if trajectory.status == "TIMEOUT"
            else "session_error"
        )
        filtered_traces: list[Trace] = []
        for trace in trajectory.traces:
            trace_metadata = dict(trace.metadata)
            current_filter = trace_metadata.get("training_filter")
            training_filter = dict(current_filter) if isinstance(current_filter, dict) else {}
            aligned_agent_timeout = (
                trainable_agent_timeout and GatewayNodeManager._has_aligned_policy_data(trace)
            )
            filter_update: dict[str, Any] = {
                "masked": not aligned_agent_timeout,
                "reason": (
                    filter_reason
                    if aligned_agent_timeout or not trainable_agent_timeout
                    else "agent_timeout_unaligned"
                ),
                "detail": trajectory.error or trajectory.status,
            }
            if trainable_agent_timeout:
                filter_update["trainable"] = aligned_agent_timeout
            training_filter.update(filter_update)
            training_filter.setdefault("original_reward", trace.reward)
            trace_metadata["training_filter"] = training_filter
            updates: dict[str, Any] = {
                "reward": 0.0,
                "metadata": trace_metadata,
            }
            if not aligned_agent_timeout:
                updates["loss_mask"] = [0] * len(trace.response_ids)
            filtered_traces.append(trace.model_copy(update=updates))

        metadata = dict(trajectory.metadata)
        evaluation = metadata.get("evaluation")
        if isinstance(evaluation, dict):
            evaluation = dict(evaluation)
            outcome_reward = evaluation.get("outcome_reward")
            trace_rewards = evaluation.get("trace_rewards")
            if outcome_reward is not None:
                evaluation.setdefault("discarded_outcome_reward", outcome_reward)
            if trace_rewards is not None:
                evaluation.setdefault("discarded_trace_rewards", trace_rewards)
            # ``reward`` and ``resolved`` are common evaluator convenience
            # fields (including Harbor).  Keep verifier_reported_reward intact
            # as the raw diagnostic while making these effective fields agree
            # with the terminal session status.
            if evaluation.get("reward") is not None:
                evaluation.setdefault("discarded_reward", evaluation["reward"])
                evaluation["reward"] = 0.0
            if evaluation.get("resolved") is not None:
                evaluation["resolved"] = False
            evaluation["outcome_reward"] = 0.0
            if isinstance(trace_rewards, list):
                evaluation["trace_rewards"] = [0.0] * len(trace_rewards)
            evaluation["reward_discarded"] = True
            evaluation["reward_discard_reason"] = filter_reason
            metadata["evaluation"] = evaluation

        return trajectory.model_copy(update={"traces": filtered_traces, "metadata": metadata})

    @staticmethod
    def _attach_agent_result_metadata(
        trajectory: Trajectory,
        agent_result: AgentRunResult,
    ) -> Trajectory:
        """Persist the trusted gateway classification of the agent lifecycle."""

        metadata = dict(trajectory.metadata)
        agent_metadata: dict[str, Any] = {
            "status": agent_result.status,
            "return_code": agent_result.return_code,
        }
        if agent_result.error:
            agent_metadata["error"] = agent_result.error
        timeout_source = agent_result.metadata.get("timeout_source")
        if timeout_source in {"agent", "session"}:
            agent_metadata["timeout_source"] = timeout_source
        timeout_stage = agent_result.metadata.get("timeout_stage")
        if timeout_stage in {"setup", "exec", "postprocess"}:
            agent_metadata["timeout_stage"] = timeout_stage
        metadata[_AGENT_RESULT_METADATA_KEY] = agent_metadata
        return trajectory.model_copy(update={"metadata": metadata})

    @staticmethod
    def _is_trainable_agent_timeout(trajectory: Trajectory) -> bool:
        if trajectory.status != "TIMEOUT":
            return False
        agent_result = trajectory.metadata.get(_AGENT_RESULT_METADATA_KEY)
        if not isinstance(agent_result, dict):
            return False
        if agent_result.get("status") != "timeout":
            return False
        if agent_result.get("timeout_source") != "agent":
            return False
        if agent_result.get("timeout_stage") != "exec":
            return False
        evaluation = trajectory.metadata.get("evaluation")
        return not (isinstance(evaluation, dict) and evaluation.get("verifier_timeout") is True)

    @staticmethod
    def _has_aligned_policy_data(trace: Trace) -> bool:
        return (
            bool(trace.response_ids)
            and len(trace.loss_mask) == len(trace.response_ids)
            and any(trace.loss_mask)
            and trace.response_logprobs is not None
            and len(trace.response_logprobs) == len(trace.response_ids)
            and all(math.isfinite(value) for value in trace.response_logprobs)
        )

    @staticmethod
    def _timeout_source_from_error(error: BaseException) -> str:
        return "agent" if str(error) == "agent execution timeout" else "session"

    @staticmethod
    def _active_timeout_source(managed: ManagedSession) -> str:
        agent_deadline = managed.agent_deadline
        session_deadline = managed.execution_deadline
        if agent_deadline is not None and (
            session_deadline is None or agent_deadline < session_deadline
        ):
            return "agent"
        return "session"

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
        environment = {
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
        # These host-generated capabilities are deliberately added after
        # caller-controlled runtime/agent env.  They exist only for the SPilot
        # harness and never reuse the externally visible session id.
        router_capability = getattr(managed, "router_capability", None)
        model_pool_admission_capability = getattr(
            managed, "model_pool_admission_capability", None
        )
        if getattr(managed, "stage", None) == SessionStage.RUNNING:
            if router_capability:
                environment[ROUTER_CAPABILITY_ENV] = router_capability
            if model_pool_admission_capability:
                environment[MODEL_POOL_ADMISSION_CAPABILITY_ENV] = (
                    model_pool_admission_capability
                )
        if runtime is not None:
            # Keep RuntimeSpec.env strictly string-valued while still exposing
            # the authoritative policy to an injected network helper. Put this
            # after both runtime and agent env so an offline task cannot regain
            # the job's HTTP proxy by overriding the marker in its payload.
            environment["POLAR_ALLOW_INTERNET"] = (
                "true" if runtime.spec.allow_internet else "false"
            )
        return environment

    @staticmethod
    def _write_exec_log(
        log_dir: Path, prefix: str, stdout: str | None, stderr: str | None
    ) -> None:
        # A session can be cancelled and cleaned while an in-flight runtime
        # command is returning. Recreate the leaf on the write side so log
        # persistence does not turn that otherwise-benign race into a session
        # execution failure.
        log_dir.mkdir(parents=True, exist_ok=True)
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

    def _cancelled_result(
        self, request: SessionDispatchRequest, timer: StageTimer
    ) -> SessionResult:
        return self._error_result(request, timer, "session cancelled")

    async def _push_result(self, callback_url: str | None, result: SessionResult) -> bool:
        """POST the terminal result to the rollout server. Return True on success."""
        if not callback_url:
            return False
        payload = result.model_dump(mode="json")
        for attempt in range(1, _CALLBACK_MAX_ATTEMPTS + 1):
            retryable = False
            last_error: Exception | None = None
            try:
                response = await self._client.post(
                    callback_url,
                    json=payload,
                    timeout=_CALLBACK_REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                return True
            except asyncio.CancelledError:
                raise
            except httpx.HTTPStatusError as exc:
                retryable = exc.response.status_code in _CALLBACK_RETRYABLE_STATUS_CODES
                last_error = exc
            except httpx.TransportError as exc:
                # A dropped response can mean the rollout server accepted the
                # first POST but its acknowledgement was lost. The callback
                # endpoint is idempotent for a terminal session id, so retrying
                # is safe and lets the gateway release its retained payload.
                retryable = True
                last_error = exc
            except Exception as exc:
                last_error = exc

            if retryable and attempt < _CALLBACK_MAX_ATTEMPTS:
                logger.debug(
                    "Callback attempt %d/%d failed for session %s (%s); retrying",
                    attempt,
                    _CALLBACK_MAX_ATTEMPTS,
                    result.session_id,
                    last_error,
                )
                await asyncio.sleep(_CALLBACK_RETRY_BACKOFF_SECONDS * attempt)
                continue

            logger.warning(
                "Failed to deliver callback for session %s to %s after %d attempt(s): %s",
                result.session_id,
                callback_url,
                attempt,
                last_error,
            )
            return False
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
        awaitable_factory,
        managed: ManagedSession,
    ):
        # Check the budget before constructing the coroutine.  Callers used to
        # pass an already-created coroutine here; when the budget had expired,
        # _remaining_budget raised before wait_for could schedule (or close) it,
        # producing "coroutine was never awaited" warnings at timeout boundaries.
        timeout = self._remaining_budget(managed)
        try:
            return await asyncio.wait_for(
                awaitable_factory(),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise GatewayExecutionTimeout("session execution timeout") from exc

    def _remaining_agent_budget(self, managed: ManagedSession) -> float:
        """Return the smaller active-agent and total-session budget."""

        total_remaining = self._remaining_budget(managed)
        deadline = managed.agent_deadline
        if deadline is None:
            return total_remaining
        agent_remaining = deadline - asyncio.get_running_loop().time()
        if agent_remaining <= 0:
            raise GatewayExecutionTimeout("agent execution timeout")
        return min(total_remaining, agent_remaining)

    async def _await_with_agent_budget(
        self,
        awaitable_factory,
        managed: ManagedSession,
    ):
        timeout = self._remaining_agent_budget(managed)
        total_deadline = managed.execution_deadline
        agent_deadline = managed.agent_deadline
        try:
            return await asyncio.wait_for(awaitable_factory(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            # If both expire together, preserve the stronger total-session cap.
            if total_deadline is not None and (
                agent_deadline is None or total_deadline <= agent_deadline
            ):
                raise GatewayExecutionTimeout("session execution timeout") from exc
            raise GatewayExecutionTimeout("agent execution timeout") from exc

    async def _await_eval_with_budget_or_cancel(
        self,
        awaitable,
        managed: ManagedSession,
    ):
        """Await evaluator work until completion, deadline, or session DELETE."""

        operation = asyncio.ensure_future(awaitable)
        cancel_wait = asyncio.create_task(managed.cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {operation, cancel_wait},
                timeout=self._remaining_budget(managed),
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Cancellation wins ties. A terminal result delivered after the
            # scheduler discarded this session must never regain trainability.
            if cancel_wait in done and managed.cancel_event.is_set():
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
                raise GatewayExecutionCancelled("session cancelled during evaluation")
            if operation in done:
                return operation.result()

            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            raise GatewayExecutionTimeout("session execution timeout")
        except BaseException:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            raise
        finally:
            cancel_wait.cancel()
            await asyncio.gather(cancel_wait, return_exceptions=True)

    @staticmethod
    def _start_execution_deadline(managed: ManagedSession) -> None:
        if managed.execution_deadline is not None:
            return
        managed.execution_deadline = (
            asyncio.get_running_loop().time() + managed.request.remaining_timeout_seconds
        )

    @staticmethod
    def _start_agent_deadline(managed: ManagedSession) -> None:
        if managed.agent_deadline is not None:
            return
        value = managed.request.metadata.get("agent_timeout")
        if value is None:
            return
        # SessionDispatchRequest validates this trusted metadata at the API
        # boundary. Keep the conversion local so the deadline uses monotonic
        # time and begins only after the READY queue grants a RUN worker.
        managed.agent_deadline = asyncio.get_running_loop().time() + float(value)

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
    ) -> bool:
        try:
            await runtime.stop()
            if not runtime.destroyed:
                raise RuntimeContainmentError(
                    f"{label} stop returned without destruction proof"
                )
            return True
        except Exception:
            logger.warning(
                "Failed to stop %s for session %s",
                label,
                session_id,
                exc_info=True,
            )
            return False

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
