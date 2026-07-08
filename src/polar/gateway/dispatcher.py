"""Stage-isolated session dispatcher for gateway execution.

Drives four stages — INIT, READY, RUNNING, POSTRUN — each with an isolated
worker pool. Queued vs. executing within a stage is the `inflight` bool on
`ManagedSession`, so the stage enum stays small. Eval-runtime prewarm is an
ad-hoc background task owned by the session handler, not a dedicated stage.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable

from polar.agent.models import AgentRunResult
from polar.rollout.models import SessionDispatchRequest, SessionResult
from polar.rollout.timer import StageTimer
from polar.runtime.base import BaseRuntime, RuntimeContainmentError
from polar.runtime.models import ExecInput

logger = logging.getLogger(__name__)

StageCallback = Callable[["ManagedSession"], Awaitable[None]]
StageTransitionCallback = Callable[["ManagedSession"], None]
_STOP = object()
_SHUTDOWN_RUNTIME_TIMEOUT_SECONDS = 60.0


class SessionStage(str, Enum):
    INIT = "INIT"
    READY = "READY"
    RUNNING = "RUNNING"
    POSTRUN = "POSTRUN"


@dataclass(slots=True)
class DispatcherSnapshot:
    init_queue_depth: int = 0
    init_inflight: int = 0
    ready_depth: int = 0
    run_inflight: int = 0
    postrun_queue_depth: int = 0
    postrun_inflight: int = 0

    @property
    def active_count(self) -> int:
        return (
            self.init_queue_depth
            + self.init_inflight
            + self.ready_depth
            + self.run_inflight
            + self.postrun_queue_depth
            + self.postrun_inflight
        )


@dataclass(slots=True)
class ManagedSession:
    """Per-session state flowing through the gateway dispatcher."""

    request: SessionDispatchRequest
    timer: StageTimer
    session_dir: Path
    artifacts_dir: Path
    router_capability: str | None = None
    model_pool_capability: str | None = None
    model_pool_admission_capability: str | None = None
    runtime: BaseRuntime | None = None
    agent_result: AgentRunResult | None = None
    final_result: SessionResult | None = None
    postrun_steps: list[ExecInput] = field(default_factory=list)
    eval_prewarm_task: asyncio.Task | None = None
    eval_runtime: BaseRuntime | None = None
    runtime_cancel_task: asyncio.Task[None] | None = None
    runtime_cancel_error: BaseException | None = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    cancel_requested: bool = False
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    execution_deadline: float | None = None
    # Starts only when a RUN worker begins active agent work. READY queue time
    # still consumes the total execution deadline, but never this model budget.
    agent_deadline: float | None = None
    stage: SessionStage = SessionStage.INIT
    inflight: bool = False

    @property
    def session_id(self) -> str:
        return self.request.session_id


class SessionDispatcher:
    """Drive INIT -> READY -> RUNNING -> POSTRUN with isolated worker pools."""

    def __init__(
        self,
        *,
        max_init_workers: int,
        max_run_workers: int,
        max_postrun_workers: int,
    ) -> None:
        if max_init_workers < 1 or max_run_workers < 1 or max_postrun_workers < 1:
            raise ValueError("all stage worker counts must be at least 1")
        self.max_init_workers = max_init_workers
        self.max_run_workers = max_run_workers
        self.max_postrun_workers = max_postrun_workers
        self.on_init: StageCallback | None = None
        self.on_run: StageCallback | None = None
        self.on_postrun: StageCallback | None = None
        self.on_stage_change: StageTransitionCallback | None = None
        # INIT is the only stage where a short fixed evaluation can otherwise
        # sit behind an already-filled fully-async training window. Use a
        # stable priority queue here: higher request priority wins, while FIFO
        # ordering is preserved within the same priority. Inflight work is
        # deliberately never preempted.
        self._init_queue: asyncio.PriorityQueue[tuple[int, int, str]] = asyncio.PriorityQueue()
        self._init_sequence = itertools.count()
        self._ready_queue: asyncio.Queue[str | object] = asyncio.Queue()
        self._postrun_queue: asyncio.Queue[str | object] = asyncio.Queue()
        self._ready_slots = asyncio.Semaphore(max_run_workers)
        self._sessions: dict[str, ManagedSession] = {}
        self._lock = asyncio.Lock()
        self._workers: list[asyncio.Task[None]] = []
        self._runtime_cancel_tasks: set[asyncio.Task[None]] = set()
        self._started = False
        self._stopping = False

    async def start(self) -> None:
        if self._started:
            return
        self._stopping = False
        self._workers = [
            *(asyncio.create_task(self._init_worker()) for _ in range(self.max_init_workers)),
            *(asyncio.create_task(self._run_worker()) for _ in range(self.max_run_workers)),
            *(
                asyncio.create_task(self._postrun_worker())
                for _ in range(self.max_postrun_workers)
            ),
        ]
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        loop = asyncio.get_running_loop()
        shutdown_deadline = loop.time() + _SHUTDOWN_RUNTIME_TIMEOUT_SECONDS
        async with self._lock:
            self._stopping = True
            self._started = False
            sessions_by_id = dict(self._sessions)
            for managed in sessions_by_id.values():
                managed.cancel_requested = True
                managed.cancel_event.set()

        async def drain_tasks(tasks: list[asyncio.Task], *, label: str) -> bool:
            if not tasks:
                return True
            remaining = shutdown_deadline - loop.time()
            if remaining <= 0:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                return False
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=remaining,
                )
                return True
            except TimeoutError:
                logger.error("Dispatcher shutdown timed out draining %s", label)
                for task in tasks:
                    if not task.done():
                        task.cancel()
                done, _pending = await asyncio.wait(tasks, timeout=0.5)
                for task in done:
                    if not task.cancelled():
                        task.exception()
                return False

        # Stop stage callbacks before taking the final runtime snapshot. INIT
        # publishes managed.runtime immediately before awaiting start(), so a
        # worker cannot create a runtime after this handoff unnoticed.
        workers = list(self._workers)
        for task in workers:
            task.cancel()
        workers_drained = await drain_tasks(workers, label="stage workers")
        self._workers.clear()

        async with self._lock:
            sessions_by_id.update(self._sessions)
            self._sessions.clear()
            sessions = list(sessions_by_id.values())
            for managed in sessions:
                managed.cancel_requested = True
                managed.cancel_event.set()
                self._schedule_runtime_cancel_locked(managed)

        # Eval prewarm is not a dispatcher worker. Cancel and join it before
        # enumerating its published runtime for the final destruction proof.
        eval_tasks = [
            managed.eval_prewarm_task
            for managed in sessions
            if managed.eval_prewarm_task is not None
            and not managed.eval_prewarm_task.done()
        ]
        for task in eval_tasks:
            task.cancel()
        eval_tasks_drained = await drain_tasks(eval_tasks, label="eval prewarm tasks")

        cancel_tasks = [
            managed.runtime_cancel_task
            for managed in sessions
            if managed.runtime_cancel_task is not None
        ]
        cancel_tasks_drained = await drain_tasks(
            cancel_tasks,
            label="runtime cancellation tasks",
        )

        runtime_owners: list[tuple[ManagedSession, BaseRuntime]] = []
        seen_runtimes: set[int] = set()
        for managed in sessions:
            for runtime in (managed.runtime, managed.eval_runtime):
                if not isinstance(runtime, BaseRuntime) or id(runtime) in seen_runtimes:
                    continue
                seen_runtimes.add(id(runtime))
                runtime_owners.append((managed, runtime))

        async def retry_stop(managed: ManagedSession, runtime: BaseRuntime) -> None:
            try:
                await runtime.stop()
                if not runtime.destroyed:
                    raise RuntimeContainmentError(
                        "runtime stop returned without destruction proof"
                    )
                managed.runtime_cancel_error = None
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                managed.runtime_cancel_error = exc

        retry_tasks = [
            asyncio.create_task(retry_stop(managed, runtime))
            for managed, runtime in runtime_owners
            if not runtime.destroyed
        ]
        retry_tasks_drained = await drain_tasks(retry_tasks, label="runtime stop retries")

        teardown_failures: list[tuple[str, BaseException]] = []
        if not workers_drained:
            teardown_failures.append(
                ("stage-workers", RuntimeContainmentError("stage workers did not stop"))
            )
        if not eval_tasks_drained:
            teardown_failures.append(
                ("eval-prewarm", RuntimeContainmentError("eval prewarm did not stop"))
            )
        if not cancel_tasks_drained or not retry_tasks_drained:
            teardown_failures.append(
                (
                    "runtime-teardown",
                    RuntimeContainmentError("runtime teardown exceeded shutdown deadline"),
                )
            )
        for managed, runtime in runtime_owners:
            if runtime.destroyed:
                continue
            error = managed.runtime_cancel_error or RuntimeContainmentError(
                "runtime teardown exceeded the dispatcher shutdown deadline"
            )
            teardown_failures.append((managed.session_id, error))
        for managed in sessions:
            managed.done_event.set()
        self._stopping = False
        if teardown_failures:
            session_ids = ", ".join(session_id for session_id, _ in teardown_failures)
            raise RuntimeContainmentError(
                "dispatcher shutdown could not prove runtime destruction for: "
                f"{session_ids}"
            ) from teardown_failures[0][1]

    async def enqueue(self, managed: ManagedSession) -> None:
        if not self._started or self._stopping:
            raise RuntimeError("dispatcher has not been started")
        async with self._lock:
            if not self._started or self._stopping:
                raise RuntimeError("dispatcher is shutting down")
            if managed.session_id in self._sessions:
                raise ValueError(f"session {managed.session_id} is already enqueued")
            self._sessions[managed.session_id] = managed
        await self._init_queue.put(
            (
                -managed.request.dispatch_priority,
                next(self._init_sequence),
                managed.session_id,
            )
        )

    async def cancel(self, session_id: str) -> asyncio.Event | None:
        """Accept cancellation without waiting for a runtime process to exit.

        Runtime kill/reap runs in a tracked task.  The post-run worker waits
        for it before finalization, and ``stop`` drains every such task, so a
        fast DELETE acknowledgement never turns into an orphan subprocess.
        """
        should_enqueue_postrun = False
        async with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None:
                return None
            if not managed.cancel_requested:
                managed.cancel_requested = True
                managed.cancel_event.set()
                # If the session is parked in READY (holding a ready slot),
                # release it and transition to POSTRUN so a worker finalizes it.
                if managed.stage == SessionStage.READY and not managed.inflight:
                    self._ready_slots.release()
                    managed.stage = SessionStage.POSTRUN
                    managed.inflight = False
                    should_enqueue_postrun = True
                elif managed.stage == SessionStage.INIT and not managed.inflight:
                    managed.stage = SessionStage.POSTRUN
                    managed.inflight = False
                    should_enqueue_postrun = True
            # A repeated DELETE can arrive after the first request marked a
            # not-yet-initialized session.  Re-check runtime availability so
            # that request remains idempotent and still schedules the kill.
            self._schedule_runtime_cancel_locked(managed)
        if should_enqueue_postrun:
            self._notify_stage_change(managed)
            await self._postrun_queue.put(session_id)
        return managed.done_event

    def _schedule_runtime_cancel_locked(
        self,
        managed: ManagedSession,
    ) -> asyncio.Task[None] | None:
        existing = managed.runtime_cancel_task
        if existing is not None:
            return existing
        if managed.runtime is None:
            return None
        task = asyncio.create_task(
            self._cancel_runtime_best_effort(managed),
            name=f"polar-runtime-cancel-{managed.session_id}",
        )
        managed.runtime_cancel_task = task
        self._runtime_cancel_tasks.add(task)
        task.add_done_callback(self._runtime_cancel_tasks.discard)
        return task

    async def _cancel_runtime_best_effort(self, managed: ManagedSession) -> None:
        runtime = managed.runtime
        if runtime is None:
            return
        try:
            await runtime.cancel()
            if isinstance(runtime, BaseRuntime) and not runtime.destroyed:
                raise RuntimeContainmentError(
                    "runtime cancel returned without destruction proof"
                )
            managed.runtime_cancel_error = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            managed.runtime_cancel_error = exc
            logger.exception(
                "Failed to cancel runtime for session %s",
                managed.session_id,
            )

    async def _await_runtime_cancel(self, managed: ManagedSession) -> None:
        async with self._lock:
            task = (
                self._schedule_runtime_cancel_locked(managed) if managed.cancel_requested else None
            )
        if task is not None:
            await asyncio.shield(task)

    async def active_count(self) -> int:
        return (await self.snapshot()).active_count

    async def snapshot(self) -> DispatcherSnapshot:
        async with self._lock:
            snap = DispatcherSnapshot()
            for managed in self._sessions.values():
                if managed.stage == SessionStage.INIT:
                    if managed.inflight:
                        snap.init_inflight += 1
                    else:
                        snap.init_queue_depth += 1
                elif managed.stage == SessionStage.READY:
                    snap.ready_depth += 1
                elif managed.stage == SessionStage.RUNNING:
                    snap.run_inflight += 1
                elif managed.stage == SessionStage.POSTRUN:
                    if managed.inflight:
                        snap.postrun_inflight += 1
                    else:
                        snap.postrun_queue_depth += 1
            return snap

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    async def _init_worker(self) -> None:
        while True:
            _, _, session_id = await self._init_queue.get()
            managed = await self._begin(session_id, SessionStage.INIT)
            if managed is None:
                continue
            if not (managed.cancel_requested or managed.final_result is not None):
                await self._safe_invoke(self.on_init, managed, SessionStage.INIT)
            await self._finish_init(managed)

    async def _run_worker(self) -> None:
        while True:
            item = await self._ready_queue.get()
            if item is _STOP:
                return
            session_id = str(item)
            managed = await self._begin(session_id, SessionStage.RUNNING, from_ready=True)
            if managed is None:
                continue
            if not (managed.cancel_requested or managed.final_result is not None):
                await self._safe_invoke(self.on_run, managed, SessionStage.RUNNING)
            await self._transition_to_postrun(managed)

    async def _postrun_worker(self) -> None:
        while True:
            item = await self._postrun_queue.get()
            if item is _STOP:
                return
            session_id = str(item)
            managed = await self._begin(session_id, SessionStage.POSTRUN)
            if managed is None:
                continue
            try:
                await self._await_runtime_cancel(managed)
                await self._safe_invoke(self.on_postrun, managed, SessionStage.POSTRUN)
            finally:
                async with self._lock:
                    self._sessions.pop(session_id, None)
                managed.done_event.set()

    async def _safe_invoke(
        self, callback: StageCallback | None, managed: ManagedSession, stage: SessionStage
    ) -> None:
        if callback is None:
            logger.error("Dispatcher stage %s has no callback", stage)
            return
        try:
            await callback(managed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Dispatcher stage %s failed for session %s", stage, managed.session_id
            )

    async def _begin(
        self,
        session_id: str,
        stage: SessionStage,
        *,
        from_ready: bool = False,
    ) -> ManagedSession | None:
        """Mark the session as inflight in *stage*. Returns None if session is gone."""
        async with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None:
                return None
            if from_ready:
                # READY slot was granted via semaphore; RUNNING consumes it.
                pass
            managed.stage = stage
            managed.inflight = True
        self._notify_stage_change(managed)
        return managed

    async def _finish_init(self, managed: ManagedSession) -> None:
        """After INIT callback returns, move to READY (waiting a run slot) or POSTRUN."""
        if managed.cancel_requested or managed.final_result is not None:
            await self._move_to_postrun(managed)
            return

        async with self._lock:
            if managed.session_id not in self._sessions:
                return
            managed.stage = SessionStage.READY
            managed.inflight = False
        self._notify_stage_change(managed)

        acquired = await self._acquire_ready_slot(managed)
        if not acquired:
            await self._move_to_postrun(managed, release_ready=False)
            return
        await self._ready_queue.put(managed.session_id)

    async def _transition_to_postrun(self, managed: ManagedSession) -> None:
        # The RUN callback is done (or was skipped). The ready slot is released
        # back to the pool on exit of RUNNING.
        self._ready_slots.release()
        await self._move_to_postrun(managed, release_ready=False)

    async def _move_to_postrun(
        self, managed: ManagedSession, *, release_ready: bool = True
    ) -> None:
        transitioned = False
        async with self._lock:
            if managed.session_id not in self._sessions:
                return
            if managed.stage == SessionStage.POSTRUN:
                return
            if release_ready and managed.stage == SessionStage.READY and not managed.inflight:
                self._ready_slots.release()
            managed.stage = SessionStage.POSTRUN
            managed.inflight = False
            transitioned = True
        if transitioned:
            self._notify_stage_change(managed)
            await self._postrun_queue.put(managed.session_id)

    async def _acquire_ready_slot(self, managed: ManagedSession) -> bool:
        """Race the semaphore against session cancellation. Returns True on acquire."""
        if managed.cancel_event.is_set() or managed.final_result is not None:
            return False
        acquire_task = asyncio.create_task(self._ready_slots.acquire())
        cancel_task = asyncio.create_task(managed.cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {acquire_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            cancel_task.cancel()
            if not acquire_task.done():
                acquire_task.cancel()
        # If we managed to acquire the semaphore despite cancellation, release it
        # so it doesn't leak to a later session.
        acquired = (
            acquire_task in done
            and not acquire_task.cancelled()
            and acquire_task.exception() is None
        )
        if managed.cancel_event.is_set() or managed.final_result is not None:
            if acquired:
                self._ready_slots.release()
            return False
        return acquired

    def _notify_stage_change(self, managed: ManagedSession) -> None:
        callback = self.on_stage_change
        if callback is None:
            return
        try:
            callback(managed)
        except Exception:
            logger.exception(
                "Dispatcher stage-change callback failed for session %s",
                managed.session_id,
            )
