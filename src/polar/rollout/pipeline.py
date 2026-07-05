"""Dispatch + collect rollout pipeline for gateway nodes."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from polar.platform.events import EventBus
from polar.rollout.balancer import NodeScheduler
from polar.rollout.models import (
    SessionContext,
    SessionDispatchRequest,
    SessionResult,
    SessionStatus,
)
from polar.trajectory.models import Trajectory

logger = logging.getLogger(__name__)

ResultCallback = Callable[[SessionResult], Awaitable[None] | None]

_CLEANUP_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_DEFINITELY_NOT_CONNECTED_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)
_DISPATCH_CONFIRM_ATTEMPTS = 5


def _trajectory_status(status: str) -> str:
    if status == SessionStatus.TIMEOUT:
        return SessionStatus.TIMEOUT
    if status == SessionStatus.COMPLETED:
        return SessionStatus.COMPLETED
    return SessionStatus.ERROR


def _is_usable_result(result: SessionResult) -> bool:
    """Return whether a completed session contains trainable trajectory data."""
    if result.status != SessionStatus.COMPLETED:
        return False
    return any(
        trace.prompt_ids and trace.response_ids and trace.loss_mask and any(trace.loss_mask)
        for trace in result.trajectory.traces
    )


class Pipeline:
    """Process rollout sessions by dispatching to gateway nodes and collecting results."""

    def __init__(
        self,
        *,
        callback_url: str,
        save_dir: str | None,
        scheduler: NodeScheduler,
        dispatch_poll_interval_seconds: float = 1.0,
        callback_grace_seconds: float = 180.0,
        http_max_connections: int = 1024,
        http_max_keepalive_connections: int = 256,
        cleanup_max_concurrency: int = 128,
        cleanup_max_attempts: int = 3,
        cleanup_retry_backoff_seconds: float = 0.1,
        event_bus: EventBus | None = None,
    ) -> None:
        if http_max_connections < 1:
            raise ValueError("http_max_connections must be positive")
        if not 0 <= http_max_keepalive_connections <= http_max_connections:
            raise ValueError(
                "http_max_keepalive_connections must be between zero and http_max_connections"
            )
        if not 1 <= cleanup_max_concurrency <= http_max_connections:
            raise ValueError(
                "cleanup_max_concurrency must be between one and http_max_connections"
            )
        if cleanup_max_attempts < 1:
            raise ValueError("cleanup_max_attempts must be positive")
        if cleanup_retry_backoff_seconds < 0:
            raise ValueError("cleanup_retry_backoff_seconds cannot be negative")

        self.callback_url = callback_url.rstrip("/")
        self.save_dir = Path(save_dir) if save_dir else None
        self.scheduler = scheduler
        self.dispatch_poll_interval_seconds = dispatch_poll_interval_seconds
        self.callback_grace_seconds = callback_grace_seconds
        self.http_max_connections = http_max_connections
        self.http_max_keepalive_connections = http_max_keepalive_connections
        self.cleanup_max_concurrency = cleanup_max_concurrency
        self.cleanup_max_attempts = cleanup_max_attempts
        self.cleanup_retry_backoff_seconds = cleanup_retry_backoff_seconds
        self.event_bus = event_bus

        self._client: httpx.AsyncClient | None = None
        self._cleanup_client: httpx.AsyncClient | None = None
        self._started = False
        self._lifecycle_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[SessionResult]] = {}
        self._pending_lock = asyncio.Lock()
        self._cleanup_slots = asyncio.Semaphore(cleanup_max_concurrency)
        self._cleanup_tasks: set[asyncio.Task[None]] = set()

    async def _emit(self, event_type: str, payload: dict) -> None:
        if self.event_bus is not None:
            await self.event_bus.publish(event_type, payload)

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._started:
                return
            control_token = os.environ.get("POLAR_CONTROL_PLANE_TOKEN", "").strip()
            control_headers = (
                {"X-Polar-Control-Token": control_token} if control_token else None
            )
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers=control_headers,
                limits=httpx.Limits(
                    max_connections=self.http_max_connections,
                    max_keepalive_connections=self.http_max_keepalive_connections,
                ),
            )
            # Keep terminal DELETEs isolated from dispatch/poll traffic.  A
            # saturated primary pool must not prevent cancellation from
            # releasing gateway runtimes and worker slots.
            self._cleanup_client = httpx.AsyncClient(
                timeout=10.0,
                headers=control_headers,
                limits=httpx.Limits(
                    max_connections=self.cleanup_max_concurrency,
                    max_keepalive_connections=min(64, self.cleanup_max_concurrency),
                ),
            )
            self._started = True

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if not self._started:
                return
            async with self._pending_lock:
                for future in self._pending.values():
                    if not future.done():
                        future.cancel()
                self._pending.clear()
            # Session workers normally await cleanup themselves.  Shielded
            # cleanup tasks may outlive a repeatedly-cancelled worker, so the
            # service lifecycle drains them before closing the shared client.
            while self._cleanup_tasks:
                await asyncio.gather(*tuple(self._cleanup_tasks), return_exceptions=True)
            if self._cleanup_client is not None:
                await self._cleanup_client.aclose()
                self._cleanup_client = None
            if self._client is not None:
                await self._client.aclose()
                self._client = None
            self._started = False

    async def run_batch(
        self,
        sessions: list[SessionContext],
        *,
        on_result: ResultCallback | None = None,
    ) -> list[SessionResult]:
        await self.start()
        if not sessions:
            return []

        threshold = sessions[0].request.early_stop_min_usable_sessions
        if threshold is None:
            return await asyncio.gather(
                *(self._dispatch_and_collect(session, on_result) for session in sessions)
            )

        collected_results: asyncio.Queue[tuple[int, SessionResult]] = asyncio.Queue()

        async def _run_one(
            index: int,
            session: SessionContext,
        ) -> SessionResult:
            return await self._dispatch_and_collect(
                session,
                on_result,
                on_collected=lambda result: collected_results.put_nowait((index, result)),
            )

        tasks = [
            asyncio.create_task(
                _run_one(index, session),
                name=f"polar-session-{session.session_id}",
            )
            for index, session in enumerate(sessions)
        ]
        ordered_results: list[SessionResult | None] = [None] * len(sessions)
        usable_sessions = 0
        early_stop_triggered = False
        try:
            for _ in sessions:
                index, result = await collected_results.get()
                ordered_results[index] = result
                if _is_usable_result(result):
                    usable_sessions += 1

                if not early_stop_triggered and usable_sessions >= threshold:
                    early_stop_triggered = True
                    # Mark every worker before cancelling it so the worker can
                    # distinguish this internal straggler stop from caller
                    # cancellation. Workers that already obtained a real
                    # result are allowed to finish persistence/callback rather
                    # than being replaced with a duplicate synthetic result.
                    for pending_session, task in zip(sessions, tasks, strict=True):
                        if task.done() or pending_session.rollout_result is not None:
                            continue
                        pending_session.early_stop_requested = True
                        pending_session.early_stop_usable_sessions = usable_sessions
                        task.cancel()

            # Result collection is deliberately notified before persistence,
            # callbacks and terminal gateway cleanup. The threshold can thus
            # cancel live stragglers immediately while already-resolved
            # workers finish their exactly-once finalization normally.
            await asyncio.gather(*tasks)
        except BaseException:
            # Caller cancellation and unexpected worker failures preserve the
            # original all-child cancellation semantics. In particular, no
            # synthetic results are emitted for an externally cancelled task.
            for session, task in zip(sessions, tasks, strict=True):
                session.early_stop_requested = False
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        assert all(result is not None for result in ordered_results)
        return [result for result in ordered_results if result is not None]

    async def accept_callback_result(self, result: SessionResult) -> bool:
        async with self._pending_lock:
            future = self._pending.get(result.session_id)
            if future is None or future.done():
                return False
            future.set_result(result)
            return True

    def status(self) -> dict[str, object]:
        return {
            "pending_sessions": len(self._pending),
        }

    def result_path_for(self, task_id: str, session_id: str) -> str | None:
        path = self._result_path(task_id, session_id)
        return None if path is None else str(path)

    async def _dispatch_and_collect(
        self,
        session: SessionContext,
        callback: ResultCallback | None,
        on_collected: Callable[[SessionResult], None] | None = None,
    ) -> SessionResult:
        if self._client is None:
            raise RuntimeError("pipeline has not been started")

        future = asyncio.get_running_loop().create_future()
        pending_registered = False
        result: SessionResult | None = None
        try:
            try:
                session.completion_future = future
                async with self._pending_lock:
                    self._pending[session.session_id] = future
                    pending_registered = True

                session.timer.mark("dispatch", "started")
                await self._emit(
                    "session.state_changed",
                    {
                        "task_id": session.task_id,
                        "session_id": session.session_id,
                        "status": "DISPATCHING",
                    },
                )
                dispatch_request = await self._dispatch_session(session)
                session.timer.mark("dispatch", "finished")
                await self._emit(
                    "session.state_changed",
                    {
                        "task_id": session.task_id,
                        "session_id": session.session_id,
                        "status": "REGISTERED",
                        "node_id": session.node_id,
                    },
                )
                result = await self._wait_for_result(session, dispatch_request, future)
            except asyncio.CancelledError:
                if not session.early_stop_requested:
                    raise
                result = self._early_stop_result(session)
            except TimeoutError as exc:
                logger.warning("Session %s timed out in rollout pipeline", session.session_id)
                result = self._failure_result(
                    session,
                    status=SessionStatus.TIMEOUT,
                    error=str(exc),
                )
            except Exception as exc:
                logger.exception("Dispatch failed for session %s", session.session_id)
                result = self._failure_result(session, error=str(exc))
            finally:
                if pending_registered:
                    async with self._pending_lock:
                        self._pending.pop(session.session_id, None)

            assert result is not None
            session.timer.mark("return", "finished")
            rollout_timing = session.timer.to_session_timing()
            result = result.model_copy(
                update={
                    "timing": result.timing.model_copy(
                        update={
                            "rollout_dispatch_ms": rollout_timing.rollout_dispatch_ms,
                            "rollout_result_wait_ms": rollout_timing.rollout_result_wait_ms,
                            "rollout_pipeline_e2e_ms": rollout_timing.rollout_pipeline_e2e_ms,
                        }
                    )
                }
            )
            # Publish this before the first finalization await. run_batch will
            # then never replace a real result whose persistence/callback is
            # already in progress with a synthetic early-stop result.
            session.rollout_result = result
            if on_collected is not None:
                on_collected(result)
            await asyncio.to_thread(self._persist_result, result)
            if callback is not None:
                maybe_awaitable = callback(result)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            return result
        finally:
            # Both external task cancellation and internal straggler stopping
            # reach this cleanup path. All session DELETEs are issued by their
            # own workers, so a batch early-stop cleans gateways concurrently.
            await self._finalize_session_cleanup(session)

    async def _dispatch_session(self, session: SessionContext) -> SessionDispatchRequest:
        if self._client is None:
            raise RuntimeError("pipeline has not been started")

        while True:
            node = self.scheduler.acquire_node()
            if node is None:
                remaining_timeout = self._remaining_timeout_seconds(session)
                await asyncio.sleep(min(self.dispatch_poll_interval_seconds, remaining_timeout))
                continue

            session.node_id = node.node_id
            session.gateway_url = node.gateway_url
            dispatch_timeout = self._remaining_timeout_seconds(session)
            dispatch_request = SessionDispatchRequest(
                session_id=session.session_id,
                task_id=session.task_id,
                instruction=session.request.instruction,
                dispatch_priority=session.request.dispatch_priority,
                remaining_timeout_seconds=session.request.timeout_seconds,
                callback_url=self.callback_url,
                runtime=session.request.runtime,
                agent=session.request.agent,
                builder=session.request.builder,
                evaluator=session.request.evaluator,
                metadata=dict(session.request.metadata),
            )
            try:
                response = await self._client.post(
                    f"{node.gateway_url}/sessions",
                    json=dispatch_request.model_dump(mode="json"),
                    timeout=min(30.0, dispatch_timeout),
                )
                response.raise_for_status()
                return dispatch_request
            except Exception as exc:
                if await self._accepted_duplicate_dispatch(
                    exc, node.gateway_url, session, dispatch_request
                ):
                    return dispatch_request
                if not isinstance(exc, _DEFINITELY_NOT_CONNECTED_ERRORS):
                    # A read/write timeout, connection reset, or HTTP failure
                    # can happen after the gateway accepted the request. Session
                    # ids are single-use only within one gateway process, so
                    # assigning the same id to another node could run two agents
                    # and race their callbacks. Keep the original placement and
                    # let the normal finally-path DELETE cancel it best-effort.
                    if isinstance(exc, httpx.TransportError):
                        self.scheduler.mark_unhealthy(node.node_id)
                    raise RuntimeError(
                        "gateway dispatch outcome is ambiguous for session "
                        f"{session.session_id} on {node.gateway_url}; refusing "
                        "cross-gateway retry"
                    ) from exc
                self.scheduler.release_reservation(node.node_id)
                self.scheduler.mark_unhealthy(node.node_id)
                try:
                    remaining_timeout = self._remaining_timeout_seconds(session)
                except TimeoutError:
                    raise TimeoutError(
                        "session timeout expired before gateway dispatch completed"
                    ) from exc
                await asyncio.sleep(min(self.dispatch_poll_interval_seconds, remaining_timeout))
                session.node_id = None
                session.gateway_url = None

    async def _accepted_duplicate_dispatch(
        self,
        exc: Exception,
        gateway_url: str,
        session: SessionContext,
        dispatch_request: SessionDispatchRequest,
    ) -> bool:
        """Treat dispatch errors after gateway accept as successful dispatch.

        A gateway can accept a session and still have the rollout server's POST
        fail before the acknowledgement arrives. A later retry of the same
        single-use session id usually returns 409. If the gateway reports that
        the session belongs to this task, the rollout server should continue to
        wait for its result instead of retrying forever.
        """
        if self._client is None:
            return False

        response = None
        confirmation_error: Exception | None = None
        for attempt in range(1, _DISPATCH_CONFIRM_ATTEMPTS + 1):
            try:
                response = await self._client.get(
                    f"{gateway_url}/sessions/{session.session_id}",
                    timeout=5.0,
                )
                response.raise_for_status()
                break
            except Exception as get_exc:
                confirmation_error = get_exc
                if attempt == _DISPATCH_CONFIRM_ATTEMPTS:
                    break
                # The same overloaded gateway may have accepted the POST but
                # be temporarily unable to answer the confirming GET. Never
                # reassign an ambiguous single-use id; give its control plane
                # a bounded chance to recover first.
                try:
                    remaining = self._remaining_timeout_seconds(session)
                except TimeoutError:
                    break
                await asyncio.sleep(
                    min(self.dispatch_poll_interval_seconds, remaining)
                )

        if response is None:
            log = logger.debug
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 409:
                log = logger.warning
            log(
                "Failed to confirm duplicate dispatch for session %s after %d "
                "attempt(s): %r",
                session.session_id,
                _DISPATCH_CONFIRM_ATTEMPTS,
                confirmation_error,
            )
            return False

        payload = response.json()
        if payload.get("task_id") != dispatch_request.task_id:
            logger.warning(
                "Duplicate session id %s exists on %s for task %s, expected %s",
                session.session_id,
                gateway_url,
                payload.get("task_id"),
                dispatch_request.task_id,
            )
            return False

        logger.info(
            "Confirmed duplicate dispatch for session %s on %s; continuing to wait for result",
            session.session_id,
            gateway_url,
        )
        return True

    async def _wait_for_result(
        self,
        session: SessionContext,
        dispatch_request: SessionDispatchRequest,
        future: asyncio.Future[SessionResult],
    ) -> SessionResult:
        if session.gateway_url is None:
            raise RuntimeError("session gateway_url was not assigned")

        # Interleave callback-wait and gateway-poll so the poll path is a
        # live safety net (not dead code). Covers two races:
        #   1. Callback HTTP POST dropped/delayed — poll GET finds the result.
        #   2. Gateway flips status→terminal a tick before serializing the
        #      result payload — we re-poll next iteration instead of
        #      synthesizing a failure.
        # REGISTERED is gateway queue time before INIT. Keep measuring it in
        # session timing, but do not spend the execution timeout until INIT starts.
        execution_timeout_started = False
        callback_deadline: float | None = None
        pre_init_poll_interval = self.dispatch_poll_interval_seconds
        result_poll_interval = max(self.dispatch_poll_interval_seconds, 5.0)

        while True:
            if execution_timeout_started:
                assert callback_deadline is not None
                remaining = callback_deadline - time.monotonic()
                if remaining <= 0:
                    break
                wait_timeout = min(result_poll_interval, remaining)
            else:
                wait_timeout = pre_init_poll_interval
            try:
                return await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=wait_timeout,
                )
            except asyncio.TimeoutError:
                pass
            try:
                status, result = await self._poll_session_state(session, timeout=30.0)
            except Exception as exc:
                logger.debug(
                    "poll_session_result failed for session %s: %s",
                    session.session_id,
                    exc,
                )
                status = None
                result = None
            if result is not None:
                if not future.done():
                    future.set_result(result)
                return result
            if (
                not execution_timeout_started
                and status is not None
                and status != SessionStatus.REGISTERED
            ):
                session.deadline_monotonic = time.monotonic() + session.request.timeout_seconds
                callback_deadline = self._callback_deadline_monotonic(session)
                execution_timeout_started = True

        raise TimeoutError(
            f"session {dispatch_request.session_id} did not return a terminal result "
            "before the callback deadline"
        )

    async def _poll_session_state(
        self,
        session: SessionContext,
        *,
        timeout: float,
    ) -> tuple[str | None, SessionResult | None]:
        if self._client is None:
            raise RuntimeError("pipeline has not been started")
        if session.gateway_url is None:
            raise RuntimeError("session gateway_url was not assigned")

        response = await self._client.get(
            f"{session.gateway_url}/sessions/{session.session_id}",
            timeout=min(30.0, timeout),
        )
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        result_payload = payload.get("result")
        status_value = str(status) if status is not None else None
        if isinstance(result_payload, dict):
            return status_value, SessionResult.model_validate(result_payload)
        # Gateway may flip status→terminal before the result payload is
        # serialized into the GET response. Returning None here keeps the
        # outer loop polling until either the payload lands or the callback
        # deadline expires — preventing synthesized empty-trace "failures"
        # that poisoned GRPO batches (see feedback_sglang_tool_parser.md
        # et al).
        return status_value, None

    async def _poll_session_result(
        self,
        session: SessionContext,
        *,
        timeout: float,
    ) -> SessionResult | None:
        _, result = await self._poll_session_state(session, timeout=timeout)
        return result

    def _remaining_timeout_seconds(self, session: SessionContext) -> float:
        remaining = session.deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("session timeout expired")
        return remaining

    def _callback_deadline_monotonic(self, session: SessionContext) -> float:
        return session.deadline_monotonic + self.callback_grace_seconds

    def _remaining_callback_window_seconds(self, session: SessionContext) -> float:
        remaining = self._callback_deadline_monotonic(session) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("session callback deadline expired")
        return remaining

    async def _cleanup_session(self, session: SessionContext) -> None:
        client = self._cleanup_client or self._client
        if client is None or session.gateway_url is None:
            return

        url = f"{session.gateway_url}/sessions/{session.session_id}"
        last_error: Exception | None = None
        for attempt in range(1, self.cleanup_max_attempts + 1):
            try:
                # Bound DELETE fan-out independently from the larger pool.
                # The slot is released before backoff so retries cannot occupy
                # the limiter and starve first attempts.
                async with self._cleanup_slots:
                    response = await client.delete(url, timeout=10.0)
                if response.status_code in {200, 404}:
                    return
                response.raise_for_status()
                return
            except asyncio.CancelledError:
                raise
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _CLEANUP_RETRYABLE_STATUS_CODES:
                    logger.warning(
                        "Failed to clean up session %s on gateway %s: HTTP %s",
                        session.session_id,
                        session.gateway_url,
                        exc.response.status_code,
                    )
                    return
                last_error = exc
            except httpx.TransportError as exc:
                # DELETE is idempotent.  PoolTimeout, connect/read timeouts and
                # transient connection failures are therefore safe to retry.
                last_error = exc
            except Exception:
                logger.warning(
                    "Failed to clean up session %s on gateway %s",
                    session.session_id,
                    session.gateway_url,
                    exc_info=True,
                )
                return

            if attempt == self.cleanup_max_attempts:
                break
            delay = self._cleanup_retry_delay(session.session_id, attempt)
            logger.debug(
                "Retrying cleanup for session %s after attempt %d/%d in %.3fs: %r",
                session.session_id,
                attempt,
                self.cleanup_max_attempts,
                delay,
                last_error,
            )
            if delay:
                await asyncio.sleep(delay)

        logger.warning(
            "Failed to clean up session %s on gateway %s after %d attempts: %r",
            session.session_id,
            session.gateway_url,
            self.cleanup_max_attempts,
            last_error,
        )

    async def _finalize_session_cleanup(self, session: SessionContext) -> None:
        """Run cleanup independently so service close can drain it after cancellation."""
        task = asyncio.create_task(
            self._cleanup_session(session),
            name=f"polar-cleanup-{session.session_id}",
        )
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        await asyncio.shield(task)

    def _cleanup_retry_delay(self, session_id: str, attempt: int) -> float:
        # A stable per-session jitter spreads a mass early-stop without using
        # global randomness (and keeps tests/replays deterministic).
        jitter = 0.75 + (sum(session_id.encode("utf-8")) % 51) / 100.0
        return self.cleanup_retry_backoff_seconds * (2 ** (attempt - 1)) * jitter

    def _persist_result(self, result: SessionResult) -> None:
        path = self._result_path(result.task_id, result.session_id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._storage_payload(result), separators=(",", ":"), default=str)
        )

    def _result_path(self, task_id: str, session_id: str) -> Path | None:
        if self.save_dir is None:
            return None
        return self.save_dir / f"task_{task_id}" / f"ses_{session_id}.json"

    @staticmethod
    def _storage_payload(result: SessionResult) -> dict[str, object]:
        """Return the persisted session artifact shape.

        The on-disk rollout result keeps session-level status/error only.
        Trajectory payloads store the structured trace data without duplicating
        terminal status information.
        """
        payload = result.model_dump(mode="json")
        trajectory = payload.get("trajectory")
        if isinstance(trajectory, dict):
            trajectory.pop("status", None)
            trajectory.pop("error", None)
        return payload

    @staticmethod
    def _failure_result(
        session: SessionContext,
        *,
        status: str = SessionStatus.ERROR,
        error: str,
    ) -> SessionResult:
        return SessionResult(
            session_id=session.session_id,
            task_id=session.task_id,
            status=_trajectory_status(status),
            trajectory=Trajectory(
                status=_trajectory_status(status),
                metadata={
                    "builder": session.request.builder.strategy,
                    "record_count": 0,
                    "task_metadata": dict(session.request.metadata),
                },
                traces=[],
                error=error,
            ),
            timing=session.timer.to_session_timing(),
            node_id=session.node_id,
            error=error,
            metadata=dict(session.request.metadata),
        )

    @staticmethod
    def _early_stop_result(session: SessionContext) -> SessionResult:
        """Build the zero-gradient placeholder backing an internal cancellation."""
        session.timer.mark("return", "finished")
        timing = session.timer.to_session_timing()
        threshold = session.request.early_stop_min_usable_sessions
        error = (
            "session cancelled after rollout batch reached its minimum usable "
            f"session count ({session.early_stop_usable_sessions}/{threshold})"
        )
        cancellation_metadata: dict[str, object] = {
            "early_stop_cancelled": True,
            "fully_masked": True,
            "early_stop_reason": "minimum_usable_sessions_reached",
            "early_stop_min_usable_sessions": threshold,
            "early_stop_usable_sessions": session.early_stop_usable_sessions,
            "early_stop_elapsed_ms": timing.rollout_pipeline_e2e_ms,
        }
        result_metadata = dict(session.request.metadata)
        result_metadata.update(cancellation_metadata)
        trajectory_metadata: dict[str, object] = {
            "builder": session.request.builder.strategy,
            "record_count": 0,
            "task_metadata": dict(session.request.metadata),
        }
        trajectory_metadata.update(cancellation_metadata)
        return SessionResult(
            session_id=session.session_id,
            task_id=session.task_id,
            status=SessionStatus.ERROR,
            trajectory=Trajectory(
                status=SessionStatus.ERROR,
                metadata=trajectory_metadata,
                traces=[],
                error=error,
            ),
            timing=timing,
            node_id=session.node_id,
            error=error,
            metadata=result_metadata,
        )
