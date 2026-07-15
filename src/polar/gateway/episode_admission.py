"""Per-model admission for complete frozen-agent episodes.

The existing model-pool concurrency limit protects individual HTTP requests.
SPilot candidates make many sequential requests, so request-level admission can
still start hundreds of complete coding agents and let their active timeout be
consumed while waiting for the same remote backend.  This module owns a second,
coarser lease that spans one complete candidate-agent process.

Leases are deliberately process-local.  A gateway restart also destroys its
session runtimes, so persisting leases would only resurrect stale ownership.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import secrets
import time
from typing import Mapping


class EpisodeAdmissionError(RuntimeError):
    """Base class for fail-closed episode-admission errors."""


class UnknownEpisodeAlias(EpisodeAdmissionError):
    """The requested alias has no configured episode admission pool."""


class EpisodeLeaseConflict(EpisodeAdmissionError):
    """A session or attempt already owns incompatible admission state."""


class EpisodeAcquireTimeout(EpisodeAdmissionError):
    """No episode slot became available inside the caller's queue budget."""


class EpisodeAcquireCancelled(EpisodeAdmissionError):
    """The owning rollout session was cancelled while queued."""


class EpisodeLeaseNotOwned(EpisodeAdmissionError):
    """A release token is unknown or belongs to another rollout session."""


class EpisodeAdmissionPoisoned(EpisodeAdmissionError):
    """The node admission boundary is permanently fail-closed."""


class EpisodeCallUnauthorized(EpisodeAdmissionError):
    """A pool call does not carry an active lease-scoped credential."""


class EpisodeLeaseClosing(EpisodeAdmissionError):
    """The lease is closing and rejects new provider requests."""


class EpisodeReleaseDraining(EpisodeAdmissionError):
    """Lease close started but upstream requests have not drained yet."""


@dataclass(frozen=True, slots=True)
class EpisodeLeaseGrant:
    """Opaque ownership returned after one alias slot is acquired."""

    lease_id: str
    session_id: str
    alias: str
    attempt_id: str
    wait_ms: int
    local_cap: int
    call_capability: str


@dataclass(frozen=True, slots=True)
class EpisodeRequestHandle:
    lease_id: str
    session_id: str
    alias: str
    request_id: str


@dataclass(slots=True)
class _PendingAttempt:
    session_id: str
    alias: str
    attempt_id: str
    enqueued_at: float
    future: asyncio.Future[EpisodeLeaseGrant]
    task: asyncio.Task[None] | None = None


@dataclass(slots=True)
class _ActiveLease:
    grant: EpisodeLeaseGrant
    acquired_at: float
    call_digest: bytes
    inflight: int
    closing: bool
    drained: asyncio.Event
    request_tasks: dict[str, asyncio.Task[object]]
    runtime_destroyed_cleanup: bool = False
    close_task: asyncio.Task[bool] | None = None


@dataclass(frozen=True, slots=True)
class _FailedAttempt:
    alias: str
    error_type: type[EpisodeAdmissionError]
    message: str

    def exception(self) -> EpisodeAdmissionError:
        return self.error_type(self.message)


def _consume_future_exception(future: asyncio.Future[object]) -> None:
    """Avoid un-retrieved exceptions after a disconnected long-poll client."""

    if future.cancelled():
        return
    future.exception()


class ModelPoolEpisodeAdmission:
    """Own bounded, idempotent full-episode leases for one gateway process."""

    def __init__(self, caps: Mapping[str, int]) -> None:
        normalized: dict[str, int] = {}
        for raw_alias, raw_cap in caps.items():
            alias = str(raw_alias).strip()
            if not alias:
                raise ValueError("episode admission aliases must be non-empty")
            if isinstance(raw_cap, bool) or not isinstance(raw_cap, int) or raw_cap <= 0:
                raise ValueError("episode admission caps must be positive integers")
            normalized[alias] = raw_cap
        self._caps = normalized
        self._slots = {
            alias: asyncio.BoundedSemaphore(cap) for alias, cap in normalized.items()
        }
        self._lock = asyncio.Lock()
        self._pending: dict[tuple[str, str], _PendingAttempt] = {}
        self._session_attempt: dict[str, tuple[str, str]] = {}
        self._active_by_attempt: dict[tuple[str, str], _ActiveLease] = {}
        self._active_by_id: dict[str, _ActiveLease] = {}
        self._active_by_call_digest: dict[bytes, _ActiveLease] = {}
        # Terminal acquire failures are tombstoned by attempt.  If an HTTP
        # response is lost, retrying the same attempt must return the same
        # timeout/cancellation rather than enqueue a second wait with a fresh
        # budget and possibly leak a grant the caller no longer expects.
        self._failed_by_attempt: dict[tuple[str, str], _FailedAttempt] = {}
        # At most two attempts are valid for the SPilot state machine.  Keeping
        # their released tokens until session teardown makes release retries
        # idempotent without retaining state across sessions.
        self._released_by_id: dict[str, str] = {}
        self._completed_attempts: set[tuple[str, str]] = set()
        self._closed = False
        self._poisoned_reason: str | None = None

    async def acquire(
        self,
        *,
        session_id: str,
        alias: str,
        attempt_id: str,
        timeout_seconds: float,
    ) -> EpisodeLeaseGrant:
        """Wait for one alias slot, idempotently by session and attempt id."""

        if alias not in self._slots:
            raise UnknownEpisodeAlias(f"model-pool alias {alias!r} has no episode cap")
        if not session_id or not attempt_id:
            raise ValueError("session_id and attempt_id must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        key = (session_id, attempt_id)
        async with self._lock:
            if self._closed:
                raise EpisodeAcquireCancelled("episode admission is shutting down")
            if self._poisoned_reason is not None:
                raise EpisodeAdmissionPoisoned("episode admission is poisoned")
            active = self._active_by_attempt.get(key)
            if active is not None:
                if active.grant.alias != alias:
                    raise EpisodeLeaseConflict("attempt already owns another model alias")
                return active.grant
            if key in self._completed_attempts:
                raise EpisodeLeaseConflict("attempt lease was already released")
            failed = self._failed_by_attempt.get(key)
            if failed is not None:
                if failed.alias != alias:
                    raise EpisodeLeaseConflict(
                        "failed attempt belongs to another model alias"
                    )
                raise failed.exception()
            pending = self._pending.get(key)
            if pending is not None:
                if pending.alias != alias:
                    raise EpisodeLeaseConflict("attempt is queued for another model alias")
                future = pending.future
            else:
                other_key = self._session_attempt.get(session_id)
                if other_key is not None and other_key != key:
                    raise EpisodeLeaseConflict(
                        "session already has a pending or active episode lease"
                    )
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                future.add_done_callback(_consume_future_exception)
                pending = _PendingAttempt(
                    session_id=session_id,
                    alias=alias,
                    attempt_id=attempt_id,
                    enqueued_at=time.monotonic(),
                    future=future,
                )
                self._pending[key] = pending
                self._session_attempt[session_id] = key
                pending.task = asyncio.create_task(
                    self._grant_pending(pending, timeout_seconds),
                    name=f"polar-episode-admission-{session_id}-{attempt_id}",
                )

        # Shield manager-owned acquisition from an HTTP client disconnect.  A
        # retry with the same attempt id receives the same eventual grant, and
        # session cancellation remains the authoritative cleanup path.
        return await asyncio.shield(future)

    async def _grant_pending(
        self,
        pending: _PendingAttempt,
        timeout_seconds: float,
    ) -> None:
        key = (pending.session_id, pending.attempt_id)
        semaphore = self._slots[pending.alias]
        acquired = False
        transferred = False
        try:
            try:
                await asyncio.wait_for(semaphore.acquire(), timeout=timeout_seconds)
                acquired = True
            except TimeoutError:
                await self._fail_pending(
                    pending,
                    EpisodeAcquireTimeout(
                        f"episode slot for {pending.alias!r} was unavailable for "
                        f"{timeout_seconds:.1f}s"
                    ),
                )
                return

            async with self._lock:
                current = self._pending.get(key)
                if (
                    self._closed
                    or self._poisoned_reason is not None
                    or current is not pending
                ):
                    return
                wait_ms = max(0, int((time.monotonic() - pending.enqueued_at) * 1000))
                call_capability = secrets.token_urlsafe(32)
                grant = EpisodeLeaseGrant(
                    lease_id=secrets.token_urlsafe(32),
                    session_id=pending.session_id,
                    alias=pending.alias,
                    attempt_id=pending.attempt_id,
                    wait_ms=wait_ms,
                    local_cap=self._caps[pending.alias],
                    call_capability=call_capability,
                )
                drained = asyncio.Event()
                drained.set()
                call_digest = hashlib.sha256(call_capability.encode("utf-8")).digest()
                active = _ActiveLease(
                    grant=grant,
                    acquired_at=time.monotonic(),
                    call_digest=call_digest,
                    inflight=0,
                    closing=False,
                    drained=drained,
                    request_tasks={},
                )
                self._pending.pop(key, None)
                self._active_by_attempt[key] = active
                self._active_by_id[grant.lease_id] = active
                self._active_by_call_digest[call_digest] = active
                transferred = True
                if not pending.future.done():
                    pending.future.set_result(grant)
        except asyncio.CancelledError:
            await self._fail_pending(
                pending,
                EpisodeAcquireCancelled("rollout session cancelled while queued"),
            )
            raise
        finally:
            if acquired and not transferred:
                semaphore.release()

    async def _fail_pending(
        self,
        pending: _PendingAttempt,
        error: EpisodeAdmissionError,
    ) -> None:
        key = (pending.session_id, pending.attempt_id)
        async with self._lock:
            if self._pending.get(key) is not pending:
                return
            self._pending.pop(key, None)
            if self._session_attempt.get(pending.session_id) == key:
                self._session_attempt.pop(pending.session_id, None)
            self._failed_by_attempt[key] = _FailedAttempt(
                alias=pending.alias,
                error_type=type(error),
                message=str(error),
            )
            if not pending.future.done():
                pending.future.set_exception(error)

    async def release(
        self,
        *,
        session_id: str,
        lease_id: str,
        wait_timeout_seconds: float | None = None,
    ) -> bool:
        """Revoke new calls, drain in-flight provider work, then free the slot."""

        first_close = False
        async with self._lock:
            if self._poisoned_reason is not None:
                raise EpisodeAdmissionPoisoned("episode admission is poisoned")
            active = self._active_by_id.get(lease_id)
            if active is None:
                if self._released_by_id.get(lease_id) == session_id:
                    return False
                raise EpisodeLeaseNotOwned("episode lease is unknown or not owned")
            grant = active.grant
            if grant.session_id != session_id:
                raise EpisodeLeaseNotOwned("episode lease is unknown or not owned")
            if not active.closing:
                first_close = True
                active.closing = True
                self._active_by_call_digest.pop(active.call_digest, None)
            if active.close_task is None:
                active.close_task = asyncio.create_task(
                    self._drain_and_finalize(active),
                    name=f"polar-episode-close-{lease_id}",
                )
            close_task = active.close_task
        try:
            if wait_timeout_seconds is None:
                await asyncio.shield(close_task)
            else:
                await asyncio.wait_for(
                    asyncio.shield(close_task),
                    timeout=wait_timeout_seconds,
                )
        except TimeoutError as exc:
            raise EpisodeReleaseDraining("episode lease is draining") from exc
        return first_close

    async def _drain_and_finalize(self, active: _ActiveLease) -> bool:
        await active.drained.wait()
        grant = active.grant
        key = (grant.session_id, grant.attempt_id)
        async with self._lock:
            if self._active_by_id.get(grant.lease_id) is not active:
                return False
            # Poison retains all active capacity until gateway shutdown even
            # after outstanding upstream requests happen to drain.
            if self._poisoned_reason is not None:
                raise EpisodeAdmissionPoisoned("episode admission is poisoned")
            self._active_by_id.pop(grant.lease_id, None)
            self._active_by_attempt.pop(key, None)
            self._active_by_call_digest.pop(active.call_digest, None)
            if self._session_attempt.get(grant.session_id) == key:
                self._session_attempt.pop(grant.session_id, None)
            self._released_by_id[grant.lease_id] = grant.session_id
            self._completed_attempts.add(key)
            semaphore = self._slots[grant.alias]
        semaphore.release()
        return True

    async def begin_request(
        self,
        *,
        call_capability: str,
        alias: str,
    ) -> EpisodeRequestHandle:
        owner_task = asyncio.current_task()
        if owner_task is None:
            raise RuntimeError("episode request has no owning asyncio task")
        request_id = secrets.token_urlsafe(16)
        digest = hashlib.sha256(call_capability.encode("utf-8")).digest()
        async with self._lock:
            if self._poisoned_reason is not None or self._closed:
                raise EpisodeAdmissionPoisoned("episode admission is poisoned")
            active = self._active_by_call_digest.get(digest)
            if active is None or active.grant.alias != alias:
                raise EpisodeCallUnauthorized("pool call capability is invalid")
            if active.closing:
                raise EpisodeLeaseClosing("episode lease is closing")
            active.inflight += 1
            active.drained.clear()
            active.request_tasks[request_id] = owner_task
            grant = active.grant
            return EpisodeRequestHandle(
                lease_id=grant.lease_id,
                session_id=grant.session_id,
                alias=grant.alias,
                request_id=request_id,
            )

    async def bind_request_task(self, handle: EpisodeRequestHandle) -> None:
        """Transfer a live request handle to the current response-owner task.

        Non-streaming requests remain owned by the route task that called
        :meth:`begin_request`.  Streaming responses may be driven by a distinct
        ASGI task after the route returns, so that task must become the owner
        before it starts sending the body.  Runtime-destruction cleanup can
        then cancel exactly the task whose ``finally`` calls ``end_request``.
        """

        owner_task = asyncio.current_task()
        if owner_task is None:
            raise RuntimeError("episode request has no owning asyncio task")
        async with self._lock:
            active = self._active_by_id.get(handle.lease_id)
            if active is None:
                raise EpisodeLeaseNotOwned("episode request lease is no longer active")
            grant = active.grant
            if grant.session_id != handle.session_id or grant.alias != handle.alias:
                raise EpisodeLeaseNotOwned("episode request ownership changed")
            if handle.request_id not in active.request_tasks:
                raise EpisodeLeaseNotOwned("episode request is no longer active")
            if active.runtime_destroyed_cleanup:
                raise EpisodeLeaseClosing("episode lease is closing")
            active.request_tasks[handle.request_id] = owner_task

    async def end_request(self, handle: EpisodeRequestHandle) -> None:
        async with self._lock:
            active = self._active_by_id.get(handle.lease_id)
            if active is None:
                return
            grant = active.grant
            if grant.session_id != handle.session_id or grant.alias != handle.alias:
                raise EpisodeLeaseNotOwned("episode request ownership changed")
            if active.request_tasks.pop(handle.request_id, None) is None:
                raise EpisodeLeaseNotOwned("episode request is no longer active")
            if active.inflight <= 0:
                raise RuntimeError("episode request inflight counter underflow")
            active.inflight -= 1
            if active.inflight == 0:
                active.drained.set()

    @staticmethod
    def _request_task_needs_cancel(task: asyncio.Task[object]) -> bool:
        if task.done():
            return False
        cancelling = getattr(task, "cancelling", None)
        return not callable(cancelling) or cancelling() == 0

    async def _cancel_requests_after_runtime_destroyed(
        self,
        *,
        session_id: str,
        lease_id: str,
    ) -> None:
        """Cancel request owners and wait for their normal finalizers.

        This is intentionally available only from the positive runtime-
        destruction path.  Cancelling a task is not itself a drain proof: the
        task must finish and ``end_request`` must remove every tracked handle.
        If either condition is not met, the lease remains active and capacity
        stays fail-closed.
        """

        current_task = asyncio.current_task()
        async with self._lock:
            active = self._active_by_id.get(lease_id)
            if active is None:
                if self._released_by_id.get(lease_id) == session_id:
                    return
                raise EpisodeLeaseNotOwned("episode lease is unknown or not owned")
            if active.grant.session_id != session_id:
                raise EpisodeLeaseNotOwned("episode lease is unknown or not owned")
            if not active.closing:
                active.closing = True
                self._active_by_call_digest.pop(active.call_digest, None)
            active.runtime_destroyed_cleanup = True
            request_tasks = tuple(set(active.request_tasks.values()))

        for task in request_tasks:
            if task is current_task:
                continue
            if self._request_task_needs_cancel(task):
                task.cancel()
        wait_tasks = tuple(task for task in request_tasks if task is not current_task)
        if wait_tasks:
            await asyncio.gather(*wait_tasks, return_exceptions=True)

        async with self._lock:
            active = self._active_by_id.get(lease_id)
            if active is None:
                return
            if active.request_tasks or active.inflight != 0:
                raise EpisodeReleaseDraining(
                    "episode request owners did not complete their drain finalizers"
                )

    async def has_active(self, *, session_id: str, alias: str) -> bool:
        if not self._caps:
            return False
        async with self._lock:
            key = self._session_attempt.get(session_id)
            if key is None:
                return False
            active = self._active_by_attempt.get(key)
            return (
                active is not None
                and active.grant.alias == alias
                and not active.closing
            )

    async def cancel_waiters(self, session_id: str) -> None:
        """Cancel a queued acquire without releasing a possibly live process."""

        if not self._caps:
            return

        task: asyncio.Task[None] | None = None
        pending: _PendingAttempt | None = None
        async with self._lock:
            key = self._session_attempt.get(session_id)
            if key is not None:
                pending = self._pending.get(key)
                if pending is not None:
                    task = pending.task
        if pending is None:
            return
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        # A task cancelled before its coroutine got its first timeslice never
        # enters `_grant_pending`'s CancelledError handler.  Make the state and
        # waiter terminal here as an idempotent second line of cleanup.
        await self._fail_pending(
            pending,
            EpisodeAcquireCancelled("rollout session cancelled while queued"),
        )

    async def release_session(self, session_id: str) -> None:
        """Compatibility helper for tests with an explicit destruction proof."""

        await self.release_after_runtime_destroyed(
            session_id,
            runtime_destroyed=True,
        )

    async def release_after_runtime_destroyed(
        self,
        session_id: str,
        *,
        runtime_destroyed: bool,
    ) -> None:
        """Close/drain leases only after positive whole-runtime destruction."""

        if not self._caps:
            return
        if not runtime_destroyed:
            raise EpisodeAdmissionError("runtime destruction was not proven")
        async with self._lock:
            if self._poisoned_reason is not None:
                raise EpisodeAdmissionPoisoned("episode admission is poisoned")

        await self.cancel_waiters(session_id)
        lease_ids: list[str] = []
        async with self._lock:
            for lease_id, active in self._active_by_id.items():
                if active.grant.session_id == session_id:
                    lease_ids.append(lease_id)
        for lease_id in lease_ids:
            try:
                await self._cancel_requests_after_runtime_destroyed(
                    session_id=session_id,
                    lease_id=lease_id,
                )
                await self.release(session_id=session_id, lease_id=lease_id)
            except EpisodeLeaseNotOwned:
                pass
        async with self._lock:
            self._completed_attempts = {
                key for key in self._completed_attempts if key[0] != session_id
            }
            self._failed_by_attempt = {
                key: failed
                for key, failed in self._failed_by_attempt.items()
                if key[0] != session_id
            }
            self._released_by_id = {
                lease_id: owner
                for lease_id, owner in self._released_by_id.items()
                if owner != session_id
            }

    async def poison(self, reason: str) -> None:
        """Reject all new work while retaining every active lease/cap slot."""

        tasks: list[asyncio.Task[None]] = []
        pendings: list[_PendingAttempt] = []
        async with self._lock:
            if self._poisoned_reason is None:
                self._poisoned_reason = reason or "unspecified fatal admission error"
            for pending in self._pending.values():
                pendings.append(pending)
                if pending.task is not None:
                    tasks.append(pending.task)
            for active in self._active_by_id.values():
                active.closing = True
                self._active_by_call_digest.pop(active.call_digest, None)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for pending in pendings:
            await self._fail_pending(
                pending,
                EpisodeAdmissionPoisoned("episode admission is poisoned"),
            )

    async def cleanup_session_records(self, session_id: str) -> bool:
        """Cancel pending work and discard safe tombstones without releasing.

        Returns ``True`` when an active lease remains.  The caller must treat
        that as a fatal retained lease: neither normal runner close/drain nor
        the runtime-destruction fallback produced a terminal acknowledgement.
        """

        if not self._caps:
            return False

        await self.cancel_waiters(session_id)
        async with self._lock:
            active_retained = any(
                active.grant.session_id == session_id
                for active in self._active_by_id.values()
            )
            self._completed_attempts = {
                key for key in self._completed_attempts if key[0] != session_id
            }
            self._failed_by_attempt = {
                key: failed
                for key, failed in self._failed_by_attempt.items()
                if key[0] != session_id
            }
            self._released_by_id = {
                lease_id: owner
                for lease_id, owner in self._released_by_id.items()
                if owner != session_id
            }
            return active_retained

    async def snapshot(self) -> dict[str, dict[str, int]]:
        if not self._caps:
            return {}
        async with self._lock:
            return {
                alias: {
                    "cap": cap,
                    "active": sum(
                        active.grant.alias == alias
                        for active in self._active_by_id.values()
                    ),
                    "queued": sum(
                        pending.alias == alias for pending in self._pending.values()
                    ),
                }
                for alias, cap in sorted(self._caps.items())
            }

    async def close(self) -> None:
        if not self._caps:
            self._closed = True
            return
        async with self._lock:
            if self._closed:
                return
            self._closed = True
        await self.poison("episode admission is shutting down")


__all__ = [
    "EpisodeAcquireCancelled",
    "EpisodeAcquireTimeout",
    "EpisodeAdmissionError",
    "EpisodeAdmissionPoisoned",
    "EpisodeCallUnauthorized",
    "EpisodeLeaseClosing",
    "EpisodeLeaseConflict",
    "EpisodeLeaseGrant",
    "EpisodeLeaseNotOwned",
    "EpisodeRequestHandle",
    "EpisodeReleaseDraining",
    "ModelPoolEpisodeAdmission",
    "UnknownEpisodeAlias",
]
