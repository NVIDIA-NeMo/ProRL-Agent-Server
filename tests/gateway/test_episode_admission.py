from __future__ import annotations

import asyncio

import pytest

from polar.gateway.episode_admission import (
    EpisodeAcquireCancelled,
    EpisodeAcquireTimeout,
    EpisodeAdmissionPoisoned,
    EpisodeCallUnauthorized,
    EpisodeLeaseConflict,
    EpisodeLeaseNotOwned,
    EpisodeReleaseDraining,
    ModelPoolEpisodeAdmission,
)


@pytest.mark.asyncio
async def test_episode_admission_enforces_alias_cap_and_is_independent() -> None:
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1, "pool/gpt": 1})
    qwen = await admission.acquire(
        session_id="qwen-a",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )
    queued = asyncio.create_task(
        admission.acquire(
            session_id="qwen-b",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=1,
        )
    )
    await asyncio.sleep(0)

    gpt = await admission.acquire(
        session_id="gpt-a",
        alias="pool/gpt",
        attempt_id="0:solve",
        timeout_seconds=1,
    )
    assert not queued.done()
    assert await admission.snapshot() == {
        "pool/gpt": {"cap": 1, "active": 1, "queued": 0},
        "pool/qwen": {"cap": 1, "active": 1, "queued": 1},
    }

    await admission.release(session_id="qwen-a", lease_id=qwen.lease_id)
    qwen_b = await queued
    assert qwen_b.session_id == "qwen-b"
    await admission.release_session("qwen-b")
    await admission.release(session_id="gpt-a", lease_id=gpt.lease_id)
    assert all(row["active"] == row["queued"] == 0 for row in (await admission.snapshot()).values())


@pytest.mark.asyncio
async def test_acquire_is_idempotent_and_release_retry_is_safe() -> None:
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    first, retry = await asyncio.gather(
        admission.acquire(
            session_id="session-a",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=1,
        ),
        admission.acquire(
            session_id="session-a",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=1,
        ),
    )
    assert first == retry
    assert await admission.release(session_id="session-a", lease_id=first.lease_id)
    assert not await admission.release(session_id="session-a", lease_id=first.lease_id)
    with pytest.raises(EpisodeLeaseConflict, match="already released"):
        await admission.acquire(
            session_id="session-a",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=1,
        )


@pytest.mark.asyncio
async def test_one_session_cannot_hold_two_attempts_or_release_another_session() -> None:
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 2})
    lease = await admission.acquire(
        session_id="session-a",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )
    with pytest.raises(EpisodeLeaseConflict, match="already has"):
        await admission.acquire(
            session_id="session-a",
            alias="pool/qwen",
            attempt_id="1:verify",
            timeout_seconds=1,
        )
    with pytest.raises(EpisodeLeaseNotOwned):
        await admission.release(session_id="session-b", lease_id=lease.lease_id)
    assert await admission.has_active(session_id="session-a", alias="pool/qwen")
    await admission.release_session("session-a")


@pytest.mark.asyncio
async def test_timeout_and_session_cancel_do_not_leak_slots() -> None:
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    owner = await admission.acquire(
        session_id="owner",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )
    with pytest.raises(EpisodeAcquireTimeout) as first_timeout:
        await admission.acquire(
            session_id="timeout",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=0.01,
        )

    waiting = asyncio.create_task(
        admission.acquire(
            session_id="cancelled",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=10,
        )
    )
    await asyncio.sleep(0)
    await admission.cancel_waiters("cancelled")
    with pytest.raises(EpisodeAcquireCancelled):
        await waiting

    await admission.release(session_id="owner", lease_id=owner.lease_id)
    # A response-loss retry is terminally idempotent: the same attempt cannot
    # silently receive a fresh queue budget after its first timeout.
    with pytest.raises(EpisodeAcquireTimeout) as retry_timeout:
        await admission.acquire(
            session_id="timeout",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=1,
        )
    assert str(retry_timeout.value) == str(first_timeout.value)
    assert not await admission.has_active(session_id="timeout", alias="pool/qwen")

    replacement = await admission.acquire(
        session_id="replacement",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )
    assert replacement.alias == "pool/qwen"
    await admission.release_session("replacement")


@pytest.mark.asyncio
async def test_close_poison_cancels_waiters_and_retains_active_leases() -> None:
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    await admission.acquire(
        session_id="owner",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )
    waiting = asyncio.create_task(
        admission.acquire(
            session_id="waiter",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=10,
        )
    )
    await asyncio.sleep(0)
    await admission.close()
    with pytest.raises(EpisodeAdmissionPoisoned):
        await waiting
    assert (await admission.snapshot())["pool/qwen"] == {
        "cap": 1,
        "active": 1,
        "queued": 0,
    }


@pytest.mark.asyncio
async def test_session_record_cleanup_never_releases_an_active_lease() -> None:
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    lease = await admission.acquire(
        session_id="owner",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )

    assert await admission.cleanup_session_records("owner") is True
    assert await admission.has_active(session_id="owner", alias="pool/qwen")

    assert await admission.release(session_id="owner", lease_id=lease.lease_id)
    assert await admission.cleanup_session_records("owner") is False
    # The safe idempotency tombstone is session-scoped and may be discarded at
    # postrun once no active lease remains.
    with pytest.raises(EpisodeLeaseNotOwned):
        await admission.release(session_id="owner", lease_id=lease.lease_id)


@pytest.mark.asyncio
async def test_release_revokes_new_calls_and_holds_cap_until_inflight_drains() -> None:
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    lease = await admission.acquire(
        session_id="owner",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )
    handle = await admission.begin_request(
        call_capability=lease.call_capability,
        alias="pool/qwen",
    )
    queued = asyncio.create_task(
        admission.acquire(
            session_id="next",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=1,
        )
    )
    await asyncio.sleep(0)

    with pytest.raises(EpisodeReleaseDraining):
        await admission.release(
            session_id="owner",
            lease_id=lease.lease_id,
            wait_timeout_seconds=0.01,
        )
    assert not queued.done()
    with pytest.raises(EpisodeCallUnauthorized, match="capability is invalid"):
        await admission.begin_request(
            call_capability=lease.call_capability,
            alias="pool/qwen",
        )

    await admission.end_request(handle)
    await admission.release(session_id="owner", lease_id=lease.lease_id)
    replacement = await queued
    assert replacement.session_id == "next"
    await admission.release_session("next")


@pytest.mark.asyncio
async def test_runtime_destroyed_release_cancels_request_owner_before_releasing() -> None:
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    lease = await admission.acquire(
        session_id="owner",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )
    request_started = asyncio.Event()
    hold_request = asyncio.Event()

    async def own_request() -> None:
        handle = await admission.begin_request(
            call_capability=lease.call_capability,
            alias="pool/qwen",
        )
        request_started.set()
        try:
            await hold_request.wait()
        finally:
            await admission.end_request(handle)

    request_task = asyncio.create_task(own_request())
    await asyncio.wait_for(request_started.wait(), timeout=1)
    queued = asyncio.create_task(
        admission.acquire(
            session_id="next",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=1,
        )
    )
    await asyncio.sleep(0)

    await admission.release_after_runtime_destroyed(
        "owner",
        runtime_destroyed=True,
    )

    with pytest.raises(asyncio.CancelledError):
        await request_task
    replacement = await asyncio.wait_for(queued, timeout=1)
    assert replacement.session_id == "next"
    await admission.release_session("next")


@pytest.mark.asyncio
async def test_runtime_destroyed_release_is_safe_with_repeated_cancel_and_release() -> None:
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    lease = await admission.acquire(
        session_id="owner",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )
    request_started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cancellation_count = 0

    async def own_request() -> None:
        nonlocal cancellation_count
        handle = await admission.begin_request(
            call_capability=lease.call_capability,
            alias="pool/qwen",
        )
        request_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_count += 1
            cancellation_seen.set()
            await finish_cleanup.wait()
        finally:
            await admission.end_request(handle)

    request_task = asyncio.create_task(own_request())
    await asyncio.wait_for(request_started.wait(), timeout=1)
    queued = asyncio.create_task(
        admission.acquire(
            session_id="next",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=1,
        )
    )
    normal_release = asyncio.create_task(
        admission.release(session_id="owner", lease_id=lease.lease_id)
    )
    await asyncio.sleep(0)
    cleanup_one = asyncio.create_task(
        admission.release_after_runtime_destroyed(
            "owner",
            runtime_destroyed=True,
        )
    )
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1)
    cleanup_two = asyncio.create_task(
        admission.release_after_runtime_destroyed(
            "owner",
            runtime_destroyed=True,
        )
    )
    await asyncio.sleep(0)
    assert not queued.done()
    assert not normal_release.done()

    finish_cleanup.set()
    await asyncio.gather(cleanup_one, cleanup_two)
    assert await normal_release is True
    await request_task
    assert cancellation_count == 1
    replacement = await asyncio.wait_for(queued, timeout=1)
    assert replacement.session_id == "next"
    await admission.release_session("next")


@pytest.mark.asyncio
async def test_runtime_destroyed_release_keeps_slot_if_owner_skips_finalizer() -> None:
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    lease = await admission.acquire(
        session_id="owner",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )

    async def abandon_request():
        return await admission.begin_request(
            call_capability=lease.call_capability,
            alias="pool/qwen",
        )

    handle = await asyncio.create_task(abandon_request())
    queued = asyncio.create_task(
        admission.acquire(
            session_id="next",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=10,
        )
    )
    await asyncio.sleep(0)

    with pytest.raises(EpisodeReleaseDraining, match="did not complete"):
        await admission.release_after_runtime_destroyed(
            "owner",
            runtime_destroyed=True,
        )
    assert (await admission.snapshot())["pool/qwen"] == {
        "cap": 1,
        "active": 1,
        "queued": 1,
    }
    assert not queued.done()

    # Test-only cleanup: production intentionally poisons the node here.
    await admission.end_request(handle)
    await admission.release(session_id="owner", lease_id=lease.lease_id)
    replacement = await asyncio.wait_for(queued, timeout=1)
    await admission.release_session(replacement.session_id)
