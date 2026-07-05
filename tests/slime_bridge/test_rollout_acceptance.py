from __future__ import annotations

from types import SimpleNamespace

import pytest

from slime_bridge import rollout as rollout_module
from slime_bridge.rollout import (
    AsyncPolarRolloutWorker,
    _convert_task_result_to_samples,
    _completed_trainable_session_count,
    _low_complete_accept_fraction_rejection_reason,
    _resolve_max_tokens,
    generate_rollout_polar_async,
    stop_global_worker,
)


def _config(threshold: float) -> SimpleNamespace:
    return SimpleNamespace(min_complete_accept_fraction=threshold)


def _worker_args() -> SimpleNamespace:
    return SimpleNamespace(
        polar_rollout_url="http://rollout:8080",
        polar_task_template={"agent": {"harness": "codex"}},
        polar_max_async_level=2,
        rollout_batch_size=4,
        n_samples_per_prompt=1,
        update_weights_interval=1,
        polar_callback_host="127.0.0.1",
    )


def _task_result(statuses: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        task_id="task-1",
        results=[
            SimpleNamespace(session_id=f"session-{i}", status=status)
            for i, status in enumerate(statuses)
        ],
    )


def _sample(
    session_index: int,
    *,
    loss_mask: list[int] | None = None,
    remove_sample: bool = False,
) -> SimpleNamespace:
    loss_mask = [1] if loss_mask is None else loss_mask
    return SimpleNamespace(
        loss_mask=loss_mask,
        response_length=len(loss_mask),
        remove_sample=remove_sample,
        metadata={"polar": {"session_id": f"session-{session_index}"}},
    )


def test_complete_accept_fraction_rejects_group_below_threshold() -> None:
    task_result = _task_result(["COMPLETED"] * 12 + ["TIMEOUT"] * 4)
    samples = [_sample(i) for i in range(12)] + [
        _sample(i, loss_mask=[0], remove_sample=True)
        for i in range(12, 16)
    ]

    reason = _low_complete_accept_fraction_rejection_reason(
        _config(0.8),
        task_result,
        samples,
    )

    assert "12/16" in reason
    assert "requires >= 13" in reason


def test_complete_accept_fraction_accepts_group_at_threshold() -> None:
    task_result = _task_result(["COMPLETED"] * 13 + ["TIMEOUT"] * 3)
    samples = [_sample(i) for i in range(13)] + [
        _sample(i, loss_mask=[0], remove_sample=True)
        for i in range(13, 16)
    ]

    assert (
        _low_complete_accept_fraction_rejection_reason(
            _config(0.8),
            task_result,
            samples,
        )
        is None
    )


def test_complete_accept_fraction_requires_trainable_completed_sessions() -> None:
    task_result = _task_result(["COMPLETED"] * 13 + ["TIMEOUT"] * 3)
    samples = [_sample(i) for i in range(12)]
    samples.append(_sample(12, loss_mask=[0]))
    samples.extend(
        _sample(i, loss_mask=[0], remove_sample=True)
        for i in range(13, 16)
    )

    assert _completed_trainable_session_count(task_result, samples) == 12
    assert (
        _low_complete_accept_fraction_rejection_reason(
            _config(0.8),
            task_result,
            samples,
        )
        is not None
    )


def test_async_worker_only_admits_requested_groups() -> None:
    worker = AsyncPolarRolloutWorker(_worker_args(), data_source=SimpleNamespace())

    assert worker._can_admit_group({}, 0) is False

    worker.request_groups(1)
    assert worker._can_admit_group({}, 0) is True
    assert worker._can_admit_group({object(): SimpleNamespace()}, 1) is False

    worker._mark_delivered(1)
    assert worker._can_admit_group({}, 0) is False


def test_fully_async_worker_prefetches_only_for_outstanding_request() -> None:
    args = _worker_args()
    args.polar_fully_async = True
    worker = AsyncPolarRolloutWorker(args, data_source=SimpleNamespace())

    assert worker._can_admit_group({}, 0) is False

    worker.request_groups(4)
    # One cold-start trainer request may fill the complete 4*2=8 async window.
    active: dict[object, SimpleNamespace] = {}
    admitted = 0
    while worker._can_admit_group(active, len(active)):
        worker._consume_fully_async_admission_credit()
        active[object()] = SimpleNamespace()
        admitted += 1
    assert len(active) == 8
    assert admitted == 8

    # Deliver the requested baseline batch. Four already-owned groups remain,
    # but exhausted request credit means they are not replaced indefinitely.
    for key in list(active)[:4]:
        del active[key]
    worker._mark_delivered(4)
    assert worker._can_admit_group(active, 4) is False

    # The next request can be satisfied synchronously from completed backlog
    # before the worker thread observes it. Its four replacement credits must
    # survive that delivery, refill the pipeline, and then stop exactly.
    worker.request_groups(4)
    active.clear()
    worker._mark_delivered(4)
    assert worker.snapshot_metrics()["polar/scheduler/requested_groups"] == 0.0
    while worker._can_admit_group(active, len(active)):
        worker._consume_fully_async_admission_credit()
        active[object()] = SimpleNamespace()
        admitted += 1
    assert len(active) == 4
    assert admitted == 12
    assert worker._can_admit_group(active, 4) is False


def test_fully_async_uses_one_window_for_all_owned_group_states() -> None:
    args = _worker_args()
    args.polar_fully_async = True
    worker = AsyncPolarRolloutWorker(args, data_source=SimpleNamespace())
    worker.request_groups(1)

    # batch=4, async level=2 => one eight-group ownership window spanning
    # active + deferred + output queue + the local completed buffer.
    worker.output_queue.put(SimpleNamespace())
    worker._completed_buffer_size = 2
    active = {object(): SimpleNamespace() for _ in range(4)}
    # 4 active + 1 output + 2 buffered = 7.
    assert worker._can_admit_group(active, 4) is True

    worker._completed_buffer_size = 3
    # 4 active + 1 output + 3 buffered = 8, so a completed backlog consumes
    # the same bounded window as rollout work in flight.
    assert worker._can_admit_group(active, 4) is False

    # A deferred group is already owned: at the full boundary it may move to
    # active without consuming fresh admission credit, but no ninth group may
    # be reserved afterward.
    worker._completed_buffer_size = 2
    deferred = SimpleNamespace()
    worker.deferred_queue.put(deferred)
    credit_before = worker.snapshot_metrics()["polar/scheduler/admission_credit"]
    assert worker._can_admit_group(active, 4) is True
    assert worker._next_group_for_submission() is deferred
    active[object()] = SimpleNamespace()
    assert worker.snapshot_metrics()["polar/scheduler/admission_credit"] == credit_before
    assert worker._can_admit_group(active, 5) is False


def test_fully_async_fresh_admission_consumes_credit_but_deferred_reuse_does_not() -> None:
    args = _worker_args()
    args.polar_fully_async = True
    group = [SimpleNamespace(group_index=0)]
    source = SimpleNamespace(get_samples=lambda count: [group] if count == 1 else [])
    worker = AsyncPolarRolloutWorker(args, data_source=source)
    worker.request_groups(4)

    before = worker.snapshot_metrics()["polar/scheduler/admission_credit"]
    fresh = worker._next_group_for_submission()
    after_fresh = worker.snapshot_metrics()["polar/scheduler/admission_credit"]
    assert fresh is not None
    assert after_fresh == before - 1

    worker.deferred_queue.put(fresh)
    assert worker._next_group_for_submission() is fresh
    assert worker.snapshot_metrics()["polar/scheduler/admission_credit"] == after_fresh


def test_max_tokens_uses_dynamic_batch_cap_with_context_parallelism() -> None:
    args = SimpleNamespace(
        max_tokens_per_gpu=20_000,
        context_parallel_size=2,
        seq_length=65_536,
    )

    assert _resolve_max_tokens(args) == 40_000


def test_max_tokens_never_exceeds_model_sequence_length() -> None:
    args = SimpleNamespace(
        max_tokens_per_gpu=40_000,
        context_parallel_size=1,
        seq_length=32_768,
    )

    assert _resolve_max_tokens(args) == 32_768


def test_max_tokens_can_fall_back_to_sequence_length() -> None:
    args = SimpleNamespace(
        max_tokens_per_gpu=None,
        context_parallel_size=1,
        seq_length=32_768,
    )

    assert _resolve_max_tokens(args) == 32_768


def test_explicit_trajectory_cap_preserves_complete_tmax_pack() -> None:
    args = SimpleNamespace(
        max_tokens_per_gpu=67_584,
        context_parallel_size=1,
        seq_length=67_584,
        polar_max_trajectory_tokens=67_584,
    )

    assert _resolve_max_tokens(args) == 67_584


def test_explicit_trajectory_cap_fails_instead_of_silently_clipping() -> None:
    args = SimpleNamespace(
        max_tokens_per_gpu=16_384,
        context_parallel_size=1,
        seq_length=67_584,
        polar_max_trajectory_tokens=67_584,
    )

    with pytest.raises(ValueError, match="exceeds trainer capacity"):
        _resolve_max_tokens(args)


def test_explicit_trajectory_cap_can_exceed_aggregate_microbatch_cap() -> None:
    args = SimpleNamespace(
        max_tokens_per_gpu=24_576,
        context_parallel_size=1,
        seq_length=67_584,
        polar_max_trajectory_tokens=67_584,
        polar_allow_single_sample_over_token_cap=True,
    )

    assert _resolve_max_tokens(args) == 67_584


def test_aggregate_microbatch_cap_opt_in_requires_boolean() -> None:
    args = SimpleNamespace(
        max_tokens_per_gpu=24_576,
        context_parallel_size=1,
        seq_length=67_584,
        polar_allow_single_sample_over_token_cap="sometimes",
    )

    with pytest.raises(ValueError, match="must be a boolean"):
        _resolve_max_tokens(args)


def test_training_conversion_isolates_one_bad_session(monkeypatch) -> None:
    bad = SimpleNamespace(session_id="bad-session")
    good = SimpleNamespace(session_id="good-session")

    def convert_one(result, *_args, **_kwargs):
        if result is bad:
            raise ValueError("unaligned token metadata")
        return [SimpleNamespace(session_id=result.session_id, remove_sample=False)]

    def placeholder(result, *_args, conversion_error, **_kwargs):
        return SimpleNamespace(
            session_id=result.session_id,
            remove_sample=True,
            conversion_error=conversion_error,
        )

    monkeypatch.setattr(rollout_module, "session_result_to_samples", convert_one)
    monkeypatch.setattr(rollout_module, "session_result_to_placeholder", placeholder)

    samples = _convert_task_result_to_samples(
        SimpleNamespace(reward_key="score"),
        SimpleNamespace(task_id="task-1", results=[bad, good]),
        [
            SimpleNamespace(group_index=3, index=10),
            SimpleNamespace(group_index=3, index=11),
        ],
        max_tokens=67_584,
    )

    assert [sample.session_id for sample in samples] == ["bad-session", "good-session"]
    assert samples[0].remove_sample is True
    assert "unaligned token metadata" in samples[0].conversion_error
    assert samples[1].remove_sample is False


def test_polar_rollout_exposes_explicit_dispose_hook() -> None:
    assert getattr(generate_rollout_polar_async, "dispose") is stop_global_worker
