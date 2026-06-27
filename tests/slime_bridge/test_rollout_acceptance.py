from __future__ import annotations

from types import SimpleNamespace

from slime_bridge.rollout import (
    AsyncPolarRolloutWorker,
    _completed_trainable_session_count,
    _low_complete_accept_fraction_rejection_reason,
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


def test_fully_async_worker_keeps_prefetch_window_warm() -> None:
    args = _worker_args()
    args.polar_fully_async = True
    worker = AsyncPolarRolloutWorker(args, data_source=SimpleNamespace())

    assert worker._can_admit_group({}, 0) is False

    worker.request_groups(1)
    worker._mark_delivered(1)
    assert worker._can_admit_group({}, 0) is True

    active = {object(): SimpleNamespace() for _ in range(8)}
    assert worker._can_admit_group(active, 8) is False
