from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from ray.exceptions import TaskCancelledError

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.types import Sample
from slime_bridge import rollout as rollout_module
from slime_bridge.rollout import (
    AsyncPolarRolloutWorker,
    PolarLowCompleteAcceptFractionError,
    _CompletedGroup,
    _PendingGroup,
    _completed_service_metrics,
    generate_rollout_polar_async,
)


def _worker_args(*, batch_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        polar_rollout_url="http://rollout:8080",
        polar_task_template={"agent": {"harness": "codex"}},
        polar_max_async_level=2,
        rollout_batch_size=batch_size,
        n_samples_per_prompt=1,
        update_weights_interval=1,
        polar_callback_host="127.0.0.1",
    )


class _ReservationSource:
    def __init__(self, reservations=()) -> None:
        self.reservations = list(reservations)
        self.reserve_calls = 0
        self.consumed: list[tuple[int, str]] = []

    def get_samples_with_reservation(self, count: int):
        assert count == 1
        self.reserve_calls += 1
        if not self.reservations:
            return []
        return [self.reservations.pop(0)]

    def get_samples(self, _count: int):
        pytest.fail("worker must use the reservation-aware API")

    def mark_consumed(self, reservation_id: int, *, outcome: str):
        self.consumed.append((reservation_id, outcome))
        return {"polar/reservations/outstanding_groups": 0.0}

    def reservation_metrics(self):
        return {"polar/reservations/outstanding_groups": 1.0}


def _group(group_index: int):
    return [SimpleNamespace(group_index=group_index, index=group_index)]


def test_worker_reserves_fresh_group_once_and_reuses_deferred_group() -> None:
    source = _ReservationSource([(41, _group(41))])
    worker = AsyncPolarRolloutWorker(_worker_args(), data_source=source)

    first = worker._next_group_for_submission()
    assert first is not None
    worker.deferred_queue.put(first)
    second = worker._next_group_for_submission()

    assert second is first
    assert second.reservation_id == 41
    assert source.reserve_calls == 1


@pytest.mark.asyncio
async def test_worker_marks_quality_rejection_as_permanently_consumed(monkeypatch) -> None:
    source = _ReservationSource()
    worker = AsyncPolarRolloutWorker(_worker_args(), data_source=source)
    pending = _PendingGroup(
        group_id=3,
        group=_group(52),
        reservation_id=52,
        submitted_rollout_id=4,
        policy_version=4,
        session_cost=1,
    )

    async def fail_attempt(*_args, **_kwargs):
        raise PolarLowCompleteAcceptFractionError("deterministic quality rejection")

    monkeypatch.setattr(worker, "_submit_attempt", fail_attempt)
    await worker._submit_and_collect(SimpleNamespace(), pending)

    assert source.consumed == [(52, "permanent_drop")]


@pytest.mark.asyncio
async def test_worker_stops_on_unknown_failure_and_keeps_it_for_resume(monkeypatch) -> None:
    source = _ReservationSource()
    worker = AsyncPolarRolloutWorker(_worker_args(), data_source=source)
    pending = _PendingGroup(
        group_id=4,
        group=_group(53),
        reservation_id=53,
        submitted_rollout_id=4,
        policy_version=4,
        session_cost=1,
    )

    async def fail_attempt(*_args, **_kwargs):
        raise RuntimeError("transient transport failure")

    monkeypatch.setattr(worker, "_submit_attempt", fail_attempt)
    await worker._submit_and_collect(SimpleNamespace(), pending)

    assert source.consumed == []
    assert worker.snapshot_metrics()["polar/replay_on_resume_groups_since_worker_start"] == 1.0
    assert worker._running is False
    with pytest.raises(RuntimeError, match="transient transport failure"):
        worker.raise_if_failed()


@pytest.mark.asyncio
async def test_worker_stop_race_does_not_consume_failed_active_reservation(monkeypatch) -> None:
    source = _ReservationSource()
    worker = AsyncPolarRolloutWorker(_worker_args(), data_source=source)
    pending = _PendingGroup(
        group_id=5,
        group=_group(54),
        reservation_id=54,
        submitted_rollout_id=4,
        policy_version=4,
        session_cost=1,
    )

    async def fail_while_stopping(*_args, **_kwargs):
        worker._running = False
        raise PolarLowCompleteAcceptFractionError("worker stopped during completion")

    monkeypatch.setattr(worker, "_submit_attempt", fail_while_stopping)
    await worker._submit_and_collect(SimpleNamespace(), pending)

    assert source.consumed == []


def test_worker_marks_stale_completed_group_consumed() -> None:
    source = _ReservationSource()
    worker = AsyncPolarRolloutWorker(_worker_args(), data_source=source)
    worker.config = replace(worker.config, max_off_policy_steps=0)
    worker._completed_buffer.append(
        _CompletedGroup(
            group_id=9,
            group=_group(63),
            reservation_id=63,
            samples=[],
            task_id="task-9",
            submitted_rollout_id=0,
            policy_version=0,
            session_count=1,
        )
    )

    assert worker.drain_completed(max_groups=1, rollout_id=1) == []
    assert source.consumed == [(63, "stale")]


def test_fully_async_dynamic_filter_drop_restores_fresh_admission_demand() -> None:
    source = _ReservationSource()
    args = _worker_args(batch_size=1)
    args.polar_fully_async = True
    worker = AsyncPolarRolloutWorker(args, data_source=source)
    worker.request_groups(1)
    worker._consume_fully_async_admission_credit()
    worker._completed_buffer.append(_completion_with_rewards(65, [0.0, 0.0]))

    completed = worker.drain_completed(max_groups=1, rollout_id=0)[0]
    assert worker.snapshot_metrics()["polar/scheduler/requested_groups"] == 0.0
    worker.mark_dynamic_filter_drop(completed, reason="zero_std_0.0")
    worker.request_groups(1)

    metrics = worker.snapshot_metrics()
    assert source.consumed == [(65, "dynamic_filter")]
    assert metrics["polar/scheduler/requested_groups"] == 1.0
    assert metrics["polar/scheduler/admission_credit"] == 2.0
    assert worker._can_admit_group({}, 0) is True


@pytest.mark.asyncio
async def test_completed_group_records_submit_to_complete_service_time(monkeypatch) -> None:
    worker = AsyncPolarRolloutWorker(_worker_args(), data_source=_ReservationSource())
    pending = _PendingGroup(
        group_id=10,
        group=_group(64),
        reservation_id=64,
        submitted_rollout_id=1,
        policy_version=1,
        session_cost=1,
    )
    task_result = SimpleNamespace(task_id="task-10", status="completed", results=[object()])

    async def complete_after_service_time(_client, _payload):
        await asyncio.sleep(0.01)
        return task_result

    monkeypatch.setattr(
        rollout_module, "_build_task_payload", lambda **_kwargs: {"task_id": "task-10"}
    )
    monkeypatch.setattr(worker, "_submit_with_callback", complete_after_service_time)
    monkeypatch.setattr(
        rollout_module,
        "_convert_task_result_to_samples",
        lambda *_args, **_kwargs: [SimpleNamespace(loss_mask=[1], remove_sample=False)],
    )

    completed = await worker._submit_attempt(SimpleNamespace(), pending)

    assert completed.submitted_at == pending.submitted_at
    assert completed.completed_at > completed.submitted_at
    assert completed.service_time_seconds == pytest.approx(
        completed.completed_at - completed.submitted_at
    )
    assert completed.service_time_seconds >= 0.01


class _GenerateWorker:
    def __init__(self, completions) -> None:
        self.config = SimpleNamespace(reward_key="score")
        self.completions = list(completions)
        self.consumed: list[tuple[list[int | None], str]] = []
        self.requested: list[int] = []
        self.dynamic_filter_drops: list[tuple[int, str | None]] = []
        self.consumed_by_outcome: dict[str, int] = {}

    def set_rollout_context(self, _rollout_id: int) -> None:
        pass

    def request_groups(self, count: int) -> None:
        self.requested.append(count)

    def drain_completed(self, *, max_groups: int, rollout_id: int):
        del rollout_id
        drained = self.completions[:max_groups]
        del self.completions[:max_groups]
        return drained

    def queue_size(self) -> int:
        return len(self.completions)

    def snapshot_metrics(self):
        return {
            "polar/reservations/outstanding_groups": 1.0,
            "polar/dropped_dynamic_filter_groups_delta": float(len(self.dynamic_filter_drops)),
        }

    def mark_dynamic_filter_drop(self, completed, *, reason: str | None):
        self.dynamic_filter_drops.append((completed.reservation_id, reason))
        return self._consume_reservations(
            [completed.reservation_id],
            outcome="dynamic_filter",
        )

    def _consume_reservations(self, reservation_ids, *, outcome: str):
        self.consumed.append((list(reservation_ids), outcome))
        self.consumed_by_outcome[outcome] = self.consumed_by_outcome.get(outcome, 0) + len(
            reservation_ids
        )
        return {
            "polar/reservations/outstanding_groups": 0.0,
            f"polar/reservations/consumed_{outcome}_since_worker_start": float(
                self.consumed_by_outcome[outcome]
            ),
        }


def _completion(
    reservation_id: int,
    *,
    service_time_seconds: float = 0.0,
    submitted_at: float = 1.0,
) -> _CompletedGroup:
    sample = SimpleNamespace(
        metadata={"polar": {}},
        reward={"score": 1.0},
        response_length=1,
    )
    return _CompletedGroup(
        group_id=reservation_id,
        group=_group(reservation_id),
        reservation_id=reservation_id,
        samples=[sample],
        task_id=f"task-{reservation_id}",
        submitted_rollout_id=0,
        policy_version=0,
        session_count=1,
        submitted_at=submitted_at,
        completed_at=submitted_at + service_time_seconds,
        service_time_seconds=service_time_seconds,
    )


def _completion_with_rewards(reservation_id: int, rewards: list[float]) -> _CompletedGroup:
    completed = _completion(reservation_id)
    completed.samples = [
        Sample(
            group_index=reservation_id,
            index=index,
            reward={"score": reward},
            response_length=1,
            loss_mask=[1],
            metadata={"polar": {}},
            status=Sample.Status.COMPLETED,
        )
        for index, reward in enumerate(rewards)
    ]
    completed.session_count = len(rewards)
    return completed


def test_generate_commits_accepted_reservations_only_after_full_output(monkeypatch) -> None:
    worker = _GenerateWorker(
        [
            _completion(71, service_time_seconds=2.0, submitted_at=1.0),
            _completion(72, service_time_seconds=5.0, submitted_at=4.0),
        ]
    )
    monkeypatch.setattr(rollout_module, "get_global_async_worker", lambda *_args: worker)
    monkeypatch.setattr(rollout_module, "_current_ray_task_is_canceled", lambda: False)
    monkeypatch.setattr(
        rollout_module,
        "_load_rollout_train_output_type",
        lambda: RolloutFnTrainOutput,
    )

    output = generate_rollout_polar_async(
        SimpleNamespace(rollout_batch_size=2),
        rollout_id=1,
        data_source=SimpleNamespace(),
    )

    assert worker.consumed == [([71, 72], "accepted")]
    assert output.metrics["polar/reservations/outstanding_groups"] == 0.0
    assert output.metrics["polar/reservations/consumed_accepted_since_worker_start"] == 2.0
    assert output.metrics["timing/service_time_max"] == 5.0
    assert output.metrics["timing/service_window"] == 8.0
    assert output.metrics["timing/pipeline_ms/rollout_collect"] >= 0.0


def test_training_dynamic_filter_replaces_zero_std_groups_and_commits_mixed_group(
    monkeypatch,
) -> None:
    all_zero = _completion_with_rewards(91, [0.0, 0.0])
    all_one = _completion_with_rewards(92, [1.0, 1.0])
    mixed = _completion_with_rewards(93, [0.0, 1.0])
    worker = _GenerateWorker([all_zero, all_one, mixed])
    monkeypatch.setattr(rollout_module, "get_global_async_worker", lambda *_args: worker)
    monkeypatch.setattr(rollout_module, "_current_ray_task_is_canceled", lambda: False)
    monkeypatch.setattr(
        rollout_module,
        "_load_rollout_train_output_type",
        lambda: RolloutFnTrainOutput,
    )

    output = generate_rollout_polar_async(
        SimpleNamespace(
            rollout_batch_size=1,
            reward_key="score",
            dynamic_sampling_filter_path=(
                "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
            ),
        ),
        rollout_id=1,
        data_source=SimpleNamespace(),
    )

    assert output.samples == [mixed.samples]
    assert worker.requested == [1, 1, 1]
    assert worker.dynamic_filter_drops == [
        (91, "zero_std_0.0"),
        (92, "zero_std_1.0"),
    ]
    assert worker.consumed == [
        ([91], "dynamic_filter"),
        ([92], "dynamic_filter"),
        ([93], "accepted"),
    ]
    assert output.metrics["rollout/dynamic_filter/drop_zero_std_0.0"] == 1.0
    assert output.metrics["rollout/dynamic_filter/drop_zero_std_1.0"] == 1.0
    assert output.metrics["polar/dropped_dynamic_filter_groups_delta"] == 2.0
    assert output.metrics["polar/reservations/consumed_dynamic_filter_since_worker_start"] == 2.0
    assert output.metrics["polar/reservations/consumed_accepted_since_worker_start"] == 1.0


def test_training_dynamic_filter_ignores_failed_zero_reward_samples(monkeypatch) -> None:
    false_mixed = _completion_with_rewards(94, [1.0, 1.0, 0.0])
    false_mixed.samples[-1].status = Sample.Status.FAILED
    false_mixed.samples[-1].loss_mask = [0]
    false_mixed.samples[-1].remove_sample = True
    true_mixed = _completion_with_rewards(95, [1.0, 0.0])
    worker = _GenerateWorker([false_mixed, true_mixed])
    monkeypatch.setattr(rollout_module, "get_global_async_worker", lambda *_args: worker)
    monkeypatch.setattr(rollout_module, "_current_ray_task_is_canceled", lambda: False)
    monkeypatch.setattr(
        rollout_module,
        "_load_rollout_train_output_type",
        lambda: RolloutFnTrainOutput,
    )

    output = generate_rollout_polar_async(
        SimpleNamespace(
            rollout_batch_size=1,
            reward_key="score",
            dynamic_sampling_filter_path=(
                "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
            ),
        ),
        rollout_id=1,
        data_source=SimpleNamespace(),
    )

    assert output.samples == [true_mixed.samples]
    assert worker.dynamic_filter_drops == [(94, "zero_std_1.0")]
    assert worker.consumed == [
        ([94], "dynamic_filter"),
        ([95], "accepted"),
    ]
    assert output.metrics["rollout/dynamic_filter/drop_zero_std_1.0"] == 1.0


def test_eval_bypasses_training_dynamic_filter(monkeypatch) -> None:
    sentinel = object()

    async def fake_eval(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(rollout_module, "_run_eval_rollout", fake_eval)
    monkeypatch.setattr(
        rollout_module,
        "_load_training_dynamic_filter",
        lambda _args: pytest.fail("eval must not load the training dynamic filter"),
    )
    monkeypatch.setattr(
        rollout_module,
        "get_global_async_worker",
        lambda *_args: pytest.fail("eval must not enter the training worker"),
    )

    result = generate_rollout_polar_async(
        SimpleNamespace(dynamic_sampling_filter_path="tests.should.not.load"),
        rollout_id=0,
        data_source=SimpleNamespace(),
        evaluation=True,
    )

    assert result is sentinel


def test_overlapping_service_window_uses_batch_production_span() -> None:
    metrics = _completed_service_metrics(
        [
            _completion(73, service_time_seconds=5.0, submitted_at=1.0),
            _completion(74, service_time_seconds=3.0, submitted_at=2.0),
        ]
    )

    assert metrics == {
        "timing/service_time_max": 5.0,
        "timing/service_window": 5.0,
        "timing/pipeline_ms/sample_conversion_mean": 0.0,
        "timing/pipeline_ms/sample_conversion_max": 0.0,
        "timing/pipeline_ms/output_queue_wait_mean": 0.0,
        "timing/pipeline_ms/output_queue_wait_max": 0.0,
    }


@pytest.mark.asyncio
async def test_output_queue_wait_is_set_before_completed_group_is_visible(
    monkeypatch,
) -> None:
    worker = AsyncPolarRolloutWorker(_worker_args(), data_source=_ReservationSource())
    completed = _completion(75, service_time_seconds=1.0)
    times = iter((10.0, 10.25))
    monkeypatch.setattr(rollout_module.time, "perf_counter", lambda: next(times))

    await worker._emit_completed(completed)
    visible = worker.output_queue.get_nowait()

    assert visible.output_queue_wait_seconds == 0.25


def test_rejected_work_timing_is_retained_as_delta_counters() -> None:
    worker = AsyncPolarRolloutWorker(_worker_args(), data_source=_ReservationSource())
    samples = [
        SimpleNamespace(
            metadata={
                "polar": {
                    "session_id": "wasted-1",
                    "timing": {
                        "e2e_ms": 100.0,
                        "agent_exec_ms": 60.0,
                        "runtime_exec_ms": 70.0,
                    },
                    "trace_index": 0,
                    "trace_metadata": {
                        "completion_metadata": [
                            {"inference_timings": [{"e2e_ms": 50.0, "queue_ms": 10.0}]}
                        ]
                    },
                }
            }
        ),
        # A second trace from the same session must not double-count it.
        SimpleNamespace(
            metadata={
                "polar": {
                    "session_id": "wasted-1",
                    "timing": {"e2e_ms": 100.0},
                }
            }
        ),
    ]

    worker._record_wasted_samples(samples)
    metrics = worker.snapshot_metrics()

    assert metrics["polar/wasted/session_count_delta"] == 1.0
    assert metrics["timing/wasted/e2e_ms_sum_delta"] == 100.0
    assert metrics["timing/wasted/agent_exec_ms_sum_delta"] == 60.0
    assert metrics["timing/wasted/runtime_exec_ms_sum_delta"] == 70.0
    assert metrics["timing/wasted/e2e_ms_per_session_mean"] == 100.0
    assert metrics["polar/wasted/inference_timing_count_delta"] == 1.0
    assert metrics["timing/wasted/inference_e2e_ms_sum_delta"] == 50.0
    assert metrics["timing/wasted/inference_queue_ms_sum_delta"] == 10.0
    assert metrics["timing/wasted/inference_queue_ms_mean"] == 10.0


def test_worker_counters_expose_lifetime_scope_and_per_rollout_delta() -> None:
    source = _ReservationSource()
    reservation_totals = {"reserved": 4.0}
    source.reservation_metrics = lambda: {
        "polar/reservations/reserved_since_worker_start": reservation_totals["reserved"],
        "polar/reservations/outstanding_groups": 1.0,
    }
    worker = AsyncPolarRolloutWorker(_worker_args(), data_source=source)
    worker._inc_metric("polar/completed_groups", 3)

    first = worker.snapshot_metrics()
    assert first["polar/completed_groups_since_worker_start"] == 3.0
    assert first["polar/completed_groups_delta"] == 3.0
    assert "polar/completed_groups" not in first
    assert first["polar/reservations/reserved_since_worker_start"] == 4.0
    assert first["polar/reservations/reserved_delta"] == 4.0

    worker._inc_metric("polar/completed_groups", 2)
    reservation_totals["reserved"] = 7.0
    second = worker.snapshot_metrics()
    assert second["polar/completed_groups_since_worker_start"] == 5.0
    assert second["polar/completed_groups_delta"] == 2.0
    assert second["polar/reservations/reserved_since_worker_start"] == 7.0
    assert second["polar/reservations/reserved_delta"] == 3.0


def test_generate_cancellation_keeps_partially_drained_reservation_outstanding(
    monkeypatch,
) -> None:
    worker = _GenerateWorker([_completion(81)])
    cancellation_checks = iter((False, True))
    stopped = []
    monkeypatch.setattr(rollout_module, "get_global_async_worker", lambda *_args: worker)
    monkeypatch.setattr(
        rollout_module,
        "_current_ray_task_is_canceled",
        lambda: next(cancellation_checks),
    )
    monkeypatch.setattr(rollout_module, "stop_global_worker", lambda: stopped.append(True))
    monkeypatch.setattr(rollout_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(TaskCancelledError):
        generate_rollout_polar_async(
            SimpleNamespace(rollout_batch_size=2),
            rollout_id=1,
            data_source=SimpleNamespace(),
        )

    assert worker.consumed == []
    assert stopped == [True]
