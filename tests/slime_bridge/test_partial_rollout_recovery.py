from __future__ import annotations

import asyncio
import copy
import random
from types import SimpleNamespace

import pytest
from ray.exceptions import TaskCancelledError

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.types import Sample
from slime_bridge.config import resolve_polar_slime_config
from slime_bridge.data_source import CeilEpochRolloutDataSourceWithBuffer
from slime_bridge import partial_rollout as partial_rollout_module
from slime_bridge.partial_rollout import (
    PartialRolloutError,
    PartialRolloutHeader,
    PartialRolloutStore,
    STATE_DROP,
    STATE_KEEP,
    STATE_PREPARED,
    STATE_RESULT_READY,
    _gc_committed_partial_rollouts,
    maybe_open_partial_rollout_store,
    partial_rollout_config_digest,
)
from slime_bridge import rollout as rollout_module
from slime_bridge.rollout import (
    AsyncPolarRolloutWorker,
    _CompletedGroup,
    _PendingGroup,
    _annotate_accepted_samples,
    generate_rollout_polar_async,
)


class _Dataset:
    def __init__(self, prompts: list[str], *, seed: int = 17) -> None:
        self.origin_samples = [Sample(prompt=prompt) for prompt in prompts]
        self.samples = list(self.origin_samples)
        self.seed = seed
        self.epoch_id = -1

    def shuffle(self, new_epoch_id: int) -> None:
        if self.epoch_id == new_epoch_id:
            return
        permutation = list(range(len(self.samples)))
        random.Random(self.seed + new_epoch_id).shuffle(permutation)
        self.samples = [self.origin_samples[index] for index in permutation]
        self.epoch_id = new_epoch_id

    def __len__(self) -> int:
        return len(self.samples)


def _args(tmp_path, *, rollout_id: int = 8, target: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        rollout_global_dataset=True,
        prompt_data=None,
        n_samples_per_prompt=1,
        rollout_shuffle=False,
        rollout_batch_size=target,
        buffer_filter_path=None,
        save=str(tmp_path),
        load=str(tmp_path),
        start_rollout_id=rollout_id,
        polar_rollout_url="http://rollout.invalid:8080",
        polar_task_template={"agent": {"harness": "test", "settings": {}}},
        polar_max_async_level=3,
        polar_fully_async=True,
        update_weights_interval=1,
        polar_callback_host="127.0.0.1",
        reward_key="score",
    )


def _source(args: SimpleNamespace, prompts: list[str]) -> CeilEpochRolloutDataSourceWithBuffer:
    source = CeilEpochRolloutDataSourceWithBuffer(args)
    source.dataset = _Dataset(prompts)
    return source


def _write_model_checkpoint_shell(tmp_path, iteration: int) -> None:
    (tmp_path / "latest_checkpointed_iteration.txt").write_text(f"{iteration}\n", encoding="utf-8")
    iteration_dir = tmp_path / f"iter_{iteration:07d}"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / ".metadata").write_bytes(b"dcp-metadata")
    (iteration_dir / "common.pt").write_bytes(b"common-state")


def _header(tmp_path, *, target: int = 2) -> PartialRolloutHeader:
    return PartialRolloutHeader(
        run_id="test-run",
        rollout_id=8,
        base_checkpoint_iteration=7,
        base_checkpoint_digest="a" * 64,
        config_digest="b" * 64,
        target_groups=target,
    )


def _training_sample(reservation_id: int, *, reward: float = 1.0) -> Sample:
    return Sample(
        group_index=reservation_id,
        index=reservation_id,
        reward={"score": reward},
        response_length=1,
        loss_mask=[1],
        metadata={
            "polar": {
                "session_id": f"session-{reservation_id}",
                "session_status": "COMPLETED",
                "placeholder": False,
            }
        },
        status=Sample.Status.COMPLETED,
    )


def _completed(
    reservation_id: int,
    group: list[Sample],
    store: PartialRolloutStore,
) -> _CompletedGroup:
    return _CompletedGroup(
        group_id=reservation_id,
        group=group,
        reservation_id=reservation_id,
        samples=[_training_sample(reservation_id)],
        task_id=f"task-{reservation_id}",
        submitted_rollout_id=8,
        policy_version=8,
        session_count=1,
        partial_store=store,
    )


def _record_result(
    store: PartialRolloutStore,
    reservation_id: int,
    group: list[Sample],
) -> _CompletedGroup:
    store.record_prepared(
        reservation_id=reservation_id,
        group=group,
        submitted_rollout_id=8,
        policy_version=8,
    )
    completed = _completed(reservation_id, group, store)
    store.record_result_ready(completed)
    return completed


def test_wal_state_machine_round_trip_and_ready_marker(tmp_path) -> None:
    store = PartialRolloutStore(tmp_path / "wal", _header(tmp_path, target=1))
    group = [Sample(group_index=0, index=0, prompt="p0")]
    completed = _record_result(store, 0, group)
    _annotate_accepted_samples(
        completed.samples,
        accepted_rollout_id=8,
        staleness=0,
        policy_version=8,
        scheduler_group_id=0,
    )
    store.record_keep(completed, accepted_rollout_id=8)

    records = store.load_records()
    assert [record["state"] for record in records] == [STATE_KEEP]
    assert store.load_sample_dicts(records[0])[0]["group_index"] == 0

    store.mark_ready([0])
    assert store.sealed is True
    reopened = PartialRolloutStore(store.directory, store.header)
    assert reopened.load_records()[0]["state"] == STATE_KEEP
    assert reopened.sealed is True


def test_wal_rejects_conflicting_terminal_transition(tmp_path) -> None:
    store = PartialRolloutStore(tmp_path / "wal", _header(tmp_path, target=1))
    group = [Sample(group_index=0, index=0, prompt="p0")]
    completed = _record_result(store, 0, group)
    _annotate_accepted_samples(
        completed.samples,
        accepted_rollout_id=8,
        staleness=0,
        policy_version=8,
        scheduler_group_id=0,
    )
    store.record_keep(completed, accepted_rollout_id=8)

    with pytest.raises(PartialRolloutError, match="cannot DROP kept"):
        store.record_drop(0, outcome="dynamic_filter", reason="zero_std")


def test_duplicate_result_is_idempotent_only_for_identical_payload(tmp_path) -> None:
    store = PartialRolloutStore(tmp_path / "wal", _header(tmp_path, target=1))
    group = [Sample(group_index=0, index=0, prompt="p0")]
    completed = _record_result(store, 0, group)
    store.record_result_ready(completed)

    conflicting = _completed(0, group, store)
    conflicting.samples[0].reward = {"score": 0.0}
    with pytest.raises(PartialRolloutError, match="conflicting result payload"):
        store.record_result_ready(conflicting)


def test_orphan_blob_after_crash_does_not_advance_prepared_decision(tmp_path) -> None:
    store = PartialRolloutStore(tmp_path / "wal", _header(tmp_path, target=1))
    group = [Sample(group_index=0, index=0, prompt="p0")]
    store.record_prepared(
        reservation_id=0,
        group=group,
        submitted_rollout_id=8,
        policy_version=8,
    )
    store._write_sample_blob(0, [_training_sample(0)])

    records = store.load_records()
    assert records[0]["state"] == STATE_PREPARED
    assert list(store.directory.glob("samples_*.pt"))


def test_atomic_record_failure_before_replace_leaves_no_partial_record(
    monkeypatch,
    tmp_path,
) -> None:
    store = PartialRolloutStore(tmp_path / "wal", _header(tmp_path, target=1))
    group = [Sample(group_index=0, index=0, prompt="p0")]

    def fail_replace(_source, _destination):
        raise OSError("injected pre-replace failure")

    monkeypatch.setattr(partial_rollout_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        store.record_prepared(
            reservation_id=0,
            group=group,
            submitted_rollout_id=8,
            policy_version=8,
        )

    assert store.load_records() == []
    assert list(store.directory.glob("*.tmp.*")) == []


def test_atomic_record_is_loadable_if_directory_fsync_reports_failure(
    monkeypatch,
    tmp_path,
) -> None:
    store = PartialRolloutStore(tmp_path / "wal", _header(tmp_path, target=1))
    group = [Sample(group_index=0, index=0, prompt="p0")]

    def fail_directory_fsync(_path):
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(partial_rollout_module, "_fsync_directory", fail_directory_fsync)
    with pytest.raises(OSError, match="injected"):
        store.record_prepared(
            reservation_id=0,
            group=group,
            submitted_rollout_id=8,
            policy_version=8,
        )

    assert store.load_records()[0]["state"] == STATE_PREPARED


def test_corrupt_sample_blob_is_fail_closed_and_quarantinable(tmp_path) -> None:
    store = PartialRolloutStore(tmp_path / "wal", _header(tmp_path, target=1))
    group = [Sample(group_index=0, index=0, prompt="p0")]
    _record_result(store, 0, group)
    record = store.load_records()[0]
    blob_path = store.directory / record["sample_blob"]["name"]
    blob_path.write_bytes(b"truncated")

    with pytest.raises(PartialRolloutError, match="missing or corrupt"):
        store.load_records()
    quarantined = store.quarantine("corrupt blob")
    assert quarantined is not None and quarantined.is_dir()
    assert not store.directory.exists()


def test_config_fingerprint_tracks_token_limits_but_redacts_credentials(tmp_path) -> None:
    args = _args(tmp_path, target=1)
    args.polar_task_template["runtime"] = {
        "env": {"NO_PROXY": "node-a,10.0.0.1", "STABLE": "value"},
        "kwargs": {
            "volumes": [
                "/tmp/polar-cache-job-a/uds/gateway:/polar/gateway:ro",
                "/stable/runtime:/opt/runtime:ro",
            ]
        },
        "internet_volumes": ["/tmp/polar-cache-job-a/uds/proxy:/polar/proxy:ro"],
    }
    args.polar_task_template["agent"]["settings"] = {
        "model_kwargs": {"max_tokens": 1024, "api_key": "secret-a"}
    }
    config = resolve_polar_slime_config(args)
    baseline = partial_rollout_config_digest(args, config)

    credential_change = copy.deepcopy(args)
    credential_change.polar_task_template["agent"]["settings"]["model_kwargs"]["api_key"] = (
        "secret-b"
    )
    assert (
        partial_rollout_config_digest(
            credential_change,
            resolve_polar_slime_config(credential_change),
        )
        == baseline
    )

    allocation_change = copy.deepcopy(args)
    allocation_change.polar_task_template["runtime"]["env"]["NO_PROXY"] = "node-b,10.0.0.2"
    allocation_change.polar_task_template["runtime"]["kwargs"]["volumes"][0] = (
        "/tmp/polar-cache-job-b/uds/gateway:/polar/gateway:ro"
    )
    allocation_change.polar_task_template["runtime"]["internet_volumes"][0] = (
        "/tmp/polar-cache-job-b/uds/proxy:/polar/proxy:ro"
    )
    assert (
        partial_rollout_config_digest(
            allocation_change,
            resolve_polar_slime_config(allocation_change),
        )
        == baseline
    )

    semantic_change = copy.deepcopy(args)
    semantic_change.polar_task_template["agent"]["settings"]["model_kwargs"]["max_tokens"] = 2048
    assert (
        partial_rollout_config_digest(
            semantic_change,
            resolve_polar_slime_config(semantic_change),
        )
        != baseline
    )


def test_committed_wal_gc_never_deletes_current_or_uncommitted_rollout(tmp_path) -> None:
    parent = tmp_path / "partial_rollout_wal"
    for rollout_id in (7, 8, 9):
        path = parent / f"rollout_{rollout_id:07d}"
        path.mkdir(parents=True)
        (path / "sentinel").write_text("x", encoding="utf-8")

    _gc_committed_partial_rollouts(
        parent,
        committed_iteration=7,
        current_rollout_id=8,
    )

    assert not (parent / "rollout_0000007").exists()
    assert (parent / "rollout_0000008").is_dir()
    assert (parent / "rollout_0000009").is_dir()


def test_rebuild_mixed_contiguous_states_without_seeking_or_skipping(tmp_path) -> None:
    args = _args(tmp_path, target=2)
    writer = _source(args, [f"p{i}" for i in range(8)])
    store = PartialRolloutStore(tmp_path / "wal", _header(tmp_path, target=2))
    reservations = writer.get_samples_with_reservation(4)

    store.record_prepared(
        reservation_id=0,
        group=reservations[0][1],
        submitted_rollout_id=8,
        policy_version=8,
    )
    ready = _record_result(store, 1, reservations[1][1])
    dropped = _record_result(store, 2, reservations[2][1])
    store.record_drop(2, outcome="dynamic_filter", reason="zero_std_1.0")
    kept = _record_result(store, 3, reservations[3][1])
    _annotate_accepted_samples(
        kept.samples,
        accepted_rollout_id=8,
        staleness=0,
        policy_version=8,
        scheduler_group_id=3,
    )
    store.record_keep(kept, accepted_rollout_id=8)

    restored = _source(args, [f"p{i}" for i in range(8)])
    records = store.load_records()
    rebuilt = restored.rebuild_partial_reservations(records)

    assert [record["state"] for record, _group in rebuilt] == [
        STATE_PREPARED,
        STATE_RESULT_READY,
        STATE_DROP,
        STATE_KEEP,
    ]
    assert restored.sample_group_index == 4
    assert restored.reservation_metrics()["polar/reservations/outstanding_groups"] == 4.0
    restored.mark_consumed(2, outcome="dynamic_filter")
    assert restored.reservation_metrics()["polar/reservations/outstanding_groups"] == 3.0
    assert ready.reservation_id == 1 and dropped.reservation_id == 2


def test_rebuild_prompt_mismatch_rolls_back_every_cursor_and_counter(tmp_path) -> None:
    args = _args(tmp_path, target=1)
    writer = _source(args, ["expected", "next"])
    store = PartialRolloutStore(tmp_path / "wal", _header(tmp_path, target=1))
    reservation_id, group = writer.get_samples_with_reservation(1)[0]
    store.record_prepared(
        reservation_id=reservation_id,
        group=group,
        submitted_rollout_id=8,
        policy_version=8,
    )
    records = store.load_records()

    restored = _source(args, ["different", "next"])
    with pytest.raises(RuntimeError, match="prompt digest mismatch"):
        restored.rebuild_partial_reservations(records)

    assert restored.sample_group_index == 0
    metrics = restored.reservation_metrics()
    assert metrics["polar/reservations/outstanding_groups"] == 0.0
    assert metrics["polar/reservations/reserved_since_worker_start"] == 0.0


def test_worker_writes_prepared_before_returning_reserved_work(tmp_path) -> None:
    args = _args(tmp_path, target=1)
    source = _source(args, ["p0", "p1"])
    store = PartialRolloutStore(tmp_path / "wal", _header(tmp_path, target=1))
    worker = AsyncPolarRolloutWorker(args, source, partial_store=store)
    worker.set_rollout_context(8)
    worker.request_groups(1)

    deferred = worker._next_group_for_submission()

    assert deferred is not None and deferred.reservation_id == 0
    assert store.load_records()[0]["state"] == STATE_PREPARED


def test_full_ready_recovery_initializes_async_credit_without_over_admission(tmp_path) -> None:
    args = _args(tmp_path, target=8)
    source = _source(args, ["p0"])
    worker = AsyncPolarRolloutWorker(args, source)
    worker.bootstrap_partial_recovery(
        deferred=[SimpleNamespace() for _ in range(16)],
        completed=[],
        held_keep_count=8,
    )

    worker.request_groups(0)
    first = worker.snapshot_metrics()
    assert first["polar/scheduler/admission_credit"] == 0.0
    assert first["polar/scheduler/recovered_held_groups"] == 8.0

    worker.request_groups(8)
    second = worker.snapshot_metrics()
    assert second["polar/scheduler/admission_credit"] == 8.0


def test_dynamic_drop_is_durable_before_reservation_consumption(tmp_path) -> None:
    args = _args(tmp_path, target=1)
    source = _source(args, ["p0", "p1"])
    store = PartialRolloutStore(tmp_path / "wal", _header(tmp_path, target=1))
    reservation_id, group = source.get_samples_with_reservation(1)[0]
    completed = _record_result(store, reservation_id, group)
    real_mark_consumed = source.mark_consumed

    def assert_wal_then_consume(reservation_id: int, *, outcome: str):
        assert store.load_records()[0]["state"] == STATE_DROP
        return real_mark_consumed(reservation_id, outcome=outcome)

    source.mark_consumed = assert_wal_then_consume
    worker = AsyncPolarRolloutWorker(args, source, partial_store=store)
    worker.mark_dynamic_filter_drop(completed, reason="zero_std_1.0")

    record = store.load_records()[0]
    assert record["state"] == STATE_DROP
    assert record["drop_outcome"] == "dynamic_filter"
    assert source.reservation_metrics()["polar/reservations/outstanding_groups"] == 0.0


def test_submit_attempt_records_result_ready_before_return(
    monkeypatch,
    tmp_path,
) -> None:
    args = _args(tmp_path, target=1)
    source = _source(args, ["p0", "p1"])
    store = PartialRolloutStore(tmp_path / "wal", _header(tmp_path, target=1))
    reservation_id, group = source.get_samples_with_reservation(1)[0]
    store.record_prepared(
        reservation_id=reservation_id,
        group=group,
        submitted_rollout_id=8,
        policy_version=8,
    )
    pending = _PendingGroup(
        group_id=0,
        group=group,
        reservation_id=reservation_id,
        submitted_rollout_id=8,
        policy_version=8,
        session_cost=1,
        partial_store=store,
    )
    worker = AsyncPolarRolloutWorker(args, source, partial_store=store)
    task_result = SimpleNamespace(
        task_id="task-0",
        status="completed",
        results=[object()],
    )

    monkeypatch.setattr(
        rollout_module,
        "_build_task_payload",
        lambda **_kwargs: {"task_id": "task-0"},
    )

    async def return_result(_client, _payload):
        return task_result

    monkeypatch.setattr(worker, "_submit_with_callback", return_result)
    monkeypatch.setattr(
        rollout_module,
        "_convert_task_result_to_samples",
        lambda *_args, **_kwargs: [_training_sample(0)],
    )

    completed = asyncio.run(worker._submit_attempt(SimpleNamespace(), pending))

    assert completed.reservation_id == 0
    assert store.load_records()[0]["state"] == STATE_RESULT_READY


class _RecoveredGenerateWorker:
    def __init__(self, plan, source) -> None:
        self.config = SimpleNamespace(reward_key="score")
        self.plan = plan
        self.source = source
        self.requested: list[int] = []
        self.completions: list[_CompletedGroup] = []
        self.released = False
        for deferred in plan.deferred:
            completed = _completed(
                int(deferred.reservation_id),
                deferred.group,
                plan.store,
            )
            _annotate_accepted_samples(
                completed.samples,
                accepted_rollout_id=8,
                staleness=0,
                policy_version=8,
                scheduler_group_id=completed.group_id,
            )
            plan.store.record_result_ready(completed)
            self.completions.append(completed)

    def set_rollout_context(self, _rollout_id: int) -> None:
        pass

    def request_groups(self, count: int) -> None:
        self.requested.append(count)

    def drain_completed(self, *, max_groups: int, rollout_id: int):
        del rollout_id
        result = self.completions[:max_groups]
        del self.completions[:max_groups]
        return result

    def queue_size(self) -> int:
        return len(self.completions)

    def snapshot_metrics(self):
        metrics = dict(self.source.reservation_metrics())
        metrics["polar/reservations/consumed_delta"] = 8.0
        metrics["polar/reservations/consumed_accepted_delta"] = 8.0
        return metrics

    def _consume_reservations(self, reservation_ids, *, outcome: str):
        return self.source.mark_consumed_many(
            [int(item) for item in reservation_ids if item is not None],
            outcome=outcome,
        )

    def release_recovered_holds(self) -> None:
        self.released = True


class _InterruptingGenerateWorker:
    def __init__(self, plan, source) -> None:
        self.config = SimpleNamespace(reward_key="score")
        self.source = source
        reservation_id, group = source.get_samples_with_reservation(1)[0]
        plan.store.record_prepared(
            reservation_id=reservation_id,
            group=group,
            submitted_rollout_id=8,
            policy_version=8,
        )
        completed = _completed(reservation_id, group, plan.store)
        plan.store.record_result_ready(completed)
        _annotate_accepted_samples(
            completed.samples,
            accepted_rollout_id=8,
            staleness=0,
            policy_version=8,
            scheduler_group_id=reservation_id,
        )
        self.completions = [completed]

    def set_rollout_context(self, _rollout_id: int) -> None:
        pass

    def request_groups(self, _count: int) -> None:
        pass

    def drain_completed(self, *, max_groups: int, rollout_id: int):
        del rollout_id
        result = self.completions[:max_groups]
        del self.completions[:max_groups]
        return result

    def queue_size(self) -> int:
        return len(self.completions)


def test_interrupted_partial_batch_returns_no_train_data_and_recovers_keep(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("RUN_ID", "test-run")
    args = _args(tmp_path, target=2)
    prompts = ["p0", "p1", "p2"]
    source = _source(args, prompts)
    source.save(7)
    _write_model_checkpoint_shell(tmp_path, 7)
    captured = {}

    def fake_get_worker(_args, data_source, plan, rollout_id):
        assert rollout_id == 8
        captured["store"] = plan.store
        return _InterruptingGenerateWorker(plan, data_source)

    cancellation_checks = iter((False, True))
    monkeypatch.setattr(rollout_module, "get_global_async_worker", fake_get_worker)
    monkeypatch.setattr(
        rollout_module,
        "_current_ray_task_is_canceled",
        lambda: next(cancellation_checks),
    )
    monkeypatch.setattr(rollout_module, "stop_global_worker", lambda: None)
    monkeypatch.setattr(rollout_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(TaskCancelledError):
        generate_rollout_polar_async(args, rollout_id=8, data_source=source)

    store = captured["store"]
    records = store.load_records()
    assert [record["state"] for record in records] == [STATE_KEEP]
    assert not store._ready_path().exists()
    assert source.reservation_metrics()["polar/reservations/outstanding_groups"] == 1.0
    assert not (tmp_path / "rollout" / "rollout_metrics_journal").exists()

    restored = _source(args, prompts)
    restored.load(7)
    recovery = rollout_module._prepare_partial_recovery(
        args,
        rollout_id=8,
        data_source=restored,
    )
    assert len(recovery.kept) == 1
    assert recovery.kept[0].reservation_id == 0


def test_kill_after_six_of_eight_recovers_six_and_generates_only_two(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("RUN_ID", "test-run")
    args = _args(tmp_path, target=8)
    prompts = [f"p{i}" for i in range(20)]
    writer = _source(args, prompts)
    writer.save(7)
    _write_model_checkpoint_shell(tmp_path, 7)

    config = resolve_polar_slime_config(args)
    store = maybe_open_partial_rollout_store(args, config, 8)
    assert store is not None
    reservations = writer.get_samples_with_reservation(8)
    for reservation_id, group in reservations:
        store.record_prepared(
            reservation_id=reservation_id,
            group=group,
            submitted_rollout_id=8,
            policy_version=8,
        )
        if reservation_id < 6:
            completed = _completed(reservation_id, group, store)
            store.record_result_ready(completed)
            _annotate_accepted_samples(
                completed.samples,
                accepted_rollout_id=8,
                staleness=0,
                policy_version=8,
                scheduler_group_id=reservation_id,
            )
            store.record_keep(completed, accepted_rollout_id=8)

    restored = _source(args, prompts)
    restored.load(7)
    captured = {}

    def fake_get_worker(_args, data_source, plan, rollout_id):
        assert data_source is restored
        assert rollout_id == 8
        captured["plan"] = plan
        worker = _RecoveredGenerateWorker(plan, restored)
        captured["worker"] = worker
        return worker

    monkeypatch.setattr(rollout_module, "get_global_async_worker", fake_get_worker)
    monkeypatch.setattr(rollout_module, "_current_ray_task_is_canceled", lambda: False)
    monkeypatch.setattr(
        rollout_module,
        "_load_rollout_train_output_type",
        lambda: RolloutFnTrainOutput,
    )

    output = generate_rollout_polar_async(args, rollout_id=8, data_source=restored)

    plan = captured["plan"]
    worker = captured["worker"]
    assert len(plan.kept) == 6
    assert len(plan.deferred) == 2
    assert worker.requested == [2]
    assert worker.released is True
    assert len(output.samples) == 8
    assert {group[0].group_index for group in output.samples} == set(range(8))
    assert restored.reservation_metrics()["polar/reservations/outstanding_groups"] == 0.0
    records = store.load_records()
    assert [record["state"] for record in records] == [STATE_KEEP] * 8
    assert store._ready_path().is_file()


def test_stale_policy_record_is_quarantined_before_reservation_rebuild(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("RUN_ID", "test-run")
    args = _args(tmp_path, target=1)
    writer = _source(args, ["p0", "p1"])
    writer.save(7)
    _write_model_checkpoint_shell(tmp_path, 7)
    store = maybe_open_partial_rollout_store(args, resolve_polar_slime_config(args), 8)
    assert store is not None
    reservation_id, group = writer.get_samples_with_reservation(1)[0]
    store.record_prepared(
        reservation_id=reservation_id,
        group=group,
        submitted_rollout_id=8,
        policy_version=8,
    )
    completed = _completed(reservation_id, group, store)
    store.record_result_ready(completed)
    record_path = store._record_path(reservation_id)
    record = partial_rollout_module.torch.load(record_path, weights_only=False)
    record["policy_version"] = 0
    partial_rollout_module._atomic_torch_save(
        store._with_record_digest(record),
        record_path,
    )

    restored = _source(args, ["p0", "p1"])
    restored.load(7)
    plan = rollout_module._prepare_partial_recovery(
        args,
        rollout_id=8,
        data_source=restored,
    )

    assert plan.store is not None
    assert plan.kept == [] and plan.deferred == [] and plan.result_ready == []
    assert restored.sample_group_index == 0
    assert list(store.directory.parent.glob(f"{store.directory.name}.invalid.*"))
