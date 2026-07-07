from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
from pathlib import Path
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
    committed_reservation_evidence_digest,
    load_checkpoint_replay_committed_reservation_ids,
    load_emitted_reservation_ids,
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
    prompt_data = args.prompt_data
    args.prompt_data = None
    try:
        source = CeilEpochRolloutDataSourceWithBuffer(args)
    finally:
        args.prompt_data = prompt_data
    source.dataset = _Dataset(prompts)
    return source


def _write_model_checkpoint_shell(tmp_path, iteration: int) -> None:
    (tmp_path / "latest_checkpointed_iteration.txt").write_text(f"{iteration}\n", encoding="utf-8")
    iteration_dir = tmp_path / f"iter_{iteration:07d}"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / ".metadata").write_bytes(b"dcp-metadata")
    (iteration_dir / "common.pt").write_bytes(b"common-state")


def _write_release_seed(tmp_path, *, seed: bytes = b"release-seed") -> None:
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release\n", encoding="utf-8")
    release = tmp_path / "release"
    release.mkdir(parents=True, exist_ok=True)
    (release / ".metadata").write_bytes(b"dcp-metadata:" + seed)
    (release / "common.pt").write_bytes(b"common-state:" + seed)
    (release / "metadata.json").write_text(
        '{"sharded_backend":"torch_dist","sharded_backend_version":1,'
        '"common_backend":"torch","common_backend_version":1}\n',
        encoding="utf-8",
    )
    (release / "__0_0.distcp").write_bytes(b"model-shard:" + seed)


def _write_prompt_data(tmp_path, prompts: list[str]) -> str:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    return str(path)


def _write_emitted_rollout_journal(
    tmp_path,
    iteration: int,
    reservation_ids: list[int],
) -> None:
    journal_dir = tmp_path / "rollout" / "rollout_metrics_journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        Sample(group_index=reservation_id, prompt=f"p-{reservation_id}")
        for reservation_id in reservation_ids
    ]
    partial_rollout_module.torch.save(
        {
            "version": 1,
            "rollout_id": iteration,
            "pending": SimpleNamespace(samples=samples),
        },
        journal_dir / f"rollout_{iteration:07d}.pending.pt",
    )
    (journal_dir / f"rollout_{iteration:07d}.emitted").write_text(
        f"version=1\nrollout_id={iteration}\n",
        encoding="utf-8",
    )


def _header(
    tmp_path,
    *,
    target: int = 2,
    committed_reservation_ids=(),
) -> PartialRolloutHeader:
    committed_reservation_ids = frozenset(committed_reservation_ids)
    live_frontier = max(committed_reservation_ids, default=-1) + 1
    return PartialRolloutHeader(
        run_id="test-run",
        rollout_id=8,
        base_checkpoint_kind="numeric",
        base_checkpoint_iteration=7,
        base_checkpoint_digest="a" * 64,
        config_digest="b" * 64,
        source_digest="c" * 64,
        data_digest="d" * 64,
        target_groups=target,
        replay_start_reservation_id=0,
        replay_live_frontier=live_frontier,
        committed_reservation_count=len(committed_reservation_ids),
        dedupe_evidence_digest=committed_reservation_evidence_digest(
            0,
            live_frontier,
            committed_reservation_ids,
        ),
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
    rollout_id = int(store.header.rollout_id)
    return _CompletedGroup(
        group_id=reservation_id,
        group=group,
        reservation_id=reservation_id,
        samples=[_training_sample(reservation_id)],
        task_id=f"task-{reservation_id}",
        submitted_rollout_id=rollout_id,
        policy_version=rollout_id,
        session_count=1,
        partial_store=store,
    )


def _record_result(
    store: PartialRolloutStore,
    reservation_id: int,
    group: list[Sample],
) -> _CompletedGroup:
    rollout_id = int(store.header.rollout_id)
    store.record_prepared(
        reservation_id=reservation_id,
        group=group,
        submitted_rollout_id=rollout_id,
        policy_version=rollout_id,
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


def test_wal_rejects_records_from_changed_dedupe_evidence(tmp_path) -> None:
    committed_ids = frozenset({0})
    header = _header(
        tmp_path,
        target=1,
        committed_reservation_ids=committed_ids,
    )
    store = PartialRolloutStore(
        tmp_path / "wal",
        header,
        committed_reservation_ids=committed_ids,
    )
    group = [Sample(group_index=0, index=0, prompt="p0")]
    store.record_prepared(
        reservation_id=0,
        group=group,
        submitted_rollout_id=8,
        policy_version=8,
    )
    store.record_resume_duplicate(0)

    changed_header = dataclasses.replace(
        header,
        committed_reservation_count=0,
        dedupe_evidence_digest=committed_reservation_evidence_digest(0, 1, ()),
    )
    reopened = PartialRolloutStore(
        store.directory,
        changed_header,
        committed_reservation_ids=(),
    )
    with pytest.raises(PartialRolloutError, match="header mismatch"):
        reopened.load_records()


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


def test_emitted_journal_extracts_exact_reservation_ids_and_ignores_speculation(
    tmp_path,
) -> None:
    _write_emitted_rollout_journal(tmp_path, 6, [1, 1, 2, 2])
    _write_emitted_rollout_journal(tmp_path, 7, [3, 3, 7, 7])

    assert load_emitted_reservation_ids(
        tmp_path,
        7,
        expected_group_count=2,
    ) == frozenset({1, 2, 3, 7})

    journal_dir = tmp_path / "rollout" / "rollout_metrics_journal"
    partial_rollout_module.torch.save(
        {
            "version": 1,
            "rollout_id": 8,
            "pending": SimpleNamespace(samples=[Sample(group_index=101, prompt="speculative")]),
        },
        journal_dir / "rollout_0000008.pending.pt",
    )
    assert load_emitted_reservation_ids(tmp_path, 8) == frozenset({1, 2, 3, 7})


def test_malformed_emitted_journal_fails_closed(tmp_path) -> None:
    _write_emitted_rollout_journal(tmp_path, 7, [3])
    pending_path = tmp_path / "rollout" / "rollout_metrics_journal" / "rollout_0000007.pending.pt"
    partial_rollout_module.torch.save(
        {
            "version": 1,
            "rollout_id": 7,
            "pending": SimpleNamespace(samples=None),
        },
        pending_path,
    )

    with pytest.raises(PartialRolloutError, match="does not retain samples"):
        load_emitted_reservation_ids(tmp_path, 7)


def test_checkpoint_dedupe_covers_emitted_steps_behind_long_lived_frontier(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("RUN_ID", "test-run")
    args = _args(tmp_path, rollout_id=8, target=2)
    source = _source(args, [f"p{i}" for i in range(8)])
    reservations = source.get_samples_with_reservation(7)
    source.mark_consumed_many(list(range(1, 7)), outcome="accepted")
    source.save(7)
    _write_model_checkpoint_shell(tmp_path, 7)
    _write_emitted_rollout_journal(tmp_path, 5, [1, 2])
    _write_emitted_rollout_journal(tmp_path, 6, [3, 4])
    _write_emitted_rollout_journal(tmp_path, 7, [5, 6])

    bounds, committed_ids = load_checkpoint_replay_committed_reservation_ids(
        tmp_path,
        7,
        expected_group_count=2,
    )

    store = maybe_open_partial_rollout_store(
        args,
        resolve_polar_slime_config(args),
        8,
    )

    assert [reservation_id for reservation_id, _group in reservations] == list(range(7))
    assert bounds == (0, 7)
    assert committed_ids == frozenset(range(1, 7))
    assert store is not None
    assert store.committed_reservation_ids == frozenset(range(1, 7))


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


def test_resume_dedupe_skips_committed_ids_without_credit_or_cross_epoch_skip(
    tmp_path,
) -> None:
    args = _args(tmp_path, target=1)
    source = _source(args, ["same-prompt-every-epoch"])
    store = PartialRolloutStore(
        tmp_path / "wal",
        _header(tmp_path, target=1, committed_reservation_ids={0, 1}),
        committed_reservation_ids={0, 1},
    )
    worker = AsyncPolarRolloutWorker(args, source, partial_store=store)
    worker.set_rollout_context(8)
    worker.request_groups(1)

    deferred = worker._next_group_for_submission()

    assert deferred is not None and deferred.reservation_id == 2
    assert deferred.group[0].prompt == "same-prompt-every-epoch"
    records = store.load_records()
    assert [record["state"] for record in records] == [
        STATE_DROP,
        STATE_DROP,
        STATE_PREPARED,
    ]
    assert [record.get("drop_outcome") for record in records[:2]] == [
        "resume_duplicate",
        "resume_duplicate",
    ]
    reservation_metrics = source.reservation_metrics()
    assert reservation_metrics["polar/reservations/reserved_since_worker_start"] == 3.0
    assert (
        reservation_metrics["polar/reservations/consumed_resume_duplicate_since_worker_start"]
        == 2.0
    )
    assert reservation_metrics["polar/reservations/outstanding_groups"] == 1.0

    metrics = worker.snapshot_metrics()
    assert metrics["polar/scheduler/admission_credit"] == 2.0
    assert metrics["polar/resume_duplicate_groups_delta"] == 2.0
    assert metrics["polar/resume_duplicate_sessions_delta"] == 2.0
    assert metrics["polar/reservations/consumed_resume_duplicate_delta"] == 2.0


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
        rollout_id = int(plan.store.header.rollout_id)
        for deferred in plan.deferred:
            completed = _completed(
                int(deferred.reservation_id),
                deferred.group,
                plan.store,
            )
            _annotate_accepted_samples(
                completed.samples,
                accepted_rollout_id=rollout_id,
                staleness=0,
                policy_version=rollout_id,
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
        rollout_id = int(plan.store.header.rollout_id)
        plan.store.record_prepared(
            reservation_id=reservation_id,
            group=group,
            submitted_rollout_id=rollout_id,
            policy_version=rollout_id,
        )
        completed = _completed(reservation_id, group, plan.store)
        plan.store.record_result_ready(completed)
        _annotate_accepted_samples(
            completed.samples,
            accepted_rollout_id=rollout_id,
            staleness=0,
            policy_version=rollout_id,
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


def test_release_seed_rollout_zero_kill_after_six_of_eight_recovers_exactly_two(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("RUN_ID", "release-seed-run")
    monkeypatch.delenv("TMAX_TRAIN_DATA_SHA256", raising=False)
    args = _args(tmp_path, rollout_id=0, target=8)
    prompts = [f"p{i}" for i in range(20)]
    args.prompt_data = _write_prompt_data(tmp_path, prompts)
    _write_release_seed(tmp_path)
    writer = _source(args, prompts)

    store = maybe_open_partial_rollout_store(
        args,
        resolve_polar_slime_config(args),
        0,
    )
    assert store is not None
    assert store.header.base_checkpoint_kind == "release"
    assert store.header.base_checkpoint_iteration == -1
    assert len(store.header.base_checkpoint_digest) == 64
    assert len(store.header.config_digest) == 64
    assert len(store.header.source_digest) == 64
    assert len(store.header.data_digest) == 64
    assert store.header.replay_start_reservation_id == 0
    assert store.header.replay_live_frontier == 0
    assert store.header.committed_reservation_count == 0
    assert store.committed_reservation_ids == frozenset()
    assert store.header.dedupe_evidence_digest == committed_reservation_evidence_digest(0, 0, ())

    reservations = writer.get_samples_with_reservation(8)
    for reservation_id, group in reservations:
        store.record_prepared(
            reservation_id=reservation_id,
            group=group,
            submitted_rollout_id=0,
            policy_version=0,
        )
        if reservation_id < 6:
            completed = _completed(reservation_id, group, store)
            store.record_result_ready(completed)
            _annotate_accepted_samples(
                completed.samples,
                accepted_rollout_id=0,
                staleness=0,
                policy_version=0,
                scheduler_group_id=reservation_id,
            )
            store.record_keep(completed, accepted_rollout_id=0)

    restored = _source(args, prompts)
    captured = {}

    def fake_get_worker(_args, data_source, plan, rollout_id):
        assert data_source is restored
        assert rollout_id == 0
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

    output = generate_rollout_polar_async(args, rollout_id=0, data_source=restored)

    plan = captured["plan"]
    worker = captured["worker"]
    assert len(plan.kept) == 6
    assert len(plan.deferred) == 2
    assert worker.requested == [2]
    assert worker.released is True
    assert len(output.samples) == 8
    assert {group[0].group_index for group in output.samples} == set(range(8))
    assert store._ready_path().is_file()
    # READY only makes optimizer input reconstructable; it never commits a
    # model update. Until actor checkpoint 0, the model pointer stays release.
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text().strip() == ("release")


def test_release_seed_change_quarantines_rollout_zero_wal(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("RUN_ID", "release-seed-run")
    monkeypatch.delenv("TMAX_TRAIN_DATA_SHA256", raising=False)
    args = _args(tmp_path, rollout_id=0, target=1)
    prompts = ["p0", "p1"]
    args.prompt_data = _write_prompt_data(tmp_path, prompts)
    _write_release_seed(tmp_path, seed=b"seed-a")
    writer = _source(args, prompts)
    store = maybe_open_partial_rollout_store(
        args,
        resolve_polar_slime_config(args),
        0,
    )
    assert store is not None
    reservation_id, group = writer.get_samples_with_reservation(1)[0]
    store.record_prepared(
        reservation_id=reservation_id,
        group=group,
        submitted_rollout_id=0,
        policy_version=0,
    )

    _write_release_seed(tmp_path, seed=b"seed-b")
    restored = _source(args, prompts)
    plan = rollout_module._prepare_partial_recovery(
        args,
        rollout_id=0,
        data_source=restored,
    )

    assert plan.store is not None
    assert plan.kept == [] and plan.result_ready == [] and plan.deferred == []
    assert not store.directory.exists()
    assert list(store.directory.parent.glob(f"{store.directory.name}.invalid.*"))
    assert restored.sample_group_index == 0


def test_release_rollout_zero_header_binds_config_source_and_data_identity(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("RUN_ID", "release-seed-run")
    monkeypatch.delenv("TMAX_TRAIN_DATA_SHA256", raising=False)
    args = _args(tmp_path, rollout_id=0, target=1)
    prompt_path = _write_prompt_data(tmp_path, ["p0"])
    args.prompt_data = prompt_path
    _write_release_seed(tmp_path)

    baseline = maybe_open_partial_rollout_store(
        args,
        resolve_polar_slime_config(args),
        0,
    )
    assert baseline is not None

    args.rollout_max_response_len = 1234
    config_changed = maybe_open_partial_rollout_store(
        args,
        resolve_polar_slime_config(args),
        0,
    )
    assert config_changed is not None
    assert config_changed.header.config_digest != baseline.header.config_digest

    args.rollout_max_response_len = None
    monkeypatch.setenv("TMAX_PRORL_GIT_COMMIT", "f" * 40)
    source_changed = maybe_open_partial_rollout_store(
        args,
        resolve_polar_slime_config(args),
        0,
    )
    assert source_changed is not None
    assert source_changed.header.source_digest != baseline.header.source_digest

    monkeypatch.delenv("TMAX_PRORL_GIT_COMMIT")
    Path(prompt_path).write_text(json.dumps({"prompt": "changed"}) + "\n")
    data_changed = maybe_open_partial_rollout_store(
        args,
        resolve_polar_slime_config(args),
        0,
    )
    assert data_changed is not None
    assert data_changed.header.data_digest != baseline.header.data_digest


@pytest.mark.parametrize("tracker_value", ["0", "7", "", "not-a-release"])
def test_nonrelease_rollout_zero_never_opens_partial_wal(
    monkeypatch,
    tmp_path,
    tracker_value,
) -> None:
    monkeypatch.setenv("RUN_ID", "not-release-run")
    monkeypatch.delenv("TMAX_TRAIN_DATA_SHA256", raising=False)
    args = _args(tmp_path, rollout_id=0, target=1)
    args.prompt_data = _write_prompt_data(tmp_path, ["p0"])
    (tmp_path / "latest_checkpointed_iteration.txt").write_text(
        tracker_value + "\n", encoding="utf-8"
    )

    store = maybe_open_partial_rollout_store(
        args,
        resolve_polar_slime_config(args),
        0,
    )

    assert store is None
    assert not (tmp_path / "rollout" / "partial_rollout_wal").exists()


def test_incomplete_release_pointer_disables_rollout_zero_wal(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("RUN_ID", "incomplete-release-run")
    args = _args(tmp_path, rollout_id=0, target=1)
    args.prompt_data = _write_prompt_data(tmp_path, ["p0"])
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release\n")
    release = tmp_path / "release"
    release.mkdir()
    (release / ".metadata").write_bytes(b"metadata")
    (release / "common.pt").write_bytes(b"common")
    (release / "metadata.json").write_text('{"sharded_backend":"torch_dist"}')

    assert (
        maybe_open_partial_rollout_store(
            args,
            resolve_polar_slime_config(args),
            0,
        )
        is None
    )


def test_prompt_identity_read_error_disables_rollout_zero_wal(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("RUN_ID", "prompt-read-error-run")
    args = _args(tmp_path, rollout_id=0, target=1)
    args.prompt_data = _write_prompt_data(tmp_path, ["p0"])
    monkeypatch.setattr(
        partial_rollout_module,
        "_release_seed_digest",
        lambda _root: "a" * 64,
    )

    def fail_hash(_path):
        raise OSError("injected prompt read error")

    monkeypatch.setattr(partial_rollout_module, "_sha256_file", fail_hash)

    assert (
        maybe_open_partial_rollout_store(
            args,
            resolve_polar_slime_config(args),
            0,
        )
        is None
    )


def test_recovery_replaces_ready_wal_group_already_committed_by_checkpoint(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("RUN_ID", "test-run")
    args = _args(tmp_path, target=1)
    prompts = ["p0", "p1", "p2"]
    writer = _source(args, prompts)
    checkpoint_reservations = writer.get_samples_with_reservation(2)
    writer.mark_consumed(1, outcome="accepted")
    writer.save(7)
    _write_model_checkpoint_shell(tmp_path, 7)
    _write_emitted_rollout_journal(tmp_path, 7, [1])

    store = maybe_open_partial_rollout_store(
        args,
        resolve_polar_slime_config(args),
        8,
    )
    assert store is not None
    assert store.committed_reservation_ids == frozenset({1})

    replay_writer = _source(args, prompts)
    replay_writer.load(7)
    replay_reservations = replay_writer.get_samples_with_reservation(2)
    assert [item[0] for item in checkpoint_reservations] == [0, 1]
    assert [item[0] for item in replay_reservations] == [0, 1]
    dropped = _record_result(store, 0, replay_reservations[0][1])
    store.record_drop(dropped.reservation_id, outcome="dynamic_filter", reason="test")
    completed = _record_result(store, 1, replay_reservations[1][1])
    _annotate_accepted_samples(
        completed.samples,
        accepted_rollout_id=8,
        staleness=0,
        policy_version=8,
        scheduler_group_id=1,
    )
    store.record_keep(completed, accepted_rollout_id=8)
    store.mark_ready([1])
    assert store.sealed is True

    restored = _source(args, prompts)
    restored.load(7)
    plan = rollout_module._prepare_partial_recovery(
        args,
        rollout_id=8,
        data_source=restored,
    )

    assert plan.store is store or plan.store is not None
    assert plan.kept == [] and plan.result_ready == [] and plan.deferred == []
    assert plan.dropped_count == 2
    assert plan.resume_duplicate_count == 1
    assert not store._ready_path().exists()
    records = store.load_records()
    assert [record["state"] for record in records] == [STATE_DROP, STATE_DROP]
    assert records[1]["drop_outcome"] == "resume_duplicate"
    assert (
        restored.reservation_metrics()[
            "polar/reservations/consumed_resume_duplicate_since_worker_start"
        ]
        == 1.0
    )

    worker = AsyncPolarRolloutWorker(args, restored, partial_store=plan.store)
    worker.set_rollout_context(8)
    worker.request_groups(1)
    replacement = worker._next_group_for_submission()
    assert replacement is not None and replacement.reservation_id == 2


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
