from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.utils.types import Sample
from slime_bridge.data_source import CeilEpochRolloutDataSourceWithBuffer


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


def _args(tmp_path, *, shuffle: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        rollout_global_dataset=True,
        prompt_data=None,
        n_samples_per_prompt=1,
        rollout_shuffle=shuffle,
        rollout_batch_size=2,
        buffer_filter_path=None,
        save=str(tmp_path),
        load=str(tmp_path),
    )


def _source(tmp_path, *, prompts: list[str], shuffle: bool = False):
    source = CeilEpochRolloutDataSourceWithBuffer(_args(tmp_path, shuffle=shuffle))
    source.dataset = _Dataset(prompts)
    if shuffle:
        source.dataset.shuffle(source.epoch_id)
    return source


def _prompt(reservation: tuple[int, list[Sample]]) -> str:
    return str(reservation[1][0].prompt)


def test_checkpoint_uses_earliest_outstanding_hole_not_outstanding_count(tmp_path) -> None:
    source = _source(tmp_path, prompts=["p0", "p1", "p2", "p3"])
    reservations = source.get_samples_with_reservation(3)

    # Only group 1 remains outstanding. Subtracting the outstanding count
    # from the live cursor (3 - 1) would incorrectly skip it and replay group 2.
    source.mark_consumed(reservations[0][0], outcome="accepted")
    source.mark_consumed(reservations[2][0], outcome="accepted")
    source.metadata = {"updated_after_reservation": True}
    source.save(7)

    path = tmp_path / "rollout" / "global_dataset_state_dict_7.pt"
    state = torch.load(path, weights_only=False)
    assert set(state) == {
        "sample_offset",
        "epoch_id",
        "sample_group_index",
        "sample_index",
        "metadata",
        "polar_reservation_frontier",
    }
    assert state["sample_group_index"] == 1
    assert state["sample_index"] == 1
    assert state["sample_offset"] == 1
    assert state["metadata"] == {"updated_after_reservation": True}
    assert state["polar_reservation_frontier"] == {
        "version": 1,
        "mode": "earliest_outstanding_at_least_once",
        "frontier_reservation_id": 1,
        "live_sample_group_index": 3,
        "outstanding_groups": 1,
        "replay_span_groups": 2,
        "potential_duplicate_groups": 1,
    }
    # Saving the frontier must not rewind the running worker's live cursor.
    assert source.sample_group_index == 3

    metrics = source.reservation_metrics()
    assert metrics["polar/reservations/outstanding_groups"] == 1.0
    assert metrics["polar/reservations/checkpoint_replay_span_groups"] == 2.0
    assert metrics["polar/reservations/checkpoint_potential_duplicate_groups"] == 1.0
    assert metrics["polar/reservations/consumed_accepted_since_worker_start"] == 2.0

    restored = _source(tmp_path, prompts=["p0", "p1", "p2", "p3"])
    restored.load(7)
    replay = restored.get_samples_with_reservation(1)[0]
    assert replay[0] == reservations[1][0] == 1
    assert _prompt(replay) == _prompt(reservations[1])
    assert restored.reservation_metrics()["polar/reservations/resumed_replay_span_groups"] == 2.0


def test_frontier_replays_cross_epoch_shuffle_suffix_in_original_order(tmp_path) -> None:
    source = _source(tmp_path, prompts=["a", "b", "c"], shuffle=True)
    reservations = source.get_samples_with_reservation(5)

    # Group 3 is reserved exactly at the epoch boundary. Group 4 completes
    # first, so a resume must replay both 3 and 4 in epoch-1 shuffle order.
    for position in (0, 1, 2, 4):
        source.mark_consumed(reservations[position][0], outcome="accepted")
    source.save(11)

    state = torch.load(
        tmp_path / "rollout" / "global_dataset_state_dict_11.pt",
        weights_only=False,
    )
    assert (state["epoch_id"], state["sample_offset"], state["sample_group_index"]) == (
        0,
        3,
        3,
    )

    restored = _source(tmp_path, prompts=["a", "b", "c"], shuffle=True)
    restored.load(11)
    replay = restored.get_samples_with_reservation(2)
    assert [reservation_id for reservation_id, _ in replay] == [3, 4]
    assert [_prompt(item) for item in replay] == [
        _prompt(reservations[3]),
        _prompt(reservations[4]),
    ]
    assert restored.epoch_id == 1


def test_nonempty_buffer_fails_closed_for_reservation_and_checkpoint(tmp_path) -> None:
    source = _source(tmp_path, prompts=["dataset"])
    source.add_samples([[Sample(prompt="buffer-only")]])

    with pytest.raises(RuntimeError, match="buffer is non-empty"):
        source.get_samples_with_reservation(1)
    with pytest.raises(RuntimeError, match="buffer is non-empty"):
        source.save(3)

    assert not (tmp_path / "rollout" / "global_dataset_state_dict_3.pt").exists()
    assert list((tmp_path / "rollout").glob("*.tmp.*")) == []


def test_bulk_consumption_validates_every_id_before_mutating_state(tmp_path) -> None:
    source = _source(tmp_path, prompts=["p0", "p1"])
    source.get_samples_with_reservation(2)

    with pytest.raises(KeyError, match="999"):
        source.mark_consumed_many([0, 999], outcome="accepted")

    metrics = source.reservation_metrics()
    assert metrics["polar/reservations/outstanding_groups"] == 2.0
    assert metrics["polar/reservations/consumed_since_worker_start"] == 0.0


def test_direct_get_cannot_bypass_frontier_after_reservation_mode_starts(tmp_path) -> None:
    source = _source(tmp_path, prompts=["p0", "p1"])
    source.get_samples_with_reservation(1)

    with pytest.raises(RuntimeError, match="cannot be mixed with reservation-aware rollout"):
        source.get_samples(1)


def test_reservation_is_registered_before_sample_validation(monkeypatch, tmp_path) -> None:
    source = _source(tmp_path, prompts=["p0", "p1"])
    original_get_samples = RolloutDataSourceWithBuffer.get_samples

    def malformed_group(self, num_samples):
        groups = original_get_samples(self, num_samples)
        groups[0][0].group_index = 999
        return groups

    monkeypatch.setattr(RolloutDataSourceWithBuffer, "get_samples", malformed_group)
    with pytest.raises(RuntimeError, match="expected 0, got.*999"):
        source.get_samples_with_reservation(1)

    # Cursor construction advanced before validation failed, but the
    # pre-registered frontier still makes the persisted state replay group 0.
    assert source.sample_group_index == 1
    source.save(13)
    state = torch.load(
        tmp_path / "rollout" / "global_dataset_state_dict_13.pt",
        weights_only=False,
    )
    assert state["sample_group_index"] == 0
    assert state["sample_offset"] == 0
    assert state["polar_reservation_frontier"]["frontier_reservation_id"] == 0
