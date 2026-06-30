"""Slime data-source wrappers used by Polar examples."""

from __future__ import annotations

import copy
import math
from typing import Any

try:
    from slime.rollout.data_source import RolloutDataSourceWithBuffer
except ImportError as _slime_import_error:
    _SLIME_IMPORT_ERROR = _slime_import_error

    class RolloutDataSourceWithBuffer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "Slime is required to use CeilEpochRolloutDataSourceWithBuffer."
            ) from _SLIME_IMPORT_ERROR


def ceil_to_batch_size(size: int, batch_size: int) -> int:
    if size <= 0:
        return 0
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return math.ceil(size / batch_size) * batch_size


class CeilEpochRolloutDataSourceWithBuffer(RolloutDataSourceWithBuffer):
    """Expose a rounded-up epoch length for fixed-size Slime rollout batches.

    Slime computes `num_rollout_per_epoch = len(data_source) // rollout_batch_size`.
    For datasets whose size is not divisible by the rollout batch size, the
    default floor behavior skips the tail prompts. Returning a rounded-up length
    lets the existing data source wrap only the final few prompts while still
    covering every prompt in the dataset once per epoch.
    """

    _FRONTIER_STATE_KEY = "polar_reservation_frontier"

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        # A reservation contains only the cursor immediately before one prompt
        # group was allocated. Rollout results remain in the worker queues and
        # are deliberately not persisted here.
        self._outstanding_reservations: dict[int, dict[str, Any]] = {}
        self._reservation_reserved_total = 0
        self._reservation_consumed_total = 0
        self._reservation_consumed_by_outcome: dict[str, int] = {}
        self._resumed_replay_span = 0
        self._resumed_potential_duplicates = 0
        self._reservation_mode_started = False

    def __len__(self) -> int:
        return ceil_to_batch_size(
            super().__len__(),
            int(getattr(self.args, "rollout_batch_size", 1) or 1),
        )

    def get_samples(self, num_samples: int) -> list[list[Any]]:
        # Keep the inherited buffer mutation serialized with reservations and
        # checkpoint snapshots, even for compatibility callers.
        with self._state_lock:
            if self._reservation_mode_started:
                raise RuntimeError(
                    "Direct get_samples cannot be mixed with reservation-aware rollout; "
                    "configure a separate eval dataset instead"
                )
            return super().get_samples(num_samples)

    def add_samples(self, samples: list[list[Any]]) -> None:
        with self._state_lock:
            super().add_samples(samples)

    def get_samples_with_reservation(
        self,
        num_samples: int,
    ) -> list[tuple[int, list[Any]]]:
        """Atomically reserve prompt groups and their pre-allocation cursors."""
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")

        reserved: list[tuple[int, list[Any]]] = []
        with self._state_lock:
            self._raise_if_buffered_locked("reserve samples")
            if num_samples:
                self._reservation_mode_started = True
            for _ in range(num_samples):
                cursor_before = {
                    "sample_offset": self.sample_offset,
                    "epoch_id": self.epoch_id,
                    "sample_group_index": self.sample_group_index,
                    "sample_index": self.sample_index,
                }
                reservation_id = int(cursor_before["sample_group_index"])
                if reservation_id in self._outstanding_reservations:
                    raise RuntimeError(f"duplicate outstanding reservation {reservation_id}")
                # Register before advancing the cursor. If sample construction
                # or validation fails, retaining this frontier is conservative:
                # a later checkpoint can replay, but can never skip, the group.
                self._outstanding_reservations[reservation_id] = cursor_before
                self._reservation_reserved_total += 1
                groups = super().get_samples(1)
                if len(groups) != 1 or not groups[0]:
                    raise RuntimeError(
                        "Slime data source did not return exactly one non-empty sample group"
                    )
                group = groups[0]
                actual_group_ids = {int(sample.group_index) for sample in group}
                if actual_group_ids != {reservation_id}:
                    raise RuntimeError(
                        "reserved sample group has inconsistent group_index values: "
                        f"expected {reservation_id}, got {sorted(actual_group_ids)}"
                    )
                reserved.append((reservation_id, group))
        return reserved

    def mark_consumed(self, reservation_id: int, *, outcome: str) -> dict[str, float]:
        """Commit a delivered or permanently discarded reservation."""
        return self.mark_consumed_many([reservation_id], outcome=outcome)

    def mark_consumed_many(
        self,
        reservation_ids: list[int],
        *,
        outcome: str,
    ) -> dict[str, float]:
        """Atomically commit a set of reservations after validating every ID."""
        outcome = str(outcome).strip().lower().replace("-", "_")
        if not outcome:
            raise ValueError("reservation outcome must be non-empty")
        normalized_ids = [int(reservation_id) for reservation_id in reservation_ids]
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("reservation_ids must not contain duplicates")
        with self._state_lock:
            missing = [
                reservation_id
                for reservation_id in normalized_ids
                if reservation_id not in self._outstanding_reservations
            ]
            if missing:
                raise KeyError(f"unknown or already consumed reservations {missing}")
            for reservation_id in normalized_ids:
                del self._outstanding_reservations[reservation_id]
            consumed_count = len(normalized_ids)
            self._reservation_consumed_total += consumed_count
            self._reservation_consumed_by_outcome[outcome] = (
                self._reservation_consumed_by_outcome.get(outcome, 0) + consumed_count
            )
            return self._reservation_metrics_locked()

    def reservation_metrics(self) -> dict[str, float]:
        with self._state_lock:
            return self._reservation_metrics_locked()

    def _reservation_metrics_locked(self) -> dict[str, float]:
        metrics = {
            "polar/reservations/reserved_since_worker_start": float(
                self._reservation_reserved_total
            ),
            "polar/reservations/consumed_since_worker_start": float(
                self._reservation_consumed_total
            ),
            "polar/reservations/outstanding_groups": float(len(self._outstanding_reservations)),
            "polar/reservations/checkpoint_replay_span_groups": float(
                self._checkpoint_replay_span_locked()
            ),
            "polar/reservations/checkpoint_potential_duplicate_groups": float(
                self._checkpoint_potential_duplicates_locked()
            ),
            "polar/reservations/resumed_replay_span_groups": float(self._resumed_replay_span),
            "polar/reservations/resumed_potential_duplicate_groups": float(
                self._resumed_potential_duplicates
            ),
        }
        for outcome, count in self._reservation_consumed_by_outcome.items():
            metrics[
                f"polar/reservations/consumed_{outcome}_since_worker_start"
            ] = float(count)
        return metrics

    def _checkpoint_state_locked(self) -> dict[str, Any]:
        self._raise_if_buffered_locked("checkpoint the data source")
        live_state = self._cursor_state_locked()
        frontier = self._frontier_locked()
        if frontier is None:
            state_dict = live_state
            frontier_reservation_id = None
        else:
            frontier_reservation_id, cursor_before = frontier
            state_dict = copy.deepcopy(cursor_before)
            # Reservation cursors describe dataset position only. Metadata may
            # legitimately advance while speculative groups are outstanding.
            state_dict["metadata"] = copy.deepcopy(live_state["metadata"])

        replay_span = max(
            0,
            int(live_state["sample_group_index"])
            - int(state_dict["sample_group_index"]),
        )
        potential_duplicates = self._checkpoint_potential_duplicates_locked()
        state_dict[self._FRONTIER_STATE_KEY] = {
            "version": 1,
            "mode": "earliest_outstanding_at_least_once",
            "frontier_reservation_id": frontier_reservation_id,
            "live_sample_group_index": int(live_state["sample_group_index"]),
            "outstanding_groups": len(self._outstanding_reservations),
            "replay_span_groups": replay_span,
            "potential_duplicate_groups": potential_duplicates,
        }
        return state_dict

    def _restore_checkpoint_state_locked(self, state_dict: dict[str, Any]) -> None:
        super()._restore_checkpoint_state_locked(state_dict)
        self._outstanding_reservations.clear()
        self._reservation_mode_started = False
        frontier_state = state_dict.get(self._FRONTIER_STATE_KEY, {})
        if isinstance(frontier_state, dict):
            self._resumed_replay_span = int(frontier_state.get("replay_span_groups", 0) or 0)
            self._resumed_potential_duplicates = int(
                frontier_state.get("potential_duplicate_groups", 0) or 0
            )
        else:
            self._resumed_replay_span = 0
            self._resumed_potential_duplicates = 0

    def _frontier_locked(self) -> tuple[int, dict[str, Any]] | None:
        if not self._outstanding_reservations:
            return None
        return min(
            self._outstanding_reservations.items(),
            key=lambda item: int(item[1]["sample_group_index"]),
        )

    def _checkpoint_replay_span_locked(self) -> int:
        frontier = self._frontier_locked()
        if frontier is None:
            return 0
        return max(
            0,
            int(self.sample_group_index)
            - int(frontier[1]["sample_group_index"]),
        )

    def _checkpoint_potential_duplicates_locked(self) -> int:
        replay_span = self._checkpoint_replay_span_locked()
        return max(0, replay_span - len(self._outstanding_reservations))

    def _raise_if_buffered_locked(self, operation: str) -> None:
        if self.buffer:
            raise RuntimeError(
                f"Cannot {operation} while the rollout data-source buffer is non-empty; "
                "buffer samples cannot be reconstructed from a dataset cursor"
            )
