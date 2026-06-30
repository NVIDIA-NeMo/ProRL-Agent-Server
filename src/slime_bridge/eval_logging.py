"""Custom Slime evaluation logging helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from numbers import Real
from typing import Any


AGGREGATE_REWARD_METRIC = "eval/aggregate/reward_weighted_mean"


def _valid_rewards(dataset_name: str, dataset: Mapping[str, Any]) -> list[float]:
    """Return all accounted outcomes and verify their count contract."""

    raw_rewards = dataset.get("rewards") or []
    if isinstance(raw_rewards, (str, bytes)) or not isinstance(raw_rewards, Sequence):
        raise TypeError(f"eval dataset {dataset_name!r} rewards must be a sequence")

    rewards: list[float] = []
    for index, reward in enumerate(raw_rewards):
        if isinstance(reward, bool) or not isinstance(reward, Real):
            raise TypeError(
                f"eval dataset {dataset_name!r} reward {index} must be numeric"
            )
        value = float(reward)
        if not math.isfinite(value):
            raise ValueError(
                f"eval dataset {dataset_name!r} reward {index} must be finite"
            )
        rewards.append(value)

    # Bridge-backed evals keep `valid_count` as the number of genuine
    # evaluator outcomes and expose `accounted_count` for the full reward
    # vector, including fallback zeros. Legacy eval functions have only
    # `valid_count`, where it continues to mean len(rewards).
    raw_valid_count = dataset.get("valid_count", len(rewards))
    if (
        isinstance(raw_valid_count, bool)
        or not isinstance(raw_valid_count, Real)
        or not math.isfinite(float(raw_valid_count))
        or int(raw_valid_count) != raw_valid_count
        or int(raw_valid_count) < 0
    ):
        raise ValueError(
            f"eval dataset {dataset_name!r} valid_count must be a non-negative integer"
        )
    valid_count = int(raw_valid_count)
    raw_accounted_count = dataset.get("accounted_count", valid_count)
    if (
        isinstance(raw_accounted_count, bool)
        or not isinstance(raw_accounted_count, Real)
        or not math.isfinite(float(raw_accounted_count))
        or int(raw_accounted_count) != raw_accounted_count
        or int(raw_accounted_count) < 0
    ):
        raise ValueError(
            f"eval dataset {dataset_name!r} accounted_count must be a non-negative integer"
        )
    accounted_count = int(raw_accounted_count)
    if accounted_count != len(rewards):
        count_label = "accounted_count" if "accounted_count" in dataset else "valid_count"
        raise ValueError(
            f"eval dataset {dataset_name!r} {count_label}={accounted_count} does not "
            f"match its {len(rewards)} rewards"
        )
    if valid_count > accounted_count:
        raise ValueError(
            f"eval dataset {dataset_name!r} valid_count={valid_count} exceeds "
            f"accounted_count={accounted_count}"
        )
    return rewards


def add_weighted_eval_metric(
    rollout_id: int,
    args: Any,
    data: Mapping[str, Mapping[str, Any]],
    extra_metrics: MutableMapping[str, Any] | None,
) -> bool:
    """Add the all-item-weighted reward mean before Slime's default logger.

    Each configured evaluation item receives exactly one vote. Infrastructure
    failures are represented by fallback zeros, as are verifier-confirmed model
    failures, so an error cannot improve the reported reward by shrinking its
    denominator. The current TMax/Terminal-Bench evaluation therefore has the
    intended 100:89 weighting without hard-coding dataset sizes.

    The hook returns ``False`` so Slime still emits every per-dataset metric and
    attaches the shared ``eval/train_step`` axis to the complete evaluation row.
    """

    del rollout_id, args
    if not isinstance(data, Mapping):
        raise TypeError("eval data must be a mapping keyed by dataset name")
    if extra_metrics is None:
        raise TypeError("weighted eval logging requires a mutable extra_metrics mapping")
    if not isinstance(extra_metrics, MutableMapping):
        raise TypeError("eval extra_metrics must be mutable")

    reward_sum = 0.0
    accounted_count = 0
    for dataset_name, dataset in data.items():
        if not isinstance(dataset, Mapping):
            raise TypeError(f"eval dataset {dataset_name!r} payload must be a mapping")
        rewards = _valid_rewards(str(dataset_name), dataset)
        reward_sum += math.fsum(rewards)
        accounted_count += len(rewards)

    # Keep empty datasets observable without manufacturing NaN/inf.
    if accounted_count:
        extra_metrics[AGGREGATE_REWARD_METRIC] = reward_sum / accounted_count
    else:
        extra_metrics.pop(AGGREGATE_REWARD_METRIC, None)
    return False
