"""Dynamic-trace reward post-processor for Slime.

Registered via Slime's ``--custom-reward-post-process-path`` hook.  A Polar
trajectory can fan out into a variable number of trace samples, and each trace
keeps its own reward.  The exchangeable unit is still the trajectory, so this
processor computes per-trace advantages against a leave-one-trajectory-out
baseline built from other trajectories in the same prompt group.

Adapter contract:
    All Slime samples produced from the same Polar ``SessionResult`` share
    ``Sample.rollout_id``. Slime uses that field to average all trace
    contributions from one trajectory as one gradient unit.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)


def post_process_rewards(
    args: Any,
    samples: list[Any],
) -> tuple[list[float], list[float]]:
    """Slime reward-post-process hook. Returns (raw_rewards, rewards)."""
    # Enforce the failure policy again at the training boundary so replaying
    # an artifact produced by an older adapter cannot resurrect a positive
    # reward. Fully failed/removed trajectories stay excluded; aligned model
    # policy failures keep trainable tokens with a fail-closed scalar zero.
    raw_rewards = [
        0.0
        if _is_failed_trajectory(sample)
        or bool(getattr(sample, "remove_sample", False))
        or _is_trainable_negative(sample)
        else float(sample.get_reward_value(args))
        for sample in samples
    ]

    if not getattr(args, "rewards_normalization", True):
        return raw_rewards, list(raw_rewards)

    estimator = getattr(args, "advantage_estimator", None)
    if estimator not in ("grpo", "gspo", "reinforce_plus_plus_baseline"):
        return raw_rewards, list(raw_rewards)

    std_norm = estimator in ("grpo", "gspo") and bool(
        getattr(args, "grpo_std_normalization", False)
    )

    traj_sample_indices: dict[tuple[Any, Any], list[int]] = {}
    traj_valid_rewards: dict[tuple[Any, Any], list[float]] = {}
    traj_failed: dict[tuple[Any, Any], bool] = {}
    group_keys: dict[Any, list[tuple[Any, Any]]] = {}
    key_by_sample: list[tuple[Any, Any]] = []

    for i, sample in enumerate(samples):
        group_idx, key = _trajectory_key(sample, i)
        key_by_sample.append(key)
        if key not in traj_sample_indices:
            traj_sample_indices[key] = []
            traj_valid_rewards[key] = []
            traj_failed[key] = False
            group_keys.setdefault(group_idx, []).append(key)
        if _is_failed_trajectory(sample):
            traj_failed[key] = True
        elif _has_trainable_tokens(sample):
            # Keep fully-masked/removed traces in ``raw_rewards`` for aligned
            # diagnostics, but never assign them an advantage. Aligned
            # parser-invalid and agent-timeout policy actions are not removed:
            # the adapter keeps their source loss mask and gives them reward
            # zero so LOO can supply the intended negative advantage.
            traj_sample_indices[key].append(i)
            traj_valid_rewards[key].append(raw_rewards[i])

    normalized_by_sample = [0.0] * len(samples)
    for keys in group_keys.values():
        valid_keys = [key for key in keys if not traj_failed[key] and traj_valid_rewards[key]]
        traj_mean = {
            key: sum(traj_valid_rewards[key]) / len(traj_valid_rewards[key]) for key in valid_keys
        }
        group_scale = _group_scale(list(traj_mean.values())) if std_norm else 1.0

        # A common zero scale means every valid trajectory has the same mean
        # reward. There is no within-prompt preference signal, so keep the
        # entire group at zero instead of manufacturing one through epsilon
        # division. Failed/fully-masked trajectories were initialized to zero
        # above as well.
        if group_scale == 0.0:
            continue

        for key in keys:
            if key not in traj_mean:
                continue
            other_means = [traj_mean[other_key] for other_key in valid_keys if other_key != key]
            baseline = sum(other_means) / len(other_means) if other_means else 0.0
            for sample_index in traj_sample_indices[key]:
                normalized_by_sample[sample_index] = (
                    raw_rewards[sample_index] - baseline
                ) / group_scale

    return raw_rewards, normalized_by_sample


def _trajectory_key(sample: Any, sample_position: int) -> tuple[Any, tuple[Any, Any]]:
    group_idx = _key_value(getattr(sample, "group_index", None), -1)
    traj_idx = getattr(sample, "rollout_id", None)
    if traj_idx is None:
        # Older Slime bridge samples used ``group_id`` for this contract.
        traj_idx = getattr(sample, "group_id", None)
    if traj_idx is None:
        traj_idx = getattr(sample, "index", None)
    return group_idx, (group_idx, _key_value(traj_idx, sample_position))


def _key_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _group_scale(trajectory_means: list[float]) -> float:
    """Return one shared standard-deviation scale for a prompt group.

    The scale must include the current trajectory. Computing a separate scale
    from each trajectory's leave-one-out peers makes a singleton binary outcome
    singular: for rewards ``[1, 0, ..., 0]``, the winner sees peers with zero
    variance and receives an advantage near ``1 / 1e-6``. The LOO *mean*
    remains trajectory-specific; only its scale is shared by the group.

    A single valid trajectory preserves the historical unscaled behavior. Two
    or more identical trajectory means return zero so the caller emits zero
    advantages for the degenerate group.
    """
    if len(trajectory_means) <= 1:
        return 1.0
    std = statistics.stdev(trajectory_means)
    return std + 1e-6 if std > 0.0 else 0.0


def _has_trainable_tokens(sample: Any) -> bool:
    if bool(getattr(sample, "remove_sample", False)):
        return False
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return int(getattr(sample, "response_length", 0) or 0) > 0
    return any(int(value) != 0 for value in loss_mask)


def _is_failed_trajectory(sample: Any) -> bool:
    """True if the sample status marks it as a fully excluded execution."""
    status = getattr(sample, "status", None)
    name = getattr(status, "name", None) or str(status).rsplit(".", 1)[-1]
    return name.upper() in ("FAILED", "ABORTED")


def _is_trainable_negative(sample: Any) -> bool:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    polar = metadata.get("polar")
    if not isinstance(polar, dict):
        return False
    training_filter = polar.get("training_filter")
    if not isinstance(training_filter, dict):
        return False
    return (
        training_filter.get("reason") in {"agent_timeout", "parser_invalid_tool_call"}
        and training_filter.get("trainable") is True
        and training_filter.get("masked") is not True
    )
