"""Post-rollout prompt-group selection for the Polar controller training batch.

Applied once, after a rollout step has assembled its groups and before they
become training data. An excluded group is not dropped but kept in the batch
and fully masked (``remove_sample``), so training draws no gradient from it
while the batch size is preserved: a shorter batch trips slime dp_schedule's
``num_steps >= 1`` assertion and kills the run. If every group would be
excluded the whole batch is left trainable for that step.

Stages, in order (each gated by a default-off arg):
  D  ``polar_drop_all_wrong_groups``      mask groups whose trajectories are all wrong
  B  ``polar_drop_all_keep_groups``       mask groups that never realized a switch
  C  ``polar_balance_all_correct_groups`` mask surplus fully-correct groups down to the
                                          number of mixed groups
"""

from __future__ import annotations

import random
from typing import Any

from slime_bridge.reward_post_process import (
    _controller_actual_action,
    _has_trainable_tokens,
    _trajectory_key,
)


def _trajectory_accuracies(group: list[Any], args: Any) -> list[float]:
    """One accuracy per trainable trajectory (traces of a trajectory share it)."""
    accuracy_by_trajectory: dict[Any, float] = {}
    for position, sample in enumerate(group):
        if not _has_trainable_tokens(sample):
            continue
        _, key = _trajectory_key(sample, position)
        if key in accuracy_by_trajectory:
            continue
        try:
            accuracy_by_trajectory[key] = float(sample.get_reward_value(args))
        except (TypeError, ValueError):
            continue
    return list(accuracy_by_trajectory.values())


def _accuracy_class(group: list[Any], args: Any) -> str:
    """Classify a group as all_correct / mixed / all_wrong / empty by accuracy."""
    accuracies = _trajectory_accuracies(group, args)
    if not accuracies:
        return "empty"
    correct = [accuracy >= 1.0 for accuracy in accuracies]
    if all(correct):
        return "all_correct"
    if not any(correct):
        return "all_wrong"
    return "mixed"


def _group_realized_a_switch(group: list[Any]) -> bool:
    """True unless the group carries actions and none is an escalate/deescalate.

    A group with no stamped action is indeterminate and treated as having a
    switch, so a rollout built without action stamping is never dropped here.
    """
    saw_action = False
    for sample in group:
        action = _controller_actual_action(sample)
        if action is None:
            continue
        saw_action = True
        if action in ("escalate", "deescalate"):
            return True
    return not saw_action


def select_training_groups(
    args: Any,
    groups: list[list[Any]],
    *,
    rollout_id: Any = None,
) -> tuple[list[list[Any]], dict[str, float]]:
    """Apply D -> B -> C group selection by masking, never shrinking the batch.

    Excluded groups stay in the returned batch but are fully masked
    (``remove_sample``) so training draws no gradient from them; dropping them
    would shorten the batch and trip dp_schedule's ``num_steps >= 1`` assertion.
    A fully-excluded batch is left entirely trainable.
    """
    drop_all_wrong = bool(getattr(args, "polar_drop_all_wrong_groups", False))
    drop_all_keep = bool(getattr(args, "polar_drop_all_keep_groups", False))
    balance_all_correct = bool(getattr(args, "polar_balance_all_correct_groups", False))
    if not (drop_all_wrong or drop_all_keep or balance_all_correct):
        return groups, {}

    metrics: dict[str, float] = {}
    classes = [_accuracy_class(group, args) for group in groups]
    metrics["polar/group_selection/all_correct_before"] = float(classes.count("all_correct"))
    metrics["polar/group_selection/mixed_before"] = float(classes.count("mixed"))
    metrics["polar/group_selection/all_wrong_before"] = float(classes.count("all_wrong"))

    kept = list(range(len(groups)))

    # D: exclude all-wrong groups.
    if drop_all_wrong:
        before = len(kept)
        kept = [i for i in kept if classes[i] != "all_wrong"]
        metrics["polar/group_selection/dropped_all_wrong"] = float(before - len(kept))

    # B: exclude groups that never realized an escalate/deescalate.
    if drop_all_keep:
        before = len(kept)
        kept = [i for i in kept if _group_realized_a_switch(groups[i])]
        metrics["polar/group_selection/dropped_all_keep"] = float(before - len(kept))

    # C: downsample fully-correct groups to the mixed-group count.
    if balance_all_correct:
        mixed = [i for i in kept if classes[i] == "mixed"]
        all_correct = [i for i in kept if classes[i] == "all_correct"]
        other = [i for i in kept if classes[i] not in ("mixed", "all_correct")]
        if len(all_correct) > len(mixed):
            seed = rollout_id if isinstance(rollout_id, int) else None
            all_correct = random.Random(seed).sample(all_correct, len(mixed))
        metrics["polar/group_selection/subsampled_all_correct"] = float(
            classes.count("all_correct") - len(all_correct)
        )
        kept = mixed + all_correct + other

    kept_set = set(kept)
    if not kept_set:
        # Nothing survived; a fully-masked batch has no training signal, so keep
        # every group trainable for this step (the historical fallback).
        metrics["polar/group_selection/empty_fallback"] = 1.0
        return groups, metrics

    excluded = [i for i in range(len(groups)) if i not in kept_set]
    for index in excluded:
        _mask_group(groups[index])

    metrics["polar/group_selection/kept"] = float(len(kept_set))
    metrics["polar/group_selection/masked"] = float(len(excluded))
    return groups, metrics


def _mask_group(group: list[Any]) -> None:
    """Keep a group scheduled in the batch but contribute no gradient from it.

    slime zeroes a ``remove_sample`` sample's loss mask (leaving it scheduled as
    a placeholder that does not dilute the loss denominator), and the reward
    post-processor already gives such samples a zero advantage and drops them
    from group statistics.
    """
    for sample in group:
        sample.remove_sample = True
