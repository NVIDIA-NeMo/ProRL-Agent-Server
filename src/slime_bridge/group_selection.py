"""Post-rollout prompt-group selection for the Polar controller training batch.

Applied once, after a rollout step has assembled its groups and before they
become training data, so dropped groups never reach the reference or policy
forward. Every stage only shrinks the batch (no backfill); if the whole batch
would be emptied the original is kept for that step so the training loop, which
cannot consume an empty batch, is never handed one.

Stages, in order (each gated by a default-off arg):
  D  ``polar_drop_all_wrong_groups``      drop groups whose trajectories are all wrong
  B  ``polar_drop_all_keep_groups``       drop groups that never realized a switch
  C  ``polar_balance_all_correct_groups`` downsample fully-correct groups to the
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
    """Apply the D -> B -> C group selection; never return an empty batch."""
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

    kept = list(zip(groups, classes))

    # D: drop all-wrong groups.
    if drop_all_wrong:
        before = len(kept)
        kept = [(group, cls) for group, cls in kept if cls != "all_wrong"]
        metrics["polar/group_selection/dropped_all_wrong"] = float(before - len(kept))

    # B: drop groups that never realized an escalate/deescalate.
    if drop_all_keep:
        before = len(kept)
        kept = [(group, cls) for group, cls in kept if _group_realized_a_switch(group)]
        metrics["polar/group_selection/dropped_all_keep"] = float(before - len(kept))

    # C: downsample fully-correct groups to the mixed-group count.
    if balance_all_correct:
        mixed = [(group, cls) for group, cls in kept if cls == "mixed"]
        all_correct = [(group, cls) for group, cls in kept if cls == "all_correct"]
        other = [(group, cls) for group, cls in kept if cls not in ("mixed", "all_correct")]
        if len(all_correct) > len(mixed):
            seed = rollout_id if isinstance(rollout_id, int) else None
            all_correct = random.Random(seed).sample(all_correct, len(mixed))
        metrics["polar/group_selection/subsampled_all_correct"] = float(
            classes.count("all_correct") - len(all_correct)
        )
        kept = mixed + all_correct + other

    selected = [group for group, _cls in kept]
    if not selected and groups:
        # The training loop cannot consume an empty batch; keep the original for
        # this step. Under GDPO with the cost gate an all-wrong/all-keep batch
        # yields near-zero advantage anyway, so this is close to a no-op.
        metrics["polar/group_selection/empty_fallback"] = 1.0
        return groups, metrics

    metrics["polar/group_selection/kept"] = float(len(selected))
    return selected, metrics
