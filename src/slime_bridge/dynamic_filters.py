"""Dynamic sampling filters for the Polar controller training rollout.

Registered through Slime's ``--dynamic-sampling-filter-path`` hook, these run
per prompt group during rollout, so a dropped group never reaches the training
batch and no reference or policy forward is spent on it.
"""

from __future__ import annotations

from typing import Any

from slime_bridge.reward_post_process import _controller_actual_action


def _group_lacks_routing_switch(samples: list[Any]) -> bool:
    """True when a group carries realized actions but never actually switched.

    A group with no stamped action is treated as indeterminate (``False``) so a
    rollout built without action stamping is never dropped for lacking a switch.
    """
    saw_action = False
    saw_switch = False
    for sample in samples:
        action = _controller_actual_action(sample)
        if action is None:
            continue
        saw_action = True
        if action in ("escalate", "deescalate"):
            saw_switch = True
    return saw_action and not saw_switch


def require_routing_action(args: Any, samples: list[Any], **kwargs: Any) -> Any:
    """Drop groups that never realized an escalate/deescalate, else defer to std.

    A group whose controller only ever kept the current worker has no routing
    signal worth a training slot, so it is dropped before batch construction.
    Every other group falls through to the standard non-degenerate reward-std
    check, preserving that behaviour.
    """
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput

    if _group_lacks_routing_switch(samples):
        return DynamicFilterOutput(keep=False, reason="no_routing_action")

    from slime.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std

    return check_reward_nonzero_std(args, samples, **kwargs)
