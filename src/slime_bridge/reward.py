"""Reward adapter for Slime custom reward model hooks."""

from __future__ import annotations

import math
from typing import Any


async def reward_func(args: Any, sample_or_samples: Any, **kwargs: Any) -> Any:
    """Read the reward already embedded in Polar-converted Slime samples."""
    del kwargs
    reward_key = str(getattr(args, "polar_reward_key", getattr(args, "reward_key", "score")))
    if isinstance(sample_or_samples, list):
        return [{reward_key: _extract_reward(sample, reward_key)} for sample in sample_or_samples]
    return {reward_key: _extract_reward(sample_or_samples, reward_key)}


def _extract_reward(sample: Any, reward_key: str) -> float:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        if reward_key in reward:
            return _finite_reward_or_zero(reward[reward_key])
        if "score" in reward:
            return _finite_reward_or_zero(reward["score"])
        for value in reward.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return _finite_reward_or_zero(value)
        return 0.0
    if isinstance(reward, (int, float)):
        return _finite_reward_or_zero(reward)
    return 0.0


def _finite_reward_or_zero(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0
