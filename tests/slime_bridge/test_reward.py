from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from slime_bridge.reward import reward_func


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed_reward",
    [float("nan"), float("inf"), float("-inf"), True, False, "not-a-number"],
)
async def test_reward_hook_fail_closes_malformed_selected_reward(
    malformed_reward: Any,
) -> None:
    sample = SimpleNamespace(reward={"score": malformed_reward})

    result = await reward_func(
        SimpleNamespace(polar_reward_key="score", reward_key="score"),
        sample,
    )

    assert result == {"score": 0.0}


@pytest.mark.asyncio
async def test_reward_hook_sanitizes_each_sample_in_a_batch() -> None:
    samples = [
        SimpleNamespace(reward={"score": 1.0}),
        SimpleNamespace(reward={"score": float("nan")}),
        SimpleNamespace(reward={"score": True}),
    ]

    result = await reward_func(
        SimpleNamespace(polar_reward_key="score", reward_key="score"),
        samples,
    )

    assert result == [{"score": 1.0}, {"score": 0.0}, {"score": 0.0}]
