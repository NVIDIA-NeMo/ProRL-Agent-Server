from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from polar.trajectory.models import EvalResult, Trace


@pytest.mark.parametrize(
    "invalid_reward",
    [float("nan"), float("inf"), float("-inf"), True, False, "not-a-number"],
)
def test_eval_result_rejects_nonfinite_boolean_or_malformed_reward(
    invalid_reward: Any,
) -> None:
    with pytest.raises(ValidationError):
        EvalResult(outcome_reward=invalid_reward)
    with pytest.raises(ValidationError):
        EvalResult(trace_rewards=[invalid_reward])


@pytest.mark.parametrize(
    "invalid_reward",
    [float("nan"), float("inf"), float("-inf"), True, False, "not-a-number"],
)
def test_trace_rejects_nonfinite_boolean_or_malformed_reward(
    invalid_reward: Any,
) -> None:
    with pytest.raises(ValidationError):
        Trace(reward=invalid_reward)


def test_reward_models_accept_finite_numeric_values() -> None:
    assert EvalResult(outcome_reward="0.25", trace_rewards=[0, 1.0, None]) == EvalResult(
        outcome_reward=0.25,
        trace_rewards=[0.0, 1.0, None],
    )
    assert Trace(reward="1").reward == 1.0
