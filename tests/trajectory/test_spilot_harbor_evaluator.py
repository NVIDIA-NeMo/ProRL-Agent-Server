from __future__ import annotations

import pytest

from polar.agent.models import AgentRunResult
from polar.trajectory.evaluator.harbor import HarborEvaluator
from polar.trajectory.evaluator.spilot_harbor import SpilotHarborEvaluator
from polar.trajectory.models import EvalResult, Trajectory


def _evaluator(
    tmp_path,
    monkeypatch,
    *,
    harbor_reward: float = 1.0,
    harbor_trace_rewards=None,
    **config,
):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(exist_ok=True)

    async def fake_harbor(_self, _trajectory, **_runtime):
        return EvalResult(
            outcome_reward=harbor_reward,
            trace_rewards=harbor_trace_rewards,
            metadata={"mode": "harbor", "reward": harbor_reward},
        )

    monkeypatch.setattr(HarborEvaluator, "evaluate", fake_harbor)
    return SpilotHarborEvaluator(tests_dir=str(tests_dir), **config)


def _agent_result(router_metadata: dict | None) -> AgentRunResult:
    metadata = {} if router_metadata is None else {"spilot_router": router_metadata}
    return AgentRunResult(status="completed", return_code=0, metadata=metadata)


@pytest.mark.asyncio
async def test_spilot_harbor_preserves_valid_action_reward(tmp_path, monkeypatch) -> None:
    evaluator = _evaluator(tmp_path, monkeypatch)
    router_metadata = {
        "action_valid": True,
        "submitted": True,
        "actions": [{"action": "ROUTE", "model_slot": "M0"}],
        "calls": [{"slot": "M0", "status": "completed"}],
        "slot_mapping": {"M0": {"model": "pool/qwen"}},
        "total_cost": 4.0,
    }

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        agent_result=_agent_result(router_metadata),
    )

    assert result.outcome_reward == 1.0
    assert result.metadata["router_action_valid"] is True
    assert result.metadata["applied_cost_penalty"] == 0.0
    assert result.metadata["spilot_router"] == router_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "router_metadata",
    [
        None,
        {},
        {"action_valid": False, "submitted": False},
        {"action_valid": True},
        {"action_valid": True, "submitted": False},
    ],
)
async def test_spilot_harbor_zeroes_invalid_or_missing_action(
    tmp_path, monkeypatch, router_metadata
) -> None:
    evaluator = _evaluator(tmp_path, monkeypatch)

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        agent_result=_agent_result(router_metadata),
    )

    assert result.outcome_reward == 0.0
    assert result.metadata["harbor_outcome_reward"] == 1.0
    assert "reward_override_reason" in result.metadata


@pytest.mark.asyncio
async def test_spilot_harbor_applies_success_gated_cost_penalty(tmp_path, monkeypatch) -> None:
    evaluator = _evaluator(
        tmp_path,
        monkeypatch,
        cost_penalty_lambda=0.2,
        cost_normalizer=2.0,
    )

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        agent_result=_agent_result(
            {"action_valid": True, "submitted": True, "total_cost": 1.0}
        ),
    )

    assert result.outcome_reward == pytest.approx(0.9)
    assert result.metadata["applied_cost_penalty"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_spilot_harbor_does_not_reward_cheap_failure(tmp_path, monkeypatch) -> None:
    evaluator = _evaluator(
        tmp_path,
        monkeypatch,
        harbor_reward=0.0,
        cost_penalty_lambda=0.5,
    )

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        agent_result=_agent_result(
            {"action_valid": True, "submitted": True, "total_cost": 0.0}
        ),
    )

    assert result.outcome_reward == 0.0


@pytest.mark.asyncio
async def test_spilot_harbor_fails_closed_on_missing_cost_when_enabled(
    tmp_path, monkeypatch
) -> None:
    evaluator = _evaluator(
        tmp_path,
        monkeypatch,
        cost_penalty_lambda=0.1,
    )

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        agent_result=_agent_result({"action_valid": True, "submitted": True}),
    )

    assert result.outcome_reward == 0.0
    assert result.metadata["reward_override_reason"] == "missing_or_invalid_total_cost"


@pytest.mark.asyncio
async def test_spilot_harbor_clears_wrapped_trace_rewards(tmp_path, monkeypatch) -> None:
    evaluator = _evaluator(
        tmp_path,
        monkeypatch,
        harbor_trace_rewards=[1.0, 1.0],
        cost_penalty_lambda=0.2,
        cost_normalizer=2.0,
    )

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        agent_result=_agent_result(
            {"action_valid": True, "submitted": True, "total_cost": 1.0}
        ),
    )

    assert result.outcome_reward == pytest.approx(0.9)
    assert result.trace_rewards is None
