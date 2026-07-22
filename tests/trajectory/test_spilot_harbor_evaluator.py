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


def _valid_router_metadata(**overrides) -> dict:
    metadata = {
        "action_valid": True,
        "submitted": True,
        "actions": [{"action": "ROUTE", "model_slot": "M0"}],
        "calls": [
            {"slot": "M0", "status": "completed", "duration_ms": 60_000.0},
            {"slot": "M1", "status": "completed", "duration_ms": 120_000.0},
        ],
        "slot_mapping": {"M0": {"model": "pool/qwen"}},
        "total_cost": 4.0,
    }
    metadata.update(overrides)
    return metadata


@pytest.mark.asyncio
async def test_latency_penalty_disabled_by_default(tmp_path, monkeypatch) -> None:
    evaluator = _evaluator(tmp_path, monkeypatch)

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        agent_result=_agent_result(_valid_router_metadata()),
    )

    assert result.outcome_reward == 1.0
    assert result.metadata["applied_latency_penalty"] == 0.0
    assert result.metadata["latency_penalty_lambda"] == 0.0
    assert result.metadata["total_latency_seconds"] == pytest.approx(180.0)
    assert result.metadata["total_latency_valid"] is True


@pytest.mark.asyncio
async def test_latency_penalty_shapes_success_gated(tmp_path, monkeypatch) -> None:
    evaluator = _evaluator(
        tmp_path,
        monkeypatch,
        latency_penalty_lambda=0.5,
        latency_normalizer=360.0,
    )

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        agent_result=_agent_result(_valid_router_metadata()),
    )

    # 180 s total latency: 0.5 * 180 / 360 = 0.25 penalty fraction.
    assert result.outcome_reward == pytest.approx(0.75)
    assert result.metadata["applied_latency_penalty"] == pytest.approx(0.25)
    assert result.metadata["applied_total_penalty"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_cost_and_latency_penalties_are_additive_and_bounded(
    tmp_path, monkeypatch
) -> None:
    evaluator = _evaluator(
        tmp_path,
        monkeypatch,
        cost_penalty_lambda=0.2,
        cost_normalizer=1.0,
        latency_penalty_lambda=1.0,
        latency_normalizer=180.0,
    )

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        agent_result=_agent_result(_valid_router_metadata()),
    )

    # cost fraction min(1, 0.2*4/1)=0.8, latency fraction min(1, 180/180)=1.0;
    # the combined penalty is bounded at 1.0 so the reward floors at zero.
    assert result.outcome_reward == 0.0
    assert result.metadata["applied_cost_penalty"] == pytest.approx(0.8)
    assert result.metadata["applied_latency_penalty"] == pytest.approx(1.0)
    assert result.metadata["applied_total_penalty"] == pytest.approx(1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "calls",
    [
        None,
        [],
        [{"slot": "M0", "status": "completed"}],
        [{"slot": "M0", "status": "completed", "duration_ms": -1.0}],
        [{"slot": "M0", "status": "completed", "duration_ms": "soon"}],
    ],
)
async def test_latency_penalty_fails_closed_on_bad_call_metadata(
    tmp_path, monkeypatch, calls
) -> None:
    evaluator = _evaluator(tmp_path, monkeypatch, latency_penalty_lambda=0.1)
    metadata = _valid_router_metadata()
    if calls is None:
        metadata.pop("calls")
    else:
        metadata["calls"] = calls

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        agent_result=_agent_result(metadata),
    )

    assert result.outcome_reward == 0.0
    assert result.metadata["reward_override_reason"] == (
        "missing_or_invalid_total_latency"
    )
    assert result.metadata["total_latency_valid"] is False


@pytest.mark.asyncio
async def test_failed_rollout_is_not_latency_shaped(tmp_path, monkeypatch) -> None:
    evaluator = _evaluator(
        tmp_path,
        monkeypatch,
        harbor_reward=0.0,
        latency_penalty_lambda=1.0,
        latency_normalizer=1.0,
    )

    result = await evaluator.evaluate(
        Trajectory(status="COMPLETED"),
        agent_result=_agent_result(_valid_router_metadata()),
    )

    # Success gating: a failed rollout is never re-shaped, and bad latency
    # metadata on a failed rollout must not flip the reason either.
    assert result.outcome_reward == 0.0
    assert result.metadata["applied_latency_penalty"] == 0.0
    assert "reward_override_reason" not in result.metadata
