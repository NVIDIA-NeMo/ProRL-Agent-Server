from __future__ import annotations

from polar.agent.models import AgentSpec
from polar.agent.presets.mini_swe_agent import MiniSweAgentHarness


def test_mini_swe_agent_uses_gateway_and_bounded_steps() -> None:
    harness = MiniSweAgentHarness(
        AgentSpec(
            harness="mini_swe_agent",
            model_name="Qwen/Qwen3.5-4B",
            settings={"step_limit": 30, "cost_limit": 0},
        )
    )

    step = harness.run_steps("Fix the quoted 'bug'")[0]

    assert 'OPENAI_API_BASE="$OPENAI_BASE_URL"' in step.command
    assert "--model=openai/Qwen3.5-4B" in step.command
    assert "--cost-limit 0" in step.command
    assert "-c mini -c agent.step_limit=30" in step.command
    assert step.env["MSWEA_CONFIGURED"] == "true"
    assert step.env["MSWEA_COST_TRACKING"] == "ignore_errors"
