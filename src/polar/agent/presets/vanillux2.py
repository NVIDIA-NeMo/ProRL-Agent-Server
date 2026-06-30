"""Tmax's Vanillux2 protocol on the shared mini-SWE-agent 2.x runtime."""

from __future__ import annotations

from polar.agent.presets.mini_swe_agent import MiniSweAgentHarness
from polar.runtime.models import ExecInput


VANILLUX2_CONFIG_PATH = "/opt/polar-mini-swe-agent/config/vanillux2.yaml"
VANILLUX2_MODEL_CLASS = "polar_mini_swe_vanillux.Vanillux2LitellmModel"
VANILLUX2_ENVIRONMENT_CLASS = (
    "polar_mini_swe_timing.Vanillux2TimedLocalEnvironment"
)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"vanillux2 {name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"vanillux2 {name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"vanillux2 {name} must be a positive integer")
    return parsed


class Vanillux2Harness(MiniSweAgentHarness):
    """Run the paper's bash-only Vanillux2 protocol without vendoring its loop.

    The portable mini-SWE runtime still owns model retries, trajectory output,
    and process lifecycle. A small config plus two injected adapters provide
    the protocol details that differ from stock mini-SWE: the official prompt,
    one bash call per turn, persistent cwd/exported environment, 10k head/tail
    observations, and the published generation/action limits.
    """

    def run_steps(self, instruction: str) -> list[ExecInput]:
        command_timeout = _positive_int(
            self.settings.get("command_timeout", 120),
            name="command_timeout",
        )
        max_format_errors = _positive_int(
            self.settings.get("max_format_errors", 64),
            name="max_format_errors",
        )
        observation_max_chars = _positive_int(
            self.settings.get("observation_max_chars", 10_000),
            name="observation_max_chars",
        )
        response_token_budget = _positive_int(
            self.settings.get("response_token_budget", 65_536),
            name="response_token_budget",
        )

        return self._run_mini_swe(
            instruction,
            config_spec=VANILLUX2_CONFIG_PATH,
            environment_class=VANILLUX2_ENVIRONMENT_CLASS,
            model_class=VANILLUX2_MODEL_CLASS,
            default_step_limit=64,
            default_model_kwargs={
                "max_tokens": 16_384,
                "temperature": 0.7,
                "top_p": 0.95,
            },
            default_model_retry_attempts=5,
            extra_config_specs=(
                f"agent.max_consecutive_format_errors={max_format_errors}",
                f"environment.timeout={command_timeout}",
                f"environment.max_output_chars={observation_max_chars}",
                f"model.response_token_budget={response_token_budget}",
            ),
        )
