"""mini-SWE-agent harness — https://github.com/swe-agent/mini-swe-agent"""

from __future__ import annotations

import shlex

from polar.agent.base import BaseHarness
from polar.runtime.base import RUNTIME_AGENT_LOG_DIR
from polar.runtime.models import ExecInput


class MiniSweAgentHarness(BaseHarness):
    """Run mini-SWE-agent non-interactively (``mini-swe-agent --yolo``).

    mini-SWE-agent talks to models only through LiteLLM. The gateway exposes an
    OpenAI-compatible endpoint and injects ``OPENAI_BASE_URL`` + ``OPENAI_API_KEY``
    into the run step, but LiteLLM's OpenAI provider reads the base URL from
    ``OPENAI_API_BASE`` — so we alias it. The model is forced onto the ``openai/``
    provider so LiteLLM uses that transport; the gateway rewrites the model id to
    the served model regardless, so the id itself is cosmetic.

    Mirrors Harbor's mini-swe-agent setup: ``--yolo`` (no confirmations),
    ``--cost-limit 0`` (disabled) plus ``MSWEA_COST_TRACKING=ignore_errors`` so the
    served model's missing price table doesn't error, ``MSWEA_CONFIGURED=true`` to
    skip the first-run interactive setup, and ``--exit-immediately`` to finish
    without prompting.
    """

    def run_steps(self, instruction: str) -> list[ExecInput]:
        model_id = (self.model_name or "gpt-5.4").rsplit("/", 1)[-1]
        cost_limit = self.settings.get("cost_limit", 0)

        flags = [
            "--yolo",
            f"--model={shlex.quote(f'openai/{model_id}')}",
            f"--task={shlex.quote(instruction)}",
            f"--cost-limit {shlex.quote(str(cost_limit))}",
            "--exit-immediately",
        ]
        # -c is a spec *list*, not a merge over the defaults: any -c suppresses
        # the packaged config, so prepend "-c mini" before layering overrides.
        step_limit = self.settings.get("step_limit")
        if step_limit is not None:
            flags.append(f"-c mini -c agent.step_limit={int(step_limit)}")
        flags_str = " ".join(flags)

        return [
            ExecInput(
                command=(
                    # uv tool drops the entry point in $HOME/.local/bin.
                    'export PATH="$HOME/.local/bin:$PATH" && '
                    # LiteLLM reads OPENAI_API_BASE; the gateway only sets OPENAI_BASE_URL.
                    'export OPENAI_API_BASE="$OPENAI_BASE_URL" && '
                    f"mini-swe-agent {flags_str} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/mini-swe-agent.txt"
                ),
                env={
                    **self.env,
                    "MSWEA_CONFIGURED": "true",
                    "MSWEA_COST_TRACKING": "ignore_errors",
                },
            )
        ]
