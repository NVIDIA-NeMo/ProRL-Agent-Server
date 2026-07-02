"""pi coding agent harness — https://github.com/mariozechner/pi-coding-agent"""

from __future__ import annotations

import json
import shlex

from polar.agent.base import BaseHarness
from polar.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR
from polar.runtime.models import ExecInput


class PiHarness(BaseHarness):
    """Run the pi coding agent CLI against the Polar gateway.

    pi bakes provider base URLs into its model registry — there is no
    ``OPENAI_BASE_URL`` equivalent — so we drop a ``models.json`` into pi's
    agent dir that overrides the chosen provider's ``baseUrl`` to the gateway
    and registers the requested model under it.

    ``model_name`` must be ``provider/model`` form (e.g., ``openai/gpt-5.4``
    or ``openai/Qwen/Qwen3-4B-Instruct-2507``). Everything after the first
    ``/`` becomes the model id; provider names match pi's built-ins.
    """

    _AGENT_DIR = "$HOME/.pi/agent"

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._AGENT_DIR}")

        if self.skills_path:
            await runtime.exec(
                f"mkdir -p {self._AGENT_DIR}/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* {self._AGENT_DIR}/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError(
                "pi harness requires model_name in 'provider/model' form "
                f"(got {self.model_name!r})"
            )
        provider, model_id = self.model_name.split("/", 1)
        api_type = str(self.settings.get("api_type", "openai-completions"))

        compat = {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "supportsStore": False,
            "maxTokensField": "max_tokens",
        }
        compat.update(dict(self.settings.get("compat") or {}))

        models_config = {
            "providers": {
                provider: {
                    "baseUrl": "__POLAR_GATEWAY_BASE_URL__",
                    "compat": compat,
                    "models": [
                        {
                            "id": model_id,
                            "api": api_type,
                            "contextWindow": int(
                                self.settings.get("context_window", 262144)
                            ),
                            "maxTokens": int(self.settings.get("max_tokens", 8192)),
                        }
                    ],
                },
            },
        }
        config_json = json.dumps(models_config)

        flags: list[str] = []
        thinking = self.settings.get("thinking")
        if thinking is not None:
            flags.append(f"--thinking {shlex.quote(str(thinking))}")
        flags_str = (" " + " ".join(flags)) if flags else ""

        escaped = shlex.quote(instruction)

        return [
            ExecInput(
                command=(
                    f"mkdir -p {self._AGENT_DIR} && "
                    # $OPENAI_BASE_URL is substituted at exec time; the
                    # placeholder keeps the JSON static (no shell quoting fun).
                    f"printf '%s' {shlex.quote(config_json)} "
                    f'| sed "s|__POLAR_GATEWAY_BASE_URL__|$OPENAI_BASE_URL|g" '
                    f"> {self._AGENT_DIR}/models.json && "
                    f"pi --print --mode json --no-session "
                    f"--provider {shlex.quote(provider)} "
                    f"--model {shlex.quote(model_id)}"
                    f"{flags_str} "
                    f"{escaped} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/pi.txt"
                ),
                env={**self.env, "PI_OFFLINE": "1"},
            )
        ]
