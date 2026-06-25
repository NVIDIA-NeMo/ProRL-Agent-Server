"""Hermes harness — https://github.com/NousResearch/hermes-agent"""

from __future__ import annotations

import json
import shlex

from polar.agent.base import BaseHarness
from polar.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR
from polar.runtime.models import ExecInput

# Isolated agent home so Hermes' sessions/skills/memory stay out of the
# workspace git diff. Read via the HERMES_HOME env var at run time.
_HERMES_HOME = "/tmp/hermes"
# A user-defined provider pins the OpenAI-compatible transport + gateway base
# URL explicitly. Hermes' built-in providers and `provider: auto` route by
# models.dev heuristics (and can mis-detect the model vendor), so we bypass them.
_PROVIDER = "polar"
# Substituted with $OPENAI_BASE_URL at exec time (the gateway env is only
# present during run steps, not setup), keeping the config static.
_BASE_URL_PLACEHOLDER = "__POLAR_GATEWAY_BASE_URL__"


class HermesHarness(BaseHarness):
    """Run NousResearch's Hermes agent non-interactively (``hermes chat -q``).

    Routes through a user-defined ``providers.polar`` entry whose ``base_url`` is
    the gateway proxy and whose ``key_env`` is ``OPENAI_API_KEY`` (injected by the
    gateway as the session id), forcing the ``openai_chat`` wire format.
    """

    async def setup(self, runtime: BaseRuntime) -> None:
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p {_HERMES_HOME}/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* "
                f"{_HERMES_HOME}/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        # Bare model id (no provider prefix); the gateway rewrites it anyway.
        model_id = (self.model_name or "gpt-5.4").rsplit("/", 1)[-1]
        config_json = json.dumps(self._build_config())
        escaped = shlex.quote(instruction)

        # -q: single non-interactive query; -Q: quiet (final response only);
        # --yolo: bypass tool-approval prompts.
        flags = [
            "--yolo",
            "chat",
            f"-q {escaped}",
            "-Q",
            f"--model {shlex.quote(model_id)}",
            f"--provider {_PROVIDER}",
        ]
        toolsets = self.settings.get("toolsets")
        if toolsets is not None:
            flags.append(f"--toolsets {shlex.quote(str(toolsets))}")
        flags_str = " ".join(flags)

        return [
            ExecInput(
                command=(
                    f"mkdir -p {_HERMES_HOME} && "
                    # base_url placeholder -> $OPENAI_BASE_URL at exec time.
                    f"printf '%s' {shlex.quote(config_json)} "
                    f'| sed "s|{_BASE_URL_PLACEHOLDER}|$OPENAI_BASE_URL|g" '
                    f"> {_HERMES_HOME}/config.yaml && "
                    'export PATH="$HOME/.local/bin:$PATH" && '
                    f"hermes {flags_str} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/hermes.txt"
                ),
                env={**self.env, "HERMES_HOME": _HERMES_HOME, "TERMINAL_ENV": "local"},
            )
        ]

    def _build_config(self) -> dict:
        config: dict = {
            "providers": {
                _PROVIDER: {
                    "base_url": _BASE_URL_PLACEHOLDER,
                    "key_env": "OPENAI_API_KEY",
                    "transport": "openai_chat",
                }
            },
            "toolsets": ["hermes-cli"],
            "model": {"context_length": int(self.settings.get("context_length", 262_144))},
            "agent": {"max_turns": int(self.settings.get("max_turns", 90))},
            # Disable the self-improvement loop so runs stay stateless and don't
            # write memory/profile files into the agent home.
            "memory": {"memory_enabled": False, "user_profile_enabled": False},
            "terminal": {"backend": "local", "timeout": 180},
            "checkpoints": {"enabled": False},
        }

        if self.mcp_servers:
            servers: dict[str, dict] = {}
            for server in self.mcp_servers:
                if server.transport == "stdio":
                    entry: dict = {"command": server.command}
                    if server.args:
                        entry["args"] = server.args
                else:
                    entry = {"url": server.url}
                servers[server.name] = entry
            config["mcp_servers"] = servers

        return config
