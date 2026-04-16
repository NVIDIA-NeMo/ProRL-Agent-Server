"""Codex CLI harness — https://github.com/openai/codex"""

from __future__ import annotations

import json
import shlex

from polar.agent.base import BaseHarness
from polar.agent.models import AgentSpec
from polar.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR
from polar.runtime.models import ExecInput


class CodexHarness(BaseHarness):
    """Run OpenAI Codex CLI in non-interactive mode."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        self._codex_home = "/root/.codex"

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._codex_home}")

        # Register MCP servers via TOML config
        if self.mcp_servers:
            toml_lines: list[str] = []
            for server in self.mcp_servers:
                toml_lines.append(f'[mcp_servers."{server.name}"]')
                if server.transport == "stdio":
                    toml_lines.append(f'command = "{server.command}"')
                    if server.args:
                        args_str = ", ".join(f'"{a}"' for a in server.args)
                        toml_lines.append(f"args = [{args_str}]")
                else:
                    toml_lines.append(f'url = "{server.url}"')
                    toml_lines.append(f'type = "{server.transport}"')
            toml_content = "\n".join(toml_lines)
            await runtime.exec(
                f"cat > {self._codex_home}/config.toml << 'POLARCFG'\n{toml_content}\nPOLARCFG"
            )

        # Copy skills
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p $HOME/.agents/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* $HOME/.agents/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)
        env: dict[str, str] = {
            **self.env,
            "CODEX_HOME": self._codex_home,
        }

        flags: list[str] = [
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
            "-c 'model_provider=\"harness_proxy\"'",
            "-c 'model_providers.harness_proxy.name=\"Harness Proxy\"'",
            '-c "model_providers.harness_proxy.base_url=\\"$OPENAI_BASE_URL\\""',
            "-c 'model_providers.harness_proxy.env_key=\"OPENAI_API_KEY\"'",
            "-c 'model_providers.harness_proxy.wire_api=\"responses\"'",
            '-c "model_providers.harness_proxy.http_headers={\\"X-Session-Id\\"=\\"$SESSION_ID\\"}"',
            "--disable responses_websockets",
            "--disable responses_websockets_v2",
            "--disable enable_request_compression",
        ]
        model = self.model_name or "o4-mini"
        flags.append(f"--model {shlex.quote(model)}")

        for key, cli in [
            ("reasoning_effort", "-c model_reasoning_effort"),
            ("reasoning_summary", "-c model_reasoning_summary"),
        ]:
            value = self.settings.get(key)
            if value is not None:
                flags.append(f"{cli}={shlex.quote(str(value))}")

        flags_str = " ".join(flags)
        return [
            ExecInput(
                command=(
                    f"set -o pipefail && "
                    f"codex exec {flags_str} -- {escaped} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/codex.txt"
                ),
                env=env,
            )
        ]
