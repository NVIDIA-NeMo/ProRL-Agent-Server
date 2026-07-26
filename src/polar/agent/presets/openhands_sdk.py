"""OpenHands SDK harness — https://github.com/OpenHands/software-agent-sdk

Unlike the CLI presets, OpenHands ships as a Python SDK, so this preset writes a
small embedded runner script that builds the SDK agent and points its LLM at the
gateway (LLM_BASE_URL=$OPENAI_BASE_URL). Same idea, different launch shape.
"""

from __future__ import annotations

import json

from polar.agent.base import BaseHarness
from polar.agent.models import AgentSpec
from polar.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR
from polar.runtime.models import ExecInput


class OpenHandsSdkHarness(BaseHarness):
    """Run OpenHands SDK agent via an embedded runner script."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        self._runner_script = "/tmp/polar_openhands_sdk_run.py"

    async def setup(self, runtime: BaseRuntime) -> None:
        # Write the embedded runner script
        script = _RUNNER_SCRIPT
        await runtime.exec(
            f"cat > {self._runner_script} << 'POLARSCRIPT'\n{script}\nPOLARSCRIPT\n"
            f"chmod +x {self._runner_script}"
        )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        model = self.model_name or "openai/gpt-5.4"
        env: dict[str, str] = {
            **self.env,
            "LLM_MODEL": model,
            "AGENT_INSTRUCTION": instruction,
        }

        # Pass MCP servers as JSON env var
        if self.mcp_servers:
            servers = [
                {
                    "name": s.name,
                    "transport": s.transport,
                    **({"command": s.command} if s.command else {}),
                    **({"args": s.args} if s.args else {}),
                    **({"url": s.url} if s.url else {}),
                }
                for s in self.mcp_servers
            ]
            env["MCP_SERVERS_JSON"] = json.dumps(servers)

        # Pass skills path
        if self.skills_path:
            env["SKILL_PATHS"] = self.skills_path

        # Map settings to env
        for key, env_key in [
            ("max_iterations", "MAX_ITERATIONS"),
            ("temperature", "LLM_TEMPERATURE"),
            ("max_output_tokens", "LLM_MAX_OUTPUT_TOKENS"),
        ]:
            value = self.settings.get(key)
            if value is not None:
                env[env_key] = str(value)

        env.setdefault("MAX_ITERATIONS", "8")

        return [
            ExecInput(
                command=(
                    'export LLM_API_KEY="$OPENAI_API_KEY" LLM_BASE_URL="$OPENAI_BASE_URL" && '
                    'PYTHON_BIN="$HOME/.venv/bin/python"; '
                    '[ -x "$PYTHON_BIN" ] || PYTHON_BIN="/opt/openhands-sdk-venv/bin/python"; '
                    '[ -x "$PYTHON_BIN" ] || PYTHON_BIN="$(command -v python3 || command -v python)"; '
                    '"$PYTHON_BIN" '
                    f"{self._runner_script} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/openhands-sdk.txt"
                ),
                env=env,
            )
        ]


_RUNNER_SCRIPT = r'''#!/usr/bin/env python3
"""OpenHands SDK runner for Polar."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _load_skills(skill_paths_raw: str) -> list[object]:
    if not skill_paths_raw:
        return []

    from openhands.sdk.context import Skill

    skills: list[object] = []
    seen: set[str] = set()
    for base_path_str in skill_paths_raw.split(":"):
        if not base_path_str:
            continue
        base_path = Path(base_path_str)
        if not base_path.exists():
            continue
        for skill_dir in base_path.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            name = skill_dir.name
            if name in seen:
                continue
            seen.add(name)
            skills.append(
                Skill(
                    name=name,
                    content=skill_file.read_text(),
                    source=str(skill_file),
                    trigger=None,
                )
            )
    return skills


def _load_mcp_config() -> dict[str, object] | None:
    raw = os.environ.get("MCP_SERVERS_JSON")
    if not raw:
        return None

    mcp_servers = json.loads(raw)
    config: dict[str, object] = {"mcpServers": {}}
    for server in mcp_servers:
        if not isinstance(server, dict):
            continue
        name = str(server.get("name") or "mcp-server")
        transport = str(server.get("transport") or "stdio")
        server_cfg: dict[str, object] = {}
        if transport == "stdio":
            command = server.get("command")
            if command:
                server_cfg["command"] = command
            args = server.get("args")
            if args:
                server_cfg["args"] = args
        else:
            url = server.get("url")
            if url:
                server_cfg["url"] = url
        config["mcpServers"][name] = server_cfg
    return config if config["mcpServers"] else None


def main():
    os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
    try:
        from openhands.sdk import Agent, AgentContext, Conversation, Tool
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.task_tracker import TaskTrackerTool
        from openhands.tools.terminal import TerminalTool
    except ImportError as e:
        print(f"openhands-sdk not installed: {e}", file=sys.stderr)
        sys.exit(1)

    # Import LLM — try top-level first (newer SDK), then submodule
    try:
        from openhands.sdk import LLM
    except ImportError:
        from openhands.sdk.llm import LLM

    instruction = os.environ.get("AGENT_INSTRUCTION", "")
    model = os.environ.get("LLM_MODEL", "openai/gpt-5.4")
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    max_iterations = int(os.environ.get("MAX_ITERATIONS", "30"))
    temperature = os.environ.get("LLM_TEMPERATURE")
    max_output_tokens = os.environ.get("LLM_MAX_OUTPUT_TOKENS")

    llm_kwargs = dict(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    if temperature:
        llm_kwargs["temperature"] = float(temperature)
    if max_output_tokens:
        llm_kwargs["max_output_tokens"] = int(max_output_tokens)
    llm = LLM(**llm_kwargs)

    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ]

    agent_kwargs = {
        "llm": llm,
        "tools": tools,
        "agent_context": AgentContext(
            skills=_load_skills(os.environ.get("SKILL_PATHS", ""))
        ),
    }
    mcp_config = _load_mcp_config()
    if mcp_config is not None:
        agent_kwargs["mcp_config"] = mcp_config

    agent = Agent(**agent_kwargs)
    workspace = os.environ.get("WORKSPACE_BASE") or os.getcwd()

    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        max_iteration_per_run=max_iterations,
    )
    conversation.send_message(instruction)
    conversation.run()
    conversation.close()

if __name__ == "__main__":
    main()
'''
