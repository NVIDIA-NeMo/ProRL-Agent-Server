from __future__ import annotations

import asyncio

from polar.agent.models import AgentSpec
from polar.agent.presets.codex import CodexHarness


class FakeRuntime:
    spec = type("Spec", (), {"workdir": "/work"})()
    runtime_session_dir = "/session"

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command: str):
        self.commands.append(command)
        return type("Result", (), {"return_code": 0, "stdout": "", "stderr": ""})()


def test_codex_empty_version_keeps_binary_check_without_version_check() -> None:
    runtime = FakeRuntime()
    harness = CodexHarness(
        AgentSpec(
            harness="codex",
            settings={"version": ""},
        )
    )

    asyncio.run(harness.setup(runtime))  # type: ignore[arg-type]

    assert any("command -v codex" in command for command in runtime.commands)
    assert not any("codex --version" in command for command in runtime.commands)
