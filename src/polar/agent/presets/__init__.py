"""Preset harnesses — ready-made launchers for popular agents.

These are *conveniences*, not integrations. Polar runs any agent unmodified:
a harness only starts the agent process and lets its LLM calls flow through
the gateway proxy (which serves the model and captures the trajectory). Each
preset here is a thin ``BaseHarness`` that writes the agent's config and emits
its run command — typically a few dozen lines.

To run an agent that isn't listed here, you do **not** add code to Polar:
- use the ``shell`` preset to wrap any command, or
- point ``agent.import_path`` at your own ``BaseHarness`` subclass.

See ``polar/agent/README.md`` for the contract and a "bring your own" guide.
"""

from polar.agent.presets.claude_code import ClaudeCodeHarness
from polar.agent.presets.codex import CodexHarness
from polar.agent.presets.gemini_cli import GeminiCliHarness
from polar.agent.presets.hermes import HermesHarness
from polar.agent.presets.mini_swe_agent import MiniSweAgentHarness
from polar.agent.presets.openclaw import OpenClawHarness
from polar.agent.presets.opencode import OpenCodeHarness
from polar.agent.presets.openhands_sdk import OpenHandsSdkHarness
from polar.agent.presets.pi import PiHarness
from polar.agent.presets.qwen_code import QwenCodeHarness
from polar.agent.presets.shell import ShellHarness

__all__ = [
    "ClaudeCodeHarness",
    "CodexHarness",
    "GeminiCliHarness",
    "HermesHarness",
    "MiniSweAgentHarness",
    "OpenClawHarness",
    "OpenCodeHarness",
    "OpenHandsSdkHarness",
    "PiHarness",
    "QwenCodeHarness",
    "ShellHarness",
]
