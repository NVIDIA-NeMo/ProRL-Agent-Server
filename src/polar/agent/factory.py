"""Harness factory with built-in name map and import_path support."""

from __future__ import annotations

from polar.agent.base import BaseHarness
from polar.agent.models import AgentSpec
from polar._imports import import_subclass


def _builtin_harness_map() -> dict[str, type[BaseHarness]]:
    """Lazy import to avoid circular imports at module level."""
    from polar.agent.presets.claude_code import ClaudeCodeHarness
    from polar.agent.presets.codex import CodexHarness
    from polar.agent.presets.gemini_cli import GeminiCliHarness
    from polar.agent.presets.hermes import HermesHarness
    from polar.agent.presets.mini_swe_agent import MiniSweAgentHarness
    from polar.agent.presets.openclaw import OpenClawHarness
    from polar.agent.presets.openhands_sdk import OpenHandsSdkHarness
    from polar.agent.presets.opencode import OpenCodeHarness
    from polar.agent.presets.pi import PiHarness
    from polar.agent.presets.qwen_code import QwenCodeHarness
    from polar.agent.presets.shell import ShellHarness

    return {
        "claude_code": ClaudeCodeHarness,
        "codex": CodexHarness,
        "gemini_cli": GeminiCliHarness,
        "hermes": HermesHarness,
        "mini_swe_agent": MiniSweAgentHarness,
        "openclaw": OpenClawHarness,
        "openhands_sdk": OpenHandsSdkHarness,
        "opencode": OpenCodeHarness,
        "pi": PiHarness,
        "qwen_code": QwenCodeHarness,
        "shell": ShellHarness,
    }


def create_harness(agent_spec: AgentSpec) -> BaseHarness:
    """Resolve and instantiate a harness from an AgentSpec."""
    if agent_spec.import_path is not None:
        cls = _import_harness_class(agent_spec.import_path)
        return cls(agent_spec)

    if agent_spec.harness is not None:
        harness_map = _builtin_harness_map()
        cls = harness_map.get(agent_spec.harness)
        if cls is None:
            raise ValueError(f"Unknown harness: {agent_spec.harness!r}")
        return cls(agent_spec)

    raise ValueError("AgentSpec must specify harness or import_path")


def _import_harness_class(import_path: str) -> type[BaseHarness]:
    return import_subclass(import_path, BaseHarness, kind="harness import path")
