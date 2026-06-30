"""Low-cardinality classification for commands executed inside runtimes.

The classifier deliberately returns only a fixed category.  Command text can
contain task data, credentials, or other high-cardinality values and must never
be copied into telemetry.
"""

from __future__ import annotations

import re


RUNTIME_EXEC_CATEGORIES: tuple[str, ...] = (
    "agent",
    "git_diff",
    "git_status",
    "git_other",
    "verifier",
    "test",
    "package_install",
    "build",
    "filesystem",
    "shell_other",
)


def classify_runtime_command(command: str) -> str:
    """Map arbitrary shell text to a stable, non-sensitive category."""

    normalized = " ".join(command.lower().split())

    # Check agent launchers first.  Their --task argument can itself mention
    # any command below and must not affect the outer runtime classification.
    if any(
        marker in normalized
        for marker in (
            "mini-swe-agent",
            "claude-code",
            "claude ",
            "codex exec",
            "gemini ",
            "openhands",
            "opencode ",
            "qwen-code",
        )
    ):
        return "agent"

    git_segment = re.search(r"(?:^|[;&|()])\s*git\b([^;&|()]*)", normalized)
    if git_segment is not None:
        git_args = git_segment.group(1)
        if re.search(r"(?:^|\s)diff(?:\s|$)", git_args):
            return "git_diff"
        if re.search(r"(?:^|\s)status(?:\s|$)", git_args):
            return "git_status"
        return "git_other"

    if "/tests/test.sh" in normalized or "/logs/verifier" in normalized:
        return "verifier"
    if re.search(
        r"(?:^|[;&|()]\s*)(?:python\s+-m\s+)?(?:pytest|tox|nox|go\s+test|cargo\s+test)\b",
        normalized,
    ) or re.search(r"\b(?:npm|pnpm|yarn)\s+(?:run\s+)?test\b", normalized):
        return "test"
    if re.search(
        r"\b(?:pip|uv\s+pip|apt-get|apt|dnf|yum|apk|npm|pnpm|yarn)\s+install\b",
        normalized,
    ):
        return "package_install"
    if re.search(
        r"(?:^|[;&|()]\s*)(?:make|cmake|ninja|cargo\s+build|go\s+build)\b",
        normalized,
    ) or re.search(r"\b(?:npm|pnpm|yarn)\s+(?:run\s+)?build\b", normalized):
        return "build"
    if re.search(
        r"(?:^|[;&|()]\s*)(?:mkdir|rm|cp|mv|chmod|chown|cat|tar|find|ls|sed|head|tail)\b",
        normalized,
    ):
        return "filesystem"
    return "shell_other"
