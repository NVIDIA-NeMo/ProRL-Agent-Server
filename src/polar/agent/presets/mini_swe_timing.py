"""Timed mini-SWE LocalEnvironment installed into the portable CLI runtime.

This file is copied as the top-level module ``polar_mini_swe_timing`` by
``prepare_mini_swe_agent.sh``.  Keep it self-contained: task containers mount
the mini-SWE runtime, not the Polar Python environment.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any

from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from minisweagent.exceptions import Submitted
from pydantic import Field


TIMING_SCHEMA_VERSION = 1
COMMAND_CATEGORIES: tuple[str, ...] = (
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
_CONTROL_RUN_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]+")
_COMPOSE_PROVIDER_RE = re.compile(
    r'\x1b\[4m>>>> Executing external compose provider "[^"]*docker-compose"\. '
    r"Please see podman-compose\(1\) for how to disable this message\. <<<<\n\n"
    r"\x1b\[0m"
)
_DOCKER_EXEC_ERROR_RE = re.compile(
    r"(?ms)^Error: executing [^\n]*(?:docker-compose|docker compose)"
    r".*?: exit status \d+\s*$"
)
_VANILLUX2_TOO_LONG_HINT = """The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a
smaller number of lines selectively. If you're using grep or find and it
produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you
can redirect output to a file and then search in that file."""


class TimedLocalEnvironmentConfig(LocalEnvironmentConfig):
    """Local environment settings plus an append-only timing destination."""

    timing_path: str = "/polar/session/logs/agent/mini-swe-command-timing.jsonl"
    # mini-SWE's packaged observation template JSON-escapes action output.
    # A 10k-character binary read full of NULs therefore becomes roughly 60k
    # prompt characters (``\\u0000`` per byte) and can overflow a 50k-token
    # model context in one turn.  Bound and sanitize the value *before* Jinja's
    # ``tojson`` filter while retaining a useful head/tail excerpt.
    max_output_chars: int = Field(default=4_000, ge=256, le=100_000)


class TimedLocalEnvironment(LocalEnvironment):
    """Record every mini-SWE action without retaining its command text."""

    def __init__(
        self,
        *,
        config_class: type = TimedLocalEnvironmentConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(config_class=config_class, **kwargs)

    def execute(
        self,
        action: dict,
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        command = str(action.get("command", ""))
        started_at = time.perf_counter()
        output: dict[str, Any] | None = None
        submitted = False
        try:
            execution_action = {
                **action,
                "command": self._command_for_execution(command),
            }
            output = super().execute(execution_action, cwd, timeout=timeout)
            self._sanitize_output(
                output, max_chars=int(self.config.max_output_chars)
            )
            return output
        except Submitted:
            submitted = True
            raise
        finally:
            duration_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
            return_code = _return_code(output, submitted=submitted)
            self._append_timing(
                {
                    "schema_version": TIMING_SCHEMA_VERSION,
                    "category": classify_mini_swe_command(command),
                    "duration_ms": duration_ms,
                    "return_code": return_code,
                    "timed_out": _timed_out(output),
                }
            )

    def _command_for_execution(self, command: str) -> str:
        return command

    def _sanitize_output(self, output: dict[str, Any], *, max_chars: int) -> None:
        _sanitize_action_output(output, max_chars=max_chars)

    def _append_timing(self, record: dict[str, object]) -> None:
        timing_path = str(getattr(self.config, "timing_path", "")).strip()
        if not timing_path:
            return
        try:
            path = Path(timing_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            encoded = (json.dumps(record, separators=(",", ":")) + "\n").encode()
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(descriptor, encoded)
            finally:
                os.close(descriptor)
        except OSError:
            # Timing must never make an agent action fail.
            return


class Vanillux2TimedLocalEnvironmentConfig(TimedLocalEnvironmentConfig):
    """Vanillux2 command-state and observation settings."""

    timeout: int = 120
    max_output_chars: int = Field(default=10_000, ge=256, le=100_000)
    state_dir: str = "/polar/session/home/.vanillux2"


class Vanillux2TimedLocalEnvironment(TimedLocalEnvironment):
    """Persist cwd/exported variables and apply Vanillux2 observation rules."""

    def __init__(
        self,
        *,
        config_class: type = Vanillux2TimedLocalEnvironmentConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(config_class=config_class, **kwargs)

    def _command_for_execution(self, command: str) -> str:
        state_dir = str(self.config.state_dir)
        state = shlex.quote(state_dir)
        cwd_path = shlex.quote(f"{state_dir}/cwd")
        env_path = shlex.quote(f"{state_dir}/env")
        script = (
            f"mkdir -p {state}\n"
            f"if [ ! -s {cwd_path} ]; then pwd > {cwd_path}; fi\n"
            f"if [ ! -s {env_path} ]; then export -p > {env_path}; fi\n"
            f'cd "$(cat {cwd_path})" 2>/dev/null || true\n'
            f". {env_path} 2>/dev/null || true\n"
            f"{command}\n"
            "_vanillux2_ec=$?\n"
            f"pwd > {cwd_path}\n"
            f"export -p > {env_path}\n"
            "exit $_vanillux2_ec"
        )
        # LocalEnvironment delegates to ``subprocess(..., shell=True)``, whose
        # shell may be dash. The protocol advertises a bash tool and official
        # Harbor execution accepts bash syntax, so select bash explicitly.
        return f"bash -c {shlex.quote(script)}"

    def _sanitize_output(self, output: dict[str, Any], *, max_chars: int) -> None:
        _sanitize_vanillux2_output(output, max_chars=max_chars)


def _sanitize_action_output(output: dict[str, Any], *, max_chars: int) -> None:
    """Prevent binary/control-heavy shell output from exploding chat prompts."""

    raw = output.get("output")
    if not isinstance(raw, str):
        return
    original_chars = len(raw)
    sanitized = _CONTROL_RUN_RE.sub("<control-bytes-elided>", raw)
    if len(sanitized) <= max_chars:
        output["output"] = sanitized
        return

    marker = (
        "\n... <command output truncated: "
        f"{original_chars} raw chars, {len(sanitized)} sanitized chars> ...\n"
    )
    excerpt_chars = max(0, max_chars - len(marker))
    head_chars = excerpt_chars // 2
    tail_chars = excerpt_chars - head_chars
    tail = sanitized[-tail_chars:] if tail_chars else ""
    output["output"] = sanitized[:head_chars] + marker + tail


def _sanitize_vanillux2_output(output: dict[str, Any], *, max_chars: int) -> None:
    """Match Vanillux2's compose cleanup and 10k head/tail observation."""

    raw = output.get("output")
    if not isinstance(raw, str):
        return
    # Keep the paper harness's raw command bytes intact. Its two targeted
    # cleanups must run before any generic ANSI/control handling, otherwise
    # stripping the leading escape byte prevents the compose regex matching.
    sanitized = _COMPOSE_PROVIDER_RE.sub("", raw)
    sanitized = _DOCKER_EXEC_ERROR_RE.sub("", sanitized).rstrip()
    if len(sanitized) <= max_chars:
        output["output"] = sanitized
        return

    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    elided = len(sanitized) - head_chars - tail_chars
    output["output"] = (
        f"{_VANILLUX2_TOO_LONG_HINT}\n\n"
        f"---- HEAD ({head_chars} chars) ----\n"
        f"{sanitized[:head_chars]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL ({tail_chars} chars) ----\n"
        f"{sanitized[-tail_chars:]}"
    )


def classify_mini_swe_command(command: str) -> str:
    """Return one fixed category without exposing task command contents."""

    normalized = " ".join(command.lower().split())
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
        r"(?:^|[;&|()]\s*)(?:mkdir|rm|cp|mv|chmod|chown|cat|tar|find|ls|sed|head|tail|rg|grep)\b",
        normalized,
    ):
        return "filesystem"
    return "shell_other"


def _return_code(output: dict[str, Any] | None, *, submitted: bool) -> int:
    if output is None:
        # Submitted is raised by LocalEnvironment after a successful submit
        # command. Other exceptional exits are conservatively marked failed.
        return 0 if submitted else -1
    try:
        return int(output.get("returncode", -1))
    except (TypeError, ValueError):
        return -1


def _timed_out(output: dict[str, Any] | None) -> bool:
    if not isinstance(output, dict):
        return False
    extra = output.get("extra")
    return isinstance(extra, dict) and extra.get("exception_type") == "TimeoutExpired"


def summarize_timing_records(path: Path, *, max_records: int = 10_000) -> dict[str, object]:
    """Parse and sanitize an action timing JSONL file for session telemetry."""

    ms_by_category = {category: 0.0 for category in COMMAND_CATEGORIES}
    count_by_category = {category: 0 for category in COMMAND_CATEGORIES}
    timeout_count = 0
    failure_count = 0
    try:
        with path.open() as stream:
            for index, line in enumerate(stream):
                if index >= max_records:
                    break
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(record, dict):
                    continue
                category = record.get("category")
                if category not in COMMAND_CATEGORIES:
                    continue
                try:
                    duration_ms = float(record.get("duration_ms"))
                    return_code = int(record.get("return_code"))
                except (TypeError, ValueError):
                    continue
                if duration_ms < 0.0 or not math.isfinite(duration_ms):
                    continue
                ms_by_category[category] += duration_ms
                count_by_category[category] += 1
                if bool(record.get("timed_out")):
                    timeout_count += 1
                elif return_code != 0:
                    failure_count += 1
    except (OSError, UnicodeError):
        pass
    return {
        "total_ms": sum(ms_by_category.values()),
        "count": sum(count_by_category.values()),
        "timeout_count": timeout_count,
        "failure_count": failure_count,
        "ms_by_category": ms_by_category,
        "count_by_category": count_by_category,
    }
