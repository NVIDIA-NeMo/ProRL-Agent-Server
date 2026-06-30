from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "src" / "polar" / "agent" / "presets" / "mini_swe_timing.py"


class _FakeLocalEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30


class _FakeSubmitted(Exception):
    pass


class _FakeLocalEnvironment:
    def __init__(self, *, config_class: type, **kwargs) -> None:
        self.config = config_class(**kwargs)
        self.last_action = None

    def execute(self, action, cwd="", *, timeout=None):
        del cwd, timeout
        self.last_action = action
        if action["command"] == "submit":
            raise _FakeSubmitted
        if action["command"] == "timeout":
            return {
                "returncode": -1,
                "extra": {"exception_type": "TimeoutExpired"},
            }
        if action["command"] == "failed":
            return {"returncode": -1, "extra": {"exception_type": "RuntimeError"}}
        if action["command"] == "binary-output":
            return {
                "returncode": 0,
                "output": "prefix\n" + ("\x00" * 7_500) + ("x" * 8_000) + "\nsuffix",
            }
        return {"returncode": 0, "output": "ok"}


def _load_timing_module(monkeypatch):
    package = ModuleType("minisweagent")
    environments = ModuleType("minisweagent.environments")
    local = ModuleType("minisweagent.environments.local")
    exceptions = ModuleType("minisweagent.exceptions")
    local.LocalEnvironment = _FakeLocalEnvironment
    local.LocalEnvironmentConfig = _FakeLocalEnvironmentConfig
    exceptions.Submitted = _FakeSubmitted
    monkeypatch.setitem(sys.modules, "minisweagent", package)
    monkeypatch.setitem(sys.modules, "minisweagent.environments", environments)
    monkeypatch.setitem(sys.modules, "minisweagent.environments.local", local)
    monkeypatch.setitem(sys.modules, "minisweagent.exceptions", exceptions)
    spec = importlib.util.spec_from_file_location("polar_mini_swe_timing_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timed_local_environment_records_git_diff_timeout_and_no_raw_command(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_timing_module(monkeypatch)
    timing_path = tmp_path / "timing.jsonl"
    environment = module.TimedLocalEnvironment(timing_path=str(timing_path))

    environment.execute({"command": "git diff -- super-secret-filename"})
    environment.execute({"command": "timeout"})
    environment.execute({"command": "failed"})

    records = [json.loads(line) for line in timing_path.read_text().splitlines()]
    assert records[0]["category"] == "git_diff"
    assert records[0]["return_code"] == 0
    assert records[1]["timed_out"] is True
    assert records[1]["return_code"] == -1
    assert records[2]["timed_out"] is False
    assert records[2]["return_code"] == -1
    assert "super-secret-filename" not in timing_path.read_text()
    summary = module.summarize_timing_records(timing_path)
    assert summary["count"] == 3
    assert summary["timeout_count"] == 1
    assert summary["failure_count"] == 1


def test_timed_local_environment_is_constructor_compatible(monkeypatch) -> None:
    module = _load_timing_module(monkeypatch)
    environment = module.TimedLocalEnvironment(env={"PAGER": "cat"}, timeout=7)

    assert isinstance(environment, _FakeLocalEnvironment)
    assert environment.config.env == {"PAGER": "cat"}
    assert environment.config.timeout == 7


def test_timed_environment_sanitizes_and_bounds_binary_output(monkeypatch) -> None:
    module = _load_timing_module(monkeypatch)
    environment = module.TimedLocalEnvironment(max_output_chars=1_000)

    result = environment.execute({"command": "binary-output"})

    assert len(result["output"]) <= 1_000
    assert "\x00" not in result["output"]
    assert "<control-bytes-elided>" in result["output"]
    assert "command output truncated" in result["output"]
    assert result["output"].startswith("prefix")
    assert result["output"].endswith("suffix")


def test_vanillux2_environment_wraps_commands_with_persistent_state(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_timing_module(monkeypatch)
    state_dir = tmp_path / "vanillux state"
    environment = module.Vanillux2TimedLocalEnvironment(state_dir=str(state_dir))

    wrapped = environment._command_for_execution(
        "cd /workspace && export VANILLUX_TEST=kept"
    )
    wrapped_tokens = shlex.split(wrapped)
    script = wrapped_tokens[2]

    assert environment.config.timeout == 120
    assert environment.config.max_output_chars == 10_000
    assert wrapped_tokens[:2] == ["bash", "-c"]
    assert f"mkdir -p '{state_dir}'" in script
    assert f". '{state_dir}/env'" in script
    assert f"pwd > '{state_dir}/cwd'" in script
    assert "cd /workspace && export VANILLUX_TEST=kept" in script
    assert "export -p >" in script
    assert "exit $_vanillux2_ec" in script


def test_vanillux2_environment_uses_paper_head_tail_observation(monkeypatch) -> None:
    module = _load_timing_module(monkeypatch)
    output = {"output": ("H" * 6_000) + ("T" * 6_000)}

    module._sanitize_vanillux2_output(output, max_chars=10_000)

    rendered = output["output"]
    assert rendered.startswith("The output of your last command was too long.")
    assert "---- HEAD (5000 chars) ----\n" + ("H" * 5_000) in rendered
    assert "---- 2000 chars elided ----" in rendered
    assert rendered.endswith("T" * 5_000)


def test_vanillux2_environment_strips_known_compose_noise_before_truncation(
    monkeypatch,
) -> None:
    module = _load_timing_module(monkeypatch)
    output = {
        "output": (
            '\x1b[4m>>>> Executing external compose provider "/usr/bin/docker-compose". '
            "Please see podman-compose(1) for how to disable this message. <<<<\n\n"
            "\x1b[0mactual output\n"
        )
    }

    module._sanitize_vanillux2_output(output, max_chars=10_000)

    assert output["output"] == "actual output"
