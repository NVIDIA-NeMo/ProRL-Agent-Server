from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

from polar.agent.factory import create_harness
from polar.agent.models import AgentRunResult
from polar.agent.models import AgentSpec
from polar.agent.presets.mini_swe_agent import MiniSweAgentHarness
from polar.agent.presets.vanillux2 import Vanillux2Harness


ROOT = Path(__file__).resolve().parents[2]


def test_mini_swe_agent_uses_gateway_and_bounded_steps() -> None:
    harness = MiniSweAgentHarness(
        AgentSpec(
            harness="mini_swe_agent",
            model_name="Qwen/Qwen3.5-4B",
            settings={"step_limit": 30, "cost_limit": 0},
        )
    )

    step = harness.run_steps("Fix the quoted 'bug'")[0]

    assert 'OPENAI_API_BASE="$OPENAI_BASE_URL"' in step.command
    assert "--model=openai/Qwen3.5-4B" in step.command
    assert "--cost-limit 0" in step.command
    assert "-c mini -c agent.step_limit=30" in step.command
    assert "-c environment.env.PYTHONPATH=" in step.command
    assert step.command.startswith("set -o pipefail; ")
    assert "--environment-class polar_mini_swe_timing.TimedLocalEnvironment" in step.command
    assert "-c environment.timing_path=/polar/session/logs/agent/" in step.command
    assert step.env["MSWEA_CONFIGURED"] == "true"
    assert step.env["MSWEA_COST_TRACKING"] == "ignore_errors"
    assert step.env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] == "3"
    assert step.env["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"


def test_mini_swe_agent_model_retry_attempts_are_configurable() -> None:
    harness = MiniSweAgentHarness(
        AgentSpec(
            harness="mini_swe_agent",
            model_name="Qwen/Qwen3.5-4B",
            settings={"model_retry_attempts": 2},
        )
    )

    step = harness.run_steps("Inspect the environment")[0]

    assert step.env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] == "2"


def test_mini_swe_agent_always_isolates_action_pythonpath() -> None:
    harness = MiniSweAgentHarness(
        AgentSpec(harness="mini_swe_agent", model_name="Qwen/Qwen3.5-4B")
    )

    step = harness.run_steps("Inspect the environment")[0]

    assert "-c mini -c environment.env.PYTHONPATH=" in step.command


def test_mini_swe_agent_passes_eval_model_kwargs_and_per_sample_seed() -> None:
    harness = MiniSweAgentHarness(
        AgentSpec(
            harness="mini_swe_agent",
            model_name="Qwen/Qwen3.5-4B",
            settings={
                "model_kwargs": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_tokens": 4096,
                    "extra_body": {"top_k": 32},
                },
                "sampling_seed": 1235,
            },
        )
    )

    step = harness.run_steps("Evaluate this task")[0]
    config_arg = next(
        token.removeprefix("model.model_kwargs=")
        for token in shlex.split(step.command)
        if token.startswith("model.model_kwargs=")
    )

    assert json.loads(config_arg) == {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 4096,
        "extra_body": {"top_k": 32},
        "seed": 1235,
    }


def test_vanillux2_uses_paper_protocol_defaults() -> None:
    harness = create_harness(
        AgentSpec(
            harness="vanillux2",
            model_name="Qwen/Qwen3.5-9B",
            settings={"sampling_seed": 7},
        )
    )

    assert isinstance(harness, Vanillux2Harness)
    step = harness.run_steps("Repair the environment")[0]
    config_arg = next(
        token.removeprefix("model.model_kwargs=")
        for token in shlex.split(step.command)
        if token.startswith("model.model_kwargs=")
    )

    assert "-c /opt/polar-mini-swe-agent/config/vanillux2.yaml" in step.command
    assert "--model-class polar_mini_swe_vanillux.Vanillux2LitellmModel" in step.command
    assert (
        "--environment-class "
        "polar_mini_swe_timing.Vanillux2TimedLocalEnvironment"
    ) in step.command
    assert "-c agent.step_limit=64" in step.command
    assert "-c agent.max_consecutive_format_errors=64" in step.command
    assert "-c environment.timeout=120" in step.command
    assert "-c environment.max_output_chars=10000" in step.command
    assert "-c model.response_token_budget=65536" in step.command
    assert json.loads(config_arg) == {
        "max_tokens": 16_384,
        "seed": 7,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    assert step.env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] == "5"


def test_vanillux2_protocol_limits_remain_configurable() -> None:
    harness = Vanillux2Harness(
        AgentSpec(
            harness="vanillux2",
            model_name="Qwen/Qwen3.5-9B",
            settings={
                "step_limit": 31,
                "command_timeout": 45,
                "observation_max_chars": 8_000,
                "max_format_errors": 9,
                "response_token_budget": 32_768,
                "model_retry_attempts": 2,
                "model_kwargs": {"max_tokens": 4_096, "temperature": 0.2},
            },
        )
    )

    step = harness.run_steps("Repair the environment")[0]
    config_arg = next(
        token.removeprefix("model.model_kwargs=")
        for token in shlex.split(step.command)
        if token.startswith("model.model_kwargs=")
    )

    assert "-c agent.step_limit=31" in step.command
    assert "-c agent.max_consecutive_format_errors=9" in step.command
    assert "-c environment.timeout=45" in step.command
    assert "-c environment.max_output_chars=8000" in step.command
    assert "-c model.response_token_budget=32768" in step.command
    assert json.loads(config_arg) == {
        "max_tokens": 4_096,
        "temperature": 0.2,
        "top_p": 0.95,
    }
    assert step.env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] == "2"


def test_shared_mini_swe_runtime_wrapper_is_relocatable() -> None:
    script = (ROOT / "examples" / "tmax_slime_grpo" / "prepare_mini_swe_agent.sh").read_text()
    wrapper = script.split("<<'SH'", 1)[1].split("\nSH", 1)[0]

    assert "--relocatable" in script
    assert "ln -sfn ../../python/bin/python3" in script
    assert "export PYTHONPATH=" not in wrapper
    assert 'export POLAR_TASK_PYTHONPATH="${PYTHONPATH}"' in wrapper
    assert "unset PYTHONPATH" in wrapper
    assert 'exec "${venv_python}" -m polar_mini_swe_runner' in wrapper
    assert 'install -m 0644 "${RUNNER_MODULE_SOURCE}"' in script
    assert "polar_mini_swe_timing.py" in script
    assert "polar_mini_swe_vanillux.py" in script
    assert '"${staging}/config/vanillux2.yaml"' in script
    assert "TIMING_SCHEMA_VERSION == 1" in script


def test_mini_swe_preflight_uses_production_mount_path() -> None:
    script = (ROOT / "examples" / "tmax_slime_grpo" / "preflight_runtime.py").read_text()

    assert 'CONTAINER_RUNTIME_DIR = "/opt/polar-mini-swe-agent"' in script
    assert 'runtime_dir / "bin/mini-swe-agent"' in script
    assert "polar_mini_swe_runner" in script
    assert "polar_mini_swe_timing" in script
    assert "polar_mini_swe_vanillux" in script
    assert "config/vanillux2.yaml" in script
    assert "TIMING_SCHEMA_VERSION == 1" in script
    assert "not in python_path" in script
    assert "p.startswith('{CONTAINER_RUNTIME_DIR}')" in script


def test_tmax_launch_contract_accepts_vanillux2() -> None:
    env_script = (ROOT / "examples" / "tmax_slime_grpo" / "env.cwdfw.sh").read_text()
    submit_script = (
        ROOT / "examples" / "tmax_slime_grpo" / "submit_slurm.sh"
    ).read_text()

    assert "mini_swe_agent|vanillux2)" in env_script
    assert "mini_swe_agent|vanillux2)" in submit_script
    assert "polar_mini_swe_vanillux" in submit_script
    assert "config/vanillux2.yaml" in submit_script


def test_mini_swe_postprocess_aggregates_fixed_categories(tmp_path: Path) -> None:
    timing_path = tmp_path / "logs" / "agent" / "mini-swe-command-timing.jsonl"
    timing_path.parent.mkdir(parents=True)
    timing_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "category": "git_diff",
                        "duration_ms": 12.5,
                        "return_code": 0,
                        "timed_out": False,
                        "command": "must not propagate",
                    }
                ),
                json.dumps(
                    {
                        "category": "test",
                        "duration_ms": 20.0,
                        "return_code": -1,
                        "timed_out": True,
                    }
                ),
                json.dumps(
                    {
                        "category": "secret-dynamic-category",
                        "duration_ms": 999.0,
                        "return_code": 0,
                    }
                ),
            ]
        )
    )
    runtime = SimpleNamespace(
        resolve_host_path=lambda _path: timing_path,
    )
    result = AgentRunResult(status="timeout", return_code=-1)
    harness = MiniSweAgentHarness(
        AgentSpec(harness="mini_swe_agent", model_name="Qwen/Qwen3.5-4B")
    )

    asyncio.run(harness.postprocess(runtime, result))

    summary = result.metadata["mini_swe_command_timing"]
    assert summary["count"] == 2
    assert summary["total_ms"] == 32.5
    assert summary["timeout_count"] == 1
    assert summary["ms_by_category"]["git_diff"] == 12.5
    assert "secret-dynamic-category" not in summary["ms_by_category"]
    assert "command" not in repr(summary)
