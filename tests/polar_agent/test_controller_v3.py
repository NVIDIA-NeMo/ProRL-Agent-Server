from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from polar.agent.factory import create_harness
from polar.agent.models import AgentSpec
from polar.agent.models import AgentRunResult
from polar.agent.presets.controller_v3 import (
    CONTROLLER_V3_CONFIG_PATH,
    CONTROLLER_V3_MODULE_PATH,
    CONTROLLER_V3_RUNNER_PATH,
    ControllerV3Harness,
)


def _spec(**settings: object) -> AgentSpec:
    return AgentSpec(
        harness="controller_v3",
        model_name="Qwen3.6-35B-A3B",
        settings=settings,
    )


def _load_runner():
    path = Path(__file__).parents[2] / "src/polar/agent/presets/controller_v3_runner.py"
    spec = importlib.util.spec_from_file_location("controller_v3_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_factory_registers_controller_v3() -> None:
    assert isinstance(create_harness(_spec()), ControllerV3Harness)


def test_runner_uses_protected_auxiliary_file_descriptors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = tmp_path / "module-without-required-suffix"
    config = tmp_path / "config-without-required-suffix"
    module.write_text("SEALED_CONTROLLER_VALUE = 7\n")
    config.write_text("agent:\n  window_events: 8\n")
    module_fd = os.open(module, os.O_RDONLY)
    config_fd = os.open(config, os.O_RDONLY)
    try:
        monkeypatch.setenv("POLAR_CONTROLLER_V3_MODULE_FD", str(module_fd))
        monkeypatch.setenv("POLAR_CONTROLLER_V3_CONFIG_FD", str(config_fd))
        runner = _load_runner()
        runner._load_controller_module()
        loaded = importlib.import_module("minisweagent.agents.oracle_controller_v3")
        assert loaded.SEALED_CONTROLLER_VALUE == 7
        assert yaml.safe_load(runner.CONFIG_PATH.read_text()) == {
            "agent": {"window_events": 8}
        }
    finally:
        os.close(module_fd)
        os.close(config_fd)


def test_harness_uploads_dynamic_sources_and_uses_protected_capabilities() -> None:
    uploaded: list[tuple[str, str]] = []

    async def upload_file(source: str, target: str) -> None:
        uploaded.append((source, target))

    harness = ControllerV3Harness(_spec(step_limit=64))
    asyncio.run(harness.setup(SimpleNamespace(upload_file=upload_file)))
    step = harness.run_steps("fix the bug")[0]

    assert {target for _, target in uploaded} == {
        CONTROLLER_V3_RUNNER_PATH,
        CONTROLLER_V3_MODULE_PATH,
        CONTROLLER_V3_CONFIG_PATH,
    }
    assert step.protected_env_keys == [
        "POLAR_ROUTER_CAPABILITY",
        "POLAR_MODEL_POOL_CAPABILITY",
    ]
    assert step.protected_argv == [
        "/opt/polar-mini-swe-agent/python/bin/python3.10",
        CONTROLLER_V3_RUNNER_PATH,
    ]
    assert "fix the bug" not in step.protected_argv
    assert "Authorization" not in repr(step.env)
    assert "POLAR_ROUTER_CAPABILITY" not in step.env
    assert "POLAR_MODEL_POOL_CAPABILITY" not in step.env


def test_one_vote_aliases_and_responses_api_contract() -> None:
    path = (
        Path(__file__).parents[2]
        / "src/polar/agent/presets/controller_v3_one_vote.yaml"
    )
    config = yaml.safe_load(path.read_text())
    assert config["agent"]["controller_model"]["model_name"] == "openai/router/policy"
    assert (
        config["agent"]["small_model"]["model_name"]
        == "openai/pool/qwen3.6-35b-a3b"
    )
    assert config["model"]["model_name"] == "openai/pool/gpt-5.6-luna"
    assert config["model"]["model_class"] == "litellm_response"
    kwargs = config["model"]["model_kwargs"]
    assert kwargs["max_output_tokens"] == 65536
    assert kwargs["reasoning"] == {"effort": "max"}
    assert kwargs["max_tokens"] is None


def test_capability_header_is_temporary_on_success_and_failure() -> None:
    runner = _load_runner()
    config = SimpleNamespace(
        model_kwargs={"extra_headers": {"X-Test": "kept"}, "timeout": 1}
    )
    original = config.model_kwargs

    def inspect() -> str:
        assert config.model_kwargs["extra_headers"]["Authorization"] == "Bearer secret"
        return "ok"

    assert runner._with_capability(config, "secret", inspect) == "ok"
    assert config.model_kwargs is original

    def fail() -> None:
        assert config.model_kwargs["extra_headers"]["Authorization"] == "Bearer secret"
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        runner._with_capability(config, "secret", fail)
    assert config.model_kwargs is original
    assert "Authorization" not in config.model_kwargs["extra_headers"]


def test_capability_guards_cover_all_three_model_queries() -> None:
    runner = _load_runner()
    responses_client = object()

    class Model:
        def __init__(self) -> None:
            self.config = SimpleNamespace(model_kwargs={})
            self.seen: list[tuple[str, object | None]] = []

        def query(self, _messages):
            self.seen.append(
                (
                    self.config.model_kwargs["extra_headers"]["Authorization"],
                    self.config.model_kwargs.get("client"),
                )
            )
            return {
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 40},
                }
            }

    small = Model()
    large = Model()
    controller = Model()

    class Agent:
        small_model = small
        large_model = large
        controller_model = controller
        usage = {"large": {}}

        def _query_controller_model(self, _messages):
            controller.seen.append(
                (
                    controller.config.model_kwargs["extra_headers"]["Authorization"],
                    controller.config.model_kwargs.get("client"),
                )
            )
            return {}

    agent = Agent()
    runner._install_capability_guards(
        agent,
        router_capability="router-secret",
        pool_capability="pool-secret",
        responses_client=responses_client,
    )
    small.query([])
    large.query([])
    agent._query_controller_model([])

    assert small.seen == [("Bearer pool-secret", None)]
    assert large.seen == [("Bearer pool-secret", responses_client)]
    assert controller.seen == [("Bearer router-secret", None)]
    assert agent.usage["large"] == {
        "input_tokens": 100,
        "cached_input_tokens": 40,
        "uncached_input_tokens": 60,
        "output_tokens": 20,
        "pricing": {
            "currency": "USD",
            "input_per_million": 1.0,
            "cached_input_per_million": 0.1,
            "output_per_million": 6.0,
            "as_of": "2026-07-09",
        },
    }
    for model in (small, large, controller):
        assert "extra_headers" not in model.config.model_kwargs
        assert "client" not in model.config.model_kwargs


def test_large_worker_prices_cached_uncached_and_output_tokens() -> None:
    runner = _load_runner()
    agent = SimpleNamespace(usage={"large": {"n_calls": 0, "cost": 0.0}})
    message = {
        "usage": {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "input_tokens_details": {"cached_tokens": 250_000},
        },
        "extra": {"cost": 0.0},
    }

    runner._price_large_worker_response(agent, message)

    assert message["extra"]["cost"] == pytest.approx(6.775)
    assert agent.usage["large"]["uncached_input_tokens"] == 750_000
    assert agent.usage["large"]["cached_input_tokens"] == 250_000


def test_format_error_turn_still_bills_large_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    exceptions_mod = types.ModuleType("minisweagent.exceptions")

    class FormatError(Exception):
        def __init__(self, *messages: dict) -> None:
            # The runtime's InterruptAgentFlow keeps messages as a TUPLE.
            self.messages = messages
            super().__init__()

    exceptions_mod.FormatError = FormatError  # type: ignore[attr-defined]
    package_mod = types.ModuleType("minisweagent")
    package_mod.exceptions = exceptions_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "minisweagent", package_mod)
    monkeypatch.setitem(sys.modules, "minisweagent.exceptions", exceptions_mod)

    runner = _load_runner()
    billed_response = {
        "usage": {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "input_tokens_details": {"cached_tokens": 250_000},
        }
    }

    class Model:
        def __init__(self, *, raises: bool) -> None:
            self.config = SimpleNamespace(model_kwargs={})
            self.raises = raises

        def query(self, _messages):
            if self.raises:
                raise FormatError(billed_response)
            return {"usage": {"input_tokens": 0, "output_tokens": 0}}

    small = Model(raises=False)
    large = Model(raises=True)

    class Agent:
        small_model = small
        large_model = large
        controller_model = Model(raises=False)
        usage = {"large": {"n_calls": 0, "cost": 0.0}}

        def _query_controller_model(self, _messages):
            return {}

    agent = Agent()
    runner._install_capability_guards(
        agent,
        router_capability="router-secret",
        pool_capability="pool-secret",
    )

    with pytest.raises(FormatError):
        large.query([])

    # The tool-call-less turn was billed before the exception propagated:
    # tokens land in usage['large'] and the dollars in the cost/n_calls
    # fields postprocess harvests into controller_v3_cost.
    assert agent.usage["large"]["input_tokens"] == 1_000_000
    assert agent.usage["large"]["output_tokens"] == 1_000_000
    assert agent.usage["large"]["cached_input_tokens"] == 250_000
    assert agent.usage["large"]["n_calls"] == 1
    assert agent.usage["large"]["cost"] == pytest.approx(6.775)
    assert billed_response["extra"]["cost"] == pytest.approx(6.775)


def test_capability_guards_initialize_zero_gpt_cost_usage() -> None:
    runner = _load_runner()

    class Model:
        def __init__(self) -> None:
            self.config = SimpleNamespace(model_kwargs={})

        def query(self, _messages):
            return {}

    agent = SimpleNamespace(
        small_model=Model(),
        large_model=Model(),
        controller_model=Model(),
        usage={"large": {"n_calls": 0, "cost": 0.0}},
        _query_controller_model=lambda _messages: {},
    )

    runner._install_capability_guards(
        agent,
        router_capability="router-secret",
        pool_capability="pool-secret",
    )

    assert agent.usage["large"]["cost"] == 0.0
    assert agent.usage["large"]["input_tokens"] == 0
    assert agent.usage["large"]["cached_input_tokens"] == 0
    assert agent.usage["large"]["uncached_input_tokens"] == 0
    assert agent.usage["large"]["output_tokens"] == 0


def test_controller_harness_collects_gpt_cost(tmp_path: Path) -> None:
    harness = ControllerV3Harness(_spec())
    trajectory = tmp_path / "mini-swe-agent.traj.json"
    trajectory.write_text(
        json.dumps(
            {
                "info": {
                    "oracle_controller_v3": {
                        "usage": {
                            "large": {
                                "cost": 0.0123,
                                "input_tokens": 1000,
                                "cached_input_tokens": 400,
                                "uncached_input_tokens": 600,
                                "output_tokens": 20,
                                "pricing": {"currency": "USD"},
                            }
                        }
                    }
                }
            }
        )
    )
    runtime = SimpleNamespace(resolve_host_path=lambda _path: trajectory)
    result = AgentRunResult(status="completed", return_code=0)

    asyncio.run(harness.postprocess(runtime, result))

    assert result.metadata["controller_v3_cost"]["cost"] == 0.0123
    assert "controller_v3_cost_error" not in result.metadata
