from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from polar.agent.factory import create_harness
from polar.agent.models import AgentSpec
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
            return {}

    small = Model()
    large = Model()
    controller = Model()

    class Agent:
        small_model = small
        large_model = large
        controller_model = controller

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
    for model in (small, large, controller):
        assert "extra_headers" not in model.config.model_kwargs
        assert "client" not in model.config.model_kwargs
