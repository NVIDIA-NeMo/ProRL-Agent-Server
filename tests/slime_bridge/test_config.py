from __future__ import annotations

from types import SimpleNamespace

import pytest

from polar.gateway.node import GatewayNodeManager
from polar.rollout.models import SessionDispatchRequest, TaskRequest
from slime_bridge.config import (
    render_instruction,
    render_task_payload,
    render_topology_template,
    resolve_polar_slime_config,
    resolve_sglang_router_base_url,
)


def _args(**overrides):
    base = {
        "polar_rollout_url": "http://rollout:8080/",
        "polar_task_template": {
            "agent": {"harness": "codex", "model_name": "{args.model_name}"},
            "runtime": {"image": "{sample.metadata.image}"},
            "metadata": {"instance": "{sample.metadata.instance_id}"},
        },
        "polar_task_id_template": "task-{rollout_id}-{sample.group_index}",
        "polar_instruction_template": "Instruction: {instruction}",
        "polar_reward_key": "score",
        "polar_max_async_level": 2,
        "polar_fully_async": True,
        "rollout_batch_size": 3,
        "n_samples_per_prompt": 4,
        "update_weights_interval": 5,
        "polar_request_timeout": 60,
        "polar_task_timeout_floor": None,
        "polar_train_agent_timeout": None,
        "polar_eval_agent_timeout": None,
        "polar_callback_host": "127.0.0.1",
        "polar_scoring_mode": "group",
        "polar_min_complete_accept_fraction": 0.0,
        "polar_candidate_pool_health_gate_enabled": False,
        "polar_candidate_pool_health_min_observed_sessions": 16,
        "polar_candidate_pool_health_min_completion_fraction": 0.1,
        "hf_checkpoint": "tokenizer-name",
        "polar_add_generation_prompt": True,
        "polar_eval_dataset_name": "eval",
        "model_name": "openai/gpt-test",
        "sglang_router_ip": "127.0.0.1",
        "sglang_router_port": 30000,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_resolve_polar_slime_config_computes_concurrency_and_normalizes_url() -> None:
    config = resolve_polar_slime_config(_args())

    assert config.rollout_server_url == "http://rollout:8080"
    assert config.max_concurrency == 6
    assert config.max_session_concurrency == 24
    assert config.fully_async is True
    assert config.max_off_policy_steps == 7
    assert config.max_consecutive_infrastructure_failures == 0
    assert config.request_timeout == 60.0
    assert config.task_timeout_floor is None
    assert config.train_agent_timeout is None
    assert config.eval_agent_timeout is None
    assert config.min_complete_accept_fraction == 0.0
    assert config.early_stop_grace_sessions == 2
    assert config.candidate_pool_health_gate_enabled is False
    assert config.candidate_pool_health_min_observed_sessions == 16
    assert config.candidate_pool_health_min_completion_fraction == 0.1


def test_resolve_polar_slime_config_requires_agent_template() -> None:
    with pytest.raises(ValueError, match="agent spec"):
        resolve_polar_slime_config(_args(polar_task_template={}))


def test_resolve_polar_slime_config_rejects_invalid_fully_async_value() -> None:
    with pytest.raises(ValueError, match="polar_fully_async"):
        resolve_polar_slime_config(_args(polar_fully_async="sometimes"))


def test_resolve_polar_slime_config_accepts_infrastructure_failure_fuse() -> None:
    config = resolve_polar_slime_config(
        _args(polar_max_consecutive_infrastructure_failures=3)
    )

    assert config.max_consecutive_infrastructure_failures == 3


@pytest.mark.parametrize("value", [-1, 1.5, True, "three"])
def test_resolve_polar_slime_config_rejects_invalid_infrastructure_failure_fuse(
    value,
) -> None:
    with pytest.raises(
        ValueError,
        match="polar_max_consecutive_infrastructure_failures",
    ):
        resolve_polar_slime_config(
            _args(polar_max_consecutive_infrastructure_failures=value)
        )


def test_resolve_polar_slime_config_accepts_complete_fraction_threshold() -> None:
    config = resolve_polar_slime_config(_args(polar_min_complete_accept_fraction=0.8))

    assert config.min_complete_accept_fraction == 0.8


def test_resolve_polar_slime_config_rejects_negative_early_stop_grace() -> None:
    with pytest.raises(ValueError, match="polar_early_stop_grace_sessions"):
        resolve_polar_slime_config(_args(polar_early_stop_grace_sessions=-1))


@pytest.mark.parametrize("value", ["sometimes", 1, None])
def test_resolve_polar_slime_config_rejects_invalid_candidate_health_enabled(value) -> None:
    with pytest.raises(ValueError, match="polar_candidate_pool_health_gate_enabled"):
        resolve_polar_slime_config(
            _args(polar_candidate_pool_health_gate_enabled=value)
        )


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "sixteen"])
def test_resolve_polar_slime_config_rejects_invalid_candidate_health_min_sessions(
    value,
) -> None:
    with pytest.raises(
        ValueError,
        match="polar_candidate_pool_health_min_observed_sessions",
    ):
        resolve_polar_slime_config(
            _args(polar_candidate_pool_health_min_observed_sessions=value)
        )


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf"), float("nan"), True])
def test_resolve_polar_slime_config_rejects_invalid_candidate_health_fraction(value) -> None:
    with pytest.raises(
        ValueError,
        match="polar_candidate_pool_health_min_completion_fraction",
    ):
        resolve_polar_slime_config(
            _args(polar_candidate_pool_health_min_completion_fraction=value)
        )


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_resolve_polar_slime_config_rejects_invalid_task_timeout_floor(value) -> None:
    with pytest.raises(ValueError, match="polar_task_timeout_floor"):
        resolve_polar_slime_config(_args(polar_task_timeout_floor=value))


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "slow"])
def test_resolve_polar_slime_config_rejects_invalid_train_agent_timeout(value) -> None:
    with pytest.raises(ValueError, match="polar_train_agent_timeout"):
        resolve_polar_slime_config(_args(polar_train_agent_timeout=value))


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "slow"])
def test_resolve_polar_slime_config_rejects_invalid_eval_agent_timeout(value) -> None:
    with pytest.raises(ValueError, match="polar_eval_agent_timeout"):
        resolve_polar_slime_config(_args(polar_eval_agent_timeout=value))


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_resolve_polar_slime_config_rejects_invalid_complete_fraction(value) -> None:
    with pytest.raises(ValueError, match="polar_min_complete_accept_fraction"):
        resolve_polar_slime_config(_args(polar_min_complete_accept_fraction=value))


def test_render_task_payload_resolves_args_and_sample_placeholders() -> None:
    args = _args()
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt="prompt",
        metadata={"image": "runtime:latest", "instance_id": "abc123"},
        group_index=9,
    )

    payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction="Fix the bug",
        rollout_id=2,
        task_position=0,
        num_rollouts=4,
    )

    assert payload["task_id"] == "task-2-9"
    assert payload["instruction"] == "Fix the bug"
    assert payload["num_samples"] == 4
    assert payload["agent"]["model_name"] == "openai/gpt-test"
    assert payload["runtime"]["image"] == "runtime:latest"
    assert payload["metadata"]["instance"] == "abc123"
    assert "early_stop_min_usable_sessions" not in payload


def test_render_task_payload_allows_trusted_dataset_step_limit_override() -> None:
    args = _args()
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt="prompt",
        metadata={
            "image": "runtime:latest",
            "instance_id": "terminal-bench/task-a",
            "agent_step_limit": 50,
        },
        group_index=9,
    )

    payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction="Complete the terminal task",
        rollout_id=2,
        task_position=0,
        num_rollouts=1,
    )

    assert payload["agent"]["settings"]["step_limit"] == 50


@pytest.mark.parametrize(
    ("dataset_timeout", "expected"),
    [(840.0, 1800.0), (2400.0, 2400.0)],
)
def test_render_task_payload_applies_timeout_floor_without_lowering_dataset_budget(
    dataset_timeout: float,
    expected: float,
) -> None:
    args = _args(
        polar_task_timeout_floor=1800,
        polar_task_template={
            "timeout_seconds": "{sample.metadata.timeout_seconds}",
            "agent": {"harness": "codex", "model_name": "model"},
        },
    )
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt="prompt",
        metadata={"timeout_seconds": dataset_timeout, "agent_timeout": 600.0},
        group_index=0,
    )

    payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction="task",
        rollout_id=1,
        task_position=0,
        num_rollouts=1,
    )

    assert payload["timeout_seconds"] == expected
    assert payload["metadata"]["agent_timeout"] == 600.0

    # The trusted budget survives both API model boundaries unchanged.
    task = TaskRequest.model_validate(payload)
    dispatch = SessionDispatchRequest(
        session_id="session-agent-timeout",
        task_id=task.task_id,
        instruction=task.instruction,
        remaining_timeout_seconds=task.timeout_seconds,
        runtime=task.runtime,
        agent=task.agent,
        builder=task.builder,
        evaluator=task.evaluator,
        callback_url=task.callback_url,
        metadata=dict(task.metadata),
    )
    assert dispatch.metadata["agent_timeout"] == 600.0


def test_render_task_payload_applies_agent_timeout_override_only_to_training() -> None:
    args = _args(
        polar_train_agent_timeout=1200,
        polar_task_template={
            "timeout_seconds": "{sample.metadata.timeout_seconds}",
            "agent": {"harness": "codex", "model_name": "model"},
        },
    )
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt="prompt",
        metadata={"timeout_seconds": 1800.0, "agent_timeout": 600.0},
        group_index=0,
    )

    training_payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction="task",
        rollout_id=1,
        task_position=0,
        num_rollouts=1,
    )
    eval_payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction="task",
        rollout_id=1,
        task_position=0,
        num_rollouts=1,
        is_eval=True,
    )

    assert training_payload["metadata"]["agent_timeout"] == 1200.0
    assert eval_payload["metadata"]["agent_timeout"] == 600.0


def test_render_task_payload_can_override_eval_agent_timeout_independently() -> None:
    args = _args(
        polar_train_agent_timeout=1200,
        polar_eval_agent_timeout=3300,
        polar_task_template={
            "timeout_seconds": "{sample.metadata.timeout_seconds}",
            "agent": {"harness": "spilot_router", "model_name": "model"},
        },
    )
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt="prompt",
        metadata={"timeout_seconds": 4500.0, "agent_timeout": 600.0},
        group_index=0,
    )

    training_payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction="task",
        rollout_id=1,
        task_position=0,
        num_rollouts=1,
    )
    eval_payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction="task",
        rollout_id=1,
        task_position=0,
        num_rollouts=1,
        is_eval=True,
    )

    assert training_payload["metadata"]["agent_timeout"] == 1200.0
    assert eval_payload["metadata"]["agent_timeout"] == 3300.0


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "slow"])
def test_render_task_payload_rejects_invalid_agent_timeout_metadata(value) -> None:
    args = _args()
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt="prompt",
        metadata={
            "image": "runtime:latest",
            "instance_id": "task",
            "agent_timeout": value,
        },
        group_index=0,
    )

    with pytest.raises(ValueError, match="sample.metadata.agent_timeout"):
        render_task_payload(
            args=args,
            config=config,
            sample=sample,
            instruction="task",
            rollout_id=1,
            task_position=0,
            num_rollouts=1,
        )


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "60"])
def test_dispatch_request_rejects_invalid_agent_timeout_metadata(value) -> None:
    with pytest.raises(ValueError, match="metadata.agent_timeout"):
        SessionDispatchRequest(
            session_id="session-bad-agent-timeout",
            task_id="task-bad-agent-timeout",
            instruction="task",
            remaining_timeout_seconds=60,
            agent={"harness": "codex"},
            metadata={"agent_timeout": value},
        )


def test_render_task_payload_merges_harbor_oci_runtime_metadata() -> None:
    args = _args(
        polar_task_template={
            "agent": {"harness": "codex", "model_name": "model"},
            "runtime": {
                "image": "{sample.metadata.image}",
                "env": {"HOME": "/polar/session/home", "PATH": "/agent:/usr/bin"},
                "direct_exec_init_command": "test ! -f /image-hook || . /image-hook",
            },
        }
    )
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt="prompt",
        metadata={
            "image": "/images/task.sqsh",
            "runtime_env": {
                "USER": "root",
                "PASSWORD": "password1",
                "PYTHONPATH": "/app:",
            },
            "runtime_init_command": (
                "mkdir -p /polar/session/logs && "
                "(supervisord -c /etc/supervisor/supervisord.conf "
                ">>/polar/session/logs/container-init.log 2>&1 &)"
            ),
        },
        group_index=0,
    )

    payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction="Complete the terminal task",
        rollout_id=1,
        task_position=0,
        num_rollouts=1,
    )

    assert payload["runtime"]["env"] == {
        "HOME": "/polar/session/home",
        "PATH": "/agent:/usr/bin",
        "USER": "root",
        "PASSWORD": "password1",
        "PYTHONPATH": "/app:",
    }
    init = payload["runtime"]["direct_exec_init_command"]
    assert init.startswith("{ test ! -f /image-hook || . /image-hook; } && {")
    assert "supervisord -c /etc/supervisor/supervisord.conf" in init


@pytest.mark.parametrize(
    "metadata",
    [
        {"runtime_env": {"KEY": 1}},
        {"runtime_env": {"": "value"}},
        {"runtime_init_command": "  "},
    ],
)
def test_render_task_payload_rejects_invalid_runtime_metadata(metadata) -> None:
    args = _args()
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt="prompt",
        metadata={"image": "runtime:latest", "instance_id": "task", **metadata},
        group_index=0,
    )

    with pytest.raises(ValueError, match="sample.metadata.runtime"):
        render_task_payload(
            args=args,
            config=config,
            sample=sample,
            instruction="task",
            rollout_id=1,
            task_position=0,
            num_rollouts=1,
        )


@pytest.mark.parametrize(("allow_internet", "expected"), [(True, "true"), (False, "false")])
def test_rendered_internet_policy_survives_dispatch_and_cannot_be_overridden(
    allow_internet: bool,
    expected: str,
) -> None:
    """Exercise the same model boundary used from Slime through the gateway."""
    args = _args(
        polar_task_template={
            "timeout_seconds": "{sample.metadata.timeout_seconds}",
            "runtime": {
                "backend": "apptainer",
                "image": "/images/{sample.metadata.task_name}.sif",
                "network": "none",
                "allow_internet": "{sample.metadata.allow_internet}",
                "internet_volumes": ["/host/proxy:/polar/proxy:ro"],
                "env": {
                    "HOME": "/polar/session/home",
                    "POLAR_ALLOW_INTERNET": "runtime-template-override",
                },
            },
            "agent": {
                "harness": "codex",
                "model_name": "{args.model_name}",
                "env": {"POLAR_ALLOW_INTERNET": "agent-template-override"},
            },
        }
    )
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt="prompt",
        metadata={
            "allow_internet": allow_internet,
            "task_name": "task_000001",
            "timeout_seconds": 600.0,
        },
        group_index=3,
    )

    payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction="Fix the bug",
        rollout_id=7,
        task_position=0,
        num_rollouts=2,
    )
    assert payload["runtime"]["allow_internet"] is allow_internet

    # FastAPI validates the submitted JSON as TaskRequest, then the rollout
    # pipeline constructs this request before serializing it to the gateway.
    task = TaskRequest.model_validate(payload)
    dispatch = SessionDispatchRequest(
        session_id="session-policy",
        task_id=task.task_id,
        instruction=task.instruction,
        remaining_timeout_seconds=task.timeout_seconds,
        runtime=task.runtime,
        agent=task.agent,
        builder=task.builder,
        evaluator=task.evaluator,
        callback_url=task.callback_url,
        metadata=dict(task.metadata),
    )
    dispatch = SessionDispatchRequest.model_validate(dispatch.model_dump(mode="json"))

    assert dispatch.runtime is not None
    assert dispatch.runtime.allow_internet is allow_internet
    assert dispatch.runtime.internet_volumes == ["/host/proxy:/polar/proxy:ro"]
    assert all(isinstance(value, str) for value in dispatch.runtime.env.values())

    runtime = SimpleNamespace(
        spec=dispatch.runtime,
        runtime_session_dir="/polar/session",
        runtime_artifacts_dir="/polar/session/artifacts",
        runtime_logs_dir="/polar/session/logs",
        runtime_agent_log_dir="/polar/session/logs/agent",
    )
    managed = SimpleNamespace(runtime=runtime)
    manager = object.__new__(GatewayNodeManager)
    manager.gateway_url = "http://gateway.test"

    environment = manager._runtime_env(
        dispatch,
        managed,  # type: ignore[arg-type]
        include_agent_env=True,
    )

    assert environment["POLAR_ALLOW_INTERNET"] == expected


def test_render_task_payload_adds_buffered_early_stop_threshold() -> None:
    args = _args(
        polar_min_complete_accept_fraction=0.5,
        polar_early_stop_grace_sessions=2,
    )
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt="prompt",
        metadata={"image": "runtime:latest", "instance_id": "abc123"},
        group_index=9,
    )

    payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction="Fix the bug",
        rollout_id=2,
        task_position=0,
        num_rollouts=8,
    )

    assert payload["early_stop_min_usable_sessions"] == 6


def test_render_instruction_uses_optional_template() -> None:
    args = _args()
    config = resolve_polar_slime_config(args)

    rendered = render_instruction(
        args=args,
        config=config,
        sample=SimpleNamespace(metadata={}),
        prompt_text="Fix the bug",
        rollout_id=1,
        task_position=0,
        num_rollouts=1,
    )

    assert rendered == "Instruction: Fix the bug"


def test_resolve_sglang_router_base_url_requires_both_ip_and_port() -> None:
    assert resolve_sglang_router_base_url(_args()) == "http://127.0.0.1:30000"
    assert resolve_sglang_router_base_url(_args(sglang_router_port=None)) is None


def test_render_topology_template_emits_inference_block(tmp_path) -> None:
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        """
rollout: {host: 127.0.0.1, port: 8080, public_url: http://127.0.0.1:8080}
gateway:
  nodes:
    - id: n1
      host: 127.0.0.1
      port: 8100
      public_url: http://127.0.0.1:8100
      model_served: Qwen/Qwen3.5-4B
      inference: {engine: sglang, base_url: http://127.0.0.1:8000}
      model_pool:
        - alias: pool/qwen3.6-27b
          model: nvidia/qwen/qwen3.6-27b
          base_url: https://integrate.api.nvidia.com/v1
          api_key_env: POLAR_NVIDIA_API_KEY
        - alias: pool/gpt-5.5
          model: openai/openai/gpt-5.5
          base_url: https://integrate.api.nvidia.com/v1
          api_key_env: POLAR_NVIDIA_API_KEY
""".strip()
    )
    rendered = render_topology_template(str(topology_path), _args())
    node = rendered["gateway"]["nodes"][0]
    assert node["inference"] == {"engine": "sglang", "base_url": "http://127.0.0.1:30000"}
    assert [(item["alias"], item["model"]) for item in node["model_pool"]] == [
        ("pool/qwen3.6-27b", "nvidia/qwen/qwen3.6-27b"),
        ("pool/gpt-5.5", "openai/openai/gpt-5.5"),
    ]
    assert {item["api_key_env"] for item in node["model_pool"]} == {
        "POLAR_NVIDIA_API_KEY"
    }
    assert "sglang" not in node
