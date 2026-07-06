from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import yaml

from polar.agent.factory import create_harness
from polar.agent.models import AgentSpec
from polar.config import TopologyConfig
from polar.gateway.engine import OpenAICompatibleEngine
from polar.gateway.proxy import InferenceClient
from polar.rollout.manager import _request_for_sample
from polar.rollout.models import TaskRequest
from slime_bridge.config import resolve_polar_slime_config
from slime_bridge.rollout import _build_task_payload


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "spilot_router_slime_grpo"


def _render_topology() -> str:
    values = {
        "POLAR_ROLLOUT_HOST": "0.0.0.0",
        "POLAR_ROLLOUT_PORT": "8080",
        "POLAR_ROLLOUT_URL": "http://head:8080",
        "POLAR_ROLLOUT_SAVE_DIR": "/tmp/results",
        "POLAR_GATEWAY_COMPLETION_QUEUE_SIZE": "1024",
        "POLAR_GATEWAY_COMPLETION_WRITE_WORKERS": "4",
        "POLAR_COMPLETION_BATCH_SIZE": "16",
        "POLAR_COMPLETION_WRITE_MAX_ATTEMPTS": "3",
        "POLAR_COMPLETION_RETRY_BACKOFF_SECONDS": "0.1",
        "POLAR_GATEWAY_HOST": "0.0.0.0",
        "POLAR_GATEWAY_PORT": "8081",
        "POLAR_GATEWAY_URL": "http://head:8081",
        "POLAR_GATEWAY_MAX_INIT_WORKERS": "32",
        "POLAR_GATEWAY_MAX_RUN_WORKERS": "64",
        "POLAR_GATEWAY_MAX_POSTRUN_WORKERS": "32",
        "POLAR_AGENT_MODEL_NAME": "Qwen/Qwen3.5-9B",
        "SGLANG_ROUTER_BASE_URL": "http://head:30000",
        "POLAR_MODEL_POOL_BASE_URL": "https://integrate.api.nvidia.com/v1",
    }
    text = (EXAMPLE / "topology.yaml").read_text()
    for name, value in values.items():
        text = text.replace("${" + name + "}", value)
    assert re.findall(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", text) == []
    return text


def test_spilot_topology_has_exact_fixed_pool_and_no_secret() -> None:
    document = yaml.safe_load(_render_topology())
    topology = TopologyConfig.model_validate(document)
    pool = topology.gateway.nodes[0].model_pool

    assert [(item.alias, item.model) for item in pool] == [
        ("pool/qwen3.6-27b", "nvidia/qwen/qwen3.6-27b"),
        ("pool/gpt-5.5", "openai/openai/gpt-5.5"),
    ]
    assert {item.api_key_env for item in pool} == {"POLAR_NVIDIA_API_KEY"}
    assert "POLAR_NVIDIA_API_KEY" not in {
        item.base_url for item in pool
    }


def test_spilot_template_variables_are_supported_by_shared_renderer() -> None:
    shared_run = (ROOT / "examples" / "swegym_slime_grpo" / "run.sh").read_text()
    placeholders: set[str] = set()
    for name in ("topology.yaml", "polar_config.yaml"):
        placeholders.update(
            re.findall(
                r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
                (EXAMPLE / name).read_text(),
            )
        )

    missing = sorted(name for name in placeholders if f'"{name}"' not in shared_run)
    assert missing == []


def test_nvidia_v1_base_url_does_not_duplicate_version_prefix() -> None:
    client = InferenceClient(
        "https://integrate.api.nvidia.com/v1",
        OpenAICompatibleEngine(),
    )

    assert client._v1_endpoint("chat/completions") == "chat/completions"


def test_spilot_agent_template_builds_registered_harness() -> None:
    text = (EXAMPLE / "polar_config.yaml").read_text()
    text = text.replace("${POLAR_AGENT_RUNTIME_VOLUME}", "")
    text = text.replace("${POLAR_INTERNET_RUNTIME_VOLUME}", "")
    document = yaml.safe_load(text)
    template = document["polar_task_template"]
    harness = create_harness(AgentSpec.model_validate(template["agent"]))

    assert harness.__class__.__name__ == "SpilotRouterHarness"
    config = harness._runner_config
    assert config["max_pool_calls"] == 2
    assert config["model_pool"]["M0"]["model"] == "pool/qwen3.6-27b"
    assert config["model_pool"]["M1"]["model"] == "pool/gpt-5.5"
    assert template["builder"]["strategy"] == "router_policy"
    assert template["evaluator"]["strategy"] == "spilot_harbor"
    assert template["evaluator"]["config"]["cost_penalty_lambda"] == 0.0


def test_spilot_fixed_eval_payload_normalizes_generic_overrides() -> None:
    text = (EXAMPLE / "polar_config.yaml").read_text()
    text = text.replace("${POLAR_AGENT_RUNTIME_VOLUME}", "")
    text = text.replace("${POLAR_INTERNET_RUNTIME_VOLUME}", "")
    document = yaml.safe_load(text)
    agent = deepcopy(document["polar_task_template"]["agent"])
    agent["model_name"] = "Qwen/Qwen3.5-9B"
    agent["settings"]["router_model_kwargs"]["temperature"] = 1.0
    agent["settings"]["router_model_kwargs"]["top_p"] = 1.0
    args = SimpleNamespace(
        polar_rollout_url="http://rollout:8080",
        polar_task_template={
            "timeout_seconds": "{sample.metadata.timeout_seconds}",
            "agent": agent,
        },
        polar_task_id_template="eval-{rollout_id}-{sample.group_index}",
        polar_max_async_level=1,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        update_weights_interval=1,
        polar_task_timeout_floor=4500,
        polar_train_agent_timeout=3300,
        polar_eval_agent_timeout=3300,
        eval_temperature=0.2,
        eval_top_p=0.9,
        eval_max_response_len=16_384,
        polar_min_complete_accept_fraction=0.0,
    )
    config = resolve_polar_slime_config(args)
    sample = SimpleNamespace(
        prompt="Fix the held-out bug",
        metadata={
            "timeout_seconds": 1320,
            "agent_timeout": 600,
            "agent_step_limit": 64,
        },
        group_index=7,
        index=0,
    )
    eval_cfg = SimpleNamespace(
        temperature=0.2,
        top_p=0.9,
        max_response_len=16_384,
    )

    payload = _build_task_payload(
        args=args,
        config=config,
        group=[sample],
        rollout_id=0,
        task_position=3,
        eval_dataset_cfg=eval_cfg,
        eval_dataset_name="tmax_holdout",
    )
    request = _request_for_sample(TaskRequest.model_validate(payload), 0)
    harness = create_harness(request.agent)
    runner_config = harness._runner_config

    assert request.timeout_seconds == 4500
    assert request.metadata["agent_timeout"] == 3300
    assert runner_config["router_max_tokens"] == 192
    assert runner_config["pool_step_limit"] == 64
    assert runner_config["router_model_kwargs"]["temperature"] == 0.2
    assert runner_config["router_model_kwargs"]["top_p"] == 0.9
    assert runner_config["router_model_kwargs"]["seed"] >= 0
    assert runner_config["slot_assignment_seed"] == runner_config["router_model_kwargs"]["seed"]
    assert "max_tokens" not in runner_config["router_model_kwargs"]


def test_spilot_submit_wrapper_pins_8_nodes_and_200_steps() -> None:
    script = (EXAMPLE / "submit_slurm.sh").read_text()
    defaults = (EXAMPLE / "experiment_defaults.sh").read_text()

    assert 'source "${SCRIPT_DIR}/experiment_defaults.sh"' in script
    assert 'NUM_NODES="${NUM_NODES:-8}"' in defaults
    assert 'PARTITION="${PARTITION:-backfill,batch}"' in defaults
    assert 'SLURM_GPUS="${SLURM_GPUS:-8}"' in defaults
    assert 'RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-8}"' in defaults
    assert 'ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"' in defaults
    assert 'ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"' in defaults
    assert 'ACTOR_TENSOR_MODEL_PARALLEL_SIZE="${ACTOR_TENSOR_MODEL_PARALLEL_SIZE:-4}"' in defaults
    assert 'TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL="${TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL:-0}"' in defaults
    assert 'ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-48}"' in defaults
    assert 'ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"' in defaults
    assert 'N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-32}"' in defaults
    assert 'GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"' in defaults
    assert 'EVAL_GLOBAL_BATCH_SIZE="${EVAL_GLOBAL_BATCH_SIZE:-256}"' in defaults
    assert 'TMAX_OVERRIDE_OPT_PARAM_SCHEDULER="${TMAX_OVERRIDE_OPT_PARAM_SCHEDULER:-1}"' in defaults
    assert 'POLAR_FULLY_ASYNC="${POLAR_FULLY_ASYNC:-true}"' in defaults
    assert 'POLAR_MULTI_GATEWAY="${POLAR_MULTI_GATEWAY:-1}"' in defaults
    assert 'TMAX_MIN_ASYNC_LEVEL="${TMAX_MIN_ASYNC_LEVEL:-3}"' in defaults
    assert 'POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-3}"' in defaults
    assert 'POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.5}"' in defaults
    assert 'POLAR_EARLY_STOP_GRACE_SESSIONS="${POLAR_EARLY_STOP_GRACE_SESSIONS:-16}"' in defaults
    assert 'MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-67584}"' in defaults
    assert (
        'TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP="${TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP:-0}"'
        in defaults
    )
    assert 'CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-0}"' in defaults
    assert 'LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-64}"' in defaults
    assert 'TMAX_OPTIMIZER_CPU_OFFLOAD="${TMAX_OPTIMIZER_CPU_OFFLOAD:-0}"' in defaults
    assert 'TMAX_NUM_ROLLOUT="${TMAX_NUM_ROLLOUT:-200}"' in defaults
    assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-5}"' in defaults
    assert 'SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL:-}"' in defaults
    assert 'WANDB_GROUP="${WANDB_GROUP:-spilot-router-qwen35-9b-8n64}"' in defaults
    assert 'TMAX_EVAL_MAX_TASKS="${TMAX_EVAL_MAX_TASKS:-100}"' in defaults
    assert 'TMAX_TRAINING_EVAL_ENABLED="${TMAX_TRAINING_EVAL_ENABLED:-1}"' in defaults
    assert 'TMAX_EXTERNAL_EVAL_ENABLED="${TMAX_EXTERNAL_EVAL_ENABLED:-0}"' in defaults
    assert 'TMAX_CONCURRENT_PRETRAIN_EVAL="${TMAX_CONCURRENT_PRETRAIN_EVAL:-0}"' in defaults
    assert 'TMAX_ONLY_READY="${TMAX_ONLY_READY:-1}"' in defaults
    assert 'TMAX_REQUIRE_EXACT_TOTAL_TASKS="${TMAX_REQUIRE_EXACT_TOTAL_TASKS:-1}"' in defaults
    assert "tmax-14598r-14498t100h-20260701T011143Z" in defaults
    assert "96a1c5929de64516eecc8a7b7ae012ccb888a2d575f15e28b8806ae6804826c8" in defaults
    assert "POLAR_NVIDIA_API_KEY" in script
    assert "POLAR_CONTROL_PLANE_TOKEN" in script
    assert "/dev/urandom" in script
    assert "runs/spilot_router_slime_grpo/current_run.env" in defaults
    assert "api_key=" not in (EXAMPLE / "topology.yaml").read_text().lower()


def test_spilot_watcher_relaunches_through_spilot_submitter() -> None:
    script = (EXAMPLE / "watch_training.sh").read_text()
    defaults = (EXAMPLE / "experiment_defaults.sh").read_text()

    assert 'source "${SCRIPT_DIR}/experiment_defaults.sh"' in script
    assert 'TMAX_SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_slurm.sh"' in script
    assert "runs/spilot_router_slime_grpo/current_run.env" in defaults
    assert "POLAR_NVIDIA_API_KEY" in script
    assert "NVIDIA_API_KEY is not set" in script


def test_spilot_shared_defaults_bootstrap_watcher_contract() -> None:
    completed = subprocess.run(
        [
            "bash",
            "-c",
            (
                "unset NUM_NODES SLURM_GPUS RAY_NUM_GPUS_PER_NODE "
                "ACTOR_NUM_NODES ACTOR_NUM_GPUS_PER_NODE "
                "ACTOR_TENSOR_MODEL_PARALLEL_SIZE "
                "TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL ROLLOUT_NUM_GPUS "
                "POLAR_SLURM_MEM_PER_NODE ROLLOUT_BATCH_SIZE "
                "N_SAMPLES_PER_PROMPT POLAR_FULLY_ASYNC TMAX_NUM_ROLLOUT "
                "MAX_TOKENS_PER_GPU TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP "
                "LOG_PROBS_CHUNK_SIZE TMAX_OPTIMIZER_CPU_OFFLOAD "
                "SAVE_INTERVAL SAVE_RETAIN_INTERVAL "
                "TMAX_AGENT_HARNESS EXPERIMENT_NAME; "
                'source "$1"; '
                "printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s' "
                '"$NUM_NODES" "$SLURM_GPUS" "$RAY_NUM_GPUS_PER_NODE" '
                '"$ACTOR_NUM_NODES" "$ACTOR_NUM_GPUS_PER_NODE" '
                '"$ACTOR_TENSOR_MODEL_PARALLEL_SIZE" '
                '"$TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL" "$ROLLOUT_NUM_GPUS" '
                '"$POLAR_SLURM_MEM_PER_NODE" "$ROLLOUT_BATCH_SIZE" '
                '"$N_SAMPLES_PER_PROMPT" "$POLAR_FULLY_ASYNC" '
                '"$TMAX_NUM_ROLLOUT" "$MAX_TOKENS_PER_GPU" '
                '"$TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP" "$LOG_PROBS_CHUNK_SIZE" '
                '"$TMAX_OPTIMIZER_CPU_OFFLOAD" '
                '"$SAVE_INTERVAL" "$SAVE_RETAIN_INTERVAL" '
                '"$TMAX_AGENT_HARNESS" "$EXPERIMENT_NAME"'
            ),
            "bash",
            str(EXAMPLE / "experiment_defaults.sh"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (
        "8|8|8|2|8|4|0|48||8|32|true|200|67584|0|64|0|5||spilot_router|"
        "spilot-router-qwen35-9b-8n64-200step"
    )


def test_spilot_uses_reference_gpu_adam_by_default() -> None:
    shared_run = (ROOT / "examples" / "swegym_slime_grpo" / "run.sh").read_text()
    run_state = (ROOT / "examples" / "tmax_slime_grpo" / "run_state.sh").read_text()
    defaults = (EXAMPLE / "experiment_defaults.sh").read_text()

    assert 'TMAX_OPTIMIZER_CPU_OFFLOAD="${TMAX_OPTIMIZER_CPU_OFFLOAD:-0}"' in defaults
    assert "--optimizer-cpu-offload" in shared_run
    assert "--optimizer-offload-fraction 1.0" in shared_run
    assert "--overlap-cpu-optimizer-d2h-h2d" in shared_run
    assert "--use-precision-aware-optimizer" in shared_run
    assert '"${OPTIMIZER_MEMORY_ARGS[@]}"' in shared_run
    assert "TMAX_OPTIMIZER_CPU_OFFLOAD" in run_state


def test_spilot_smoke_is_one_node_one_step_without_dynamic_filtering() -> None:
    script = (EXAMPLE / "submit_smoke.sh").read_text()

    assert 'NUM_NODES="${NUM_NODES:-1}"' in script
    assert 'ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"' in script
    assert 'ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"' in script
    assert 'MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-24576}"' in script
    assert (
        'TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP="${TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP:-1}"'
        in script
    )
    assert 'TMAX_OPTIMIZER_CPU_OFFLOAD="${TMAX_OPTIMIZER_CPU_OFFLOAD:-1}"' in script
    assert 'ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"' in script
    assert 'N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"' in script
    assert 'GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"' in script
    assert 'EVAL_GLOBAL_BATCH_SIZE="${EVAL_GLOBAL_BATCH_SIZE:-8}"' in script
    assert 'TMAX_NUM_ROLLOUT="${TMAX_NUM_ROLLOUT:-1}"' in script
    assert 'TMAX_DYNAMIC_SAMPLING_FILTER_PATH=""' in script
    assert 'TMAX_EVAL_ENABLED="${TMAX_EVAL_ENABLED:-0}"' in script
    run_state = (ROOT / "examples" / "tmax_slime_grpo" / "run_state.sh").read_text()
    assert "NVIDIA_API_KEY" not in run_state
    assert "POLAR_NVIDIA_API_KEY" not in run_state
    assert "POLAR_CONTROL_PLANE_TOKEN" not in run_state


def test_spilot_shell_wrappers_parse() -> None:
    for name in (
        "experiment_defaults.sh",
        "run.sh",
        "submit_slurm.sh",
        "submit_smoke.sh",
        "watch_training.sh",
    ):
        completed = subprocess.run(
            ["bash", "-n", str(EXAMPLE / name)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
