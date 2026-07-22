from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "examples/controller_v3_tmax_slime_grpo"


def test_controller_v3_polar_contract() -> None:
    text = (EXAMPLE / "polar_config.yaml").read_text()
    assert 'harness: "controller_v3"' in text
    assert 'strategy: "router_policy"' in text
    assert 'strategy: "harbor"' in text
    assert "cost_penalty" not in text
    assert "latency_penalty" not in text


def test_controller_v3_topology_routes_local_qwen_and_nvidia_luna() -> None:
    text = (EXAMPLE / "topology.yaml").read_text()
    assert "pool/qwen3.6-35b-a3b" in text
    assert "nvidia/qwen/qwen3.6-35b-a3b" in text
    assert "${CONTROLLER_V3_SMALL_ROUTER_BASE_URL}" in text
    assert "pool/gpt-5.6-luna" in text
    assert "openai/openai/gpt-5.6-luna" in text
    assert "${POLAR_MODEL_POOL_BASE_URL}" in text
    assert "${CONTROLLER_V3_QWEN_GATEWAY_MAX_CONCURRENCY}" in text
    assert "${CONTROLLER_V3_GPT_GATEWAY_MAX_CONCURRENCY}" in text


def test_split_wrapper_reserves_exact_gpu_layout() -> None:
    text = (EXAMPLE / "run.sh").read_text()
    assert 'export RAY_LAST_NODE_NUM_GPUS="${RAY_LAST_NODE_NUM_GPUS:-2}"' in text
    assert 'export RAY_NUM_GPUS_PER_NODE=8' in text
    assert 'CUDA_VISIBLE_DEVICES="${first_gpu},${second_gpu}"' in text
    assert "for replica in 0 1 2" in text
    assert "--tp-size 2" in text
    assert "--ep-size 2" in text
    assert "sglang_router.launch_router" in text


def test_slurm_submit_preserves_last_node_ray_gpu_limit() -> None:
    text = (ROOT / "examples/swegym_slime_grpo/submit_slurm.sh").read_text()
    assert "RAY_NUM_*|RAY_LAST_NODE_NUM_GPUS|" in text


def test_slurm_dry_run_is_side_effect_free_and_reports_full_topology() -> None:
    result = subprocess.run(
        ["bash", str(EXAMPLE / "submit_slurm.sh"), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        check=True,
        text=True,
        capture_output=True,
    )
    output = result.stdout
    assert "no command executed" in output
    assert "nodes=3 gpus_per_node=8 total_gpus=24" in output
    assert "actor=2x8 controller_rollout=1x2 frozen_qwen=3x2" in output
    assert "gateways=3" in output
    assert "api=responses reasoning=max" in output
    assert "request_caps_per_gateway=qwen:1,gpt:4" in output
    assert "TMax ready rows=1007 images=1000" in output
    assert "/data/training_data/tmax/tmax-15k" in output
    assert "/tmax-15k-open-instruct/enroot-images" in output
    assert "sbatch --nodes=3" in output


def test_two_node_smoke_dry_run_uses_one_prompt_and_two_trajectories() -> None:
    result = subprocess.run(
        ["bash", str(EXAMPLE / "smoke_2n.sh"), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        check=True,
        text=True,
        capture_output=True,
    )
    output = result.stdout
    assert "no command executed" in output
    assert "TMax ready rows=1 images=1" in output
    assert "nodes=2 gpus_per_node=8 total_gpus=16 partition=interactive" in output
    assert "actor=1x8 controller_rollout=1x2 frozen_qwen=3x2" in output
    assert "gateways=2" in output
    assert "request_caps_per_gateway=qwen:1,gpt:4" in output
    assert "sbatch --nodes=2" in output
