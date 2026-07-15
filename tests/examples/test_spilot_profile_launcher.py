from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "examples" / "spilot_router_slime_grpo" / "profile"
TMAX = ROOT / "examples" / "tmax_slime_grpo"
SHARED = ROOT / "examples" / "swegym_slime_grpo"


def run_profile(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(PROFILE / "submit_profile.sh"), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_profile_plan_is_read_only_and_exposes_controlled_arms(tmp_path: Path) -> None:
    data_root = tmp_path / "must-not-be-created"
    env = os.environ.copy()
    env.update(
        POLAR_DATA_ROOT=str(data_root),
        PROFILE_ID="unit",
        PROFILE_STEPS="4",
    )

    result = run_profile(
        "plan",
        "async-16t16r-l1",
        "async-8t24r-l3",
        "collocate-16shared",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "action=plan profile_id=unit steps=4 num_rollout=4" in result.stdout
    assert "async-16t16r-l1" in result.stdout
    assert "async-8t24r-l3" in result.stdout
    assert "collocate-16shared" in result.stdout
    assert "4x8" in result.stdout
    assert "4x4" in result.stdout
    assert not data_root.exists()


def test_profile_plan_exposes_requested_8_32_and_equal_budget_collocate(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env.update(POLAR_DATA_ROOT=str(tmp_path / "unused"), PROFILE_ID="alloc-sweep")

    result = run_profile(
        "plan",
        "async-16t16r-l3",
        "async-8t32r-l3",
        "collocate-32shared",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "steps=3 num_rollout=3" in result.stdout
    assert "seed=release model=Qwen/Qwen3.5-9B" in result.stdout
    assert "max_tokens_per_gpu=32768 early_stop=0.5+2" in result.stdout
    assert "async-8t32r-l3" in result.stdout
    assert "collocate-32shared" in result.stdout
    assert "5x8" in result.stdout
    assert "4x8" in result.stdout


def test_profile_numbered_seed_uses_exclusive_rollout_boundary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("47\n")
    train_data = tmp_path / "train.jsonl"
    train_data.write_text("{}\n")
    env = os.environ.copy()
    env.update(
        PROFILE_ID="numbered",
        PROFILE_STEPS="4",
        PROFILE_LOAD_DIR=str(checkpoint),
        PROFILE_TRAIN_DATA=str(train_data),
    )

    result = run_profile("plan", "async-16t16r-l3", env=env)

    assert result.returncode == 0, result.stderr
    assert "num_rollout=52" in result.stdout


def test_profile_numbered_seed_requires_matching_train_data(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("12\n")
    env = os.environ.copy()
    env.update(PROFILE_LOAD_DIR=str(checkpoint))
    env.pop("PROFILE_TRAIN_DATA", None)

    result = run_profile("plan", "async-16t16r-l3", env=env)

    assert result.returncode == 1
    assert "requires the exact PROFILE_TRAIN_DATA" in result.stderr


def test_profile_preflights_unknown_and_duplicate_arms() -> None:
    unknown = run_profile("plan", "async-16t16r-l1", "not-an-arm")
    duplicate = run_profile(
        "plan", "collocate-16", "collocate-16shared"
    )

    assert unknown.returncode == 1
    assert "unknown profile arm" in unknown.stderr
    assert duplicate.returncode == 1
    assert "duplicate profile arm" in duplicate.stderr


def topology_env(tmp_path: Path, *, mode: str, fully_async: str) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "USER": os.environ.get("USER", "test"),
        "POLAR_DATA_ROOT": str(tmp_path / "data"),
        "NUM_NODES": "4",
        "SLURM_GPUS": "4",
        "RAY_NUM_GPUS_PER_NODE": "4",
        "ACTOR_NUM_NODES": "4",
        "ACTOR_NUM_GPUS_PER_NODE": "4",
        "ACTOR_TENSOR_MODEL_PARALLEL_SIZE": "4",
        "ROLLOUT_NUM_GPUS": "16",
        "ROLLOUT_NUM_GPUS_PER_ENGINE": "1",
        "TMAX_TRAIN_MODE": mode,
        "POLAR_FULLY_ASYNC": fully_async,
        "TMAX_MIN_ASYNC_LEVEL": "1",
        "POLAR_MAX_ASYNC_LEVEL": "1",
        "TMAX_ENABLE_GRACEFUL_EXIT": "0",
        "TMAX_PROFILE_DISABLE_CHECKPOINT": "1",
        "TMAX_REQUIRE_WANDB": "0",
        "TMAX_EVAL_ENABLED": "0",
        "TMAX_TRAINING_EVAL_ENABLED": "0",
        "TMAX_CONCURRENT_PRETRAIN_EVAL": "0",
        "TMAX_REQUIRE_FULL_GPU_ALLOCATION": "1",
        "PARTITION": "batch",
        "WALL_TIME": "04:00:00",
        "TMAX_MIN_WALL_TIME": "04:00:00",
    }


def test_tmax_resource_validator_counts_shared_gpus_for_colocate(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{TMAX / "env.cwdfw.sh"}" >/dev/null; printf %s "$TMAX_TRAIN_MODE"',
        ],
        cwd=ROOT,
        env=topology_env(tmp_path, mode="colocate", fully_async="false"),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "colocate"


def test_same_topology_is_rejected_as_disjoint_async(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", "-c", f'source "{TMAX / "env.cwdfw.sh"}"'],
        cwd=ROOT,
        env=topology_env(tmp_path, mode="fully_async", fully_async="true"),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "require 32 GPUs, exceeding Ray capacity 16" in result.stderr


def test_disposable_profile_allows_short_batch_with_canonical_admission(
    tmp_path: Path,
) -> None:
    train_data = tmp_path / "train.jsonl"
    holdout_data = tmp_path / "holdout.jsonl"
    train_data.write_text("{}\n")
    holdout_data.write_text("{}\n")
    env = topology_env(tmp_path, mode="fully_async", fully_async="true")
    env.update(
        SLURM_GPUS="8",
        RAY_NUM_GPUS_PER_NODE="8",
        ACTOR_NUM_NODES="2",
        ACTOR_NUM_GPUS_PER_NODE="8",
        ROLLOUT_NUM_GPUS="16",
        TMAX_EVAL_ENABLED="1",
        TMAX_TRAIN_DATA=str(train_data),
        TMAX_EXCLUDE_DATA=str(holdout_data),
        TMAX_EVAL_DATA=str(holdout_data),
        TMAX_VALIDATE_EXISTING_ASSETS="0",
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -e; "
            f'source "{ROOT / "examples/spilot_router_slime_grpo/experiment_defaults.sh"}"; '
            f'source "{TMAX / "env.cwdfw.sh"}" >/dev/null; '
            'printf "%s|%s" "$PARTITION" "$WALL_TIME"',
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "batch|04:00:00"
    assert "disposable profile" in result.stderr


def test_8_train_32_rollout_uses_five_ray_nodes_and_four_gateways(
    tmp_path: Path,
) -> None:
    train_data = tmp_path / "train.jsonl"
    holdout_data = tmp_path / "holdout.jsonl"
    train_data.write_text("{}\n")
    holdout_data.write_text("{}\n")
    env = topology_env(tmp_path, mode="fully_async", fully_async="true")
    env.update(
        NUM_NODES="5",
        SLURM_GPUS="8",
        RAY_NUM_GPUS_PER_NODE="8",
        ACTOR_NUM_NODES="1",
        ACTOR_NUM_GPUS_PER_NODE="8",
        ROLLOUT_NUM_GPUS="32",
        TMAX_MIN_ASYNC_LEVEL="3",
        POLAR_MAX_ASYNC_LEVEL="3",
        POLAR_MULTI_GATEWAY="1",
        POLAR_GATEWAY_COUNT_OVERRIDE="4",
        SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT="4",
        TMAX_EVAL_ENABLED="1",
        TMAX_TRAIN_DATA=str(train_data),
        TMAX_EXCLUDE_DATA=str(holdout_data),
        TMAX_EVAL_DATA=str(holdout_data),
        TMAX_VALIDATE_EXISTING_ASSETS="0",
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -e; "
            f'source "{ROOT / "examples/spilot_router_slime_grpo/experiment_defaults.sh"}"; '
            f'source "{TMAX / "env.cwdfw.sh"}" >/dev/null; '
            'printf "%s|%s|%s" "$NUM_NODES" '
            '"$POLAR_GATEWAY_COUNT_OVERRIDE" "$SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT"',
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "5|4|4"


def test_shared_launcher_mode_and_no_checkpoint_hooks_are_opt_in() -> None:
    launcher = (SHARED / "run.sh").read_text()
    profile_submit = (PROFILE / "submit_profile.sh").read_text()
    run_state = (TMAX / "run_state.sh").read_text()

    assert 'TMAX_TRAIN_MODE="${TMAX_TRAIN_MODE:-fully_async}"' in launcher
    assert 'SLIME_TRAIN_ENTRYPOINT="${SLIME_DIR}/train.py"' in launcher
    assert "TRAIN_MODE_ARGS=(--colocate)" in launcher
    assert 'if [ "${TMAX_PROFILE_DISABLE_CHECKPOINT}" = "0" ]; then' in launcher
    assert 'SAVE_PATH_ARGS=(--save "${SAVE_DIR}")' in launcher
    assert "SAVE_PATH_ARGS=()" in launcher
    assert '"${SAVE_PATH_ARGS[@]}"' in launcher
    assert '"${SAVE_INTERVAL_ARGS[@]}"' in launcher
    assert "TMAX_TRAIN_MODE TMAX_PROFILE_DISABLE_CHECKPOINT" in run_state
    assert "TMAX_PROFILE_ARM TMAX_PROFILE_BATCH_ID" in run_state
    assert 'POLAR_GATEWAY_COUNT="${POLAR_GATEWAY_COUNT_OVERRIDE:-${RAY_NUM_NODES}}"' in launcher
    assert '"${_polar_gateway_hosts[@]:0:${POLAR_GATEWAY_COUNT}}"' in launcher
    assert "POLAR_GATEWAY_COUNT_OVERRIDE" in run_state
    assert 'WANDB_MODE="${PROFILE_WANDB_MODE:-offline}"' in profile_submit
    assert 'TMAX_REQUIRE_WANDB="${PROFILE_REQUIRE_WANDB:-0}"' in profile_submit
    assert r'\"PYTORCH_ALLOC_CONF\": \"${TMAX_PYTORCH_ALLOC_CONF}\"' in launcher


def test_allocator_defaults_are_mode_aware_and_overrideable() -> None:
    command = (
        f'source "{SHARED / "launcher_utils.sh"}"; '
        'printf "%s|%s|%s" '
        '"$(polar_select_pytorch_allocator_config colocate)" '
        '"$(polar_select_pytorch_allocator_config fully_async)" '
        '"$(polar_select_pytorch_allocator_config colocate max_split_size_mb:512)"'
    )
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    colocate, fully_async, override = result.stdout.split("|")
    assert colocate == "max_split_size_mb:2048"
    assert "expandable_segments" not in colocate
    assert fully_async == "max_split_size_mb:2048,expandable_segments:True"
    assert override == "max_split_size_mb:512"


def test_profile_submit_pins_release_model_memory_and_completion_contract(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    hf_checkpoint = data_root / "checkpoints" / "Qwen3.5-9B"
    ref_load = data_root / "checkpoints" / "Qwen3.5-9B_torch_dist"
    hf_checkpoint.mkdir(parents=True)
    ref_load.mkdir(parents=True)
    (hf_checkpoint / "config.json").write_text("{}\n")
    (ref_load / "latest_checkpointed_iteration.txt").write_text("release\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured_env = tmp_path / "submitted.env"
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        'env | sort >"${CAPTURE_ENV}"\n'
        'mkdir -p "$(dirname "${TMAX_SUBMIT_RECEIPT_FILE}")"\n'
        "printf '%s\\n' 'export POLAR_SUBMITTED_JOB_ID=4242' "
        '>"${TMAX_SUBMIT_RECEIPT_FILE}"\n'
    )
    fake_bash.chmod(0o755)

    env = os.environ.copy()
    for name in (
        "PROFILE_LOAD_DIR",
        "PROFILE_HF_CHECKPOINT",
        "PROFILE_REF_LOAD",
        "PROFILE_TORCH_DIST_DIR",
        "PROFILE_MODEL_ARGS_FILE",
        "PROFILE_AGENT_MODEL_NAME",
    ):
        env.pop(name, None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CAPTURE_ENV": str(captured_env),
            "POLAR_DATA_ROOT": str(data_root),
            "PROFILE_ID": "unit-release-contract",
            "HF_CHECKPOINT": "/stale/Qwen3.5-4B",
            "REF_LOAD": "/stale/Qwen3.5-4B_torch_dist",
            "TORCH_DIST_DIR": "/stale/Qwen3.5-4B_torch_dist",
            "MODEL_ARGS_FILE": "/stale/qwen4-model-args.sh",
            "POLAR_AGENT_MODEL_NAME": "Qwen/Qwen3.5-4B",
            "LOAD_DIR": "/stale/numeric-step-49",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )

    result = subprocess.run(
        [
            "/bin/bash",
            str(PROFILE / "submit_profile.sh"),
            "submit",
            "collocate-16shared",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    submitted = dict(
        line.split("=", 1)
        for line in captured_env.read_text().splitlines()
        if "=" in line
    )
    assert submitted["HF_CHECKPOINT"] == str(hf_checkpoint)
    assert submitted["REF_LOAD"] == str(ref_load)
    assert submitted["TORCH_DIST_DIR"] == str(ref_load)
    assert submitted["MODEL_ARGS_FILE"] == str(TMAX / "model_args.sh")
    assert submitted["POLAR_AGENT_MODEL_NAME"] == "Qwen/Qwen3.5-9B"
    assert submitted["LOAD_DIR"] == str(ref_load)
    assert submitted["TMAX_NUM_ROLLOUT"] == "3"
    assert submitted["MAX_TOKENS_PER_GPU"] == "32768"
    assert submitted["TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP"] == "1"
    assert submitted["POLAR_MIN_COMPLETE_ACCEPT_FRACTION"] == "0.5"
    assert submitted["POLAR_EARLY_STOP_GRACE_SESSIONS"] == "2"
    assert submitted["TMAX_PYTORCH_ALLOC_CONF"] == "max_split_size_mb:2048"
    assert "PYTORCH_ALLOC_CONF" not in submitted
    assert "PYTORCH_CUDA_ALLOC_CONF" not in submitted
