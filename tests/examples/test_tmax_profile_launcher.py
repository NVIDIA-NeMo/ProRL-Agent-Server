from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "examples" / "tmax_slime_grpo" / "profile"


def test_tmax_profile_default_plan_has_comparable_gpu_arms(tmp_path: Path) -> None:
    checkpoint = tmp_path / "release"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("release\n")
    train_data = tmp_path / "train.jsonl"
    train_data.write_text('{"prompt": "test"}\n')
    env = os.environ.copy()
    env.update(
        {
            "POLAR_DATA_ROOT": str(tmp_path / "data"),
            "PROFILE_ID": "unit-plan",
            "PROFILE_STEPS": "3",
            "PROFILE_LOAD_DIR": str(checkpoint),
            "PROFILE_TRAIN_DATA": str(train_data),
        }
    )

    result = subprocess.run(
        ["bash", str(PROFILE / "submit_profile.sh"), "plan"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "steps=3 num_rollout=3" in result.stdout
    assert "harness=mini_swe_agent" in result.stdout
    assert "async-16t16r-l4" in result.stdout and "16      16" in result.stdout
    assert "async-8t24r-l4" in result.stdout and "8      24" in result.stdout
    assert "async-8t32r-l4" in result.stdout and "5x8" in result.stdout
    assert "collocate-32shared" in result.stdout
    assert not (tmp_path / "data").exists()


def test_tmax_profile_is_checkpoint_free_and_isolated() -> None:
    launcher = (PROFILE / "submit_profile.sh").read_text()
    assert "TMAX_PROFILE_DISABLE_CHECKPOINT=1" in launcher
    assert "TMAX_ENABLE_GRACEFUL_EXIT=0" in launcher
    assert "TMAX_AGENT_HARNESS=mini_swe_agent" in launcher
    assert "TMAX_EVAL_ENABLED=0" in launcher
    assert 'run_id="tmax-prof-${canonical_arm}-r${repeat}-${PROFILE_ID}"' in launcher
    assert "PROFILE_AFTER_JOB_ID" in launcher
    assert 'PROFILE_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP:-0' in launcher


def test_tmax_profile_enforces_the_safe_per_sample_token_cap() -> None:
    launcher = (PROFILE / "submit_profile.sh").read_text()
    polar_config = (ROOT / "examples" / "tmax_slime_grpo" / "polar_config.yaml").read_text()

    assert 'PROFILE_MAX_TOKENS_PER_GPU="${PROFILE_MAX_TOKENS_PER_GPU:-24576}"' in launcher
    assert 'TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP="${PROFILE_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP}"' in launcher
    assert (
        "polar_allow_single_sample_over_token_cap: ${TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP}"
    ) in polar_config


def test_profile_monitor_is_a_serial_cpu_relay() -> None:
    launcher = (PROFILE / "submit_monitor.sh").read_text()
    assert "MONITOR_PARTITION:-cpu_short" in launcher
    assert (
        'MONITOR_ACCOUNT="${MONITOR_ACCOUNT:-${ACCOUNT:-${SBATCH_ACCOUNT:-nvr_lpr_llm}}}"'
        in launcher
    )
    assert '--account="${MONITOR_ACCOUNT}"' in launcher
    assert 'dependency_args=(--dependency="afterany:${previous_job_id}")' in launcher
    assert "monitor_profile.py" in launcher
    assert "--status-json" in launcher
    assert "--job" in launcher
    assert "--handoff-on-timeout" in launcher


def test_profile_report_is_chained_after_all_gpu_arms() -> None:
    launcher = (PROFILE / "submit_report.sh").read_text()
    assert (
        'REPORT_ACCOUNT="${REPORT_ACCOUNT:-${ACCOUNT:-${SBATCH_ACCOUNT:-nvr_lpr_llm}}}"'
        in launcher
    )
    assert '--account="${REPORT_ACCOUNT}"' in launcher
    assert '--dependency="afterany:${after_job_id}"' in launcher
    assert "run_profile_report.py" in launcher
    assert "--spilot-manifest" in launcher
    assert "--tmax-manifest" in launcher
    assert "gpu-allocation-report.html" in launcher


def test_tmax_profile_submit_requires_explicit_seed_and_data(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("PROFILE_LOAD_DIR", None)
    env.pop("PROFILE_TRAIN_DATA", None)
    env.update(
        {
            "POLAR_DATA_ROOT": str(tmp_path / "data"),
            "PROFILE_ID": "unit-required-inputs",
        }
    )

    missing_seed = subprocess.run(
        ["bash", str(PROFILE / "submit_profile.sh"), "submit", "async-8t24r-l4"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert missing_seed.returncode != 0
    assert "submit requires PROFILE_LOAD_DIR" in missing_seed.stderr

    checkpoint = tmp_path / "release"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("release\n")
    env["PROFILE_LOAD_DIR"] = str(checkpoint)
    missing_data = subprocess.run(
        ["bash", str(PROFILE / "submit_profile.sh"), "submit", "async-8t24r-l4"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert missing_data.returncode != 0
    assert "submit requires PROFILE_TRAIN_DATA" in missing_data.stderr


def test_tmax_profile_replaces_stale_model_and_data_environment(tmp_path: Path) -> None:
    checkpoint = tmp_path / "release"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("release\n")
    train_data = tmp_path / "train.jsonl"
    train_data.write_text('{"prompt": "test"}\n')

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

    data_root = tmp_path / "data"
    env = os.environ.copy()
    for name in (
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
            "PROFILE_ID": "unit-lineage",
            "PROFILE_LOAD_DIR": str(checkpoint),
            "PROFILE_TRAIN_DATA": str(train_data),
            # Simulate a shell previously used for a Qwen3.5-4B run.
            "HF_CHECKPOINT": "/stale/Qwen3.5-4B",
            "REF_LOAD": "/stale/Qwen3.5-4B_torch_dist",
            "TORCH_DIST_DIR": "/stale/torch_dist",
            "MODEL_ARGS_FILE": "/stale/model_args.sh",
            "POLAR_AGENT_MODEL_NAME": "Qwen/Qwen3.5-4B",
            "TMAX_TRAIN_DATA": "/stale/train.jsonl",
            "PROMPT_DATA": "/stale/prompts.jsonl",
        }
    )

    result = subprocess.run(
        [
            "/bin/bash",
            str(PROFILE / "submit_profile.sh"),
            "submit",
            "async-8t24r-l4",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    submitted = dict(
        line.split("=", 1) for line in captured_env.read_text().splitlines() if "=" in line
    )
    assert submitted["HF_CHECKPOINT"] == str(data_root / "checkpoints" / "Qwen3.5-9B")
    assert submitted["REF_LOAD"] == str(data_root / "checkpoints" / "Qwen3.5-9B_torch_dist")
    assert submitted["TORCH_DIST_DIR"] == submitted["REF_LOAD"]
    assert submitted["MODEL_ARGS_FILE"] == str(
        ROOT / "examples" / "tmax_slime_grpo" / "model_args.sh"
    )
    assert submitted["POLAR_AGENT_MODEL_NAME"] == "Qwen/Qwen3.5-9B"
    assert submitted["LOAD_DIR"] == str(checkpoint)
    assert submitted["TMAX_TRAIN_DATA"] == str(train_data)
    assert submitted["PROMPT_DATA"] == str(train_data)
