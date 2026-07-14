from __future__ import annotations

import json
import hashlib
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from polar.config.topology import TopologyConfig


ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "examples" / "swegym_slime_grpo"
TMAX = ROOT / "examples" / "tmax_slime_grpo"
SPILOT = ROOT / "examples" / "spilot_router_slime_grpo"


def spilot_entrypoint_env() -> dict[str, str]:
    return {
        "TMAX_SUBMIT_SCRIPT": str(SPILOT / "submit_slurm.sh"),
        "POLAR_TRAIN_RUN_SCRIPT": str(SPILOT / "run.sh"),
        "POLAR_CONFIG_TEMPLATE": str(SPILOT / "polar_config.yaml"),
        "TOPOLOGY_TEMPLATE": str(SPILOT / "topology.yaml"),
    }


def run_bash(script: str, *, env: dict[str, str] | None = None, check: bool = True):
    return subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def clean_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir()
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "USER": os.environ.get("USER", "test"),
        "POLAR_DATA_ROOT": str(tmp_path / "data"),
    }


def test_shared_launcher_enables_post_train_full_trajectory_examples() -> None:
    launcher = (SHARED / "run.sh").read_text()

    assert 'POLAR_ROLLOUT_EXAMPLE_INTERVAL="${POLAR_ROLLOUT_EXAMPLE_INTERVAL:-10}"' in launcher
    assert 'POLAR_ROLLOUT_EXAMPLE_COUNT="${POLAR_ROLLOUT_EXAMPLE_COUNT:-2}"' in launcher
    assert '--custom-rollout-log-function-path "${CUSTOM_ROLLOUT_LOG_FUNCTION_PATH}"' in launcher
    assert "slime_bridge.rollout.log_rollout_trajectory_examples" in launcher


def test_launchers_keep_runtime_outputs_under_the_data_root() -> None:
    shared_run = (SHARED / "run.sh").read_text()
    shared_submit = (SHARED / "submit_slurm.sh").read_text()
    swegym_sif_submit = (SHARED / "submit_build_sifs_slurm.sh").read_text()
    tmax_sif_submit = (ROOT / "examples" / "tmax-15k" / "submit_build_sifs_slurm.sh").read_text()
    hf_export = (TMAX / "export_hf_checkpoint.sh").read_text()

    assert 'export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}/wandb}"' in shared_run
    assert 'mkdir -p "${RUN_DIR}" "${WANDB_DIR}"' in shared_run
    assert '\\"WANDB_DIR\\": \\"${WANDB_DIR}\\"' in shared_run
    assert '${PROJECT_ROOT}/logs' not in shared_run

    assert 'LOG_DIR="${POLAR_SLURM_LOG_DIR:-${DATA_ROOT}/logs/slurm}"' in shared_submit
    assert '${PROJECT_ROOT}/logs/slurm' not in shared_submit
    assert (
        'LOG_DIR="${SIF_BUILD_LOG_DIR:-${POLAR_DATA_ROOT}/logs/slurm}"'
        in swegym_sif_submit
    )
    assert (
        'LOG_DIR="${TMAX_SIF_BUILD_LOG_DIR:-${TMAX_DATA_ROOT}/logs/slurm}"'
        in tmax_sif_submit
    )
    assert 'log_dir="${TMAX_HF_EXPORT_LOG_DIR:-${DATA_ROOT}/logs/slurm}"' in hf_export


def test_shared_launcher_defaults_and_overrides_physical_wandb_run_id() -> None:
    launcher = (SHARED / "run.sh").read_text()
    assignment = re.search(
        r'^export WANDB_RUN_ID="\$\{WANDB_RUN_ID:-\$\{RUN_ID\}\}"$',
        launcher,
        re.M,
    )

    assert assignment is not None
    default = run_bash(
        f'RUN_ID=logical-run\nunset WANDB_RUN_ID\n{assignment.group(0)}\nprintf %s "$WANDB_RUN_ID"'
    )
    override = run_bash(
        f'RUN_ID=logical-run\nWANDB_RUN_ID=telemetry-run\n{assignment.group(0)}\nprintf %s "$WANDB_RUN_ID"'
    )

    assert default.stdout == "logical-run"
    assert override.stdout == "telemetry-run"
    assert '"--wandb-run-id" "$WANDB_RUN_ID"' in launcher
    assert '\\"WANDB_RUN_ID\\": \\"${WANDB_RUN_ID}\\"' in launcher
    assert '--wandb-run-id "$WANDB_RUN_ID" \\' in launcher
    assert '"--wandb-run-id" "$RUN_ID"' not in launcher
    assert '\\"WANDB_RUN_ID\\": \\"${RUN_ID}\\"' not in launcher
    assert '--wandb-run-id "$RUN_ID" \\' not in launcher


def write_command(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


def write_distcp_metadata(model_dir: Path, *shard_names: str) -> None:
    payload = {
        "storage_data": [
            {"relative_path": name, "offset": 0, "length": len(b"weights")}
            for name in shard_names
        ]
    }
    (model_dir / ".metadata").write_bytes(pickle.dumps(payload, protocol=4))


def tmax_submit_env(tmp_path: Path, *, load_pointer: str) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "srun", "exit 0\n")
    fake_python = bin_dir / "python"
    write_command(
        fake_python,
        """
if [[ "${1:-}" == *validate_data_integrity.py || "${1:-}" == -c ]]; then
    exec "$REAL_PYTHON" "$@"
fi
output=
validate_existing=0
while [ "$#" -gt 0 ]; do
    if [ "$1" = --output ]; then
        output="$2"
        shift 2
    elif [ "$1" = --validate-existing ]; then
        validate_existing=1
        shift
    else
        shift
    fi
done
test -n "$output"
if [ "$validate_existing" = 1 ]; then
    test -s "$output"
    exit 0
fi
mkdir -p "$(dirname "$output")"
task_name=fake_train
case "$(basename "$output")" in
    *terminal_bench*) task_name=fake_external_eval ;;
    *eval*) task_name=fake_eval ;;
esac
printf '{"prompt": [], "metadata": {"task_name": "%s"}}\n' "$task_name" > "$output"
""",
    )
    agent_dir = tmp_path / "agent"
    (agent_dir / "bin").mkdir(parents=True)
    write_command(agent_dir / "bin" / "codex", "exit 0\n")
    megatron_dir = tmp_path / "Megatron-LM"
    tokenizer = megatron_dir / "megatron" / "training" / "tokenizer" / "tokenizer.py"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("")
    load_dir = tmp_path / "load"
    load_dir.mkdir()
    (load_dir / "latest_checkpointed_iteration.txt").write_text(f"{load_pointer}\n")

    env = clean_env(tmp_path)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        REAL_PYTHON=sys.executable,
        RUN_ID="fresh-release",
        LOAD_DIR=str(load_dir),
        REF_LOAD=str(load_dir),
        TMAX_SIF_PYTHON_BIN=str(fake_python),
        TMAX_AGENT_HARNESS="codex",
        AGENT_CLI_DIR=str(agent_dir),
        TMAX_DATASET_DIR=str(tmp_path / "dataset"),
        APPTAINER_IMAGE_DIR=str(tmp_path / "images"),
        POLR_TRAIN_SQSH="registry.example/train:latest",
        POLR_TRAIN_VENV=str(tmp_path / "venv"),
        SLIME_DIR=str(ROOT.parent / "slime"),
        MEGATRON_DIR=str(megatron_dir),
        TMAX_REQUIRE_WANDB="0",
        TMAX_TRAIN_ABI_PREFLIGHT="0",
        TMAX_PERSIST_RUN_STATE="0",
        SUBMIT_DRY_RUN="1",
        SUBMIT_BACKEND="sbatch",
    )
    return env


def tmax_numeric_seed_env(tmp_path: Path) -> dict[str, str]:
    env = tmax_submit_env(tmp_path, load_pointer="47")
    train_data = tmp_path / "reused.jsonl"
    train_data.write_text("{}\n" * 100)
    env.update(
        TMAX_TRAIN_DATA=str(train_data),
        TMAX_PREPARE_DATA="0",
        TMAX_VALIDATE_EXISTING_ASSETS="0",
        TMAX_EVAL_ENABLED="0",
        TMAX_NUM_ROLLOUT="50",
    )
    return env


def test_tmax_submit_fails_closed_on_training_abi_preflight(tmp_path: Path) -> None:
    env = tmax_submit_env(tmp_path, load_pointer="release")
    abi_python = tmp_path / "bin" / "abi-python"
    write_command(abi_python, "exit 41\n")
    env.update(
        TMAX_TRAIN_ABI_PREFLIGHT="1",
        TMAX_TRAIN_PYTHON_BIN=str(abi_python),
    )

    result = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "training ABI preflight failed before Slurm submission" in result.stderr
    assert "Dry run only" not in result.stdout


@pytest.mark.parametrize(
    ("credential_env", "expected_error"),
    [
        ({}, "SPilot Router requires POLAR_CONTROL_PLANE_TOKEN"),
        (
            {"POLAR_CONTROL_PLANE_TOKEN": "c" * 32},
            "SPilot Router requires POLAR_NVIDIA_API_KEY",
        ),
        (
            {
                "POLAR_CONTROL_PLANE_TOKEN": "too-short",
                "POLAR_NVIDIA_API_KEY": "test-model-pool-key",
                "POLAR_MODEL_POOL_BASE_URL": "https://model-pool.invalid/v1",
            },
            "POLAR_CONTROL_PLANE_TOKEN must be a 32-128 character opaque token",
        ),
        (
            {
                "POLAR_CONTROL_PLANE_TOKEN": "c" * 32,
                "POLAR_NVIDIA_API_KEY": "test-model-pool-key",
            },
            "SPilot Router requires POLAR_MODEL_POOL_BASE_URL",
        ),
    ],
)
def test_tmax_submit_fails_before_sbatch_when_spilot_wrapper_was_bypassed(
    tmp_path: Path,
    credential_env: dict[str, str],
    expected_error: str,
) -> None:
    env = tmax_submit_env(tmp_path, load_pointer="release")
    env["TMAX_AGENT_HARNESS"] = "spilot_router"
    env.update(spilot_entrypoint_env())
    env.update(credential_env)

    result = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert expected_error in result.stderr
    assert "Dry run only" not in result.stdout


def test_tmax_submit_rejects_spilot_credentials_with_generic_runtime_entrypoint(
    tmp_path: Path,
) -> None:
    env = tmax_submit_env(tmp_path, load_pointer="release")
    env.update(
        spilot_entrypoint_env(),
        TMAX_AGENT_HARNESS="spilot_router",
        POLAR_CONTROL_PLANE_TOKEN="c" * 32,
        POLAR_NVIDIA_API_KEY="test-model-pool-key",
        POLAR_MODEL_POOL_BASE_URL="https://model-pool.invalid/v1",
        POLAR_TRAIN_RUN_SCRIPT=str(TMAX / "run.sh"),
    )

    result = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "submission preflight: SPilot Router requires POLAR_TRAIN_RUN_SCRIPT=" in result.stderr
    assert "Dry run only" not in result.stdout


def test_tmax_allocation_rejects_missing_spilot_credentials_before_run_dir(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "must-not-exist"
    env = clean_env(tmp_path)
    env.update(
        spilot_entrypoint_env(),
        TMAX_AGENT_HARNESS="spilot_router",
        RUN_ID="router-run",
        RUN_DIR=str(run_dir),
    )

    result = subprocess.run(
        ["bash", str(TMAX / "run.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "allocation startup: SPilot Router requires POLAR_CONTROL_PLANE_TOKEN" in result.stderr
    assert not run_dir.exists()


def test_tmax_submit_allows_explicit_release_seed_with_fresh_data(
    tmp_path: Path,
) -> None:
    env = tmax_submit_env(tmp_path, load_pointer="release")

    result = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Dry run only; sbatch wrapper:" in result.stdout
    assert (tmp_path / "data" / "runs" / "fresh-release" / "tmax-train.jsonl").is_file()


@pytest.mark.parametrize(
    ("wandb_run_id", "expected"),
    [(None, "fresh-release"), ("telemetry-run", "telemetry-run")],
)
def test_tmax_submit_snapshots_physical_wandb_run_id_without_changing_lineage(
    tmp_path: Path, wandb_run_id: str | None, expected: str
) -> None:
    env = tmax_submit_env(tmp_path, load_pointer="release")
    if wandb_run_id is not None:
        env["WANDB_RUN_ID"] = wandb_run_id

    result = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    logical_run_dir = tmp_path / "data" / "runs" / "fresh-release"
    snapshots = list((logical_run_dir / "submit").glob("env-*.sh"))
    assert len(snapshots) == 1
    assert f"export WANDB_RUN_ID={expected}\n" in snapshots[0].read_text()
    assert not (tmp_path / "data" / "runs" / "telemetry-run").exists()


def test_sbatch_submission_scrubs_parent_slurm_memory_contract(
    tmp_path: Path,
) -> None:
    env = tmax_submit_env(tmp_path, load_pointer="release")
    bin_dir = tmp_path / "bin"
    captured_env = tmp_path / "sbatch.env"
    captured_args = tmp_path / "sbatch.args"
    write_command(
        bin_dir / "sbatch",
        """
/usr/bin/env | LC_ALL=C sort > "$SBATCH_ENV_CAPTURE"
printf '%s\n' "$@" > "$SBATCH_ARGS_CAPTURE"
printf '424242;test-cluster\n'
""",
    )
    inherited_memory_vars = {
        "SLURM_MEM_PER_NODE": "2048",
        "SLURM_MEM_PER_CPU": "1024",
        "SLURM_MEM_PER_GPU": "512",
        "SBATCH_MEM_PER_NODE": "2G",
        "SBATCH_MEM_PER_CPU": "1G",
        "SBATCH_MEM_PER_GPU": "1G",
    }
    env.update(
        inherited_memory_vars,
        SUBMIT_DRY_RUN="0",
        SLURM_JOB_ID="111111",
        SLURM_JOBID="111111",
        SBATCH_ENV_CAPTURE=str(captured_env),
        SBATCH_ARGS_CAPTURE=str(captured_args),
    )

    result = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    submitted_env = captured_env.read_text().splitlines()
    for name in inherited_memory_vars:
        assert not any(line.startswith(f"{name}=") for line in submitted_env)
    assert "--mem=0" in captured_args.read_text().splitlines()
    assert "Submitted SLURM job: 424242" in result.stdout


def test_sbatch_submission_supports_explicit_partial_node_memory(
    tmp_path: Path,
) -> None:
    env = tmax_submit_env(tmp_path, load_pointer="release")
    bin_dir = tmp_path / "bin"
    captured_args = tmp_path / "sbatch.args"
    write_command(
        bin_dir / "sbatch",
        """
printf '%s\n' "$@" > "$SBATCH_ARGS_CAPTURE"
printf '424243;test-cluster\n'
""",
    )
    env.update(
        SUBMIT_DRY_RUN="0",
        POLAR_SLURM_MEM_PER_NODE="250G",
        SBATCH_ARGS_CAPTURE=str(captured_args),
    )

    result = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--mem=250G" in captured_args.read_text().splitlines()


def test_tmax_submit_rejects_stale_mini_swe_timing_runtime(tmp_path: Path) -> None:
    env = tmax_submit_env(tmp_path, load_pointer="release")
    runtime = tmp_path / "mini-swe-runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "venv" / "bin").mkdir(parents=True)
    installed_module = runtime / "polar_mini_swe_timing.py"
    installed_runner = runtime / "polar_mini_swe_runner.py"
    installed_vanillux = runtime / "polar_mini_swe_vanillux.py"
    installed_vanillux_config = runtime / "config" / "vanillux2.yaml"
    installed_vanillux_config.parent.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "src" / "polar" / "agent" / "presets" / "mini_swe_timing.py",
        installed_module,
    )
    shutil.copyfile(
        ROOT / "src" / "polar" / "agent" / "presets" / "mini_swe_runner.py",
        installed_runner,
    )
    shutil.copyfile(
        ROOT / "src" / "polar" / "agent" / "presets" / "mini_swe_vanillux.py",
        installed_vanillux,
    )
    shutil.copyfile(
        ROOT / "src" / "polar" / "agent" / "presets" / "vanillux2.yaml",
        installed_vanillux_config,
    )
    current_wrapper = 'export POLAR_TASK_PYTHONPATH="${PYTHONPATH:-}"\nunset PYTHONPATH\nexit 0\n'
    write_command(runtime / "bin" / "mini-swe-agent", current_wrapper)
    write_command(
        runtime / "venv" / "bin" / "python",
        """
if [[ " $* " == *" polar_mini_swe_runner "* ]]; then
    printf '%s\n' "$FAKE_MINI_SWE_RUNNER_MODULE"
elif [[ " $* " == *" polar_mini_swe_vanillux "* ]]; then
    printf '%s\n' "$FAKE_MINI_SWE_VANILLUX_MODULE"
else
    printf '%s\n' "$FAKE_MINI_SWE_TIMING_MODULE"
fi
""",
    )
    env.update(
        TMAX_AGENT_HARNESS="mini_swe_agent",
        MINI_SWE_AGENT_RUNTIME_DIR=str(runtime),
        MINI_SWE_AGENT_BIN=str(runtime / "bin" / "mini-swe-agent"),
        FAKE_MINI_SWE_TIMING_MODULE=str(installed_module),
        FAKE_MINI_SWE_RUNNER_MODULE=str(installed_runner),
        FAKE_MINI_SWE_VANILLUX_MODULE=str(installed_vanillux),
    )

    current = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert current.returncode == 0, current.stderr

    write_command(runtime / "bin" / "mini-swe-agent", "exit 0\n")
    stale_wrapper = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert stale_wrapper.returncode == 1
    assert "runtime is missing or stale" in stale_wrapper.stderr
    assert "Dry run only" not in stale_wrapper.stdout

    write_command(runtime / "bin" / "mini-swe-agent", current_wrapper)

    installed_module.write_text("TIMING_SCHEMA_VERSION = 1\n# stale copy\n")
    stale = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert stale.returncode == 1
    assert "runtime is missing or stale" in stale.stderr
    assert "must match this checkout" in stale.stderr
    assert "prepare_mini_swe_agent.sh" in stale.stderr
    assert "Dry run only" not in stale.stdout


def test_tmax_delegation_routes_the_container_back_to_tmax() -> None:
    tmax_submit = (TMAX / "submit_slurm.sh").read_text()
    shared_submit = (SHARED / "submit_slurm.sh").read_text()
    container_entrypoint = (SHARED / "run_in_container.sh").read_text()
    tmax_run = (TMAX / "run.sh").read_text()

    assert (
        'export POLAR_TRAIN_RUN_SCRIPT="${POLAR_TRAIN_RUN_SCRIPT:-${SCRIPT_DIR}/run.sh}"'
        in tmax_submit
    )
    assert 'TRAIN_RUN_SCRIPT="${POLAR_TRAIN_RUN_SCRIPT:-${SCRIPT_DIR}/run.sh}"' in shared_submit
    assert (
        'TRAIN_RUN_SCRIPT="${POLAR_TRAIN_RUN_SCRIPT:?set POLAR_TRAIN_RUN_SCRIPT}"'
        in container_entrypoint
    )
    assert 'exec bash "${TRAIN_RUN_SCRIPT}"' in container_entrypoint
    assert "export REQUIRE_SWEGYM_HARNESS=0" in tmax_run
    assert (
        'export TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${SCRIPT_DIR}/topology.yaml}"' in tmax_run
    )
    assert (
        'export POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${SCRIPT_DIR}/polar_config.yaml}"'
        in tmax_run
    )


def test_fixed_eval_completion_marker_is_wired_with_dataset_hash() -> None:
    shared_run = (SHARED / "run.sh").read_text()
    env = (TMAX / "env.cwdfw.sh").read_text()
    run_state = (TMAX / "run_state.sh").read_text()

    assert "--final-eval-complete-marker" in shared_run
    assert "--final-eval-data-sha256" in shared_run
    assert "FINAL_EVAL_DATA_SHA256" in shared_run
    assert "FINAL_EVAL_COMPLETE_MARKER" in env
    assert "FINAL_EVAL_COMPLETE_MARKER" in run_state
    assert "TMAX_EVAL_DATA_SHA256" in run_state


def test_launcher_records_real_startup_barriers_without_fixed_ray_sleep() -> None:
    shared_run = (SHARED / "run.sh").read_text()
    shared_submit = (SHARED / "submit_slurm.sh").read_text()
    container_entrypoint = (SHARED / "run_in_container.sh").read_text()

    assert "Waiting for ${RAY_NUM_NODES} Ray ranks to report ready" in shared_run
    assert 'touch "${RAY_READY_DIR}/head_ready"' in shared_run
    assert 'ray.cluster_resources().get("GPU", 0)' in shared_run
    assert "sleep 45" not in shared_run
    for marker in (
        "SLIME_SUBMIT_UNIX_NS",
        "SLIME_SLURM_BATCH_START_UNIX_NS",
        "SLIME_CONTAINER_ENTRY_UNIX_NS",
        "SLIME_JOB_SCRIPT_START_UNIX_NS",
        "SLIME_RAY_READY_UNIX_NS",
        "SLIME_ROLLOUT_SERVICE_START_UNIX_NS",
        "SLIME_ROLLOUT_SERVICE_READY_UNIX_NS",
        "SLIME_GATEWAY_START_UNIX_NS",
        "SLIME_GATEWAY_READY_UNIX_NS",
        "SLIME_UDS_TUNNEL_START_UNIX_NS",
        "SLIME_UDS_TUNNEL_READY_UNIX_NS",
        "SLIME_SERVICES_READY_UNIX_NS",
        "SLIME_RAY_JOB_SUBMIT_UNIX_NS",
    ):
        assert f'\\"{marker}\\"' in shared_run
    assert "export SLIME_SLURM_BATCH_START_UNIX_NS=\\$(date +%s%N)" in shared_submit
    assert '_SLIME_CONTAINER_ENTRY_UNIX_NS="$(date +%s%N)"' in container_entrypoint
    assert "SLIME_POLAR_" not in shared_run
    assert "SLIME_POLAR_" not in container_entrypoint


def test_ray_primary_receives_gpu_axis_topology() -> None:
    shared_run = (SHARED / "run.sh").read_text()
    tmax_run = (TMAX / "run.sh").read_text()

    assert 'export GPU_MONITOR_PREFIX="${GPU_MONITOR_PREFIX:-polar_tmax_system}"' in tmax_run
    assert '\\"GPU_MONITOR_PREFIX\\": \\"${GPU_MONITOR_PREFIX:-polar_system}\\"' in shared_run
    assert '\\"GPU_MONITOR_NODE_ROLE\\": \\"${GPU_MONITOR_NODE_ROLE:-}\\"' in shared_run
    assert '\\"SLURM_NNODES\\": \\"${RAY_NUM_NODES}\\"' in shared_run


def test_tmax_uses_paper_aligned_dppo_optimizer_defaults() -> None:
    env = (TMAX / "env.cwdfw.sh").read_text()
    run_state = (TMAX / "run_state.sh").read_text()
    shared_run = (SHARED / "run.sh").read_text()
    shared_submit = (SHARED / "submit_slurm.sh").read_text()

    assert 'export POLICY_LOSS_TYPE="${POLICY_LOSS_TYPE:-dppo}"' in env
    assert 'export DPPO_DIVERGENCE_TYPE="${DPPO_DIVERGENCE_TYPE:-tv}"' in env
    assert 'export DPPO_DIVERGENCE_THRESHOLD="${DPPO_DIVERGENCE_THRESHOLD:-0.1}"' in env
    assert 'export USE_TIS="${USE_TIS:-0}"' in env
    assert 'export TRAIN_LR="${TRAIN_LR:-1e-6}"' in env
    assert 'export KL_LOSS_COEF="${KL_LOSS_COEF:-0}"' in env
    assert 'export GRPO_STD_NORMALIZATION="${GRPO_STD_NORMALIZATION:-0}"' in env
    assert 'export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"' in env
    assert '--lr "${TRAIN_LR:-1e-6}"' in shared_run
    assert '--policy-loss-type "${POLICY_LOSS_TYPE}"' in shared_run
    assert "--use-rollout-logprobs" in shared_run
    assert "--dppo-divergence-threshold" in shared_run
    assert "--disable-grpo-std-normalization" in shared_run
    assert shared_run.count("--use-kl-loss") == 1
    assert '"${KL_LOSS_ARGS[@]}"' in shared_run
    assert '"${GRPO_NORMALIZATION_ARGS[@]}"' in shared_run
    assert "POLICY_LOSS_TYPE USE_TIS" in run_state
    assert "GRPO_STD_NORMALIZATION" in run_state
    assert "DPPO_DIVERGENCE_TYPE DPPO_DIVERGENCE_THRESHOLD" in run_state
    assert "TRAIN_LR|KL_LOSS_COEF|POLICY_LOSS_TYPE|USE_TIS" in shared_submit


def test_sglang_deterministic_inference_is_opt_in_and_forwards_backend(
    tmp_path: Path,
) -> None:
    shared_run = (SHARED / "run.sh").read_text()
    start = shared_run.index("configure_sglang_inference_args() {")
    end = shared_run.index("\n}\n", start) + 2
    function_source = shared_run[start:end]
    script = f"""
set -euo pipefail
{function_source}
configure_sglang_inference_args
printf '<%s>|<%s>' \
    "${{SGLANG_DETERMINISTIC_ARGS[*]}}" \
    "${{SGLANG_ATTENTION_BACKEND_ARGS[*]}}"
"""

    base_env = clean_env(tmp_path)
    default = run_bash(script, env=base_env)
    assert default.stdout == "<>|<>"

    enabled_env = base_env.copy()
    enabled_env.update(
        SGLANG_ENABLE_DETERMINISTIC_INFERENCE="true",
        SGLANG_ATTENTION_BACKEND="fa3",
    )
    enabled = run_bash(script, env=enabled_env)
    assert enabled.stdout.splitlines()[-1] == (
        "<--sglang-enable-deterministic-inference>|<--sglang-attention-backend fa3>"
    )
    assert "Using SGLang deterministic inference" in enabled.stdout

    invalid_env = base_env.copy()
    invalid_env["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "yes"
    invalid = run_bash(script, env=invalid_env, check=False)
    assert invalid.returncode != 0
    assert "must be 0/1/false/true" in invalid.stderr

    assert '"${SGLANG_DETERMINISTIC_ARGS[@]}" \\' in shared_run
    assert '"${SGLANG_ATTENTION_BACKEND_ARGS[@]}" \\' in shared_run


def test_hf_export_dry_run_is_generic_and_uses_minimal_slurm_environment(
    tmp_path: Path,
) -> None:
    script = TMAX / "export_hf_checkpoint.sh"
    checkpoint = tmp_path / "checkpoint"
    model_dir = checkpoint / "iter_0000047"
    model_dir.mkdir(parents=True)
    (model_dir / "common.pt").write_bytes(b"common")
    write_distcp_metadata(model_dir, "__0_0.distcp")
    (model_dir / "__0_0.distcp").write_bytes(b"weights")
    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / "config.json").write_text(
        json.dumps({"model_type": "test_model", "vocab_size": 123})
    )
    (origin / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    slime_dir = tmp_path / "slime"
    (slime_dir / "tools").mkdir(parents=True)
    (slime_dir / "tools" / "convert_torch_dist_to_hf.py").write_text("")

    env = os.environ.copy()
    env.update(
        TMAX_HF_EXPORT_RUN_ID="test-run",
        TMAX_HF_EXPORT_CKPT_ROOT=str(checkpoint),
        TMAX_HF_EXPORT_OUTPUT_ROOT=str(tmp_path / "output"),
        TMAX_HF_EXPORT_ORIGIN=str(origin),
        TMAX_HF_EXPORT_PYTHON=sys.executable,
        TMAX_HF_EXPORT_LOG_DIR=str(tmp_path / "logs"),
        SLIME_DIR=str(slime_dir),
    )

    result = subprocess.run(
        ["bash", str(script), "--dry-run", "47"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    source = script.read_text()
    assert "tmax-8n64-qwen35-4b-lr1e6" not in source
    assert "Qwen3.5-4B" not in source
    assert '--vocab-size "${origin_vocab_size}"' in source
    assert "--export=ALL" not in source
    assert "--export=" in result.stdout
    assert "TMAX_HF_EXPORT_RUN_ID=test-run" in result.stdout
    assert "iter_0000047-bf16" in result.stdout

    missing_origin_env = env.copy()
    missing_origin_env.pop("TMAX_HF_EXPORT_ORIGIN")
    missing_origin = subprocess.run(
        ["bash", str(script), "--dry-run", "47"],
        cwd=ROOT,
        env=missing_origin_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_origin.returncode != 0
    assert "set TMAX_HF_EXPORT_ORIGIN" in missing_origin.stderr

    relative_checkpoint_env = env.copy()
    relative_checkpoint_env["TMAX_HF_EXPORT_CKPT_ROOT"] = "relative/checkpoint"
    relative_checkpoint = subprocess.run(
        ["bash", str(script), "--dry-run", "47"],
        cwd=ROOT,
        env=relative_checkpoint_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert relative_checkpoint.returncode != 0
    assert (
        "TMAX_HF_EXPORT_CKPT_ROOT must be an absolute path"
        in relative_checkpoint.stderr
    )


def test_explicit_num_rollout_is_forwarded_persisted_and_targets_n_minus_one(
    tmp_path: Path,
) -> None:
    env_script = (TMAX / "env.cwdfw.sh").read_text()
    run_state = (TMAX / "run_state.sh").read_text()
    watcher = (TMAX / "watch_training.sh").read_text()
    shared_run = (SHARED / "run.sh").read_text()
    submit = (TMAX / "submit_slurm.sh").read_text()

    assert "TMAX_NUM_ROLLOUT TMAX_TARGET_ITER" in run_state
    assert 'TRAIN_LENGTH_ARGS=(--num-epoch "${NUM_EPOCH:-1}")' in shared_run
    assert 'TRAIN_LENGTH_ARGS=(--num-rollout "${TMAX_NUM_ROLLOUT}")' in shared_run
    assert '"${TRAIN_LENGTH_ARGS[@]}" \\' in shared_run
    assert shared_run.count('--num-epoch "${NUM_EPOCH:-1}"') == 1
    assert "printf '%s\\n' \"$((TMAX_NUM_ROLLOUT - 1))\"" in watcher
    assert '_tmax_seed_target="$((TMAX_NUM_ROLLOUT - 1))"' in submit
    assert "tmax_validate_numbered_checkpoint" in submit
    assert "global_dataset_state_dict_${value}.pt" in (TMAX / "lifecycle.sh").read_text()
    assert "TMAX_NUM_ROLLOUT - 1" in env_script

    env = clean_env(tmp_path)
    env["TMAX_NUM_ROLLOUT"] = "50"
    result = run_bash(
        f"source {TMAX / 'env.cwdfw.sh'} >/dev/null; "
        'printf \'%s/%s\' "$TMAX_NUM_ROLLOUT" "$TMAX_TARGET_ITER"',
        env=env,
    )
    assert result.stdout == "50/49"

    mismatch_env = env.copy()
    mismatch_env["TMAX_TARGET_ITER"] = "48"
    mismatch = run_bash(
        f"source {TMAX / 'env.cwdfw.sh'} >/dev/null",
        env=mismatch_env,
        check=False,
    )
    assert mismatch.returncode != 0
    assert "must equal TMAX_NUM_ROLLOUT-1=49" in mismatch.stderr


def test_checkpoint_retention_is_optional_validated_and_forwarded(tmp_path: Path) -> None:
    env_script = (TMAX / "env.cwdfw.sh").read_text()
    shared_run = (SHARED / "run.sh").read_text()
    shared_submit = (SHARED / "submit_slurm.sh").read_text()
    run_state = (TMAX / "run_state.sh").read_text()

    assert 'export SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL:-}"' in env_script
    assert 'SAVE_RETENTION_ARGS+=(--save-retain-interval "${SAVE_RETAIN_INTERVAL}")' in shared_run
    assert '"${SAVE_RETENTION_ARGS[@]}" \\' in shared_run
    assert "SAVE_INTERVAL|SAVE_RETAIN_INTERVAL|SEQUENCE_PARALLEL" in shared_submit
    assert "SAVE_INTERVAL SAVE_RETAIN_INTERVAL" in run_state

    optional_env = clean_env(tmp_path)
    optional = run_bash(
        f"source {TMAX / 'env.cwdfw.sh'} >/dev/null; "
        'printf \'%s|%s\' "$SAVE_INTERVAL" "$SAVE_RETAIN_INTERVAL"',
        env=optional_env,
    )
    assert optional.stdout == "10|"

    valid_env = optional_env.copy()
    valid_env.update(SAVE_INTERVAL="5", SAVE_RETAIN_INTERVAL="20")
    valid = run_bash(
        f"source {TMAX / 'env.cwdfw.sh'} >/dev/null; "
        'printf \'%s|%s\' "$SAVE_INTERVAL" "$SAVE_RETAIN_INTERVAL"',
        env=valid_env,
    )
    assert valid.stdout == "5|20"

    invalid_cases = (
        ({"SAVE_INTERVAL": "0"}, "SAVE_INTERVAL must be a positive integer"),
        (
            {"SAVE_INTERVAL": "5", "SAVE_RETAIN_INTERVAL": "0"},
            "SAVE_RETAIN_INTERVAL must be a positive integer",
        ),
        (
            {"SAVE_INTERVAL": "6", "SAVE_RETAIN_INTERVAL": "20"},
            "must be divisible by SAVE_INTERVAL=6",
        ),
    )
    for overrides, message in invalid_cases:
        invalid_env = optional_env.copy()
        invalid_env.update(overrides)
        invalid = run_bash(
            f"source {TMAX / 'env.cwdfw.sh'} >/dev/null",
            env=invalid_env,
            check=False,
        )
        assert invalid.returncode != 0
        assert message in invalid.stderr


def test_resumed_checkpoint_eval_flag_only_targets_first_external_numeric_seed(
    tmp_path: Path,
) -> None:
    shared_run = (SHARED / "run.sh").read_text()
    start = shared_run.index("configure_resumed_checkpoint_eval_args() {")
    end = shared_run.index("\n}\n", start) + 2
    function_source = shared_run[start:end]
    seed_dir = tmp_path / "external-seed"
    save_dir = tmp_path / "logical-run"
    seed_dir.mkdir()
    save_dir.mkdir()
    tracker = seed_dir / "latest_checkpointed_iteration.txt"
    tracker.write_text("39\n")

    script = f"""
set -euo pipefail
polar_checkpoint_is_release_seed() {{
    [ "$(tr -d '[:space:]' <"$1/latest_checkpointed_iteration.txt")" = release ]
}}
{function_source}
configure_resumed_checkpoint_eval_args
printf '<%s>' "${{RESUMED_CHECKPOINT_EVAL_ARGS[*]}}"
"""
    base_env = clean_env(tmp_path)
    base_env.update(
        TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN="1",
        TMAX_CONCURRENT_PRETRAIN_EVAL="0",
        TMAX_EVAL_ENABLED="1",
        REQUESTED_LOAD_DIR=str(seed_dir),
        LOAD_DIR=str(seed_dir),
        SAVE_DIR=str(save_dir),
    )

    external_seed = run_bash(script, env=base_env)
    assert external_seed.stdout.splitlines()[-1] == ("<--eval-resumed-checkpoint-before-train>")
    assert "checkpoint 39 before rollout 40" in external_seed.stdout

    internal_resume_env = base_env.copy()
    internal_resume_env["LOAD_DIR"] = str(save_dir)
    internal_resume = run_bash(script, env=internal_resume_env)
    assert internal_resume.stdout == "<>"

    tracker.write_text("release\n")
    release_seed = run_bash(script, env=base_env)
    assert release_seed.stdout == "<>"
    tracker.write_text("39\n")

    for name, value, expected_error in (
        (
            "TMAX_CONCURRENT_PRETRAIN_EVAL",
            "1",
            "requires TMAX_CONCURRENT_PRETRAIN_EVAL=0",
        ),
        (
            "TMAX_EVAL_ENABLED",
            "0",
            "requires training-time eval",
        ),
        (
            "TMAX_TRAINING_EVAL_ENABLED",
            "0",
            "requires training-time eval",
        ),
    ):
        invalid_env = base_env.copy()
        invalid_env[name] = value
        invalid = run_bash(script, env=invalid_env, check=False)
        assert invalid.returncode != 0
        assert expected_error in invalid.stderr


def test_training_eval_schedule_can_be_disabled_without_disabling_holdout_contract(
    tmp_path: Path,
) -> None:
    env = clean_env(tmp_path)
    env.update(
        TMAX_EVAL_ENABLED="1",
        TMAX_TRAINING_EVAL_ENABLED="0",
    )
    result = run_bash(
        f"source {TMAX / 'env.cwdfw.sh'} >/dev/null; "
        'printf \'%s|%s\' "$TMAX_EVAL_ENABLED" "$TMAX_TRAINING_EVAL_ENABLED"',
        env=env,
    )

    assert result.stdout == "1|0"
    shared_run = (SHARED / "run.sh").read_text()
    run_state = (TMAX / "run_state.sh").read_text()
    watcher = (TMAX / "watch_training.sh").read_text()
    assert "TMAX_TRAINING_EVAL_ENABLED:-${TMAX_EVAL_ENABLED:-0}" in shared_run
    assert "Training-time eval disabled" in shared_run
    assert "TMAX_TRAINING_EVAL_ENABLED" in run_state
    assert "TMAX_TRAINING_EVAL_ENABLED:-${TMAX_EVAL_ENABLED}" in watcher


def test_tmax_uses_paper_faithful_full_groups_and_dynamic_sampling() -> None:
    env = (TMAX / "env.cwdfw.sh").read_text()
    run_state = (TMAX / "run_state.sh").read_text()
    watcher = (TMAX / "watch_training.sh").read_text()
    shared_run = (SHARED / "run.sh").read_text()

    filter_path = "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    assert (
        'export POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0}"'
        in env
    )
    assert 'export POLAR_EARLY_STOP_GRACE_SESSIONS="${POLAR_EARLY_STOP_GRACE_SESSIONS:-0}"' in env
    assert filter_path in env
    assert "TMAX_DYNAMIC_SAMPLING_FILTER_PATH" in run_state
    assert 'TMAX_DYNAMIC_SAMPLING_FILTER_PATH=""' in watcher
    assert "TRAIN_DYNAMIC_SAMPLING_ARGS=()" in shared_run
    assert "--dynamic-sampling-filter-path" in shared_run
    assert '"${TMAX_DYNAMIC_SAMPLING_FILTER_PATH}"' in shared_run
    assert '"${TRAIN_DYNAMIC_SAMPLING_ARGS[@]}"' in shared_run


def test_tmax_allows_persisted_empty_dynamic_filter_for_legacy_run(tmp_path: Path) -> None:
    env = clean_env(tmp_path)
    env["TMAX_DYNAMIC_SAMPLING_FILTER_PATH"] = ""

    result = run_bash(
        f"source {TMAX / 'env.cwdfw.sh'} >/dev/null; "
        "printf '<%s>' \"$TMAX_DYNAMIC_SAMPLING_FILTER_PATH\"",
        env=env,
    )

    assert result.stdout == "<>"


@pytest.mark.parametrize(
    ("coefficient", "expected"),
    [
        ("0", ""),
        ("0.001", "--use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl"),
    ],
)
def test_shared_launcher_only_enables_reference_kl_for_positive_coefficient(
    coefficient: str,
    expected: str,
) -> None:
    shared_run = (SHARED / "run.sh").read_text()
    start = shared_run.index("KL_LOSS_ARGS=()")
    end = shared_run.index("\nSGLANG_REASONING_ARGS=()", start)
    snippet = shared_run[start:end]
    script = f"""
set -euo pipefail
PYTHON_BIN={sys.executable}
KL_LOSS_COEF={coefficient}
{snippet}
printf '%s' "${{KL_LOSS_ARGS[*]}}"
"""

    assert run_bash(script).stdout == expected


def test_tmax_context_parallel_override_is_validated_and_forwarded(tmp_path: Path) -> None:
    env = clean_env(tmp_path)
    env.update(CONTEXT_PARALLEL_SIZE="2", MAX_TOKENS_PER_GPU="33792")
    script = f"""
source {TMAX / "env.cwdfw.sh"} >/dev/null
actor_gpus=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
actor_dp=$((actor_gpus / (ACTOR_TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE)))
printf '%s' "$CONTEXT_PARALLEL_SIZE|$actor_dp|$((MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE))"
"""

    assert run_bash(script, env=env).stdout == "2|1|67584"
    assert '--context-parallel-size "$CONTEXT_PARALLEL_SIZE"' in (SHARED / "run.sh").read_text()


def test_tmax_rejects_microbatch_cap_below_full_pack_by_default(tmp_path: Path) -> None:
    env = clean_env(tmp_path)
    env["MAX_TOKENS_PER_GPU"] = "24576"

    result = run_bash(f"source {TMAX / 'env.cwdfw.sh'}", env=env, check=False)

    assert result.returncode != 0
    assert "trainer token capacity MAX_TOKENS_PER_GPU*CP=24576" in result.stderr
    assert "TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP=1" in result.stderr


def test_tmax_explicitly_allows_aggregate_microbatch_cap_below_pack(tmp_path: Path) -> None:
    env = clean_env(tmp_path)
    env.update(
        MAX_TOKENS_PER_GPU="24576",
        TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP="1",
    )

    result = run_bash(
        f"source {TMAX / 'env.cwdfw.sh'} >/dev/null; "
        'printf \'%s|%s\' "$MAX_TOKENS_PER_GPU" '
        '"$TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP"',
        env=env,
    )

    assert result.stdout == "24576|1"
    assert "TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP" in (TMAX / "run_state.sh").read_text()


def test_tmax_enables_true_trainer_fp32_lm_head_by_default(tmp_path: Path) -> None:
    env = clean_env(tmp_path)
    result = run_bash(
        f"source {TMAX / 'env.cwdfw.sh'} >/dev/null; printf '%s' \"$TMAX_ENABLE_FP32_LM_HEAD\"",
        env=env,
    )

    assert result.stdout == "1"
    shared_run = (SHARED / "run.sh").read_text()
    assert "TRAINER_FP32_LM_HEAD_ARGS=(--enable-fp32-lm-head)" in shared_run
    assert '"${TRAINER_FP32_LM_HEAD_ARGS[@]}"' in shared_run
    assert "TMAX_ENABLE_FP32_LM_HEAD" in (TMAX / "run_state.sh").read_text()


def test_tmax_cpu_optimizer_offload_is_explicit_opt_in(tmp_path: Path) -> None:
    env = clean_env(tmp_path)
    result = run_bash(
        f"source {TMAX / 'env.cwdfw.sh'} >/dev/null; "
        'printf \'%s\' "$TMAX_OPTIMIZER_CPU_OFFLOAD"',
        env=env,
    )

    assert result.stdout == "0"

    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()
    invalid_env = clean_env(invalid_root)
    invalid_env["TMAX_OPTIMIZER_CPU_OFFLOAD"] = "true"
    invalid = run_bash(f"source {TMAX / 'env.cwdfw.sh'}", env=invalid_env, check=False)
    assert invalid.returncode != 0
    assert "TMAX_OPTIMIZER_CPU_OFFLOAD must be 0 or 1" in invalid.stderr


def test_tmax_context_parallel_rejects_invalid_actor_topology(tmp_path: Path) -> None:
    env = clean_env(tmp_path)
    env.update(CONTEXT_PARALLEL_SIZE="3")

    result = run_bash(f"source {TMAX / 'env.cwdfw.sh'}", env=env, check=False)

    assert result.returncode != 0
    assert "must be divisible by tensor*context parallel size 4*3=12" in result.stderr


def test_tmax_enables_interleaved_qwen35_reasoning_end_to_end() -> None:
    env = (TMAX / "env.cwdfw.sh").read_text()
    polar_config = (TMAX / "polar_config.yaml").read_text()
    run_state = (TMAX / "run_state.sh").read_text()
    shared_run = (SHARED / "run.sh").read_text()

    assert 'POLAR_AGENT_ENABLE_THINKING="${POLAR_AGENT_ENABLE_THINKING:-true}"' in env
    assert 'POLAR_AGENT_TEMPERATURE="${POLAR_AGENT_TEMPERATURE:-1.0}"' in env
    assert 'POLAR_AGENT_MAX_TOKENS="${POLAR_AGENT_MAX_TOKENS:-16384}"' in env
    assert 'SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-qwen3}"' in env
    assert "enable_thinking: ${POLAR_AGENT_ENABLE_THINKING}" in polar_config
    assert "temperature: ${POLAR_AGENT_TEMPERATURE}" in polar_config
    assert "max_tokens: ${POLAR_AGENT_MAX_TOKENS}" in polar_config
    assert "POLAR_AGENT_MAX_TOKENS POLAR_AGENT_ENABLE_THINKING" in run_state
    assert "SGLANG_REASONING_PARSER" in run_state
    assert '--sglang-reasoning-parser "${SGLANG_REASONING_PARSER}"' in shared_run


def test_tmax_enables_sglang_fp32_lm_head_projection() -> None:
    env = (TMAX / "env.cwdfw.sh").read_text()
    run_state = (TMAX / "run_state.sh").read_text()
    shared_run = (SHARED / "run.sh").read_text()

    assert 'SGLANG_ENABLE_FP32_LM_HEAD="${SGLANG_ENABLE_FP32_LM_HEAD:-1}"' in env
    assert "SGLANG_ENABLE_FP32_LM_HEAD" in run_state
    assert "SGLANG_FP32_LM_HEAD_ARGS=(--sglang-enable-fp32-lm-head)" in shared_run
    assert '"${SGLANG_FP32_LM_HEAD_ARGS[@]}"' in shared_run


def test_tmax_enables_token_level_loss_and_exports_it_to_slurm() -> None:
    env = (TMAX / "env.cwdfw.sh").read_text()
    run_state = (TMAX / "run_state.sh").read_text()
    shared_run = (SHARED / "run.sh").read_text()
    shared_submit = (SHARED / "submit_slurm.sh").read_text()

    assert 'CALCULATE_PER_TOKEN_LOSS="${CALCULATE_PER_TOKEN_LOSS:-1}"' in env
    assert "CALCULATE_PER_TOKEN_LOSS" in run_state
    assert "PER_TOKEN_LOSS_ARGS=(--calculate-per-token-loss)" in shared_run
    assert '"${PER_TOKEN_LOSS_ARGS[@]}"' in shared_run
    assert "CALCULATE_PER_TOKEN_LOSS" in shared_submit


def test_tmax_enables_and_persists_logprob_fail_fast_guard() -> None:
    env = (TMAX / "env.cwdfw.sh").read_text()
    run_state = (TMAX / "run_state.sh").read_text()
    shared_run = (SHARED / "run.sh").read_text()
    shared_submit = (SHARED / "submit_slurm.sh").read_text()

    assert "MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF:-1.0" in env
    assert "MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF" in run_state
    assert "--max-train-rollout-logprob-abs-diff" in shared_run
    assert "MAX_TRAIN_ROLLOUT_LOGPROB_ABS_DIFF" in shared_submit


def test_tmax_qwen35_9b_model_selection_is_explicit_and_persisted() -> None:
    env = (TMAX / "env.cwdfw.sh").read_text()
    model_args = (TMAX / "model_args.sh").read_text()
    topology = (TMAX / "topology.yaml").read_text()
    run_state = (TMAX / "run_state.sh").read_text()
    shared_run = (SHARED / "run.sh").read_text()
    shared_submit = (SHARED / "submit_slurm.sh").read_text()
    converter = (SHARED / "convert_weights.sh").read_text()
    watcher = (TMAX / "watch_training.sh").read_text()

    assert 'Qwen3.5-9B}"' in env
    assert "Qwen3.5-9B_torch_dist" in env
    assert 'MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${SCRIPT_DIR}/model_args.sh}"' in env
    assert (
        'FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${POLAR_DATA_ROOT}/kernel-cache/'
        in env
    )
    assert 'POLAR_AGENT_MODEL_NAME="${POLAR_AGENT_MODEL_NAME:-Qwen/Qwen3.5-9B}"' in env
    assert "--hidden-size 4096" in model_args
    assert "--ffn-hidden-size 12288" in model_args
    assert "--untie-embeddings-and-output-weights" in model_args
    assert "--linear-attention-freq 4" in model_args
    assert "--no-rope-fusion" in model_args
    assert "--no-persist-layer-norm" in model_args
    assert "--use-gated-attention" not in model_args
    # Qwen3.5 checkpoints store zero-centred RMSNorm gamma deltas.
    assert "--apply-layernorm-1p" in model_args
    assert "model_served: ${POLAR_AGENT_MODEL_NAME}" in topology
    assert "HF_CHECKPOINT REF_LOAD TORCH_DIST_DIR MODEL_ARGS_FILE" in run_state
    assert "ACTOR_TENSOR_MODEL_PARALLEL_SIZE" in run_state
    assert "SGLANG_MEM_FRACTION_STATIC" in run_state
    assert "FLASHINFER_WORKSPACE_BASE" in run_state
    assert 'source "${MODEL_ARGS_FILE}"' in shared_run
    assert 'polar_validate_model_args "${MODEL_ARGS[@]}"' in shared_run
    assert '--tensor-model-parallel-size "${ACTOR_TENSOR_MODEL_PARALLEL_SIZE:-2}"' in shared_run
    assert '--sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.8}"' in shared_run
    assert 'source "${MODEL_ARGS_FILE}"' in converter
    assert 'source "${SCRIPT_DIR}/launcher_utils.sh"' in converter
    assert 'polar_validate_model_args "${MODEL_ARGS[@]}"' in converter
    assert "REF_LOAD|TORCH_DIST_DIR|MODEL_ARGS_FILE" in shared_submit
    assert "FLASHINFER_*" in shared_submit
    assert "legacy run state: preserving Qwen3.5-4B model lineage" in watcher
    assert '! tmax_run_state_has_export "${TMAX_RUN_STATE_FILE}" HF_CHECKPOINT' in watcher


def test_shared_qwen35_model_args_preflight_requires_layernorm_1p() -> None:
    script = f"""
source {SHARED / "launcher_utils.sh"}
qwen35_without_fix=(--spec slime_plugins.models.qwen3_5 get_qwen3_5_spec)
if polar_validate_model_args "${{qwen35_without_fix[@]}}"; then
    printf 'unexpected-qwen35-success\n'
    exit 1
fi
polar_validate_model_args \
    --spec slime_plugins.models.qwen3_5 get_qwen3_5_spec \
    --apply-layernorm-1p
polar_validate_model_args \
    --spec=slime_plugins.models.qwen3_5 get_qwen3_5_spec \
    --apply-layernorm-1p
polar_validate_model_args \
    --spec another.models.module get_model_spec
printf 'validated\n'
"""

    result = run_bash(script)
    assert result.stdout.splitlines() == ["validated"]
    assert "unexpected-qwen35-success" not in result.stdout
    assert (
        "MODEL_ARGS selecting slime_plugins.models.qwen3_5 must include --apply-layernorm-1p"
    ) in result.stderr


@pytest.mark.parametrize(
    ("nodes", "gpus_per_node", "wall_time", "expected_partition"),
    [
        (1, 8, "2:00:00", "interactive,batch_short,backfill,batch"),
        (2, 8, "2:00:00", "interactive,batch_short,backfill,batch"),
        (2, 8, "2:00:01", "interactive,backfill,batch"),
        (2, 9, "2:00:00", "batch_short"),
        (3, 8, "2:00:00", "batch_short"),
        (4, 8, "2:00:00", "batch_short"),
        (4, 8, "2:00:01", "backfill,batch"),
        (5, 8, "2:00:00", "backfill,batch"),
    ],
)
def test_tmax_selects_partitions_by_resources_and_wall_time(
    tmp_path: Path,
    nodes: int,
    gpus_per_node: int,
    wall_time: str,
    expected_partition: str,
) -> None:
    env = clean_env(tmp_path)
    env.update(
        NUM_NODES=str(nodes),
        SLURM_GPUS=str(gpus_per_node),
        WALL_TIME=wall_time,
        TMAX_MIN_WALL_TIME=wall_time,
        TMAX_REQUIRE_FULL_GPU_ALLOCATION="0",
        ACTOR_NUM_NODES="1",
        ACTOR_NUM_GPUS_PER_NODE="2",
        ROLLOUT_NUM_GPUS="2",
    )
    script = f"""
source {TMAX / "env.cwdfw.sh"} >/dev/null
printf '%s|%s|%s\n' "$WALL_TIME" "$TMAX_MIN_WALL_TIME" "$PARTITION"
"""

    result = run_bash(script, env=env)

    assert result.stdout.strip() == f"{wall_time}|{wall_time}|{expected_partition}"


def test_tmax_default_allocation_uses_four_node_backfill_batch(tmp_path: Path) -> None:
    env = clean_env(tmp_path)
    script = f"""
source {TMAX / "env.cwdfw.sh"} >/dev/null
printf '%s|%s|%s|%s\n' "$NUM_NODES" "$WALL_TIME" "$TMAX_MIN_WALL_TIME" "$PARTITION"
"""

    result = run_bash(script, env=env)

    assert result.stdout.strip() == "4|4:00:00|4:00:00|backfill,batch"


def test_tmax_partition_selection_uses_wall_time_after_minimum_raise(
    tmp_path: Path,
) -> None:
    env = clean_env(tmp_path)
    env.update(WALL_TIME="1:00:00", TMAX_MIN_WALL_TIME="3:00:00")
    script = f"""
source {TMAX / "env.cwdfw.sh"} >/dev/null
printf '%s|%s\n' "$WALL_TIME" "$PARTITION"
"""

    result = run_bash(script, env=env)

    assert result.stdout.strip() == "3:00:00|backfill,batch"
    assert "raising wall time 1:00:00 -> 3:00:00" in result.stderr


@pytest.mark.parametrize(
    ("partition", "nodes", "gpus_per_node", "wall_time", "expected_error"),
    [
        (
            "interactive,backfill,batch",
            3,
            8,
            "2:00:00",
            "interactive, which is limited to at most 2 nodes and 16 total GPUs",
        ),
        (
            "backfill,interactive,batch",
            2,
            9,
            "2:00:00",
            "interactive, which is limited to at most 2 nodes and 16 total GPUs",
        ),
        (
            "batch_short,backfill,batch",
            5,
            8,
            "2:00:00",
            "batch_short, which is limited to at most 4 nodes and 2:00:00",
        ),
        (
            "backfill,batch_short,batch",
            4,
            8,
            "2:00:01",
            "batch_short, which is limited to at most 4 nodes and 2:00:00",
        ),
    ],
)
def test_tmax_rejects_explicit_partition_qos_violations(
    tmp_path: Path,
    partition: str,
    nodes: int,
    gpus_per_node: int,
    wall_time: str,
    expected_error: str,
) -> None:
    env = clean_env(tmp_path)
    env.update(
        PARTITION=partition,
        NUM_NODES=str(nodes),
        SLURM_GPUS=str(gpus_per_node),
        WALL_TIME=wall_time,
    )

    result = run_bash(f"source {TMAX / 'env.cwdfw.sh'}", env=env, check=False)

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_tmax_partition_validation_matches_complete_comma_tokens(tmp_path: Path) -> None:
    env = clean_env(tmp_path)
    env.update(
        PARTITION="interactive_debug,batch_short_debug,backfill",
        NUM_NODES="5",
        WALL_TIME="4:00:00",
        TMAX_REQUIRE_FULL_GPU_ALLOCATION="0",
        ACTOR_NUM_NODES="1",
        ACTOR_NUM_GPUS_PER_NODE="2",
        ROLLOUT_NUM_GPUS="2",
    )

    result = run_bash(
        f"source {TMAX / 'env.cwdfw.sh'} >/dev/null; printf '%s' \"$PARTITION\"",
        env=env,
    )

    assert result.stdout == "interactive_debug,batch_short_debug,backfill"


def test_shared_launcher_cleanup_never_waits_unbounded() -> None:
    shared_run = (SHARED / "run.sh").read_text()
    cleanup_start = shared_run.index("cleanup() {")
    cleanup_end = shared_run.index("\ntrap cleanup EXIT", cleanup_start)
    cleanup = shared_run[cleanup_start:cleanup_end]

    assert cleanup.index('touch "$RUN_DONE_FILE"') < cleanup.index("polar_stop_ray_bounded")
    assert "polar_terminate_pids_bounded" in cleanup
    assert "wait 2>/dev/null || true" not in cleanup
    assert "kill -KILL" in shared_run
    assert "PROCESS_GROUPS=()" in shared_run
    assert 'setsid "${monitor_command[@]}" &' in shared_run
    assert "polar_append_descendant_pids" in cleanup
    assert "polar_terminate_process_groups_bounded" in cleanup
    assert "POLAR_RAY_STOP_TIMEOUT_SECONDS" in shared_run
    assert "POLAR_BACKGROUND_SHUTDOWN_GRACE_SECONDS" in shared_run
    assert "POLAR_GATEWAY_SHUTDOWN_GRACE_SECONDS" in shared_run
    assert "polar_shutdown_gateway_bounded" in cleanup
    assert 'exit "${final_status}"' in cleanup


def test_shared_launcher_bounded_cleanup_kills_stuck_child() -> None:
    shared_run = (SHARED / "run.sh").read_text()
    helpers_start = shared_run.index("polar_pid_is_active() {")
    helpers_end = shared_run.index("\ncleanup() {", helpers_start)
    helpers = shared_run[helpers_start:helpers_end]
    script = f"""
set -euo pipefail
{helpers}
trap '' TERM
while true; do sleep 1; done &
pid=$!
trap - TERM
pids=("$pid")
for _ in $(seq 1 50); do
    pids=("$pid")
    polar_append_descendant_pids "$pid" pids
    [ "${{#pids[@]}}" -gt 1 ] && break
    sleep 0.02
done
[ "${{#pids[@]}}" -gt 1 ]
POLAR_BACKGROUND_KILL_GRACE_SECONDS=1 \
    polar_terminate_pids_bounded 0 "${{pids[@]}}"
for child in "${{pids[@]}}"; do
    if polar_pid_is_active "$child"; then
        echo "child $child remained alive" >&2
        exit 1
    fi
done
"""

    result = run_bash(script)

    assert result.returncode == 0


@pytest.mark.parametrize(
    ("term_action", "expected"),
    [
        ("exit 0", "clean"),
        ("exit 7", "failed"),
        ("", "killed"),
    ],
)
def test_shared_launcher_gateway_shutdown_propagates_failure(
    term_action: str,
    expected: str,
) -> None:
    shared_run = (SHARED / "run.sh").read_text()
    helpers_start = shared_run.index("polar_pid_is_active() {")
    helpers_end = shared_run.index("\ncleanup() {", helpers_start)
    helpers = shared_run[helpers_start:helpers_end]
    if term_action:
        grace = 2
    else:
        grace = 0
    script = f"""
set -euo pipefail
{helpers}
ready=$(mktemp)
action={json.dumps(term_action)}
bash -c 'trap "$2" TERM; printf ready >"$1"; while true; do read -r -t 1 _ || true; done' _ "$ready" "$action" &
pid=$!
for _ in $(seq 1 100); do
    [ -s "$ready" ] && break
    sleep 0.01
done
[ -s "$ready" ]
if POLAR_BACKGROUND_KILL_GRACE_SECONDS=1 \
    polar_shutdown_gateway_bounded "$pid" {grace}; then
    printf clean
else
    if [ {grace} -eq 0 ]; then printf killed; else printf failed; fi
fi
"""

    result = run_bash(script)

    assert result.stdout == expected


def test_shared_launcher_gateway_shutdown_does_not_repeat_pre_sent_term() -> None:
    shared_run = (SHARED / "run.sh").read_text()
    helpers_start = shared_run.index("polar_pid_is_active() {")
    helpers_end = shared_run.index("\ncleanup() {", helpers_start)
    helpers = shared_run[helpers_start:helpers_end]
    script = f"""
set -euo pipefail
{helpers}
ready=$(mktemp)
python -c 'import pathlib, signal, sys, time; signal.signal(signal.SIGTERM, lambda *_: (signal.signal(signal.SIGTERM, signal.SIG_DFL), time.sleep(0.2), sys.exit(0))); pathlib.Path(sys.argv[1]).write_text("ready"); signal.pause()' "$ready" &
pid=$!
for _ in $(seq 1 100); do
    [ -s "$ready" ] && break
    sleep 0.01
done
[ -s "$ready" ]
kill -TERM "$pid"
POLAR_BACKGROUND_KILL_GRACE_SECONDS=1 \
    polar_shutdown_gateway_bounded "$pid" 2 1
"""

    result = run_bash(script)

    assert result.returncode == 0, result.stderr


def test_spilot_gateway_shutdown_budget_covers_http_and_runtime_teardown() -> None:
    launcher = (SHARED / "run.sh").read_text()

    assert 'POLAR_GATEWAY_SHUTDOWN_GRACE_SECONDS:-150' in launcher
    assert 'POLAR_GATEWAY_SHUTDOWN_GRACE_SECONDS}" -lt 150' in launcher
    assert "60s HTTP drain + 60s runtime proof + 30s margin" in launcher


def test_shared_launcher_bounded_cleanup_kills_stuck_process_group() -> None:
    if shutil.which("setsid") is None:
        pytest.skip("setsid is required for the process-group cleanup path")
    shared_run = (SHARED / "run.sh").read_text()
    helpers_start = shared_run.index("polar_pid_is_active() {")
    helpers_end = shared_run.index("\ncleanup() {", helpers_start)
    helpers = shared_run[helpers_start:helpers_end]
    script = f"""
set -euo pipefail
{helpers}
setsid bash -c 'trap "" TERM; while true; do sleep 1; done' &
leader=$!
for _ in $(seq 1 50); do
    polar_process_group_is_active "$leader" && break
    sleep 0.02
done
polar_process_group_is_active "$leader"
POLAR_BACKGROUND_KILL_GRACE_SECONDS=1 \
    polar_terminate_process_groups_bounded 0 "$leader"
wait "$leader" 2>/dev/null || true
if polar_process_group_is_active "$leader"; then
    echo "process group remained alive" >&2
    exit 1
fi
"""

    result = run_bash(script)

    assert result.returncode == 0


def test_tmax_submit_numeric_seed_still_requires_explicit_matching_data(
    tmp_path: Path,
) -> None:
    env = tmax_submit_env(tmp_path, load_pointer="47")

    result = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "requires explicit TMAX_TRAIN_DATA and TMAX_PREPARE_DATA=0" in result.stderr
    assert not (tmp_path / "data" / "runs" / "fresh-release" / "tmax-train.jsonl").exists()


def test_tmax_submit_numeric_seed_requires_complete_model_checkpoint(
    tmp_path: Path,
) -> None:
    env = tmax_numeric_seed_env(tmp_path)

    missing = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert missing.returncode == 1
    assert "has no matching model checkpoint directory" in missing.stderr
    assert "iter_0000047" in missing.stderr
    assert "Dry run only" not in missing.stdout

    model_dir = Path(env["LOAD_DIR"]) / "iter_0000047"
    model_dir.mkdir()
    (model_dir / "common.pt").write_bytes(b"model")
    broken_metadata = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert broken_metadata.returncode == 1
    assert "model checkpoint 47 is incomplete" in broken_metadata.stderr
    assert ".metadata" in broken_metadata.stderr

    write_distcp_metadata(model_dir, "__0_0.distcp")
    (model_dir / "common.pt").write_bytes(b"")
    broken_common = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert broken_common.returncode == 1
    assert "model checkpoint 47 is incomplete" in broken_common.stderr
    assert "common.pt" in broken_common.stderr


def test_numbered_checkpoint_helper_requires_complete_nonempty_distcp_shard_set(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    model_dir = checkpoint / "iter_0000047"
    model_dir.mkdir(parents=True)
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("47\n")
    (model_dir / "common.pt").write_bytes(b"common")
    write_distcp_metadata(model_dir, "__0_0.distcp")
    state = checkpoint / "rollout" / "global_dataset_state_dict_47.pt"
    state.parent.mkdir()
    state.write_bytes(b"state")
    env = os.environ.copy()
    env["CHECKPOINT_ROOT"] = str(checkpoint)
    script = (
        f"source {TMAX / 'lifecycle.sh'}; "
        'tmax_validate_numbered_checkpoint "$CHECKPOINT_ROOT" "ERROR: test"'
    )

    missing = run_bash(script, env=env, check=False)
    assert missing.returncode != 0
    assert "metadata references missing shard(s): __0_0.distcp" in missing.stderr

    shard = model_dir / "__0_0.distcp"
    shard.touch()
    empty = run_bash(script, env=env, check=False)
    assert empty.returncode != 0
    assert "empty or invalid shard(s): __0_0.distcp" in empty.stderr

    shard.write_bytes(b"x")
    truncated = run_bash(script, env=env, check=False)
    assert truncated.returncode != 0
    assert "expected 7 bytes, found 1" in truncated.stderr

    shard.write_bytes(b"weights")
    write_distcp_metadata(model_dir, "__0_0.distcp", "__1_0.distcp")
    partial = run_bash(script, env=env, check=False)
    assert partial.returncode != 0
    assert "metadata references missing shard(s): __1_0.distcp" in partial.stderr

    (model_dir / "__1_0.distcp").write_bytes(b"weights")
    ready = run_bash(script, env=env)
    assert ready.stdout == "47\n"


def test_tmax_submit_numeric_seed_requires_matching_rollout_state(
    tmp_path: Path,
) -> None:
    env = tmax_numeric_seed_env(tmp_path)
    model_dir = Path(env["LOAD_DIR"]) / "iter_0000047"
    model_dir.mkdir()
    (model_dir / "common.pt").write_bytes(b"model")
    write_distcp_metadata(model_dir, "__0_0.distcp")
    (model_dir / "__0_0.distcp").write_bytes(b"weights")

    missing = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert missing.returncode == 1
    assert "has no matching rollout state" in missing.stderr
    assert "global_dataset_state_dict_47.pt" in missing.stderr
    assert "Dry run only" not in missing.stdout

    rollout_state = Path(env["LOAD_DIR"]) / "rollout" / "global_dataset_state_dict_47.pt"
    rollout_state.parent.mkdir()
    rollout_state.write_bytes(b"state")
    accepted = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "Dry run only; sbatch wrapper:" in accepted.stdout


def test_tmax_submit_validates_sifs_in_reused_prompt_file(tmp_path: Path) -> None:
    env = tmax_submit_env(tmp_path, load_pointer="release")
    task_dir = Path(env["TMAX_DATASET_DIR"]) / "task_a"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "task.toml").write_text("")
    (task_dir / "instruction.md").write_text("fix it")
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\n")
    train_data = tmp_path / "reused.jsonl"
    train_data.write_text(
        json.dumps(
            {
                "prompt": [],
                "metadata": {
                    "task_name": "task_a",
                    "sif_path": str(Path(env["APPTAINER_IMAGE_DIR"]) / "task_a.sif"),
                },
            }
        )
        + "\n"
    )
    env.update(
        TMAX_SIF_PYTHON_BIN=sys.executable,
        TMAX_TRAIN_DATA=str(train_data),
        TMAX_PREPARE_DATA="0",
        TMAX_MAX_TASKS="1",
        TMAX_EVAL_ENABLED="0",
    )

    result = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "Missing or empty SIF(s) for 1/1" in result.stderr
    assert "Dry run only" not in result.stdout


def test_container_entrypoint_uses_job_private_cwd_and_rollout_dir(tmp_path: Path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    write_command(venv / "bin" / "python3", "exit 0\n")
    train = tmp_path / "train.sh"
    write_command(
        train,
        'printf \'%s\\n\' "$PWD" "$HOME" "$RUN_DIR" "$POLAR_ROLLOUT_SAVE_DIR" "$SGLANG_ROUTER_PORT"\n',
    )
    job_id = str(9_000_000 + os.getpid())
    cache_root = Path("/tmp") / f"polar-cache-{job_id}"
    env = clean_env(tmp_path)
    env.update(
        POLAR_TRAIN_PROJECT_ROOT=str(ROOT),
        POLAR_TRAIN_RUN_SCRIPT=str(train),
        POLR_TRAIN_VENV=str(venv),
        RUN_ID="isolated-run",
        SLURM_JOB_ID=job_id,
    )

    try:
        result = subprocess.run(
            ["bash", str(SHARED / "run_in_container.sh")],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)

    assert result.returncode == 0, result.stderr
    pwd, home, run_dir, rollout_dir, router_port = result.stdout.splitlines()[-5:]
    expected_run_dir = tmp_path / "data" / "runs" / "isolated-run" / f"job-{job_id}"
    assert pwd == home == str(cache_root / "home")
    assert Path(pwd) != tmp_path / "data"
    assert run_dir == str(expected_run_dir)
    assert rollout_dir == str(expected_run_dir / "rollout_results")
    assert Path(rollout_dir).is_absolute()
    assert router_port == "8680"


def test_container_entrypoint_rejects_stale_runtime_timing_markers(tmp_path: Path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    write_command(venv / "bin" / "python3", "exit 0\n")
    train = tmp_path / "show-markers.sh"
    write_command(
        train,
        "printf '%s\\n' \"$SLIME_CONTAINER_ENTRY_UNIX_NS\" "
        '"$SLIME_SLURM_BATCH_START_UNIX_NS" '
        '"${SLIME_JOB_SCRIPT_START_UNIX_NS-unset}" '
        '"${SLIME_RAY_READY_UNIX_NS-unset}"\n',
    )
    env_file = tmp_path / "stale-env.sh"
    env_file.write_text(
        "export SLIME_CONTAINER_ENTRY_UNIX_NS=1\n"
        "export SLIME_SLURM_BATCH_START_UNIX_NS=2\n"
        "export SLIME_JOB_SCRIPT_START_UNIX_NS=3\n"
        "export SLIME_RAY_READY_UNIX_NS=4\n"
    )
    job_id = str(9_100_000 + os.getpid())
    cache_root = Path("/tmp") / f"polar-cache-{job_id}"
    env = clean_env(tmp_path)
    env.update(
        POLAR_TRAIN_PROJECT_ROOT=str(ROOT),
        POLAR_TRAIN_RUN_SCRIPT=str(train),
        POLAR_TRAIN_ENV_FILE=str(env_file),
        POLR_TRAIN_VENV=str(venv),
        RUN_ID="marker-test",
        SLURM_JOB_ID=job_id,
        SLIME_SLURM_BATCH_START_UNIX_NS="222",
    )

    try:
        result = subprocess.run(
            ["bash", str(SHARED / "run_in_container.sh")],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)

    assert result.returncode == 0, result.stderr
    container_entry, batch_start, job_start, ray_ready = result.stdout.splitlines()[-4:]
    assert int(container_entry) > 4
    assert batch_start == "222"
    assert job_start == "unset"
    assert ray_ready == "unset"


def test_topology_templates_render_an_explicit_absolute_rollout_dir(tmp_path: Path):
    rollout_dir = tmp_path / "job" / "rollout_results"
    replacements = {
        "POLAR_ROLLOUT_HOST": "127.0.0.1",
        "POLAR_ROLLOUT_PORT": "18080",
        "POLAR_ROLLOUT_URL": "http://127.0.0.1:18080",
        "POLAR_ROLLOUT_SAVE_DIR": str(rollout_dir),
        "POLAR_GATEWAY_HOST": "127.0.0.1",
        "POLAR_GATEWAY_PORT": "18100",
        "POLAR_GATEWAY_URL": "http://127.0.0.1:18100",
        "POLAR_GATEWAY_MAX_INIT_WORKERS": "4",
        "POLAR_GATEWAY_MAX_RUN_WORKERS": "8",
        "POLAR_GATEWAY_MAX_POSTRUN_WORKERS": "4",
        "POLAR_COMPLETION_QUEUE_SIZE": "32768",
        "POLAR_COMPLETION_WRITE_WORKERS": "16",
        "POLAR_GATEWAY_COMPLETION_QUEUE_SIZE": "32768",
        "POLAR_GATEWAY_COMPLETION_WRITE_WORKERS": "16",
        "POLAR_COMPLETION_BATCH_SIZE": "16",
        "POLAR_COMPLETION_WRITE_MAX_ATTEMPTS": "3",
        "POLAR_COMPLETION_RETRY_BACKOFF_SECONDS": "0.1",
        "SGLANG_ROUTER_BASE_URL": "http://127.0.0.1:9000",
    }

    for template in (SHARED / "topology.yaml", TMAX / "topology.yaml"):
        text = template.read_text()
        assert 'save_dir: "${POLAR_ROLLOUT_SAVE_DIR}"' in text
        for name, value in replacements.items():
            text = text.replace("${" + name + "}", value)
        rendered = tmp_path / f"{template.parent.name}-topology.yaml"
        rendered.write_text(text)
        topology = TopologyConfig.load(rendered)
        assert topology.rollout.save_dir == str(rollout_dir)
        assert Path(topology.rollout.save_dir).is_absolute()

    assert '"POLAR_ROLLOUT_SAVE_DIR",' in (SHARED / "run.sh").read_text()


def test_allocation_port_and_load_seed_precedence(tmp_path: Path):
    save = tmp_path / "save"
    seed = tmp_path / "seed"
    ref = tmp_path / "ref"
    for directory in (save, seed, ref):
        directory.mkdir()
    script = f"""
source {SHARED / "launcher_utils.sh"}
export SLIME_EPHEMERAL_PORT_LOWER_BOUND=9000
p1=$(polar_rollout_base_port_for_allocation 13200027)
p2=$(polar_rollout_base_port_for_allocation 13200028)
p_array_1=$(polar_rollout_base_port_for_allocation 13200027_1)
p_array_2=$(polar_rollout_base_port_for_allocation 13200027_2)
printf '%s %s %s %s %s\n' "$p1" "$p2" "$p_array_1" "$p_array_2" "$(polar_select_load_dir {save} {ref} {seed})"
printf 1 > {save / "latest_checkpointed_iteration.txt"}
polar_select_load_dir {save} {ref} {seed}
"""
    lines = run_bash(script).stdout.splitlines()
    p1, p2, p_array_1, p_array_2, chosen = lines[0].split()
    assert p1 != p2
    assert p_array_1 != p_array_2
    for port in (p1, p2, p_array_1, p_array_2):
        base = int(port)
        assert 2048 <= base <= 8128
        assert (base - 2048) % 320 == 0
        assert base + 320 <= 9000
    assert chosen == str(seed)
    assert lines[1] == str(save)


def test_allocation_ports_read_kernel_ephemeral_range_and_keep_twenty_slots(tmp_path: Path):
    port_range = tmp_path / "ip_local_port_range"
    port_range.write_text("9000 65000\n")
    script = f"""
source {SHARED / "launcher_utils.sh"}
unset SLIME_EPHEMERAL_PORT_LOWER_BOUND
export SLIME_IP_LOCAL_PORT_RANGE_PATH={port_range}
printf '%s\n' "$(polar_ephemeral_port_lower_bound)"
for allocation in $(seq 0 19); do
    polar_rollout_base_port_for_allocation "$allocation"
done
"""

    lines = run_bash(script).stdout.splitlines()
    assert lines[0] == "9000"
    ports = [int(value) for value in lines[1:]]
    assert ports == [2048 + slot * 320 for slot in range(20)]
    assert max(ports) + 320 == 8448 < 9000


def test_rollout_port_validation_rejects_ephemeral_overlap():
    script = f"""
source {SHARED / "launcher_utils.sh"}
export SLIME_EPHEMERAL_PORT_LOWER_BOUND=9000
polar_validate_rollout_base_port 2048
polar_validate_rollout_base_port 8680
polar_validate_rollout_base_port 8681
"""

    result = run_bash(script, check=False)
    assert result.returncode != 0
    assert "320-port block" in result.stderr
    assert "base <= 8680" in result.stderr


def test_rollout_port_configure_propagates_unsafe_override_failure():
    script = f"""
source {SHARED / "launcher_utils.sh"}
export SLIME_EPHEMERAL_PORT_LOWER_BOUND=9000
export SLIME_ROLLOUT_BASE_PORT=20000
if polar_configure_rollout_base_port; then
    printf 'unexpected-success\n'
    exit 0
fi
exit 7
"""

    result = run_bash(script, check=False)
    assert result.returncode == 7
    assert "SLIME_ROLLOUT_BASE_PORT=20000 is unsafe" in result.stderr
    assert "unexpected-success" not in result.stdout
    assert "[launcher]" not in result.stdout


def test_sglang_router_default_is_outside_ephemeral_and_engine_blocks():
    script = f"""
source {SHARED / "launcher_utils.sh"}
export SLIME_EPHEMERAL_PORT_LOWER_BOUND=9000
unset SGLANG_ROUTER_PORT
for allocation in $(seq 0 19); do
    export SLIME_ROLLOUT_BASE_PORT=$(polar_rollout_base_port_for_allocation "$allocation")
    polar_validate_sglang_router_port 8680
done
export SLIME_ROLLOUT_BASE_PORT=6208
polar_configure_sglang_router_port
printf '%s\n' "$SGLANG_ROUTER_PORT"
"""

    lines = run_bash(script).stdout.splitlines()
    assert lines[-2:] == [
        "[launcher] SGLang router port=8680 (safe default)",
        "8680",
    ]


def test_sglang_router_rejects_ephemeral_engine_and_ray_port_conflicts():
    script = f"""
source {SHARED / "launcher_utils.sh"}
export SLIME_EPHEMERAL_PORT_LOWER_BOUND=9000
export SLIME_ROLLOUT_BASE_PORT=6208
for port in 9000 6208 6300 6379 8265 1023 not-a-port; do
    if polar_validate_sglang_router_port "$port"; then
        printf 'unexpected-success:%s\n' "$port"
        exit 1
    fi
done
polar_validate_sglang_router_port 8680
"""

    result = run_bash(script)
    assert "unexpected-success" not in result.stdout
    assert "non-ephemeral port" in result.stderr
    assert "overlaps the reserved rollout-engine port block" in result.stderr
    assert "conflicts with a fixed Ray control-plane port" in result.stderr
    assert "must be an integer" in result.stderr


def test_release_seed_detection_distinguishes_training_checkpoint(tmp_path: Path):
    release = tmp_path / "release-seed"
    numeric = tmp_path / "training-checkpoint"
    missing = tmp_path / "missing-tracker"
    for directory in (release, numeric, missing):
        directory.mkdir()
    (release / "latest_checkpointed_iteration.txt").write_text("release\n")
    (numeric / "latest_checkpointed_iteration.txt").write_text("17\n")

    script = f"""
source {SHARED / "launcher_utils.sh"}
if polar_checkpoint_is_release_seed {release}; then printf 'release\\n'; fi
if polar_checkpoint_is_release_seed {numeric}; then printf 'bad-numeric\\n'; else printf 'numeric\\n'; fi
if polar_checkpoint_is_release_seed {missing}; then printf 'bad-missing\\n'; else printf 'missing\\n'; fi
"""

    assert run_bash(script).stdout.splitlines() == ["release", "numeric", "missing"]


def test_tmax_topology_rejects_unassigned_slurm_gpus(tmp_path: Path):
    env = clean_env(tmp_path)
    env.update(
        NUM_NODES="2",
        SLURM_GPUS="8",
        RAY_NUM_GPUS_PER_NODE="8",
        ACTOR_NUM_NODES="1",
        ACTOR_NUM_GPUS_PER_NODE="8",
        ROLLOUT_NUM_GPUS="4",
    )

    result = run_bash(f"source {TMAX / 'env.cwdfw.sh'}", env=env, check=False)

    assert result.returncode != 0
    assert "must use all 16 allocated GPUs" in result.stderr


def test_tmax_cross_node_tensor_parallel_requires_explicit_opt_in(tmp_path: Path):
    base = clean_env(tmp_path)
    base.update(
        NUM_NODES="8",
        SLURM_GPUS="1",
        RAY_NUM_GPUS_PER_NODE="1",
        ACTOR_NUM_NODES="4",
        ACTOR_NUM_GPUS_PER_NODE="1",
        ACTOR_TENSOR_MODEL_PARALLEL_SIZE="4",
        ROLLOUT_NUM_GPUS="4",
        ROLLOUT_BATCH_SIZE="1",
        N_SAMPLES_PER_PROMPT="8",
        POLAR_FULLY_ASYNC="false",
    )

    rejected = run_bash(
        f"source {TMAX / 'env.cwdfw.sh'}",
        env=base,
        check=False,
    )
    assert rejected.returncode != 0
    assert "TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL=1" in rejected.stderr

    allowed_env = base.copy()
    allowed_env["TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL"] = "1"
    allowed = run_bash(
        f"source {TMAX / 'env.cwdfw.sh'} >/dev/null; printf ok",
        env=allowed_env,
    )
    assert allowed.stdout == "ok"
    shared_run = (SHARED / "run.sh").read_text()
    assert '--num-gpus-per-node "$RAY_NUM_GPUS_PER_NODE"' in shared_run


def test_tmax_four_node_defaults_keep_both_gpu_pools_fed(tmp_path: Path):
    env = clean_env(tmp_path)
    script = f"""
source {TMAX / "env.cwdfw.sh"} >/dev/null
actor_gpus=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
actor_dp=$((actor_gpus / (ACTOR_TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE)))
global_batch=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT))
active_sessions=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT * POLAR_MAX_ASYNC_LEVEL))
printf '%s\n' \
  "$NUM_NODES|$actor_gpus|$ROLLOUT_NUM_GPUS" \
  "$ACTOR_TENSOR_MODEL_PARALLEL_SIZE|$CONTEXT_PARALLEL_SIZE|$ROLLOUT_NUM_GPUS_PER_ENGINE|$MAX_TOKENS_PER_GPU|$SGLANG_MEM_FRACTION_STATIC" \
  "$ROLLOUT_BATCH_SIZE|$N_SAMPLES_PER_PROMPT|$NUM_STEPS_PER_ROLLOUT|$global_batch" \
  "$actor_dp|$((global_batch / actor_dp))" \
  "$POLAR_MAX_ASYNC_LEVEL|$active_sessions|$((active_sessions / ROLLOUT_NUM_GPUS))" \
  "$POLAR_MAX_INIT_WORKERS|$POLAR_MAX_RUN_WORKERS|$POLAR_MAX_POSTRUN_WORKERS" \
  "$TMAX_TRAIN_AGENT_TIMEOUT_SECONDS|$TMAX_TRAIN_TASK_TIMEOUT_RESERVE_SECONDS|$POLAR_REQUEST_TIMEOUT|$POLAR_TASK_TIMEOUT_FLOOR_SECONDS|$TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU" \
  "$POLAR_APPTAINER_PERSISTENT_BROKER|$POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY|$POLAR_APPTAINER_BROKER_START_CONCURRENCY|$POLAR_APPTAINER_BROKER_START_PER_IMAGE_CONCURRENCY" \
  "$SLURM_STEP_CPUS_PER_TASK|$TMAX_MAX_TASKS|$TMAX_ONLY_READY" \
  "$TMAX_EVAL_ENABLED|$TMAX_EVAL_START_INDEX|$TMAX_EVAL_MAX_TASKS|$TMAX_EVAL_SAMPLES_PER_PROMPT|$TMAX_EVAL_MIN_VALID_SAMPLES" \
  "$ROLLOUT_MAX_PROMPT_LEN|$ROLLOUT_MAX_RESPONSE_LEN|$TMAX_MAX_TOTAL_RESPONSE_LEN|$TMAX_TRAIN_PACK_LENGTH|$SEQ_LENGTH" \
  "$POLAR_MULTI_GATEWAY|$POLAR_APPTAINER_DIRECT_EXEC_RETRIES|$CALCULATE_PER_TOKEN_LOSS"
"""

    lines = run_bash(script, env=env).stdout.splitlines()

    assert lines == [
        "4|8|24",
        "4|1|1|67584|0.7",
        "8|32|1|256",
        "2|128",
        "4|1024|42",
        "48|384|192",
        "1200|600|3600|1800|8",
        "0|2|32|2",
        "120|14501|0",
        "1|14501|100|1|100",
        "2048|16384|65536|67584|67584",
        "0|3|1",
    ]


def test_tmax_rejects_undersized_postrun_pool(tmp_path: Path) -> None:
    env = clean_env(tmp_path)
    env["POLAR_MAX_POSTRUN_WORKERS"] = "191"

    result = run_bash(f"source {TMAX / 'env.cwdfw.sh'}", env=env, check=False)

    assert result.returncode != 0
    assert "POLAR_MAX_POSTRUN_WORKERS=191 is too small" in result.stderr
    assert "require at least 192" in result.stderr


def test_tmax_rejects_task_floor_without_training_timeout_reserve(tmp_path: Path) -> None:
    env = clean_env(tmp_path)
    env["POLAR_TASK_TIMEOUT_FLOOR_SECONDS"] = "1799"

    result = run_bash(f"source {TMAX / 'env.cwdfw.sh'}", env=env, check=False)

    assert result.returncode != 0
    assert "must cover TMAX_TRAIN_AGENT_TIMEOUT_SECONDS" in result.stderr


def test_tmax_rejects_request_timeout_below_task_floor(tmp_path: Path) -> None:
    env = clean_env(tmp_path)
    env.update(
        POLAR_REQUEST_TIMEOUT="1799",
        POLAR_TASK_TIMEOUT_FLOOR_SECONDS="1800",
    )

    result = run_bash(f"source {TMAX / 'env.cwdfw.sh'}", env=env, check=False)

    assert result.returncode != 0
    assert "POLAR_REQUEST_TIMEOUT must be at least" in result.stderr


def test_tmax_restores_single_gateway_direct_exec_by_default(tmp_path: Path):
    env = clean_env(tmp_path)

    result = run_bash(
        f"source {TMAX / 'env.cwdfw.sh'} >/dev/null; printf '%s' \"$POLAR_MULTI_GATEWAY\"",
        env=env,
    )

    assert result.stdout == "0"


def test_shared_launcher_splits_gateway_capacity_without_amplifying_totals():
    launcher = (SHARED / "run.sh").read_text()
    start = launcher.index('case "${POLAR_MULTI_GATEWAY:-0}"')
    end = launcher.index("\nPOLAR_ROLLOUT_LOCAL_URL=", start)
    capacity_block = launcher[start:end]
    script = f"""
set -euo pipefail
POLAR_MULTI_GATEWAY=1
RAY_NUM_NODES=4
POLAR_MAX_INIT_WORKERS=48
POLAR_MAX_RUN_WORKERS=384
POLAR_MAX_POSTRUN_WORKERS=192
POLAR_COMPLETION_QUEUE_SIZE=32768
POLAR_COMPLETION_WRITE_WORKERS=16
{capacity_block}
printf '%s\n' \
  "$POLAR_GATEWAY_COUNT" \
  "$POLAR_GATEWAY_MAX_INIT_WORKERS/$POLAR_GATEWAY_MAX_RUN_WORKERS/$POLAR_GATEWAY_MAX_POSTRUN_WORKERS" \
  "$POLAR_GATEWAY_COMPLETION_QUEUE_SIZE/$POLAR_GATEWAY_COMPLETION_WRITE_WORKERS"
"""

    lines = run_bash(script).stdout.splitlines()
    assert lines[-3:] == ["4", "12/96/48", "8192/4"]


def test_spilot_canonical_episode_admission_defaults_and_timeout_formula(
    tmp_path: Path,
) -> None:
    env = clean_env(tmp_path)
    script = f"""
source {SPILOT / "experiment_defaults.sh"}
printf '%s\n' \
  "$NUM_NODES|$POLAR_MAX_INIT_WORKERS|$POLAR_MAX_RUN_WORKERS|$TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU" \
  "$SPILOT_EPISODE_ADMISSION_ENABLED|$SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS|$SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT" \
  "$SPILOT_QWEN_MAX_ACTIVE_EPISODES|$SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES|$SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY|$SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES" \
  "$SPILOT_GPT_MAX_ACTIVE_EPISODES|$SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES|$SPILOT_GPT_GATEWAY_MAX_CONCURRENCY|$SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES" \
  "$TMAX_TRAIN_AGENT_TIMEOUT_SECONDS|$POLAR_TASK_TIMEOUT_FLOOR_SECONDS|$POLAR_REQUEST_TIMEOUT" \
  "$PARTITION|$WALL_TIME|$TMAX_MIN_WALL_TIME|$TMAX_GRACEFUL_EXIT_BUFFER_SECONDS|$TMAX_MIN_GRACEFUL_EXIT_BUFFER_SECONDS|$SAVE_INTERVAL"
"""

    assert run_bash(script, env=env).stdout.splitlines() == [
        "8|96|576|12",
        "true|14400|8",
        "8|1|1|8",
        "32|4|4|32",
        "17700|36600|37200",
        "backfill|2-00:00:00|2-00:00:00|43200|43200|5",
    ]


def test_spilot_defaults_reject_nondivisible_aggregate_episode_cap(
    tmp_path: Path,
) -> None:
    env = clean_env(tmp_path)
    env["SPILOT_QWEN_MAX_ACTIVE_EPISODES"] = "9"

    result = run_bash(
        f"source {SPILOT / 'experiment_defaults.sh'}",
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert "must be positive and divisible" in result.stderr


@pytest.mark.parametrize(
    "name",
    [
        "POLAR_APPTAINER_NO_MOUNT_HOSTFS",
        "POLAR_APPTAINER_NO_MOUNT_TMP",
        "POLAR_APPTAINER_ISOLATE_PID",
        "POLAR_APPTAINER_ISOLATE_IPC",
        "POLAR_APPTAINER_CLEANENV",
    ],
)
def test_spilot_defaults_reject_isolation_opt_out(
    tmp_path: Path,
    name: str,
) -> None:
    env = clean_env(tmp_path)
    env[name] = "0"

    result = run_bash(
        f"source {SPILOT / 'experiment_defaults.sh'}",
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert f"formal SPilot requires {name}=1" in result.stderr


def test_spilot_launcher_rejects_timeout_without_ready_wave_reserve(
    tmp_path: Path,
) -> None:
    env = clean_env(tmp_path)
    env["POLAR_TASK_TIMEOUT_FLOOR_SECONDS"] = "36599"
    result = run_bash(
        f"source {SPILOT / 'experiment_defaults.sh'}; "
        f"source {TMAX / 'env.cwdfw.sh'}",
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "SPilot admission timeout formula requires "
        "agent/task/request=17700/36600/37200"
    ) in result.stderr


def test_spilot_launcher_rejects_wall_time_shorter_than_request_and_grace(
    tmp_path: Path,
) -> None:
    env = clean_env(tmp_path)
    env.update(WALL_TIME="4:00:00", TMAX_MIN_WALL_TIME="4:00:00")
    result = run_bash(
        f"source {SPILOT / 'experiment_defaults.sh'}; "
        f"source {TMAX / 'env.cwdfw.sh'}",
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "SPilot WALL_TIME must cover POLAR_REQUEST_TIMEOUT + "
        "TMAX_GRACEFUL_EXIT_BUFFER_SECONDS (14400 < 80400)"
    ) in result.stderr


def test_spilot_launcher_rejects_grace_buffer_without_checkpoint_reserve(
    tmp_path: Path,
) -> None:
    env = clean_env(tmp_path)
    env.update(
        TMAX_GRACEFUL_EXIT_BUFFER_SECONDS="40799",
        TMAX_MIN_GRACEFUL_EXIT_BUFFER_SECONDS="40799",
    )
    result = run_bash(
        f"source {SPILOT / 'experiment_defaults.sh'}; "
        f"source {TMAX / 'env.cwdfw.sh'}",
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "SPilot TMAX_GRACEFUL_EXIT_BUFFER_SECONDS must cover "
        "POLAR_REQUEST_TIMEOUT + 3600 (40799 < 40800)"
    ) in result.stderr


def test_shared_launcher_preserves_spilot_effective_caps_across_eight_gateways():
    launcher = (SHARED / "run.sh").read_text()
    start = launcher.index('case "${POLAR_MULTI_GATEWAY:-0}"')
    end = launcher.index("\nPOLAR_ROLLOUT_LOCAL_URL=", start)
    capacity_block = launcher[start:end]
    script = f"""
set -euo pipefail
TMAX_AGENT_HARNESS=spilot_router
POLAR_MULTI_GATEWAY=1
RAY_NUM_NODES=8
POLAR_MAX_INIT_WORKERS=96
POLAR_MAX_RUN_WORKERS=576
POLAR_MAX_POSTRUN_WORKERS=384
POLAR_COMPLETION_QUEUE_SIZE=32768
POLAR_COMPLETION_WRITE_WORKERS=16
SPILOT_EPISODE_ADMISSION_ENABLED=true
SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS=14400
SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT=8
SPILOT_QWEN_MAX_ACTIVE_EPISODES=8
SPILOT_GPT_MAX_ACTIVE_EPISODES=32
SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES=1
SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES=4
SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY=1
SPILOT_GPT_GATEWAY_MAX_CONCURRENCY=4
SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES=8
SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES=32
{capacity_block}
printf '%s\n' \
  "$POLAR_GATEWAY_COUNT" \
  "$POLAR_GATEWAY_MAX_INIT_WORKERS/$POLAR_GATEWAY_MAX_RUN_WORKERS" \
  "$SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES/$SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES"
"""

    lines = run_bash(script).stdout.splitlines()

    assert lines[-3:] == ["8", "12/72", "1/4"]


def test_legacy_disabled_spilot_resume_keeps_single_gateway_topology():
    launcher = (SHARED / "run.sh").read_text()
    start = launcher.index('case "${POLAR_MULTI_GATEWAY:-0}"')
    end = launcher.index("\nPOLAR_ROLLOUT_LOCAL_URL=", start)
    capacity_block = launcher[start:end]
    script = f"""
set -euo pipefail
TMAX_AGENT_HARNESS=spilot_router
POLAR_MULTI_GATEWAY=0
RAY_NUM_NODES=8
POLAR_MAX_INIT_WORKERS=96
POLAR_MAX_RUN_WORKERS=576
POLAR_MAX_POSTRUN_WORKERS=384
POLAR_COMPLETION_QUEUE_SIZE=32768
POLAR_COMPLETION_WRITE_WORKERS=16
SPILOT_EPISODE_ADMISSION_ENABLED=false
# Legacy migration records the historical node count, but disabled admission
# has no per-gateway lease split and must not force multi-gateway execution.
SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT=8
{capacity_block}
printf '%s|%s/%s/%s\n' \
  "$POLAR_GATEWAY_COUNT" \
  "$POLAR_GATEWAY_MAX_INIT_WORKERS" \
  "$POLAR_GATEWAY_MAX_RUN_WORKERS" \
  "$POLAR_GATEWAY_MAX_POSTRUN_WORKERS"
"""

    assert run_bash(script).stdout.splitlines()[-1] == "1|96/576/384"


@pytest.mark.parametrize("admission_enabled", ["true", "false"])
def test_spilot_gateway_health_monitor_writes_sanitized_job_marker(
    tmp_path: Path,
    admission_enabled: str,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(
        bin_dir / "curl",
        """
output=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o) output="$2"; shift 2 ;;
        -w) shift 2 ;;
        *) shift ;;
    esac
done
printf '%s' "$FAKE_HEALTH_PAYLOAD" > "$output"
printf '%s' "$FAKE_HEALTH_STATUS"
""",
    )
    state = tmp_path / "current_run.env"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {
        "status": "error",
        "node_id": "slurm-rank-2",
        "model_pool_episode_admission_health": {
            "healthy": False,
            "fatal_retained": True,
            "retained_session_ids": ["secret-session-a", "secret-session-b"],
        },
    }
    env = clean_env(tmp_path)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        TMAX_AGENT_HARNESS="spilot_router",
        SPILOT_EPISODE_ADMISSION_ENABLED=admission_enabled,
        POLAR_GATEWAY_LOCAL_URL="http://127.0.0.1:18100",
        RUN_DIR=str(run_dir),
        PYTHON_BIN=sys.executable,
        TMAX_RUN_STATE_FILE=str(state),
        SLURM_JOB_ID="123",
        RAY_NODE_RANK="2",
        FAKE_HEALTH_STATUS="503",
        FAKE_HEALTH_PAYLOAD=json.dumps(payload),
    )
    script = f"""
source {SHARED / 'launcher_utils.sh'}
if polar_check_spilot_admission_health; then
    printf healthy
else
    printf 'fatal:%s' "$?"
fi
"""

    result = run_bash(script, env=env)

    assert result.stdout == "fatal:70"
    assert "[spilot admission fatal] job=123 rank=2 retained_count=2" in result.stderr
    marker = Path(f"{state}.job-123.admission-fatal.rank-2.json")
    marker_payload = json.loads(marker.read_text())
    assert marker_payload == {
        "schema_version": 1,
        "kind": "model_pool_episode_admission_fatal_retained",
        "job_id": "123",
        "rank": 2,
        "node_id": "slurm-rank-2",
        "observed_at": marker_payload["observed_at"],
        "retained_count": 2,
    }
    assert "secret-session" not in marker.read_text()

    env.update(FAKE_HEALTH_STATUS="200", FAKE_HEALTH_PAYLOAD='{"status":"ok"}')
    env["SLURM_JOB_ID"] = "124"
    healthy = run_bash(script, env=env)
    assert healthy.stdout == "healthy"
    assert not Path(f"{state}.job-124.admission-fatal.rank-2.json").exists()


def test_shared_launcher_checks_spilot_gateway_health_in_head_and_worker_loops() -> None:
    launcher = (SHARED / "run.sh").read_text()

    assert launcher.count("if ! polar_check_spilot_admission_health; then") == 2
    assert "polar/spilot_router/admission_fatal_job_count_total=" in launcher


def test_spilot_templates_render_both_alias_caps_and_runner_admission(
    tmp_path: Path,
) -> None:
    topology_values = {
        "POLAR_ROLLOUT_HOST": "0.0.0.0",
        "POLAR_ROLLOUT_PORT": "18080",
        "POLAR_ROLLOUT_URL": "http://rollout.test:18080",
        "POLAR_ROLLOUT_SAVE_DIR": "/tmp/rollout",
        "POLAR_GATEWAY_COMPLETION_QUEUE_SIZE": "4096",
        "POLAR_GATEWAY_COMPLETION_WRITE_WORKERS": "2",
        "POLAR_COMPLETION_BATCH_SIZE": "16",
        "POLAR_COMPLETION_WRITE_MAX_ATTEMPTS": "3",
        "POLAR_COMPLETION_RETRY_BACKOFF_SECONDS": "0.1",
        "POLAR_GATEWAY_HOST": "0.0.0.0",
        "POLAR_GATEWAY_PORT": "18100",
        "POLAR_GATEWAY_URL": "http://gateway.test:18100",
        "POLAR_GATEWAY_MAX_INIT_WORKERS": "6",
        "POLAR_GATEWAY_MAX_RUN_WORKERS": "6",
        "POLAR_GATEWAY_MAX_POSTRUN_WORKERS": "48",
        "POLAR_AGENT_MODEL_NAME": "Qwen/Qwen3.5-9B",
        "SGLANG_ROUTER_BASE_URL": "http://router.test:30000",
        "POLAR_MODEL_POOL_BASE_URL": "https://integrate.api.nvidia.com/v1",
        "SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY": "1",
        "SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES": "1",
        "SPILOT_GPT_GATEWAY_MAX_CONCURRENCY": "4",
        "SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES": "4",
    }
    topology_text = (SPILOT / "topology.yaml").read_text()
    for name, value in topology_values.items():
        topology_text = topology_text.replace("${" + name + "}", value)
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(topology_text)

    topology = TopologyConfig.load(topology_path)
    candidates = topology.gateway.nodes[0].model_pool
    assert [
        (candidate.max_concurrency, candidate.max_active_episodes)
        for candidate in candidates
    ] == [(1, 1), (4, 4)]

    config_text = (SPILOT / "polar_config.yaml").read_text()
    config_names = set(
        re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", config_text)
    )
    config_values = {name: "1" for name in config_names}
    config_values.update(
        {
            "POLAR_FULLY_ASYNC": "true",
            "POLAR_SANDBOX_NETWORK": "none",
            "POLAR_AGENT_ENABLE_THINKING": "true",
            "SPILOT_EPISODE_ADMISSION_ENABLED": "true",
            "SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS": "14400",
            "POLAR_AGENT_RUNTIME_VOLUME": "",
            "POLAR_INTERNET_RUNTIME_VOLUME": "",
        }
    )
    for name, value in config_values.items():
        config_text = config_text.replace("${" + name + "}", value)
    config = yaml.safe_load(config_text)
    settings = config["polar_task_template"]["agent"]["settings"]
    assert settings["pool_episode_admission_enabled"] is True
    assert settings["pool_episode_admission_wait_budget_seconds"] == 14400
    assert settings["total_timeout_seconds"] == 3000

    disabled_topology_values = dict(topology_values)
    disabled_topology_values.update(
        {
            "SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY": "32",
            "SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES": "null",
            "SPILOT_GPT_GATEWAY_MAX_CONCURRENCY": "32",
            "SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES": "null",
        }
    )
    disabled_topology_text = (SPILOT / "topology.yaml").read_text()
    for name, value in disabled_topology_values.items():
        disabled_topology_text = disabled_topology_text.replace(
            "${" + name + "}", value
        )
    disabled_topology_path = tmp_path / "topology-disabled.yaml"
    disabled_topology_path.write_text(disabled_topology_text)
    disabled_candidates = TopologyConfig.load(
        disabled_topology_path
    ).gateway.nodes[0].model_pool
    assert [
        (candidate.max_concurrency, candidate.max_active_episodes)
        for candidate in disabled_candidates
    ] == [(32, None), (32, None)]

    disabled_config_text = (SPILOT / "polar_config.yaml").read_text()
    disabled_config_values = dict(config_values)
    disabled_config_values.update(
        {
            "SPILOT_EPISODE_ADMISSION_ENABLED": "false",
            "SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS": "0",
        }
    )
    for name, value in disabled_config_values.items():
        disabled_config_text = disabled_config_text.replace(
            "${" + name + "}", value
        )
    disabled_config = yaml.safe_load(disabled_config_text)
    disabled_settings = disabled_config["polar_task_template"]["agent"]["settings"]
    assert disabled_settings["pool_episode_admission_enabled"] is False
    assert disabled_settings["pool_episode_admission_wait_budget_seconds"] == 0


def test_shared_launcher_rejects_per_gateway_capacity_amplification():
    launcher = (SHARED / "run.sh").read_text()
    start = launcher.index('case "${POLAR_MULTI_GATEWAY:-0}"')
    end = launcher.index("\nPOLAR_ROLLOUT_LOCAL_URL=", start)
    capacity_block = launcher[start:end]
    script = f"""
set -euo pipefail
POLAR_MULTI_GATEWAY=1
RAY_NUM_NODES=4
POLAR_MAX_INIT_WORKERS=48
POLAR_MAX_RUN_WORKERS=384
POLAR_MAX_POSTRUN_WORKERS=192
POLAR_COMPLETION_QUEUE_SIZE=32768
POLAR_COMPLETION_WRITE_WORKERS=16
POLAR_GATEWAY_MAX_RUN_WORKERS=97
{capacity_block}
"""

    result = run_bash(script, check=False)
    assert result.returncode != 0
    assert "exceeds aggregate POLAR_MAX_RUN_WORKERS=384" in result.stderr


def test_shared_launcher_starts_one_local_gateway_per_rank_before_ray_submit():
    launcher = (SHARED / "run.sh").read_text()

    assert 'node_index="${SLURM_NODEID:-${RAY_NODE_RANK}}"' in launcher
    assert 'node_id="slurm-rank-${node_index}"' in launcher
    assert 'touch "${RAY_READY_DIR}/gateway_ready_rank_${RAY_NODE_RANK}"' in launcher
    assert 'wait_gateway_fleet_ready "${POLAR_GATEWAY_COUNT}"' in launcher
    assert '"${POLAR_ROLLOUT_LOCAL_URL}/nodes"' in launcher
    assert "wait_for_run_done_with_sidecars" in launcher
    assert "render_gateway_topology.py" in launcher
    assert launcher.index('wait_gateway_fleet_ready "${POLAR_GATEWAY_COUNT}"') < launcher.index(
        "# ── Step 3: Slime"
    )


def test_shared_launcher_bypasses_proxy_for_every_slurm_gateway(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(
        bin_dir / "scontrol",
        """
test "${1:-}" = show
test "${2:-}" = hostnames
printf '%s\n' gw-1 gw-2 gw-3 gw-4
""",
    )
    write_command(
        bin_dir / "getent",
        """
test "${1:-}" = ahostsv4
host="${2:?missing host}"
printf '10.8.0.%s STREAM %s\n' "${host##*-}" "$host"
""",
    )
    launcher = (SHARED / "run.sh").read_text()
    functions = launcher[
        launcher.index("resolve_host_ip() {") : launcher.index("\n# ── External deps")
    ]
    script = f"""
set -euo pipefail
PYTHON_BIN={sys.executable}
SLURM_JOB_NODELIST='gw-[1-4]'
{functions}
slurm_allocation_proxy_bypass_hosts
"""
    env = clean_env(tmp_path)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    result = run_bash(script, env=env)

    assert result.stdout == ("gw-1,10.8.0.1,gw-2,10.8.0.2,gw-3,10.8.0.3,gw-4,10.8.0.4")
    assert 'CLUSTER_PROXY_BYPASS_HOSTS="$(slurm_allocation_proxy_bypass_hosts)"' in launcher
    assert 'PROXY_BYPASS_HOSTS="${PROXY_BYPASS_HOSTS},${CLUSTER_PROXY_BYPASS_HOSTS}"' in launcher
    assert 'export no_proxy="${no_proxy:+${no_proxy},}${PROXY_BYPASS_HOSTS}"' in launcher
    assert 'export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${PROXY_BYPASS_HOSTS}"' in launcher


@pytest.mark.parametrize(
    ("nodelist_var", "nodelist", "expected"),
    [
        (
            "SLURM_JOB_NODELIST",
            "pool0-[01281,01409]",
            ["pool0-01281", "pool0-01409"],
        ),
        (
            "SLURM_NODELIST",
            "rack[0-1]-node[01-03:2],standalone",
            [
                "rack0-node01",
                "rack0-node03",
                "rack1-node01",
                "rack1-node03",
                "standalone",
            ],
        ),
    ],
)
def test_shared_launcher_expands_slurm_hosts_without_scontrol(
    tmp_path: Path,
    nodelist_var: str,
    nodelist: str,
    expected: list[str],
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    launcher = (SHARED / "run.sh").read_text()
    functions = launcher[
        launcher.index("resolve_host_ip() {") : launcher.index("\n# ── External deps")
    ]
    script = f"""
set -euo pipefail
PYTHON_BIN={sys.executable}
{nodelist_var}={nodelist!r}
{functions}
slurm_allocation_hosts
"""
    env = clean_env(tmp_path)
    # Exclude the cluster Slurm module directory while retaining bash itself;
    # this proves discovery is independent of the host's scontrol installation.
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"

    result = run_bash(script, env=env)

    assert result.stdout.splitlines() == expected


def test_multi_gateway_topology_has_no_hard_scontrol_dependency():
    launcher = (SHARED / "run.sh").read_text()

    assert "multi-gateway topology requires scontrol" not in launcher
    assert 'if ! _polar_gateway_host_output="$(slurm_allocation_hosts)"' in launcher
    assert "multi-gateway topology requires SLURM_JOB_NODELIST or SLURM_NODELIST" in launcher
    assert "Slurm hostlist expanded to ${#_polar_gateway_hosts[@]} hosts" in launcher


def test_slurm_host_discovery_reports_missing_allocation_environment(tmp_path: Path):
    launcher = (SHARED / "run.sh").read_text()
    functions = launcher[
        launcher.index("resolve_host_ip() {") : launcher.index("\n# ── External deps")
    ]
    result = run_bash(
        f"PYTHON_BIN={sys.executable}\n{functions}\nslurm_allocation_hosts",
        env=clean_env(tmp_path),
        check=False,
    )

    assert result.returncode != 0
    assert "neither SLURM_JOB_NODELIST nor SLURM_NODELIST is set" in result.stderr


def test_shared_readiness_directory_exists_before_worker_gateway_start():
    launcher = (SHARED / "run.sh").read_text()
    initialized = launcher.index('touch "${RAY_READY_DIR}/initialized"')
    worker_wait = launcher.index('[ -f "${RAY_READY_DIR}/initialized" ] && break')
    gateway_call = launcher.index("\n    start_polar_gateway\n", worker_wait)

    assert initialized < worker_wait < gateway_call


def test_tmax_external_eval_defaults_pin_terminal_bench_2_0(tmp_path: Path):
    env = clean_env(tmp_path)
    script = f"""
source {TMAX / "env.cwdfw.sh"} >/dev/null
printf '%s\n' \
  "$TMAX_EXTERNAL_EVAL_DATASET_NAME" \
  "$TMAX_HARBOR_EVAL_TASKS_DIR" \
  "$TMAX_HARBOR_EVAL_IMAGE_DIR" \
  "$TMAX_HARBOR_EVAL_REVISION"
"""

    lines = run_bash(script, env=env).stdout.splitlines()

    data_root = tmp_path / "data" / "benchmarks" / "terminal-bench-2.0"
    assert lines == [
        "terminal_bench_2_0",
        str(data_root / "terminal-bench"),
        str(data_root / "enroot-images"),
        "terminal-bench@2.0@69671fbaac6d67a7ef0dfec016cc38a64ef7a77c",
    ]


def test_tmax_fully_async_rejects_sparse_rollout_capacity(tmp_path: Path):
    env = clean_env(tmp_path)
    env.update(
        ROLLOUT_BATCH_SIZE="4",
        TMAX_MIN_ASYNC_LEVEL="1",
        POLAR_MAX_ASYNC_LEVEL="1",
    )

    result = run_bash(f"source {TMAX / 'env.cwdfw.sh'}", env=env, check=False)

    assert result.returncode != 0
    assert "fully-async capacity 128 sessions is too small" in result.stderr
    assert "for 24 rollout GPUs; require at least 384" in result.stderr


def test_tmax_fixed_eval_rejects_overlap_with_training_window(tmp_path: Path):
    env = clean_env(tmp_path)
    env.update(TMAX_EVAL_SOURCE="tmax", TMAX_EVAL_START_INDEX="999")

    result = run_bash(f"source {TMAX / 'env.cwdfw.sh'}", env=env, check=False)

    assert result.returncode != 0
    assert "before the training window ends at 14501" in result.stderr


def test_tmax_eval_source_selects_source_specific_defaults(tmp_path: Path):
    env = clean_env(tmp_path)
    env.update(TMAX_EVAL_SOURCE="tmax", TMAX_EVAL_START_INDEX="1000")
    script = f"""
source {TMAX / "env.cwdfw.sh"} >/dev/null
printf '%s|%s|%s|%s\n' \
  "$TMAX_EVAL_DATASET_NAME" "$TMAX_EVAL_MAX_TASKS" \
  "$TMAX_EVAL_SAMPLES_PER_PROMPT" "$TMAX_EVAL_MIN_VALID_SAMPLES"
"""

    result = run_bash(script, env=env)

    assert result.stdout.strip() == "tmax_holdout|100|1|100"


def test_external_eval_allows_unbounded_training_window(tmp_path: Path):
    env = clean_env(tmp_path)
    env.update(TMAX_MAX_TASKS="-1", TMAX_EVAL_SOURCE="harbor")

    result = run_bash(f"source {TMAX / 'env.cwdfw.sh'} >/dev/null; printf ok", env=env)

    assert result.stdout == "ok"


def test_tmax_fixed_eval_rejects_minimum_above_configured_sample_count(
    tmp_path: Path,
):
    env = clean_env(tmp_path)
    env.update(TMAX_EVAL_MIN_VALID_SAMPLES="101")

    result = run_bash(f"source {TMAX / 'env.cwdfw.sh'}", env=env, check=False)

    assert result.returncode != 0
    assert "TMAX_EVAL_MIN_VALID_SAMPLES=101" in result.stderr
    assert "exceeds configured eval sample count 100" in result.stderr


def test_tmax_submit_rejects_train_eval_canonical_path_alias(tmp_path: Path):
    env = tmax_submit_env(tmp_path, load_pointer="release")
    shared_data = tmp_path / "shared.jsonl"
    env.update(
        TMAX_TRAIN_DATA=str(shared_data),
        TMAX_EVAL_DATA=str(shared_data.parent / "." / shared_data.name),
    )

    result = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "must resolve to different files" in result.stderr
    assert not shared_data.exists()


def test_tmax_submit_rejects_eval_content_changed_from_existing_manifest(tmp_path: Path):
    env = tmax_submit_env(tmp_path, load_pointer="release")
    first = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert first.returncode == 0, first.stderr

    eval_data = tmp_path / "data" / "runs" / "fresh-release" / "terminal_bench_2_0-eval.jsonl"
    eval_data.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "changed after baseline"}],
                "metadata": {"task_name": "fake_external_eval_changed"},
            }
        )
        + "\n"
    )
    env.update(TMAX_PREPARE_DATA="0", TMAX_PREPARE_EVAL_DATA="0")

    second = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert second.returncode == 1
    assert "fixed external eval data changed from pinned sha256" in second.stderr


def test_tmax_submit_rejects_train_content_changed_from_existing_manifest(tmp_path: Path):
    env = tmax_submit_env(tmp_path, load_pointer="release")
    first = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert first.returncode == 0, first.stderr

    train_data = tmp_path / "data" / "runs" / "fresh-release" / "tmax-train.jsonl"
    train_data.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "mutated checkpoint prompt"}],
                "metadata": {"task_name": "fake_train"},
            }
        )
        + "\n"
    )
    env.update(TMAX_PREPARE_DATA="0", TMAX_PREPARE_EVAL_DATA="0")

    second = subprocess.run(
        ["bash", str(TMAX / "submit_slurm.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert second.returncode == 1
    assert "training data changed from pinned sha256" in second.stderr


def test_tmax_runtime_revalidates_and_exports_immutable_eval_manifest():
    tmax_run = (TMAX / "run.sh").read_text()
    shared_run = (SHARED / "run.sh").read_text()

    assert 'if [ "${SLURM_PROCID:-0}" = "0" ]; then' in tmax_run
    assert tmax_run.count("--validate-existing") >= 2
    assert "validate_data_integrity.py" in tmax_run
    assert "POLAR_EVAL_DATA_INTEGRITY_B64" in tmax_run
    assert "TMAX_TRAIN_DATA_SHA256" in tmax_run
    assert "unset TMAX_TRAIN_DATA_SHA256" in tmax_run
    assert '\\"POLAR_EVAL_DATA_INTEGRITY_B64\\"' in shared_run


def test_tmax_full_dataset_mode_can_disable_fixed_eval(tmp_path: Path):
    env = clean_env(tmp_path)
    env.update(TMAX_MAX_TASKS="-1", TMAX_EVAL_ENABLED="0")

    result = run_bash(f"source {TMAX / 'env.cwdfw.sh'}", env=env, check=False)

    assert result.returncode == 0


def test_tmax_rejects_invalid_local_spawn_concurrency(tmp_path: Path):
    env = clean_env(tmp_path)
    env.update(POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY="1025")

    result = run_bash(f"source {TMAX / 'env.cwdfw.sh'}", env=env, check=False)

    assert result.returncode != 0
    assert "POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY must be at most 1024" in result.stderr


def test_shared_launcher_balances_dynamic_training_batches():
    launcher = (SHARED / "run.sh").read_text()

    assert "--balance-data" in launcher
    assert "--balance-by-flops" not in launcher
    assert "8 × 32 trajectories which form one GBS=256 step" in launcher
    assert "24 × 8 trajectories split across three GBS=64 steps" not in launcher


def test_tmax_readme_checkpoint_example_matches_default_batch():
    readme = (TMAX / "README.md").read_text()

    assert "900 training prompts and batch eight, final rollout id 112" in readme
    assert "1,000 prompts and batch 16, iteration 62" not in readme


def test_shared_launcher_suppresses_sglang_access_log_by_default():
    launcher = (SHARED / "run.sh").read_text()

    assert '--sglang-log-level-http "${SGLANG_LOG_LEVEL_HTTP:-warning}"' in launcher
    assert (
        'SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-${_POLAR_SGLANG_ROUTER_PORT_DEFAULT}}"'
        in launcher
    )
    assert 'polar_validate_sglang_router_port "${SGLANG_ROUTER_PORT}"' in launcher


def test_shared_launcher_keeps_gateway_uds_available_for_host_and_none_network():
    launcher_path = SHARED / "run.sh"
    launcher = launcher_path.read_text()

    syntax = subprocess.run(
        ["bash", "-n", str(launcher_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert syntax.returncode == 0, syntax.stderr
    assert 'export POLAR_SANDBOX_NETWORK="${POLAR_SANDBOX_NETWORK:-host}"' in launcher
    assert "${POLAR_JOB_CACHE_ROOT:-/tmp/polar-${SLURM_JOB_ID:-$$}}/uds" in launcher
    assert "${POLAR_UDS_DIR}/gateway" in launcher
    assert "${POLAR_UDS_DIR}/proxy" in launcher
    assert (
        'export POLAR_SANDBOX_GATEWAY_UDS="${POLAR_SANDBOX_GATEWAY_UDS:-/polar/gateway/gateway.sock}"'
        in launcher
    )
    assert '"${POLAR_GATEWAY_UDS_SOCKET}=127.0.0.1:${POLAR_GATEWAY_PORT}"' in launcher
    assert 'mappings+=("${POLAR_PROXY_UDS_SOCKET}=${POLAR_PROXY_TCP_TARGET}")' in launcher
    assert 'POLAR_EFFECTIVE_HTTP_PROXY="${http_proxy:-}"' in launcher
    assert '"${PYTHON_BIN}" -m polar.runtime.uds_tunnel' in launcher
    assert '--ready-file "${POLAR_UDS_TUNNEL_READY_FILE}"' in launcher
    assert 'export POLAR_UDS_TUNNEL_BACKLOG="${POLAR_UDS_TUNNEL_BACKLOG:-4096}"' in launcher
    assert '--backlog "${POLAR_UDS_TUNNEL_BACKLOG}"' in launcher
    assert 'PIDS+=("${POLAR_UDS_TUNNEL_PID}")' in launcher
    assert (
        'wait_http_ok "Polar gateway ${node_id}" "${POLAR_GATEWAY_LOCAL_URL}/health" 60'
        in launcher
    )
    gateway_ready = launcher.index('wait_http_ok "Polar gateway ${node_id}"')
    tunnel_start = launcher.index("\n    start_sandbox_uds_tunnel\n", gateway_ready)
    slime_start = launcher.index("# ── Step 3", tunnel_start)
    assert gateway_ready < tunnel_start < slime_start
    tunnel_function = launcher.split("start_sandbox_uds_tunnel() {", 1)[1].split(
        "wait_ray_dashboard() {", 1
    )[0]
    assert "POLAR_SANDBOX_NETWORK" not in tunnel_function

    render_block = launcher.split("for name in (", 1)[1].split("):\n", 1)[0]
    for name in (
        "POLAR_SANDBOX_NETWORK",
        "POLAR_UDS_DIR",
        "POLAR_GATEWAY_UDS_DIR",
        "POLAR_PROXY_UDS_DIR",
        "POLAR_GATEWAY_UDS_SOCKET",
        "POLAR_PROXY_UDS_SOCKET",
        "POLAR_PROXY_UDS_ENABLED",
        "POLAR_UDS_TUNNEL_READY_FILE",
        "POLAR_UDS_TUNNEL_BACKLOG",
        "POLAR_SANDBOX_GATEWAY_UDS",
        "POLAR_SANDBOX_HTTP_PROXY_UDS",
        "POLAR_SANDBOX_HTTP_PROXY_PORT",
        "POLAR_INTERNET_RUNTIME_VOLUME",
    ):
        assert f'    "{name}",' in render_block

    assert 'export POLAR_SANDBOX_GATEWAY_UDS=""' not in launcher
    assert 'export POLAR_SANDBOX_HTTP_PROXY_UDS=""' in launcher
    task_template = (TMAX / "polar_config.yaml").read_text()
    assert "${POLAR_GATEWAY_UDS_DIR}:/polar/gateway:ro" in task_template
    assert "${POLAR_INTERNET_RUNTIME_VOLUME}" in task_template
    assert "${POLAR_UDS_DIR}:/polar/gateway:ro" not in task_template


def test_shared_launcher_renders_and_validates_early_stop_grace():
    launcher = (SHARED / "run.sh").read_text()
    render_block = launcher.split("for name in (", 1)[1].split("):\n", 1)[0]
    rendered_names = set(re.findall(r'^\s*"([A-Za-z_][A-Za-z0-9_]*)",$', render_block, re.M))
    template_names: set[str] = set()
    for template in (
        SHARED / "topology.yaml",
        SHARED / "polar_config.yaml",
        TMAX / "topology.yaml",
        TMAX / "polar_config.yaml",
        SPILOT / "topology.yaml",
        SPILOT / "polar_config.yaml",
    ):
        template_names.update(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", template.read_text()))

    assert "export POLAR_EARLY_STOP_GRACE_SESSIONS=" in launcher
    assert template_names <= rendered_names
    assert "unresolved template variable(s)" in launcher
    submitter = (SHARED / "submit_slurm.sh").read_text()
    for name in (
        "SPILOT_EPISODE_ADMISSION_ENABLED",
        "SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS",
        "SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT",
        "SPILOT_QWEN_MAX_ACTIVE_EPISODES",
        "SPILOT_GPT_MAX_ACTIVE_EPISODES",
        "SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES",
        "SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES",
        "SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY",
        "SPILOT_GPT_GATEWAY_MAX_CONCURRENCY",
        "SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES",
        "SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES",
    ):
        assert name in submitter
    assert "SPILOT_*" not in submitter


def test_graceful_deadline_prefers_slurm_end_time():
    script = f"""
source {TMAX / "lifecycle.sh"}
export WALL_TIME=4:00:00 TMAX_ENABLE_GRACEFUL_EXIT=1
export TMAX_GRACEFUL_EXIT_BUFFER_SECONDS=1800 TMAX_NOW_UNIX=1000
export SLURM_JOB_END_TIME=5000
tmax_configure_graceful_deadline
printf '%s|%s\n' "$SLIME_GRACEFUL_EXIT_AT_UNIX_TIME" "$TMAX_GRACEFUL_DEADLINE_SOURCE"
"""
    assert run_bash(script).stdout.strip() == "3200|SLURM_JOB_END_TIME"


def test_run_state_drops_legacy_port_and_persists_topology(tmp_path: Path):
    state = tmp_path / "state.env"
    env = clean_env(tmp_path)
    env.update(
        RUN_ID="run-a",
        SAVE_DIR=str(tmp_path / "save"),
        FINAL_EVAL_COMPLETE_MARKER=str(tmp_path / "save" / "FINAL_EVAL_COMPLETE"),
        TMAX_EVAL_DATA_SHA256="a" * 64,
        NUM_NODES="2",
        ACTOR_NUM_NODES="1",
        ROLLOUT_NUM_GPUS="8",
        POLAR_SLURM_MEM_PER_NODE="250G",
        TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL="1",
        SAVE_INTERVAL="1",
        SAVE_RETAIN_INTERVAL="200",
        POLAR_APPTAINER_NO_MOUNT_TMP="1",
        POLAR_APPTAINER_ISOLATE_PID="1",
        POLAR_APPTAINER_ISOLATE_IPC="1",
        POLAR_APPTAINER_CLEANENV="1",
        POLAR_SANDBOX_NETWORK="none",
        POLAR_COMPLETION_QUEUE_SIZE="4096",
        POLAR_COMPLETION_WRITE_WORKERS="12",
        POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY="7",
        POLAR_APPTAINER_PERSISTENT_BROKER="0",
        HF_CHECKPOINT=str(tmp_path / "Qwen3.5-9B"),
        REF_LOAD=str(tmp_path / "Qwen3.5-9B_torch_dist"),
        MODEL_ARGS_FILE=str(TMAX / "model_args.sh"),
        ACTOR_TENSOR_MODEL_PARALLEL_SIZE="4",
        SGLANG_MEM_FRACTION_STATIC="0.7",
        SGLANG_ROUTER_PORT="8681",
        TMAX_DYNAMIC_SAMPLING_FILTER_PATH=(
            "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
        ),
        SLIME_ROLLOUT_BASE_PORT="29440",
        STATE=str(state),
    )
    script = f"""
source {TMAX / "run_state.sh"}
tmax_write_run_state "$STATE"
unset SLIME_ROLLOUT_BASE_PORT
tmax_load_run_state "$STATE"
printf '%s' "${{SLIME_ROLLOUT_BASE_PORT-unset}}"
"""
    assert run_bash(script, env=env).stdout == "unset"
    content = state.read_text()
    assert "NUM_NODES=2" in content
    assert "FINAL_EVAL_COMPLETE_MARKER=" in content
    assert "FINAL_EVAL_COMPLETE" in content
    assert f"TMAX_EVAL_DATA_SHA256={'a' * 64}" in content
    assert "ACTOR_NUM_NODES=1" in content
    assert "POLAR_SLURM_MEM_PER_NODE=250G" in content
    assert "TMAX_ALLOW_CROSS_NODE_TENSOR_PARALLEL=1" in content
    assert "SAVE_INTERVAL=1" in content
    assert "SAVE_RETAIN_INTERVAL=200" in content
    assert "POLAR_APPTAINER_NO_MOUNT_TMP=1" in content
    assert "POLAR_APPTAINER_ISOLATE_PID=1" in content
    assert "POLAR_APPTAINER_ISOLATE_IPC=1" in content
    assert "POLAR_APPTAINER_CLEANENV=1" in content
    assert "POLAR_SANDBOX_NETWORK=none" in content
    assert "POLAR_COMPLETION_QUEUE_SIZE=4096" in content
    assert "POLAR_COMPLETION_WRITE_WORKERS=12" in content
    assert "POLAR_LOCAL_SUBPROCESS_SPAWN_CONCURRENCY=7" in content
    assert "POLAR_APPTAINER_PERSISTENT_BROKER=0" in content
    assert f"HF_CHECKPOINT={tmp_path / 'Qwen3.5-9B'}" in content
    assert f"REF_LOAD={tmp_path / 'Qwen3.5-9B_torch_dist'}" in content
    assert f"MODEL_ARGS_FILE={TMAX / 'model_args.sh'}" in content
    assert "ACTOR_TENSOR_MODEL_PARALLEL_SIZE=4" in content
    assert "SGLANG_MEM_FRACTION_STATIC=0.7" in content
    assert "SGLANG_ROUTER_PORT=8681" in content
    assert "TMAX_DYNAMIC_SAMPLING_FILTER_PATH=" in content
    assert "check_reward_nonzero_std" in content
    assert "SLIME_ROLLOUT_BASE_PORT" not in content


def test_run_state_round_trips_physical_wandb_run_id(tmp_path: Path):
    state = tmp_path / "state.env"
    env = clean_env(tmp_path)
    env.update(
        RUN_ID="logical-run",
        WANDB_RUN_ID="telemetry-run",
        SAVE_DIR=str(tmp_path / "save"),
        STATE=str(state),
    )
    script = f"""
source {TMAX / "run_state.sh"}
tmax_write_run_state "$STATE"
unset WANDB_RUN_ID
tmax_load_run_state "$STATE"
printf '%s' "$WANDB_RUN_ID"
"""

    assert run_bash(script, env=env).stdout == "telemetry-run"
    assert "export WANDB_RUN_ID=telemetry-run" in state.read_text()
    assert (
        'export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_ID}}"'
        in (TMAX / "submit_slurm.sh").read_text()
    )


def test_run_state_round_trips_mini_swe_agent_runtime(tmp_path: Path):
    state = tmp_path / "state.env"
    runtime_dir = tmp_path / "mini-swe-runtime"
    agent_bin = runtime_dir / "bin" / "mini-swe-agent"
    env = clean_env(tmp_path)
    env.update(
        RUN_ID="logical-run",
        SAVE_DIR=str(tmp_path / "save"),
        MINI_SWE_AGENT_RUNTIME_DIR=str(runtime_dir),
        MINI_SWE_AGENT_BIN=str(agent_bin),
        MINI_SWE_AGENT_SPEC="spilot_router",
        STATE=str(state),
    )
    script = f"""
source {TMAX / "run_state.sh"}
tmax_write_run_state "$STATE"
unset MINI_SWE_AGENT_RUNTIME_DIR MINI_SWE_AGENT_BIN MINI_SWE_AGENT_SPEC
tmax_load_run_state "$STATE"
printf '%s|%s|%s' \
    "$MINI_SWE_AGENT_RUNTIME_DIR" "$MINI_SWE_AGENT_BIN" "$MINI_SWE_AGENT_SPEC"
"""

    assert run_bash(script, env=env).stdout == (
        f"{runtime_dir}|{agent_bin}|spilot_router"
    )
    content = state.read_text()
    assert f"export MINI_SWE_AGENT_RUNTIME_DIR={runtime_dir}\n" in content
    assert f"export MINI_SWE_AGENT_BIN={agent_bin}\n" in content
    assert "export MINI_SWE_AGENT_SPEC=spilot_router\n" in content


def test_run_state_round_trips_spilot_episode_admission_contract(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.env"
    expected = {
        "SPILOT_EPISODE_ADMISSION_ENABLED": "true",
        "SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS": "14400",
        "SPILOT_EPISODE_ADMISSION_GATEWAY_COUNT": "8",
        "SPILOT_QWEN_MAX_ACTIVE_EPISODES": "8",
        "SPILOT_GPT_MAX_ACTIVE_EPISODES": "32",
        "SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES": "1",
        "SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES": "4",
        "SPILOT_QWEN_GATEWAY_MAX_CONCURRENCY": "1",
        "SPILOT_GPT_GATEWAY_MAX_CONCURRENCY": "4",
        "SPILOT_QWEN_EFFECTIVE_MAX_ACTIVE_EPISODES": "8",
        "SPILOT_GPT_EFFECTIVE_MAX_ACTIVE_EPISODES": "32",
        "PARTITION": "backfill",
        "WALL_TIME": "2-00:00:00",
        "TMAX_MIN_WALL_TIME": "2-00:00:00",
        "SAVE_INTERVAL": "5",
        "TMAX_GRACEFUL_EXIT_BUFFER_SECONDS": "43200",
        "TMAX_MIN_GRACEFUL_EXIT_BUFFER_SECONDS": "43200",
        "TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT": "0",
    }
    env = clean_env(tmp_path)
    env.update(STATE=str(state), RUN_ID="router-run", SAVE_DIR="/tmp/router-save")
    env.update(expected)
    names = " ".join(expected)
    print_args = " ".join(f'"${{{name}}}"' for name in expected)
    script = f"""
source {TMAX / "run_state.sh"}
tmax_write_run_state "$STATE"
unset {names}
tmax_load_run_state "$STATE"
printf '%s\\n' {print_args}
"""

    values = run_bash(script, env=env).stdout.splitlines()

    assert values == list(expected.values())
    content = state.read_text()
    for name, value in expected.items():
        assert f"export {name}={value}\n" in content


def test_run_state_detects_fields_present_in_legacy_files(tmp_path: Path):
    state = tmp_path / "state.env"
    state.write_text(
        "export RUN_ID=legacy-run\nexport SAVE_DIR=/tmp/legacy-save\nexport TMAX_EVAL_ENABLED=1\n"
    )
    script = f"""
source {TMAX / "run_state.sh"}
if tmax_run_state_has_export "$STATE" TMAX_EVAL_ENABLED; then printf enabled; fi
if tmax_run_state_has_export "$STATE" TMAX_EVAL_SOURCE; then printf bad; else printf '|source-missing'; fi
"""
    env = clean_env(tmp_path)
    env["STATE"] = str(state)

    assert run_bash(script, env=env).stdout == "enabled|source-missing"


def test_run_state_persists_resume_entrypoints_but_not_spilot_credentials(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state.env"
    entrypoints = {
        "TMAX_SUBMIT_SCRIPT": "/repo/examples/spilot_router_slime_grpo/submit_slurm.sh",
        "POLAR_TRAIN_RUN_SCRIPT": "/repo/examples/spilot_router_slime_grpo/run.sh",
        "POLAR_CONFIG_TEMPLATE": "/repo/examples/spilot_router_slime_grpo/polar_config.yaml",
        "TOPOLOGY_TEMPLATE": "/repo/examples/spilot_router_slime_grpo/topology.yaml",
    }
    env = clean_env(tmp_path)
    env.update(
        STATE=str(state),
        RUN_ID="router-run",
        SAVE_DIR="/tmp/router-save",
        POLAR_CONTROL_PLANE_TOKEN="c" * 32,
        POLAR_NVIDIA_API_KEY="private-model-pool-key",
        **entrypoints,
    )
    script = f"""
source {TMAX / "run_state.sh"}
tmax_write_run_state "$STATE"
"""

    run_bash(script, env=env)
    content = state.read_text()
    for name, value in entrypoints.items():
        assert f"export {name}={value}\n" in content
    assert "POLAR_CONTROL_PLANE_TOKEN" not in content
    assert "POLAR_NVIDIA_API_KEY" not in content


def watcher_env(tmp_path: Path, bin_dir: Path) -> dict[str, str]:
    save = tmp_path / "save"
    save.mkdir()
    train = tmp_path / "train.jsonl"
    train.write_text('{"prompt": []}\n')
    eval_data = tmp_path / "eval.jsonl"
    eval_data.write_text('{"prompt": [], "metadata": {"task_name": "eval-1"}}\n')
    env = clean_env(tmp_path)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        TMAX_RUN_STATE_FILE=str(tmp_path / "state.env"),
        RUN_ID="run-a",
        SAVE_DIR=str(save),
        TMAX_TRAIN_DATA=str(train),
        TMAX_EVAL_DATA=str(eval_data),
        TMAX_PREPARE_DATA="0",
        TMAX_EVAL_ENABLED="1",
        TMAX_EXTERNAL_EVAL_ENABLED="0",
        JOB_NAME="polar-tmax-run-a",
        TMAX_WATCH_MAX_QUICK_FAILURES="3",
        TMAX_WATCH_QUICK_FAILURE_SECONDS="900",
    )
    return env


def enabled_spilot_watcher_env(tmp_path: Path, bin_dir: Path) -> dict[str, str]:
    env = watcher_env(tmp_path, bin_dir)
    # The canonical exclusion file lives in the real reference-data tree.
    # These watcher unit tests use a one-row synthetic dataset, so select that
    # explicit smoke-mode contract before sourcing the wrapper defaults.
    env["TMAX_EXCLUDE_DATA"] = ""
    env["TMAX_EVAL_ENABLED"] = "0"
    env["TMAX_TRAINING_EVAL_ENABLED"] = "0"
    completed = run_bash(
        f"source {SPILOT / 'experiment_defaults.sh'}; env -0",
        env=env,
    )
    for item in completed.stdout.split("\0"):
        if "=" in item:
            name, value = item.split("=", 1)
            env[name] = value
    # Give admission-enabled watcher tests a self-consistent resumable wall
    # contract so they reach terminal-job accounting rather than preflight.
    env.update(
        PARTITION="backfill",
        TMAX_GRACEFUL_EXIT_BUFFER_SECONDS="40800",
        WALL_TIME="1-00:00:00",
    )
    env["POLAR_NVIDIA_API_KEY"] = "test-model-pool-key"
    return env


def write_spilot_admission_fatal_marker(
    state_file: Path,
    *,
    job_id: str,
    rank: int = 0,
) -> Path:
    marker = Path(
        f"{state_file}.job-{job_id}.admission-fatal.rank-{rank}.json"
    )
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "model_pool_episode_admission_fatal_retained",
                "job_id": job_id,
                "rank": rank,
                "node_id": f"slurm-rank-{rank}",
                "observed_at": "2026-07-07T12:00:00+00:00",
                "retained_count": 1,
            }
        )
    )
    return marker


def write_candidate_pool_health_incident(
    save_dir: Path,
    *,
    job_id: str,
    rollout_id: int = 7,
    triggered: bool = True,
    fallback_state_file: Path | None = None,
) -> Path:
    incident_dir = (
        Path(f"{fallback_state_file}.candidate_pool_health_incidents")
        if fallback_state_file is not None
        else save_dir / "rollout" / "candidate_pool_health_incidents"
    )
    incident_dir.mkdir(parents=True, exist_ok=True)
    incident = incident_dir / f"rollout_{rollout_id:07d}.json"
    incident.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "slurm_job_id": job_id,
                "rollout_id": rollout_id,
                "triggered": triggered,
                "trigger_reasons": ["C1:completion_fraction_below_threshold"],
            }
        )
    )
    return incident


@pytest.mark.parametrize(
    ("submit_script", "nvidia_key", "expected_error"),
    [
        (
            str(TMAX / "submit_slurm.sh"),
            "test-model-pool-key",
            "watcher preflight: SPilot Router requires TMAX_SUBMIT_SCRIPT=",
        ),
        (
            str(ROOT / "examples" / "spilot_router_slime_grpo" / "submit_slurm.sh"),
            None,
            "SPilot Router watcher requires NVIDIA_API_KEY",
        ),
    ],
)
def test_generic_watcher_fails_closed_on_incomplete_spilot_resume_environment(
    tmp_path: Path,
    submit_script: str,
    nvidia_key: str | None,
    expected_error: str,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "exit 0\n")
    env = watcher_env(tmp_path, bin_dir)
    env.update(
        TMAX_AGENT_HARNESS="spilot_router",
        TMAX_SUBMIT_SCRIPT=submit_script,
        POLAR_TRAIN_RUN_SCRIPT=str(SPILOT / "run.sh"),
        POLAR_CONFIG_TEMPLATE=str(SPILOT / "polar_config.yaml"),
        TOPOLOGY_TEMPLATE=str(SPILOT / "topology.yaml"),
    )
    if nvidia_key is not None:
        env["POLAR_NVIDIA_API_KEY"] = nvidia_key

    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert expected_error in result.stderr


def test_watcher_loaded_spilot_state_without_entrypoints_cannot_fall_back(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "exit 0\n")
    env = watcher_env(tmp_path, bin_dir)
    env["POLAR_NVIDIA_API_KEY"] = "test-model-pool-key"
    state = Path(env["TMAX_RUN_STATE_FILE"])
    original = (
        f"export RUN_ID={env['RUN_ID']}\n"
        f"export SAVE_DIR={env['SAVE_DIR']}\n"
        f"export TMAX_TRAIN_DATA={env['TMAX_TRAIN_DATA']}\n"
        f"export TMAX_EVAL_DATA={env['TMAX_EVAL_DATA']}\n"
        "export TMAX_AGENT_HARNESS=spilot_router\n"
        "export JOB_NAME=polar-tmax-run-a\n"
    )
    state.write_text(original)

    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "watcher preflight: SPilot Router requires TMAX_SUBMIT_SCRIPT=" in result.stderr
    assert state.read_text() == original


def test_watcher_legacy_state_ignores_inherited_new_eval_defaults(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "exit 0\n")
    env = watcher_env(tmp_path, bin_dir)
    state = Path(env["TMAX_RUN_STATE_FILE"])
    state.write_text(
        f"export RUN_ID=legacy-run\n"
        f"export SAVE_DIR={env['SAVE_DIR']}\n"
        f"export TMAX_TRAIN_DATA={env['TMAX_TRAIN_DATA']}\n"
        f"export TMAX_EVAL_DATA={env['TMAX_EVAL_DATA']}\n"
        "export JOB_NAME=polar-tmax-legacy-run\n"
    )
    env.pop("RUN_ID")
    env.update(TMAX_EVAL_ENABLED="1", TMAX_EVAL_SOURCE="harbor")
    (Path(env["SAVE_DIR"]) / "TRAINING_COMPLETE").write_text("legacy marker\n")

    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "training complete marker found" in result.stdout
    assert "final eval is incomplete" not in result.stdout


def test_watcher_legacy_spilot_state_disables_admission_and_keeps_old_timeouts(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "exit 0\n")
    env = watcher_env(tmp_path, bin_dir)
    env.update(
        RUN_ID="legacy-spilot",
        POLAR_NVIDIA_API_KEY="test-model-pool-key",
        **spilot_entrypoint_env(),
    )
    state = Path(env["TMAX_RUN_STATE_FILE"])
    state.write_text(
        f"export RUN_ID={env['RUN_ID']}\n"
        f"export SAVE_DIR={env['SAVE_DIR']}\n"
        f"export TMAX_TRAIN_DATA={env['TMAX_TRAIN_DATA']}\n"
        f"export TMAX_EVAL_DATA={env['TMAX_EVAL_DATA']}\n"
        "export TMAX_AGENT_HARNESS=spilot_router\n"
        "export TMAX_EVAL_ENABLED=0\n"
        "export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS=3300\n"
        "export POLAR_TASK_TIMEOUT_FLOOR_SECONDS=4500\n"
        "export POLAR_REQUEST_TIMEOUT=5100\n"
        "export JOB_NAME=polar-tmax-legacy-spilot\n"
    )
    (Path(env["SAVE_DIR"]) / "TRAINING_COMPLETE").write_text("legacy marker\n")

    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "legacy SPilot run state: episode admission disabled" in result.stderr
    content = state.read_text()
    assert "export SPILOT_EPISODE_ADMISSION_ENABLED=false\n" in content
    assert "export SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS=0\n" in content
    assert "export SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES=null\n" in content
    assert "export SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES=null\n" in content
    assert "export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS=3300\n" in content
    assert "export POLAR_TASK_TIMEOUT_FLOOR_SECONDS=4500\n" in content
    assert "export POLAR_REQUEST_TIMEOUT=5100\n" in content


def test_watcher_rejects_partial_spilot_admission_run_state(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "exit 0\n")
    env = watcher_env(tmp_path, bin_dir)
    env.update(
        RUN_ID="partial-spilot",
        POLAR_NVIDIA_API_KEY="test-model-pool-key",
        **spilot_entrypoint_env(),
    )
    state = Path(env["TMAX_RUN_STATE_FILE"])
    state.write_text(
        f"export RUN_ID={env['RUN_ID']}\n"
        f"export SAVE_DIR={env['SAVE_DIR']}\n"
        "export TMAX_AGENT_HARNESS=spilot_router\n"
        "export SPILOT_EPISODE_ADMISSION_ENABLED=true\n"
    )

    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "partial SPilot episode-admission contract" in result.stderr


def write_checkpoint_pair(save_dir: Path, iteration: int) -> None:
    (save_dir / "latest_checkpointed_iteration.txt").write_text(f"{iteration}\n")
    model_dir = save_dir / f"iter_{iteration:07d}"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "common.pt").write_bytes(b"common")
    write_distcp_metadata(model_dir, "__0_0.distcp")
    (model_dir / "__0_0.distcp").write_bytes(b"weights")
    state = save_dir / "rollout" / f"global_dataset_state_dict_{iteration}.pt"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("state\n")


def test_watcher_requires_matching_final_eval_marker_when_eval_enabled(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "exit 0\n")
    env = watcher_env(tmp_path, bin_dir)
    env["TMAX_TARGET_ITER"] = "5"
    save = tmp_path / "save"
    write_checkpoint_pair(save, 5)
    (save / "TRAINING_COMPLETE").write_text("legacy marker\n")

    incomplete = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert incomplete.returncode == 0, incomplete.stderr
    assert "final checkpoint reached" in incomplete.stdout
    assert "final eval is incomplete" in incomplete.stdout
    assert "training complete marker found" not in incomplete.stdout

    (save / "FINAL_EVAL_COMPLETE").write_text(
        json.dumps(
            {
                "final_rollout_id": 5,
                "model_iteration": 5,
                "num_rollout": 6,
                "eval_data_sha256": hashlib.sha256(
                    Path(env["TMAX_EVAL_DATA"]).read_bytes()
                ).hexdigest(),
            }
        )
    )
    complete = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert complete.returncode == 0, complete.stderr
    assert "final eval complete marker found" in complete.stdout

    write_checkpoint_pair(save, 6)
    overshot = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert overshot.returncode == 0, overshot.stderr
    assert "final eval complete marker found" not in overshot.stdout

    write_checkpoint_pair(save, 5)

    Path(env["TMAX_EVAL_DATA"]).write_text(
        '{"prompt": [], "metadata": {"task_name": "mutated-eval"}}\n'
    )
    stale = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert stale.returncode == 0, stale.stderr
    assert "final eval marker is stale or malformed" in stale.stderr
    assert "final eval complete marker found" not in stale.stdout


def test_watcher_relaunches_final_checkpoint_for_eval_only_recovery(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "exit 0\n")
    submit = tmp_path / "fake-submit.sh"
    write_command(
        submit,
        """
mkdir -p "$(dirname "$TMAX_SUBMIT_RECEIPT_FILE")"
printf 'export POLAR_SUBMITTED_JOB_ID=333\\nexport POLAR_SUBMITTED_AT_UNIX=12345\\n' > "$TMAX_SUBMIT_RECEIPT_FILE"
""",
    )
    env = watcher_env(tmp_path, bin_dir)
    env.update(TMAX_TARGET_ITER="5", TMAX_SUBMIT_SCRIPT=str(submit))
    write_checkpoint_pair(tmp_path / "save", 5)

    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "next resume will run eval only" in result.stdout
    assert "submission succeeded: job=333" in result.stdout
    state = (tmp_path / "state.env").read_text()
    assert "export FINAL_EVAL_COMPLETE_MARKER=" in state
    assert "FINAL_EVAL_COMPLETE" in state


def test_watcher_keeps_checkpoint_only_completion_when_eval_disabled(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "exit 0\n")
    env = watcher_env(tmp_path, bin_dir)
    env.update(TMAX_TARGET_ITER="5", TMAX_EVAL_ENABLED="0")
    write_checkpoint_pair(tmp_path / "save", 5)

    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "target reached: latest_iter=5 target=5" in result.stdout


def test_watcher_stops_after_consecutive_no_progress_even_when_state_changes(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "printf '%s\\n' \"$SACCT_RECORD\"\n")

    env = watcher_env(tmp_path, bin_dir)
    env.update(
        TMAX_LAST_JOB_ID="101",
        TMAX_LAST_JOB_CHECKPOINT_ITER="-1",
        TMAX_WATCH_FAILURE_COUNT="2",
        TMAX_WATCH_FAILURE_SIGNATURE="FAILED/exit=1:0/checkpoint=-1",
        SACCT_RECORD="101|FAILED|1800|1:0|",
    )
    failed = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert failed.returncode == 1
    assert "refusing another automatic submission" in failed.stderr

    # A long exit-0 allocation with no final marker and no newer checkpoint is
    # still no progress; it must not reset the retry guard and loop forever.
    normal_dir = tmp_path / "normal"
    normal_dir.mkdir()
    normal_env = watcher_env(normal_dir, bin_dir)
    normal_env.update(
        TMAX_LAST_JOB_ID="102",
        TMAX_LAST_JOB_CHECKPOINT_ITER="-1",
        TMAX_WATCH_FAILURE_COUNT="2",
        TMAX_WATCH_FAILURE_SIGNATURE="FAILED/exit=1:0/checkpoint=-1",
        SACCT_RECORD="102|COMPLETED|2400|0:0|",
    )
    normal = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh")],
        cwd=ROOT,
        env=normal_env,
        text=True,
        capture_output=True,
    )
    assert normal.returncode == 1
    assert "no-progress failure 3/3" in normal.stderr
    assert "export TMAX_WATCH_FAILURE_COUNT=3" in (normal_dir / "state.env").read_text()


@pytest.mark.parametrize("terminal_state", ["FAILED", "OUT_OF_MEMORY", "TIMEOUT"])
def test_watcher_fail_closes_first_quick_deterministic_failure(
    tmp_path: Path, terminal_state: str
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "printf '%s\\n' \"$SACCT_RECORD\"\n")
    submit = tmp_path / "fake-submit.sh"
    submit_called = tmp_path / "submit-called"
    write_command(
        submit,
        """
touch "$SUBMIT_CALLED"
mkdir -p "$(dirname "$TMAX_SUBMIT_RECEIPT_FILE")"
printf 'export POLAR_SUBMITTED_JOB_ID=333\\nexport POLAR_SUBMITTED_AT_UNIX=12345\\n' > "$TMAX_SUBMIT_RECEIPT_FILE"
""",
    )
    env = watcher_env(tmp_path, bin_dir)
    env.update(
        TMAX_LAST_JOB_ID="101",
        TMAX_LAST_JOB_CHECKPOINT_ITER="-1",
        TMAX_SUBMIT_SCRIPT=str(submit),
        SUBMIT_CALLED=str(submit_called),
        SACCT_RECORD=f"101|{terminal_state}|120|1:0|",
    )

    first = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert first.returncode == 1
    assert "fail-closed quick failure" in first.stderr
    assert "refusing automatic resubmission" in first.stderr
    assert not submit_called.exists()
    state = (tmp_path / "state.env").read_text()
    assert "export TMAX_WATCH_FAILURE_COUNT=1" in state
    assert f"quick-fail-closed/{terminal_state}/exit=1:0/checkpoint=-1" in state

    # The fail-closed outcome is persisted. A service restart must not bypass
    # it and submit the same broken allocation; an operator explicitly resets
    # the latch only after fixing the root cause.
    restarted_env = env.copy()
    restarted_env.pop("RUN_ID")
    restarted = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=restarted_env,
        text=True,
        capture_output=True,
    )

    assert restarted.returncode == 1
    assert "fail-closed quick-failure latch remains set" in restarted.stderr
    assert not submit_called.exists()

    reset_env = restarted_env.copy()
    reset_env["TMAX_WATCH_RESET_FAILURES"] = "1"
    reset = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=reset_env,
        text=True,
        capture_output=True,
    )

    assert reset.returncode == 0, reset.stderr
    assert "submission succeeded: job=333" in reset.stdout
    assert submit_called.exists()


@pytest.mark.parametrize("use_fallback_location", [False, True])
def test_watcher_fail_closes_candidate_pool_health_incident_for_terminal_job(
    tmp_path: Path,
    use_fallback_location: bool,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "printf '%s\\n' \"$SACCT_RECORD\"\n")
    submit_called = tmp_path / "submit-called"
    bash_proxy = bin_dir / "bash"
    bash_proxy.write_text(
        """#!/bin/sh
if [ "${1:-}" = "$EXPECTED_SUBMIT_SCRIPT" ]; then
    touch "$SUBMIT_CALLED"
    exit 0
fi
exec /bin/bash "$@"
"""
    )
    bash_proxy.chmod(0o755)
    env = enabled_spilot_watcher_env(tmp_path, bin_dir)
    env.update(
        TMAX_LAST_JOB_ID="201",
        TMAX_LAST_JOB_CHECKPOINT_ITER="-1",
        EXPECTED_SUBMIT_SCRIPT=str(SPILOT / "submit_slurm.sh"),
        SUBMIT_CALLED=str(submit_called),
        SACCT_RECORD="201|COMPLETED|7200|0:0|",
    )
    write_candidate_pool_health_incident(
        Path(env["SAVE_DIR"]),
        job_id="201",
        rollout_id=50,
        fallback_state_file=(
            Path(env["TMAX_RUN_STATE_FILE"])
            if use_fallback_location
            else None
        ),
    )

    result = subprocess.run(
        ["/bin/bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert not submit_called.exists()
    assert "candidate-pool health incident: job=201" in result.stderr
    assert "refusing automatic resubmission" in result.stderr
    state = Path(env["TMAX_RUN_STATE_FILE"]).read_text()
    assert "export TMAX_WATCH_LAST_ACCOUNTED_JOB_ID=201\n" in state
    assert "quick-fail-closed/candidate-pool-health/job=201" in state
    assert r"incident=1\|50\|C1:completion_fraction_below_threshold" in state


def test_watcher_ignores_candidate_pool_health_incident_from_old_job(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "printf '%s\\n' \"$SACCT_RECORD\"\n")
    submit_called = tmp_path / "submit-called"
    bash_proxy = bin_dir / "bash"
    bash_proxy.write_text(
        """#!/bin/sh
if [ "${1:-}" = "$EXPECTED_SUBMIT_SCRIPT" ]; then
    touch "$SUBMIT_CALLED"
    mkdir -p "$(dirname "$TMAX_SUBMIT_RECEIPT_FILE")"
    printf 'export POLAR_SUBMITTED_JOB_ID=333\nexport POLAR_SUBMITTED_AT_UNIX=12345\n' > "$TMAX_SUBMIT_RECEIPT_FILE"
    exit 0
fi
exec /bin/bash "$@"
"""
    )
    bash_proxy.chmod(0o755)
    env = enabled_spilot_watcher_env(tmp_path, bin_dir)
    env.update(
        TMAX_LAST_JOB_ID="202",
        TMAX_LAST_JOB_CHECKPOINT_ITER="-1",
        EXPECTED_SUBMIT_SCRIPT=str(SPILOT / "submit_slurm.sh"),
        SUBMIT_CALLED=str(submit_called),
        SACCT_RECORD="202|PREEMPTED|120|1:0|",
    )
    write_candidate_pool_health_incident(
        Path(env["SAVE_DIR"]),
        job_id="201",
        rollout_id=49,
    )

    result = subprocess.run(
        ["/bin/bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert submit_called.exists()
    assert "candidate-pool health incident" not in result.stderr
    assert "submission succeeded: job=333" in result.stdout


@pytest.mark.parametrize("admission_enabled", ["true", "false"])
def test_watcher_relaunches_quick_spilot_retained_fatal_and_counts_run_metric(
    tmp_path: Path,
    admission_enabled: str,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "printf '%s\\n' \"$SACCT_RECORD\"\n")
    submit_called = tmp_path / "submit-called"
    bash_proxy = bin_dir / "bash"
    bash_proxy.write_text(
        """#!/bin/sh
if [ "${1:-}" = "$EXPECTED_SUBMIT_SCRIPT" ]; then
    touch "$SUBMIT_CALLED"
    mkdir -p "$(dirname "$TMAX_SUBMIT_RECEIPT_FILE")"
    printf 'export POLAR_SUBMITTED_JOB_ID=333\nexport POLAR_SUBMITTED_AT_UNIX=12345\n' > "$TMAX_SUBMIT_RECEIPT_FILE"
    exit 0
fi
exec /bin/bash "$@"
"""
    )
    bash_proxy.chmod(0o755)
    env = enabled_spilot_watcher_env(tmp_path, bin_dir)
    env.update(
        TMAX_LAST_JOB_ID="101",
        TMAX_LAST_JOB_CHECKPOINT_ITER="-1",
        SPILOT_EPISODE_ADMISSION_ENABLED=admission_enabled,
        EXPECTED_SUBMIT_SCRIPT=str(SPILOT / "submit_slurm.sh"),
        SUBMIT_CALLED=str(submit_called),
        SACCT_RECORD="101|FAILED|120|70:0|",
    )
    if admission_enabled == "false":
        env.update(
            SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS="0",
            SPILOT_QWEN_GATEWAY_MAX_ACTIVE_EPISODES="null",
            SPILOT_GPT_GATEWAY_MAX_ACTIVE_EPISODES="null",
        )
    state_file = Path(env["TMAX_RUN_STATE_FILE"])
    write_spilot_admission_fatal_marker(state_file, job_id="101", rank=3)

    result = subprocess.run(
        ["/bin/bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert submit_called.exists()
    assert "SPilot admission fatal job metric: count_total=1" in result.stderr
    assert "fail-closed quick failure" not in result.stderr
    assert "submission succeeded: job=333" in result.stdout
    state = state_file.read_text()
    assert "export TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT=1\n" in state
    assert (
        "recoverable/model-pool-episode-admission-retained/FAILED/exit=70:0/checkpoint=-1"
        in state
    )


def test_watcher_stops_after_three_no_progress_spilot_retained_fatals(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "printf '%s\\n' \"$SACCT_RECORD\"\n")
    publisher = bin_dir / "wandb-publisher-python"
    publish_args = tmp_path / "wandb-publish-args"
    write_command(publisher, "printf '%s\\n' \"$@\" > \"$PUBLISH_ARGS\"\n")
    env = enabled_spilot_watcher_env(tmp_path, bin_dir)
    env.update(
        TMAX_LAST_JOB_ID="103",
        TMAX_LAST_JOB_CHECKPOINT_ITER="-1",
        TMAX_WATCH_FAILURE_COUNT="2",
        TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT="2",
        SACCT_RECORD="103|FAILED|120|70:0|",
        WANDB_API_KEY="test-wandb-key",
        WANDB_MODE="online",
        TMAX_SPILOT_FATAL_WANDB_PYTHON_BIN=str(publisher),
        PUBLISH_ARGS=str(publish_args),
    )
    state_file = Path(env["TMAX_RUN_STATE_FILE"])
    write_spilot_admission_fatal_marker(state_file, job_id="103")

    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "no-progress failure 3/3" in result.stderr
    assert "refusing another automatic submission" in result.stderr
    state = state_file.read_text()
    assert "export TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT=3\n" in state
    assert "export TMAX_WATCH_FAILURE_COUNT=3\n" in state
    assert publish_args.exists()
    published = publish_args.read_text().splitlines()
    assert "--one-shot-static" in published
    assert "polar/spilot_router/admission_fatal_job_count_total=3" in published
    assert "published final SPilot admission fatal count=3 to W&B" in result.stderr


def test_watcher_does_not_trust_malformed_spilot_retained_fatal_marker(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "printf '%s\\n' \"$SACCT_RECORD\"\n")
    env = enabled_spilot_watcher_env(tmp_path, bin_dir)
    env.update(
        TMAX_LAST_JOB_ID="104",
        TMAX_LAST_JOB_CHECKPOINT_ITER="-1",
        SACCT_RECORD="104|FAILED|120|70:0|",
    )
    state_file = Path(env["TMAX_RUN_STATE_FILE"])
    marker = Path(f"{state_file}.job-104.admission-fatal.rank-0.json")
    marker.write_text('{"job_id":"wrong-job"}\n')

    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "malformed SPilot admission fatal marker" in result.stderr
    assert "fail-closed quick failure" in result.stderr
    state = state_file.read_text()
    assert "export TMAX_SPILOT_ADMISSION_FATAL_JOB_COUNT=0\n" in state


@pytest.mark.parametrize("terminal_state", ["PREEMPTED", "NODE_FAIL", "REVOKED"])
def test_watcher_keeps_quick_infrastructure_failures_recoverable(
    tmp_path: Path, terminal_state: str
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "printf '%s\\n' \"$SACCT_RECORD\"\n")
    submit = tmp_path / "fake-submit.sh"
    submit_called = tmp_path / "submit-called"
    write_command(
        submit,
        """
touch "$SUBMIT_CALLED"
mkdir -p "$(dirname "$TMAX_SUBMIT_RECEIPT_FILE")"
printf 'export POLAR_SUBMITTED_JOB_ID=333\\nexport POLAR_SUBMITTED_AT_UNIX=12345\\n' > "$TMAX_SUBMIT_RECEIPT_FILE"
""",
    )
    env = watcher_env(tmp_path, bin_dir)
    env.update(
        TMAX_LAST_JOB_ID="101",
        TMAX_LAST_JOB_CHECKPOINT_ITER="-1",
        TMAX_SUBMIT_SCRIPT=str(submit),
        SUBMIT_CALLED=str(submit_called),
        SACCT_RECORD=f"101|{terminal_state}|120|1:0|",
    )

    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert submit_called.exists()
    assert "submission succeeded: job=333" in result.stdout
    assert "fail-closed quick failure" not in result.stderr
    state = (tmp_path / "state.env").read_text()
    assert "export TMAX_WATCH_FAILURE_COUNT=1" in state
    assert f"{terminal_state}/exit=1:0/checkpoint=-1" in state


def test_watcher_records_fresh_receipt_before_next_poll(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "exit 0\n")
    submit = tmp_path / "fake-submit.sh"
    write_command(
        submit,
        """
mkdir -p "$(dirname "$TMAX_SUBMIT_RECEIPT_FILE")"
printf 'export TMAX_PRORL_GIT_COMMIT=%040d\\nexport TMAX_SLIME_GIT_COMMIT=%040d\\nexport TMAX_MEGATRON_GIT_COMMIT=%040d\\n' 1 2 3 >> "$TMAX_RUN_STATE_FILE"
printf 'export POLAR_SUBMITTED_JOB_ID=222\\nexport POLAR_SUBMITTED_AT_UNIX=12345\\n' > "$TMAX_SUBMIT_RECEIPT_FILE"
""",
    )
    env = watcher_env(tmp_path, bin_dir)
    env["TMAX_SUBMIT_SCRIPT"] = str(submit)
    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "submission succeeded: job=222" in result.stdout
    state = (tmp_path / "state.env").read_text()
    assert "export TMAX_LAST_JOB_ID=222" in state
    assert f"export TMAX_PRORL_GIT_COMMIT={'1'.zfill(40)}" in state
    assert f"export TMAX_SLIME_GIT_COMMIT={'2'.zfill(40)}" in state
    assert f"export TMAX_MEGATRON_GIT_COMMIT={'3'.zfill(40)}" in state


def test_watcher_retains_source_lock_after_submission_failure_without_receipt(
    tmp_path: Path,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "exit 0\n")
    submit = tmp_path / "fake-submit.sh"
    write_command(
        submit,
        """
printf 'export TMAX_PRORL_GIT_COMMIT=%040d\\nexport TMAX_SLIME_GIT_COMMIT=%040d\\nexport TMAX_MEGATRON_GIT_COMMIT=%040d\\n' 4 5 6 >> "$TMAX_RUN_STATE_FILE"
exit 17
""",
    )
    env = watcher_env(tmp_path, bin_dir)
    env["TMAX_SUBMIT_SCRIPT"] = str(submit)

    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "submission failure 1/3" in result.stderr
    state = (tmp_path / "state.env").read_text()
    assert "export TMAX_WATCH_FAILURE_COUNT=1" in state
    assert f"export TMAX_PRORL_GIT_COMMIT={'4'.zfill(40)}" in state
    assert f"export TMAX_SLIME_GIT_COMMIT={'5'.zfill(40)}" in state
    assert f"export TMAX_MEGATRON_GIT_COMMIT={'6'.zfill(40)}" in state


def test_watcher_adopts_unique_named_job_when_tracked_receipt_is_stale(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(
        bin_dir / "squeue",
        """
case " $* " in
  *" -n "*) printf '222 R polar-tmax-run-a node-1\\n' ;;
esac
""",
    )
    write_command(bin_dir / "sacct", "exit 0\n")
    env = watcher_env(tmp_path, bin_dir)
    env["TMAX_LAST_JOB_ID"] = "111"

    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "adopted active job 222" in result.stdout
    assert "job_id=222" in result.stdout
    assert "export TMAX_LAST_JOB_ID=222" in (tmp_path / "state.env").read_text()


def test_watcher_aborts_on_multiple_exact_named_jobs(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(
        bin_dir / "squeue",
        """
case " $* " in
  *" -n "*) printf '222 R polar-tmax-run-a node-1\\n223 PD polar-tmax-run-a Priority\\n' ;;
esac
""",
    )
    write_command(bin_dir / "sacct", "exit 0\n")
    env = watcher_env(tmp_path, bin_dir)

    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "multiple active jobs match polar-tmax-run-a" in result.stderr


def test_watcher_rejects_model_pointer_without_rollout_state(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_command(bin_dir / "squeue", "exit 0\n")
    write_command(bin_dir / "sacct", "exit 0\n")
    env = watcher_env(tmp_path, bin_dir)
    (tmp_path / "save" / "latest_checkpointed_iteration.txt").write_text("5\n")
    model_dir = tmp_path / "save" / "iter_0000005"
    model_dir.mkdir()
    (model_dir / "common.pt").write_bytes(b"common")
    write_distcp_metadata(model_dir, "__0_0.distcp")
    (model_dir / "__0_0.distcp").write_bytes(b"weights")

    result = subprocess.run(
        ["bash", str(TMAX / "watch_training.sh"), "--relaunch"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "checkpoint 5 has no matching rollout state" in result.stderr
