from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TMAX = ROOT / "examples" / "tmax_slime_grpo"
SUBMIT_MATRIX = TMAX / "submit_matrix.sh"
WATCH_MATRIX = TMAX / "watch_matrix.sh"
LEGACY_SUBMIT_MATRIX = TMAX / "submit_4node_matrix.sh"
LEGACY_WATCH_MATRIX = TMAX / "watch_4node_matrix.sh"
MATRIX_SETTINGS_SCRIPT = TMAX / "matrix_settings.sh"

MATRIX_SETTINGS = (
    "qwen35-4b-fidelity",
    "qwen35-4b-fidelity-8n",
    "qwen35-4b-fidelity-8n-b16n8-traj",
    "qwen35-9b-baseline-a2-full65k",
    "qwen35-9b-b16n16-a2-full65k",
    "qwen35-9b-async4-full65k",
    "qwen35-9b-lr5e7-a2-full65k",
    "qwen35-9b-lr2e6-a2-full65k",
)
EIGHT_NODE_SETTINGS = (
    "qwen35-4b-fidelity-8n",
    "qwen35-4b-fidelity-8n-b16n8-traj",
)
FOUR_NODE_SETTINGS = tuple(
    setting for setting in MATRIX_SETTINGS if setting not in EIGHT_NODE_SETTINGS
)
SHORT_CAP_SETTINGS = (
    "qwen35-9b-baseline-a2-mt4k",
    "qwen35-9b-b16n16-a2-mt4k",
    "qwen35-9b-async4-mt4k",
    "qwen35-9b-lr5e7-a2-mt4k",
    "qwen35-9b-a2-mt8k",
)


def _case_arm(script: str, setting: str) -> str:
    start = script.index(f"        {setting})")
    end = script.index("            ;;", start)
    return script[start:end]


def _bash_array(script: str, name: str) -> tuple[str, ...]:
    marker = f"readonly -a {name}=("
    start = script.index(marker) + len(marker)
    end = script.index("\n)", start)
    return tuple(line.strip() for line in script[start:end].splitlines() if line.strip())


def _select_matrix_settings(
    scope: str, *settings: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                'set -euo pipefail; source "$1"; shift; scope="$1"; shift; '
                "declare -a selected; "
                'tmax_matrix_select_settings selected "$scope" "$@"; '
                'printf "%s\\n" "${selected[@]}"'
            ),
            "matrix-selector",
            str(MATRIX_SETTINGS_SCRIPT),
            scope,
            *settings,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_tmax_matrix_scripts_remain_valid_bash() -> None:
    subprocess.run(
        [
            "bash",
            "-n",
            str(SUBMIT_MATRIX),
            str(WATCH_MATRIX),
            str(LEGACY_SUBMIT_MATRIX),
            str(LEGACY_WATCH_MATRIX),
            str(MATRIX_SETTINGS_SCRIPT),
        ],
        cwd=ROOT,
        check=True,
    )


def test_matrix_source_run_is_explicit_absolute_and_help_is_asset_free() -> None:
    env = os.environ.copy()
    env.pop("TMAX_MATRIX_SOURCE_RUN", None)
    env.pop("TMAX_MATRIX_TOPOLOGY_SCOPE", None)

    missing = subprocess.run(
        ["bash", str(SUBMIT_MATRIX), "plan", "qwen35-4b-fidelity"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode != 0
    assert "set TMAX_MATRIX_SOURCE_RUN to the reviewed matrix data directory" in missing.stderr

    help_result = subprocess.run(
        ["bash", str(SUBMIT_MATRIX), "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "TMAX_MATRIX_SOURCE_RUN is required" in help_result.stdout
    assert "TMAX_MATRIX_TOPOLOGY_SCOPE=all|4n32|8n64" in help_result.stdout

    relative_env = {
        **env,
        "TMAX_MATRIX_SOURCE_RUN": "relative/matrix-data",
    }
    relative = subprocess.run(
        ["bash", str(SUBMIT_MATRIX), "plan", "qwen35-4b-fidelity"],
        cwd=ROOT,
        env=relative_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert relative.returncode == 2
    assert "TMAX_MATRIX_SOURCE_RUN must be an absolute path" in relative.stderr


def test_legacy_matrix_wrappers_delegate_to_generic_scripts() -> None:
    submit = LEGACY_SUBMIT_MATRIX.read_text()
    watch = LEGACY_WATCH_MATRIX.read_text()

    assert "SUBMIT_4NODE_MATRIX" in submit
    assert "SUBMIT_TMAX_MATRIX" in submit
    assert "export TMAX_MATRIX_TOPOLOGY_SCOPE=4n32" in submit
    assert "export TMAX_MATRIX_TOPOLOGY_SCOPE=4n32" in watch
    assert 'exec bash "${SCRIPT_DIR}/submit_matrix.sh" "$@"' in submit
    assert 'exec bash "${SCRIPT_DIR}/watch_matrix.sh" "$@"' in watch
    assert 'CONFIRM_TOKEN="SUBMIT_TMAX_MATRIX"' in SUBMIT_MATRIX.read_text()


def test_matrix_selector_scopes_all_and_rejects_cross_topology_settings() -> None:
    generic = _select_matrix_settings("all", "all")
    assert generic.returncode == 0, generic.stderr
    assert tuple(generic.stdout.splitlines()) == MATRIX_SETTINGS

    legacy = _select_matrix_settings("4n32", "all")
    assert legacy.returncode == 0, legacy.stderr
    assert tuple(legacy.stdout.splitlines()) == FOUR_NODE_SETTINGS

    eight_node = _select_matrix_settings("8n64", "all")
    assert eight_node.returncode == 0, eight_node.stderr
    assert tuple(eight_node.stdout.splitlines()) == EIGHT_NODE_SETTINGS

    rejected = _select_matrix_settings("4n32", EIGHT_NODE_SETTINGS[0])
    assert rejected.returncode == 2
    assert "outside 4n32 scope" in rejected.stderr

    invalid_selector = _select_matrix_settings("not-a-topology", "all")
    assert invalid_selector.returncode == 2
    assert "scope must be all, 4n32, or 8n64" in invalid_selector.stderr

    duplicate = _select_matrix_settings("all", MATRIX_SETTINGS[0], MATRIX_SETTINGS[0])
    assert duplicate.returncode == 2
    assert f"duplicate matrix setting: {MATRIX_SETTINGS[0]}" in duplicate.stderr

    invalid_watcher = subprocess.run(
        ["bash", str(WATCH_MATRIX)],
        cwd=ROOT,
        env={
            **os.environ,
            "TMAX_MATRIX_STAMP": "invalid-scope-test",
            "TMAX_MATRIX_TOPOLOGY_SCOPE": "not-a-topology",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert invalid_watcher.returncode == 2
    assert "scope must be all, 4n32, or 8n64" in invalid_watcher.stderr


def _matrix_test_environment(tmp_path: Path, *, stamp: str) -> dict[str, str]:
    source = tmp_path / "source"
    source.mkdir()
    train = source / "tmax-train.jsonl"
    holdout = source / "tmax_holdout-eval.jsonl"
    train.write_text("{}\n" * 14_498)
    holdout.write_text("{}\n" * 100)

    qwen4 = tmp_path / "qwen4"
    qwen9_hf = tmp_path / "qwen9-hf"
    qwen9_ref = tmp_path / "qwen9-ref"
    for path in (qwen4, qwen9_hf, qwen9_ref):
        path.mkdir()
    (qwen4 / "latest_checkpointed_iteration.txt").write_text("release\n")
    (qwen9_hf / "config.json").write_text("{}\n")
    (qwen9_ref / "latest_checkpointed_iteration.txt").write_text("release\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_marker = tmp_path / "sbatch-called"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        f"touch {shlex.quote(str(sbatch_marker))}\n"
        "printf '12345\n'\n"
    )
    fake_sbatch.chmod(0o755)

    return {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "POLAR_DATA_ROOT": str(tmp_path / "data"),
        "TMAX_MATRIX_SOURCE_RUN": str(source),
        "TMAX_MATRIX_TRAIN_SHA256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "TMAX_MATRIX_HOLDOUT_SHA256": hashlib.sha256(holdout.read_bytes()).hexdigest(),
        "TMAX_MATRIX_EVAL_BUNDLE_SHA256": hashlib.sha256(holdout.read_bytes()).hexdigest(),
        "TMAX_MATRIX_QWEN4_REF_LOAD": str(qwen4),
        "TMAX_MATRIX_QWEN9_HF_CHECKPOINT": str(qwen9_hf),
        "TMAX_MATRIX_QWEN9_REF_LOAD": str(qwen9_ref),
        "TMAX_MATRIX_STAMP": stamp,
    }


def test_matrix_plan_replay_preserves_all_effective_inputs(tmp_path: Path) -> None:
    env = {
        **_matrix_test_environment(tmp_path, stamp="replay-test"),
        "TMAX_MATRIX_TOPOLOGY_SCOPE": "4n32",
        "TMAX_MATRIX_PARTITIONS": "batch_short",
        "TMAX_MATRIX_EXCLUDE_NODES": "node-a,node-b",
        "TMAX_MATRIX_WANDB_PROJECT": "test-project",
        "TMAX_MATRIX_WANDB_GROUP": "test-group",
    }
    completed = subprocess.run(
        ["bash", str(SUBMIT_MATRIX), "plan", MATRIX_SETTINGS[0]],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    replay = next(line.strip() for line in completed.stdout.splitlines() if "TMAX_MATRIX_CONFIRM=" in line)
    tokens = shlex.split(replay)
    bash_index = tokens.index("bash")
    assignments = dict(token.split("=", 1) for token in tokens[:bash_index])
    for name in (
        "POLAR_DATA_ROOT",
        "TMAX_MATRIX_SOURCE_RUN",
        "TMAX_MATRIX_TRAIN_SHA256",
        "TMAX_MATRIX_HOLDOUT_SHA256",
        "TMAX_MATRIX_EVAL_BUNDLE_SHA256",
        "TMAX_MATRIX_TOPOLOGY_SCOPE",
        "TMAX_MATRIX_PARTITIONS",
        "TMAX_MATRIX_EXCLUDE_NODES",
        "TMAX_MATRIX_WANDB_PROJECT",
        "TMAX_MATRIX_WANDB_GROUP",
        "TMAX_MATRIX_QWEN4_REF_LOAD",
        "TMAX_MATRIX_QWEN9_HF_CHECKPOINT",
        "TMAX_MATRIX_QWEN9_REF_LOAD",
    ):
        assert assignments[name] == env[name]


def test_matrix_preflights_every_arm_before_first_submission(tmp_path: Path) -> None:
    stamp = "preflight-test"
    env = {
        **_matrix_test_environment(tmp_path, stamp=stamp),
        "TMAX_MATRIX_CONFIRM": "SUBMIT_TMAX_MATRIX",
    }
    blocked = Path(env["POLAR_DATA_ROOT"]) / "runs" / f"tmax-4n32-{MATRIX_SETTINGS[3]}-{stamp}"
    blocked.mkdir(parents=True)
    completed = subprocess.run(
        ["bash", str(SUBMIT_MATRIX), "submit", MATRIX_SETTINGS[0], MATRIX_SETTINGS[3]],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "refusing to reuse non-fresh run paths" in completed.stderr
    assert not (tmp_path / "sbatch-called").exists()


def test_tmax_matrix_uses_only_full65k_qwen9b_sweep_arms() -> None:
    submit = SUBMIT_MATRIX.read_text()

    for setting in MATRIX_SETTINGS:
        assert setting in submit
    for setting in SHORT_CAP_SETTINGS:
        assert setting not in submit
    assert "mt4k" not in submit
    assert "mt8k" not in submit

    baseline = _case_arm(submit, "qwen35-9b-baseline-a2-full65k")
    assert "configure_qwen9b 1e-6 2" in baseline
    assert "ROLLOUT_BATCH_SIZE" not in baseline
    assert "N_SAMPLES_PER_PROMPT" not in baseline

    batch16 = _case_arm(submit, "qwen35-9b-b16n16-a2-full65k")
    assert "configure_qwen9b 1e-6 2" in batch16
    assert "export ROLLOUT_BATCH_SIZE=16" in batch16
    assert "export N_SAMPLES_PER_PROMPT=16" in batch16
    assert "export GLOBAL_BATCH_SIZE=256" in batch16

    assert "configure_qwen9b 1e-6 4" in _case_arm(submit, "qwen35-9b-async4-full65k")
    assert "configure_qwen9b 5e-7 2" in _case_arm(submit, "qwen35-9b-lr5e7-a2-full65k")
    assert "configure_qwen9b 2e-6 2" in _case_arm(submit, "qwen35-9b-lr2e6-a2-full65k")


def test_matrix_restores_single_gateway_fresh_direct_apptainer() -> None:
    submit = SUBMIT_MATRIX.read_text()

    assert "export POLAR_MULTI_GATEWAY=0" in submit
    assert "export POLAR_APPTAINER_PERSISTENT_BROKER=0" in submit
    assert "export POLAR_APPTAINER_NO_INSTANCE=1" in submit
    assert "export POLAR_APPTAINER_DIRECT_EXEC_RETRIES=3" in submit
    assert "export POLAR_REQUEST_TIMEOUT=3600" in submit
    assert "export POLAR_TASK_TIMEOUT_FLOOR_SECONDS=1800" in submit
    assert "export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS=1200" in submit
    assert "export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS=600" in submit
    assert "export TMAX_TRAIN_TASK_TIMEOUT_RESERVE_SECONDS=600" in submit
    assert "export TMAX_MIN_RUN_WORKERS_PER_ROLLOUT_GPU=12" in submit
    assert "export TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU=8" in submit
    assert (
        'POLAR_MAX_POSTRUN_WORKERS="$((ROLLOUT_NUM_GPUS * TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU))"'
        in submit
    )
    assert (
        'expected_postrun_workers="$((ROLLOUT_NUM_GPUS * TMAX_MIN_POSTRUN_WORKERS_PER_ROLLOUT_GPU))"'
        in submit
    )
    assert (
        "4n single gateway; 8n one gateway per node; fresh direct Apptainer exec; no broker/instance; retries=3"
        in submit
    )
    assert 'export SLURM_EXCLUDE="${MATRIX_EXCLUDE_NODES}"' in submit
    assert 'echo "  exclude:     ${MATRIX_EXCLUDE_NODES:-none}"' in submit

    shared_submit = (ROOT / "examples" / "swegym_slime_grpo" / "submit_slurm.sh").read_text()
    assert 'SBATCH_EXCLUDE_ARG=(--exclude="${SLURM_EXCLUDE}")' in shared_submit
    assert '"${SBATCH_EXCLUDE_ARG[@]}"' in shared_submit
    assert 'SRUN_CMD+=(--exclude="${SLURM_EXCLUDE}")' in shared_submit


def test_matrix_keeps_two_full_trajectory_examples_every_ten_train_steps() -> None:
    submit = SUBMIT_MATRIX.read_text()

    assert "export POLAR_ROLLOUT_EXAMPLE_INTERVAL=10" in submit
    assert "export POLAR_ROLLOUT_EXAMPLE_COUNT=2" in submit
    assert "export POLAR_ROLLOUT_EXAMPLES_WANDB=1" in submit


def test_all_matrix_arms_match_released_full_trajectory_shape() -> None:
    submit = SUBMIT_MATRIX.read_text()
    run_state = (TMAX / "run_state.sh").read_text()
    qwen4b = _case_arm(submit, "qwen35-4b-fidelity")
    qwen4b_8n = _case_arm(submit, "qwen35-4b-fidelity-8n")

    expected_exports = (
        "export ACTOR_NUM_NODES=1",
        "export ACTOR_NUM_GPUS_PER_NODE=8",
        "export ACTOR_TENSOR_MODEL_PARALLEL_SIZE=4",
        "export CONTEXT_PARALLEL_SIZE=1",
        "export SEQUENCE_PARALLEL=1",
        "export ROLLOUT_NUM_GPUS=24",
        "export ROLLOUT_NUM_GPUS_PER_ENGINE=1",
        "export SEQ_LENGTH=67584",
        "export ROLLOUT_MAX_PROMPT_LEN=2048",
        "export ROLLOUT_MAX_RESPONSE_LEN=16384",
        "export TMAX_MAX_TOTAL_RESPONSE_LEN=65536",
        "export TMAX_TRAIN_PACK_LENGTH=67584",
        "export MAX_TOKENS_PER_GPU=67584",
        "export LOG_PROBS_CHUNK_SIZE=64",
        "export TMAX_ENABLE_FP32_LM_HEAD=1",
        "export SGLANG_ENABLE_FP32_LM_HEAD=1",
        "export CALCULATE_PER_TOKEN_LOSS=1",
    )
    for expected in expected_exports:
        assert expected in submit
    assert "LOG_PROBS_CHUNK_SIZE" in run_state

    # Model-specific arms inherit one common topology and token budget.
    assert "ACTOR_NUM_NODES" not in qwen4b
    assert "ROLLOUT_MAX_RESPONSE_LEN" not in qwen4b
    assert "export TMAX_ENABLE_FP32_LM_HEAD=0" in qwen4b
    assert "export SGLANG_ENABLE_FP32_LM_HEAD=0" in qwen4b
    assert "export TMAX_MIN_ASYNC_LEVEL=3" in qwen4b
    assert "export POLAR_MAX_ASYNC_LEVEL=3" in qwen4b
    assert "export ROLLOUT_BATCH_SIZE=8" in qwen4b
    assert "export N_SAMPLES_PER_PROMPT=8" in qwen4b
    assert "export NUM_STEPS_PER_ROLLOUT=1" in qwen4b
    assert "export GLOBAL_BATCH_SIZE=64" in qwen4b
    assert "export EVAL_GLOBAL_BATCH_SIZE=64" in qwen4b
    assert "export SAVE_INTERVAL=5" in qwen4b
    assert "export TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU=8" in qwen4b
    assert "export TMAX_TRAIN_AGENT_TIMEOUT_SECONDS=600" in qwen4b
    assert "export TMAX_CONCURRENT_PRETRAIN_EVAL=0" in qwen4b
    assert "export TRAIN_LR=5e-7" in qwen4b
    assert "export POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0.5" in qwen4b
    assert "export POLAR_EARLY_STOP_GRACE_SESSIONS=2" in qwen4b

    assert "export NUM_NODES=8" in qwen4b_8n
    assert "export ACTOR_NUM_NODES=2" in qwen4b_8n
    assert "export ROLLOUT_NUM_GPUS=48" in qwen4b_8n
    assert "export POLAR_MULTI_GATEWAY=1" in qwen4b_8n
    assert "export ROLLOUT_BATCH_SIZE=8" in qwen4b_8n
    assert "export N_SAMPLES_PER_PROMPT=16" in qwen4b_8n
    assert "export GLOBAL_BATCH_SIZE=128" in qwen4b_8n
    assert "export EVAL_GLOBAL_BATCH_SIZE=128" in qwen4b_8n
    assert "export TMAX_EVAL_INTERVAL=10" in qwen4b_8n
    assert "export TMAX_OVERRIDE_OPT_PARAM_SCHEDULER=1" in qwen4b_8n
    assert 'export LOAD_DIR="${MATRIX_QWEN4_LOAD_DIR}"' in qwen4b_8n
    assert "export TMAX_MIN_ASYNC_LEVEL=3" in qwen4b_8n
    assert "export POLAR_MAX_ASYNC_LEVEL=3" in qwen4b_8n
    assert "export TMAX_MIN_ACTIVE_SESSIONS_PER_ROLLOUT_GPU=8" in qwen4b_8n
    assert "export POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0.5" in qwen4b_8n
    assert "export POLAR_EARLY_STOP_GRACE_SESSIONS=4" in qwen4b_8n
    assert "export TRAIN_LR=5e-7" in qwen4b_8n
    assert "30-minute idle-GPU reaper" in qwen4b
    assert "tied 4B model must keep native LM-head precision" in submit
    assert "untied 9B arm requires matching FP32 LM heads" in submit
    assert 'pack_tokens="$((ROLLOUT_MAX_PROMPT_LEN + TMAX_MAX_TOTAL_RESPONSE_LEN))"' in submit
    assert 'trainer_token_capacity="$((MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE))"' in submit
    assert '"${TMAX_TRAIN_PACK_LENGTH}" -ne "${pack_tokens}"' in submit
    assert '"${SEQ_LENGTH}" -ne "${TMAX_TRAIN_PACK_LENGTH}"' in submit
    assert '"${trainer_token_capacity}" -lt "${pack_tokens}"' in submit
    assert (
        "tokens=turn${ROLLOUT_MAX_RESPONSE_LEN}/total_response${TMAX_MAX_TOTAL_RESPONSE_LEN}"
        in submit
    )
    assert "8 trainer TP4/CP1/DP2 + 24 TP1 rollout GPUs" in submit
    assert "16 trainer TP4/CP1/DP4 + 48 TP1 rollout GPUs" in submit
    assert '"$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))" -ne' in submit
    assert '"$((GLOBAL_BATCH_SIZE * NUM_STEPS_PER_ROLLOUT))"' in submit


def test_b16n8_trajectory_arm_is_bounded_without_leaking_to_old_arms() -> None:
    submit = SUBMIT_MATRIX.read_text()
    trajectory = _case_arm(submit, "qwen35-4b-fidelity-8n-b16n8-traj")
    existing_8n = _case_arm(submit, "qwen35-4b-fidelity-8n")
    reset_start = submit.index("reset_matrix_overrides() {")
    reset_end = submit.index("\n}", reset_start)
    reset = submit[reset_start:reset_end]

    for expected in (
        "export NUM_NODES=8",
        "export ACTOR_NUM_NODES=2",
        "export ROLLOUT_NUM_GPUS=48",
        "export POLAR_MULTI_GATEWAY=1",
        "export ROLLOUT_BATCH_SIZE=16",
        "export N_SAMPLES_PER_PROMPT=8",
        "export NUM_STEPS_PER_ROLLOUT=1",
        "export GLOBAL_BATCH_SIZE=128",
        "export EVAL_GLOBAL_BATCH_SIZE=128",
        "export CALCULATE_PER_TOKEN_LOSS=0",
        "export GRPO_STD_NORMALIZATION=0",
        "export TMAX_EVAL_INTERVAL=10",
        "export SAVE_INTERVAL=5",
        "export TRAIN_LR=5e-7",
        "export POLAR_EARLY_STOP_GRACE_SESSIONS=4",
        "export TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN=1",
    ):
        assert expected in trajectory

    assert 'MATRIX_NUM_ROLLOUT="${TMAX_MATRIX_NUM_ROLLOUT:-}"' in submit
    assert 'export TMAX_NUM_ROLLOUT="${MATRIX_NUM_ROLLOUT}"' in trajectory
    assert 'export TMAX_TARGET_ITER="$((TMAX_NUM_ROLLOUT - 1))"' in trajectory
    assert "TMAX_NUM_ROLLOUT TMAX_TARGET_ITER" in reset
    assert "TMAX_MATRIX_NUM_ROLLOUT" not in reset
    assert "TMAX_NUM_ROLLOUT" not in existing_8n
    assert "TMAX_TARGET_ITER" not in existing_8n
    assert "export TMAX_NUM_ROLLOUT=50" not in submit
    assert "TMAX_MATRIX_NUM_ROLLOUT=%q" in submit
    assert "export TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN=0" in submit
    assert "export TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN=1" not in existing_8n
    assert "resume_seed_eval=${TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN}" in submit


def test_8node_continuation_seed_and_afterok_dependency_are_auditable() -> None:
    submit = SUBMIT_MATRIX.read_text()
    shared_run = (ROOT / "examples" / "swegym_slime_grpo" / "run.sh").read_text()
    run_state = (TMAX / "run_state.sh").read_text()
    lifecycle = (TMAX / "lifecycle.sh").read_text()
    checkpoint_validator = (TMAX / "validate_torch_dist_checkpoint.py").read_text()

    assert 'MATRIX_QWEN4_LOAD_DIR="${TMAX_MATRIX_QWEN4_LOAD_DIR:-}"' in submit
    assert '"ERROR: Qwen3.5-4B continuation"' in submit
    assert "tmax_validate_numbered_checkpoint" in submit
    assert "common.pt" in lifecycle
    assert ".metadata" in lifecycle
    assert 'validate_torch_dist_checkpoint.py" "${model_dir}"' in lifecycle
    assert "expected_sizes = referenced_distcp_shard_sizes(metadata_path)" in checkpoint_validator
    assert (
        'present = {path.name for path in model_dir.glob("*.distcp")}'
        in checkpoint_validator
    )
    assert "actual_size != expected_size" in checkpoint_validator
    assert "global_dataset_state_dict_${value}.pt" in lifecycle
    assert "^afterok:[0-9]+(:[0-9]+)*$" in submit
    assert "afterany" not in submit
    assert (
        "eval_interval=${TMAX_EVAL_INTERVAL} resume_seed_eval=${TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN} scheduler_override=${TMAX_OVERRIDE_OPT_PARAM_SCHEDULER} load_dir=${LOAD_DIR:-release}"
        in submit
    )
    assert "TMAX_MATRIX_QWEN4_LOAD_DIR=%q" in submit
    assert "export TMAX_OVERRIDE_OPT_PARAM_SCHEDULER=0" in submit
    assert 'case "${TMAX_OVERRIDE_OPT_PARAM_SCHEDULER:-0}"' in shared_run
    assert "OPT_PARAM_SCHEDULER_ARGS=(--override-opt-param-scheduler)" in shared_run
    assert '"${OPT_PARAM_SCHEDULER_ARGS[@]}"' in shared_run
    assert "TMAX_OVERRIDE_OPT_PARAM_SCHEDULER" in run_state


def test_matrix_can_serialize_pretrain_eval_and_exports_logprob_chunk_size() -> None:
    submit = SUBMIT_MATRIX.read_text()
    run_state = (TMAX / "run_state.sh").read_text()
    shared = ROOT / "examples" / "swegym_slime_grpo"
    shared_run = (shared / "run.sh").read_text()
    shared_submit = (shared / "submit_slurm.sh").read_text()

    assert "export TMAX_CONCURRENT_PRETRAIN_EVAL=1" in submit
    assert "export TMAX_VALIDATE_EXISTING_ASSETS=0" in submit
    assert 'case "${TMAX_CONCURRENT_PRETRAIN_EVAL:-1}"' in shared_run
    assert "PRETRAIN_EVAL_ARGS=(--concurrent-pretrain-eval)" in shared_run
    assert "--eval-resumed-checkpoint-before-train" in shared_run
    assert '"${PRETRAIN_EVAL_ARGS[@]}"' in shared_run
    assert "SGLANG_ENABLE_DETERMINISTIC_INFERENCE" in run_state
    assert "SGLANG_ATTENTION_BACKEND" in run_state
    assert "TMAX_CONCURRENT_PRETRAIN_EVAL" in run_state
    assert "TMAX_EVAL_RESUMED_CHECKPOINT_BEFORE_TRAIN" in run_state
    assert "TMAX_VALIDATE_EXISTING_ASSETS" in run_state
    assert "LOG_PROBS_CHUNK_SIZE" in shared_submit

    tmax_submit = (TMAX / "submit_slurm.sh").read_text()
    tmax_run = (TMAX / "run.sh").read_text()
    assert 'TMAX_VALIDATE_EXISTING_ASSETS="${TMAX_VALIDATE_EXISTING_ASSETS:-1}"' in tmax_submit
    assert 'TMAX_VALIDATE_EXISTING_ASSETS="${TMAX_VALIDATE_EXISTING_ASSETS:-1}"' in tmax_run
    assert tmax_submit.count('[ "${TMAX_VALIDATE_EXISTING_ASSETS}" = "1" ]') >= 3
    assert tmax_run.count('[ "${TMAX_VALIDATE_EXISTING_ASSETS}" = "1" ]') >= 3


def test_matrix_uses_all_ready_complement_and_only_tmax_eval() -> None:
    submit = SUBMIT_MATRIX.read_text()
    run_state = (TMAX / "run_state.sh").read_text()

    assert "tmax-14598r-14498t100h-20260701T011143Z" not in submit
    assert 'MATRIX_SOURCE_RUN="${TMAX_MATRIX_SOURCE_RUN:?' in submit
    assert "TMAX_MATRIX_SOURCE_RUN=%q" in submit
    assert "TMAX_MATRIX_TRAIN_SHA256=%q" in submit
    assert "TMAX_MATRIX_HOLDOUT_SHA256=%q" in submit
    assert "TMAX_MATRIX_EVAL_BUNDLE_SHA256=%q" in submit
    assert 'verify_jsonl train "${MATRIX_SOURCE_RUN}/tmax-train.jsonl" 14498' in submit
    assert 'export TMAX_EXCLUDE_DATA="${TMAX_EVAL_DATA}"' in submit
    assert "export TMAX_MAX_TASKS=-1" in submit
    assert "export TMAX_TOTAL_TASKS=14601" in submit
    assert "export TMAX_ONLY_READY=1" in submit
    assert "export TMAX_REQUIRE_EXACT_TOTAL_TASKS=1" in submit
    assert "export TMAX_EXTERNAL_EVAL_ENABLED=0" in submit
    assert "verify_jsonl terminal_bench_2_0" not in submit
    assert "14,498 train (14,598 ready minus fixed 100)" in submit
    assert "TMAX_EXCLUDE_DATA" in run_state


def test_submit_and_watcher_matrix_setting_sets_match_exactly() -> None:
    shared = _bash_array(MATRIX_SETTINGS_SCRIPT.read_text(), "TMAX_MATRIX_SETTINGS")
    submit = SUBMIT_MATRIX.read_text()
    watch = WATCH_MATRIX.read_text()

    assert shared == MATRIX_SETTINGS
    assert len(shared) == len(set(shared))
    assert 'source "${SCRIPT_DIR}/matrix_settings.sh"' in submit
    assert 'source "${SCRIPT_DIR}/matrix_settings.sh"' in watch
    assert "readonly -a MATRIX_SETTINGS" not in submit
    assert "readonly -a ALL_SETTINGS" not in watch


def test_matrix_watcher_discovers_submitted_subset_for_older_stamps() -> None:
    watch = WATCH_MATRIX.read_text()

    assert 'TMAX_MATRIX_WATCH_SETTINGS:-}" = all' in watch
    assert 'SETTINGS+=("${setting}")' in watch
    assert "no submitted matrix run states found for stamp" in watch
    assert 'tmax_matrix_settings_for_scope SETTINGS "${MATRIX_TOPOLOGY_SCOPE}"' in watch
    assert 'tmax_matrix_settings_for_scope SCOPED_SETTINGS "${MATRIX_TOPOLOGY_SCOPE}"' in watch
    assert 'tmax_matrix_select_settings SETTINGS "${MATRIX_TOPOLOGY_SCOPE}"' in watch
    assert 'topology_tag="$(tmax_matrix_setting_topology "${setting}")"' in watch
    assert "matrix_run_id" in watch
