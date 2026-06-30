from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TMAX = ROOT / "examples" / "tmax_slime_grpo"
SUBMIT_MATRIX = TMAX / "submit_4node_matrix.sh"
WATCH_MATRIX = TMAX / "watch_4node_matrix.sh"

MATRIX_SETTINGS = (
    "qwen35-4b-fidelity",
    "qwen35-9b-baseline-a2-full65k",
    "qwen35-9b-b16n16-a2-full65k",
    "qwen35-9b-async4-full65k",
    "qwen35-9b-lr5e7-a2-full65k",
    "qwen35-9b-lr2e6-a2-full65k",
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


def test_four_node_matrix_scripts_remain_valid_bash() -> None:
    subprocess.run(
        ["bash", "-n", str(SUBMIT_MATRIX), str(WATCH_MATRIX)],
        cwd=ROOT,
        check=True,
    )


def test_four_node_matrix_uses_only_full65k_qwen9b_sweep_arms() -> None:
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
    assert "single gateway; fresh direct Apptainer exec; no broker/instance; retries=3" in submit


def test_matrix_keeps_two_full_trajectory_examples_every_ten_train_steps() -> None:
    submit = SUBMIT_MATRIX.read_text()

    assert "export POLAR_ROLLOUT_EXAMPLE_INTERVAL=10" in submit
    assert "export POLAR_ROLLOUT_EXAMPLE_COUNT=2" in submit
    assert "export POLAR_ROLLOUT_EXAMPLES_WANDB=1" in submit


def test_all_matrix_arms_match_released_full_trajectory_shape() -> None:
    submit = SUBMIT_MATRIX.read_text()
    run_state = (TMAX / "run_state.sh").read_text()
    qwen4b = _case_arm(submit, "qwen35-4b-fidelity")

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
    assert "export ROLLOUT_BATCH_SIZE=24" in qwen4b
    assert "export N_SAMPLES_PER_PROMPT=8" in qwen4b
    assert "export NUM_STEPS_PER_ROLLOUT=3" in qwen4b
    assert "export GLOBAL_BATCH_SIZE=64" in qwen4b
    assert "export EVAL_GLOBAL_BATCH_SIZE=64" in qwen4b
    assert "export TRAIN_LR=5e-7" in qwen4b
    assert "export POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0.5" in qwen4b
    assert "export POLAR_EARLY_STOP_GRACE_SESSIONS=2" in qwen4b
    assert "Slurm job 13241355" in qwen4b
    assert "tied 4B model must keep native LM-head precision" in submit
    assert "untied 9B model requires matching FP32 LM heads" in submit
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
    assert '"$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))" -ne' in submit
    assert '"$((GLOBAL_BATCH_SIZE * NUM_STEPS_PER_ROLLOUT))"' in submit


def test_matrix_watcher_discovers_submitted_subset_for_older_stamps() -> None:
    watch = WATCH_MATRIX.read_text()

    assert 'TMAX_MATRIX_WATCH_SETTINGS:-}" = all' in watch
    assert 'SETTINGS+=("${setting}")' in watch
    assert "no submitted matrix run states found for stamp" in watch
