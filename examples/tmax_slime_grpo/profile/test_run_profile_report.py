from __future__ import annotations

import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_profile_report as report_runner  # noqa: E402


def _write_run(data_root: Path, log_root: Path, run_id: str, job_id: str) -> None:
    run = data_root / "runs" / run_id
    run.mkdir(parents=True)
    (run / "run_state.env").write_text(
        "\n".join(
            [
                f"export RUN_ID={run_id}",
                "export TMAX_PROFILE_ARM=async-16t16r-l4",
                "export TMAX_TRAIN_MODE=fully_async",
                "export POLAR_FULLY_ASYNC=true",
                "export POLAR_MAX_ASYNC_LEVEL=4",
                "export ACTOR_NUM_NODES=2",
                "export ACTOR_NUM_GPUS_PER_NODE=8",
                "export ROLLOUT_NUM_GPUS=16",
                "export NUM_NODES=4",
                "export SLURM_GPUS=8",
                "export GLOBAL_BATCH_SIZE=256",
                "export ROLLOUT_BATCH_SIZE=8",
                "export N_SAMPLES_PER_PROMPT=32",
                "export POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0.5",
                "export POLAR_EARLY_STOP_GRACE_SESSIONS=16",
                "export POLAR_AGENT_MODEL_NAME=Qwen/Qwen3.5-9B",
                "export TMAX_AGENT_HARNESS=mini_swe_agent",
                "export LOAD_DIR=/checkpoint/release",
                f"export TMAX_TRAIN_DATA_SHA256={'a' * 64}",
                f"export TMAX_PRORL_GIT_COMMIT={'b' * 40}",
                f"export TMAX_SLIME_GIT_COMMIT={'c' * 40}",
                f"export TMAX_MEGATRON_GIT_COMMIT={'d' * 40}",
            ]
        )
        + "\n"
    )
    lines: list[str] = []
    for step in range(3):
        lines.extend(
            [
                f"[2026-07-14 00:00:{10 + step:02d}] train - perf {step}: "
                "{'timing/train_wait_time': 2, 'timing/train_time': 8, "
                "'timing/step_time': 10}",
                f"[2026-07-14 00:00:{10 + step:02d}] rollout - perf {step}: "
                "{'polar/rollout_trainable_sessions': 16, "
                "'polar/rollout_trainable_session_fraction': 1, "
                "'polar/rollout_success_rate': 1, "
                "'polar/terminal_timeout_sessions': 0, "
                "'polar/terminal_error_sessions': 0, "
                "'polar/staleness/mean': 0, "
                "'timing/inference/e2e_ms_mean': 100, "
                "'polar/scheduler/completed_buffer': 0, "
                "'polar/scheduler/output_queue': 0, "
                "'polar/scheduler/deferred_queue': 0}",
            ]
        )
    (log_root / f"profile-{job_id}.out").write_text("\n".join(lines) + "\n")
    (log_root / f"profile-{job_id}.err").write_text(
        f"Ray job polar-{job_id} finished with status SUCCEEDED\n"
    )


def _write_manifest(path: Path, run_id: str, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "run_id\tarm\trepeat\tmode\tactor_gpus\trollout_gpus\t"
        "async_level\tallocated_gpus\tjob_id\n"
        f"{run_id}\tasync-16t16r-l4\t1\tfully_async\t16\t16\t4\t32\t{job_id}\n"
    )


def test_build_bundle_writes_html_and_audit_artifacts(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    log_root = data_root / "logs" / "slurm"
    log_root.mkdir(parents=True)
    _write_run(data_root, log_root, "spilot-run", "101")
    _write_run(data_root, log_root, "tmax-run", "102")
    spilot_manifest = tmp_path / "spilot.tsv"
    tmax_manifest = tmp_path / "tmax.tsv"
    _write_manifest(spilot_manifest, "spilot-run", "101")
    _write_manifest(tmax_manifest, "tmax-run", "102")
    output = tmp_path / "report"

    result = report_runner.main(
        [
            "--data-root",
            str(data_root),
            "--log-root",
            str(log_root),
            "--spilot-manifest",
            str(spilot_manifest),
            "--tmax-manifest",
            str(tmax_manifest),
            "--output-dir",
            str(output),
            "--log-timezone",
            "UTC",
        ]
    )

    assert result == 0
    assert (output / "REPORT_COMPLETE").is_file()
    assert "Technical summary" in (output / "gpu-allocation-report.html").read_text()
    analysis = json.loads((output / "gpu-allocation-analysis.json").read_text())
    assert [suite["suite"] for suite in analysis["suites"]] == [
        "spilot_router",
        "tmax",
    ]
    assert analysis["suites"][0]["rows"][0]["valid"] is True
    assert analysis["suites"][1]["rows"][0]["valid"] is True
    assert (output / "gpu-allocation-arms.csv").is_file()
    assert (output / "gpu-role-utilization.csv").is_file()
