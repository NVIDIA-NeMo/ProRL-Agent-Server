from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "tmax_slime_grpo" / "profile"))
import run_profile_report as report_runner  # noqa: E402


SPILOT_ARMS = [
    ("async-16t16r-l3", "fully_async", 16, 16, 3, 32),
    ("async-8t24r-l3", "fully_async", 8, 24, 3, 32),
    ("async-8t32r-l3", "fully_async", 8, 32, 3, 40),
    ("collocate-32shared", "colocate", 32, 32, 1, 32),
]
TMAX_ARMS = [
    ("async-16t16r-l4", "fully_async", 16, 16, 4, 32),
    ("async-8t24r-l4", "fully_async", 8, 24, 4, 32),
    ("async-8t32r-l4", "fully_async", 8, 32, 4, 40),
    ("collocate-32shared", "colocate", 32, 32, 1, 32),
]


def _write_run(
    data_root: Path,
    log_root: Path,
    run_id: str,
    job_id: str,
    arm: tuple[str, str, int, int, int, int],
    *,
    harness: str,
    steps: int = 3,
    terminal_status: str = "SUCCEEDED",
) -> None:
    arm_name, mode, actor_gpus, rollout_gpus, async_level, allocated_gpus = arm
    run = data_root / "runs" / run_id
    run.mkdir(parents=True)
    (run / "run_state.env").write_text(
        "\n".join(
            [
                f"export RUN_ID={run_id}",
                f"export TMAX_PROFILE_ARM={arm_name}",
                f"export TMAX_TRAIN_MODE={mode}",
                f"export POLAR_FULLY_ASYNC={'true' if mode == 'fully_async' else 'false'}",
                f"export POLAR_MAX_ASYNC_LEVEL={async_level}",
                f"export ACTOR_NUM_NODES={actor_gpus // 8}",
                "export ACTOR_NUM_GPUS_PER_NODE=8",
                f"export ROLLOUT_NUM_GPUS={rollout_gpus}",
                f"export NUM_NODES={allocated_gpus // 8}",
                "export SLURM_GPUS=8",
                "export GLOBAL_BATCH_SIZE=256",
                "export ROLLOUT_BATCH_SIZE=8",
                "export N_SAMPLES_PER_PROMPT=32",
                "export CONTEXT_PARALLEL_SIZE=1",
                "export MAX_TOKENS_PER_GPU=32768",
                "export TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP=0",
                "export TMAX_OPTIMIZER_CPU_OFFLOAD=0",
                "export POLAR_MIN_COMPLETE_ACCEPT_FRACTION=0.5",
                "export POLAR_EARLY_STOP_GRACE_SESSIONS=2",
                "export POLAR_AGENT_MODEL_NAME=Qwen/Qwen3.5-9B",
                f"export TMAX_AGENT_HARNESS={harness}",
                "export LOAD_DIR=/checkpoint/release",
                f"export TMAX_PRORL_GIT_COMMIT={'b' * 40}",
                f"export TMAX_SLIME_GIT_COMMIT={'c' * 40}",
                f"export TMAX_MEGATRON_GIT_COMMIT={'d' * 40}",
            ]
        )
        + "\n"
    )
    # Evaluation-disabled profile submissions persist the data digest in the
    # profile contract because the regular TMax submit path clears eval hashes.
    (run / "profile.env").write_text(
        f"export TMAX_TRAIN_DATA_SHA256={'a' * 64}\n"
    )
    lines: list[str] = []
    for step in range(steps):
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
        f"Ray job polar-{job_id} finished with status {terminal_status}\n"
    )


def _write_suite(
    data_root: Path,
    log_root: Path,
    manifest: Path,
    *,
    suite: str,
    arms: list[tuple[str, str, int, int, int, int]],
    first_job_id: int,
    steps: int = 3,
) -> None:
    rows: list[dict[str, str | int]] = []
    for offset, arm in enumerate(arms):
        job_id = str(first_job_id + offset)
        run_id = f"{suite}-run-{offset}"
        _write_run(
            data_root,
            log_root,
            run_id,
            job_id,
            arm,
            harness=suite,
            steps=steps,
        )
        arm_name, mode, actor, rollout, async_level, allocated = arm
        rows.append(
            {
                "run_id": run_id,
                "arm": arm_name,
                "repeat": 1,
                "mode": mode,
                "actor_gpus": actor,
                "rollout_gpus": rollout,
                "async_level": async_level,
                "allocated_gpus": allocated,
                "job_id": job_id,
            }
        )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _prepare_inputs(
    tmp_path: Path,
    *,
    steps: int = 3,
) -> tuple[Path, Path, Path, Path]:
    data_root = tmp_path / "data"
    log_root = data_root / "logs" / "slurm"
    log_root.mkdir(parents=True)
    spilot_manifest = tmp_path / "spilot.tsv"
    tmax_manifest = tmp_path / "tmax.tsv"
    _write_suite(
        data_root,
        log_root,
        spilot_manifest,
        suite="spilot_router",
        arms=SPILOT_ARMS,
        first_job_id=101,
        steps=steps,
    )
    _write_suite(
        data_root,
        log_root,
        tmax_manifest,
        suite="tmax",
        arms=TMAX_ARMS,
        first_job_id=201,
        steps=steps,
    )
    return data_root, log_root, spilot_manifest, tmax_manifest


def _run_args(
    data_root: Path,
    log_root: Path,
    spilot_manifest: Path,
    tmax_manifest: Path,
    output: Path,
    *,
    expected_steps: int = 3,
) -> list[str]:
    terminal_evidence = output.parent / f"{output.name}-slurm-accounting.json"
    jobs: dict[str, dict[str, str | int]] = {}
    for manifest in (spilot_manifest, tmax_manifest):
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in rows:
            job_id = row["job_id"]
            allocated_gpus = int(row["allocated_gpus"])
            jobs[job_id] = {
                "state": "COMPLETED",
                "exit_code": "0:0",
                "start": "2026-07-15T01:00:00",
                "end": "2026-07-15T02:00:00",
                "elapsed": "01:00:00",
                "allocated_nodes": allocated_gpus // 8,
                "allocated_gpus": allocated_gpus,
            }
    terminal_evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "captured_at_utc": "2026-07-15T02:00:01+00:00",
                "complete": True,
                "jobs": jobs,
            }
        )
        + "\n"
    )
    return [
        "--data-root",
        str(data_root),
        "--log-root",
        str(log_root),
        "--spilot-manifest",
        str(spilot_manifest),
        "--tmax-manifest",
        str(tmax_manifest),
        "--slurm-terminal-evidence",
        str(terminal_evidence),
        "--output-dir",
        str(output),
        "--expected-steps",
        str(expected_steps),
        "--log-timezone",
        "UTC",
    ]


def test_report_defaults_to_compute_node_utc() -> None:
    args = report_runner.parser().parse_args(
        [
            "--data-root",
            "/data",
            "--log-root",
            "/logs",
            "--spilot-manifest",
            "/spilot.tsv",
            "--tmax-manifest",
            "/tmax.tsv",
            "--slurm-terminal-evidence",
            "/terminal.json",
            "--output-dir",
            "/report",
        ]
    )
    assert args.log_timezone == "UTC"


def test_complete_gate_accepts_eight_valid_directional_arms(tmp_path: Path) -> None:
    data_root, log_root, spilot_manifest, tmax_manifest = _prepare_inputs(tmp_path)
    output = tmp_path / "report"

    result = report_runner.main(
        _run_args(data_root, log_root, spilot_manifest, tmax_manifest, output)
    )

    assert result == 0
    assert (output / "REPORT_COMPLETE").is_file()
    assert not (output / "REPORT_INCOMPLETE.json").exists()
    assert "Technical summary" in (output / "gpu-allocation-report.html").read_text()
    analysis = json.loads((output / "gpu-allocation-analysis.json").read_text())
    assert [suite["suite"] for suite in analysis["suites"]] == [
        "spilot_router",
        "tmax",
    ]
    assert analysis["measurement_contract"]["minimum_steady_steps_for_measured"] == 5
    for suite in analysis["suites"]:
        assert suite["status"] == "directional"
        assert suite["recommended_signature"] is None
        assert suite["directional_candidate_signature"] is not None
        assert len(suite["rows"]) == 4
        assert all(row["valid"] is True for row in suite["rows"])
    inputs = json.loads((output / "report-inputs.json").read_text())
    assert inputs["completion_gate"]["complete"] is True
    assert inputs["completion_gate"]["valid_arm_count"] == 8
    assert inputs["completion_gate"]["slurm_terminal_valid"] is True
    assert inputs["slurm_terminal_validation"]["valid"] is True
    assert all(item["valid"] for item in inputs["manifest_validation"].values())
    assert (output / "gpu-allocation-arms.csv").is_file()
    assert (output / "gpu-role-utilization.csv").is_file()
    assert (output / "slurm-terminal-evidence.json").is_file()


def test_slurm_failure_blocks_completion_even_when_ray_succeeds(tmp_path: Path) -> None:
    data_root, log_root, spilot_manifest, tmax_manifest = _prepare_inputs(tmp_path)
    output = tmp_path / "report"
    args = _run_args(data_root, log_root, spilot_manifest, tmax_manifest, output)
    evidence_path = Path(args[args.index("--slurm-terminal-evidence") + 1])
    evidence = json.loads(evidence_path.read_text())
    evidence["jobs"]["201"]["state"] = "FAILED"
    evidence["jobs"]["201"]["exit_code"] = "70:0"
    evidence_path.write_text(json.dumps(evidence) + "\n")

    result = report_runner.main(args)

    assert result == 1
    incomplete = json.loads((output / "REPORT_INCOMPLETE.json").read_text())
    assert any("Slurm job 201 state is FAILED" in reason for reason in incomplete["reasons"])
    assert any("Slurm job 201 exit code is 70:0" in reason for reason in incomplete["reasons"])


def test_manifest_must_contain_exactly_the_expected_four_arms(tmp_path: Path) -> None:
    data_root, log_root, spilot_manifest, tmax_manifest = _prepare_inputs(tmp_path)
    lines = spilot_manifest.read_text().splitlines()
    spilot_manifest.write_text("\n".join(lines[:-1]) + "\n")
    output = tmp_path / "report"

    result = report_runner.main(
        _run_args(data_root, log_root, spilot_manifest, tmax_manifest, output)
    )

    assert result == 1
    assert not (output / "REPORT_COMPLETE").exists()
    incomplete = json.loads((output / "REPORT_INCOMPLETE.json").read_text())
    assert incomplete["stage"] == "completion_gate"
    assert any("expected exactly 4" in reason for reason in incomplete["reasons"])
    assert any("missing expected arms" in reason for reason in incomplete["reasons"])
    # A strict completion failure still leaves the analysis and reader-facing
    # diagnostic report available for root-cause inspection.
    assert (output / "gpu-allocation-analysis.json").is_file()
    assert (output / "gpu-allocation-report.html").is_file()


def test_manifest_rejects_duplicate_topology_signature(tmp_path: Path) -> None:
    data_root, log_root, spilot_manifest, tmax_manifest = _prepare_inputs(tmp_path)
    with spilot_manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for field in ("mode", "actor_gpus", "rollout_gpus", "async_level", "allocated_gpus"):
        rows[1][field] = rows[0][field]
    with spilot_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "report"

    result = report_runner.main(
        _run_args(data_root, log_root, spilot_manifest, tmax_manifest, output)
    )

    assert result == 1
    incomplete = json.loads((output / "REPORT_INCOMPLETE.json").read_text())
    assert any("duplicate arm signatures" in reason for reason in incomplete["reasons"])
    assert any("has signature" in reason for reason in incomplete["reasons"])
    assert (output / "gpu-allocation-report.md").is_file()


def test_invalid_arm_keeps_diagnostics_but_blocks_completion(tmp_path: Path) -> None:
    data_root, log_root, spilot_manifest, tmax_manifest = _prepare_inputs(tmp_path)
    (log_root / "profile-201.err").write_text(
        "Ray job polar-201 finished with status FAILED\n"
    )
    output = tmp_path / "report"

    result = report_runner.main(
        _run_args(data_root, log_root, spilot_manifest, tmax_manifest, output)
    )

    assert result == 1
    incomplete = json.loads((output / "REPORT_INCOMPLETE.json").read_text())
    assert any("tmax has 3/4 valid arms" in reason for reason in incomplete["reasons"])
    assert any("analysis has 7/8 valid arms" in reason for reason in incomplete["reasons"])
    analysis = json.loads((output / "gpu-allocation-analysis.json").read_text())
    tmax = next(suite for suite in analysis["suites"] if suite["suite"] == "tmax")
    assert sum(row["valid"] is True for row in tmax["rows"]) == 3
    assert (output / "gpu-allocation-report.html").is_file()


def test_missing_required_report_artifact_blocks_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root, log_root, spilot_manifest, tmax_manifest = _prepare_inputs(tmp_path)
    monkeypatch.setattr(
        report_runner,
        "REQUIRED_REPORT_ARTIFACTS",
        (*report_runner.REQUIRED_REPORT_ARTIFACTS, "deliberately-missing.txt"),
    )
    output = tmp_path / "report"
    output.mkdir()
    # A stale artifact from an earlier attempt must not satisfy this run's
    # completion gate.
    (output / "deliberately-missing.txt").write_text("stale\n")

    result = report_runner.main(
        _run_args(data_root, log_root, spilot_manifest, tmax_manifest, output)
    )

    assert result == 1
    incomplete = json.loads((output / "REPORT_INCOMPLETE.json").read_text())
    assert any("deliberately-missing.txt" in reason for reason in incomplete["reasons"])
    assert not (output / "deliberately-missing.txt").exists()
    assert not (output / "REPORT_COMPLETE").exists()
