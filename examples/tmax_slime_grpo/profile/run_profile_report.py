#!/usr/bin/env python3
"""Summarize two completed profile batches and render the final report bundle."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


PROFILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROFILE_DIR.parents[2]
SUMMARY_SCRIPT = (
    PROJECT_ROOT
    / "examples"
    / "spilot_router_slime_grpo"
    / "profile"
    / "profile_summary.py"
)
RECOMMEND_SCRIPT = PROFILE_DIR / "recommend_gpu_allocation.py"


def read_manifest(path: Path, data_root: Path) -> tuple[list[dict[str, str]], list[Path]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"profile manifest contains no arms: {path}")
    job_dirs: list[Path] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        run_id = str(row.get("run_id") or "")
        job_id = str(row.get("job_id") or "")
        if not run_id or not job_id.isdigit() or int(job_id) <= 0:
            raise ValueError(f"invalid run_id/job_id row in {path}: {row}")
        key = (run_id, job_id)
        if key in seen:
            raise ValueError(f"duplicate run_id/job_id row in {path}: {key}")
        seen.add(key)
        # Include job-<id> even when a job failed before creating the directory.
        # The common summarizer can then recover config from the parent run and
        # match external Slurm stdout/stderr by job id.
        job_dirs.append(data_root / "runs" / run_id / f"job-{job_id}")
    return rows, job_dirs


def run_checked(command: Sequence[str]) -> str:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    return result.stdout


def summarize(
    job_dirs: Sequence[Path],
    *,
    output_json: Path,
    output_table: Path,
    log_root: Path,
    log_timezone: str,
    warmup_steps: int,
) -> None:
    command = [
        sys.executable,
        str(SUMMARY_SCRIPT),
        *[str(path) for path in job_dirs],
        "--warmup-steps",
        str(warmup_steps),
        "--log-root",
        str(log_root),
        "--log-timezone",
        log_timezone,
        "--json",
        str(output_json),
    ]
    output_table.write_text(run_checked(command), encoding="utf-8")


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spilot_rows, spilot_jobs = read_manifest(args.spilot_manifest, args.data_root)
    tmax_rows, tmax_jobs = read_manifest(args.tmax_manifest, args.data_root)

    spilot_summary = args.output_dir / "spilot-summary.json"
    tmax_summary = args.output_dir / "tmax-summary.json"
    summarize(
        spilot_jobs,
        output_json=spilot_summary,
        output_table=args.output_dir / "spilot-summary.txt",
        log_root=args.log_root,
        log_timezone=args.log_timezone,
        warmup_steps=args.warmup_steps,
    )
    summarize(
        tmax_jobs,
        output_json=tmax_summary,
        output_table=args.output_dir / "tmax-summary.txt",
        log_root=args.log_root,
        log_timezone=args.log_timezone,
        warmup_steps=args.warmup_steps,
    )

    run_checked(
        [
            sys.executable,
            str(RECOMMEND_SCRIPT),
            "--suite",
            f"spilot_router={spilot_summary}",
            "--suite",
            f"tmax={tmax_summary}",
            "--expected-steps",
            str(args.expected_steps),
            "--warmup-steps",
            str(args.warmup_steps),
            "--markdown",
            str(args.output_dir / "gpu-allocation-report.md"),
            "--html",
            str(args.output_dir / "gpu-allocation-report.html"),
            "--json",
            str(args.output_dir / "gpu-allocation-analysis.json"),
            "--arm-csv",
            str(args.output_dir / "gpu-allocation-arms.csv"),
            "--gpu-role-csv",
            str(args.output_dir / "gpu-role-utilization.csv"),
        ]
    )
    inputs = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "expected_steps": args.expected_steps,
        "warmup_steps": args.warmup_steps,
        "log_timezone": args.log_timezone,
        "log_root": str(args.log_root),
        "spilot_manifest": str(args.spilot_manifest),
        "tmax_manifest": str(args.tmax_manifest),
        "spilot_rows": spilot_rows,
        "tmax_rows": tmax_rows,
    }
    (args.output_dir / "report-inputs.json").write_text(
        json.dumps(inputs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "REPORT_COMPLETE").write_text(
        inputs["generated_at"] + "\n",
        encoding="utf-8",
    )
    return inputs


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-root", required=True, type=Path)
    result.add_argument("--log-root", required=True, type=Path)
    result.add_argument("--spilot-manifest", required=True, type=Path)
    result.add_argument("--tmax-manifest", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--expected-steps", type=int, default=3)
    result.add_argument("--warmup-steps", type=int, default=1)
    result.add_argument("--log-timezone", default="America/Los_Angeles")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.expected_steps <= 0:
        parser().error("--expected-steps must be positive")
    if args.warmup_steps < 0 or args.warmup_steps >= args.expected_steps:
        parser().error("--warmup-steps must be non-negative and less than expected steps")
    for path, label in (
        (args.data_root, "data root"),
        (args.log_root, "log root"),
        (args.spilot_manifest, "SPilot manifest"),
        (args.tmax_manifest, "TMax manifest"),
    ):
        if not path.exists():
            parser().error(f"{label} does not exist: {path}")
    build_bundle(args)
    print(args.output_dir / "gpu-allocation-report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
