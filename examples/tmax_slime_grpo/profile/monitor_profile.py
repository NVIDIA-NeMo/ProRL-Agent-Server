#!/usr/bin/env python3
"""Continuously record Slurm state for a profiling/report dependency chain."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Sequence


TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
SUCCESS_STATES = {"COMPLETED"}
DEFAULT_SQUEUE = "/cm/shared/apps/slurm/current/bin/squeue"
DEFAULT_SACCT = "/cm/shared/apps/slurm/current/bin/sacct"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_job(value: str) -> tuple[str, str]:
    label, separator, job_id = value.partition("=")
    if not separator or not label or not job_id.isdigit() or int(job_id) <= 0:
        raise argparse.ArgumentTypeError("--job must be LABEL=NUMERIC_JOB_ID")
    return label, job_id


def _run(command: Sequence[str]) -> tuple[str, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return "", str(error)
    error = result.stderr.strip()
    if result.returncode != 0 and not error:
        error = f"command exited {result.returncode}"
    return result.stdout, error


def query_squeue(job_ids: Sequence[str], executable: str) -> tuple[dict[str, dict[str, str]], str]:
    stdout, error = _run(
        [
            executable,
            "-h",
            "-j",
            ",".join(job_ids),
            "-o",
            "%i|%T|%M|%R",
        ]
    )
    rows: dict[str, dict[str, str]] = {}
    for line in stdout.splitlines():
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        job_id, state, elapsed, reason = (part.strip() for part in parts)
        rows[job_id] = {
            "state": state.split("+", 1)[0],
            "elapsed": elapsed,
            "reason": reason,
            "exit_code": "",
            "source": "squeue",
        }
    return rows, error


def query_sacct(job_ids: Sequence[str], executable: str) -> tuple[dict[str, dict[str, str]], str]:
    stdout, error = _run(
        [
            executable,
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(job_ids),
            "--format=JobIDRaw,State,Elapsed,ExitCode",
        ]
    )
    rows: dict[str, dict[str, str]] = {}
    wanted = set(job_ids)
    for line in stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        job_id, raw_state, elapsed, exit_code = (part.strip() for part in parts[:4])
        if job_id not in wanted:
            continue
        rows[job_id] = {
            "state": raw_state.split()[0].split("+", 1)[0],
            "elapsed": elapsed,
            "reason": "",
            "exit_code": exit_code,
            "source": "sacct",
        }
    return rows, error


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def snapshot(
    jobs: Sequence[tuple[str, str]],
    *,
    squeue: str,
    sacct: str,
    previous: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    job_ids = [job_id for _, job_id in jobs]
    queue_rows, queue_error = query_squeue(job_ids, squeue)
    missing = [job_id for job_id in job_ids if job_id not in queue_rows]
    account_rows: dict[str, dict[str, str]] = {}
    account_error = ""
    if missing:
        account_rows, account_error = query_sacct(missing, sacct)
    result: dict[str, dict[str, Any]] = {}
    previous = previous or {}
    for label, job_id in jobs:
        row: dict[str, Any] = (
            queue_rows.get(job_id)
            or account_rows.get(job_id)
            or dict(previous.get(job_id) or {})
            or {
                "state": "UNKNOWN",
                "elapsed": "",
                "reason": "scheduler returned no record",
                "exit_code": "",
                "source": "none",
            }
        )
        state = str(row.get("state") or "UNKNOWN").split("+", 1)[0]
        row.update(
            {
                "label": label,
                "state": state,
                "terminal": state in TERMINAL_STATES,
                "success": state in SUCCESS_STATES,
            }
        )
        result[job_id] = row
    return {
        "generated_at": utc_now(),
        "jobs": result,
        "all_terminal": all(row["terminal"] for row in result.values()),
        "any_failed": any(row["terminal"] and not row["success"] for row in result.values()),
        "scheduler_errors": {"squeue": queue_error, "sacct": account_error},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", action="append", type=parse_job, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=300.0)
    parser.add_argument("--max-seconds", type=float, default=172800.0)
    parser.add_argument("--squeue", default=DEFAULT_SQUEUE)
    parser.add_argument("--sacct", default=DEFAULT_SACCT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.poll_seconds <= 0 or args.heartbeat_seconds <= 0 or args.max_seconds <= 0:
        raise SystemExit("poll, heartbeat, and max seconds must be positive")
    started = time.monotonic()
    last_heartbeat = -args.heartbeat_seconds
    previous: dict[str, dict[str, Any]] = {}
    previous_states: dict[str, str] = {}
    while True:
        report = snapshot(
            args.job,
            squeue=args.squeue,
            sacct=args.sacct,
            previous=previous,
        )
        elapsed = time.monotonic() - started
        report["monitor_started_at"] = dt.datetime.fromtimestamp(
            time.time() - elapsed,
            tz=dt.timezone.utc,
        ).isoformat()
        report["monitor_elapsed_seconds"] = round(elapsed, 3)
        atomic_json(args.status_json, report)
        previous = report["jobs"]

        states = {job_id: str(row["state"]) for job_id, row in previous.items()}
        changed = states != previous_states
        heartbeat = elapsed - last_heartbeat >= args.heartbeat_seconds
        if changed or heartbeat:
            summary = ", ".join(
                f"{row['label']}={row['state']}" for row in previous.values()
            )
            print(f"[{report['generated_at']}] {summary}", flush=True)
            last_heartbeat = elapsed
            previous_states = states

        if report["all_terminal"]:
            return 1 if report["any_failed"] else 0
        if elapsed >= args.max_seconds:
            print(
                f"[{utc_now()}] monitor time budget reached with nonterminal jobs",
                flush=True,
            )
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
