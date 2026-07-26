#!/usr/bin/env python3
"""Capture exact Slurm allocation terminal evidence for profile manifests."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Sequence


DEFAULT_SACCT = Path("/cm/shared/apps/slurm/current/bin/sacct")
SACCT_FIELDS = (
    "JobIDRaw",
    "State",
    "ExitCode",
    "Start",
    "End",
    "Elapsed",
    "NNodes",
    "AllocTRES",
)


def _manifest_job_ids(paths: Sequence[Path]) -> list[str]:
    job_ids: list[str] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if not rows:
            raise ValueError(f"manifest contains no jobs: {path}")
        for row in rows:
            job_id = str(row.get("job_id") or "").strip()
            if not job_id.isdigit() or int(job_id) <= 0:
                raise ValueError(f"manifest has invalid job_id={job_id!r}: {path}")
            if job_id in job_ids:
                raise ValueError(f"job id {job_id} is repeated across manifests")
            job_ids.append(job_id)
    return job_ids


def _allocated_gpus(alloc_tres: str) -> int:
    values: dict[str, str] = {}
    for item in alloc_tres.split(","):
        key, separator, value = item.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    raw = values.get("gres/gpu")
    if raw is None or not raw.isdigit() or int(raw) <= 0:
        raise ValueError(f"AllocTRES has no positive gres/gpu count: {alloc_tres!r}")
    return int(raw)


def parse_sacct(output: str, expected_job_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    expected = set(expected_job_ids)
    jobs: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        if not raw_line.strip():
            continue
        fields = raw_line.split("|")
        if len(fields) != len(SACCT_FIELDS):
            raise ValueError(
                f"sacct row {line_number} has {len(fields)} fields; "
                f"expected {len(SACCT_FIELDS)}"
            )
        job_id, state, exit_code, start, end, elapsed, nodes, alloc_tres = fields
        job_id = job_id.strip()
        if job_id not in expected:
            raise ValueError(f"sacct returned unexpected allocation job {job_id!r}")
        if job_id in jobs:
            raise ValueError(f"sacct returned duplicate allocation job {job_id}")
        if not nodes.strip().isdigit() or int(nodes) <= 0:
            raise ValueError(f"sacct job {job_id} has invalid NNodes={nodes!r}")
        jobs[job_id] = {
            "state": state.strip().upper(),
            "exit_code": exit_code.strip(),
            "start": start.strip(),
            "end": end.strip(),
            "elapsed": elapsed.strip(),
            "allocated_nodes": int(nodes),
            "allocated_gpus": _allocated_gpus(alloc_tres.strip()),
            "alloc_tres": alloc_tres.strip(),
        }
    missing = sorted(expected - set(jobs), key=int)
    if missing:
        raise ValueError(f"sacct returned no allocation row for jobs: {', '.join(missing)}")
    return jobs


def capture(
    manifests: Sequence[Path],
    *,
    sacct: Path = DEFAULT_SACCT,
) -> dict[str, Any]:
    job_ids = _manifest_job_ids(manifests)
    command = [
        str(sacct),
        "--allocations",
        "--noheader",
        "--parsable2",
        "--jobs",
        ",".join(job_ids),
        "--format=" + ",".join(SACCT_FIELDS),
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        detail = " ".join(result.stderr.splitlines())
        raise RuntimeError(f"sacct failed ({result.returncode}): {detail}")
    jobs = parse_sacct(result.stdout, job_ids)
    complete = all(
        record["state"] == "COMPLETED" and record["exit_code"] == "0:0"
        for record in jobs.values()
    )
    return {
        "schema_version": 1,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "complete": complete,
        "manifests": [str(path.resolve()) for path in manifests],
        "job_ids": job_ids,
        "jobs": jobs,
    }


def _write_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", action="append", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--sacct", default=DEFAULT_SACCT, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        payload = capture(args.manifest, sacct=args.sacct)
        _write_atomic(args.output, payload)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Slurm terminal evidence capture failed: {exc}")
        return 1
    if not payload["complete"]:
        print(f"Slurm terminal evidence is not complete: {args.output}")
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
