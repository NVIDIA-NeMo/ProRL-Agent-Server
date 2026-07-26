from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "tmax_slime_grpo" / "profile"))
import capture_slurm_terminal_evidence as capture  # noqa: E402


def test_parse_sacct_requires_exact_allocation_rows() -> None:
    output = (
        "101|COMPLETED|0:0|2026-07-15T01:00:00|2026-07-15T02:00:00|"
        "01:00:00|4|billing=32,cpu=512,gres/gpu=32,node=4\n"
        "102|FAILED|1:0|2026-07-15T01:00:00|2026-07-15T02:00:00|"
        "01:00:00|5|billing=40,cpu=640,gres/gpu=40,node=5\n"
    )

    jobs = capture.parse_sacct(output, ["101", "102"])

    assert jobs["101"]["allocated_gpus"] == 32
    assert jobs["101"]["allocated_nodes"] == 4
    assert jobs["102"]["state"] == "FAILED"


def test_parse_sacct_fails_closed_on_missing_job() -> None:
    output = (
        "101|COMPLETED|0:0|2026-07-15T01:00:00|2026-07-15T02:00:00|"
        "01:00:00|4|billing=32,gres/gpu=32,node=4\n"
    )

    with pytest.raises(ValueError, match="no allocation row for jobs: 102"):
        capture.parse_sacct(output, ["101", "102"])


def test_manifest_job_ids_are_unique(tmp_path: Path) -> None:
    manifests = []
    for name in ("a.tsv", "b.tsv"):
        path = tmp_path / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["job_id"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"job_id": "101"})
        manifests.append(path)

    with pytest.raises(ValueError, match="repeated across manifests"):
        capture._manifest_job_ids(manifests)
