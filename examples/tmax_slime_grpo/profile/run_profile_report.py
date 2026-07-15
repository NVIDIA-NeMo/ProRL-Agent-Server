#!/usr/bin/env python3
"""Summarize two completed profile batches and render the final report bundle."""

from __future__ import annotations

import argparse
from collections import Counter
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

ArmSignature = tuple[str, int, int, int, int]
EXPECTED_MANIFEST_ARMS: dict[str, dict[str, ArmSignature]] = {
    "spilot_router": {
        "async-16t16r-l3": ("fully_async", 16, 16, 3, 32),
        "async-8t24r-l3": ("fully_async", 8, 24, 3, 32),
        "async-8t32r-l3": ("fully_async", 8, 32, 3, 40),
        "collocate-32shared": ("collocate", 32, 32, 1, 32),
    },
    "tmax": {
        "async-16t16r-l4": ("fully_async", 16, 16, 4, 32),
        "async-8t24r-l4": ("fully_async", 8, 24, 4, 32),
        "async-8t32r-l4": ("fully_async", 8, 32, 4, 40),
        "collocate-32shared": ("collocate", 32, 32, 1, 32),
    },
}
REQUIRED_REPORT_ARTIFACTS = (
    "spilot-summary.json",
    "spilot-summary.txt",
    "tmax-summary.json",
    "tmax-summary.txt",
    "gpu-allocation-report.md",
    "gpu-allocation-report.html",
    "gpu-allocation-analysis.json",
    "gpu-allocation-arms.csv",
    "gpu-role-utilization.csv",
    "report-inputs.json",
)
COMPLETE_MARKER = "REPORT_COMPLETE"
INCOMPLETE_MARKER = "REPORT_INCOMPLETE.json"


class ReportIncompleteError(RuntimeError):
    """Raised after diagnostic artifacts record an unmet completion gate."""


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _clear_previous_outputs(output_dir: Path) -> None:
    """Prevent artifacts from an earlier attempt satisfying the current gate."""

    for name in (*REQUIRED_REPORT_ARTIFACTS, COMPLETE_MARKER, INCOMPLETE_MARKER):
        (output_dir / name).unlink(missing_ok=True)


def _integer_field(row: dict[str, str], field: str) -> int:
    raw = str(row.get(field) or "").strip()
    if not raw.isdigit() or int(raw) <= 0:
        raise ValueError(f"{field}={raw!r} is not a positive integer")
    return int(raw)


def _manifest_signature(row: dict[str, str]) -> ArmSignature:
    mode = str(row.get("mode") or "").strip()
    # Launchers persist the historical spelling ``colocate`` in manifests;
    # profile_summary canonicalizes it to ``collocate`` in the analysis.
    if mode == "colocate":
        mode = "collocate"
    if mode not in {"fully_async", "collocate"}:
        raise ValueError(
            f"mode={mode!r} is neither fully_async nor colocate/collocate"
        )
    return (
        mode,
        _integer_field(row, "actor_gpus"),
        _integer_field(row, "rollout_gpus"),
        _integer_field(row, "async_level"),
        _integer_field(row, "allocated_gpus"),
    )


def _format_signature(signature: ArmSignature) -> str:
    mode, actor, rollout, async_level, allocated = signature
    return f"{mode}:{actor}t:{rollout}r:{allocated}g:l{async_level}"


def validate_manifest_contract(
    suite: str,
    rows: Sequence[dict[str, str]],
) -> dict[str, Any]:
    """Validate the exact four-arm final-report contract."""

    expected = EXPECTED_MANIFEST_ARMS[suite]
    errors: list[str] = []
    observed_arms = [str(row.get("arm") or "").strip() for row in rows]
    arm_counts = Counter(observed_arms)
    duplicate_arms = sorted(arm for arm, count in arm_counts.items() if count > 1)
    if len(rows) != len(expected):
        errors.append(
            f"manifest contains {len(rows)} rows; expected exactly {len(expected)}"
        )
    if duplicate_arms:
        errors.append(f"duplicate arm names: {', '.join(duplicate_arms)}")
    missing_arms = sorted(set(expected) - set(observed_arms))
    unexpected_arms = sorted(set(observed_arms) - set(expected))
    if missing_arms:
        errors.append(f"missing expected arms: {', '.join(missing_arms)}")
    if unexpected_arms:
        errors.append(f"unexpected arms: {', '.join(unexpected_arms)}")

    observed_signatures: list[str] = []
    parsed_signatures: list[ArmSignature] = []
    for index, row in enumerate(rows, start=1):
        arm = observed_arms[index - 1] or "<missing>"
        try:
            signature = _manifest_signature(row)
        except ValueError as exc:
            errors.append(f"row {index} ({arm}): {exc}")
            continue
        parsed_signatures.append(signature)
        observed_signatures.append(_format_signature(signature))
        expected_signature = expected.get(arm)
        if expected_signature is not None and signature != expected_signature:
            errors.append(
                f"arm {arm} has signature {_format_signature(signature)}; expected "
                f"{_format_signature(expected_signature)}"
            )

    signature_counts = Counter(parsed_signatures)
    duplicate_signatures = sorted(
        _format_signature(signature)
        for signature, count in signature_counts.items()
        if count > 1
    )
    if duplicate_signatures:
        errors.append(f"duplicate arm signatures: {', '.join(duplicate_signatures)}")

    return {
        "suite": suite,
        "valid": not errors,
        "expected_arms": list(expected),
        "observed_arms": observed_arms,
        "expected_signatures": [
            _format_signature(signature) for signature in expected.values()
        ],
        "observed_signatures": observed_signatures,
        "errors": errors,
    }


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


def _analysis_row_signature(row: dict[str, Any]) -> ArmSignature:
    mode = str(row.get("mode") or "").strip()
    if mode not in {"fully_async", "collocate"}:
        raise ValueError(f"mode={mode!r} is neither fully_async nor collocate")
    parsed: list[int] = []
    for field in ("actor_gpus", "rollout_gpus", "async_level", "allocated_gpus"):
        value = row.get(field)
        if isinstance(value, bool):
            raise ValueError(f"{field}={value!r} is not a positive integer")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}={value!r} is not a positive integer") from exc
        if integer <= 0 or str(value).strip() not in {str(integer), f"{integer}.0"}:
            raise ValueError(f"{field}={value!r} is not a positive integer")
        parsed.append(integer)
    actor, rollout, async_level, allocated = parsed
    return mode, actor, rollout, async_level, allocated


def evaluate_completion_gate(
    *,
    analysis: dict[str, Any],
    manifest_validation: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Return auditable evidence for whether the final report is complete."""

    reasons: list[str] = []
    for suite, validation in manifest_validation.items():
        for error in validation["errors"]:
            reasons.append(f"{suite} manifest: {error}")

    raw_suites = analysis.get("suites")
    suites = raw_suites if isinstance(raw_suites, list) else []
    suite_counts = Counter(
        str(suite.get("suite") or "") for suite in suites if isinstance(suite, dict)
    )
    expected_suite_names = set(EXPECTED_MANIFEST_ARMS)
    observed_suite_names = set(suite_counts)
    if observed_suite_names != expected_suite_names:
        reasons.append(
            "analysis suites differ from the required spilot_router/tmax pair: "
            f"observed={sorted(observed_suite_names)}"
        )
    duplicate_suites = sorted(name for name, count in suite_counts.items() if count > 1)
    if duplicate_suites:
        reasons.append(f"analysis contains duplicate suites: {', '.join(duplicate_suites)}")

    suite_evidence: dict[str, Any] = {}
    total_valid_arms = 0
    for suite_name, expected_arms in EXPECTED_MANIFEST_ARMS.items():
        matching = [
            suite
            for suite in suites
            if isinstance(suite, dict) and suite.get("suite") == suite_name
        ]
        if len(matching) != 1:
            suite_evidence[suite_name] = {
                "present_once": False,
                "valid_arm_count": 0,
                "expected_arm_count": len(expected_arms),
                "status": None,
                "candidate": None,
                "topology_valid": False,
            }
            continue
        suite = matching[0]
        raw_rows = suite.get("rows")
        rows = raw_rows if isinstance(raw_rows, list) else []
        valid_rows = [
            row for row in rows if isinstance(row, dict) and row.get("valid") is True
        ]
        total_valid_arms += len(valid_rows)
        if len(rows) != len(expected_arms):
            reasons.append(
                f"{suite_name} analysis contains {len(rows)} rows; "
                f"expected {len(expected_arms)}"
            )
        if len(valid_rows) != len(expected_arms):
            reasons.append(
                f"{suite_name} has {len(valid_rows)}/{len(expected_arms)} valid arms"
            )

        observed_signatures: list[ArmSignature] = []
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                reasons.append(f"{suite_name} analysis row {index} is not an object")
                continue
            try:
                observed_signatures.append(_analysis_row_signature(row))
            except ValueError as exc:
                reasons.append(f"{suite_name} analysis row {index}: {exc}")
        expected_signatures = set(expected_arms.values())
        observed_signature_set = set(observed_signatures)
        topology_valid = (
            len(observed_signatures) == len(expected_signatures)
            and len(observed_signature_set) == len(observed_signatures)
            and observed_signature_set == expected_signatures
        )
        if not topology_valid:
            reasons.append(
                f"{suite_name} analyzed topologies do not exactly match the four-arm contract"
            )

        status = suite.get("status")
        candidate = (
            suite.get("recommended_signature")
            if status == "measured"
            else suite.get("directional_candidate_signature")
            if status == "directional"
            else None
        )
        if status not in {"directional", "measured"} or not candidate:
            reasons.append(
                f"{suite_name} has no directional/measured candidate "
                f"(status={status!r})"
            )
        suite_evidence[suite_name] = {
            "present_once": True,
            "valid_arm_count": len(valid_rows),
            "expected_arm_count": len(expected_arms),
            "status": status,
            "candidate": candidate,
            "topology_valid": topology_valid,
            "observed_signatures": [
                _format_signature(signature) for signature in observed_signatures
            ],
        }

    if total_valid_arms != 8:
        reasons.append(f"analysis has {total_valid_arms}/8 valid arms")

    artifact_status = {
        name: (output_dir / name).is_file() and (output_dir / name).stat().st_size > 0
        for name in REQUIRED_REPORT_ARTIFACTS
    }
    missing_artifacts = sorted(
        name for name, present in artifact_status.items() if not present
    )
    if missing_artifacts:
        reasons.append(
            f"missing or empty report artifacts: {', '.join(missing_artifacts)}"
        )

    return {
        "complete": not reasons,
        "valid_arm_count": total_valid_arms,
        "required_valid_arm_count": 8,
        "suites": suite_evidence,
        "artifacts": artifact_status,
        "reasons": list(dict.fromkeys(reasons)),
    }


def _write_incomplete_marker(
    output_dir: Path,
    *,
    reasons: Sequence[str],
    stage: str,
    details: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / COMPLETE_MARKER).unlink(missing_ok=True)
    marker = output_dir / INCOMPLETE_MARKER
    _write_json(
        marker,
        {
            "status": "incomplete",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stage": stage,
            "reasons": list(reasons),
            "details": details or {},
        },
    )
    return marker


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_outputs(args.output_dir)
    spilot_rows, spilot_jobs = read_manifest(args.spilot_manifest, args.data_root)
    tmax_rows, tmax_jobs = read_manifest(args.tmax_manifest, args.data_root)
    manifest_validation = {
        "spilot_router": validate_manifest_contract("spilot_router", spilot_rows),
        "tmax": validate_manifest_contract("tmax", tmax_rows),
    }

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
        "manifest_validation": manifest_validation,
    }
    _write_json(args.output_dir / "report-inputs.json", inputs)
    analysis_path = args.output_dir / "gpu-allocation-analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if not isinstance(analysis, dict):
        raise ValueError(f"analysis is not a JSON object: {analysis_path}")
    completion_gate = evaluate_completion_gate(
        analysis=analysis,
        manifest_validation=manifest_validation,
        output_dir=args.output_dir,
    )
    inputs["completion_gate"] = completion_gate
    _write_json(args.output_dir / "report-inputs.json", inputs)
    if not completion_gate["complete"]:
        _write_incomplete_marker(
            args.output_dir,
            reasons=completion_gate["reasons"],
            stage="completion_gate",
            details=completion_gate,
        )
        raise ReportIncompleteError("; ".join(completion_gate["reasons"]))

    (args.output_dir / COMPLETE_MARKER).write_text(
        inputs["generated_at"] + "\n",
        encoding="utf-8",
    )
    (args.output_dir / INCOMPLETE_MARKER).unlink(missing_ok=True)
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
    result.add_argument("--log-timezone", default="UTC")
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
    try:
        build_bundle(args)
    except ReportIncompleteError as exc:
        print(f"profile report is incomplete: {exc}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, ValueError) as exc:
        _write_incomplete_marker(
            args.output_dir,
            reasons=[str(exc)],
            stage="generation",
        )
        print(f"profile report generation failed: {exc}", file=sys.stderr)
        return 1
    print(args.output_dir / "gpu-allocation-report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
