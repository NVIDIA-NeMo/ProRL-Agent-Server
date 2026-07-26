#!/usr/bin/env python3
"""Publish a polished top-level HTML report from a strict 8-arm profile bundle.

The strict report job owns measurement and its fail-closed completion gate.  This
script is a deterministic presentation layer: it revalidates that gate, freezes
the strict inputs into a content-addressed read-only snapshot, creates one
canonical Data Analytics ``artifact.json``, and asks the packaged portable report
builder to produce the final self-contained HTML file.

No final output is touched unless all eight arms are strict-valid.  Any profile
below the strict analysis's measured-step threshold is described as directional.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence


PROFILE_DIR = Path(__file__).resolve().parent
DEFAULT_WORKSPACE_ROOT = PROFILE_DIR.parents[4]
FINALIZER_SCHEMA_VERSION = 2
FINALIZER_PATH = Path(__file__).resolve()
FINALIZER_REPOSITORY_PATH = FINALIZER_PATH.relative_to(DEFAULT_WORKSPACE_ROOT).as_posix()

SUITE_ORDER = ("spilot_router", "tmax")
SUITE_LABELS = {
    "spilot_router": "SPilot Router",
    "tmax": "Direct RL / TMax",
}
SUITE_SHORT_LABELS = {
    "spilot_router": "SPilot",
    "tmax": "Direct RL",
}

# label -> (mode, actor GPUs, rollout GPUs, async level, allocated GPUs)
EXPECTED_ARMS: dict[str, dict[str, tuple[str, int, int, int, int]]] = {
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

STRICT_REQUIRED_ARTIFACTS = (
    "spilot-summary.json",
    "spilot-summary.txt",
    "tmax-summary.json",
    "tmax-summary.txt",
    "gpu-allocation-report.md",
    "gpu-allocation-report.html",
    "gpu-allocation-analysis.json",
    "gpu-allocation-arms.csv",
    "gpu-role-utilization.csv",
    "slurm-terminal-evidence.json",
    "report-inputs.json",
)

SOURCE_SNAPSHOT_PREFIX = "gpu_allocation_profiling_source_snapshot_"
SOURCE_SNAPSHOT_MANIFEST = "source-snapshot-manifest.json"

FINAL_EVIDENCE = "gpu_allocation_profiling_final_evidence.json"
FINAL_ARTIFACT = "gpu_allocation_profiling_final_artifact.json"
FINAL_HTML = "gpu_allocation_profiling_final_report.html"
FINAL_RECEIPT = "gpu_allocation_profiling_final_delivery_receipt.json"
FINAL_COMPLETE = "gpu_allocation_profiling_final_complete.json"


class FinalizationError(RuntimeError):
    """Raised when a strict bundle is unsafe to publish as a final report."""


DeliveryRunner = Callable[[Path, Path], Mapping[str, Any]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizationError(message)


def _load_json_snapshot(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FinalizationError(f"{label} is not a JSON object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot_id(source_hashes: Mapping[str, str]) -> str:
    identity = {
        "schema_version": 1,
        "files": dict(sorted(source_hashes.items())),
    }
    return hashlib.sha256(_json_text(identity).encode("utf-8")).hexdigest()


def _validate_source_snapshot(
    snapshot_dir: Path,
    *,
    snapshot_id: str,
    source_hashes: Mapping[str, str],
) -> tuple[Path, str]:
    """Revalidate a content-addressed snapshot without consulting its live source."""

    _require(
        snapshot_dir.is_dir() and not snapshot_dir.is_symlink(),
        "source snapshot path is not a regular directory",
    )
    manifest_path = snapshot_dir / SOURCE_SNAPSHOT_MANIFEST
    _require(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        "source snapshot manifest is not a regular file",
    )
    manifest, manifest_sha256 = _load_json_snapshot(
        manifest_path, "source snapshot manifest"
    )
    _require(
        manifest.get("schema_version") == 1,
        "source snapshot manifest has an unsupported schema version",
    )
    _require(
        manifest.get("snapshot_id") == snapshot_id,
        "source snapshot manifest has the wrong content identity",
    )
    raw_files = manifest.get("files")
    _require(isinstance(raw_files, dict), "source snapshot manifest has no files object")
    _require(
        set(raw_files) == set(source_hashes),
        "source snapshot manifest file set differs from the validated strict bundle",
    )
    expected_entries = set(source_hashes) | {SOURCE_SNAPSHOT_MANIFEST}
    observed_entries = {path.name for path in snapshot_dir.iterdir()}
    _require(
        observed_entries == expected_entries,
        "source snapshot directory contains missing or unexpected entries",
    )
    for name, expected_sha256 in source_hashes.items():
        record = raw_files.get(name)
        _require(isinstance(record, dict), f"source snapshot record is invalid: {name}")
        _require(
            record.get("sha256") == expected_sha256,
            f"source snapshot manifest hash differs for {name}",
        )
        path = snapshot_dir / name
        _require(
            path.is_file() and not path.is_symlink() and _sha256(path) == expected_sha256,
            f"source snapshot content differs for {name}",
        )
        _require(
            record.get("size_bytes") == path.stat().st_size,
            f"source snapshot size differs for {name}",
        )
    return manifest_path, manifest_sha256


def materialize_source_snapshot(
    validated: Mapping[str, Any],
    output_dir: Path,
    workspace_root: Path,
) -> dict[str, Any]:
    """Atomically freeze validated report inputs into a content-addressed directory."""

    strict_dir = Path(validated["strict_dir"])
    source_hashes = dict(validated["source_hashes"])
    snapshot_id = _source_snapshot_id(source_hashes)
    snapshot_dir = output_dir / f"{SOURCE_SNAPSHOT_PREFIX}{snapshot_id}"
    if snapshot_dir.exists():
        manifest_path, manifest_sha256 = _validate_source_snapshot(
            snapshot_dir,
            snapshot_id=snapshot_id,
            source_hashes=source_hashes,
        )
        return {
            "snapshot_id": snapshot_id,
            "directory": snapshot_dir,
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha256,
            "source_hashes": source_hashes,
        }

    with tempfile.TemporaryDirectory(
        prefix=".gpu-profile-source-snapshot-", dir=output_dir
    ) as raw_temp:
        staged = Path(raw_temp) / snapshot_dir.name
        staged.mkdir()
        file_records: dict[str, dict[str, Any]] = {}
        for name, expected_sha256 in sorted(source_hashes.items()):
            source_path = strict_dir / name
            try:
                raw = source_path.read_bytes()
            except OSError as exc:
                raise FinalizationError(
                    f"cannot snapshot strict source {name}: {exc}"
                ) from exc
            observed_sha256 = hashlib.sha256(raw).hexdigest()
            _require(
                observed_sha256 == expected_sha256,
                f"strict source changed while creating snapshot: {name}",
            )
            destination = staged / name
            destination.write_bytes(raw)
            file_records[name] = {
                "sha256": expected_sha256,
                "size_bytes": len(raw),
            }
        manifest = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "source_strict_output_dir": _relative_path(
                strict_dir, workspace_root, "strict output directory"
            ),
            "strict_generated_at": validated["generated_at"],
            "files": file_records,
        }
        manifest_path = staged / SOURCE_SNAPSHOT_MANIFEST
        manifest_path.write_text(_json_text(manifest), encoding="utf-8")

        # A mutation before or during the copy fails closed. Once this check
        # passes, subsequent live-source changes are irrelevant because every
        # report source resolves to the immutable snapshot.
        for name, expected_sha256 in source_hashes.items():
            source_path = strict_dir / name
            _require(
                source_path.is_file() and _sha256(source_path) == expected_sha256,
                f"strict source changed while creating snapshot: {name}",
            )
        try:
            os.replace(staged, snapshot_dir)
        except FileExistsError:
            # Another finalizer may have published the same content identity.
            pass

    # Apply read-only permissions only after the atomic rename. Some managed
    # filesystems reject renaming a directory after its write bit is removed.
    for path in snapshot_dir.iterdir():
        path.chmod(0o444)
    snapshot_dir.chmod(0o555)

    manifest_path, manifest_sha256 = _validate_source_snapshot(
        snapshot_dir,
        snapshot_id=snapshot_id,
        source_hashes=source_hashes,
    )
    return {
        "snapshot_id": snapshot_id,
        "directory": snapshot_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "source_hashes": source_hashes,
    }


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise FinalizationError(f"{label} is not a finite number: {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"{label} is not a finite number: {value!r}") from exc
    if not math.isfinite(parsed):
        raise FinalizationError(f"{label} is not a finite number: {value!r}")
    return parsed


def _optional_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _number(value, label)


def _positive_int(value: Any, label: str) -> int:
    parsed = _number(value, label)
    if parsed <= 0 or not parsed.is_integer():
        raise FinalizationError(f"{label} is not a positive integer: {value!r}")
    return int(parsed)


def _nonnegative_int(value: Any, label: str) -> int:
    parsed = _number(value, label)
    if parsed < 0 or not parsed.is_integer():
        raise FinalizationError(f"{label} is not a non-negative integer: {value!r}")
    return int(parsed)


def _utc_iso(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalizationError(f"{label} is missing")
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise FinalizationError(f"{label} is not an ISO timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise FinalizationError(f"{label} has no timezone: {value!r}")
    normalized = parsed.astimezone(dt.timezone.utc).isoformat(timespec="seconds")
    return normalized.replace("+00:00", "Z")


def _relative_path(path: Path, workspace_root: Path, label: str) -> str:
    try:
        relative = path.resolve().relative_to(workspace_root.resolve())
    except ValueError as exc:
        raise FinalizationError(
            f"{label} must be inside workspace root {workspace_root}: {path}"
        ) from exc
    _require(".." not in relative.parts, f"unsafe relative path for {label}: {relative}")
    return relative.as_posix()


def _row_signature(row: Mapping[str, Any], suite: str, index: int) -> tuple[str, int, int, int, int]:
    prefix = f"{suite} analysis row {index}"
    mode = str(row.get("mode") or "")
    _require(mode in {"fully_async", "collocate"}, f"{prefix} has invalid mode {mode!r}")
    return (
        mode,
        _positive_int(row.get("actor_gpus"), f"{prefix}.actor_gpus"),
        _positive_int(row.get("rollout_gpus"), f"{prefix}.rollout_gpus"),
        _positive_int(row.get("async_level"), f"{prefix}.async_level"),
        _positive_int(row.get("allocated_gpus"), f"{prefix}.allocated_gpus"),
    )


def _candidate_signature(suite: Mapping[str, Any]) -> str:
    status = suite.get("status")
    if status == "measured":
        value = suite.get("recommended_signature")
    elif status == "directional":
        value = suite.get("directional_candidate_signature")
    else:
        value = None
    _require(isinstance(value, str) and bool(value), f"suite {suite.get('suite')} has no candidate")
    return value


def _validate_slurm_terminal_snapshot(
    payload: Mapping[str, Any],
    normalized_suites: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Revalidate normalized allocation-level Slurm evidence for the exact eight jobs."""

    _require(
        payload.get("schema_version") == 1,
        "Slurm terminal evidence has an unsupported schema version",
    )
    _require(payload.get("valid") is True, "Slurm terminal evidence is not valid")
    _require(payload.get("errors") == [], "Slurm terminal evidence contains errors")
    raw_jobs = payload.get("jobs")
    _require(isinstance(raw_jobs, dict), "Slurm terminal evidence has no jobs object")
    expected: dict[str, dict[str, Any]] = {}
    for suite_record in normalized_suites:
        suite = str(suite_record["suite"])
        for row in suite_record["rows"]:
            job_id = str(row["job_id"])
            allocated_gpus = _positive_int(
                row.get("allocated_gpus"), f"{suite}.{row.get('label')}.allocated_gpus"
            )
            _require(
                allocated_gpus % 8 == 0,
                f"{suite}.{row.get('label')} allocation is not divisible by 8 GPUs/node",
            )
            expected[job_id] = {
                "suite": suite,
                "arm": str(row["label"]),
                "allocated_gpus": allocated_gpus,
                "allocated_nodes": allocated_gpus // 8,
            }
    _require(len(expected) == 8, "Slurm terminal evidence contract is not eight jobs")
    _require(
        {str(job_id) for job_id in raw_jobs} == set(expected),
        "Slurm terminal evidence job ids differ from the strict eight-arm analysis",
    )
    _require(
        _positive_int(
            payload.get("expected_job_count"),
            "Slurm terminal evidence expected_job_count",
        )
        == 8,
        "Slurm terminal evidence does not expect eight jobs",
    )
    _require(
        _positive_int(
            payload.get("observed_job_count"),
            "Slurm terminal evidence observed_job_count",
        )
        == 8,
        "Slurm terminal evidence does not contain eight observed jobs",
    )
    normalized_jobs: dict[str, dict[str, Any]] = {}
    for job_id, expected_record in expected.items():
        record = raw_jobs.get(job_id)
        _require(isinstance(record, dict), f"Slurm terminal job {job_id} is not an object")
        _require(
            str(record.get("suite") or "") == expected_record["suite"]
            and str(record.get("arm") or "") == expected_record["arm"],
            f"Slurm terminal job {job_id} has the wrong suite/arm binding",
        )
        _require(
            str(record.get("state") or "").upper() == "COMPLETED",
            f"Slurm terminal job {job_id} is not COMPLETED",
        )
        _require(
            str(record.get("exit_code") or "") == "0:0",
            f"Slurm terminal job {job_id} does not have exit code 0:0",
        )
        for field in ("allocated_gpus", "allocated_nodes"):
            _require(
                _positive_int(record.get(field), f"Slurm terminal job {job_id}.{field}")
                == expected_record[field],
                f"Slurm terminal job {job_id} has the wrong {field}",
            )
        for field in ("start", "end", "elapsed"):
            _require(
                isinstance(record.get(field), str) and bool(record[field].strip()),
                f"Slurm terminal job {job_id} has no {field} accounting",
            )
        normalized_jobs[job_id] = dict(record)
    return {
        "schema_version": 1,
        "valid": True,
        "captured_at_utc": payload.get("captured_at_utc"),
        "expected_job_count": 8,
        "observed_job_count": 8,
        "jobs": normalized_jobs,
        "errors": [],
    }


def validate_strict_bundle(strict_dir: Path) -> dict[str, Any]:
    """Validate the strict 8/8 bundle and return normalized report inputs."""

    strict_dir = strict_dir.resolve()
    _require(strict_dir.is_dir(), f"strict output directory does not exist: {strict_dir}")
    complete_path = strict_dir / "REPORT_COMPLETE"
    incomplete_path = strict_dir / "REPORT_INCOMPLETE.json"
    _require(complete_path.is_file() and complete_path.stat().st_size > 0, "REPORT_COMPLETE is missing")
    _require(not incomplete_path.exists(), "REPORT_INCOMPLETE.json is present")
    for name in STRICT_REQUIRED_ARTIFACTS:
        path = strict_dir / name
        _require(path.is_file() and path.stat().st_size > 0, f"strict artifact is missing or empty: {name}")

    inputs, inputs_sha256 = _load_json_snapshot(
        strict_dir / "report-inputs.json", "strict report inputs"
    )
    analysis, analysis_sha256 = _load_json_snapshot(
        strict_dir / "gpu-allocation-analysis.json", "strict analysis"
    )
    generated_at = _utc_iso(analysis.get("generated_at"), "analysis.generated_at")
    _utc_iso(inputs.get("generated_at"), "report-inputs.generated_at")
    _require(inputs.get("log_timezone") == "UTC", "strict report did not use UTC log timestamps")
    try:
        marker_bytes = complete_path.read_bytes()
        marker = marker_bytes.decode("utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise FinalizationError(f"cannot read REPORT_COMPLETE: {exc}") from exc
    _require(marker == str(inputs.get("generated_at") or ""), "REPORT_COMPLETE does not match report-inputs.generated_at")

    expected_steps = _positive_int(inputs.get("expected_steps"), "report-inputs.expected_steps")
    warmup_steps = _nonnegative_int(inputs.get("warmup_steps"), "report-inputs.warmup_steps")
    _require(warmup_steps < expected_steps, "warmup_steps must be less than expected_steps")
    _require(analysis.get("expected_steps") == expected_steps, "analysis expected_steps differs from report inputs")
    _require(analysis.get("warmup_steps") == warmup_steps, "analysis warmup_steps differs from report inputs")
    steady_steps = expected_steps - warmup_steps

    gate = inputs.get("completion_gate")
    _require(isinstance(gate, dict), "report-inputs has no completion_gate object")
    _require(gate.get("complete") is True, "strict completion gate is not complete")
    _require(gate.get("valid_arm_count") == 8, "strict completion gate does not have 8 valid arms")
    _require(gate.get("required_valid_arm_count") == 8, "strict completion gate does not require 8 valid arms")
    _require(
        gate.get("slurm_terminal_valid") is True,
        "strict completion gate did not validate Slurm terminal evidence",
    )
    _require(gate.get("reasons") == [], "strict completion gate contains failure reasons")
    artifact_gate = gate.get("artifacts")
    _require(isinstance(artifact_gate, dict), "strict completion gate has no artifact inventory")
    for name in STRICT_REQUIRED_ARTIFACTS:
        _require(artifact_gate.get(name) is True, f"strict gate did not validate artifact {name}")

    manifest_validation = inputs.get("manifest_validation")
    _require(isinstance(manifest_validation, dict), "manifest validation evidence is missing")
    gate_suites = gate.get("suites")
    _require(isinstance(gate_suites, dict), "completion gate suite evidence is missing")
    for suite_name in SUITE_ORDER:
        manifest = manifest_validation.get(suite_name)
        _require(isinstance(manifest, dict), f"manifest validation is missing {suite_name}")
        _require(manifest.get("valid") is True, f"{suite_name} manifest is not valid")
        _require(manifest.get("errors") == [], f"{suite_name} manifest has errors")
        suite_gate = gate_suites.get(suite_name)
        _require(isinstance(suite_gate, dict), f"completion gate is missing {suite_name}")
        _require(suite_gate.get("present_once") is True, f"{suite_name} is not present exactly once")
        _require(suite_gate.get("topology_valid") is True, f"{suite_name} topology gate failed")
        _require(suite_gate.get("valid_arm_count") == 4, f"{suite_name} does not have four valid arms")
        _require(suite_gate.get("expected_arm_count") == 4, f"{suite_name} gate does not require four arms")
        _require(suite_gate.get("status") in {"directional", "measured"}, f"{suite_name} has no publishable status")
        _require(bool(suite_gate.get("candidate")), f"{suite_name} gate has no candidate")

    raw_suites = analysis.get("suites")
    _require(isinstance(raw_suites, list), "analysis.suites is not a list")
    suite_names = [suite.get("suite") for suite in raw_suites if isinstance(suite, dict)]
    _require(suite_names == list(SUITE_ORDER), f"analysis suite order/content is invalid: {suite_names}")

    normalized_suites: list[dict[str, Any]] = []
    job_ids: set[str] = set()
    for suite_name, suite in zip(SUITE_ORDER, raw_suites):
        _require(isinstance(suite, dict), f"analysis suite {suite_name} is not an object")
        status = suite.get("status")
        _require(status in {"directional", "measured"}, f"{suite_name} has status {status!r}")
        raw_rows = suite.get("rows")
        _require(isinstance(raw_rows, list) and len(raw_rows) == 4, f"{suite_name} does not have exactly four rows")
        rows_by_label: dict[str, dict[str, Any]] = {}
        fingerprints: set[str] = set()
        for index, raw_row in enumerate(raw_rows, start=1):
            _require(isinstance(raw_row, dict), f"{suite_name} row {index} is not an object")
            row = dict(raw_row)
            label = str(row.get("label") or "")
            _require(label in EXPECTED_ARMS[suite_name], f"{suite_name} has unexpected arm label {label!r}")
            _require(label not in rows_by_label, f"{suite_name} repeats arm label {label}")
            _require(_row_signature(row, suite_name, index) == EXPECTED_ARMS[suite_name][label], f"{suite_name} arm {label} has the wrong topology")
            _require(row.get("valid") is True, f"{suite_name} arm {label} is not strict-valid")
            _require(row.get("job_status") == "SUCCEEDED", f"{suite_name} arm {label} did not finish SUCCEEDED")
            _require(not row.get("exclusion_reasons"), f"{suite_name} arm {label} has exclusion reasons")
            _require(_positive_int(row.get("step_count_total"), f"{suite_name}.{label}.step_count_total") == expected_steps, f"{suite_name} arm {label} has the wrong total step count")
            _require(_positive_int(row.get("step_count_steady"), f"{suite_name}.{label}.step_count_steady") == steady_steps, f"{suite_name} arm {label} has the wrong steady step count")
            for field in (
                "steps_per_hour",
                "trainable_sessions_per_second",
                "trainable_sessions_per_gpu_hour",
                "throughput_relative_to_best",
                "efficiency_relative_to_best",
            ):
                value = _number(row.get(field), f"{suite_name}.{label}.{field}")
                _require(value > 0, f"{suite_name} arm {label} has non-positive {field}")
            wait = _number(row.get("weighted_wait_ratio"), f"{suite_name}.{label}.weighted_wait_ratio")
            success = _number(row.get("rollout_success_rate"), f"{suite_name}.{label}.rollout_success_rate")
            _require(0 <= wait <= 1, f"{suite_name} arm {label} has invalid wait ratio")
            _require(0 <= success <= 1, f"{suite_name} arm {label} has invalid rollout success rate")
            _require(_number(row.get("staleness_mean"), f"{suite_name}.{label}.staleness_mean") >= 0, f"{suite_name} arm {label} has negative staleness")
            _require(_number(row.get("terminal_timeout_sessions_mean"), f"{suite_name}.{label}.terminal_timeout_sessions_mean") >= 0, f"{suite_name} arm {label} has negative timeout count")
            _require(_number(row.get("terminal_error_sessions_mean"), f"{suite_name}.{label}.terminal_error_sessions_mean") >= 0, f"{suite_name} arm {label} has negative error count")
            coverage = _optional_number(
                row.get("gpu_coverage_fraction"),
                f"{suite_name}.{label}.gpu_coverage_fraction",
            )
            publishable = row.get("utilization_publishable") is True
            if publishable:
                _require(
                    coverage is not None and coverage >= 0.90,
                    f"{suite_name} arm {label} publishes GPU utilization below 90% coverage",
                )
            else:
                _require(
                    all(
                        row.get(field) is None
                        for field in (
                            "gpu_util_pct",
                            "actor_gpu_util_pct",
                            "rollout_gpu_util_pct",
                            "shared_gpu_util_pct",
                        )
                    ),
                    f"{suite_name} arm {label} exposes GPU utilization without the 90% coverage gate",
                )
            fingerprint = row.get("comparability_fingerprint")
            _require(isinstance(fingerprint, str) and bool(fingerprint), f"{suite_name} arm {label} has no comparability fingerprint")
            fingerprints.add(fingerprint)
            job_id = str(row.get("job_id") or "")
            _require(job_id.isdigit() and int(job_id) > 0, f"{suite_name} arm {label} has invalid job id")
            _require(job_id not in job_ids, f"duplicate job id in analysis: {job_id}")
            job_ids.add(job_id)
            rows_by_label[label] = row
        _require(set(rows_by_label) == set(EXPECTED_ARMS[suite_name]), f"{suite_name} arm set is incomplete")
        _require(len(fingerprints) == 1, f"{suite_name} arms have different comparability fingerprints")
        candidate_signature = _candidate_signature(suite)
        matching_candidates = [row for row in rows_by_label.values() if row.get("signature") == candidate_signature]
        _require(len(matching_candidates) == 1, f"{suite_name} candidate does not resolve to exactly one arm")
        candidate = matching_candidates[0]
        _require(gate_suites[suite_name].get("candidate") == candidate_signature, f"{suite_name} gate candidate differs from analysis")
        normalized_suites.append(
            {
                "suite": suite_name,
                "status": status,
                "analysis": suite,
                "rows": [rows_by_label[label] for label in EXPECTED_ARMS[suite_name]],
                "candidate": candidate,
            }
        )

    contract = analysis.get("measurement_contract")
    _require(isinstance(contract, dict), "analysis measurement contract is missing")
    minimum_measured = _positive_int(
        contract.get("minimum_steady_steps_for_measured"),
        "measurement_contract.minimum_steady_steps_for_measured",
    )
    evidence_strength = (
        "directional"
        if steady_steps < minimum_measured
        or any(suite["status"] == "directional" for suite in normalized_suites)
        else "measured"
    )
    if steady_steps < minimum_measured:
        _require(
            all(suite["status"] == "directional" for suite in normalized_suites),
            "a suite claims measured status with fewer than the required steady steps",
        )

    slurm_terminal_raw, slurm_terminal_sha256 = _load_json_snapshot(
        strict_dir / "slurm-terminal-evidence.json",
        "normalized Slurm terminal evidence",
    )
    slurm_terminal_validation = _validate_slurm_terminal_snapshot(
        slurm_terminal_raw,
        normalized_suites,
    )
    _require(
        inputs.get("slurm_terminal_validation") == slurm_terminal_raw,
        "report inputs and normalized Slurm terminal evidence differ",
    )

    campaign_path = strict_dir / "campaign.json"
    _require(
        campaign_path.is_file() and campaign_path.stat().st_size > 0,
        "authoritative campaign.json is missing",
    )
    campaign, campaign_sha256 = _load_json_snapshot(
        campaign_path, "profiling campaign metadata"
    )
    _require(
        campaign.get("authoritative") is True,
        "profiling campaign metadata is not authoritative",
    )
    campaign_suites = campaign.get("suites")
    _require(
        isinstance(campaign_suites, dict),
        "profiling campaign metadata has no suites object",
    )
    analysis_bindings = {
        suite_record["suite"]: {
            str(row["label"]): str(row["job_id"])
            for row in suite_record["rows"]
        }
        for suite_record in normalized_suites
    }
    campaign_bindings: dict[str, dict[str, str]] = {}
    campaign_job_ids: set[str] = set()
    for campaign_suite, campaign_suite_record in campaign_suites.items():
        _require(
            isinstance(campaign_suite_record, dict),
            f"profiling campaign suite {campaign_suite} is not an object",
        )
        campaign_jobs = campaign_suite_record.get("jobs")
        _require(
            isinstance(campaign_jobs, dict) and bool(campaign_jobs),
            f"profiling campaign suite {campaign_suite} has no jobs",
        )
        arm_labels = set(campaign_jobs)
        matching_suites = [
            suite_name
            for suite_name, expected_arms in EXPECTED_ARMS.items()
            if arm_labels == set(expected_arms)
        ]
        _require(
            len(matching_suites) == 1,
            f"profiling campaign suite {campaign_suite} has an invalid arm set",
        )
        logical_suite = matching_suites[0]
        _require(
            logical_suite not in campaign_bindings,
            f"profiling campaign repeats logical suite {logical_suite}",
        )
        logical_bindings: dict[str, str] = {}
        for arm_label, raw_job_id in campaign_jobs.items():
            campaign_job_id = str(raw_job_id)
            _require(
                campaign_job_id.isdigit() and int(campaign_job_id) > 0,
                f"profiling campaign job {campaign_suite}/{arm_label} is invalid",
            )
            _require(
                campaign_job_id not in campaign_job_ids,
                f"profiling campaign repeats job id {campaign_job_id}",
            )
            campaign_job_ids.add(campaign_job_id)
            logical_bindings[str(arm_label)] = campaign_job_id
        campaign_bindings[logical_suite] = logical_bindings
    _require(
        campaign_job_ids == job_ids,
        "profiling campaign job ids differ from the strict eight-arm analysis",
    )
    _require(
        campaign_bindings == analysis_bindings,
        "profiling campaign arm-to-job bindings differ from the strict analysis",
    )
    experiment_contract = campaign.get("experiment_contract")
    _require(
        isinstance(experiment_contract, dict),
        "profiling campaign metadata has no experiment_contract object",
    )
    has_dependencies = experiment_contract.get("profile_jobs_have_dependencies")
    _require(
        isinstance(has_dependencies, bool),
        "profiling campaign dependency policy is missing",
    )
    requested = experiment_contract.get("requested_topology")
    _require(
        isinstance(requested, dict),
        "profiling campaign requested topology is missing",
    )
    requested_jobs = _positive_int(
        requested.get("profile_jobs"), "campaign requested profile jobs"
    )
    requested_gpus = _positive_int(
        requested.get("total_gpus"), "campaign requested GPUs"
    )
    requested_nodes = _positive_int(
        requested.get("total_nodes"), "campaign requested nodes"
    )
    analysis_gpus = sum(
        int(row["allocated_gpus"])
        for suite_record in normalized_suites
        for row in suite_record["rows"]
    )
    _require(requested_jobs == len(job_ids), "campaign requested job count is not eight")
    _require(requested_gpus == analysis_gpus, "campaign requested GPU count differs from analysis")
    snapshot = campaign.get("snapshot")
    _require(isinstance(snapshot, dict), "profiling campaign snapshot is missing")
    if not has_dependencies:
        authoritative_jobs = _positive_int(
            snapshot.get("authoritative_concurrent_profile_jobs"),
            "campaign authoritative concurrent profile jobs",
        )
        authoritative_gpus = _positive_int(
            snapshot.get("authoritative_concurrent_profile_gpus"),
            "campaign authoritative concurrent profile GPUs",
        )
        authoritative_nodes = _positive_int(
            snapshot.get("authoritative_concurrent_profile_nodes"),
            "campaign authoritative concurrent profile nodes",
        )
        _require(
            (authoritative_jobs, authoritative_gpus, authoritative_nodes)
            == (requested_jobs, requested_gpus, requested_nodes),
            "campaign authoritative concurrency snapshot differs from requested topology",
        )
        recorded_jobs = _positive_int(
            snapshot.get("total_concurrent_profile_jobs"),
            "campaign recorded concurrent profile jobs",
        )
        recorded_gpus = _positive_int(
            snapshot.get("total_concurrent_profile_gpus"),
            "campaign recorded concurrent profile GPUs",
        )
        overlap_ids = (
            snapshot.get("overlapping_preliminary_jobs")
            if "overlapping_preliminary_jobs" in snapshot
            else snapshot.get("overlapping_preliminary_r4_jobs")
        )
        _require(
            isinstance(overlap_ids, list),
            "campaign overlapping preliminary jobs is not a list",
        )
        normalized_overlap_ids: set[str] = set()
        for raw_overlap_id in overlap_ids:
            overlap_id = str(raw_overlap_id)
            _require(
                overlap_id.isdigit() and int(overlap_id) > 0,
                "campaign contains an invalid overlapping preliminary job id",
            )
            _require(
                overlap_id not in normalized_overlap_ids
                and overlap_id not in campaign_job_ids,
                f"campaign repeats overlapping job id {overlap_id}",
            )
            normalized_overlap_ids.add(overlap_id)
        raw_overlap_gpus = (
            snapshot.get("overlapping_preliminary_gpus")
            if "overlapping_preliminary_gpus" in snapshot
            else snapshot.get("overlapping_preliminary_r4_gpus")
        )
        overlap_gpus = _nonnegative_int(
            raw_overlap_gpus,
            "campaign overlapping preliminary GPUs",
        )
        _require(
            recorded_jobs == authoritative_jobs + len(overlap_ids),
            "campaign concurrency snapshot job count does not reconcile with recorded overlap",
        )
        _require(
            recorded_gpus == authoritative_gpus + overlap_gpus,
            "campaign concurrency snapshot GPU count does not reconcile with recorded overlap",
        )

    source_hashes = {
        name: _sha256(strict_dir / name)
        for name in (*STRICT_REQUIRED_ARTIFACTS, "REPORT_COMPLETE")
    }
    source_hashes["report-inputs.json"] = inputs_sha256
    source_hashes["gpu-allocation-analysis.json"] = analysis_sha256
    source_hashes["slurm-terminal-evidence.json"] = slurm_terminal_sha256
    source_hashes["REPORT_COMPLETE"] = hashlib.sha256(marker_bytes).hexdigest()
    source_hashes["campaign.json"] = campaign_sha256

    return {
        "strict_dir": strict_dir,
        "inputs": inputs,
        "analysis": analysis,
        "source_hashes": source_hashes,
        "generated_at": generated_at,
        "expected_steps": expected_steps,
        "warmup_steps": warmup_steps,
        "steady_steps": steady_steps,
        "minimum_measured_steps": minimum_measured,
        "evidence_strength": evidence_strength,
        "suites": normalized_suites,
        "slurm_terminal_validation": slurm_terminal_validation,
        "campaign": campaign,
    }


def _arm_name(row: Mapping[str, Any]) -> str:
    if row.get("mode") == "collocate":
        return f"{row.get('allocated_gpus')} shared GPUs (collocate)"
    return (
        f"{row.get('actor_gpus')} train / {row.get('rollout_gpus')} rollout "
        f"(fully async, L{row.get('async_level')})"
    )


def _chart_arm_name(suite_name: str, row: Mapping[str, Any]) -> str:
    prefix = SUITE_SHORT_LABELS[suite_name]
    if row.get("mode") == "collocate":
        return f"{prefix} · {row.get('allocated_gpus')} shared"
    return f"{prefix} · {row.get('actor_gpus')}/{row.get('rollout_gpus')} async"


def _short_chart_arm_name(suite_name: str, row: Mapping[str, Any]) -> str:
    prefix = "SP" if suite_name == "spilot_router" else "RL"
    if row.get("mode") == "collocate":
        return f"{prefix} shared"
    return f"{prefix} {row.get('actor_gpus')}/{row.get('rollout_gpus')}"


def _gpu_fraction(value: Any, label: str) -> float | None:
    parsed = _optional_number(value, label)
    return None if parsed is None else parsed / 100.0


def _round_text(value: Any, digits: int = 2) -> str:
    return f"{_number(value, 'display metric'):.{digits}f}"


def _execution_context(campaign: Mapping[str, Any]) -> dict[str, Any]:
    contract = campaign.get("experiment_contract")
    _require(isinstance(contract, dict), "profiling campaign experiment contract is invalid")
    has_dependencies = contract.get("profile_jobs_have_dependencies")
    _require(isinstance(has_dependencies, bool), "profiling campaign dependency policy is invalid")
    requested = contract.get("requested_topology")
    if not isinstance(requested, dict):
        requested = {}
    snapshot = campaign.get("snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {}

    def optional_positive_int(field: str, source: Mapping[str, Any]) -> int | None:
        value = source.get(field)
        return None if value is None else _positive_int(value, f"campaign.{field}")

    recorded_jobs = optional_positive_int("total_concurrent_profile_jobs", snapshot)
    recorded_gpus = optional_positive_int("total_concurrent_profile_gpus", snapshot)
    authoritative_jobs = optional_positive_int(
        "authoritative_concurrent_profile_jobs", snapshot
    )
    authoritative_gpus = optional_positive_int(
        "authoritative_concurrent_profile_gpus", snapshot
    )
    overlap_gpus = (
        snapshot.get("overlapping_preliminary_gpus")
        if "overlapping_preliminary_gpus" in snapshot
        else snapshot.get("overlapping_preliminary_r4_gpus")
    )
    if overlap_gpus is not None:
        overlap_gpus = _nonnegative_int(
            overlap_gpus, "campaign.overlapping_preliminary_gpus"
        )
    overlap_raw = (
        snapshot.get("overlapping_preliminary_jobs")
        if "overlapping_preliminary_jobs" in snapshot
        else snapshot.get("overlapping_preliminary_r4_jobs")
    ) or []
    _require(
        isinstance(overlap_raw, list),
        "campaign overlapping preliminary jobs is not a list",
    )
    overlap_ids: list[str] = []
    for raw_job_id in overlap_raw:
        job_id = str(raw_job_id)
        _require(
            job_id.isdigit() and int(job_id) > 0,
            "campaign contains an invalid overlapping preliminary job id",
        )
        overlap_ids.append(job_id)

    if not has_dependencies:
        concurrency_detail = (
            f" The recorded concurrency snapshot contained {recorded_jobs} profile jobs "
            f"using {recorded_gpus} GPUs"
            if recorded_jobs is not None and recorded_gpus is not None
            else ""
        )
        overlap_detail = (
            f", including {len(overlap_ids)} retained preliminary jobs"
            if overlap_ids
            else ""
        )
        summary = (
            "All eight authoritative arms were submitted independently with no job dependencies."
            f"{concurrency_detail}{overlap_detail}."
        )
        limitation = (
            "The authoritative arms ran concurrently rather than in isolated windows."
            f"{concurrency_detail}{overlap_detail}; shared provider quotas, network paths, "
            "filesystem/I/O, and cluster load can depress absolute rates or affect arms unevenly."
        )
        repeat_control = (
            "Repeat the leading arms without preliminary-job overlap and with balanced, isolated "
            "provider/network/filesystem capacity; record exact scheduler overlap."
        )
        mode = "concurrent"
        limitation_heading = "Concurrent-load confounding"
    else:
        summary = (
            "The campaign metadata records job dependencies, so arms were not guaranteed to run "
            "as one fully concurrent wave."
        )
        limitation = (
            "Dependency/order effects are confounded with provider, network, filesystem, and "
            "time-of-day conditions."
        )
        repeat_control = (
            "Repeat the leading arms in counterbalanced order and record exact scheduler overlap."
        )
        mode = "dependency_ordered"
        limitation_heading = "Dependency/order confounding"

    return {
        "mode": mode,
        "limitation_heading": limitation_heading,
        "profile_jobs_have_dependencies": has_dependencies,
        "scheduler_policy": contract.get("scheduler_policy"),
        "requested_profile_jobs": optional_positive_int("profile_jobs", requested),
        "requested_gpus": optional_positive_int("total_gpus", requested),
        "requested_nodes": optional_positive_int("total_nodes", requested),
        "authoritative_concurrent_profile_jobs": authoritative_jobs,
        "authoritative_concurrent_profile_gpus": authoritative_gpus,
        "recorded_concurrent_profile_jobs": recorded_jobs,
        "recorded_concurrent_profile_gpus": recorded_gpus,
        # Compatibility aliases for existing evidence consumers. These values
        # are a recorded scheduler snapshot, not a claim of an audited peak.
        "peak_concurrent_profile_jobs": recorded_jobs,
        "peak_concurrent_profile_gpus": recorded_gpus,
        "overlapping_preliminary_gpus": overlap_gpus,
        "overlapping_preliminary_job_ids": overlap_ids,
        "summary": summary,
        "limitation": limitation,
        "repeat_control": repeat_control,
    }


def build_evidence(
    validated: Mapping[str, Any],
    workspace_root: Path,
    *,
    structural_only: bool = False,
) -> dict[str, Any]:
    snapshot_info = validated.get("source_snapshot")
    if isinstance(snapshot_info, Mapping):
        source_dir = Path(snapshot_info["directory"])
        source_snapshot = {
            "snapshot_id": str(snapshot_info["snapshot_id"]),
            "directory": _relative_path(
                source_dir, workspace_root, "source snapshot directory"
            ),
            "manifest": _relative_path(
                Path(snapshot_info["manifest_path"]),
                workspace_root,
                "source snapshot manifest",
            ),
            "manifest_sha256": str(snapshot_info["manifest_sha256"]),
        }
    else:
        # Direct evidence-construction tests may operate on the validated live
        # bundle. finalize_bundle always supplies a content-addressed snapshot.
        source_dir = Path(validated["strict_dir"])
        source_snapshot = None
    source_hashes = dict(validated["source_hashes"])
    suites: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    for suite_record in validated["suites"]:
        suite_name = suite_record["suite"]
        candidate = suite_record["candidate"]
        analysis_suite = suite_record["analysis"]
        suite_rows: list[dict[str, Any]] = []
        for row in suite_record["rows"]:
            arm = {
                "suite": suite_name,
                "suite_label": SUITE_LABELS[suite_name],
                "arm": row["label"],
                "arm_name": _arm_name(row),
                "chart_label": _chart_arm_name(suite_name, row),
                "short_chart_label": _short_chart_arm_name(suite_name, row),
                "job_id": str(row["job_id"]),
                "mode": row["mode"],
                "actor_gpus": row["actor_gpus"],
                "rollout_gpus": row["rollout_gpus"],
                "async_level": row["async_level"],
                "allocated_gpus": row["allocated_gpus"],
                "steady_steps": row["step_count_steady"],
                "candidate": row["signature"] == candidate["signature"],
                "trainable_sessions_per_second": row["trainable_sessions_per_second"],
                "trainable_sessions_per_gpu_hour": row["trainable_sessions_per_gpu_hour"],
                "trainable_tokens_per_second": row.get("trainable_tokens_per_second"),
                "trainable_tokens_per_gpu_hour": row.get("trainable_tokens_per_gpu_hour"),
                "throughput_relative_to_best": row["throughput_relative_to_best"],
                "efficiency_relative_to_best": row["efficiency_relative_to_best"],
                "weighted_wait_ratio": row["weighted_wait_ratio"],
                "staleness_mean": row["staleness_mean"],
                "rollout_success_rate": row["rollout_success_rate"],
                "inference_e2e_ms_mean": row.get("inference_e2e_ms_mean"),
                "gpu_coverage_fraction": row.get("gpu_coverage_fraction"),
                "utilization_publishable": row.get("utilization_publishable") is True,
                "gpu_utilization_fraction": _gpu_fraction(
                    row.get("gpu_util_pct"), f"{suite_name}.{row['label']}.gpu_util_pct"
                ),
                "actor_gpu_utilization_fraction": _gpu_fraction(
                    row.get("actor_gpu_util_pct"),
                    f"{suite_name}.{row['label']}.actor_gpu_util_pct",
                ),
                "rollout_gpu_utilization_fraction": _gpu_fraction(
                    row.get("rollout_gpu_util_pct"),
                    f"{suite_name}.{row['label']}.rollout_gpu_util_pct",
                ),
                "shared_gpu_utilization_fraction": _gpu_fraction(
                    row.get("shared_gpu_util_pct"),
                    f"{suite_name}.{row['label']}.shared_gpu_util_pct",
                ),
                "terminal_timeout_sessions_mean": row["terminal_timeout_sessions_mean"],
                "terminal_error_sessions_mean": row["terminal_error_sessions_mean"],
                "accepted_group_fraction": row.get("accepted_group_fraction"),
                "bottleneck": row.get("bottleneck"),
            }
            suite_rows.append(arm)
            arm_rows.append(arm)
        suites.append(
            {
                "suite": suite_name,
                "suite_label": SUITE_LABELS[suite_name],
                "status": suite_record["status"],
                "candidate_arm": candidate["label"],
                "candidate_name": _arm_name(candidate),
                "candidate_signature": candidate["signature"],
                "candidate_sessions_per_second": candidate["trainable_sessions_per_second"],
                "candidate_sessions_per_gpu_hour": candidate["trainable_sessions_per_gpu_hour"],
                "candidate_weighted_wait_ratio": candidate["weighted_wait_ratio"],
                "candidate_staleness_mean": candidate["staleness_mean"],
                "candidate_allocated_gpus": candidate["allocated_gpus"],
                "fastest_arm": analysis_suite.get("fastest_label"),
                "most_efficient_arm": analysis_suite.get("most_efficient_label"),
                "decision_blockers": list(analysis_suite.get("decision_blockers") or []),
                "comparability": analysis_suite.get("comparability") or {},
                "arms": suite_rows,
            }
        )
    execution = _execution_context(validated["campaign"])
    campaign = validated["campaign"]
    campaign_contract = campaign.get("experiment_contract")
    _require(
        isinstance(campaign_contract, Mapping),
        "profiling campaign experiment contract is invalid",
    )
    campaign_context = {
        "campaign_id": campaign.get("campaign_id"),
        "supersedes_campaign": campaign.get("supersedes_campaign"),
        "reason_for_supersession": campaign.get("reason_for_supersession"),
        "model": campaign_contract.get("model"),
        "checkpoint_tracker": campaign_contract.get("checkpoint_tracker"),
        "start_rollout_id": campaign_contract.get("start_rollout_id"),
        "max_tokens_per_gpu": campaign_contract.get("max_tokens_per_gpu"),
        "context_parallel_size": campaign_contract.get("context_parallel_size"),
        "allow_single_sample_over_token_cap": campaign_contract.get(
            "allow_single_sample_over_token_cap"
        ),
        "optimizer_cpu_offload": campaign_contract.get("optimizer_cpu_offload"),
        "oversize_policy": campaign_contract.get("oversize_policy"),
    }
    return {
        "schema_version": FINALIZER_SCHEMA_VERSION,
        "report_kind": "final_gpu_allocation_profile",
        "generated_at": validated["generated_at"],
        "presentation_provenance": {
            "finalizer_path": FINALIZER_REPOSITORY_PATH,
            "finalizer_sha256": _sha256(FINALIZER_PATH),
            "artifact_schema_version": FINALIZER_SCHEMA_VERSION,
        },
        "presentation_qa": (
            {
                "verification_mode": "structural_only",
                "verification_scope": (
                    "Structural-only presentation QA was explicitly requested. Canonical validation, "
                    "packaging, and semantic HTML fallback structure are required; browser interaction "
                    "and responsive-layout verification are not claimed."
                ),
                "browser_isolation": (
                    "A full synthetic report hit a scrollbar-width horizontal-overflow check in the "
                    "packaged reader's full-bleed 100vw header, while a three-block, one-chart artifact "
                    "from the same payload passed desktop and mobile browser verification."
                ),
            }
            if structural_only
            else {
                "verification_mode": "browser",
                "verification_scope": (
                    "Full desktop and mobile browser verification is required before publication; "
                    "an implicit structural-only fallback is rejected."
                ),
            }
        ),
        "strict_output_dir": _relative_path(
            source_dir, workspace_root, "strict report source directory"
        ),
        "source_snapshot": source_snapshot,
        "source_hashes": source_hashes,
        "completion": {
            "valid_arms": 8,
            "required_arms": 8,
            "expected_steps_per_arm": validated["expected_steps"],
            "warmup_steps_per_arm": validated["warmup_steps"],
            "steady_steps_per_arm": validated["steady_steps"],
            "minimum_steady_steps_for_measured": validated["minimum_measured_steps"],
            "log_timezone": "UTC",
            "strict_gate_complete": True,
            "ray_terminal_gate_complete": True,
            "slurm_terminal_gate_complete": True,
        },
        "evidence_strength": validated["evidence_strength"],
        "decision_rule": validated["analysis"].get("decision_rule"),
        "measurement_contract": validated["analysis"].get("measurement_contract"),
        "consensus": validated["analysis"].get("consensus"),
        "execution_context": execution,
        "campaign_context": campaign_context,
        "slurm_terminal_validation": validated["slurm_terminal_validation"],
        "suites": suites,
        "arm_metrics": arm_rows,
        "chart_map": [
            {
                "section": f"{SUITE_LABELS[suite]} throughput",
                "question": "Which allocation has the highest accepted-session wall throughput?",
                "family": "Comparison and ranking",
                "type": "bar",
                "fields": ["arm_label", "sessions_per_second"],
                "normalization": "Absolute accepted sessions per steady elapsed second; zero baseline; one workload only.",
                "palette_policy": "single-root preferred",
            }
            for suite in SUITE_ORDER
        ]
        + [
            {
                "section": f"{SUITE_LABELS[suite]} GPU-hour efficiency",
                "question": "Which allocation produces the most accepted sessions per allocated GPU-hour?",
                "family": "Comparison and ranking",
                "type": "bar",
                "fields": ["arm_label", "sessions_per_gpu_hour"],
                "normalization": "Absolute accepted sessions per allocated GPU-hour; zero baseline; one workload only.",
                "palette_policy": "single-root preferred",
            }
            for suite in SUITE_ORDER
        ]
        + [
            {
                "section": "Trainer wait ratio",
                "question": "What fraction of steady optimizer-step time is spent waiting for rollout data?",
                "family": "Comparison and ranking",
                "type": "bar",
                "fields": ["arm_label", "weighted_wait_ratio"],
                "normalization": "Absolute fraction of steady step time on its own percentage scale.",
                "palette_policy": "single-root preferred",
            },
            {
                "section": "Sample staleness",
                "question": "How stale are accepted samples under each allocation?",
                "family": "Comparison and ranking",
                "type": "bar",
                "fields": ["arm_label", "staleness_mean"],
                "normalization": "Mean optimizer-step age on a separate numeric scale from wait ratio.",
                "palette_policy": "single-root preferred",
            },
        ],
        "limitations": [
            f"One submitted run per topology and {validated['steady_steps']} post-warmup steady optimizer steps per arm.",
            execution["limitation"],
            "Workload rates are ranked only within SPilot Router or direct RL / TMax and are never pooled.",
            "GPU utilization is publishable only where timestamp-window telemetry coverage is at least 90%.",
            "A systems allocation recommendation is not a held-out training-quality result.",
        ],
    }


def _source(
    source_id: str,
    label: str,
    path: str,
    generated_at: str,
    description: str,
    *,
    filters: Sequence[str] = (),
    metric_definitions: Sequence[str] = (),
) -> dict[str, Any]:
    escaped = path.replace("'", "''")
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": f"SELECT CAST(readfile('{escaped}') AS TEXT) AS source_json;",
            "description": description,
            "executed_at": generated_at,
            "filters": list(filters),
            "metric_definitions": list(metric_definitions),
        },
    }


def _candidate_summary(evidence_suite: Mapping[str, Any]) -> str:
    strength = "directional candidate" if evidence_suite["status"] == "directional" else "measured recommendation"
    return (
        f"- **{evidence_suite['suite_label']}:** `{evidence_suite['candidate_name']}` is the "
        f"{strength}, at {_round_text(evidence_suite['candidate_sessions_per_second'], 3)} accepted "
        f"sessions/s and {_round_text(evidence_suite['candidate_sessions_per_gpu_hour'])} accepted "
        "sessions per allocated GPU-hour."
    )


def build_artifact(
    evidence: Mapping[str, Any],
    *,
    workspace_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    generated_at = str(evidence["generated_at"])
    strict_dir = workspace_root / str(evidence["strict_output_dir"])
    analysis_path = _relative_path(
        strict_dir / "gpu-allocation-analysis.json", workspace_root, "strict analysis"
    )
    inputs_path = _relative_path(
        strict_dir / "report-inputs.json", workspace_root, "strict report inputs"
    )
    evidence_path = _relative_path(output_dir / FINAL_EVIDENCE, workspace_root, "final evidence")
    sources = [
        _source(
            "final_evidence",
            "Reviewed final GPU-allocation evidence",
            evidence_path,
            generated_at,
            "Loads the deterministic 8/8 evidence snapshot used to materialize this report.",
            filters=(
                "Exactly eight strict-valid arms",
                "One warmup optimizer step excluded per arm",
                "UTC log and GPU telemetry alignment",
            ),
            metric_definitions=(
                "Accepted sessions per GPU-hour: accepted trainable sessions divided by allocated GPUs times steady elapsed hours.",
                "Throughput relative to best: accepted sessions per second divided by the maximum within the same workload.",
                "Weighted train wait: total steady train-wait seconds divided by total steady step seconds.",
            ),
        ),
        _source(
            "strict_analysis",
            "Strict eight-arm allocation analysis",
            analysis_path,
            generated_at,
            "Loads the fail-closed ranking analysis produced from strict UTC per-arm summaries.",
            filters=("Eight contract-valid arms", "Warmup excluded", "Ranked within workload"),
        ),
        _source(
            "strict_inputs",
            "Strict report completion evidence",
            inputs_path,
            generated_at,
            "Loads manifest validation, artifact inventory, UTC configuration, and the 8/8 completion gate.",
        ),
        _source(
            "slurm_terminal",
            "Normalized Slurm allocation terminal evidence",
            _relative_path(
                strict_dir / "slurm-terminal-evidence.json",
                workspace_root,
                "Slurm terminal evidence",
            ),
            generated_at,
            "Loads allocation-level terminal state, exit code, node/GPU allocation, and timing for the exact eight jobs.",
            filters=(
                "Exact strict-analysis job-id set",
                "Slurm COMPLETED with exit code 0:0",
                "Allocated GPUs and nodes match each arm contract",
            ),
        ),
    ]
    if evidence["execution_context"]["mode"] != "unknown":
        campaign_path = _relative_path(
            strict_dir / "campaign.json", workspace_root, "campaign metadata"
        )
        sources.append(
            _source(
                "campaign_metadata",
                "Authoritative profiling campaign metadata",
                campaign_path,
                generated_at,
                "Loads recorded source-revision identifiers, job identities, dependency policy, and the scheduler concurrency snapshot.",
                filters=(
                    "Authoritative campaign only",
                    "Campaign job IDs exactly match the strict eight-arm analysis",
                ),
            )
        )

    suite_by_name = {suite["suite"]: suite for suite in evidence["suites"]}
    spilot = suite_by_name["spilot_router"]
    tmax = suite_by_name["tmax"]
    completion = evidence["completion"]
    execution = evidence["execution_context"]
    consensus = evidence.get("consensus") or {}
    campaign_context = evidence.get("campaign_context") or {}
    if consensus.get("signature"):
        consensus_sentence = (
            f"Both workloads select the same topology signature `{consensus['signature']}`, but their raw "
            "rates remain separate and are not pooled."
        )
    else:
        consensus_sentence = (
            "The workload-specific candidates differ, so there is no single cross-workload allocation; "
            "use the candidate for the workload being trained."
        )
    strength_sentence = (
        f"Each arm contains {completion['steady_steps_per_arm']} post-warmup steady steps, below the "
        f"{completion['minimum_steady_steps_for_measured']}-step measured threshold. The result is "
        "directional, not a production optimum."
        if evidence["evidence_strength"] == "directional"
        else "Every workload satisfies the measured-step threshold and the strict confounder gate."
    )
    candidate_kind = (
        "directional candidate"
        if evidence["evidence_strength"] == "directional"
        else "measured recommendation"
    )

    arm_rows = []
    for arm in evidence["arm_metrics"]:
        suite_status = suite_by_name[arm["suite"]]["status"]
        arm_rows.append(
            {
                "suite_key": arm["suite"],
                "suite": arm["suite_label"],
                "arm": arm["arm_name"],
                "arm_label": arm["chart_label"],
                "short_chart_label": arm["short_chart_label"],
                "job_id": arm["job_id"],
                "candidate_status": (
                    "Directional candidate"
                    if arm["candidate"] and suite_status == "directional"
                    else "Measured recommendation"
                    if arm["candidate"]
                    else "Comparator"
                ),
                "allocated_gpus": arm["allocated_gpus"],
                "steady_steps": arm["steady_steps"],
                "sessions_per_second": arm["trainable_sessions_per_second"],
                "sessions_per_gpu_hour": arm["trainable_sessions_per_gpu_hour"],
                "throughput_relative_to_best": arm["throughput_relative_to_best"],
                "efficiency_relative_to_best": arm["efficiency_relative_to_best"],
                "weighted_wait_ratio": arm["weighted_wait_ratio"],
                "staleness_mean": arm["staleness_mean"],
                "rollout_success_rate": arm["rollout_success_rate"],
                "gpu_utilization": arm["gpu_utilization_fraction"],
                "gpu_coverage": arm["gpu_coverage_fraction"],
                "actor_gpu_utilization": arm["actor_gpu_utilization_fraction"],
                "rollout_gpu_utilization": arm["rollout_gpu_utilization_fraction"],
                "shared_gpu_utilization": arm["shared_gpu_utilization_fraction"],
                "utilization_publishable": arm["utilization_publishable"],
                "terminal_errors_mean": arm["terminal_error_sessions_mean"],
                "terminal_timeouts_mean": arm["terminal_timeout_sessions_mean"],
                "diagnosis": arm["bottleneck"],
            }
        )
    def chart_rows(suite_name: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "arm_label": row["short_chart_label"],
                "suite": row["suite"],
                "sessions_per_second": row["sessions_per_second"],
                "sessions_per_gpu_hour": row["sessions_per_gpu_hour"],
                "weighted_wait_ratio": row["weighted_wait_ratio"],
                "staleness_mean": row["staleness_mean"],
                "rollout_success_rate": row["rollout_success_rate"],
                "allocated_gpus": row["allocated_gpus"],
                "candidate_status": row["candidate_status"],
            }
            for row in arm_rows
            if suite_name is None or row["suite_key"] == suite_name
        ]

    spilot_chart_rows = chart_rows("spilot_router")
    tmax_chart_rows = chart_rows("tmax")
    all_chart_rows = chart_rows()
    gpu_role_rows = [
        {
            "arm_label": row["arm_label"],
            "coverage": row["gpu_coverage"],
            "overall_utilization": row["gpu_utilization"],
            "actor_utilization": row["actor_gpu_utilization"],
            "rollout_utilization": row["rollout_gpu_utilization"],
            "shared_utilization": row["shared_gpu_utilization"],
            "publishability": (
                "Published (coverage >= 90%)"
                if row["utilization_publishable"]
                else "Withheld (coverage < 90% or unavailable)"
            ),
        }
        for row in arm_rows
    ]
    candidate_rows = [
        {
            "suite": suite["suite_label"],
            "evidence_strength": suite["status"].title(),
            "candidate": suite["candidate_name"],
            "accepted_sessions_per_second": suite["candidate_sessions_per_second"],
            "accepted_sessions_per_gpu_hour": suite["candidate_sessions_per_gpu_hour"],
            "weighted_wait_ratio": suite["candidate_weighted_wait_ratio"],
            "mean_staleness": suite["candidate_staleness_mean"],
            "allocated_gpus": suite["candidate_allocated_gpus"],
            "fastest_arm": suite["fastest_arm"],
            "most_efficient_arm": suite["most_efficient_arm"],
            "decision_blockers": "; ".join(suite["decision_blockers"]) or "None",
        }
        for suite in evidence["suites"]
    ]
    comparability_rows = []
    for suite in evidence["suites"]:
        comp = suite["comparability"]
        comparability_rows.append(
            {
                "suite": suite["suite_label"],
                "model": comp.get("model"),
                "checkpoint": Path(str(comp.get("checkpoint") or "")).name,
                "data_sha256": str(comp.get("data_sha256") or "")[:12],
                "global_batch_size": comp.get("global_batch_size"),
                "rollout_batch_size": comp.get("rollout_batch_size"),
                "samples_per_prompt": comp.get("samples_per_prompt"),
                "harness": comp.get("harness"),
                "max_tokens_per_gpu": comp.get("max_tokens_per_gpu"),
                "context_parallel_size": comp.get("context_parallel_size"),
                "allow_single_sample_over_token_cap": comp.get(
                    "allow_single_sample_over_token_cap"
                ),
                "optimizer_cpu_offload": comp.get("optimizer_cpu_offload"),
                "code_revision": str(comp.get("code_revision") or "")[:12],
            }
        )

    cards = [
        {
            "id": "completion_card",
            "description": "All planned allocation arms passed the strict manifest, topology, Ray and Slurm terminal-state, and record-completeness gates.",
            "dataset": "campaign_completion",
            "sourceId": "final_evidence",
            "metrics": [
                {"label": "Strict-valid arms", "field": "valid_arms", "format": "number"},
                {"label": "Required arms", "field": "required_arms", "format": "number"},
            ],
        },
        {
            "id": "steady_depth_card",
            "description": "Post-warmup optimizer steps available per arm versus the threshold for a measured recommendation.",
            "dataset": "campaign_completion",
            "sourceId": "final_evidence",
            "metrics": [
                {"label": "Steady steps per arm", "field": "steady_steps", "format": "number"},
                {"label": "Measured threshold", "field": "measured_threshold", "format": "number"},
            ],
        },
        {
            "id": "spilot_candidate_card",
            "description": "Accepted work for the strict SPilot candidate; all allocated GPUs are included in the cost denominator.",
            "dataset": "spilot_candidate",
            "sourceId": "final_evidence",
            "metrics": [
                {"label": "SPilot sessions / GPU-hour", "field": "sessions_per_gpu_hour", "format": "number"},
                {"label": "Accepted sessions / second", "field": "sessions_per_second", "format": "number"},
                {"label": "Allocated GPUs", "field": "allocated_gpus", "format": "number"},
            ],
        },
        {
            "id": "tmax_candidate_card",
            "description": "Accepted work for the strict direct-RL/TMax candidate; all allocated GPUs are included in the cost denominator.",
            "dataset": "tmax_candidate",
            "sourceId": "final_evidence",
            "metrics": [
                {"label": "Direct RL sessions / GPU-hour", "field": "sessions_per_gpu_hour", "format": "number"},
                {"label": "Accepted sessions / second", "field": "sessions_per_second", "format": "number"},
                {"label": "Allocated GPUs", "field": "allocated_gpus", "format": "number"},
            ],
        },
    ]

    def comparison_chart(
        *,
        chart_id: str,
        title: str,
        subtitle: str,
        question: str,
        rationale: str,
        dataset: str,
        field: str,
        label: str,
        value_format: str,
        denominator: str,
        unit: str,
        semantic_family: str,
    ) -> dict[str, Any]:
        return {
            "id": chart_id,
            "title": title,
            "subtitle": subtitle,
            "intent": "comparison",
            "question": question,
            "rationale": rationale,
            "comparisonContext": {
                "grain": "allocation arm",
                "denominator": denominator,
                "normalization": "absolute value on a zero-baseline category scale",
                "unit": unit,
                "semanticFamily": semantic_family,
            },
            "type": "bar",
            "dataset": dataset,
            "sourceId": "strict_analysis",
            "encodings": {
                "x": {"field": "arm_label", "type": "nominal", "label": "Allocation arm"},
                "y": {
                    "field": field,
                    "type": "quantitative",
                    "label": label,
                    "format": value_format,
                },
                "tooltip": [
                    {"field": "suite", "type": "nominal", "label": "Workload"},
                    {
                        "field": "sessions_per_second",
                        "type": "quantitative",
                        "label": "Accepted sessions/s",
                        "format": "number",
                    },
                    {
                        "field": "sessions_per_gpu_hour",
                        "type": "quantitative",
                        "label": "Accepted sessions/GPU-hour",
                        "format": "number",
                    },
                    {
                        "field": "weighted_wait_ratio",
                        "type": "quantitative",
                        "label": "Weighted wait ratio",
                        "format": "percent",
                    },
                    {
                        "field": "staleness_mean",
                        "type": "quantitative",
                        "label": "Mean staleness",
                        "format": "number",
                    },
                    {
                        "field": "allocated_gpus",
                        "type": "quantitative",
                        "label": "Allocated GPUs",
                        "format": "number",
                    },
                    {"field": "candidate_status", "type": "nominal", "label": "Decision role"},
                ],
            },
            "valueFormat": value_format,
            "unit": unit,
            "layout": "full",
            "labels": {"values": "all"},
            "settings": {"orientation": "vertical", "showValues": True, "sort": "none"},
            "surface": {"surface": "card", "viewMode": "both", "showControls": False},
        }

    charts = [
        comparison_chart(
            chart_id="spilot_throughput_chart",
            title="SPilot accepted-session throughput",
            subtitle="Four strict-valid SPilot arms; absolute accepted sessions per steady elapsed second, shown from zero.",
            question="Which SPilot allocation has the highest accepted-session wall throughput?",
            rationale="A workload-specific zero-baseline category comparison avoids pooling SPilot rates with the direct-RL harness.",
            dataset="spilot_arm_metrics",
            field="sessions_per_second",
            label="Accepted sessions per second",
            value_format="number",
            denominator="summed steady optimizer-step seconds",
            unit="sessions per second",
            semantic_family="absolute throughput",
        ),
        comparison_chart(
            chart_id="spilot_efficiency_chart",
            title="SPilot accepted sessions per allocated GPU-hour",
            subtitle="Four strict-valid SPilot arms; all allocated GPUs are charged for the complete steady elapsed window, shown from zero.",
            question="Which SPilot allocation produces the most accepted sessions per allocated GPU-hour?",
            rationale="A separate zero-baseline efficiency chart makes the 40-GPU arm pay explicitly for its additional rollout capacity.",
            dataset="spilot_arm_metrics",
            field="sessions_per_gpu_hour",
            label="Accepted sessions per GPU-hour",
            value_format="number",
            denominator="allocated GPUs multiplied by summed steady elapsed hours",
            unit="sessions per GPU-hour",
            semantic_family="absolute GPU-hour efficiency",
        ),
        comparison_chart(
            chart_id="tmax_throughput_chart",
            title="Direct RL / TMax accepted-session throughput",
            subtitle="Four strict-valid direct-RL arms; absolute accepted sessions per steady elapsed second, shown from zero.",
            question="Which direct-RL/TMax allocation has the highest accepted-session wall throughput?",
            rationale="A workload-specific zero-baseline category comparison keeps the mini-swe-agent harness on its own rate scale.",
            dataset="tmax_arm_metrics",
            field="sessions_per_second",
            label="Accepted sessions per second",
            value_format="number",
            denominator="summed steady optimizer-step seconds",
            unit="sessions per second",
            semantic_family="absolute throughput",
        ),
        comparison_chart(
            chart_id="tmax_efficiency_chart",
            title="Direct RL / TMax accepted sessions per allocated GPU-hour",
            subtitle="Four strict-valid direct-RL arms; all allocated GPUs are charged for the complete steady elapsed window, shown from zero.",
            question="Which direct-RL/TMax allocation produces the most accepted sessions per allocated GPU-hour?",
            rationale="A separate zero-baseline efficiency chart shows whether higher GPU and worker capacity earns back its total allocation cost.",
            dataset="tmax_arm_metrics",
            field="sessions_per_gpu_hour",
            label="Accepted sessions per GPU-hour",
            value_format="number",
            denominator="allocated GPUs multiplied by summed steady elapsed hours",
            unit="sessions per GPU-hour",
            semantic_family="absolute GPU-hour efficiency",
        ),
        comparison_chart(
            chart_id="wait_ratio_chart",
            title="Weighted trainer wait ratio by allocation arm",
            subtitle="Eight strict-valid arms; total steady train-wait seconds divided by total steady optimizer-step seconds.",
            question="What fraction of steady optimizer-step time is spent waiting for rollout data?",
            rationale="Wait ratio uses its own percentage scale so it cannot be mistaken for throughput, efficiency, or sample age.",
            dataset="all_arm_metrics",
            field="weighted_wait_ratio",
            label="Weighted trainer wait ratio",
            value_format="percent",
            denominator="total steady optimizer-step seconds",
            unit="fraction",
            semantic_family="wait composition",
        ),
        comparison_chart(
            chart_id="staleness_chart",
            title="Mean accepted-sample staleness by allocation arm",
            subtitle="Eight strict-valid arms; mean optimizer-step age on a separate numeric scale from wait ratio.",
            question="How stale are accepted samples under each allocation?",
            rationale="A separate staleness chart preserves the sample-age unit and prevents a mixed-scale comparison with wait percentage.",
            dataset="all_arm_metrics",
            field="staleness_mean",
            label="Mean staleness (optimizer steps)",
            value_format="number",
            denominator="accepted trainable sessions in steady optimizer steps",
            unit="optimizer steps",
            semantic_family="sample staleness",
        ),
    ]

    tables = [
        {
            "id": "candidate_summary_table",
            "title": "Workload-specific allocation candidates",
            "subtitle": (
                "Candidate, fastest arm, and GPU-hour leader under the strict decision rule; "
                f"{completion['steady_steps_per_arm']} steady steps per arm yield "
                f"{evidence['evidence_strength']} evidence."
            ),
            "dataset": "suite_candidates",
            "sourceId": "strict_analysis",
            "defaultSort": {"field": "suite", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "suite", "label": "Workload", "type": "text"},
                {"field": "evidence_strength", "label": "Strength", "type": "text"},
                {"field": "candidate", "label": "Candidate", "type": "text"},
                {"field": "accepted_sessions_per_second", "label": "Accepted sessions/s", "format": "number"},
                {"field": "accepted_sessions_per_gpu_hour", "label": "Accepted sessions/GPU-hour", "format": "number"},
                {"field": "weighted_wait_ratio", "label": "Weighted wait", "format": "percent"},
                {"field": "mean_staleness", "label": "Mean staleness", "format": "number"},
                {"field": "allocated_gpus", "label": "Allocated GPUs", "format": "number"},
                {"field": "fastest_arm", "label": "Fastest arm", "type": "text"},
                {"field": "most_efficient_arm", "label": "GPU-hour leader", "type": "text"},
                {"field": "decision_blockers", "label": "Decision blockers", "type": "text"},
            ],
        },
        {
            "id": "all_arms_table",
            "title": "Strict-valid eight-arm metrics",
            "subtitle": (
                f"One run and {completion['steady_steps_per_arm']} post-warmup optimizer steps per "
                "topology; rates are comparable only within workload."
            ),
            "dataset": "arm_metrics",
            "sourceId": "strict_analysis",
            "defaultSort": {"field": "arm_label", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "arm_label", "label": "Arm", "type": "text"},
                {"field": "candidate_status", "label": "Decision role", "type": "text"},
                {"field": "allocated_gpus", "label": "GPUs", "format": "number"},
                {"field": "sessions_per_second", "label": "Accepted sessions/s", "format": "number"},
                {"field": "sessions_per_gpu_hour", "label": "Accepted sessions/GPU-hour", "format": "number"},
                {"field": "weighted_wait_ratio", "label": "Weighted wait", "format": "percent"},
                {"field": "staleness_mean", "label": "Mean staleness", "format": "number"},
                {"field": "rollout_success_rate", "label": "Rollout success", "format": "percent"},
                {"field": "terminal_errors_mean", "label": "Mean terminal errors", "format": "number"},
                {"field": "terminal_timeouts_mean", "label": "Mean timeouts", "format": "number"},
                {"field": "diagnosis", "label": "Bottleneck diagnosis", "type": "text"},
            ],
        },
        {
            "id": "gpu_role_table",
            "title": "Timestamp-aligned GPU utilization by role",
            "subtitle": "Role values are published only when the strict telemetry window covers at least 90% of expected samples.",
            "dataset": "gpu_role_metrics",
            "sourceId": "strict_analysis",
            "defaultSort": {"field": "arm_label", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "arm_label", "label": "Arm", "type": "text"},
                {"field": "coverage", "label": "Telemetry coverage", "format": "percent"},
                {"field": "overall_utilization", "label": "Overall GPU utilization", "format": "percent"},
                {"field": "actor_utilization", "label": "Trainer / actor", "format": "percent"},
                {"field": "rollout_utilization", "label": "Rollout", "format": "percent"},
                {"field": "shared_utilization", "label": "Shared / collocate", "format": "percent"},
                {"field": "publishability", "label": "90% coverage gate", "type": "text"},
            ],
        },
        {
            "id": "comparability_table",
            "title": "Model, data, and batching contract",
            "subtitle": "Validated suite-level fingerprint fields; harness-specific rates remain separate.",
            "dataset": "comparability",
            "sourceId": "strict_analysis",
            "defaultSort": {"field": "suite", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "suite", "label": "Workload", "type": "text"},
                {"field": "model", "label": "Model", "type": "text"},
                {"field": "checkpoint", "label": "Checkpoint", "type": "text"},
                {"field": "data_sha256", "label": "Data SHA-256", "type": "text"},
                {"field": "global_batch_size", "label": "Global batch", "format": "number"},
                {"field": "rollout_batch_size", "label": "Rollout batch", "format": "number"},
                {"field": "samples_per_prompt", "label": "Samples/prompt", "format": "number"},
                {"field": "max_tokens_per_gpu", "label": "Max tokens/GPU", "format": "number"},
                {"field": "context_parallel_size", "label": "Context parallel", "format": "number"},
                {"field": "allow_single_sample_over_token_cap", "label": "Allow oversize sample", "type": "text"},
                {"field": "optimizer_cpu_offload", "label": "Optimizer CPU offload", "type": "text"},
                {"field": "harness", "label": "Harness", "type": "text"},
                {"field": "code_revision", "label": "Code revision", "type": "text"},
            ],
        },
    ]

    title = "GPU Allocation Profiling for SPilot Router and Direct RL"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "final_evidence",
            "body": "\n".join(
                [
                    "## Technical Summary",
                    "",
                    "- **The strict campaign is complete:** all eight planned arms passed the exact topology, Ray and Slurm terminal-state, record-completeness, provenance, and artifact gates.",
                    f"- **Execution context:** {execution['summary']}",
                    f"- **Memory-safe training contract:** {campaign_context.get('model')} starts from rollout {campaign_context.get('start_rollout_id')} with a shared {campaign_context.get('max_tokens_per_gpu')}-token/GPU cap, CP={campaign_context.get('context_parallel_size')}, oversize admission={campaign_context.get('allow_single_sample_over_token_cap')}, and optimizer CPU offload={campaign_context.get('optimizer_cpu_offload')}.",
                    _candidate_summary(spilot),
                    _candidate_summary(tmax),
                    f"- **Cross-workload interpretation:** {consensus_sentence}",
                    f"- **Confidence:** {strength_sentence}",
                ]
            ),
        },
        {
            "id": "campaign_correction",
            "type": "markdown",
            "sourceId": "campaign_metadata",
            "body": (
                "## Why the final campaign uses this memory contract\n\n"
                f"Campaign `{campaign_context.get('campaign_id')}` supersedes "
                f"`{campaign_context.get('supersedes_campaign')}`. "
                f"{campaign_context.get('reason_for_supersession')}\n\n"
                f"The final comparison therefore applies one {campaign_context.get('max_tokens_per_gpu')}-token/GPU cap to every topology. "
                f"Oversize handling is causal and aligned: {campaign_context.get('oversize_policy')}"
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "completion_card",
                "steady_depth_card",
                "spilot_candidate_card",
                "tmax_candidate_card",
            ],
        },
        {
            "id": "candidate_interpretation",
            "type": "markdown",
            "sourceId": "final_evidence",
            "body": (
                "## The decision is workload-specific\n\n"
                f"{_candidate_summary(spilot)[2:]}\n\n{_candidate_summary(tmax)[2:]}\n\n"
                "A candidate must remain within 95% of its workload's best accepted-session throughput, then maximize accepted sessions per allocated GPU-hour within the 5% efficiency tie band. Lower staleness and fewer GPUs break close ties."
            ),
        },
        {"id": "candidate_summary", "type": "table", "tableId": "candidate_summary_table"},
        {
            "id": "spilot_throughput_finding",
            "type": "markdown",
            "sourceId": "final_evidence",
            "body": (
                "## SPilot throughput defines the near-best pool\n\n"
                "The zero-baseline chart compares absolute accepted sessions per second across the four SPilot allocations. The strict rule retains arms within 95% of the fastest SPilot arm before considering GPU-hour efficiency; direct-RL rates are excluded from this scale."
            ),
        },
        {"id": "spilot_throughput", "type": "chart", "chartId": "spilot_throughput_chart"},
        {
            "id": "spilot_efficiency_finding",
            "type": "markdown",
            "sourceId": "final_evidence",
            "body": (
                "## SPilot GPU-hour efficiency prices every device\n\n"
                "The second zero-baseline SPilot chart divides accepted sessions by all allocated GPU-hours. It tests whether the 40-GPU 8/32 throughput ceiling earns back the additional eight devices before staleness and GPU-count tie-breaks are applied."
            ),
        },
        {"id": "spilot_efficiency", "type": "chart", "chartId": "spilot_efficiency_chart"},
        {
            "id": "tmax_throughput_finding",
            "type": "markdown",
            "sourceId": "final_evidence",
            "body": (
                "## Direct-RL throughput stays on its own workload scale\n\n"
                "The direct-RL/TMax zero-baseline chart compares absolute accepted sessions per second for the mini-swe-agent harness only. Its four-arm 95% candidate pool is computed independently from SPilot because agent execution and worker scaling differ."
            ),
        },
        {"id": "tmax_throughput", "type": "chart", "chartId": "tmax_throughput_chart"},
        {
            "id": "tmax_efficiency_finding",
            "type": "markdown",
            "sourceId": "final_evidence",
            "body": (
                "## Direct-RL GPU-hour efficiency resolves its near-throughput ties\n\n"
                "The direct-RL/TMax efficiency chart charges every trainer and rollout GPU for the full steady elapsed window. Higher GPU and CPU-worker capacity is preferred only when accepted work per allocated GPU-hour remains competitive."
            ),
        },
        {"id": "tmax_efficiency", "type": "chart", "chartId": "tmax_efficiency_chart"},
        {
            "id": "wait_ratio_finding",
            "type": "markdown",
            "sourceId": "final_evidence",
            "body": (
                "## Trainer wait identifies rollout-side pressure\n\n"
                "Weighted wait is total steady train-wait time divided by total steady optimizer-step time. It is shown on a dedicated percentage scale: a high value indicates that trainer capacity is frequently idle, but local rollout GPUs are not necessarily the cause when their utilization is low."
            ),
        },
        {"id": "wait_ratio", "type": "chart", "chartId": "wait_ratio_chart"},
        {
            "id": "staleness_finding",
            "type": "markdown",
            "sourceId": "final_evidence",
            "body": (
                "## Staleness is the quality-of-overlap tie-breaker\n\n"
                "Mean accepted-sample age is plotted separately in optimizer steps rather than sharing the wait-ratio axis. Lower staleness breaks close throughput and GPU-hour ties because aggressive asynchronous overlap can otherwise train on older trajectories."
            ),
        },
        {"id": "staleness", "type": "chart", "chartId": "staleness_chart"},
        {
            "id": "exact_metrics_finding",
            "type": "markdown",
            "sourceId": "final_evidence",
            "body": (
                "## Exact metrics expose the operating tradeoffs\n\n"
                "The audit table keeps accepted-work throughput, GPU-hour efficiency, wait, staleness, rollout quality, telemetry coverage, and bottleneck diagnosis together. GPU utilization is blank when the strict parser cannot publish a timestamp-aligned window with at least 90% coverage."
            ),
        },
        {"id": "all_arms", "type": "table", "tableId": "all_arms_table"},
        {
            "id": "gpu_role_finding",
            "type": "markdown",
            "sourceId": "final_evidence",
            "body": (
                "## Role-specific GPU utilization is coverage-gated\n\n"
                "Trainer/actor, rollout, and shared/collocate utilization are shown only for timestamp-aligned telemetry windows with at least 90% coverage. Blank role cells are an explicit withholding decision, not zero utilization."
            ),
        },
        {"id": "gpu_role_utilization", "type": "table", "tableId": "gpu_role_table"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "sourceId": "final_evidence",
            "body": (
                    "## Scope, data, and metric definitions\n\n"
                    f"The comparison uses exactly {completion['expected_steps_per_arm']} optimizer steps per arm: {completion['warmup_steps_per_arm']} warmup and {completion['steady_steps_per_arm']} steady. Useful work is the strict `polar/rollout_trainable_sessions` count. Wall throughput is accepted sessions divided by summed steady-step seconds. GPU-hour efficiency divides accepted sessions by allocated GPUs multiplied by the same elapsed hours.\n\n"
                    "SPilot Router and direct RL / TMax are separate workload blocks. Their model, checkpoint, data, batch, and rollout-policy fields are audited, but harness behavior differs, so raw rates are never pooled.\n\n"
                    f"Execution context: {execution['summary']}"
            ),
        },
        {"id": "comparability", "type": "table", "tableId": "comparability_table"},
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "final_evidence",
            "body": (
                "## Selection methodology and guardrails\n\n"
                f"{evidence['decision_rule']} The strict layer rejects incomplete step records, a non-successful Ray run status or Slurm allocation terminal state, inconsistent fingerprints, missing trainable-session provenance, or missing decision telemetry. Provider latency, rollout success, trainable fraction, terminal errors/timeouts, staleness, and queue slope can downgrade a result from measured to directional."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "final_evidence",
            "body": "\n".join(
                [
                    "## Limitations, uncertainty, and robustness",
                    "",
                    f"- **Minimal sample:** one run and {completion['steady_steps_per_arm']} steady steps per topology; the measured threshold is {completion['minimum_steady_steps_for_measured']} steady steps.",
                    f"- **{execution['limitation_heading']}:** {execution['limitation']}",
                    "- **Dynamic task mix:** replacement and early-completion behavior means identical seeds do not create paired trajectories across topologies.",
                    "- **Topology-specific implementation:** collocation includes phase switching/offload behavior, while direct RL scales CPU agent workers with rollout capacity.",
                    "- **Systems evidence only:** accepted-work throughput does not establish held-out reward or coding quality.",
                    "",
                    (
                        "These limitations make the directional ordering a screening result that requires controlled replication before it can support a production optimum."
                        if evidence["evidence_strength"] == "directional"
                        else "These limitations bound generalization beyond the strict measured campaign and still require a held-out quality check."
                    ),
                ]
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "sourceId": "final_evidence",
            "body": (
                "## Recommended next steps\n\n"
                f"1. Confirm each workload's {candidate_kind} and nearest competitor with at least six post-warmup steady optimizer steps per repeat and at least two repeats from the same checkpoint.\n"
                f"2. {execution['repeat_control']}\n"
                "3. Require the same strict UTC, topology, provenance, error, staleness, and queue guards.\n"
                "4. Run a held-out coding-quality comparison before changing a long-running production allocation."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- Does the candidate remain stable when task and response-length mix changes?\n"
                "- When trainer wait is high but rollout GPU utilization is low, is admission/provider latency the actual bottleneck?\n"
                "- Does collocation's GPU-hour saving persist after offload/onload and weight-transfer cost?\n"
                "- Does the systems winner preserve reward, completion quality, and training stability over a longer horizon?"
            ),
        },
    ]

    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "campaign_completion": [
                {
                    "valid_arms": completion["valid_arms"],
                    "required_arms": completion["required_arms"],
                    "steady_steps": completion["steady_steps_per_arm"],
                    "measured_threshold": completion["minimum_steady_steps_for_measured"],
                }
            ],
            "spilot_candidate": [
                {
                    "sessions_per_gpu_hour": spilot["candidate_sessions_per_gpu_hour"],
                    "sessions_per_second": spilot["candidate_sessions_per_second"],
                    "allocated_gpus": spilot["candidate_allocated_gpus"],
                }
            ],
            "tmax_candidate": [
                {
                    "sessions_per_gpu_hour": tmax["candidate_sessions_per_gpu_hour"],
                    "sessions_per_second": tmax["candidate_sessions_per_second"],
                    "allocated_gpus": tmax["candidate_allocated_gpus"],
                }
            ],
            "suite_candidates": candidate_rows,
            "spilot_arm_metrics": spilot_chart_rows,
            "tmax_arm_metrics": tmax_chart_rows,
            "all_arm_metrics": all_chart_rows,
            "arm_metrics": arm_rows,
            "gpu_role_metrics": gpu_role_rows,
            "comparability": comparability_rows,
        },
        "accessIssues": [],
    }
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                f"Strict 8/8 GPU-allocation comparison generated {generated_at}. "
                f"Evidence strength: {evidence['evidence_strength']}."
            ),
            "generatedAt": generated_at,
            "blocks": blocks,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
        },
        "snapshot": snapshot,
        "sources": sources,
    }


def discover_builder(explicit: Path | None = None) -> Path:
    if explicit is not None:
        builder = explicit.expanduser().resolve()
        _require(builder.is_file(), f"portable report builder does not exist: {builder}")
        return builder
    environment = os.environ.get("DATA_ANALYTICS_REPORT_BUILDER")
    if environment:
        return discover_builder(Path(environment))
    root = Path.home() / ".codex" / "plugins" / "cache" / "openai-curated-remote" / "data-analytics"
    candidates = sorted(
        root.glob("*/skills/build-report/scripts/deliver_portable_artifact.mjs")
    )
    _require(bool(candidates), "no installed Data Analytics portable report builder was found")
    _require(
        len(candidates) == 1,
        "multiple Data Analytics report builders are installed; pass --builder explicitly",
    )
    return candidates[0].resolve()


def _validate_delivery_receipt(
    receipt: Mapping[str, Any],
    html_path: Path,
    *,
    required_verification: str | None = None,
) -> None:
    _require(receipt.get("ok") is True, "portable report delivery receipt is not successful")
    stages = receipt.get("stages")
    _require(isinstance(stages, dict), "portable report delivery receipt has no stages")
    _require(stages.get("validation") == "passed", "portable artifact validation did not pass")
    _require(stages.get("package") == "passed", "portable artifact packaging did not pass")
    _require(
        stages.get("verification") in {"passed", "structural_only"},
        "portable artifact verification did not pass",
    )
    if required_verification is not None:
        _require(
            stages.get("verification") == required_verification,
            f"portable artifact verification must be {required_verification}",
        )
    _require(
        html_path.is_file() and html_path.stat().st_size > 0,
        "portable report delivery wrote no HTML",
    )


def deliver_with_builder(
    artifact_path: Path,
    html_path: Path,
    *,
    builder: Path,
    node_bin: str,
    structural_only: bool = False,
) -> dict[str, Any]:
    command = [
        node_bin,
        str(builder),
        "--input",
        str(artifact_path),
        "--output",
        str(html_path),
    ]
    environment = None
    if structural_only:
        environment = dict(os.environ)
        environment["CHROMIUM_EXECUTABLE_PATH"] = (
            "/tmp/data-analytics-structural-only-no-browser"
        )
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise FinalizationError(
            f"portable report builder failed ({result.returncode}): {detail}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    _require(bool(lines), "portable report builder returned no receipt")
    try:
        receipt = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise FinalizationError("portable report builder returned an invalid receipt") from exc
    _require(isinstance(receipt, dict), "portable report builder receipt is not an object")
    _validate_delivery_receipt(
        receipt,
        html_path,
        required_verification="structural_only" if structural_only else None,
    )
    if structural_only:
        receipt["verification_policy"] = "structural_only_for_shared_reader_scrollbar_overflow"
        receipt["qa_limitations"] = [
            "Canonical validation, packaging, and semantic fallback structure passed; browser interaction and responsive-layout verification were not claimed because the packaged reader's full-bleed 100vw header triggers a scrollbar-width overflow on long reports in this environment."
        ]
    return dict(receipt)


def finalize_bundle(
    strict_dir: Path,
    output_dir: Path,
    *,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
    builder: Path | None = None,
    node_bin: str = "node",
    delivery_runner: DeliveryRunner | None = None,
    structural_only: bool = False,
) -> dict[str, Any]:
    """Validate, build, and atomically publish the final report artifacts."""

    workspace_root = workspace_root.resolve()
    strict_dir = strict_dir.resolve()
    output_dir = output_dir.resolve()
    _relative_path(strict_dir, workspace_root, "strict output directory")
    _relative_path(output_dir, workspace_root, "final output directory")
    validated = validate_strict_bundle(strict_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot = materialize_source_snapshot(
        validated,
        output_dir,
        workspace_root,
    )
    validated = dict(validated)
    validated["source_snapshot"] = source_snapshot
    evidence = build_evidence(
        validated,
        workspace_root,
        structural_only=structural_only,
    )
    artifact = build_artifact(
        evidence,
        workspace_root=workspace_root,
        output_dir=output_dir,
    )

    resolved_builder = None
    if delivery_runner is None:
        resolved_builder = discover_builder(builder)
    with tempfile.TemporaryDirectory(prefix=".gpu-profile-final-", dir=output_dir) as raw_temp:
        temp = Path(raw_temp)
        evidence_temp = temp / FINAL_EVIDENCE
        artifact_temp = temp / FINAL_ARTIFACT
        html_temp = temp / FINAL_HTML
        receipt_temp = temp / FINAL_RECEIPT
        complete_temp = temp / FINAL_COMPLETE
        evidence_temp.write_text(_json_text(evidence), encoding="utf-8")
        artifact_temp.write_text(_json_text(artifact), encoding="utf-8")
        if delivery_runner is not None:
            receipt = dict(delivery_runner(artifact_temp, html_temp))
        else:
            assert resolved_builder is not None
            receipt = deliver_with_builder(
                artifact_temp,
                html_temp,
                builder=resolved_builder,
                node_bin=node_bin,
                structural_only=structural_only,
            )
        _validate_delivery_receipt(
            receipt,
            html_temp,
            required_verification=("structural_only" if structural_only else "passed"),
        )
        if (receipt.get("stages") or {}).get("verification") == "structural_only":
            receipt.setdefault(
                "qa_limitations",
                [
                    "Portable browser interaction and responsive-layout verification were not completed; canonical validation, packaging, and semantic fallback structure passed."
                ],
            )
        presentation_provenance = {
            "finalizer_path": evidence["presentation_provenance"]["finalizer_path"],
            "finalizer_sha256": evidence["presentation_provenance"]["finalizer_sha256"],
            "portable_builder_name": (
                resolved_builder.name if resolved_builder is not None else "injected_delivery_runner"
            ),
            "portable_builder_sha256": (
                _sha256(resolved_builder) if resolved_builder is not None else None
            ),
        }
        receipt["presentation_provenance"] = presentation_provenance
        receipt["html"] = _relative_path(output_dir / FINAL_HTML, workspace_root, "final HTML")
        receipt["artifact"] = _relative_path(output_dir / FINAL_ARTIFACT, workspace_root, "final artifact")
        receipt["evidence"] = _relative_path(output_dir / FINAL_EVIDENCE, workspace_root, "final evidence")
        snapshot_relative = _relative_path(
            Path(source_snapshot["directory"]),
            workspace_root,
            "source snapshot directory",
        )
        receipt["source_snapshot"] = snapshot_relative
        # Compatibility key: this now resolves to the immutable snapshot, not
        # the mutable strict-report working directory.
        receipt["strict_source"] = snapshot_relative
        receipt_temp.write_text(_json_text(receipt), encoding="utf-8")
        complete = {
            "status": "complete",
            "generated_at": validated["generated_at"],
            "evidence_strength": validated["evidence_strength"],
            "strict_report_complete_sha256": evidence["source_hashes"]["REPORT_COMPLETE"],
            "source_snapshot": {
                "snapshot_id": source_snapshot["snapshot_id"],
                "directory": snapshot_relative,
                "manifest": _relative_path(
                    Path(source_snapshot["manifest_path"]),
                    workspace_root,
                    "source snapshot manifest",
                ),
                "manifest_sha256": source_snapshot["manifest_sha256"],
                "files": dict(sorted(evidence["source_hashes"].items())),
            },
            "outputs": {
                FINAL_EVIDENCE: _sha256(evidence_temp),
                FINAL_ARTIFACT: _sha256(artifact_temp),
                FINAL_HTML: _sha256(html_temp),
                FINAL_RECEIPT: _sha256(receipt_temp),
            },
            "delivery_verification": (receipt.get("stages") or {}).get("verification"),
            "presentation_provenance": presentation_provenance,
            "qa_limitations": list(receipt.get("qa_limitations") or []),
        }
        complete_temp.write_text(_json_text(complete), encoding="utf-8")

        _, observed_manifest_sha256 = _validate_source_snapshot(
            Path(source_snapshot["directory"]),
            snapshot_id=str(source_snapshot["snapshot_id"]),
            source_hashes=evidence["source_hashes"],
        )
        _require(
            observed_manifest_sha256 == source_snapshot["manifest_sha256"],
            "source snapshot manifest changed during finalization",
        )

        # Invalidate an old marker before replacing any payload.  A process or
        # filesystem failure can then leave an incomplete bundle, never an old
        # trusted marker pointing at a mixture of old and new payloads.
        published_marker = output_dir / FINAL_COMPLETE
        if published_marker.exists():
            os.replace(published_marker, temp / ".previous-final-complete.invalidated")
        for name, source in (
            (FINAL_EVIDENCE, evidence_temp),
            (FINAL_ARTIFACT, artifact_temp),
            (FINAL_HTML, html_temp),
            (FINAL_RECEIPT, receipt_temp),
        ):
            os.replace(source, output_dir / name)
        _, observed_manifest_sha256 = _validate_source_snapshot(
            Path(source_snapshot["directory"]),
            snapshot_id=str(source_snapshot["snapshot_id"]),
            source_hashes=evidence["source_hashes"],
        )
        _require(
            observed_manifest_sha256 == source_snapshot["manifest_sha256"],
            "source snapshot manifest changed during publication",
        )
        # Publish the completion marker last.  Consumers should trust a final
        # bundle only when this marker exists and its hashes match.
        os.replace(complete_temp, published_marker)
    return {
        "status": "complete",
        "evidence_strength": validated["evidence_strength"],
        "strict_valid_arms": 8,
        "html": str(output_dir / FINAL_HTML),
        "artifact": str(output_dir / FINAL_ARTIFACT),
        "evidence": str(output_dir / FINAL_EVIDENCE),
        "receipt": str(output_dir / FINAL_RECEIPT),
        "complete_marker": str(output_dir / FINAL_COMPLETE),
        "source_snapshot": str(source_snapshot["directory"]),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--strict-output-dir", required=True, type=Path)
    result.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--builder", type=Path)
    result.add_argument("--node-bin", default="node")
    result.add_argument(
        "--structural-only",
        action="store_true",
        help=(
            "Use the packaged builder's structural-only fallback for the known shared-reader "
            "100vw/vertical-scrollbar overflow; records that browser QA was not claimed."
        ),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_dir = args.output_dir or args.workspace_root / "report"
    try:
        result = finalize_bundle(
            args.strict_output_dir,
            output_dir,
            workspace_root=args.workspace_root,
            builder=args.builder,
            node_bin=args.node_bin,
            structural_only=args.structural_only,
        )
    except (FinalizationError, OSError) as exc:
        print(f"final profile report was not published: {exc}", file=__import__("sys").stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
