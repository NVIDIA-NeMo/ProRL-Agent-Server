#!/usr/bin/env python3
"""Build a cross-suite GPU-allocation recommendation from profile summaries.

This is the reproducible analysis layer for the final reader-facing report.
It writes Markdown, JSON, and audit CSVs.  It never treats failed startup runs
or incomplete arms as performance samples.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 2
COMPARABILITY_FIELDS = (
    "model",
    "checkpoint",
    "data_sha256",
    "global_batch_size",
    "code_revision",
    "slime_revision",
    "megatron_revision",
    "harness",
    "rollout_batch_size",
    "samples_per_prompt",
    "context_parallel_size",
    "max_tokens_per_gpu",
    "allow_single_sample_over_token_cap",
    "optimizer_cpu_offload",
    "min_complete_accept_fraction",
    "early_stop_grace_sessions",
)
CROSS_SUITE_INVARIANTS = (
    "model",
    "checkpoint",
    "data_sha256",
    "global_batch_size",
    "rollout_batch_size",
    "samples_per_prompt",
    "context_parallel_size",
    "max_tokens_per_gpu",
    "allow_single_sample_over_token_cap",
    "optimizer_cpu_offload",
    "min_complete_accept_fraction",
    "early_stop_grace_sessions",
)
TRAINABLE_SESSION_SOURCE = "polar/rollout_trainable_sessions"
MIN_STEADY_STEPS_FOR_MEASURED = 5
PROVIDER_LATENCY_RELATIVE_TOLERANCE = 0.10
SUCCESS_RATE_ABSOLUTE_TOLERANCE = 0.02
TRAINABLE_FRACTION_ABSOLUTE_TOLERANCE = 0.02
STALENESS_ABSOLUTE_TOLERANCE = 0.50
QUEUE_SLOPE_ABSOLUTE_TOLERANCE = 1.0
EFFICIENCY_TIE_RELATIVE_TOLERANCE = 0.05

CONFOUNDER_STEP_METRICS = (
    "inference_e2e_ms_mean",
    "rollout_success_rate",
    "terminal_timeout_session_count",
    "terminal_error_session_count",
    "staleness_mean",
    "queue_backlog_groups",
    "trainable_session_fraction",
)


def number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _nonempty(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _integer(value: Any) -> int | None:
    parsed = number(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def comparability_fingerprint(comparability: Mapping[str, Any]) -> str:
    """Return the canonical SHA256 for the explicit comparability contract."""

    canonical = {key: comparability.get(key) for key in COMPARABILITY_FIELDS}
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_comparability(
    job: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    raw = job.get("comparability")
    if not isinstance(raw, Mapping):
        return None, ["missing comparability contract"]

    normalized: dict[str, Any] = {}
    reasons: list[str] = []
    for key in (
        "model",
        "checkpoint",
        "harness",
    ):
        value = _nonempty(raw.get(key))
        if value is None:
            reasons.append(f"comparability.{key} is missing")
        else:
            normalized[key] = value

    data_sha256 = _nonempty(raw.get("data_sha256"))
    if data_sha256 is None or re.fullmatch(r"[0-9a-fA-F]{64}", data_sha256) is None:
        reasons.append("comparability.data_sha256 is not a 64-character SHA256")
    else:
        normalized["data_sha256"] = data_sha256.lower()

    global_batch_size = _integer(raw.get("global_batch_size"))
    if global_batch_size is None or global_batch_size <= 0:
        reasons.append("comparability.global_batch_size is not a positive integer")
    else:
        normalized["global_batch_size"] = global_batch_size
        config_batch = _integer(config.get("global_batch_size"))
        if config_batch != global_batch_size:
            reasons.append(
                "config.global_batch_size does not match comparability.global_batch_size"
            )

    for key in ("code_revision", "slime_revision", "megatron_revision"):
        revision = _nonempty(raw.get(key))
        if revision is None or re.fullmatch(r"[0-9a-fA-F]{7,64}", revision) is None:
            reasons.append(f"comparability.{key} is not a Git revision hash")
        else:
            normalized[key] = revision.lower()

    for key in (
        "rollout_batch_size",
        "samples_per_prompt",
        "context_parallel_size",
        "max_tokens_per_gpu",
    ):
        value = _integer(raw.get(key))
        if value is None or value <= 0:
            reasons.append(f"comparability.{key} is not a positive integer")
        else:
            normalized[key] = value
            config_value = _integer(config.get(key))
            if config_value != value:
                reasons.append(f"config.{key} does not match comparability.{key}")

    for key in (
        "allow_single_sample_over_token_cap",
        "optimizer_cpu_offload",
    ):
        value = raw.get(key)
        if not isinstance(value, bool):
            reasons.append(f"comparability.{key} is not a boolean")
        else:
            normalized[key] = value
            if config.get(key) != value:
                reasons.append(f"config.{key} does not match comparability.{key}")

    early_stop_grace = _integer(raw.get("early_stop_grace_sessions"))
    if early_stop_grace is None or early_stop_grace < 0:
        reasons.append(
            "comparability.early_stop_grace_sessions is not a non-negative integer"
        )
    else:
        normalized["early_stop_grace_sessions"] = early_stop_grace
        if _integer(config.get("early_stop_grace_sessions")) != early_stop_grace:
            reasons.append(
                "config.early_stop_grace_sessions does not match comparability."
                "early_stop_grace_sessions"
            )

    accept_fraction = number(raw.get("min_complete_accept_fraction"))
    if accept_fraction is None or not 0 <= accept_fraction <= 1:
        reasons.append(
            "comparability.min_complete_accept_fraction is not between zero and one"
        )
    else:
        normalized["min_complete_accept_fraction"] = accept_fraction
        config_accept_fraction = number(config.get("min_complete_accept_fraction"))
        if config_accept_fraction != accept_fraction:
            reasons.append(
                "config.min_complete_accept_fraction does not match comparability."
                "min_complete_accept_fraction"
            )

    supplied_fingerprint = _nonempty(raw.get("fingerprint"))
    if not reasons:
        expected_fingerprint = comparability_fingerprint(normalized)
        if supplied_fingerprint != expected_fingerprint:
            reasons.append("comparability fingerprint does not match its fields")
        else:
            normalized["fingerprint"] = expected_fingerprint
    elif supplied_fingerprint is not None:
        normalized["fingerprint"] = supplied_fingerprint
    return (normalized or None), reasons


def _validate_completion(job: Mapping[str, Any]) -> list[str]:
    state = str(job.get("job_status") or "").strip().upper()
    if state != "SUCCEEDED":
        return [f"terminal Ray job status is not SUCCEEDED: {state or 'missing'}"]
    return []


def _mean_from_steps(steps: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [number(step.get(key)) for step in steps]
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None) / len(values)


def _slope_from_steps(steps: Sequence[Mapping[str, Any]], key: str) -> float | None:
    points: list[tuple[int, float]] = []
    for step in steps:
        step_id = _integer(step.get("step"))
        value = number(step.get(key))
        if step_id is None or value is None:
            return None
        points.append((step_id, value))
    if len(points) < 2:
        return None
    points.sort()
    distance = points[-1][0] - points[0][0]
    if distance <= 0:
        return None
    return (points[-1][1] - points[0][1]) / distance


def mean_metric(job: Mapping[str, Any], key: str) -> float | None:
    return number(nested(job, "steady_state", "metrics", key, "mean"))


def throughput(job: Mapping[str, Any], key: str) -> float | None:
    return number(nested(job, "steady_state", "throughput", key))


def fmt(value: Any, digits: int = 2) -> str:
    parsed = number(value)
    if parsed is None:
        return "-"
    if 0 < abs(parsed) < 0.01 or abs(parsed) >= 10000:
        return f"{parsed:.2e}"
    return f"{parsed:.{digits}f}"


def parse_suite(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("--suite must be NAME=SUMMARY_JSON")
    return name, Path(raw_path).expanduser()


def signature(config: Mapping[str, Any]) -> str:
    mode = str(config.get("mode") or "unknown")
    actor = config.get("actor_gpus")
    rollout = config.get("rollout_gpus")
    allocated = config.get("allocated_gpus")
    return f"{mode}:{actor}t:{rollout}r:{allocated}g"


def role_utilization(job: Mapping[str, Any], role: str) -> float | None:
    return number(nested(job, "gpu", "roles", role, "utilization_gpu_pct", "mean"))


def bottleneck(row: Mapping[str, Any]) -> str:
    wait = number(row.get("weighted_wait_ratio"))
    if wait is None:
        return "inconclusive"
    if row.get("mode") == "collocate":
        if wait >= 0.20:
            return "rollout/phase-switch wait; shared GPU trace cannot split roles"
        if wait <= 0.05:
            return "train or offload/onload path"
        return "balanced/transition; shared GPU trace"
    actor_util = number(row.get("actor_gpu_util_pct"))
    rollout_util = number(row.get("rollout_gpu_util_pct"))
    if wait >= 0.20:
        if rollout_util is not None and rollout_util >= 60:
            return "local rollout GPU critical"
        if rollout_util is not None and rollout_util < 30:
            return "agent/provider/CPU pipeline critical; more rollout GPU unlikely to help"
        return "rollout-side critical"
    if wait <= 0.05:
        if actor_util is not None and actor_util >= 60:
            return "trainer compute critical"
        return "update-weight/communication or trainer scheduling critical"
    return "balanced/transition"


def extract_row(
    suite: str,
    job: Mapping[str, Any],
    expected_steps: int,
    warmup_steps: int,
) -> dict[str, Any]:
    config = job.get("config") if isinstance(job.get("config"), Mapping) else {}
    steps = job.get("steps") if isinstance(job.get("steps"), list) else []
    reasons: list[str] = []
    expected_steady = expected_steps - warmup_steps
    total_steps = _integer(job.get("step_count_total"))
    steady_step_count = _integer(job.get("step_count_steady"))
    if total_steps != expected_steps:
        reasons.append(f"completed {total_steps or 0}/{expected_steps} expected steps exactly")
    if steady_step_count != expected_steady:
        reasons.append(
            f"observed {steady_step_count or 0}/{expected_steady} expected steady steps exactly"
        )
    if len(steps) != expected_steps:
        reasons.append(f"found {len(steps)}/{expected_steps} exact step records")

    step_ids = [_integer(step.get("step")) for step in steps if isinstance(step, Mapping)]
    if len(step_ids) != len(steps) or any(step_id is None for step_id in step_ids):
        reasons.append("one or more step records have no integer step id")
    elif len(set(step_ids)) != len(step_ids):
        reasons.append("duplicate step records observed")

    if any(not isinstance(step, Mapping) or not isinstance(step.get("steady"), bool) for step in steps):
        reasons.append("step records do not explicitly identify warmup versus steady state")
    steady_records = [
        step for step in steps if isinstance(step, Mapping) and step.get("steady") is True
    ]
    if len(steady_records) != expected_steady:
        reasons.append(
            f"found {len(steady_records)}/{expected_steady} exact steady step records"
        )

    warmup_ids = job.get("warmup_step_ids")
    nonsteady_ids = {
        _integer(step.get("step"))
        for step in steps
        if isinstance(step, Mapping) and step.get("steady") is False
    }
    parsed_warmup_ids = (
        {_integer(step_id) for step_id in warmup_ids}
        if isinstance(warmup_ids, list)
        else set()
    )
    if (
        not isinstance(warmup_ids, list)
        or None in parsed_warmup_ids
        or len(parsed_warmup_ids) != warmup_steps
        or parsed_warmup_ids != nonsteady_ids
    ):
        reasons.append("warmup_step_ids do not exactly match non-steady records")

    elapsed = 0.0
    accepted_sessions = 0.0
    train_wait = 0.0
    wait_complete = True
    for step in steady_records:
        step_id = step.get("step")
        sources = set(step.get("sources") or [])
        missing = {"train", "rollout"} - sources
        if missing:
            reasons.append(
                f"steady step {step_id} missing {'/'.join(sorted(missing))} perf record"
            )
        step_time = number(step.get("step_time_s"))
        if step_time is None or step_time <= 0:
            reasons.append(f"steady step {step_id} has no positive step_time_s")
        else:
            elapsed += step_time
        accepted_source = step.get("accepted_session_count_metric")
        if accepted_source != TRAINABLE_SESSION_SOURCE:
            reasons.append(
                f"steady step {step_id} accepted_session_count is not proven from "
                f"{TRAINABLE_SESSION_SOURCE}"
            )
        trainable_sessions = number(step.get("accepted_session_count"))
        if trainable_sessions is None or trainable_sessions < 0:
            reasons.append(
                f"steady step {step_id} has no non-negative accepted_session_count"
            )
        else:
            accepted_sessions += trainable_sessions
        wait = number(step.get("train_wait_time_s"))
        if wait is None or wait < 0:
            wait_complete = False
        else:
            train_wait += wait

    allocated_gpus = _integer(config.get("allocated_gpus"))
    actor_gpus = _integer(config.get("actor_gpus"))
    rollout_gpus = _integer(config.get("rollout_gpus"))
    mode = str(config.get("mode") or "")
    if mode not in {"fully_async", "collocate"}:
        reasons.append("mode is neither fully_async nor collocate")
    if allocated_gpus is None or allocated_gpus <= 0:
        reasons.append("allocated_gpus is not a positive integer")
    if actor_gpus is None or actor_gpus <= 0:
        reasons.append("actor_gpus is not a positive integer")
    if rollout_gpus is None or rollout_gpus <= 0:
        reasons.append("rollout_gpus is not a positive integer")

    reasons.extend(_validate_completion(job))
    comparability, comparability_reasons = _validate_comparability(job, config)
    reasons.extend(comparability_reasons)

    if elapsed <= 0:
        reasons.append("steady elapsed time is not positive")
    if accepted_sessions <= 0:
        reasons.append("no trainable sessions were observed in steady steps")
    estimated_gpu_hours = (
        elapsed * allocated_gpus / 3600.0
        if elapsed > 0 and allocated_gpus is not None and allocated_gpus > 0
        else None
    )
    steps_per_hour = expected_steady * 3600.0 / elapsed if elapsed > 0 else None
    steps_per_gpu_hour = (
        expected_steady / estimated_gpu_hours if estimated_gpu_hours else None
    )
    sessions_per_second = accepted_sessions / elapsed if elapsed > 0 else None
    sessions_per_gpu_hour = (
        accepted_sessions / estimated_gpu_hours if estimated_gpu_hours else None
    )

    confounder_values = {
        key: _mean_from_steps(steady_records, key) for key in CONFOUNDER_STEP_METRICS
    }
    confounder_missing = [
        key for key, value in confounder_values.items() if value is None
    ]
    queue_backlog_slope = _slope_from_steps(steady_records, "queue_backlog_groups")
    if queue_backlog_slope is None and "queue_backlog_groups" not in confounder_missing:
        confounder_missing.append("queue_backlog_slope")

    gpu_coverage = number(nested(job, "gpu", "steady_coverage_fraction"))
    utilization_publishable = gpu_coverage is not None and gpu_coverage >= 0.90
    row: dict[str, Any] = {
        "suite": suite,
        "job_id": job.get("job_id"),
        "run_path": job.get("run_path"),
        "label": config.get("label") or job.get("job") or "unknown",
        "signature": signature(config),
        "mode": mode,
        "actor_gpus": actor_gpus,
        "rollout_gpus": rollout_gpus,
        "allocated_gpus": allocated_gpus,
        "async_level": config.get("async_level"),
        "step_count_total": total_steps or 0,
        "step_count_steady": steady_step_count or 0,
        "valid": not reasons,
        "exclusion_reasons": list(dict.fromkeys(reasons)),
        "job_status": job.get("job_status"),
        "comparability": comparability,
        "comparability_fingerprint": (
            comparability.get("fingerprint") if comparability else None
        ),
        "steps_per_hour": steps_per_hour,
        "steps_per_gpu_hour": steps_per_gpu_hour,
        "trainable_sessions_per_second": sessions_per_second,
        "trainable_sessions_per_gpu_hour": sessions_per_gpu_hour,
        "trainable_tokens_per_second": throughput(job, "accepted_trainable_tokens_per_s"),
        "trainable_tokens_per_gpu_hour": throughput(
            job, "accepted_trainable_tokens_per_gpu_hour"
        ),
        "accepted_group_fraction": throughput(job, "accepted_group_fraction"),
        "weighted_wait_ratio": train_wait / elapsed if wait_complete and elapsed > 0 else None,
        "staleness_mean": confounder_values["staleness_mean"],
        "queue_backlog_slope": queue_backlog_slope,
        "trainable_session_fraction": confounder_values["trainable_session_fraction"],
        "rollout_success_rate": confounder_values["rollout_success_rate"],
        "terminal_timeout_sessions_mean": confounder_values[
            "terminal_timeout_session_count"
        ],
        "terminal_error_sessions_mean": confounder_values[
            "terminal_error_session_count"
        ],
        "inference_e2e_ms_mean": confounder_values["inference_e2e_ms_mean"],
        "confounder_missing_metrics": sorted(confounder_missing),
        "gpu_coverage_fraction": gpu_coverage,
        "utilization_publishable": utilization_publishable,
        "gpu_util_pct": (
            number(nested(job, "gpu", "steady", "utilization_gpu_pct", "mean"))
            if utilization_publishable
            else None
        ),
        "actor_gpu_util_pct": role_utilization(job, "actor") if utilization_publishable else None,
        "rollout_gpu_util_pct": role_utilization(job, "rollout") if utilization_publishable else None,
        "shared_gpu_util_pct": role_utilization(job, "shared") if utilization_publishable else None,
        "warnings": list(job.get("warnings") or []),
    }
    row["bottleneck"] = bottleneck(row)
    return row


def _relative(value: float | None, best: float | None) -> float | None:
    if value is None or best is None or best <= 0:
        return None
    return value / best


def _range(values: Sequence[float]) -> float:
    return max(values) - min(values)


def _confounder_blockers(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    for row in rows:
        for metric in row.get("confounder_missing_metrics") or []:
            missing.append(f"{row.get('label')}: missing {metric}")
    if missing:
        return sorted(set(missing)), []

    blockers: list[str] = []
    latencies = [number(row.get("inference_e2e_ms_mean")) or 0.0 for row in rows]
    if min(latencies) <= 0:
        blockers.append("provider/inference latency is not positive for every arm")
    elif max(latencies) / min(latencies) > 1 + PROVIDER_LATENCY_RELATIVE_TOLERANCE:
        blockers.append("provider/inference latency differs by more than 10% across arms")

    success = [number(row.get("rollout_success_rate")) or 0.0 for row in rows]
    if min(success) < 1 - SUCCESS_RATE_ABSOLUTE_TOLERANCE:
        blockers.append("at least one rollout success rate is below 98%")
    if _range(success) > SUCCESS_RATE_ABSOLUTE_TOLERANCE:
        blockers.append("rollout success rate differs by more than 2 percentage points")

    trainable = [number(row.get("trainable_session_fraction")) or 0.0 for row in rows]
    if _range(trainable) > TRAINABLE_FRACTION_ABSOLUTE_TOLERANCE:
        blockers.append(
            "trainable-session fraction differs by more than 2 percentage points"
        )

    if any((number(row.get("terminal_timeout_sessions_mean")) or 0) > 0 for row in rows):
        blockers.append("terminal rollout timeouts were observed")
    if any((number(row.get("terminal_error_sessions_mean")) or 0) > 0 for row in rows):
        blockers.append("terminal rollout errors were observed")

    staleness = [number(row.get("staleness_mean")) or 0.0 for row in rows]
    if _range(staleness) > STALENESS_ABSOLUTE_TOLERANCE:
        blockers.append("mean staleness differs by more than 0.5 steps across arms")

    queue_slopes = [number(row.get("queue_backlog_slope")) or 0.0 for row in rows]
    if any(abs(value) > QUEUE_SLOPE_ABSOLUTE_TOLERANCE for value in queue_slopes):
        blockers.append("queue backlog changes by more than 1 group per step")
    return [], blockers


def analyze_suite(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    if not valid:
        return {
            "suite": name,
            "status": "no_valid_arms",
            "recommended_signature": None,
            "recommended_label": None,
            "directional_candidate_signature": None,
            "directional_candidate_label": None,
            "decision_blockers": ["no arm satisfies the measurement contract"],
            "rows": rows,
        }
    fingerprints = {
        str(row["comparability_fingerprint"])
        for row in valid
        if row.get("comparability_fingerprint")
    }
    if len(fingerprints) != 1:
        return {
            "suite": name,
            "status": "inconclusive",
            "recommended_signature": None,
            "recommended_label": None,
            "directional_candidate_signature": None,
            "directional_candidate_label": None,
            "comparability_fingerprint": None,
            "comparability": None,
            "decision_blockers": [
                "valid arms do not share one model/checkpoint/data/GBS/code fingerprint"
            ],
            "rows": rows,
        }
    best_throughput = max(
        number(row["trainable_sessions_per_second"]) or 0 for row in valid
    )
    best_efficiency = max(
        number(row["trainable_sessions_per_gpu_hour"]) or 0 for row in valid
    )
    for row in valid:
        row["throughput_relative_to_best"] = _relative(
            number(row["trainable_sessions_per_second"]), best_throughput
        )
        row["efficiency_relative_to_best"] = _relative(
            number(row["trainable_sessions_per_gpu_hour"]), best_efficiency
        )
        row["suite_score"] = 0.65 * (row["throughput_relative_to_best"] or 0) + 0.35 * (
            row["efficiency_relative_to_best"] or 0
        )
    near_best = [
        row for row in valid if (row.get("throughput_relative_to_best") or 0) >= 0.95
    ]
    best_near_efficiency = max(
        number(row["trainable_sessions_per_gpu_hour"]) or 0 for row in near_best
    )
    efficiency_tie_pool = [
        row
        for row in near_best
        if (number(row["trainable_sessions_per_gpu_hour"]) or 0)
        >= (1 - EFFICIENCY_TIE_RELATIVE_TOLERANCE) * best_near_efficiency
    ]
    candidate = min(
        efficiency_tie_pool,
        key=lambda row: (
            number(row.get("staleness_mean"))
            if number(row.get("staleness_mean")) is not None
            else math.inf,
            number(row.get("allocated_gpus")) or math.inf,
            -(number(row.get("trainable_sessions_per_second")) or 0),
            str(row.get("label")),
        ),
    )
    fastest = max(valid, key=lambda row: number(row["trainable_sessions_per_second"]) or 0)
    efficient = max(
        valid, key=lambda row: number(row["trainable_sessions_per_gpu_hour"]) or 0
    )
    insufficient_telemetry, confounder_blockers = _confounder_blockers(valid)
    low_sample = min(row["step_count_steady"] for row in valid) < MIN_STEADY_STEPS_FOR_MEASURED
    decision_blockers = list(insufficient_telemetry) + list(confounder_blockers)
    insufficient_arms = len(valid) < 2
    if insufficient_arms:
        decision_blockers.append("fewer than two contract-valid arms remain for comparison")
    if low_sample:
        decision_blockers.append(
            f"fewer than {MIN_STEADY_STEPS_FOR_MEASURED} steady steps per arm"
        )
    if insufficient_telemetry or insufficient_arms:
        status = "inconclusive"
    elif confounder_blockers or low_sample:
        status = "directional"
    else:
        status = "measured"
    return {
        "suite": name,
        "status": status,
        "comparability_fingerprint": next(iter(fingerprints)),
        "comparability": dict(valid[0]["comparability"]),
        "decision_blockers": decision_blockers,
        "recommended_signature": candidate["signature"] if status == "measured" else None,
        "recommended_label": candidate["label"] if status == "measured" else None,
        "directional_candidate_signature": candidate["signature"],
        "directional_candidate_label": candidate["label"],
        "fastest_signature": fastest["signature"],
        "fastest_label": fastest["label"],
        "most_efficient_signature": efficient["signature"],
        "most_efficient_label": efficient["label"],
        "rows": rows,
    }


def consensus(suites: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid_suites = [
        suite
        for suite in suites
        if suite.get("status") in {"measured", "directional"}
        and suite.get("directional_candidate_signature")
    ]
    if len(valid_suites) < 2:
        return {
            "status": "inconclusive",
            "reason": "fewer than two suites contain valid arms",
            "signature": None,
        }
    if len(valid_suites) != len(suites):
        return {
            "status": "inconclusive",
            "reason": "at least one suite is invalid or inconclusive",
            "signature": None,
        }
    invariant_blocks = {
        json.dumps(
            {
                key: nested(suite, "comparability", key)
                for key in CROSS_SUITE_INVARIANTS
            },
            sort_keys=True,
        )
        for suite in valid_suites
    }
    if len(invariant_blocks) != 1:
        return {
            "status": "inconclusive",
            "reason": (
                "cross-suite model/checkpoint/data/batch or rollout-policy invariants differ"
            ),
            "signature": None,
        }
    directions = {
        suite.get("directional_candidate_signature") for suite in valid_suites
    }
    if len(directions) != 1:
        return {
            "status": "inconclusive",
            "reason": "suite-specific winners differ; no cross-suite winner is reported",
            "signature": None,
        }
    winner = next(iter(directions))
    all_measured = all(suite.get("status") == "measured" for suite in valid_suites)
    return {
        "status": "measured_consistent" if all_measured else "directionally_consistent",
        "reason": (
            "all suite blocks select the same contract-valid candidate; raw metrics were "
            "normalized only within each suite and were not pooled"
        ),
        "signature": winner,
    }


def build_analysis(
    suite_inputs: Sequence[tuple[str, Path]],
    expected_steps: int,
    warmup_steps: int,
) -> dict[str, Any]:
    if expected_steps <= 0:
        raise ValueError("expected_steps must be positive")
    if warmup_steps < 0 or warmup_steps >= expected_steps:
        raise ValueError("warmup_steps must be non-negative and less than expected_steps")
    suites: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for name, path in suite_inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        jobs = payload.get("jobs") if isinstance(payload, Mapping) else None
        if not isinstance(jobs, list):
            raise ValueError(f"summary has no jobs list: {path}")
        rows = [extract_row(name, job, expected_steps, warmup_steps) for job in jobs]
        suites.append(analyze_suite(name, rows))
        sources.append({"suite": name, "summary_json": str(path)})
    consensus_result = consensus(suites)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "expected_steps": expected_steps,
        "warmup_steps": warmup_steps,
        "confidence": consensus_result["status"],
        "decision_rule": (
            "Exclude incomplete/invalid arms; keep arms within 95% of best trainable-session "
            "throughput; identify the best trainable sessions per allocated GPU-hour; treat "
            "efficiencies within 5% as tied, then prefer lower staleness and fewer GPUs."
        ),
        "measurement_contract": {
            "trainable_session_source": TRAINABLE_SESSION_SOURCE,
            "minimum_steady_steps_for_measured": MIN_STEADY_STEPS_FOR_MEASURED,
            "provider_latency_relative_tolerance": PROVIDER_LATENCY_RELATIVE_TOLERANCE,
            "success_rate_absolute_tolerance": SUCCESS_RATE_ABSOLUTE_TOLERANCE,
            "trainable_fraction_absolute_tolerance": TRAINABLE_FRACTION_ABSOLUTE_TOLERANCE,
            "staleness_absolute_tolerance": STALENESS_ABSOLUTE_TOLERANCE,
            "queue_slope_absolute_tolerance": QUEUE_SLOPE_ABSOLUTE_TOLERANCE,
            "efficiency_tie_relative_tolerance": EFFICIENCY_TIE_RELATIVE_TOLERANCE,
        },
        "sources": sources,
        "suites": suites,
        "consensus": consensus_result,
    }


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "Suite",
        "Arm",
        "GPU A/R",
        "Valid",
        "steps/h",
        "sessions/s",
        "sessions/GPUh",
        "wait",
        "GPU util",
        "Diagnosis",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("suite")),
                    str(row.get("label")),
                    f"{row.get('allocated_gpus') or '-'} ({row.get('actor_gpus') or '-'}/{row.get('rollout_gpus') or '-'})",
                    "yes" if row.get("valid") else "no: " + "; ".join(row.get("exclusion_reasons") or []),
                    fmt(row.get("steps_per_hour")),
                    fmt(row.get("trainable_sessions_per_second"), 3),
                    fmt(row.get("trainable_sessions_per_gpu_hour")),
                    (fmt((number(row.get("weighted_wait_ratio")) or 0) * 100, 1) + "%")
                    if number(row.get("weighted_wait_ratio")) is not None
                    else "-",
                    fmt(row.get("gpu_util_pct"), 1),
                    str(row.get("bottleneck")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_markdown(analysis: Mapping[str, Any]) -> str:
    suites = list(analysis.get("suites") or [])
    rows = [row for suite in suites for row in suite.get("rows", [])]
    consensus_result = analysis.get("consensus") or {}
    suite_bullets = []
    for suite in suites:
        if suite.get("status") == "no_valid_arms":
            suite_bullets.append(f"- **{suite['suite']}:** no valid arm; no recommendation.")
        elif suite.get("status") == "inconclusive":
            blockers = "; ".join(suite.get("decision_blockers") or [])
            suite_bullets.append(
                f"- **{suite['suite']}:** inconclusive; no recommendation. {blockers}"
            )
        elif suite.get("status") == "directional":
            blockers = "; ".join(suite.get("decision_blockers") or [])
            suite_bullets.append(
                f"- **{suite['suite']}:** `{suite['directional_candidate_label']}` is only a "
                f"directional candidate. {blockers}"
            )
        else:
            suite_bullets.append(
                f"- **{suite['suite']}:** `{suite['recommended_label']}` is the near-best "
                "throughput recommendation under the GPU-hour tie-break rule."
            )
    consensus_text = (
        f"Cross-suite result: **{consensus_result.get('status')}**, common candidate signature "
        f"`{consensus_result.get('signature')}`. {consensus_result.get('reason', '')}"
        if consensus_result.get("signature")
        else f"Cross-suite result: **inconclusive**. {consensus_result.get('reason', '')}"
    )
    invalid = [row for row in rows if not row.get("valid")]
    expected_steps = _integer(analysis.get("expected_steps")) or 0
    warmup_steps = _integer(analysis.get("warmup_steps")) or 0
    steady_steps = expected_steps - warmup_steps
    comparability_lines = [
        "| Suite | Model | Checkpoint | Data SHA256 | GBS | Rollout batch | Samples/prompt | Harness | Code revision |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for suite in suites:
        comp = suite.get("comparability") or {}
        comparability_lines.append(
            "| "
            + " | ".join(
                [
                    str(suite.get("suite") or "-"),
                    str(comp.get("model") or "-"),
                    str(comp.get("checkpoint") or "-"),
                    str(comp.get("data_sha256") or "-")[:12],
                    str(comp.get("global_batch_size") or "-"),
                    str(comp.get("rollout_batch_size") or "-"),
                    str(comp.get("samples_per_prompt") or "-"),
                    str(comp.get("harness") or "-"),
                    str(comp.get("code_revision") or "-")[:12],
                ]
            )
            + " |"
        )
    return "\n".join(
        [
            "# Training GPU allocation profile",
            "",
            "## Technical summary",
            "",
            consensus_text,
            "",
            *suite_bullets,
            "",
            f"The requested contract contains exactly {expected_steps} optimizer-step records per "
            f"arm, of which {warmup_steps} are warmup and {steady_steps} are measured. Statuses "
            "marked directional or inconclusive are not definitive allocation winners.",
            "",
            "## Throughput and GPU-hour efficiency determine different winners",
            "",
            markdown_table(rows),
            "",
            "## Scope, data, and metric definitions",
            "",
            *comparability_lines,
            "",
            "These are validated input fields, not inferred experiment facts. Within a suite, every "
            "ranked arm must have the same canonical fingerprint. Across suites, harness and code "
            "revisions may form different blocks; raw rates are normalized only within a suite and "
            "are never pooled. Steady duration is the sum of positive per-step `step_time_s`. Useful "
            f"work is accepted sessions whose recorded source is exactly `{TRAINABLE_SESSION_SOURCE}`; "
            "cost is allocated GPUs multiplied by steady wall time.",
            "",
            "## Selection methodology",
            "",
            analysis.get("decision_rule", ""),
            " Trainer wait is total wait divided by total step time. GPU utilization "
            "is published only when timestamp-window telemetry covers at least 90% of allocated GPU-hours.",
            "",
            "## Limitations and robustness checks",
            "",
            f"- {len(invalid)} arm(s) were excluded as incomplete or invalid and do not contribute to rankings.",
            "- A valid arm requires exact step counts, both train and rollout records for every steady "
            "step, positive step time, terminal Ray status `SUCCEEDED`, and the strict trainable-session source.",
            "- Provider latency may differ by at most 10%; success and trainable fractions by at most "
            "2 percentage points; staleness by at most 0.5 step. Timeouts/errors or queue slope above "
            "1 group per step block a definitive winner.",
            "- Collocate utilization is a shared physical trace; it cannot be labeled actor versus rollout.",
            "",
            "## Recommended next steps",
            "",
            "- Confirm any directional candidate with at least 6 steps × 2 repeats.",
            "- Reverse arm order in the second repeat to reduce time-of-day/provider bias.",
            "- Run a held-out quality comparison from the same checkpoint before changing long training.",
            "",
            "## Further questions",
            "",
            "- Does the winner remain stable when task and response-length mix changes?",
            "- If trainer wait is high but rollout GPU utilization is low, is admission/provider latency the real bottleneck?",
            "- Does collocation's lower GPU count compensate for offload/onload and weight-transfer overhead?",
            "",
        ]
    )


def _html(value: Any) -> str:
    return html.escape(str(value if value is not None else "-"), quote=True)


def render_html(analysis: Mapping[str, Any]) -> str:
    """Render a self-contained, reader-facing technical report."""

    suites = list(analysis.get("suites") or [])
    rows = [row for suite in suites for row in suite.get("rows", [])]
    consensus_result = analysis.get("consensus") or {}
    expected_steps = _integer(analysis.get("expected_steps")) or 0
    warmup_steps = _integer(analysis.get("warmup_steps")) or 0

    if consensus_result.get("signature"):
        consensus_html = (
            f"<strong>{_html(consensus_result.get('status'))}</strong>: common candidate "
            f"<code>{_html(consensus_result.get('signature'))}</code>. "
            f"{_html(consensus_result.get('reason'))}"
        )
    else:
        consensus_html = (
            "<strong>Inconclusive.</strong> "
            + _html(consensus_result.get("reason"))
        )

    suite_items: list[str] = []
    for suite in suites:
        status = str(suite.get("status") or "inconclusive")
        blockers = "; ".join(suite.get("decision_blockers") or [])
        if status == "measured":
            outcome = (
                f"recommendation <code>{_html(suite.get('recommended_label'))}</code>"
            )
        elif status == "directional":
            outcome = (
                "directional candidate "
                f"<code>{_html(suite.get('directional_candidate_label'))}</code>"
            )
        else:
            outcome = "no recommendation"
        suite_items.append(
            f"<li><strong>{_html(suite.get('suite'))}</strong> — {_html(status)}; "
            f"{outcome}. {_html(blockers)}</li>"
        )

    arm_rows: list[str] = []
    for row in rows:
        validity = (
            "yes"
            if row.get("valid")
            else "no: " + "; ".join(row.get("exclusion_reasons") or [])
        )
        arm_rows.append(
            "<tr>"
            f"<td>{_html(row.get('suite'))}</td>"
            f"<td>{_html(row.get('label'))}</td>"
            f"<td>{_html(row.get('allocated_gpus'))} "
            f"({_html(row.get('actor_gpus'))}/{_html(row.get('rollout_gpus'))})</td>"
            f"<td>{_html(validity)}</td>"
            f"<td>{_html(fmt(row.get('steps_per_hour')))}</td>"
            f"<td>{_html(fmt(row.get('trainable_sessions_per_second'), 3))}</td>"
            f"<td>{_html(fmt(row.get('trainable_sessions_per_gpu_hour')))}</td>"
            f"<td>{_html(fmt(row.get('staleness_mean')))}</td>"
            f"<td>{_html(row.get('bottleneck'))}</td>"
            "</tr>"
        )

    comparability_rows: list[str] = []
    for suite in suites:
        comp = suite.get("comparability") or {}
        comparability_rows.append(
            "<tr>"
            f"<td>{_html(suite.get('suite'))}</td>"
            f"<td>{_html(comp.get('model'))}</td>"
            f"<td>{_html(comp.get('checkpoint'))}</td>"
            f"<td><code>{_html(str(comp.get('data_sha256') or '-')[:12])}</code></td>"
            f"<td>{_html(comp.get('global_batch_size'))}</td>"
            f"<td>{_html(comp.get('rollout_batch_size'))}</td>"
            f"<td>{_html(comp.get('samples_per_prompt'))}</td>"
            f"<td>{_html(comp.get('harness'))}</td>"
            f"<td><code>{_html(str(comp.get('code_revision') or '-')[:12])}</code></td>"
            "</tr>"
        )

    invalid_count = sum(not bool(row.get("valid")) for row in rows)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Training GPU allocation profile</title>
  <style>
    :root {{ color-scheme: light; --ink: #172033; --muted: #5d687a; --line: #d9dee8; --panel: #f6f8fb; --accent: #3157b7; }}
    body {{ margin: 0; background: #fff; color: var(--ink); font: 15px/1.55 system-ui, -apple-system, sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 40px 28px 64px; }}
    h1 {{ font-size: 30px; margin: 0 0 8px; }}
    h2 {{ margin-top: 38px; padding-top: 10px; border-top: 1px solid var(--line); font-size: 21px; }}
    .lede {{ color: var(--muted); max-width: 850px; }}
    .summary {{ border-left: 4px solid var(--accent); background: var(--panel); padding: 14px 18px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 840px; }}
    th, td {{ padding: 9px 11px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: var(--panel); white-space: nowrap; }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{ font: 0.92em ui-monospace, SFMono-Regular, Consolas, monospace; }}
    .fine {{ color: var(--muted); font-size: 13px; }}
  </style>
</head>
<body>
<main>
  <h1>Training GPU allocation profile</h1>
  <p class="lede">Contract-validated comparison of wall throughput, useful work per allocated GPU-hour, and operating confounders.</p>

  <h2>Technical summary</h2>
  <div class="summary"><p>{consensus_html}</p><ul>{''.join(suite_items)}</ul></div>
  <p>The input requires exactly {_html(expected_steps)} optimizer-step records per arm: {_html(warmup_steps)} warmup and {_html(expected_steps - warmup_steps)} steady.</p>

  <h2>Throughput and GPU-hour efficiency</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Suite</th><th>Arm</th><th>GPU total (train/rollout)</th><th>Valid</th><th>steps/h</th><th>sessions/s</th><th>sessions/GPUh</th><th>Staleness</th><th>Diagnosis</th></tr></thead>
    <tbody>{''.join(arm_rows)}</tbody>
  </table></div>

  <h2>Scope, data, and metric definitions</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Suite</th><th>Model</th><th>Checkpoint</th><th>Data SHA256</th><th>GBS</th><th>Rollout batch</th><th>Samples/prompt</th><th>Harness</th><th>Code revision</th></tr></thead>
    <tbody>{''.join(comparability_rows)}</tbody>
  </table></div>
  <p>These values come from the validated comparability contract. Every ranked arm within a suite must have one canonical fingerprint. Cross-suite harness and revision blocks are kept separate: rates are normalized within a suite and never pooled. Useful work is accepted sessions sourced exactly from <code>{_html(TRAINABLE_SESSION_SOURCE)}</code>.</p>

  <h2>Selection methodology</h2>
  <p>{_html(analysis.get('decision_rule'))}</p>
  <p>Throughput and GPU-hour efficiency are recomputed from strict steady-step records. A measured recommendation additionally requires complete confounder telemetry and at least {_html(MIN_STEADY_STEPS_FOR_MEASURED)} steady steps per arm.</p>

  <h2>Limitations and robustness checks</h2>
  <ul>
    <li>{_html(invalid_count)} arm(s) were excluded and do not contribute to ranking.</li>
    <li>Every steady step must contain train and rollout records, positive step time, and strict trainable-session provenance; the Ray job must finish with <code>SUCCEEDED</code>.</li>
    <li>Provider latency may differ by at most 10%; success and trainable fractions by 2 percentage points; staleness by 0.5 step. Timeouts/errors or queue growth above 1 group per step block a definitive winner.</li>
    <li>GPU utilization is published only at 90% or greater timestamp-window coverage. Collocate traces cannot be split into actor and rollout roles.</li>
  </ul>

  <h2>Recommended next steps</h2>
  <ul><li>Confirm directional candidates with at least 6 steps × 2 repeats.</li><li>Reverse arm order to reduce provider and time-of-day bias.</li><li>Check held-out training quality before changing a long-running allocation.</li></ul>

  <h2>Further questions</h2>
  <ul><li>Does the candidate remain stable when task and response-length mix changes?</li><li>Is low rollout utilization caused by provider/admission latency rather than local GPUs?</li><li>Does collocation save enough GPU-hours to offset phase-switch and offload costs?</li></ul>
  <p class="fine">Generated at {_html(analysis.get('generated_at'))}. Schema version {_html(analysis.get('schema_version'))}.</p>
</main>
</body>
</html>
"""


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_audit_csvs(analysis: Mapping[str, Any], per_step: Path, gpu_role: Path) -> None:
    # The summary JSON remains the full per-step source. These flat files hold
    # the decision-facing arm and role records for easy audit and plotting.
    rows = [row for suite in analysis.get("suites", []) for row in suite.get("rows", [])]
    write_csv(
        per_step,
        rows,
        [
            "suite",
            "label",
            "signature",
            "job_id",
            "valid",
            "step_count_total",
            "step_count_steady",
            "steps_per_hour",
            "trainable_sessions_per_second",
            "trainable_sessions_per_gpu_hour",
            "weighted_wait_ratio",
            "staleness_mean",
            "queue_backlog_slope",
            "bottleneck",
        ],
    )
    role_rows: list[dict[str, Any]] = []
    for row in rows:
        for role, key in (
            ("all", "gpu_util_pct"),
            ("actor", "actor_gpu_util_pct"),
            ("rollout", "rollout_gpu_util_pct"),
            ("shared", "shared_gpu_util_pct"),
        ):
            role_rows.append(
                {
                    "suite": row["suite"],
                    "label": row["label"],
                    "job_id": row["job_id"],
                    "role": role,
                    "utilization_gpu_pct": row.get(key),
                    "coverage_fraction": row.get("gpu_coverage_fraction"),
                    "publishable": row.get("utilization_publishable"),
                }
            )
    write_csv(
        gpu_role,
        role_rows,
        [
            "suite",
            "label",
            "job_id",
            "role",
            "utilization_gpu_pct",
            "coverage_fraction",
            "publishable",
        ],
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--suite", action="append", required=True, type=parse_suite)
    result.add_argument("--expected-steps", type=int, default=3)
    result.add_argument("--warmup-steps", type=int, default=1)
    result.add_argument("--markdown", required=True, type=Path)
    result.add_argument("--html", required=True, type=Path)
    result.add_argument("--json", required=True, type=Path)
    result.add_argument("--arm-csv", required=True, type=Path)
    result.add_argument("--gpu-role-csv", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if (
        args.expected_steps <= 0
        or args.warmup_steps < 0
        or args.warmup_steps >= args.expected_steps
    ):
        parser().error(
            "expected steps must be positive and warmup steps must be non-negative "
            "and less than expected steps"
        )
    analysis = build_analysis(args.suite, args.expected_steps, args.warmup_steps)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(analysis), encoding="utf-8")
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(render_html(analysis), encoding="utf-8")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_audit_csvs(analysis, args.arm_csv, args.gpu_role_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
