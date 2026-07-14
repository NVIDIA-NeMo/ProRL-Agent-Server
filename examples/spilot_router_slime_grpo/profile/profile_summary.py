#!/usr/bin/env python3
"""Summarize SPilot/Slime profiling runs from local logs and GPU CSV files.

The parser intentionally uses only the Python standard library.  It accepts a
job directory directly, or a logical run directory containing ``job-*``
children.  Missing telemetry is reported as a warning rather than treated as a
fatal error, which makes the tool useful for interrupted profiling jobs too.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import fnmatch
import json
import math
import os
import re
import shlex
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PERF_RE = re.compile(r"\bperf\s+(-?\d+)\s*:\s*(\{.*)")
METRIC_PAIR_RE = re.compile(
    r"[\"']([^\"']+)[\"']\s*:\s*"
    r"(?:(?:np\.)?(?:float(?:16|32|64)?|int(?:16|32|64)?)\s*\(\s*)?"
    r"([-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?)"
)
EXPORT_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
JOB_ID_RE = re.compile(r"^job-(\d+)$")
NODE_ID_RE = re.compile(r"(?:^|[_-])node[_-]?(\d+)(?:$|[_-])", re.IGNORECASE)

# Never retain or emit arbitrary environment values: submit snapshots can
# contain credentials.  This allowlist is deliberately limited to topology and
# experiment-label fields needed by the profiler.
SAFE_ENV_KEYS = {
    "PROFILE_ARM",
    "PROFILE_MODE",
    "TMAX_PROFILE_ARM",
    "TMAX_PROFILE_BATCH",
    "TMAX_PROFILE_BATCH_ID",
    "EXPERIMENT_NAME",
    "RUN_ID",
    "SLIME_TRAIN_MODE",
    "TMAX_TRAIN_MODE",
    "POLAR_FULLY_ASYNC",
    "POLAR_MAX_ASYNC_LEVEL",
    "POLAR_MULTI_GATEWAY",
    "POLAR_GATEWAY_COUNT_OVERRIDE",
    "COLOCATE",
    "COLOCATED",
    "SLIME_COLOCATE",
    "ACTOR_NUM_NODES",
    "ACTOR_NUM_GPUS_PER_NODE",
    "ROLLOUT_NUM_GPUS",
    "ROLLOUT_NUM_GPUS_PER_ENGINE",
    "RAY_NUM_NODES",
    "NUM_NODES",
    "RAY_NUM_GPUS_PER_NODE",
    "SLURM_GPUS",
    "GPU_MONITOR_NODE_ROLE",
    "GPU_MONITOR_TRAIN_GPUS",
    "GPU_MONITOR_ROLLOUT_GPUS",
    "GLOBAL_BATCH_SIZE",
    "ROLLOUT_BATCH_SIZE",
    "N_SAMPLES_PER_PROMPT",
    "POLAR_SUBMITTED_JOB_ID",
    "SLURM_JOB_ID",
}

LOG_PATTERNS = (
    "output_pool*.log",
    "slurm*.out",
    "slurm*.log",
    "train*.log",
    "stdout*.log",
    "profile*.log",
)

PRUNED_TELEMETRY_DIRS = {
    ".git",
    "__pycache__",
    "compiler_cache",
    "partial_rollouts",
    "rollout_results",
    "sessions",
    "startup",
    "trajectory_examples",
}


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any) -> int | None:
    number = _float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def stats(values: Iterable[Any]) -> dict[str, float | int | None]:
    clean = sorted(value for item in values if (value := _float(item)) is not None)
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
            "sum": None,
        }
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "p50": _percentile(clean, 0.50),
        "p95": _percentile(clean, 0.95),
        "min": clean[0],
        "max": clean[-1],
        "sum": sum(clean),
    }


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _shell_value(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    try:
        pieces = shlex.split(raw, comments=False, posix=True)
    except ValueError:
        return raw.strip("'\"")
    return pieces[0] if len(pieces) == 1 else " ".join(pieces)


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = EXPORT_RE.match(line.rstrip("\n"))
                if not match or match.group(1) not in SAFE_ENV_KEYS:
                    continue
                values[match.group(1)] = _shell_value(match.group(2))
    except OSError:
        return {}
    return values


def _simple_polar_config(path: Path) -> dict[str, str]:
    wanted = {"polar_fully_async", "polar_max_async_level"}
    result: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.split("#", 1)[0].strip()
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                if key.strip() in wanted:
                    result[key.strip()] = value.strip().strip("'\"")
    except OSError:
        pass
    return result


def _job_id(job_dir: Path) -> str | None:
    match = JOB_ID_RE.match(job_dir.name)
    return match.group(1) if match else None


def discover_job_dirs(inputs: Sequence[Path]) -> list[Path]:
    """Expand logical run directories to their direct ``job-*`` children."""

    jobs: list[Path] = []
    seen: set[str] = set()
    for raw_path in inputs:
        path = raw_path.expanduser()
        if path.is_file():
            path = path.parent
        children: list[Path] = []
        if path.is_dir() and not JOB_ID_RE.match(path.name):
            try:
                children = sorted(
                    child
                    for child in path.iterdir()
                    if child.is_dir() and JOB_ID_RE.match(child.name)
                )
            except OSError:
                children = []
        candidates = children or [path]
        for candidate in candidates:
            key = str(candidate.resolve(strict=False))
            if key not in seen:
                seen.add(key)
                jobs.append(candidate)
    return jobs


def _load_config(job_dir: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    warnings: list[str] = []
    sources: list[Path] = []
    job_id = _job_id(job_dir)
    run_root = job_dir.parent if job_id else job_dir

    base_candidates = [
        run_root / "run_state.env",
        job_dir / "run_state.env",
        run_root / "profile.env",
        job_dir / "profile.env",
        run_root / "profile_config.env",
        job_dir / "profile_config.env",
        job_dir / "profile_env.sh",
    ]
    env: dict[str, str] = {}
    for candidate in base_candidates:
        if candidate.is_file():
            env.update(parse_env_file(candidate))
            sources.append(candidate)

    submit_files: list[Path] = []
    for root in {run_root, job_dir}:
        submit_dir = root / "submit"
        if submit_dir.is_dir():
            submit_files.extend(sorted(submit_dir.glob("env-*.sh")))
    matching: list[tuple[Path, dict[str, str]]] = []
    fallback: list[tuple[Path, dict[str, str]]] = []
    for submit_file in sorted(set(submit_files)):
        parsed = parse_env_file(submit_file)
        fallback.append((submit_file, parsed))
        if job_id and parsed.get("POLAR_SUBMITTED_JOB_ID") == job_id:
            matching.append((submit_file, parsed))
    selected = matching[-1:] or fallback[-1:]
    for path, parsed in selected:
        env.update(parsed)
        sources.append(path)

    polar: dict[str, str] = {}
    for polar_path in (run_root / "polar_config.yaml", job_dir / "polar_config.yaml"):
        if polar_path.is_file():
            polar.update(_simple_polar_config(polar_path))
            sources.append(polar_path)

    mode_hint = (
        env.get("PROFILE_MODE")
        or env.get("SLIME_TRAIN_MODE")
        or env.get("TMAX_TRAIN_MODE")
        or ""
    ).strip().lower().replace("-", "_")
    colocated = mode_hint in {"collocate", "colocate", "collocated", "sync_collocate"}
    colocated = colocated or any(
        _truthy(env.get(key)) for key in ("COLOCATE", "COLOCATED", "SLIME_COLOCATE")
    )
    fully_async_value = env.get("POLAR_FULLY_ASYNC", polar.get("polar_fully_async"))
    if colocated:
        mode = "collocate"
    elif mode_hint in {"fully_async", "async"} or _truthy(fully_async_value):
        mode = "fully_async"
    elif mode_hint:
        mode = mode_hint
    elif fully_async_value is not None:
        mode = "sync"
    else:
        mode = "unknown"

    actor_nodes = _int(env.get("ACTOR_NUM_NODES"))
    actor_gpus_per_node = _int(env.get("ACTOR_NUM_GPUS_PER_NODE"))
    rollout_gpus = _int(env.get("ROLLOUT_NUM_GPUS"))
    actor_gpus = (
        actor_nodes * actor_gpus_per_node
        if actor_nodes is not None and actor_gpus_per_node is not None
        else None
    )
    if mode == "collocate":
        logical_gpus = max(value for value in (actor_gpus, rollout_gpus) if value is not None) if (
            actor_gpus is not None or rollout_gpus is not None
        ) else None
    elif actor_gpus is not None and rollout_gpus is not None:
        logical_gpus = actor_gpus + rollout_gpus
    else:
        logical_gpus = None

    async_level = _int(env.get("POLAR_MAX_ASYNC_LEVEL"))
    if async_level is None:
        async_level = _int(polar.get("polar_max_async_level"))
    num_nodes = _int(env.get("RAY_NUM_NODES")) or _int(env.get("NUM_NODES"))
    gpus_per_node = _int(env.get("RAY_NUM_GPUS_PER_NODE")) or _int(env.get("SLURM_GPUS"))
    allocated_gpus = (
        num_nodes * gpus_per_node
        if num_nodes is not None and gpus_per_node is not None
        else logical_gpus
    )

    if not sources:
        warnings.append("no run_state, submit environment, or polar_config found")
    config: dict[str, Any] = {
        "label": (
            env.get("TMAX_PROFILE_ARM")
            or env.get("PROFILE_ARM")
            or env.get("EXPERIMENT_NAME")
            or run_root.name
        ),
        "profile_batch": env.get("TMAX_PROFILE_BATCH_ID")
        or env.get("TMAX_PROFILE_BATCH"),
        "run_id": env.get("RUN_ID"),
        "mode": mode,
        "async_level": async_level,
        "actor_nodes": actor_nodes,
        "actor_gpus_per_node": actor_gpus_per_node,
        "actor_gpus": actor_gpus,
        "rollout_gpus": rollout_gpus,
        "rollout_gpus_per_engine": _int(env.get("ROLLOUT_NUM_GPUS_PER_ENGINE")),
        "logical_gpus": logical_gpus,
        "allocated_gpus": allocated_gpus,
        "num_nodes": num_nodes,
        "gpus_per_node": gpus_per_node,
        "gateway_count": _int(env.get("POLAR_GATEWAY_COUNT_OVERRIDE"))
        or (num_nodes if _truthy(env.get("POLAR_MULTI_GATEWAY")) else 1),
        "global_batch_size": _int(env.get("GLOBAL_BATCH_SIZE")),
        "rollout_batch_size": _int(env.get("ROLLOUT_BATCH_SIZE")),
        "samples_per_prompt": _int(env.get("N_SAMPLES_PER_PROMPT")),
        "gpu_monitor_node_role": env.get("GPU_MONITOR_NODE_ROLE"),
    }
    return config, [str(path) for path in dict.fromkeys(sources)], warnings


def _discover_logs(job_dir: Path) -> list[Path]:
    if not job_dir.is_dir():
        return []
    found: set[Path] = set()

    def collect(root_path: Path, max_depth: int) -> None:
        if not root_path.is_dir():
            return
        base_depth = len(root_path.parts)
        for root, directories, filenames in os.walk(root_path, onerror=lambda _error: None):
            depth = len(Path(root).parts) - base_depth
            directories[:] = [name for name in directories if name not in PRUNED_TELEMETRY_DIRS]
            if depth >= max_depth:
                directories[:] = []
            current = Path(root)
            for filename in filenames:
                if any(fnmatch.fnmatch(filename, pattern) for pattern in LOG_PATTERNS):
                    found.add(current / filename)

    # Current launchers place logs under job/wandb/.../files.  Search those
    # small telemetry roots first and use only a depth-bounded fallback.  A
    # blind rglob of the job would traverse thousands of rollout task trees.
    for telemetry_root in (job_dir / "wandb", job_dir / "logs"):
        collect(telemetry_root, max_depth=6)
    try:
        for child in job_dir.iterdir():
            if child.is_file() and any(fnmatch.fnmatch(child.name, pattern) for pattern in LOG_PATTERNS):
                found.add(child)
    except OSError:
        pass
    if not found:
        collect(job_dir, max_depth=5)
    return sorted(found)


def _numeric_dict(text: str) -> dict[str, float]:
    # Ray prefixes and line wrapping can leave text after the dict.  Slice at
    # the final brace first; if literal_eval still fails, recover scalar pairs.
    end = text.rfind("}")
    candidate = text[: end + 1] if end >= 0 else text
    try:
        parsed = ast.literal_eval(candidate)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        parsed = None
    result: dict[str, float] = {}
    if isinstance(parsed, Mapping):
        for key, value in parsed.items():
            number = _float(value)
            if isinstance(key, str) and number is not None:
                result[key] = number
    if result:
        return result
    for key, raw_value in METRIC_PAIR_RE.findall(candidate):
        number = _float(raw_value)
        if number is not None:
            result[key] = number
    return result


def parse_perf_logs(paths: Sequence[Path]) -> tuple[dict[int, dict[str, float]], dict[int, set[str]], int]:
    by_step: dict[int, dict[str, float]] = defaultdict(dict)
    sources: dict[int, set[str]] = defaultdict(set)
    parse_failures = 0
    for path in paths:
        try:
            handle = path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            continue
        with handle:
            for raw_line in handle:
                line = ANSI_RE.sub("", raw_line)
                match = PERF_RE.search(line)
                if not match:
                    continue
                step = int(match.group(1))
                metrics = _numeric_dict(match.group(2))
                if not metrics:
                    parse_failures += 1
                    continue
                by_step[step].update(metrics)
                if "timing/train_wait_time" in metrics or "timing/train_time" in metrics:
                    sources[step].add("train")
                if any(
                    key.startswith(("polar/", "rollout/"))
                    or key in {"perf/rollout_time", "timing/service_time_max"}
                    for key in metrics
                ):
                    sources[step].add("rollout")
    return dict(by_step), dict(sources), parse_failures


def _first(metrics: Mapping[str, float], *keys: str) -> float | None:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def _matching_metric(metrics: Mapping[str, float], fragments: Sequence[str]) -> float | None:
    for key, value in metrics.items():
        normalized = key.lower().replace("-", "_")
        if all(fragment in normalized for fragment in fragments):
            return value
    return None


def canonical_step(step: int, metrics: Mapping[str, float], sources: set[str]) -> dict[str, Any]:
    wait = _first(metrics, "timing/train_wait_time")
    train = _first(metrics, "timing/train_time")
    step_time = _first(metrics, "timing/step_time")
    if step_time is None and wait is not None and train is not None:
        step_time = wait + train
    wait_ratio = _first(metrics, "perf/wait_time_ratio")
    if wait_ratio is None and wait is not None and step_time:
        wait_ratio = wait / step_time

    accepted_groups = _first(
        metrics,
        "polar/accepted/group_count",
        "polar/candidate/accepted_group_count",
        "polar/decision_window/accepted_group_count",
    )
    accepted_sessions = _first(
        metrics,
        "polar/accepted/accounted_sessions",
        "rollout/session_reward_accounted_sessions",
        "polar/reward_accounted_sessions",
        "polar/session_outcome/accounted_count",
        "polar/candidate/accounted_sessions",
        "polar/candidate/trainable_sessions",
    )
    finished_groups = _first(
        metrics,
        "polar/completed_groups",
        "polar/completed_count",
        "polar/scheduler/completed_groups_delta",
    )
    finished_sessions = _first(
        metrics,
        "polar/completed_sessions",
        "polar/completed_session_count",
        "polar/session_status/completed_count",
    )
    token_count = _first(metrics, "polar/session_trainable_response_tokens/count")
    token_mean = _first(metrics, "polar/session_trainable_response_tokens/mean")
    accepted_tokens = token_count * token_mean if token_count is not None and token_mean is not None else None

    sample_age_mean = _first(
        metrics,
        "rollout/sample_age/mean",
        "polar/sample_age/mean",
        "sample_age/mean",
    )
    if sample_age_mean is None:
        sample_age_mean = _matching_metric(metrics, ("sample", "age", "mean"))
    sample_age_p95 = _first(
        metrics,
        "rollout/sample_age/p95",
        "polar/sample_age/p95",
        "sample_age/p95",
    )
    if sample_age_p95 is None:
        sample_age_p95 = _matching_metric(metrics, ("sample", "age", "p95"))

    queue_completed_buffer = _first(metrics, "polar/scheduler/completed_buffer")
    queue_output = _first(metrics, "polar/scheduler/output_queue")
    queue_deferred = _first(metrics, "polar/scheduler/deferred_queue")
    queue_parts = [queue_completed_buffer, queue_output, queue_deferred]
    queue_backlog = (
        sum(value for value in queue_parts if value is not None)
        if any(value is not None for value in queue_parts)
        else None
    )

    return {
        "step": step,
        "sources": sorted(sources),
        "train_wait_time_s": wait,
        "train_time_s": train,
        "step_time_s": step_time,
        "wait_ratio": wait_ratio,
        "actor_train_tokens_per_s": _first(metrics, "perf/actor_train_tok_per_s"),
        "rollout_time_s": _first(metrics, "perf/rollout_time"),
        "service_time_max_s": _first(metrics, "timing/service_time_max"),
        "service_window_s": _first(metrics, "timing/service_window"),
        "rollout_collect_s": (
            _first(metrics, "timing/pipeline_ms/rollout_collect") / 1000.0
            if _first(metrics, "timing/pipeline_ms/rollout_collect") is not None
            else None
        ),
        "staleness_mean": _first(metrics, "polar/staleness/mean"),
        "sample_age_mean": sample_age_mean,
        "sample_age_p95": sample_age_p95,
        "accepted_group_count": accepted_groups,
        "accepted_session_count": accepted_sessions,
        "accepted_trainable_tokens_estimate": accepted_tokens,
        "finished_group_count": finished_groups,
        "finished_session_count": finished_sessions,
        "queue_active_groups": _first(metrics, "polar/scheduler/active_groups"),
        "queue_completed_buffer": queue_completed_buffer,
        "queue_output": queue_output,
        "queue_deferred": queue_deferred,
        "queue_backlog_groups": queue_backlog,
        "queue_outstanding_groups": _first(metrics, "polar/reservations/outstanding_groups"),
    }


def _normalized_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _row_value(row: Mapping[str, str], *aliases: str) -> str | None:
    normalized = {_normalized_header(key): value for key, value in row.items() if key is not None}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def _timestamp(value: str | None) -> float | None:
    number = _float(value)
    if number is not None:
        return number
    if not value:
        return None
    cleaned = value.strip()
    formats = (
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    )
    for fmt in formats:
        try:
            return dt.datetime.strptime(cleaned, fmt).replace(tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            pass
    return None


def _node_id(path: Path) -> int | None:
    match = NODE_ID_RE.search(path.stem + "_")
    return int(match.group(1)) if match else None


def _gpu_role(path: Path, node: int | None, gpu: int | None, config: Mapping[str, Any]) -> str:
    if config.get("mode") == "collocate":
        return "shared"
    stem = path.stem.lower()
    if "actor" in stem or "train" in stem:
        return "actor"
    if "rollout" in stem or "infer" in stem:
        return "rollout"
    explicit = str(config.get("gpu_monitor_node_role") or "").lower()
    if explicit in {"actor", "train"}:
        return "actor"
    if explicit in {"rollout", "inference"}:
        return "rollout"
    if explicit in {"shared", "collocate", "collocated"}:
        return "shared"

    actor_nodes = config.get("actor_nodes")
    actor_gpus = config.get("actor_gpus")
    gpus_per_node = config.get("gpus_per_node")
    num_nodes = config.get("num_nodes")
    if node is not None and actor_nodes is not None and (num_nodes or 0) > 1:
        return "actor" if node < actor_nodes else "rollout"
    if (
        node is not None
        and gpu is not None
        and actor_gpus is not None
        and gpus_per_node is not None
    ):
        return "actor" if node * gpus_per_node + gpu < actor_gpus else "rollout"
    return "unknown"


def parse_gpu_csv(paths: Sequence[Path], config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    for path in paths:
        node = _node_id(path)
        try:
            handle = path.open("r", encoding="utf-8", errors="replace", newline="")
        except OSError:
            continue
        with handle:
            try:
                reader = csv.DictReader(handle)
                for csv_row in reader:
                    gpu = _int(_row_value(csv_row, "gpu", "index", "gpu_index"))
                    util = _float(
                        _row_value(
                            csv_row,
                            "util_gpu_pct",
                            "utilization_gpu",
                            "utilization_gpu_pct",
                            "gpu_utilization",
                        )
                    )
                    timestamp = _timestamp(
                        _row_value(csv_row, "sample_time", "timestamp", "time")
                    )
                    memory_used = _float(_row_value(csv_row, "memory_used_mb", "memory_used"))
                    memory_total = _float(_row_value(csv_row, "memory_total_mb", "memory_total"))
                    if gpu is None and util is None and timestamp is None:
                        malformed += 1
                        continue
                    rows.append(
                        {
                            "file": str(path),
                            "node": node,
                            "gpu": gpu,
                            "device": f"{node if node is not None else path.stem}:{gpu if gpu is not None else '?'}",
                            "role": _gpu_role(path, node, gpu, config),
                            "timestamp": timestamp,
                            "train_step": _int(_row_value(csv_row, "train_step", "step")),
                            "utilization_gpu_pct": util,
                            "memory_used_mb": memory_used,
                            "memory_total_mb": memory_total,
                            "memory_fraction": _safe_div(memory_used, memory_total),
                            "power_w": _float(
                                _row_value(csv_row, "power_draw_w", "power_w", "power_draw")
                            ),
                            "temperature_c": _float(
                                _row_value(csv_row, "temperature_c", "temperature_gpu", "temperature")
                            ),
                        }
                    )
            except csv.Error:
                malformed += 1
    return rows, malformed


def _gpu_hours(rows: Sequence[Mapping[str, Any]]) -> float | None:
    times: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row.get("timestamp"))
        if value is not None:
            times[str(row["device"])].append(value)
    if not times:
        return None
    total_seconds = 0.0
    for values in times.values():
        ordered = sorted(set(values))
        if len(ordered) >= 2:
            total_seconds += ordered[-1] - ordered[0]
    return total_seconds / 3600.0


def _gpu_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "device_count": len({str(row["device"]) for row in rows}),
        "gpu_hours_observed": _gpu_hours(rows),
        "utilization_gpu_pct": stats(row.get("utilization_gpu_pct") for row in rows),
        "memory_used_mb": stats(row.get("memory_used_mb") for row in rows),
        "memory_fraction": stats(row.get("memory_fraction") for row in rows),
        "power_w": stats(row.get("power_w") for row in rows),
        "temperature_c": stats(row.get("temperature_c") for row in rows),
    }


def summarize_gpu(rows: Sequence[Mapping[str, Any]], steady_steps: set[int]) -> dict[str, Any]:
    matching = [row for row in rows if row.get("train_step") in steady_steps]
    steady_rows = matching if matching else list(rows)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in steady_rows:
        grouped[str(row.get("role") or "unknown")].append(row)
    return {
        "steady_filter": "train_step" if matching else "all_samples",
        "all_samples": _gpu_group(rows),
        "steady": _gpu_group(steady_rows),
        "roles": {role: _gpu_group(role_rows) for role, role_rows in sorted(grouped.items())},
    }


STEP_METRICS = (
    "train_wait_time_s",
    "train_time_s",
    "step_time_s",
    "wait_ratio",
    "actor_train_tokens_per_s",
    "rollout_time_s",
    "service_time_max_s",
    "service_window_s",
    "rollout_collect_s",
    "staleness_mean",
    "sample_age_mean",
    "sample_age_p95",
    "accepted_group_count",
    "accepted_session_count",
    "accepted_trainable_tokens_estimate",
    "finished_group_count",
    "finished_session_count",
    "queue_active_groups",
    "queue_completed_buffer",
    "queue_output",
    "queue_deferred",
    "queue_backlog_groups",
    "queue_outstanding_groups",
)


def _per_step_slope(steps: Sequence[Mapping[str, Any]], key: str) -> float | None:
    points = [
        (int(step["step"]), value)
        for step in steps
        if (value := _float(step.get(key))) is not None
    ]
    if len(points) < 2 or points[-1][0] == points[0][0]:
        return None
    return (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])


def _steady_summary(steps: Sequence[Mapping[str, Any]], gpu_count: int | None) -> dict[str, Any]:
    metric_stats = {key: stats(step.get(key) for step in steps) for key in STEP_METRICS}
    elapsed = _float(metric_stats["step_time_s"]["sum"])
    accepted_groups = _float(metric_stats["accepted_group_count"]["sum"])
    accepted_sessions = _float(metric_stats["accepted_session_count"]["sum"])
    accepted_tokens = _float(metric_stats["accepted_trainable_tokens_estimate"]["sum"])
    finished_groups = _float(metric_stats["finished_group_count"]["sum"])
    finished_sessions = _float(metric_stats["finished_session_count"]["sum"])
    estimated_gpu_hours = (
        elapsed * gpu_count / 3600.0
        if elapsed is not None and gpu_count is not None and gpu_count > 0
        else None
    )
    throughput = {
        "elapsed_step_time_s": elapsed,
        "train_steps_per_hour": _safe_div(len(steps) * 3600.0, elapsed),
        "accepted_groups_per_s": _safe_div(accepted_groups, elapsed),
        "accepted_sessions_per_s": _safe_div(accepted_sessions, elapsed),
        "accepted_trainable_tokens_per_s": _safe_div(accepted_tokens, elapsed),
        "finished_groups_per_s": _safe_div(finished_groups, elapsed),
        "finished_sessions_per_s": _safe_div(finished_sessions, elapsed),
        "gpu_hours_estimate": estimated_gpu_hours,
        "accepted_groups_per_gpu_hour": _safe_div(accepted_groups, estimated_gpu_hours),
        "accepted_sessions_per_gpu_hour": _safe_div(accepted_sessions, estimated_gpu_hours),
        "accepted_trainable_tokens_per_gpu_hour": _safe_div(accepted_tokens, estimated_gpu_hours),
    }
    queue_trends = {
        key: _per_step_slope(steps, key)
        for key in (
            "queue_backlog_groups",
            "queue_active_groups",
            "queue_outstanding_groups",
        )
    }
    return {
        "metrics": metric_stats,
        "throughput": throughput,
        "queue_trend_per_step": queue_trends,
    }


def _discover_gpu_csv(job_dir: Path) -> list[Path]:
    if not job_dir.is_dir():
        return []
    found: set[Path] = set()
    monitor_dir = job_dir / "gpu_monitor"
    if monitor_dir.is_dir():
        try:
            found.update(path for path in monitor_dir.glob("*.csv") if path.is_file())
        except OSError:
            pass
    try:
        found.update(
            path
            for path in job_dir.glob("*gpu*.csv")
            if path.is_file()
        )
    except OSError:
        pass
    if not found:
        base_depth = len(job_dir.parts)
        for root, directories, filenames in os.walk(job_dir, onerror=lambda _error: None):
            depth = len(Path(root).parts) - base_depth
            directories[:] = [name for name in directories if name not in PRUNED_TELEMETRY_DIRS]
            if depth >= 4:
                directories[:] = []
            root_path = Path(root)
            for filename in filenames:
                if filename.lower().endswith(".csv") and (
                    "gpu" in filename.lower() or "gpu" in str(root_path).lower()
                ):
                    found.add(root_path / filename)
    return sorted(found)


def summarize_job(job_dir: Path, warmup_steps: int = 1) -> dict[str, Any]:
    config, env_sources, warnings = _load_config(job_dir)
    log_paths = _discover_logs(job_dir)
    by_step, step_sources, parse_failures = parse_perf_logs(log_paths)
    canonical = [canonical_step(step, by_step[step], step_sources.get(step, set())) for step in sorted(by_step)]
    train_ids = [step["step"] for step in canonical if "train" in step["sources"]]
    observed_ids = train_ids or [step["step"] for step in canonical]
    warmup_ids = set(observed_ids[:warmup_steps])
    for step in canonical:
        step["steady"] = step["step"] not in warmup_ids
    steady_steps = [step for step in canonical if step["steady"]]
    steady_ids = {int(step["step"]) for step in steady_steps}

    gpu_paths = _discover_gpu_csv(job_dir)
    gpu_rows, malformed_gpu_rows = parse_gpu_csv(gpu_paths, config)
    gpu = summarize_gpu(gpu_rows, steady_ids)
    observed_gpu_count = _int(gpu["all_samples"].get("device_count"))
    gpu_count = _int(config.get("allocated_gpus")) or observed_gpu_count
    steady = _steady_summary(steady_steps, gpu_count)

    if not job_dir.exists():
        warnings.append("input directory does not exist")
    if not log_paths:
        warnings.append("no profiling log files found")
    elif not canonical:
        warnings.append("no parseable perf records found")
    if parse_failures:
        warnings.append(f"{parse_failures} perf record(s) could not be parsed")
    if canonical and not train_ids:
        warnings.append("no train timing records found; warmup applied to all perf step ids")
    if not any("rollout" in step["sources"] for step in canonical):
        warnings.append("no rollout metric records found")
    if not gpu_paths:
        warnings.append("no GPU monitor CSV files found")
    elif not gpu_rows:
        warnings.append("GPU monitor CSV files contained no usable samples")
    if malformed_gpu_rows:
        warnings.append(f"{malformed_gpu_rows} malformed GPU CSV row(s) skipped")
    if gpu_rows and gpu["steady_filter"] != "train_step":
        warnings.append("GPU CSV has no matching train_step values; utilization includes all samples")
    if len(steady_steps) == 0:
        warnings.append("no steady-state steps remain after warmup")

    job_id = _job_id(job_dir)
    return {
        "job": job_dir.name,
        "job_id": job_id,
        "path": str(job_dir),
        "run_path": str(job_dir.parent if job_id else job_dir),
        "config": config,
        "sources": {
            "config": env_sources,
            "logs": [str(path) for path in log_paths],
            "gpu_csv": [str(path) for path in gpu_paths],
        },
        "warmup_step_ids": sorted(warmup_ids),
        "step_count_total": len(canonical),
        "step_count_steady": len(steady_steps),
        "steps": canonical,
        "steady_state": steady,
        "gpu": gpu,
        "warnings": warnings,
    }


def build_report(inputs: Sequence[Path], warmup_steps: int = 1) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "warmup_steps": warmup_steps,
        "jobs": [summarize_job(path, warmup_steps=warmup_steps) for path in discover_job_dirs(inputs)],
    }


def _fmt(value: Any, digits: int = 2) -> str:
    number = _float(value)
    if number is None:
        return "-"
    if abs(number) >= 10000 or (0 < abs(number) < 0.01):
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def render_table(report: Mapping[str, Any]) -> str:
    headers = [
        "job",
        "mode",
        "A/R GPU",
        "L",
        "steps",
        "step_s",
        "wait%",
        "acc_s/s",
        "stale",
        "q_d/step",
        "GPU%",
        "actor%",
        "rollout%",
        "acc_s/GPUh",
        "warn",
    ]
    rows: list[list[str]] = []
    for job in report.get("jobs", []):
        config = job.get("config", {})
        roles = _nested(job, "gpu", "roles") or {}
        rows.append(
            [
                str(job.get("job_id") or job.get("job") or "-")[-20:],
                str(config.get("mode") or "-")[:11],
                f"{config.get('actor_gpus') or '-'}/{config.get('rollout_gpus') or '-'}",
                str(config.get("async_level") if config.get("async_level") is not None else "-"),
                str(job.get("step_count_steady", 0)),
                _fmt(_nested(job, "steady_state", "metrics", "step_time_s", "mean"), 1),
                _fmt(
                    (_nested(job, "steady_state", "metrics", "wait_ratio", "mean") or 0) * 100
                    if _nested(job, "steady_state", "metrics", "wait_ratio", "mean") is not None
                    else None,
                    1,
                ),
                _fmt(_nested(job, "steady_state", "throughput", "accepted_sessions_per_s"), 3),
                _fmt(_nested(job, "steady_state", "metrics", "staleness_mean", "mean"), 2),
                _fmt(
                    _nested(
                        job,
                        "steady_state",
                        "queue_trend_per_step",
                        "queue_backlog_groups",
                    ),
                    2,
                ),
                _fmt(_nested(job, "gpu", "steady", "utilization_gpu_pct", "mean"), 1),
                _fmt(_nested(roles, "actor", "utilization_gpu_pct", "mean"), 1),
                _fmt(_nested(roles, "rollout", "utilization_gpu_pct", "mean"), 1),
                _fmt(
                    _nested(
                        job,
                        "steady_state",
                        "throughput",
                        "accepted_sessions_per_gpu_hour",
                    ),
                    2,
                ),
                str(len(job.get("warnings", []))),
            ]
        )
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    lines = ["  ".join(header.ljust(width) for header, width in zip(headers, widths))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(value.ljust(width) for value, width in zip(row, widths)) for row in rows)
    if not rows:
        lines.append("(no inputs)")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="job directories or logical run directories containing job-*",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="number of earliest train steps to exclude (default: 1)",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="also write the complete report as JSON; use '-' for stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.warmup_steps < 0:
        _parser().error("--warmup-steps must be non-negative")
    report = build_report(args.paths, warmup_steps=args.warmup_steps)
    table = render_table(report)
    if args.json == "-":
        print(table, file=sys.stderr)
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(table)
        if args.json:
            destination = Path(args.json).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
