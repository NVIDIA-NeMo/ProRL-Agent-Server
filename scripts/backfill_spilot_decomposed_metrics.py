#!/usr/bin/env python3
"""Reconstruct SPilot reward/cost metrics and optionally append them to W&B.

The default mode is deliberately local and read-only.  It loads every durable
rollout metrics journal from step 0 through ``train_progress.step`` (inclusive),
reconstructs one-session-one-vote metrics, cross-checks the legacy aggregates,
and prints JSON Lines.  No W&B module is imported and no network call is made
unless ``--commit`` is supplied.

The commit path never attempts to write old W&B ``_step`` values.  It resumes
the exact existing run with ``resume="must"`` and appends rows at the history
tail.  A namespaced business axis (``postrun_v1/rollout_step`` by default)
keeps the reconstructed rollout number independent from W&B's append-only
history step.

The journal files are trusted training outputs and require ``torch.load(...,
weights_only=False)`` because they contain Slime ``Sample`` objects.  Run this
script only against journals produced by the trusted training job.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


JOURNAL_VERSION = 1
DEFAULT_METRIC_PREFIX = "postrun_v1"
DEFAULT_READBACK_TIMEOUT_S = 120.0
DEFAULT_READBACK_INTERVAL_S = 5.0
_METRIC_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class BackfillError(RuntimeError):
    """Raised when local or remote evidence is unsafe to backfill."""


@dataclass(frozen=True)
class SessionRecord:
    """The bounded, trace-invariant fields used for one session's metrics."""

    session_id: str
    accuracy_outcome: float
    cost_penalty_fraction: float
    total_cost: float
    candidate_alias_pair: tuple[str, str] | None
    route_candidate: str | None
    cost_candidate_c0: float
    cost_candidate_c1: float
    cost_unattributed: float
    cost_solve: float
    cost_verify: float
    cost_other_role: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="run checkpoint directory containing train_progress.step and rollout/",
    )
    parser.add_argument(
        "--entity",
        default="",
        help="exact W&B entity; required with --commit (never inferred from login defaults)",
    )
    parser.add_argument(
        "--project",
        default="",
        help="exact W&B project; required with --commit",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="exact W&B run ID; defaults to the checkpoint directory name in dry-run mode",
    )
    parser.add_argument(
        "--metric-prefix",
        default=DEFAULT_METRIC_PREFIX,
        help="isolated W&B namespace for reconstructed rows",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="first journal step to reconstruct; the safe/default full backfill starts at 0",
    )
    parser.add_argument(
        "--expected-completed-step",
        type=int,
        default=None,
        help="abort unless train_progress.step has this value",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="append missing rows to the exact existing W&B run; absent means local dry-run",
    )
    parser.add_argument(
        "--wandb-dir",
        type=Path,
        default=None,
        help="local W&B staging directory; defaults to a fresh directory under /tmp",
    )
    parser.add_argument(
        "--readback-timeout-s",
        type=float,
        default=DEFAULT_READBACK_TIMEOUT_S,
        help="maximum wait for Public API readback after a commit",
    )
    parser.add_argument(
        "--readback-interval-s",
        type=float,
        default=DEFAULT_READBACK_INTERVAL_S,
        help="poll interval for post-commit Public API readback",
    )
    args = parser.parse_args(argv)

    if args.start_step < 0:
        parser.error("--start-step must be non-negative")
    if args.expected_completed_step is not None and args.expected_completed_step < 0:
        parser.error("--expected-completed-step must be non-negative")
    if not math.isfinite(args.readback_timeout_s) or args.readback_timeout_s <= 0:
        parser.error("--readback-timeout-s must be finite and positive")
    if not math.isfinite(args.readback_interval_s) or args.readback_interval_s <= 0:
        parser.error("--readback-interval-s must be finite and positive")
    try:
        args.metric_prefix = _normalize_metric_prefix(args.metric_prefix)
    except BackfillError as exc:
        parser.error(str(exc))
    if not args.run_id:
        args.run_id = args.checkpoint_root.resolve().name
    if args.commit:
        missing = [name for name in ("entity", "project", "run_id") if not getattr(args, name)]
        if missing:
            parser.error("--commit requires explicit --" + ", --".join(missing))
        if not os.environ.get("WANDB_API_KEY"):
            parser.error(
                "--commit requires WANDB_API_KEY from the approved secret injection; "
                "default .netrc credentials are intentionally rejected"
            )
    return args


def _normalize_metric_prefix(value: str) -> str:
    prefix = value.strip().strip("/")
    if not prefix:
        raise BackfillError("metric prefix must not be empty")
    components = prefix.split("/")
    if any(not _METRIC_COMPONENT_RE.fullmatch(component) for component in components):
        raise BackfillError(
            "metric prefix components may contain only letters, digits, '_', '-', and '.'"
        )
    if prefix.startswith("_"):
        raise BackfillError("metric prefix must not use a reserved leading underscore")
    return prefix


def _metric(prefix: str, suffix: str) -> str:
    return f"{prefix}/{suffix}"


def _axis_key(prefix: str) -> str:
    return _metric(prefix, "rollout_step")


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise BackfillError(f"{field} must be numeric, not boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BackfillError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise BackfillError(f"{field} must be finite, got {parsed!r}")
    return parsed


def _unit_interval_float(value: Any, *, field: str) -> float:
    parsed = _finite_float(value, field=field)
    if not 0.0 <= parsed <= 1.0:
        raise BackfillError(f"{field} must be in [0, 1], got {parsed!r}")
    return parsed


def _nonnegative_float(value: Any, *, field: str) -> float:
    parsed = _finite_float(value, field=field)
    if parsed < 0.0:
        raise BackfillError(f"{field} must be non-negative, got {parsed!r}")
    return parsed


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BackfillError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BackfillError(f"{field} must be a sequence")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)


def _require_close(left: float, right: float, *, field: str) -> None:
    if not _close(left, right):
        raise BackfillError(f"{field} mismatch: {left!r} != {right!r}")


def read_completed_step(checkpoint_root: Path) -> int:
    progress_path = checkpoint_root / "train_progress.step"
    try:
        raw = progress_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise BackfillError(f"cannot read completed-step marker {progress_path}: {exc}") from exc
    try:
        step = int(raw)
    except ValueError as exc:
        raise BackfillError(f"invalid completed-step marker {progress_path}: {raw!r}") from exc
    if step < 0:
        raise BackfillError(f"completed step must be non-negative, got {step}")
    return step


def journal_paths(checkpoint_root: Path, step: int) -> tuple[Path, Path]:
    journal_dir = checkpoint_root / "rollout" / "rollout_metrics_journal"
    stem = f"rollout_{step:07d}"
    return journal_dir / f"{stem}.pending.pt", journal_dir / f"{stem}.emitted"


def _validate_emitted_marker(path: Path, expected_step: int) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BackfillError(f"cannot read emitted marker {path}: {exc}") from exc
    fields: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            fields[key] = value
    if fields != {"version": str(JOURNAL_VERSION), "rollout_id": str(expected_step)}:
        raise BackfillError(f"invalid emitted marker {path}: {fields!r}")


def load_journal(path: Path, expected_step: int) -> tuple[list[Any], Mapping[str, Any]]:
    """Load a trusted Slime journal with the training Python environment."""

    try:
        import torch
    except ImportError as exc:
        raise BackfillError(
            "torch is required to read pending.pt; use the training Python environment "
            "rather than the host's default python"
        ) from exc

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise BackfillError(f"failed to load trusted journal {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BackfillError(f"journal {path} does not contain a dict payload")
    if payload.get("version") != JOURNAL_VERSION:
        raise BackfillError(f"journal {path} has unsupported version {payload.get('version')!r}")
    if payload.get("rollout_id") != expected_step:
        raise BackfillError(
            f"journal {path} rollout id {payload.get('rollout_id')!r} != {expected_step}"
        )
    pending = payload.get("pending")
    samples = getattr(pending, "samples", None)
    if not isinstance(samples, list) or not samples:
        raise BackfillError(
            f"journal {path} has no retained Sample list; compact journals cannot be decomposed"
        )
    extra_metrics = getattr(pending, "extra_metrics", None)
    if not isinstance(extra_metrics, Mapping):
        raise BackfillError(f"journal {path} has no extra_metrics mapping")
    return samples, extra_metrics


def _candidate_maps(
    router: Mapping[str, Any], *, session_id: str
) -> tuple[dict[str, str], dict[str, str], tuple[str, str] | None]:
    slot_mapping = router.get("slot_mapping")
    # Match the live fail-closed attribution: invalid-action/error sessions can
    # retain valid accuracy/cost evidence without a usable slot map. Their
    # route and call cost stay in the explicit unattributed buckets.
    if not isinstance(slot_mapping, Mapping):
        return {}, {}, None
    aliases_by_slot: dict[str, str] = {}
    for raw_slot, raw_candidate in slot_mapping.items():
        candidate = _mapping(raw_candidate, field=f"{session_id}.slot_mapping[{raw_slot!r}]")
        alias = candidate.get("model")
        if not isinstance(alias, str) or not alias:
            raise BackfillError(f"{session_id}.slot_mapping[{raw_slot!r}].model is invalid")
        aliases_by_slot[str(raw_slot).upper()] = alias
    aliases = sorted(set(aliases_by_slot.values()))
    if len(aliases_by_slot) != 2 or len(aliases) != 2:
        return {}, {}, None
    alias_pair = (aliases[0], aliases[1])
    candidate_by_alias = {alias: f"C{index}" for index, alias in enumerate(aliases)}
    candidate_by_slot = {
        slot: candidate_by_alias[alias] for slot, alias in aliases_by_slot.items()
    }
    return candidate_by_slot, candidate_by_alias, alias_pair


def _excluded_router_sample(sample: Any) -> bool:
    status = getattr(sample, "status", None)
    status_name = str(getattr(status, "name", status) or "").rsplit(".", 1)[-1].upper()
    loss_mask = getattr(sample, "loss_mask", None)
    has_trainable_token = bool(
        loss_mask is not None and any(int(value) != 0 for value in loss_mask)
    )
    return (
        getattr(sample, "remove_sample", False) is True
        and status_name in {"FAILED", "ABORTED"}
        and not has_trainable_token
    )


def _sample_session_id(sample: Any) -> str:
    metadata = _mapping(getattr(sample, "metadata", None), field="sample.metadata")
    polar = _mapping(metadata.get("polar"), field="sample.metadata.polar")
    session_id = polar.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise BackfillError("sample has no stable polar.session_id")
    return session_id


def session_record_from_sample(sample: Any, expected_step: int) -> SessionRecord | None:
    metadata = _mapping(getattr(sample, "metadata", None), field="sample.metadata")
    polar = _mapping(metadata.get("polar"), field="sample.metadata.polar")
    accepted_step = polar.get("accepted_rollout_id")
    try:
        accepted_step_int = int(accepted_step)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BackfillError(f"invalid accepted_rollout_id {accepted_step!r}") from exc
    if accepted_step_int != expected_step:
        raise BackfillError(
            f"sample accepted_rollout_id {accepted_step_int} != journal step {expected_step}"
        )
    session_id = _sample_session_id(sample)

    trajectory = _mapping(
        polar.get("trajectory_metadata"), field=f"{session_id}.trajectory_metadata"
    )
    evaluation_value = trajectory.get("evaluation")
    if not isinstance(evaluation_value, Mapping):
        if _excluded_router_sample(sample):
            return None
        raise BackfillError(f"{session_id}.evaluation must be a mapping")
    evaluation = evaluation_value
    router_value = evaluation.get("spilot_router")
    if not isinstance(router_value, Mapping):
        if _excluded_router_sample(sample):
            return None
        raise BackfillError(f"{session_id}.spilot_router must be a mapping")
    router = router_value

    accuracy = _unit_interval_float(
        evaluation.get("harbor_outcome_reward"),
        field=f"{session_id}.harbor_outcome_reward",
    )
    penalty = _unit_interval_float(
        evaluation.get("applied_cost_penalty"),
        field=f"{session_id}.applied_cost_penalty",
    )
    total_cost = _nonnegative_float(
        evaluation.get("total_cost"), field=f"{session_id}.evaluation.total_cost"
    )
    router_total_cost = _nonnegative_float(
        router.get("total_cost"), field=f"{session_id}.spilot_router.total_cost"
    )
    _require_close(total_cost, router_total_cost, field=f"{session_id}.total_cost")

    candidate_by_slot, candidate_by_alias, candidate_alias_pair = _candidate_maps(
        router, session_id=session_id
    )
    route_candidate: str | None = None
    actions = _sequence(router.get("actions", []), field=f"{session_id}.actions")
    for raw_action in actions:
        action = _mapping(raw_action, field=f"{session_id}.action")
        if action.get("valid") is True and str(action.get("action") or "").upper() == "ROUTE":
            route_candidate = candidate_by_slot.get(str(action.get("model_slot") or "").upper())
            break

    cost_by_candidate = {"C0": 0.0, "C1": 0.0, "unattributed": 0.0}
    cost_by_role = {"solve": 0.0, "verify": 0.0, "other": 0.0}
    calls = _sequence(router.get("calls", []), field=f"{session_id}.calls")
    for index, raw_call in enumerate(calls):
        call = _mapping(raw_call, field=f"{session_id}.calls[{index}]")
        cost = _nonnegative_float(call.get("cost"), field=f"{session_id}.calls[{index}].cost")
        model_candidate = candidate_by_alias.get(call.get("model"))
        slot_candidate = candidate_by_slot.get(str(call.get("slot") or "").upper())
        candidate = (
            model_candidate
            if model_candidate is not None and model_candidate == slot_candidate
            else "unattributed"
        )
        cost_by_candidate[candidate] += cost
        role = str(call.get("role") or "").lower()
        cost_by_role[role if role in {"solve", "verify"} else "other"] += cost

    _require_close(
        sum(cost_by_candidate.values()),
        total_cost,
        field=f"{session_id}.call_cost_sum",
    )
    _require_close(
        sum(cost_by_role.values()),
        total_cost,
        field=f"{session_id}.role_cost_sum",
    )
    return SessionRecord(
        session_id=session_id,
        accuracy_outcome=accuracy,
        cost_penalty_fraction=penalty,
        total_cost=total_cost,
        candidate_alias_pair=candidate_alias_pair,
        route_candidate=route_candidate,
        cost_candidate_c0=cost_by_candidate["C0"],
        cost_candidate_c1=cost_by_candidate["C1"],
        cost_unattributed=cost_by_candidate["unattributed"],
        cost_solve=cost_by_role["solve"],
        cost_verify=cost_by_role["verify"],
        cost_other_role=cost_by_role["other"],
    )


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def _reconstruct_step_with_candidate_pair(
    step: int,
    samples: Sequence[Any],
    extra_metrics: Mapping[str, Any],
    *,
    metric_prefix: str = DEFAULT_METRIC_PREFIX,
) -> tuple[dict[str, float | int], tuple[str, str] | None]:
    records: dict[str, SessionRecord] = {}
    excluded_sessions: set[str] = set()
    for sample in samples:
        session_id = _sample_session_id(sample)
        record = session_record_from_sample(sample, step)
        if record is None:
            if session_id in records:
                raise BackfillError(f"session {session_id!r} mixes excluded and evaluated traces")
            excluded_sessions.add(session_id)
            continue
        if session_id in excluded_sessions:
            raise BackfillError(f"session {session_id!r} mixes excluded and evaluated traces")
        previous = records.setdefault(record.session_id, record)
        if previous != record:
            raise BackfillError(
                f"session {record.session_id!r} has conflicting trace-level decomposition fields"
            )
    if not records:
        raise BackfillError(f"journal step {step} contains no sessions")

    sessions = list(records.values())
    candidate_alias_pairs = {
        record.candidate_alias_pair
        for record in sessions
        if record.candidate_alias_pair is not None
    }
    if len(candidate_alias_pairs) > 1:
        raise BackfillError(
            f"journal step {step} contains multiple sorted candidate alias pairs: "
            f"{sorted(candidate_alias_pairs)!r}"
        )
    candidate_alias_pair = next(iter(candidate_alias_pairs), None)
    raw_mean = _mean(record.accuracy_outcome for record in sessions)
    reward_accounted_count_value = _nonnegative_float(
        extra_metrics.get("polar/spilot_router/reward_accounted_session_count"),
        field="polar/spilot_router/reward_accounted_session_count",
    )
    reward_accounted_count = int(reward_accounted_count_value)
    if reward_accounted_count_value != reward_accounted_count:
        raise BackfillError(
            "polar/spilot_router/reward_accounted_session_count must be an integer, "
            f"got {reward_accounted_count_value!r}"
        )
    if reward_accounted_count > len(sessions):
        raise BackfillError(
            "polar/spilot_router/reward_accounted_session_count exceeds Router "
            f"evaluation coverage: {reward_accounted_count} > {len(sessions)}"
        )
    if reward_accounted_count > 0.0:
        shaped_mean = _finite_float(
            extra_metrics.get("polar/spilot_router/reward_mean"),
            field="polar/spilot_router/reward_mean",
        )
    else:
        shaped_mean = 0.0
    penalty_fraction_mean = _mean(record.cost_penalty_fraction for record in sessions)
    penalty_contributions = [
        record.accuracy_outcome * record.cost_penalty_fraction for record in sessions
    ]
    penalty_contribution_mean = _mean(penalty_contributions)
    cost_total = sum(record.total_cost for record in sessions)
    per_session_costs = [record.total_cost for record in sessions]
    route_c0 = [record for record in sessions if record.route_candidate == "C0"]
    route_c1 = [record for record in sessions if record.route_candidate == "C1"]
    route_unattributed_count = len(sessions) - len(route_c0) - len(route_c1)

    prefix = _normalize_metric_prefix(metric_prefix)
    row: dict[str, float | int] = {
        _axis_key(prefix): step,
        _metric(prefix, "source_journal_version"): JOURNAL_VERSION,
        _metric(prefix, "trace_count"): len(samples),
        _metric(prefix, "session_count"): len(sessions),
        _metric(prefix, "excluded_session_count"): len(excluded_sessions),
        _metric(prefix, "accuracy_outcome"): raw_mean,
        _metric(prefix, "accuracy_outcome_accounted_session_count"): len(sessions),
        _metric(prefix, "accuracy_outcome_positive_count"): sum(
            record.accuracy_outcome > 0.0 for record in sessions
        ),
        _metric(prefix, "reward_shaped"): shaped_mean,
        _metric(prefix, "reward_accounted_session_count"): reward_accounted_count,
        _metric(prefix, "cost_penalty_fraction"): penalty_fraction_mean,
        _metric(prefix, "cost_penalty_contribution"): penalty_contribution_mean,
        _metric(prefix, "cost_penalty_contribution_total"): sum(penalty_contributions),
        _metric(prefix, "cost_total"): cost_total,
        _metric(prefix, "cost_accounted_session_count"): len(sessions),
        _metric(prefix, "cost_per_session"): cost_total / len(sessions),
        _metric(prefix, "cost_per_session_std"): (
            statistics.pstdev(per_session_costs) if len(per_session_costs) > 1 else 0.0
        ),
        _metric(prefix, "cost_per_session_median"): statistics.median(per_session_costs),
        _metric(prefix, "cost_per_session_min"): min(per_session_costs),
        _metric(prefix, "cost_per_session_max"): max(per_session_costs),
        _metric(prefix, "cost_candidate_c0_total"): sum(
            record.cost_candidate_c0 for record in sessions
        ),
        _metric(prefix, "cost_candidate_c1_total"): sum(
            record.cost_candidate_c1 for record in sessions
        ),
        _metric(prefix, "cost_unattributed_total"): sum(
            record.cost_unattributed for record in sessions
        ),
        _metric(prefix, "cost_solve_total"): sum(record.cost_solve for record in sessions),
        _metric(prefix, "cost_verify_total"): sum(record.cost_verify for record in sessions),
        _metric(prefix, "cost_other_role_total"): sum(
            record.cost_other_role for record in sessions
        ),
        _metric(prefix, "route_candidate_c0_count"): len(route_c0),
        _metric(prefix, "route_candidate_c1_count"): len(route_c1),
        _metric(prefix, "route_unattributed_count"): route_unattributed_count,
        _metric(prefix, "accuracy_outcome_candidate_c0_accounted_session_count"): len(route_c0),
        _metric(prefix, "accuracy_outcome_candidate_c1_accounted_session_count"): len(route_c1),
    }
    if route_c0:
        row[_metric(prefix, "accuracy_outcome_candidate_c0")] = _mean(
            record.accuracy_outcome for record in route_c0
        )
    if route_c1:
        row[_metric(prefix, "accuracy_outcome_candidate_c1")] = _mean(
            record.accuracy_outcome for record in route_c1
        )
    _validate_row_decomposition(row, prefix=prefix)
    _cross_check_legacy_metrics(row, extra_metrics, prefix=prefix, step=step)
    return row, candidate_alias_pair


def reconstruct_step(
    step: int,
    samples: Sequence[Any],
    extra_metrics: Mapping[str, Any],
    *,
    metric_prefix: str = DEFAULT_METRIC_PREFIX,
) -> dict[str, float | int]:
    row, _candidate_alias_pair = _reconstruct_step_with_candidate_pair(
        step,
        samples,
        extra_metrics,
        metric_prefix=metric_prefix,
    )
    return row


def _validate_row_decomposition(row: Mapping[str, float | int], *, prefix: str) -> None:
    cost_total = float(row[_metric(prefix, "cost_total")])
    _require_close(
        cost_total,
        float(row[_metric(prefix, "cost_candidate_c0_total")])
        + float(row[_metric(prefix, "cost_candidate_c1_total")])
        + float(row[_metric(prefix, "cost_unattributed_total")]),
        field="candidate cost decomposition",
    )
    _require_close(
        cost_total,
        float(row[_metric(prefix, "cost_solve_total")])
        + float(row[_metric(prefix, "cost_verify_total")])
        + float(row[_metric(prefix, "cost_other_role_total")]),
        field="role cost decomposition",
    )
    _require_close(
        float(row[_metric(prefix, "session_count")]),
        float(row[_metric(prefix, "route_candidate_c0_count")])
        + float(row[_metric(prefix, "route_candidate_c1_count")])
        + float(row[_metric(prefix, "route_unattributed_count")]),
        field="route count decomposition",
    )


def _cross_check_legacy_metrics(
    row: Mapping[str, float | int],
    extra_metrics: Mapping[str, Any],
    *,
    prefix: str,
    step: int,
) -> None:
    checks = {
        "polar/spilot_router/session_count": _metric(prefix, "session_count"),
        "polar/spilot_router/reward_mean": _metric(prefix, "reward_shaped"),
        "polar/spilot_router/total_cost": _metric(prefix, "cost_total"),
        "polar/spilot_router/route_candidate_c0_count": _metric(
            prefix, "route_candidate_c0_count"
        ),
        "polar/spilot_router/route_candidate_c1_count": _metric(
            prefix, "route_candidate_c1_count"
        ),
    }
    for legacy_key, reconstructed_key in checks.items():
        if legacy_key not in extra_metrics:
            raise BackfillError(f"journal step {step} is missing legacy metric {legacy_key!r}")
        legacy_value = _finite_float(extra_metrics[legacy_key], field=legacy_key)
        _require_close(
            legacy_value,
            float(row[reconstructed_key]),
            field=f"step {step} cross-check {legacy_key}",
        )


JournalLoader = Callable[[Path, int], tuple[list[Any], Mapping[str, Any]]]


def reconstruct_rows(
    checkpoint_root: Path,
    *,
    metric_prefix: str = DEFAULT_METRIC_PREFIX,
    start_step: int = 0,
    expected_completed_step: int | None = None,
    loader: JournalLoader = load_journal,
) -> tuple[int, list[dict[str, float | int]]]:
    checkpoint_root = checkpoint_root.resolve()
    completed_before = read_completed_step(checkpoint_root)
    if expected_completed_step is not None and completed_before != expected_completed_step:
        raise BackfillError(
            f"train_progress.step is {completed_before}, expected {expected_completed_step}"
        )
    if start_step > completed_before:
        raise BackfillError(f"start step {start_step} is after completed step {completed_before}")

    rows: list[dict[str, float | int]] = []
    canonical_candidate_alias_pair: tuple[str, str] | None = None
    for step in range(start_step, completed_before + 1):
        pending_path, emitted_path = journal_paths(checkpoint_root, step)
        if not pending_path.is_file():
            raise BackfillError(f"missing durable journal {pending_path}")
        if not emitted_path.is_file():
            raise BackfillError(
                f"journal step {step} has no emitted marker; refusing an uncommitted batch"
            )
        _validate_emitted_marker(emitted_path, step)
        samples, extra_metrics = loader(pending_path, step)
        try:
            row, step_candidate_alias_pair = _reconstruct_step_with_candidate_pair(
                step,
                samples,
                extra_metrics,
                metric_prefix=metric_prefix,
            )
        except BackfillError as exc:
            raise BackfillError(f"journal step {step}: {exc}") from exc
        if step_candidate_alias_pair is not None:
            if canonical_candidate_alias_pair is None:
                canonical_candidate_alias_pair = step_candidate_alias_pair
            elif step_candidate_alias_pair != canonical_candidate_alias_pair:
                raise BackfillError(
                    "sorted candidate alias pair drift at journal step "
                    f"{step}: expected {canonical_candidate_alias_pair!r}, "
                    f"got {step_candidate_alias_pair!r}"
                )
        rows.append(row)

    if canonical_candidate_alias_pair is None:
        raise BackfillError("no canonical two-candidate alias pair was found in the journals")

    completed_after = read_completed_step(checkpoint_root)
    if completed_after != completed_before:
        raise BackfillError(
            "train_progress.step changed while journals were read "
            f"({completed_before} -> {completed_after}); wait for all writers and retry"
        )
    return completed_before, rows


def _remote_project_name(remote_run: Any) -> str:
    project = getattr(remote_run, "project", None)
    return str(getattr(project, "name", project) or "")


def preflight_remote_run(api: Any, *, entity: str, project: str, run_id: str) -> Any:
    path = f"{entity}/{project}/{run_id}"
    try:
        remote_run = api.run(path)
    except Exception as exc:
        raise BackfillError(f"cannot access exact W&B run {path}: {exc}") from exc
    actual = (
        str(getattr(remote_run, "entity", "")),
        _remote_project_name(remote_run),
        str(getattr(remote_run, "id", "")),
    )
    expected = (entity, project, run_id)
    if actual != expected:
        raise BackfillError(f"W&B identity mismatch: expected {expected!r}, got {actual!r}")
    state = str(getattr(remote_run, "state", "")).lower()
    if state != "finished":
        raise BackfillError(
            f"W&B run state is {state or 'unavailable'}; only 'finished' is safe for backfill"
        )
    return remote_run


def _step_from_remote(value: Any, *, axis_key: str) -> int:
    parsed = _finite_float(value, field=axis_key)
    step = int(parsed)
    if parsed != step or step < 0:
        raise BackfillError(f"remote {axis_key} must be a non-negative integer, got {value!r}")
    return step


def scan_remote_rows(
    remote_run: Any,
    *,
    axis_key: str,
    metric_keys: Sequence[str],
    expected_keys_by_step: Mapping[int, set[str]] | None = None,
) -> dict[int, dict[str, float | int]]:
    requested_keys = [axis_key, *metric_keys]
    rows: dict[int, dict[str, float | int]] = {}
    try:
        history = remote_run.scan_history(
            keys=requested_keys,
            page_size=1_000,
            use_cache=False,
        )
        for raw_row in history:
            if axis_key not in raw_row or raw_row[axis_key] is None:
                continue
            step = _step_from_remote(raw_row[axis_key], axis_key=axis_key)
            if step in rows:
                raise BackfillError(
                    f"remote history has duplicate {axis_key}={step}; manual repair is required"
                )
            expected_keys = (
                expected_keys_by_step.get(step, set())
                if expected_keys_by_step is not None
                else set(metric_keys)
            )
            missing = [key for key in expected_keys if key not in raw_row or raw_row[key] is None]
            if missing:
                raise BackfillError(
                    f"remote history row {axis_key}={step} is partial; missing {missing!r}"
                )
            rows[step] = {
                axis_key: step,
                **{
                    key: _finite_float(raw_row[key], field=f"remote {key}")
                    for key in metric_keys
                    if key in raw_row and raw_row[key] is not None
                },
            }
    except BackfillError:
        raise
    except Exception as exc:
        raise BackfillError(f"failed to scan existing W&B backfill history: {exc}") from exc
    return rows


def _metric_keys(rows: Sequence[Mapping[str, float | int]], *, axis_key: str) -> list[str]:
    if not rows:
        raise BackfillError("no local rows were reconstructed")
    return sorted(set().union(*(row.keys() for row in rows)) - {axis_key})


def _expected_keys_by_step(
    rows: Sequence[Mapping[str, float | int]], *, axis_key: str
) -> dict[int, set[str]]:
    expected: dict[int, set[str]] = {}
    for row in rows:
        step = _step_from_remote(row[axis_key], axis_key=axis_key)
        if step in expected:
            raise BackfillError(f"duplicate local step {step}")
        expected[step] = set(row) - {axis_key}
    return expected


def compare_remote_rows(
    local_rows: Sequence[Mapping[str, float | int]],
    remote_rows: Mapping[int, Mapping[str, float | int]],
    *,
    axis_key: str,
) -> list[dict[str, float | int]]:
    local_by_step: dict[int, Mapping[str, float | int]] = {}
    for row in local_rows:
        step = _step_from_remote(row[axis_key], axis_key=axis_key)
        if step in local_by_step:
            raise BackfillError(f"duplicate local step {step}")
        local_by_step[step] = row
    unexpected = sorted(set(remote_rows) - set(local_by_step))
    if unexpected:
        raise BackfillError(
            f"remote namespace contains steps outside the requested local range: {unexpected!r}"
        )

    missing: list[dict[str, float | int]] = []
    for step in sorted(local_by_step):
        local_row = local_by_step[step]
        remote_row = remote_rows.get(step)
        if remote_row is None:
            missing.append(dict(local_row))
            continue
        if set(remote_row) != set(local_row):
            raise BackfillError(f"remote {axis_key}={step} schema differs from the local schema")
        for key, local_value in local_row.items():
            remote_value = remote_row[key]
            if key == axis_key:
                equal = int(remote_value) == int(local_value)
            else:
                equal = _close(float(remote_value), float(local_value))
            if not equal:
                raise BackfillError(
                    f"remote {axis_key}={step} differs at {key}: "
                    f"{remote_value!r} != {local_value!r}"
                )
    return missing


def _assert_progress_unchanged(checkpoint_root: Path, expected_step: int) -> None:
    observed = read_completed_step(checkpoint_root.resolve())
    if observed != expected_step:
        raise BackfillError(
            f"train_progress.step changed before commit ({expected_step} -> {observed}); aborting"
        )


def _wandb_staging_dir(configured: Path | None) -> Path:
    if configured is not None:
        configured = configured.resolve()
        configured.mkdir(parents=True, exist_ok=True)
        return configured
    return Path(tempfile.mkdtemp(prefix="spilot-wandb-backfill-"))


def append_missing_rows(
    wandb_module: Any,
    missing_rows: Sequence[Mapping[str, float | int]],
    *,
    entity: str,
    project: str,
    run_id: str,
    axis_key: str,
    metric_keys: Sequence[str],
    wandb_dir: Path,
) -> None:
    if not missing_rows:
        return
    settings = wandb_module.Settings(
        mode="online",
        console="off",
        x_disable_stats=True,
    )
    run = wandb_module.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="must",
        dir=str(wandb_dir),
        settings=settings,
    )
    if run is None:
        raise BackfillError("wandb.init returned no run")
    actual = (
        str(getattr(run, "entity", "")),
        str(getattr(run, "project", "")),
        str(getattr(run, "id", "")),
    )
    expected = (entity, project, run_id)
    if actual != expected:
        raise BackfillError(f"wandb.init identity mismatch: expected {expected!r}, got {actual!r}")

    try:
        run.define_metric(axis_key, summary="none")
        for key in metric_keys:
            run.define_metric(key, step_metric=axis_key, summary="none")
        for row in missing_rows:
            # Intentionally omit the SDK step= argument.  `_step` must append
            # at the history tail; axis_key carries the historical rollout id.
            run.log(dict(row))
    except Exception:
        # Best effort flush without labeling the original training run failed.
        # Readback will determine exactly which rows reached the server.
        try:
            run.finish(exit_code=0)
        finally:
            raise
    else:
        run.finish(exit_code=0)


def wait_for_readback(
    api: Any,
    local_rows: Sequence[Mapping[str, float | int]],
    *,
    entity: str,
    project: str,
    run_id: str,
    axis_key: str,
    metric_keys: Sequence[str],
    timeout_s: float,
    interval_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: BackfillError | None = None
    while True:
        try:
            remote_run = api.run(f"{entity}/{project}/{run_id}")
            remote_rows = scan_remote_rows(
                remote_run,
                axis_key=axis_key,
                metric_keys=metric_keys,
                expected_keys_by_step=_expected_keys_by_step(local_rows, axis_key=axis_key),
            )
            missing = compare_remote_rows(
                local_rows,
                remote_rows,
                axis_key=axis_key,
            )
            if not missing:
                return
            last_error = BackfillError(
                "readback is still missing rollout steps "
                + repr([int(row[axis_key]) for row in missing])
            )
        except BackfillError as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            raise BackfillError(f"W&B readback did not converge: {last_error}")
        sleep(interval_s)


def commit_rows(
    rows: Sequence[Mapping[str, float | int]],
    *,
    checkpoint_root: Path,
    completed_step: int,
    entity: str,
    project: str,
    run_id: str,
    metric_prefix: str,
    wandb_dir: Path | None,
    readback_timeout_s: float,
    readback_interval_s: float,
    wandb_module: Any | None = None,
) -> int:
    if not os.environ.get("WANDB_API_KEY"):
        raise BackfillError("WANDB_API_KEY is required from the approved secret injection")
    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise BackfillError("wandb is required only for --commit") from exc

    api = wandb_module.Api()
    remote_run = preflight_remote_run(api, entity=entity, project=project, run_id=run_id)
    axis_key = _axis_key(metric_prefix)
    metric_keys = _metric_keys(rows, axis_key=axis_key)
    expected_keys_by_step = _expected_keys_by_step(rows, axis_key=axis_key)
    existing = scan_remote_rows(
        remote_run,
        axis_key=axis_key,
        metric_keys=metric_keys,
        expected_keys_by_step=expected_keys_by_step,
    )
    missing = compare_remote_rows(rows, existing, axis_key=axis_key)
    _assert_progress_unchanged(checkpoint_root, completed_step)
    if missing:
        staging_dir = _wandb_staging_dir(wandb_dir)
        append_missing_rows(
            wandb_module,
            missing,
            entity=entity,
            project=project,
            run_id=run_id,
            axis_key=axis_key,
            metric_keys=metric_keys,
            wandb_dir=staging_dir,
        )
    wait_for_readback(
        api,
        rows,
        entity=entity,
        project=project,
        run_id=run_id,
        axis_key=axis_key,
        metric_keys=metric_keys,
        timeout_s=readback_timeout_s,
        interval_s=readback_interval_s,
    )
    return len(missing)


def _print_rows(
    *,
    mode: str,
    checkpoint_root: Path,
    completed_step: int,
    metric_prefix: str,
    rows: Sequence[Mapping[str, float | int]],
) -> None:
    print(
        json.dumps(
            {
                "type": "summary",
                "mode": mode,
                "checkpoint_root": str(checkpoint_root.resolve()),
                "completed_step": completed_step,
                "metric_prefix": metric_prefix,
                "row_count": len(rows),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    for row in rows:
        print(json.dumps(row, sort_keys=True, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        completed_step, rows = reconstruct_rows(
            args.checkpoint_root,
            metric_prefix=args.metric_prefix,
            start_step=args.start_step,
            expected_completed_step=args.expected_completed_step,
        )
        _print_rows(
            mode="commit" if args.commit else "dry-run",
            checkpoint_root=args.checkpoint_root,
            completed_step=completed_step,
            metric_prefix=args.metric_prefix,
            rows=rows,
        )
        if not args.commit:
            print(
                json.dumps(
                    {
                        "type": "result",
                        "status": "dry-run-only",
                        "message": "no network call or W&B write was attempted; pass --commit explicitly",
                    },
                    sort_keys=True,
                )
            )
            return 0

        appended = commit_rows(
            rows,
            checkpoint_root=args.checkpoint_root,
            completed_step=completed_step,
            entity=args.entity,
            project=args.project,
            run_id=args.run_id,
            metric_prefix=args.metric_prefix,
            wandb_dir=args.wandb_dir,
            readback_timeout_s=args.readback_timeout_s,
            readback_interval_s=args.readback_interval_s,
        )
        print(
            json.dumps(
                {
                    "type": "result",
                    "status": "verified",
                    "appended_rows": appended,
                    "verified_rows": len(rows),
                },
                sort_keys=True,
            )
        )
        return 0
    except BackfillError as exc:
        print(f"backfill refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
