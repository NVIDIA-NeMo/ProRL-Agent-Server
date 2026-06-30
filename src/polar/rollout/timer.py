"""Per-session stage timing utilities."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from polar.rollout.models import SessionTiming
from polar.runtime.command_timing import RUNTIME_EXEC_CATEGORIES


# Finer marks (``build``, ``eval``, ``teardown``) still write into the mark
# dictionary for debug logs but are rolled into ``postrun_ms`` in the public
# schema.
_POSTRUN_MARKS: tuple[str, ...] = ("postrun", "build", "eval", "teardown")


@dataclass(slots=True)
class StageTimer:
    """Record monotonic timestamps for session stages."""

    _marks: dict[str, float] = field(default_factory=dict)
    _runtime_exec_ms: dict[str, float] = field(default_factory=dict)
    _runtime_exec_count: dict[str, int] = field(default_factory=dict)
    _runtime_exec_timeout_count: int = 0
    _runtime_exec_failure_count: int = 0
    _runtime_exec_exception_count: int = 0
    _runtime_exec_cancelled_count: int = 0
    _mini_swe_command_ms: dict[str, float] = field(default_factory=dict)
    _mini_swe_command_count: dict[str, int] = field(default_factory=dict)
    _mini_swe_command_timeout_count: int = 0
    _mini_swe_command_failure_count: int = 0

    def mark(self, stage: str, event: str) -> None:
        """Mark a stage start or finish."""
        self._marks[f"{stage}_{event}"] = time.monotonic()

    def add_runtime_exec_summary(self, summary: dict[str, object]) -> None:
        """Merge one runtime's fixed-schema exec aggregate."""

        self._merge_command_summary(
            summary,
            ms_target=self._runtime_exec_ms,
            count_target=self._runtime_exec_count,
            counter_prefix="runtime",
        )

    def add_mini_swe_command_summary(self, summary: object) -> None:
        """Merge sanitized mini-SWE action timings collected by its environment."""

        if not isinstance(summary, dict):
            return
        self._merge_command_summary(
            summary,
            ms_target=self._mini_swe_command_ms,
            count_target=self._mini_swe_command_count,
            counter_prefix="mini_swe",
        )

    def to_session_timing(self) -> SessionTiming:
        """Return durations for the init/run/post-run lifecycle."""
        container_start_ms = self._duration_ms("runtime_validation")
        eval_container_start_ms = self._duration_ms("eval_runtime_validation")
        return SessionTiming(
            register_to_init_queue_ms=self._span_ms("dispatch_started", "init_started"),
            rollout_dispatch_ms=self._duration_ms("dispatch"),
            rollout_result_wait_ms=self._span_ms(
                "dispatch_finished", "return_finished"
            ),
            rollout_pipeline_e2e_ms=self._span_ms(
                "dispatch_started", "return_finished"
            ),
            init_ms=self._duration_ms("init"),
            ready_queue_ms=self._span_ms("init_finished", "run_started"),
            run_ms=self._duration_ms("run"),
            postrun_queue_ms=self._span_ms("run_finished", "postrun_started"),
            postrun_ms=self._postrun_span_ms(),
            container_start_ms=container_start_ms,
            eval_container_start_ms=eval_container_start_ms,
            runtime_validation_ms=container_start_ms,
            eval_runtime_validation_ms=eval_container_start_ms,
            prepare_ms=self._duration_ms("prepare"),
            eval_prepare_ms=self._duration_ms("eval_prepare"),
            agent_setup_ms=self._duration_ms("agent_setup"),
            agent_exec_ms=self._duration_ms("agent_exec"),
            agent_postprocess_ms=self._duration_ms("agent_postprocess"),
            build_ms=self._duration_ms("build"),
            eval_ms=self._duration_ms("eval"),
            postrun_exec_ms=self._duration_ms("postrun_exec"),
            runtime_stop_ms=self._duration_ms("runtime_stop"),
            e2e_ms=self._span_ms("dispatch_started", "return_finished"),
            runtime_exec_ms=sum(self._runtime_exec_ms.values()),
            runtime_exec_count=sum(self._runtime_exec_count.values()),
            runtime_exec_timeout_count=self._runtime_exec_timeout_count,
            runtime_exec_failure_count=self._runtime_exec_failure_count,
            runtime_exec_exception_count=self._runtime_exec_exception_count,
            runtime_exec_cancelled_count=self._runtime_exec_cancelled_count,
            runtime_exec_ms_by_category=dict(self._runtime_exec_ms),
            runtime_exec_count_by_category=dict(self._runtime_exec_count),
            mini_swe_command_ms=sum(self._mini_swe_command_ms.values()),
            mini_swe_command_count=sum(self._mini_swe_command_count.values()),
            mini_swe_command_timeout_count=self._mini_swe_command_timeout_count,
            mini_swe_command_failure_count=self._mini_swe_command_failure_count,
            mini_swe_command_ms_by_category=dict(self._mini_swe_command_ms),
            mini_swe_command_count_by_category=dict(self._mini_swe_command_count),
        )

    def _merge_command_summary(
        self,
        summary: dict[str, object],
        *,
        ms_target: dict[str, float],
        count_target: dict[str, int],
        counter_prefix: str,
    ) -> None:
        ms_by_category = summary.get("ms_by_category")
        count_by_category = summary.get("count_by_category")
        if isinstance(ms_by_category, dict):
            for category in RUNTIME_EXEC_CATEGORIES:
                value = _nonnegative_float(ms_by_category.get(category))
                if value is not None:
                    ms_target[category] = ms_target.get(category, 0.0) + value
        if isinstance(count_by_category, dict):
            for category in RUNTIME_EXEC_CATEGORIES:
                value = _nonnegative_int(count_by_category.get(category))
                if value is not None:
                    count_target[category] = count_target.get(category, 0) + value

        timeout_count = _nonnegative_int(summary.get("timeout_count")) or 0
        failure_count = _nonnegative_int(summary.get("failure_count")) or 0
        if counter_prefix == "runtime":
            self._runtime_exec_timeout_count += timeout_count
            self._runtime_exec_failure_count += failure_count
            self._runtime_exec_exception_count += (
                _nonnegative_int(summary.get("exception_count")) or 0
            )
            self._runtime_exec_cancelled_count += (
                _nonnegative_int(summary.get("cancelled_count")) or 0
            )
        else:
            self._mini_swe_command_timeout_count += timeout_count
            self._mini_swe_command_failure_count += failure_count

    def _duration_ms(self, stage: str) -> float:
        started = self._marks.get(f"{stage}_started")
        finished = self._marks.get(f"{stage}_finished") or started
        if started is None or finished is None:
            return 0.0
        return max(0.0, (finished - started) * 1000.0)

    def _span_ms(self, start_mark: str, end_mark: str) -> float:
        started = self._marks.get(start_mark)
        finished = self._marks.get(end_mark)
        if started is None or finished is None:
            return 0.0
        return max(0.0, (finished - started) * 1000.0)

    def _postrun_span_ms(self) -> float:
        """Earliest postrun-family started to latest postrun-family finished."""
        starts = [
            self._marks[f"{stage}_started"]
            for stage in _POSTRUN_MARKS
            if f"{stage}_started" in self._marks
        ]
        finishes = [
            self._marks[f"{stage}_finished"]
            for stage in _POSTRUN_MARKS
            if f"{stage}_finished" in self._marks
        ]
        if not starts or not finishes:
            return 0.0
        return max(0.0, (max(finishes) - min(starts)) * 1000.0)


def _nonnegative_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0.0 or not math.isfinite(parsed):
        return None
    return parsed


def _nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed
