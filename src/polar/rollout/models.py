"""Shared data models for rollout orchestration and gateway-node execution."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from polar.agent.models import AgentSpec
from polar.runtime.models import RuntimeSpec
from polar.trajectory.models import EvaluatorSpec, StrategySpec, Trajectory

if TYPE_CHECKING:
    from polar.rollout.timer import StageTimer


class SessionStatus(StrEnum):
    """Canonical session lifecycle statuses.

    StrEnum instances serialize to their string values, so wire compatibility
    with older clients that read plain status strings is preserved.
    """

    REGISTERED = "REGISTERED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    POST_RUN = "POST_RUN"
    BUILDING = "BUILDING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"

    @classmethod
    def terminal(cls) -> frozenset["SessionStatus"]:
        return frozenset({cls.COMPLETED, cls.ERROR, cls.TIMEOUT})

    @classmethod
    def active(cls) -> frozenset["SessionStatus"]:
        return frozenset(set(cls) - cls.terminal())


def _new_stage_timer() -> "StageTimer":
    from polar.rollout.timer import StageTimer

    return StageTimer()


def _default_builder_spec() -> StrategySpec:
    return StrategySpec(strategy="per_request")


def _validate_agent_timeout_metadata(metadata: dict[str, object]) -> None:
    value = metadata.get("agent_timeout")
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("metadata.agent_timeout must be a positive finite number")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("metadata.agent_timeout must be a positive finite number")


class TaskRequest(BaseModel):
    """Task submitted by the trainer."""

    task_id: str
    instruction: str
    num_samples: int = Field(default=1, ge=1)
    # Higher values enter gateway INIT before lower-priority sessions that
    # have not started yet. Running sessions are never preempted.
    dispatch_priority: int = Field(default=0, ge=0, le=100)
    early_stop_min_usable_sessions: int | None = Field(default=None, ge=1)
    timeout_seconds: float = Field(default=600.0, gt=0)
    runtime: RuntimeSpec | None = None
    agent: AgentSpec
    builder: StrategySpec = Field(default_factory=_default_builder_spec)
    evaluator: EvaluatorSpec | None = None
    callback_url: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_early_stop_threshold(self) -> "TaskRequest":
        threshold = self.early_stop_min_usable_sessions
        if threshold is not None and threshold > self.num_samples:
            raise ValueError("early_stop_min_usable_sessions cannot exceed num_samples")
        _validate_agent_timeout_metadata(self.metadata)
        return self


class SessionDispatchRequest(BaseModel):
    """Session lifecycle request sent from the rollout server to a gateway node.

    `remaining_timeout_seconds` is the execution budget the gateway starts
    counting when the session enters INIT. The field name is kept for wire
    compatibility with existing clients. If trusted metadata contains
    `agent_timeout`, that independent budget starts only when RUN begins.
    """

    session_id: str
    task_id: str
    instruction: str
    dispatch_priority: int = Field(default=0, ge=0, le=100)
    remaining_timeout_seconds: float = Field(gt=0)
    runtime: RuntimeSpec | None = None
    agent: AgentSpec
    builder: StrategySpec = Field(default_factory=_default_builder_spec)
    evaluator: EvaluatorSpec | None = None
    callback_url: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_agent_timeout(self) -> "SessionDispatchRequest":
        _validate_agent_timeout_metadata(self.metadata)
        return self


class SessionDispatchResponse(BaseModel):
    """Acknowledgement returned by a gateway node when a session is accepted."""

    session_id: str
    task_id: str
    status: SessionStatus
    node_id: str | None = None


class SessionTiming(BaseModel):
    """Per-session durations in milliseconds.

    The original four aggregate fields remain for wire compatibility.  The
    finer fields make container startup, preparation, agent execution,
    evaluation, and teardown independently observable.  Runtime and mini-SWE
    command summaries use fixed low-cardinality categories; raw commands are
    intentionally never included.
    """

    model_config = ConfigDict(extra="forbid")

    register_to_init_queue_ms: float = 0.0
    rollout_dispatch_ms: float = 0.0
    rollout_result_wait_ms: float = 0.0
    rollout_pipeline_e2e_ms: float = 0.0
    init_ms: float = 0.0
    ready_queue_ms: float = 0.0
    run_ms: float = 0.0
    postrun_queue_ms: float = 0.0
    postrun_ms: float = 0.0
    # Explicit aliases for the legacy ``runtime_validation`` names. In
    # direct-exec Apptainer mode runtime.start() launches the long-lived broker
    # container and waits for its readiness ping, so this is the per-session
    # container startup/entry cost users expect to see in telemetry.
    container_start_ms: float = 0.0
    eval_container_start_ms: float = 0.0
    runtime_validation_ms: float = 0.0
    eval_runtime_validation_ms: float = 0.0
    prepare_ms: float = 0.0
    eval_prepare_ms: float = 0.0
    agent_setup_ms: float = 0.0
    agent_exec_ms: float = 0.0
    agent_postprocess_ms: float = 0.0
    build_ms: float = 0.0
    eval_ms: float = 0.0
    postrun_exec_ms: float = 0.0
    runtime_stop_ms: float = 0.0
    e2e_ms: float = 0.0
    runtime_exec_ms: float = 0.0
    runtime_exec_count: int = 0
    runtime_exec_timeout_count: int = 0
    runtime_exec_failure_count: int = 0
    runtime_exec_exception_count: int = 0
    runtime_exec_cancelled_count: int = 0
    runtime_exec_ms_by_category: dict[str, float] = Field(default_factory=dict)
    runtime_exec_count_by_category: dict[str, int] = Field(default_factory=dict)
    mini_swe_command_ms: float = 0.0
    mini_swe_command_count: int = 0
    mini_swe_command_timeout_count: int = 0
    mini_swe_command_failure_count: int = 0
    mini_swe_command_ms_by_category: dict[str, float] = Field(default_factory=dict)
    mini_swe_command_count_by_category: dict[str, int] = Field(default_factory=dict)


class SessionResult(BaseModel):
    """Terminal node result returned to the rollout server."""

    session_id: str
    task_id: str
    status: SessionStatus
    trajectory: Trajectory
    timing: SessionTiming = Field(default_factory=SessionTiming)
    node_id: str | None = None
    error: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """Blocking response returned once all rollout sessions resolve."""

    task_id: str
    status: str  # Task-level status vocabulary: "running" | "completed" | "failed"
    results: list[SessionResult]
    result_paths: list[str] = Field(default_factory=list)


class TaskStatus(BaseModel):
    """Monitoring view for a task that may still be running."""

    task_id: str
    status: str
    total_sessions: int
    completed_sessions: int
    results: list[SessionResult] = Field(default_factory=list)
    result_paths: list[str] = Field(default_factory=list)


class NodeRegistrationRequest(BaseModel):
    """Payload sent by a gateway node when registering with the rollout server."""

    node_id: str
    gateway_url: str
    max_init_workers: int = Field(ge=1)
    max_run_workers: int = Field(ge=1)
    max_postrun_workers: int = Field(ge=1)
    heartbeat_interval_seconds: int = Field(default=30, ge=1)


class NodeStageMetrics(BaseModel):
    """Per-node stage occupancy and queue depths."""

    init_queue_depth: int = Field(default=0, ge=0)
    init_inflight: int = Field(default=0, ge=0)
    ready_depth: int = Field(default=0, ge=0)
    run_inflight: int = Field(default=0, ge=0)
    postrun_queue_depth: int = Field(default=0, ge=0)
    postrun_inflight: int = Field(default=0, ge=0)

    @property
    def total_sessions(self) -> int:
        return (
            self.init_queue_depth
            + self.init_inflight
            + self.ready_depth
            + self.run_inflight
            + self.postrun_queue_depth
            + self.postrun_inflight
        )


class NodeHeartbeatRequest(BaseModel):
    """Heartbeat payload sent by a gateway node."""

    metrics: NodeStageMetrics = Field(default_factory=NodeStageMetrics)


class GatewayNodeInfo(BaseModel):
    """External view of one schedulable gateway node."""

    node_id: str
    gateway_url: str
    max_init_workers: int
    max_run_workers: int
    max_postrun_workers: int
    metrics: NodeStageMetrics = Field(default_factory=NodeStageMetrics)
    dispatch_reservations: int = Field(default=0, ge=0)
    healthy: bool
    draining: bool = False
    heartbeat_interval_seconds: int
    last_heartbeat: datetime


@dataclass(slots=True)
class SessionContext:
    """Internal state that flows through dispatch and collection."""

    session_id: str
    task_id: str
    request: TaskRequest
    deadline_monotonic: float = field(default_factory=time.monotonic)
    node_id: str | None = None
    gateway_url: str | None = None
    timer: "StageTimer" = field(default_factory=_new_stage_timer)
    rollout_result: SessionResult | None = None
    # Set only by Pipeline.run_batch before it internally cancels a straggler.
    # This keeps early-stop cancellation distinct from caller/task cancellation.
    early_stop_requested: bool = field(default=False, repr=False)
    early_stop_usable_sessions: int = field(default=0, repr=False)
    # Set by the duplicate-dispatch confirmation when the gateway answered an
    # authoritative 404 for this session id: the failed POST never landed, so
    # the dispatch loop may safely retry instead of failing closed.
    dispatch_confirmed_not_landed: bool = field(default=False, repr=False)
    completion_future: asyncio.Future[SessionResult] | None = field(
        default=None,
        repr=False,
    )
