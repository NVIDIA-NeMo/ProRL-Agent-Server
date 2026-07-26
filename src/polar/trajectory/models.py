"""Shared completion-session and trajectory schemas used by rollout nodes."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Strategy / evaluator specs
# ---------------------------------------------------------------------------


class StrategySpec(BaseModel):
    """Identifies a builder or evaluator strategy with optional per-request config."""

    strategy: str
    config: dict[str, Any] = Field(default_factory=dict)


class EvaluatorSpec(BaseModel):
    """Evaluator configuration for a rollout session."""

    strategy: str
    config: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    refresh_runtime: bool = False


# ---------------------------------------------------------------------------
# Evaluator result
# ---------------------------------------------------------------------------


class EvalResult(BaseModel):
    """Evaluator output — the gateway merges rewards into the trajectory."""

    outcome_reward: float | None = None
    trace_rewards: list[float | None] | None = None
    outcome_reward_components: dict[str, float] | None = None
    trace_reward_components: list[dict[str, float] | None] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("outcome_reward", mode="before")
    @classmethod
    def _validate_outcome_reward(cls, value: Any) -> float | None:
        return _finite_reward_or_none(value)

    @field_validator("trace_rewards", mode="before")
    @classmethod
    def _validate_trace_rewards(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            return value
        return [_finite_reward_or_none(item) for item in value]

    @field_validator("outcome_reward_components", mode="before")
    @classmethod
    def _validate_outcome_reward_components(
        cls, value: Any
    ) -> dict[str, float] | None:
        return _finite_reward_components_or_none(value)

    @field_validator("trace_reward_components", mode="before")
    @classmethod
    def _validate_trace_reward_components(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            return value
        return [
            _finite_reward_components_or_none(components)
            for components in value
        ]


# ---------------------------------------------------------------------------
# Completion and trajectory models
# ---------------------------------------------------------------------------


class CompletionRecord(BaseModel):
    """One normalized upstream completion payload."""

    model_config = ConfigDict(extra="allow")

    completion_id: str
    timestamp: str | None = None
    request: dict[str, Any] = Field(default_factory=dict)
    original_request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompletionSession(BaseModel):
    """Builder-facing session payload with metadata and ordered completions."""

    model_config = ConfigDict(extra="allow")

    session_id: str
    created_at: str | None = None
    completion_count: int = 0
    task_id: str | None = None
    model_requested: str | None = None
    model_used: str | None = None
    api_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    completions: list[CompletionRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sort_completions(self) -> "CompletionSession":
        self.completions = sorted(
            self.completions,
            key=lambda completion: (
                str(completion.timestamp or ""),
                completion.completion_id,
            ),
        )
        return self


class Trace(BaseModel):
    """One reconstructed completion interaction."""

    prompt_ids: list[int] = Field(default_factory=list)
    response_ids: list[int] = Field(default_factory=list)
    loss_mask: list[int] = Field(default_factory=list)
    prompt_messages: list[dict[str, Any]] = Field(default_factory=list)
    response_messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    response_logprobs: list[float] | None = None
    reward: float | None = None
    reward_components: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reward", mode="before")
    @classmethod
    def _validate_reward(cls, value: Any) -> float | None:
        return _finite_reward_or_none(value)

    @field_validator("reward_components", mode="before")
    @classmethod
    def _validate_reward_components(cls, value: Any) -> dict[str, float]:
        return _finite_reward_components_or_none(value) or {}

    @field_validator("loss_mask")
    @classmethod
    def _validate_loss_mask_values(cls, value: list[int]) -> list[int]:
        normalized: list[int] = []
        for item in value:
            mask_value = int(item)
            if mask_value not in (0, 1):
                raise ValueError("loss_mask values must be 0 or 1")
            normalized.append(mask_value)
        return normalized

    @model_validator(mode="after")
    def _validate_response_lengths(self) -> "Trace":
        if self.loss_mask and len(self.loss_mask) != len(self.response_ids):
            raise ValueError("loss_mask length must match response_ids length")
        if self.response_logprobs is not None and len(self.response_logprobs) != len(
            self.response_ids
        ):
            raise ValueError("response_logprobs length must match response_ids length")
        return self


def _finite_reward_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("reward must be numeric, not boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reward must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError("reward must be finite")
    return parsed


def _finite_reward_components_or_none(
    value: Any,
) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("reward components must be a mapping")

    normalized: dict[str, float] = {}
    for key, component in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("reward component names must be non-empty strings")
        parsed = _finite_reward_or_none(component)
        if parsed is None:
            raise ValueError(f"reward component {key!r} must not be null")
        normalized[key] = parsed
    return normalized


class Trajectory(BaseModel):
    """Structured trajectory reconstructed from session completion records."""

    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    traces: list[Trace] = Field(default_factory=list)
    error: str | None = None

    @field_validator("status")
    @classmethod
    def _validate_status(cls, value: str) -> str:
        allowed = {"COMPLETED", "TIMEOUT", "ERROR"}
        if value not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return value
