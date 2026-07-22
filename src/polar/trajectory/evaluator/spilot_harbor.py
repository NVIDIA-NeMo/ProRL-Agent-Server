"""Harbor evaluation with SPilot action validation and optional cost shaping."""

from __future__ import annotations

import math
from typing import Any

from polar.trajectory.evaluator.harbor import HarborEvaluator
from polar.trajectory.models import EvalResult, Trajectory


class SpilotHarborEvaluator(HarborEvaluator):
    """Grade final runtime state while enforcing the Router action contract.

    The Harbor verifier always runs so invalid Router actions still leave useful
    diagnostics.  When ``require_valid_action`` is enabled, however, only an
    explicit harness-validated ``agent_result.metadata.spilot_router.action_valid=true``
    and ``submitted=true`` can retain the verifier reward.

    Cost and latency shaping are disabled by default.  With positive lambdas
    the reward is success gated with a bounded additive penalty::

        cost_frac    = min(1, lambda_c * total_cost / cost_normalizer)
        latency_frac = min(1, lambda_l * total_latency_s / latency_normalizer)
        shaped = harbor_reward * (1 - min(1, cost_frac + latency_frac))

    Thus a cheaper/faster failed rollout never outranks a more expensive or
    slower failed one.  Total latency is the sum of ``calls[].duration_ms``
    from the router metadata, in seconds.  Missing or malformed cost/latency
    metadata fails closed to reward zero only when the respective shaping
    term is enabled.
    """

    MODE = "spilot_harbor"

    def __init__(
        self,
        *,
        require_valid_action: bool = True,
        cost_penalty_lambda: float = 0.0,
        cost_normalizer: float = 1.0,
        latency_penalty_lambda: float = 0.0,
        latency_normalizer: float = 1.0,
        **harbor_config: Any,
    ) -> None:
        if not isinstance(require_valid_action, bool):
            raise ValueError("require_valid_action must be a boolean")
        self.require_valid_action = require_valid_action
        self.cost_penalty_lambda = float(cost_penalty_lambda)
        self.cost_normalizer = float(cost_normalizer)
        self.latency_penalty_lambda = float(latency_penalty_lambda)
        self.latency_normalizer = float(latency_normalizer)
        if not math.isfinite(self.cost_penalty_lambda) or self.cost_penalty_lambda < 0:
            raise ValueError("cost_penalty_lambda must be finite and non-negative")
        if not math.isfinite(self.cost_normalizer) or self.cost_normalizer <= 0:
            raise ValueError("cost_normalizer must be finite and greater than zero")
        if (
            not math.isfinite(self.latency_penalty_lambda)
            or self.latency_penalty_lambda < 0
        ):
            raise ValueError("latency_penalty_lambda must be finite and non-negative")
        if not math.isfinite(self.latency_normalizer) or self.latency_normalizer <= 0:
            raise ValueError("latency_normalizer must be finite and greater than zero")
        super().__init__(**harbor_config)

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        harbor_result = await super().evaluate(trajectory, **runtime)
        router_metadata = _router_metadata(runtime.get("agent_result"))
        action_valid_value = router_metadata.get("action_valid")
        action_valid = action_valid_value is True
        submitted_value = router_metadata.get("submitted")
        submitted = submitted_value is True
        terminal_action_valid = action_valid and submitted

        raw_reward = float(harbor_result.outcome_reward or 0.0)
        shaped_reward = raw_reward
        invalid_reason: str | None = None
        if self.require_valid_action and not terminal_action_valid:
            shaped_reward = 0.0
            if action_valid_value is None:
                invalid_reason = "missing_action_valid"
            elif not action_valid:
                invalid_reason = "invalid_router_action"
            elif submitted_value is None:
                invalid_reason = "missing_submit"
            else:
                invalid_reason = "router_did_not_submit"

        total_cost, cost_valid = _nonnegative_finite_float(
            router_metadata.get("total_cost")
        )
        total_latency_seconds, latency_valid = _total_call_latency_seconds(
            router_metadata.get("calls")
        )
        applied_cost_penalty = 0.0
        applied_latency_penalty = 0.0
        if shaped_reward > 0.0 and self.cost_penalty_lambda > 0.0:
            if not cost_valid:
                shaped_reward = 0.0
                invalid_reason = "missing_or_invalid_total_cost"
            else:
                applied_cost_penalty = min(
                    1.0,
                    self.cost_penalty_lambda * total_cost / self.cost_normalizer,
                )
        if shaped_reward > 0.0 and self.latency_penalty_lambda > 0.0:
            if not latency_valid:
                shaped_reward = 0.0
                invalid_reason = "missing_or_invalid_total_latency"
            else:
                applied_latency_penalty = min(
                    1.0,
                    self.latency_penalty_lambda
                    * total_latency_seconds
                    / self.latency_normalizer,
                )
        applied_total_penalty = min(
            1.0, applied_cost_penalty + applied_latency_penalty
        )
        if shaped_reward > 0.0 and applied_total_penalty > 0.0:
            shaped_reward *= 1.0 - applied_total_penalty

        metadata = dict(harbor_result.metadata)
        metadata.update(
            {
                "mode": self.MODE,
                # The harness has schema-validated and capped this
                # artifact at 128 KiB. Persist it under evaluation metadata so
                # post-training analysis can recover route distributions,
                # pool failures, slot randomization, and SUBMIT/VERIFY rates.
                "spilot_router": dict(router_metadata),
                "harbor_outcome_reward": raw_reward,
                "reward": shaped_reward,
                "router_action_valid": action_valid,
                "router_action_valid_value_present": action_valid_value is not None,
                "router_submitted": submitted,
                "router_submitted_value_present": submitted_value is not None,
                "router_terminal_action_valid": terminal_action_valid,
                "require_valid_action": self.require_valid_action,
                "cost_penalty_lambda": self.cost_penalty_lambda,
                "cost_normalizer": self.cost_normalizer,
                "total_cost": total_cost if cost_valid else None,
                "total_cost_valid": cost_valid,
                "applied_cost_penalty": applied_cost_penalty,
                "latency_penalty_lambda": self.latency_penalty_lambda,
                "latency_normalizer": self.latency_normalizer,
                "total_latency_seconds": (
                    total_latency_seconds if latency_valid else None
                ),
                "total_latency_valid": latency_valid,
                "applied_latency_penalty": applied_latency_penalty,
                "applied_total_penalty": applied_total_penalty,
            }
        )
        if invalid_reason is not None:
            metadata["reward_override_reason"] = invalid_reason
        return EvalResult(
            outcome_reward=shaped_reward,
            # Gateway merge gives trace_rewards precedence over outcome_reward.
            # Clear any wrapped per-trace values so the shaped terminal reward
            # is broadcast consistently to every Router trace.
            trace_rewards=None,
            metadata=metadata,
        )


def _router_metadata(agent_result: Any) -> dict[str, Any]:
    agent_metadata = getattr(agent_result, "metadata", None)
    if not isinstance(agent_metadata, dict) and isinstance(agent_result, dict):
        agent_metadata = agent_result.get("metadata")
    if not isinstance(agent_metadata, dict):
        return {}
    router_metadata = agent_metadata.get("spilot_router")
    return router_metadata if isinstance(router_metadata, dict) else {}


def _total_call_latency_seconds(calls: Any) -> tuple[float, bool]:
    """Sum ``duration_ms`` across router calls, in seconds.

    Fails closed (valid=False) when the calls list is missing/empty or any
    entry lacks a finite non-negative ``duration_ms`` — mirroring the strict
    total_cost semantics so a latency-shaped run never silently under-counts.
    """
    if not isinstance(calls, list) or not calls:
        return 0.0, False
    total_ms = 0.0
    for call in calls:
        if not isinstance(call, dict):
            return 0.0, False
        duration_ms, valid = _nonnegative_finite_float(call.get("duration_ms"))
        if not valid:
            return 0.0, False
        total_ms += duration_ms
    return total_ms / 1000.0, True


def _nonnegative_finite_float(value: Any) -> tuple[float, bool]:
    if isinstance(value, bool) or value is None:
        return 0.0, False
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    if not math.isfinite(converted) or converted < 0.0:
        return 0.0, False
    return converted, True
