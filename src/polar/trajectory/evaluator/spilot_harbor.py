"""Harbor evaluation with SPilot action validation and optional cost shaping."""

from __future__ import annotations

import json
import math
import os
from typing import Any

from polar.trajectory.evaluator.harbor import HarborEvaluator
from polar.trajectory.models import EvalResult, Trajectory

_COST_PENALTY_MODES = ("multiplicative", "additive")
_DIFFICULTY_CLASSES = ("easy", "hard", "unknown")
_ADDITIVE_REWARD_FLOOR = -1.0
_LEDGER_MAX_BYTES = 256 * 1024 * 1024


class SpilotHarborEvaluator(HarborEvaluator):
    """Grade final runtime state while enforcing the Router action contract.

    The Harbor verifier always runs so invalid Router actions still leave useful
    diagnostics.  When ``require_valid_action`` is enabled, however, only an
    explicit harness-validated ``agent_result.metadata.spilot_router.action_valid=true``
    and ``submitted=true`` can retain the verifier reward.

    Cost and latency shaping are disabled by default.  With positive lambdas
    and ``cost_penalty_mode="multiplicative"`` (the historical default) the
    reward is success gated with a bounded penalty::

        cost_frac    = min(1, lambda_c * total_cost / cost_normalizer)
        latency_frac = min(1, lambda_l * total_latency_s / latency_normalizer)
        shaped = harbor_reward * (1 - min(1, cost_frac + latency_frac))

    ``cost_penalty_mode="additive"`` subtracts the same bounded fraction from
    the outcome instead, floored at -1.0::

        shaped = max(-1.0, harbor_reward - min(1, cost_frac + latency_frac))

    so an expensive failure ranks strictly below a cheap one.  In both modes
    an invalid/unsubmitted Router episode stays at exactly 0.0 (the action
    contract, not spend, is its training signal), and missing cost/latency
    metadata fails closed to reward zero when the respective term is enabled.

    Optional difficulty conditioning multiplies the cost lambda per task via
    a ledger JSON (``{"tasks": {"<row-key>": {"class": "easy|hard|unknown"}}}``)
    built offline from within-group counterfactuals: routing spend on a task
    siblings solved all-qwen ("easy") can be penalized harder than spend on a
    task only GPT ever solved ("hard").  The ledger is reloaded on mtime
    change and any read/parse problem falls back to multiplier 1.0 — shaping
    must never turn an evaluator outage into a reward outage.
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
        cost_penalty_mode: str | None = None,
        difficulty_ledger_path: str | None = None,
        difficulty_easy_multiplier: float | None = None,
        difficulty_hard_multiplier: float | None = None,
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
        mode = "multiplicative" if cost_penalty_mode is None else str(cost_penalty_mode)
        if mode not in _COST_PENALTY_MODES:
            raise ValueError(
                f"cost_penalty_mode must be one of {_COST_PENALTY_MODES}; got {mode!r}"
            )
        self.cost_penalty_mode = mode
        ledger_path = (
            str(difficulty_ledger_path).strip()
            if difficulty_ledger_path is not None
            else ""
        )
        if ledger_path and not ledger_path.startswith("/"):
            raise ValueError("difficulty_ledger_path must be an absolute path")
        self.difficulty_ledger_path = ledger_path or None
        self.difficulty_easy_multiplier = (
            1.0
            if difficulty_easy_multiplier is None
            else float(difficulty_easy_multiplier)
        )
        self.difficulty_hard_multiplier = (
            1.0
            if difficulty_hard_multiplier is None
            else float(difficulty_hard_multiplier)
        )
        for name, value in (
            ("difficulty_easy_multiplier", self.difficulty_easy_multiplier),
            ("difficulty_hard_multiplier", self.difficulty_hard_multiplier),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        self._ledger_cache: dict[str, str] = {}
        self._ledger_mtime_ns: int | None = None
        super().__init__(**harbor_config)

    # ------------------------------------------------------------------
    # Difficulty ledger

    @staticmethod
    def _ledger_key(task_id: Any) -> str | None:
        """Stable per-dataset-row key from a harness task id.

        Harness task ids look like ``polar-spilot-router-<group>-<row>``; the
        trailing integer is the dataset row and is stable across rollouts,
        lanes, and restarts, while the group index is not.
        """

        if not isinstance(task_id, str) or not task_id:
            return None
        tail = task_id.rsplit("-", 1)[-1]
        return tail if tail.isdigit() else task_id

    def _difficulty_class(self, task_id: Any) -> str:
        if self.difficulty_ledger_path is None:
            return "unknown"
        key = self._ledger_key(task_id)
        if key is None:
            return "unknown"
        try:
            stat = os.stat(self.difficulty_ledger_path)
        except OSError:
            # File absent/unreadable right now: fail open without caching so
            # a ledger that appears later is picked up immediately.
            return "unknown"
        if stat.st_mtime_ns != self._ledger_mtime_ns:
            cache: dict[str, str] = {}
            try:
                if stat.st_size > _LEDGER_MAX_BYTES:
                    raise ValueError("ledger file implausibly large")
                with open(self.difficulty_ledger_path, encoding="utf-8") as fh:
                    payload = json.load(fh)
                if isinstance(payload, dict):
                    tasks = payload.get("tasks")
                    if isinstance(tasks, dict):
                        for raw_key, entry in tasks.items():
                            cls = (
                                entry.get("class")
                                if isinstance(entry, dict)
                                else None
                            )
                            if cls in _DIFFICULTY_CLASSES:
                                cache[str(raw_key)] = str(cls)
            except Exception:  # noqa: BLE001
                # ANY malformed content (non-dict JSON, wrong nesting, binary
                # garbage, decode/recursion errors, ...) must fail open to
                # "unknown" — the ledger conditions shaping, it must never
                # break evaluation itself.
                cache = {}
            # Cache the (possibly empty) result AT this mtime so a broken
            # file is not re-parsed for every session; the next rebuild
            # changes the mtime and is re-read.
            self._ledger_cache = cache
            self._ledger_mtime_ns = stat.st_mtime_ns
        return self._ledger_cache.get(key, "unknown")

    def _difficulty_multiplier(self, difficulty: str) -> float:
        if difficulty == "easy":
            return self.difficulty_easy_multiplier
        if difficulty == "hard":
            return self.difficulty_hard_multiplier
        return 1.0

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
        # Penalties are computed for every terminally-valid episode in
        # additive mode (failures included), but only for successes in the
        # historical multiplicative mode.
        additive = self.cost_penalty_mode == "additive"
        contract_ok = terminal_action_valid or not self.require_valid_action
        cost_shaping_active = self.cost_penalty_lambda > 0.0 and (
            shaped_reward > 0.0 or (additive and contract_ok)
        )
        latency_shaping_active = self.latency_penalty_lambda > 0.0 and (
            shaped_reward > 0.0 or (additive and contract_ok)
        )
        difficulty = self._difficulty_class(
            (trajectory.metadata or {}).get("task_id")
            if isinstance(getattr(trajectory, "metadata", None), dict)
            else None
        )
        effective_cost_lambda = self.cost_penalty_lambda * self._difficulty_multiplier(
            difficulty
        )
        applied_cost_penalty = 0.0
        applied_latency_penalty = 0.0
        if cost_shaping_active:
            if not cost_valid:
                shaped_reward = 0.0
                invalid_reason = "missing_or_invalid_total_cost"
                additive = False
            else:
                applied_cost_penalty = min(
                    1.0,
                    effective_cost_lambda * total_cost / self.cost_normalizer,
                )
        if latency_shaping_active and invalid_reason != "missing_or_invalid_total_cost":
            if not latency_valid:
                shaped_reward = 0.0
                invalid_reason = "missing_or_invalid_total_latency"
                additive = False
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
        if additive and contract_ok and applied_total_penalty > 0.0:
            shaped_reward = max(
                _ADDITIVE_REWARD_FLOOR, shaped_reward - applied_total_penalty
            )
        elif shaped_reward > 0.0 and applied_total_penalty > 0.0:
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
                "cost_penalty_mode": self.cost_penalty_mode,
                "difficulty_class": difficulty,
                "effective_cost_lambda": effective_cost_lambda,
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
