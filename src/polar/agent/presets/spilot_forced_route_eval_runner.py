"""Eval-only forced-route entrypoint for controlled SPilot pool comparisons.

The normal SPilot runner remains the single implementation of candidate slot
assignment, mini-SWE execution, workspace mutation, and result metadata.  This
entrypoint replaces only the Router client with a deterministic one-shot
choice and forces ``max_pool_calls=1`` so the selected candidate is followed by
an automatic submit.

It is uploaded only when the host-side harness validates the explicit
eval-only acknowledgement.  The runtime acknowledgement below is a second
fail-closed boundary; ordinary training never sets it.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

try:  # package import in Polar; sibling import in the portable task runtime
    from . import spilot_router_runner as core
except ImportError:  # pragma: no cover - exercised by the uploaded script
    import spilot_router_runner as core


EVAL_ONLY_ACK = "SPILOT_FORCED_ROUTE_EVAL_ONLY_V1"
_RUNTIME_ACK_ENV = "SPILOT_FORCED_ROUTE_EVAL_RUNTIME_ACK"


class ForcedRouteClient:
    """Return exactly one deterministic ROUTE action and never call a model."""

    def __init__(self) -> None:
        self.slot: str | None = None
        self.call_count = 0

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        timeout_seconds: float,
        model_kwargs: dict[str, object],
    ) -> core.RouterCompletion:
        del model, messages, timeout_seconds, model_kwargs
        if self.slot is None:
            raise core.GatewayInfrastructureError("forced route slot was not initialized")
        if self.call_count:
            raise core.GatewayInfrastructureError(
                "forced-route eval attempted more than one Router decision"
            )
        self.call_count += 1
        return core.RouterCompletion(
            content=json.dumps(
                {"action": "ROUTE", "model_slot": self.slot},
                separators=(",", ":"),
            ),
            request_id="forced-route-eval-no-actor",
            finish_reason="forced_eval_only",
        )


def run_forced_eval(config: dict[str, object], task: str) -> dict[str, object]:
    """Execute one forced candidate and return normal SPilot result metadata."""

    if os.environ.get(_RUNTIME_ACK_ENV) != EVAL_ONLY_ACK:
        raise ValueError("forced-route eval runtime acknowledgement is missing")
    forced = config.get("forced_route_eval")
    if not isinstance(forced, dict) or forced.get("acknowledgement") != EVAL_ONLY_ACK:
        raise ValueError("forced-route eval config acknowledgement is missing")
    if config.get("max_pool_calls") != 1:
        raise ValueError("forced-route eval requires max_pool_calls=1")
    timeout_names = (
        "pool_timeout_seconds",
        "router_timeout_seconds",
        "total_timeout_seconds",
        "reserve_evaluator_seconds",
        "deadline_margin_seconds",
    )
    timeouts: dict[str, float] = {}
    for name in timeout_names:
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"forced-route eval {name} must be numeric")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError(f"forced-route eval {name} must be finite and non-negative")
        timeouts[name] = parsed
    minimum_total = (
        timeouts["pool_timeout_seconds"]
        + timeouts["router_timeout_seconds"]
        + timeouts["reserve_evaluator_seconds"]
        + timeouts["deadline_margin_seconds"]
    )
    if timeouts["total_timeout_seconds"] < minimum_total:
        raise ValueError(
            "forced-route eval total_timeout_seconds must cover pool + Router + "
            f"evaluator reserve + deadline margin ({timeouts['total_timeout_seconds']:g} "
            f"< {minimum_total:g})"
        )
    candidate_model = forced.get("candidate_model")
    if not isinstance(candidate_model, str) or not candidate_model:
        raise ValueError("forced-route eval candidate_model must be non-empty")

    core_config = dict(config)
    core_config.pop("forced_route_eval", None)
    router = ForcedRouteClient()
    pool = core.MiniSwePoolExecutor(core_config)
    admission = (
        core.GatewayEpisodeAdmissionClient()
        if core_config.get("pool_episode_admission_enabled") is True
        else None
    )
    model_pool_capability = (
        None
        if admission is not None
        else core._read_protected_capability(core._MODEL_POOL_CAPABILITY_ENV)
    )
    try:
        orchestrator = core.SpilotOrchestrator(
            config=core_config,
            task=task,
            router=router,
            pool=pool,
            admission=admission,
            model_pool_capability=model_pool_capability,
        )
        matching = [
            candidate
            for candidate in orchestrator.candidates
            if candidate.model == candidate_model
        ]
        if len(matching) != 1:
            raise ValueError(
                f"forced candidate {candidate_model!r} matched {len(matching)} pool entries; expected 1"
            )
        router.slot = matching[0].slot
        result = orchestrator.run()
        if router.call_count != 1:
            raise ValueError(
                "forced-route eval did not execute exactly one deterministic decision"
            )
        if len(result.get("calls", [])) != 1 or not result.get("submitted"):
            raise ValueError(
                "forced-route eval did not execute one pool call followed by submit"
            )
        result.update(
            {
                "eval_only": True,
                "actor_invoked": False,
                "forced_candidate_model": candidate_model,
                "forced_route_acknowledgement": EVAL_ONLY_ACK,
            }
        )
        return result
    finally:
        if admission is not None:
            admission.close()


def main() -> int:
    result_path: Path | None = None
    orchestrator_result: dict[str, object] | None = None
    try:
        config_obj = json.loads(core._decode_env_b64(core._CONFIG_ENV))
        task = core._decode_env_b64(core._TASK_ENV).decode("utf-8")
        if not isinstance(config_obj, dict):
            raise ValueError("router config must be a JSON object")
        result_path = Path(str(config_obj.get("result_path", "/tmp/router_result.json")))
        orchestrator_result = run_forced_eval(config_obj, task)
        core._write_result(result_path, orchestrator_result)
        return 0
    except (ValueError, UnicodeError, json.JSONDecodeError, core.PoolInfrastructureError) as exc:
        if result_path is not None and orchestrator_result is not None:
            try:
                core._write_result(result_path, orchestrator_result)
            except OSError:
                pass
        print(f"SPilot forced-route eval error: {core._bounded_text(str(exc), 500)}", file=sys.stderr)
        return 2
    except Exception as exc:  # fail closed for unexpected eval-control bugs
        print(
            "SPilot forced-route eval unexpected error: "
            + core._bounded_text(f"{type(exc).__name__}: {exc}", 500),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
