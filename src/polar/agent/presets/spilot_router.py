"""SPilot router harness.

The harness uploads a small, self-contained orchestrator into the rollout
runtime.  That orchestrator asks the trainable policy for a strict routing
action, runs the selected frozen mini-SWE agent on the live workspace, and
optionally asks the policy whether to submit or spend one more model call on
verification/repair.

Only the router requests are intended to be persisted by the gateway.  Pool
model aliases are explicit in ``settings.model_pool`` so the gateway can route
those calls to frozen OpenAI-compatible backends without exposing credentials
to the task container.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from polar.agent.base import BaseHarness
from polar.agent.models import AgentRunResult, AgentSpec
from polar.runtime.base import (
    BaseRuntime,
    RUNTIME_AGENT_LOG_DIR,
    RUNTIME_ARTIFACTS_DIR,
    RUNTIME_SESSION_DIR,
)
from polar.runtime.models import ExecInput


SPILOT_RUNNER_PATH = f"{RUNTIME_SESSION_DIR}/spilot_router_runner.py"
SPILOT_FORCED_EVAL_RUNNER_PATH = (
    f"{RUNTIME_SESSION_DIR}/spilot_forced_route_eval_runner.py"
)
SPILOT_RESULT_PATH = f"{RUNTIME_ARTIFACTS_DIR}/router_result.json"
SPILOT_LOG_PATH = f"{RUNTIME_AGENT_LOG_DIR}/spilot-router.txt"
_DEFAULT_PORTABLE_PYTHON = "/opt/polar-mini-swe-agent/venv/bin/python"
_DEFAULT_MINI_SWE_BIN = "/opt/polar-mini-swe-agent/bin/mini-swe-agent"
_DEFAULT_ROUTER_REQUEST_MODEL = "router/policy"
_MAX_EPISODE_ADMISSION_WAIT_SECONDS = 86_400.0
_FORCED_EVAL_ACK = "SPILOT_FORCED_ROUTE_EVAL_ONLY_V1"
_FORCED_EVAL_AGENT_ACK_ENV = "SPILOT_FORCED_ROUTE_EVAL_ACK"
_FORCED_EVAL_RUNTIME_ACK_ENV = "SPILOT_FORCED_ROUTE_EVAL_RUNTIME_ACK"


class SpilotRouterHarness(BaseHarness):
    """Launch the bounded SPilot router state machine in one mutable runtime."""

    def __init__(self, agent_spec: AgentSpec) -> None:
        forced_eval, normalized_spec = _extract_forced_eval_config(agent_spec)
        super().__init__(normalized_spec)
        if not self.model_name:
            raise ValueError("spilot_router requires agent.model_name for the router")
        self._runner_config = _build_runner_config(normalized_spec)
        self._forced_eval = forced_eval
        if forced_eval is not None:
            if self._runner_config["max_pool_calls"] != 1:
                raise ValueError("forced-route eval requires max_pool_calls=1")
            self._runner_config["forced_route_eval"] = forced_eval

    async def setup(self, runtime: BaseRuntime) -> None:
        """Upload the portable orchestrator without modifying the task image."""

        source = Path(__file__).with_name("spilot_router_runner.py")
        if not source.is_file():
            raise RuntimeError(f"SPilot router runner is missing: {source}")
        await runtime.upload_file(str(source), SPILOT_RUNNER_PATH)
        if self._forced_eval is not None:
            forced_source = Path(__file__).with_name("spilot_forced_route_eval_runner.py")
            if not forced_source.is_file():
                raise RuntimeError(
                    f"SPilot forced-route eval runner is missing: {forced_source}"
                )
            await runtime.upload_file(
                str(forced_source),
                SPILOT_FORCED_EVAL_RUNNER_PATH,
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        config_b64 = _encode_json_b64(self._runner_config)
        task_b64 = base64.b64encode(instruction.encode("utf-8")).decode("ascii")
        python = str(self._runner_config["runner_python"])
        runner_path = (
            SPILOT_FORCED_EVAL_RUNNER_PATH
            if self._forced_eval is not None
            else SPILOT_RUNNER_PATH
        )
        protected_env_keys: list[str] = []
        if self._forced_eval is None:
            protected_env_keys.append("POLAR_ROUTER_CAPABILITY")
        if self._runner_config["pool_episode_admission_enabled"]:
            protected_env_keys.append("POLAR_MODEL_POOL_ADMISSION_CAPABILITY")
        else:
            protected_env_keys.append("POLAR_MODEL_POOL_CAPABILITY")
        core_source = Path(__file__).with_name("spilot_router_runner.py")
        protected_file_digests = {
            SPILOT_RUNNER_PATH: hashlib.sha256(core_source.read_bytes()).hexdigest()
        }
        if self._forced_eval is not None:
            forced_source = Path(__file__).with_name("spilot_forced_route_eval_runner.py")
            protected_file_digests[SPILOT_FORCED_EVAL_RUNNER_PATH] = hashlib.sha256(
                forced_source.read_bytes()
            ).hexdigest()
        return [
            ExecInput(
                protected_argv=[python, runner_path],
                protected_env_keys=protected_env_keys,
                protected_file_digests=protected_file_digests,
                env={
                    **self.env,
                    "SPILOT_ROUTER_CONFIG_B64": config_b64,
                    "SPILOT_TASK_B64": task_b64,
                    "MSWEA_CONFIGURED": "true",
                    "MSWEA_COST_TRACKING": "ignore_errors",
                    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                    **(
                        {_FORCED_EVAL_RUNTIME_ACK_ENV: _FORCED_EVAL_ACK}
                        if self._forced_eval is not None
                        else {}
                    ),
                },
            )
        ]

    async def postprocess(self, runtime: BaseRuntime, result: AgentRunResult) -> None:
        """Attach bounded, secret-free router metadata for reward/evaluation."""

        path = runtime.resolve_host_path(SPILOT_RESULT_PATH)
        if path is None or not path.is_file():
            result.metadata["spilot_router"] = {
                "schema_version": 1,
                "action_valid": False,
                "actions": [],
                "calls": [],
                "submitted": False,
                "total_cost": 0.0,
                "termination_reason": "result_artifact_missing",
            }
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result.metadata["spilot_router"] = _validate_result_metadata(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            result.metadata["spilot_router"] = {
                "schema_version": 1,
                "action_valid": False,
                "actions": [],
                "calls": [],
                "submitted": False,
                "total_cost": 0.0,
                "termination_reason": "result_artifact_invalid",
                "error": _bounded_text(str(exc), 240),
            }


def _build_runner_config(agent_spec: AgentSpec) -> dict[str, Any]:
    settings = dict(agent_spec.settings)
    model_pool = settings.pop("model_pool", None)
    if not isinstance(model_pool, (list, dict)) or not model_pool:
        raise ValueError("spilot_router settings.model_pool must be a non-empty list or mapping")

    # Slime's fixed-eval path applies generic harness settings. Normalize the
    # subset that has a clear SPilot meaning, while deliberately ignoring its
    # large generic max_tokens value so Router actions retain their own bound.
    eval_model_kwargs = _request_mapping(
        settings.pop("model_kwargs", {}), "model_kwargs"
    )
    raw_sampling_seed = settings.pop("sampling_seed", None)
    sampling_seed = (
        None
        if raw_sampling_seed is None
        else _bounded_int(
            raw_sampling_seed,
            name="sampling_seed",
            minimum=0,
            maximum=2**63 - 1,
        )
    )
    eval_step_limit = settings.pop("step_limit", None)
    configured_pool_step_limit = settings.pop("pool_step_limit", 64)

    router_model_kwargs = _request_mapping(
        settings.pop("router_model_kwargs", {}), "router_model_kwargs"
    )
    for key in ("temperature", "top_p"):
        if key in eval_model_kwargs:
            router_model_kwargs[key] = eval_model_kwargs[key]
    if sampling_seed is not None:
        router_model_kwargs["seed"] = sampling_seed

    routing_mode_setting = _routing_mode(settings.pop("routing_mode", "task_level"))
    max_pool_calls = _bounded_int(
        settings.pop("max_pool_calls", 2),
        name="max_pool_calls",
        minimum=1,
        # Task-level routing is ROUTE + optional VERIFY.  Turn-level routes
        # every agent STEP and budgets by pool_step_limit; max_pool_calls is
        # accepted there only so shared lane configs keep rendering, and is
        # otherwise unused.
        maximum=2 if routing_mode_setting == "task_level" else 1000,
    )
    config: dict[str, Any] = {
        "schema_version": 1,
        # The gateway rewrites this reserved request alias to agent.model_name.
        # Keeping the action-policy namespace distinct prevents accidental
        # local-policy calls made by a pool subprocess from entering the Router
        # trajectory builder.
        "router_model": str(
            settings.pop("router_request_model", _DEFAULT_ROUTER_REQUEST_MODEL)
        ),
        "model_pool": model_pool,
        "max_pool_calls": max_pool_calls,
        "shuffle_slots": _strict_bool(settings.pop("shuffle_slots", True), "shuffle_slots"),
        # "real_names" (default since 2026-07-21) labels candidates with their
        # actual pool model names; "anonymous" is the historical M0/M1 protocol
        # (TB2.1 decision probes proved an M0 token anchor under it) and must be
        # requested explicitly when resuming pre-switch checkpoints.
        "slot_label_mode": _slot_label_mode(settings.pop("slot_label_mode", "real_names")),
        # "task_level" = one ROUTE assigns the whole attempt (+ optional final
        # VERIFY); "turn_level" = per-STEP routing: one turn is one step() —
        # a single pool-model completion plus the execution of the one action
        # it emitted, inside a shared Vanillux2 conversation — and the router
        # re-decides (ROUTE any candidate or SUBMIT) after every step, bounded
        # by pool_step_limit.
        "routing_mode": routing_mode_setting,
        # turn_level only: how the shared conversation is presented to a
        # newly routed candidate.  "shared" (default) is the historical
        # raw-transcript behaviour; "switch_notice" appends an attribution
        # notice on every model switch; "model_tagged" prefixes each
        # assistant turn with the producing model's slot label;
        # "reset_context" collapses history into a bounded executed-step
        # digest at every switch.
        "context_handoff": _context_handoff(
            settings.pop("context_handoff", "shared"), routing_mode_setting
        ),
        # turn_level only: "full" (default) keeps the router's own past ROUTE
        # completions in its context; "markov" rebuilds a fresh decision
        # context every step (task + cards + aggregate route history + latest
        # digest) so the policy cannot verbatim-copy its previous action.
        "router_memory": _router_memory(
            settings.pop("router_memory", "full"), routing_mode_setting
        ),
        "shuffle_seed": _bounded_int(
            settings.pop("shuffle_seed", 0),
            name="shuffle_seed",
            minimum=0,
            maximum=2**63 - 1,
        ),
        "router_max_tokens": _bounded_int(
            settings.pop("router_max_tokens", 192),
            name="router_max_tokens",
            minimum=16,
            maximum=4096,
        ),
        "router_timeout_seconds": _positive_number(
            settings.pop("router_timeout_seconds", 180), "router_timeout_seconds"
        ),
        "pool_timeout_seconds": _positive_number(
            settings.pop("pool_timeout_seconds", 1200), "pool_timeout_seconds"
        ),
        "pool_episode_admission_enabled": _strict_bool(
            settings.pop("pool_episode_admission_enabled", False),
            "pool_episode_admission_enabled",
        ),
        "pool_episode_admission_wait_budget_seconds": _nonnegative_number(
            settings.pop("pool_episode_admission_wait_budget_seconds", 0),
            "pool_episode_admission_wait_budget_seconds",
        ),
        "total_timeout_seconds": _positive_number(
            settings.pop("total_timeout_seconds", 3000), "total_timeout_seconds"
        ),
        "reserve_evaluator_seconds": _nonnegative_number(
            settings.pop("reserve_evaluator_seconds", 300),
            "reserve_evaluator_seconds",
        ),
        "deadline_margin_seconds": _nonnegative_number(
            settings.pop("deadline_margin_seconds", 5), "deadline_margin_seconds"
        ),
        "pool_step_limit": _bounded_int(
            eval_step_limit
            if eval_step_limit is not None
            else configured_pool_step_limit,
            name="pool_step_limit",
            minimum=1,
            # turn_level records one call per step; the artifact validator's
            # bounded-entry cap requires the tighter step budget there.
            maximum=1000 if routing_mode_setting == "task_level" else 128,
        ),
        "router_observation_max_chars": _bounded_int(
            settings.pop("router_observation_max_chars", 1500),
            name="router_observation_max_chars",
            minimum=256,
            maximum=10_000,
        ),
        "pool_cost_limit": _nonnegative_number(
            settings.pop("pool_cost_limit", 0), "pool_cost_limit"
        ),
        "pool_command_timeout": _bounded_int(
            settings.pop("pool_command_timeout", 120),
            name="pool_command_timeout",
            minimum=1,
            maximum=86_400,
        ),
        "pool_max_format_errors": _bounded_int(
            settings.pop("pool_max_format_errors", 64),
            name="pool_max_format_errors",
            minimum=1,
            maximum=10_000,
        ),
        "pool_response_token_budget": _bounded_int(
            settings.pop("pool_response_token_budget", 65_536),
            name="pool_response_token_budget",
            minimum=1,
            maximum=10_000_000,
        ),
        "pool_model_retry_attempts": _bounded_int(
            settings.pop("pool_model_retry_attempts", 5),
            name="pool_model_retry_attempts",
            minimum=1,
            maximum=20,
        ),
        "observation_max_chars": _bounded_int(
            settings.pop("observation_max_chars", 10_000),
            name="observation_max_chars",
            minimum=256,
            maximum=100_000,
        ),
        "log_tail_chars": _bounded_int(
            settings.pop("log_tail_chars", 6_000),
            name="log_tail_chars",
            minimum=0,
            maximum=50_000,
        ),
        "router_model_kwargs": router_model_kwargs,
        "pool_model_kwargs": _request_mapping(
            settings.pop("pool_model_kwargs", {}), "pool_model_kwargs"
        ),
        "runner_python": str(settings.pop("runner_python", _DEFAULT_PORTABLE_PYTHON)),
        "mini_swe_bin": str(settings.pop("mini_swe_bin", _DEFAULT_MINI_SWE_BIN)),
        "result_path": SPILOT_RESULT_PATH,
        "agent_log_dir": RUNTIME_AGENT_LOG_DIR,
        # Fixed evaluation supplies a seed stable across baseline/final runs.
        # Training leaves this unset so each episode still randomizes slots via
        # its unique session/task identity.
        "slot_assignment_seed": sampling_seed,
    }
    if config["reserve_evaluator_seconds"] >= config["total_timeout_seconds"]:
        raise ValueError(
            "spilot_router reserve_evaluator_seconds must be smaller than total_timeout_seconds"
        )
    if bool(config["pool_episode_admission_enabled"]) != bool(
        config["pool_episode_admission_wait_budget_seconds"]
    ):
        raise ValueError(
            "spilot_router pool episode admission must be enabled exactly when "
            "its wait budget is positive"
        )
    if (
        config["pool_episode_admission_wait_budget_seconds"]
        > _MAX_EPISODE_ADMISSION_WAIT_SECONDS
    ):
        raise ValueError(
            "spilot_router pool_episode_admission_wait_budget_seconds must be "
            f"at most {_MAX_EPISODE_ADMISSION_WAIT_SECONDS:g}"
        )
    if config["router_model"] != _DEFAULT_ROUTER_REQUEST_MODEL:
        raise ValueError(
            "spilot_router router_request_model must be 'router/policy'"
        )
    _validate_pool_request_kwargs(model_pool)
    if settings:
        unknown = ", ".join(sorted(settings))
        raise ValueError(f"unknown spilot_router settings: {unknown}")
    return config


def _extract_forced_eval_config(
    agent_spec: AgentSpec,
) -> tuple[dict[str, str] | None, AgentSpec]:
    """Remove and validate the deliberately hard-to-enable eval-only mode."""

    settings = dict(agent_spec.settings)
    raw = settings.pop("forced_route_eval", None)
    if raw is None:
        return None, agent_spec
    if not isinstance(raw, dict):
        raise ValueError("spilot_router forced_route_eval must be a mapping")
    allowed = {"enabled", "acknowledgement", "candidate_model"}
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(
            "unknown forced_route_eval settings: " + ", ".join(sorted(unknown))
        )
    if raw.get("enabled") is not True:
        raise ValueError("forced_route_eval.enabled must be true")
    if raw.get("acknowledgement") != _FORCED_EVAL_ACK:
        raise ValueError("forced_route_eval acknowledgement is missing")
    if agent_spec.env.get(_FORCED_EVAL_AGENT_ACK_ENV) != _FORCED_EVAL_ACK:
        raise ValueError("forced_route_eval agent acknowledgement is missing")
    candidate_model = raw.get("candidate_model")
    if not isinstance(candidate_model, str) or not candidate_model.strip():
        raise ValueError("forced_route_eval.candidate_model must be non-empty")
    candidate_model = candidate_model.strip()

    pool = settings.get("model_pool")
    values = pool.values() if isinstance(pool, dict) else pool
    matches = 0
    if isinstance(values, list) or hasattr(values, "__iter__"):
        for candidate in values:
            if isinstance(candidate, dict) and candidate.get("model") == candidate_model:
                matches += 1
    if matches != 1:
        raise ValueError(
            f"forced_route_eval candidate_model matched {matches} pool entries; expected 1"
        )

    normalized = agent_spec.model_copy(update={"settings": settings})
    return {
        "acknowledgement": _FORCED_EVAL_ACK,
        "candidate_model": candidate_model,
    }, normalized


def _encode_json_b64(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.b64encode(encoded).decode("ascii")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"spilot_router {name} must be a mapping")
    return dict(value)


def _request_mapping(value: object, name: str) -> dict[str, Any]:
    result = _mapping(value, name)
    _reject_credential_fields(result, name)
    return result


def _validate_pool_request_kwargs(model_pool: object) -> None:
    values = model_pool.values() if isinstance(model_pool, dict) else model_pool
    for index, candidate in enumerate(values):
        if not isinstance(candidate, dict):
            continue
        kwargs = candidate.get("model_kwargs", {})
        if not isinstance(kwargs, dict):
            raise ValueError(
                f"spilot_router model_pool candidate {index} model_kwargs must be a mapping"
            )
        _reject_credential_fields(kwargs, f"model_pool candidate {index} model_kwargs")


def _reject_credential_fields(value: object, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if (
                normalized in {"api_key", "apikey", "authorization", "password", "secret", "token"}
                or normalized.endswith("_api_key")
                or normalized.endswith("_token")
                or normalized.endswith("_password")
                or normalized.endswith("_secret")
            ):
                raise ValueError(f"spilot_router {path} must not contain credential field {key!r}")
            _reject_credential_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_credential_fields(child, f"{path}[{index}]")


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"spilot_router {name} must be a boolean")
    return value


def _slot_label_mode(value: object) -> str:
    if value not in ("anonymous", "real_names"):
        raise ValueError(
            "spilot_router slot_label_mode must be 'anonymous' or 'real_names'; "
            f"got {value!r}"
        )
    return str(value)


def _routing_mode(value: object) -> str:
    if value not in ("task_level", "turn_level"):
        raise ValueError(
            "spilot_router routing_mode must be 'task_level' or 'turn_level'; "
            f"got {value!r}"
        )
    return str(value)


def _router_memory(value: object, routing_mode: str) -> str:
    if value not in ("full", "markov"):
        raise ValueError(
            f"spilot_router router_memory must be 'full' or 'markov'; got {value!r}"
        )
    if value != "full" and routing_mode != "turn_level":
        raise ValueError(
            "spilot_router router_memory='markov' requires routing_mode='turn_level'"
        )
    return str(value)


def _context_handoff(value: object, routing_mode: str) -> str:
    if value not in ("shared", "switch_notice", "model_tagged", "reset_context"):
        raise ValueError(
            "spilot_router context_handoff must be one of 'shared', "
            f"'switch_notice', 'model_tagged', 'reset_context'; got {value!r}"
        )
    if value != "shared" and routing_mode != "turn_level":
        raise ValueError(
            "spilot_router context_handoff variants require "
            "routing_mode='turn_level'"
        )
    return str(value)


def _bounded_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"spilot_router {name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"spilot_router {name} must be between {minimum} and {maximum}")
    return value


def _positive_number(value: object, name: str) -> float:
    parsed = _nonnegative_number(value, name)
    if parsed <= 0:
        raise ValueError(f"spilot_router {name} must be positive")
    return parsed


def _nonnegative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"spilot_router {name} must be a number")
    parsed = float(value)
    if parsed < 0 or not math.isfinite(parsed):
        raise ValueError(f"spilot_router {name} must be finite and non-negative")
    return parsed


def _validate_result_metadata(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("router result must be an object")
    required = {
        "schema_version",
        "action_valid",
        "actions",
        "calls",
        "submitted",
        "total_cost",
        "slot_mapping",
        "slot_mapping_fingerprint",
        "termination_reason",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"router result missing fields: {', '.join(sorted(missing))}")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported router result schema_version")
    if not isinstance(payload.get("action_valid"), bool):
        raise ValueError("router result action_valid must be boolean")
    if not isinstance(payload.get("submitted"), bool):
        raise ValueError("router result submitted must be boolean")
    # task_level episodes record at most 2 decisions/calls; turn_level records
    # one call per agent step (step budget capped at 128) plus up to one more
    # router decision than executed steps.  The 128 KiB encoded cap below is
    # the real payload bound; these entry caps just reject unbounded lists.
    if not isinstance(payload.get("actions"), list) or len(payload["actions"]) > 129:
        raise ValueError("router result actions must contain at most 129 entries")
    if not isinstance(payload.get("calls"), list) or len(payload["calls"]) > 128:
        raise ValueError("router result calls must contain at most 128 entries")
    if not isinstance(payload.get("slot_mapping"), dict):
        raise ValueError("router result slot_mapping must be an object")
    total_cost = payload.get("total_cost")
    if isinstance(total_cost, bool) or not isinstance(total_cost, (int, float)):
        raise ValueError("router result total_cost must be numeric")

    # Round-trip through JSON to detach pydantic/custom mapping subclasses and
    # cap the artifact before it is attached to an in-memory session result.
    sanitized = json.loads(json.dumps(payload, ensure_ascii=False))
    encoded = json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 128_000:
        raise ValueError("router result exceeds 128 KiB")
    return sanitized


def _bounded_text(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."


__all__ = ["SpilotRouterHarness"]
