"""Portable SPilot router orchestrator.

This file is uploaded into each task runtime by :mod:`spilot_router`.  Keep it
free of Polar imports: the portable mini-SWE Python environment contains only
mini-SWE-agent and its dependencies.

The state machine is intentionally small::

    ROUTE(slot) -> pool solve -> [SUBMIT | VERIFY(slot) -> pool repair] -> submit

Pool agents share the current working directory, but each starts with a fresh
conversation.  Pool failures and timeouts are observations for the second
router decision; malformed router actions and control-plane failures are kept
separate so training can zero-reward the former and mask the latter.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Protocol


_CONFIG_ENV = "SPILOT_ROUTER_CONFIG_B64"
_TASK_ENV = "SPILOT_TASK_B64"
_ROUTER_CAPABILITY_ENV = "POLAR_ROUTER_CAPABILITY"
_MODEL_POOL_CAPABILITY_ENV = "POLAR_MODEL_POOL_CAPABILITY"
_MINI_SWE_TASK_B64_ENV = "POLAR_MINI_SWE_TASK_B64"
_VANILLUX2_CONFIG_PATH = "/opt/polar-mini-swe-agent/config/vanillux2.yaml"
_VANILLUX2_MODEL_CLASS = "polar_mini_swe_vanillux.Vanillux2LitellmModel"
_VANILLUX2_ENVIRONMENT_CLASS = "polar_mini_swe_timing.Vanillux2TimedLocalEnvironment"
_SLOT_RE = re.compile(r"^M(?:0|[1-9][0-9]*)$")
_MAX_ERROR_CHARS = 500
_MAX_CARD_CHARS = 4_000
_GIT_COMMAND_TIMEOUT_SECONDS = 10.0
_FORBIDDEN_REQUEST_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
}


class RouterProtocolError(ValueError):
    """A sampled router completion did not satisfy the action grammar."""


class GatewayInfrastructureError(RuntimeError):
    """The trainable router could not reach or use its serving gateway."""


class PoolInfrastructureError(RuntimeError):
    """The local pool-agent executable could not be launched."""


def _classify_process_failure(
    *, return_code: int, timed_out: bool
) -> tuple[str | None, int | None, str | None]:
    """Classify a child failure without confusing timeout ``-1`` with SIGHUP."""

    if timed_out:
        return "timeout", None, None
    if return_code == 0:
        return None, None, None
    if return_code < 0:
        signal_number = -return_code
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"UNKNOWN_SIGNAL_{signal_number}"
        return "signal", signal_number, signal_name
    return "exit_code", None, None


@dataclass(frozen=True)
class Candidate:
    slot: str
    model: str
    card: object
    cost_weight: float = 1.0
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def public_metadata(self) -> dict[str, object]:
        return {
            "model": self.model,
            "card": self.card,
            "cost_weight": self.cost_weight,
        }


@dataclass
class RouterCompletion:
    content: str
    usage: dict[str, int] = field(default_factory=dict)
    request_id: str | None = None
    finish_reason: str | None = None


@dataclass
class PoolCallResult:
    slot: str
    model: str
    role: str
    status: str
    return_code: int
    duration_ms: int
    attempted: bool
    timed_out: bool
    log_file: str
    log_tail: str
    git_status: str
    git_diff_stat: str
    workspace_fingerprint: str
    error: str | None = None
    failure_kind: str | None = None
    signal_number: int | None = None
    signal_name: str | None = None

    def metadata(self, *, index: int, cost: float) -> dict[str, object]:
        result: dict[str, object] = {
            "index": index,
            "slot": self.slot,
            "model": self.model,
            "role": self.role,
            "status": self.status,
            "return_code": self.return_code,
            "duration_ms": self.duration_ms,
            "attempted": self.attempted,
            "timed_out": self.timed_out,
            "cost": cost,
            "log_file": self.log_file,
            "workspace_fingerprint": self.workspace_fingerprint,
        }
        if self.error:
            result["error"] = _bounded_text(self.error, _MAX_ERROR_CHARS)
        if self.failure_kind is not None:
            result["failure_kind"] = self.failure_kind
        if self.signal_number is not None:
            result["signal_number"] = self.signal_number
        if self.signal_name is not None:
            result["signal_name"] = self.signal_name
        return result

    def observation(self, max_chars: int) -> str:
        payload = {
            "model_slot": self.slot,
            "role": self.role,
            "status": self.status,
            "return_code": self.return_code,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
            "failure_kind": self.failure_kind,
            "signal_number": self.signal_number,
            "signal_name": self.signal_name,
            "workspace_fingerprint": self.workspace_fingerprint,
            "git_status": self.git_status,
            "git_diff_stat": self.git_diff_stat,
            "agent_log_tail": self.log_tail,
            "error": self.error,
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return _bounded_text(text, max_chars)


class RouterClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        timeout_seconds: float,
        model_kwargs: dict[str, Any],
    ) -> RouterCompletion: ...


class PoolExecutor(Protocol):
    def run(
        self,
        *,
        candidate: Candidate,
        task: str,
        role: str,
        call_index: int,
        timeout_seconds: float,
    ) -> PoolCallResult: ...


class OpenAIGatewayClient:
    """Small OpenAI-chat client with optional Unix-domain-socket transport."""

    def __init__(self) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - portable image contract
            raise GatewayInfrastructureError("portable runtime is missing httpx") from exc

        base_url = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
        if not base_url:
            raise GatewayInfrastructureError("OPENAI_BASE_URL is not configured")
        self._url = (
            f"{base_url}/chat/completions"
            if base_url.endswith("/v1")
            else f"{base_url}/v1/chat/completions"
        )
        socket_path = os.environ.get("POLAR_GATEWAY_UDS", "").strip()
        if socket_path:
            transport = httpx.HTTPTransport(uds=socket_path)
            self._client = httpx.Client(transport=transport, trust_env=False)
        else:
            self._client = httpx.Client(trust_env=False)

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        timeout_seconds: float,
        model_kwargs: dict[str, Any],
    ) -> RouterCompletion:
        payload = _openai_chat_payload(
            model=model,
            messages=messages,
            model_kwargs=model_kwargs,
        )
        headers = {"Content-Type": "application/json"}
        api_key = (
            os.environ.get(_ROUTER_CAPABILITY_ENV, "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            response = self._client.post(
                self._url,
                json=payload,
                headers=headers,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            detail = _http_exception_detail(exc)
            raise GatewayInfrastructureError(f"router gateway request failed: {detail}") from exc

        try:
            choice = body["choices"][0]
            message = choice["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError) as exc:
            raise GatewayInfrastructureError(
                "router gateway returned an invalid chat-completion response"
            ) from exc
        # A valid HTTP completion can consume its budget in reasoning and end
        # with content=null/empty. That is a sampled invalid action, not a
        # serving failure, so preserve it for the strict parser to zero-reward.
        if content is None:
            content = ""
        elif not isinstance(content, str):
            raise GatewayInfrastructureError("router completion content must be text")
        usage = _sanitize_usage(body.get("usage")) if isinstance(body, dict) else {}
        request_id = body.get("id") if isinstance(body, dict) else None
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        return RouterCompletion(
            content=content,
            usage=usage,
            request_id=request_id if isinstance(request_id, str) else None,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        )


class MiniSwePoolExecutor:
    """Run each frozen candidate as a complete mini-SWE coding agent."""

    def __init__(self, config: dict[str, Any], *, cwd: Path | None = None) -> None:
        self.config = config
        self.cwd = cwd or Path.cwd()
        self.log_dir = Path(str(config["agent_log_dir"]))
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        candidate: Candidate,
        task: str,
        role: str,
        call_index: int,
        timeout_seconds: float,
    ) -> PoolCallResult:
        log_path = self.log_dir / f"spilot-pool-{call_index:02d}-{role}.txt"
        timing_path = self.log_dir / f"spilot-pool-{call_index:02d}-timing.jsonl"
        # Vanillux2 persists cwd and exported variables between turns.  A
        # verifier is a fresh agent sharing only the mutable task workspace,
        # so every pool call needs an isolated protocol-state directory.
        state_dir = self.log_dir / f"spilot-pool-{call_index:02d}-state"
        instruction = task if role == "solve" else _verification_instruction(task)
        model_kwargs = dict(self.config.get("pool_model_kwargs", {}))
        model_kwargs.update(candidate.model_kwargs)
        args = self._command(
            model=candidate.model,
            timing_path=str(timing_path),
            state_dir=str(state_dir),
            model_kwargs=model_kwargs,
        )
        child_env = dict(os.environ)
        # The frozen coding agent gets only the model-pool-scoped capability.
        # Keep the Router capability and outer protocol/config out of its
        # ordinary child environment.
        child_env.pop(_CONFIG_ENV, None)
        child_env.pop(_TASK_ENV, None)
        child_env.pop(_ROUTER_CAPABILITY_ENV, None)
        child_env[_MINI_SWE_TASK_B64_ENV] = base64.b64encode(
            instruction.encode("utf-8")
        ).decode("ascii")
        pool_capability = child_env.pop(_MODEL_POOL_CAPABILITY_ENV, "").strip()
        if pool_capability:
            child_env["OPENAI_API_KEY"] = pool_capability
        base_url = child_env.get("OPENAI_BASE_URL", "")
        if base_url:
            child_env["OPENAI_API_BASE"] = base_url
        child_env.update(
            {
                "MSWEA_CONFIGURED": "true",
                "MSWEA_COST_TRACKING": "ignore_errors",
                "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": str(
                    self.config["pool_model_retry_attempts"]
                ),
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            }
        )

        started = time.monotonic()
        process: subprocess.Popen[str] | None = None
        timed_out = False
        error: str | None = None
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as stream:
                try:
                    process = subprocess.Popen(
                        args,
                        cwd=self.cwd,
                        env=child_env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                    )
                except (FileNotFoundError, PermissionError, OSError) as exc:
                    raise PoolInfrastructureError(
                        f"could not launch mini-SWE pool agent: {_bounded_text(str(exc), 240)}"
                    ) from exc
                try:
                    return_code = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process_group(process)
                    return_code = -1
                    error = f"pool agent exceeded {timeout_seconds:.1f}s timeout"
        except PoolInfrastructureError:
            raise
        except (OSError, UnicodeError) as exc:
            raise PoolInfrastructureError(
                f"could not write pool-agent log: {_bounded_text(str(exc), 240)}"
            ) from exc

        duration_ms = max(0, int((time.monotonic() - started) * 1000))
        failure_kind, signal_number, signal_name = _classify_process_failure(
            return_code=return_code,
            timed_out=timed_out,
        )
        if timed_out:
            status = "timeout"
        elif return_code == 0:
            status = "completed"
        elif failure_kind == "signal":
            status = "failed"
            error = f"pool agent terminated by signal {signal_name} ({signal_number})"
        else:
            status = "failed"
            error = f"pool agent exited with code {return_code}"
        git_status, git_diff_stat, fingerprint = _workspace_summary(self.cwd)
        return PoolCallResult(
            slot=candidate.slot,
            model=candidate.model,
            role=role,
            status=status,
            return_code=return_code,
            duration_ms=duration_ms,
            attempted=True,
            timed_out=timed_out,
            log_file=str(log_path),
            log_tail=_read_tail(log_path, int(self.config["log_tail_chars"])),
            git_status=git_status,
            git_diff_stat=git_diff_stat,
            workspace_fingerprint=fingerprint,
            error=error,
            failure_kind=failure_kind,
            signal_number=signal_number,
            signal_name=signal_name,
        )

    def deadline_result(
        self,
        *,
        candidate: Candidate,
        role: str,
        call_index: int,
    ) -> PoolCallResult:
        git_status, git_diff_stat, fingerprint = _workspace_summary(self.cwd)
        return PoolCallResult(
            slot=candidate.slot,
            model=candidate.model,
            role=role,
            status="timeout",
            return_code=-1,
            duration_ms=0,
            attempted=False,
            timed_out=True,
            log_file=str(self.log_dir / f"spilot-pool-{call_index:02d}-{role}.txt"),
            log_tail="",
            git_status=git_status,
            git_diff_stat=git_diff_stat,
            workspace_fingerprint=fingerprint,
            error="pool call skipped because the episode deadline was exhausted",
            failure_kind="timeout",
        )

    def _command(
        self,
        *,
        model: str,
        timing_path: str,
        state_dir: str,
        model_kwargs: dict[str, Any],
    ) -> list[str]:
        model_id = model if model.startswith("openai/") else f"openai/{model}"
        args = [
            str(self.config["mini_swe_bin"]),
            "--yolo",
            "--environment-class",
            _VANILLUX2_ENVIRONMENT_CLASS,
            "--model-class",
            _VANILLUX2_MODEL_CLASS,
            f"--model={model_id}",
            "--cost-limit",
            str(self.config["pool_cost_limit"]),
            "--exit-immediately",
            "-c",
            _VANILLUX2_CONFIG_PATH,
            "-c",
            f"agent.step_limit={self.config['pool_step_limit']}",
            "-c",
            f"agent.max_consecutive_format_errors={self.config['pool_max_format_errors']}",
            "-c",
            f"environment.timeout={self.config['pool_command_timeout']}",
            "-c",
            f"environment.max_output_chars={self.config['observation_max_chars']}",
            "-c",
            f"environment.state_dir={state_dir}",
            "-c",
            f"model.response_token_budget={self.config['pool_response_token_budget']}",
            "-c",
            "environment.env.PYTHONPATH=",
            "-c",
            f"environment.timing_path={timing_path}",
        ]
        if model_kwargs:
            args.extend(
                [
                    "-c",
                    "model.model_kwargs="
                    + json.dumps(model_kwargs, separators=(",", ":"), sort_keys=True),
                ]
            )
        return args


class SpilotOrchestrator:
    """Execute and record one bounded SPilot routing episode."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        task: str,
        router: RouterClient,
        pool: PoolExecutor,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = _validate_config(config)
        self.task = task
        self.router = router
        self.pool = pool
        self.clock = clock
        self.candidates = _assign_slots(
            self.config["model_pool"],
            shuffle=bool(self.config["shuffle_slots"]),
            seed=int(self.config["shuffle_seed"]),
            stable_seed=self.config.get("slot_assignment_seed"),
            session_id=os.environ.get("SESSION_ID", ""),
            task_id=os.environ.get("TASK_ID", ""),
        )
        self.candidate_by_slot = {candidate.slot: candidate for candidate in self.candidates}
        mapping = {candidate.slot: candidate.public_metadata() for candidate in self.candidates}
        mapping_json = json.dumps(mapping, separators=(",", ":"), sort_keys=True)
        self.result: dict[str, Any] = {
            "schema_version": 1,
            "action_valid": True,
            "actions": [],
            "calls": [],
            "submitted": False,
            "total_cost": 0.0,
            "slot_mapping": mapping,
            "slot_mapping_fingerprint": hashlib.sha256(mapping_json.encode()).hexdigest(),
            "termination_reason": "not_started",
        }
        usable_seconds = float(self.config["total_timeout_seconds"]) - float(
            self.config["reserve_evaluator_seconds"]
        )
        self.deadline = self.clock() + usable_seconds

    def run(self) -> dict[str, Any]:
        initial_messages = self._initial_messages()
        first = self._router_completion(initial_messages)
        first_action = self._record_action(first, expected="ROUTE", step=0)
        if first_action is None:
            return self.result

        first_candidate = self.candidate_by_slot[first_action["model_slot"]]
        first_call = self._pool_call(first_candidate, role="solve", call_index=0)
        self._record_call(first_call, first_candidate)

        if int(self.config["max_pool_calls"]) == 1:
            self.result["submitted"] = True
            self.result["termination_reason"] = "m0_auto_submit"
            self.result["final_workspace_fingerprint"] = first_call.workspace_fingerprint
            return self.result

        observation = first_call.observation(int(self.config["observation_max_chars"]))
        followup_messages = [
            *initial_messages,
            {"role": "assistant", "content": first.content},
            {
                "role": "user",
                "content": (
                    "The selected coding agent has finished. This observation contains "
                    "only public execution state; hidden evaluator tests have not run.\n"
                    f"OBSERVATION:\n{observation}\n\n"
                    "Choose exactly one next action. Either submit the current workspace:\n"
                    '{"action":"SUBMIT"}\n'
                    "or spend the final pool call on a fresh verifier/repair agent:\n"
                    '{"action":"VERIFY","model_slot":"M0"}\n'
                    "Use one available slot and output only the JSON object."
                ),
            },
        ]
        second = self._router_completion(followup_messages)
        second_action = self._record_action(second, expected="FINAL", step=1)
        if second_action is None:
            self.result["final_workspace_fingerprint"] = first_call.workspace_fingerprint
            return self.result

        if second_action["action"] == "SUBMIT":
            self.result["submitted"] = True
            self.result["termination_reason"] = "router_submit"
            self.result["final_workspace_fingerprint"] = first_call.workspace_fingerprint
            return self.result

        candidate = self.candidate_by_slot[second_action["model_slot"]]
        verify_call = self._pool_call(candidate, role="verify", call_index=1)
        self._record_call(verify_call, candidate)
        self.result["submitted"] = True
        self.result["termination_reason"] = "verify_auto_submit"
        self.result["final_workspace_fingerprint"] = verify_call.workspace_fingerprint
        return self.result

    def mark_infrastructure_error(self, exc: BaseException) -> None:
        self.result["action_valid"] = False
        self.result["submitted"] = False
        self.result["termination_reason"] = "infrastructure_error"
        self.result["infrastructure_error"] = _bounded_text(str(exc), _MAX_ERROR_CHARS)

    def _initial_messages(self) -> list[dict[str, str]]:
        cards = [
            {"model_slot": candidate.slot, "model_card": candidate.card}
            for candidate in self.candidates
        ]
        return [
            {
                "role": "system",
                "content": (
                    "You are SPilot, a routing policy for software-engineering agents. "
                    "Do not solve the task yourself. Select one candidate to run a full "
                    "coding-agent attempt. Your response must be exactly one JSON object, "
                    "with no markdown, commentary, or extra keys."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"TASK:\n{self.task}\n\n"
                    "AVAILABLE MODEL SLOTS:\n"
                    f"{json.dumps(cards, ensure_ascii=False, sort_keys=True)}\n\n"
                    "Choose the first action using exactly this schema:\n"
                    '{"action":"ROUTE","model_slot":"M0"}\n'
                    "Replace M0 with one available slot. Output only the JSON object."
                ),
            },
        ]

    def _router_completion(self, messages: list[dict[str, str]]) -> RouterCompletion:
        remaining = self.deadline - self.clock()
        margin = float(self.config["deadline_margin_seconds"])
        if remaining <= margin:
            raise GatewayInfrastructureError("episode deadline exhausted before router call")
        timeout = min(float(self.config["router_timeout_seconds"]), remaining - margin)
        kwargs = dict(self.config["router_model_kwargs"])
        kwargs.setdefault("max_tokens", int(self.config["router_max_tokens"]))
        return self.router.complete(
            model=str(self.config["router_model"]),
            messages=messages,
            timeout_seconds=timeout,
            model_kwargs=kwargs,
        )

    def _record_action(
        self,
        completion: RouterCompletion,
        *,
        expected: str,
        step: int,
    ) -> dict[str, str] | None:
        try:
            action = parse_router_action(
                completion.content,
                expected=expected,
                allowed_slots=set(self.candidate_by_slot),
            )
        except RouterProtocolError as exc:
            self.result["action_valid"] = False
            self.result["submitted"] = False
            self.result["termination_reason"] = f"invalid_action_step_{step}"
            self.result["actions"].append(
                {
                    "step": step,
                    "valid": False,
                    "error": _bounded_text(str(exc), _MAX_ERROR_CHARS),
                    "response_excerpt": _bounded_text(completion.content, 500),
                    "usage": completion.usage,
                    "finish_reason": completion.finish_reason,
                }
            )
            return None
        self.result["actions"].append(
            {
                "step": step,
                "valid": True,
                **action,
                "usage": completion.usage,
                "finish_reason": completion.finish_reason,
            }
        )
        return action

    def _pool_call(
        self,
        candidate: Candidate,
        *,
        role: str,
        call_index: int,
    ) -> PoolCallResult:
        remaining = self.deadline - self.clock()
        margin = float(self.config["deadline_margin_seconds"])
        reserve = margin
        if call_index == 0 and int(self.config["max_pool_calls"]) > 1:
            reserve += float(self.config["router_timeout_seconds"])
        available = remaining - reserve
        if available <= 0:
            deadline_result = getattr(self.pool, "deadline_result", None)
            if callable(deadline_result):
                return deadline_result(
                    candidate=candidate,
                    role=role,
                    call_index=call_index,
                )
            return PoolCallResult(
                slot=candidate.slot,
                model=candidate.model,
                role=role,
                status="timeout",
                return_code=-1,
                duration_ms=0,
                attempted=False,
                timed_out=True,
                log_file="",
                log_tail="",
                git_status="",
                git_diff_stat="",
                workspace_fingerprint="unavailable",
                error="pool call skipped because the episode deadline was exhausted",
                failure_kind="timeout",
            )
        timeout = min(float(self.config["pool_timeout_seconds"]), available)
        return self.pool.run(
            candidate=candidate,
            task=self.task,
            role=role,
            call_index=call_index,
            timeout_seconds=timeout,
        )

    def _record_call(self, call: PoolCallResult, candidate: Candidate) -> None:
        cost = candidate.cost_weight if call.attempted else 0.0
        self.result["total_cost"] = float(self.result["total_cost"]) + cost
        self.result["calls"].append(call.metadata(index=len(self.result["calls"]), cost=cost))


def parse_router_action(
    text: str,
    *,
    expected: str,
    allowed_slots: set[str],
) -> dict[str, str]:
    """Parse the deliberately tiny action language without repair/coercion."""

    if not isinstance(text, str) or not text:
        raise RouterProtocolError("router response is empty")
    # JSON permits surrounding whitespace, and many chat templates append one
    # terminal newline.  Accept that syntax while still rejecting markdown,
    # prose, coercions, duplicate keys, and extra object fields.
    text = text.strip()

    def reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RouterProtocolError(f"router response repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicate_pairs)
    except RouterProtocolError:
        raise
    except json.JSONDecodeError as exc:
        raise RouterProtocolError(f"router response is not strict JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise RouterProtocolError("router action must be a JSON object")
    action = value.get("action")
    if expected == "ROUTE":
        if set(value) != {"action", "model_slot"} or action != "ROUTE":
            raise RouterProtocolError("first action must have exactly action=ROUTE and model_slot")
    elif expected == "FINAL":
        if action == "SUBMIT":
            if set(value) != {"action"}:
                raise RouterProtocolError("SUBMIT must not contain extra fields")
            return {"action": "SUBMIT"}
        if set(value) != {"action", "model_slot"} or action != "VERIFY":
            raise RouterProtocolError(
                "second action must be exactly SUBMIT or VERIFY with model_slot"
            )
    else:  # pragma: no cover - internal programming error
        raise ValueError(f"unknown expected action phase: {expected}")

    slot = value.get("model_slot")
    if not isinstance(slot, str) or slot not in allowed_slots:
        raise RouterProtocolError(f"model_slot must be one of {sorted(allowed_slots)}")
    return {"action": str(action), "model_slot": slot}


def _assign_slots(
    raw_pool: object,
    *,
    shuffle: bool,
    seed: int,
    stable_seed: int | None,
    session_id: str,
    task_id: str,
) -> list[Candidate]:
    if isinstance(raw_pool, list):
        labels = [f"M{index}" for index in range(len(raw_pool))]
        raw_candidates = list(raw_pool)
    elif isinstance(raw_pool, dict):
        labels = list(raw_pool)
        if not labels or any(
            not isinstance(label, str) or not _SLOT_RE.fullmatch(label) for label in labels
        ):
            raise ValueError("model_pool mapping keys must be logical slots such as M0")
        labels.sort(key=lambda value: int(value[1:]))
        raw_candidates = [raw_pool[label] for label in labels]
    else:
        raise ValueError("model_pool must be a non-empty list or mapping")
    if not raw_candidates:
        raise ValueError("model_pool must not be empty")

    parsed = [_parse_candidate(value) for value in raw_candidates]
    if shuffle and len(parsed) > 1:
        if stable_seed is None:
            material = f"{seed}\0{session_id}\0{task_id}".encode()
            derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        else:
            derived_seed = int(stable_seed)
        random.Random(derived_seed).shuffle(parsed)
    return [
        Candidate(
            slot=slot,
            model=item["model"],
            card=item["card"],
            cost_weight=item["cost_weight"],
            model_kwargs=item["model_kwargs"],
        )
        for slot, item in zip(labels, parsed)
    ]


def _parse_candidate(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        model = value.strip()
        if not model:
            raise ValueError("candidate model alias must not be empty")
        return {
            "model": model,
            "card": {"name": model},
            "cost_weight": 1.0,
            "model_kwargs": {},
        }
    if not isinstance(value, dict):
        raise ValueError("each model_pool candidate must be a string or mapping")
    allowed = {"model", "card", "description", "cost_weight", "model_kwargs"}
    extra = set(value).difference(allowed)
    if extra:
        raise ValueError(f"unknown candidate fields: {', '.join(sorted(extra))}")
    model = value.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("candidate.model must be a non-empty string")
    card = value.get("card", value.get("description", {"name": model}))
    encoded_card = json.dumps(card, ensure_ascii=False)
    if len(encoded_card) > _MAX_CARD_CHARS:
        raise ValueError(f"candidate card exceeds {_MAX_CARD_CHARS} characters")
    raw_cost = value.get("cost_weight", 1.0)
    if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
        raise ValueError("candidate.cost_weight must be numeric")
    cost = float(raw_cost)
    if cost < 0 or not math.isfinite(cost):
        raise ValueError("candidate.cost_weight must be finite and non-negative")
    kwargs = value.get("model_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("candidate.model_kwargs must be a mapping")
    _reject_secret_request_fields(kwargs, "candidate.model_kwargs")
    return {
        "model": model.strip(),
        "card": card,
        "cost_weight": cost,
        "model_kwargs": dict(kwargs),
    }


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "router_model",
        "model_pool",
        "max_pool_calls",
        "shuffle_slots",
        "shuffle_seed",
        "router_max_tokens",
        "router_timeout_seconds",
        "pool_timeout_seconds",
        "total_timeout_seconds",
        "reserve_evaluator_seconds",
        "deadline_margin_seconds",
        "pool_step_limit",
        "pool_cost_limit",
        "pool_command_timeout",
        "pool_max_format_errors",
        "pool_response_token_budget",
        "pool_model_retry_attempts",
        "observation_max_chars",
        "log_tail_chars",
        "router_model_kwargs",
        "pool_model_kwargs",
        "slot_assignment_seed",
        "mini_swe_bin",
        "agent_log_dir",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"runner config missing fields: {', '.join(sorted(missing))}")
    if config["max_pool_calls"] not in (1, 2):
        raise ValueError("max_pool_calls must be 1 or 2")
    slot_assignment_seed = config.get("slot_assignment_seed")
    if slot_assignment_seed is not None and (
        isinstance(slot_assignment_seed, bool)
        or not isinstance(slot_assignment_seed, int)
        or slot_assignment_seed < 0
    ):
        raise ValueError("slot_assignment_seed must be a non-negative integer or null")
    if not isinstance(config["router_model_kwargs"], dict):
        raise ValueError("router_model_kwargs must be a mapping")
    if not isinstance(config["pool_model_kwargs"], dict):
        raise ValueError("pool_model_kwargs must be a mapping")
    _reject_secret_request_fields(config["router_model_kwargs"], "router_model_kwargs")
    _reject_secret_request_fields(config["pool_model_kwargs"], "pool_model_kwargs")
    reserved = {"model", "messages", "stream"}
    overlap = reserved.intersection(config["router_model_kwargs"])
    if overlap:
        raise ValueError(f"router_model_kwargs cannot override: {', '.join(sorted(overlap))}")
    return config


def _reject_secret_request_fields(value: object, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if (
                normalized in _FORBIDDEN_REQUEST_KEYS
                or normalized.endswith("_api_key")
                or normalized.endswith("_token")
                or normalized.endswith("_password")
                or normalized.endswith("_secret")
            ):
                raise ValueError(f"{path} must not contain credential field {key!r}")
            _reject_secret_request_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_request_fields(child, f"{path}[{index}]")


def _verification_instruction(task: str) -> str:
    return (
        "You are the final verifier and repair agent. A previous coding agent has "
        "already attempted the task in this same workspace. Inspect the current "
        "files and diff, run relevant public tests, preserve correct work, and fix "
        "any remaining issue. Hidden evaluator tests are unavailable. Complete the "
        "implementation rather than only reviewing it.\n\nORIGINAL TASK:\n"
        f"{task}"
    )


def _workspace_summary(cwd: Path) -> tuple[str, str, str]:
    status = _run_bounded_command(
        ["git", "status", "--short", "--untracked-files=normal"], cwd=cwd, limit=8_000
    )
    diff_stat = _run_bounded_command(
        ["git", "diff", "--no-ext-diff", "--stat"], cwd=cwd, limit=8_000
    )
    digest = hashlib.sha256(
        ("status\0" + status + "\0diff-stat\0" + diff_stat).encode("utf-8", errors="replace")
    ).hexdigest()
    return status, diff_stat, digest


def _run_bounded_command(args: list[str], *, cwd: Path, limit: int) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=_GIT_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _bounded_text(f"unavailable: {exc}", min(limit, 240))
    return _bounded_text(result.stdout or "", limit)


def _read_tail(path: Path, limit: int) -> str:
    if limit <= 0:
        return ""
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit * 4))
            data = stream.read()
        return data.decode("utf-8", errors="replace")[-limit:]
    except OSError as exc:
        return _bounded_text(f"log unavailable: {exc}", 240)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5.0)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


def _sanitize_usage(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[key] = item
    return result


def _openai_chat_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Match OpenAI SDK/LiteLLM ``extra_body`` merge semantics explicitly."""

    kwargs = dict(model_kwargs)
    extra_body = kwargs.pop("extra_body", {})
    if not isinstance(extra_body, dict):
        raise GatewayInfrastructureError("router_model_kwargs.extra_body must be an object")
    protected = {"model", "messages", "stream"}
    forbidden = protected.intersection(extra_body)
    if forbidden:
        raise GatewayInfrastructureError(
            "router extra_body cannot override " + ", ".join(sorted(forbidden))
        )
    overlap = set(kwargs).intersection(extra_body)
    if overlap:
        raise GatewayInfrastructureError(
            "router extra_body duplicates request fields: " + ", ".join(sorted(overlap))
        )
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        **kwargs,
        **extra_body,
    }


def _http_exception_detail(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", "unknown")
        try:
            detail = response.text
        except Exception:
            detail = ""
        return _bounded_text(f"HTTP {status}: {detail}", _MAX_ERROR_CHARS)
    return _bounded_text(f"{type(exc).__name__}: {exc}", _MAX_ERROR_CHARS)


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


def _decode_env_b64(name: str) -> bytes:
    raw = os.environ.get(name)
    if raw is None:
        raise ValueError(f"required environment variable {name} is missing")
    try:
        return base64.b64decode(raw, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} is not valid base64") from exc


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    try:
        config_obj = json.loads(_decode_env_b64(_CONFIG_ENV))
        task = _decode_env_b64(_TASK_ENV).decode("utf-8")
        if not isinstance(config_obj, dict):
            raise ValueError("router config must be a JSON object")
        result_path = Path(str(config_obj.get("result_path", "/tmp/router_result.json")))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"SPilot configuration error: {_bounded_text(str(exc), 500)}", file=sys.stderr)
        return 2

    gateway: OpenAIGatewayClient | None = None
    orchestrator: SpilotOrchestrator | None = None
    try:
        gateway = OpenAIGatewayClient()
        pool = MiniSwePoolExecutor(config_obj)
        orchestrator = SpilotOrchestrator(
            config=config_obj,
            task=task,
            router=gateway,
            pool=pool,
        )
        result = orchestrator.run()
        _write_result(result_path, result)
        # Invalid actions and pool model failures are sampled-policy outcomes,
        # not process failures.  Their metadata lets the evaluator force zero
        # reward while preserving aligned router logprobs.
        return 0
    except (GatewayInfrastructureError, PoolInfrastructureError, ValueError) as exc:
        if orchestrator is not None:
            orchestrator.mark_infrastructure_error(exc)
            try:
                _write_result(result_path, orchestrator.result)
            except OSError:
                pass
        print(f"SPilot infrastructure error: {_bounded_text(str(exc), 500)}", file=sys.stderr)
        return 2
    except Exception as exc:  # fail closed for unexpected control-plane bugs
        if orchestrator is not None:
            orchestrator.mark_infrastructure_error(exc)
            try:
                _write_result(result_path, orchestrator.result)
            except OSError:
                pass
        print(
            f"SPilot unexpected infrastructure error: "
            f"{_bounded_text(f'{type(exc).__name__}: {exc}', 500)}",
            file=sys.stderr,
        )
        return 2
    finally:
        if gateway is not None:
            gateway.close()


if __name__ == "__main__":
    raise SystemExit(main())
