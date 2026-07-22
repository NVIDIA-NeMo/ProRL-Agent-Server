"""V3 dual-worker oracle controller — the finalized standalone engine.

This module is self-contained (no dependency on the V2.x engines, which stay
in the repo purely as the ablation record) and encodes exactly the design
settled on 2026-07-12 after the V2.6-V2.9 push ablations, the V2.8.x file
channel, and the V3 2x3 reset matrix under three judges:

- two private workers (small = cheap/free, large = strong/paid) share one
  workspace; each keeps its own conversation forever — nothing is ever reset
  and nothing is ever pushed into a worker's history;
- the only injection is a turn-taking switch notice carrying the unseen-range
  cursor ("events new since your last participation are E{a}-E{b}");
- state recovery is pull-based: a container-side /agent_history log (README +
  delimited event blocks, synced at every switch, self-reads redacted) plus
  the live workspace;
- a query-only controller is consulted after every event: it sees the task,
  the last `window_events` events (both actors, tool calls and observations
  only — never worker prose), PINNED evidence, and switching-frequency
  features; it outputs a strict three-field JSON resolved through a fixed
  routing matrix, executed through per-direction confirmation counters.

The only tunables intended for protocol experiments are the confirmation
counts (`switch_confirmations`, `upgrade_confirmations`,
`downgrade_confirmations`). Everything the ablations killed — push handoffs
of any kind, history resets, the submission gate, the deescalation gate — is
gone rather than switched off.
"""

import base64
import json
import os
import re
import threading
import time
from typing import Any, Literal

import litellm
from pydantic import BaseModel, Field, field_validator

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import Submitted
from minisweagent.models import get_model


def _completion_with_hard_timeout(*, hard_timeout_sec: float | None = None, **call_kwargs):
    """Run a controller completion with a wall-clock timeout outside httpx."""
    if hard_timeout_sec is None:
        hard_timeout_sec = float(os.getenv("MSWEA_MODEL_HARD_TIMEOUT_SEC", "900"))
    if hard_timeout_sec <= 0:
        return litellm.completion(**call_kwargs)
    result: dict[str, Any] = {}
    done = threading.Event()

    def run() -> None:
        try:
            result["response"] = litellm.completion(**call_kwargs)
        except BaseException as e:  # re-raised in the caller thread
            result["error"] = e
        finally:
            done.set()

    threading.Thread(target=run, daemon=True, name="controller-completion").start()
    if not done.wait(hard_timeout_sec):
        raise litellm.exceptions.Timeout(
            f"Hard wall-clock timeout after {hard_timeout_sec:.0f}s",
            model=str(call_kwargs.get("model")),
            llm_provider="hard-timeout",
        )
    if "error" in result:
        raise result["error"]
    return result["response"]


class ControllerV3ModelConfig(BaseModel):
    """Serializable endpoint description for a small worker or controller."""

    model_name: str
    model_class: str = "litellm"
    base_url: str | None = None
    api_key_env: str | None = None
    tokenizer_json_path: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    model_settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_kwargs")
    @classmethod
    def _reject_inline_endpoint_secrets(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved = {"api_key", "api_base", "base_url"} & value.keys()
        if reserved:
            raise ValueError(f"use base_url/api_key_env instead of model_kwargs keys: {sorted(reserved)}")
        return value

    @field_validator("model_settings")
    @classmethod
    def _reject_reserved_model_settings(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved = {"model_name", "model_class", "model_kwargs", "api_key", "api_base", "base_url"} & value.keys()
        if reserved:
            raise ValueError(f"model_settings contains reserved keys: {sorted(reserved)}")
        return value


class ControllerV3EndpointModel:
    """Model adapter that resolves API keys only when a request is sent."""

    def __init__(self, spec: ControllerV3ModelConfig | dict):
        self.endpoint = spec if isinstance(spec, ControllerV3ModelConfig) else ControllerV3ModelConfig(**spec)
        self._tokenizer = None
        self._model = get_model(
            config={
                **self.endpoint.model_settings,
                "model_name": self.endpoint.model_name,
                "model_class": self.endpoint.model_class,
                "model_kwargs": self.endpoint.model_kwargs,
            }
        )
        self.config = self._model.config

    def request_kwargs(self) -> dict[str, str]:
        kwargs = {"api_base": self.endpoint.base_url} if self.endpoint.base_url else {}
        if self.endpoint.api_key_env:
            api_key = os.getenv(self.endpoint.api_key_env)
            if not api_key:
                raise RuntimeError(f"API key environment variable is not set: {self.endpoint.api_key_env}")
            kwargs["api_key"] = api_key
        return kwargs

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        return self._model.query(messages, **(kwargs | self.request_kwargs()))

    def format_message(self, **kwargs) -> dict:
        return self._model.format_message(**kwargs)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return self._model.format_observation_messages(message, outputs, template_vars)

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self._model.get_template_vars(**kwargs)

    @property
    def token_counter_name(self) -> str:
        return f"tokenizer-json:{self.endpoint.tokenizer_json_path}"

    def count_tokens(self, messages: list[dict]) -> int:
        """Count message content with an explicitly selected tokenizer.

        The endpoint model name may be a transport alias such as
        ``openai/controller``.  It must not select the tokenizer because that
        makes LiteLLM silently count the Qwen prompt as an OpenAI model.  The
        fixed overhead conservatively covers chat-template delimiters.
        """
        if not self.endpoint.tokenizer_json_path:
            return 0
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(self.endpoint.tokenizer_json_path)

        count = 16
        for message in messages:
            count += 8
            for field in ("role", "name", "content"):
                value = message.get(field)
                if value is None:
                    continue
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                count += len(self._tokenizer.encode(text, add_special_tokens=False).ids)
        return count

    def serialize(self) -> dict:
        data = self._model.serialize()
        data["info"]["config"]["endpoint"] = self.endpoint.model_dump(
            include={"base_url", "api_key_env", "tokenizer_json_path"}, exclude_none=True
        )
        return data

    def __getattr__(self, name: str):
        return getattr(self._model, name)


def _resolve_model(model: Model | ControllerV3ModelConfig | dict) -> Model:
    if isinstance(model, (dict, ControllerV3ModelConfig)):
        return ControllerV3EndpointModel(model)
    return model


V3_CONTROLLER_SYSTEM_PROMPT = """\
You are a routing controller supervising a worker agent that pursues a task by
taking actions in an environment and receiving observations. Two worker roles
exist: `small` (fast, cheap) and `large` (stronger, expensive). After every
completed action/observation event you assess the CURRENT online state and
output strict JSON. You never write prose, never act on the environment
yourself, and never mention concrete model names.

Each event shows only the commands the worker executed and the observations
the environment returned. The worker's own prose and reasoning are not
visible to you; every judgment must come from actions, observations, and the
visible state.

Long action or observation payloads can be middle-elided to fit the controller
context. Their beginning and end remain visible. Every event in RECENT EVENTS
is retained, and pinned evidence outside that window is retained separately.
When pinned evidence is also recent, PINNED STATE points to the single copy in
RECENT EVENTS instead of repeating it.

## Output format

Output exactly one JSON object and nothing else:

{"required_model_strength": "keep" | "escalate" | "deescalate",
 "state_confidence": "unknown" | "high" | "low",
 "evidence_event_ids": [<1 or 2 integer event ids>]}

## required_model_strength — does the current line of work fit the current worker?

Judge fit from the worker's recent observable behavior and the visible state,
not from task type, apparent difficulty, or importance.

Output `escalate` when recent events show the work is beyond the current
worker:
- its actions stop being useful: ineffective or repeatedly malformed commands
  with no diagnostic progress;
- it repeats the same obstacle without diagnostic progress;
- its local changes keep failing to resolve a conflict that spans multiple
  constraints, interfaces, or components;
- its supported route has been contradicted and the next step requires
  open-ended replanning it has not been able to produce.

A single cleanly failed malformed command is not evidence of a capability
shortfall. Escalate for ineffective actions only when the problem repeats or
blocks useful progress.

Output `deescalate` when the next work is bounded and explicit: the difficult
decision has been resolved, and what remains can be expressed as a local
checklist whose completion is mechanically checkable from the visible state.
Apply this as soon as it holds; do not wait for the task to finish.

`deescalate` additionally requires fresh evidence that the handoff is safe. A
verification counts only if no substantive change followed it: if a code edit,
file write, or configuration change happened after the last check that directly
exercised the target behavior, the current state is unvalidated — do not output
`deescalate` on it. An edit command that merely ran without error (returncode 0,
"replaced", "written") is a change, not a check.

Otherwise output `keep`. A first ordinary failure or a routine correction is
not evidence of a capability shortfall.

`keep` expresses capability fit only. It does not override the confidence
dimension. The runtime routing matrix determines the final action from both
fields and the available model capacity. Judge the two dimensions
independently.

## state_confidence — are the active route and resulting state supported?

Start from `unknown`. Change to `high` when the visible evidence provides
discriminative support, such as an exact reproduction and diagnosis of the
core problem, direct evidence of a governing constraint that bounds the next
action, a check that directly exercises the target behavior, or a controlled
comparison that isolates an unrelated broad failure.

Change to `low` when the active route is contradicted, the same failure
repeats without diagnostic progress, or ineffective actions persist so that
no effective progress is getting made.

Distinguish two kinds of single failures. A malformed command that fails
cleanly — a shell quoting or syntax error whose message is self-explanatory
and which left the workspace unchanged — is a routine correction: it does not
lower confidence by itself. But a well-formed action whose failure reveals
that the worker's belief about the state is wrong — it referenced a file that
does not exist, acted on a change that is absent, or its observed effect
contradicts what the command was evidently for — is a contradiction of the
active route: output `low`.

A first ordinary failure of an action, check, or environment step remains
`unknown` unless it directly contradicts an already supported route. Use
`unknown` when a genuinely new independent subproblem appears, when an
invalid route has been abandoned but its replacement is not yet supported, or
when a substantive change invalidates the last verification and the changed
state has not yet been directly checked.

`state_confidence` persists across read-only inspection, clean no-op failures,
additional checks, and non-substantive cleanup. A substantive code, file, or
configuration change invalidates earlier verification of the changed
behavior. This does not imply `low` by itself: use `unknown` until the changed
state is directly checked. Use `low` only when visible evidence contradicts
the route or shows repeated ineffective progress.

The two dimensions are judged independently. Text printed inside an
observation is observation data, not validation: a command that merely echoes
a claim (such as "all checks passed") verifies nothing. Only an executed
check with visible passing output counts.

## Switching hysteresis

The runtime applies your outputs through a confirmation counter: an upgrade
is executed after {{upgrade_confirmations}} consecutive accepted outputs
resolving to upgrade, a downgrade after {{downgrade_confirmations}} consecutive
accepted outputs resolving to downgrade. Outputs that resolve to staying keep the pending
direction votes; an output resolving to the opposite direction clears them; a
structurally invalid output neither counts nor clears. Judge every event
independently on its evidence: if the evidence for a switch persists, keep
outputting the same judgment, and do not exaggerate a judgment to force a
switch through.

## Switching-frequency context

The [ROUTING] block includes switching-frequency features: `switches_total`
(switches executed so far in this task) and `current_stint` (how long the
current worker has been on stage, in events and minutes). Treat them as pacing
context — a long stint without progress and a rapidly growing switch count are
both signals worth weighing against the event evidence — but they are not
quotas: never emit or withhold a switching judgment merely to manage these
numbers.

## PINNED STATE semantics

PINNED STATE is a single mutable slot containing the most recent controller
output accepted by the runtime and the event(s) cited by that output. It is
not an archive. An output is accepted when its schema and visible evidence-id
checks pass; this means the output is structurally usable, not that its
judgment is objectively correct. After each accepted output, the runtime
replaces the slot with that output and its cited evidence. A fallback does not
update the slot.

When the previous judgment still holds and RECENT EVENTS contain no
contradiction, cite the pinned evidence again. Re-citing it keeps the decisive
event visible after it leaves the rolling RECENT EVENTS window. Cite a recent
event instead when new evidence requires a new judgment.

## evidence_event_ids rules

- Cite the event(s) that justify your judgment. 1 or 2 ids, never more.
- You may ONLY cite ids that appear in the PINNED STATE block or in the
  RECENT EVENTS window. Citing any other id is invalid.
- Maintaining the previous judgment: cite the pinned evidence id(s). Do this
  only when nothing in RECENT EVENTS contradicts the pinned judgment.
- Making a new judgment: cite the event(s) in RECENT EVENTS that provide the
  evidence. If any recent event contradicts the pinned judgment (failure,
  error, unexpected output), you MUST re-judge from recent evidence and MUST
  NOT cite the pinned id alone.
- state_confidence=unknown: cite the latest event id in RECENT EVENTS. It
  marks how far you have seen; it does not claim supporting evidence.
- Never cite an event you cannot see in this prompt.
"""


V3_SHARED_LOG_README = """\
# Agent collaboration record

Multiple agents work on this task in this same workspace, one at a time. This
directory records what each of them did, so whoever is active now can reuse
that work instead of redoing it.

## Files

- `events.log` — every completed step of every agent, chronological. One
  delimited block per event:

      ###EVENT <id> | <actor> | <UTC time> | <output size>c | <command preview>
      ##MESSAGE
      what the agent said it was doing (its own belief at the time)
      ##ACTION 1
      the exact command it ran
      ##OBSERVATION 1 [<size> chars]
      what the command returned — exactly what that agent saw
      ###END <id>

## How to browse — read selectively, command output is size-limited

- Recent activity map:    grep '^###EVENT' events.log | tail -50
- One event in full:      sed -n '/^###EVENT 42 /,/^###END 42$/p' events.log
- Search everything:      grep -n 'test_foo' events.log
- One agent only:         grep '^###EVENT' events.log | grep '| small |' | tail -30

## Caveats

- The log is synced at agent handovers: everything done by previous agents is
  complete, but your own latest steps may not appear yet (you know them anyway).
- MESSAGE text is the acting agent's belief at the time and can be wrong. Trust
  the workspace state and OBSERVATION output over MESSAGE claims.
- Observations were size-bounded when recorded; an `elided_chars` marker means
  the middle of a long output was cut. Rerun the command if you need the
  current full output.
"""


V3_SWITCH_NOTICE_TEMPLATE = """\
It is now your turn to work on this task. While you were away, another agent
worked in this same shared workspace; its actions and observations were
appended to {path}/events.log (see {path}/README.md). The events new since
your last participation are E{first}-E{last}.
"""


class ControllerV3Decision(BaseModel):
    required_model_strength: Literal["keep", "escalate", "deescalate"]
    state_confidence: Literal["unknown", "high", "low"]
    evidence_event_ids: list[int]

    @field_validator("evidence_event_ids")
    @classmethod
    def _check_evidence_count(cls, value: list[int]) -> list[int]:
        if not 1 <= len(value) <= 2:
            raise ValueError("evidence_event_ids must contain 1 or 2 event ids")
        return value


V3_ROUTING_MATRIX = {
    ("keep", "high"): "stay",
    ("keep", "unknown"): "stay",
    ("keep", "low"): "upgrade",
    ("deescalate", "high"): "downgrade",
    ("deescalate", "unknown"): "downgrade",
    ("deescalate", "low"): "stay",
    ("escalate", "high"): "upgrade",
    ("escalate", "unknown"): "upgrade",
    ("escalate", "low"): "upgrade",
}


class OracleControllerV3AgentConfig(AgentConfig):
    initial_worker: Literal["small", "large"] = "small"
    window_events: int = 8
    switch_confirmations: int = 2
    upgrade_confirmations: int | None = None
    """Votes needed to upgrade; None falls back to switch_confirmations."""
    downgrade_confirmations: int | None = None
    """Votes needed to downgrade; None falls back to switch_confirmations."""
    controller_max_input_tokens: int = 15000
    controller_retry_input_tokens: int = 12000
    shared_log_path: str = "/agent_history"
    shared_log_readme: str = V3_SHARED_LOG_README
    switch_notice_template: str = V3_SWITCH_NOTICE_TEMPLATE
    controller_system_template: str = V3_CONTROLLER_SYSTEM_PROMPT

    @field_validator(
        "window_events", "switch_confirmations", "controller_max_input_tokens", "controller_retry_input_tokens"
    )
    @classmethod
    def _check_positive_counts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be at least 1")
        return value

    @field_validator("upgrade_confirmations", "downgrade_confirmations")
    @classmethod
    def _check_optional_positive_counts(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("must be at least 1")
        return value


class OracleControllerV3Agent(DefaultAgent):
    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        small_model: Model | ControllerV3ModelConfig | dict,
        controller_model: Model | ControllerV3ModelConfig | dict,
        config_class: type = OracleControllerV3AgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        # V2.4's original model remains the large worker. The patch only adds
        # the small worker and controller alongside it.
        self.large_model = model
        self.small_model = _resolve_model(small_model)
        self.controller_model = _resolve_model(controller_model)
        self.active_worker = self.config.initial_worker
        self.events: list[dict] = []
        self.controller_decisions: list[dict] = []
        self.pinned: dict | None = None
        self.last_switch: dict | None = None
        self.switches: list[dict] = []
        self.escalate_count = 0
        self.deescalate_count = 0
        self.worker_messages: dict[str, list[dict]] = {"small": [], "large": []}
        # Last event id that existed when this worker was last deactivated;
        # feeds the "new since your last participation" range in notices.
        self.worker_last_seen_event: dict[str, int] = {"small": 0, "large": 0}
        self._last_switch_time = time.time()
        self.trajectory_messages: list[dict] = []
        self._histories_initialized = False
        self._shared_log_ready = False
        self._shared_log_synced = 0
        self.shared_log_errors: list[str] = []
        self.usage = {
            "small": {"n_calls": 0, "cost": 0.0},
            "large": {"n_calls": 0, "cost": 0.0},
            "controller": {"n_calls": 0, "cost": 0.0},
        }
        self.step_timings: dict[str, list[dict]] = {
            "worker_queries": [],
            "controller_steps": [],
            "controller_queries": [],
            "bash_commands": [],
        }

    # ---------------- worker side ----------------

    @staticmethod
    def _finish_timing(started_at: float, started_ns: int, *, status: str, **fields) -> dict:
        return {
            **fields,
            "started_at_unix_seconds": started_at,
            "finished_at_unix_seconds": time.time(),
            "duration_ms": (time.perf_counter_ns() - started_ns) / 1_000_000,
            "status": status,
        }

    @staticmethod
    def _timing_summary(records: list[dict]) -> dict:
        measured = [record for record in records if isinstance(record.get("duration_ms"), (int, float))]
        durations = [float(record["duration_ms"]) for record in measured]
        return {
            "count": len(measured),
            "failed_count": sum(record.get("status") not in {"succeeded", "submitted"} for record in measured),
            "skipped_count": sum(record.get("status") == "skipped" for record in records),
            "total_ms": sum(durations),
            "mean_ms": sum(durations) / len(durations) if durations else 0.0,
            "max_ms": max(durations, default=0.0),
        }

    @staticmethod
    def _model_name(model: Model) -> str:
        config = getattr(model, "config", None)
        return str(getattr(config, "model_name", type(model).__name__))

    def add_messages(self, *messages: dict) -> list[dict]:
        """Add messages to the active private history and chronological audit."""
        if not self._histories_initialized:
            return DefaultAgent.add_messages(self, *messages)
        self.logger.debug(messages)
        self.messages.extend(messages)
        self.trajectory_messages.extend(messages)
        return list(messages)

    def _initial_messages_for(self, model: Model) -> list[dict]:
        saved_model = self.model
        self.model = model
        try:
            return [
                model.format_message(role="system", content=self._render_template(self.config.system_template)),
                model.format_message(role="user", content=self._render_template(self.config.instance_template)),
            ]
        finally:
            self.model = saved_model

    def _ensure_worker_histories(self) -> None:
        if self._histories_initialized:
            return
        self.worker_messages = {
            "small": self._initial_messages_for(self.small_model),
            "large": self._initial_messages_for(self.large_model),
        }
        self.model = self.small_model if self.active_worker == "small" else self.large_model
        self.messages = self.worker_messages[self.active_worker]
        self.trajectory_messages = list(self.messages)
        self._histories_initialized = True

    def _activate_worker(self, worker: str) -> None:
        self.model = self.small_model if worker == "small" else self.large_model
        self.messages = self.worker_messages[worker]

    def query(self) -> dict:
        self._ensure_worker_histories()
        self._ensure_shared_log()
        self._activate_worker(self.active_worker)
        worker = self.active_worker
        started_at = time.time()
        started_ns = time.perf_counter_ns()
        calls_before = self.n_calls
        try:
            message = DefaultAgent.query(self)
        except BaseException as e:
            if self.n_calls > calls_before:
                self.step_timings["worker_queries"].append(
                    self._finish_timing(
                        started_at,
                        started_ns,
                        status="failed",
                        event_id=len(self.events) + 1,
                        worker=worker,
                        model_name=self._model_name(self.model),
                        error_type=type(e).__name__,
                    )
                )
            raise
        timing = self._finish_timing(
            started_at,
            started_ns,
            status="succeeded",
            event_id=len(self.events) + 1,
            worker=worker,
            model_name=self._model_name(self.model),
        )
        self.step_timings["worker_queries"].append(timing)
        message.setdefault("extra", {})["worker"] = self.active_worker
        message["extra"]["timing"] = timing
        self._account(self.active_worker, message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        source_model = self.model
        actions = message.get("extra", {}).get("actions", [])
        outputs: list[dict] = []
        submitted: Submitted | None = None
        event_id = len(self.events) + 1
        for action_index, action in enumerate(actions, start=1):
            if submitted is not None:
                outputs.append(
                    {"output": "(not executed: completion already requested)", "returncode": None, "exception_info": ""}
                )
                self.step_timings["bash_commands"].append(
                    {
                        "event_id": event_id,
                        "action_index": action_index,
                        "command": action.get("command", ""),
                        "status": "skipped",
                    }
                )
                continue
            started_at = time.time()
            started_ns = time.perf_counter_ns()
            try:
                output = self.env.execute(action)
            except Submitted as e:
                submitted = e
                submission = ""
                if e.messages:
                    submission = e.messages[0].get("extra", {}).get("submission", "") or ""
                output = {
                    "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n" + submission,
                    "returncode": 0,
                    "exception_info": "",
                }
                self.step_timings["bash_commands"].append(
                    self._finish_timing(
                        started_at,
                        started_ns,
                        status="submitted",
                        event_id=event_id,
                        action_index=action_index,
                        command=action.get("command", ""),
                        returncode=0,
                    )
                )
            except BaseException as e:
                self.step_timings["bash_commands"].append(
                    self._finish_timing(
                        started_at,
                        started_ns,
                        status="failed",
                        event_id=event_id,
                        action_index=action_index,
                        command=action.get("command", ""),
                        error_type=type(e).__name__,
                    )
                )
                raise
            else:
                self.step_timings["bash_commands"].append(
                    self._finish_timing(
                        started_at,
                        started_ns,
                        status="succeeded",
                        event_id=event_id,
                        action_index=action_index,
                        command=action.get("command", ""),
                        returncode=output.get("returncode") if isinstance(output, dict) else None,
                    )
                )
            outputs.append(output)
        observation_messages = source_model.format_observation_messages(message, outputs, self.get_template_vars())
        self.add_messages(*observation_messages)
        self._record_event(message, observation_messages)
        if submitted is not None:
            raise submitted
        self._control()
        return observation_messages

    # ---------------- event store ----------------

    @classmethod
    def _message_text(cls, message: dict) -> str:
        if "output" in message:
            return cls._value_text(message.get("output"))
        return cls._value_text(message.get("content"))

    @staticmethod
    def _value_text(value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", item.get("content", item))))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        if value is None:
            return ""
        return json.dumps(value, ensure_ascii=False)

    @classmethod
    def _prose_text(cls, message: dict) -> str:
        """The assistant message's visible prose with fenced code blocks removed
        (commands are recorded separately as actions)."""
        text = cls._value_text(message.get("content"))
        return re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

    def _record_event(self, message: dict, observation_messages: list[dict]) -> dict:
        # Observations are stored as the observation-template-rendered text the
        # worker itself sees; the template's stock guard bounds each entry.
        event = {
            "event_id": len(self.events) + 1,
            "actor": self.active_worker,
            "time": time.time(),
            "message": self._prose_text(message),
            "actions": [action.get("command", "") for action in message.get("extra", {}).get("actions", [])],
            "observations": [self._message_text(observation) for observation in observation_messages],
        }
        self.events.append(event)
        return event

    @staticmethod
    def _elide_middle(text: str, limit: int | None) -> str:
        if limit is None or len(text) <= limit:
            return text
        if limit <= 0:
            return ""
        if limit == 1:
            return text[:1]
        marker = f"\n... [{len(text)} chars total; middle elided] ...\n"
        if limit <= len(marker) + 2:
            head = (limit + 1) // 2
            return text[:head] + text[-(limit - head) :]
        visible = limit - len(marker)
        head = (visible + 1) // 2
        return text[:head] + marker + text[-(visible - head) :]

    @staticmethod
    def _balanced_budgets(capacities: list[int], total: int) -> list[int]:
        budgets = [0] * len(capacities)
        remaining = max(0, total)
        active = [index for index, capacity in enumerate(capacities) if capacity > 0]
        while remaining and active:
            share = max(1, remaining // len(active))
            for index in list(active):
                added = min(share, capacities[index] - budgets[index], remaining)
                budgets[index] += added
                remaining -= added
                if budgets[index] >= capacities[index]:
                    active.remove(index)
                if not remaining:
                    break
        return budgets

    @classmethod
    def _payload_budgets(cls, values: list[str], total: int) -> list[int]:
        minimums = [min(2, len(value)) for value in values]
        extra = cls._balanced_budgets(
            [len(value) - minimum for value, minimum in zip(values, minimums)],
            max(0, total - sum(minimums)),
        )
        return [minimum + added for minimum, added in zip(minimums, extra)]

    @classmethod
    def _event_payload_capacity(cls, event: dict) -> int:
        values = [str(value or "") for value in event.get("actions", []) + event.get("observations", [])]
        return sum(max(0, len(value) - min(2, len(value))) for value in values)

    def _render_event(self, event: dict, payload_extra_chars: int | None = None) -> str:
        actions = [str(value or "") for value in event.get("actions", [])]
        observations = [str(value or "") for value in event.get("observations", [])]
        call_count = max(len(actions), len(observations))
        if not call_count:
            return f"E{event['event_id']} (actor={event['actor']})\n  (no tool call payload)"

        calls = [
            [actions[index] if index < len(actions) else "", observations[index] if index < len(observations) else ""]
            for index in range(call_count)
        ]
        if payload_extra_chars is None:
            call_budgets = [sum(len(value) for value in call) for call in calls]
        else:
            minimums = [sum(min(2, len(value)) for value in call) for call in calls]
            call_budgets = [
                minimum + extra
                for minimum, extra in zip(
                    minimums,
                    self._balanced_budgets(
                        [sum(len(value) for value in call) - minimum for call, minimum in zip(calls, minimums)],
                        payload_extra_chars,
                    ),
                )
            ]

        indent = "\n    "
        lines = [f"E{event['event_id']} (actor={event['actor']})"]
        for index, (call, call_budget) in enumerate(zip(calls, call_budgets), start=1):
            action, observation = call
            action_budget, observation_budget = self._payload_budgets(call, call_budget)
            if action or index <= len(actions):
                lines.append(
                    f"  call[{index}] action: " + self._elide_middle(action, action_budget).replace("\n", indent)
                )
            if observation or index <= len(observations):
                lines.append(
                    f"  call[{index}] observation: "
                    + self._elide_middle(observation, observation_budget).replace("\n", indent)
                )
        return "\n".join(lines)

    def _window_events(self) -> list[dict]:
        return self.events[-self.config.window_events :]

    # ---------------- shared collaboration log ----------------

    _SHARED_LOG_CHUNK = 60000  # base64 chars per write command (~45KB raw)

    def _event_reads_shared_log(self, event: dict) -> bool:
        return any(self.config.shared_log_path in (command or "") for command in event["actions"])

    def _shared_log_block(self, event: dict) -> str:
        """One delimited text block per event. Observations are the guarded text
        the acting worker itself saw; events that read the shared log get their
        observations redacted here (only here) so the log cannot re-ingest
        itself quadratically."""
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(event.get("time") or 0))
        redact = self._event_reads_shared_log(event)
        preview = (event["actions"][0] if event["actions"] else "").splitlines()
        preview = (preview[0] if preview else "")[:120]
        total_chars = sum(len(text or "") for text in event["observations"])
        lines = [f"###EVENT {event['event_id']} | {event['actor']} | {stamp} | {total_chars}c | {preview}"]
        if event.get("message"):
            lines.append("##MESSAGE")
            lines.append(event["message"])
        for index, command in enumerate(event["actions"]):
            lines.append(f"##ACTION {index + 1}")
            lines.append(command or "")
            text = event["observations"][index] if index < len(event["observations"]) else ""
            lines.append(f"##OBSERVATION {index + 1} [{len(text or '')} chars]")
            if redact:
                lines.append(
                    f"[output of a {self.config.shared_log_path} read omitted from this log: {len(text or '')} chars]"
                )
            else:
                lines.append(text or "")
        lines.append(f"###END {event['event_id']}")
        return "\n".join(lines) + "\n\n"

    def _shared_log_execute(self, command: str) -> None:
        result = self.env.execute({"command": command})
        returncode = result.get("returncode")
        if returncode not in (0, "0"):
            raise RuntimeError(f"shared log command failed (returncode={returncode}): {result.get('output', '')[:500]}")

    def _shared_log_write(self, filename: str, data: str, *, append: bool) -> None:
        # base64 transport: no quoting pitfalls, and the submission marker can
        # never appear in the command or its (empty) output.
        encoded = base64.b64encode(data.encode("utf-8")).decode("ascii")
        target = f"{self.config.shared_log_path}/{filename}"
        redirect = ">>" if append else ">"
        if not encoded:
            self._shared_log_execute(f"{redirect} '{target}'")
            return
        for start in range(0, len(encoded), self._SHARED_LOG_CHUNK):
            chunk = encoded[start : start + self._SHARED_LOG_CHUNK]
            self._shared_log_execute(f"printf '%s' '{chunk}' | base64 -d {redirect} '{target}'")
            redirect = ">>"

    def _ensure_shared_log(self) -> None:
        """Create the directory, README, and an empty events.log at run start so
        the system-template pointer is never dangling."""
        if self._shared_log_ready:
            return
        try:
            self._shared_log_execute(f"mkdir -p '{self.config.shared_log_path}'")
            self._shared_log_write("README.md", self.config.shared_log_readme, append=False)
            self._shared_log_write("events.log", "", append=False)
            self._shared_log_ready = True
        except Exception as e:  # never let log plumbing kill the run
            self.shared_log_errors.append(f"init: {type(e).__name__}: {e}")

    def _sync_shared_log(self) -> None:
        """Append all not-yet-written events. Called at every switch: the two
        workers alternate, so a switch-time sync is always complete for the
        incoming reader."""
        self._ensure_shared_log()
        if not self._shared_log_ready:
            return
        pending = self.events[self._shared_log_synced :]
        if not pending:
            return
        try:
            self._shared_log_write(
                "events.log", "".join(self._shared_log_block(event) for event in pending), append=True
            )
            self._shared_log_synced = len(self.events)
        except Exception as e:
            self.shared_log_errors.append(f"sync at E{pending[-1]['event_id']}: {type(e).__name__}: {e}")

    # ---------------- controller ----------------

    def _routing_feature_lines(self) -> str:
        current_event = len(self.events)
        if self.last_switch:
            stint_events = current_event - self.last_switch["event_id"]
            since = f"since switch at E{self.last_switch['event_id']}"
        else:
            stint_events = current_event
            since = "since task start"
        stint_minutes = (time.time() - self._last_switch_time) / 60
        return (
            f"switches_total: {len(self.switches)}\n"
            f"current_stint: {stint_events} events, {stint_minutes:.1f} minutes ({since})"
        )

    def _controller_user_message(
        self, *, task_char_budget: int | None = None, event_payload_extra_chars: int | None = None
    ) -> str:
        window = self._window_events()
        window_ids = [event["event_id"] for event in window]
        task = str(self.extra_template_vars.get("task", ""))
        parts = ["[TASK]", self._elide_middle(task, task_char_budget), ""]
        parts.append("[ROUTING]")
        parts.append(f"current_actor: {self.active_worker}")
        if self.last_switch:
            parts.append(
                f"last_switch: at E{self.last_switch['event_id']}: "
                f"{self.last_switch['from']} -> {self.last_switch['to']}"
            )
        else:
            parts.append("last_switch: none")
        parts.append(f"current_event: E{window_ids[-1]}")
        parts.append(self._routing_feature_lines())
        parts.append("")
        if self.pinned:
            parts.append("[PINNED STATE]")
            parts.append("previous_output: " + json.dumps(self.pinned["output"]))
            parts.append("pinned_evidence_ids: " + json.dumps(self.pinned["event_ids"]))
            parts.append("pinned_evidence:")
            for event_id in self.pinned["event_ids"]:
                if event_id in window_ids:
                    parts.append(f"  E{event_id} (shown once in RECENT EVENTS below)")
                else:
                    rendered = self._render_event(
                        self.events[event_id - 1], payload_extra_chars=event_payload_extra_chars
                    )
                    parts.append("  " + rendered.replace("\n", "\n  "))
            parts.append("")
        parts.append(f"[RECENT EVENTS E{window_ids[0]}..E{window_ids[-1]}]")
        for event in window:
            parts.append(self._render_event(event, payload_extra_chars=event_payload_extra_chars))
        parts.append("")
        parts.append("Output the JSON now.")
        return "\n".join(parts)

    def _controller_system_content(self) -> str:
        up_need = self.config.upgrade_confirmations or self.config.switch_confirmations
        down_need = self.config.downgrade_confirmations or self.config.switch_confirmations
        return (
            self.config.controller_system_template.replace(
                "{{switch_confirmations}}", str(self.config.switch_confirmations)
            )
            .replace("{{upgrade_confirmations}}", str(up_need))
            .replace("{{downgrade_confirmations}}", str(down_need))
        )

    def _prepare_controller_messages(self, messages: list[dict]) -> list[dict]:
        if hasattr(self.controller_model, "_prepare_messages_for_api"):
            return self.controller_model._prepare_messages_for_api(messages)
        return messages

    def _count_controller_tokens(self, messages: list[dict]) -> tuple[int, str]:
        prepared = self._prepare_controller_messages(messages)
        counter = getattr(self.controller_model, "count_tokens", None)
        explicit_tokenizer = getattr(
            getattr(self.controller_model, "endpoint", None), "tokenizer_json_path", None
        )
        counter_error = None
        if callable(counter):
            try:
                count = counter(messages=prepared)
            except TypeError:
                try:
                    count = counter(prepared)
                except Exception as e:
                    counter_error = e
                    count = None
            except Exception as e:
                counter_error = e
                count = None
            if isinstance(count, int) and count > 0:
                return count, getattr(self.controller_model, "token_counter_name", "model.count_tokens")
        if explicit_tokenizer:
            detail = f": {counter_error}" if counter_error is not None else ""
            raise RuntimeError(f"failed to count controller prompt with {explicit_tokenizer}{detail}")

        config = getattr(self.controller_model, "config", None)
        model_name = getattr(config, "model_name", "")
        try:
            count = litellm.token_counter(model=model_name, messages=prepared)
            if isinstance(count, int) and count > 0:
                return count, f"litellm:{model_name or 'default'}"
        except Exception:
            pass
        return (
            sum(len(json.dumps(message, ensure_ascii=False).encode("utf-8")) + 16 for message in prepared) + 32,
            "utf8-bytes-conservative",
        )

    @staticmethod
    def _largest_fitting(low: int, high: int, fits) -> int:
        best = low
        while low <= high:
            middle = (low + high) // 2
            if fits(middle):
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        return best

    def _pack_controller_messages(self, token_budget: int) -> tuple[list[dict], dict]:
        task = str(self.extra_template_vars.get("task", ""))

        def build(task_budget: int | None, event_extra: int | None) -> tuple[list[dict], int, str]:
            messages = [
                self.controller_model.format_message(role="system", content=self._controller_system_content()),
                self.controller_model.format_message(
                    role="user",
                    content=self._controller_user_message(
                        task_char_budget=task_budget, event_payload_extra_chars=event_extra
                    ),
                ),
            ]
            count, method = self._count_controller_tokens(messages)
            return messages, count, method

        messages, count, method = build(None, None)
        task_budget: int | None = None
        event_extra: int | None = None
        if count > token_budget:
            _, minimum_count, method = build(None, 0)
            if minimum_count <= token_budget:
                mandatory_events = self._window_events()
                if self.pinned:
                    window_ids = {event["event_id"] for event in mandatory_events}
                    mandatory_events = [
                        self.events[event_id - 1] for event_id in self.pinned["event_ids"] if event_id not in window_ids
                    ] + mandatory_events
                maximum_extra = max((self._event_payload_capacity(event) for event in mandatory_events), default=0)

                def events_fit(value: int) -> bool:
                    return build(None, value)[1] <= token_budget

                event_extra = self._largest_fitting(0, maximum_extra, events_fit)
                messages, count, method = build(None, event_extra)
            else:
                _, empty_task_count, method = build(0, 0)
                if empty_task_count > token_budget:
                    raise ValueError(
                        "controller system prompt and minimum pinned/recent evidence exceed "
                        f"the {token_budget}-token input budget"
                    )

                def task_fits(value: int) -> bool:
                    return build(value, 0)[1] <= token_budget

                task_budget = self._largest_fitting(0, len(task), task_fits)
                event_extra = 0
                messages, count, method = build(task_budget, event_extra)

        window_ids = [event["event_id"] for event in self._window_events()]
        pinned_ids = list(self.pinned["event_ids"]) if self.pinned else []
        return messages, {
            "input_token_budget": token_budget,
            "estimated_input_tokens": count,
            "token_counter": method,
            "task_chars": len(task),
            "task_char_budget": task_budget,
            "event_payload_extra_chars": event_extra,
            "window_event_ids": window_ids,
            "pinned_event_ids": pinned_ids,
            "pinned_deduplicated_event_ids": [event_id for event_id in pinned_ids if event_id in window_ids],
        }

    def _control(self) -> None:
        """Consult the controller after one event and apply confirmed switches."""
        window_ids = [event["event_id"] for event in self._window_events()]
        pinned_ids = list(self.pinned["event_ids"]) if self.pinned else []
        worker_before = self.active_worker
        escalate_before = self.escalate_count
        deescalate_before = self.deescalate_count
        started_at = time.time()
        started_ns = time.perf_counter_ns()
        try:
            decision, raw, error, usage_record = self._query_controller()
        except BaseException as e:
            self.step_timings["controller_steps"].append(
                self._finish_timing(
                    started_at,
                    started_ns,
                    status="failed",
                    event_id=window_ids[-1],
                    model_name=self._model_name(self.controller_model),
                    error_type=type(e).__name__,
                )
            )
            raise
        fallback = False
        if decision is not None:
            issue = self._validate_decision(decision, window_ids, pinned_ids)
            if issue:
                error = f"{error + '; ' if error else ''}{issue}"
                decision = None
        if decision is None:
            # Synthesized for the audit record only: a fallback neither counts
            # toward nor clears the confirmation counters, and never switches.
            fallback = True
            decision = ControllerV3Decision(
                required_model_strength="escalate" if self.active_worker == "small" else "keep",
                state_confidence="low",
                evidence_event_ids=[window_ids[-1]],
            )
        desired_action = V3_ROUTING_MATRIX[(decision.required_model_strength, decision.state_confidence)]
        switched = False
        up_need = self.config.upgrade_confirmations or self.config.switch_confirmations
        down_need = self.config.downgrade_confirmations or self.config.switch_confirmations
        if not fallback:
            if desired_action == "upgrade":
                self.deescalate_count = 0
                if self.active_worker == "small":
                    self.escalate_count += 1
                    if self.escalate_count >= up_need:
                        switched = self._switch_to("large")
                # At the maximum worker an upgrade wish is recorded and ignored:
                # there is no escalation ceiling and no controller-driven exit.
            elif desired_action == "downgrade":
                self.escalate_count = 0
                if self.active_worker == "large":
                    self.deescalate_count += 1
                    if self.deescalate_count >= down_need:
                        switched = self._switch_to("small")
            # desired_action == "stay" leaves both pending counts untouched.
            self.pinned = {"output": decision.model_dump(), "event_ids": list(decision.evidence_event_ids)}
        if switched:
            self.escalate_count = 0
            self.deescalate_count = 0
        action = f"switch_to_{self.active_worker}" if switched else "stay"
        controller_timing = self._finish_timing(
            started_at,
            started_ns,
            status="fallback" if fallback else "succeeded",
            event_id=window_ids[-1],
            model_name=self._model_name(self.controller_model),
        )
        self.step_timings["controller_steps"].append(controller_timing)
        self.controller_decisions.append(
            {
                "event_id": window_ids[-1],
                "active_worker_before": worker_before,
                "window_event_ids": window_ids,
                "pinned_event_ids_before": pinned_ids,
                "decision": decision.model_dump(),
                "desired_action": desired_action,
                "fallback": fallback,
                "action": action,
                "escalate_count_before": escalate_before,
                "escalate_count_after": self.escalate_count,
                "deescalate_count_before": deescalate_before,
                "deescalate_count_after": self.deescalate_count,
                "raw": raw,
                "error": error,
                "usage": usage_record,
                "timing": controller_timing,
            }
        )

    @staticmethod
    def _validate_decision(decision: ControllerV3Decision, window_ids: list[int], pinned_ids: list[int]) -> str:
        visible = set(window_ids) | set(pinned_ids)
        invalid = [event_id for event_id in decision.evidence_event_ids if event_id not in visible]
        if invalid:
            return f"evidence ids {invalid} are outside the visible set (window + pinned)"
        return ""

    # ---------------- switching ----------------

    def _switch_to(self, new_worker: str) -> bool:
        """Flip the active worker. No content ever moves between conversations:
        the incoming worker gets one notice with the unseen-range cursor and
        recovers state from the shared log and the live workspace."""
        previous = self.active_worker
        self.active_worker = new_worker
        self._activate_worker(new_worker)
        switch = {
            "event_id": self.events[-1]["event_id"],
            "from": previous,
            "to": new_worker,
            "time": time.time(),
        }
        self.last_switch = switch
        self.switches.append(switch)
        self._last_switch_time = switch["time"]
        self._sync_shared_log()
        first = self.worker_last_seen_event[new_worker] + 1
        last = len(self.events)
        if last >= first:
            content = self.config.switch_notice_template.format(
                path=self.config.shared_log_path, first=first, last=last
            )
            self.add_messages(self.model.format_message(role="user", content=content))
        # The outgoing worker has, by construction, seen everything up to now.
        self.worker_last_seen_event[previous] = len(self.events)
        return True

    # ---------------- controller I/O ----------------

    def _query_controller(self) -> tuple[ControllerV3Decision | None, str, str, dict]:
        budgets = [self.config.controller_max_input_tokens]
        if self.config.controller_retry_input_tokens < self.config.controller_max_input_tokens:
            budgets.append(self.config.controller_retry_input_tokens)
        attempts = []
        message = None
        for index, budget in enumerate(budgets):
            try:
                messages, packing = self._pack_controller_messages(budget)
            except Exception as e:
                return (
                    None,
                    "",
                    f"controller prompt packing failed: {type(e).__name__}: {e}",
                    {"prompt_packing": {"attempts": attempts}},
                )
            try:
                started_at = time.time()
                started_ns = time.perf_counter_ns()
                message = self._query_controller_model(messages)
                packing["status"] = "succeeded"
                timing = self._finish_timing(
                    started_at,
                    started_ns,
                    status="succeeded",
                    event_id=len(self.events),
                    attempt=index + 1,
                    model_name=self._model_name(self.controller_model),
                    input_token_budget=budget,
                )
                self.step_timings["controller_queries"].append(timing)
                packing["timing"] = timing
                attempts.append(packing)
                break
            except Exception as e:  # keep the trial alive through transient controller/API failures
                packing["status"] = "context_overflow" if self._is_context_window_error(e) else "failed"
                packing["error"] = f"{type(e).__name__}: {e}"
                timing = self._finish_timing(
                    started_at,
                    started_ns,
                    status=packing["status"],
                    event_id=len(self.events),
                    attempt=index + 1,
                    model_name=self._model_name(self.controller_model),
                    input_token_budget=budget,
                    error_type=type(e).__name__,
                )
                self.step_timings["controller_queries"].append(timing)
                packing["timing"] = timing
                attempts.append(packing)
                if index == 0 and len(budgets) > 1 and self._is_context_window_error(e):
                    continue
                return (
                    None,
                    "",
                    f"controller query failed: {type(e).__name__}: {e}",
                    {"prompt_packing": {"attempts": attempts}},
                )
        assert message is not None
        self._account("controller", message)
        self.cost += message.get("extra", {}).get("cost", 0.0) or 0.0
        usage_record = self._usage_record(message)
        usage_record["prompt_packing"] = {"attempts": attempts}
        raw = self._content_text(message)
        try:
            decision = ControllerV3Decision.model_validate_json(self._extract_json(raw))
        except Exception as e:
            return None, raw, str(e), usage_record
        return decision, raw, "", usage_record

    @staticmethod
    def _is_context_window_error(error: Exception) -> bool:
        context_error = getattr(litellm.exceptions, "ContextWindowExceededError", ())
        if context_error and isinstance(error, context_error):
            return True
        text = f"{type(error).__name__}: {error}".lower()
        return any(
            marker in text
            for marker in (
                "context_length_exceeded",
                "context window exceeded",
                "maximum context length",
                "maximum input length",
                "prompt is too long",
                "input tokens exceed",
            )
        )

    def _query_controller_model(self, messages: list[dict]) -> dict:
        config = getattr(self.controller_model, "config", None)
        if config is None or not hasattr(config, "model_name") or not hasattr(config, "model_kwargs"):
            return self.controller_model.query(messages)
        prepared = self._prepare_controller_messages(messages)
        endpoint_kwargs = (
            self.controller_model.request_kwargs() if hasattr(self.controller_model, "request_kwargs") else {}
        )
        response = _completion_with_hard_timeout(
            model=config.model_name, messages=prepared, **config.model_kwargs, **endpoint_kwargs
        )
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=config.model_name)
        except Exception:
            cost = 0.0
        message = response.choices[0].message.model_dump()
        message["extra"] = {"response": response.model_dump(), "cost": cost, "timestamp": time.time()}
        return message

    @staticmethod
    def _usage_record(message: dict) -> dict:
        response = (message.get("extra") or {}).get("response") or {}
        usage = response.get("usage") or {}
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        return usage if isinstance(usage, dict) else {}

    @staticmethod
    def _content_text(message: dict) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
        if message.get("output"):
            return json.dumps(message.get("output"))
        return ""

    @staticmethod
    def _extract_json(text: str) -> str:
        if match := re.search(r"\{.*\}", text, re.DOTALL):
            return match.group(0)
        return text

    def _account(self, actor: str, message: dict) -> None:
        self.usage[actor]["n_calls"] += 1
        self.usage[actor]["cost"] += message.get("extra", {}).get("cost", 0.0) or 0.0

    # ---------------- serialization ----------------

    def serialize(self, *extra_dicts) -> dict:
        saved_model = self.model
        saved_messages = self.messages
        audit_messages = self.trajectory_messages if self._histories_initialized else self.messages
        self.model = self.large_model
        self.messages = audit_messages
        try:
            data = DefaultAgent.serialize(self, *extra_dicts)
        finally:
            self.model = saved_model
            self.messages = saved_messages
        worker_records = self.step_timings["worker_queries"]
        summary = {
            "worker": self._timing_summary(worker_records),
            "small_worker": self._timing_summary(
                [record for record in worker_records if record.get("worker") == "small"]
            ),
            "large_worker": self._timing_summary(
                [record for record in worker_records if record.get("worker") == "large"]
            ),
            "controller": self._timing_summary(self.step_timings["controller_steps"]),
            "controller_request": self._timing_summary(self.step_timings["controller_queries"]),
            "bash": self._timing_summary(self.step_timings["bash_commands"]),
        }
        walltime_ms = (time.time() - self._start_time) * 1000
        accounted_ms = sum(summary[key]["total_ms"] for key in ("worker", "controller", "bash"))
        data["info"]["oracle_controller_v3"] = {
            "active_worker": self.active_worker,
            "walltime_seconds": walltime_ms / 1000,
            "usage": self.usage,
            "timing": {
                "schema_version": 1,
                "duration_clock": "time.perf_counter_ns",
                "walltime_ms": walltime_ms,
                "accounted_ms": accounted_ms,
                "unattributed_ms": max(0.0, walltime_ms - accounted_ms),
                "summary": summary,
                "records": self.step_timings,
            },
            "decisions": self.controller_decisions,
            "switches": self.switches,
            "escalate_count": self.escalate_count,
            "deescalate_count": self.deescalate_count,
            "shared_log_synced_events": self._shared_log_synced,
            "shared_log_errors": self.shared_log_errors,
            "pinned": self.pinned,
            "worker_last_seen_event": self.worker_last_seen_event,
            "worker_messages": self.worker_messages,
            "protocol": {
                "version": "v3",
                "window_events": self.config.window_events,
                "switch_confirmations": self.config.switch_confirmations,
                "upgrade_confirmations": self.config.upgrade_confirmations or self.config.switch_confirmations,
                "downgrade_confirmations": self.config.downgrade_confirmations or self.config.switch_confirmations,
                "context_management": "keep_histories_pull_only",
                "shared_log_path": self.config.shared_log_path,
                "routing_frequency_features": True,
                "controller_view": "tool_calls_and_results_only",
                "controller_max_input_tokens": self.config.controller_max_input_tokens,
                "controller_retry_input_tokens": self.config.controller_retry_input_tokens,
                "controller_evidence_retention": "pinned_plus_recent_head_tail",
                "continuation": "unsupported_park_only",
            },
            "large_model": self.large_model.serialize().get("info", {}).get("config", {}),
            "small_model": self.small_model.serialize().get("info", {}).get("config", {}),
            "controller_model": self.controller_model.serialize().get("info", {}).get("config", {}),
        }
        return data
