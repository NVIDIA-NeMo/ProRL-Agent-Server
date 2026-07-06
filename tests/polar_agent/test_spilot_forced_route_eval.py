from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from polar.agent.models import AgentSpec
from polar.agent.presets.spilot_forced_route_eval_runner import (
    EVAL_ONLY_ACK,
    ForcedRouteClient,
    run_forced_eval,
)
from polar.agent.presets.spilot_router import SpilotRouterHarness
from polar.agent.presets.spilot_router_runner import Candidate, PoolCallResult
from polar.trajectory.builder.spilot_forced_eval import SpilotForcedEvalBuilder
from polar.trajectory.models import CompletionRecord, CompletionSession, StrategySpec
from polar.trajectory.registry import default_builder_registry


def _settings(**updates: object) -> dict[str, object]:
    settings: dict[str, object] = {
        "model_pool": {
            "M0": {"model": "pool/qwen3.6-27b", "card": {"name": "Qwen"}},
            "M1": {"model": "pool/gpt-5.5", "card": {"name": "GPT"}},
        },
        "max_pool_calls": 1,
        "shuffle_slots": True,
        "sampling_seed": 17,
        "forced_route_eval": {
            "enabled": True,
            "acknowledgement": EVAL_ONLY_ACK,
            "candidate_model": "pool/qwen3.6-27b",
        },
    }
    settings.update(updates)
    return settings


def _spec(**updates: object) -> AgentSpec:
    values: dict[str, object] = {
        "harness": "spilot_router",
        "model_name": "eval-only/no-router",
        "settings": _settings(),
        "env": {"SPILOT_FORCED_ROUTE_EVAL_ACK": EVAL_ONLY_ACK},
    }
    values.update(updates)
    return AgentSpec.model_validate(values)


def test_forced_eval_harness_requires_both_acknowledgements() -> None:
    no_config_ack = _settings()
    no_config_ack["forced_route_eval"] = {
        "enabled": True,
        "acknowledgement": "wrong",
        "candidate_model": "pool/qwen3.6-27b",
    }
    with pytest.raises(ValueError, match="acknowledgement"):
        SpilotRouterHarness(_spec(settings=no_config_ack))

    with pytest.raises(ValueError, match="agent acknowledgement"):
        SpilotRouterHarness(_spec(env={}))


def test_forced_eval_harness_rejects_unknown_or_duplicate_candidate() -> None:
    settings = _settings()
    forced = dict(settings["forced_route_eval"])  # type: ignore[arg-type]
    forced["candidate_model"] = "pool/missing"
    settings["forced_route_eval"] = forced
    with pytest.raises(ValueError, match="matched 0"):
        SpilotRouterHarness(_spec(settings=settings))


def test_forced_eval_harness_uploads_separate_runner_and_never_uses_normal_entrypoint() -> None:
    harness = SpilotRouterHarness(_spec())
    assert harness._runner_config["max_pool_calls"] == 1
    assert harness._runner_config["forced_route_eval"] == {
        "acknowledgement": EVAL_ONLY_ACK,
        "candidate_model": "pool/qwen3.6-27b",
    }

    uploaded: list[tuple[str, str]] = []

    async def upload_file(source: str, target: str) -> None:
        uploaded.append((source, target))

    asyncio.run(harness.setup(SimpleNamespace(upload_file=upload_file)))
    assert [Path(source).name for source, _target in uploaded] == [
        "spilot_router_runner.py",
        "spilot_forced_route_eval_runner.py",
    ]
    step = harness.run_steps("Fix it")[0]
    assert "/polar/session/spilot_forced_route_eval_runner.py" in step.command
    assert "SPILOT_FORCED_ROUTE_EVAL_RUNTIME_ACK" in step.env
    config = json.loads(
        __import__("base64").b64decode(step.env["SPILOT_ROUTER_CONFIG_B64"])
    )
    assert config["forced_route_eval"]["candidate_model"] == "pool/qwen3.6-27b"


class _FakePool:
    def __init__(self, _config: dict[str, object]) -> None:
        self.calls: list[tuple[str, str]] = []

    def run(
        self,
        *,
        candidate: Candidate,
        task: str,
        role: str,
        call_index: int,
        timeout_seconds: float,
    ) -> PoolCallResult:
        del task, call_index, timeout_seconds
        self.calls.append((candidate.model, role))
        return PoolCallResult(
            slot=candidate.slot,
            model=candidate.model,
            role=role,
            status="completed",
            return_code=0,
            duration_ms=25,
            attempted=True,
            timed_out=False,
            log_file="/tmp/pool.log",
            log_tail="done",
            git_status=" M answer.py",
            git_diff_stat="answer.py | 1 +",
            workspace_fingerprint="abc",
        )


def test_forced_runner_selects_model_identity_and_auto_submits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polar.agent.presets import spilot_forced_route_eval_runner as module

    harness = SpilotRouterHarness(_spec())
    monkeypatch.setenv("SPILOT_FORCED_ROUTE_EVAL_RUNTIME_ACK", EVAL_ONLY_ACK)
    monkeypatch.setattr(module.core, "MiniSwePoolExecutor", _FakePool)

    result = run_forced_eval(harness._runner_config, "Fix it")

    assert result["eval_only"] is True
    assert result["actor_invoked"] is False
    assert result["submitted"] is True
    assert result["termination_reason"] == "m0_auto_submit"
    assert result["forced_candidate_model"] == "pool/qwen3.6-27b"
    assert result["actions"] == [
        {
            "step": 0,
            "valid": True,
            "action": "ROUTE",
            "model_slot": result["calls"][0]["slot"],
            "usage": {},
            "finish_reason": "forced_eval_only",
        }
    ]
    assert result["calls"][0]["model"] == "pool/qwen3.6-27b"
    assert len(result["calls"]) == 1


def test_forced_route_client_refuses_second_decision() -> None:
    client = ForcedRouteClient()
    client.slot = "M1"
    first = client.complete(model="unused", messages=[], timeout_seconds=1, model_kwargs={})
    assert json.loads(first.content) == {"action": "ROUTE", "model_slot": "M1"}
    with pytest.raises(Exception, match="more than one"):
        client.complete(model="unused", messages=[], timeout_seconds=1, model_kwargs={})


@pytest.mark.asyncio
async def test_eval_builder_is_guarded_completed_and_permanently_trace_free() -> None:
    completion = CompletionRecord(
        completion_id="pool-output-that-must-never-train",
        metadata={"completion_role": "unexpected"},
    )
    session = CompletionSession(
        session_id="session-1",
        task_id="task-1",
        metadata={"dataset_index": 7},
        completions=[completion],
    )
    with pytest.raises(ValueError, match="acknowledgement"):
        SpilotForcedEvalBuilder(acknowledgement="wrong")

    builder = default_builder_registry().create(
        StrategySpec(
            strategy=(
                "polar.trajectory.builder.spilot_forced_eval:SpilotForcedEvalBuilder"
            ),
            config={"acknowledgement": EVAL_ONLY_ACK},
        )
    )
    trajectory = await builder.build(session)

    assert trajectory.status == "COMPLETED"
    assert trajectory.traces == []
    assert trajectory.metadata["trainable"] is False
    assert trajectory.metadata["raw_record_count"] == 1
    assert trajectory.metadata["record_count"] == 0
