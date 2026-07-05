from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from polar.agent.factory import create_harness
from polar.agent.models import AgentRunResult, AgentSpec
from polar.agent.presets.spilot_router import SpilotRouterHarness
from polar.agent.presets.spilot_router_runner import (
    Candidate,
    MiniSwePoolExecutor,
    OpenAIGatewayClient,
    PoolCallResult,
    RouterCompletion,
    RouterProtocolError,
    SpilotOrchestrator,
    _openai_chat_payload,
    parse_router_action,
)
from polar.agent.presets import spilot_router_runner


def _runner_config(**updates: object) -> dict[str, object]:
    config: dict[str, object] = {
        "schema_version": 1,
        "router_model": "router/policy",
        "model_pool": {
            "M0": {
                "model": "pool/qwen3.6-27b",
                "card": {"family": "qwen", "strengths": ["coding"]},
                "cost_weight": 1.0,
                "model_kwargs": {
                    "max_tokens": 16384,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                },
            },
            "M1": {
                "model": "pool/gpt-5.5",
                "card": {"family": "gpt", "strengths": ["reasoning"]},
                "cost_weight": 2.0,
                "model_kwargs": {"max_completion_tokens": 16384},
            },
        },
        "max_pool_calls": 2,
        "shuffle_slots": False,
        "shuffle_seed": 0,
        "router_max_tokens": 192,
        "router_timeout_seconds": 30.0,
        "pool_timeout_seconds": 120.0,
        "total_timeout_seconds": 300.0,
        "reserve_evaluator_seconds": 30.0,
        "deadline_margin_seconds": 1.0,
        "pool_step_limit": 30,
        "pool_cost_limit": 0.0,
        "pool_model_retry_attempts": 3,
        "observation_max_chars": 12_000,
        "log_tail_chars": 6_000,
        "router_model_kwargs": {},
        "pool_model_kwargs": {},
        "slot_assignment_seed": None,
        "runner_python": "/opt/polar-mini-swe-agent/venv/bin/python",
        "mini_swe_bin": "/opt/polar-mini-swe-agent/bin/mini-swe-agent",
        "result_path": "/polar/session/artifacts/router_result.json",
        "agent_log_dir": "/polar/session/logs/agent",
    }
    config.update(updates)
    return config


def _call_result(
    candidate: Candidate,
    role: str,
    call_index: int,
    *,
    status: str = "completed",
) -> PoolCallResult:
    return PoolCallResult(
        slot=candidate.slot,
        model=candidate.model,
        role=role,
        status=status,
        return_code=0 if status == "completed" else 1,
        duration_ms=100 + call_index,
        attempted=True,
        timed_out=False,
        log_file=f"/logs/call-{call_index}.txt",
        log_tail=f"agent {call_index} {status}",
        git_status=" M solution.py",
        git_diff_stat=" solution.py | 2 ++",
        workspace_fingerprint=f"fingerprint-{call_index}",
        error=None if status == "completed" else "model failed",
    )


class FakeRouter:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        timeout_seconds: float,
        model_kwargs: dict[str, object],
    ) -> RouterCompletion:
        self.requests.append(
            {
                "model": model,
                "messages": messages,
                "timeout_seconds": timeout_seconds,
                "model_kwargs": model_kwargs,
            }
        )
        return RouterCompletion(
            content=self.responses.pop(0),
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


class FakePool:
    def __init__(self, *, status: str = "completed", workspace: Path | None = None) -> None:
        self.status = status
        self.workspace = workspace
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
        assert timeout_seconds > 0
        self.calls.append((candidate.slot, role))
        if self.workspace is not None:
            marker = self.workspace / "shared.txt"
            if role == "solve":
                marker.write_text("first", encoding="utf-8")
            else:
                assert marker.read_text(encoding="utf-8") == "first"
                marker.write_text("verified", encoding="utf-8")
        return _call_result(candidate, role, call_index, status=self.status)


@pytest.mark.parametrize(
    ("text", "expected", "result"),
    [
        ('{"action":"ROUTE","model_slot":"M0"}\n', "ROUTE", "ROUTE"),
        ('{"action":"SUBMIT"}', "FINAL", "SUBMIT"),
        ('{"action":"VERIFY","model_slot":"M1"}', "FINAL", "VERIFY"),
    ],
)
def test_strict_action_parser_accepts_only_the_bounded_grammar(
    text: str, expected: str, result: str
) -> None:
    action = parse_router_action(text, expected=expected, allowed_slots={"M0", "M1"})

    assert action["action"] == result


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"action":"SUBMIT"}\n```',
        '{"action":"SUBMIT","reason":"done"}',
        '{"action":"ROUTE","model_slot":"M9"}',
        '{"action":"ROUTE","action":"ROUTE","model_slot":"M0"}',
        'I choose {"action":"SUBMIT"}',
    ],
)
def test_strict_action_parser_rejects_repairs_extra_keys_and_unknown_slots(text: str) -> None:
    phase = "ROUTE" if "ROUTE" in text else "FINAL"
    with pytest.raises(RouterProtocolError):
        parse_router_action(text, expected=phase, allowed_slots={"M0", "M1"})


def test_route_then_submit_records_router_contract_and_bounded_observation() -> None:
    router = FakeRouter(
        '{"action":"ROUTE","model_slot":"M0"}',
        '{"action":"SUBMIT"}',
    )
    pool = FakePool(status="failed")
    orchestrator = SpilotOrchestrator(
        config=_runner_config(),
        task="Fix the bug",
        router=router,
        pool=pool,
    )

    result = orchestrator.run()

    assert result["action_valid"] is True
    assert result["submitted"] is True
    assert result["termination_reason"] == "router_submit"
    assert [action["action"] for action in result["actions"]] == ["ROUTE", "SUBMIT"]
    assert len(result["calls"]) == 1
    assert result["calls"][0]["status"] == "failed"
    assert result["total_cost"] == 1.0
    second_prompt = router.requests[1]["messages"][-1]["content"]
    assert '"status": "failed"' in second_prompt
    assert "hidden evaluator tests have not run" in second_prompt


def test_verify_runs_fresh_agent_on_same_mutable_workspace(tmp_path: Path) -> None:
    router = FakeRouter(
        '{"action":"ROUTE","model_slot":"M0"}',
        '{"action":"VERIFY","model_slot":"M1"}',
    )
    pool = FakePool(workspace=tmp_path)
    result = SpilotOrchestrator(
        config=_runner_config(),
        task="Implement the feature",
        router=router,
        pool=pool,
    ).run()

    assert pool.calls == [("M0", "solve"), ("M1", "verify")]
    assert (tmp_path / "shared.txt").read_text(encoding="utf-8") == "verified"
    assert result["submitted"] is True
    assert result["termination_reason"] == "verify_auto_submit"
    assert result["final_workspace_fingerprint"] == "fingerprint-1"
    assert result["total_cost"] == 3.0


def test_invalid_router_action_is_trainable_policy_failure_not_pool_failure() -> None:
    router = FakeRouter('{"action":"SUBMIT"}')
    pool = FakePool()
    result = SpilotOrchestrator(
        config=_runner_config(), task="Task", router=router, pool=pool
    ).run()

    assert result["action_valid"] is False
    assert result["submitted"] is False
    assert result["termination_reason"] == "invalid_action_step_0"
    assert result["calls"] == []
    assert pool.calls == []


def test_m0_routes_once_then_auto_submits() -> None:
    router = FakeRouter('{"action":"ROUTE","model_slot":"M1"}')
    pool = FakePool()
    result = SpilotOrchestrator(
        config=_runner_config(max_pool_calls=1),
        task="Task",
        router=router,
        pool=pool,
    ).run()

    assert len(router.requests) == 1
    assert pool.calls == [("M1", "solve")]
    assert result["submitted"] is True
    assert result["termination_reason"] == "m0_auto_submit"


def test_pool_command_merges_global_and_per_candidate_request_kwargs(tmp_path: Path) -> None:
    config = _runner_config(
        agent_log_dir=str(tmp_path),
        pool_model_kwargs={"timeout": 60},
        mini_swe_bin="mini-swe-agent",
    )
    executor = MiniSwePoolExecutor(config, cwd=tmp_path)
    candidate = Candidate(
        slot="M1",
        model="pool/gpt-5.5",
        card={},
        model_kwargs={"max_completion_tokens": 8192},
    )

    command = executor._command(
        model=candidate.model,
        instruction="Task",
        timing_path="/tmp/timing.jsonl",
        model_kwargs={**config["pool_model_kwargs"], **candidate.model_kwargs},
    )
    kwargs_arg = next(
        value.removeprefix("model.model_kwargs=")
        for value in command
        if value.startswith("model.model_kwargs=")
    )

    assert "--model=openai/pool/gpt-5.5" in command
    assert json.loads(kwargs_arg) == {
        "max_completion_tokens": 8192,
        "timeout": 60,
    }
    assert "temperature" not in json.loads(kwargs_arg)
    assert "top_p" not in json.loads(kwargs_arg)


def test_pool_child_does_not_inherit_outer_router_protocol_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_env: dict[str, str] = {}
    real_popen = subprocess.Popen

    class Process:
        pid = 123

        def wait(self, timeout: float) -> int:
            assert timeout > 0
            return 0

    def fake_popen(*_args, **kwargs):
        if "env" not in kwargs:
            return real_popen(*_args, **kwargs)
        captured_env.update(kwargs["env"])
        return Process()

    monkeypatch.setenv("SPILOT_ROUTER_CONFIG_B64", "private-router-config")
    monkeypatch.setenv("SPILOT_TASK_B64", "private-task")
    monkeypatch.setenv("POLAR_ROUTER_CAPABILITY", "router-only-capability")
    monkeypatch.setenv("POLAR_MODEL_POOL_CAPABILITY", "pool-only-capability")
    monkeypatch.setattr(
        "polar.agent.presets.spilot_router_runner.subprocess.Popen",
        fake_popen,
    )
    config = _runner_config(
        agent_log_dir=str(tmp_path),
        mini_swe_bin="mini-swe-agent",
    )
    executor = MiniSwePoolExecutor(config, cwd=tmp_path)

    result = executor.run(
        candidate=Candidate(slot="M0", model="pool/test", card={}),
        task="Task",
        role="solve",
        call_index=0,
        timeout_seconds=10,
    )

    assert result.status == "completed"
    assert "SPILOT_ROUTER_CONFIG_B64" not in captured_env
    assert "SPILOT_TASK_B64" not in captured_env
    assert "POLAR_ROUTER_CAPABILITY" not in captured_env
    assert "POLAR_MODEL_POOL_CAPABILITY" not in captured_env
    assert captured_env["OPENAI_API_KEY"] == "pool-only-capability"


def test_harness_is_builtin_and_uploads_portable_runner() -> None:
    harness = create_harness(
        AgentSpec(
            harness="spilot_router",
            model_name="Qwen/Qwen3.5-9B",
            settings={
                "model_pool": ["pool/qwen3.6-27b", "pool/gpt-5.5"],
                "shuffle_slots": False,
            },
        )
    )
    assert isinstance(harness, SpilotRouterHarness)
    assert harness._runner_config["router_model"] == "router/policy"

    step = harness.run_steps("Fix quoted 'bug'")[0]
    assert "/opt/polar-mini-swe-agent/venv/bin/python" in step.command
    assert "/polar/session/spilot_router_runner.py" in step.command
    assert 'export OPENAI_API_BASE="$OPENAI_BASE_URL"' in step.command
    assert step.command.startswith("set -o pipefail; ")
    assert "SPILOT_ROUTER_CONFIG_B64" in step.env
    assert "SPILOT_TASK_B64" in step.env

    uploaded: list[tuple[str, str]] = []

    async def upload_file(source: str, target: str) -> None:
        uploaded.append((source, target))

    asyncio.run(harness.setup(SimpleNamespace(upload_file=upload_file)))
    assert Path(uploaded[0][0]).name == "spilot_router_runner.py"
    assert uploaded[0][1] == "/polar/session/spilot_router_runner.py"


def test_harness_normalizes_fixed_eval_settings_without_expanding_action_budget() -> None:
    harness = SpilotRouterHarness(
        AgentSpec(
            harness="spilot_router",
            model_name="Qwen/Qwen3.5-9B",
            settings={
                "model_pool": ["pool/qwen", "pool/gpt"],
                "router_max_tokens": 192,
                "pool_step_limit": 30,
                "router_model_kwargs": {
                    "temperature": 1.0,
                    "extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                },
                # Added by Slime's fixed-eval path.
                "model_kwargs": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_tokens": 16_384,
                    "extra_body": {"top_k": 20},
                },
                "sampling_seed": 1234,
                "step_limit": 64,
            },
        )
    )

    config = harness._runner_config
    assert config["router_max_tokens"] == 192
    assert config["router_model_kwargs"] == {
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 1234,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    assert "max_tokens" not in config["router_model_kwargs"]
    assert config["pool_step_limit"] == 64
    assert config["slot_assignment_seed"] == 1234


def test_fixed_eval_seed_makes_slot_mapping_independent_of_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runner_config(shuffle_slots=True, slot_assignment_seed=9876)
    monkeypatch.setenv("SESSION_ID", "baseline-session")
    baseline = SpilotOrchestrator(
        config=config,
        task="Task",
        router=FakeRouter('{"action":"ROUTE","model_slot":"M0"}'),
        pool=FakePool(),
    )
    monkeypatch.setenv("SESSION_ID", "final-session")
    final = SpilotOrchestrator(
        config=config,
        task="Task",
        router=FakeRouter('{"action":"ROUTE","model_slot":"M0"}'),
        pool=FakePool(),
    )

    assert baseline.result["slot_mapping"] == final.result["slot_mapping"]


def test_harness_postprocess_attaches_spilot_router_metadata(tmp_path: Path) -> None:
    result_payload = {
        "schema_version": 1,
        "action_valid": True,
        "actions": [{"step": 0, "valid": True, "action": "ROUTE", "model_slot": "M0"}],
        "calls": [],
        "submitted": True,
        "total_cost": 0.0,
        "slot_mapping": {"M0": {"model": "pool/test"}},
        "slot_mapping_fingerprint": "abc",
        "termination_reason": "m0_auto_submit",
    }
    artifact = tmp_path / "router_result.json"
    artifact.write_text(json.dumps(result_payload), encoding="utf-8")
    runtime = SimpleNamespace(resolve_host_path=lambda _path: artifact)
    harness = SpilotRouterHarness(
        AgentSpec(
            harness="spilot_router",
            model_name="Qwen/Qwen3.5-9B",
            settings={"model_pool": ["pool/test"]},
        )
    )
    result = AgentRunResult(status="completed", return_code=0)

    asyncio.run(harness.postprocess(runtime, result))

    assert result.metadata["spilot_router"] == result_payload


def test_candidate_request_kwargs_reject_credentials() -> None:
    config = _runner_config(
        model_pool={"M0": {"model": "pool/test", "model_kwargs": {"api_key": "bad"}}}
    )

    with pytest.raises(ValueError, match="credential"):
        SpilotOrchestrator(
            config=config,
            task="Task",
            router=FakeRouter('{"action":"ROUTE","model_slot":"M0"}'),
            pool=FakePool(),
        )


def test_router_extra_body_is_flattened_like_openai_sdk() -> None:
    payload = _openai_chat_payload(
        model="router",
        messages=[{"role": "user", "content": "route"}],
        model_kwargs={
            "temperature": 0.7,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
    )

    assert "extra_body" not in payload
    assert payload["temperature"] == 0.7
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_router_extra_body_cannot_silently_override_standard_fields() -> None:
    with pytest.raises(Exception, match="duplicates request fields"):
        _openai_chat_payload(
            model="router",
            messages=[],
            model_kwargs={"max_tokens": 10, "extra_body": {"max_tokens": 20}},
        )


def test_router_null_content_is_sampled_invalid_action_not_infrastructure_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "id": "completion-1",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": None},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"completion_tokens": 192},
            }

    captured_headers: dict[str, str] = {}

    class FakeHttpClient:
        def post(self, *args: object, **kwargs: object) -> FakeResponse:
            captured_headers.update(kwargs["headers"])
            return FakeResponse()

    client = OpenAIGatewayClient.__new__(OpenAIGatewayClient)
    client._url = "http://gateway/v1/chat/completions"
    client._client = FakeHttpClient()
    monkeypatch.setenv("OPENAI_API_KEY", "session-id")
    monkeypatch.setenv("POLAR_ROUTER_CAPABILITY", "router-only-capability")

    completion = client.complete(
        model="router",
        messages=[{"role": "user", "content": "route"}],
        timeout_seconds=1.0,
        model_kwargs={},
    )

    assert completion.content == ""
    assert completion.finish_reason == "length"
    assert captured_headers["Authorization"] == "Bearer router-only-capability"
    with pytest.raises(RouterProtocolError, match="empty"):
        parse_router_action(completion.content, expected="ROUTE", allowed_slots={"M0"})


def test_main_returns_success_for_pool_model_failure_and_writes_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _runner_config(
        result_path=str(tmp_path / "router_result.json"),
        agent_log_dir=str(tmp_path / "logs"),
    )
    encoded_config = base64.b64encode(json.dumps(config).encode()).decode()
    encoded_task = base64.b64encode(b"Fix it").decode()
    monkeypatch.setenv("SPILOT_ROUTER_CONFIG_B64", encoded_config)
    monkeypatch.setenv("SPILOT_TASK_B64", encoded_task)

    class MainRouter(FakeRouter):
        def __init__(self) -> None:
            super().__init__(
                '{"action":"ROUTE","model_slot":"M0"}',
                '{"action":"SUBMIT"}',
            )

        def close(self) -> None:
            return None

    class MainPool(FakePool):
        def __init__(self, config: dict[str, object]) -> None:
            super().__init__(status="failed")

    monkeypatch.setattr(spilot_router_runner, "OpenAIGatewayClient", MainRouter)
    monkeypatch.setattr(spilot_router_runner, "MiniSwePoolExecutor", MainPool)

    assert spilot_router_runner.main() == 0
    payload = json.loads((tmp_path / "router_result.json").read_text(encoding="utf-8"))
    assert payload["action_valid"] is True
    assert payload["calls"][0]["status"] == "failed"
    assert payload["submitted"] is True
