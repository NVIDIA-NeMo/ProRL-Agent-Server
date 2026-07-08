from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from polar.agent.factory import create_harness
from polar.agent.models import AgentRunResult, AgentSpec
from polar.agent.presets.spilot_router import SpilotRouterHarness
from polar.agent.presets.spilot_router_runner import (
    Candidate,
    EpisodeLeaseGrant,
    GatewayEpisodeAdmissionClient,
    GatewayInfrastructureError,
    MiniSwePoolExecutor,
    OpenAIGatewayClient,
    PoolCallResult,
    RouterCompletion,
    RouterProtocolError,
    SpilotOrchestrator,
    UnreapedPoolProcessError,
    _classify_process_failure,
    _openai_chat_payload,
    _verification_instruction,
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
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
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
        "pool_step_limit": 64,
        "pool_cost_limit": 0.0,
        "pool_command_timeout": 120,
        "pool_max_format_errors": 64,
        "pool_response_token_budget": 65_536,
        "pool_model_retry_attempts": 5,
        "observation_max_chars": 10_000,
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
        self.capabilities: list[str] = []

    def run(
        self,
        *,
        candidate: Candidate,
        task: str,
        role: str,
        call_index: int,
        timeout_seconds: float,
        model_call_capability: str = "",
    ) -> PoolCallResult:
        assert timeout_seconds > 0
        self.calls.append((candidate.slot, role))
        self.capabilities.append(model_call_capability)
        if self.workspace is not None:
            marker = self.workspace / "shared.txt"
            if role == "solve":
                marker.write_text("first", encoding="utf-8")
            else:
                assert marker.read_text(encoding="utf-8") == "first"
                marker.write_text("verified", encoding="utf-8")
        return _call_result(candidate, role, call_index, status=self.status)


def _bypass_pool_capability_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        spilot_router_runner,
        "_deliver_pool_call_capability",
        lambda **_kwargs: None,
    )


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
        model_pool_capability="session-pool-capability",
    )

    result = orchestrator.run()

    assert result["action_valid"] is True
    assert result["submitted"] is True
    assert result["termination_reason"] == "router_submit"
    assert [action["action"] for action in result["actions"]] == ["ROUTE", "SUBMIT"]
    assert len(result["calls"]) == 1
    assert result["calls"][0]["status"] == "failed"
    assert result["total_cost"] == 1.0
    assert pool.capabilities == ["session-pool-capability"]
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
        model_pool_capability="session-pool-capability",
    ).run()

    assert pool.calls == [("M0", "solve"), ("M1", "verify")]
    assert pool.capabilities == [
        "session-pool-capability",
        "session-pool-capability",
    ]
    assert (tmp_path / "shared.txt").read_text(encoding="utf-8") == "verified"
    assert result["submitted"] is True
    assert result["termination_reason"] == "verify_auto_submit"
    assert result["final_workspace_fingerprint"] == "fingerprint-1"
    assert result["total_cost"] == 3.0


def test_episode_admission_wraps_each_selected_candidate_before_pool_run() -> None:
    events: list[str] = []

    class EventRouter(FakeRouter):
        def complete(self, **kwargs) -> RouterCompletion:
            events.append("router")
            return super().complete(**kwargs)

    class EventPool(FakePool):
        def run(self, **kwargs) -> PoolCallResult:
            events.append(f"pool:{kwargs['candidate'].model}")
            return super().run(**kwargs)

    class Admission:
        def __init__(self) -> None:
            self.count = 0

        def acquire(
            self, *, model: str, attempt_id: str, timeout_seconds: float
        ) -> EpisodeLeaseGrant:
            assert timeout_seconds > 0
            events.append(f"acquire:{model}:{attempt_id}")
            self.count += 1
            return EpisodeLeaseGrant(
                lease_id=f"lease-{self.count}",
                model=model,
                attempt_id=attempt_id,
                wait_ms=2_000,
                local_cap=4,
                call_capability=f"call-capability-{self.count}",
            )

        def release(self, lease_id: str) -> None:
            events.append(f"release:{lease_id}")

        def close(self) -> None:
            return None

    router = EventRouter(
        '{"action":"ROUTE","model_slot":"M0"}',
        '{"action":"VERIFY","model_slot":"M1"}',
    )
    pool = EventPool()
    result = SpilotOrchestrator(
        config=_runner_config(
            pool_episode_admission_enabled=True,
            pool_episode_admission_wait_budget_seconds=10.0,
        ),
        task="Task",
        router=router,
        pool=pool,
        admission=Admission(),
        model_pool_capability="must-not-use-session-capability",
    ).run()

    assert events == [
        "router",
        "acquire:pool/qwen3.6-27b:0:solve",
        "pool:pool/qwen3.6-27b",
        "release:lease-1",
        "router",
        "acquire:pool/gpt-5.5:1:verify",
        "pool:pool/gpt-5.5",
        "release:lease-2",
    ]
    assert result["admission_wait_ms"] == 4_000
    assert [call["admission_wait_ms"] for call in result["calls"]] == [2_000, 2_000]
    assert [call["admission_local_cap"] for call in result["calls"]] == [4, 4]
    assert pool.capabilities == ["call-capability-1", "call-capability-2"]
    # Queue pressure is infrastructure telemetry, not a policy observation.
    assert "admission_wait" not in router.requests[1]["messages"][-1]["content"]


def test_episode_admission_timeout_records_local_wait_without_candidate_outcome() -> None:
    now = [100.0]

    class TimeoutAdmission:
        def acquire(self, **_kwargs) -> EpisodeLeaseGrant:
            now[0] += 7.25
            raise GatewayInfrastructureError("episode admission request timed out")

        def release(self, _lease_id: str) -> None:
            raise AssertionError("an acquire without a grant must not release")

        def close(self) -> None:
            return None

    orchestrator = SpilotOrchestrator(
        config=_runner_config(
            pool_episode_admission_enabled=True,
            pool_episode_admission_wait_budget_seconds=10.0,
        ),
        task="Task",
        router=FakeRouter('{"action":"ROUTE","model_slot":"M0"}'),
        pool=FakePool(),
        admission=TimeoutAdmission(),
        clock=lambda: now[0],
    )

    with pytest.raises(GatewayInfrastructureError, match="timed out"):
        orchestrator.run()
    orchestrator.mark_infrastructure_error(
        GatewayInfrastructureError("episode admission request timed out")
    )

    assert orchestrator.result["admission_wait_ms"] == 7_250
    assert orchestrator.result["admission_failure"] == {
        "model": "pool/qwen3.6-27b",
        "attempt_id": "0:solve",
        "wait_ms": 7_250,
        "error": "GatewayInfrastructureError: episode admission request timed out",
    }
    assert orchestrator.result["calls"] == []
    assert orchestrator.result["total_cost"] == 0.0
    assert orchestrator.result["termination_reason"] == "infrastructure_error"


def test_unreaped_pool_process_leaves_lease_for_gateway_fatal_retention() -> None:
    released: list[str] = []

    class Admission:
        def acquire(self, **_kwargs) -> EpisodeLeaseGrant:
            return EpisodeLeaseGrant(
                lease_id="lease-still-live",
                model="pool/qwen3.6-27b",
                attempt_id="0:solve",
                wait_ms=0,
                local_cap=1,
                call_capability="lease-call-capability",
            )

        def release(self, lease_id: str) -> None:
            released.append(lease_id)

        def close(self) -> None:
            return None

    class UnreapedPool(FakePool):
        def run(self, **_kwargs) -> PoolCallResult:
            raise UnreapedPoolProcessError("still alive")

    orchestrator = SpilotOrchestrator(
        config=_runner_config(
            pool_episode_admission_enabled=True,
            pool_episode_admission_wait_budget_seconds=10.0,
        ),
        task="Task",
        router=FakeRouter('{"action":"ROUTE","model_slot":"M0"}'),
        pool=UnreapedPool(),
        admission=Admission(),
    )

    with pytest.raises(UnreapedPoolProcessError):
        orchestrator.run()
    assert released == []


def test_process_scope_proof_error_retains_episode_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[str] = []

    class Admission:
        def acquire(self, **_kwargs) -> EpisodeLeaseGrant:
            return EpisodeLeaseGrant(
                lease_id="lease-unverifiable-scope",
                model="pool/qwen3.6-27b",
                attempt_id="0:solve",
                wait_ms=0,
                local_cap=1,
                call_capability="lease-call-capability",
            )

        def release(self, lease_id: str) -> None:
            released.append(lease_id)

        def close(self) -> None:
            return None

    class Process:
        pid = 12_345

        def wait(self, timeout: float) -> int:
            assert timeout > 0
            return 42

    monkeypatch.setattr(
        "polar.agent.presets.spilot_router_runner.subprocess.Popen",
        lambda *_args, **_kwargs: Process(),
    )
    monkeypatch.setattr(
        "polar.agent.presets.spilot_router_runner._terminate_process_scope",
        lambda _process, _scope: False,
    )
    _bypass_pool_capability_delivery(monkeypatch)
    pool = MiniSwePoolExecutor(
        _runner_config(agent_log_dir=str(tmp_path), mini_swe_bin="mini-swe-agent"),
        cwd=tmp_path,
    )
    orchestrator = SpilotOrchestrator(
        config=_runner_config(
            pool_episode_admission_enabled=True,
            pool_episode_admission_wait_budget_seconds=10.0,
        ),
        task="Task",
        router=FakeRouter('{"action":"ROUTE","model_slot":"M0"}'),
        pool=pool,
        admission=Admission(),
    )

    with pytest.raises(UnreapedPoolProcessError, match="failed mini-SWE process scope"):
        orchestrator.run()
    assert released == []


def test_successful_pool_call_preserves_setsid_double_fork_task_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[str] = []
    daemon_pid_path = tmp_path / "daemon.pid"

    class Admission:
        def acquire(self, **_kwargs) -> EpisodeLeaseGrant:
            return EpisodeLeaseGrant(
                lease_id="lease-daemon-contained",
                model="pool/qwen3.6-27b",
                attempt_id="0:solve",
                wait_ms=0,
                local_cap=1,
                call_capability="lease-call-capability",
            )

        def release(self, lease_id: str) -> None:
            assert daemon_pid_path.is_file()
            daemon_pid = int(daemon_pid_path.read_text(encoding="utf-8"))
            assert Path(f"/proc/{daemon_pid}/stat").exists()
            released.append(lease_id)

        def close(self) -> None:
            return None

    daemon_code = r"""
import ctypes
import os
from pathlib import Path
import signal
import sys
import time

secret_read = int(os.environ.pop("POLAR_POOL_CALL_CAPABILITY_FD"))
ready_write = int(os.environ.pop("POLAR_POOL_CALL_READY_FD"))
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(4, 0, 0, 0, 0) != 0 or libc.prctl(3, 0, 0, 0, 0) != 0:
    raise SystemExit(4)
os.write(ready_write, b"1")
os.close(ready_write)
if not os.read(secret_read, 16384).rstrip(b"\r\n"):
    raise SystemExit(5)
os.close(secret_read)
ready_read, ready_write = os.pipe()
first = os.fork()
if first:
    os.close(ready_write)
    if os.read(ready_read, 1) != b"1":
        raise SystemExit(3)
    os.close(ready_read)
    os.waitpid(first, 0)
    raise SystemExit(0)

os.close(ready_read)
os.setsid()
second = os.fork()
if second:
    os._exit(0)

signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")
os.write(ready_write, b"1")
os.close(ready_write)
while True:
    time.sleep(10)
"""
    monkeypatch.setattr(
        MiniSwePoolExecutor,
        "_command",
        lambda *_args, **_kwargs: [
            sys.executable,
            "-c",
            daemon_code,
            str(daemon_pid_path),
        ],
    )
    monkeypatch.setattr(
        spilot_router_runner,
        "_PROCESS_SCOPE_TERM_TIMEOUT_SECONDS",
        0.1,
    )
    monkeypatch.setattr(
        spilot_router_runner,
        "_PROCESS_SCOPE_KILL_TIMEOUT_SECONDS",
        2.0,
    )
    monkeypatch.setattr(
        spilot_router_runner,
        "_workspace_summary",
        lambda _cwd: ("", "", "fingerprint"),
    )
    pool = MiniSwePoolExecutor(
        _runner_config(agent_log_dir=str(tmp_path), mini_swe_bin="mini-swe-agent"),
        cwd=tmp_path,
    )
    orchestrator = SpilotOrchestrator(
        config=_runner_config(
            max_pool_calls=1,
            pool_episode_admission_enabled=True,
            pool_episode_admission_wait_budget_seconds=10.0,
        ),
        task="Task",
        router=FakeRouter('{"action":"ROUTE","model_slot":"M0"}'),
        pool=pool,
        admission=Admission(),
    )

    try:
        result = orchestrator.run()
        assert result["submitted"] is True
        assert released == ["lease-daemon-contained"]
        daemon_pid = int(daemon_pid_path.read_text(encoding="utf-8"))
        assert Path(f"/proc/{daemon_pid}/stat").exists()
    finally:
        if daemon_pid_path.is_file():
            daemon_pid = int(daemon_pid_path.read_text(encoding="utf-8"))
            try:
                os.kill(daemon_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 2
            while Path(f"/proc/{daemon_pid}/stat").exists() and time.monotonic() < deadline:
                try:
                    os.waitpid(daemon_pid, os.WNOHANG)
                except ChildProcessError:
                    pass
                time.sleep(0.01)


def test_episode_admission_wait_budget_matches_gateway_api_limit() -> None:
    with pytest.raises(ValueError, match="between 0 and 86400"):
        SpilotOrchestrator(
            config=_runner_config(
                pool_episode_admission_enabled=True,
                pool_episode_admission_wait_budget_seconds=86_400.001,
            ),
            task="Task",
            router=FakeRouter('{"action":"SUBMIT"}'),
            pool=FakePool(),
        )

    with pytest.raises(ValueError, match="at most 86400"):
        create_harness(
            AgentSpec(
                harness="spilot_router",
                model_name="Qwen/Qwen3.5-9B",
                settings={
                    "model_pool": ["pool/qwen3.6-27b", "pool/gpt-5.5"],
                    "pool_episode_admission_enabled": True,
                    "pool_episode_admission_wait_budget_seconds": 86_400.001,
                },
            )
        )


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
        model_pool_capability="session-pool-capability",
    ).run()

    assert len(router.requests) == 1
    assert pool.calls == [("M1", "solve")]
    assert pool.capabilities == ["session-pool-capability"]
    assert result["submitted"] is True
    assert result["termination_reason"] == "m0_auto_submit"


def test_uncapped_pool_call_without_session_capability_fails_closed() -> None:
    orchestrator = SpilotOrchestrator(
        config=_runner_config(max_pool_calls=1),
        task="Task",
        router=FakeRouter('{"action":"ROUTE","model_slot":"M0"}'),
        pool=FakePool(),
    )

    with pytest.raises(
        GatewayInfrastructureError,
        match="session-scoped model-pool capability is unavailable",
    ):
        orchestrator.run()


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
        timing_path="/tmp/timing.jsonl",
        state_dir="/tmp/pool-state",
        model_kwargs={**config["pool_model_kwargs"], **candidate.model_kwargs},
    )
    kwargs_arg = next(
        value.removeprefix("model.model_kwargs=")
        for value in command
        if value.startswith("model.model_kwargs=")
    )

    assert "--model=openai/pool/gpt-5.5" in command
    assert "--model-class" in command
    assert "polar_mini_swe_vanillux.Vanillux2LitellmModel" in command
    assert "polar_mini_swe_timing.Vanillux2TimedLocalEnvironment" in command
    assert "/opt/polar-mini-swe-agent/config/vanillux2.yaml" in command
    assert "agent.step_limit=64" in command
    assert "agent.max_consecutive_format_errors=64" in command
    assert "environment.timeout=120" in command
    assert "environment.max_output_chars=10000" in command
    assert "environment.state_dir=/tmp/pool-state" in command
    assert "model.response_token_budget=65536" in command
    assert not any(arg == "--task" or arg.startswith("--task=") for arg in command)
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
    captured_close_fds: list[bool] = []
    captured_pass_fds: list[tuple[int, ...]] = []
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
        captured_close_fds.append(kwargs["close_fds"])
        captured_pass_fds.append(kwargs["pass_fds"])
        return Process()

    monkeypatch.setenv("SPILOT_ROUTER_CONFIG_B64", "private-router-config")
    monkeypatch.setenv("SPILOT_TASK_B64", "private-task")
    monkeypatch.setenv("POLAR_ROUTER_CAPABILITY", "router-only-capability")
    monkeypatch.setenv("POLAR_MODEL_POOL_CAPABILITY", "pool-only-capability")
    monkeypatch.setenv("SESSION_ID", "legacy-session-credential")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-openai-credential")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "legacy-anthropic-credential")
    monkeypatch.setenv("GOOGLE_API_KEY", "legacy-google-credential")
    monkeypatch.setenv("NVIDIA_API_KEY", "host-provider-credential")
    monkeypatch.setenv("POLAR_CONTROL_PLANE_TOKEN", "control-plane-credential")
    monkeypatch.setenv("POLAR_GATEWAY_UDS", "/polar/gateway/gateway.sock")
    monkeypatch.setenv("POLAR_HTTP_PROXY_UDS", "/polar/proxy/proxy.sock")
    monkeypatch.setenv("POLAR_HTTP_PROXY_PORT", "28100")
    monkeypatch.setenv("POLAR_HTTP_PROXY_BROKER_READY", "true")
    monkeypatch.setenv("POLAR_ALLOW_INTERNET", "true")
    monkeypatch.setenv("POLAR_APT_HTTP_SOURCE_POLICY", "https")
    monkeypatch.setenv("POLAR_TASK_PYTHONPATH", "/task/python")
    monkeypatch.setenv(
        "POLAR_MODEL_POOL_ADMISSION_CAPABILITY", "admission-only-capability"
    )
    monkeypatch.setattr(
        "polar.agent.presets.spilot_router_runner.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "polar.agent.presets.spilot_router_runner._workspace_summary",
        lambda _cwd: ("", "", "fingerprint"),
    )
    _bypass_pool_capability_delivery(monkeypatch)
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
        model_call_capability="lease-call-capability",
    )

    assert result.status == "completed"
    assert "SPILOT_ROUTER_CONFIG_B64" not in captured_env
    assert "SPILOT_TASK_B64" not in captured_env
    assert "POLAR_ROUTER_CAPABILITY" not in captured_env
    assert "POLAR_MODEL_POOL_CAPABILITY" not in captured_env
    assert "POLAR_MODEL_POOL_ADMISSION_CAPABILITY" not in captured_env
    assert "POLAR_ROUTER_CAPABILITY_FD" not in captured_env
    assert "POLAR_MODEL_POOL_ADMISSION_CAPABILITY_FD" not in captured_env
    assert captured_close_fds == [True]
    assert len(captured_pass_fds) == 1
    assert len(captured_pass_fds[0]) == 2
    assert "OPENAI_API_KEY" not in captured_env
    assert "ANTHROPIC_API_KEY" not in captured_env
    assert "GOOGLE_API_KEY" not in captured_env
    assert "NVIDIA_API_KEY" not in captured_env
    assert "POLAR_CONTROL_PLANE_TOKEN" not in captured_env
    assert "SESSION_ID" not in captured_env
    assert "lease-call-capability" not in captured_env.values()
    assert captured_env["POLAR_GATEWAY_UDS"] == "/polar/gateway/gateway.sock"
    assert captured_env["POLAR_HTTP_PROXY_UDS"] == "/polar/proxy/proxy.sock"
    assert captured_env["POLAR_HTTP_PROXY_PORT"] == "28100"
    assert captured_env["POLAR_HTTP_PROXY_BROKER_READY"] == "true"
    assert captured_env["POLAR_ALLOW_INTERNET"] == "true"
    assert captured_env["POLAR_APT_HTTP_SOURCE_POLICY"] == "https"
    assert captured_env["POLAR_TASK_PYTHONPATH"] == "/task/python"
    assert "POLAR_POOL_CALL_CAPABILITY_FD" in captured_env
    assert "POLAR_POOL_CALL_READY_FD" in captured_env
    assert captured_env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] == "5"
    assert (
        base64.b64decode(captured_env["POLAR_MINI_SWE_TASK_B64"], validate=True).decode("utf-8")
        == "Task"
    )


def test_pool_task_round_trips_for_solve_and_verify_without_entering_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[list[str], dict[str, str]]] = []

    class Process:
        pid = 123

        def wait(self, timeout: float) -> int:
            assert timeout > 0
            return 0

    def fake_popen(args, **kwargs):
        captured.append((list(args), dict(kwargs["env"])))
        return Process()

    monkeypatch.setattr(
        "polar.agent.presets.spilot_router_runner.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(
        "polar.agent.presets.spilot_router_runner._workspace_summary",
        lambda _cwd: ("", "", "fingerprint"),
    )
    _bypass_pool_capability_delivery(monkeypatch)
    config = _runner_config(agent_log_dir=str(tmp_path), mini_swe_bin="mini-swe-agent")
    executor = MiniSwePoolExecutor(config, cwd=tmp_path)
    candidate = Candidate(slot="M0", model="pool/test", card={})
    task = "Stop stale helpers with pkill -f polar-danger-marker-9f27\n雪 'quoted'"

    executor.run(
        candidate=candidate,
        task=task,
        role="solve",
        call_index=0,
        timeout_seconds=10,
        model_call_capability="lease-call-capability",
    )
    executor.run(
        candidate=candidate,
        task=task,
        role="verify",
        call_index=1,
        timeout_seconds=10,
        model_call_capability="lease-call-capability",
    )

    decoded_tasks = [
        base64.b64decode(env["POLAR_MINI_SWE_TASK_B64"], validate=True).decode("utf-8")
        for _, env in captured
    ]
    assert decoded_tasks == [task, _verification_instruction(task)]
    for args, _ in captured:
        os_argv = "\0".join(args)
        assert "polar-danger-marker-9f27" not in os_argv
        assert not any(arg == "--task" or arg.startswith("--task=") for arg in args)
    state_dirs = [
        next(arg for arg in args if arg.startswith("environment.state_dir="))
        for args, _ in captured
    ]
    assert state_dirs == [
        f"environment.state_dir={tmp_path}/spilot-pool-00-state",
        f"environment.state_dir={tmp_path}/spilot-pool-01-state",
    ]
    assert state_dirs[0] != state_dirs[1]


@pytest.mark.parametrize(
    ("return_code", "timed_out", "expected"),
    [
        (-signal.SIGTERM, False, ("signal", signal.SIGTERM, "SIGTERM")),
        (42, False, ("exit_code", None, None)),
        (-1, True, ("timeout", None, None)),
    ],
)
def test_pool_process_failure_classification_prioritizes_timeout(
    return_code: int,
    timed_out: bool,
    expected: tuple[str, int | None, str | None],
) -> None:
    assert _classify_process_failure(return_code=return_code, timed_out=timed_out) == expected


@pytest.mark.parametrize(
    ("mode", "expected_status", "expected_kind", "expected_signal"),
    [
        ("sigterm", "failed", "signal", signal.SIGTERM),
        ("exit42", "failed", "exit_code", None),
        ("timeout", "timeout", "timeout", None),
    ],
)
def test_pool_result_exposes_signal_exit_and_timeout_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_status: str,
    expected_kind: str,
    expected_signal: int | None,
) -> None:
    class Process:
        pid = 123

        def wait(self, timeout: float) -> int:
            if mode == "timeout":
                raise subprocess.TimeoutExpired("mini-swe-agent", timeout)
            if mode == "sigterm":
                return -signal.SIGTERM
            return 42

    monkeypatch.setattr(
        "polar.agent.presets.spilot_router_runner.subprocess.Popen",
        lambda *_args, **_kwargs: Process(),
    )
    monkeypatch.setattr(
        "polar.agent.presets.spilot_router_runner._terminate_process_scope",
        lambda _process, _scope: True,
    )
    monkeypatch.setattr(
        "polar.agent.presets.spilot_router_runner._workspace_summary",
        lambda _cwd: ("", "", "fingerprint"),
    )
    _bypass_pool_capability_delivery(monkeypatch)
    executor = MiniSwePoolExecutor(
        _runner_config(agent_log_dir=str(tmp_path), mini_swe_bin="mini-swe-agent"),
        cwd=tmp_path,
    )

    result = executor.run(
        candidate=Candidate(slot="M0", model="pool/test", card={}),
        task="Task",
        role="solve",
        call_index=0,
        timeout_seconds=0.1,
        model_call_capability="lease-call-capability",
    )
    metadata = result.metadata(index=0, cost=1.0)

    assert result.status == expected_status
    assert metadata["failure_kind"] == expected_kind
    assert metadata.get("signal_number") == expected_signal
    if expected_signal is not None:
        assert metadata["signal_name"] == "SIGTERM"
    else:
        assert "signal_name" not in metadata


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
    assert harness._runner_config["pool_step_limit"] == 64
    assert harness._runner_config["pool_command_timeout"] == 120
    assert harness._runner_config["pool_max_format_errors"] == 64
    assert harness._runner_config["pool_response_token_budget"] == 65_536
    assert harness._runner_config["pool_model_retry_attempts"] == 5
    assert harness._runner_config["observation_max_chars"] == 10_000

    step = harness.run_steps("Fix quoted 'bug'")[0]
    assert step.command is None
    assert step.protected_argv == [
        "/opt/polar-mini-swe-agent/venv/bin/python",
        "/polar/session/spilot_router_runner.py",
    ]
    assert step.protected_env_keys == [
        "POLAR_ROUTER_CAPABILITY",
        "POLAR_MODEL_POOL_CAPABILITY",
    ]
    runner_source = Path(spilot_router_runner.__file__)
    assert step.protected_file_digests == {
        "/polar/session/spilot_router_runner.py": hashlib.sha256(
            runner_source.read_bytes()
        ).hexdigest()
    }
    assert "SPILOT_ROUTER_CONFIG_B64" in step.env
    assert "SPILOT_TASK_B64" in step.env
    assert "POLAR_ROUTER_CAPABILITY" not in step.env
    assert "POLAR_MODEL_POOL_CAPABILITY" not in step.env
    assert "POLAR_MODEL_POOL_ADMISSION_CAPABILITY" not in step.env

    uploaded: list[tuple[str, str]] = []

    async def upload_file(source: str, target: str) -> None:
        uploaded.append((source, target))

    asyncio.run(harness.setup(SimpleNamespace(upload_file=upload_file)))
    assert Path(uploaded[0][0]).name == "spilot_router_runner.py"
    assert uploaded[0][1] == "/polar/session/spilot_router_runner.py"

    admitted_harness = create_harness(
        AgentSpec(
            harness="spilot_router",
            model_name="Qwen/Qwen3.5-9B",
            settings={
                "model_pool": ["pool/qwen3.6-27b", "pool/gpt-5.5"],
                "pool_episode_admission_enabled": True,
                "pool_episode_admission_wait_budget_seconds": 60,
            },
        )
    )
    admitted_step = admitted_harness.run_steps("Fix it")[0]
    assert admitted_step.protected_env_keys == [
        "POLAR_ROUTER_CAPABILITY",
        "POLAR_MODEL_POOL_ADMISSION_CAPABILITY",
    ]
    assert "POLAR_MODEL_POOL_CAPABILITY" not in admitted_step.protected_env_keys


def test_runner_fetches_capabilities_over_fresh_peer_authenticated_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    response = json.dumps(
        {
            "ok": True,
            "protected_env": {
                "POLAR_ROUTER_CAPABILITY": "router-secret",
                "POLAR_MODEL_POOL_CAPABILITY": "pool-secret",
                "POLAR_MODEL_POOL_ADMISSION_CAPABILITY": "admission-secret",
            },
        }
    ).encode() + b"\n"

    class ProtectedConnection:
        def __init__(self) -> None:
            self.response_pending = True

        def settimeout(self, _timeout: float) -> None:
            return None

        def connect(self, path: str) -> None:
            assert path == "/protected/socket"

        def getsockopt(self, *_args: object) -> bytes:
            return spilot_router_runner.struct.pack("3i", os.getpid(), os.getuid(), os.getgid())

        def sendall(self, data: bytes) -> None:
            observed.update(json.loads(data))

        def recv(self, _size: int) -> bytes:
            if self.response_pending:
                self.response_pending = False
                return response
            return b""

        def close(self) -> None:
            return None

    hardened: list[bool] = []
    monkeypatch.setattr(
        spilot_router_runner,
        "_set_and_verify_non_dumpable",
        lambda: hardened.append(True),
    )
    monkeypatch.setattr(
        spilot_router_runner.socket,
        "socket",
        lambda *_args, **_kwargs: ProtectedConnection(),
    )
    monkeypatch.setattr(spilot_router_runner, "_PROTECTED_ENV_CACHE", None)
    monkeypatch.setenv("POLAR_PROTECTED_EXEC_SOCKET", "/protected/socket")
    monkeypatch.setenv("POLAR_PROTECTED_EXEC_REQUEST_ID", "request-123")
    monkeypatch.setenv("POLAR_PROTECTED_EXEC_BROKER_PID", str(os.getpid()))
    ready_read, ready_write = os.pipe()
    os.write(ready_write, b"1")
    os.close(ready_write)
    monkeypatch.setenv("POLAR_PROTECTED_EXEC_READY_FD", str(ready_read))
    monkeypatch.delenv("POLAR_ROUTER_CAPABILITY", raising=False)
    monkeypatch.delenv("POLAR_MODEL_POOL_CAPABILITY", raising=False)
    monkeypatch.delenv("POLAR_MODEL_POOL_ADMISSION_CAPABILITY", raising=False)

    values = spilot_router_runner._receive_protected_environment()

    assert hardened == [True]
    assert observed == {"operation": "protected_child_ready", "id": "request-123"}
    assert values == {
        "POLAR_ROUTER_CAPABILITY": "router-secret",
        "POLAR_MODEL_POOL_CAPABILITY": "pool-secret",
        "POLAR_MODEL_POOL_ADMISSION_CAPABILITY": "admission-secret",
    }
    assert "POLAR_PROTECTED_EXEC_SOCKET" not in os.environ
    assert "POLAR_PROTECTED_EXEC_REQUEST_ID" not in os.environ
    assert "POLAR_PROTECTED_EXEC_BROKER_PID" not in os.environ
    assert "POLAR_PROTECTED_EXEC_READY_FD" not in os.environ


def test_admission_http_retries_response_loss_with_stable_ids() -> None:
    lease_id = "lease-response-loss-stable-id"

    class Response:
        status_code = 200

        def __init__(self, body: dict[str, object]) -> None:
            self._body = body

        def json(self) -> dict[str, object]:
            return self._body

    class Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object], float]] = []
            self.outcomes: list[object] = [
                OSError("response lost after acquire commit"),
                Response(
                    {
                        "lease_id": lease_id,
                        "model": "pool/qwen",
                        "attempt_id": "0:solve",
                        "wait_ms": 7,
                        "local_cap": 1,
                        "call_capability": "lease-call-capability",
                    }
                ),
                OSError("response lost after release commit"),
                Response({"released": False}),
            ]

        def post(self, url: str, **kwargs: object) -> Response:
            self.calls.append((url, dict(kwargs["json"]), float(kwargs["timeout"])))
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            assert isinstance(outcome, Response)
            return outcome

    transport = Client()
    client = GatewayEpisodeAdmissionClient.__new__(GatewayEpisodeAdmissionClient)
    client._acquire_url = "http://gateway/internal/acquire"
    client._release_url = "http://gateway/internal/release"
    client._headers = {"Authorization": "Bearer redacted"}
    client._client = transport
    client._clock = spilot_router_runner.time.monotonic
    client._sleep = lambda _seconds: None

    grant = client.acquire(
        model="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=10,
    )
    client.release(grant.lease_id)

    assert [call[1]["attempt_id"] for call in transport.calls[:2]] == [
        "0:solve",
        "0:solve",
    ]
    assert [call[1]["lease_id"] for call in transport.calls[2:]] == [
        lease_id,
        lease_id,
    ]
    assert all(call[2] > 0 for call in transport.calls)


def test_admission_http_retry_never_exceeds_total_budget() -> None:
    class Clock:
        now = 100.0

        def __call__(self) -> float:
            return self.now

    clock = Clock()

    class Client:
        timeouts: list[float] = []

        def post(self, _url: str, **kwargs: object) -> object:
            timeout = float(kwargs["timeout"])
            self.timeouts.append(timeout)
            clock.now += timeout
            raise TimeoutError("simulated full request timeout")

    transport = Client()
    client = GatewayEpisodeAdmissionClient.__new__(GatewayEpisodeAdmissionClient)
    client._headers = {"Authorization": "Bearer redacted"}
    client._client = transport
    client._clock = clock
    client._sleep = lambda seconds: setattr(clock, "now", clock.now + seconds)

    with pytest.raises(GatewayInfrastructureError, match="after 1 attempt"):
        client._post(
            "http://gateway/internal/acquire",
            payload={"attempt_id": "same-attempt"},
            budget_seconds=2.5,
        )
    assert transport.timeouts == [2.5]
    assert clock.now == 102.5


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
    client._headers = {
        "Authorization": "Bearer router-only-capability",
        "Content-Type": "application/json",
    }

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
    capability_reads: list[str] = []

    def read_capability(key: str) -> str:
        capability_reads.append(key)
        return "session-pool-capability"

    monkeypatch.setattr(
        spilot_router_runner,
        "_read_protected_capability",
        read_capability,
    )

    assert spilot_router_runner.main() == 0
    assert capability_reads == ["POLAR_MODEL_POOL_CAPABILITY"]
    payload = json.loads((tmp_path / "router_result.json").read_text(encoding="utf-8"))
    assert payload["action_valid"] is True
    assert payload["calls"][0]["status"] == "failed"
    assert payload["submitted"] is True
