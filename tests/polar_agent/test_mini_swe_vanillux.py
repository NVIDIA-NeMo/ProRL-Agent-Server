from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "src" / "polar" / "agent" / "presets" / "mini_swe_vanillux.py"


class _FakeFormatError(Exception):
    def __init__(self, *messages: dict) -> None:
        self.messages = messages
        super().__init__()


class _FakeLimitsExceeded(Exception):
    def __init__(self, *messages: dict) -> None:
        self.messages = messages
        super().__init__()


class _FakeLitellmModel:
    def __init__(self, **kwargs) -> None:
        model_kwargs = kwargs.get("model_kwargs")
        if model_kwargs is None:
            model_kwargs = {
                "temperature": 0.7,
                "max_tokens": 100,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            }
        self.config = SimpleNamespace(
            model_name=kwargs.get("model_name", "openai/Qwen3.5-9B"),
            model_kwargs=dict(model_kwargs),
            format_error_template="{{ error }} (finish={{ finish_reason }})",
        )

    def _prepare_messages_for_api(self, messages):
        return messages

    def query(self, messages, **kwargs):
        return self._query(self._prepare_messages_for_api(messages), **kwargs)

    def _parse_actions(self, response):
        return response.actions


def _load_vanillux_module(monkeypatch):
    litellm = ModuleType("litellm")
    litellm.completion = lambda **_kwargs: None
    litellm.exceptions = SimpleNamespace(AuthenticationError=RuntimeError)
    jinja2 = ModuleType("jinja2")

    class Template:
        def __init__(self, source, **_kwargs) -> None:
            self.source = source

        def render(self, **values) -> str:
            result = self.source
            for key, value in values.items():
                result = result.replace("{{ " + key + " }}", str(value))
            return result

    jinja2.StrictUndefined = object()
    jinja2.Template = Template
    package = ModuleType("minisweagent")
    exceptions = ModuleType("minisweagent.exceptions")
    models = ModuleType("minisweagent.models")
    litellm_model = ModuleType("minisweagent.models.litellm_model")
    exceptions.FormatError = _FakeFormatError
    exceptions.LimitsExceeded = _FakeLimitsExceeded
    litellm_model.LitellmModel = _FakeLitellmModel
    monkeypatch.setitem(sys.modules, "litellm", litellm)
    monkeypatch.setitem(sys.modules, "jinja2", jinja2)
    monkeypatch.setitem(sys.modules, "minisweagent", package)
    monkeypatch.setitem(sys.modules, "minisweagent.exceptions", exceptions)
    monkeypatch.setitem(sys.modules, "minisweagent.models", models)
    monkeypatch.setitem(sys.modules, "minisweagent.models.litellm_model", litellm_model)
    spec = importlib.util.spec_from_file_location("polar_mini_swe_vanillux_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vanillux_model_exposes_one_persistent_bash_tool(monkeypatch) -> None:
    module = _load_vanillux_module(monkeypatch)
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return "response"

    monkeypatch.setattr(module.litellm, "completion", fake_completion)
    model = module.Vanillux2LitellmModel()

    assert model._query([{"role": "user", "content": "task"}], timeout=12) == "response"
    assert captured["model"] == "openai/Qwen3.5-9B"
    assert captured["temperature"] == 0.7
    assert captured["timeout"] == 12
    assert captured["tools"] == [module.VANILLUX2_BASH_TOOL]
    assert captured["tools"][0]["function"]["name"] == "bash"
    assert "persistent shell" in captured["tools"][0]["function"]["description"]


def test_lease_capability_is_delivered_after_hardening_and_scoped_to_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_vanillux_module(monkeypatch)
    secret_read, secret_write = os.pipe()
    ready_read, ready_write = os.pipe()
    capability = "lease-call-capability"
    os.write(secret_write, (capability + "\n").encode())
    os.close(secret_write)
    monkeypatch.setenv("POLAR_POOL_CALL_CAPABILITY_FD", str(secret_read))
    monkeypatch.setenv("POLAR_POOL_CALL_READY_FD", str(ready_write))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    prctl_operations: list[int] = []

    class Libc:
        def prctl(self, operation, *_args) -> int:
            prctl_operations.append(operation)
            return 0

    monkeypatch.setattr(module.ctypes, "CDLL", lambda *_args, **_kwargs: Libc())
    observed_api_keys: list[str | None] = []

    def fake_completion(**_kwargs):
        observed_api_keys.append(os.environ.get("OPENAI_API_KEY"))
        return "response"

    monkeypatch.setattr(module.litellm, "completion", fake_completion)
    model = module.Vanillux2LitellmModel()

    assert os.read(ready_read, 1) == b"1"
    os.close(ready_read)
    assert prctl_operations == [module._PR_SET_DUMPABLE, module._PR_GET_DUMPABLE]
    assert "POLAR_POOL_CALL_CAPABILITY_FD" not in os.environ
    assert "POLAR_POOL_CALL_READY_FD" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ

    assert model.query([{"role": "user", "content": "task"}]) == "response"
    assert observed_api_keys == [capability]
    assert "OPENAI_API_KEY" not in os.environ


def test_vanillux_cumulative_budget_uses_exact_prompt_growth_and_clamps_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_vanillux_module(monkeypatch)
    tokenize_counts = iter((100, 150, 164))
    tokenize_requests: list[tuple[str, dict, dict]] = []
    completion_requests: list[dict] = []

    class TokenizeResponse:
        def __init__(self, count: int) -> None:
            self._count = count

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "tokens": list(range(self._count)),
                "count": self._count,
                "max_model_len": 262_144,
            }

    class TokenizeClient:
        def post(self, url, *, json, headers, timeout):
            tokenize_requests.append((url, json, headers))
            return TokenizeResponse(next(tokenize_counts))

    def fake_completion(**kwargs):
        completion_requests.append(kwargs)
        return "response"

    monkeypatch.setattr(module.litellm, "client_session", TokenizeClient(), raising=False)
    monkeypatch.setattr(module.litellm, "completion", fake_completion)
    monkeypatch.setenv("OPENAI_API_BASE", "http://polar.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "session-1")
    model = module.Vanillux2LitellmModel(response_token_budget=64)

    first_messages = [{"role": "user", "content": "task"}]
    second_messages = [
        *first_messages,
        {"role": "assistant", "content": "inspect", "tool_calls": []},
        {"role": "tool", "content": "large observation", "tool_call_id": "call-1"},
    ]
    assert model.query(first_messages) == "response"
    assert model.query(second_messages) == "response"
    with pytest.raises(_FakeLimitsExceeded) as exc_info:
        model.query(second_messages)

    assert [request["max_tokens"] for request in completion_requests] == [64, 14]
    assert model.initial_prompt_tokens == 100
    assert model.current_prompt_tokens == 164
    assert model.used_response_tokens == 64
    assert exc_info.value.messages[0]["extra"] == {
        "exit_status": "ResponseTokenBudgetExceeded",
        "submission": "",
        "response_token_budget": 64,
        "used_response_tokens": 64,
    }
    assert len(tokenize_requests) == 3
    url, payload, headers = tokenize_requests[0]
    assert url == "http://polar.invalid/v1/tokenize"
    assert payload["tools"] == [module.VANILLUX2_BASH_TOOL]
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert headers["Authorization"] == "Bearer session-1"
    assert tokenize_requests[1][1]["messages"] == second_messages


def test_vanillux_budget_uses_only_max_completion_tokens_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_vanillux_module(monkeypatch)
    tokenize_counts = iter((100, 150))
    completion_requests: list[dict] = []

    class TokenizeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"count": next(tokenize_counts)}

    class TokenizeClient:
        def post(self, *_args, **_kwargs):
            return TokenizeResponse()

    def fake_completion(**kwargs):
        completion_requests.append(kwargs)
        return "response"

    monkeypatch.setattr(module.litellm, "client_session", TokenizeClient(), raising=False)
    monkeypatch.setattr(module.litellm, "completion", fake_completion)
    monkeypatch.setenv("OPENAI_API_BASE", "http://polar.invalid/v1")
    model = module.Vanillux2LitellmModel(
        response_token_budget=64,
        model_name="openai/pool/gpt-5.5",
        # Recursive mini-SWE config merging supplies both fields.  The modern
        # field must win and remain the field clamped on every turn.
        model_kwargs={
            "max_tokens": 16_384,
            "max_completion_tokens": 40,
            "temperature": 0.7,
        },
    )

    assert model.query([{"role": "user", "content": "task"}]) == "response"
    assert model.query([{"role": "user", "content": "larger task"}]) == "response"

    assert "max_tokens" not in model.config.model_kwargs
    assert [request["max_completion_tokens"] for request in completion_requests] == [40, 14]
    assert all("max_tokens" not in request for request in completion_requests)


def test_vanillux_budget_fails_closed_when_gateway_count_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_vanillux_module(monkeypatch)

    class InvalidResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"count": [3]}

    class InvalidClient:
        def post(self, *args, **kwargs):
            return InvalidResponse()

    monkeypatch.setattr(module.litellm, "client_session", InvalidClient(), raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "http://polar.invalid/v1")
    model = module.Vanillux2LitellmModel(response_token_budget=64)

    with pytest.raises(RuntimeError, match="invalid scalar token count"):
        model.query([{"role": "user", "content": "task"}])


def test_vanillux_tokenize_retries_transient_failures_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_vanillux_module(monkeypatch)
    attempts = 0
    backoffs: list[float] = []

    class FlakyClient:
        def post(self, url, *, json, headers, timeout):
            nonlocal attempts
            attempts += 1
            request = httpx.Request("POST", url)
            if attempts == 1:
                raise httpx.ReadTimeout("tokenizer timed out", request=request)
            if attempts == 2:
                return httpx.Response(
                    503,
                    json={"error": "tokenizer warming up"},
                    request=request,
                )
            return httpx.Response(200, json={"count": 37}, request=request)

    monkeypatch.setattr(module.litellm, "client_session", FlakyClient(), raising=False)
    monkeypatch.setattr(module.time, "sleep", backoffs.append)
    monkeypatch.setenv("OPENAI_API_BASE", "http://polar.invalid/v1")
    model = module.Vanillux2LitellmModel(response_token_budget=64)

    assert model._tokenize_prompt([{"role": "user", "content": "task"}]) == 37
    assert attempts == 3
    assert backoffs == [0.1, 0.2]


def test_vanillux_tokenize_does_not_retry_permanent_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_vanillux_module(monkeypatch)
    attempts = 0
    backoffs: list[float] = []

    class BadRequestClient:
        def post(self, url, *, json, headers, timeout):
            nonlocal attempts
            attempts += 1
            return httpx.Response(
                400,
                json={"error": "invalid tokenize request"},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(module.litellm, "client_session", BadRequestClient(), raising=False)
    monkeypatch.setattr(module.time, "sleep", backoffs.append)
    monkeypatch.setenv("OPENAI_API_BASE", "http://polar.invalid/v1")
    model = module.Vanillux2LitellmModel(response_token_budget=64)

    with pytest.raises(RuntimeError, match=r"1/5 attempts: HTTP 400.*invalid"):
        model._tokenize_prompt([{"role": "user", "content": "task"}])

    assert attempts == 1
    assert backoffs == []


def test_vanillux_model_accepts_exactly_one_action(monkeypatch) -> None:
    module = _load_vanillux_module(monkeypatch)
    model = module.Vanillux2LitellmModel()
    action = {"command": "pwd", "tool_call_id": "call-1"}
    response = SimpleNamespace(
        actions=[action],
        choices=[SimpleNamespace(finish_reason="tool_calls")],
    )

    assert model._parse_actions(response) == [action]


@pytest.mark.parametrize("action_count", [0, 2])
def test_vanillux_model_rejects_any_non_single_action(monkeypatch, action_count: int) -> None:
    module = _load_vanillux_module(monkeypatch)
    model = module.Vanillux2LitellmModel()
    response = SimpleNamespace(
        actions=[{"command": f"echo {index}"} for index in range(action_count)],
        choices=[SimpleNamespace(finish_reason="tool_calls")],
    )

    with pytest.raises(_FakeFormatError) as exc_info:
        model._parse_actions(response)

    message = exc_info.value.messages[0]
    assert message["role"] == "user"
    assert f"received {action_count}" in message["content"]
    assert "finish=tool_calls" in message["content"]
    assert message["extra"]["interrupt_type"] == "FormatError"
