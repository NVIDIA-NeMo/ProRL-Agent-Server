from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "spilot_router_slime_grpo" / "serve_tokenizer.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("spilot_tokenizer_service", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


class FakeTokenizer:
    model_max_length = 262_144

    def __init__(self, encoded=None):
        self.encoded = encoded if encoded is not None else [1, 2, 3, 4]
        self.calls: list[tuple[object, dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.encoded


def test_count_chat_tokens_forwards_tools_and_template_kwargs() -> None:
    tokenizer = FakeTokenizer()
    count = module.count_chat_tokens(
        tokenizer,
        {
            "model": "ignored-served-model",
            "messages": [{"role": "user", "content": "fix it"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )

    assert count == 4
    assert tokenizer.calls == [
        (
            [{"role": "user", "content": "fix it"}],
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": True,
                "tools": [{"type": "function", "function": {"name": "bash"}}],
            },
        )
    ]


def test_count_chat_tokens_accepts_batched_input_ids() -> None:
    tokenizer = FakeTokenizer({"input_ids": [[10, 11, 12]]})
    assert module.count_chat_tokens(tokenizer, {"messages": []}) == 3


def test_count_chat_tokens_normalizes_openai_tool_argument_strings() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "/workspace"},
    ]

    assert module.count_chat_tokens(tokenizer, {"messages": messages}) == 4
    normalized = tokenizer.calls[0][0]
    assert normalized[1]["content"] == ""
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == {
        "command": "pwd"
    }
    assert messages[1]["content"] is None
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"command":"pwd"}'


@pytest.mark.parametrize(
    "payload,error",
    [
        ([], "JSON body"),
        ({}, "messages"),
        ({"messages": [], "tools": {}}, "tools"),
        ({"messages": [], "chat_template_kwargs": []}, "chat_template_kwargs"),
    ],
)
def test_count_chat_tokens_rejects_malformed_payload(payload, error: str) -> None:
    with pytest.raises(module.TokenizerRequestError, match=error):
        module.count_chat_tokens(FakeTokenizer(), payload)


def test_loopback_server_exposes_only_health_workers_and_tokenize() -> None:
    tokenizer = FakeTokenizer([7, 8, 9])
    try:
        server = module.TokenizerHTTPServer(("127.0.0.1", 0), tokenizer)
    except PermissionError:
        pytest.skip("test sandbox forbids loopback socket binding")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(f"http://{host}:{port}/health", timeout=2) as response:
            assert json.load(response)["status"] == "ok"
        with opener.open(f"http://{host}:{port}/workers", timeout=2) as response:
            assert json.load(response) == {"workers": []}
        request = Request(
            f"http://{host}:{port}/v1/tokenize",
            data=json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(request, timeout=2) as response:
            assert json.load(response) == {"count": 3, "max_model_len": 262_144}
        with pytest.raises(HTTPError) as exc_info:
            opener.open(f"http://{host}:{port}/v1/models", timeout=2)
        assert exc_info.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
