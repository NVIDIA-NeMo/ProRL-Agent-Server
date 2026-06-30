from __future__ import annotations

import asyncio
import json

import httpx
import orjson
import pytest

from polar.gateway.engine import SGLangEngine
from polar.gateway.proxy import InferenceClient
from polar.gateway import proxy as proxy_module


def test_sglang_router_workers_are_used_for_direct_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, str]] = []
    decode_calls = 0
    real_loads = orjson.loads

    def tracked_loads(payload: bytes) -> object:
        nonlocal decode_calls
        decode_calls += 1
        return real_loads(payload)

    monkeypatch.setattr(proxy_module.orjson, "loads", tracked_loads)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.host or "", request.url.path))
        if request.url.path == "/workers":
            return httpx.Response(
                200,
                json={
                    "workers": [
                        {"id": "w0", "url": "http://worker-a:15000", "worker_type": "regular"}
                    ]
                },
            )
        if request.url.host == "worker-a" and request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            assert body["return_prompt_token_ids"] is True
            assert body["return_meta_info"] is True
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "prompt_token_ids": [1, 2, 3],
                            "message": {"role": "assistant", "content": "x"},
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [{"token": "x", "logprob": -0.1, "bytes": [120]}]
                            },
                            "meta_info": {"output_token_logprobs": [[-0.1, 4, "x"]]},
                        }
                    ]
                },
            )
        return httpx.Response(500, json={"error": f"unexpected {request.url}"})

    async def run() -> dict:
        client = InferenceClient("http://router:9000", SGLangEngine())
        client._client = httpx.AsyncClient(
            base_url="http://router:9000",
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.completion({"messages": [{"role": "user", "content": "hi"}]})
        finally:
            await client.close()

    response = asyncio.run(run())
    choice = response["choices"][0]

    assert ("router", "/workers") in requests
    assert ("worker-a", "/v1/chat/completions") in requests
    assert ("router", "/v1/chat/completions") not in requests
    assert choice["input_token_ids"] == [1, 2, 3]
    assert choice["token_ids"] == [4]
    assert choice["logprobs"]["content"][0]["token_id"] == 4
    assert decode_calls == 1


def test_eval_sampling_overrides_reach_sglang_http_request() -> None:
    forwarded: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/workers":
            return httpx.Response(200, json={"workers": []})
        if request.url.path == "/v1/chat/completions":
            forwarded.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        return httpx.Response(500)

    async def run() -> None:
        client = InferenceClient("http://router:9000", SGLangEngine())
        client._client = httpx.AsyncClient(
            base_url="http://router:9000",
            transport=httpx.MockTransport(handler),
        )
        try:
            await client.completion(
                {
                    "model": "served-model",
                    "messages": [{"role": "user", "content": "fix it"}],
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_tokens": 4096,
                    "seed": 1235,
                    "top_k": 32,
                    "min_tokens": 4,
                }
            )
        finally:
            await client.close()

    asyncio.run(run())

    assert len(forwarded) == 1
    body = forwarded[0]
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9
    assert body["max_tokens"] == 4096
    assert body["seed"] == 1235
    assert body["top_k"] == 32
    assert body["min_tokens"] == 4


def test_tokenize_forwards_exact_chat_payload_without_generation_mutation() -> None:
    forwarded: list[dict] = []
    destinations: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        destinations.append((request.url.host or "", request.url.path))
        if request.url.path == "/workers":
            return httpx.Response(
                200,
                json={
                    "workers": [
                        {
                            "id": "w0",
                            "url": "http://worker-a:15000",
                            "worker_type": "regular",
                        }
                    ]
                },
            )
        if request.url.path == "/v1/tokenize":
            forwarded.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"tokens": [1, 2, 3, 4], "count": 4, "max_model_len": 262144},
            )
        return httpx.Response(500)

    async def run() -> dict:
        client = InferenceClient("http://router:9000", SGLangEngine())
        client._client = httpx.AsyncClient(
            base_url="http://router:9000",
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.tokenize(
                {
                    "model": "served-model",
                    "messages": [{"role": "user", "content": "fix it"}],
                    "tools": [{"type": "function", "function": {"name": "bash"}}],
                    "chat_template_kwargs": {"enable_thinking": True},
                }
            )
        finally:
            await client.close()

    response = asyncio.run(run())

    assert response == {
        "tokens": [1, 2, 3, 4],
        "count": 4,
        "max_model_len": 262144,
    }
    assert forwarded == [
        {
            "model": "served-model",
            "messages": [{"role": "user", "content": "fix it"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
            "chat_template_kwargs": {"enable_thinking": True},
        }
    ]
    assert ("worker-a", "/v1/tokenize") in destinations
    assert ("router", "/v1/tokenize") not in destinations
