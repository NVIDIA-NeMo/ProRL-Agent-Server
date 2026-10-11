from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import yaml

from polar.config import TopologyConfig
from polar.gateway import server
from polar.gateway.detection import APIType
from polar.gateway.engine import SGLangEngine, VLLMEngine
from polar.gateway.proxy import InferenceClient, UpstreamHTTPError
from polar.gateway.transform.openai_chat import OpenAIChatTransformer


def _completion_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "model": "test-model",
            "prompt_token_ids": [1, 2],
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                        "reasoning": "because",
                    },
                    "finish_reason": "stop",
                    "token_ids": [10, 11],
                    "logprobs": {
                        "content": [
                            {"token": "o", "logprob": -0.1},
                            {"token": "k", "logprob": -0.2},
                        ]
                    },
                }
            ],
        },
    )


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    scheduler: str = "thunderagent",
) -> tuple[InferenceClient, httpx.AsyncClient]:
    client = InferenceClient(
        "http://inference:9000",
        VLLMEngine(),
        scheduler=scheduler,
        program_namespace="node-a",
    )
    transport_client = httpx.AsyncClient(
        base_url="http://inference:9000",
        transport=httpx.MockTransport(handler),
    )
    client._client = transport_client
    return client, transport_client


@pytest.mark.asyncio
async def test_thunderagent_requires_session_identity() -> None:
    client, _ = _make_client(lambda _: _completion_response())
    try:
        with pytest.raises(ValueError, match="require a session_id"):
            await client.completion({"messages": []})
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_release_before_first_completion_fences_session() -> None:
    client, _ = _make_client(lambda _: _completion_response())
    assert await client.release_program("session-a") is False
    with pytest.raises(UpstreamHTTPError) as exc_info:
        await client.completion({"messages": []}, session_id="session-a")
    assert exc_info.value.status_code == 409
    await client.close()


@pytest.mark.asyncio
async def test_default_scheduler_sends_no_session_header_and_release_is_noop() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["program_id"] == "caller-controlled"
        assert body["extra_body"]["program_id"] == "caller-controlled"
        return _completion_response()

    client, _ = _make_client(handler, scheduler="none")
    try:
        await client.completion(
            {
                "messages": [],
                "program_id": "caller-controlled",
                "extra_body": {"program_id": "caller-controlled"},
            },
            session_id="session-a",
        )
        assert await client.release_program("session-a") is False
    finally:
        await client.close()

    assert [request.url.path for request in requests] == ["/v1/chat/completions"]
    assert "x-session-id" not in requests[0].headers


@pytest.mark.asyncio
async def test_thunderagent_session_identity_and_vllm_training_fields_are_preserved() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/chat/completions":
            return _completion_response()
        if request.url.path == "/programs/release":
            return httpx.Response(200, json={"released": True})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client, _ = _make_client(handler)
    responses: list[dict] = []
    try:
        for session_id in ("session-a", "session-a", "session-b"):
            responses.append(
                await client.completion(
                    {
                        "messages": [],
                        "program_id": "caller-controlled",
                        "extra_body": {"program_id": "caller-controlled"},
                    },
                    session_id=session_id,
                )
            )
        assert await client.release_program("session-a") is True
        assert await client.release_program("session-b") is True
    finally:
        await client.close()

    completions = [request for request in requests if request.url.path == "/v1/chat/completions"]
    assert [request.headers["x-session-id"] for request in completions] == [
        "node-a:session-a",
        "node-a:session-a",
        "node-a:session-b",
    ]
    for request in completions:
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["logprobs"] is True
        assert body["return_token_ids"] is True
        assert body["top_logprobs"] == 0
        assert "program_id" not in body
        assert "program_id" not in body["extra_body"]

    for response in responses:
        choice = response["choices"][0]
        assert response["prompt_token_ids"] == [1, 2]
        assert choice["token_ids"] == [10, 11]
        assert [entry["token_id"] for entry in choice["logprobs"]["content"]] == [10, 11]
        assert choice["message"]["reasoning_content"] == "because"


@pytest.mark.asyncio
async def test_gateway_namespace_isolates_equal_session_ids() -> None:
    program_ids: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            program_ids.append(request.headers["x-session-id"])
            return _completion_response()
        program_ids.append(json.loads(request.content)["program_id"])
        return httpx.Response(200, json={"released": True})

    for namespace in ("node-a", "node-b"):
        client = InferenceClient(
            "http://inference:9000",
            VLLMEngine(),
            scheduler="thunderagent",
            program_namespace=namespace,
        )
        client._client = httpx.AsyncClient(
            base_url="http://inference:9000",
            transport=httpx.MockTransport(handler),
        )
        await client.completion({"messages": []}, session_id="same-session")
        await client.release_program("same-session")
        await client.close()

    assert program_ids == [
        "node-a:same-session",
        "node-a:same-session",
        "node-b:same-session",
        "node-b:same-session",
    ]


@pytest.mark.asyncio
async def test_thunderagent_preserves_sglang_training_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/programs/release":
            return httpx.Response(200, json={"released": True})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "prompt_token_ids": [1, 2],
                        "message": {"role": "assistant", "content": "ok"},
                        "logprobs": {
                            "content": [
                                {"token": "o", "logprob": -0.1},
                                {"token": "k", "logprob": -0.2},
                            ]
                        },
                        "meta_info": {
                            "output_token_logprobs": [
                                [-0.1, 10, "o"],
                                [-0.2, 11, "k"],
                            ]
                        },
                    }
                ]
            },
        )

    client = InferenceClient(
        "http://inference:9000",
        SGLangEngine(),
        scheduler="thunderagent",
        program_namespace="node-a",
    )
    client._client = httpx.AsyncClient(
        base_url="http://inference:9000",
        transport=httpx.MockTransport(handler),
    )
    try:
        response = await client.completion({"messages": []}, session_id="session-a")
        assert await client.release_program("session-a") is True
    finally:
        await client.close()

    completion = requests[0]
    body = json.loads(completion.content)
    assert completion.headers["x-session-id"] == "node-a:session-a"
    assert body["return_prompt_token_ids"] is True
    assert body["return_meta_info"] is True
    choice = response["choices"][0]
    assert choice["input_token_ids"] == [1, 2]
    assert choice["token_ids"] == [10, 11]
    assert [entry["token_id"] for entry in choice["logprobs"]["content"]] == [10, 11]


@pytest.mark.asyncio
async def test_concurrent_release_of_one_program_posts_once() -> None:
    release_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return _completion_response()
        if request.url.path == "/programs/release":
            release_requests.append(request)
            return httpx.Response(200, json={"released": True})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client, _ = _make_client(handler)
    try:
        await client.completion({"messages": []}, session_id="session-a")
        results = await asyncio.gather(
            client.release_program("session-a"),
            client.release_program("session-a"),
        )
    finally:
        await client.close()

    assert sorted(results) == [False, True]
    assert len(release_requests) == 1
    assert json.loads(release_requests[0].content) == {"program_id": "node-a:session-a"}


@pytest.mark.asyncio
async def test_release_waits_for_paused_completion_and_fences_late_requests() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/v1/chat/completions":
            return _completion_response()
        return httpx.Response(200, json={"released": True})

    client, _ = _make_client(handler)
    await client.pause_generation()
    completion_task = asyncio.create_task(
        client.completion({"messages": []}, session_id="session-a")
    )
    await asyncio.sleep(0)
    assert client._program_inflight == {"session-a": 1}

    release_task = asyncio.create_task(client.release_program("session-a"))
    await asyncio.sleep(0)
    assert not release_task.done()

    await client.resume_generation()
    await completion_task
    assert await release_task is True

    assert paths == ["/v1/chat/completions", "/programs/release"]
    with pytest.raises(UpstreamHTTPError) as exc_info:
        await client.completion({"messages": []}, session_id="session-a")
    assert exc_info.value.status_code == 409
    await client.close()


@pytest.mark.asyncio
async def test_failed_release_can_be_retried() -> None:
    release_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal release_attempts
        if request.url.path == "/v1/chat/completions":
            return _completion_response()
        if request.url.path == "/programs/release":
            release_attempts += 1
            if release_attempts == 1:
                return httpx.Response(500, json={"error": {"message": "release failed"}})
            return httpx.Response(200, json={"released": True})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client, _ = _make_client(handler)
    try:
        await client.completion({"messages": []}, session_id="session-a")
        with pytest.raises(UpstreamHTTPError, match="release failed"):
            await client.release_program("session-a")
        assert await client.release_program("session-a") is True
    finally:
        await client.close()

    assert release_attempts == 2


@pytest.mark.asyncio
async def test_concurrent_waiter_retries_failed_release() -> None:
    first_release_started = asyncio.Event()
    finish_first_release = asyncio.Event()
    release_attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal release_attempts
        if request.url.path == "/v1/chat/completions":
            return _completion_response()
        release_attempts += 1
        if release_attempts == 1:
            first_release_started.set()
            await finish_first_release.wait()
            return httpx.Response(500, json={"error": {"message": "release failed"}})
        return httpx.Response(200, json={"released": True})

    client = InferenceClient(
        "http://inference:9000",
        VLLMEngine(),
        scheduler="thunderagent",
        program_namespace="node-a",
    )
    client._client = httpx.AsyncClient(
        base_url="http://inference:9000",
        transport=httpx.MockTransport(handler),
    )
    await client.completion({"messages": []}, session_id="session-a")
    first = asyncio.create_task(client.release_program("session-a"))
    await first_release_started.wait()
    second = asyncio.create_task(client.release_program("session-a"))
    finish_first_release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)

    assert release_attempts == 2
    assert sum(result is True for result in results) == 1
    assert sum(isinstance(result, UpstreamHTTPError) for result in results) == 1
    await client.close()


@pytest.mark.asyncio
async def test_close_releases_each_active_program_once() -> None:
    released_programs: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return _completion_response()
        if request.url.path == "/programs/release":
            released_programs.append(json.loads(request.content)["program_id"])
            return httpx.Response(200, json={"released": True})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client, transport_client = _make_client(handler)
    await client.completion({"messages": []}, session_id="session-a")
    await client.completion({"messages": []}, session_id="session-a")
    await client.completion({"messages": []}, session_id="session-b")

    await client.close()
    await client.close()

    assert Counter(released_programs) == Counter({"node-a:session-a": 1, "node-a:session-b": 1})
    assert transport_client.is_closed


@pytest.mark.asyncio
async def test_close_releases_program_while_completion_is_in_flight() -> None:
    completion_started = asyncio.Event()
    allow_completion = asyncio.Event()
    released_programs: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            completion_started.set()
            await allow_completion.wait()
            return _completion_response()
        if request.url.path == "/programs/release":
            released_programs.append(json.loads(request.content)["program_id"])
            return httpx.Response(200, json={"released": True})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = InferenceClient(
        "http://inference:9000",
        VLLMEngine(),
        scheduler="thunderagent",
        program_namespace="node-a",
    )
    client._client = httpx.AsyncClient(
        base_url="http://inference:9000",
        transport=httpx.MockTransport(handler),
    )
    completion_task = asyncio.create_task(
        client.completion({"messages": []}, session_id="session-a")
    )
    await completion_started.wait()
    completion_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await completion_task

    await client.close()
    allow_completion.set()

    assert released_programs == ["node-a:session-a"]


@pytest.mark.asyncio
async def test_close_drains_completion_waiting_at_pause_gate() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/v1/chat/completions":
            return _completion_response()
        return httpx.Response(200, json={"released": True})

    client, _ = _make_client(handler)
    await client.pause_generation()
    completion_task = asyncio.create_task(
        client.completion({"messages": []}, session_id="session-a")
    )
    await asyncio.sleep(0)
    assert client._program_inflight == {"session-a": 1}

    await asyncio.gather(completion_task, client.close())

    assert paths == ["/v1/chat/completions", "/programs/release"]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_gateway_handlers_forward_session_identity(
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    response = _completion_response().json()
    inference = SimpleNamespace(completion=AsyncMock(return_value=response))
    storage = SimpleNamespace(save_message=Mock())
    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(
            inference=inference,
            storage=storage,
        ),
    )
    openai_request = {
        "model": "served-model",
        "messages": [],
        "stream": streaming,
    }
    handler = server._handle_streaming if streaming else server._handle_non_streaming

    await handler(
        APIType.OPENAI_CHAT,
        OpenAIChatTransformer(),
        openai_request,
        {"model": "requested-model", "stream": streaming},
        "session-a",
        original_model="requested-model",
        session_info=None,
    )

    expected_request = {**openai_request, "stream": False} if streaming else openai_request
    inference.completion.assert_awaited_once_with(expected_request, session_id="session-a")


@pytest.mark.parametrize(
    ("scheduler", "expects_releaser"),
    [("none", False), ("thunderagent", True)],
)
@pytest.mark.asyncio
async def test_build_state_wires_scheduler_opt_in(
    tmp_path,
    scheduler: str,
    expects_releaser: bool,
) -> None:
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        yaml.safe_dump(
            {
                "gateway": {
                    "nodes": [
                        {
                            "id": "node-a",
                            "public_url": "http://gateway:8100",
                            "inference": {
                                "engine": "vllm",
                                "scheduler": scheduler,
                                "base_url": "http://inference:9000",
                            },
                        }
                    ]
                }
            }
        )
    )

    state = server._build_state(TopologyConfig.load(topology_path), "node-a")
    try:
        assert state.inference.scheduler == scheduler
        assert state.inference.program_namespace == "node-a"
        assert (state.node_manager._program_releaser is not None) is expects_releaser
    finally:
        await state.node_manager.close()
        await state.inference.close()
        state.storage.close()
        await state.completion_writer.close()


@pytest.mark.asyncio
async def test_lifespan_releases_inference_if_node_shutdown_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_manager = SimpleNamespace(
        start=AsyncMock(),
        close=AsyncMock(side_effect=RuntimeError("node shutdown failed")),
    )
    inference = SimpleNamespace(close=AsyncMock())
    storage = SimpleNamespace(close=Mock())
    completion_writer = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(
            node_manager=node_manager,
            inference=inference,
            storage=storage,
            completion_writer=completion_writer,
        ),
    )

    with pytest.raises(RuntimeError, match="node shutdown failed"):
        async with server._lifespan(None):
            pass

    inference.close.assert_awaited_once()
    storage.close.assert_called_once()
    completion_writer.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_removes_completion_persisted_while_release_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_started = asyncio.Event()
    finish_release = asyncio.Event()
    storage = server.SessionStore()
    registry = server.SessionRegistry()
    registry.register("session-a")
    storage.ensure_session("session-a", None, None, None)

    async def release_program(session_id: str) -> None:
        release_started.set()
        await finish_release.wait()

    node_manager = SimpleNamespace(
        cancel=AsyncMock(return_value=True),
        release_program=release_program,
    )
    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(
            node_manager=node_manager,
            session_registry=registry,
            storage=storage,
        ),
    )

    delete_task = asyncio.create_task(server.delete_session("session-a"))
    await release_started.wait()
    storage.save_message(
        "session-a",
        {"model": "test"},
        {"choices": []},
    )
    finish_release.set()
    response = await delete_task

    assert response.messages_deleted == 1
    assert registry.get("session-a") is None
    assert storage.get_session_metadata("session-a") is None
