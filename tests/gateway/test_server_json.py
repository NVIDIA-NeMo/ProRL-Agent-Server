from __future__ import annotations

from types import SimpleNamespace

from fastapi.responses import Response
import orjson
import pytest
import uvicorn

from polar.gateway.detection import APIType
from polar.gateway import server


def test_serve_exits_nonzero_when_lifespan_shutdown_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_kwargs: dict = {}
    fake_server = SimpleNamespace(
        started=True,
        lifespan=SimpleNamespace(shutdown_failed=True),
        run=lambda: None,
    )
    monkeypatch.setattr(server, "configure_server", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(node=SimpleNamespace(host="127.0.0.1", port=8081)),
    )
    def fake_config(*_args, **kwargs):
        config_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(uvicorn, "Config", fake_config)
    monkeypatch.setattr(uvicorn, "Server", lambda *, config: fake_server)

    with pytest.raises(SystemExit) as error:
        server.serve("topology.yaml")

    assert error.value.code == 1
    assert config_kwargs["timeout_graceful_shutdown"] == 60


@pytest.mark.asyncio
async def test_tokenize_route_rewrites_model_and_never_persists_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forwarded: list[dict] = []

    class Inference:
        async def tokenize(self, request: dict) -> dict:
            forwarded.append(request)
            return {"tokens": [1, 2, 3], "count": 3, "max_model_len": 262144}

    class Storage:
        def save_message(self, *args, **kwargs) -> str:
            raise AssertionError("tokenization must not create a completion record")

    class JsonRequest:
        async def json(self) -> dict:
            return {
                "model": "requested-model",
                "messages": [{"role": "user", "content": "fix it"}],
                "tools": [{"type": "function", "function": {"name": "bash"}}],
            }

    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(
            inference=Inference(),
            node=SimpleNamespace(model_served="served-model"),
            storage=Storage(),
        ),
    )

    response = await server.tokenize_request(JsonRequest())  # type: ignore[arg-type]

    assert isinstance(response, Response)
    assert orjson.loads(response.body) == {
        "count": 3,
        "max_model_len": 262144,
    }
    assert forwarded == [
        {
            "model": "served-model",
            "messages": [{"role": "user", "content": "fix it"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
        }
    ]


@pytest.mark.asyncio
async def test_non_streaming_response_uses_orjson(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "id": "completion",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    }

    class Inference:
        async def completion(self, request: dict) -> dict:
            return payload

    class Storage:
        def save_message(self, *args, **kwargs) -> str:
            return "message"

    class Transformer:
        def transform_response(self, response: dict, original_request: dict) -> dict:
            return response

    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(inference=Inference(), storage=Storage()),
    )

    response = await server._handle_non_streaming(
        APIType.OPENAI_CHAT,
        Transformer(),  # type: ignore[arg-type]
        {"model": "served-model", "messages": []},
        {"model": "requested-model", "messages": []},
        "session",
        original_model="requested-model",
        session_info=None,
    )

    assert isinstance(response, Response)
    assert response.headers["content-type"] == "application/json"
    assert orjson.loads(response.body) == payload


async def _non_streaming_upstream_error(
    monkeypatch: pytest.MonkeyPatch,
    api_type: APIType,
    body: dict,
) -> Response:
    class Inference:
        async def completion(self, request: dict) -> dict:
            raise server.UpstreamHTTPError(400, body)

    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(inference=Inference()),
    )
    return await server._handle_non_streaming(
        api_type,
        object(),  # type: ignore[arg-type]
        {"model": "served-model", "messages": []},
        {"model": "requested-model", "messages": []},
        "session",
        original_model="requested-model",
        session_info=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("api_type", [APIType.OPENAI_CHAT, APIType.OPENAI_RESPONSES])
@pytest.mark.parametrize(
    "message",
    [
        (
            "Requested token count exceeds the model's maximum context length of "
            "50000 tokens. You requested a total of 52233 tokens."
        ),
        (
            "This model's maximum context length is 4097 tokens. However, your "
            "messages resulted in 5000 tokens."
        ),
        "The requested token count exceeds the context window.",
    ],
)
async def test_non_streaming_openai_context_error_is_standardized(
    monkeypatch: pytest.MonkeyPatch,
    api_type: APIType,
    message: str,
) -> None:
    upstream_body = {
        "error": {
            "message": message,
            "type": "BadRequestError",
            "code": "upstream_code",
            "param": None,
            "upstream_detail": "preserved",
        },
        "request_id": "request-1",
    }

    response = await _non_streaming_upstream_error(
        monkeypatch,
        api_type,
        upstream_body,
    )

    assert response.status_code == 400
    assert orjson.loads(response.body) == {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": "messages",
            "upstream_detail": "preserved",
        },
        "request_id": "request-1",
    }


@pytest.mark.asyncio
async def test_non_streaming_openai_other_error_preserves_upstream_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream_body = {
        "error": {
            "message": "The upstream worker is temporarily unavailable.",
            "type": "server_error",
            "code": "worker_unavailable",
            "param": None,
        },
        "request_id": "request-2",
    }

    response = await _non_streaming_upstream_error(
        monkeypatch,
        APIType.OPENAI_CHAT,
        upstream_body,
    )

    assert orjson.loads(response.body) == upstream_body


@pytest.mark.asyncio
async def test_non_streaming_anthropic_context_error_keeps_anthropic_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = "Requested token count exceeds the model's maximum context length."
    response = await _non_streaming_upstream_error(
        monkeypatch,
        APIType.ANTHROPIC,
        {"error": {"message": message, "type": "api_error"}},
    )

    assert orjson.loads(response.body) == {
        "type": "error",
        "error": {"type": "api_error", "message": message},
    }


@pytest.mark.asyncio
async def test_streaming_openai_context_error_preserves_upstream_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = "Requested token count exceeds the model's maximum context length."
    upstream_body = {
        "error": {
            "message": message,
            "type": "BadRequestError",
            "code": None,
            "param": None,
        }
    }

    class Inference:
        async def completion(self, request: dict) -> dict:
            raise server.UpstreamHTTPError(400, upstream_body)

    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(inference=Inference()),
    )
    response = await server._handle_streaming(
        APIType.OPENAI_CHAT,
        object(),  # type: ignore[arg-type]
        {"model": "served-model", "messages": [], "stream": True},
        {"model": "requested-model", "messages": [], "stream": True},
        "session",
        original_model="requested-model",
        session_info=None,
    )

    assert response.status_code == 400
    assert orjson.loads(response.body) == upstream_body
