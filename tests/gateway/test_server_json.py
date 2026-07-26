from __future__ import annotations

from contextlib import contextmanager
import signal
from types import SimpleNamespace

from fastapi.responses import Response
import orjson
import pytest
import uvicorn

from polar.gateway.detection import APIType
from polar.gateway import server


class _FakeUvicornServer:
    def __init__(
        self,
        *,
        signals: list[int] | None = None,
        started: bool = True,
        force_exit: bool = False,
        shutdown_failed: bool = False,
    ) -> None:
        self.started = started
        self.force_exit = force_exit
        self.lifespan = SimpleNamespace(shutdown_failed=shutdown_failed)
        self._captured_signals: list[int] = []
        self._signals_to_capture = list(signals or [])
        self.reraised_signals: list[int] = []

    @contextmanager
    def capture_signals(self):
        yield
        # Model Uvicorn 0.44's post-context behavior without sending a real
        # signal to pytest. The gateway wrapper must clear the source list
        # before control reaches this point.
        self.reraised_signals.extend(reversed(self._captured_signals))
        if self._captured_signals:
            raise AssertionError("Uvicorn would have re-raised a captured signal")

    def run(self) -> None:
        with self.capture_signals():
            self._captured_signals.extend(self._signals_to_capture)


def _patch_gateway_server(
    monkeypatch: pytest.MonkeyPatch,
    fake_server: _FakeUvicornServer,
) -> dict:
    config_kwargs: dict = {}
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
    return config_kwargs


def test_serve_exits_nonzero_when_lifespan_shutdown_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_server = _FakeUvicornServer(shutdown_failed=True)
    config_kwargs = _patch_gateway_server(monkeypatch, fake_server)

    with pytest.raises(SystemExit) as error:
        server.serve("topology.yaml")

    assert error.value.code == 1
    assert config_kwargs["timeout_graceful_shutdown"] == 60


@pytest.mark.parametrize("signals", [[], [signal.SIGTERM]])
def test_serve_accepts_only_normal_or_single_sigterm_clean_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    signals: list[int],
) -> None:
    fake_server = _FakeUvicornServer(signals=signals)
    _patch_gateway_server(monkeypatch, fake_server)

    server.serve("topology.yaml")

    assert fake_server.reraised_signals == []
    assert fake_server._captured_signals == []


@pytest.mark.parametrize(
    "signals",
    [
        [signal.SIGINT],
        [signal.SIGTERM, signal.SIGTERM],
        [signal.SIGTERM, signal.SIGINT],
    ],
)
def test_serve_rejects_unexpected_or_repeated_shutdown_signals(
    monkeypatch: pytest.MonkeyPatch,
    signals: list[int],
) -> None:
    fake_server = _FakeUvicornServer(signals=signals)
    _patch_gateway_server(monkeypatch, fake_server)

    with pytest.raises(SystemExit) as error:
        server.serve("topology.yaml")

    assert error.value.code == 1
    assert fake_server.reraised_signals == []


@pytest.mark.parametrize(
    "server_state",
    [
        {"started": False},
        {"force_exit": True},
    ],
)
def test_serve_rejects_unproven_server_state_after_sigterm(
    monkeypatch: pytest.MonkeyPatch,
    server_state: dict,
) -> None:
    fake_server = _FakeUvicornServer(signals=[signal.SIGTERM], **server_state)
    _patch_gateway_server(monkeypatch, fake_server)

    with pytest.raises(SystemExit) as error:
        server.serve("topology.yaml")

    assert error.value.code == 1


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
