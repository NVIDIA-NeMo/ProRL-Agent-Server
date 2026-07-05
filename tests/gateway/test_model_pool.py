from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import orjson
import pytest
from starlette.requests import Request

from polar.config import TopologyConfig
from polar.agent.models import AgentSpec
from polar.gateway import server
from polar.gateway.engine import OpenAICompatibleEngine
from polar.gateway.proxy import InferenceClient
from polar.gateway.session import (
    MODEL_POOL_CAPABILITY_SCOPE,
    ROUTER_CAPABILITY_SCOPE,
    SessionRegistry,
)
from polar.gateway.storage import SessionStore
from polar.gateway.transform import TransformManager
from polar.rollout.models import SessionDispatchRequest, SessionStatus


def _request(
    path: str,
    body: dict,
    *,
    authorization: str | None = "Bearer sandbox-session",
    extra_headers: dict[str, str] | None = None,
) -> Request:
    payload = orjson.dumps(body)
    delivered = False

    async def receive() -> dict:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": payload, "more_body": False}

    headers = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))
    for key, value in (extra_headers or {}).items():
        headers.append((key.lower().encode(), value.encode()))
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": headers,
            "client": ("sandbox", 1234),
            "server": ("gateway", 8081),
        },
        receive,
    )


def test_model_pool_topology_is_strict_and_keeps_only_env_name(tmp_path: Path) -> None:
    path = tmp_path / "topology.yaml"
    path.write_text(
        """
gateway:
  nodes:
    - id: node-a
      public_url: http://127.0.0.1:8100
      model_pool:
        - alias: pool/qwen
          model: nvidia/qwen/qwen3.6-27b
          base_url: https://integrate.api.nvidia.com/v1/
          api_key_env: NVIDIA_API_KEY
"""
    )

    candidate = TopologyConfig.load(path).gateway.nodes[0].model_pool[0]

    assert candidate.alias == "pool/qwen"
    assert candidate.model == "nvidia/qwen/qwen3.6-27b"
    assert candidate.base_url == "https://integrate.api.nvidia.com/v1"
    assert candidate.api_key_env == "NVIDIA_API_KEY"
    assert candidate.max_concurrency == 32
    assert "api_key" not in candidate.model_dump()


@pytest.mark.parametrize(
    ("model_pool", "message"),
    [
        (
            [
                {
                    "alias": "qwen",
                    "model": "upstream-qwen",
                    "base_url": "https://example.test/v1",
                    "api_key_env": "NVIDIA_API_KEY",
                }
            ],
            "must start with 'pool/'",
        ),
        (
            [
                {
                    "alias": "pool/qwen",
                    "model": "upstream-qwen",
                    "base_url": "https://example.test/v1",
                    "api_key_env": "NVIDIA_API_KEY",
                },
                {
                    "alias": "pool/qwen",
                    "model": "other-qwen",
                    "base_url": "https://example.test/v1",
                    "api_key_env": "NVIDIA_API_KEY",
                },
            ],
            "Duplicate model pool alias",
        ),
    ],
)
def test_model_pool_topology_rejects_unsafe_or_duplicate_aliases(
    tmp_path: Path,
    model_pool: list[dict],
    message: str,
) -> None:
    path = tmp_path / "topology.yaml"
    path.write_text(
        json.dumps(
            {
                "gateway": {
                    "nodes": [
                        {
                            "id": "node-a",
                            "public_url": "http://127.0.0.1:8100",
                            "model_pool": model_pool,
                        }
                    ]
                }
            }
        )
    )

    with pytest.raises(ValueError, match=message):
        TopologyConfig.load(path)


@pytest.mark.asyncio
async def test_pool_alias_uses_host_auth_and_is_not_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forwarded: list[dict] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer host-nvidia-secret"
        body = json.loads(request.content)
        forwarded.append(body)
        assert body["model"] == "private/upstream-model"
        assert "logprobs" not in body
        assert "return_prompt_token_ids" not in body
        assert "return_meta_info" not in body
        return httpx.Response(
            200,
            json={
                "id": "pool-completion",
                "model": "private/upstream-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    pool_client = InferenceClient(
        "https://nvidia.example/v1",
        OpenAICompatibleEngine(),
        default_headers={"Authorization": "Bearer host-nvidia-secret"},
    )
    pool_client._client = httpx.AsyncClient(
        base_url="https://nvidia.example/v1",
        transport=httpx.MockTransport(upstream),
    )

    class PolicyInference:
        async def completion(self, request: dict) -> dict:
            raise AssertionError("pool aliases must not use local policy inference")

    class Storage:
        def save_message(self, *args, **kwargs) -> None:
            raise AssertionError("pool completions must not enter SessionStore")

    registry = SessionRegistry()
    registry.register(
        "sandbox-session",
        registered=True,
        status=SessionStatus.RUNNING,
    )
    capability = registry.issue_capability(
        "sandbox-session",
        scope=MODEL_POOL_CAPABILITY_SCOPE,
    )
    state = SimpleNamespace(
        inference=PolicyInference(),
        model_pool={
            "pool/M0": server.ModelPoolRoute(
                model="private/upstream-model",
                inference=pool_client,
            )
        },
        storage=Storage(),
        node=SimpleNamespace(model_served="local-router-policy"),
        transform_manager=TransformManager(),
        session_registry=registry,
    )
    monkeypatch.setattr(server, "get_state", lambda: state)

    try:
        response = await server.proxy_request(
            _request(
                "/v1/chat/completions",
                {
                    "model": "pool/M0",
                    "messages": [{"role": "user", "content": "fix it"}],
                    "logprobs": True,
                    "return_prompt_token_ids": True,
                },
                authorization=f"Bearer {capability}",
            ),
            "v1/chat/completions",
        )
    finally:
        await pool_client.close()

    assert response.status_code == 200
    assert orjson.loads(response.body)["model"] == "pool/M0"
    assert forwarded == [
        {
            "model": "private/upstream-model",
            "messages": [{"role": "user", "content": "fix it"}],
            "stream": False,
        }
    ]


@pytest.mark.asyncio
async def test_unknown_pool_alias_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = SessionRegistry()
    registry.register(
        "sandbox-session",
        registered=True,
        status=SessionStatus.RUNNING,
    )
    capability = registry.issue_capability(
        "sandbox-session",
        scope=MODEL_POOL_CAPABILITY_SCOPE,
    )
    state = SimpleNamespace(
        model_pool={},
        session_registry=registry,
    )
    monkeypatch.setattr(server, "get_state", lambda: state)

    response = await server.proxy_request(
        _request(
            "/v1/chat/completions",
            {"model": "pool/not-allowlisted", "messages": []},
            authorization=f"Bearer {capability}",
        ),
        "v1/chat/completions",
    )

    assert response.status_code == 400
    assert orjson.loads(response.body)["error"]["code"] == "unknown_pool_model"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["pool/M0", "router/policy"])
@pytest.mark.parametrize(
    ("credential_state", "expected_status", "expected_code"),
    [
        ("missing", 401, "missing_session_credential"),
        ("unknown", 401, "invalid_session_capability"),
        ("exposed_session_id", 401, "invalid_session_capability"),
        ("wrong_scope", 401, "invalid_session_capability"),
        ("terminal", 403, "inactive_session_credential"),
    ],
)
async def test_privileged_alias_requires_registered_active_session(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    credential_state: str,
    expected_status: int,
    expected_code: str,
) -> None:
    registry = SessionRegistry()
    required_scope = (
        MODEL_POOL_CAPABILITY_SCOPE if model.startswith("pool/") else ROUTER_CAPABILITY_SCOPE
    )
    authorization: str | None = None
    if credential_state == "unknown":
        authorization = "Bearer not-a-real-capability"
    elif credential_state == "exposed_session_id":
        registry.register("sandbox-session", registered=True)
        # This is exactly what the unauthenticated POST /sessions endpoint can
        # create and status endpoints may expose.  The public id is not a
        # privileged capability.
        authorization = "Bearer sandbox-session"
    elif credential_state == "wrong_scope":
        registry.register("sandbox-session", registered=True)
        opposite_scope = (
            ROUTER_CAPABILITY_SCOPE
            if required_scope == MODEL_POOL_CAPABILITY_SCOPE
            else MODEL_POOL_CAPABILITY_SCOPE
        )
        capability = registry.issue_capability(
            "sandbox-session",
            scope=opposite_scope,
        )
        authorization = f"Bearer {capability}"
    elif credential_state == "terminal":
        registry.register(
            "sandbox-session",
            registered=True,
        )
        capability = registry.issue_capability(
            "sandbox-session",
            scope=required_scope,
        )
        registry.set_status("sandbox-session", SessionStatus.COMPLETED)
        authorization = f"Bearer {capability}"

    state = SimpleNamespace(model_pool={}, session_registry=registry)
    monkeypatch.setattr(server, "get_state", lambda: state)
    response = await server.proxy_request(
        _request(
            "/v1/chat/completions",
            {"model": model, "messages": []},
            authorization=authorization,
        ),
        "v1/chat/completions",
    )

    assert response.status_code == expected_status
    assert orjson.loads(response.body)["error"]["code"] == expected_code
    if credential_state == "unknown":
        assert registry.get("sandbox-session") is None


@pytest.mark.asyncio
async def test_publicly_created_session_id_cannot_access_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SessionRegistry()
    storage = SessionStore()
    state = SimpleNamespace(
        model_pool={},
        session_registry=registry,
        storage=storage,
    )
    monkeypatch.setattr(server, "get_state", lambda: state)
    try:
        created = await server.create_session(
            _request(
                "/sessions",
                {"session_id": "attacker-chosen-session"},
                authorization=None,
            )
        )
        assert created.session_id == "attacker-chosen-session"

        response = await server.proxy_request(
            _request(
                "/v1/chat/completions",
                {"model": "pool/M0", "messages": []},
                authorization="Bearer attacker-chosen-session",
            ),
            "v1/chat/completions",
        )
    finally:
        storage.close()

    assert response.status_code == 401
    assert orjson.loads(response.body)["error"]["code"] == "invalid_session_capability"


@pytest.mark.asyncio
async def test_spilot_gateway_dispatch_requires_control_plane_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_manager = SimpleNamespace(dispatch=AsyncMock())
    state = SimpleNamespace(
        node_manager=node_manager,
        node=SimpleNamespace(id="node-a"),
    )
    monkeypatch.setenv("POLAR_CONTROL_PLANE_TOKEN", "trusted-control-token")
    monkeypatch.setattr(server, "get_state", lambda: state)
    payload = SessionDispatchRequest(
        session_id="session-a",
        task_id="task-a",
        instruction="Fix it",
        remaining_timeout_seconds=60,
        agent=AgentSpec(harness="spilot_router"),
    ).model_dump(mode="json")

    denied = await server.create_session(
        _request("/sessions", payload, authorization=None)
    )
    assert denied.status_code == 401
    assert orjson.loads(denied.body)["error"]["code"] == "invalid_control_plane_credential"
    node_manager.dispatch.assert_not_awaited()

    accepted = await server.create_session(
        _request(
            "/sessions",
            payload,
            authorization=None,
            extra_headers={"X-Polar-Control-Token": "trusted-control-token"},
        )
    )
    assert accepted.session_id == "session-a"
    node_manager.dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_local_completion_gets_trusted_router_policy_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: list[dict] = []

    class Inference:
        async def completion(self, request: dict) -> dict:
            return {
                "id": "policy-completion",
                "choices": [{"message": {"role": "assistant", "content": "{}"}}],
            }

    class Storage:
        def save_message(self, *args, **kwargs) -> None:
            saved.append(kwargs)

    class Transformer:
        def transform_response(self, response: dict, original_request: dict) -> dict:
            return response

    state = SimpleNamespace(inference=Inference(), storage=Storage())
    monkeypatch.setattr(server, "get_state", lambda: state)
    session_info = SimpleNamespace(
        session_id="session",
        task_id="task",
        created_at=SimpleNamespace(isoformat=lambda: "now"),
        metadata={"completion_role": "untrusted", "source": "test"},
    )

    await server._handle_non_streaming(
        server.APIType.OPENAI_CHAT,
        Transformer(),  # type: ignore[arg-type]
        {"model": "local-router-policy", "messages": []},
        {"model": "router", "messages": []},
        "session",
        original_model="router/policy",
        session_info=session_info,
        completion_role="router_policy",
    )

    assert saved[0]["metadata"]["completion_role"] == "router_policy"
    assert saved[0]["metadata"]["source"] == "test"


@pytest.mark.asyncio
async def test_reserved_router_alias_is_tagged_through_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: list[dict] = []

    class Inference:
        async def completion(self, request: dict) -> dict:
            return {
                "id": "policy-completion",
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
            }

    class Storage:
        def save_message(self, *args, **kwargs) -> None:
            saved.append(kwargs)

    registry = SessionRegistry()
    registry.register(
        "sandbox-session",
        task_id="task",
        registered=True,
        status=SessionStatus.RUNNING,
        metadata={"completion_role": "untrusted", "source": "test"},
    )
    capability = registry.issue_capability(
        "sandbox-session",
        scope=ROUTER_CAPABILITY_SCOPE,
    )
    state = SimpleNamespace(
        inference=Inference(),
        model_pool={},
        storage=Storage(),
        node=SimpleNamespace(model_served="Qwen/Qwen3.5-9B"),
        transform_manager=TransformManager(),
        session_registry=registry,
    )
    monkeypatch.setattr(server, "get_state", lambda: state)

    response = await server.proxy_request(
        _request(
            "/v1/chat/completions",
            {
                "model": "router/policy",
                "messages": [{"role": "user", "content": "route"}],
            },
            authorization=f"Bearer {capability}",
        ),
        "v1/chat/completions",
    )

    assert response.status_code == 200
    assert saved[0]["model_requested"] == "router/policy"
    assert saved[0]["metadata"]["completion_role"] == "router_policy"
    assert saved[0]["metadata"]["source"] == "test"


@pytest.mark.asyncio
async def test_other_local_completion_is_not_router_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: list[dict] = []

    class Inference:
        async def completion(self, request: dict) -> dict:
            return {
                "id": "ordinary-policy-completion",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }

    class Storage:
        def save_message(self, *args, **kwargs) -> None:
            saved.append(kwargs)

    class Transformer:
        def transform_response(self, response: dict, original_request: dict) -> dict:
            return response

    state = SimpleNamespace(inference=Inference(), storage=Storage())
    monkeypatch.setattr(server, "get_state", lambda: state)

    await server._handle_non_streaming(
        server.APIType.OPENAI_CHAT,
        Transformer(),  # type: ignore[arg-type]
        {"model": "local-router-policy", "messages": []},
        {"model": "cosmetic-model-name", "messages": []},
        "session",
        original_model="cosmetic-model-name",
        session_info=None,
    )

    assert saved[0]["metadata"]["completion_role"] == "policy"


@pytest.mark.asyncio
async def test_unknown_router_alias_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = SessionRegistry()
    registry.register(
        "sandbox-session",
        registered=True,
        status=SessionStatus.RUNNING,
    )
    capability = registry.issue_capability(
        "sandbox-session",
        scope=ROUTER_CAPABILITY_SCOPE,
    )
    state = SimpleNamespace(model_pool={}, session_registry=registry)
    monkeypatch.setattr(server, "get_state", lambda: state)

    response = await server.proxy_request(
        _request(
            "/v1/chat/completions",
            {"model": "router/not-policy", "messages": []},
            authorization=f"Bearer {capability}",
        ),
        "v1/chat/completions",
    )

    assert response.status_code == 400
    assert orjson.loads(response.body)["error"]["code"] == "unknown_router_model"


@pytest.mark.asyncio
async def test_pool_client_enforces_max_concurrency() -> None:
    active = 0
    peak = 0
    saturated = asyncio.Event()
    release = asyncio.Event()

    async def upstream(_: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            saturated.set()
        await release.wait()
        active -= 1
        return httpx.Response(200, json={"choices": []})

    client = InferenceClient(
        "https://nvidia.example/v1",
        OpenAICompatibleEngine(),
        max_concurrency=2,
    )
    client._client = httpx.AsyncClient(
        base_url="https://nvidia.example/v1",
        transport=httpx.MockTransport(upstream),
    )

    tasks = [asyncio.create_task(client.completion({"model": "m"})) for _ in range(5)]
    try:
        await asyncio.wait_for(saturated.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert peak == 2
        release.set()
        await asyncio.gather(*tasks)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()

    assert peak == 2


@pytest.mark.asyncio
async def test_pool_429_is_propagated_without_logging_upstream_body(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class RateLimitedPool:
        async def completion(self, request: dict) -> dict:
            raise server.UpstreamHTTPError(
                429,
                {"error": {"message": "sensitive-upstream-detail"}},
            )

    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(inference=object(), storage=object()),
    )

    with caplog.at_level("WARNING"):
        response = await server._handle_non_streaming(
            server.APIType.OPENAI_CHAT,
            object(),  # type: ignore[arg-type]
            {"model": "private/upstream-model", "messages": []},
            {"model": "pool/M0", "messages": []},
            "session",
            original_model="pool/M0",
            session_info=None,
            inference=RateLimitedPool(),  # type: ignore[arg-type]
            persist_completion=False,
            response_model_alias="pool/M0",
        )

    assert response.status_code == 429
    assert "sensitive-upstream-detail" not in caplog.text
    assert "upstream_http_error" in caplog.text
