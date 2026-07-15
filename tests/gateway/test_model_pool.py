from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import orjson
import pytest
from starlette.requests import ClientDisconnect, Request

from polar.config import TopologyConfig
from polar.agent.models import AgentSpec
from polar.gateway import server
from polar.gateway.engine import OpenAICompatibleEngine
from polar.gateway.episode_admission import (
    EpisodeReleaseDraining,
    ModelPoolEpisodeAdmission,
)
from polar.gateway.proxy import InferenceClient
from polar.gateway.session import (
    MODEL_POOL_ADMISSION_CAPABILITY_SCOPE,
    MODEL_POOL_CAPABILITY_SCOPE,
    ROUTER_CAPABILITY_SCOPE,
    SessionRegistry,
)
from polar.gateway.storage import SessionStore
from polar.gateway.transform import TransformManager
from polar.rollout.models import NodeStageMetrics, SessionDispatchRequest, SessionStatus


def _request(
    path: str,
    body: dict,
    *,
    method: str = "POST",
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
            "method": method,
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
    assert candidate.max_active_episodes is None
    assert "api_key" not in candidate.model_dump()


def test_model_pool_episode_cap_cannot_exceed_request_cap(tmp_path: Path) -> None:
    path = tmp_path / "topology.yaml"
    path.write_text(
        """
gateway:
  nodes:
    - id: node-a
      public_url: http://127.0.0.1:8100
      model_pool:
        - alias: pool/qwen
          model: upstream-qwen
          base_url: https://example.test/v1
          api_key_env: NVIDIA_API_KEY
          max_concurrency: 2
          max_active_episodes: 3
"""
    )

    with pytest.raises(ValueError, match="cannot exceed max_concurrency"):
        TopologyConfig.load(path)


@pytest.mark.asyncio
async def test_gateway_health_is_503_after_fatal_retained_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission_health = {
        "healthy": False,
        "fatal_retained": True,
        "retained_session_ids": ["session-retained"],
    }
    state = SimpleNamespace(
        node_manager=SimpleNamespace(
            stage_metrics=AsyncMock(return_value=NodeStageMetrics()),
            episode_admission_health=lambda: admission_health,
        ),
        episode_admission=SimpleNamespace(
            snapshot=AsyncMock(
                return_value={"pool/qwen": {"cap": 1, "active": 1, "queued": 0}}
            )
        ),
        inference=SimpleNamespace(health=AsyncMock(return_value={"status": "ok"})),
        node=SimpleNamespace(
            id="node-a",
            public_url="http://gateway.test",
            max_init_workers=1,
            max_run_workers=1,
            max_postrun_workers=1,
        ),
        completion_writer=SimpleNamespace(stats=lambda: {}),
        session_registry=SimpleNamespace(
            active_status_counts=lambda: {},
            active_sessions=lambda: [],
        ),
    )
    monkeypatch.setattr(server, "get_state", lambda: state)

    response = await server.health()

    assert response.status_code == 503
    body = json.loads(response.body)
    assert body["status"] == "error"
    assert body["model_pool_episode_admission_health"] == admission_health


@pytest.mark.asyncio
async def test_spilot_tokenize_requires_exact_active_pool_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenize = AsyncMock(return_value={"count": 3, "max_model_len": 1024})
    registry = SessionRegistry()
    registry.register(
        "tokenize-session",
        registered=True,
        status=SessionStatus.RUNNING,
    )
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    grant = await admission.acquire(
        session_id="tokenize-session",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )
    session_pool_capability = registry.issue_capability(
        "tokenize-session",
        scope=MODEL_POOL_CAPABILITY_SCOPE,
    )
    state = SimpleNamespace(
        inference=SimpleNamespace(tokenize=tokenize),
        model_pool={
            "pool/qwen": server.ModelPoolRoute(
                model="upstream-qwen",
                inference=SimpleNamespace(),  # type: ignore[arg-type]
                max_active_episodes=1,
            )
        },
        episode_admission=admission,
        session_registry=registry,
        node=SimpleNamespace(model_served="local-router-policy"),
    )
    monkeypatch.setattr(server, "get_state", lambda: state)

    denied = await server.tokenize_request(
        _request(
            "/v1/tokenize",
            {"model": "pool/qwen", "messages": []},
            authorization=None,
        )
    )
    session_capability_denied = await server.tokenize_request(
        _request(
            "/v1/tokenize",
            {"model": "pool/qwen", "messages": []},
            authorization=f"Bearer {session_pool_capability}",
        )
    )
    accepted = await server.tokenize_request(
        _request(
            "/v1/tokenize",
            {"model": "pool/qwen", "messages": []},
            authorization=f"Bearer {grant.call_capability}",
        )
    )

    assert denied.status_code == 401
    assert orjson.loads(denied.body)["error"]["code"] == (
        "missing_pool_call_capability"
    )
    assert session_capability_denied.status_code == 401
    assert orjson.loads(session_capability_denied.body)["error"]["code"] == (
        "invalid_pool_call_capability"
    )
    assert accepted.status_code == 200
    assert orjson.loads(accepted.body) == {"count": 3, "max_model_len": 1024}
    tokenize.assert_awaited_once()
    # The tokenization request was drained immediately, so lease release does
    # not wait for GC or an unrelated response lifecycle.
    assert await admission.release(
        session_id="tokenize-session",
        lease_id=grant.lease_id,
    ) is True


@pytest.mark.asyncio
async def test_uncapped_spilot_tokenize_uses_session_pool_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenize = AsyncMock(return_value={"count": 7, "max_model_len": 2048})
    registry = SessionRegistry()
    registry.register(
        "uncapped-tokenize-session",
        registered=True,
        status=SessionStatus.RUNNING,
    )
    pool_capability = registry.issue_capability(
        "uncapped-tokenize-session",
        scope=MODEL_POOL_CAPABILITY_SCOPE,
    )
    router_capability = registry.issue_capability(
        "uncapped-tokenize-session",
        scope=ROUTER_CAPABILITY_SCOPE,
    )
    state = SimpleNamespace(
        inference=SimpleNamespace(tokenize=tokenize),
        model_pool={
            "pool/qwen": server.ModelPoolRoute(
                model="upstream-qwen",
                inference=SimpleNamespace(),  # type: ignore[arg-type]
                max_active_episodes=None,
            )
        },
        session_registry=registry,
        node=SimpleNamespace(model_served="local-router-policy"),
    )
    monkeypatch.setattr(server, "get_state", lambda: state)

    missing = await server.tokenize_request(
        _request(
            "/v1/tokenize",
            {"model": "pool/qwen", "messages": []},
            authorization=None,
        )
    )
    wrong_scope = await server.tokenize_request(
        _request(
            "/v1/tokenize",
            {"model": "pool/qwen", "messages": []},
            authorization=f"Bearer {router_capability}",
        )
    )
    accepted = await server.tokenize_request(
        _request(
            "/v1/tokenize",
            {"model": "pool/qwen", "messages": []},
            authorization=f"Bearer {pool_capability}",
        )
    )

    assert missing.status_code == 401
    assert orjson.loads(missing.body)["error"]["code"] == "missing_session_credential"
    assert wrong_scope.status_code == 401
    assert orjson.loads(wrong_scope.body)["error"]["code"] == (
        "invalid_session_capability"
    )
    assert accepted.status_code == 200
    assert orjson.loads(accepted.body) == {"count": 7, "max_model_len": 2048}
    tokenize.assert_awaited_once_with(
        {"model": "local-router-policy", "messages": []}
    )


@pytest.mark.asyncio
async def test_cancelled_tokenize_finishes_lease_cleanup_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream_started = asyncio.Event()
    block_upstream = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def tokenize(_request: dict) -> dict:
        upstream_started.set()
        await block_upstream.wait()
        return {"count": 1}

    registry = SessionRegistry()
    registry.register("tokenize-cancel", registered=True, status=SessionStatus.RUNNING)
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    grant = await admission.acquire(
        session_id="tokenize-cancel",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )
    real_end_request = admission.end_request

    async def slow_end_request(handle) -> None:
        cleanup_started.set()
        await release_cleanup.wait()
        await real_end_request(handle)

    monkeypatch.setattr(admission, "end_request", slow_end_request)
    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(
            inference=SimpleNamespace(tokenize=tokenize),
            model_pool={
                "pool/qwen": server.ModelPoolRoute(
                    model="upstream-qwen",
                    inference=SimpleNamespace(),  # type: ignore[arg-type]
                    max_active_episodes=1,
                )
            },
            episode_admission=admission,
            session_registry=registry,
            node=SimpleNamespace(model_served="local-router-policy"),
        ),
    )

    task = asyncio.create_task(
        server.tokenize_request(
            _request(
                "/v1/tokenize",
                {"model": "pool/qwen", "messages": []},
                authorization=f"Bearer {grant.call_capability}",
            )
        )
    )
    await asyncio.wait_for(upstream_started.wait(), timeout=1)
    task.cancel()
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await admission.release(
        session_id="tokenize-cancel",
        lease_id=grant.lease_id,
    ) is True


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
async def test_episode_lease_api_authenticates_separately_and_gates_pool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forwarded: list[dict] = []

    class Inference:
        async def completion(self, request: dict) -> dict:
            forwarded.append(request)
            return {
                "id": "pool-completion",
                "model": "upstream-qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            }

    class Storage:
        def save_message(self, *args, **kwargs) -> None:
            raise AssertionError("pool completions must not be persisted")

    registry = SessionRegistry()
    registry.register(
        "sandbox-session",
        registered=True,
        status=SessionStatus.RUNNING,
    )
    pool_capability = registry.issue_capability(
        "sandbox-session", scope=MODEL_POOL_CAPABILITY_SCOPE
    )
    admission_capability = registry.issue_capability(
        "sandbox-session", scope=MODEL_POOL_ADMISSION_CAPABILITY_SCOPE
    )
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    state = SimpleNamespace(
        inference=SimpleNamespace(),
        model_pool={
            "pool/qwen": server.ModelPoolRoute(
                model="upstream-qwen",
                inference=Inference(),  # type: ignore[arg-type]
                max_active_episodes=1,
            )
        },
        episode_admission=admission,
        storage=Storage(),
        node=SimpleNamespace(model_served="local-router-policy"),
        transform_manager=TransformManager(),
        session_registry=registry,
    )
    monkeypatch.setattr(server, "get_state", lambda: state)

    wrong_scope = await server.acquire_model_pool_episode_lease(
        _request(
            "/internal/model-pool/episode-leases/acquire",
            {},
            authorization=f"Bearer {pool_capability}",
        ),
        server.EpisodeLeaseAcquireRequest(
            model="pool/qwen",
            attempt_id="0:solve",
            wait_timeout_seconds=1,
        ),
    )
    assert wrong_scope.status_code == 401

    denied = await server.proxy_request(
        _request(
            "/v1/chat/completions",
            {"model": "pool/qwen", "messages": []},
            authorization=f"Bearer {pool_capability}",
        ),
        "v1/chat/completions",
    )
    assert denied.status_code == 401
    assert orjson.loads(denied.body)["error"]["code"] == "invalid_pool_call_capability"
    assert forwarded == []

    granted = await server.acquire_model_pool_episode_lease(
        _request(
            "/internal/model-pool/episode-leases/acquire",
            {},
            authorization=f"Bearer {admission_capability}",
        ),
        server.EpisodeLeaseAcquireRequest(
            model="pool/qwen",
            attempt_id="0:solve",
            wait_timeout_seconds=1,
        ),
    )
    assert granted["model"] == "pool/qwen"

    allowed = await server.proxy_request(
        _request(
            "/v1/chat/completions",
            {"model": "pool/qwen", "messages": []},
            authorization=f"Bearer {granted['call_capability']}",
        ),
        "v1/chat/completions",
    )
    assert allowed.status_code == 200
    assert len(forwarded) == 1

    released = await server.release_model_pool_episode_lease(
        _request(
            "/internal/model-pool/episode-leases/release",
            {},
            authorization=f"Bearer {admission_capability}",
        ),
        server.EpisodeLeaseReleaseRequest(lease_id=granted["lease_id"]),
    )
    assert released == {"released": True}


@pytest.mark.asyncio
async def test_blocked_pool_upstream_keeps_episode_slot_until_request_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream_started = asyncio.Event()
    upstream_finish = asyncio.Event()

    class Inference:
        async def completion(self, _request: dict) -> dict:
            upstream_started.set()
            await upstream_finish.wait()
            return {
                "id": "pool-completion",
                "model": "upstream-qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            }

    class Storage:
        def save_message(self, *_args, **_kwargs) -> None:
            raise AssertionError("pool completions must not be persisted")

    registry = SessionRegistry()
    admission_capabilities: dict[str, str] = {}
    for session_id in ("session-one", "session-two"):
        registry.register(
            session_id,
            registered=True,
            status=SessionStatus.RUNNING,
        )
        admission_capabilities[session_id] = registry.issue_capability(
            session_id,
            scope=MODEL_POOL_ADMISSION_CAPABILITY_SCOPE,
        )
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    state = SimpleNamespace(
        inference=SimpleNamespace(),
        model_pool={
            "pool/qwen": server.ModelPoolRoute(
                model="upstream-qwen",
                inference=Inference(),  # type: ignore[arg-type]
                max_active_episodes=1,
            )
        },
        episode_admission=admission,
        storage=Storage(),
        node=SimpleNamespace(model_served="local-router-policy"),
        transform_manager=TransformManager(),
        session_registry=registry,
    )
    monkeypatch.setattr(server, "get_state", lambda: state)

    first = await server.acquire_model_pool_episode_lease(
        _request(
            "/internal/model-pool/episode-leases/acquire",
            {},
            authorization=f"Bearer {admission_capabilities['session-one']}",
        ),
        server.EpisodeLeaseAcquireRequest(
            model="pool/qwen",
            attempt_id="0:solve",
            wait_timeout_seconds=1,
        ),
    )
    pool_task = asyncio.create_task(
        server.proxy_request(
            _request(
                "/v1/chat/completions",
                {"model": "pool/qwen", "messages": []},
                authorization=f"Bearer {first['call_capability']}",
            ),
            "v1/chat/completions",
        )
    )
    await asyncio.wait_for(upstream_started.wait(), timeout=1)

    queued_acquire = asyncio.create_task(
        server.acquire_model_pool_episode_lease(
            _request(
                "/internal/model-pool/episode-leases/acquire",
                {},
                authorization=f"Bearer {admission_capabilities['session-two']}",
            ),
            server.EpisodeLeaseAcquireRequest(
                model="pool/qwen",
                attempt_id="0:solve",
                wait_timeout_seconds=1,
            ),
        )
    )
    await asyncio.sleep(0)
    assert not queued_acquire.done()

    draining = await server.release_model_pool_episode_lease(
        _request(
            "/internal/model-pool/episode-leases/release",
            {},
            authorization=f"Bearer {admission_capabilities['session-one']}",
        ),
        server.EpisodeLeaseReleaseRequest(
            lease_id=first["lease_id"],
            wait_timeout_seconds=0.01,
        ),
    )
    assert draining.status_code == 202
    assert orjson.loads(draining.body) == {"released": False, "draining": True}
    assert not queued_acquire.done()

    rejected = await server.proxy_request(
        _request(
            "/v1/chat/completions",
            {"model": "pool/qwen", "messages": []},
            authorization=f"Bearer {first['call_capability']}",
        ),
        "v1/chat/completions",
    )
    assert rejected.status_code == 401
    assert orjson.loads(rejected.body)["error"]["code"] == "invalid_pool_call_capability"

    upstream_finish.set()
    response = await asyncio.wait_for(pool_task, timeout=1)
    assert response.status_code == 200
    second = await asyncio.wait_for(queued_acquire, timeout=1)
    assert second["model"] == "pool/qwen"

    release_retry = await server.release_model_pool_episode_lease(
        _request(
            "/internal/model-pool/episode-leases/release",
            {},
            authorization=f"Bearer {admission_capabilities['session-one']}",
        ),
        server.EpisodeLeaseReleaseRequest(lease_id=first["lease_id"]),
    )
    assert release_retry == {"released": False}
    await admission.release(
        session_id="session-two",
        lease_id=second["lease_id"],
    )


@pytest.mark.asyncio
async def test_runtime_destroyed_fallback_cancels_non_streaming_pool_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream_started = asyncio.Event()
    upstream_cancelled = asyncio.Event()

    class Inference:
        async def completion(self, _request: dict) -> dict:
            upstream_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                upstream_cancelled.set()

    registry = SessionRegistry()
    registry.register("owner", registered=True, status=SessionStatus.RUNNING)
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    first = await admission.acquire(
        session_id="owner",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )
    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(
            inference=SimpleNamespace(),
            model_pool={
                "pool/qwen": server.ModelPoolRoute(
                    model="upstream-qwen",
                    inference=Inference(),  # type: ignore[arg-type]
                    max_active_episodes=1,
                )
            },
            episode_admission=admission,
            storage=SimpleNamespace(),
            node=SimpleNamespace(model_served="local-router-policy"),
            transform_manager=TransformManager(),
            session_registry=registry,
        ),
    )
    request_task = asyncio.create_task(
        server.proxy_request(
            _request(
                "/v1/chat/completions",
                {"model": "pool/qwen", "messages": []},
                authorization=f"Bearer {first.call_capability}",
            ),
            "v1/chat/completions",
        )
    )
    await asyncio.wait_for(upstream_started.wait(), timeout=1)
    queued = asyncio.create_task(
        admission.acquire(
            session_id="next",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=1,
        )
    )

    await admission.release_after_runtime_destroyed(
        "owner",
        runtime_destroyed=True,
    )

    with pytest.raises(asyncio.CancelledError):
        await request_task
    assert upstream_cancelled.is_set()
    replacement = await asyncio.wait_for(queued, timeout=1)
    await admission.release_session(replacement.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disconnect_early",
    [False, True],
    ids=["body-exhausted", "client-disconnect"],
)
async def test_pool_stream_holds_episode_slot_until_body_closes(
    monkeypatch: pytest.MonkeyPatch,
    disconnect_early: bool,
) -> None:
    class Inference:
        async def completion(self, _request: dict) -> dict:
            return {
                "id": "pool-completion",
                "model": "upstream-qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            }

    class Storage:
        def save_message(self, *_args, **_kwargs) -> None:
            raise AssertionError("pool completions must not be persisted")

    registry = SessionRegistry()
    for session_id in ("stream-one", "stream-two"):
        registry.register(
            session_id,
            registered=True,
            status=SessionStatus.RUNNING,
        )
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    first = await admission.acquire(
        session_id="stream-one",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )
    state = SimpleNamespace(
        inference=SimpleNamespace(),
        model_pool={
            "pool/qwen": server.ModelPoolRoute(
                model="upstream-qwen",
                inference=Inference(),  # type: ignore[arg-type]
                max_active_episodes=1,
            )
        },
        episode_admission=admission,
        storage=Storage(),
        node=SimpleNamespace(model_served="local-router-policy"),
        transform_manager=TransformManager(),
        session_registry=registry,
    )
    monkeypatch.setattr(server, "get_state", lambda: state)

    response = await server.proxy_request(
        _request(
            "/v1/chat/completions",
            {"model": "pool/qwen", "messages": [], "stream": True},
            authorization=f"Bearer {first.call_capability}",
        ),
        "v1/chat/completions",
    )
    queued = asyncio.create_task(
        admission.acquire(
            session_id="stream-two",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=1,
        )
    )
    await asyncio.sleep(0)
    with pytest.raises(EpisodeReleaseDraining):
        await admission.release(
            session_id="stream-one",
            lease_id=first.lease_id,
            wait_timeout_seconds=0.01,
        )
    assert not queued.done()

    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)
        if (
            disconnect_early
            and message["type"] == "http.response.body"
            and message.get("more_body") is True
        ):
            raise OSError("synthetic client disconnect")

    async def receive() -> dict:
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [],
        "client": ("sandbox", 1234),
        "server": ("gateway", 8081),
    }
    if disconnect_early:
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)  # type: ignore[misc]
    else:
        await response(scope, receive, send)  # type: ignore[misc]

    second = await asyncio.wait_for(queued, timeout=1)
    assert await admission.release(
        session_id="stream-one",
        lease_id=first.lease_id,
    ) is False
    assert await admission.release(
        session_id="stream-two",
        lease_id=second.lease_id,
    ) is True


@pytest.mark.asyncio
async def test_runtime_destroyed_fallback_cancels_stream_response_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Inference:
        async def completion(self, _request: dict) -> dict:
            return {
                "id": "pool-completion",
                "model": "upstream-qwen",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            }

    registry = SessionRegistry()
    registry.register("stream-owner", registered=True, status=SessionStatus.RUNNING)
    admission = ModelPoolEpisodeAdmission({"pool/qwen": 1})
    first = await admission.acquire(
        session_id="stream-owner",
        alias="pool/qwen",
        attempt_id="0:solve",
        timeout_seconds=1,
    )
    monkeypatch.setattr(
        server,
        "get_state",
        lambda: SimpleNamespace(
            inference=SimpleNamespace(),
            model_pool={
                "pool/qwen": server.ModelPoolRoute(
                    model="upstream-qwen",
                    inference=Inference(),  # type: ignore[arg-type]
                    max_active_episodes=1,
                )
            },
            episode_admission=admission,
            storage=SimpleNamespace(),
            node=SimpleNamespace(model_served="local-router-policy"),
            transform_manager=TransformManager(),
            session_registry=registry,
        ),
    )

    # Resolve the route in one task, then drive its response in another.  The
    # response boundary must transfer admission ownership to the ASGI task.
    route_task = asyncio.create_task(
        server.proxy_request(
            _request(
                "/v1/chat/completions",
                {"model": "pool/qwen", "messages": [], "stream": True},
                authorization=f"Bearer {first.call_capability}",
            ),
            "v1/chat/completions",
        )
    )
    response = await route_task
    send_started = asyncio.Event()

    async def send(_message: dict) -> None:
        send_started.set()
        await asyncio.Event().wait()

    async def receive() -> dict:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [],
        "client": ("sandbox", 1234),
        "server": ("gateway", 8081),
    }
    response_task = asyncio.create_task(response(scope, receive, send))  # type: ignore[misc]
    await asyncio.wait_for(send_started.wait(), timeout=1)
    queued = asyncio.create_task(
        admission.acquire(
            session_id="stream-next",
            alias="pool/qwen",
            attempt_id="0:solve",
            timeout_seconds=1,
        )
    )

    await admission.release_after_runtime_destroyed(
        "stream-owner",
        runtime_destroyed=True,
    )

    with pytest.raises(asyncio.CancelledError):
        await response_task
    replacement = await asyncio.wait_for(queued, timeout=1)
    await admission.release_session(replacement.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["response_start", "cancelled"])
async def test_stream_response_outer_boundary_always_runs_finalizer_once(
    failure_point: str,
) -> None:
    calls = 0
    send_started = asyncio.Event()
    hold_send = asyncio.Event()
    finalizer_started = asyncio.Event()
    release_finalizer = asyncio.Event()
    if failure_point == "response_start":
        release_finalizer.set()

    async def finalize() -> None:
        nonlocal calls
        calls += 1
        finalizer_started.set()
        await release_finalizer.wait()

    async def body():
        yield b"chunk"

    response = server._FinalizingStreamingResponse(
        body(),
        finalizer=server._AsyncOnce(finalize),
    )
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [],
        "client": ("sandbox", 1234),
        "server": ("gateway", 8081),
    }

    async def receive() -> dict:
        return {"type": "http.disconnect"}

    async def send(_message: dict) -> None:
        send_started.set()
        if failure_point == "response_start":
            raise OSError("response start failed")
        await hold_send.wait()

    if failure_point == "response_start":
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, send)
    else:
        task = asyncio.create_task(response(scope, receive, send))
        await asyncio.wait_for(send_started.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(finalizer_started.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release_finalizer.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert calls == 1


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
    assert denied.status_code == 403
    assert orjson.loads(denied.body)["error"]["code"] == "control_plane_forbidden"
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
async def test_spilot_candidate_cannot_use_gateway_control_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UDS transport access grants no session/admin control authority."""

    class Registry:
        def active_sessions(self):
            raise AssertionError("unauthorized caller must not list sessions")

    inference = SimpleNamespace(
        generation_status=lambda: (_ for _ in ()).throw(
            AssertionError("unauthorized caller must not inspect inference state")
        ),
        pause_generation=AsyncMock(),
        resume_generation=AsyncMock(),
    )
    node_manager = SimpleNamespace(cancel=AsyncMock())
    state = SimpleNamespace(
        inference=inference,
        model_pool={"pool/qwen": object()},
        node=SimpleNamespace(model_pool=[object()]),
        node_manager=node_manager,
        session_registry=Registry(),
    )
    monkeypatch.setenv("POLAR_CONTROL_PLANE_TOKEN", "trusted-control-token")
    monkeypatch.setattr(server, "get_state", lambda: state)

    no_auth_get = _request("/sessions", {}, method="GET", authorization=None)
    no_auth_delete = _request(
        "/sessions/victim", {}, method="DELETE", authorization=None
    )
    no_auth_pause = _request(
        "/admin/inference/pause", {}, method="POST", authorization=None
    )
    no_auth_resume = _request(
        "/admin/inference/resume", {}, method="POST", authorization=None
    )

    responses = [
        await server.list_sessions(no_auth_get, status=None, task_id=None, limit=200),
        await server.delete_session(no_auth_delete, "victim"),
        await server.pause_inference_generation(no_auth_pause),
        await server.resume_inference_generation(no_auth_resume),
    ]

    assert [response.status_code for response in responses] == [403, 403, 403, 403]
    assert {
        orjson.loads(response.body)["error"]["code"] for response in responses
    } == {"control_plane_forbidden"}
    node_manager.cancel.assert_not_awaited()
    inference.pause_generation.assert_not_awaited()
    inference.resume_generation.assert_not_awaited()


@pytest.mark.asyncio
async def test_spilot_session_cannot_bypass_scoped_alias_with_served_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SessionRegistry()
    registry.register(
        "spilot-session",
        registered=True,
        status=SessionStatus.RUNNING,
        metadata={"_polar_agent_harness": "spilot_router"},
    )
    state = SimpleNamespace(model_pool={}, session_registry=registry)
    monkeypatch.setattr(server, "get_state", lambda: state)

    response = await server.proxy_request(
        _request(
            "/v1/chat/completions",
            {"model": "Qwen/Qwen3.5-9B", "messages": []},
            authorization="Bearer spilot-session",
        ),
        "v1/chat/completions",
    )

    assert response.status_code == 403
    assert orjson.loads(response.body)["error"]["code"] == (
        "spilot_unscoped_model_forbidden"
    )


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
