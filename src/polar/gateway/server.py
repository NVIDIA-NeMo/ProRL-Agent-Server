"""FastAPI gateway proxy server and gateway-node lifecycle entry point."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import orjson
from pydantic import BaseModel, ConfigDict, Field

from polar.config import GatewayNodeConfig, TopologyConfig
from polar.gateway.completion_writer import CompletionWriter
from polar.gateway.detection import APIType, detect, extract_model
from polar.gateway.engine import (
    OpenAICompatibleEngine,
    POLAR_INFERENCE_TIMINGS_KEY,
    get_engine,
    sanitize_inference_timings,
)
from polar.gateway.episode_admission import (
    EpisodeAcquireCancelled,
    EpisodeAcquireTimeout,
    EpisodeAdmissionPoisoned,
    EpisodeCallUnauthorized,
    EpisodeLeaseClosing,
    EpisodeLeaseConflict,
    EpisodeLeaseNotOwned,
    EpisodeReleaseDraining,
    ModelPoolEpisodeAdmission,
    UnknownEpisodeAlias,
)
from polar.gateway.node import GatewayNodeManager
from polar.gateway.proxy import (
    InferenceClient,
    UpstreamError,
    UpstreamHTTPError,
    UpstreamTimeoutError,
)
from polar.gateway.session import (
    extract_api_key,
    InvalidSessionIdError,
    MODEL_POOL_ADMISSION_CAPABILITY_SCOPE,
    MODEL_POOL_CAPABILITY_SCOPE,
    ROUTER_CAPABILITY_SCOPE,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionDeleteResponse,
    SessionRegistry,
    SessionStatusResponse,
    clean_session_id,
    generate_session_id,
    resolve_session_id,
)
from polar.gateway.storage import SessionStore
from polar.gateway.transform import TransformManager
from polar.gateway.transform.base import BaseTransformer
from polar.http_logging import uvicorn_access_log_enabled
from polar.platform.events import SSE_HEADERS, EventBus
from polar.rollout.models import SessionDispatchRequest, SessionDispatchResponse, SessionStatus
from polar.trajectory.registry import default_builder_registry, default_evaluator_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_ROUTER_POLICY_MODEL_ALIAS = "router/policy"
_CONTROL_PLANE_TOKEN_ENV = "POLAR_CONTROL_PLANE_TOKEN"
_CONTROL_PLANE_TOKEN_HEADER = "x-polar-control-token"
_MAX_EPISODE_ADMISSION_WAIT_SECONDS = 86_400.0
_GATEWAY_HTTP_DRAIN_TIMEOUT_SECONDS = 60


class _StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EpisodeLeaseAcquireRequest(_StrictRequestModel):
    model: str = Field(min_length=1, max_length=256, pattern=r"^pool/\S+$")
    attempt_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )
    wait_timeout_seconds: float = Field(
        gt=0,
        le=_MAX_EPISODE_ADMISSION_WAIT_SECONDS,
    )


class EpisodeLeaseReleaseRequest(_StrictRequestModel):
    lease_id: str = Field(min_length=16, max_length=256)
    wait_timeout_seconds: float = Field(default=2.0, gt=0, le=10.0)


@dataclass(frozen=True, slots=True)
class ModelPoolRoute:
    """Trusted host-side destination for one opaque sandbox model alias."""

    model: str
    inference: InferenceClient
    max_active_episodes: int | None = None


@dataclass(slots=True)
class GatewayState:
    topology: TopologyConfig
    node: GatewayNodeConfig
    inference: InferenceClient
    model_pool: dict[str, ModelPoolRoute]
    episode_admission: ModelPoolEpisodeAdmission
    storage: SessionStore
    transform_manager: TransformManager
    session_registry: SessionRegistry
    node_manager: GatewayNodeManager
    completion_writer: CompletionWriter
    event_bus: EventBus


class _AsyncOnce:
    """Run one async cleanup exactly once, shielding it from disconnect cancellation."""

    def __init__(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._callback = callback
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

    async def __call__(self) -> None:
        async with self._lock:
            if self._task is None:

                async def invoke() -> None:
                    await self._callback()

                self._task = asyncio.create_task(invoke())
            task = self._task
        await _await_task_despite_cancellation(task)


async def _await_task_despite_cancellation(task: asyncio.Task[None]) -> None:
    """Finish security cleanup before propagating any caller cancellation."""

    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
            if task.done():
                break
    # Retrieve/propagate cleanup failure before an unrelated caller cancel.
    task.result()
    if cancellation is not None:
        raise cancellation


async def _end_request_despite_cancellation(
    admission: ModelPoolEpisodeAdmission,
    handle: Any,
) -> None:
    async def finish() -> None:
        await admission.end_request(handle)

    await _await_task_despite_cancellation(asyncio.create_task(finish()))


class _FinalizingStreamingResponse(StreamingResponse):
    """Own a lease/request finalizer across every ASGI send/disconnect path."""

    def __init__(self, *args: Any, finalizer: _AsyncOnce, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_finalizer = finalizer

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette does not guarantee aclose() when response-start/body
            # send fails or the response task is cancelled. Close best-effort
            # for generator-local cleanup, then unconditionally release the
            # admission handle from this response ownership boundary.
            aclose = getattr(self.body_iterator, "aclose", None)
            if callable(aclose):
                try:
                    await asyncio.shield(aclose())
                except (Exception, asyncio.CancelledError) as exc:
                    logger.debug("Could not close streaming body during disconnect: %s", exc)
            await self._request_finalizer()


_state: GatewayState | None = None
_configured_topology_path: str | None = None
_configured_node_id: str | None = None


def configure_server(topology_path: str = "topology.yaml", *, node_id: str | None = None) -> None:
    global _configured_topology_path, _configured_node_id, _state
    _configured_topology_path = topology_path
    _configured_node_id = node_id
    _state = None


def _build_state(topology: TopologyConfig, node_id: str | None) -> GatewayState:
    node = topology.select_gateway_node(node_id)
    inference = InferenceClient(node.inference_base_url, get_engine(node.engine))
    model_pool: dict[str, ModelPoolRoute] = {}
    for candidate in node.model_pool:
        api_key = os.environ.get(candidate.api_key_env)
        if not api_key:
            raise ValueError(
                f"Model pool alias {candidate.alias!r} requires non-empty host "
                f"environment variable {candidate.api_key_env!r}"
            )
        model_pool[candidate.alias] = ModelPoolRoute(
            model=candidate.model,
            inference=InferenceClient(
                candidate.base_url,
                OpenAICompatibleEngine(),
                default_headers={"Authorization": f"Bearer {api_key}"},
                max_concurrency=candidate.max_concurrency,
            ),
            max_active_episodes=candidate.max_active_episodes,
        )
    episode_admission = ModelPoolEpisodeAdmission(
        {
            candidate.alias: candidate.max_active_episodes
            for candidate in node.model_pool
            if candidate.max_active_episodes is not None
        }
    )
    persistence_config = topology.gateway.completion_persistence
    save_dir = topology.rollout.save_dir
    completion_writer = CompletionWriter(
        save_dir=save_dir if save_dir else None,
        max_field_bytes=persistence_config.max_field_bytes,
        queue_size=persistence_config.queue_size,
        write_workers=persistence_config.write_workers,
        batch_size=persistence_config.batch_size,
        write_max_attempts=persistence_config.write_max_attempts,
        retry_backoff_seconds=persistence_config.retry_backoff_seconds,
        enabled=persistence_config.enabled and bool(save_dir),
    )
    storage = SessionStore(completion_writer=completion_writer)
    transform_manager = TransformManager()
    session_registry = SessionRegistry()
    builder_registry = default_builder_registry()
    evaluator_registry = default_evaluator_registry()
    event_bus = EventBus()
    # Wrap session_registry methods to emit events on state changes.
    _wrap_registry_for_events(session_registry, event_bus)
    node_manager = GatewayNodeManager(
        node_id=node.id,
        gateway_url=node.public_url,
        max_init_workers=node.max_init_workers,
        max_run_workers=node.max_run_workers,
        max_postrun_workers=node.max_postrun_workers,
        storage=storage,
        session_registry=session_registry,
        builders=builder_registry,
        evaluators=evaluator_registry,
        default_runtime=node.default_runtime,
        rollout_server_url=topology.gateway.rollout_server_url or None,
        heartbeat_interval_seconds=topology.gateway.heartbeat_interval_seconds,
        episode_admission=episode_admission,
    )
    return GatewayState(
        topology=topology,
        node=node,
        inference=inference,
        model_pool=model_pool,
        episode_admission=episode_admission,
        storage=storage,
        transform_manager=transform_manager,
        session_registry=session_registry,
        node_manager=node_manager,
        completion_writer=completion_writer,
        event_bus=event_bus,
    )


def _wrap_registry_for_events(registry: SessionRegistry, bus: EventBus) -> None:
    """Monkey-patch set_status / set_result so changes emit events to the bus."""
    original_set_status = registry.set_status
    original_set_result = registry.set_result

    def _bus_publish(event_type: str, payload: dict[str, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        bus.publish_threadsafe(loop, event_type, payload)

    def patched_set_status(session_id: str, status: str):
        info = original_set_status(session_id, status)
        if info is not None:
            _bus_publish(
                "session.state_changed",
                {
                    "session_id": session_id,
                    "task_id": info.task_id,
                    "status": status,
                },
            )
        return info

    def patched_set_result(session_id: str, result):
        info = original_set_result(session_id, result)
        if info is not None:
            _bus_publish(
                "session.state_changed",
                {
                    "session_id": session_id,
                    "task_id": info.task_id,
                    "status": str(info.status),
                },
            )
        return info

    registry.set_status = patched_set_status  # type: ignore[assignment]
    registry.set_result = patched_set_result  # type: ignore[assignment]


def get_state() -> GatewayState:
    global _state
    if _state is None:
        topology_path = _configured_topology_path or os.environ.get(
            "POLAR_TOPOLOGY",
            "topology.yaml",
        )
        node_id = _configured_node_id or os.environ.get("POLAR_GATEWAY_NODE_ID")
        _state = _build_state(TopologyConfig.load(topology_path), node_id)
    return _state


@asynccontextmanager
async def _lifespan(_: FastAPI):
    state = get_state()
    await state.completion_writer.start()
    await state.node_manager.start()
    try:
        yield
    finally:
        await state.node_manager.close()
        await state.inference.close()
        await asyncio.gather(*(route.inference.close() for route in state.model_pool.values()))
        state.storage.close()
        await state.completion_writer.close()


app = FastAPI(title="Polar Gateway", version="0.1.0", lifespan=_lifespan)


@app.api_route("/", methods=["GET", "HEAD"])
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "polar-gateway"}


def _format_anthropic_events(events: list[dict[str, Any]]) -> str:
    parts = []
    for event in events:
        event_type = event.get("type", "unknown")
        parts.append(f"event: {event_type}\ndata: {json.dumps(event)}\n\n")
    return "".join(parts)


def _format_openai_sse(chunk: dict[str, Any]) -> str:
    return f"data: {json.dumps(chunk, default=str)}\n\n"


def _format_responses_events(events: list[dict[str, Any]]) -> str:
    parts = []
    for event in events:
        event_type = event.get("type", "unknown")
        parts.append(f"event: {event_type}\ndata: {json.dumps(event)}\n\n")
    return "".join(parts)


def _format_google_sse(chunk: dict[str, Any]) -> str:
    return f"data: {json.dumps(chunk)}\n\n"


def _error_type_name(exc: Exception) -> str:
    if isinstance(exc, UpstreamTimeoutError):
        return "timeout_error"
    if isinstance(exc, UpstreamHTTPError):
        return "upstream_http_error"
    if isinstance(exc, UpstreamError):
        return "upstream_error"
    return type(exc).__name__


def _build_error_body(
    api_type: APIType,
    message: str,
    *,
    error_type: str,
    upstream_body: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    if api_type == APIType.ANTHROPIC:
        if isinstance(upstream_body, dict):
            if upstream_body.get("type") == "error" and isinstance(
                upstream_body.get("error"), dict
            ):
                return upstream_body
            error = upstream_body.get("error")
            if isinstance(error, dict):
                return {
                    "type": "error",
                    "error": {
                        "type": error.get("type", "api_error"),
                        "message": error.get("message", message),
                    },
                }
        return {"type": "error", "error": {"type": "api_error", "message": message}}

    if isinstance(upstream_body, dict) and "error" in upstream_body:
        return upstream_body

    if api_type == APIType.GOOGLE:
        status = "DEADLINE_EXCEEDED" if error_type == "timeout_error" else "INTERNAL"
        return {"error": {"message": message, "status": status}}

    return {"error": {"message": message, "type": error_type}}


def _is_openai_context_length_error(message: str) -> bool:
    """Return whether an upstream message unambiguously reports a context limit."""

    normalized = " ".join(message.lower().split())
    if any(
        phrase in normalized
        for phrase in (
            "context length exceeded",
            "context window exceeded",
            "exceeds the context length",
            "exceeds context length",
            "exceeds the context window",
            "exceeds context window",
        )
    ):
        return True

    # Match both SGLang's "requested token count exceeds the model's maximum
    # context length" and OpenAI's "model's maximum context length ... your
    # messages resulted in ... tokens" without reclassifying generic errors
    # that merely mention context.
    return "maximum context" in normalized and any(
        marker in normalized
        for marker in ("exceed", "requested", "token count", "messages resulted")
    )


def _standardize_openai_context_length_error(
    api_type: APIType,
    upstream_body: dict[str, Any] | str | None,
) -> dict[str, Any] | str | None:
    """Add the OpenAI fields LiteLLM uses to identify context overflow."""

    if api_type not in (APIType.OPENAI_CHAT, APIType.OPENAI_RESPONSES):
        return upstream_body
    if not isinstance(upstream_body, dict):
        return upstream_body
    error = upstream_body.get("error")
    if not isinstance(error, dict):
        return upstream_body
    message = error.get("message")
    if not isinstance(message, str) or not _is_openai_context_length_error(message):
        return upstream_body

    return {
        **upstream_body,
        "error": {
            **error,
            "message": message,
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": "messages",
        },
    }


def _upstream_error_response(
    api_type: APIType,
    exc: Exception,
    *,
    standardize_openai_context_length: bool = False,
) -> JSONResponse:
    if isinstance(exc, UpstreamHTTPError):
        status_code = exc.status_code
        upstream_body = exc.body
    elif isinstance(exc, UpstreamTimeoutError):
        status_code = 504
        upstream_body = None
    elif isinstance(exc, UpstreamError):
        status_code = 502
        upstream_body = None
    else:
        status_code = 502
        upstream_body = None

    if standardize_openai_context_length:
        upstream_body = _standardize_openai_context_length_error(api_type, upstream_body)

    return JSONResponse(
        _build_error_body(
            api_type,
            str(exc),
            error_type=_error_type_name(exc),
            upstream_body=upstream_body,
        ),
        status_code=status_code,
    )


def _stream_error_output(api_type: APIType, exc: Exception) -> str:
    message = str(exc)
    error_type = _error_type_name(exc)

    if api_type == APIType.ANTHROPIC:
        return _format_anthropic_events(
            [
                {
                    "type": "error",
                    "error": {"type": error_type, "message": message},
                }
            ]
        )
    if api_type == APIType.OPENAI_RESPONSES:
        return _format_responses_events([{"type": "error", "message": message}])
    if api_type == APIType.GOOGLE:
        status = "DEADLINE_EXCEEDED" if error_type == "timeout_error" else "INTERNAL"
        return _format_google_sse({"error": {"message": message, "status": status}})
    return _format_openai_sse({"error": {"message": message, "type": error_type}})


def _resolve_session_id(
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    query_session_id: str | None = None,
    registry: SessionRegistry | None = None,
) -> str:
    return resolve_session_id(
        registry or get_state().session_registry,
        headers,
        body,
        query_session_id=query_session_id,
    )


def _resolve_privileged_session_id(
    headers: dict[str, str],
    registry: SessionRegistry,
    *,
    scope: str,
) -> tuple[str | None, JSONResponse | None]:
    """Authenticate a Router/model-pool request against a live rollout session.

    Ordinary policy proxy calls retain the gateway's legacy implicit-session
    behavior.  Reserved ``router/*`` and ``pool/*`` aliases can reach training
    data or paid host credentials, so they require the bearer injected by the
    rollout dispatcher and must never auto-register an unknown caller.
    """

    credential = extract_api_key(headers)
    if credential is None:
        return None, _privileged_session_error(
            "A live rollout-session credential is required",
            status_code=401,
            code="missing_session_credential",
        )
    info = registry.resolve_capability(credential, scope=scope)
    if info is None:
        return None, _privileged_session_error(
            "The rollout-session capability is invalid",
            status_code=401,
            code="invalid_session_capability",
        )
    if not info.registered or info.status != SessionStatus.RUNNING:
        return None, _privileged_session_error(
            "The rollout session is not running",
            status_code=403,
            code="inactive_session_credential",
        )

    registry.update_activity(info.session_id)
    return info.session_id, None


def _resolve_legacy_proxy_session_id(
    headers: dict[str, str],
    registry: SessionRegistry,
) -> tuple[str | None, JSONResponse | None]:
    """Resolve an ordinary model call without implicit session creation."""

    credential = extract_api_key(headers)
    if credential is None:
        return None, _privileged_session_error(
            "A live rollout-session credential is required",
            status_code=401,
            code="missing_session_credential",
        )
    info = registry.get(credential)
    if info is None or not info.registered:
        return None, _privileged_session_error(
            "The rollout-session credential is invalid",
            status_code=401,
            code="invalid_session_credential",
        )
    if info.status != SessionStatus.RUNNING:
        return None, _privileged_session_error(
            "The rollout session is not running",
            status_code=403,
            code="inactive_session_credential",
        )
    if info.metadata.get("_polar_agent_harness") == "spilot_router":
        return None, _privileged_session_error(
            "SPilot sessions may call only scoped Router and model-pool aliases",
            status_code=403,
            code="spilot_unscoped_model_forbidden",
        )
    registry.update_activity(info.session_id)
    return info.session_id, None


def _require_control_plane_request(
    request: Request,
    *,
    state: GatewayState | Any | None = None,
) -> JSONResponse | None:
    """Protect gateway control/diagnostic routes from task sandboxes.

    SPilot task containers intentionally share the gateway's UDS transport so
    they can reach the scoped Router/model-pool APIs.  Transport reachability
    is therefore not authority to inspect or mutate gateway state.  Preserve
    legacy deployments that have neither a control token nor a model pool,
    while failing closed whenever model-pool credentials are configured.
    """

    current_state = state if state is not None else get_state()
    expected = os.environ.get(_CONTROL_PLANE_TOKEN_ENV, "").strip()
    requires_protection = bool(expected) or bool(
        getattr(getattr(current_state, "node", None), "model_pool", None)
        or getattr(current_state, "model_pool", None)
    )
    if not requires_protection:
        return None
    if not expected:
        return _control_plane_error(
            "Gateway control-plane authentication is not configured",
            status_code=503,
            code="control_plane_auth_unconfigured",
        )
    supplied = request.headers.get(_CONTROL_PLANE_TOKEN_HEADER, "")
    if not secrets.compare_digest(supplied, expected):
        return _control_plane_error(
            "This gateway route is restricted to the control plane",
            status_code=403,
            code="control_plane_forbidden",
        )
    return None


def _coerce_datetime(value: str | None) -> datetime:
    if value:
        return datetime.fromisoformat(value)
    return datetime.now(timezone.utc)


def _session_response(session_id: str) -> SessionStatusResponse:
    state = get_state()
    metadata = state.storage.get_session_metadata(session_id)
    info = state.session_registry.get(session_id)
    result = info.result if info is not None else None
    if info is None and metadata is None and result is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if info is not None:
        task_id = info.task_id
        created_at = info.created_at
        status = info.status
    else:
        task_id = (metadata or {}).get("task_id") or (result.task_id if result else None)
        created_at = _coerce_datetime((metadata or {}).get("created_at"))
        status = result.status if result is not None else SessionStatus.REGISTERED

    completion_count = int((metadata or {}).get("completion_count", 0))
    return SessionStatusResponse(
        session_id=session_id,
        task_id=task_id,
        created_at=created_at,
        completion_count=completion_count,
        status=status,
        result=result,
    )


def _format_stream_events(api_type: APIType, events: list[dict[str, Any]]) -> str:
    if api_type == APIType.ANTHROPIC:
        return _format_anthropic_events(events)
    if api_type == APIType.OPENAI_RESPONSES:
        return _format_responses_events(events)
    if api_type == APIType.GOOGLE:
        return _format_google_sse(events[0]) if events else ""
    return _format_openai_sse(events[0]) if events else ""


def _completion_metadata(
    session_info: Any | None,
    response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(getattr(session_info, "metadata", None) or {})
    if session_info is not None:
        metadata.setdefault("session_id", session_info.session_id)
        if session_info.task_id is not None:
            metadata.setdefault("task_id", session_info.task_id)
    if response is not None:
        # Internal engine annotations must not leak into the OpenAI payload.
        # Store only the sanitized values alongside this exact completion so
        # trajectory-level aggregation can attribute inference time correctly.
        timings = sanitize_inference_timings(response.pop(POLAR_INFERENCE_TIMINGS_KEY, None))
        if timings:
            metadata["inference_timings"] = timings
    return metadata


def _policy_completion_metadata(
    session_info: Any | None,
    response: dict[str, Any],
    *,
    completion_role: str,
) -> dict[str, Any]:
    metadata = _completion_metadata(session_info, response)
    # Stamp this at the gateway persistence boundary and deliberately overwrite
    # similarly named session metadata supplied by a caller.  Only the reserved
    # Router request alias is eligible for SPilot training; other local-policy
    # requests remain persisted for legacy builders but are excluded by the
    # RouterPolicyBuilder.
    metadata["completion_role"] = completion_role
    return metadata


def format_stream_output(
    api_type: APIType,
    transformer: BaseTransformer,
    chunk: dict[str, Any],
    original_request: dict[str, Any],
    is_first: bool,
) -> str:
    transformed = transformer.transform_stream_chunk(chunk, original_request, is_first=is_first)
    if api_type == APIType.ANTHROPIC:
        return _format_anthropic_events(transformed)
    if api_type == APIType.OPENAI_RESPONSES:
        if isinstance(transformed, list):
            return _format_responses_events(transformed)
        return _format_responses_events([transformed]) if transformed else ""
    if api_type == APIType.GOOGLE:
        return _format_google_sse(transformed)
    return _format_openai_sse(transformed)


@app.get("/v1/models")
async def list_models(request: Request):
    state = get_state()
    if auth_error := _require_control_plane_request(request, state=state):
        return auth_error
    try:
        return await state.inference.list_models()
    except Exception as exc:
        logger.error("Failed to list models: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=502)


@app.post("/v1/tokenize")
async def tokenize_request(request: Request) -> Response:
    """Proxy exact chat-template tokenization without recording a completion.

    Agent-side trajectory budgets need the same tokenizer, tool schema and
    chat template as generation.  Sending this through the gateway keeps
    rootless network-isolated sandboxes on their existing UDS transport while
    intentionally bypassing completion persistence and generation accounting.
    """

    state = get_state()
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON body must be an object"}, status_code=400)

    pool_request_handle = None
    # A formal Router gateway exposes its UDS to untrusted task sandboxes.
    # Tokenization still consumes local serving capacity, so bind it to the
    # same active alias-specific episode lease as the candidate completion.
    if getattr(state, "model_pool", None):
        requested_model = body.get("model")
        pool_route = (
            state.model_pool.get(requested_model)
            if isinstance(requested_model, str)
            else None
        )
        if pool_route is None or pool_route.max_active_episodes is None:
            return _privileged_session_error(
                "Tokenization requires an active capped model-pool lease",
                status_code=403,
                code="tokenize_model_forbidden",
            )
        credential = extract_api_key({key: value for key, value in request.headers.items()})
        if credential is None:
            return _privileged_session_error(
                "A lease-scoped pool-call credential is required",
                status_code=401,
                code="missing_pool_call_capability",
            )
        try:
            pool_request_handle = await state.episode_admission.begin_request(
                call_capability=credential,
                alias=requested_model,
            )
        except EpisodeCallUnauthorized:
            return _privileged_session_error(
                "The pool-call capability is invalid",
                status_code=401,
                code="invalid_pool_call_capability",
            )
        except EpisodeLeaseClosing:
            return _episode_admission_error(
                "The episode lease is closing",
                status_code=409,
                code="episode_lease_closing",
            )
        except EpisodeAdmissionPoisoned:
            return _episode_admission_error(
                "Model-pool episode admission is poisoned",
                status_code=503,
                code="episode_admission_poisoned",
            )
        info = state.session_registry.get(pool_request_handle.session_id)
        if info is None or info.status != SessionStatus.RUNNING:
            await _end_request_despite_cancellation(
                state.episode_admission,
                pool_request_handle,
            )
            return _privileged_session_error(
                "The rollout session is not running",
                status_code=403,
                code="inactive_pool_call_capability",
            )

    tokenize_body = dict(body)
    tokenize_body["model"] = state.node.model_served
    try:
        try:
            response = await state.inference.tokenize(tokenize_body)
        except UpstreamError as exc:
            logger.warning("Upstream tokenization error: %s", exc)
            return _upstream_error_response(APIType.OPENAI_CHAT, exc)
    finally:
        if pool_request_handle is not None:
            await _end_request_despite_cancellation(
                state.episode_admission,
                pool_request_handle,
            )
    # SGLang also returns every token id.  The budget client only needs the
    # scalar count; dropping the potentially 262k-element list avoids copying
    # it over the sandbox UDS and serializing it a second time.
    compact_response = {
        key: response[key]
        for key in ("count", "max_model_len")
        if key in response
    }
    return Response(
        content=orjson.dumps(compact_response),
        media_type="application/json",
    )


@app.get("/health")
async def health():
    state = get_state()
    metrics = await state.node_manager.stage_metrics()
    episode_admission = await state.episode_admission.snapshot()
    admission_health = state.node_manager.episode_admission_health()
    try:
        upstream = await state.inference.health()
    except Exception as exc:
        upstream = {"status": "error", "error": str(exc)}
    payload = {
        "status": "ok" if admission_health["healthy"] else "error",
        "node_id": state.node.id,
        "gateway_url": state.node.public_url,
        "inference": upstream,
        "metrics": metrics.model_dump(mode="json"),
        "completion_persistence": state.completion_writer.stats(),
        "active_status_counts": state.session_registry.active_status_counts(),
        "model_pool_episode_admission": episode_admission,
        "model_pool_episode_admission_health": admission_health,
        "available_init": max(0, state.node.max_init_workers - metrics.init_inflight),
        "available_run": max(0, state.node.max_run_workers - metrics.run_inflight),
        "available_postrun": max(0, state.node.max_postrun_workers - metrics.postrun_inflight),
    }
    if not admission_health["healthy"]:
        return JSONResponse(payload, status_code=503)
    return payload


@app.get("/admin/inference/status")
async def inference_generation_status(request: Request):
    state = get_state()
    if auth_error := _require_control_plane_request(request, state=state):
        return auth_error
    return state.inference.generation_status()


@app.post("/admin/inference/pause")
async def pause_inference_generation(request: Request, timeout_seconds: float = 300.0):
    state = get_state()
    if auth_error := _require_control_plane_request(request, state=state):
        return auth_error
    try:
        status = await state.inference.pause_generation(timeout_seconds=timeout_seconds)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Timed out waiting for inference requests to drain after {timeout_seconds}s",
        ) from exc
    logger.info(
        "Paused inference generation proxy for weight update; inflight=%s",
        status["inflight"],
    )
    return status


@app.post("/admin/inference/resume")
async def resume_inference_generation(request: Request):
    state = get_state()
    if auth_error := _require_control_plane_request(request, state=state):
        return auth_error
    status = await state.inference.resume_generation()
    logger.info("Resumed inference generation proxy")
    return status


@app.get("/sessions")
async def list_sessions(
    request: Request,
    status: str | None = Query(default=None),
    task_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    """List sessions on this gateway (active + recently terminal)."""
    state = get_state()
    if auth_error := _require_control_plane_request(request, state=state):
        return auth_error
    active = state.session_registry.active_sessions()
    rows: list[dict[str, Any]] = []
    for entry in active:
        if status and entry.get("status") != status:
            continue
        if task_id and entry.get("task_id") != task_id:
            continue
        metadata = state.storage.get_session_metadata(entry["session_id"]) or {}
        rows.append(
            {
                **entry,
                "completion_count": int(metadata.get("completion_count") or 0),
                "model_requested": metadata.get("model_requested"),
                "model_used": metadata.get("model_used"),
                "api_type": metadata.get("api_type"),
                "created_at": metadata.get("created_at"),
                "node_id": state.node.id,
            }
        )
    return {"sessions": rows[:limit], "node_id": state.node.id}


@app.get("/sessions/{session_id}/completions")
async def list_session_completions(request: Request, session_id: str) -> dict[str, Any]:
    """In-memory completions for an active or recently-completed session."""
    state = get_state()
    if auth_error := _require_control_plane_request(request, state=state):
        return auth_error
    try:
        safe = clean_session_id(session_id)
    except InvalidSessionIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if safe is None:
        raise HTTPException(status_code=400, detail="Session id required")
    completions = state.storage.get_completions(safe)
    return {
        "session_id": safe,
        "completions": completions,
        "node_id": state.node.id,
    }


@app.get("/events")
async def stream_events(request: Request):
    state = get_state()
    if auth_error := _require_control_plane_request(request, state=state):
        return auth_error

    async def iterator():
        async for chunk in state.event_bus.stream_events(heartbeat_seconds=15.0):
            if await request.is_disconnected():
                break
            yield chunk

    return StreamingResponse(
        iterator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/sessions", response_model=SessionCreateResponse | SessionDispatchResponse)
async def create_session(request: Request):
    state = get_state()
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if auth_error := _require_control_plane_request(request, state=state):
        return auth_error
    if "agent" in body and "session_id" in body:
        expected_control_token = os.environ.get(_CONTROL_PLANE_TOKEN_ENV, "").strip()
        agent = body.get("agent")
        is_spilot_dispatch = isinstance(agent, dict) and agent.get("harness") == "spilot_router"
        if not expected_control_token and is_spilot_dispatch:
            return _control_plane_error(
                "SPilot dispatch requires a configured control-plane token",
                status_code=503,
                code="control_plane_auth_unconfigured",
            )
        if expected_control_token:
            supplied_control_token = request.headers.get(_CONTROL_PLANE_TOKEN_HEADER, "")
            if not secrets.compare_digest(supplied_control_token, expected_control_token):
                return _control_plane_error(
                    "The control-plane credential is invalid",
                    status_code=401,
                    code="invalid_control_plane_credential",
                )
        dispatch_request = SessionDispatchRequest.model_validate(body)
        try:
            await state.node_manager.dispatch(dispatch_request)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SessionDispatchResponse(
            session_id=dispatch_request.session_id,
            task_id=dispatch_request.task_id,
            status=SessionStatus.REGISTERED,
            node_id=state.node.id,
        )

    create_request = SessionCreateRequest.model_validate(body)
    try:
        session_id = clean_session_id(create_request.session_id) or generate_session_id()
    except InvalidSessionIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    info = state.session_registry.register(
        session_id,
        task_id=create_request.task_id,
        registered=True,
        status=SessionStatus.REGISTERED,
    )
    metadata = state.storage.ensure_session(
        info.session_id,
        model_requested=None,
        model_used=None,
        api_type=None,
        task_id=info.task_id,
        created_at=info.created_at.isoformat(),
    )
    return SessionCreateResponse(
        session_id=info.session_id,
        task_id=info.task_id,
        created_at=info.created_at,
        completion_count=int(metadata.get("completion_count", 0)),
        status=info.status,
    )


@app.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session(request: Request, session_id: str):
    if auth_error := _require_control_plane_request(request):
        return auth_error
    try:
        safe_session_id = clean_session_id(session_id)
    except InvalidSessionIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if safe_session_id is None:
        raise HTTPException(status_code=400, detail="Session ID cannot be empty")
    return _session_response(safe_session_id)


@app.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(request: Request, session_id: str):
    state = get_state()
    if auth_error := _require_control_plane_request(request, state=state):
        return auth_error
    try:
        safe_session_id = clean_session_id(session_id)
    except InvalidSessionIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if safe_session_id is None:
        raise HTTPException(status_code=400, detail="Session ID cannot be empty")

    cancellation_accepted = await state.node_manager.cancel(safe_session_id)
    if cancellation_accepted:
        # Runtime kill/reap and session-directory removal continue in tracked
        # gateway tasks.  Acknowledge immediately so hundreds of concurrent
        # early-stop DELETEs do not occupy the rollout client's connections.
        return SessionDeleteResponse(
            session_id=safe_session_id,
            deleted=True,
            messages_deleted=0,
        )

    info = state.session_registry.get(safe_session_id)
    deleted_count = state.storage.delete_session(safe_session_id)
    if info is None and deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")

    state.session_registry.remove(safe_session_id)
    return SessionDeleteResponse(
        session_id=safe_session_id,
        deleted=True,
        messages_deleted=deleted_count,
    )


def _episode_admission_error(
    message: str,
    *,
    status_code: int,
    code: str,
) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": "episode_admission_error",
                "code": code,
            }
        },
        status_code=status_code,
    )


@app.post("/internal/model-pool/episode-leases/acquire")
async def acquire_model_pool_episode_lease(
    request: Request,
    payload: EpisodeLeaseAcquireRequest,
):
    state = get_state()
    session_id, auth_error = _resolve_privileged_session_id(
        {key: value for key, value in request.headers.items()},
        state.session_registry,
        scope=MODEL_POOL_ADMISSION_CAPABILITY_SCOPE,
    )
    if auth_error is not None:
        return auth_error
    assert session_id is not None
    try:
        grant = await state.episode_admission.acquire(
            session_id=session_id,
            alias=payload.model,
            attempt_id=payload.attempt_id,
            timeout_seconds=payload.wait_timeout_seconds,
        )
    except UnknownEpisodeAlias as exc:
        return _episode_admission_error(
            str(exc), status_code=400, code="unknown_episode_model"
        )
    except EpisodeLeaseConflict as exc:
        return _episode_admission_error(
            str(exc), status_code=409, code="episode_lease_conflict"
        )
    except EpisodeAcquireTimeout as exc:
        return _episode_admission_error(
            str(exc), status_code=503, code="episode_admission_timeout"
        )
    except EpisodeAcquireCancelled as exc:
        return _episode_admission_error(
            str(exc), status_code=409, code="episode_admission_cancelled"
        )
    except EpisodeAdmissionPoisoned as exc:
        return _episode_admission_error(
            str(exc), status_code=503, code="episode_admission_poisoned"
        )
    return {
        "lease_id": grant.lease_id,
        "model": grant.alias,
        "attempt_id": grant.attempt_id,
        "wait_ms": grant.wait_ms,
        "local_cap": grant.local_cap,
        "call_capability": grant.call_capability,
    }


@app.post("/internal/model-pool/episode-leases/release")
async def release_model_pool_episode_lease(
    request: Request,
    payload: EpisodeLeaseReleaseRequest,
):
    state = get_state()
    session_id, auth_error = _resolve_privileged_session_id(
        {key: value for key, value in request.headers.items()},
        state.session_registry,
        scope=MODEL_POOL_ADMISSION_CAPABILITY_SCOPE,
    )
    if auth_error is not None:
        return auth_error
    assert session_id is not None
    try:
        released = await state.episode_admission.release(
            session_id=session_id,
            lease_id=payload.lease_id,
            wait_timeout_seconds=payload.wait_timeout_seconds,
        )
    except EpisodeReleaseDraining:
        return JSONResponse(
            {"released": False, "draining": True},
            status_code=202,
        )
    except EpisodeAdmissionPoisoned as exc:
        return _episode_admission_error(
            str(exc), status_code=503, code="episode_admission_poisoned"
        )
    except EpisodeLeaseNotOwned as exc:
        return _episode_admission_error(
            str(exc), status_code=404, code="episode_lease_not_owned"
        )
    return {"released": released}


@app.api_route("/{path:path}", methods=["POST"])
async def proxy_request(request: Request, path: str):
    state = get_state()
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    headers = {k: v for k, v in request.headers.items()}
    full_path = request.url.path
    api_type = detect(full_path, headers, body)
    original_model = extract_model(api_type, body)
    is_router_alias = isinstance(original_model, str) and original_model.startswith(
        "router/"
    )
    is_pool_alias = isinstance(original_model, str) and original_model.startswith(
        "pool/"
    )
    if is_router_alias:
        session_id, auth_error = _resolve_privileged_session_id(
            headers,
            state.session_registry,
            scope=ROUTER_CAPABILITY_SCOPE,
        )
        if auth_error is not None:
            return auth_error
        assert session_id is not None
    elif not is_pool_alias:
        session_id, auth_error = _resolve_legacy_proxy_session_id(
            headers,
            state.session_registry,
        )
        if auth_error is not None:
            return auth_error
        assert session_id is not None
    else:
        # A capped pool alias authenticates below with the lease-scoped call
        # token. Keep the legacy session capability only for uncapped routes.
        session_id = ""
        if original_model not in state.model_pool:
            session_id, auth_error = _resolve_privileged_session_id(
                headers,
                state.session_registry,
                scope=MODEL_POOL_CAPABILITY_SCOPE,
            )
            if auth_error is not None:
                return auth_error
            assert session_id is not None

    pool_route: ModelPoolRoute | None = None
    pool_request_handle = None
    if isinstance(original_model, str) and original_model.startswith("router/"):
        if original_model != _ROUTER_POLICY_MODEL_ALIAS:
            return _model_pool_error(
                f"Unknown Router policy alias: {original_model}",
                code="unknown_router_model",
            )
        if api_type != APIType.OPENAI_CHAT or "/v1/chat/completions" not in full_path:
            return _model_pool_error(
                "The Router policy alias supports only /v1/chat/completions",
                code="unsupported_router_api",
            )
    if isinstance(original_model, str) and original_model.startswith("pool/"):
        if api_type != APIType.OPENAI_CHAT or "/v1/chat/completions" not in full_path:
            return _model_pool_error(
                "Model-pool aliases support only /v1/chat/completions",
                code="unsupported_pool_api",
            )
        pool_route = state.model_pool.get(original_model)
        if pool_route is None:
            return _model_pool_error(
                f"Unknown model-pool alias: {original_model}",
                code="unknown_pool_model",
            )
        if pool_route.max_active_episodes is not None:
            admission = getattr(state, "episode_admission", None)
            if admission is None:
                return _episode_admission_error(
                    "Model-pool episode admission is not initialized",
                    status_code=503,
                    code="episode_admission_unavailable",
                )
            credential = extract_api_key(headers)
            if credential is None:
                return _privileged_session_error(
                    "A lease-scoped pool-call credential is required",
                    status_code=401,
                    code="missing_pool_call_capability",
                )
            try:
                pool_request_handle = await admission.begin_request(
                    call_capability=credential,
                    alias=original_model,
                )
            except EpisodeCallUnauthorized:
                return _privileged_session_error(
                    "The pool-call capability is invalid",
                    status_code=401,
                    code="invalid_pool_call_capability",
                )
            except EpisodeLeaseClosing:
                return _episode_admission_error(
                    "The episode lease is closing",
                    status_code=409,
                    code="episode_lease_closing",
                )
            except EpisodeAdmissionPoisoned:
                return _episode_admission_error(
                    "Model-pool episode admission is poisoned",
                    status_code=503,
                    code="episode_admission_poisoned",
                )
            session_id = pool_request_handle.session_id
            info = state.session_registry.get(session_id)
            if info is None or info.status != SessionStatus.RUNNING:
                await _end_request_despite_cancellation(
                    admission,
                    pool_request_handle,
                )
                pool_request_handle = None
                return _privileged_session_error(
                    "The rollout session is not running",
                    status_code=403,
                    code="inactive_pool_call_capability",
                )
            state.session_registry.update_activity(session_id)
        else:
            session_id, auth_error = _resolve_privileged_session_id(
                headers,
                state.session_registry,
                scope=MODEL_POOL_CAPABILITY_SCOPE,
            )
            if auth_error is not None:
                return auth_error
            assert session_id is not None
    try:
        transformer = state.transform_manager.get(api_type)
        session_info = state.session_registry.get(session_id)

        logger.debug(
            "← %s %s | api=%s model=%s session=%s",
            request.method,
            full_path,
            api_type.value,
            original_model,
            session_id,
        )

        if api_type == APIType.GOOGLE and "streamGenerateContent" in full_path:
            body["_streaming"] = True

        served_model = pool_route.model if pool_route is not None else state.node.model_served
        inference = pool_route.inference if pool_route is not None else state.inference
        completion_role = (
            "router_policy"
            if original_model == _ROUTER_POLICY_MODEL_ALIAS
            else "policy"
        )
        transformed_body = body.copy()
        transformed_body["_polar_model_served"] = served_model
        openai_request = transformer.transform_request(transformed_body)
        openai_request["model"] = served_model
        is_streaming = openai_request.get("stream", False)

        if is_streaming:
            stream_finalizer: Callable[[], Awaitable[None]] | None = None
            if pool_request_handle is not None:
                request_handle = pool_request_handle

                async def stream_finalizer() -> None:
                    await state.episode_admission.end_request(request_handle)

            response = await _handle_streaming(
                api_type,
                transformer,
                openai_request,
                body,
                session_id,
                original_model=original_model,
                session_info=session_info,
                inference=inference,
                persist_completion=pool_route is None,
                response_model_alias=original_model if pool_route is not None else None,
                completion_role=completion_role,
                request_finalizer=stream_finalizer,
            )
            if isinstance(response, StreamingResponse) and pool_request_handle is not None:
                # The response body's async-generator now owns the lease
                # request handle through normal exhaustion or disconnect.
                pool_request_handle = None
            return response
        return await _handle_non_streaming(
            api_type,
            transformer,
            openai_request,
            body,
            session_id,
            original_model=original_model,
            session_info=session_info,
            inference=inference,
            persist_completion=pool_route is None,
            response_model_alias=original_model if pool_route is not None else None,
            completion_role=completion_role,
        )
    finally:
        if pool_request_handle is not None:
            await _end_request_despite_cancellation(
                state.episode_admission,
                pool_request_handle,
            )


def _model_pool_error(message: str, *, code: str) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code,
            }
        },
        status_code=400,
    )


def _privileged_session_error(
    message: str,
    *,
    status_code: int,
    code: str,
) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": "authentication_error",
                "code": code,
            }
        },
        status_code=status_code,
        headers={"WWW-Authenticate": "Bearer"} if status_code == 401 else None,
    )


def _control_plane_error(
    message: str,
    *,
    status_code: int,
    code: str,
) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": "authentication_error",
                "code": code,
            }
        },
        status_code=status_code,
    )


async def _handle_non_streaming(
    api_type: APIType,
    transformer: BaseTransformer,
    openai_request: dict[str, Any],
    original_request: dict[str, Any],
    session_id: str,
    *,
    original_model: str,
    session_info: Any | None,
    inference: InferenceClient | None = None,
    persist_completion: bool = True,
    response_model_alias: str | None = None,
    completion_role: str = "policy",
) -> Response:
    state = get_state()
    inference = inference or state.inference
    try:
        response = await inference.completion(openai_request)
    except UpstreamError as exc:
        if persist_completion:
            logger.warning("Non-streaming upstream error for session %s: %s", session_id, exc)
        else:
            logger.warning(
                "Model-pool upstream error for session %s: %s",
                session_id,
                _error_type_name(exc),
            )
        return _upstream_error_response(
            api_type,
            exc,
            standardize_openai_context_length=True,
        )

    if response_model_alias is not None:
        response["model"] = response_model_alias
    if persist_completion:
        state.storage.save_message(
            session_id,
            openai_request,
            response,
            original_request=original_request,
            model_requested=original_model,
            model_used=openai_request["model"],
            api_type=api_type.value,
            task_id=session_info.task_id if session_info else None,
            created_at=session_info.created_at.isoformat() if session_info else None,
            metadata=_policy_completion_metadata(
                session_info,
                response,
                completion_role=completion_role,
            ),
        )
    transformed = transformer.transform_response(response, original_request)
    # Non-streaming SGLang responses carry large token/logprob arrays.  Use the
    # optimized encoder so serializing them does not monopolize the gateway's
    # only event-loop thread.
    return Response(content=orjson.dumps(transformed), media_type="application/json")


async def _handle_streaming(
    api_type: APIType,
    transformer: BaseTransformer,
    openai_request: dict[str, Any],
    original_request: dict[str, Any],
    session_id: str,
    *,
    original_model: str,
    session_info: Any | None,
    inference: InferenceClient | None = None,
    persist_completion: bool = True,
    response_model_alias: str | None = None,
    completion_role: str = "policy",
    request_finalizer: Callable[[], Awaitable[None]] | None = None,
) -> StreamingResponse | JSONResponse:
    state = get_state()
    inference = inference or state.inference
    non_stream_request = {k: v for k, v in openai_request.items() if k != "stream_options"}
    non_stream_request["stream"] = False
    try:
        response = await inference.completion(non_stream_request)
    except UpstreamError as exc:
        if persist_completion:
            logger.warning("Upstream error for streaming session %s: %s", session_id, exc)
        else:
            logger.warning(
                "Model-pool streaming upstream error for session %s: %s",
                session_id,
                _error_type_name(exc),
            )
        return _upstream_error_response(api_type, exc)

    if response_model_alias is not None:
        response["model"] = response_model_alias
    if persist_completion:
        state.storage.save_message(
            session_id,
            openai_request,
            response,
            original_request=original_request,
            model_requested=original_model,
            model_used=openai_request["model"],
            api_type=api_type.value,
            task_id=session_info.task_id if session_info else None,
            created_at=session_info.created_at.isoformat() if session_info else None,
            metadata=_policy_completion_metadata(
                session_info,
                response,
                completion_role=completion_role,
            ),
        )

    synthetic_chunk = _response_to_stream_chunk(response)
    stream_state = transformer.create_stream_state(original_request)
    finalizer_once = _AsyncOnce(request_finalizer) if request_finalizer is not None else None

    async def generate():
        try:
            if stream_state is not None:
                events = stream_state.process_chunk(synthetic_chunk, is_first=True)
                if events:
                    yield _format_stream_events(api_type, events)
                final_events = stream_state.finalize()
                if final_events:
                    yield _format_stream_events(api_type, final_events)
            else:
                output = format_stream_output(
                    api_type,
                    transformer,
                    synthetic_chunk,
                    original_request,
                    True,
                )
                if output:
                    yield output
            if api_type == APIType.OPENAI_CHAT:
                yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("Synthetic stream error: %s", exc)
            yield _stream_error_output(api_type, exc)
        finally:
            if finalizer_once is not None:
                await finalizer_once()

    response_kwargs = {
        "media_type": "text/event-stream",
        "headers": {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    }
    if finalizer_once is not None:
        return _FinalizingStreamingResponse(
            generate(),
            finalizer=finalizer_once,
            **response_kwargs,
        )
    return StreamingResponse(generate(), **response_kwargs)


def _response_to_stream_chunk(response: dict[str, Any]) -> dict[str, Any]:
    """Convert a non-streaming chat completion into a single 'delta' chunk
    suitable for a transformer's stream_state.process_chunk / transform_stream_chunk."""
    choices = response.get("choices") or [{}]
    choice = choices[0]
    message = choice.get("message", {}) or {}

    tool_calls_delta: list[dict[str, Any]] = []
    for i, tc in enumerate(message.get("tool_calls") or []):
        func = tc.get("function", {}) or {}
        tool_calls_delta.append(
            {
                "index": i,
                "id": tc.get("id"),
                "type": tc.get("type", "function"),
                "function": {
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", ""),
                },
            }
        )

    delta: dict[str, Any] = {"role": "assistant"}
    if message.get("content") is not None:
        delta["content"] = message.get("content")
    if message.get("reasoning_content") is not None:
        delta["reasoning_content"] = message.get("reasoning_content")
    if tool_calls_delta:
        delta["tool_calls"] = tool_calls_delta

    return {
        "id": response.get("id"),
        "object": "chat.completion.chunk",
        "created": response.get("created"),
        "model": response.get("model"),
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": choice.get("finish_reason"),
            }
        ],
        "usage": response.get("usage"),
    }


def serve(
    topology_path: str = "topology.yaml",
    *,
    node_id: str | None = None,
    log_level: str = "info",
) -> None:
    import uvicorn

    configure_server(topology_path, node_id=node_id)
    state = get_state()
    config = uvicorn.Config(
        app,
        host=state.node.host,
        port=state.node.port,
        log_level=log_level,
        access_log=uvicorn_access_log_enabled(),
        # Bound active HTTP/provider streams before ASGI lifespan shutdown.
        # Node.close then gets its separate dispatcher/runtime proof budget;
        # launcher teardown covers both phases plus a margin.
        timeout_graceful_shutdown=_GATEWAY_HTTP_DRAIN_TIMEOUT_SECONDS,
    )
    server = uvicorn.Server(config=config)
    server.run()
    lifespan = getattr(server, "lifespan", None)
    if not server.started or bool(getattr(lifespan, "shutdown_failed", False)):
        # uvicorn.run() reports startup failures nonzero but silently returns
        # zero when ASGI lifespan shutdown fails. A gateway teardown failure
        # means runtime containment is unproven, so surface it to Slurm and the
        # launcher instead of publishing a successful experiment exit.
        raise SystemExit(1)


def main() -> None:
    serve(
        os.environ.get("POLAR_TOPOLOGY", "topology.yaml"),
        node_id=os.environ.get("POLAR_GATEWAY_NODE_ID"),
    )


if __name__ == "__main__":
    main()
