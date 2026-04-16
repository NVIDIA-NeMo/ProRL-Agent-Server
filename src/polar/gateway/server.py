"""FastAPI gateway proxy server and gateway-node lifecycle entry point."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from polar.config import GatewayNodeConfig, TopologyConfig
from polar.gateway.control import RolloutControlClient
from polar.gateway.detection import APIType, detect, extract_model
from polar.gateway.node import GatewayNodeManager
from polar.gateway.proxy import (
    SGLangClient,
    UpstreamError,
    UpstreamHTTPError,
    UpstreamTimeoutError,
)
from polar.gateway.session import (
    InvalidSessionIdError,
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
from polar.gateway.streaming import (
    StreamAccumulator,
)
from polar.gateway.transform import TransformManager
from polar.gateway.transform.base import BaseTransformer
from polar.rollout.models import SessionDispatchRequest, SessionDispatchResponse
from polar.runtime.models import RuntimeSpec
from polar.trajectory.registry import default_builder_registry, default_evaluator_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GatewayState:
    topology: TopologyConfig
    node: GatewayNodeConfig
    sglang: SGLangClient
    storage: SessionStore
    transform_manager: TransformManager
    session_registry: SessionRegistry
    node_manager: GatewayNodeManager
    control_client: RolloutControlClient | None


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
    sglang = SGLangClient(node.sglang_base_url, timeout=node.sglang_timeout)
    storage = SessionStore()
    transform_manager = TransformManager()
    session_registry = SessionRegistry()
    builder_registry = default_builder_registry()
    evaluator_registry = default_evaluator_registry()
    node_manager = GatewayNodeManager(
        node_id=node.id,
        gateway_url=node.public_url,
        max_init_workers=node.max_init_workers,
        max_run_workers=node.max_run_workers,
        max_postrun_workers=node.max_postrun_workers,
        ready_buffer_target=node.ready_buffer_target,
        storage=storage,
        session_registry=session_registry,
        builders=builder_registry,
        evaluators=evaluator_registry,
        default_runtime=node.default_runtime,
    )
    control_client = (
        RolloutControlClient(
            rollout_server_url=topology.gateway.rollout_server_url,
            node_id=node.id,
            gateway_url=node.public_url,
            max_init_workers=node.max_init_workers,
            max_run_workers=node.max_run_workers,
            max_postrun_workers=node.max_postrun_workers,
            ready_buffer_target=node.ready_buffer_target,
            heartbeat_interval_seconds=topology.gateway.heartbeat_interval_seconds,
            node_manager=node_manager,
        )
        if topology.gateway.rollout_server_url
        else None
    )
    return GatewayState(
        topology=topology,
        node=node,
        sglang=sglang,
        storage=storage,
        transform_manager=transform_manager,
        session_registry=session_registry,
        node_manager=node_manager,
        control_client=control_client,
    )


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
    await state.node_manager.start()
    if state.control_client is not None:
        await state.control_client.start()
    try:
        yield
    finally:
        if state.control_client is not None:
            await state.control_client.close()
        await state.node_manager.close()
        await state.sglang.close()
        state.storage.close()


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
            if upstream_body.get("type") == "error" and isinstance(upstream_body.get("error"), dict):
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


def _upstream_error_response(api_type: APIType, exc: Exception) -> JSONResponse:
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
        return _format_anthropic_events([{
            "type": "error",
            "error": {"type": error_type, "message": message},
        }])
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
        status = result.status if result is not None else "REGISTERED"

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
async def list_models():
    state = get_state()
    try:
        return await state.sglang.list_models()
    except Exception as exc:
        logger.error("Failed to list models: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=502)


@app.get("/health")
async def health():
    state = get_state()
    metrics = await state.node_manager.stage_metrics()
    try:
        upstream = await state.sglang.health()
    except Exception as exc:
        upstream = {"status": "error", "error": str(exc)}
    return {
        "status": "ok",
        "node_id": state.node.id,
        "gateway_url": state.node.public_url,
        "sglang": upstream,
        "metrics": metrics.model_dump(mode="json"),
        "active_status_counts": state.session_registry.active_status_counts(),
        "active_sessions": state.session_registry.active_sessions(),
        "available_init": max(0, state.node.max_init_workers - metrics.init_inflight),
        "available_run": max(0, state.node.max_run_workers - metrics.run_inflight),
        "available_postrun": max(0, state.node.max_postrun_workers - metrics.postrun_inflight),
    }


@app.post("/sessions", response_model=SessionCreateResponse | SessionDispatchResponse)
async def create_session(request: Request):
    state = get_state()
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if "agent" in body and "session_id" in body:
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
            status="REGISTERED",
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
        status="REGISTERED",
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
async def get_session(session_id: str):
    try:
        safe_session_id = clean_session_id(session_id)
    except InvalidSessionIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if safe_session_id is None:
        raise HTTPException(status_code=400, detail="Session ID cannot be empty")
    return _session_response(safe_session_id)


@app.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str):
    state = get_state()
    try:
        safe_session_id = clean_session_id(session_id)
    except InvalidSessionIdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if safe_session_id is None:
        raise HTTPException(status_code=400, detail="Session ID cannot be empty")

    await state.node_manager.cancel(safe_session_id)
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
    try:
        session_id = _resolve_session_id(
            headers,
            body,
            query_session_id=(
                request.query_params.get("session_id")
                or request.query_params.get("key")
            ),
        )
    except InvalidSessionIdError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    original_model = extract_model(api_type, body)
    transformer = state.transform_manager.get(api_type)
    session_info = state.session_registry.get(session_id)

    logger.info(
        "← %s %s | api=%s model=%s session=%s",
        request.method, full_path, api_type.value, original_model, session_id,
    )

    # Debug: log request body keys and previous_response_id for diagnostics
    body_keys = sorted(body.keys()) if isinstance(body, dict) else "not-a-dict"
    prev_resp_id = body.get("previous_response_id") if isinstance(body, dict) else None
    input_type = type(body.get("input", "")).__name__ if isinstance(body, dict) else "?"
    input_len = len(body.get("input", "")) if isinstance(body, dict) and isinstance(body.get("input"), (str, list)) else 0
    logger.info(
        "  body_keys=%s prev_response_id=%s input_type=%s input_len=%s stream=%s",
        body_keys, prev_resp_id, input_type, input_len, body.get("stream"),
    )

    if api_type == APIType.GOOGLE and "streamGenerateContent" in full_path:
        body["_streaming"] = True

    # Resolve previous_response_id for multi-turn Responses API conversations
    if (
        api_type == APIType.OPENAI_RESPONSES
        and isinstance(body, dict)
        and body.get("previous_response_id")
    ):
        prev_id = body["previous_response_id"]
        logger.info("  Resolving previous_response_id=%s from session %s", prev_id, session_id)
        session_data = state.storage.load_completion_session(session_id)
        if session_data and session_data.completions:
            # Rebuild conversation history from stored completions
            history_items: list[dict[str, Any]] = []
            for rec in session_data.completions:
                req_msgs = rec.request.get("messages", [])
                resp_choices = rec.response.get("choices", [])
                # Add the request messages (skip system — instructions handles that)
                for msg in req_msgs:
                    role = msg.get("role", "")
                    if role == "system":
                        continue
                    if role == "user":
                        history_items.append({"type": "message", "role": "user", "content": msg.get("content", "")})
                    elif role == "tool":
                        history_items.append({
                            "type": "function_call_output",
                            "call_id": msg.get("tool_call_id", ""),
                            "output": msg.get("content", ""),
                        })
                # Add the assistant response
                if resp_choices:
                    resp_msg = resp_choices[0].get("message", {})
                    content = resp_msg.get("content", "")
                    tool_calls = resp_msg.get("tool_calls", [])
                    if content:
                        history_items.append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}],
                        })
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        history_items.append({
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", "{}"),
                        })

            # Merge: history_items + current input items
            current_input = body.get("input", [])
            if isinstance(current_input, str):
                current_input = [{"type": "message", "role": "user", "content": current_input}]
            elif not isinstance(current_input, list):
                current_input = []
            body["input"] = history_items + current_input
            logger.info(
                "  Resolved history: %d records, %d history items + %d current items",
                len(session_data.completions), len(history_items), len(current_input),
            )
        else:
            logger.warning("  No session data found for previous_response_id=%s", prev_id)

    openai_request = transformer.transform_request(body)
    openai_request["model"] = state.node.model_served
    is_streaming = openai_request.get("stream", False)

    if is_streaming:
        return await _handle_streaming(
            api_type,
            transformer,
            openai_request,
            body,
            session_id,
            original_model=original_model,
            session_info=session_info,
        )
    return await _handle_non_streaming(
        api_type,
        transformer,
        openai_request,
        body,
        session_id,
        original_model=original_model,
        session_info=session_info,
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
) -> JSONResponse:
    state = get_state()
    try:
        response = await state.sglang.completion(openai_request)
    except UpstreamError as exc:
        logger.warning("Non-streaming upstream error for session %s: %s", session_id, exc)
        return _upstream_error_response(api_type, exc)

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
    )
    transformed = transformer.transform_response(response, original_request)
    return JSONResponse(transformed)


async def _handle_streaming(
    api_type: APIType,
    transformer: BaseTransformer,
    openai_request: dict[str, Any],
    original_request: dict[str, Any],
    session_id: str,
    *,
    original_model: str,
    session_info: Any | None,
) -> StreamingResponse | JSONResponse:
    state = get_state()
    try:
        raw_stream = await state.sglang.open_completion_stream(openai_request)
    except UpstreamError as exc:
        logger.warning("Streaming setup error for session %s: %s", session_id, exc)
        return _upstream_error_response(api_type, exc)

    accumulator = StreamAccumulator()
    stream_state = transformer.create_stream_state(original_request)
    outcome = {"persist": False}
    save_done = asyncio.Event()
    state.storage.register_pending_save(session_id, save_done)

    async def generate():
        is_first = True
        had_error = False
        try:
            async for chunk in raw_stream.aiter_chunks():
                accumulator.accumulate(chunk)
                if stream_state is not None:
                    transformed = stream_state.process_chunk(chunk, is_first=is_first)
                    output = _format_stream_events(api_type, transformed) if transformed else ""
                else:
                    output = format_stream_output(
                        api_type,
                        transformer,
                        chunk,
                        original_request,
                        is_first,
                    )
                if output:
                    yield output
                is_first = False
        except Exception as exc:
            had_error = True
            logger.error("Stream error: %s", exc)
            yield _stream_error_output(api_type, exc)
        finally:
            await raw_stream.aclose()

        if not had_error:
            if stream_state is not None:
                final_events = stream_state.finalize()
                if final_events:
                    yield _format_stream_events(api_type, final_events)
            if api_type == APIType.OPENAI_CHAT:
                yield "data: [DONE]\n\n"
            outcome["persist"] = True

    async def finalize() -> None:
        try:
            if not outcome["persist"]:
                return
            try:
                response = accumulator.to_response()
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
                )
            except Exception as exc:
                logger.error("Failed to save streaming response: %s", exc)
        finally:
            save_done.set()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
        background=BackgroundTask(finalize),
    )


def serve(
    topology_path: str = "topology.yaml",
    *,
    node_id: str | None = None,
    log_level: str = "info",
) -> None:
    import uvicorn

    configure_server(topology_path, node_id=node_id)
    state = get_state()
    uvicorn.run(
        app,
        host=state.node.host,
        port=state.node.port,
        log_level=log_level,
    )


def main() -> None:
    serve(
        os.environ.get("POLAR_TOPOLOGY", "topology.yaml"),
        node_id=os.environ.get("POLAR_GATEWAY_NODE_ID"),
    )


if __name__ == "__main__":
    main()
