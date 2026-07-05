"""FastAPI server for rollout orchestration."""

from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from polar.config import RolloutServiceConfig, TopologyConfig
from polar.http_logging import uvicorn_access_log_enabled
from polar.platform.events import SSE_HEADERS, EventBus
from polar.rollout.balancer import NodeScheduler
from polar.rollout.manager import RolloutManager
from polar.rollout.models import (
    GatewayNodeInfo,
    NodeHeartbeatRequest,
    NodeRegistrationRequest,
    SessionResult,
    TaskRequest,
    TaskStatus,
)
from polar.rollout.pipeline import Pipeline
from polar.runtime.assets import RuntimeAssetUnavailableError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_CONTROL_PLANE_TOKEN_ENV = "POLAR_CONTROL_PLANE_TOKEN"
_CONTROL_PLANE_TOKEN_HEADER = "x-polar-control-token"


def _require_control_plane_request(
    request: Request,
    *,
    required: bool = False,
) -> None:
    expected = os.environ.get(_CONTROL_PLANE_TOKEN_ENV, "").strip()
    if not expected:
        if required:
            raise HTTPException(
                status_code=503,
                detail="SPilot submission requires a configured control-plane token",
            )
        return
    supplied = request.headers.get(_CONTROL_PLANE_TOKEN_HEADER, "")
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid control-plane credential")


@dataclass(slots=True)
class RolloutState:
    topology: TopologyConfig
    rollout: RolloutServiceConfig
    scheduler: NodeScheduler
    pipeline: Pipeline
    manager: RolloutManager
    event_bus: EventBus


_state: RolloutState | None = None
_configured_topology_path: str | None = None


def configure_server(topology_path: str = "topology.yaml") -> None:
    global _configured_topology_path, _state
    _configured_topology_path = topology_path
    _state = None


def _build_state(topology: TopologyConfig) -> RolloutState:
    rollout = topology.rollout
    scheduler = NodeScheduler(bootstrap_nodes=topology.bootstrap_nodes)
    event_bus = EventBus()
    pipeline = Pipeline(
        callback_url=f"{rollout.public_url}/callbacks/session_result",
        save_dir=rollout.save_dir,
        scheduler=scheduler,
        dispatch_poll_interval_seconds=rollout.dispatch_poll_interval_seconds,
        callback_grace_seconds=rollout.callback_grace_seconds,
        http_max_connections=rollout.http_max_connections,
        http_max_keepalive_connections=rollout.http_max_keepalive_connections,
        cleanup_max_concurrency=rollout.cleanup_max_concurrency,
        cleanup_max_attempts=rollout.cleanup_max_attempts,
        cleanup_retry_backoff_seconds=rollout.cleanup_retry_backoff_seconds,
        event_bus=event_bus,
    )
    manager = RolloutManager(pipeline=pipeline, scheduler=scheduler, event_bus=event_bus)
    return RolloutState(
        topology=topology,
        rollout=rollout,
        scheduler=scheduler,
        pipeline=pipeline,
        manager=manager,
        event_bus=event_bus,
    )


def get_state() -> RolloutState:
    global _state
    if _state is None:
        topology_path = _configured_topology_path or os.environ.get(
            "POLAR_TOPOLOGY",
            "topology.yaml",
        )
        _state = _build_state(TopologyConfig.load(topology_path))
    return _state


@asynccontextmanager
async def _lifespan(_: FastAPI):
    state = get_state()
    await state.pipeline.start()
    try:
        yield
    finally:
        await state.manager.close()
        await state.pipeline.close()


app = FastAPI(title="Polar Rollout", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health():
    state = get_state()
    return {"status": "ok", "nodes": len(state.scheduler.list_nodes())}


@app.post("/rollout/task/submit")
async def submit_task_async(http_request: Request, request: TaskRequest):
    """Non-blocking task submission. Returns immediately with task_id.

    Poll ``GET /rollout/task/{task_id}`` until status becomes terminal.
    """
    if request.agent.harness == "spilot_router":
        _require_control_plane_request(http_request, required=True)

    state = get_state()
    try:
        task_id = await state.manager.submit_task(request)
    except RuntimeAssetUnavailableError as exc:
        logger.error("Rejecting rollout task %s: %s", request.task_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"task_id": task_id, "status": "running"}


@app.get("/rollout/task/{task_id}", response_model=TaskStatus)
async def get_task(task_id: str):
    task = get_state().manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.delete("/rollout/task/{task_id}", response_model=TaskStatus)
async def cancel_task(
    task_id: str,
    register_if_missing: bool = Query(default=False),
):
    """Cancel a rollout task and all gateway sessions owned by it."""
    task = await get_state().manager.cancel_task(
        task_id,
        register_if_missing=register_if_missing,
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/rollout/status")
async def rollout_status():
    return get_state().manager.status()


@app.post("/nodes/register", response_model=GatewayNodeInfo)
async def register_node(http_request: Request, request: NodeRegistrationRequest):
    _require_control_plane_request(http_request)
    return get_state().scheduler.register_node(request)


@app.post("/nodes/{node_id}/heartbeat", response_model=GatewayNodeInfo)
async def node_heartbeat(
    node_id: str,
    http_request: Request,
    request: NodeHeartbeatRequest,
):
    _require_control_plane_request(http_request)
    try:
        return get_state().scheduler.heartbeat(node_id, metrics=request.metrics)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/nodes", response_model=list[GatewayNodeInfo])
async def list_nodes():
    return get_state().scheduler.list_nodes()


@app.get("/nodes/{node_id}", response_model=GatewayNodeInfo)
async def get_node(node_id: str):
    node = get_state().scheduler.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@app.delete("/nodes/{node_id}", response_model=GatewayNodeInfo)
async def drain_node(node_id: str):
    try:
        return get_state().scheduler.drain_node(node_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/callbacks/session_result")
async def session_result_callback(http_request: Request, result: SessionResult):
    _require_control_plane_request(http_request)
    await get_state().pipeline.accept_callback_result(result)
    return {"status": "accepted"}


@app.get("/tasks")
async def list_tasks(
    status: str | None = Query(default=None),
    harness: str | None = Query(default=None),
    since: float | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
):
    """List tasks tracked in-memory by RolloutManager."""
    state = get_state()
    tasks = state.manager.list_tasks()
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    if harness:
        tasks = [t for t in tasks if t.get("harness") == harness]
    if since is not None:
        tasks = [t for t in tasks if (t.get("updated_at") or 0) >= since]
    tasks.sort(key=lambda t: (t.get("updated_at") or 0), reverse=True)
    return {"tasks": tasks[:limit]}


@app.get("/tasks/{task_id}/sessions")
async def list_task_sessions(task_id: str):
    """Per-session summaries for a task currently in memory."""
    state = get_state()
    sessions = state.manager.list_sessions_for(task_id)
    if sessions is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "sessions": sessions}


@app.get("/events")
async def stream_events(request: Request):
    state = get_state()

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


def serve(topology_path: str = "topology.yaml", *, log_level: str = "info") -> None:
    import uvicorn

    configure_server(topology_path)
    state = get_state()
    uvicorn.run(
        app,
        host=state.rollout.host,
        port=state.rollout.port,
        log_level=log_level,
        access_log=uvicorn_access_log_enabled(),
    )


def main() -> None:
    serve(os.environ.get("POLAR_TOPOLOGY", "topology.yaml"))


if __name__ == "__main__":
    main()
