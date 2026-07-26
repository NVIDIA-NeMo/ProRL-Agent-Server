"""SSE multiplex of upstream events to browser clients."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from polar.platform.events import SSE_HEADERS, EventBus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/events")
async def stream_events(request: Request) -> StreamingResponse:
    state = request.app.state.platform
    bus: EventBus = state.event_bus

    async def iterator():
        async for chunk in bus.stream_events(heartbeat_seconds=15.0):
            if await request.is_disconnected():
                break
            yield chunk

    return StreamingResponse(
        iterator(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get("/health")
async def platform_health(request: Request) -> dict[str, Any]:
    state = request.app.state.platform
    rollout = await state.rollout_client.safe_get_json("/health", default=None)
    gateways: dict[str, Any] = {}
    for node_id, client in state.gateway_clients.items():
        gateways[node_id] = await client.safe_get_json("/health", default=None)
    return {
        "status": "ok",
        "rollout_reachable": bool(rollout and rollout.get("status") == "ok"),
        "rollout_url": state.config.rollout_url,
        "gateways": gateways,
    }


__all__ = ["router"]
