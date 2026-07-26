"""Live topology view: rollout + gateways + inference engine + worker pools."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request

from polar.platform.upstream import UpstreamClient

router = APIRouter(prefix="/api")


def _node_view(node) -> dict[str, Any]:
    return {
        "node_id": node.id,
        "host": node.host,
        "port": node.port,
        "gateway_url": node.public_url,
        "model_served": node.model_served,
        "engine": node.engine,
        "inference_base_url": node.inference_base_url,
        "max_init_workers": node.max_init_workers,
        "max_run_workers": node.max_run_workers,
        "max_postrun_workers": node.max_postrun_workers,
    }


@router.get("/topology")
async def get_topology(request: Request) -> dict[str, Any]:
    state = request.app.state.platform
    topology = state.config.topology

    rollout_health = await state.rollout_client.safe_get_json(
        "/health", default={"status": "unreachable"}
    )
    rollout_status = await state.rollout_client.safe_get_json(
        "/rollout/status", default={"tasks": {}, "pipeline": {}, "nodes": {}}
    )

    async def _fetch_gateway(node) -> dict[str, Any]:
        client: UpstreamClient = state.gateway_clients.get(node.id)
        health: dict[str, Any] = {"status": "unreachable"}
        if client is not None:
            health = await client.safe_get_json(
                "/health", default={"status": "unreachable"}
            )
        return {
            **_node_view(node),
            "health": health,
            "reachable": health.get("status") == "ok",
        }

    gateways = await asyncio.gather(
        *(_fetch_gateway(node) for node in topology.gateway.nodes)
    )

    return {
        "rollout": {
            "url": topology.rollout.public_url,
            "host": topology.rollout.host,
            "port": topology.rollout.port,
            "save_dir": str(state.config.save_dir),
            "health": rollout_health,
            "status": rollout_status,
            "reachable": rollout_health.get("status") == "ok",
        },
        "gateways": gateways,
    }


__all__ = ["router"]
