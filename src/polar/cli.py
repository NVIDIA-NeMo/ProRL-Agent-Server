"""Command-line interface for Polar."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse

import httpx
import yaml

from polar.config import TopologyConfig
from polar.gateway.server import serve as serve_gateway
from polar.platform.cli import add_subcommand as add_platform_subcommand
from polar.platform.cli import handle as handle_platform
from polar.rollout.server import serve as serve_rollout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polar",
        description="Run Polar services and interact with a Polar topology.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rollout_parser = subparsers.add_parser(
        "serve_rollout",
        help="Start the rollout service.",
    )
    rollout_parser.add_argument(
        "-c",
        "--config",
        default="topology.yaml",
        help="Path to topology.yaml",
    )

    add_platform_subcommand(subparsers)

    gateway_parser = subparsers.add_parser(
        "serve_gateway",
        help="Start one gateway node from topology.yaml.",
    )
    gateway_parser.add_argument(
        "-c",
        "--config",
        default="topology.yaml",
        help="Path to topology.yaml",
    )
    gateway_parser.add_argument(
        "--node-id",
        help="Gateway node id to launch. Required when topology has more than one node.",
    )

    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit a task file to the rollout service.",
    )
    submit_parser.add_argument("task_file", help="Path to a task JSON or YAML file")
    submit_parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Optional topology.yaml used to discover the rollout URL.",
    )
    submit_parser.add_argument(
        "--rollout-url",
        default=None,
        help="Override the rollout server URL.",
    )
    submit_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw response JSON.",
    )
    submit_parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between task-status polls (default: 2.0).",
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Show rollout health and gateway topology.",
    )
    status_parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Optional topology.yaml used to discover the rollout URL.",
    )
    status_parser.add_argument(
        "--rollout-url",
        default=None,
        help="Override the rollout server URL.",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw response JSON.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "serve_rollout":
            serve_rollout(args.config)
            return 0
        if args.command == "serve_gateway":
            serve_gateway(args.config, node_id=args.node_id)
            return 0
        if args.command == "dashboard":
            return handle_platform(args)
        if args.command == "submit":
            return _handle_submit(args)
        if args.command == "status":
            return _handle_status(args)
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.strip()
        if body:
            print(
                f"error: rollout server returned {exc.response.status_code}: {body}",
                file=sys.stderr,
            )
        else:
            print(
                f"error: rollout server returned {exc.response.status_code}",
                file=sys.stderr,
            )
        return 1
    except httpx.HTTPError as exc:
        print(f"error: could not reach the rollout service: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def _handle_submit(args: argparse.Namespace) -> int:
    rollout_url = _resolve_rollout_url(args.config, args.rollout_url)
    payload = _load_structured_file(args.task_file)
    timeout = httpx.Timeout(None, connect=30.0)
    control_token = os.environ.get("POLAR_CONTROL_PLANE_TOKEN", "").strip()
    headers = {"X-Polar-Control-Token": control_token} if control_token else None
    with httpx.Client(base_url=rollout_url, timeout=timeout, headers=headers) as client:
        submit_resp = client.post("/rollout/task/submit", json=payload)
        submit_resp.raise_for_status()
        task_id = submit_resp.json()["task_id"]

        poll_interval = max(0.1, float(args.poll_interval))
        while True:
            time.sleep(poll_interval)
            status_resp = client.get(f"/rollout/task/{task_id}")
            status_resp.raise_for_status()
            result = status_resp.json()
            if str(result.get("status")) in ("completed", "failed"):
                break
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    sessions = result.get("results") or []
    completed = sum(1 for session in sessions if session.get("status") == "COMPLETED")
    rewarded = sum(
        1
        for session in sessions
        if ((session.get("trajectory") or {}).get("traces") or [{}])[-1].get("reward") == 1.0
    )
    print(f"Task: {result.get('task_id')}")
    print(f"Status: {result.get('status')}")
    print(f"Completed sessions: {completed}/{len(sessions)}")
    print(f"Reward=1.0 sessions: {rewarded}")
    result_paths = result.get("result_paths") or []
    if result_paths:
        print("Result paths:")
        for path in result_paths:
            print(f"  {path}")
    return 0


def _handle_status(args: argparse.Namespace) -> int:
    rollout_url = _resolve_rollout_url(args.config, args.rollout_url)
    timeout = httpx.Timeout(10.0, connect=5.0)
    with httpx.Client(base_url=rollout_url, timeout=timeout) as client:
        health = client.get("/health").json()
        status = client.get("/rollout/status").json()

    gateway_health_by_node: dict[str, dict[str, Any]] = {}
    if args.config:
        topology = TopologyConfig.load(args.config)
        for node in topology.gateway.nodes:
            try:
                with httpx.Client(base_url=node.public_url, timeout=timeout) as client:
                    gateway_health_by_node[node.id] = client.get("/health").json()
            except httpx.HTTPError:
                gateway_health_by_node[node.id] = {
                    "status": "unreachable",
                    "gateway_url": node.public_url,
                }

    if args.json:
        print(
            json.dumps(
                {
                    "rollout_url": rollout_url,
                    "health": health,
                    "status": status,
                    "gateway_health": gateway_health_by_node,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    topology = TopologyConfig.load(args.config) if args.config else None
    node_rows = ((status.get("nodes") or {}).get("nodes")) or []

    overview_lines = [
        f"Rollout: {_endpoint_label('rollout', rollout_url)}",
        f"Health: {str(health.get('status', 'unknown')).upper()}",
        f"Registered Nodes: {health.get('nodes', 0)}",
        f"Running Sessions: {_format_count((status.get('pipeline') or {}).get('pending_sessions', 0), 'session')}",
    ]
    print(_boxed_section("Polar Status", overview_lines))

    if topology is not None:
        print()
        print(_boxed_section("Topology", _build_topology_lines(topology, node_rows)))

    if node_rows:
        print()
        print(_boxed_section("Live Load", _build_live_load_lines(node_rows, gateway_health_by_node)))
    return 0


def _build_topology_lines(
    topology: TopologyConfig,
    node_rows: list[dict[str, Any]],
) -> list[str]:
    node_status_by_id = {str(node.get("node_id")): node for node in node_rows}
    lines = [f"{_endpoint_label('rollout', topology.rollout.public_url)}"]
    last_index = len(topology.gateway.nodes) - 1
    for index, node in enumerate(topology.gateway.nodes):
        branch = "└──" if index == last_index else "├──"
        status = node_status_by_id.get(node.id, {})
        badge = _node_badge(
            healthy=bool(status.get("healthy", False)),
            draining=bool(status.get("draining", False)),
            reachable=bool(status),
        )
        gateway_label = _gateway_label(node.id)
        inference_label = _endpoint_label(node.engine, node.inference_base_url)
        lines.append(f"{branch} {gateway_label} [{badge}] ── {inference_label}")
    return lines


def _build_live_load_lines(
    node_rows: list[dict[str, Any]],
    gateway_health_by_node: dict[str, dict[str, Any]],
) -> list[str]:
    gateway_labels = {
        str(node.get("node_id")): _gateway_label(str(node.get("node_id")))
        for node in node_rows
    }
    name_width = max((len(label) for label in gateway_labels.values()), default=0)
    lines: list[str] = []
    for node in node_rows:
        node_id = str(node.get("node_id"))
        metrics = node.get("metrics") or {}
        node_health = gateway_health_by_node.get(node_id, {})
        status_counts = node_health.get("active_status_counts") or {}
        compact_status = ""
        if status_counts:
            compact = ", ".join(
                f"{name.lower()}={count}"
                for name, count in sorted(status_counts.items())
            )
            compact_status = f"  statuses[{compact}]"
        lines.append(
            f"{gateway_labels[node_id].ljust(name_width)}  "
            f"init {metrics.get('init_inflight', 0)}/{node.get('max_init_workers')}  "
            f"runtime_pool {metrics.get('ready_depth', 0)}  "
            f"run {metrics.get('run_inflight', 0)}/{node.get('max_run_workers')}  "
            f"postrun {metrics.get('postrun_inflight', 0)}/{node.get('max_postrun_workers')}"
            f"{compact_status}"
        )
    return lines


def _boxed_section(title: str, lines: list[str]) -> str:
    content = lines or ["(empty)"]
    width = max(len(title), *(len(line) for line in content))
    top = f"┌─ {title} " + "─" * max(0, width - len(title) - 1) + "┐"
    body = [f"│ {line.ljust(width)} │" for line in content]
    bottom = "└" + "─" * (width + 2) + "┘"
    return "\n".join([top, *body, bottom])


def _endpoint_label(prefix: str, url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or parsed.netloc or url
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{prefix}_{host}{port}"


def _format_count(count: object, noun: str) -> str:
    value = int(count or 0)
    suffix = noun if value == 1 else f"{noun}s"
    return f"{value} {suffix}"


def _gateway_label(node_id: str) -> str:
    return f"gateway_{node_id}"


def _node_badge(*, healthy: bool, draining: bool, reachable: bool) -> str:
    if not reachable:
        return "DOWN"
    if draining:
        return "DRAIN"
    if healthy:
        return "UP"
    return "DOWN"


def _resolve_rollout_url(config_path: str | None, explicit_url: str | None) -> str:
    if explicit_url:
        return explicit_url.rstrip("/")
    if config_path:
        topology = TopologyConfig.load(config_path)
        return topology.rollout.public_url
    return "http://127.0.0.1:8080"


def _load_structured_file(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    raw = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw)
    else:
        payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Task file {path} must contain a top-level mapping")
    return payload


if __name__ == "__main__":
    sys.exit(main())
