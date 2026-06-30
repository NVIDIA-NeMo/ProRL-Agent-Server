#!/usr/bin/env python3
"""Expand one rendered gateway prototype into a Slurm gateway fleet.

The shell launcher performs the deliberately narrow environment substitution.
This helper then clones the already-validated node shape once per Slurm host,
assigns deterministic rank-based ids/public URLs, validates the complete Polar
topology, and publishes it with an atomic rename for the other ranks.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit, urlunsplit

import yaml

from polar.config import TopologyConfig


def expand_gateway_nodes(document: dict, hosts: list[str]) -> dict:
    """Return a topology whose gateway prototype is cloned for every host."""

    hosts = [host.strip() for host in hosts]
    if not hosts:
        raise ValueError("at least one gateway host is required")
    if any(not host for host in hosts):
        raise ValueError("gateway hosts must be non-empty")
    if len(set(hosts)) != len(hosts):
        raise ValueError("gateway hosts must be unique")

    gateway = document.get("gateway")
    if not isinstance(gateway, dict):
        raise ValueError("topology gateway block must be a mapping")
    configured_nodes = gateway.get("nodes")
    if not isinstance(configured_nodes, list) or len(configured_nodes) != 1:
        raise ValueError("gateway topology expansion requires exactly one prototype node")

    prototype = configured_nodes[0]
    if not isinstance(prototype, dict):
        raise ValueError("gateway node prototype must be a mapping")
    public_url = str(prototype.get("public_url") or "")
    parsed = urlsplit(public_url)
    if parsed.scheme not in {"http", "https"} or parsed.port is None:
        raise ValueError("gateway node prototype must have an explicit HTTP(S) public_url port")

    nodes = []
    for rank, host in enumerate(hosts):
        url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
        node = deepcopy(prototype)
        node["id"] = f"slurm-rank-{rank}"
        node["public_url"] = urlunsplit(
            (
                parsed.scheme,
                f"{url_host}:{parsed.port}",
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
        nodes.append(node)

    expanded = deepcopy(document)
    expanded["gateway"]["nodes"] = nodes
    return expanded


def write_topology_atomic(document: dict, output: Path) -> None:
    """Validate and atomically publish one topology on the shared filesystem."""

    TopologyConfig.model_validate(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            yaml.safe_dump(document, temporary, sort_keys=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gateway-host", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    loaded = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit("rendered topology must be a mapping")
    document = expand_gateway_nodes(loaded, args.gateway_host) if args.gateway_host else loaded
    write_topology_atomic(document, args.output)
    persistence = document["gateway"].get("completion_persistence", {})
    for node in document["gateway"]["nodes"]:
        public_host = urlsplit(node["public_url"]).hostname
        print(
            "[polar topology] "
            f"node_id={node['id']} host={public_host} bind_host={node.get('host')} "
            f"public_url={node['public_url']} "
            "quotas="
            f"{node.get('max_init_workers')}/"
            f"{node.get('max_run_workers')}/"
            f"{node.get('max_postrun_workers')} "
            "completion="
            f"{persistence.get('queue_size', 'default')}/"
            f"{persistence.get('write_workers', 'default')}"
        )


if __name__ == "__main__":
    main()
