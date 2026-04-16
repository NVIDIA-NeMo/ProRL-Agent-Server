"""Load and validate a single Polar topology file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
from typing import Any
from urllib.parse import urlparse

import yaml

from polar.runtime.models import RuntimeSpec


@dataclass(frozen=True, slots=True)
class GatewayNodeConfig:
    id: str
    host: str
    port: int
    public_url: str
    model_served: str
    sglang_base_url: str
    sglang_timeout: float
    max_init_workers: int
    max_run_workers: int
    max_postrun_workers: int
    ready_buffer_target: int
    default_runtime: RuntimeSpec | None = None


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    heartbeat_interval_seconds: int
    rollout_server_url: str | None
    nodes: tuple[GatewayNodeConfig, ...]


@dataclass(frozen=True, slots=True)
class RolloutServiceConfig:
    host: str
    port: int
    public_url: str
    save_dir: str | None
    dispatch_poll_interval_seconds: float
    callback_grace_seconds: float


@dataclass(frozen=True, slots=True)
class TopologyConfig:
    path: Path
    rollout: RolloutServiceConfig
    gateway: GatewayConfig

    @classmethod
    def load(cls, config_path: str | Path = "topology.yaml") -> "TopologyConfig":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Topology file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Topology path is not a file: {path}")

        try:
            with path.open() as handle:
                loaded = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Topology file {path} must contain a top-level mapping")

        rollout = cls._parse_rollout(loaded.get("rollout"))
        gateway = cls._parse_gateway(loaded.get("gateway"), rollout.public_url)
        return cls(path=path, rollout=rollout, gateway=gateway)

    @property
    def bootstrap_nodes(self) -> list[dict[str, object]]:
        return [
            {
                "node_id": node.id,
                "gateway_url": node.public_url,
                "max_init_workers": node.max_init_workers,
                "max_run_workers": node.max_run_workers,
                "max_postrun_workers": node.max_postrun_workers,
                "ready_buffer_target": node.ready_buffer_target,
                "heartbeat_interval_seconds": self.gateway.heartbeat_interval_seconds,
            }
            for node in self.gateway.nodes
        ]

    def select_gateway_node(self, node_id: str | None = None) -> GatewayNodeConfig:
        if node_id:
            match = next((node for node in self.gateway.nodes if node.id == node_id), None)
            if match is None:
                raise ValueError(f"Unknown gateway node id: {node_id}")
            return match
        if len(self.gateway.nodes) != 1:
            raise ValueError(
                "Topology defines multiple gateway nodes; pass --node-id to choose one."
            )
        return self.gateway.nodes[0]

    @staticmethod
    def _parse_rollout(raw: Any) -> RolloutServiceConfig:
        rollout = _require_mapping(raw, "rollout")
        host = _require_non_empty_string(rollout.get("host", "0.0.0.0"), "rollout.host")
        port = _coerce_port(rollout.get("port", 8080), "rollout.port")
        public_url_raw = rollout.get("public_url")
        public_url = (
            _coerce_http_url(public_url_raw, "rollout.public_url")
            if public_url_raw is not None
            else _default_public_url(host, port)
        )
        save_dir = rollout.get("save_dir")
        if save_dir is not None:
            save_dir = _require_non_empty_string(save_dir, "rollout.save_dir")
        return RolloutServiceConfig(
            host=host,
            port=port,
            public_url=public_url,
            save_dir=save_dir,
            dispatch_poll_interval_seconds=_coerce_positive_float(
                rollout.get("dispatch_poll_interval_seconds", 1.0),
                "rollout.dispatch_poll_interval_seconds",
            ),
            callback_grace_seconds=_coerce_non_negative_float(
                rollout.get("callback_grace_seconds", 5.0),
                "rollout.callback_grace_seconds",
            ),
        )

    @staticmethod
    def _parse_gateway(raw: Any, rollout_public_url: str) -> GatewayConfig:
        gateway = _require_mapping(raw, "gateway")
        nodes_raw = gateway.get("nodes")
        if not isinstance(nodes_raw, list) or not nodes_raw:
            raise ValueError("gateway.nodes must be a non-empty list")

        seen_ids: set[str] = set()
        nodes: list[GatewayNodeConfig] = []
        for index, entry in enumerate(nodes_raw):
            node = _require_mapping(entry, f"gateway.nodes[{index}]")
            node_id = _require_non_empty_string(
                node.get("id", socket.gethostname()),
                f"gateway.nodes[{index}].id",
            )
            if node_id in seen_ids:
                raise ValueError(f"Duplicate gateway node id: {node_id}")
            seen_ids.add(node_id)

            host = _require_non_empty_string(
                node.get("host", "0.0.0.0"),
                f"gateway.nodes[{index}].host",
            )
            port = _coerce_port(node.get("port", 8081), f"gateway.nodes[{index}].port")
            public_url_raw = node.get("public_url")
            public_url = (
                _coerce_http_url(public_url_raw, f"gateway.nodes[{index}].public_url")
                if public_url_raw is not None
                else _default_public_url(host, port)
            )
            sglang = _require_mapping(
                node.get("vllm") or node.get("sglang"),
                f"gateway.nodes[{index}].vllm",
            )
            default_runtime_raw = node.get("default_runtime")
            default_runtime = None
            if default_runtime_raw is not None:
                if not isinstance(default_runtime_raw, dict):
                    raise ValueError(
                        f"gateway.nodes[{index}].default_runtime must be a mapping"
                    )
                default_runtime = RuntimeSpec.model_validate(default_runtime_raw)

            max_run_workers = _coerce_positive_int(
                node.get("max_run_workers", node.get("capacity", 1)),
                f"gateway.nodes[{index}].max_run_workers",
            )
            nodes.append(
                GatewayNodeConfig(
                    id=node_id,
                    host=host,
                    port=port,
                    public_url=public_url,
                    model_served=str(node.get("model_served", "")),
                    sglang_base_url=_coerce_http_url(
                        sglang.get("base_url", "http://127.0.0.1:8000"),
                        f"gateway.nodes[{index}].sglang.base_url",
                    ),
                    sglang_timeout=_coerce_positive_float(
                        sglang.get("timeout", 300.0),
                        f"gateway.nodes[{index}].sglang.timeout",
                    ),
                    max_init_workers=_coerce_positive_int(
                        node.get("max_init_workers", 4),
                        f"gateway.nodes[{index}].max_init_workers",
                    ),
                    max_run_workers=max_run_workers,
                    max_postrun_workers=_coerce_positive_int(
                        node.get("max_postrun_workers", 4),
                        f"gateway.nodes[{index}].max_postrun_workers",
                    ),
                    ready_buffer_target=_coerce_positive_int(
                        node.get("ready_buffer_target", max_run_workers),
                        f"gateway.nodes[{index}].ready_buffer_target",
                    ),
                    default_runtime=default_runtime,
                )
            )

        rollout_server_url_raw = gateway.get("rollout_server_url")
        rollout_server_url = (
            _coerce_http_url(rollout_server_url_raw, "gateway.rollout_server_url")
            if rollout_server_url_raw is not None
            else rollout_public_url
        )
        return GatewayConfig(
            heartbeat_interval_seconds=_coerce_positive_int(
                gateway.get("heartbeat_interval_seconds", 30),
                "gateway.heartbeat_interval_seconds",
            ),
            rollout_server_url=rollout_server_url,
            nodes=tuple(nodes),
        )


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _require_non_empty_string(value: object, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _coerce_port(value: object, field_name: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{field_name} must be between 1 and 65535")
    return port


def _coerce_positive_int(value: object, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return parsed


def _coerce_positive_float(value: object, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return parsed


def _coerce_non_negative_float(value: object, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0")
    return parsed


def _coerce_http_url(value: object, field_name: str) -> str:
    text = _require_non_empty_string(value, field_name)
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an http:// or https:// URL")
    return text.rstrip("/")


def _default_public_url(host: str, port: int) -> str:
    public_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{public_host}:{port}"
