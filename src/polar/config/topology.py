"""Load and validate a single Polar topology file."""

from __future__ import annotations

from pathlib import Path
import socket
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from polar.runtime.models import RuntimeSpec


class _StrictModel(BaseModel):
    """Pydantic base that rejects unknown keys so removed knobs fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class _InferenceConfig(_StrictModel):
    engine: Literal["sglang", "vllm"] = "sglang"
    scheduler: Literal["none", "thunderagent"] = "none"
    base_url: str = "http://127.0.0.1:8000"

    @field_validator("base_url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        return _normalize_http_url(value, "gateway.nodes[].inference.base_url")


class GatewayNodeConfig(_StrictModel):
    id: str = Field(default_factory=socket.gethostname)
    host: str = "0.0.0.0"
    port: int = Field(default=8081, ge=1, le=65535)
    public_url: str
    model_served: str = ""
    inference: _InferenceConfig = Field(default_factory=_InferenceConfig)
    max_init_workers: int = Field(default=4, gt=0)
    max_run_workers: int = Field(default=2, gt=0)
    max_postrun_workers: int = Field(default=4, gt=0)
    default_runtime: RuntimeSpec | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_public_url(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("public_url") in (None, ""):
            host = str(data.get("host", "0.0.0.0")).strip() or "0.0.0.0"
            port = int(data.get("port", 8081))
            data = {**data, "public_url": _default_public_url(host, port)}
        return data

    @field_validator("id", "host")
    @classmethod
    def _strip_non_empty(cls, value: str, info) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError(f"gateway.nodes[].{info.field_name} must be a non-empty string")
        return text

    @field_validator("model_served")
    @classmethod
    def _strip_model(cls, value: str) -> str:
        return (value or "").strip()

    @field_validator("public_url")
    @classmethod
    def _validate_public_url(cls, value: str) -> str:
        return _normalize_http_url(value, "gateway.nodes[].public_url")

    @property
    def inference_base_url(self) -> str:
        return self.inference.base_url

    @property
    def engine(self) -> str:
        return self.inference.engine


class _CompletionPersistenceConfig(_StrictModel):
    enabled: bool = True
    max_field_bytes: int = Field(default=1 * 1024 * 1024, gt=0)
    queue_size: int = Field(default=1024, gt=0)


class GatewayConfig(_StrictModel):
    heartbeat_interval_seconds: int = Field(default=30, gt=0)
    rollout_server_url: str | None = None
    nodes: tuple[GatewayNodeConfig, ...] = Field(min_length=1)
    completion_persistence: _CompletionPersistenceConfig = Field(
        default_factory=_CompletionPersistenceConfig
    )

    @field_validator("rollout_server_url")
    @classmethod
    def _validate_rollout_server_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_http_url(value, "gateway.rollout_server_url")

    @model_validator(mode="after")
    def _unique_node_ids(self) -> "GatewayConfig":
        seen: set[str] = set()
        for node in self.nodes:
            if node.id in seen:
                raise ValueError(f"Duplicate gateway node id: {node.id}")
            seen.add(node.id)
        return self


class RolloutServiceConfig(_StrictModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)
    public_url: str = ""
    save_dir: str | None = None
    dispatch_poll_interval_seconds: float = Field(default=1.0, gt=0)
    callback_grace_seconds: float = Field(default=120.0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _default_public_url(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("public_url") in (None, ""):
            host = str(data.get("host", "0.0.0.0")).strip() or "0.0.0.0"
            port = int(data.get("port", 8080))
            data = {**data, "public_url": _default_public_url(host, port)}
        return data

    @field_validator("host")
    @classmethod
    def _strip_host(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("rollout.host must be a non-empty string")
        return text

    @field_validator("save_dir")
    @classmethod
    def _strip_save_dir(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("rollout.save_dir must be a non-empty string")
        return text

    @field_validator("public_url")
    @classmethod
    def _validate_public_url(cls, value: str) -> str:
        return _normalize_http_url(value, "rollout.public_url")


class TopologyConfig(_StrictModel):
    rollout: RolloutServiceConfig = Field(default_factory=RolloutServiceConfig)
    gateway: GatewayConfig
    path: Path | None = None

    @classmethod
    def load(cls, config_path: str | Path = "topology.yaml") -> "TopologyConfig":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Topology file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Topology path is not a file: {path}")

        try:
            with path.open() as handle:
                loaded = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

        if not isinstance(loaded, dict):
            raise ValueError(f"Topology file {path} must contain a top-level mapping")

        loaded = {**loaded, "path": path}
        model = cls.model_validate(loaded)
        # Default gateway.rollout_server_url to rollout.public_url when unset.
        if model.gateway.rollout_server_url is None:
            gateway = model.gateway.model_copy(
                update={"rollout_server_url": model.rollout.public_url}
            )
            model = model.model_copy(update={"gateway": gateway})
        return model

    @property
    def bootstrap_nodes(self) -> list[dict[str, Any]]:
        return [
            {
                "node_id": node.id,
                "gateway_url": node.public_url,
                "max_init_workers": node.max_init_workers,
                "max_run_workers": node.max_run_workers,
                "max_postrun_workers": node.max_postrun_workers,
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


def _normalize_http_url(value: str, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an http:// or https:// URL")
    return text.rstrip("/")


def _default_public_url(host: str, port: int) -> str:
    public_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{public_host}:{port}"
