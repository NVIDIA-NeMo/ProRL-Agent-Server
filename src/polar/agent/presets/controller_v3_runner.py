#!/usr/bin/env python3
"""Protected direct runner for dynamically uploaded Controller V3."""

from __future__ import annotations

import base64
import ctypes
import importlib.machinery
import importlib.util
import json
import os
import socket
import struct
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

RUNNER_DIR = Path(__file__).resolve().parent
BASE_CONFIG_PATH = Path("/opt/polar-mini-swe-agent/config/controller-v3-v1.yaml")
OUTPUT_PATH = Path("/polar/session/logs/agent/mini-swe-agent.traj.json")

_ROUTER_CAPABILITY = "POLAR_ROUTER_CAPABILITY"
_POOL_CAPABILITY = "POLAR_MODEL_POOL_CAPABILITY"
_SOCKET_ENV = "POLAR_PROTECTED_EXEC_SOCKET"
_BROKER_PID_ENV = "POLAR_PROTECTED_EXEC_BROKER_PID"
_REQUEST_ID_ENV = "POLAR_PROTECTED_EXEC_REQUEST_ID"
_READY_FD_ENV = "POLAR_PROTECTED_EXEC_READY_FD"
_MODULE_FD_ENV = "POLAR_CONTROLLER_V3_MODULE_FD"
_CONFIG_FD_ENV = "POLAR_CONTROLLER_V3_CONFIG_FD"
_PR_SET_DUMPABLE = 4
_PR_GET_DUMPABLE = 3
_MAX_SECRET_BYTES = 16 * 1024
_GPT_INPUT_USD_PER_MILLION = 1.0
_GPT_CACHED_INPUT_USD_PER_MILLION = 0.1
_GPT_OUTPUT_USD_PER_MILLION = 6.0
_GPT_PRICING_AS_OF = "2026-07-09"


class ProtectedExecutionError(RuntimeError):
    pass


def _protected_source_path(env_name: str, fallback: Path) -> Path:
    descriptor = os.environ.pop(env_name, "")
    if not descriptor:
        return fallback
    if not descriptor.isdigit() or int(descriptor) < 3:
        raise ProtectedExecutionError(f"invalid protected file descriptor: {env_name}")
    return Path("/proc/self/fd") / descriptor


MODULE_PATH = _protected_source_path(
    _MODULE_FD_ENV, RUNNER_DIR / "oracle_controller_v3.py"
)
CONFIG_PATH = _protected_source_path(
    _CONFIG_FD_ENV, RUNNER_DIR / "controller_v3_one_vote.yaml"
)


def _set_non_dumpable() -> None:
    if sys.platform != "linux":
        raise ProtectedExecutionError("protected runner requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise ProtectedExecutionError("could not harden protected runner")
    if libc.prctl(_PR_GET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise ProtectedExecutionError("protected runner remains dumpable")


def _receive_capabilities() -> dict[str, str]:
    for key in (_ROUTER_CAPABILITY, _POOL_CAPABILITY):
        if os.environ.pop(key, ""):
            raise ProtectedExecutionError(
                "capability appeared in the runner initial environment"
            )

    _set_non_dumpable()
    try:
        ready_fd = int(os.environ.pop(_READY_FD_ENV))
        broker_pid = int(os.environ.pop(_BROKER_PID_ENV))
        socket_path = os.environ.pop(_SOCKET_ENV)
        request_id = os.environ.pop(_REQUEST_ID_ENV)
    except (KeyError, ValueError) as exc:
        raise ProtectedExecutionError("protected channel is unavailable") from exc
    if ready_fd < 3 or broker_pid <= 0 or not socket_path or not request_id:
        raise ProtectedExecutionError("protected channel metadata is invalid")

    try:
        if os.read(ready_fd, 1) != b"1":
            raise ProtectedExecutionError("protected readiness gate was not released")
    finally:
        os.close(ready_fd)

    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(30.0)
    try:
        connection.connect(socket_path)
        peer_pid, _, _ = struct.unpack(
            "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
        if peer_pid != broker_pid:
            raise ProtectedExecutionError("protected broker identity mismatch")
        request = json.dumps(
            {"operation": "protected_child_ready", "id": request_id},
            separators=(",", ":"),
        ).encode()
        connection.sendall(request + b"\n")
        response = bytearray()
        while b"\n" not in response:
            chunk = connection.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
            if len(response) > _MAX_SECRET_BYTES:
                raise ProtectedExecutionError("protected response is too large")
    finally:
        connection.close()

    try:
        payload = json.loads(bytes(response).split(b"\n", 1)[0])
    except (ValueError, json.JSONDecodeError) as exc:
        raise ProtectedExecutionError("protected response is invalid") from exc
    values = payload.get("protected_env") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise ProtectedExecutionError("protected broker rejected the request")
    if not isinstance(values, dict):
        raise ProtectedExecutionError("protected capabilities are missing")

    result: dict[str, str] = {}
    for key in (_ROUTER_CAPABILITY, _POOL_CAPABILITY):
        value = values.get(key)
        if not isinstance(value, str) or not value:
            raise ProtectedExecutionError(f"required capability is missing: {key}")
        result[key] = value
    return result


def _load_controller_module() -> None:
    name = "minisweagent.agents.oracle_controller_v3"
    loader = importlib.machinery.SourceFileLoader(name, str(MODULE_PATH))
    spec = importlib.util.spec_from_loader(name, loader)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load dynamic Controller V3 module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def _configure_gateway_transport() -> tuple[Any | None, Any | None]:
    socket_path = os.environ.get("POLAR_GATEWAY_UDS", "").strip()
    if not socket_path:
        return None, None
    import httpx
    import litellm
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = httpx.Client(
        transport=httpx.HTTPTransport(uds=socket_path),
        trust_env=False,
    )
    litellm.client_session = client
    # LiteLLM's native Responses path does not consult client_session.  It
    # accepts its own HTTPHandler via the per-request ``client`` argument.
    return client, HTTPHandler(client=client)


def _with_capability(
    config: Any,
    capability: str,
    call: Callable[..., Any],
    *args: Any,
    request_client: Any | None = None,
    **kwargs: Any,
) -> Any:
    original = config.model_kwargs
    headers = original.get("extra_headers")
    headers = headers if isinstance(headers, dict) else {}
    config.model_kwargs = {
        **original,
        **({"client": request_client} if request_client is not None else {}),
        "extra_headers": {
            **headers,
            "Authorization": f"Bearer {capability}",
        },
    }
    try:
        return call(*args, **kwargs)
    finally:
        config.model_kwargs = original


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("token count must be an integer")
    parsed = int(value or 0)
    if parsed < 0:
        raise ValueError("token count must be nonnegative")
    return parsed


def _price_large_worker_response(agent: Any, message: dict[str, Any]) -> None:
    usage = message.get("usage")
    if not isinstance(usage, dict):
        response = (message.get("extra") or {}).get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        raise RuntimeError("GPT response is missing usage")

    input_tokens = _nonnegative_int(
        usage.get("input_tokens", usage.get("prompt_tokens"))
    )
    output_tokens = _nonnegative_int(
        usage.get("output_tokens", usage.get("completion_tokens"))
    )
    details = usage.get("input_tokens_details")
    if not isinstance(details, dict):
        details = usage.get("prompt_tokens_details")
    cached_tokens = _nonnegative_int(
        details.get("cached_tokens") if isinstance(details, dict) else 0
    )
    if cached_tokens > input_tokens:
        raise RuntimeError("GPT cached input tokens exceed total input tokens")
    uncached_tokens = input_tokens - cached_tokens
    cost = (
        uncached_tokens * _GPT_INPUT_USD_PER_MILLION
        + cached_tokens * _GPT_CACHED_INPUT_USD_PER_MILLION
        + output_tokens * _GPT_OUTPUT_USD_PER_MILLION
    ) / 1_000_000

    message.setdefault("extra", {})["cost"] = cost
    totals = agent.usage["large"]
    for key, value in (
        ("input_tokens", input_tokens),
        ("cached_input_tokens", cached_tokens),
        ("uncached_input_tokens", uncached_tokens),
        ("output_tokens", output_tokens),
    ):
        totals[key] = int(totals.get(key, 0)) + value
    totals["pricing"] = {
        "currency": "USD",
        "input_per_million": _GPT_INPUT_USD_PER_MILLION,
        "cached_input_per_million": _GPT_CACHED_INPUT_USD_PER_MILLION,
        "output_per_million": _GPT_OUTPUT_USD_PER_MILLION,
        "as_of": _GPT_PRICING_AS_OF,
    }


def _install_capability_guards(
    agent: Any,
    *,
    router_capability: str,
    pool_capability: str,
    responses_client: Any | None = None,
) -> None:
    try:
        from minisweagent.exceptions import FormatError
    except ImportError:  # pragma: no cover - always present in the sealed runtime

        class FormatError(Exception):  # type: ignore[no-redef]
            """Stand-in so the guards import outside the runtime; never raised there."""

    large_usage = agent.usage["large"]
    for key in (
        "input_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
    ):
        large_usage.setdefault(key, 0)
    large_usage.setdefault(
        "pricing",
        {
            "currency": "USD",
            "input_per_million": _GPT_INPUT_USD_PER_MILLION,
            "cached_input_per_million": _GPT_CACHED_INPUT_USD_PER_MILLION,
            "output_per_million": _GPT_OUTPUT_USD_PER_MILLION,
            "as_of": _GPT_PRICING_AS_OF,
        },
    )
    for model in (agent.small_model, agent.large_model):
        original_query = model.query
        config = model.config
        request_client = responses_client if model is agent.large_model else None
        price_response = model is agent.large_model

        def guarded_query(
            messages: list[dict[str, Any]],
            _query: Callable[..., Any] = original_query,
            _config: Any = config,
            _request_client: Any | None = request_client,
            _price_response: bool = price_response,
            **kwargs: Any,
        ) -> dict[str, Any]:
            try:
                message = _with_capability(
                    _config,
                    pool_capability,
                    _query,
                    messages,
                    request_client=_request_client,
                    **kwargs,
                )
            except FormatError as exc:
                # The paid responses() call already happened before the tool
                # call parse raised; the model stashes the billed response on
                # the exception (InterruptAgentFlow keeps it as a TUPLE). Bill
                # it (best-effort) so a tool-call-less turn still counts toward
                # cost, then let FormatError propagate. _account never runs on
                # this path, so mirror it here to land the dollars in the
                # usage['large']['cost'] field postprocess harvests.
                if _price_response:
                    error_messages = getattr(exc, "messages", None)
                    if isinstance(error_messages, (list, tuple)) and error_messages:
                        try:
                            _price_large_worker_response(agent, error_messages[0])
                            billed = (
                                error_messages[0].get("extra", {}).get("cost", 0.0)
                                or 0.0
                            )
                            totals = agent.usage["large"]
                            totals["n_calls"] = int(totals.get("n_calls", 0)) + 1
                            totals["cost"] = float(totals.get("cost", 0.0)) + billed
                        except Exception:
                            pass
                raise
            if _price_response:
                _price_large_worker_response(agent, message)
            return message

        model.query = guarded_query

    original_controller_query = agent._query_controller_model
    controller_config = agent.controller_model.config

    def guarded_controller_query(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return _with_capability(
            controller_config,
            router_capability,
            original_controller_query,
            messages,
        )

    agent._query_controller_model = guarded_controller_query


def _decode_task() -> str:
    encoded = os.environ.pop("CONTROLLER_V3_TASK_B64", "")
    if not encoded:
        raise RuntimeError("Controller V3 task is missing")
    try:
        return base64.b64decode(encoded, validate=True).decode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("Controller V3 task is invalid") from exc


def _optional_number(name: str, cast: Callable[[str], Any]) -> Any | None:
    value = os.environ.pop(name, "")
    return cast(value) if value else None


def _build_agent(task: str) -> Any:
    from minisweagent.agents import get_agent
    from minisweagent.config import get_config_from_spec
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model
    from minisweagent.utils.serialize import recursive_merge

    overrides: dict[str, Any] = {
        "run": {"task": task},
        "agent": {
            "mode": "yolo",
            "confirm_exit": False,
            "output_path": OUTPUT_PATH,
        },
        "environment": {"environment_class": "local"},
    }
    step_limit = _optional_number("CONTROLLER_V3_STEP_LIMIT", int)
    cost_limit = _optional_number("CONTROLLER_V3_COST_LIMIT", float)
    wall_time = _optional_number("CONTROLLER_V3_WALL_TIME_LIMIT_SECONDS", float)
    if step_limit is not None:
        overrides["agent"]["step_limit"] = step_limit
    if cost_limit is not None:
        overrides["agent"]["cost_limit"] = cost_limit
    if wall_time is not None:
        overrides["agent"]["wall_time_limit_seconds"] = wall_time

    config = recursive_merge(
        get_config_from_spec(str(BASE_CONFIG_PATH)),
        yaml.safe_load(CONFIG_PATH.read_text()),
        overrides,
    )
    # A Responses request must not inherit the chat-completions token field.
    if config.get("model", {}).get("model_class") == "litellm_response":
        config["model"]["model_kwargs"].pop("max_tokens", None)

    model = get_model(config=config["model"])
    environment = get_environment(config["environment"], default_type="local")
    return get_agent(
        model,
        environment,
        config["agent"],
        default_type="interactive",
    )


def main() -> int:
    capabilities = _receive_capabilities()
    task = _decode_task()
    _load_controller_module()
    gateway_client, responses_client = _configure_gateway_transport()
    try:
        agent = _build_agent(task)
        _install_capability_guards(
            agent,
            router_capability=capabilities.pop(_ROUTER_CAPABILITY),
            pool_capability=capabilities.pop(_POOL_CAPABILITY),
            responses_client=responses_client,
        )
        agent.run(task)
        return 0
    finally:
        capabilities.clear()
        if gateway_client is not None:
            gateway_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
