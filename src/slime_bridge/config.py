"""Configuration helpers for Slime-driven Polar rollouts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

from polar.config import TopologyConfig

_PLACEHOLDER_RE = re.compile(r"{([^{}]+)}")


@dataclass(frozen=True, slots=True)
class PolarSlimeConfig:
    rollout_server_url: str
    task_template: dict[str, Any]
    task_id_template: str
    instruction_template: str | None
    reward_key: str
    max_concurrency: int
    max_session_concurrency: int
    max_async_level: int
    fully_async: bool
    max_off_policy_steps: int
    request_timeout: float | None
    task_timeout_floor: float | None
    train_agent_timeout: float | None
    callback_host: str
    scoring_mode: str
    min_complete_accept_fraction: float
    early_stop_grace_sessions: int
    tokenizer_name_or_path: str | None
    add_generation_prompt: bool
    eval_dataset_name: str


def resolve_polar_slime_config(args: Any) -> PolarSlimeConfig:
    rollout_server_url = getattr(args, "polar_rollout_url", None)
    topology_path = getattr(args, "polar_topology_path", None)
    if rollout_server_url is None and topology_path:
        rollout_server_url = TopologyConfig.load(topology_path).rollout.public_url
    if rollout_server_url is None:
        raise ValueError(
            "Polar rollout URL is not configured. Set polar_rollout_url or polar_topology_path "
            "in Slime's custom config YAML."
        )

    task_template = deepcopy(getattr(args, "polar_task_template", None) or {})
    if not isinstance(task_template, dict):
        raise ValueError("polar_task_template must be a mapping")
    if "agent" not in task_template:
        raise ValueError("polar_task_template must include an agent spec")

    max_async_level = int(getattr(args, "polar_max_async_level", 2))
    if max_async_level <= 0:
        raise ValueError("polar_max_async_level must be greater than 0")

    fully_async_value = getattr(args, "polar_fully_async", False)
    if isinstance(fully_async_value, str):
        normalized = fully_async_value.strip().lower()
        if normalized not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError("polar_fully_async must be a boolean")
        fully_async = normalized in {"1", "true", "yes", "on"}
    else:
        fully_async = bool(fully_async_value)

    rollout_batch_size = int(getattr(args, "rollout_batch_size", 1) or 1)
    if rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size must be greater than 0")

    group_size = int(getattr(args, "n_samples_per_prompt", 1) or 1)
    if group_size <= 0:
        raise ValueError("n_samples_per_prompt must be greater than 0")

    update_weights_interval = int(getattr(args, "update_weights_interval", 1) or 1)
    if update_weights_interval <= 0:
        raise ValueError("update_weights_interval must be greater than 0")

    max_concurrency = rollout_batch_size * max_async_level
    max_session_concurrency = max_concurrency * group_size
    max_off_policy_steps = max_async_level + update_weights_interval

    request_timeout = getattr(args, "polar_request_timeout", None)
    if request_timeout is not None:
        request_timeout = float(request_timeout)
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("polar_request_timeout must be greater than 0")

    task_timeout_floor = getattr(args, "polar_task_timeout_floor", None)
    if task_timeout_floor is not None:
        task_timeout_floor = float(task_timeout_floor)
        if not math.isfinite(task_timeout_floor) or task_timeout_floor <= 0:
            raise ValueError("polar_task_timeout_floor must be greater than 0")

    train_agent_timeout = getattr(args, "polar_train_agent_timeout", None)
    if train_agent_timeout is not None:
        train_agent_timeout = _positive_finite_number(
            train_agent_timeout,
            field="polar_train_agent_timeout",
        )

    callback_host = str(getattr(args, "polar_callback_host", "127.0.0.1")).strip()
    if not callback_host:
        raise ValueError("polar_callback_host must be a non-empty host or IP")
    if callback_host in {"0.0.0.0", "::"}:
        raise ValueError(
            "polar_callback_host must be reachable by the rollout server, not a wildcard bind address"
        )

    scoring_mode = str(getattr(args, "polar_scoring_mode", "group")).strip().lower()
    if scoring_mode not in {"group", "individual"}:
        raise ValueError("polar_scoring_mode must be 'group' or 'individual'")

    min_complete_accept_fraction = float(
        getattr(args, "polar_min_complete_accept_fraction", 0.0) or 0.0
    )
    if not 0.0 <= min_complete_accept_fraction <= 1.0:
        raise ValueError("polar_min_complete_accept_fraction must be between 0 and 1")
    early_stop_grace_sessions = int(getattr(args, "polar_early_stop_grace_sessions", 2) or 0)
    if early_stop_grace_sessions < 0:
        raise ValueError("polar_early_stop_grace_sessions must be non-negative")

    return PolarSlimeConfig(
        rollout_server_url=str(rollout_server_url).rstrip("/"),
        task_template=task_template,
        task_id_template=str(
            getattr(
                args, "polar_task_id_template", "polar-slime-{rollout_id}-{sample.group_index}"
            )
        ),
        instruction_template=getattr(args, "polar_instruction_template", None),
        reward_key=str(
            getattr(args, "polar_reward_key", None) or getattr(args, "reward_key", None) or "score"
        ),
        max_concurrency=max_concurrency,
        max_session_concurrency=max_session_concurrency,
        max_async_level=max_async_level,
        fully_async=fully_async,
        max_off_policy_steps=max_off_policy_steps,
        request_timeout=request_timeout,
        task_timeout_floor=task_timeout_floor,
        train_agent_timeout=train_agent_timeout,
        callback_host=callback_host,
        scoring_mode=scoring_mode,
        min_complete_accept_fraction=min_complete_accept_fraction,
        early_stop_grace_sessions=early_stop_grace_sessions,
        tokenizer_name_or_path=getattr(args, "hf_checkpoint", None),
        add_generation_prompt=bool(getattr(args, "polar_add_generation_prompt", True)),
        eval_dataset_name=str(getattr(args, "polar_eval_dataset_name", "polar_eval")),
    )


def resolve_sglang_router_base_url(args: Any) -> str | None:
    ip = getattr(args, "sglang_router_ip", None)
    port = getattr(args, "sglang_router_port", None)
    if ip in (None, "") or port in (None, ""):
        return None
    return f"http://{ip}:{port}"


def render_task_payload(
    *,
    args: Any,
    config: PolarSlimeConfig,
    sample: Any,
    instruction: str,
    rollout_id: int,
    task_position: int,
    num_rollouts: int,
    is_eval: bool = False,
) -> dict[str, Any]:
    context = _build_context(
        args=args,
        sample=sample,
        instruction=instruction,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=num_rollouts,
    )
    payload = _render_template_value(deepcopy(config.task_template), context)
    if not isinstance(payload, dict):
        raise ValueError("polar_task_template must render to a mapping")

    payload["task_id"] = str(_render_template_value(config.task_id_template, context))
    payload["instruction"] = instruction
    payload["num_samples"] = num_rollouts
    _apply_task_timeout_floor(payload, config.task_timeout_floor)
    metadata = getattr(sample, "metadata", None) or {}
    if isinstance(metadata, dict):
        _apply_sample_runtime_metadata(payload, metadata)
        _apply_sample_agent_timeout(payload, metadata)
    if config.train_agent_timeout is not None and not is_eval:
        task_metadata = payload.setdefault("metadata", {})
        if not isinstance(task_metadata, dict):
            raise ValueError("rendered task payload metadata must be a mapping")
        task_metadata["agent_timeout"] = config.train_agent_timeout
    if isinstance(metadata, dict) and metadata.get("agent_step_limit") is not None:
        try:
            step_limit = int(metadata["agent_step_limit"])
        except (TypeError, ValueError) as exc:
            raise ValueError("sample.metadata.agent_step_limit must be an integer") from exc
        if step_limit <= 0:
            raise ValueError("sample.metadata.agent_step_limit must be positive")
        agent = payload.get("agent")
        if not isinstance(agent, dict):
            raise ValueError("rendered task payload must include an agent mapping")
        settings = agent.setdefault("settings", {})
        if not isinstance(settings, dict):
            raise ValueError("rendered task payload agent.settings must be a mapping")
        settings["step_limit"] = step_limit
    if 0.0 < config.min_complete_accept_fraction < 1.0:
        required = math.ceil(num_rollouts * config.min_complete_accept_fraction)
        # Waiting for a small buffer above the eventual trainer acceptance
        # threshold absorbs parser-invalid/empty completions while avoiding
        # the slowest group members becoming a hard all-session barrier.
        payload["early_stop_min_usable_sessions"] = min(
            num_rollouts,
            required + config.early_stop_grace_sessions,
        )
    return payload


def _positive_finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return parsed


def _apply_task_timeout_floor(payload: dict[str, Any], timeout_floor: float | None) -> None:
    """Clamp the rendered infrastructure budget without changing dataset rows."""

    if timeout_floor is None:
        return
    rendered_timeout = payload.get("timeout_seconds")
    if rendered_timeout is None:
        payload["timeout_seconds"] = timeout_floor
        return
    parsed_timeout = _positive_finite_number(
        rendered_timeout,
        field="rendered task timeout_seconds",
    )
    payload["timeout_seconds"] = max(parsed_timeout, timeout_floor)


def _apply_sample_agent_timeout(payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Copy the benchmark's agent budget into the trusted task metadata."""

    if "agent_timeout" not in metadata:
        return
    agent_timeout = _positive_finite_number(
        metadata["agent_timeout"],
        field="sample.metadata.agent_timeout",
    )
    task_metadata = payload.setdefault("metadata", {})
    if not isinstance(task_metadata, dict):
        raise ValueError("rendered task payload metadata must be a mapping")
    task_metadata["agent_timeout"] = agent_timeout


def _apply_sample_runtime_metadata(payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Merge trusted dataset runtime semantics that cannot be YAML-templated.

    Slime's scalar placeholder renderer cannot splice an arbitrary environment
    mapping into ``RuntimeSpec.env``.  Exported Harbor datasets need that for
    OCI ``ENV`` values, and occasionally need an image ``CMD``/``ENTRYPOINT``
    started once by the long-lived direct-exec broker.  Keep the extension
    narrow, typed, and fail-closed so malformed dataset rows never silently run
    under different container semantics.
    """

    runtime_env = metadata.get("runtime_env")
    runtime_init = metadata.get("runtime_init_command")
    if runtime_env is None and runtime_init is None:
        return

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("sample runtime metadata requires a rendered runtime mapping")

    if runtime_env is not None:
        if not isinstance(runtime_env, dict) or not all(
            isinstance(key, str) and key and isinstance(value, str)
            for key, value in runtime_env.items()
        ):
            raise ValueError("sample.metadata.runtime_env must map non-empty strings to strings")
        environment = runtime.setdefault("env", {})
        if not isinstance(environment, dict):
            raise ValueError("rendered runtime.env must be a mapping")
        environment.update(runtime_env)

    if runtime_init is not None:
        if not isinstance(runtime_init, str) or not runtime_init.strip():
            raise ValueError("sample.metadata.runtime_init_command must be a non-empty string")
        existing_init = runtime.get("direct_exec_init_command")
        if existing_init is None:
            runtime["direct_exec_init_command"] = runtime_init.strip()
        elif not isinstance(existing_init, str) or not existing_init.strip():
            raise ValueError("rendered runtime.direct_exec_init_command must be non-empty")
        else:
            existing = existing_init.strip().rstrip(";").rstrip()
            additional = runtime_init.strip().rstrip(";").rstrip()
            runtime["direct_exec_init_command"] = f"{{ {existing}; }} && {{ {additional}; }}"


def render_instruction(
    *,
    args: Any,
    config: PolarSlimeConfig,
    sample: Any,
    prompt_text: str,
    rollout_id: int,
    task_position: int,
    num_rollouts: int,
) -> str:
    template = config.instruction_template
    if not template:
        return prompt_text
    context = _build_context(
        args=args,
        sample=sample,
        instruction=prompt_text,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=num_rollouts,
    )
    rendered = _render_template_value(template, context)
    if not isinstance(rendered, str):
        raise ValueError("polar_instruction_template must render to a string")
    return rendered


def render_topology_template(topology_path: str | Path, args: Any) -> dict[str, Any]:
    """Load a topology template and point every gateway node at Slime's router."""
    router_url = resolve_sglang_router_base_url(args)
    if router_url is None:
        raise ValueError("sglang_router_ip and sglang_router_port must be set to render topology")

    topology = TopologyConfig.load(topology_path)
    return {
        "rollout": {
            "host": topology.rollout.host,
            "port": topology.rollout.port,
            "public_url": topology.rollout.public_url,
            "save_dir": topology.rollout.save_dir,
            "dispatch_poll_interval_seconds": topology.rollout.dispatch_poll_interval_seconds,
            "callback_grace_seconds": topology.rollout.callback_grace_seconds,
            "http_max_connections": topology.rollout.http_max_connections,
            "http_max_keepalive_connections": (topology.rollout.http_max_keepalive_connections),
            "cleanup_max_concurrency": topology.rollout.cleanup_max_concurrency,
            "cleanup_max_attempts": topology.rollout.cleanup_max_attempts,
            "cleanup_retry_backoff_seconds": (topology.rollout.cleanup_retry_backoff_seconds),
        },
        "gateway": {
            "heartbeat_interval_seconds": topology.gateway.heartbeat_interval_seconds,
            "rollout_server_url": topology.gateway.rollout_server_url,
            "nodes": [
                {
                    "id": node.id,
                    "host": node.host,
                    "port": node.port,
                    "public_url": node.public_url,
                    "model_served": node.model_served,
                    "inference": {
                        "engine": "sglang",
                        "base_url": router_url,
                    },
                    "max_init_workers": node.max_init_workers,
                    "max_run_workers": node.max_run_workers,
                    "max_postrun_workers": node.max_postrun_workers,
                    **(
                        {"default_runtime": node.default_runtime.model_dump(mode="python")}
                        if node.default_runtime is not None
                        else {}
                    ),
                }
                for node in topology.gateway.nodes
            ],
        },
    }


def _build_context(
    *,
    args: Any,
    sample: Any,
    instruction: str,
    rollout_id: int,
    task_position: int,
    num_rollouts: int,
) -> dict[str, Any]:
    args_namespace = SimpleNamespace(**vars(args)) if hasattr(args, "__dict__") else args
    metadata = deepcopy(getattr(sample, "metadata", None) or {})
    return {
        "args": args_namespace,
        "instruction": instruction,
        "num_rollouts": num_rollouts,
        "rollout_id": rollout_id,
        "sglang": SimpleNamespace(router_base_url=resolve_sglang_router_base_url(args)),
        "sample": SimpleNamespace(
            prompt=deepcopy(getattr(sample, "prompt", "")),
            response=deepcopy(getattr(sample, "response", "")),
            label=getattr(sample, "label", None),
            metadata=_to_namespace(metadata),
            index=getattr(sample, "index", None),
            group_index=getattr(sample, "group_index", None),
            status=getattr(sample, "status", None),
        ),
        "task_position": task_position,
    }


def _render_template_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        if match := re.fullmatch(r"{([^{}]+)}", value):
            resolved = deepcopy(_resolve_path(context, match.group(1)))
            return _from_namespace(resolved)

        def replace(match: re.Match[str]) -> str:
            resolved = _resolve_path(context, match.group(1))
            return "" if resolved is None else str(resolved)

        return _PLACEHOLDER_RE.sub(replace, value)

    if isinstance(value, list):
        return [_render_template_value(item, context) for item in value]

    if isinstance(value, dict):
        return {str(key): _render_template_value(item, context) for key, item in value.items()}

    return value


def _resolve_path(context: dict[str, Any], path: str) -> Any:
    current: Any = context
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise ValueError(f"Unknown template variable: {path}")
            current = current[part]
            continue

        if hasattr(current, part):
            current = getattr(current, part)
            continue

        raise ValueError(f"Unknown template variable: {path}")
    return current


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _from_namespace(value: Any) -> Any:
    """Convert SimpleNamespace trees back to plain dicts for JSON serialization."""
    if isinstance(value, SimpleNamespace):
        return {k: _from_namespace(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: _from_namespace(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_namespace(item) for item in value]
    return value
