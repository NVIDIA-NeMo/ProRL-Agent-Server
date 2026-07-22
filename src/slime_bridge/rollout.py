"""Slime rollout bridge for Polar-managed agent sessions.

Single entrypoint ``generate_rollout_polar_async`` routes training to a
persistent background worker and evaluation to a one-shot submit+poll batch.
Both paths speak Polar's async-only HTTP surface (``/rollout/task/submit`` +
``/rollout/task/{task_id}``).
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import copy
import hashlib
import json
import logging
import math
import os
import queue
import re
import statistics
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request

from polar.http_logging import uvicorn_access_log_enabled
from polar.rollout.models import TaskResult, TaskStatus
from polar.runtime.command_timing import RUNTIME_EXEC_CATEGORIES
from slime_bridge._messages import prompt_to_instruction_text
from slime_bridge.adapter import (
    RolloutLogprobError,
    session_result_to_placeholder,
    session_result_to_samples,
)
from slime_bridge.config import (
    PolarSlimeConfig,
    render_instruction,
    render_task_payload,
    resolve_polar_slime_config,
)
from slime_bridge.partial_rollout import (
    PartialRolloutError,
    PartialRolloutStore,
    STATE_DROP,
    STATE_KEEP,
    STATE_PREPARED,
    STATE_RESULT_READY,
    maybe_open_partial_rollout_store,
)

logger = logging.getLogger(__name__)

_WEIGHT_UPDATE_GATEWAY_STATE_LOCK = threading.Lock()
_WEIGHT_UPDATE_PAUSED_GATEWAYS: tuple[str, ...] = ()


def _control_plane_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("POLAR_CONTROL_PLANE_TOKEN", "").strip()
    if token:
        headers["X-Polar-Control-Token"] = token
    return headers


def _weight_update_gateway_urls(args: Any) -> tuple[str, ...]:
    """Discover the complete registered gateway fleet from the rollout service."""

    rollout_url = str(getattr(args, "polar_rollout_url", "") or "").rstrip("/")
    if not rollout_url:
        raise RuntimeError("polar_rollout_url is required for weight-update coordination")
    with httpx.Client(timeout=30.0, headers=_control_plane_headers()) as client:
        response = client.get(f"{rollout_url}/nodes")
        response.raise_for_status()
        nodes = response.json()
    if not isinstance(nodes, list) or not nodes:
        raise RuntimeError("Polar rollout service returned an empty or malformed gateway fleet")

    urls: list[str] = []
    for node in nodes:
        gateway_url = node.get("gateway_url") if isinstance(node, dict) else None
        if not isinstance(gateway_url, str) or not gateway_url.strip():
            raise RuntimeError("Polar rollout service returned a node without gateway_url")
        urls.append(gateway_url.rstrip("/"))
    if len(set(urls)) != len(urls):
        raise RuntimeError("Polar rollout service returned duplicate gateway URLs")
    return tuple(urls)


def _gateway_generation_control(
    gateway_url: str,
    action: str,
    *,
    pause_timeout_seconds: float,
) -> dict[str, Any]:
    request_timeout = pause_timeout_seconds + 30.0 if action == "pause" else 30.0
    params = {"timeout_seconds": pause_timeout_seconds} if action == "pause" else None
    with httpx.Client(
        timeout=request_timeout,
        headers=_control_plane_headers(),
    ) as client:
        response = client.post(
            f"{gateway_url}/admin/inference/{action}",
            params=params,
        )
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Gateway {gateway_url} returned malformed {action} status")
    expected_paused = action == "pause"
    if payload.get("paused") is not expected_paused:
        raise RuntimeError(
            f"Gateway {gateway_url} did not enter expected paused={expected_paused} state"
        )
    inflight = payload.get("inflight")
    if action == "pause" and (type(inflight) is not int or inflight != 0):
        raise RuntimeError(
            f"Gateway {gateway_url} pause returned nonzero or malformed inflight={inflight!r}"
        )
    return payload


def _control_gateway_fleet(
    gateway_urls: tuple[str, ...],
    action: str,
    *,
    pause_timeout_seconds: float,
) -> None:
    errors: list[str] = []
    with ThreadPoolExecutor(
        max_workers=len(gateway_urls),
        thread_name_prefix=f"polar-gateway-{action}",
    ) as executor:
        futures = {
            executor.submit(
                _gateway_generation_control,
                url,
                action,
                pause_timeout_seconds=pause_timeout_seconds,
            ): url
            for url in gateway_urls
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError(
            f"Failed to {action} Polar gateway inference fleet: " + "; ".join(errors)
        )


def pause_for_weight_update(args: Any) -> None:
    """Freeze new Router generations and drain every gateway before weight sync.

    Slime may keep a fully-async Polar window alive after ``generate`` returns.
    Draining the gateway proxies closes that race before SGLang's destructive
    pause/flush cycle.  A partial pause is rolled back before the error is
    surfaced, so a failed precondition cannot strand rollout traffic.
    """

    global _WEIGHT_UPDATE_PAUSED_GATEWAYS
    timeout = float(getattr(args, "polar_weight_update_pause_timeout", 300.0))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("polar_weight_update_pause_timeout must be positive and finite")
    with _WEIGHT_UPDATE_GATEWAY_STATE_LOCK:
        if _WEIGHT_UPDATE_PAUSED_GATEWAYS:
            raise RuntimeError("Polar gateway fleet is already paused for a weight update")
    gateway_urls = _weight_update_gateway_urls(args)
    try:
        _control_gateway_fleet(
            gateway_urls,
            "pause",
            pause_timeout_seconds=timeout,
        )
    except Exception:
        try:
            _control_gateway_fleet(
                gateway_urls,
                "resume",
                pause_timeout_seconds=timeout,
            )
        except Exception:
            logger.exception("Failed to roll back a partial Polar gateway pause")
        raise
    with _WEIGHT_UPDATE_GATEWAY_STATE_LOCK:
        _WEIGHT_UPDATE_PAUSED_GATEWAYS = gateway_urls
    logger.info("Paused and drained %d Polar gateways for weight update", len(gateway_urls))


def resume_after_weight_update(args: Any) -> None:
    """Resume the exact gateway fleet frozen by :func:`pause_for_weight_update`."""

    global _WEIGHT_UPDATE_PAUSED_GATEWAYS
    timeout = float(getattr(args, "polar_weight_update_pause_timeout", 300.0))
    with _WEIGHT_UPDATE_GATEWAY_STATE_LOCK:
        gateway_urls = _WEIGHT_UPDATE_PAUSED_GATEWAYS
    if not gateway_urls:
        raise RuntimeError("Polar gateway fleet was not paused for a weight update")
    _control_gateway_fleet(
        gateway_urls,
        "resume",
        pause_timeout_seconds=timeout,
    )
    with _WEIGHT_UPDATE_GATEWAY_STATE_LOCK:
        _WEIGHT_UPDATE_PAUSED_GATEWAYS = ()
    logger.info("Resumed %d Polar gateways after weight update", len(gateway_urls))


_POLL_INTERVAL = 2.0  # seconds between task-status polls (eval / no-callback path)
_CALLBACK_FALLBACK_POLL_SECONDS = 60.0  # defensive backstop for dropped callbacks
_TASK_STATUS_GET_MAX_ATTEMPTS = 5
_TASK_STATUS_GET_RETRY_BACKOFF_SECONDS = 0.5
_TASK_STATUS_GET_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_TRAJECTORY_EXAMPLE_INTERVAL = 10
_TRAJECTORY_EXAMPLE_COUNT = 2
_EVAL_DATA_INTEGRITY_ENV = "POLAR_EVAL_DATA_INTEGRITY_B64"
_EVAL_SAMPLING_SEED_METADATA_KEY = "eval_sampling_seed_base"
_EVAL_DISPATCH_PRIORITY = 100

_EVAL_STANDARD_MODEL_KWARGS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("temperature", "temperature", ("eval_temperature", "rollout_temperature")),
    ("top_p", "top_p", ("eval_top_p", "rollout_top_p")),
    (
        "max_response_len",
        "max_tokens",
        ("eval_max_response_len", "rollout_max_response_len"),
    ),
    ("stop", "stop", ("rollout_stop",)),
)
_EVAL_SGLANG_EXTRA_BODY: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("top_k", "top_k", ("eval_top_k", "rollout_top_k")),
    ("stop_token_ids", "stop_token_ids", ("rollout_stop_token_ids",)),
    ("min_new_tokens", "min_tokens", ("eval_min_new_tokens",)),
    ("repetition_penalty", "repetition_penalty", ()),
    ("skip_special_tokens", "skip_special_tokens", ("rollout_skip_special_tokens",)),
    ("no_stop_trim", "no_stop_trim", ()),
)

_eval_tokenizer_cache: dict[str, Any] = {}

_SESSION_STAGE_TIMING_FIELDS: tuple[tuple[str, str], ...] = (
    ("register_to_init_queue_ms", "register_to_init_queue_mean"),
    ("rollout_dispatch_ms", "rollout_dispatch_mean"),
    ("rollout_result_wait_ms", "rollout_result_wait_mean"),
    ("rollout_pipeline_e2e_ms", "rollout_pipeline_e2e_mean"),
    ("init_ms", "init_mean"),
    ("ready_queue_ms", "ready_queue_mean"),
    ("container_start_ms", "container_start_mean"),
    ("eval_container_start_ms", "eval_container_start_mean"),
    ("runtime_validation_ms", "runtime_validation_mean"),
    ("eval_runtime_validation_ms", "eval_runtime_validation_mean"),
    ("prepare_ms", "prepare_mean"),
    ("eval_prepare_ms", "eval_prepare_mean"),
    ("run_ms", "run_mean"),
    ("postrun_queue_ms", "postrun_queue_mean"),
    ("agent_setup_ms", "agent_setup_mean"),
    ("agent_exec_ms", "agent_exec_mean"),
    ("agent_postprocess_ms", "agent_postprocess_mean"),
    ("postrun_ms", "postrun_mean"),
    ("build_ms", "build_mean"),
    ("eval_ms", "eval_mean"),
    ("postrun_exec_ms", "postrun_exec_mean"),
    ("runtime_stop_ms", "runtime_stop_mean"),
    ("e2e_ms", "e2e_mean"),
)

_INFERENCE_TIMING_FIELDS: tuple[str, ...] = (
    "e2e_ms",
    "api_dispatch_ms",
    "request_to_forward_ms",
    "queue_ms",
    "forward_ms",
    "prefill_forward_ms",
    "decode_forward_ms",
    "decode_ms",
    "inference_service_ms",
    "num_running_reqs",
    "num_waiting_reqs",
    "num_retractions",
    "prompt_tokens",
    "completion_tokens",
    "pd_prefill_bootstrap_queue_ms",
    "pd_prefill_bootstrap_ms",
    "pd_prefill_alloc_wait_ms",
    "pd_prefill_forward_ms",
    "pd_prefill_transfer_queue_ms",
    "pd_transfer_speed_gb_s",
    "pd_transfer_total_mb",
    "pd_prefill_retry_count",
    "pd_decode_prealloc_ms",
    "pd_decode_bootstrap_ms",
    "pd_decode_alloc_wait_ms",
    "pd_decode_transfer_ms",
    "pd_decode_forward_ms",
)


class PolarRolloutSchedulerError(RuntimeError):
    """Raised when the async Polar scheduler cannot safely make progress."""


class PolarUntrainableGroupError(PolarRolloutSchedulerError):
    """Raised when a completed group contains no loss-bearing tokens."""

    def __init__(self, message: str, *, infrastructure_only: bool) -> None:
        super().__init__(message)
        self.infrastructure_only = infrastructure_only


class PolarLowCompleteAcceptFractionError(PolarRolloutSchedulerError):
    """Raised when a completed task has too few trainable completed sessions."""


class CandidatePoolHealthGateError(PolarRolloutSchedulerError):
    """Raised before training when one required pool candidate is unavailable."""


class PolarEvalDataIntegrityError(ValueError):
    """Raised when an eval JSONL no longer matches its launcher manifest."""


@dataclass(slots=True)
class _DeferredGroup:
    group: list[Any]
    reservation_id: int | None = None
    submitted_rollout_id: int | None = None
    policy_version: int | None = None
    partial_store: PartialRolloutStore | None = None


@dataclass(slots=True)
class _PendingGroup:
    group_id: int
    group: list[Any]
    reservation_id: int | None
    submitted_rollout_id: int
    policy_version: int
    session_cost: int
    partial_store: PartialRolloutStore | None = None
    submitted_at: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class _CompletedGroup:
    group_id: int
    group: list[Any]
    reservation_id: int | None
    samples: list[Any]
    task_id: str
    submitted_rollout_id: int
    policy_version: int
    session_count: int
    submitted_at: float = 0.0
    completed_at: float = field(default_factory=time.monotonic)
    service_time_seconds: float = 0.0
    sample_conversion_seconds: float = 0.0
    output_queue_wait_seconds: float = 0.0
    partial_store: PartialRolloutStore | None = None


@dataclass(slots=True)
class _PartialRecoveryPlan:
    store: PartialRolloutStore | None = None
    kept: list[_CompletedGroup] = field(default_factory=list)
    result_ready: list[_CompletedGroup] = field(default_factory=list)
    deferred: list[_DeferredGroup] = field(default_factory=list)
    candidate_only: list[_CompletedGroup] = field(default_factory=list)
    dropped_count: int = 0
    resume_duplicate_count: int = 0
    dynamic_filter_metrics: dict[str, float] = field(default_factory=dict)
    reservation_metrics: dict[str, float] = field(default_factory=dict)


_WASTED_TIMING_FIELDS: tuple[str, ...] = (
    "rollout_pipeline_e2e_ms",
    "e2e_ms",
    "runtime_validation_ms",
    "ready_queue_ms",
    "agent_exec_ms",
    "eval_ms",
    "runtime_exec_ms",
    "mini_swe_command_ms",
)
_WASTED_INFERENCE_FIELDS: tuple[str, ...] = (
    "e2e_ms",
    "queue_ms",
    "prefill_forward_ms",
    "decode_ms",
    "inference_service_ms",
)

# ---------------------------------------------------------------------------
# Global worker singleton
# ---------------------------------------------------------------------------
_global_async_worker: "AsyncPolarRolloutWorker | None" = None
_worker_lock = threading.Lock()


def get_global_async_worker(
    args: Any,
    data_source: Any,
    partial_recovery: _PartialRecoveryPlan | None = None,
    rollout_id: int | None = None,
) -> "AsyncPolarRolloutWorker":
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is None or not _global_async_worker.is_alive():
            logger.info("Creating new async Polar rollout worker")
            recovery = partial_recovery or _PartialRecoveryPlan()
            _global_async_worker = AsyncPolarRolloutWorker(
                args,
                data_source,
                partial_store=recovery.store,
            )
            if rollout_id is not None:
                _global_async_worker.set_rollout_context(rollout_id)
            _global_async_worker.bootstrap_partial_recovery(
                deferred=recovery.deferred,
                completed=recovery.result_ready,
                held_keep_count=len(recovery.kept),
            )
            _global_async_worker.start()
        return _global_async_worker


def stop_global_worker() -> None:
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is not None:
            _global_async_worker.stop()
            _global_async_worker = None


def _current_ray_task_is_canceled() -> bool:
    """Cooperatively observe ``ray.cancel`` from a regular actor method."""
    try:
        import ray

        return bool(ray.get_runtime_context().is_canceled())
    except (ImportError, RuntimeError):
        # Unit tests and standalone callers do not run inside a Ray worker.
        return False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _build_task_payload(
    *,
    args: Any,
    config: PolarSlimeConfig,
    group: list[Any],
    rollout_id: int,
    task_position: int,
    eval_dataset_cfg: Any | None = None,
    eval_dataset_name: str | None = None,
) -> dict[str, Any]:
    first_sample = group[0]
    prompt_text = prompt_to_instruction_text(getattr(first_sample, "prompt", ""))
    instruction = render_instruction(
        args=args,
        config=config,
        sample=first_sample,
        prompt_text=prompt_text,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=len(group),
    )
    payload = render_task_payload(
        args=args,
        config=config,
        sample=first_sample,
        instruction=instruction,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=len(group),
        is_eval=eval_dataset_cfg is not None or eval_dataset_name is not None,
    )
    if eval_dataset_cfg is not None or eval_dataset_name is not None:
        _apply_eval_runtime_overrides(
            payload,
            args=args,
            dataset_cfg=eval_dataset_cfg,
            dataset_name=eval_dataset_name or "polar_eval",
            task_position=task_position,
        )
    return payload


def _apply_eval_runtime_overrides(
    payload: dict[str, Any],
    *,
    args: Any,
    dataset_cfg: Any | None,
    dataset_name: str,
    task_position: int,
) -> None:
    """Attach eval-only sampling controls to the agent and stable seed metadata."""

    # Fixed eval must wait for every configured seed. Reusing training's
    # straggler early-stop would compare whichever sessions happened to finish
    # first at baseline versus final time.
    payload["early_stop_min_usable_sessions"] = int(payload["num_samples"])
    # A fully-async training call may enqueue hundreds of sessions at the same
    # instant as the fixed pre-train baseline. Let eval overtake only sessions
    # still waiting for gateway INIT; already-initializing/running sessions are
    # untouched. This shortens the baseline tail without changing its model,
    # sampling, execution timeout, evaluator, or initial-policy weight barrier.
    payload["dispatch_priority"] = max(
        int(payload.get("dispatch_priority", 0)),
        _EVAL_DISPATCH_PRIORITY,
    )

    agent = payload.get("agent")
    if not isinstance(agent, dict):
        raise ValueError("polar eval task payload agent must be a mapping")
    settings = agent.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError("polar eval task payload agent.settings must be a mapping")

    existing_model_kwargs = settings.get("model_kwargs") or {}
    if not isinstance(existing_model_kwargs, dict):
        raise ValueError("polar eval agent.settings.model_kwargs must be a mapping")
    model_kwargs = copy.deepcopy(existing_model_kwargs)
    for dataset_attr, request_key, arg_attrs in _EVAL_STANDARD_MODEL_KWARGS:
        value = _eval_runtime_value(dataset_cfg, dataset_attr, args, arg_attrs)
        if value is not None:
            model_kwargs[request_key] = copy.deepcopy(value)

    extra_body = model_kwargs.get("extra_body") or {}
    if not isinstance(extra_body, dict):
        raise ValueError("polar eval agent model_kwargs.extra_body must be a mapping")
    extra_body = copy.deepcopy(extra_body)
    for dataset_attr, request_key, arg_attrs in _EVAL_SGLANG_EXTRA_BODY:
        value = _eval_runtime_value(dataset_cfg, dataset_attr, args, arg_attrs)
        if value is not None:
            extra_body[request_key] = copy.deepcopy(value)
    if extra_body:
        model_kwargs["extra_body"] = extra_body

    settings = {**settings, "model_kwargs": model_kwargs}
    agent["settings"] = settings

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("polar eval task metadata must be a mapping")
    # Stable across baseline/final evaluations and independent of rollout id.
    # RolloutManager adds the within-prompt sample index before dispatch.
    seed_material = f"{dataset_name}\0{task_position}".encode()
    seed_base = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
    payload["metadata"] = {
        **metadata,
        _EVAL_SAMPLING_SEED_METADATA_KEY: seed_base,
    }


def _eval_runtime_value(
    dataset_cfg: Any | None,
    dataset_attr: str,
    args: Any,
    arg_attrs: tuple[str, ...],
) -> Any:
    if dataset_cfg is not None:
        value = getattr(dataset_cfg, dataset_attr, None)
        if value is not None:
            return value
    for attr in arg_attrs:
        value = getattr(args, attr, None)
        if value is not None:
            return value
    return None


def _attach_scheduler_metadata(
    payload: dict[str, Any],
    *,
    group_id: int,
    policy_version: int,
    rollout_step: int,
) -> None:
    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("polar task metadata must be a mapping when provided")
    payload["metadata"] = {
        **metadata,
        "group_id": group_id,
        "policy_version": policy_version,
        "rollout_step": rollout_step,
    }


async def _get_task_status_with_retry(
    client: httpx.AsyncClient,
    base_url: str,
    task_id: str,
) -> TaskStatus:
    """Read an existing task status through transient transport failures."""
    url = f"{base_url}/rollout/task/{task_id}"
    for attempt in range(1, _TASK_STATUS_GET_MAX_ATTEMPTS + 1):
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if (
                exc.response.status_code not in _TASK_STATUS_GET_RETRYABLE_STATUS_CODES
                or attempt >= _TASK_STATUS_GET_MAX_ATTEMPTS
            ):
                raise
            error_text = str(exc)
        except httpx.TransportError as exc:
            if attempt >= _TASK_STATUS_GET_MAX_ATTEMPTS:
                raise
            error_text = str(exc)
        else:
            return TaskStatus.model_validate(response.json())

        backoff_seconds = _TASK_STATUS_GET_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
        logger.warning(
            "Polar task %s status GET failed (attempt %s/%s); retrying in %.1fs: %s",
            task_id,
            attempt,
            _TASK_STATUS_GET_MAX_ATTEMPTS,
            backoff_seconds,
            error_text,
        )
        await asyncio.sleep(backoff_seconds)

    raise AssertionError("unreachable task-status retry state")


async def _submit_and_wait_for_task(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
    *,
    poll_interval: float = _POLL_INTERVAL,
) -> TaskResult:
    """Submit one task via the async endpoint and poll until terminal."""
    resp = await client.post(
        f"{base_url}/rollout/task/submit",
        json=payload,
        headers=_control_plane_headers(),
    )
    resp.raise_for_status()
    task_id = resp.json()["task_id"]

    while True:
        await asyncio.sleep(poll_interval)
        try:
            status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
            status_resp.raise_for_status()
        except (
            httpx.HTTPStatusError,
            httpx.TimeoutException,
            httpx.TransportError,
        ) as exc:
            logger.warning("Polling Polar task %s failed; continuing: %s", task_id, exc)
            continue
        status = TaskStatus.model_validate(status_resp.json())
        if status.status in ("completed", "failed"):
            break

    return TaskResult(
        task_id=task_id,
        status=status.status,
        results=status.results,
        result_paths=status.result_paths,
    )


def _resolve_max_tokens(args: Any) -> int | None:
    """Resolve the safe per-sample trajectory cap.

    By default the dynamic microbatch budget is also treated as a per-sample
    bound.  Slime's scheduler can instead admit an oversize individual sample
    in a microbatch by itself; an explicit opt-in then leaves ``seq_length`` as
    the hard trajectory bound while ``max_tokens_per_gpu`` controls only the
    aggregate tokens packed into a microbatch.
    """
    mtpg = getattr(args, "max_tokens_per_gpu", None)
    seq_length = getattr(args, "seq_length", None)
    raw_allow_oversize = getattr(
        args,
        "polar_allow_single_sample_over_token_cap",
        False,
    )
    if isinstance(raw_allow_oversize, bool):
        allow_oversize = raw_allow_oversize
    elif isinstance(raw_allow_oversize, int) and raw_allow_oversize in (0, 1):
        allow_oversize = bool(raw_allow_oversize)
    elif isinstance(raw_allow_oversize, str) and raw_allow_oversize.strip().lower() in {
        "0",
        "1",
        "false",
        "true",
        "no",
        "yes",
        "off",
        "on",
    }:
        allow_oversize = raw_allow_oversize.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    else:
        raise ValueError("polar_allow_single_sample_over_token_cap must be a boolean")

    caps: list[int] = []
    if mtpg and not allow_oversize:
        cp_size = int(getattr(args, "context_parallel_size", 1) or 1)
        caps.append(int(mtpg) * cp_size)
    if seq_length:
        caps.append(int(seq_length))
    positive_caps = [cap for cap in caps if cap > 0]
    hard_cap = min(positive_caps) if positive_caps else None

    configured = getattr(args, "polar_max_trajectory_tokens", None)
    if configured is None:
        return hard_cap
    configured = int(configured)
    if configured <= 0:
        raise ValueError("polar_max_trajectory_tokens must be positive")
    if hard_cap is not None and configured > hard_cap:
        raise ValueError(
            "polar_max_trajectory_tokens exceeds trainer capacity: "
            f"configured={configured}, capacity={hard_cap} "
            f"(max_tokens_per_gpu={mtpg}, context_parallel_size="
            f"{getattr(args, 'context_parallel_size', 1)}, seq_length={seq_length}, "
            f"allow_single_sample_over_token_cap={allow_oversize})"
        )
    return configured


def _convert_task_result_to_samples(
    config: PolarSlimeConfig,
    task_result: TaskResult,
    group: list[Any],
    *,
    max_tokens: int | None = None,
) -> list[Any]:
    """Convert one task's session results into flat Slime samples.

    Each session → one trajectory → N traces → N samples, all tagged
    with the same ``Sample.index`` so the reward post-processor groups
    them as one trajectory.  The index is taken from the originating
    group sample at matching position, falling back to the position
    within the task result.
    """
    group_index = _group_index_for(group)
    group_samples: list[Any] = []
    for pos, session_result in enumerate(task_result.results):
        source = group[pos] if pos < len(group) else None
        traj_idx = int(getattr(source, "index", pos) if source is not None else pos)
        try:
            group_samples.extend(
                session_result_to_samples(
                    session_result,
                    group_index,
                    trajectory_index=traj_idx,
                    reward_key=config.reward_key,
                    max_tokens=max_tokens,
                )
            )
        except Exception as exc:
            # One corrupt/misaligned trajectory must not tear down a healthy
            # asynchronous trainer. Keep group cardinality with a fully masked
            # placeholder; the remaining sessions still contribute normally.
            detail = f"{type(exc).__name__}: {' '.join(str(exc).splitlines())}"[:1000]
            logger.exception(
                "Training session %s could not be converted; replacing only "
                "this session with a zero-gradient placeholder",
                getattr(session_result, "session_id", f"{task_result.task_id}:{pos}"),
            )
            group_samples.append(
                session_result_to_placeholder(
                    session_result,
                    group_index,
                    trajectory_index=traj_idx,
                    reward_key=config.reward_key,
                    conversion_error=detail,
                )
            )
    return group_samples


def _convert_eval_task_result_to_samples(
    config: PolarSlimeConfig,
    task_result: TaskResult,
    group: list[Any],
    *,
    dataset_name: str,
    max_tokens: int | None = None,
) -> list[Any]:
    """Best-effort eval conversion with one-session failure isolation.

    Training conversion remains fail-closed because malformed policy samples
    must never reach the optimizer.  Evaluation has no gradient path, so a
    malformed result is logged and later represented by one zero reward.
    """

    group_index = _group_index_for(group)
    group_samples: list[Any] = []
    for pos, session_result in enumerate(task_result.results[: len(group)]):
        source = group[pos] if pos < len(group) else None
        traj_idx = int(getattr(source, "index", pos) if source is not None else pos)
        try:
            group_samples.extend(
                session_result_to_samples(
                    session_result,
                    group_index,
                    trajectory_index=traj_idx,
                    reward_key=config.reward_key,
                    max_tokens=max_tokens,
                )
            )
        except Exception as exc:
            detail = " ".join(str(exc).splitlines())
            logger.warning(
                "Eval dataset %s session %s could not be converted; assigning "
                "reward 0: %s: %.500s",
                dataset_name,
                getattr(session_result, "session_id", f"{task_result.task_id}:{pos}"),
                type(exc).__name__,
                detail,
            )
    return group_samples


def _trainable_token_count(sample: Any) -> int:
    if bool(getattr(sample, "remove_sample", False)):
        return 0
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return int(getattr(sample, "response_length", 0) or 0)
    return sum(1 for value in loss_mask if int(value) != 0)


def _has_trainable_tokens(samples: list[Any]) -> bool:
    return any(_trainable_token_count(sample) > 0 for sample in samples)


def _low_complete_accept_fraction_rejection_reason(
    config: PolarSlimeConfig,
    task_result: TaskResult,
    samples: list[Any],
) -> str | None:
    threshold = config.min_complete_accept_fraction
    if threshold <= 0.0:
        return None

    total_sessions = len(task_result.results)
    if total_sessions <= 0:
        return "empty task results"

    completed_trainable = _completed_trainable_session_count(task_result, samples)
    required = math.ceil(total_sessions * threshold)
    if completed_trainable >= required:
        return None

    fraction = completed_trainable / total_sessions
    return (
        f"completed trainable sessions {completed_trainable}/{total_sessions} "
        f"({fraction:.3f}) below polar_min_complete_accept_fraction={threshold:g} "
        f"(requires >= {required})"
    )


def _completed_trainable_session_count(
    task_result: TaskResult,
    samples: list[Any],
) -> int:
    trainable_session_ids: set[str] = set()
    for sample in samples:
        if _trainable_token_count(sample) <= 0:
            continue
        session_id = _sample_session_id(sample)
        if session_id:
            trainable_session_ids.add(session_id)

    count = 0
    for result in task_result.results:
        if (
            _status_value(result.status) == "COMPLETED"
            and result.session_id in trainable_session_ids
        ):
            count += 1
    return count


def _completed_service_metrics(completed_groups: list[_CompletedGroup]) -> dict[str, float]:
    """Summarize task latency and the production window for one accepted batch."""
    timed_groups = [
        completed
        for completed in completed_groups
        if (
            completed.service_time_seconds > 0
            and math.isfinite(completed.service_time_seconds)
            and completed.completed_at > completed.submitted_at
        )
    ]
    if not timed_groups:
        return {}

    service_window = max(group.completed_at for group in timed_groups) - min(
        group.submitted_at for group in timed_groups
    )
    metrics = {
        "timing/service_time_max": max(group.service_time_seconds for group in timed_groups),
        "timing/pipeline_ms/sample_conversion_mean": 1000.0
        * sum(group.sample_conversion_seconds for group in timed_groups)
        / len(timed_groups),
        "timing/pipeline_ms/sample_conversion_max": 1000.0
        * max(group.sample_conversion_seconds for group in timed_groups),
        "timing/pipeline_ms/output_queue_wait_mean": 1000.0
        * sum(group.output_queue_wait_seconds for group in timed_groups)
        / len(timed_groups),
        "timing/pipeline_ms/output_queue_wait_max": 1000.0
        * max(group.output_queue_wait_seconds for group in timed_groups),
    }
    if service_window > 0 and math.isfinite(service_window):
        metrics["timing/service_window"] = service_window
    return metrics


def _sample_session_id(sample: Any) -> str | None:
    polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar", {})
    session_id = polar_meta.get("session_id") or getattr(sample, "session_id", None)
    return str(session_id) if session_id else None


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status))


def _is_zero_trainable_error(exc: BaseException) -> bool:
    return "zero trainable tokens" in str(exc)


def _task_result_is_infrastructure_only(task_result: TaskResult) -> bool:
    if not task_result.results:
        return False
    for result in task_result.results:
        if _status_value(result.status).upper() != "ERROR":
            return False
        trajectory = getattr(result, "trajectory", None)
        if getattr(trajectory, "traces", None):
            return False
    return True


def _annotate_accepted_samples(
    samples: list[Any],
    *,
    accepted_rollout_id: int,
    staleness: int,
    policy_version: int,
    scheduler_group_id: int,
) -> None:
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata
        polar_meta = metadata.setdefault("polar", {})
        if not isinstance(polar_meta, dict):
            polar_meta = {}
            metadata["polar"] = polar_meta
        polar_meta.update(
            {
                "accepted_rollout_id": int(accepted_rollout_id),
                "policy_staleness": int(staleness),
                "policy_version": int(policy_version),
                "scheduler_group_id": int(scheduler_group_id),
            }
        )
        train_metadata = getattr(sample, "train_metadata", None)
        if train_metadata is None:
            train_metadata = {}
            sample.train_metadata = train_metadata
        train_metadata.update(
            {
                "policy_staleness": int(staleness),
                "policy_version": int(policy_version),
            }
        )


# ---------------------------------------------------------------------------
# Persistent training worker
# ---------------------------------------------------------------------------
class AsyncPolarRolloutWorker:
    """Persistent background worker that continuously submits Polar tasks.

    Runs in its own thread with a dedicated asyncio event loop.  Pulls
    sample groups from ``data_source``, submits them to the async
    ``/rollout/task/submit`` endpoint, polls until completion, converts
    results, and pushes them into ``output_queue``.  Training loops call
    ``drain_completed()`` to collect finished groups.
    """

    def __init__(
        self,
        args: Any,
        data_source: Any,
        *,
        partial_store: PartialRolloutStore | None = None,
    ) -> None:
        self.args = args
        self.data_source = data_source
        self.config = resolve_polar_slime_config(args)
        batch_size = int(getattr(args, "rollout_batch_size", 1) or 1)
        # Output queue is a handoff channel; the durable overflow buffer is
        # `_completed_buffer`, which is drained in bounded chunks by training.
        queue_maxsize = max(32, batch_size * self.config.max_async_level * 2)
        self.output_queue: queue.Queue[_CompletedGroup] = queue.Queue(maxsize=queue_maxsize)
        # Rejected groups can still contain the only trustworthy evidence that
        # a frozen candidate is unavailable.  Keep that evidence on a separate
        # unbounded handoff: it is consumed by the trainer immediately and is
        # never eligible for optimizer input.
        self.health_output_queue: queue.Queue[_CompletedGroup] = queue.Queue()
        self.deferred_queue: queue.Queue[_DeferredGroup] = queue.Queue()
        self._completed_buffer: deque[_CompletedGroup] = deque()
        self._running = True
        self._thread: threading.Thread | None = None
        self._group_counter = 0
        self._batch_size = batch_size
        self._current_rollout_id = int(getattr(args, "start_rollout_id", 0) or 0)
        self._requested_groups = 0
        # Fully-async requests grant a finite number of fresh admissions.  The
        # first request may fill the entire async window; each later request
        # grants only enough credit to replace the batch it consumes.  Keeping
        # this separate from outstanding demand closes the race where a request
        # is satisfied immediately from the completed backlog before the worker
        # thread gets a chance to refill any rollout work.
        self._fully_async_request_count = 0
        self._fully_async_admission_credit = 0
        self._fatal_error: BaseException | None = None
        self._consecutive_infrastructure_failures = 0
        self._state_lock = threading.RLock()
        self._metrics: dict[str, float] = {}
        self._last_reported_counters: dict[str, float] = {}
        self._active_groups = 0
        self._active_sessions = 0
        self._completed_buffer_size = 0
        # Per-task callback plumbing: event fires when the rollout server POSTs
        # the terminal TaskResult to our local listener.
        self._task_events: dict[str, asyncio.Event] = {}
        self._task_results: dict[str, TaskResult] = {}
        self._callback_url: str | None = None
        self._partial_store = partial_store
        self._resume_committed_reservation_ids = (
            partial_store.committed_reservation_ids if partial_store is not None else frozenset()
        )
        # Recovered KEEP groups live in generate_rollout_polar_async rather
        # than a worker queue, but they still own reservations and must count
        # against the bounded async window until the complete batch commits.
        self._recovered_held_groups = 0

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="polar-async-rollout"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- results ---------------------------------------------------------------

    def set_rollout_context(self, rollout_id: int) -> None:
        with self._state_lock:
            self._current_rollout_id = int(rollout_id)

    def bootstrap_partial_recovery(
        self,
        *,
        deferred: list[_DeferredGroup],
        completed: list[_CompletedGroup],
        held_keep_count: int,
    ) -> None:
        """Seed reconstructed work before the background thread starts."""

        if self._thread is not None:
            raise RuntimeError("partial recovery must be bootstrapped before worker start")
        with self._state_lock:
            self._recovered_held_groups = int(held_keep_count)
            for item in deferred:
                self.deferred_queue.put_nowait(item)
            for item in completed:
                self.output_queue.put_nowait(item)

    def release_recovered_holds(self) -> None:
        with self._state_lock:
            self._recovered_held_groups = 0

    def request_groups(self, count: int) -> None:
        count = int(count)
        if count < 0:
            return
        with self._state_lock:
            self._requested_groups += count
            if self.config.fully_async:
                if self._fully_async_request_count == 0:
                    max_window = self._batch_size * self.config.max_async_level
                    reconstructed_owned = (
                        self._recovered_held_groups
                        + self.deferred_queue.qsize()
                        + self.output_queue.qsize()
                        + self._completed_buffer_size
                    )
                    self._fully_async_admission_credit += max(
                        0,
                        max_window - reconstructed_owned,
                    )
                    self._fully_async_request_count += 1
                elif count > 0:
                    self._fully_async_admission_credit += count
                    self._fully_async_request_count += 1

    def raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise PolarRolloutSchedulerError(str(self._fatal_error)) from self._fatal_error

    def drain_completed(
        self,
        *,
        max_groups: int,
        rollout_id: int,
    ) -> list[_CompletedGroup]:
        self.raise_if_failed()

        with self._state_lock:
            # Move the handoff queue and publish its mirrored size atomically
            # with respect to admission checks.  Otherwise qsize() can fall
            # before `_completed_buffer_size` rises, briefly hiding owned work
            # and allowing the single window to be exceeded.
            while True:
                try:
                    self._completed_buffer.append(self.output_queue.get_nowait())
                except queue.Empty:
                    break
            self._completed_buffer_size = len(self._completed_buffer)

        accepted: list[_CompletedGroup] = []
        while self._completed_buffer and len(accepted) < max_groups:
            completed = self._completed_buffer.popleft()
            staleness = max(0, int(rollout_id) - completed.policy_version)
            if staleness > self.config.max_off_policy_steps:
                self._inc_metric("polar/stale_groups")
                reason = (
                    f"staleness {staleness} exceeded max_off_policy_steps="
                    f"{self.config.max_off_policy_steps}"
                )
                self._inc_metric("polar/dropped_groups")
                self._inc_metric("polar/dropped_stale_groups")
                self._inc_metric("polar/dropped_sessions", completed.session_count)
                logger.warning(
                    "Dropping stale Polar group %s task=%s: %s",
                    completed.group_id,
                    completed.task_id,
                    reason,
                )
                self._emit_health_observation(completed)
                self._record_wasted_samples(completed.samples)
                if completed.partial_store is not None:
                    completed.partial_store.record_drop(
                        completed.reservation_id,
                        outcome="stale",
                        reason=reason,
                    )
                self._consume_reservation(completed.reservation_id, outcome="stale")
                self._restore_fully_async_admission_credit(1)
                continue

            _annotate_accepted_samples(
                completed.samples,
                accepted_rollout_id=rollout_id,
                staleness=staleness,
                policy_version=completed.policy_version,
                scheduler_group_id=completed.group_id,
            )
            accepted.append(completed)

        if accepted:
            self._mark_delivered(len(accepted))
        with self._state_lock:
            self._completed_buffer_size = len(self._completed_buffer)
        return accepted

    def drain_health_observations(self) -> list[_CompletedGroup]:
        """Drain rejected pre-filter groups for fail-closed pool monitoring."""

        observations: list[_CompletedGroup] = []
        while True:
            try:
                observations.append(self.health_output_queue.get_nowait())
            except queue.Empty:
                return observations

    def _emit_health_observation(self, completed: _CompletedGroup) -> None:
        if self.config.candidate_pool_health_gate_enabled:
            self.health_output_queue.put_nowait(completed)

    def queue_size(self) -> int:
        with self._state_lock:
            return (
                self.output_queue.qsize()
                + self._completed_buffer_size
                + self.deferred_queue.qsize()
            )

    def mark_dynamic_filter_drop(
        self,
        completed: _CompletedGroup,
        *,
        reason: str | None,
    ) -> dict[str, float]:
        """Permanently consume a completed group rejected by active sampling."""

        self._inc_metric("polar/dropped_groups")
        self._inc_metric("polar/dropped_dynamic_filter_groups")
        self._inc_metric("polar/dropped_sessions", completed.session_count)
        self._record_wasted_samples(completed.samples)
        logger.info(
            "Dynamic sampling filtered Polar group %s task=%s reason=%s",
            completed.group_id,
            completed.task_id,
            reason or "unspecified",
        )
        if completed.partial_store is not None:
            completed.partial_store.record_drop(
                completed.reservation_id,
                outcome="dynamic_filter",
                reason=reason,
            )
        return self._consume_reservation(
            completed.reservation_id,
            outcome="dynamic_filter",
        )

    def snapshot_metrics(self) -> dict[str, float]:
        with self._state_lock:
            raw_counters = dict(self._metrics)
            out: dict[str, float] = {}
            out["polar/scheduler/active_groups"] = float(self._active_groups)
            out["polar/scheduler/active_sessions"] = float(self._active_sessions)
            out["polar/scheduler/completed_buffer"] = float(self._completed_buffer_size)
            out["polar/scheduler/output_queue"] = float(self.output_queue.qsize())
            out["polar/scheduler/deferred_queue"] = float(self.deferred_queue.qsize())
            out["polar/scheduler/recovered_held_groups"] = float(self._recovered_held_groups)
            out["polar/scheduler/requested_groups"] = float(self._requested_groups)
            if self.config.fully_async:
                out["polar/scheduler/admission_credit"] = float(self._fully_async_admission_credit)
        reservation_metrics = getattr(self.data_source, "reservation_metrics", None)
        if callable(reservation_metrics):
            for key, value in reservation_metrics().items():
                value = float(value)
                if key.endswith("_since_worker_start"):
                    raw_counters[key] = value
                else:
                    out[key] = value

        with self._state_lock:
            for key, value in raw_counters.items():
                if key.endswith("_since_worker_start"):
                    lifetime_key = key
                    delta_key = f"{key.removesuffix('_since_worker_start')}_delta"
                else:
                    lifetime_key = f"{key}_since_worker_start"
                    delta_key = f"{key}_delta"
                previous = self._last_reported_counters.get(key, 0.0)
                out[lifetime_key] = value
                out[delta_key] = max(0.0, value - previous)
                self._last_reported_counters[key] = value
        wasted_sessions = out.get("polar/wasted/session_count_delta", 0.0)
        if wasted_sessions > 0.0:
            for timing_field in _WASTED_TIMING_FIELDS:
                total = out.get(f"timing/wasted/{timing_field}_sum_delta")
                if total is not None:
                    out[f"timing/wasted/{timing_field}_per_session_mean"] = total / wasted_sessions
        wasted_inference = out.get("polar/wasted/inference_timing_count_delta", 0.0)
        if wasted_inference > 0.0:
            for timing_field in _WASTED_INFERENCE_FIELDS:
                total = out.get(f"timing/wasted/inference_{timing_field}_sum_delta")
                if total is not None:
                    out[f"timing/wasted/inference_{timing_field}_mean"] = total / wasted_inference
        return out

    # -- internal --------------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.run(self._async_loop())

    async def _async_loop(self) -> None:
        logger.info("Async Polar rollout worker started")
        active: dict[asyncio.Task[None], _PendingGroup] = {}
        active_session_cost = 0
        wakeup = asyncio.Event()

        callback_server, callback_task = await self._start_callback_listener()
        timeout = (
            None
            if self.config.request_timeout is None
            else httpx.Timeout(self.config.request_timeout)
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                while self._running:
                    done = [t for t in active if t.done()]
                    for t in done:
                        pending = active.pop(t)
                        active_session_cost -= pending.session_cost
                        try:
                            t.result()
                        except Exception as exc:
                            logger.exception("Polar async task failed")
                            self._set_fatal(exc)
                            self._running = False
                    self._record_active_counts(active, active_session_cost)

                    while self._running and self._can_admit_group(active, active_session_cost):
                        try:
                            next_group = self._next_group_for_submission()
                        except Exception as exc:
                            self._set_fatal(exc)
                            self._running = False
                            break
                        if next_group is None:
                            break
                        session_cost = len(next_group.group)
                        if session_cost > self.config.max_session_concurrency:
                            self._set_fatal(
                                PolarRolloutSchedulerError(
                                    f"Prompt group needs {session_cost} sessions but "
                                    f"derived max_session_concurrency is "
                                    f"{self.config.max_session_concurrency}"
                                )
                            )
                            self._running = False
                            break
                        if (
                            active_session_cost + session_cost
                            > self.config.max_session_concurrency
                        ):
                            self.deferred_queue.put(next_group)
                            break

                        gid = self._group_counter
                        self._group_counter += 1
                        current_rollout_id, current_policy_version = self._rollout_context()
                        submitted_rollout_id = (
                            current_rollout_id
                            if next_group.submitted_rollout_id is None
                            else int(next_group.submitted_rollout_id)
                        )
                        policy_version = (
                            current_policy_version
                            if next_group.policy_version is None
                            else int(next_group.policy_version)
                        )
                        pending = _PendingGroup(
                            group_id=gid,
                            group=next_group.group,
                            reservation_id=next_group.reservation_id,
                            submitted_rollout_id=submitted_rollout_id,
                            policy_version=policy_version,
                            session_cost=session_cost,
                            partial_store=next_group.partial_store,
                        )
                        task = asyncio.create_task(
                            self._submit_and_collect(client, pending),
                            name=f"polar-rollout-task-{gid}",
                        )
                        task.add_done_callback(lambda _: wakeup.set())
                        active[task] = pending
                        active_session_cost += session_cost
                        self._record_active_counts(active, active_session_cost)

                    if self._running:
                        try:
                            await asyncio.wait_for(wakeup.wait(), timeout=0.5)
                        except asyncio.TimeoutError:
                            pass
                        wakeup.clear()

                if active:
                    # These groups were only speculative prefetch. Once training is
                    # disposing this worker they cannot enter a committed model
                    # step, so waiting for full agent timeouts just delays the
                    # graceful checkpoint exit and can consume the wall-time
                    # reserve. Cancelling each waiter also sends a task-level
                    # DELETE that tears down its gateway sessions.
                    logger.info("Cancelling %d uncommitted in-flight Polar tasks", len(active))
                    for task in active:
                        task.cancel()
                    await asyncio.gather(*active.keys(), return_exceptions=True)
        finally:
            callback_server.should_exit = True
            try:
                await asyncio.wait_for(callback_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Callback listener did not shut down within 5s")
        logger.info("Async Polar rollout worker stopped")

    async def _start_callback_listener(self) -> tuple[uvicorn.Server, asyncio.Task[None]]:
        """Bind a FastAPI listener for TaskResult callbacks."""
        app = FastAPI()

        @app.post("/callback/task_result")
        async def on_task_result(request: Request) -> dict[str, Any]:
            payload = await request.json()
            task_id = payload.get("task_id") if isinstance(payload, dict) else None
            if not task_id:
                return {"ok": False, "reason": "missing task_id"}
            try:
                result = TaskResult.model_validate(payload)
            except Exception:
                logger.exception("Invalid callback payload for task %s", task_id)
                return {"ok": False, "reason": "invalid payload"}
            self._task_results[task_id] = result
            event = self._task_events.get(task_id)
            if event is not None:
                event.set()
            return {"ok": True}

        config = uvicorn.Config(
            app=app,
            host=self.config.callback_host,
            port=0,
            log_level="warning",
            lifespan="on",
            access_log=uvicorn_access_log_enabled(),
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve(), name="polar-callback-listener")
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        self._callback_url = f"http://{self.config.callback_host}:{port}/callback/task_result"
        logger.info("Polar trainer callback listener bound to %s", self._callback_url)
        return server, task

    async def _submit_and_collect(self, client: httpx.AsyncClient, pending: _PendingGroup) -> None:
        last_error: BaseException | None = None

        if self._running:
            try:
                completed = await self._submit_attempt(client, pending)
                await self._emit_completed(completed)
                return
            except Exception as exc:
                last_error = exc

        if last_error is None:
            return

        # An external dispose may race with an HTTP waiter failing. A stopped
        # worker must leave every uncommitted reservation at the checkpoint
        # frontier instead of turning teardown into a permanent skip.
        if not self._running:
            return

        fuse_error: PolarRolloutSchedulerError | None = None
        if _is_zero_trainable_error(last_error):
            category_metric = "polar/dropped_zero_trainable_groups"
            reason = "zero trainable tokens"
            permanently_consumed = True
            if (
                isinstance(last_error, PolarUntrainableGroupError)
                and last_error.infrastructure_only
            ):
                fuse_error = self._note_infrastructure_failure()
            else:
                self._reset_infrastructure_failures()
        elif isinstance(last_error, PolarLowCompleteAcceptFractionError):
            self._reset_infrastructure_failures()
            category_metric = "polar/dropped_low_complete_fraction_groups"
            reason = "low complete accept fraction"
            permanently_consumed = True
        elif isinstance(last_error, RolloutLogprobError):
            self._reset_infrastructure_failures()
            category_metric = "polar/dropped_logprob_error_groups"
            reason = "rollout logprob error"
            permanently_consumed = True
        else:
            self._reset_infrastructure_failures()
            category_metric = "polar/dropped_failed_groups"
            reason = "task failure"
            permanently_consumed = False

        self._inc_metric("polar/dropped_groups")
        self._inc_metric(category_metric)
        self._inc_metric("polar/dropped_sessions", pending.session_cost)
        logger.warning(
            "Dropping Polar group %s because of %s: %s",
            pending.group_id,
            reason,
            last_error,
        )
        if permanently_consumed:
            if pending.partial_store is not None:
                pending.partial_store.record_drop(
                    pending.reservation_id,
                    outcome="permanent_drop",
                    reason=reason,
                )
            self._consume_reservation(pending.reservation_id, outcome="permanent_drop")
            self._restore_fully_async_admission_credit(1)
            if fuse_error is not None:
                self._set_fatal(fuse_error)
                self._running = False
        else:
            if pending.reservation_id is not None:
                self._inc_metric("polar/replay_on_resume_groups")
            logger.warning(
                "Leaving reservation %s outstanding and stopping for checkpoint replay after %s",
                pending.reservation_id,
                reason,
            )
            self._set_fatal(last_error)
            self._running = False
        return

    async def _submit_attempt(
        self,
        client: httpx.AsyncClient,
        pending: _PendingGroup,
    ) -> _CompletedGroup:
        payload = _build_task_payload(
            args=self.args,
            config=self.config,
            group=pending.group,
            rollout_id=pending.group_id,
            task_position=0,
        )
        payload["task_id"] = str(payload["task_id"])
        _attach_scheduler_metadata(
            payload,
            group_id=pending.group_id,
            policy_version=pending.policy_version,
            rollout_step=pending.submitted_rollout_id,
        )
        pending.submitted_at = time.monotonic()
        task_result = await self._submit_with_callback(client, payload)
        completed_at = time.monotonic()
        service_time_seconds = max(0.0, completed_at - pending.submitted_at)

        rejection_reason = self._task_rejection_reason(task_result, pending.group)
        if rejection_reason is not None:
            self._record_wasted_results(task_result.results)
            raise PolarRolloutSchedulerError(
                f"Task {task_result.task_id} cannot be accepted: {rejection_reason}"
            )

        conversion_started = time.perf_counter()
        try:
            group_samples = _convert_task_result_to_samples(
                self.config,
                task_result,
                pending.group,
                max_tokens=_resolve_max_tokens(self.args),
            )
        except Exception:
            self._record_wasted_results(task_result.results)
            raise
        sample_conversion_seconds = time.perf_counter() - conversion_started
        if not group_samples:
            self._record_wasted_results(task_result.results)
            raise PolarRolloutSchedulerError(
                f"Task {task_result.task_id} converted to zero samples"
            )
        completed = _CompletedGroup(
            group_id=pending.group_id,
            group=pending.group,
            reservation_id=pending.reservation_id,
            samples=group_samples,
            task_id=task_result.task_id,
            submitted_rollout_id=pending.submitted_rollout_id,
            policy_version=pending.policy_version,
            session_count=len(task_result.results),
            submitted_at=pending.submitted_at,
            completed_at=completed_at,
            service_time_seconds=service_time_seconds,
            sample_conversion_seconds=sample_conversion_seconds,
            partial_store=pending.partial_store,
        )
        if not _has_trainable_tokens(group_samples):
            self._emit_health_observation(completed)
            self._record_wasted_samples(group_samples)
            raise PolarUntrainableGroupError(
                f"Task {task_result.task_id} produced zero trainable tokens",
                infrastructure_only=_task_result_is_infrastructure_only(task_result),
            )
        rejection_reason = _low_complete_accept_fraction_rejection_reason(
            self.config, task_result, group_samples
        )
        if rejection_reason is not None:
            self._emit_health_observation(completed)
            self._record_wasted_samples(group_samples)
            raise PolarLowCompleteAcceptFractionError(
                f"Task {task_result.task_id} cannot be accepted: {rejection_reason}"
            )
        if pending.partial_store is not None:
            pending.partial_store.record_result_ready(completed)
        return completed

    async def _emit_completed(self, completed: _CompletedGroup) -> None:
        wait_started = time.perf_counter()
        while self._running:
            try:
                completed.output_queue_wait_seconds = time.perf_counter() - wait_started
                self.output_queue.put_nowait(completed)
                self._reset_infrastructure_failures()
                self._inc_metric("polar/completed_groups")
                return
            except queue.Full:
                self._inc_metric("polar/output_queue_full_waits")
                await asyncio.sleep(0.1)

    def _next_group_for_submission(self) -> _DeferredGroup | None:
        try:
            deferred = self.deferred_queue.get_nowait()
            self._inc_metric("polar/deferred_queue_dequeues")
            return deferred
        except queue.Empty:
            pass

        while True:
            reservation_getter = getattr(
                self.data_source,
                "get_samples_with_reservation",
                None,
            )
            if callable(reservation_getter):
                reservations = reservation_getter(1)
                if not reservations:
                    return None
                if len(reservations) != 1:
                    raise PolarRolloutSchedulerError(
                        "Slime data source returned an invalid reservation batch"
                    )
                reservation_id, group = reservations[0]
            else:
                groups = self.data_source.get_samples(1)
                if not groups:
                    return None
                reservation_id = None
                group = groups[0]
            if not group:
                raise PolarRolloutSchedulerError(
                    "Slime data source returned an empty sample group"
                )
            submitted_rollout_id, policy_version = self._rollout_context()
            partial_store = self._partial_store
            if partial_store is not None and partial_store.sealed:
                partial_store = None

            if (
                reservation_id is not None
                and int(reservation_id) in self._resume_committed_reservation_ids
            ):
                self._skip_resume_duplicate(
                    reservation_id=int(reservation_id),
                    group=group,
                    submitted_rollout_id=submitted_rollout_id,
                    policy_version=policy_version,
                    partial_store=partial_store,
                )
                # A replay skip never owned remote work and therefore consumes
                # no fully-async admission credit. Continue synchronously until
                # the first genuinely admissible reservation is found.
                continue

            if partial_store is not None:
                if reservation_id is None:
                    raise PolarRolloutSchedulerError(
                        "partial rollout WAL requires reservation-aware data source"
                    )
                partial_store.record_prepared(
                    reservation_id=reservation_id,
                    group=group,
                    submitted_rollout_id=submitted_rollout_id,
                    policy_version=policy_version,
                )
            self._consume_fully_async_admission_credit()
            return _DeferredGroup(
                group=group,
                reservation_id=reservation_id,
                submitted_rollout_id=submitted_rollout_id,
                policy_version=policy_version,
                partial_store=partial_store,
            )

    def _skip_resume_duplicate(
        self,
        *,
        reservation_id: int,
        group: list[Any],
        submitted_rollout_id: int,
        policy_version: int,
        partial_store: PartialRolloutStore | None,
    ) -> None:
        """Consume a base-checkpoint duplicate before any remote submission."""

        if partial_store is not None:
            partial_store.record_prepared(
                reservation_id=reservation_id,
                group=group,
                submitted_rollout_id=submitted_rollout_id,
                policy_version=policy_version,
            )
            partial_store.record_resume_duplicate(reservation_id)
        self._consume_reservation(reservation_id, outcome="resume_duplicate")
        self._inc_metric("polar/resume_duplicate_groups")
        self._inc_metric("polar/resume_duplicate_sessions", len(group))
        logger.info(
            "Skipping reservation %s already committed by the base checkpoint",
            reservation_id,
        )

    def _can_admit_group(
        self,
        active: dict[asyncio.Task[None], _PendingGroup],
        active_session_cost: int,
    ) -> bool:
        with self._state_lock:
            if len(active) >= self.config.max_concurrency:
                return False
            if active_session_cost >= self.config.max_session_concurrency:
                return False

            deferred_groups = self.deferred_queue.qsize()
            # A deferred group was admitted earlier and already belongs to the
            # window.  Promoting it to active work neither consumes fresh credit
            # nor increases ownership, so let it finish even after delivery.
            if deferred_groups > 0:
                return True

            max_window = self._batch_size * self.config.max_async_level
            active_or_deferred = len(active) + deferred_groups
            completed_backlog = (
                self.output_queue.qsize()
                + self._completed_buffer_size
                + self._recovered_held_groups
            )
            owned_groups = active_or_deferred + completed_backlog
            if owned_groups >= max_window:
                return False

            if self.config.fully_async:
                # A single ownership window covers every group reserved on
                # behalf of the trainer, regardless of whether it is active or
                # complete and awaiting delivery.  Finite per-request credit
                # permits a backlog-satisfied request to refill its consumed
                # batch, but cannot restart unlimited production afterward.
                return self._fully_async_admission_credit > 0

            requested_groups = self._requested_groups
            if requested_groups <= 0:
                return False
            return owned_groups < min(requested_groups, max_window)

    def _task_rejection_reason(self, task_result: TaskResult, group: list[Any]) -> str | None:
        if task_result.status != "completed":
            return f"task status={task_result.status}"
        if not task_result.results:
            return "empty task results"
        if len(task_result.results) != len(group):
            return f"session count {len(task_result.results)} != expected {len(group)}"
        return None

    def _rollout_context(self) -> tuple[int, int]:
        with self._state_lock:
            return self._current_rollout_id, self._current_rollout_id

    def _mark_delivered(self, count: int) -> None:
        with self._state_lock:
            self._requested_groups = max(0, self._requested_groups - int(count))

    def _consume_fully_async_admission_credit(self) -> None:
        if not self.config.fully_async:
            return
        with self._state_lock:
            if self._fully_async_admission_credit <= 0:
                raise PolarRolloutSchedulerError(
                    "Fully-async scheduler attempted a fresh admission without request credit"
                )
            self._fully_async_admission_credit -= 1

    def _restore_fully_async_admission_credit(self, count: int) -> None:
        """Replace rejected owned work only while a trainer still needs data."""

        if not self.config.fully_async or count <= 0:
            return
        with self._state_lock:
            if self._requested_groups > 0:
                self._fully_async_admission_credit += int(count)

    def _consume_reservation(
        self,
        reservation_id: int | None,
        *,
        outcome: str,
    ) -> dict[str, float]:
        if reservation_id is None:
            return {}
        marker = getattr(self.data_source, "mark_consumed", None)
        if not callable(marker):
            raise PolarRolloutSchedulerError(
                "data source returned a reservation but does not expose mark_consumed"
            )
        metrics = marker(reservation_id, outcome=outcome)
        return metrics if isinstance(metrics, dict) else {}

    def _consume_reservations(
        self,
        reservation_ids: list[int | None],
        *,
        outcome: str,
    ) -> dict[str, float]:
        concrete_ids = [
            reservation_id for reservation_id in reservation_ids if reservation_id is not None
        ]
        if not concrete_ids:
            return {}
        marker_many = getattr(self.data_source, "mark_consumed_many", None)
        if callable(marker_many):
            metrics = marker_many(concrete_ids, outcome=outcome)
            return metrics if isinstance(metrics, dict) else {}

        metrics: dict[str, float] = {}
        for reservation_id in concrete_ids:
            metrics = self._consume_reservation(reservation_id, outcome=outcome)
        return metrics

    def _record_active_counts(
        self,
        active: dict[asyncio.Task[None], _PendingGroup],
        active_session_cost: int,
    ) -> None:
        with self._state_lock:
            self._active_groups = len(active)
            self._active_sessions = active_session_cost

    def _inc_metric(self, key: str, amount: float = 1.0) -> None:
        with self._state_lock:
            self._metrics[key] = self._metrics.get(key, 0.0) + amount

    def _record_wasted_results(self, results: list[Any]) -> None:
        """Retain timing totals for completed work rejected before training."""

        seen: set[str] = set()
        for result in results:
            session_id = str(getattr(result, "session_id", "") or "")
            if not session_id or session_id in seen:
                continue
            seen.add(session_id)
            timing = getattr(result, "timing", None)
            values = timing.model_dump(mode="python") if timing is not None else {}
            self._record_wasted_timing(values)
            trajectory = getattr(result, "trajectory", None)
            for trace in getattr(trajectory, "traces", []) or []:
                self._record_wasted_inference(getattr(trace, "metadata", None))

    def _record_wasted_samples(self, samples: list[Any]) -> None:
        seen: set[str] = set()
        seen_traces: set[tuple[str, int]] = set()
        for sample in samples:
            polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar", {})
            session_id = str(polar_meta.get("session_id") or "")
            if not session_id:
                continue
            if session_id not in seen:
                seen.add(session_id)
                self._record_wasted_timing(polar_meta.get("timing") or {})
            trace_key = (session_id, int(polar_meta.get("trace_index", 0) or 0))
            if trace_key not in seen_traces:
                seen_traces.add(trace_key)
                self._record_wasted_inference(polar_meta.get("trace_metadata"))

    def _record_wasted_timing(self, timing: Any) -> None:
        if not isinstance(timing, dict):
            return
        self._inc_metric("polar/wasted/session_count")
        for timing_field in _WASTED_TIMING_FIELDS:
            value = _optional_nonnegative_finite_float(timing.get(timing_field))
            if value is not None:
                self._inc_metric(
                    f"timing/wasted/{timing_field}_sum",
                    value,
                )

    def _record_wasted_inference(self, trace_metadata: Any) -> None:
        for timing in _iter_inference_timings(trace_metadata):
            self._inc_metric("polar/wasted/inference_timing_count")
            for timing_field in _WASTED_INFERENCE_FIELDS:
                value = _optional_nonnegative_finite_float(timing.get(timing_field))
                if value is not None:
                    self._inc_metric(
                        f"timing/wasted/inference_{timing_field}_sum",
                        value,
                    )

    def _set_fatal(self, exc: BaseException) -> None:
        with self._state_lock:
            if self._fatal_error is None:
                self._fatal_error = exc

    def _note_infrastructure_failure(self) -> PolarRolloutSchedulerError | None:
        with self._state_lock:
            self._consecutive_infrastructure_failures += 1
            count = self._consecutive_infrastructure_failures
            self._metrics["polar/consecutive_infrastructure_failures"] = float(count)
            limit = self.config.max_consecutive_infrastructure_failures
        if limit > 0 and count >= limit:
            return PolarRolloutSchedulerError(
                "Polar rollout stopped after "
                f"{count} consecutive infrastructure-only untrainable groups "
                f"(limit={limit})"
            )
        return None

    def _reset_infrastructure_failures(self) -> None:
        with self._state_lock:
            if self._consecutive_infrastructure_failures == 0:
                return
            self._consecutive_infrastructure_failures = 0
            self._metrics["polar/consecutive_infrastructure_failures"] = 0.0

    async def _submit_with_callback(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> TaskResult:
        """Submit a task, wait on its completion event, and fall back to polling."""
        task_id = payload["task_id"]
        # Register event BEFORE submit so a fast callback cannot arrive first.
        event = asyncio.Event()
        self._task_events[task_id] = event
        payload["callback_url"] = self._callback_url
        base_url = self.config.rollout_server_url
        try:
            resp = await client.post(
                f"{base_url}/rollout/task/submit",
                json=payload,
                headers=_control_plane_headers(),
            )
            resp.raise_for_status()
            return await self._await_task_result(client, task_id, event)
        except asyncio.CancelledError:
            # Cancel the server-side task as well as this local waiter. The
            # tombstone query closes the race where DELETE reaches the rollout
            # server before the submit request is registered.
            try:
                response = await client.delete(
                    f"{base_url}/rollout/task/{task_id}",
                    params={"register_if_missing": "true"},
                    timeout=10.0,
                )
                response.raise_for_status()
            except Exception:
                logger.warning(
                    "Failed to cancel remote Polar task %s during worker shutdown",
                    task_id,
                    exc_info=True,
                )
            raise
        finally:
            self._task_events.pop(task_id, None)
            self._task_results.pop(task_id, None)

    async def _await_task_result(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        event: asyncio.Event,
    ) -> TaskResult:
        """Wait on the completion event with a defensive 60s fallback poll."""
        base_url = self.config.rollout_server_url
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=_CALLBACK_FALLBACK_POLL_SECONDS)
            except asyncio.TimeoutError:
                status = await _get_task_status_with_retry(client, base_url, task_id)
                if status.status in ("completed", "failed"):
                    return TaskResult(
                        task_id=task_id,
                        status=status.status,
                        results=status.results,
                        result_paths=status.result_paths,
                    )
                continue
            result = self._task_results.get(task_id)
            if result is not None:
                return result
            # Race: event set but result missing — re-poll once.
            status = await _get_task_status_with_retry(client, base_url, task_id)
            return TaskResult(
                task_id=task_id,
                status=status.status,
                results=status.results,
                result_paths=status.result_paths,
            )


# ---------------------------------------------------------------------------
# One-shot eval rollout
# ---------------------------------------------------------------------------
async def _run_eval_rollout(
    args: Any,
    rollout_id: int,
    data_source: Any,
) -> Any:
    config = resolve_polar_slime_config(args)
    eval_datasets = list(getattr(args, "eval_datasets", []) or [])
    if eval_datasets:
        data: dict[str, dict[str, Any]] = {}
        metrics: dict[str, Any] = {}
        for dataset_cfg in eval_datasets:
            dataset_name, dataset_data, dataset_metrics = await _run_eval_dataset(
                args=args,
                config=config,
                rollout_id=rollout_id,
                dataset_cfg=dataset_cfg,
            )
            if dataset_name in data:
                raise ValueError(
                    f"Duplicate eval dataset name {dataset_name!r}; metric namespaces must be unique"
                )
            data[dataset_name] = dataset_data
            metrics.update(_prefix_eval_metrics(dataset_name, dataset_metrics))

        RolloutFnEvalOutput = _load_rollout_eval_output_type()
        return RolloutFnEvalOutput(data=data, metrics=metrics)

    logger.warning(
        "Polar eval called without args.eval_datasets; falling back to the training data source. "
        "Pass --eval-prompt-data to evaluate validation prompts."
    )
    sample_groups = _pull_sample_groups(data_source, args.rollout_batch_size)
    dataset_data, metrics = await _submit_eval_groups(
        args=args,
        config=config,
        dataset_name=config.eval_dataset_name,
        rollout_id=rollout_id,
        sample_groups=sample_groups,
        dataset_cfg=None,
    )
    RolloutFnEvalOutput = _load_rollout_eval_output_type()
    return RolloutFnEvalOutput(
        data={config.eval_dataset_name: dataset_data},
        metrics=_prefix_eval_metrics(config.eval_dataset_name, metrics),
    )


async def _run_eval_dataset(
    *,
    args: Any,
    config: PolarSlimeConfig,
    rollout_id: int,
    dataset_cfg: Any,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    dataset_name = str(getattr(dataset_cfg, "name", "") or config.eval_dataset_name)
    sample_groups = _load_eval_sample_groups(args, dataset_cfg)
    dataset_data, metrics = await _submit_eval_groups(
        args=args,
        config=config,
        dataset_name=dataset_name,
        rollout_id=rollout_id,
        sample_groups=sample_groups,
        dataset_cfg=dataset_cfg,
    )
    return dataset_name, dataset_data, metrics


async def _submit_eval_groups(
    *,
    args: Any,
    config: PolarSlimeConfig,
    dataset_name: str,
    rollout_id: int,
    sample_groups: list[list[Any]],
    dataset_cfg: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not sample_groups:
        raise ValueError("Polar eval dataset produced no sample groups")

    timeout = None if config.request_timeout is None else httpx.Timeout(config.request_timeout)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def _run_one(position: int, group: list[Any]) -> TaskResult:
        async with semaphore:
            payload: dict[str, Any] | None = None
            fallback_task_id = _eval_task_id(
                "payload-error",
                dataset_name=dataset_name,
                rollout_id=rollout_id,
                position=position,
            )
            try:
                payload = _build_task_payload(
                    args=args,
                    config=config,
                    group=group,
                    rollout_id=rollout_id,
                    task_position=position,
                    eval_dataset_cfg=dataset_cfg,
                    eval_dataset_name=dataset_name,
                )
                payload["task_id"] = _eval_task_id(
                    payload["task_id"],
                    dataset_name=dataset_name,
                    rollout_id=rollout_id,
                    position=position,
                )
                _attach_scheduler_metadata(
                    payload,
                    group_id=position,
                    policy_version=rollout_id,
                    rollout_step=rollout_id,
                )
                return await _submit_and_wait_for_task(
                    client,
                    config.rollout_server_url,
                    payload,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Evaluation is observational: one broken container, HTTP
                # request, or malformed task response must not take down the
                # trainer.  Preserve a task-shaped failure so the expected
                # sessions are zero-filled below and remain visible in the
                # dataset error counters.
                detail = " ".join(str(exc).splitlines())
                task_id = (
                    str(payload.get("task_id", fallback_task_id))
                    if isinstance(payload, dict)
                    else fallback_task_id
                )
                logger.warning(
                    "Eval dataset %s task/group %s failed before producing a result; "
                    "assigning reward 0 to its %d session(s): %s: %.500s",
                    dataset_name,
                    task_id,
                    len(group),
                    type(exc).__name__,
                    detail,
                )
                return TaskResult(
                    task_id=task_id,
                    status="failed",
                    results=[],
                    result_paths=[],
                )

    async with httpx.AsyncClient(timeout=timeout) as client:
        task_results = await asyncio.gather(
            *(_run_one(pos, g) for pos, g in enumerate(sample_groups))
        )

    output_groups: list[list[Any]] = []
    max_tokens = _resolve_max_tokens(args)
    for group, task_result in zip(sample_groups, task_results, strict=True):
        output_groups.append(
            _convert_eval_task_result_to_samples(
                config,
                task_result,
                group,
                dataset_name=dataset_name,
                max_tokens=max_tokens,
            )
        )

    flat_samples = [sample for group in output_groups for sample in group]
    session_outcomes, completed_count, model_failure_count = _eval_session_outcomes(
        flat_samples,
        config.reward_key,
        dataset_name=dataset_name,
    )
    trusted_rewards = [reward for reward, _sample in session_outcomes]
    reward_samples = [sample for _reward, sample in session_outcomes]
    expected_session_count = sum(len(group) for group in sample_groups)
    valid_count = len(trusted_rewards)
    error_count = max(0, expected_session_count - valid_count)
    # Every fixed eval item receives one vote.  Infrastructure failures and
    # malformed/missing results mean the model did not solve that item, so
    # account them as zero instead of silently shrinking the denominator.
    rewards = trusted_rewards + [0.0] * error_count
    accounted_count = len(rewards)
    _log_eval_zero_fill_diagnostics(
        dataset_name=dataset_name,
        sample_groups=sample_groups,
        task_results=task_results,
        flat_samples=flat_samples,
        zero_filled_count=error_count,
    )
    try:
        metrics = _build_metrics(
            config,
            task_results,
            output_groups,
            reward_filter="completed",
        )
        if not isinstance(metrics, dict):
            raise TypeError("eval metric builder must return a mapping")
    except Exception as exc:
        # Timing/diagnostic telemetry is optional. A malformed sample must not
        # stop the core per-item reward/count row from being emitted.
        detail = " ".join(str(exc).splitlines())
        logger.warning(
            "Eval dataset %s optional metric aggregation failed; continuing "
            "with reward/count metrics: %s: %.500s",
            dataset_name,
            type(exc).__name__,
            detail,
        )
        metrics = {}
    min_eval_samples = _eval_runtime_value(
        dataset_cfg,
        "min_eval_samples",
        args,
        ("min_eval_samples",),
    )
    metrics["polar/valid_count"] = float(valid_count)
    metrics["polar/completed_count"] = float(completed_count)
    metrics["polar/model_failure_count"] = float(model_failure_count)
    metrics["polar/error_count"] = float(error_count)
    metrics["polar/accounted_count"] = float(accounted_count)
    if min_eval_samples is not None:
        metrics["polar/min_valid_count"] = float(min_eval_samples)

    # `_build_metrics` receives trace-level samples because those are still
    # needed for timing and parser diagnostics. Replace only its primary reward
    # summary with one outcome per fixed-seed session so sessions that happen to
    # reconstruct into multiple traces cannot receive extra weight.
    # Trace-level reward diagnostics can be both differently weighted and
    # contaminated by a malformed raw reward (for example NaN or bool). Eval's
    # authoritative view is the finite, one-vote-per-session vector above.
    for metric_name in tuple(metrics):
        if metric_name.startswith("polar/reward"):
            metrics.pop(metric_name)
    if rewards:
        metrics["polar/reward_mean"] = sum(rewards) / len(rewards)
    if trusted_rewards:
        metrics["polar/reward_mean_valid"] = sum(trusted_rewards) / len(trusted_rewards)
        metrics["polar/reward_std_valid"] = (
            statistics.pstdev(trusted_rewards) if len(trusted_rewards) > 1 else 0.0
        )
    if len(rewards) > 1:
        metrics["polar/reward_std"] = statistics.pstdev(rewards)
    elif rewards:
        metrics["polar/reward_std"] = 0.0
    metrics["polar/reward_accounted_sessions"] = float(accounted_count)

    return {
        "rewards": rewards,
        "all_rewards": list(rewards),
        "truncated": [_is_truncated(s) for s in reward_samples] + [False] * error_count,
        "all_truncated": [_is_truncated(s) for s in flat_samples],
        "samples": flat_samples,
        "valid_count": valid_count,
        "accounted_count": accounted_count,
        "completed_count": completed_count,
        "model_failure_count": model_failure_count,
        "error_count": error_count,
        "min_eval_samples": min_eval_samples,
    }, metrics


def _eval_task_id(base_task_id: Any, *, dataset_name: str, rollout_id: int, position: int) -> str:
    """Namespace eval task ids away from train task ids.

    Training ids commonly use ``{rollout_id}-{sample.group_index}``; eval uses
    ``position`` as group index, so eval 11 / item 11 would collide with train
    group 11. A suffix keeps task polling and persisted result dirs separate.
    """
    safe_dataset = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in dataset_name)
    return f"{base_task_id}-eval-{safe_dataset}-{rollout_id}-{position}"


def _completed_session_samples(samples: list[Any]) -> list[Any]:
    return [
        sample
        for sample in samples
        if _sample_session_status(sample) == "COMPLETED"
        and not bool((getattr(sample, "metadata", {}) or {}).get("polar", {}).get("placeholder"))
    ]


def _eval_session_outcomes(
    samples: list[Any],
    reward_key: str,
    *,
    dataset_name: str,
) -> tuple[list[tuple[float, Any]], int, int]:
    """Return one trusted eval outcome per unique completed session.

    An ERROR/FAILED session never contributes an evaluator reward, even when
    its verifier happened to finish successfully.  The caller gives every such
    fixed eval item a fallback zero, so it remains an unsolved vote in the
    denominator while ``valid_count`` continues to mean completed executions.
    ``model_failure_count`` separately records errors with a trustworthy
    verifier result for diagnosis.
    """

    samples_by_session: dict[str, list[Any]] = {}
    for sample_index, sample in enumerate(samples):
        metadata = getattr(sample, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            logger.warning(
                "Eval dataset %s sample %d has malformed metadata; assigning reward 0",
                dataset_name,
                sample_index,
            )
            continue
        polar_meta = metadata.get("polar", {})
        if not isinstance(polar_meta, dict):
            logger.warning(
                "Eval dataset %s sample %d has malformed polar metadata; assigning reward 0",
                dataset_name,
                sample_index,
            )
            continue
        session_id = polar_meta.get("session_id")
        if session_id is None:
            logger.warning(
                "Eval dataset %s sample %d has no session_id; assigning reward 0",
                dataset_name,
                sample_index,
            )
            continue
        samples_by_session.setdefault(str(session_id), []).append(sample)

    outcomes: list[tuple[float, Any]] = []
    completed_count = 0
    model_failure_count = 0
    for session_samples in samples_by_session.values():
        non_placeholder = [
            sample
            for sample in session_samples
            if not bool(
                (getattr(sample, "metadata", {}) or {}).get("polar", {}).get("placeholder")
            )
        ]
        if not non_placeholder:
            continue

        completed = [
            sample
            for sample in non_placeholder
            if str(_sample_session_status(sample) or "").upper() == "COMPLETED"
        ]
        if completed:
            completed_rewards = [
                _extract_eval_sample_reward(
                    sample,
                    reward_key,
                    dataset_name=dataset_name,
                )
                for sample in completed
            ]
            # Any malformed trace makes this session an untrusted zero rather
            # than silently selecting another trace and hiding the failure.
            if all(reward is not None for reward in completed_rewards):
                finite_rewards = [float(reward) for reward in completed_rewards]
                outcomes.append((sum(finite_rewards) / len(finite_rewards), completed[0]))
                completed_count += 1
            continue

        trusted_failure = next(
            (
                sample
                for sample in non_placeholder
                if str(_sample_session_status(sample) or "").upper() in {"ERROR", "FAILED"}
                and _has_trusted_eval_failure(sample)
            ),
            None,
        )
        if trusted_failure is not None:
            # Do not append an outcome here.  The fixed-denominator zero-fill
            # represents this failed item below and also makes it visible in
            # error_count/logs.  This avoids classifying an ERROR execution as
            # a valid evaluation merely because its verifier observed a solved
            # final container state.
            model_failure_count += 1

    return outcomes, completed_count, model_failure_count


def _extract_eval_sample_reward(
    sample: Any,
    reward_key: str,
    *,
    dataset_name: str,
) -> float | None:
    """Strict finite eval reward extraction with per-session diagnostics."""

    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        if reward_key in reward:
            raw_value = reward[reward_key]
        elif "score" in reward:
            raw_value = reward["score"]
        else:
            raw_value = None
    else:
        raw_value = reward

    try:
        if isinstance(raw_value, bool) or raw_value is None:
            raise TypeError("reward is missing or boolean")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("reward is not finite")
    except (TypeError, ValueError, OverflowError) as exc:
        polar_meta = _sample_polar_metadata(sample)
        session_id = polar_meta.get("session_id") or getattr(sample, "session_id", None)
        logger.warning(
            "Eval dataset %s session %s has malformed reward; assigning reward 0: %s",
            dataset_name,
            session_id or "UNKNOWN",
            exc,
        )
        return None
    return value


def _log_eval_zero_fill_diagnostics(
    *,
    dataset_name: str,
    sample_groups: list[list[Any]],
    task_results: list[TaskResult],
    flat_samples: list[Any],
    zero_filled_count: int,
) -> None:
    """Log actionable details for eval items represented by fallback zeros."""

    if zero_filled_count <= 0:
        return

    for group, task_result in zip(sample_groups, task_results, strict=True):
        expected = len(group)
        returned = len(task_result.results)
        if task_result.status != "completed" or returned != expected:
            logger.warning(
                "Eval dataset %s task %s status=%s returned %d/%d session "
                "result(s); missing outcomes are assigned reward 0",
                dataset_name,
                task_result.task_id,
                task_result.status,
                returned,
                expected,
            )

    logged_sessions: set[str] = set()
    for sample in flat_samples:
        polar_meta = _sample_polar_metadata(sample)
        if not polar_meta:
            continue
        session_id = str(polar_meta.get("session_id") or "")
        if not session_id or session_id in logged_sessions:
            continue
        status = str(_sample_session_status(sample) or "").upper()
        placeholder = bool(polar_meta.get("placeholder"))
        if not placeholder and status == "COMPLETED":
            continue
        logged_sessions.add(session_id)
        result_error = polar_meta.get("result_error")
        trajectory_error = polar_meta.get("trajectory_error")
        detail = result_error or trajectory_error or "no trusted evaluator outcome"
        logger.warning(
            "Eval dataset %s session %s status=%s produced no trusted outcome; "
            "assigning reward 0: %.500s",
            dataset_name,
            session_id,
            status or "UNKNOWN",
            " ".join(str(detail).splitlines()),
        )

    logger.warning(
        "Eval dataset %s zero-filled %d failed/missing sample(s); evaluation "
        "continues and the failures remain in error_count",
        dataset_name,
        zero_filled_count,
    )


def _has_trusted_eval_failure(sample: Any) -> bool:
    polar_meta = _sample_polar_metadata(sample)
    trajectory_metadata = polar_meta.get("trajectory_metadata")
    if not isinstance(trajectory_metadata, dict):
        return False
    evaluation = trajectory_metadata.get("evaluation")
    if not isinstance(evaluation, dict):
        return False
    exit_code = evaluation.get("verifier_exit_code")
    return (
        evaluation.get("verifier_reward_accepted") is True
        and not isinstance(exit_code, bool)
        and exit_code == 0
    )


def _sample_session_status(sample: Any) -> str | None:
    polar_meta = _sample_polar_metadata(sample)
    status = polar_meta.get("session_status")
    return getattr(status, "value", status)


def _sample_polar_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return {}
    polar_meta = metadata.get("polar", {})
    return polar_meta if isinstance(polar_meta, dict) else {}


def _eval_data_integrity_records() -> dict[str, dict[str, str]]:
    """Decode the optional launcher-pinned eval manifest.

    The manifest itself is copied into the Ray runtime environment rather than
    re-read from a mutable file. This makes its expected digests stable for the
    lifetime of the rollout manager while still leaving an on-disk manifest for
    experiment auditing.
    """

    encoded = os.environ.get(_EVAL_DATA_INTEGRITY_ENV, "").strip()
    if not encoded:
        return {}
    try:
        payload = base64.b64decode(encoded, validate=True)
        manifest = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise PolarEvalDataIntegrityError(
            f"Invalid {_EVAL_DATA_INTEGRITY_ENV} manifest: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise PolarEvalDataIntegrityError(
            f"{_EVAL_DATA_INTEGRITY_ENV} must contain a schema_version=1 object"
        )
    if manifest.get("algorithm") != "sha256":
        raise PolarEvalDataIntegrityError("Eval data integrity manifest must use sha256")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise PolarEvalDataIntegrityError("Eval data integrity manifest has no datasets")

    records: dict[str, dict[str, str]] = {}
    for index, entry in enumerate(datasets):
        if not isinstance(entry, dict):
            raise PolarEvalDataIntegrityError(
                f"Eval data integrity dataset {index} is not an object"
            )
        name = entry.get("name")
        path = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(name, str) or not name:
            raise PolarEvalDataIntegrityError(f"Eval data integrity dataset {index} has no name")
        if not isinstance(path, str) or not path:
            raise PolarEvalDataIntegrityError(f"Eval data integrity dataset {name!r} has no path")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise PolarEvalDataIntegrityError(
                f"Eval data integrity dataset {name!r} has an invalid sha256"
            )
        canonical = str(Path(path).expanduser().resolve(strict=False))
        normalized = {"name": name, "sha256": digest.lower()}
        previous = records.get(canonical)
        if previous is not None and previous != normalized:
            raise PolarEvalDataIntegrityError(
                f"Conflicting eval data integrity entries for {canonical}"
            )
        records[canonical] = normalized
    return records


def _expected_eval_data_sha256(dataset_cfg: Any) -> str | None:
    records = _eval_data_integrity_records()
    if not records:
        return None
    path = str(getattr(dataset_cfg, "path"))
    canonical = str(Path(path).expanduser().resolve(strict=False))
    record = records.get(canonical)
    if record is None:
        protected = ", ".join(sorted(records))
        raise PolarEvalDataIntegrityError(
            f"Eval dataset {canonical} is absent from the immutable integrity manifest; "
            f"protected paths: {protected}"
        )
    return record["sha256"]


def _load_eval_sample_groups(args: Any, dataset_cfg: Any) -> list[list[Any]]:
    Sample = _load_sample_type()
    path = str(getattr(dataset_cfg, "path"))
    expected_sha256 = _expected_eval_data_sha256(dataset_cfg)
    input_key = getattr(dataset_cfg, "input_key", None) or getattr(args, "input_key", "prompt")
    label_key = getattr(dataset_cfg, "label_key", None) or getattr(args, "label_key", None)
    metadata_key = getattr(dataset_cfg, "metadata_key", None) or getattr(
        args, "metadata_key", "metadata"
    )
    tool_key = getattr(dataset_cfg, "tool_key", None) or getattr(args, "tool_key", None)
    group_size = int(
        getattr(dataset_cfg, "n_samples_per_eval_prompt", None)
        or getattr(args, "n_samples_per_eval_prompt", None)
        or 1
    )
    if group_size <= 0:
        raise ValueError("n_samples_per_eval_prompt must be positive")
    max_prompt_len = getattr(dataset_cfg, "max_prompt_len", None)
    if max_prompt_len is None:
        max_prompt_len = getattr(args, "eval_max_prompt_len", None)
    if max_prompt_len is not None and int(max_prompt_len) <= 0:
        raise ValueError("eval_max_prompt_len must be positive when provided")

    groups: list[list[Any]] = []
    sample_index = 0
    for prompt_index, row in enumerate(_read_jsonl_rows(path, expected_sha256=expected_sha256)):
        if input_key not in row:
            raise KeyError(f"Eval row {prompt_index} in {path} missing input key {input_key!r}")
        prompt = row[input_key]
        if max_prompt_len is not None:
            prompt_tokens = _eval_prompt_token_length(args, prompt)
            if prompt_tokens > int(max_prompt_len):
                logger.warning(
                    "Skipping eval row %d in %s: prompt has %d tokens (limit=%d)",
                    prompt_index,
                    path,
                    prompt_tokens,
                    int(max_prompt_len),
                )
                continue

        metadata = _inject_eval_metadata(dataset_cfg, row.get(metadata_key))
        if tool_key and tool_key in row:
            tools = row[tool_key]
            if isinstance(tools, str):
                tools = json.loads(tools)
            metadata["tools"] = tools

        group: list[Any] = []
        for _ in range(group_size):
            sample = Sample(
                prompt=copy.deepcopy(prompt),
                label=row.get(label_key) if label_key else None,
                metadata=copy.deepcopy(metadata),
                group_index=prompt_index,
                index=sample_index,
            )
            sample.generate_function_path = getattr(
                dataset_cfg, "custom_generate_function_path", None
            )
            group.append(sample)
            sample_index += 1
        groups.append(group)

    return groups


def _eval_prompt_token_length(args: Any, prompt: Any) -> int:
    checkpoint = str(getattr(args, "hf_checkpoint", "") or "").strip()
    if not checkpoint:
        raise ValueError("hf_checkpoint is required to enforce eval_max_prompt_len")
    tokenizer = _eval_tokenizer_cache.get(checkpoint)
    if tokenizer is None:
        from slime.utils.processing_utils import load_tokenizer

        tokenizer = load_tokenizer(checkpoint, trust_remote_code=True)
        _eval_tokenizer_cache[checkpoint] = tokenizer
    encoded = tokenizer(
        prompt_to_instruction_text(prompt),
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"]
    return len(input_ids)


def _read_jsonl_rows(
    path: str,
    *,
    expected_sha256: str | None = None,
) -> list[dict[str, Any]]:
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise PolarEvalDataIntegrityError(f"Cannot read eval dataset {path}: {exc}") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise PolarEvalDataIntegrityError(
            f"Eval dataset changed after launcher validation: {path}; "
            f"expected sha256={expected_sha256}, actual sha256={actual_sha256}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Eval dataset {path} is not UTF-8: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Eval row {line_number} in {path} is not a JSON object")
        rows.append(row)
    return rows


def _inject_eval_metadata(dataset_cfg: Any, sample_metadata: Any) -> dict[str, Any]:
    inject = getattr(dataset_cfg, "inject_metadata", None)
    if callable(inject):
        metadata = inject(sample_metadata)
    elif isinstance(sample_metadata, dict):
        metadata = dict(sample_metadata)
    else:
        metadata = {}
    return metadata


def _prefix_eval_metrics(dataset_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    prefixed: dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("timing/"):
            prefixed[f"timing/eval/{dataset_name}/{key.removeprefix('timing/')}"] = value
        elif key.startswith("polar/"):
            prefixed[f"eval/{dataset_name}/{key.removeprefix('polar/')}"] = value
        else:
            prefixed[f"eval/{dataset_name}/{key}"] = value
    return prefixed


def _pull_sample_groups(data_source: Any, batch_size: int) -> list[list[Any]]:
    getter = getattr(data_source, "get_samples", None)
    if callable(getter):
        groups = getter(batch_size)
    elif callable(data_source):
        groups = data_source(batch_size)
    else:
        raise ValueError("data_source must expose get_samples(num_samples) or be callable")
    if not isinstance(groups, list):
        raise ValueError("data_source.get_samples must return a list of sample groups")
    for group in groups:
        if not group:
            raise ValueError("Slime data source returned an empty sample group")
    return groups


def _build_metrics(
    config: PolarSlimeConfig,
    task_results: list[TaskResult],
    output_groups: list[list[Any]],
    *,
    reward_filter: str = "all",
) -> dict[str, Any]:
    flat_samples = [sample for group in output_groups for sample in group]
    all_rewards = [_extract_sample_reward(s, config.reward_key) for s in flat_samples]
    completed_rewards = [
        _extract_sample_reward(s, config.reward_key)
        for s in _completed_session_samples(flat_samples)
    ]
    if reward_filter == "all":
        rewards = all_rewards
    elif reward_filter == "completed":
        rewards = completed_rewards
    else:
        raise ValueError("reward_filter must be 'all' or 'completed'")
    metrics: dict[str, Any] = {}
    metrics.update(_polar_extra_metrics(flat_samples, rewards, config.reward_key))
    return metrics


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def _load_training_dynamic_filter(args: Any) -> Any | None:
    path = getattr(args, "dynamic_sampling_filter_path", None)
    if not path:
        return None

    from slime.utils.misc import load_function

    return load_function(path)


def _call_training_dynamic_filter(dynamic_filter: Any, args: Any, samples: list[Any]) -> Any:
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput, call_dynamic_filter

    # Failed, parser-invalid, placeholder, and otherwise fully masked samples
    # are diagnostic-only.  Feeding their fail-closed reward=0 into a dynamic
    # nonzero-std filter can manufacture an apparent mixed-reward group from
    # valid rewards [1, 1, ...] plus an infrastructure/agent error.  That group
    # has no real preference signal once reward post-processing correctly
    # removes the failed trajectory.  Filter on exactly the trajectories that
    # can contribute gradients instead.
    trainable_samples = [sample for sample in samples if _sample_has_trainable_tokens(sample)]
    if not trainable_samples:
        return DynamicFilterOutput(keep=False, reason="no_trainable_samples")
    return call_dynamic_filter(dynamic_filter, args, trainable_samples)


_CANDIDATE_SESSION_COUNT_METRICS = {
    "attempted_sessions": "polar/rollout_attempted_sessions",
    "trainable_sessions": "polar/rollout_trainable_sessions",
    "fully_masked_sessions": "polar/rollout_fully_masked_sessions",
    "successful_sessions": "polar/rollout_successful_sessions",
    "terminal_timeout_sessions": "polar/terminal_timeout_sessions",
    "terminal_error_sessions": "polar/terminal_error_sessions",
    "timeout_agent_exec_sessions": "polar/timeout_agent_exec_sessions",
    "timeout_agent_postprocess_sessions": "polar/timeout_agent_postprocess_sessions",
    "timeout_trainable_sessions": "polar/timeout_trainable_sessions",
    "timeout_masked_sessions": "polar/timeout_masked_sessions",
}

_CANDIDATE_DECOMPOSED_METRICS = {
    name: (
        f"polar/spilot_router/{name}_accounted_session_count",
        f"polar/spilot_router/{name}_mean",
        f"polar/spilot_router/{name}_std",
    )
    for name in (
        "accuracy_outcome",
        "total_cost",
        "cost_penalty_fraction",
        "cost_penalty_reward_delta",
        "cost_adjusted_reward",
        "total_latency_seconds",
        "latency_penalty_fraction",
        "latency_penalty_reward_delta",
    )
}


@dataclass(frozen=True, slots=True)
class _CandidatePoolHealthGateConfig:
    candidate_aliases: tuple[str, ...]
    min_observed_sessions: int
    min_completion_fraction: float


def _candidate_pool_health_gate_config(
    config: Any,
) -> _CandidatePoolHealthGateConfig | None:
    """Validate the opt-in candidate-availability gate and its model identities."""

    enabled = getattr(config, "candidate_pool_health_gate_enabled", False)
    if type(enabled) is not bool:
        raise ValueError("candidate_pool_health_gate_enabled must be a boolean")
    if not enabled:
        return None

    raw_min_sessions = getattr(config, "candidate_pool_health_min_observed_sessions", 16)
    if isinstance(raw_min_sessions, bool):
        raise ValueError("candidate_pool_health_min_observed_sessions must be an integer")
    try:
        min_observed_sessions = int(raw_min_sessions)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "candidate_pool_health_min_observed_sessions must be an integer"
        ) from exc
    if min_observed_sessions <= 0 or min_observed_sessions != raw_min_sessions:
        raise ValueError("candidate_pool_health_min_observed_sessions must be positive")

    raw_min_fraction = getattr(config, "candidate_pool_health_min_completion_fraction", 0.1)
    if isinstance(raw_min_fraction, bool):
        raise ValueError("candidate_pool_health_min_completion_fraction must be numeric")
    try:
        min_completion_fraction = float(raw_min_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "candidate_pool_health_min_completion_fraction must be numeric"
        ) from exc
    if not math.isfinite(min_completion_fraction) or not 0.0 <= min_completion_fraction <= 1.0:
        raise ValueError(
            "candidate_pool_health_min_completion_fraction must be between 0 and 1"
        )

    task_template = getattr(config, "task_template", None)
    agent = task_template.get("agent") if isinstance(task_template, dict) else None
    settings = agent.get("settings") if isinstance(agent, dict) else None
    raw_pool = settings.get("model_pool") if isinstance(settings, dict) else None
    if isinstance(raw_pool, dict):
        raw_candidates = list(raw_pool.values())
    elif isinstance(raw_pool, list):
        raw_candidates = list(raw_pool)
    else:
        raise ValueError(
            "candidate pool health gate requires agent.settings.model_pool"
        )

    aliases: list[str] = []
    for candidate in raw_candidates:
        alias = candidate if isinstance(candidate, str) else None
        if isinstance(candidate, dict):
            alias = candidate.get("model")
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError(
                "candidate pool health gate requires a model alias for every candidate"
            )
        aliases.append(alias.strip())
    if len(aliases) < 2 or len(set(aliases)) != len(aliases):
        raise ValueError(
            "candidate pool health gate requires at least two unique model aliases"
        )
    return _CandidatePoolHealthGateConfig(
        candidate_aliases=tuple(sorted(aliases)),
        min_observed_sessions=min_observed_sessions,
        min_completion_fraction=min_completion_fraction,
    )


@dataclass(slots=True)
class _CandidatePoolHealthAccumulator:
    """Retain per-session pool availability before dynamic sampling selection."""

    expected_aliases: tuple[str, ...]
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    conflicting_sessions: set[str] = field(default_factory=set)
    missing_session_id_telemetry_count: int = 0
    group_count: int = 0
    reservation_ids: set[int] = field(default_factory=set)
    policy_versions: set[int] = field(default_factory=set)

    def add(self, completed: _CompletedGroup) -> None:
        self.group_count += 1
        if completed.reservation_id is not None:
            self.reservation_ids.add(int(completed.reservation_id))
        self.policy_versions.add(int(completed.policy_version))
        for sample in completed.samples:
            sample_metadata = getattr(sample, "metadata", None)
            polar = sample_metadata.get("polar") if isinstance(sample_metadata, dict) else None
            if not isinstance(polar, dict):
                continue
            trajectory = polar.get("trajectory_metadata")
            evaluation = (
                trajectory.get("evaluation") if isinstance(trajectory, dict) else None
            )
            router = evaluation.get("spilot_router") if isinstance(evaluation, dict) else None
            if not isinstance(router, dict):
                continue
            session_id = polar.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                self.missing_session_id_telemetry_count += 1
                continue
            if session_id in self.conflicting_sessions:
                continue
            previous = self.sessions.get(session_id)
            if previous is None:
                self.sessions[session_id] = router
            elif not _strict_telemetry_equal(previous, router):
                self.sessions.pop(session_id, None)
                self.conflicting_sessions.add(session_id)

    def report(
        self,
        gate_config: _CandidatePoolHealthGateConfig,
        *,
        rollout_id: int,
        accepted_group_count: int,
    ) -> dict[str, Any]:
        expected = set(self.expected_aliases)
        if tuple(sorted(expected)) != gate_config.candidate_aliases:
            raise ValueError("candidate pool health accumulator/config aliases differ")

        labels = {
            alias: f"C{index}" for index, alias in enumerate(gate_config.candidate_aliases)
        }
        candidate_stats: dict[str, dict[str, Any]] = {
            alias: {
                "label": labels[alias],
                "alias": alias,
                "observed_session_count": 0,
                "available_session_count": 0,
                "unavailable_session_count": 0,
                "telemetry_error_session_count": 0,
                "attempted_call_count": 0,
                "completed_call_count": 0,
                "failed_call_count": 0,
                "timeout_call_count": 0,
                "unknown_status_call_count": 0,
                "skipped_call_count": 0,
                "admission_failure_session_count": 0,
                "pre_call_infrastructure_failure_session_count": 0,
            }
            for alias in gate_config.candidate_aliases
        }
        globally_invalid_sessions: set[str] = set(self.conflicting_sessions)
        unattributed_observation_sessions: set[str] = set()
        observed_session_ids: set[str] = set()

        for session_id, router in self.sessions.items():
            aliases_by_slot: dict[str, str] = {}
            slot_mapping = router.get("slot_mapping")
            if isinstance(slot_mapping, dict):
                for raw_slot, raw_candidate in slot_mapping.items():
                    alias = (
                        raw_candidate.get("model")
                        if isinstance(raw_candidate, dict)
                        else None
                    )
                    if isinstance(alias, str) and alias:
                        aliases_by_slot[str(raw_slot).upper()] = alias
            mapping_valid = (
                len(aliases_by_slot) == len(expected)
                and set(aliases_by_slot.values()) == expected
            )
            expected_action_calls: list[tuple[str, str, str]] = []
            actions = router.get("actions")
            if actions is not None and not isinstance(actions, list):
                globally_invalid_sessions.add(session_id)
                actions = []
            route_actions_seen = 0
            for action in actions or []:
                if not isinstance(action, dict) or action.get("valid") is not True:
                    continue
                action_name = str(action.get("action") or "").upper()
                if action_name not in {"ROUTE", "VERIFY"}:
                    continue
                action_slot = str(action.get("model_slot") or "").upper()
                action_alias = aliases_by_slot.get(action_slot)
                if action_alias not in expected:
                    globally_invalid_sessions.add(session_id)
                    unattributed_observation_sessions.add(session_id)
                    continue
                if action_name == "ROUTE":
                    # Task-level episodes have one ROUTE ("solve"); turn-level
                    # episodes route every pool call, and every ROUTE after the
                    # first runs a continuation agent.
                    role = "solve" if route_actions_seen == 0 else "continue"
                    route_actions_seen += 1
                else:
                    role = "verify"
                expected_action_calls.append((role, action_slot, action_alias))

            evidence: dict[str, dict[str, bool]] = {
                alias: {
                    "completed": False,
                    "unavailable": False,
                    "telemetry_error": False,
                }
                for alias in gate_config.candidate_aliases
            }
            covered_action_keys: set[tuple[str, str]] = set()
            calls = router.get("calls")
            if calls is not None and not isinstance(calls, list):
                globally_invalid_sessions.add(session_id)
                calls = []
            for call in calls or []:
                if not isinstance(call, dict):
                    globally_invalid_sessions.add(session_id)
                    continue
                call_model = call.get("model")
                model_alias = call_model if call_model in expected else None
                call_slot = str(call.get("slot") or "").upper()
                slot_alias = aliases_by_slot.get(call_slot)
                alias = model_alias if model_alias == slot_alias else None
                if alias is None:
                    globally_invalid_sessions.add(session_id)
                    if model_alias is not None:
                        evidence[model_alias]["telemetry_error"] = True
                    if slot_alias in expected:
                        evidence[slot_alias]["telemetry_error"] = True

                attempted = call.get("attempted")
                if attempted is False:
                    if alias is not None:
                        candidate_stats[alias]["skipped_call_count"] += 1
                    continue
                if attempted is not True:
                    globally_invalid_sessions.add(session_id)
                    if alias is not None:
                        evidence[alias]["telemetry_error"] = True
                    continue
                if alias is None:
                    unattributed_observation_sessions.add(session_id)
                    continue

                observed_session_ids.add(session_id)
                role = str(call.get("role") or "").lower()
                if role in {"solve", "verify", "continue"}:
                    covered_action_keys.add((role, call_slot))
                else:
                    globally_invalid_sessions.add(session_id)
                    evidence[alias]["telemetry_error"] = True
                stats = candidate_stats[alias]
                stats["attempted_call_count"] += 1
                status = str(call.get("status") or "").lower()
                if status == "completed":
                    stats["completed_call_count"] += 1
                    evidence[alias]["completed"] = True
                elif status in {"failed", "timeout"}:
                    stats[f"{status}_call_count"] += 1
                    evidence[alias]["unavailable"] = True
                else:
                    stats["unknown_status_call_count"] += 1
                    evidence[alias]["unavailable"] = True
                    evidence[alias]["telemetry_error"] = True
                    globally_invalid_sessions.add(session_id)

            admission_failure = router.get("admission_failure")
            if isinstance(admission_failure, dict):
                failure_model = admission_failure.get("model")
                if failure_model in expected:
                    evidence[failure_model]["unavailable"] = True
                    candidate_stats[failure_model]["admission_failure_session_count"] += 1
                    observed_session_ids.add(session_id)
                    # Admission fails before a call record exists. Attribute it
                    # to the latest unmatched action for that stable alias so
                    # the infrastructure reconciliation below does not count
                    # the same failure twice.
                    for role, slot, alias in reversed(expected_action_calls):
                        key = (role, slot)
                        if alias == failure_model and key not in covered_action_keys:
                            covered_action_keys.add(key)
                            break
                else:
                    globally_invalid_sessions.add(session_id)
                    unattributed_observation_sessions.add(session_id)

            if router.get("termination_reason") == "infrastructure_error":
                pre_call_failure_aliases: set[str] = set()
                for role, slot, alias in expected_action_calls:
                    if (role, slot) not in covered_action_keys:
                        evidence[alias]["unavailable"] = True
                        pre_call_failure_aliases.add(alias)
                        observed_session_ids.add(session_id)
                for alias in pre_call_failure_aliases:
                    candidate_stats[alias][
                        "pre_call_infrastructure_failure_session_count"
                    ] += 1
                if not expected_action_calls:
                    globally_invalid_sessions.add(session_id)
                    unattributed_observation_sessions.add(session_id)

            has_candidate_evidence = any(
                item["completed"] or item["unavailable"] for item in evidence.values()
            )
            if not mapping_valid and has_candidate_evidence:
                globally_invalid_sessions.add(session_id)
                for alias, item in evidence.items():
                    if item["completed"] or item["unavailable"]:
                        item["telemetry_error"] = True

            for alias, item in evidence.items():
                if not (item["completed"] or item["unavailable"]):
                    continue
                stats = candidate_stats[alias]
                stats["observed_session_count"] += 1
                if item["completed"]:
                    stats["available_session_count"] += 1
                else:
                    stats["unavailable_session_count"] += 1
                if item["telemetry_error"]:
                    stats["telemetry_error_session_count"] += 1

        triggered_candidates: list[str] = []
        trigger_reasons: list[str] = []
        for alias in gate_config.candidate_aliases:
            stats = candidate_stats[alias]
            observed = int(stats["observed_session_count"])
            available = int(stats["available_session_count"])
            fraction = available / observed if observed else None
            eligible = observed >= gate_config.min_observed_sessions
            stats["completion_fraction"] = fraction
            stats["eligible"] = eligible
            stats["insufficient_evidence"] = not eligible
            stats["triggered"] = False
            reasons: list[str] = []
            if eligible and fraction is not None:
                if fraction < gate_config.min_completion_fraction:
                    reasons.append("completion_fraction_below_threshold")
                if int(stats["telemetry_error_session_count"]) > 0:
                    reasons.append("candidate_telemetry_integrity_error")
            if reasons:
                stats["triggered"] = True
                stats["trigger_reasons"] = reasons
                triggered_candidates.append(labels[alias])
                trigger_reasons.extend(f"{labels[alias]}:{reason}" for reason in reasons)

        total_observed_sessions = len(observed_session_ids)
        health_evidence_sessions = (
            observed_session_ids
            | globally_invalid_sessions
            | unattributed_observation_sessions
        )
        if (
            len(health_evidence_sessions) >= gate_config.min_observed_sessions
            and globally_invalid_sessions
        ):
            trigger_reasons.append("global:candidate_telemetry_integrity_error")
        if len(unattributed_observation_sessions) >= gate_config.min_observed_sessions:
            trigger_reasons.append("global:unattributed_candidate_observations")
        if self.missing_session_id_telemetry_count >= gate_config.min_observed_sessions:
            trigger_reasons.append("global:missing_session_identity_telemetry")

        return {
            "schema_version": 1,
            "rollout_id": int(rollout_id),
            "accepted_group_count": int(accepted_group_count),
            "decision_window_group_count": int(self.group_count),
            "decision_window_session_count": len(self.sessions),
            "reservation_ids": sorted(self.reservation_ids),
            "policy_versions": sorted(self.policy_versions),
            "thresholds": {
                "min_observed_sessions": gate_config.min_observed_sessions,
                "min_completion_fraction": gate_config.min_completion_fraction,
            },
            "candidates": {
                labels[alias]: candidate_stats[alias]
                for alias in gate_config.candidate_aliases
            },
            "telemetry": {
                "conflicting_session_count": len(self.conflicting_sessions),
                "globally_invalid_session_count": len(globally_invalid_sessions),
                "missing_session_id_telemetry_count": self.missing_session_id_telemetry_count,
                "unattributed_observation_session_count": len(
                    unattributed_observation_sessions
                ),
                "total_observed_session_count": total_observed_sessions,
            },
            "triggered": bool(trigger_reasons),
            "triggered_candidates": triggered_candidates,
            "trigger_reasons": trigger_reasons,
        }


def _candidate_pool_health_metrics(report: dict[str, Any]) -> dict[str, float]:
    prefix = "polar/candidate_pool_health"
    metrics = {
        f"{prefix}/gate_triggered": float(bool(report.get("triggered"))),
        f"{prefix}/decision_window_group_count": float(
            report.get("decision_window_group_count", 0)
        ),
    }
    for label, raw_stats in (report.get("candidates") or {}).items():
        if not isinstance(label, str) or not isinstance(raw_stats, dict):
            continue
        candidate_prefix = f"{prefix}/{label.lower()}"
        for field_name in (
            "observed_session_count",
            "available_session_count",
            "unavailable_session_count",
            "telemetry_error_session_count",
            "attempted_call_count",
            "completed_call_count",
            "failed_call_count",
            "timeout_call_count",
            "unknown_status_call_count",
            "skipped_call_count",
            "admission_failure_session_count",
            "pre_call_infrastructure_failure_session_count",
        ):
            metrics[f"{candidate_prefix}/{field_name}"] = float(
                raw_stats.get(field_name, 0)
            )
        metrics[f"{candidate_prefix}/eligible"] = float(bool(raw_stats.get("eligible")))
        completion_fraction = raw_stats.get("completion_fraction")
        if completion_fraction is not None:
            metrics[f"{candidate_prefix}/completion_fraction"] = float(completion_fraction)
    return metrics


def _persist_candidate_pool_health_incident(
    args: Any,
    report: dict[str, Any],
) -> Path:
    save_root = getattr(args, "save", None)
    state_file = os.environ.get("TMAX_RUN_STATE_FILE")
    incident_dirs: list[Path] = []
    if save_root:
        incident_dirs.append(
            Path(save_root) / "rollout" / "candidate_pool_health_incidents"
        )
    if state_file:
        # This independent lineage-scoped location is a durability fallback if
        # the checkpoint tree itself becomes temporarily unwritable.  The
        # watcher scans both locations and still binds payloads to SLURM_JOB_ID.
        incident_dirs.append(
            Path(f"{state_file}.candidate_pool_health_incidents")
        )
    incident_dirs = list(dict.fromkeys(incident_dirs))
    if not incident_dirs:
        raise OSError(
            "candidate-pool health incident has neither args.save nor "
            "TMAX_RUN_STATE_FILE storage"
        )

    rollout_id = int(report["rollout_id"])
    payload = dict(report)
    payload["created_unix_time"] = time.time()
    # The watcher must only latch an incident produced by the terminal Slurm
    # job it is accounting.  Keeping the scheduler identity in the durable
    # payload lets a later, explicitly restarted job reuse the same SAVE_DIR
    # without being blocked by a stale incident from an older attempt.
    payload["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")

    failures: list[Exception] = []
    for incident_dir in incident_dirs:
        destination = incident_dir / f"rollout_{rollout_id:07d}.json"
        temporary = incident_dir / (
            f".{destination.name}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            incident_dir.mkdir(parents=True, exist_ok=True)
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(incident_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return destination
        except Exception as exc:
            failures.append(exc)
            logger.exception(
                "Failed to persist candidate-pool health incident under %s",
                incident_dir,
            )
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # The directory itself may be the failed component.  Cleanup
                # must not suppress the attempt at the independent fallback.
                pass
    raise OSError(
        "candidate-pool health incident could not be persisted to any durable location"
    ) from failures[-1]


def _enforce_candidate_pool_health_gate(
    args: Any,
    accumulator: _CandidatePoolHealthAccumulator,
    gate_config: _CandidatePoolHealthGateConfig,
    *,
    rollout_id: int,
    accepted_group_count: int,
    partial_store: PartialRolloutStore | None = None,
) -> dict[str, Any]:
    report = accumulator.report(
        gate_config,
        rollout_id=rollout_id,
        accepted_group_count=accepted_group_count,
    )
    if not report["triggered"]:
        return report

    try:
        stop_global_worker()
    except Exception:
        logger.exception("Failed to stop Polar worker after candidate-pool health gate")
    # Accepted groups have durable KEEP records before the complete decision
    # window is available.  They are intentionally not sealed with READY, but
    # ordinary partial recovery would still reuse those bad provider results.
    # Quarantine the whole uncommitted WAL so the exact checkpoint cursor
    # regenerates the batch after the backend is healthy.
    wal_quarantine_succeeded: bool | None = None
    wal_quarantine_error_type: str | None = None
    if partial_store is not None:
        try:
            partial_store.quarantine(
                "candidate-pool health gate rejected the uncommitted decision window"
            )
            wal_quarantine_succeeded = True
        except Exception as exc:
            wal_quarantine_succeeded = False
            wal_quarantine_error_type = type(exc).__name__
            report["trigger_reasons"].append(
                "global:partial_wal_quarantine_failed"
            )
            logger.exception(
                "Failed to quarantine partial rollout WAL after candidate-pool health gate"
            )
    report["partial_wal"] = {
        "present": partial_store is not None,
        "quarantine_succeeded": wal_quarantine_succeeded,
        "quarantine_error_type": wal_quarantine_error_type,
    }
    try:
        incident_path = _persist_candidate_pool_health_incident(args, report)
    except Exception as exc:
        logger.exception(
            "Candidate-pool health incident persistence failed in every durable location"
        )
        raise CandidatePoolHealthGateError(
            f"candidate-pool health gate rejected rollout {rollout_id}, but its "
            "incident could not be persisted; refusing to continue"
        ) from exc
    logger.error(
        "Candidate-pool health gate rejected rollout %d before training: %s incident=%s",
        rollout_id,
        ", ".join(report["trigger_reasons"]),
        incident_path or "unavailable",
    )
    raise CandidatePoolHealthGateError(
        f"candidate-pool health gate rejected rollout {rollout_id}: "
        + ", ".join(report["trigger_reasons"])
    )


@dataclass(slots=True)
class _CandidateQualityAccumulator:
    """Streaming quality summary before dynamic-sampling selection."""

    group_count: int = 0
    telemetry_error_count: int = 0
    accounted_sessions: float = 0.0
    quality_eligible_sessions: float = 0.0
    early_stop_cancelled_sessions: float = 0.0
    reward_sum: float = 0.0
    reward_square_sum: float = 0.0
    quality_group_count: int = 0
    group_reward_sum: float = 0.0
    group_reward_square_sum: float = 0.0
    trainable_samples: int = 0
    trainable_reward_sum: float = 0.0
    decomposed_counts: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in _CANDIDATE_DECOMPOSED_METRICS}
    )
    decomposed_sums: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in _CANDIDATE_DECOMPOSED_METRICS}
    )
    decomposed_square_sums: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in _CANDIDATE_DECOMPOSED_METRICS}
    )
    session_counts: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in _CANDIDATE_SESSION_COUNT_METRICS}
    )
    category_counts: dict[str, int] = field(
        default_factory=lambda: {
            "all_correct": 0,
            "all_wrong": 0,
            "mixed": 0,
            "constant_other": 0,
            "no_trainable": 0,
        }
    )

    def add(self, completed: _CompletedGroup, *, reward_key: str) -> None:
        """Atomically add one candidate group's optional quality telemetry."""

        samples = completed.samples
        trainable_rewards = [
            _extract_sample_reward(sample, reward_key)
            for sample in samples
            if _sample_has_trainable_tokens(sample)
        ]
        if not trainable_rewards:
            category = "no_trainable"
        else:
            unique_rewards = set(trainable_rewards)
            if len(unique_rewards) > 1:
                category = "mixed"
            elif trainable_rewards[0] == 0.0:
                category = "all_wrong"
            elif trainable_rewards[0] == 1.0:
                category = "all_correct"
            else:
                category = "constant_other"

        flat_rewards = [_extract_sample_reward(sample, reward_key) for sample in samples]
        quality = _polar_extra_metrics(samples, flat_rewards, reward_key)
        accounted = float(quality.get("polar/reward_accounted_sessions", 0.0))
        reward_mean = quality.get("polar/reward_mean")
        reward_std = float(quality.get("polar/reward_std", 0.0))
        parsed_reward_mean: float | None = None
        if accounted > 0.0 and reward_mean is not None:
            parsed_reward_mean = float(reward_mean)

        cancelled = float(quality.get("polar/early_stop/cancelled_sessions", 0.0))
        quality_eligible_sessions = max(
            0.0,
            float(completed.session_count) - cancelled,
        )
        session_counts = {
            name: float(quality.get(source, 0.0))
            for name, source in _CANDIDATE_SESSION_COUNT_METRICS.items()
        }
        decomposed: dict[str, tuple[float, float, float]] = {}
        for name, (count_key, mean_key, std_key) in _CANDIDATE_DECOMPOSED_METRICS.items():
            count = float(quality.get(count_key, 0.0))
            mean = quality.get(mean_key)
            if count <= 0.0 or mean is None:
                continue
            parsed_mean = float(mean)
            parsed_std = float(quality.get(std_key, 0.0))
            if not all(math.isfinite(value) for value in (count, parsed_mean, parsed_std)):
                raise ValueError(f"non-finite candidate {name} telemetry")
            if parsed_std < 0.0:
                raise ValueError(f"negative candidate {name} standard deviation")
            decomposed[name] = (count, parsed_mean, parsed_std)

        # Commit only after every optional extraction and conversion succeeds.
        # The caller can therefore count a telemetry error without retaining a
        # partially-observed candidate group.
        self.group_count += 1
        self.category_counts[category] += 1
        self.trainable_samples += len(trainable_rewards)
        self.trainable_reward_sum += sum(trainable_rewards)
        for name, count in session_counts.items():
            self.session_counts[name] += count
        for name, (count, mean, std) in decomposed.items():
            self.decomposed_counts[name] += count
            self.decomposed_sums[name] += count * mean
            self.decomposed_square_sums[name] += count * (std**2 + mean**2)
        if parsed_reward_mean is not None:
            self.accounted_sessions += accounted
            self.reward_sum += accounted * parsed_reward_mean
            self.reward_square_sum += accounted * (reward_std**2 + parsed_reward_mean**2)
            self.quality_group_count += 1
            self.group_reward_sum += parsed_reward_mean
            self.group_reward_square_sum += parsed_reward_mean**2
        self.early_stop_cancelled_sessions += cancelled
        self.quality_eligible_sessions += quality_eligible_sessions

    def record_group_error(self) -> None:
        """Account for a candidate whose optional telemetry could not be read."""

        self.group_count += 1
        self.telemetry_error_count += 1

    def record_export_error(self) -> None:
        self.telemetry_error_count += 1

    def as_metrics(self, *, accepted_group_count: int) -> dict[str, float]:
        if self.group_count <= 0:
            return {}
        group_count = float(self.group_count)
        metrics = {
            "polar/candidate/group_count": group_count,
            "polar/candidate/accepted_group_count": float(accepted_group_count),
            "polar/candidate/accept_fraction": float(accepted_group_count) / group_count,
            "polar/candidate/telemetry_error_count": float(self.telemetry_error_count),
            "polar/candidate/accounted_sessions": self.accounted_sessions,
            "polar/candidate/quality_eligible_sessions": self.quality_eligible_sessions,
            "polar/candidate/early_stop_cancelled_sessions": (self.early_stop_cancelled_sessions),
            "polar/candidate/trainable_samples": float(self.trainable_samples),
        }
        for name, count in self.session_counts.items():
            metrics[f"polar/candidate/{name}"] = count
        for name in _CANDIDATE_DECOMPOSED_METRICS:
            count = self.decomposed_counts[name]
            metrics[f"polar/candidate/{name}_accounted_sessions"] = count
            if count <= 0.0:
                continue
            mean = self.decomposed_sums[name] / count
            metrics[f"polar/candidate/{name}_mean"] = mean
            metrics[f"polar/candidate/{name}_std"] = (
                max(
                    0.0,
                    self.decomposed_square_sums[name] / count - mean**2,
                )
                ** 0.5
            )
            if name in {"total_cost", "cost_penalty_reward_delta"}:
                metrics[f"polar/candidate/{name}_total"] = self.decomposed_sums[name]
        attempted_sessions = self.session_counts["attempted_sessions"]
        total_sessions = attempted_sessions + self.early_stop_cancelled_sessions
        if total_sessions > 0.0:
            metrics["polar/candidate/attempted_session_fraction"] = (
                attempted_sessions / total_sessions
            )
        if attempted_sessions > 0.0:
            metrics["polar/candidate/trainable_session_fraction"] = (
                self.session_counts["trainable_sessions"] / attempted_sessions
            )
            metrics["polar/candidate/fully_masked_session_fraction"] = (
                self.session_counts["fully_masked_sessions"] / attempted_sessions
            )
            metrics["polar/candidate/rollout_success_rate"] = (
                self.session_counts["successful_sessions"] / attempted_sessions
            )
        for category, count in self.category_counts.items():
            metrics[f"polar/candidate/{category}_fraction"] = float(count) / group_count
            metrics[f"polar/candidate/{category}_count"] = float(count)
        if self.accounted_sessions > 0.0:
            reward_mean = self.reward_sum / self.accounted_sessions
            metrics["polar/candidate/reward_mean"] = reward_mean
            metrics["polar/candidate/reward_std"] = (
                max(0.0, self.reward_square_sum / self.accounted_sessions - reward_mean**2) ** 0.5
            )
        if self.quality_group_count > 0:
            quality_group_count = float(self.quality_group_count)
            group_reward_mean = self.group_reward_sum / quality_group_count
            metrics["polar/candidate/group_reward_mean"] = group_reward_mean
            metrics["polar/candidate/group_reward_std"] = (
                max(
                    0.0,
                    self.group_reward_square_sum / quality_group_count - group_reward_mean**2,
                )
                ** 0.5
            )
        if self.trainable_samples > 0:
            metrics["polar/candidate/trainable_reward_mean"] = (
                self.trainable_reward_sum / self.trainable_samples
            )
        if self.quality_eligible_sessions > 0.0:
            metrics["polar/candidate/quality_coverage_fraction"] = (
                self.accounted_sessions / self.quality_eligible_sessions
            )
        return metrics


def _candidate_quality_metrics_fail_open(
    accumulator: _CandidateQualityAccumulator,
    *,
    accepted_group_count: int,
) -> dict[str, float]:
    try:
        return accumulator.as_metrics(accepted_group_count=accepted_group_count)
    except Exception:
        accumulator.record_export_error()
        logger.warning(
            "Candidate-quality telemetry export failed; continuing without optional details",
            exc_info=True,
        )
        group_count = float(accumulator.group_count)
        accepted_count = float(accepted_group_count)
        accept_fraction = accepted_count / group_count if group_count > 0.0 else 0.0
        return {
            "polar/candidate/group_count": group_count,
            "polar/candidate/accepted_group_count": accepted_count,
            "polar/candidate/accept_fraction": accept_fraction,
            "polar/candidate/telemetry_error_count": float(accumulator.telemetry_error_count),
        }


def _decision_window_metrics(scheduler_metrics: dict[str, float]) -> dict[str, float]:
    consumed_key = "polar/reservations/consumed_delta"
    if consumed_key not in scheduler_metrics:
        return {}
    consumed = float(scheduler_metrics[consumed_key])
    accepted = float(scheduler_metrics.get("polar/reservations/consumed_accepted_delta", 0.0))
    return {
        "polar/decision_window/consumed_window_group_count": consumed,
        "polar/decision_window/accepted_group_count": accepted,
        "polar/decision_window/end_to_end_consumed_accept_fraction": (
            accepted / consumed if consumed > 0.0 else 0.0
        ),
    }


def _can_bootstrap_partial_recovery() -> bool:
    with _worker_lock:
        return _global_async_worker is None or not _global_async_worker.is_alive()


def _recovered_completed_group(
    *,
    record: dict[str, Any],
    group: list[Any],
    samples: list[Any],
    store: PartialRolloutStore,
) -> _CompletedGroup:
    return _CompletedGroup(
        group_id=int(record.get("scheduler_group_id", record["reservation_id"])),
        group=group,
        reservation_id=int(record["reservation_id"]),
        samples=samples,
        task_id=str(record.get("task_id", f"recovered-{record['reservation_id']}")),
        submitted_rollout_id=int(record["submitted_rollout_id"]),
        policy_version=int(record["policy_version"]),
        session_count=int(record.get("session_count", len(group))),
        # Monotonic timestamps cannot be compared across processes.  Preserve
        # correctness by omitting cross-process service-window telemetry.
        submitted_at=0.0,
        completed_at=0.0,
        service_time_seconds=0.0,
        sample_conversion_seconds=0.0,
        output_queue_wait_seconds=0.0,
        partial_store=store,
    )


def _validate_recovered_samples(
    *,
    record: dict[str, Any],
    samples: list[Any],
    rollout_id: int,
    max_off_policy_steps: int,
) -> None:
    reservation_id = int(record["reservation_id"])
    policy_version = int(record["policy_version"])
    staleness = int(rollout_id) - policy_version
    if staleness < 0 or staleness > int(max_off_policy_steps):
        raise PartialRolloutError(
            f"reservation {reservation_id} has unsafe recovered policy staleness {staleness}"
        )
    if not samples:
        raise PartialRolloutError(f"reservation {reservation_id} recovered an empty sample group")
    for sample in samples:
        if int(getattr(sample, "group_index", -1)) != reservation_id:
            raise PartialRolloutError(
                f"reservation {reservation_id} recovered a sample with group_index="
                f"{getattr(sample, 'group_index', None)}"
            )
    if not _has_trainable_tokens(samples):
        raise PartialRolloutError(f"reservation {reservation_id} recovered zero trainable tokens")

    if record["state"] != STATE_KEEP:
        return
    if int(record.get("accepted_rollout_id", -1)) != int(rollout_id):
        raise PartialRolloutError(f"reservation {reservation_id} KEEP belongs to another rollout")
    for sample in samples:
        polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar", {})
        train_meta = getattr(sample, "train_metadata", None) or {}
        if (
            int(polar_meta.get("accepted_rollout_id", -1)) != int(rollout_id)
            or int(polar_meta.get("policy_version", -1)) != policy_version
            or int(train_meta.get("policy_version", -1)) != policy_version
        ):
            raise PartialRolloutError(
                f"reservation {reservation_id} KEEP sample policy metadata mismatch"
            )


def _prepare_partial_recovery(
    args: Any,
    *,
    rollout_id: int,
    data_source: Any,
) -> _PartialRecoveryPlan:
    """Load, validate, and replay a partial WAL before starting the worker."""

    if not _can_bootstrap_partial_recovery():
        return _PartialRecoveryPlan()
    if not bool(getattr(args, "rollout_global_dataset", False)):
        return _PartialRecoveryPlan()

    config = resolve_polar_slime_config(args)
    store = maybe_open_partial_rollout_store(args, config, rollout_id)
    if store is None:
        return _PartialRecoveryPlan()
    rebuilder = getattr(data_source, "rebuild_partial_reservations", None)
    if not callable(rebuilder):
        logger.warning(
            "Partial rollout recovery disabled: data source does not expose "
            "rebuild_partial_reservations"
        )
        return _PartialRecoveryPlan()

    try:
        records = store.load_records()
        committed_record_ids = {
            int(record["reservation_id"])
            for record in records
            if int(record["reservation_id"]) in store.committed_reservation_ids
        }
        for reservation_id in sorted(committed_record_ids):
            store.record_resume_duplicate(reservation_id)
        if committed_record_ids:
            records = store.load_records()
        recovered_owned = sum(record["state"] != STATE_DROP for record in records)
        if recovered_owned > config.max_concurrency:
            raise PartialRolloutError(
                f"partial WAL owns {recovered_owned} groups, exceeding async window "
                f"{config.max_concurrency}"
            )
        Sample = _load_sample_type()
        preloaded_samples: dict[int, list[Any]] = {}
        for record in records:
            reuse_samples = record["state"] in (STATE_RESULT_READY, STATE_KEEP) or (
                record["state"] == STATE_DROP
                and record.get("drop_outcome") == "dynamic_filter"
                and record.get("sample_blob") is not None
            )
            if reuse_samples:
                samples = [
                    Sample.from_dict(sample_dict)
                    for sample_dict in store.load_sample_dicts(record)
                ]
                if record["state"] == STATE_KEEP:
                    reservation_id = int(record["reservation_id"])
                    policy_version = int(record["policy_version"])
                    for sample in samples:
                        polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar", {})
                        train_meta = getattr(sample, "train_metadata", None) or {}
                        for metadata, field_name, expected in (
                            (
                                polar_meta,
                                "accepted_rollout_id",
                                int(record["accepted_rollout_id"]),
                            ),
                            (polar_meta, "policy_version", policy_version),
                            (train_meta, "policy_version", policy_version),
                        ):
                            present = metadata.get(field_name)
                            if present is not None and int(present) != expected:
                                raise PartialRolloutError(
                                    f"reservation {reservation_id} KEEP blob has "
                                    f"conflicting {field_name}={present}"
                                )
                    _annotate_accepted_samples(
                        samples,
                        accepted_rollout_id=int(record["accepted_rollout_id"]),
                        staleness=int(rollout_id) - policy_version,
                        policy_version=policy_version,
                        scheduler_group_id=int(record.get("scheduler_group_id", reservation_id)),
                    )
                _validate_recovered_samples(
                    record=record,
                    samples=samples,
                    rollout_id=rollout_id,
                    max_off_policy_steps=config.max_off_policy_steps,
                )
                preloaded_samples[int(record["reservation_id"])] = samples
        rebuilt = rebuilder(records)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        try:
            store.quarantine(reason)
        except Exception:
            logger.exception("Failed to quarantine invalid partial rollout WAL")
            raise
        # Continue with the existing at-least-once replay path and a fresh WAL.
        return _PartialRecoveryPlan(
            store=PartialRolloutStore(
                store.directory,
                store.header,
                committed_reservation_ids=store.committed_reservation_ids,
            )
        )

    plan = _PartialRecoveryPlan(store=store)
    drop_records: list[dict[str, Any]] = []
    for record, group in rebuilt:
        state = record["state"]
        reservation_id = int(record["reservation_id"])
        if state == STATE_PREPARED:
            plan.deferred.append(
                _DeferredGroup(
                    group=group,
                    reservation_id=reservation_id,
                    submitted_rollout_id=int(record["submitted_rollout_id"]),
                    policy_version=int(record["policy_version"]),
                    partial_store=store,
                )
            )
            continue

        reuse_samples = state in (STATE_RESULT_READY, STATE_KEEP) or (
            state == STATE_DROP
            and record.get("drop_outcome") == "dynamic_filter"
            and record.get("sample_blob") is not None
        )
        if reuse_samples:
            completed = _recovered_completed_group(
                record=record,
                group=group,
                samples=preloaded_samples[reservation_id],
                store=store,
            )
        else:
            completed = None

        if state == STATE_RESULT_READY:
            assert completed is not None
            plan.result_ready.append(completed)
        elif state == STATE_KEEP:
            assert completed is not None
            plan.kept.append(completed)
        elif state == STATE_DROP:
            drop_records.append(record)
            if completed is not None:
                plan.candidate_only.append(completed)
            if record.get("drop_outcome") == "dynamic_filter":
                reason = record.get("drop_reason")
                if reason:
                    key = f"rollout/dynamic_filter/drop_{reason}"
                    plan.dynamic_filter_metrics[key] = (
                        plan.dynamic_filter_metrics.get(key, 0.0) + 1.0
                    )
        else:
            raise PartialRolloutError(f"unsupported recovered state {state}")

    marker_many = getattr(data_source, "mark_consumed_many", None)
    marker_one = getattr(data_source, "mark_consumed", None)
    plan.dropped_count = len(drop_records)
    plan.resume_duplicate_count = sum(
        record.get("drop_outcome") == "resume_duplicate" for record in drop_records
    )
    for record in drop_records:
        reservation_id = int(record["reservation_id"])
        outcome = str(record.get("drop_outcome") or "recovered_drop")
        if callable(marker_one):
            metrics = marker_one(reservation_id, outcome=outcome)
        elif callable(marker_many):
            metrics = marker_many([reservation_id], outcome=outcome)
        else:
            raise PartialRolloutError(
                "data source cannot consume a reconstructed DROP reservation"
            )
        if isinstance(metrics, dict):
            plan.reservation_metrics.update(metrics)

    logger.info(
        "Recovered partial rollout %s: keep=%d result_ready=%d prepared=%d "
        "drop=%d resume_duplicate=%d",
        rollout_id,
        len(plan.kept),
        len(plan.result_ready),
        len(plan.deferred),
        len(drop_records),
        plan.resume_duplicate_count,
    )
    return plan


def generate_rollout_polar_async(
    args: Any, rollout_id: int, data_source: Any, evaluation: bool = False
) -> Any:
    """Slime-compatible async rollout entrypoint.

    Training runs are served by a persistent background worker that pulls
    from ``data_source`` and drains completed groups on each call.
    Evaluation runs are served by a one-shot submit+poll batch over the
    same async HTTP surface.
    """
    if evaluation:
        return asyncio.run(_run_eval_rollout(args, rollout_id, data_source))

    dynamic_filter = _load_training_dynamic_filter(args)
    partial_recovery = _prepare_partial_recovery(
        args,
        rollout_id=rollout_id,
        data_source=data_source,
    )
    if partial_recovery.store is None:
        async_worker = get_global_async_worker(args, data_source)
    else:
        async_worker = get_global_async_worker(
            args,
            data_source,
            partial_recovery,
            rollout_id,
        )
    candidate_pool_gate_config = _candidate_pool_health_gate_config(async_worker.config)
    candidate_pool_health = (
        _CandidatePoolHealthAccumulator(candidate_pool_gate_config.candidate_aliases)
        if candidate_pool_gate_config is not None
        else None
    )
    async_worker.set_rollout_context(rollout_id)
    target = int(getattr(args, "rollout_batch_size", 1))
    if len(partial_recovery.kept) > target:
        raise PartialRolloutError(
            f"recovered {len(partial_recovery.kept)} KEEP groups for target {target}"
        )

    data: list[list[Any]] = [completed.samples for completed in partial_recovery.kept]
    accepted_completions: list[_CompletedGroup] = list(partial_recovery.kept)

    def observe_candidate_health(completed: _CompletedGroup) -> None:
        if candidate_pool_health is None or candidate_pool_gate_config is None:
            return
        candidate_pool_health.add(completed)
        # Enforce incrementally, before any DROP/KEEP record is consumed or a
        # replacement request is admitted.  Otherwise a provider outage whose
        # groups all fail the low-complete or dynamic-sampling filters can loop
        # forever without ever reaching a full accepted batch.
        _enforce_candidate_pool_health_gate(
            args,
            candidate_pool_health,
            candidate_pool_gate_config,
            rollout_id=rollout_id,
            accepted_group_count=len(data),
            partial_store=partial_recovery.store,
        )

    def drain_candidate_health_observations() -> None:
        drainer = getattr(async_worker, "drain_health_observations", None)
        if not callable(drainer):
            return
        for observation in drainer():
            observe_candidate_health(observation)

    candidate_quality = _CandidateQualityAccumulator()
    for completed in (*partial_recovery.candidate_only, *partial_recovery.kept):
        observe_candidate_health(completed)
        try:
            candidate_quality.add(completed, reward_key=async_worker.config.reward_key)
        except Exception:
            candidate_quality.record_group_error()
            logger.warning(
                "Recovered candidate-quality telemetry failed for Polar group %s",
                completed.group_id,
                exc_info=True,
            )
    async_worker.request_groups(target - len(partial_recovery.kept))
    dynamic_filter_metrics: dict[str, float] = dict(partial_recovery.dynamic_filter_metrics)
    dynamic_filter_reservation_metrics: dict[str, float] = dict(
        partial_recovery.reservation_metrics
    )
    partial_recovery_metrics: dict[str, float] = {}
    recovered_group_count = (
        len(partial_recovery.kept)
        + len(partial_recovery.result_ready)
        + len(partial_recovery.deferred)
        + partial_recovery.dropped_count
    )
    if recovered_group_count:
        partial_recovery_metrics = {
            "polar/partial_recovery/replayed_group_count": float(recovered_group_count),
            "polar/partial_recovery/restored_keep_group_count": float(len(partial_recovery.kept)),
            "polar/partial_recovery/restored_result_ready_group_count": float(
                len(partial_recovery.result_ready)
            ),
            "polar/partial_recovery/resubmitted_prepared_group_count": float(
                len(partial_recovery.deferred)
            ),
            "polar/partial_recovery/restored_drop_group_count": float(
                partial_recovery.dropped_count
            ),
            "polar/partial_recovery/resume_duplicate_group_count": float(
                partial_recovery.resume_duplicate_count
            ),
        }
    start = time.monotonic()
    last_progress = start

    while len(data) < target:
        if _current_ray_task_is_canceled():
            from ray.exceptions import TaskCancelledError

            logger.info("Stopping Polar rollout worker after Ray task cancellation")
            stop_global_worker()
            raise TaskCancelledError(error_message="Polar rollout generation was cancelled")
        made_progress = False
        # Permanent worker-side rejections (for example, a whole provider
        # cohort failing the trainable-completion floor) never enter the normal
        # completed queue.  Inspect them before raise_if_failed and before
        # asking for any replacement work.
        drain_candidate_health_observations()
        completed_groups = async_worker.drain_completed(
            max_groups=target - len(data),
            rollout_id=rollout_id,
        )
        # drain_completed can itself reject stale groups and publish their
        # pre-filter health evidence.
        drain_candidate_health_observations()
        replacement_groups = 0
        for completed in completed_groups:
            observe_candidate_health(completed)
            try:
                candidate_quality.add(
                    completed,
                    reward_key=async_worker.config.reward_key,
                )
            except Exception:
                candidate_quality.record_group_error()
                logger.warning(
                    "Candidate-quality telemetry failed for Polar group %s task=%s; "
                    "continuing with filtering and reservation handling",
                    completed.group_id,
                    completed.task_id,
                    exc_info=True,
                )
            if dynamic_filter is not None:
                filter_output = _call_training_dynamic_filter(
                    dynamic_filter,
                    args,
                    completed.samples,
                )
                if not filter_output.keep:
                    reason = filter_output.reason
                    if reason:
                        metric = f"rollout/dynamic_filter/drop_{reason}"
                        dynamic_filter_metrics[metric] = (
                            dynamic_filter_metrics.get(metric, 0.0) + 1.0
                        )
                    dynamic_filter_reservation_metrics.update(
                        async_worker.mark_dynamic_filter_drop(
                            completed,
                            reason=reason,
                        )
                    )
                    replacement_groups += 1
                    made_progress = True
                    continue
            if partial_recovery.store is not None:
                if completed.partial_store is not partial_recovery.store:
                    raise PartialRolloutError(
                        "completed group is not owned by the active partial WAL"
                    )
                partial_recovery.store.record_keep(
                    completed,
                    accepted_rollout_id=rollout_id,
                )
            data.append(completed.samples)
            accepted_completions.append(completed)
            made_progress = True
        if replacement_groups:
            # drain_completed has already satisfied scheduler demand for every
            # returned group. Restore exactly the filtered demand so fully-async
            # admission creates fresh prompt reservations without growing the
            # bounded ownership window.
            async_worker.request_groups(replacement_groups)

        now = time.monotonic()
        if made_progress:
            last_progress = now
        elif now - last_progress > 60:
            logger.warning(
                "No progress for 60s. Queue=%d, accepted=%d/%d",
                async_worker.queue_size(),
                len(data),
                target,
            )
            last_progress = now

        if len(data) < target:
            time.sleep(0.05)

    elapsed = time.monotonic() - start
    logger.info(
        "Async rollout collected %d groups in %.1fs (queue=%d)",
        len(data),
        elapsed,
        async_worker.queue_size(),
    )
    drain_candidate_health_observations()
    candidate_pool_health_report: dict[str, Any] | None = None
    if candidate_pool_health is not None and candidate_pool_gate_config is not None:
        candidate_pool_health_report = _enforce_candidate_pool_health_gate(
            args,
            candidate_pool_health,
            candidate_pool_gate_config,
            rollout_id=rollout_id,
            accepted_group_count=len(data),
            partial_store=partial_recovery.store,
        )
    if partial_recovery.store is not None:
        partial_recovery.store.mark_ready(
            completed.reservation_id for completed in accepted_completions
        )

    RolloutFnTrainOutput = _load_rollout_train_output_type()
    flat = [s for g in data for s in g]
    rewards = [_extract_sample_reward(s, async_worker.config.reward_key) for s in flat]
    metrics: dict[str, Any] = dict(dynamic_filter_metrics)
    metrics.update(dynamic_filter_reservation_metrics)
    metrics.update(partial_recovery_metrics)
    if candidate_pool_health_report is not None:
        metrics.update(_candidate_pool_health_metrics(candidate_pool_health_report))
    metrics.update(
        _candidate_quality_metrics_fail_open(
            candidate_quality,
            accepted_group_count=len(data),
        )
    )
    accepted_quality = _polar_extra_metrics(flat, rewards, async_worker.config.reward_key)
    metrics.update(accepted_quality)
    if "polar/spilot_router/session_count" in accepted_quality:
        metrics["polar/spilot_router/rollout_step"] = float(rollout_id)
    metrics["polar/accepted/group_count"] = float(len(data))
    for source, suffix in (
        ("polar/reward_mean", "reward_mean"),
        ("polar/reward_std", "reward_std"),
        ("polar/reward_accounted_sessions", "accounted_sessions"),
        ("polar/reward_mean_completed", "reward_mean_completed"),
        (
            "polar/spilot_router/accuracy_outcome_accounted_session_count",
            "accuracy_outcome_accounted_sessions",
        ),
        ("polar/spilot_router/accuracy_outcome_mean", "accuracy_outcome_mean"),
        (
            "polar/spilot_router/total_cost_accounted_session_count",
            "total_cost_accounted_sessions",
        ),
        ("polar/spilot_router/total_cost_mean", "total_cost_mean"),
        (
            "polar/spilot_router/cost_penalty_fraction_mean",
            "cost_penalty_fraction_mean",
        ),
        (
            "polar/spilot_router/cost_penalty_reward_delta_mean",
            "cost_penalty_reward_delta_mean",
        ),
        (
            "polar/spilot_router/cost_adjusted_reward_mean",
            "cost_adjusted_reward_mean",
        ),
        (
            "polar/spilot_router/total_latency_seconds_mean",
            "total_latency_seconds_mean",
        ),
        (
            "polar/spilot_router/latency_penalty_fraction_mean",
            "latency_penalty_fraction_mean",
        ),
        (
            "polar/spilot_router/latency_penalty_reward_delta_mean",
            "latency_penalty_reward_delta_mean",
        ),
    ):
        if source in accepted_quality:
            metrics[f"polar/accepted/{suffix}"] = accepted_quality[source]
    # Slime's built-in rollout/raw_reward is intentionally trace/sample
    # weighted and includes zero-gradient placeholders. Publish the
    # exchangeable one-session-one-vote quality next to it so dashboards do
    # not mistake a diagnostic transport field for Router outcome quality.
    if "polar/reward_mean" in accepted_quality:
        metrics["rollout/session_reward_mean"] = accepted_quality["polar/reward_mean"]
    if "polar/reward_accounted_sessions" in accepted_quality:
        metrics["rollout/session_reward_accounted_sessions"] = accepted_quality[
            "polar/reward_accounted_sessions"
        ]
    for source, target in (
        ("polar/spilot_router/accuracy_outcome_mean", "rollout/accuracy_outcome_mean"),
        ("polar/spilot_router/total_cost_mean", "rollout/total_cost_mean"),
        (
            "polar/spilot_router/cost_penalty_fraction_mean",
            "rollout/cost_penalty_fraction_mean",
        ),
        (
            "polar/spilot_router/cost_penalty_reward_delta_mean",
            "rollout/cost_penalty_reward_delta_mean",
        ),
        (
            "polar/spilot_router/cost_adjusted_reward_mean",
            "rollout/cost_adjusted_reward_mean",
        ),
        (
            "polar/spilot_router/total_latency_seconds_mean",
            "rollout/total_latency_seconds_mean",
        ),
        (
            "polar/spilot_router/latency_penalty_fraction_mean",
            "rollout/latency_penalty_fraction_mean",
        ),
        (
            "polar/spilot_router/latency_penalty_reward_delta_mean",
            "rollout/latency_penalty_reward_delta_mean",
        ),
    ):
        if source in accepted_quality:
            metrics[target] = accepted_quality[source]
    metrics.update(_completed_service_metrics(accepted_completions))
    metrics["timing/pipeline_ms/rollout_collect"] = elapsed * 1000.0
    output = RolloutFnTrainOutput(samples=data, metrics=metrics)
    # Commit accepted reservations only after a complete batch has been built.
    # A cancellation after a partial drain must leave every partial group
    # outstanding so the checkpoint frontier replays it on resume. Ray has no
    # atomic "return-and-ack" handshake for actor cancellation; train_async's
    # checkpoint-before-prefetch ordering is the durable at-least-once guard
    # for the tiny race between this final check and the acknowledgement.
    if _current_ray_task_is_canceled():
        from ray.exceptions import TaskCancelledError

        logger.info("Stopping Polar rollout worker before committing a cancelled batch")
        stop_global_worker()
        raise TaskCancelledError(error_message="Polar rollout generation was cancelled")
    post_commit_metrics = async_worker._consume_reservations(
        [completed.reservation_id for completed in accepted_completions],
        outcome="accepted",
    )
    if partial_recovery.store is not None:
        async_worker.release_recovered_holds()
    # Snapshot exactly once per delivered rollout, after reservation commit,
    # so *_delta means "during this rollout" and lifetime counters carry an
    # explicit worker-local scope across Slurm restarts.
    scheduler_metrics = async_worker.snapshot_metrics()
    metrics.update(scheduler_metrics)
    metrics.update(_decision_window_metrics(scheduler_metrics))
    metrics.update(post_commit_metrics)
    return output


def _configured_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s=%r; using %d", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %d", name, raw, default)
        return default
    return value


def _sample_reward_for_example(sample: Any) -> float:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        value = reward.get("score", next(iter(reward.values()), 0.0))
    else:
        value = reward
    return _finite_float_or_zero(value)


def _trajectory_example_payload(session_id: str, samples: list[Any]) -> dict[str, Any]:
    ordered = sorted(
        samples,
        key=lambda sample: int(
            ((getattr(sample, "metadata", {}) or {}).get("polar") or {}).get("trace_index", 0) or 0
        ),
    )
    trace_rewards = [_sample_reward_for_example(sample) for sample in ordered]
    traces: list[dict[str, Any]] = []
    for sample in ordered:
        polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar") or {}
        trace_debug = polar_meta.get("trace_debug") or {}
        status = getattr(sample, "status", None)
        status_value = getattr(status, "value", status)
        traces.append(
            {
                "trace_index": polar_meta.get("trace_index"),
                "finish_reason": trace_debug.get("finish_reason"),
                "response_length": int(getattr(sample, "response_length", 0) or 0),
                "status": str(status_value) if status_value is not None else None,
                "remove_sample": bool(getattr(sample, "remove_sample", False)),
                "training_filter": polar_meta.get("training_filter"),
                "token_clipping": polar_meta.get("token_clipping"),
                "prompt_messages": (
                    sample.prompt if isinstance(getattr(sample, "prompt", None), list) else []
                ),
                # This is deliberately unabridged. W&B receives the same JSON
                # file as the local log so a full multi-turn trajectory can be
                # inspected without placing 65k-token text in scalar history.
                "response_messages": trace_debug.get("response_messages") or [],
            }
        )

    first = ordered[0]
    first_meta = (getattr(first, "metadata", {}) or {}).get("polar") or {}
    total_tokens = sum(int(getattr(sample, "response_length", 0) or 0) for sample in ordered)
    return {
        "session_id": session_id,
        "task_id": first_meta.get("task_id"),
        "node_id": first_meta.get("node_id"),
        "session_status": first_meta.get("session_status"),
        "result_error": first_meta.get("result_error"),
        "trajectory_status": first_meta.get("trajectory_status"),
        "trajectory_error": first_meta.get("trajectory_error"),
        "total_response_tokens": total_tokens,
        "session_reward_mean": (sum(trace_rewards) / len(trace_rewards) if trace_rewards else 0.0),
        "trace_rewards": trace_rewards,
        "num_traces": len(traces),
        "traces": traces,
    }


def _select_trajectory_examples(
    by_session: dict[str, list[Any]], count: int
) -> list[dict[str, Any]]:
    payloads = [
        _trajectory_example_payload(session_id, samples)
        for session_id, samples in by_session.items()
        if samples
    ]
    payloads = [payload for payload in payloads if payload["total_response_tokens"] > 0]
    if len(payloads) <= count:
        selected = sorted(
            payloads,
            key=lambda payload: (
                -payload["session_reward_mean"],
                -payload["total_response_tokens"],
                payload["session_id"],
            ),
        )
        return [
            {**payload, "selection_reason": "available_by_reward_and_length"}
            for payload in selected
        ]

    # Prefer one high-reward and one low-reward long trajectory. This exposes
    # both successful and unsuccessful behavior; if every reward is identical,
    # fall back to the longest distinct sessions.
    high = max(
        payloads,
        key=lambda payload: (
            payload["session_reward_mean"],
            payload["total_response_tokens"],
            payload["session_id"],
        ),
    )
    low = max(
        payloads,
        key=lambda payload: (
            -payload["session_reward_mean"],
            payload["total_response_tokens"],
            payload["session_id"],
        ),
    )
    selected = [{**high, "selection_reason": "highest_reward_longest"}]
    if low["session_id"] != high["session_id"]:
        selected.append({**low, "selection_reason": "lowest_reward_longest"})
    for payload in sorted(
        payloads,
        key=lambda item: (-item["total_response_tokens"], item["session_id"]),
    ):
        if len(selected) >= count:
            break
        if all(payload["session_id"] != item["session_id"] for item in selected):
            selected.append({**payload, "selection_reason": "longest_remaining"})
    return selected[:count]


def _trajectory_examples_output_dir() -> Path | None:
    configured = os.environ.get("POLAR_ROLLOUT_EXAMPLES_DIR", "").strip()
    if not configured:
        rollout_dir = os.environ.get("POLAR_ROLLOUT_SAVE_DIR", "").strip()
        if rollout_dir:
            configured = str(Path(rollout_dir) / "trajectory_examples")
    if not configured:
        return None
    output_dir = Path(configured)
    if not output_dir.is_absolute():
        logger.warning("POLAR_ROLLOUT_EXAMPLES_DIR must be absolute: %s", output_dir)
        return None
    return output_dir


_AUTHORIZATION_SECRET_RE = re.compile(
    r"(?i)(\bauthorization\s*:\s*(?:bearer|basic)\s+)[^\s,;}\]]+"
)
_NAMED_SECRET_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret|cookie)"
    r"\b\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}\]]+)"
)
_ENV_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?im)(\b(?:[A-Z][A-Z0-9]*_)*(?:API_KEY|TOKEN|SECRET_ACCESS_KEY|SECRET|"
    r"PASSWORD|PASSWD|COOKIE|AUTHORIZATION)\b\s*=\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}\]]+)"
)
_COMMON_SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_SECRET_MAPPING_KEYS = frozenset(
    {
        "authorization",
        "proxy_authorization",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "auth_token",
        "refresh_token",
        "id_token",
        "password",
        "passwd",
        "cookie",
        "set_cookie",
        "secret",
        "client_secret",
        "private_key",
        "credentials",
        "secret_access_key",
    }
)
_SECRET_MAPPING_KEY_SUFFIXES = tuple(
    f"_{key}" for key in _SECRET_MAPPING_KEYS if key not in {"authorization", "apikey", "secret"}
)


def _is_secret_mapping_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return normalized in _SECRET_MAPPING_KEYS or normalized.endswith(_SECRET_MAPPING_KEY_SUFFIXES)


def _redact_trajectory_value(value: Any) -> tuple[Any, int]:
    """Recursively remove credentials without truncating trajectory content."""

    if isinstance(value, str):
        redacted = value
        count = 0
        for pattern, replacement in (
            (_PRIVATE_KEY_RE, "<redacted-private-key>"),
            (_AUTHORIZATION_SECRET_RE, r"\1<redacted>"),
            (_ENV_SECRET_ASSIGNMENT_RE, r"\1<redacted>"),
            (_NAMED_SECRET_RE, r"\1<redacted>"),
            (_COMMON_SECRET_RE, "<redacted-secret>"),
        ):
            redacted, replacements = pattern.subn(replacement, redacted)
            count += replacements
        return redacted, count
    if isinstance(value, dict):
        output: dict[Any, Any] = {}
        count = 0
        for key, item in value.items():
            if isinstance(key, str) and _is_secret_mapping_key(key):
                output[key] = "<redacted>"
                count += 1
                continue
            output_item, replacements = _redact_trajectory_value(item)
            output[key] = output_item
            count += replacements
        return output, count
    if isinstance(value, list):
        output_list: list[Any] = []
        count = 0
        for item in value:
            output_item, replacements = _redact_trajectory_value(item)
            output_list.append(output_item)
            count += replacements
        return output_list, count
    if isinstance(value, tuple):
        output_tuple: list[Any] = []
        count = 0
        for item in value:
            output_item, replacements = _redact_trajectory_value(item)
            output_tuple.append(output_item)
            count += replacements
        return output_tuple, count
    return value, 0


def _telemetry_error_detail(exc: Exception) -> str:
    detail = " ".join(str(exc).splitlines())
    redacted, _ = _redact_trajectory_value(detail)
    return str(redacted)[:300]


def _flatten_trajectory_samples(samples: Any) -> list[Any]:
    if isinstance(samples, list | tuple):
        flattened: list[Any] = []
        for item in samples:
            flattened.extend(_flatten_trajectory_samples(item))
        return flattened
    return [samples]


def _sample_rollout_key(sample: Any, position: int) -> Any:
    key = getattr(sample, "rollout_id", None)
    if key is None:
        key = getattr(sample, "index", None)
    if key is None:
        key = _sample_session_id(sample)
    if key is None:
        return ("position", position)
    try:
        hash(key)
    except TypeError:
        return ("value", repr(key))
    return ("value", key)


def _train_step_sample_slices(
    args: Any,
    rollout_id: int,
    samples: list[Any],
    interval: int,
) -> list[tuple[int, list[Any]]]:
    """Return exact optimizer-step subsets that cross the configured cadence."""

    rollout_batch_size = int(getattr(args, "rollout_batch_size"))
    samples_per_prompt = int(getattr(args, "n_samples_per_prompt"))
    global_batch_size = int(getattr(args, "global_batch_size"))
    product = rollout_batch_size * samples_per_prompt
    if min(rollout_batch_size, samples_per_prompt, global_batch_size) <= 0:
        raise ValueError("rollout/global batch sizes must be positive")
    if product % global_batch_size:
        raise ValueError(
            "rollout_batch_size*n_samples_per_prompt must be divisible by global_batch_size"
        )
    steps_per_rollout = product // global_batch_size
    configured_steps = getattr(args, "num_steps_per_rollout", None)
    if configured_steps is not None and int(configured_steps) != steps_per_rollout:
        raise ValueError(
            f"num_steps_per_rollout={configured_steps} does not match {steps_per_rollout}"
        )

    rollout_order: list[Any] = []
    samples_by_rollout: dict[Any, list[Any]] = {}
    for position, sample in enumerate(samples):
        key = _sample_rollout_key(sample, position)
        if key not in samples_by_rollout:
            rollout_order.append(key)
            samples_by_rollout[key] = []
        samples_by_rollout[key].append(sample)
    if len(rollout_order) < product:
        raise ValueError(
            f"received {len(rollout_order)} unique trajectories, expected at least {product}"
        )

    selected_steps: list[tuple[int, list[Any]]] = []
    first_train_step = int(rollout_id) * steps_per_rollout
    for local_step in range(steps_per_rollout):
        train_step = first_train_step + local_step
        # train/step is zero-based; cadence 10 means the first periodic sample
        # is attached to the completed optimizer step 10, then 20, 30, ... .
        if train_step <= 0 or train_step % interval:
            continue
        begin = local_step * global_batch_size
        end = begin + global_batch_size
        step_keys = rollout_order[begin:end]
        step_samples = [sample for key in step_keys for sample in samples_by_rollout[key]]
        selected_steps.append((train_step, step_samples))
    return selected_steps


def _trajectory_example_records(
    train_step: int,
    samples: list[Any],
    count: int,
) -> list[dict[str, Any]]:
    by_session: dict[str, list[Any]] = {}
    for position, sample in enumerate(samples):
        session_id = _sample_session_id(sample) or f"unknown-{position}"
        by_session.setdefault(session_id, []).append(sample)
    selected = _select_trajectory_examples(by_session, count)
    records: list[dict[str, Any]] = []
    for selection_rank, example in enumerate(selected):
        raw_session_id = str(example.get("session_id") or "unknown")
        example_to_redact = dict(example)
        # Session ids such as ``sk-polar-*`` resemble API credentials but are
        # only internal correlation handles. Keep them in the mode-0600 local
        # log and precompute a distinct non-reversible value for W&B.
        example_to_redact["session_id"] = "<internal-session-id>"
        redacted, redaction_count = _redact_trajectory_value(example_to_redact)
        redacted["session_id"] = raw_session_id
        records.append(
            {
                "schema_version": 2,
                "train_step": int(train_step),
                "selection_rank": selection_rank,
                "redaction_count": redaction_count,
                "session_hash": hashlib.sha256(raw_session_id.encode()).hexdigest()[:16],
                "trajectory": redacted,
            }
        )
    return records


def _persist_trajectory_example_records(
    train_step: int,
    records: list[dict[str, Any]],
) -> Path:
    output_dir = _trajectory_examples_output_dir()
    if output_dir is None:
        raise ValueError(
            "set POLAR_ROLLOUT_EXAMPLES_DIR or POLAR_ROLLOUT_SAVE_DIR to an absolute path"
        )
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    output_path = output_dir / f"trajectory_examples_step_{train_step:06d}.jsonl"
    temporary_path = output_dir / (f".{output_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, default=str))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o600)
        temporary_path.replace(output_path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
    return output_path


def _log_trajectory_examples_to_wandb(
    args: Any,
    train_step: int,
    records: list[dict[str, Any]],
) -> None:
    if os.environ.get("POLAR_ROLLOUT_EXAMPLES_WANDB", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    } or not bool(getattr(args, "use_wandb", False)):
        return

    import wandb
    from slime.utils import wandb_utils

    if getattr(wandb, "run", None) is None:
        return
    rows: list[list[Any]] = []
    for record in records:
        trajectory = dict(record["trajectory"])
        session_hash = str(record["session_hash"])
        trajectory["session_id"] = session_hash
        rows.append(
            [
                int(train_step),
                session_hash,
                trajectory.get("task_id"),
                trajectory.get("session_reward_mean"),
                trajectory.get("total_response_tokens"),
                trajectory.get("selection_reason"),
                json.dumps(trajectory, ensure_ascii=False, indent=2, default=str),
            ]
        )
    table = wandb.Table(
        columns=[
            "train_step",
            "session_hash",
            "task_id",
            "reward",
            "response_tokens",
            "selection_reason",
            "full_trajectory_json",
        ],
        data=rows,
        log_mode="IMMUTABLE",
    )
    metrics = {
        "train/step": int(train_step),
        "examples/rollout_trajectories": table,
    }
    wandb_utils.define_logged_metric_axes(metrics, step_metric="train/step")
    wandb.log(metrics)


def log_rollout_trajectory_examples(
    rollout_id: int,
    args: Any,
    samples: Any,
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Slime post-train hook for periodic full trajectory examples.

    Slime invokes this from ``commit_rollout_metrics`` only after every actor
    optimizer step in the rollout batch succeeds. Telemetry is deliberately
    fail-open: local or W&B failures can never fail or cancel training.
    """

    del rollout_extra_metrics, rollout_time
    if bool(getattr(args, "debug_rollout_only", False)):
        # Slime invokes custom rollout log hooks before any actor train in this
        # diagnostics-only mode; do not label such generations as committed
        # optimizer-step examples.
        return False
    try:
        interval = _configured_positive_int(
            "POLAR_ROLLOUT_EXAMPLE_INTERVAL", _TRAJECTORY_EXAMPLE_INTERVAL
        )
        count = _configured_positive_int("POLAR_ROLLOUT_EXAMPLE_COUNT", _TRAJECTORY_EXAMPLE_COUNT)
        flat_samples = _flatten_trajectory_samples(samples)
        step_slices = _train_step_sample_slices(
            args,
            rollout_id,
            flat_samples,
            interval,
        )
    except Exception as exc:
        logger.warning(
            "Skipping trajectory-example telemetry for rollout %s (%s): %.300s",
            rollout_id,
            type(exc).__name__,
            _telemetry_error_detail(exc),
        )
        return False

    for train_step, step_samples in step_slices:
        try:
            records = _trajectory_example_records(train_step, step_samples, count)
        except Exception as exc:
            logger.warning(
                "Failed to prepare trajectory examples at train step %d (%s): %.300s",
                train_step,
                type(exc).__name__,
                _telemetry_error_detail(exc),
            )
            continue
        if not records:
            continue

        try:
            output_path = _persist_trajectory_example_records(train_step, records)
        except Exception as exc:
            logger.warning(
                "Failed to persist trajectory examples at train step %d (%s): %.300s",
                train_step,
                type(exc).__name__,
                _telemetry_error_detail(exc),
            )
        else:
            summary = ",".join(
                f"{record['session_hash']}@"
                f"r={record['trajectory']['session_reward_mean']:.3g}/"
                f"t={record['trajectory']['total_response_tokens']}"
                for record in records
            )
            logger.info(
                "Saved %d full trajectory example(s) for train step %d to %s [%s]",
                len(records),
                train_step,
                output_path,
                summary,
            )

        try:
            _log_trajectory_examples_to_wandb(args, train_step, records)
        except Exception as exc:
            logger.warning(
                "Failed to log full-trajectory W&B Table at train step %d (%s): %.300s",
                train_step,
                type(exc).__name__,
                _telemetry_error_detail(exc),
            )
    return False


def _group_index_for(group: list[Any]) -> int:
    if group and getattr(group[0], "group_index", None) is not None:
        return int(group[0].group_index)
    return -1


def _extract_sample_reward(sample: Any, reward_key: str) -> float:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        if reward_key in reward:
            return _finite_float_or_zero(reward[reward_key])
        if "score" in reward:
            return _finite_float_or_zero(reward["score"])
    if isinstance(reward, (int, float)):
        return _finite_float_or_zero(reward)
    return 0.0


def _finite_float_or_zero(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _is_trainable_agent_timeout_sample(sample: Any) -> bool:
    if bool(getattr(sample, "remove_sample", False)):
        return False
    status = getattr(sample, "status", None)
    status_name = getattr(status, "name", None) or str(status).rsplit(".", 1)[-1]
    if status_name.upper() in {"FAILED", "ABORTED"}:
        return False
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is not None and not any(int(value) != 0 for value in loss_mask):
        return False
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    polar_meta = metadata.get("polar")
    if not isinstance(polar_meta, dict):
        return False
    training_filter = polar_meta.get("training_filter")
    return (
        isinstance(training_filter, dict)
        and training_filter.get("reason") == "agent_timeout"
        and training_filter.get("trainable") is True
        and training_filter.get("masked") is not True
    )


def _effective_trainable_reward(sample: Any, reward_key: str) -> float:
    if _is_trainable_agent_timeout_sample(sample):
        return 0.0
    return _extract_sample_reward(sample, reward_key)


def _sample_trainable_response_tokens(sample: Any) -> float:
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return _nonnegative_finite_float(getattr(sample, "response_length", 0))
    return sum(float(value) for value in loss_mask)


def _session_status_bucket(status: Any) -> str:
    status_name = str(getattr(status, "value", status) or "").upper()
    if status_name == "COMPLETED":
        return "completed"
    if status_name == "TIMEOUT":
        return "timeout"
    if status_name in {"ERROR", "FAILED"}:
        return "error"
    return "unknown"


def _add_distribution_metrics(
    out: dict[str, float],
    prefix: str,
    values: list[float],
    *,
    include_count: bool = False,
) -> None:
    if not values:
        return
    if include_count:
        out[f"{prefix}/count"] = float(len(values))
    out[f"{prefix}/mean"] = sum(values) / len(values)
    out[f"{prefix}/median"] = statistics.median(values)
    out[f"{prefix}/min"] = min(values)
    out[f"{prefix}/max"] = max(values)


def _add_session_distribution_metrics(
    out: dict[str, float],
    prefix: str,
    name: str,
    values: list[float],
    *,
    include_total: bool = False,
) -> None:
    """Publish one-session-one-vote scalar telemetry with explicit coverage."""

    out[f"{prefix}/{name}_accounted_session_count"] = float(len(values))
    if not values:
        return
    out[f"{prefix}/{name}_mean"] = sum(values) / len(values)
    out[f"{prefix}/{name}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    out[f"{prefix}/{name}_median"] = statistics.median(values)
    out[f"{prefix}/{name}_min"] = min(values)
    out[f"{prefix}/{name}_max"] = max(values)
    if include_total:
        out[f"{prefix}/{name}_total"] = sum(values)


def _strict_telemetry_equal(left: Any, right: Any) -> bool:
    """Compare bounded telemetry without Python's ``True == 1`` coercion."""

    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        return left.keys() == right.keys() and all(
            _strict_telemetry_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right):
            return False
        return len(left) == len(right) and all(
            _strict_telemetry_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    return left == right


def _spilot_router_metrics(
    sessions: dict[str, dict[str, Any]],
    session_rewards: dict[str, float],
    session_evaluations: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Aggregate bounded Router telemetry with one vote per session.

    Model identities are deliberately absent from metric names.  Slots are the
    stable action vocabulary, while their per-episode mapping remains available
    in trajectory metadata for offline analysis.
    """
    if not sessions:
        return {}

    prefix = "polar/spilot_router"
    candidate_alias_pairs: set[tuple[str, str]] = set()
    for metadata in sessions.values():
        slot_mapping = metadata.get("slot_mapping")
        if not isinstance(slot_mapping, dict):
            continue
        aliases = [
            candidate.get("model")
            for candidate in slot_mapping.values()
            if isinstance(candidate, dict)
            and isinstance(candidate.get("model"), str)
            and candidate.get("model")
        ]
        if len(aliases) == 2 and len(set(aliases)) == 2:
            candidate_alias_pairs.add(tuple(sorted(aliases)))
    # Candidate labels are meaningful only if the entire accepted batch uses
    # one canonical pair.  A mixed pair must not silently relabel a model as
    # C0/C1 and blend incomparable candidate aggregates.
    canonical_candidate_aliases = (
        next(iter(candidate_alias_pairs)) if len(candidate_alias_pairs) == 1 else None
    )
    action_valid_sessions = 0
    submitted_sessions = 0
    route_counts = {"M0": 0, "M1": 0}
    verify_counts = {"M0": 0, "M1": 0}
    route_candidate_counts = {"C0": 0, "C1": 0}
    verify_candidate_counts = {"C0": 0, "C1": 0}
    direct_submit_sessions = 0
    pool_call_count = 0
    pool_unattributed_call_count = 0
    pool_status_counts = {"completed": 0, "failed": 0, "timeout": 0}
    pool_status_candidate_counts = {
        candidate: {"completed": 0, "failed": 0, "timeout": 0} for candidate in ("C0", "C1")
    }
    pool_duration_ms_by_candidate: dict[str, list[float]] = {
        "C0": [],
        "C1": [],
    }
    pool_costs: list[float] = []
    pool_costs_by_candidate: dict[str, list[float]] = {"C0": [], "C1": []}
    pool_costs_by_role: dict[str, list[float]] = {"solve": [], "verify": [], "continue": []}
    pool_unattributed_cost = 0.0
    admission_wait_ms_by_candidate: dict[str, list[float]] = {
        "C0": [],
        "C1": [],
    }
    total_cost = 0.0
    admission_session_count = 0
    admission_waits_ms: list[float] = []
    admission_waited_session_count = 0
    admission_local_caps: list[float] = []
    admission_failure_count = 0
    admission_failure_candidate_counts = {"C0": 0, "C1": 0}
    admission_failure_wait_ms_by_candidate: dict[str, list[float]] = {
        "C0": [],
        "C1": [],
    }
    admission_fatal_retained_session_count = 0
    admission_node_health_accounted_session_count = 0
    admission_node_healthy_session_count = 0
    initial_slot_by_session: dict[str, str] = {}
    initial_candidate_by_session: dict[str, str] = {}
    accuracy_outcome_by_session: dict[str, float] = {}
    total_cost_by_session: dict[str, float] = {}
    cost_penalty_fraction_by_session: dict[str, float] = {}
    cost_penalty_reward_delta_by_session: dict[str, float] = {}
    total_latency_seconds_by_session: dict[str, float] = {}
    latency_penalty_fraction_by_session: dict[str, float] = {}
    latency_penalty_reward_delta_by_session: dict[str, float] = {}

    for session_id, metadata in sessions.items():
        admission_enabled = metadata.get("admission_enabled") is True
        if admission_enabled:
            admission_session_count += 1
            admission_wait_ms = _optional_nonnegative_finite_float(
                metadata.get("admission_wait_ms")
            )
            if admission_wait_ms is not None:
                admission_waits_ms.append(admission_wait_ms)
                if admission_wait_ms > 0:
                    admission_waited_session_count += 1
            fatal_retained = metadata.get("admission_fatal_retained") is True
            if fatal_retained:
                admission_fatal_retained_session_count += 1
            node_healthy = metadata.get("admission_node_healthy")
            if node_healthy is True or node_healthy is False:
                admission_node_health_accounted_session_count += 1
            # Missing health telemetry is unknown, never an implicit healthy
            # vote. This keeps partially written/legacy metadata fail closed.
            if node_healthy is True:
                admission_node_healthy_session_count += 1

        if metadata.get("action_valid") is True:
            action_valid_sessions += 1
        if metadata.get("submitted") is True:
            submitted_sessions += 1

        parsed_cost = _optional_nonnegative_finite_float(metadata.get("total_cost"))
        if parsed_cost is not None:
            total_cost += parsed_cost
            total_cost_by_session[session_id] = parsed_cost

        evaluation = (
            session_evaluations.get(session_id) if isinstance(session_evaluations, dict) else None
        )
        if isinstance(evaluation, dict):
            accuracy_outcome = _optional_unit_interval_float(
                evaluation.get("harbor_outcome_reward")
            )
            if accuracy_outcome is not None:
                accuracy_outcome_by_session[session_id] = accuracy_outcome
            cost_penalty_fraction = _optional_unit_interval_float(
                evaluation.get("applied_cost_penalty")
            )
            if cost_penalty_fraction is not None:
                cost_penalty_fraction_by_session[session_id] = cost_penalty_fraction
            if accuracy_outcome is not None and cost_penalty_fraction is not None:
                cost_penalty_reward_delta_by_session[session_id] = (
                    accuracy_outcome * cost_penalty_fraction
                )
            total_latency_seconds = _optional_nonnegative_finite_float(
                evaluation.get("total_latency_seconds")
            )
            if total_latency_seconds is not None:
                total_latency_seconds_by_session[session_id] = total_latency_seconds
            latency_penalty_fraction = _optional_unit_interval_float(
                evaluation.get("applied_latency_penalty")
            )
            if latency_penalty_fraction is not None:
                latency_penalty_fraction_by_session[session_id] = (
                    latency_penalty_fraction
                )
            if accuracy_outcome is not None and latency_penalty_fraction is not None:
                latency_penalty_reward_delta_by_session[session_id] = (
                    accuracy_outcome * latency_penalty_fraction
                )

        candidate_by_slot: dict[str, str] = {}
        candidate_by_alias: dict[str, str] = {}
        slot_mapping = metadata.get("slot_mapping")
        if isinstance(slot_mapping, dict):
            aliases_by_slot: dict[str, str] = {}
            for raw_slot, raw_candidate in slot_mapping.items():
                if not isinstance(raw_candidate, dict):
                    continue
                alias = raw_candidate.get("model")
                if isinstance(alias, str) and alias:
                    aliases_by_slot[str(raw_slot).upper()] = alias
            # Candidate labels are stable under the per-episode M0/M1 shuffle:
            # C0 is the lexicographically first model alias, C1 the second.
            # Duplicate aliases cannot be disambiguated safely, so omit their
            # candidate-level attribution while retaining slot diagnostics.
            sorted_aliases = tuple(sorted(set(aliases_by_slot.values())))
            if (
                len(sorted_aliases) == 2
                and len(aliases_by_slot) == 2
                and sorted_aliases == canonical_candidate_aliases
            ):
                candidate_by_alias = {
                    alias: f"C{index}" for index, alias in enumerate(sorted_aliases)
                }
                candidate_by_slot = {
                    slot: candidate_by_alias[alias]
                    for slot, alias in aliases_by_slot.items()
                    if alias in candidate_by_alias
                }

        admission_failure = metadata.get("admission_failure")
        if admission_enabled and isinstance(admission_failure, dict):
            admission_failure_count += 1
            failure_model = admission_failure.get("model")
            failure_candidate = (
                candidate_by_alias.get(failure_model) if isinstance(failure_model, str) else None
            )
            if failure_candidate in admission_failure_candidate_counts:
                admission_failure_candidate_counts[failure_candidate] += 1
                failure_wait_ms = _optional_nonnegative_finite_float(
                    admission_failure.get("wait_ms")
                )
                if failure_wait_ms is not None:
                    admission_failure_wait_ms_by_candidate[failure_candidate].append(
                        failure_wait_ms
                    )

        actions = metadata.get("actions")
        if isinstance(actions, list):
            initial_route: dict[str, Any] | None = None
            submit_seen = False
            verify_action: dict[str, Any] | None = None
            for action in actions:
                if not isinstance(action, dict) or action.get("valid") is not True:
                    continue
                action_name = str(action.get("action") or "").upper()
                if action_name == "ROUTE" and initial_route is None:
                    initial_route = action
                elif action_name == "VERIFY" and verify_action is None:
                    verify_action = action
                elif action_name == "SUBMIT":
                    submit_seen = True

            if initial_route is not None:
                slot = str(initial_route.get("model_slot") or "").upper()
                if slot in route_counts:
                    route_counts[slot] += 1
                    initial_slot_by_session[session_id] = slot
                    candidate = candidate_by_slot.get(slot)
                    if candidate in route_candidate_counts:
                        route_candidate_counts[candidate] += 1
                        initial_candidate_by_session[session_id] = candidate
            if verify_action is not None:
                slot = str(verify_action.get("model_slot") or "").upper()
                if slot in verify_counts:
                    verify_counts[slot] += 1
                    candidate = candidate_by_slot.get(slot)
                    if candidate in verify_candidate_counts:
                        verify_candidate_counts[candidate] += 1
            if submit_seen:
                direct_submit_sessions += 1

        calls = metadata.get("calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                pool_call_count += 1
                status = str(call.get("status") or "").lower()
                if status in pool_status_counts:
                    pool_status_counts[status] += 1
                call_model = call.get("model")
                call_slot = str(call.get("slot") or "").upper()
                model_candidate = (
                    candidate_by_alias.get(call_model) if isinstance(call_model, str) else None
                )
                slot_candidate = candidate_by_slot.get(call_slot)
                call_candidate = (
                    model_candidate
                    if model_candidate is not None and model_candidate == slot_candidate
                    else None
                )
                if call_candidate in pool_status_candidate_counts:
                    if status in pool_status_counts:
                        pool_status_candidate_counts[call_candidate][status] += 1
                    duration_ms = _optional_nonnegative_finite_float(call.get("duration_ms"))
                    if duration_ms is not None:
                        pool_duration_ms_by_candidate[call_candidate].append(duration_ms)
                    call_admission_wait_ms = _optional_nonnegative_finite_float(
                        call.get("admission_wait_ms")
                    )
                    if call_admission_wait_ms is not None:
                        admission_wait_ms_by_candidate[call_candidate].append(
                            call_admission_wait_ms
                        )
                else:
                    pool_unattributed_call_count += 1
                call_cost = _optional_nonnegative_finite_float(call.get("cost"))
                if call_cost is not None:
                    pool_costs.append(call_cost)
                    if call_candidate in pool_costs_by_candidate:
                        pool_costs_by_candidate[call_candidate].append(call_cost)
                    else:
                        pool_unattributed_cost += call_cost
                    call_role = str(call.get("role") or "").lower()
                    if call_role in pool_costs_by_role:
                        pool_costs_by_role[call_role].append(call_cost)
                if admission_enabled:
                    local_cap = _optional_nonnegative_finite_float(call.get("admission_local_cap"))
                    if local_cap is not None and local_cap > 0:
                        admission_local_caps.append(local_cap)

    session_count = len(sessions)
    metrics: dict[str, float] = {
        f"{prefix}/session_count": float(session_count),
        f"{prefix}/action_valid_count": float(action_valid_sessions),
        f"{prefix}/action_valid_fraction": action_valid_sessions / session_count,
        f"{prefix}/submitted_count": float(submitted_sessions),
        f"{prefix}/submitted_fraction": submitted_sessions / session_count,
        f"{prefix}/candidate_alias_pair_count": float(len(candidate_alias_pairs)),
        f"{prefix}/candidate_alias_pair_conflict": float(len(candidate_alias_pairs) > 1),
        f"{prefix}/route_m0_count": float(route_counts["M0"]),
        f"{prefix}/route_m1_count": float(route_counts["M1"]),
        f"{prefix}/verify_m0_count": float(verify_counts["M0"]),
        f"{prefix}/verify_m1_count": float(verify_counts["M1"]),
        f"{prefix}/route_candidate_c0_count": float(route_candidate_counts["C0"]),
        f"{prefix}/route_candidate_c1_count": float(route_candidate_counts["C1"]),
        f"{prefix}/verify_candidate_c0_count": float(verify_candidate_counts["C0"]),
        f"{prefix}/verify_candidate_c1_count": float(verify_candidate_counts["C1"]),
        f"{prefix}/direct_submit_count": float(direct_submit_sessions),
        f"{prefix}/pool_call_count": float(pool_call_count),
        f"{prefix}/pool_unattributed_call_count": float(pool_unattributed_call_count),
        f"{prefix}/pool_completed_count": float(pool_status_counts["completed"]),
        f"{prefix}/pool_failed_count": float(pool_status_counts["failed"]),
        f"{prefix}/pool_timeout_count": float(pool_status_counts["timeout"]),
        f"{prefix}/pool_cost_accounted_call_count": float(len(pool_costs)),
        f"{prefix}/pool_cost_total": sum(pool_costs),
        f"{prefix}/pool_unattributed_cost_total": pool_unattributed_cost,
        f"{prefix}/pool_cost_reconciliation_delta": total_cost - sum(pool_costs),
        f"{prefix}/total_cost": total_cost,
        f"{prefix}/admission_session_count": float(admission_session_count),
        f"{prefix}/admission_wait_ms_total": sum(admission_waits_ms),
        f"{prefix}/admission_wait_accounted_session_count": float(len(admission_waits_ms)),
        f"{prefix}/admission_waited_session_count": float(admission_waited_session_count),
        f"{prefix}/admission_fatal_retained_session_count": float(
            admission_fatal_retained_session_count
        ),
        f"{prefix}/admission_failure_count": float(admission_failure_count),
        f"{prefix}/admission_node_health_accounted_session_count": float(
            admission_node_health_accounted_session_count
        ),
        f"{prefix}/admission_node_healthy_session_count": float(
            admission_node_healthy_session_count
        ),
    }
    if admission_waits_ms:
        metrics[f"{prefix}/admission_wait_ms_mean"] = sum(admission_waits_ms) / len(
            admission_waits_ms
        )
        metrics[f"{prefix}/admission_wait_ms_max"] = max(admission_waits_ms)
    if admission_session_count:
        metrics[f"{prefix}/admission_waited_session_fraction"] = (
            admission_waited_session_count / admission_session_count
        )
        metrics[f"{prefix}/admission_fatal_retained_session_fraction"] = (
            admission_fatal_retained_session_count / admission_session_count
        )
        metrics[f"{prefix}/admission_node_healthy_session_fraction"] = (
            admission_node_healthy_session_count / admission_session_count
        )
        metrics[f"{prefix}/admission_node_health_accounted_session_fraction"] = (
            admission_node_health_accounted_session_count / admission_session_count
        )
    if admission_local_caps:
        metrics[f"{prefix}/admission_local_cap_observation_count"] = float(
            len(admission_local_caps)
        )
        metrics[f"{prefix}/admission_local_cap_mean"] = sum(admission_local_caps) / len(
            admission_local_caps
        )
        metrics[f"{prefix}/admission_local_cap_min"] = min(admission_local_caps)
        metrics[f"{prefix}/admission_local_cap_max"] = max(admission_local_caps)

    decomposed_metrics = {
        "accuracy_outcome": accuracy_outcome_by_session,
        "total_cost": total_cost_by_session,
        "cost_penalty_fraction": cost_penalty_fraction_by_session,
        "cost_penalty_reward_delta": cost_penalty_reward_delta_by_session,
        "cost_adjusted_reward": session_rewards,
        "total_latency_seconds": total_latency_seconds_by_session,
        "latency_penalty_fraction": latency_penalty_fraction_by_session,
        "latency_penalty_reward_delta": latency_penalty_reward_delta_by_session,
    }
    for metric_name, values_by_session in decomposed_metrics.items():
        values = [
            values_by_session[session_id]
            for session_id in sessions
            if session_id in values_by_session
        ]
        _add_session_distribution_metrics(
            metrics,
            prefix,
            metric_name,
            values,
            include_total=metric_name in {"total_cost", "cost_penalty_reward_delta"},
        )
    accuracy_outcomes = list(accuracy_outcome_by_session.values())
    metrics[f"{prefix}/accuracy_outcome_positive_count"] = float(
        sum(value > 0.0 for value in accuracy_outcomes)
    )
    if accuracy_outcomes:
        metrics[f"{prefix}/accuracy_outcome_positive_fraction"] = sum(
            value > 0.0 for value in accuracy_outcomes
        ) / len(accuracy_outcomes)

    for breakdown_name, initial_by_session in (
        ("m0", {key: value for key, value in initial_slot_by_session.items() if value == "M0"}),
        ("m1", {key: value for key, value in initial_slot_by_session.items() if value == "M1"}),
        (
            "candidate_c0",
            {key: value for key, value in initial_candidate_by_session.items() if value == "C0"},
        ),
        (
            "candidate_c1",
            {key: value for key, value in initial_candidate_by_session.items() if value == "C1"},
        ),
    ):
        for metric_name, values_by_session in decomposed_metrics.items():
            values = [
                values_by_session[session_id]
                for session_id in initial_by_session
                if session_id in values_by_session
            ]
            metrics[f"{prefix}/{metric_name}_{breakdown_name}_accounted_session_count"] = float(
                len(values)
            )
            if values:
                metrics[f"{prefix}/{metric_name}_{breakdown_name}_mean"] = sum(values) / len(
                    values
                )

    for candidate in ("C0", "C1"):
        candidate_name = candidate.lower()
        metrics[f"{prefix}/admission_failure_candidate_{candidate_name}_count"] = float(
            admission_failure_candidate_counts[candidate]
        )
        failure_waits = admission_failure_wait_ms_by_candidate[candidate]
        metrics[f"{prefix}/admission_failure_wait_candidate_{candidate_name}_accounted_count"] = (
            float(len(failure_waits))
        )
        if failure_waits:
            metrics[f"{prefix}/admission_failure_wait_candidate_{candidate_name}_mean_ms"] = sum(
                failure_waits
            ) / len(failure_waits)
            metrics[f"{prefix}/admission_failure_wait_candidate_{candidate_name}_max_ms"] = max(
                failure_waits
            )
        for status in ("completed", "failed", "timeout"):
            metrics[f"{prefix}/pool_{status}_candidate_{candidate_name}_count"] = float(
                pool_status_candidate_counts[candidate][status]
            )
        candidate_durations = pool_duration_ms_by_candidate[candidate]
        metrics[f"{prefix}/pool_duration_candidate_{candidate_name}_count"] = float(
            len(candidate_durations)
        )
        if candidate_durations:
            metrics[f"{prefix}/pool_duration_candidate_{candidate_name}_mean_ms"] = sum(
                candidate_durations
            ) / len(candidate_durations)
            metrics[f"{prefix}/pool_duration_candidate_{candidate_name}_max_ms"] = max(
                candidate_durations
            )
        candidate_admission_waits = admission_wait_ms_by_candidate[candidate]
        metrics[f"{prefix}/admission_wait_candidate_{candidate_name}_accounted_count"] = float(
            len(candidate_admission_waits)
        )
        if candidate_admission_waits:
            metrics[f"{prefix}/admission_wait_candidate_{candidate_name}_mean_ms"] = sum(
                candidate_admission_waits
            ) / len(candidate_admission_waits)
            metrics[f"{prefix}/admission_wait_candidate_{candidate_name}_max_ms"] = max(
                candidate_admission_waits
            )
        candidate_costs = pool_costs_by_candidate[candidate]
        metrics[f"{prefix}/pool_cost_candidate_{candidate_name}_accounted_call_count"] = float(
            len(candidate_costs)
        )
        metrics[f"{prefix}/pool_cost_candidate_{candidate_name}_total"] = sum(candidate_costs)
        if candidate_costs:
            metrics[f"{prefix}/pool_cost_candidate_{candidate_name}_mean"] = sum(
                candidate_costs
            ) / len(candidate_costs)

    for role, role_costs in pool_costs_by_role.items():
        metrics[f"{prefix}/pool_cost_{role}_accounted_call_count"] = float(len(role_costs))
        metrics[f"{prefix}/pool_cost_{role}_total"] = sum(role_costs)
        if role_costs:
            metrics[f"{prefix}/pool_cost_{role}_mean"] = sum(role_costs) / len(role_costs)

    router_rewards = [
        session_rewards[session_id] for session_id in sessions if session_id in session_rewards
    ]
    metrics[f"{prefix}/reward_accounted_session_count"] = float(len(router_rewards))
    if router_rewards:
        metrics[f"{prefix}/reward_mean"] = sum(router_rewards) / len(router_rewards)

    for slot in ("M0", "M1"):
        slot_rewards = [
            session_rewards[session_id]
            for session_id, initial_slot in initial_slot_by_session.items()
            if initial_slot == slot and session_id in session_rewards
        ]
        slot_name = slot.lower()
        metrics[f"{prefix}/reward_{slot_name}_accounted_session_count"] = float(len(slot_rewards))
        if slot_rewards:
            metrics[f"{prefix}/reward_{slot_name}_mean"] = sum(slot_rewards) / len(slot_rewards)

    for candidate in ("C0", "C1"):
        candidate_rewards = [
            session_rewards[session_id]
            for session_id, initial_candidate in initial_candidate_by_session.items()
            if initial_candidate == candidate and session_id in session_rewards
        ]
        candidate_name = candidate.lower()
        metrics[f"{prefix}/reward_candidate_{candidate_name}_count"] = float(
            len(candidate_rewards)
        )
        if candidate_rewards:
            metrics[f"{prefix}/reward_candidate_{candidate_name}_mean"] = sum(
                candidate_rewards
            ) / len(candidate_rewards)

    return metrics


def _polar_extra_metrics(
    flat_samples: list[Any],
    rewards: list[float],
    reward_key: str,
) -> dict[str, float]:
    """Compact user-facing Polar metrics for W&B."""
    out: dict[str, float] = {}
    seen: set[str] = set()
    stage_timing_values = {field: [] for field, _ in _SESSION_STAGE_TIMING_FIELDS}
    runtime_exec_values = {
        "runtime_exec_ms": [],
        "runtime_exec_count": [],
        "runtime_exec_timeout_count": [],
        "runtime_exec_failure_count": [],
        "runtime_exec_exception_count": [],
        "runtime_exec_cancelled_count": [],
        "mini_swe_command_ms": [],
        "mini_swe_command_count": [],
        "mini_swe_command_timeout_count": [],
        "mini_swe_command_failure_count": [],
    }
    command_category_totals = {
        "runtime_exec_ms_by_category": {category: 0.0 for category in RUNTIME_EXEC_CATEGORIES},
        "runtime_exec_count_by_category": {category: 0.0 for category in RUNTIME_EXEC_CATEGORIES},
        "mini_swe_command_ms_by_category": {category: 0.0 for category in RUNTIME_EXEC_CATEGORIES},
        "mini_swe_command_count_by_category": {
            category: 0.0 for category in RUNTIME_EXEC_CATEGORIES
        },
    }
    inference_timing_values: dict[str, list[float]] = {
        field: [] for field in _INFERENCE_TIMING_FIELDS
    }
    inference_timing_count = 0
    seen_inference_traces: set[tuple[str, int]] = set()
    timed_session_count = 0
    session_is_placeholder: dict[str, bool] = {}
    session_report: dict[str, dict[str, Any]] = {}
    completed_session_trace_rewards: dict[str, list[float]] = {}
    agent_timeout_session_trace_rewards: dict[str, list[float]] = {}
    trusted_model_failure_sessions: set[str] = set()
    policy_staleness: list[float] = []
    parser_invalid_traces = 0
    parser_invalid_sessions: set[str] = set()
    agent_timeout_traces = 0
    agent_timeout_sessions: set[str] = set()
    trainable_sessions: set[str] = set()
    terminal_timeout_sessions: set[str] = set()
    terminal_error_sessions: set[str] = set()
    timeout_agent_exec_sessions: set[str] = set()
    timeout_agent_postprocess_sessions: set[str] = set()
    early_stop_cancelled_sessions: set[str] = set()
    early_stop_elapsed_ms: list[float] = []
    session_trainable_response_tokens: dict[str, float] = {}
    session_raw_response_tokens: dict[str, float] = {}
    session_real_trace_counts: dict[str, int] = {}
    session_truncated_trace_counts: dict[str, int] = {}
    session_agent_timeout_trace_counts: dict[str, int] = {}
    session_status_buckets: dict[str, str] = {}
    spilot_router_sessions: dict[str, dict[str, Any]] = {}
    spilot_router_evaluations: dict[str, dict[str, Any]] = {}
    spilot_router_conflicting_sessions: set[str] = set()
    trainable_traces = 0
    trajectory_rewards_by_group: dict[Any, dict[Any, list[float]]] = {}
    for sample in flat_samples:
        polar_meta = sample.metadata.get("polar", {})
        training_filter = polar_meta.get("training_filter") or {}
        parser_invalid = (
            isinstance(training_filter, dict)
            and training_filter.get("reason") == "parser_invalid_tool_call"
        )
        agent_timeout = _is_trainable_agent_timeout_sample(sample)
        if parser_invalid:
            parser_invalid_traces += 1
        if agent_timeout:
            agent_timeout_traces += 1
        if "policy_staleness" in polar_meta:
            policy_staleness.append(float(polar_meta["policy_staleness"]))
        session_id = polar_meta.get("session_id")
        session_key = str(session_id) if session_id else None
        session_status = str(_sample_session_status(sample) or "").upper()
        trace_index = int(polar_meta.get("trace_index", 0) or 0)
        inference_trace_key = (str(session_id or ""), trace_index)
        if inference_trace_key not in seen_inference_traces:
            seen_inference_traces.add(inference_trace_key)
            trace_metadata = polar_meta.get("trace_metadata")
            for inference_timing in _iter_inference_timings(trace_metadata):
                inference_timing_count += 1
                for field, values in inference_timing_values.items():
                    value = _optional_nonnegative_finite_float(inference_timing.get(field))
                    if value is not None:
                        values.append(value)
        if parser_invalid and session_key:
            parser_invalid_sessions.add(session_key)
        if agent_timeout and session_key:
            agent_timeout_sessions.add(session_key)
        result_metadata = polar_meta.get("result_metadata") or {}
        early_stop_cancelled = (
            session_id
            and isinstance(result_metadata, dict)
            and result_metadata.get("early_stop_cancelled") is True
        )
        if early_stop_cancelled:
            early_stop_cancelled_sessions.add(session_key)
            elapsed = _optional_nonnegative_finite_float(
                result_metadata.get("early_stop_elapsed_ms")
            )
            if elapsed is not None:
                early_stop_elapsed_ms.append(elapsed)
        sample_is_trainable = _sample_has_trainable_tokens(sample)
        if sample_is_trainable:
            trainable_traces += 1
            if session_key:
                trainable_sessions.add(session_key)
            group_id = getattr(sample, "group_index", None)
            trajectory_id = getattr(sample, "rollout_id", None)
            if trajectory_id is None:
                trajectory_id = getattr(sample, "index", None)
            if trajectory_id is None:
                trajectory_id = session_id or id(sample)
            trajectory_rewards_by_group.setdefault(group_id, {}).setdefault(
                trajectory_id, []
            ).append(_effective_trainable_reward(sample, reward_key))
        is_placeholder = bool(polar_meta.get("placeholder"))
        if not session_key:
            continue
        trajectory_metadata = polar_meta.get("trajectory_metadata")
        evaluation = (
            trajectory_metadata.get("evaluation")
            if isinstance(trajectory_metadata, dict)
            else None
        )
        router_metadata = evaluation.get("spilot_router") if isinstance(evaluation, dict) else None
        if (
            isinstance(router_metadata, dict)
            and session_key not in spilot_router_conflicting_sessions
        ):
            previous_router = spilot_router_sessions.get(session_key)
            previous_evaluation = spilot_router_evaluations.get(session_key)
            evaluation_fields = {
                field: evaluation.get(field)
                for field in ("harbor_outcome_reward", "applied_cost_penalty")
            }
            previous_evaluation_fields = (
                {
                    field: previous_evaluation.get(field)
                    for field in ("harbor_outcome_reward", "applied_cost_penalty")
                }
                if isinstance(previous_evaluation, dict)
                else None
            )
            if previous_router is None:
                spilot_router_sessions[session_key] = router_metadata
                spilot_router_evaluations[session_key] = evaluation
            elif not _strict_telemetry_equal(
                previous_router, router_metadata
            ) or not _strict_telemetry_equal(previous_evaluation_fields, evaluation_fields):
                # Conflicting trace copies are unsafe for one-session-one-vote
                # telemetry. Omit the whole session rather than selecting an
                # arbitrary first trace based on arrival order.
                spilot_router_sessions.pop(session_key, None)
                spilot_router_evaluations.pop(session_key, None)
                spilot_router_conflicting_sessions.add(session_key)
        status_bucket = _session_status_bucket(session_status)
        if session_status_buckets.get(session_key, "unknown") == "unknown":
            session_status_buckets[session_key] = status_bucket
        if not is_placeholder:
            session_trainable_response_tokens[session_key] = session_trainable_response_tokens.get(
                session_key, 0.0
            ) + _sample_trainable_response_tokens(sample)
            session_raw_response_tokens[session_key] = session_raw_response_tokens.get(
                session_key, 0.0
            ) + _nonnegative_finite_float(getattr(sample, "response_length", 0))
            session_real_trace_counts[session_key] = (
                session_real_trace_counts.get(session_key, 0) + 1
            )
            if _is_truncated(sample):
                session_truncated_trace_counts[session_key] = (
                    session_truncated_trace_counts.get(session_key, 0) + 1
                )
                if agent_timeout:
                    session_agent_timeout_trace_counts[session_key] = (
                        session_agent_timeout_trace_counts.get(session_key, 0) + 1
                    )
        session_is_placeholder[session_key] = (
            session_is_placeholder.get(session_key, True) and is_placeholder
        )
        if not early_stop_cancelled:
            if session_status == "TIMEOUT":
                terminal_timeout_sessions.add(session_key)
                trajectory_metadata = polar_meta.get("trajectory_metadata")
                agent_result = (
                    trajectory_metadata.get("agent_result")
                    if isinstance(trajectory_metadata, dict)
                    else None
                )
                if isinstance(agent_result, dict):
                    timeout_source = str(agent_result.get("timeout_source") or "").lower()
                    timeout_stage = str(agent_result.get("timeout_stage") or "").lower()
                    if timeout_source == "agent" and timeout_stage == "exec":
                        timeout_agent_exec_sessions.add(session_key)
                    elif timeout_source == "agent" and timeout_stage == "postprocess":
                        timeout_agent_postprocess_sessions.add(session_key)
            elif session_status in {"ERROR", "FAILED"}:
                terminal_error_sessions.add(session_key)
        if session_key not in seen:
            seen.add(session_key)
            timing = polar_meta.get("timing") or {}
            # Synthetic straggler placeholders contain rollout-server time to
            # cancellation, not completed gateway stage timings. Keep them out
            # of stage means and report their elapsed time separately below.
            if isinstance(timing, dict) and timing and not early_stop_cancelled:
                timed_session_count += 1
                for field, _ in _SESSION_STAGE_TIMING_FIELDS:
                    stage_timing_values[field].append(_nonnegative_finite_float(timing.get(field)))
                for field, values in runtime_exec_values.items():
                    values.append(_nonnegative_finite_float(timing.get(field)))
                for field, category_totals in command_category_totals.items():
                    category_values = timing.get(field)
                    if not isinstance(category_values, dict):
                        continue
                    for category in RUNTIME_EXEC_CATEGORIES:
                        category_totals[category] += _nonnegative_finite_float(
                            category_values.get(category)
                        )
            evaluation = (polar_meta.get("trajectory_metadata") or {}).get("evaluation") or {}
            report = evaluation.get("report") or {}
            if isinstance(report, dict) and report:
                session_report[session_key] = report
        if not early_stop_cancelled:
            if agent_timeout and not is_placeholder:
                # The model exhausted its own agent budget. This is a real,
                # aligned zero-reward policy outcome rather than missing
                # infrastructure, so include it in primary quality metrics.
                agent_timeout_session_trace_rewards.setdefault(session_key, []).append(0.0)
            elif session_status == "COMPLETED" and not is_placeholder:
                # Include parser-invalid/fully-masked real traces as zero
                # quality outcomes. The adapter already fail-closes their
                # scalar reward, and quality reporting should not make model
                # failures disappear merely because they are untrainable.
                completed_session_trace_rewards.setdefault(session_key, []).append(
                    _extract_sample_reward(sample, reward_key) if sample_is_trainable else 0.0
                )
            elif session_status in {"ERROR", "FAILED"} and _has_trusted_eval_failure(sample):
                # The verifier ran successfully and accepted the outcome, so
                # this is a real model failure rather than missing infra data.
                # Force zero even if a stale artifact carries positive reward.
                trusted_model_failure_sessions.add(session_key)

    if timed_session_count:
        for field, metric_suffix in _SESSION_STAGE_TIMING_FIELDS:
            out[f"timing/session_ms/{metric_suffix}"] = (
                sum(stage_timing_values[field]) / timed_session_count
            )
        for field, values in runtime_exec_values.items():
            family = "runtime_exec" if field.startswith("runtime_exec") else "mini_swe_command"
            suffix = field.removeprefix(f"{family}_")
            namespace = "timing" if suffix == "ms" else "polar"
            out[f"{namespace}/{family}/{suffix}_per_session_mean"] = (
                sum(values) / timed_session_count
            )

        for family in ("runtime_exec", "mini_swe_command"):
            total_ms = sum(runtime_exec_values[f"{family}_ms"])
            total_count = sum(runtime_exec_values[f"{family}_count"])
            if total_count > 0.0:
                out[f"timing/{family}/ms_per_command_mean"] = total_ms / total_count

        category_metric_names = (
            ("runtime_exec_ms_by_category", "runtime_exec", "ms"),
            ("runtime_exec_count_by_category", "runtime_exec", "count"),
            ("mini_swe_command_ms_by_category", "mini_swe_command", "ms"),
            ("mini_swe_command_count_by_category", "mini_swe_command", "count"),
        )
        for field, family, unit in category_metric_names:
            for category, total in command_category_totals[field].items():
                if total > 0.0:
                    namespace = "timing" if unit == "ms" else "polar"
                    out[f"{namespace}/{family}/{category}_{unit}_per_session_mean"] = (
                        total / timed_session_count
                    )
                    if unit == "ms":
                        count = command_category_totals[
                            field.replace("ms_by_category", "count_by_category")
                        ][category]
                        if count > 0.0:
                            out[f"timing/{family}/{category}_ms_per_command_mean"] = total / count
    if inference_timing_count:
        out["polar/inference/timed_completion_count"] = float(inference_timing_count)
        for field, values in inference_timing_values.items():
            if not values:
                continue
            namespace = "timing" if field.endswith("_ms") else "polar"
            out[f"{namespace}/inference/{field}_mean"] = sum(values) / len(values)
            out[f"{namespace}/inference/{field}_p95"] = _nearest_rank_percentile(values, 0.95)
            out[f"{namespace}/inference/{field}_max"] = max(values)
    if rewards:
        # Retain the old trace/placeholder-weighted view under an explicit
        # diagnostic name. It is not an exchangeable GRPO outcome: a session
        # can emit multiple traces, while an early-stop cancellation emits a
        # synthetic zero-gradient placeholder.
        safe_rewards = [_finite_float_or_zero(reward) for reward in rewards]
        out["polar/reward_mean_all_samples"] = sum(safe_rewards) / len(safe_rewards)
    completed_session_rewards_by_key = {
        session_id: sum(trace_rewards) / len(trace_rewards)
        for session_id, trace_rewards in completed_session_trace_rewards.items()
        if trace_rewards
    }
    agent_timeout_session_rewards_by_key = {
        session_id: sum(trace_rewards) / len(trace_rewards)
        for session_id, trace_rewards in agent_timeout_session_trace_rewards.items()
        if trace_rewards
    }
    accounted_session_rewards_by_key = dict(completed_session_rewards_by_key)
    accounted_session_rewards_by_key.update(agent_timeout_session_rewards_by_key)
    for session_id in trusted_model_failure_sessions:
        accounted_session_rewards_by_key.setdefault(session_id, 0.0)
    out.update(
        _spilot_router_metrics(
            spilot_router_sessions,
            accounted_session_rewards_by_key,
            spilot_router_evaluations,
        )
    )
    if spilot_router_sessions or spilot_router_conflicting_sessions:
        out["polar/spilot_router/telemetry_conflict_session_count"] = float(
            len(spilot_router_conflicting_sessions)
        )
    completed_session_rewards = list(completed_session_rewards_by_key.values())
    accounted_session_rewards = list(accounted_session_rewards_by_key.values())
    if completed_session_rewards:
        reward_mean_completed = sum(completed_session_rewards) / len(completed_session_rewards)
        out["polar/reward_mean_completed"] = reward_mean_completed
    if accounted_session_rewards:
        # Primary quality is one outcome per real session: completed sessions
        # contribute their mean trace reward, trustworthy model failures
        # contribute zero, and intentional early-stop cancellations are absent.
        # This avoids both trace fan-out weighting and synthetic placeholder
        # dilution while still making genuine model errors hurt quality.
        out["polar/reward_mean"] = sum(accounted_session_rewards) / len(accounted_session_rewards)
        out["polar/reward_std"] = (
            statistics.pstdev(accounted_session_rewards)
            if len(accounted_session_rewards) > 1
            else 0.0
        )
        out["polar/reward_accounted_sessions"] = float(len(accounted_session_rewards))
        out["polar/reward_trainable_agent_timeout_sessions"] = float(
            len(agent_timeout_session_trace_rewards)
        )
        out["polar/reward_model_failure_sessions"] = float(
            len(
                (trusted_model_failure_sessions | agent_timeout_session_trace_rewards.keys())
                - completed_session_trace_rewards.keys()
            )
        )

    length_session_ids = set(session_trainable_response_tokens) - early_stop_cancelled_sessions
    if length_session_ids:
        ordered_length_session_ids = sorted(length_session_ids)
        effective_lengths = [
            session_trainable_response_tokens[session_id]
            for session_id in ordered_length_session_ids
        ]
        raw_lengths = [
            session_raw_response_tokens[session_id] for session_id in ordered_length_session_ids
        ]
        _add_distribution_metrics(
            out,
            "polar/session_trainable_response_tokens",
            effective_lengths,
            include_count=True,
        )
        _add_distribution_metrics(out, "polar/session_raw_response_tokens", raw_lengths)

        for status_bucket in ("completed", "timeout", "error", "unknown"):
            status_lengths = [
                session_trainable_response_tokens[session_id]
                for session_id in ordered_length_session_ids
                if session_status_buckets.get(session_id, "unknown") == status_bucket
            ]
            if status_lengths:
                out[f"polar/session_trainable_response_tokens/by_status/{status_bucket}_mean"] = (
                    sum(status_lengths) / len(status_lengths)
                )

        real_trace_count = sum(
            session_real_trace_counts[session_id] for session_id in length_session_ids
        )
        truncated_trace_count = sum(
            session_truncated_trace_counts.get(session_id, 0) for session_id in length_session_ids
        )
        agent_timeout_trace_count = sum(
            session_agent_timeout_trace_counts.get(session_id, 0)
            for session_id in length_session_ids
        )
        if real_trace_count:
            out["polar/trace_truncation/truncated_fraction"] = (
                truncated_trace_count / real_trace_count
            )
            out["polar/trace_truncation/agent_timeout_fraction"] = (
                agent_timeout_trace_count / real_trace_count
            )
            out["polar/trace_truncation/non_agent_timeout_fraction"] = (
                truncated_trace_count - agent_timeout_trace_count
            ) / real_trace_count

        truncated_session_ids = {
            session_id
            for session_id in length_session_ids
            if session_truncated_trace_counts.get(session_id, 0) > 0
        }
        agent_timeout_truncated_session_ids = {
            session_id
            for session_id in length_session_ids
            if session_agent_timeout_trace_counts.get(session_id, 0) > 0
        }
        non_agent_timeout_session_ids = truncated_session_ids - agent_timeout_truncated_session_ids
        length_session_count = len(length_session_ids)
        out["polar/session_truncation/session_count"] = float(length_session_count)
        out["polar/session_truncation/truncated_count"] = float(len(truncated_session_ids))
        out["polar/session_truncation/truncated_fraction"] = (
            len(truncated_session_ids) / length_session_count
        )
        out["polar/session_truncation/agent_timeout_fraction"] = (
            len(agent_timeout_truncated_session_ids) / length_session_count
        )
        out["polar/session_truncation/non_agent_timeout_fraction"] = (
            len(non_agent_timeout_session_ids) / length_session_count
        )

    quality_eligible_session_ids = seen - early_stop_cancelled_sessions
    if quality_eligible_session_ids:
        quality_eligible_session_count = len(quality_eligible_session_ids)
        for status_bucket in ("completed", "timeout", "error", "unknown"):
            status_count = sum(
                1
                for session_id in quality_eligible_session_ids
                if session_status_buckets.get(session_id, "unknown") == status_bucket
            )
            out[f"polar/session_status/{status_bucket}_count"] = float(status_count)
            out[f"polar/session_status/{status_bucket}_fraction"] = (
                status_count / quality_eligible_session_count
            )

        accounted_session_ids = (
            quality_eligible_session_ids & accounted_session_rewards_by_key.keys()
        )
        unaccounted_session_ids = quality_eligible_session_ids - accounted_session_ids
        for outcome_name, session_ids in (
            ("accounted", accounted_session_ids),
            ("unaccounted", unaccounted_session_ids),
        ):
            out[f"polar/session_outcome/{outcome_name}_count"] = float(len(session_ids))
            out[f"polar/session_outcome/{outcome_name}_fraction"] = (
                len(session_ids) / quality_eligible_session_count
            )

        outcome_session_ids = {
            "positive": {
                session_id
                for session_id in accounted_session_ids
                if accounted_session_rewards_by_key[session_id] > 0.0
            },
            "zero": {
                session_id
                for session_id in accounted_session_ids
                if accounted_session_rewards_by_key[session_id] == 0.0
            },
            "negative": {
                session_id
                for session_id in accounted_session_ids
                if accounted_session_rewards_by_key[session_id] < 0.0
            },
        }
        accounted_session_count = len(accounted_session_ids)
        for outcome_name, session_ids in outcome_session_ids.items():
            out[f"polar/session_outcome/{outcome_name}_count"] = float(len(session_ids))
            if accounted_session_count:
                out[f"polar/session_outcome/{outcome_name}_fraction_of_accounted"] = (
                    len(session_ids) / accounted_session_count
                )
            outcome_lengths = [
                session_trainable_response_tokens[session_id]
                for session_id in session_ids & length_session_ids
            ]
            if outcome_lengths:
                out[f"polar/session_trainable_response_tokens/by_outcome/{outcome_name}_mean"] = (
                    sum(outcome_lengths) / len(outcome_lengths)
                )
    if policy_staleness:
        out["polar/staleness/mean"] = sum(policy_staleness) / len(policy_staleness)

    if flat_samples:
        out["polar/training_filter/parser_invalid_trace_fraction"] = parser_invalid_traces / len(
            flat_samples
        )
        out["polar/training_filter/agent_timeout_trace_fraction"] = agent_timeout_traces / len(
            flat_samples
        )
        out["polar/training_filter/trainable_trace_fraction"] = trainable_traces / len(
            flat_samples
        )

    group_count = len(trajectory_rewards_by_group)
    if group_count:
        mixed_groups = 0
        trainable_reward_groups = 0
        for trajectories in trajectory_rewards_by_group.values():
            means = [sum(values) / len(values) for values in trajectories.values() if values]
            distinct = {float(value) for value in means}
            if len(distinct) > 1:
                mixed_groups += 1
            # This mirrors the custom LOO post-processor: with two or more
            # trajectories, an all-equal group has zero scale and therefore
            # zero advantages. A single valid trajectory retains an unscaled
            # nonzero reward against the empty-peer baseline.
            if len(distinct) > 1 or (len(means) == 1 and means[0] != 0.0):
                trainable_reward_groups += 1
        out["polar/reward_groups/count"] = float(group_count)
        out["polar/reward_groups/mixed_fraction"] = mixed_groups / group_count
        out["polar/reward_groups/trainable_fraction"] = trainable_reward_groups / group_count

    total_sessions = len(seen)
    empty_sessions = sum(1 for p in session_is_placeholder.values() if p)
    if total_sessions > 0:
        # Slot completion keeps the scheduler/capacity view, where intentional
        # early-stop placeholders occupy real requested slots. Model-quality
        # success excludes those deliberately unexecuted stragglers and treats
        # both unexpected empty sessions and model agent timeouts as failures.
        out["polar/rollout_slot_completion_rate"] = (
            total_sessions - empty_sessions
        ) / total_sessions
        attempted_sessions = seen - early_stop_cancelled_sessions
        attempted_trainable_sessions = attempted_sessions & trainable_sessions
        fully_masked_sessions = attempted_sessions - attempted_trainable_sessions
        attempted_terminal_timeouts = attempted_sessions & terminal_timeout_sessions
        attempted_terminal_errors = attempted_sessions & terminal_error_sessions
        trainable_timeout_sessions = attempted_terminal_timeouts & trainable_sessions
        masked_timeout_sessions = attempted_terminal_timeouts - trainable_timeout_sessions
        out["polar/rollout_attempted_sessions"] = float(len(attempted_sessions))
        out["polar/rollout_attempted_session_fraction"] = len(attempted_sessions) / total_sessions
        out["polar/rollout_trainable_sessions"] = float(len(attempted_trainable_sessions))
        out["polar/rollout_fully_masked_sessions"] = float(len(fully_masked_sessions))
        out["polar/terminal_timeout_sessions"] = float(len(attempted_terminal_timeouts))
        out["polar/terminal_error_sessions"] = float(len(attempted_terminal_errors))
        out["polar/timeout_agent_exec_sessions"] = float(
            len(timeout_agent_exec_sessions & attempted_sessions)
        )
        out["polar/timeout_agent_postprocess_sessions"] = float(
            len(timeout_agent_postprocess_sessions & attempted_sessions)
        )
        out["polar/timeout_trainable_sessions"] = float(len(trainable_timeout_sessions))
        out["polar/timeout_masked_sessions"] = float(len(masked_timeout_sessions))
        if attempted_sessions:
            out["polar/rollout_trainable_session_fraction"] = len(
                attempted_trainable_sessions
            ) / len(attempted_sessions)
            out["polar/rollout_fully_masked_session_fraction"] = len(fully_masked_sessions) / len(
                attempted_sessions
            )
            failed_attempted_sessions = {
                session_id
                for session_id, placeholder in session_is_placeholder.items()
                if placeholder and session_id in attempted_sessions
            } | (agent_timeout_sessions & attempted_sessions)
            failed_attempted_sessions |= attempted_terminal_timeouts
            failed_attempted_sessions |= attempted_terminal_errors
            successful_sessions = attempted_sessions - failed_attempted_sessions
            out["polar/rollout_successful_sessions"] = float(len(successful_sessions))
            out["polar/rollout_success_rate"] = len(successful_sessions) / len(attempted_sessions)
            out["polar/training_filter/parser_invalid_session_fraction"] = len(
                parser_invalid_sessions & attempted_sessions
            ) / len(attempted_sessions)
            out["polar/training_filter/agent_timeout_session_fraction"] = len(
                agent_timeout_sessions & attempted_sessions
            ) / len(attempted_sessions)
        out["polar/early_stop/cancelled_sessions"] = float(len(early_stop_cancelled_sessions))
        out["polar/early_stop/cancelled_session_fraction"] = (
            len(early_stop_cancelled_sessions) / total_sessions
        )
        if early_stop_elapsed_ms:
            out["timing/early_stop/elapsed_ms_mean"] = sum(early_stop_elapsed_ms) / len(
                early_stop_elapsed_ms
            )
            out["timing/early_stop/elapsed_ms_max"] = max(early_stop_elapsed_ms)
    if session_report:
        graded_sessions = len(session_report)
        resolved = sum(1 for r in session_report.values() if r.get("resolved"))
        out["polar/resolved_rate"] = resolved / graded_sessions
    return out


def _nonnegative_finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if parsed < 0.0 or not math.isfinite(parsed):
        return 0.0
    return parsed


def _optional_nonnegative_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0.0 or not math.isfinite(parsed):
        return None
    return parsed


def _optional_unit_interval_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        return None
    return parsed


def _iter_inference_timings(trace_metadata: Any):
    """Yield sanitized per-choice timings without duplicating merged metadata."""

    if not isinstance(trace_metadata, dict):
        return
    completion_metadata = trace_metadata.get("completion_metadata")
    metadata_items = (
        completion_metadata if isinstance(completion_metadata, list) else [trace_metadata]
    )
    for metadata in metadata_items:
        if not isinstance(metadata, dict):
            continue
        timings = metadata.get("inference_timings")
        if not isinstance(timings, list):
            continue
        for timing in timings:
            if isinstance(timing, dict):
                yield timing


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _sample_has_trainable_tokens(sample: Any) -> bool:
    if bool(getattr(sample, "remove_sample", False)):
        return False
    status = getattr(sample, "status", None)
    status_name = getattr(status, "name", None) or str(status).rsplit(".", 1)[-1]
    if status_name.upper() in ("FAILED", "ABORTED"):
        return False
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return int(getattr(sample, "response_length", 0) or 0) > 0
    return any(int(value) != 0 for value in loss_mask)


def _is_truncated(sample: Any) -> bool:
    status = getattr(sample, "status", None)
    return getattr(status, "value", status) == "truncated"


def _load_rollout_train_output_type() -> Any:
    try:
        from slime.rollout.base_types import RolloutFnTrainOutput
    except ImportError as exc:
        raise ImportError("Slime is required to run Polar rollouts from a Slime trainer.") from exc
    return RolloutFnTrainOutput


def _load_rollout_eval_output_type() -> Any:
    try:
        from slime.rollout.base_types import RolloutFnEvalOutput
    except ImportError as exc:
        raise ImportError(
            "Slime is required to run Polar evaluation rollouts from a Slime trainer."
        ) from exc
    return RolloutFnEvalOutput


def _load_sample_type() -> Any:
    try:
        from slime.utils.types import Sample
    except ImportError as exc:
        raise ImportError(
            "Slime is required to build Polar evaluation samples from eval datasets."
        ) from exc
    return Sample


# Slime's RolloutManager recognizes an optional callable ``dispose`` attribute
# on a custom rollout function. Stop speculative work before it finishes W&B
# and releases the Ray actor.
setattr(generate_rollout_polar_async, "dispose", stop_global_worker)
# The paired weight-update hooks freeze new gateway-to-SGLang generations and
# drain in-flight calls before Slime performs its destructive pause/cache flush.
setattr(generate_rollout_polar_async, "pause_for_weight_update", pause_for_weight_update)
setattr(
    generate_rollout_polar_async,
    "resume_after_weight_update",
    resume_after_weight_update,
)

atexit.register(stop_global_worker)
