#!/usr/bin/env python3
"""Run a paired, eval-only forced-route comparison through Polar.

This command submits the Cartesian product of a fixed TMax JSONL slice and the
two configured SPilot pool candidates.  Every task executes exactly one frozen
mini-SWE candidate and auto-submits to the existing ``spilot_harbor``
evaluator.  It never starts Slime, loads an actor checkpoint, calls the Router,
or emits trainable trajectory traces.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
from types import SimpleNamespace
from typing import Any, Iterable

import httpx
import yaml

from polar.agent.presets.spilot_forced_route_eval_runner import EVAL_ONLY_ACK
from slime_bridge._messages import prompt_to_instruction_text
from slime_bridge.config import (
    render_instruction,
    render_task_payload,
    resolve_polar_slime_config,
)


_CONTROL_TOKEN_ENV = "POLAR_CONTROL_PLANE_TOKEN"
_CONTROL_TOKEN_HEADER = "X-Polar-Control-Token"
_AGENT_ACK_ENV = "SPILOT_FORCED_ROUTE_EVAL_ACK"
_BUILDER = "polar.trajectory.builder.spilot_forced_eval:SpilotForcedEvalBuilder"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    pool_model: str
    endpoint_model: str
    label: str

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "-", self.label).strip("-").lower()


@dataclass(frozen=True, slots=True)
class EvalItem:
    dataset_index: int
    prompt: object
    metadata: dict[str, Any]

    @property
    def task_name(self) -> str:
        value = self.metadata.get("task_name")
        return str(value) if value is not None else f"dataset-row-{self.dataset_index}"


@dataclass(frozen=True, slots=True)
class WorkItem:
    item: EvalItem
    candidate: CandidateSpec
    replicate: int
    pair_seed: int


DEFAULT_CANDIDATES = (
    CandidateSpec(
        pool_model="pool/qwen3.6-27b",
        endpoint_model="nvidia/qwen/qwen3.6-27b",
        label="qwen3.6-27b",
    ),
    CandidateSpec(
        pool_model="pool/gpt-5.5",
        endpoint_model="openai/openai/gpt-5.5",
        label="gpt-5.5",
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Fixed evaluation JSONL")
    parser.add_argument(
        "--polar-config",
        type=Path,
        required=True,
        help="Rendered Polar YAML from the allocation that hosts the rollout service",
    )
    parser.add_argument("--rollout-url", help="Override polar_rollout_url from the YAML")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-tasks",
        type=int,
        required=True,
        help="Required spend guard; number of JSONL rows in the paired comparison",
    )
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument(
        "--forward-seed-to-pool",
        action="store_true",
        help="Also pass the pair seed as model_kwargs.seed; use only if both endpoints support it",
    )
    parser.add_argument(
        "--i-understand-eval-only",
        action="store_true",
        help="Required acknowledgement that this bypasses the trainable Router",
    )
    args = parser.parse_args(argv)
    if not args.i_understand_eval_only:
        parser.error("--i-understand-eval-only is required")
    if not _RUN_ID_RE.fullmatch(args.run_id):
        parser.error("--run-id must match [A-Za-z0-9][A-Za-z0-9._-]{0,79}")
    for name in ("start_index", "seed"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("max_tasks", "replicates", "max_concurrency"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("poll_seconds", "request_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_eval_slice(path: Path, *, start_index: int, max_tasks: int) -> list[EvalItem]:
    if not path.is_file():
        raise ValueError(f"evaluation data does not exist: {path}")
    items: list[EvalItem] = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index < start_index:
                continue
            if len(items) >= max_tasks:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at dataset row {index}: {exc}") from exc
            if not isinstance(row, dict) or "prompt" not in row:
                raise ValueError(f"dataset row {index} must be an object with prompt")
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"dataset row {index} metadata must be an object")
            items.append(EvalItem(index, row["prompt"], dict(metadata)))
    if len(items) != max_tasks:
        raise ValueError(
            f"requested {max_tasks} tasks at start index {start_index}, found {len(items)}"
        )
    return items


def load_rendered_config(
    path: Path,
    *,
    rollout_url: str | None,
    max_concurrency: int,
) -> tuple[SimpleNamespace, Any]:
    if not path.is_file():
        raise ValueError(f"rendered Polar config does not exist: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("rendered Polar config must be a mapping")
    document = dict(document)
    if rollout_url:
        document["polar_rollout_url"] = rollout_url.rstrip("/")
    document.update(
        {
            "rollout_batch_size": max_concurrency,
            "n_samples_per_prompt": 1,
            "update_weights_interval": 1,
            "polar_max_async_level": 1,
        }
    )
    args = SimpleNamespace(**document)
    return args, resolve_polar_slime_config(args)


def validate_candidates(config: Any, candidates: Iterable[CandidateSpec]) -> None:
    agent = config.task_template.get("agent")
    settings = agent.get("settings") if isinstance(agent, dict) else None
    pool = settings.get("model_pool") if isinstance(settings, dict) else None
    values = list(pool.values()) if isinstance(pool, dict) else list(pool or [])
    configured = [item.get("model") for item in values if isinstance(item, dict)]
    for candidate in candidates:
        if configured.count(candidate.pool_model) != 1:
            raise ValueError(
                f"candidate {candidate.pool_model!r} must occur exactly once in model_pool"
            )


def pair_seed(seed: int, dataset_index: int, replicate: int) -> int:
    material = f"{seed}\0{dataset_index}\0{replicate}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def make_work_items(
    items: list[EvalItem],
    *,
    candidates: tuple[CandidateSpec, ...],
    replicates: int,
    seed: int,
) -> list[WorkItem]:
    task_order = list(items)
    random.Random(seed).shuffle(task_order)
    result: list[WorkItem] = []
    for replicate in range(replicates):
        for item in task_order:
            ordered_candidates = list(candidates)
            if pair_seed(seed, item.dataset_index, replicate) & 1:
                ordered_candidates.reverse()
            for candidate in ordered_candidates:
                result.append(
                    WorkItem(
                        item=item,
                        candidate=candidate,
                        replicate=replicate,
                        pair_seed=pair_seed(seed, item.dataset_index, replicate),
                    )
                )
    return result


def build_payload(
    work: WorkItem,
    *,
    args: Any,
    config: Any,
    run_id: str,
    data_sha256: str,
    forward_seed_to_pool: bool,
) -> dict[str, Any]:
    sample = SimpleNamespace(
        prompt=work.item.prompt,
        metadata=work.item.metadata,
        group_index=work.item.dataset_index,
        index=work.item.dataset_index,
    )
    prompt = prompt_to_instruction_text(work.item.prompt)
    instruction = render_instruction(
        args=args,
        config=config,
        sample=sample,
        prompt_text=prompt,
        rollout_id=0,
        task_position=work.item.dataset_index,
        num_rollouts=1,
    )
    payload = render_task_payload(
        args=args,
        config=config,
        sample=sample,
        instruction=instruction,
        rollout_id=0,
        task_position=work.item.dataset_index,
        num_rollouts=1,
        is_eval=True,
    )
    payload["task_id"] = (
        f"spilot-forced-eval-{run_id}-{work.candidate.slug}-"
        f"{work.item.dataset_index}-r{work.replicate}"
    )
    payload["num_samples"] = 1
    payload["early_stop_min_usable_sessions"] = 1
    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("rendered task metadata must be a mapping")
    metadata.update(
        {
            "forced_route_eval": True,
            "dataset_sha256": data_sha256,
            "dataset_index": work.item.dataset_index,
            "eval_seed": work.pair_seed,
            "eval_replicate": work.replicate,
            "candidate_model": work.candidate.pool_model,
            "candidate_endpoint_model": work.candidate.endpoint_model,
        }
    )

    agent = payload.get("agent")
    if not isinstance(agent, dict) or agent.get("harness") != "spilot_router":
        raise ValueError("forced-route eval requires agent.harness=spilot_router")
    # No Router request should exist, but use an intentionally unroutable
    # identity so a future regression fails closed instead of touching an
    # actor endpoint.
    agent["model_name"] = "eval-only/forced-route-no-actor"
    settings = agent.setdefault("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("rendered agent.settings must be a mapping")
    settings["max_pool_calls"] = 1
    settings["sampling_seed"] = work.pair_seed
    settings["forced_route_eval"] = {
        "enabled": True,
        "acknowledgement": EVAL_ONLY_ACK,
        "candidate_model": work.candidate.pool_model,
    }
    if forward_seed_to_pool:
        pool_kwargs = settings.setdefault("pool_model_kwargs", {})
        if not isinstance(pool_kwargs, dict):
            raise ValueError("agent.settings.pool_model_kwargs must be a mapping")
        if "seed" in pool_kwargs and pool_kwargs["seed"] != work.pair_seed:
            raise ValueError("pool_model_kwargs.seed conflicts with paired eval seed")
        pool_kwargs["seed"] = work.pair_seed
    environment = agent.setdefault("env", {})
    if not isinstance(environment, dict):
        raise ValueError("rendered agent.env must be a mapping")
    environment[_AGENT_ACK_ENV] = EVAL_ONLY_ACK

    payload["builder"] = {
        "strategy": _BUILDER,
        "config": {"acknowledgement": EVAL_ONLY_ACK},
    }
    evaluator = payload.get("evaluator")
    if not isinstance(evaluator, dict) or evaluator.get("strategy") != "spilot_harbor":
        raise ValueError("forced-route eval requires the spilot_harbor evaluator")
    evaluator_config = evaluator.setdefault("config", {})
    if not isinstance(evaluator_config, dict):
        raise ValueError("evaluator.config must be a mapping")
    evaluator_config["require_valid_action"] = True
    return payload


def _finite_reward(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def result_row(work: WorkItem, task_id: str, task_status: dict[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {
        "dataset_index": work.item.dataset_index,
        "task_name": work.item.task_name,
        "replicate": work.replicate,
        "pair_seed": work.pair_seed,
        "candidate_model": work.candidate.pool_model,
        "candidate_endpoint_model": work.candidate.endpoint_model,
        "task_id": task_id,
        "task_status": task_status.get("status", "unknown"),
        "valid": False,
        "reward": 0.0,
    }
    results = task_status.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        base["error"] = "task did not return exactly one session result"
        return base
    session = results[0]
    trajectory = session.get("trajectory")
    timing = session.get("timing") if isinstance(session.get("timing"), dict) else {}
    if not isinstance(trajectory, dict):
        base["error"] = "session trajectory is missing"
        return base
    trajectory_metadata = trajectory.get("metadata")
    trajectory_metadata = trajectory_metadata if isinstance(trajectory_metadata, dict) else {}
    evaluation = trajectory_metadata.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    router = evaluation.get("spilot_router")
    router = router if isinstance(router, dict) else {}
    traces = trajectory.get("traces")
    integrity_errors: list[str] = []
    if trajectory_metadata.get("builder") != "spilot_forced_eval":
        integrity_errors.append("unexpected builder")
    if trajectory_metadata.get("eval_only") is not True:
        integrity_errors.append("missing trajectory eval_only marker")
    if traces != []:
        integrity_errors.append("forced evaluation emitted trainable traces")
    if router.get("eval_only") is not True or router.get("actor_invoked") is not False:
        integrity_errors.append("missing no-actor result marker")
    if router.get("forced_route_acknowledgement") != EVAL_ONLY_ACK:
        integrity_errors.append("missing forced-route result acknowledgement")
    if router.get("forced_candidate_model") != work.candidate.pool_model:
        integrity_errors.append("forced candidate result mismatch")
    actions = router.get("actions")
    calls = router.get("calls")
    if not isinstance(actions, list) or len(actions) != 1:
        integrity_errors.append("expected one forced ROUTE action")
    if not isinstance(calls, list) or len(calls) != 1:
        integrity_errors.append("expected one candidate call")
    if isinstance(actions, list) and actions:
        action = actions[0] if isinstance(actions[0], dict) else {}
        if action.get("action") != "ROUTE" or action.get("valid") is not True:
            integrity_errors.append("forced ROUTE action is invalid")
    call = calls[0] if isinstance(calls, list) and calls and isinstance(calls[0], dict) else {}
    if call.get("model") != work.candidate.pool_model:
        integrity_errors.append("candidate call model mismatch")
    if router.get("action_valid") is not True or router.get("submitted") is not True:
        integrity_errors.append("forced route was not valid and submitted")
    if router.get("termination_reason") != "m0_auto_submit":
        integrity_errors.append("forced route did not auto-submit")
    reported_reward = _finite_reward(evaluation.get("reward"))
    session_status = str(session.get("status", "unknown"))
    valid = not integrity_errors and session_status == "COMPLETED" and reported_reward is not None
    base.update(
        {
            "session_id": session.get("session_id"),
            "session_status": session_status,
            "valid": valid,
            "reward": reported_reward if valid else 0.0,
            "reported_reward": reported_reward,
            "harbor_outcome_reward": _finite_reward(evaluation.get("harbor_outcome_reward")),
            "router_action_valid": router.get("action_valid"),
            "router_submitted": router.get("submitted"),
            "termination_reason": router.get("termination_reason"),
            "pool_status": call.get("status"),
            "pool_return_code": call.get("return_code"),
            "pool_timed_out": call.get("timed_out"),
            "pool_duration_ms": call.get("duration_ms"),
            "e2e_ms": timing.get("e2e_ms"),
            "run_ms": timing.get("run_ms"),
            "eval_ms": timing.get("eval_ms"),
            "error": session.get("error"),
            "integrity_errors": integrity_errors,
        }
    )
    return base


async def submit_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    rollout_url: str,
    token: str,
    poll_seconds: float,
    work: WorkItem,
    payload: dict[str, Any],
) -> dict[str, Any]:
    task_id = str(payload["task_id"])
    submitted = False
    try:
        async with semaphore:
            response = await client.post(
                f"{rollout_url}/rollout/task/submit",
                json=payload,
                headers={_CONTROL_TOKEN_HEADER: token},
            )
            response.raise_for_status()
            submitted = True
            returned_id = response.json().get("task_id")
            if returned_id != task_id:
                raise ValueError(f"rollout server returned unexpected task id {returned_id!r}")
            while True:
                await asyncio.sleep(poll_seconds)
                status_response = await client.get(f"{rollout_url}/rollout/task/{task_id}")
                status_response.raise_for_status()
                status = status_response.json()
                if status.get("status") in {"completed", "failed"}:
                    return result_row(work, task_id, status)
    except asyncio.CancelledError:
        if submitted:
            try:
                await client.delete(
                    f"{rollout_url}/rollout/task/{task_id}",
                    params={"register_if_missing": "true"},
                    headers={_CONTROL_TOKEN_HEADER: token},
                )
            except Exception:
                pass
        raise
    except Exception as exc:
        if submitted:
            try:
                await client.delete(
                    f"{rollout_url}/rollout/task/{task_id}",
                    params={"register_if_missing": "true"},
                    headers={_CONTROL_TOKEN_HEADER: token},
                )
            except Exception:
                pass
        return {
            "dataset_index": work.item.dataset_index,
            "task_name": work.item.task_name,
            "replicate": work.replicate,
            "pair_seed": work.pair_seed,
            "candidate_model": work.candidate.pool_model,
            "candidate_endpoint_model": work.candidate.endpoint_model,
            "task_id": task_id,
            "task_status": "submission_error",
            "valid": False,
            "reward": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def summarize(rows: list[dict[str, Any]], candidates: Iterable[CandidateSpec]) -> dict[str, Any]:
    candidates = tuple(candidates)
    if len(candidates) != 2:
        raise ValueError("paired forced-route summary requires exactly two candidates")
    by_candidate: dict[str, Any] = {}
    for candidate in candidates:
        selected = [row for row in rows if row["candidate_model"] == candidate.pool_model]
        valid = [row for row in selected if row.get("valid") is True]
        e2e = [float(row["e2e_ms"]) for row in selected if _finite_reward(row.get("e2e_ms")) is not None]
        rewards = [float(row.get("reward", 0.0)) for row in selected]
        by_candidate[candidate.pool_model] = {
            "endpoint_model": candidate.endpoint_model,
            "count": len(selected),
            "valid_count": len(valid),
            "error_count": len(selected) - len(valid),
            "reward_sum": sum(rewards),
            "reward_mean": sum(rewards) / len(rewards) if rewards else None,
            "pool_status_counts": _counts(row.get("pool_status") for row in selected),
            "session_status_counts": _counts(row.get("session_status") for row in selected),
            "e2e_ms": {
                "mean": statistics.fmean(e2e) if e2e else None,
                "p50": _percentile(e2e, 0.50),
                "p95": _percentile(e2e, 0.95),
                "max": max(e2e) if e2e else None,
            },
        }

    paired: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        paired.setdefault((int(row["dataset_index"]), int(row["replicate"])), {})[
            str(row["candidate_model"])
        ] = row
    first, second = candidates
    comparable = [
        pair
        for pair in paired.values()
        if first.pool_model in pair and second.pool_model in pair
    ]
    deltas = [
        float(pair[second.pool_model]["reward"]) - float(pair[first.pool_model]["reward"])
        for pair in comparable
    ]
    return {
        "candidate_metrics": by_candidate,
        "paired": {
            "pair_count": len(comparable),
            "delta_definition": f"{second.pool_model} - {first.pool_model}",
            "mean_reward_delta": statistics.fmean(deltas) if deltas else None,
            "second_wins": sum(delta > 0 for delta in deltas),
            "first_wins": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
        },
    }


def _counts(values: Iterable[object]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value) if value is not None else "<missing>"
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def async_main(args: argparse.Namespace) -> int:
    token = os.environ.get(_CONTROL_TOKEN_ENV, "").strip()
    if not token:
        raise ValueError(f"{_CONTROL_TOKEN_ENV} is required")
    items = load_eval_slice(
        args.data,
        start_index=args.start_index,
        max_tasks=args.max_tasks,
    )
    config_args, config = load_rendered_config(
        args.polar_config,
        rollout_url=args.rollout_url,
        max_concurrency=args.max_concurrency,
    )
    validate_candidates(config, DEFAULT_CANDIDATES)
    data_sha = sha256_file(args.data)
    config_sha = sha256_file(args.polar_config)
    work_items = make_work_items(
        items,
        candidates=DEFAULT_CANDIDATES,
        replicates=args.replicates,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "eval_only": True,
        "actor_training": False,
        "actor_invoked": False,
        "acknowledgement": EVAL_ONLY_ACK,
        "run_id": args.run_id,
        "data_path": str(args.data.resolve()),
        "data_sha256": data_sha,
        "polar_config_path": str(args.polar_config.resolve()),
        "polar_config_sha256": config_sha,
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "rollout_url": config.rollout_server_url,
        "start_index": args.start_index,
        "max_tasks": args.max_tasks,
        "replicates": args.replicates,
        "seed": args.seed,
        "forward_seed_to_pool": args.forward_seed_to_pool,
        "max_concurrency": args.max_concurrency,
        "dataset_indices": [item.dataset_index for item in items],
        "candidates": [
            {
                "pool_model": candidate.pool_model,
                "endpoint_model": candidate.endpoint_model,
                "label": candidate.label,
            }
            for candidate in DEFAULT_CANDIDATES
        ],
        "expected_result_count": len(work_items),
    }
    write_json(args.output_dir / "manifest.json", manifest)

    payloads = [
        build_payload(
            work,
            args=config_args,
            config=config,
            run_id=args.run_id,
            data_sha256=data_sha,
            forward_seed_to_pool=args.forward_seed_to_pool,
        )
        for work in work_items
    ]
    timeout = httpx.Timeout(args.request_timeout)
    limits = httpx.Limits(
        max_connections=args.max_concurrency,
        max_keepalive_connections=args.max_concurrency,
    )
    semaphore = asyncio.Semaphore(args.max_concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
        rows = await asyncio.gather(
            *(
                submit_one(
                    client,
                    semaphore,
                    rollout_url=config.rollout_server_url,
                    token=token,
                    poll_seconds=args.poll_seconds,
                    work=work,
                    payload=payload,
                )
                for work, payload in zip(work_items, payloads, strict=True)
            )
        )
    rows.sort(key=lambda row: (row["dataset_index"], row["replicate"], row["candidate_model"]))
    write_jsonl(args.output_dir / "results.jsonl", rows)
    summary = {**manifest, **summarize(rows, DEFAULT_CANDIDATES)}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["candidate_metrics"], indent=2, sort_keys=True))
    print(json.dumps(summary["paired"], indent=2, sort_keys=True))
    return 0 if all(row.get("valid") is True for row in rows) else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"forced-route eval error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
