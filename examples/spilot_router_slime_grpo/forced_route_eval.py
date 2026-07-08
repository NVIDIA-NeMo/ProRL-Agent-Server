#!/usr/bin/env python3
"""Run a paired, eval-only forced-route comparison through Polar.

This command submits the Cartesian product of a fixed TMax JSONL slice and the
two configured SPilot pool candidates, plus an optional Qwen3.5-9B baseline.
Every task executes exactly one frozen
mini-SWE candidate and auto-submits to the existing ``spilot_harbor``
evaluator.  It never starts Slime, loads an actor checkpoint, calls the Router,
or emits trainable trajectory traces.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import fcntl
import hashlib
from importlib import metadata as importlib_metadata
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
_OWNER_FD_ENV = "SPILOT_FORCED_EVAL_OWNER_FD"
_AGENT_ACK_ENV = "SPILOT_FORCED_ROUTE_EVAL_ACK"
_BUILDER = "polar.trajectory.builder.spilot_forced_eval:SpilotForcedEvalBuilder"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_FORCED_EVAL_ROUTER_TIMEOUT_SECONDS = 180
_FORCED_EVAL_RESERVE_EVALUATOR_SECONDS = 300
_FORCED_EVAL_DEADLINE_MARGIN_SECONDS = 5
_FORCED_EVAL_ADMISSION_WAIT_SECONDS = 300
_REPO_ROOT = Path(__file__).resolve().parents[2]
_IMPLEMENTATION_FILES = (
    "examples/spilot_router_slime_grpo/forced_route_eval.py",
    "examples/spilot_router_slime_grpo/run_forced_route_eval.py",
    "examples/spilot_router_slime_grpo/run_forced_route_eval.sh",
    "examples/spilot_router_slime_grpo/submit_forced_route_eval.sh",
    "examples/spilot_router_slime_grpo/serve_tokenizer.py",
    "examples/spilot_router_slime_grpo/polar_config.yaml",
    "src/polar/agent/presets/spilot_router.py",
    "src/polar/agent/presets/spilot_router_runner.py",
    "src/polar/agent/presets/spilot_forced_route_eval_runner.py",
    "src/slime_bridge/config.py",
)
_IMPLEMENTATION_TREES = (
    "src/polar/agent",
    "src/polar/config",
    "src/polar/gateway",
    "src/polar/rollout",
    "src/polar/runtime",
    "src/polar/trajectory",
    "src/slime_bridge",
)
_IMPLEMENTATION_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".json", ".jinja", ".sh"})
_SNAPSHOT_TREES = ("src", "examples/spilot_router_slime_grpo")
_TOKENIZER_ASSETS = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
_DEPENDENCY_DISTRIBUTIONS = (
    "fastapi",
    "httpx",
    "pydantic",
    "PyYAML",
    "transformers",
    "uvicorn",
)
_IDENTITY_METADATA_FIELDS = (
    "forced_eval_plan_sha256",
    "forced_eval_work_sha256",
    "dataset_row_sha256",
)


class PendingWorkError(RuntimeError):
    """Control-plane failure that must remain pending, not become benchmark data."""


class TaskIdentityError(RuntimeError):
    """A terminal remote task does not belong to the expected immutable work item."""


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
QWEN35_BASELINE_CANDIDATE = CandidateSpec(
    pool_model="pool/qwen3.5-9b-baseline",
    endpoint_model="nvidia/qwen/qwen3.5-9b",
    label="qwen3.5-9b-baseline",
)


def selected_candidates(*, include_qwen35_baseline: bool) -> tuple[CandidateSpec, ...]:
    return DEFAULT_CANDIDATES + (
        (QWEN35_BASELINE_CANDIDATE,) if include_qwen35_baseline else ()
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
    parser.add_argument(
        "--semantic-identity",
        type=Path,
        required=True,
        help="Secret-free allocation identity produced before child services start",
    )
    parser.add_argument("--rollout-url", help="Override polar_rollout_url from the YAML")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--allocation-attempt-id",
        required=True,
        help="Launcher-generated identity for teardown verification of this allocation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing output only when its immutable semantic plan matches",
    )
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
    parser.add_argument("--control-plane-retry-attempts", type=int, default=5)
    parser.add_argument("--control-plane-retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument(
        "--max-paid-attempts-per-work",
        type=int,
        default=1,
        help=(
            "Global paid-attempt cap per task/candidate work item; values above 1 "
            "explicitly opt in to potentially paying for a full candidate retry"
        ),
    )
    parser.add_argument(
        "--allow-ambiguous-paid-retry",
        action="store_true",
        help=(
            "Explicitly authorize a fresh allocation to abandon an ambiguous prior "
            "submission and spend another paid attempt"
        ),
    )
    parser.add_argument("--pool-timeout-seconds", type=int, default=1200)
    parser.add_argument("--runner-total-timeout-seconds", type=int, default=3000)
    parser.add_argument(
        "--forward-seed-to-pool",
        action="store_true",
        help="Also pass the pair seed as model_kwargs.seed; use only if both endpoints support it",
    )
    parser.add_argument(
        "--include-qwen35-baseline",
        action="store_true",
        help="Also evaluate NVIDIA Qwen3.5-9B on the identical task/seed matrix",
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
    if not re.fullmatch(r"[0-9a-f]{64}", args.allocation_attempt_id):
        parser.error("--allocation-attempt-id must be a lowercase SHA-256 digest")
    for name in ("start_index", "seed"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in (
        "max_tasks",
        "replicates",
        "max_concurrency",
        "pool_timeout_seconds",
        "runner_total_timeout_seconds",
        "control_plane_retry_attempts",
        "max_paid_attempts_per_work",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "poll_seconds",
        "request_timeout",
        "control_plane_retry_backoff_seconds",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    minimum_runner_total = (
        args.pool_timeout_seconds
        + _FORCED_EVAL_ROUTER_TIMEOUT_SECONDS
        + _FORCED_EVAL_RESERVE_EVALUATOR_SECONDS
        + _FORCED_EVAL_DEADLINE_MARGIN_SECONDS
        + _FORCED_EVAL_ADMISSION_WAIT_SECONDS
    )
    if args.runner_total_timeout_seconds < minimum_runner_total:
        parser.error(
            "--runner-total-timeout-seconds must cover pool + Router + evaluator "
            f"reserve + deadline margin ({args.runner_total_timeout_seconds} < "
            f"{minimum_runner_total})"
        )
    return args


def validate_timeout_contract(config: Any, args: argparse.Namespace) -> dict[str, float]:
    agent = config.task_template.get("agent")
    settings = agent.get("settings") if isinstance(agent, dict) else None
    if not isinstance(settings, dict):
        raise ValueError("forced-route eval requires SPilot agent settings")
    names = (
        "pool_timeout_seconds",
        "router_timeout_seconds",
        "total_timeout_seconds",
        "reserve_evaluator_seconds",
        "deadline_margin_seconds",
        "pool_episode_admission_wait_budget_seconds",
    )
    values: dict[str, float] = {}
    for name in names:
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"agent.settings.{name} must be numeric")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError(f"agent.settings.{name} must be finite and non-negative")
        values[name] = parsed
    if settings.get("max_pool_calls") != 1:
        raise ValueError("forced-route eval timeout contract requires max_pool_calls=1")
    if settings.get("pool_episode_admission_enabled") is not True:
        raise ValueError("forced-route paid calls require pool episode admission")
    if values["pool_episode_admission_wait_budget_seconds"] <= 0:
        raise ValueError("forced-route admission wait budget must be positive")
    if values["pool_timeout_seconds"] != float(args.pool_timeout_seconds):
        raise ValueError("rendered pool timeout does not match --pool-timeout-seconds")
    if values["total_timeout_seconds"] != float(args.runner_total_timeout_seconds):
        raise ValueError(
            "rendered runner total timeout does not match --runner-total-timeout-seconds"
        )
    minimum_total = (
        values["pool_timeout_seconds"]
        + values["router_timeout_seconds"]
        + values["reserve_evaluator_seconds"]
        + values["deadline_margin_seconds"]
        + values["pool_episode_admission_wait_budget_seconds"]
    )
    if values["total_timeout_seconds"] < minimum_total:
        raise ValueError(
            "forced-route runner total timeout must cover pool + Router + evaluator "
            f"reserve + deadline margin ({values['total_timeout_seconds']:g} < "
            f"{minimum_total:g})"
        )
    outer_agent_timeout = config.eval_agent_timeout
    if outer_agent_timeout is None or float(outer_agent_timeout) < values["total_timeout_seconds"]:
        raise ValueError(
            "forced-route outer eval agent timeout must cover the runner total timeout"
        )
    values["outer_agent_timeout_seconds"] = float(outer_agent_timeout)
    return values


def sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _tree_content_manifest(
    root: Path,
    *,
    allowed_suffixes: frozenset[str] | None = None,
) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"content tree does not exist: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            entries.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "target": os.readlink(path),
                }
            )
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unsupported content-tree entry: {path}")
        if allowed_suffixes is not None and path.suffix not in allowed_suffixes:
            continue
        stat = path.stat()
        entries.append(
            {
                "path": relative,
                "type": "file",
                "size": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise ValueError(f"content tree has no hashable files: {root}")
    return {
        "root": str(root.resolve()),
        "entries": entries,
        "sha256": _canonical_sha256(entries),
    }


def build_implementation_manifest(repo_root: Path = _REPO_ROOT) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in _IMPLEMENTATION_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"declared forced-eval implementation file is missing: {path}")
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    trees: list[dict[str, Any]] = []
    for relative in _IMPLEMENTATION_TREES:
        tree = _tree_content_manifest(
            repo_root / relative,
            allowed_suffixes=_IMPLEMENTATION_SUFFIXES,
        )
        trees.append(
            {
                "path": relative,
                "entries": tree["entries"],
                "sha256": tree["sha256"],
            }
        )
    payload = {"files": files, "trees": trees}
    return {**payload, "sha256": _canonical_sha256(payload)}


def build_task_asset_manifest(items: Iterable[EvalItem]) -> dict[str, Any]:
    file_cache: dict[Path, tuple[int, str]] = {}
    tree_cache: dict[Path, dict[str, Any]] = {}
    tasks: list[dict[str, Any]] = []
    for item in items:
        sif_value = str(item.metadata.get("sif_path", "")).strip()
        tests_value = str(item.metadata.get("tests_dir", "")).strip()
        if not sif_value or not tests_value:
            raise ValueError(f"dataset row {item.dataset_index} requires sif_path and tests_dir")
        sif_path = Path(sif_value).expanduser().resolve()
        tests_dir = Path(tests_value).expanduser().resolve()
        if not sif_path.is_file():
            raise ValueError(f"dataset row {item.dataset_index} SIF is missing: {sif_path}")
        if not tests_dir.is_dir():
            raise ValueError(
                f"dataset row {item.dataset_index} verifier tree is missing: {tests_dir}"
            )
        if sif_path not in file_cache:
            file_cache[sif_path] = (sif_path.stat().st_size, sha256_file(sif_path))
        if tests_dir not in tree_cache:
            tree_cache[tests_dir] = _tree_content_manifest(tests_dir)
        sif_size, sif_hash = file_cache[sif_path]
        tests_tree = tree_cache[tests_dir]
        tasks.append(
            {
                "dataset_index": item.dataset_index,
                "dataset_row_sha256": dataset_row_sha256(item),
                "sif_path": str(sif_path),
                "sif_size": sif_size,
                "sif_sha256": sif_hash,
                "tests_dir": str(tests_dir),
                "tests_tree_sha256": tests_tree["sha256"],
                "tests_entries": tests_tree["entries"],
            }
        )
    return {"tasks": tasks, "sha256": _canonical_sha256(tasks)}


def _normalize_semantic_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_semantic_config(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_normalize_semantic_config(item) for item in value]
    if isinstance(value, str):
        normalized = re.sub(
            r"/tmp/polar-forced-eval-[^/\s\"']+",
            "<FORCED_EVAL_JOB_LOCAL_ROOT>",
            value,
        )
        return re.sub(
            r"http://127\.0\.0\.1:\d+",
            "<FORCED_EVAL_LOOPBACK_URL>",
            normalized,
        )
    return value


def build_semantic_config_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"rendered Polar config does not exist: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("rendered Polar config must be a mapping")
    document = dict(document)
    for transport_key in (
        "polar_rollout_url",
        "polar_gateway_url",
        "polar_callback_host",
    ):
        document.pop(transport_key, None)
    normalized = _normalize_semantic_config(document)
    return {
        "document": normalized,
        "sha256": _canonical_sha256(normalized),
    }


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


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"identity file does not exist: {path}")
    return {"size": path.stat().st_size, "sha256": sha256_file(path)}


def _identity_tree_paths(root: Path) -> list[Path]:
    ignored_names = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}
    return [
        path
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if not any(part in ignored_names for part in path.relative_to(root).parts)
        and path.suffix not in {".pyc", ".pyo"}
    ]


def _identity_tree(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"identity tree does not exist: {root}")
    before = _identity_tree_paths(root)
    entries: list[dict[str, Any]] = []
    total_size = 0
    for path in before:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries.append({"path": relative, "type": "symlink", "target": os.readlink(path)})
        elif path.is_dir():
            continue
        elif path.is_file():
            identity = _identity_file(path)
            total_size += int(identity["size"])
            entries.append({"path": relative, "type": "file", **identity})
        else:
            raise ValueError(f"unsupported identity-tree entry: {path}")
    after = _identity_tree_paths(root)
    before_names = [path.relative_to(root).as_posix() for path in before]
    after_names = [path.relative_to(root).as_posix() for path in after]
    if before_names != after_names:
        raise ValueError(f"content tree changed while hashing: {root}")
    return {
        "sha256": _canonical_sha256(entries),
        "entry_count": len(entries),
        "total_file_bytes": total_size,
    }


def _source_snapshot_identity(root: Path) -> dict[str, Any]:
    trees = [{"path": relative, **_identity_tree(root / relative)} for relative in _SNAPSHOT_TREES]
    return {"trees": trees, "sha256": _canonical_sha256(trees)}


def _dependency_identity() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in _DEPENDENCY_DISTRIBUTIONS:
        try:
            result[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def load_semantic_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"semantic identity does not exist: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"semantic identity is invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("semantic identity must be a JSON object")
    semantic = document.get("semantic")
    if not isinstance(semantic, dict):
        raise ValueError("semantic identity has no semantic document")
    if document.get("semantic_sha256") != _canonical_sha256(semantic):
        raise ValueError("semantic identity digest does not match its document")
    return document


def verify_semantic_identity(document: dict[str, Any]) -> None:
    semantic = document.get("semantic")
    verification = document.get("verification")
    if not isinstance(semantic, dict) or not isinstance(verification, dict):
        raise ValueError("semantic identity is missing semantic or verification data")
    if document.get("semantic_sha256") != _canonical_sha256(semantic):
        raise ValueError("semantic identity digest does not match its document")
    snapshot_root = Path(str(verification["source_snapshot_root"]))
    if _source_snapshot_identity(snapshot_root) != semantic.get("source_snapshot"):
        raise ValueError("source snapshot content identity changed")
    tokenizer_path = Path(str(verification["tokenizer_path"]))
    tokenizer = semantic.get("tokenizer")
    expected_tokenizer = tokenizer.get("assets") if isinstance(tokenizer, dict) else None
    actual_tokenizer = {
        name: _identity_file(tokenizer_path / name) for name in _TOKENIZER_ASSETS
    }
    if actual_tokenizer != expected_tokenizer:
        raise ValueError("tokenizer content identity changed")
    expected_trees = semantic.get("runtime_trees")
    actual_trees = {
        "mini_swe_agent_runtime": _identity_tree(
            Path(str(verification["mini_swe_agent_runtime_path"]))
        ),
        "agent_cli_opt_node": _identity_tree(
            Path(str(verification["agent_cli_opt_node_path"]))
        ),
    }
    if actual_trees != expected_trees:
        raise ValueError("mini-SWE or agent-cli runtime identity changed")
    expected_files = semantic.get("runtime_files")
    if not isinstance(expected_files, dict):
        raise ValueError("semantic runtime file identity is invalid")
    actual_files = {
        "train_container": _identity_file(Path(str(verification["train_container_path"]))),
        "python_executable": _identity_file(
            Path(str(verification["python_executable_path"]))
        ),
    }
    apptainer = verification.get("apptainer_executable_path")
    if apptainer is not None:
        actual_files["apptainer_executable"] = _identity_file(Path(str(apptainer)))
    if actual_files != expected_files:
        raise ValueError("container or executable identity changed")
    if _dependency_identity() != semantic.get("dependencies"):
        raise ValueError("Python dependency identity changed")
    python_identity = semantic.get("python")
    if python_identity != {
        "implementation": sys.implementation.name,
        "version": list(sys.version_info[:3]),
    }:
        raise ValueError("Python runtime identity changed")
    for name in ("topology", "polar_config"):
        path = Path(str(verification[f"{name}_path"]))
        if _identity_file(path) != verification.get(f"{name}_file"):
            raise ValueError(f"rendered {name} content changed")


def dataset_row_sha256(item: EvalItem) -> str:
    return _canonical_sha256(
        {
            "dataset_index": item.dataset_index,
            "prompt": item.prompt,
            "metadata": item.metadata,
        }
    )


def work_sha256(work: WorkItem, *, plan_sha256: str) -> str:
    return _canonical_sha256(
        {
            "plan_sha256": plan_sha256,
            "dataset_index": work.item.dataset_index,
            "dataset_row_sha256": dataset_row_sha256(work.item),
            "replicate": work.replicate,
            "pair_seed": work.pair_seed,
            "candidate_model": work.candidate.pool_model,
            "candidate_endpoint_model": work.candidate.endpoint_model,
        }
    )


def forced_task_id(work: WorkItem, *, run_id: str, plan_sha256: str) -> str:
    fingerprint = work_sha256(work, plan_sha256=plan_sha256)
    return (
        f"spilot-fe-{run_id[:20]}-{work.candidate.slug[:16]}-"
        f"{work.item.dataset_index}-r{work.replicate}-{fingerprint[:24]}"
    )


def build_payload(
    work: WorkItem,
    *,
    args: Any,
    config: Any,
    run_id: str,
    data_sha256: str,
    plan_sha256: str,
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
    task_id = forced_task_id(work, run_id=run_id, plan_sha256=plan_sha256)
    work_fingerprint = work_sha256(work, plan_sha256=plan_sha256)
    row_fingerprint = dataset_row_sha256(work.item)
    payload["task_id"] = task_id
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
            "forced_eval_plan_sha256": plan_sha256,
            "forced_eval_work_sha256": work_fingerprint,
            "dataset_row_sha256": row_fingerprint,
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


def _verify_terminal_task_identity(
    task_id: str,
    task_status: dict[str, Any],
    expected_identity: dict[str, Any],
) -> None:
    reported_task_id = task_status.get("task_id")
    if reported_task_id is not None and reported_task_id != task_id:
        raise TaskIdentityError(
            f"remote task id mismatch: expected {task_id!r}, got {reported_task_id!r}"
        )
    if task_status.get("status") != "completed":
        raise PendingWorkError("only a completed rollout task is a benchmark outcome")
    results = task_status.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise PendingWorkError("completed task has no unique result identity")
    trajectory = results[0].get("trajectory")
    trajectory_metadata = trajectory.get("metadata") if isinstance(trajectory, dict) else None
    task_metadata = (
        trajectory_metadata.get("task_metadata") if isinstance(trajectory_metadata, dict) else None
    )
    if not isinstance(task_metadata, dict):
        raise PendingWorkError("completed task is missing echoed task metadata")
    for field in _IDENTITY_METADATA_FIELDS:
        if task_metadata.get(field) != expected_identity[field]:
            raise TaskIdentityError(
                f"completed task identity field {field!r} does not match immutable plan"
            )


def _validated_benchmark_outcome(
    work: WorkItem,
    task_status: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return audited session/evaluation/router/call or leave the work pending.

    A rollout task reaching its terminal ``completed`` state only means that
    the orchestration pipeline stopped.  It is not itself evidence that a
    candidate episode completed.  In particular, pipeline/session timeouts,
    missing forced-eval acknowledgements, and missing/failed calls must never
    be converted into benchmark reward zero.
    """

    results = task_status.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise PendingWorkError("completed task has no unique candidate session outcome")
    session = results[0]
    session_status = session.get("status")
    if session_status != "COMPLETED":
        raise PendingWorkError(
            f"candidate session ended as {session_status!r}, not a benchmark outcome"
        )
    trajectory = session.get("trajectory")
    if not isinstance(trajectory, dict):
        raise PendingWorkError("completed candidate session has no trajectory")
    trajectory_metadata = trajectory.get("metadata")
    if not isinstance(trajectory_metadata, dict):
        raise PendingWorkError("completed candidate session has no trajectory metadata")
    evaluation = trajectory_metadata.get("evaluation")
    if not isinstance(evaluation, dict):
        raise PendingWorkError("completed candidate session has no evaluator outcome")
    router = evaluation.get("spilot_router")
    if not isinstance(router, dict):
        raise PendingWorkError("completed candidate session has no forced-route outcome")

    if trajectory_metadata.get("builder") != "spilot_forced_eval":
        raise TaskIdentityError("candidate session used an unexpected trajectory builder")
    if trajectory_metadata.get("eval_only") is not True:
        raise TaskIdentityError("candidate session is missing the eval-only marker")
    if trajectory.get("traces") != []:
        raise TaskIdentityError("forced evaluation emitted trainable traces")
    if router.get("eval_only") is not True or router.get("actor_invoked") is not False:
        raise TaskIdentityError("forced evaluation did not prove the actor was bypassed")

    acknowledgement = router.get("forced_route_acknowledgement")
    if acknowledgement is None:
        raise PendingWorkError("forced-route acknowledgement is missing")
    if acknowledgement != EVAL_ONLY_ACK:
        raise TaskIdentityError("forced-route acknowledgement does not match this evaluator")
    forced_candidate = router.get("forced_candidate_model")
    if forced_candidate is None:
        raise PendingWorkError("forced candidate acknowledgement is missing")
    if forced_candidate != work.candidate.pool_model:
        raise TaskIdentityError("forced candidate acknowledgement does not match the work item")

    actions = router.get("actions")
    calls = router.get("calls")
    if not isinstance(actions, list) or not actions:
        raise PendingWorkError("forced ROUTE action is missing")
    if not isinstance(calls, list) or not calls:
        raise PendingWorkError("forced candidate call is missing")
    if len(actions) != 1 or not isinstance(actions[0], dict):
        raise TaskIdentityError("forced evaluation did not emit exactly one ROUTE action")
    if len(calls) != 1 or not isinstance(calls[0], dict):
        raise TaskIdentityError("forced evaluation did not emit exactly one candidate call")
    action = actions[0]
    call = calls[0]
    if action.get("action") != "ROUTE" or action.get("valid") is not True:
        raise TaskIdentityError("forced ROUTE action is invalid")
    if router.get("action_valid") is not True or router.get("submitted") is not True:
        raise TaskIdentityError("forced route was not valid and submitted")
    if router.get("termination_reason") != "m0_auto_submit":
        raise TaskIdentityError("forced route did not auto-submit after the candidate call")
    if call.get("model") != work.candidate.pool_model:
        raise TaskIdentityError("candidate call model does not match the work item")
    if action.get("model_slot") != call.get("slot"):
        raise TaskIdentityError("forced ROUTE action and candidate call slots do not match")
    if call.get("attempted") is not True:
        raise PendingWorkError("candidate call was not explicitly attempted")

    call_status = call.get("status")
    timed_out = call.get("timed_out")
    if call_status == "completed":
        if timed_out is not False or call.get("return_code") != 0:
            raise PendingWorkError("candidate completed-call markers are inconsistent")
    elif call_status in {"timeout", "timed_out"}:
        if timed_out is not True or call.get("failure_kind") != "timeout":
            raise PendingWorkError("candidate timeout is not explicitly attributable to the call")
    else:
        raise PendingWorkError(
            f"candidate call ended as {call_status!r}, not an attributable benchmark outcome"
        )

    if _finite_reward(evaluation.get("reward")) is None:
        raise PendingWorkError("candidate evaluator reward is missing or non-finite")
    if _finite_reward(evaluation.get("harbor_outcome_reward")) is None:
        raise PendingWorkError("candidate harbor outcome reward is missing or non-finite")
    if not isinstance(session.get("session_id"), str) or not session["session_id"]:
        raise PendingWorkError("candidate session id is missing")
    duration = _finite_reward(call.get("duration_ms"))
    if duration is None or duration < 0:
        raise PendingWorkError("candidate call duration is missing or invalid")
    return session, evaluation, router, call


def result_row(
    work: WorkItem,
    task_id: str,
    task_status: dict[str, Any],
    *,
    expected_identity: dict[str, Any],
    allocation_attempt_id: str = "0" * 64,
) -> dict[str, Any]:
    session, evaluation, router, call = _validated_benchmark_outcome(work, task_status)
    base: dict[str, Any] = {
        "dataset_index": work.item.dataset_index,
        "task_name": work.item.task_name,
        "replicate": work.replicate,
        "pair_seed": work.pair_seed,
        "candidate_model": work.candidate.pool_model,
        "candidate_endpoint_model": work.candidate.endpoint_model,
        "task_id": task_id,
        "task_status": task_status.get("status", "unknown"),
        "allocation_attempt_id": allocation_attempt_id,
        **{field: expected_identity[field] for field in _IDENTITY_METADATA_FIELDS},
        "valid": True,
    }
    trajectory = session.get("trajectory")
    timing = session.get("timing") if isinstance(session.get("timing"), dict) else {}
    assert isinstance(trajectory, dict)
    trajectory_metadata = trajectory.get("metadata")
    assert isinstance(trajectory_metadata, dict)
    reported_reward = _finite_reward(evaluation.get("reward"))
    assert reported_reward is not None
    session_status = str(session.get("status", "unknown"))
    base.update(
        {
            "session_id": session.get("session_id"),
            "session_status": session_status,
            "reward": reported_reward,
            "reported_reward": reported_reward,
            "harbor_outcome_reward": _finite_reward(evaluation.get("harbor_outcome_reward")),
            "eval_only": router.get("eval_only"),
            "actor_invoked": router.get("actor_invoked"),
            "forced_route_acknowledgement": router.get("forced_route_acknowledgement"),
            "forced_candidate_model": router.get("forced_candidate_model"),
            "router_action_valid": router.get("action_valid"),
            "router_submitted": router.get("submitted"),
            "termination_reason": router.get("termination_reason"),
            "pool_status": call.get("status"),
            "pool_attempted": call.get("attempted"),
            "pool_return_code": call.get("return_code"),
            "pool_timed_out": call.get("timed_out"),
            "pool_failure_kind": call.get("failure_kind"),
            "pool_duration_ms": call.get("duration_ms"),
            "e2e_ms": timing.get("e2e_ms"),
            "run_ms": timing.get("run_ms"),
            "eval_ms": timing.get("eval_ms"),
            "error": session.get("error"),
            "integrity_errors": [],
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
    retry_attempts: int,
    retry_backoff_seconds: float,
    work: WorkItem,
    payload: dict[str, Any],
    attempt_ledger: AttemptLedger | None = None,
    allocation_attempt_id: str = "0" * 64,
    allow_ambiguous_paid_retry: bool = False,
) -> dict[str, Any]:
    task_id = str(payload["task_id"])
    payload_metadata = payload.get("metadata")
    if not isinstance(payload_metadata, dict):
        raise ValueError("forced-eval payload metadata must be a mapping")
    expected_identity = {field: payload_metadata.get(field) for field in _IDENTITY_METADATA_FIELDS}
    if not all(isinstance(value, str) and value for value in expected_identity.values()):
        raise ValueError("forced-eval payload is missing immutable identity metadata")
    work_id = str(expected_identity["forced_eval_work_sha256"])
    managed_task = False

    async def post_new_attempt(reason: str) -> dict[str, Any]:
        nonlocal managed_task
        attempt_id: int | None = None
        if attempt_ledger is not None:
            attempt_id = attempt_ledger.reserve_not_sent(
                work_id,
                task_id=task_id,
                allocation_attempt_id=allocation_attempt_id,
                reason=reason,
            )
            # Persist ambiguity before entering httpx. A crash after this line
            # can never be mistaken for proof that the POST was unsent.
            attempt_ledger.mark_submit_started(work_id, attempt_id=attempt_id)
        try:
            response = await client.post(
                f"{rollout_url}/rollout/task/submit",
                json=payload,
                headers={_CONTROL_TOKEN_HEADER: token},
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
            if attempt_ledger is not None and attempt_id is not None:
                attempt_ledger.reclaim_not_sent(
                    work_id,
                    attempt_id=attempt_id,
                    reason="httpx_connect_before_send",
                )
            raise
        response.raise_for_status()
        submitted = response.json()
        if not isinstance(submitted, dict):
            raise ValueError("rollout submit response must be a JSON object")
        returned_id = submitted.get("task_id")
        if returned_id != task_id:
            raise TaskIdentityError(
                f"rollout server returned unexpected task id {returned_id!r}"
            )
        if attempt_ledger is not None:
            attempt_ledger.mark_accepted(work_id, task_id=task_id)
        managed_task = True
        return submitted

    try:
        async with semaphore:
            consecutive_failures = 0
            terminal_resubmits = 0
            ambiguous_missing_polls = 0
            while True:
                try:
                    # A process can die after Polar accepted or completed a
                    # task but before its POST response or local result became
                    # durable. GET-first closes both ACK-loss windows.
                    status_response = await client.get(f"{rollout_url}/rollout/task/{task_id}")
                    if status_response.status_code == 404:
                        status: dict[str, Any] | Any = {"status": "missing"}
                    else:
                        status_response.raise_for_status()
                        status = status_response.json()
                        if not isinstance(status, dict):
                            raise ValueError("rollout task status must be a JSON object")
                        managed_task = True

                    state = status.get("status")
                    if state == "missing":
                        active = (
                            attempt_ledger.active_attempt(work_id)
                            if attempt_ledger is not None
                            else None
                        )
                        if (
                            active is not None
                            and active.get("state")
                            in {"submit_started_ambiguous", "accepted_inflight"}
                            and active.get("allocation_attempt_id") == allocation_attempt_id
                        ):
                            ambiguous_missing_polls += 1
                            if ambiguous_missing_polls < retry_attempts:
                                await asyncio.sleep(retry_backoff_seconds)
                                continue
                        if attempt_ledger is not None:
                            can_submit = attempt_ledger.reconcile_missing_remote(
                                work_id,
                                current_allocation_attempt_id=allocation_attempt_id,
                                allow_ambiguous_paid_retry=allow_ambiguous_paid_retry,
                            )
                            if not can_submit:
                                raise PendingWorkError(
                                    f"work {work_id} has no authorized paid attempt remaining"
                                )
                        status = await post_new_attempt("remote_missing")
                        state = status.get("status")
                        ambiguous_missing_polls = 0

                    if state in {"running", "completed", "failed", "cancelled"}:
                        if attempt_ledger is not None:
                            active = attempt_ledger.active_attempt(work_id)
                            if active is not None and active.get("state") != "accepted_inflight":
                                attempt_ledger.mark_accepted(work_id, task_id=task_id)

                    if state in {"failed", "cancelled"}:
                        if attempt_ledger is not None:
                            attempt_ledger.mark_terminal(
                                work_id,
                                outcome=f"remote_{state}",
                            )
                        terminal_resubmits += 1
                        if terminal_resubmits > retry_attempts:
                            raise PendingWorkError(
                                f"remote task {task_id} repeatedly ended as {state!r}; "
                                "leaving it pending"
                            )
                        if attempt_ledger is not None and attempt_ledger.remaining_attempts(
                            work_id
                        ) <= 0:
                            raise PendingWorkError(
                                f"work {work_id} exhausted its global paid-attempt budget"
                            )
                        status = await post_new_attempt(f"remote_{state}")
                        state = status.get("status")

                    if state == "completed":
                        try:
                            _verify_terminal_task_identity(
                                task_id,
                                status,
                                expected_identity,
                            )
                            return result_row(
                                work,
                                task_id,
                                status,
                                expected_identity=expected_identity,
                                allocation_attempt_id=allocation_attempt_id,
                            )
                        except PendingWorkError as exc:
                            # A task-level completion can still be an
                            # orchestration failure (for example a TIMEOUT
                            # session or a missing forced-call record). Replace
                            # the terminal task under the same deterministic id
                            # only after durably reserving another paid attempt.
                            terminal_resubmits += 1
                            if terminal_resubmits > retry_attempts:
                                raise PendingWorkError(
                                    f"remote task {task_id} repeatedly lacked a benchmark "
                                    f"outcome: {exc}"
                                ) from exc
                            if attempt_ledger is not None:
                                attempt_ledger.mark_terminal(
                                    work_id,
                                    outcome=f"completed_without_outcome:{exc}",
                                )
                                if attempt_ledger.remaining_attempts(work_id) <= 0:
                                    raise PendingWorkError(
                                        f"work {work_id} exhausted its global paid-attempt budget"
                                    ) from exc
                            await post_new_attempt(f"completed_without_outcome:{exc}")
                            await asyncio.sleep(retry_backoff_seconds)
                            continue
                    if state == "cancelled":
                        raise PendingWorkError(f"remote task {task_id} remains cancelled")
                    if state != "running":
                        raise ValueError(f"unexpected rollout task state {state!r}")
                    consecutive_failures = 0
                    ambiguous_missing_polls = 0
                    await asyncio.sleep(poll_seconds)
                except TaskIdentityError:
                    raise
                except PendingWorkError:
                    raise
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    consecutive_failures += 1
                    if consecutive_failures >= retry_attempts:
                        raise PendingWorkError(
                            f"task {task_id} remains pending after {retry_attempts} "
                            f"control-plane attempts: {type(exc).__name__}: {exc}"
                        ) from exc
                    await asyncio.sleep(retry_backoff_seconds * min(consecutive_failures, 5))
    except asyncio.CancelledError:
        if managed_task:
            try:
                await client.delete(
                    f"{rollout_url}/rollout/task/{task_id}",
                    params={"register_if_missing": "true"},
                    headers={_CONTROL_TOKEN_HEADER: token},
                )
            except Exception:
                pass
        raise


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def summarize(
    rows: list[dict[str, Any]],
    candidates: Iterable[CandidateSpec],
    *,
    expected_rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidates = tuple(candidates)
    if len(candidates) < 2 or len({candidate.pool_model for candidate in candidates}) != len(
        candidates
    ):
        raise ValueError("forced-route summary requires at least two unique candidates")
    by_candidate: dict[str, Any] = {}
    expected_values = list(expected_rows) if expected_rows is not None else list(rows)
    for candidate in candidates:
        selected = [row for row in rows if row["candidate_model"] == candidate.pool_model]
        expected_selected = [
            row for row in expected_values if row["candidate_model"] == candidate.pool_model
        ]
        valid = [row for row in selected if row.get("valid") is True]
        e2e = [
            float(row["e2e_ms"])
            for row in selected
            if _finite_reward(row.get("e2e_ms")) is not None
        ]
        rewards = [float(row.get("reward", 0.0)) for row in selected]
        by_candidate[candidate.pool_model] = {
            "endpoint_model": candidate.endpoint_model,
            "expected_count": len(expected_selected),
            "count": len(selected),
            "missing_count": len(expected_selected) - len(selected),
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
    expected_paired: dict[tuple[int, int], set[str]] = {}
    for row in expected_values:
        expected_paired.setdefault(
            (int(row["dataset_index"]), int(row["replicate"])), set()
        ).add(str(row["candidate_model"]))

    def comparison(first: CandidateSpec, second: CandidateSpec) -> dict[str, Any]:
        comparable = [
            pair
            for pair in paired.values()
            if first.pool_model in pair and second.pool_model in pair
        ]
        deltas = [
            float(pair[second.pool_model]["reward"])
            - float(pair[first.pool_model]["reward"])
            for pair in comparable
        ]
        expected_pair_count = sum(
            first.pool_model in models and second.pool_model in models
            for models in expected_paired.values()
        )
        return {
            "expected_pair_count": expected_pair_count,
            "pair_count": len(comparable),
            "missing_pair_count": expected_pair_count - len(comparable),
            "delta_definition": f"{second.pool_model} - {first.pool_model}",
            "mean_reward_delta": statistics.fmean(deltas) if deltas else None,
            "second_wins": sum(delta > 0 for delta in deltas),
            "first_wins": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
        }

    by_model = {candidate.pool_model: candidate for candidate in candidates}
    qwen = by_model["pool/qwen3.6-27b"]
    gpt = by_model["pool/gpt-5.5"]
    gpt_vs_qwen = comparison(qwen, gpt)
    paired_comparisons = {"gpt_vs_qwen": gpt_vs_qwen}
    baseline = by_model.get(QWEN35_BASELINE_CANDIDATE.pool_model)
    if baseline is not None:
        paired_comparisons.update(
            {
                "gpt_vs_qwen35_baseline": comparison(baseline, gpt),
                "qwen36_vs_qwen35_baseline": comparison(baseline, qwen),
            }
        )
    return {
        "candidate_metrics": by_candidate,
        # Backward-compatible canonical pool comparison.
        "paired": gpt_vs_qwen,
        "paired_comparisons": paired_comparisons,
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
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _row_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(row["dataset_index"]),
        int(row["replicate"]),
        str(row["candidate_model"]),
    )


def _plan_sha256(plan: dict[str, Any]) -> str:
    return _canonical_sha256(plan)


def _expected_result_fields(
    work: WorkItem,
    task_id: str,
    *,
    plan_sha256: str,
) -> dict[str, Any]:
    return {
        "dataset_index": work.item.dataset_index,
        "task_name": work.item.task_name,
        "replicate": work.replicate,
        "pair_seed": work.pair_seed,
        "candidate_model": work.candidate.pool_model,
        "candidate_endpoint_model": work.candidate.endpoint_model,
        "task_id": task_id,
        "forced_eval_plan_sha256": plan_sha256,
        "forced_eval_work_sha256": work_sha256(work, plan_sha256=plan_sha256),
        "dataset_row_sha256": dataset_row_sha256(work.item),
    }


class OutputDirectoryLock:
    """Process-scoped exclusive lock preventing concurrent snapshot writers."""

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _validate_allocation_owner(output_dir)
        self.path = output_dir / ".forced-eval.lock"
        self._stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._stream.close()
            raise ValueError(
                f"another forced-eval process holds the output lock: {self.path}"
            ) from exc
        self._stream.seek(0)
        self._stream.truncate()
        self._stream.write(f"pid={os.getpid()}\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        if self._stream.closed:
            return
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()

    def __enter__(self) -> OutputDirectoryLock:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _validate_allocation_owner(output_dir: Path) -> None:
    owner_path = output_dir / ".forced-eval.owner.lock"
    inherited = os.environ.get(_OWNER_FD_ENV, "").strip()
    if inherited:
        try:
            descriptor = int(inherited)
            descriptor_stat = os.fstat(descriptor)
            path_stat = owner_path.stat()
        except (OSError, ValueError) as exc:
            raise ValueError("invalid inherited forced-eval owner descriptor") from exc
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise ValueError("inherited forced-eval owner descriptor does not match output")
        return
    if not owner_path.exists():
        return
    with owner_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "forced-eval output is owned by an active allocation; resume there"
            ) from exc
        finally:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


class AttemptLedger:
    """Crash-durable paid-call state machine keyed by semantic work."""

    _ACTIVE_STATES = {
        "reserved_not_sent",
        "submit_started_ambiguous",
        "accepted_inflight",
    }

    def __init__(
        self,
        path: Path,
        *,
        plan_sha256: str,
        work_ids: Iterable[str],
        max_paid_attempts: int,
    ) -> None:
        self.path = path
        if (
            isinstance(max_paid_attempts, bool)
            or not isinstance(max_paid_attempts, int)
            or max_paid_attempts <= 0
        ):
            raise ValueError("max_paid_attempts must be a positive integer")
        expected_ids = sorted(set(work_ids))
        if path.exists():
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid attempt ledger: {exc}") from exc
            if not isinstance(document, dict):
                raise ValueError("attempt ledger must be a JSON object")
            if document.get("schema_version") != 2:
                raise ValueError("attempt ledger schema is not resumable by this evaluator")
            if document.get("plan_sha256") != plan_sha256:
                raise ValueError("attempt ledger belongs to a different semantic plan")
            if document.get("max_paid_attempts_per_work") != max_paid_attempts:
                raise ValueError("attempt ledger paid-attempt policy mismatch")
            works = document.get("works")
            if not isinstance(works, dict) or sorted(works) != expected_ids:
                raise ValueError("attempt ledger work set mismatch")
            for work_id, work in works.items():
                if not isinstance(work, dict):
                    raise ValueError(f"attempt ledger work {work_id!r} must be an object")
                paid = work.get("paid_attempts")
                if (
                    isinstance(paid, bool)
                    or not isinstance(paid, int)
                    or paid < 0
                    or paid > max_paid_attempts
                ):
                    raise ValueError(f"attempt ledger work {work_id!r} has invalid paid count")
                if not isinstance(work.get("completed"), bool):
                    raise ValueError(f"attempt ledger work {work_id!r} has invalid completion")
                attempts = work.get("attempts")
                if not isinstance(attempts, list) or not all(
                    isinstance(attempt, dict) for attempt in attempts
                ):
                    raise ValueError(f"attempt ledger work {work_id!r} has invalid attempts")
                if [attempt.get("attempt_id") for attempt in attempts] != list(
                    range(1, len(attempts) + 1)
                ):
                    raise ValueError(f"attempt ledger work {work_id!r} attempt ids are invalid")
                active = [
                    attempt
                    for attempt in attempts
                    if attempt.get("state") in self._ACTIVE_STATES
                ]
                if len(active) > 1 or (active and active[0] is not attempts[-1]):
                    raise ValueError(f"attempt ledger work {work_id!r} has multiple active attempts")
                valid_states = self._ACTIVE_STATES | {"terminal"}
                if any(attempt.get("state") not in valid_states for attempt in attempts):
                    raise ValueError(f"attempt ledger work {work_id!r} has invalid attempt state")
                if any(
                    not isinstance(attempt.get("charged"), bool)
                    or not isinstance(attempt.get("allocation_attempt_id"), str)
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", str(attempt.get("allocation_attempt_id"))
                    )
                    for attempt in attempts
                ):
                    raise ValueError(f"attempt ledger work {work_id!r} attempt metadata is invalid")
                if sum(attempt["charged"] is True for attempt in attempts) != paid:
                    raise ValueError(f"attempt ledger work {work_id!r} paid count is inconsistent")
                if work["completed"] is True and active:
                    raise ValueError(f"attempt ledger work {work_id!r} is completed but active")
                events = work.get("events")
                if not isinstance(events, list) or not all(
                    isinstance(event, dict) for event in events
                ):
                    raise ValueError(f"attempt ledger work {work_id!r} has invalid events")
            self.document = document
        else:
            self.document = {
                "schema_version": 2,
                "plan_sha256": plan_sha256,
                "max_paid_attempts_per_work": max_paid_attempts,
                "works": {
                    work_id: {
                        "paid_attempts": 0,
                        "completed": False,
                        "attempts": [],
                        "events": [],
                    }
                    for work_id in expected_ids
                },
            }
            self._persist()

    def paid_attempts(self, work_id: str) -> int:
        return int(self.document["works"][work_id]["paid_attempts"])

    def remaining_attempts(self, work_id: str) -> int:
        return max(
            0,
            int(self.document["max_paid_attempts_per_work"])
            - self.paid_attempts(work_id),
        )

    def active_attempt(self, work_id: str) -> dict[str, Any] | None:
        work = self.document["works"].get(work_id)
        if not isinstance(work, dict):
            raise ValueError(f"unknown attempt-ledger work id: {work_id}")
        attempts = work.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError("attempt ledger attempts must be a list")
        if attempts and attempts[-1].get("state") in self._ACTIVE_STATES:
            return attempts[-1]
        return None

    def reserve_not_sent(
        self,
        work_id: str,
        *,
        task_id: str,
        allocation_attempt_id: str,
        reason: str,
    ) -> int:
        work = self.document["works"].get(work_id)
        if not isinstance(work, dict):
            raise ValueError(f"unknown attempt-ledger work id: {work_id}")
        if work.get("completed") is True:
            raise ValueError(f"cannot reserve another paid attempt for completed work {work_id}")
        active = self.active_attempt(work_id)
        if active is not None:
            if active.get("state") == "reserved_not_sent":
                return int(active["attempt_id"])
            raise PendingWorkError(
                f"work {work_id} has outstanding {active.get('state')} attempt; reconcile first"
            )
        current = int(work.get("paid_attempts", 0))
        maximum = int(self.document["max_paid_attempts_per_work"])
        if current >= maximum:
            raise PendingWorkError(
                f"work {work_id} exhausted its global paid-attempt budget ({maximum})"
            )
        attempts = work["attempts"]
        attempt_id = len(attempts) + 1
        attempts.append(
            {
                "attempt_id": attempt_id,
                "task_id": task_id,
                "allocation_attempt_id": allocation_attempt_id,
                "state": "reserved_not_sent",
                "charged": False,
                "reason": reason,
                "terminal_outcome": None,
            }
        )
        self._event(work, "reserved_not_sent", attempt_id=attempt_id, reason=reason)
        self._persist()
        return attempt_id

    def reserve_paid_attempt(
        self,
        work_id: str,
        *,
        task_id: str,
        reason: str,
        allocation_attempt_id: str = "0" * 64,
    ) -> int:
        """Compatibility wrapper: reservation alone proves no POST was sent."""

        return self.reserve_not_sent(
            work_id,
            task_id=task_id,
            allocation_attempt_id=allocation_attempt_id,
            reason=reason,
        )

    def mark_submit_started(self, work_id: str, *, attempt_id: int) -> None:
        work, attempt = self._active(work_id, attempt_id)
        if attempt.get("state") != "reserved_not_sent":
            raise ValueError("submit can start only from reserved_not_sent")
        if self.paid_attempts(work_id) >= int(self.document["max_paid_attempts_per_work"]):
            raise PendingWorkError(f"work {work_id} exhausted its global paid-attempt budget")
        attempt["state"] = "submit_started_ambiguous"
        attempt["charged"] = True
        work["paid_attempts"] = self.paid_attempts(work_id) + 1
        self._event(work, "submit_started_ambiguous", attempt_id=attempt_id)
        self._persist()

    def reclaim_not_sent(self, work_id: str, *, attempt_id: int, reason: str) -> None:
        work, attempt = self._active(work_id, attempt_id)
        state = attempt.get("state")
        if state not in {"reserved_not_sent", "submit_started_ambiguous"}:
            raise ValueError("only an unaccepted submission can be reclaimed")
        if attempt.get("charged") is True:
            work["paid_attempts"] = max(0, self.paid_attempts(work_id) - 1)
            attempt["charged"] = False
        attempt["state"] = "terminal"
        attempt["terminal_outcome"] = "proven_not_sent"
        self._event(work, "terminal", attempt_id=attempt_id, reason=reason)
        self._persist()

    def mark_accepted(self, work_id: str, *, task_id: str) -> None:
        work = self.document["works"].get(work_id)
        if not isinstance(work, dict):
            raise ValueError(f"unknown attempt-ledger work id: {work_id}")
        attempt = self.active_attempt(work_id)
        if attempt is None:
            raise PendingWorkError(f"work {work_id} has no reserved submission to accept")
        if attempt.get("task_id") != task_id:
            raise ValueError("accepted task id does not match attempt ledger")
        if attempt.get("state") == "reserved_not_sent":
            if self.paid_attempts(work_id) >= int(
                self.document["max_paid_attempts_per_work"]
            ):
                raise PendingWorkError(f"work {work_id} exhausted its paid-attempt budget")
            attempt["charged"] = True
            work["paid_attempts"] = self.paid_attempts(work_id) + 1
        attempt["state"] = "accepted_inflight"
        self._event(work, "accepted_inflight", attempt_id=int(attempt["attempt_id"]))
        self._persist()

    def mark_terminal(self, work_id: str, *, outcome: str) -> None:
        work = self.document["works"].get(work_id)
        if not isinstance(work, dict):
            raise ValueError(f"unknown attempt-ledger work id: {work_id}")
        attempt = self.active_attempt(work_id)
        if attempt is None:
            return
        attempt["state"] = "terminal"
        attempt["terminal_outcome"] = outcome
        self._event(work, "terminal", attempt_id=int(attempt["attempt_id"]), reason=outcome)
        self._persist()

    def reconcile_missing_remote(
        self,
        work_id: str,
        *,
        current_allocation_attempt_id: str,
        allow_ambiguous_paid_retry: bool,
    ) -> bool:
        """Return True only when creating a new reservation is safe/authorized."""

        attempt = self.active_attempt(work_id)
        if attempt is None:
            return self.remaining_attempts(work_id) > 0
        state = str(attempt.get("state"))
        if state == "reserved_not_sent":
            self.reclaim_not_sent(
                work_id,
                attempt_id=int(attempt["attempt_id"]),
                reason="resume_observed_reserved_not_sent",
            )
            return self.remaining_attempts(work_id) > 0
        prior_allocation = str(attempt.get("allocation_attempt_id"))
        if prior_allocation == current_allocation_attempt_id:
            raise PendingWorkError(
                f"work {work_id} has {state} submission with no remote record; "
                "ACK loss remains ambiguous"
            )
        if not allow_ambiguous_paid_retry:
            raise PendingWorkError(
                f"work {work_id} has {state} from a prior allocation; explicit "
                "--allow-ambiguous-paid-retry is required"
            )
        self.mark_terminal(work_id, outcome="explicit_cross_allocation_abandon")
        return self.remaining_attempts(work_id) > 0

    def mark_completed(self, work_id: str, *, task_id: str) -> None:
        work = self.document["works"].get(work_id)
        if not isinstance(work, dict):
            raise ValueError(f"unknown attempt-ledger work id: {work_id}")
        attempt = self.active_attempt(work_id)
        if attempt is not None:
            if attempt.get("task_id") != task_id:
                raise ValueError("completed task id does not match active attempt")
            attempt["state"] = "terminal"
            attempt["terminal_outcome"] = "benchmark_result_persisted"
        work["completed"] = True
        events = work.setdefault("events", [])
        if not isinstance(events, list):
            raise ValueError("attempt ledger events must be a list")
        if not events or events[-1].get("event") != "completed":
            events.append({"event": "completed", "task_id": task_id})
        self._persist()

    def _active(self, work_id: str, attempt_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
        work = self.document["works"].get(work_id)
        if not isinstance(work, dict):
            raise ValueError(f"unknown attempt-ledger work id: {work_id}")
        attempt = self.active_attempt(work_id)
        if attempt is None or attempt.get("attempt_id") != attempt_id:
            raise ValueError("attempt is not the active work reservation")
        return work, attempt

    @staticmethod
    def _event(work: dict[str, Any], event: str, **fields: Any) -> None:
        events = work.setdefault("events", [])
        if not isinstance(events, list):
            raise ValueError("attempt ledger events must be a list")
        events.append({"event": event, **fields})

    def _persist(self) -> None:
        write_json(self.path, self.document)


class ResultStore:
    """Crash-durable, duplicate-safe result snapshot for one immutable plan."""

    def __init__(
        self,
        output_dir: Path,
        *,
        plan: dict[str, Any],
        expected: dict[str, dict[str, Any]],
        allocation_attempts: list[dict[str, Any]],
        candidates: tuple[CandidateSpec, ...],
        current_allocation_attempt_id: str | None,
    ) -> None:
        self.output_dir = output_dir
        self.plan = plan
        self.expected = expected
        self.allocation_attempts = allocation_attempts
        self.candidates = candidates
        self.current_allocation_attempt_id = current_allocation_attempt_id
        self.content_integrity: dict[str, Any] = {
            "status": "verified",
            "failures": [],
        }
        self._rows: dict[str, dict[str, Any]] = {}

    @classmethod
    def open(
        cls,
        output_dir: Path,
        *,
        plan: dict[str, Any],
        expected: dict[str, dict[str, Any]],
        resume: bool = False,
        allocation_attempt: dict[str, Any] | None = None,
        candidates: Iterable[CandidateSpec] = DEFAULT_CANDIDATES,
    ) -> ResultStore:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.is_dir():
            raise ValueError(f"forced-eval output is not a directory: {output_dir}")
        manifest_path = output_dir / "manifest.json"
        allocation_attempts: list[dict[str, Any]] = []
        if manifest_path.exists():
            if not resume:
                raise ValueError(
                    "forced-eval output already exists; pass --resume only for the "
                    "same immutable benchmark plan"
                )
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid existing manifest: {exc}") from exc
            if not isinstance(existing_manifest, dict):
                raise ValueError("existing manifest must be a JSON object")
            expected_plan_hash = _plan_sha256(plan)
            if existing_manifest.get("plan_sha256") != expected_plan_hash:
                raise ValueError(
                    "existing output belongs to a different forced-eval plan; "
                    "use a fresh run id and output directory"
                )
            for key, value in plan.items():
                if existing_manifest.get(key) != value:
                    raise ValueError(f"existing manifest field {key!r} does not match resume plan")
            existing_attempts = existing_manifest.get("allocation_attempts", [])
            if not isinstance(existing_attempts, list) or not all(
                isinstance(item, dict) for item in existing_attempts
            ):
                raise ValueError("existing allocation_attempts must be a list of objects")
            seen_attempt_ids: set[str] = set()
            for item in existing_attempts:
                attempt_id = item.get("allocation_attempt_id")
                teardown = item.get("teardown_verification")
                if (
                    not isinstance(attempt_id, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", attempt_id)
                    or attempt_id in seen_attempt_ids
                ):
                    raise ValueError("existing allocation attempt id is invalid or duplicated")
                seen_attempt_ids.add(attempt_id)
                if (
                    not isinstance(teardown, dict)
                    or teardown.get("status") not in {"pending", "verified", "failed"}
                ):
                    raise ValueError("existing allocation teardown state is invalid")
            allocation_attempts = list(existing_attempts)
            existing_integrity = existing_manifest.get("content_integrity")
            if not isinstance(existing_integrity, dict):
                raise ValueError("existing manifest has no content_integrity state")
            if existing_integrity.get("status") != "verified":
                raise ValueError(
                    "existing forced-eval output failed content-integrity verification"
                )
        else:
            if resume:
                raise ValueError("--resume requires an existing forced-eval manifest")
            entries = list(output_dir.iterdir())
            stale_temporaries = [
                path
                for path in entries
                if re.fullmatch(
                    r"\.(?:manifest\.json|summary\.json|results\.jsonl)\.\d+\.tmp",
                    path.name,
                )
            ]
            unexpected = [
                path
                for path in entries
                if path not in stale_temporaries
                and path.name not in {".forced-eval.lock", ".forced-eval.owner.lock"}
            ]
            if unexpected:
                raise ValueError(
                    "existing forced-eval output has no manifest; use a fresh output directory"
                )
            for path in stale_temporaries:
                path.unlink(missing_ok=True)
        if allocation_attempt is not None:
            attempt = dict(allocation_attempt)
            attempt_id = attempt.get("allocation_attempt_id")
            if not isinstance(attempt_id, str) or not re.fullmatch(
                r"[0-9a-f]{64}", attempt_id
            ):
                raise ValueError("allocation_attempt_id must be a SHA-256 digest")
            teardown = attempt.get("teardown_verification")
            if not isinstance(teardown, dict) or teardown.get("status") != "pending":
                raise ValueError("new allocation attempt must begin pending teardown")
            immutable_attempt = {
                key: value
                for key, value in attempt.items()
                if key not in {"teardown_verification", "attempt_sha256"}
            }
            attempt["attempt_sha256"] = _canonical_sha256(immutable_attempt)
            matching = [
                item
                for item in allocation_attempts
                if item.get("allocation_attempt_id") == attempt_id
            ]
            if matching:
                existing = matching[0]
                existing_immutable = {
                    key: value
                    for key, value in existing.items()
                    if key not in {"teardown_verification", "attempt_sha256"}
                }
                if existing_immutable != immutable_attempt:
                    raise ValueError("allocation attempt identity was reused with new metadata")
                existing_teardown = existing.get("teardown_verification", {})
                if existing_teardown.get("status") != "pending":
                    raise ValueError("cannot rerun an allocation attempt after teardown finalized")
            else:
                allocation_attempts.append(attempt)
        store = cls(
            output_dir,
            plan=plan,
            expected=expected,
            allocation_attempts=allocation_attempts,
            candidates=tuple(candidates),
            current_allocation_attempt_id=(
                str(allocation_attempt["allocation_attempt_id"])
                if allocation_attempt is not None
                else None
            ),
        )
        if manifest_path.exists():
            store.content_integrity = dict(existing_integrity)
        if not manifest_path.exists():
            # Establish the immutable resume identity before creating any
            # other artifact. If the process dies after mkdir or this atomic
            # write, the next invocation can safely reconstruct empty state.
            write_json(
                manifest_path,
                {
                    **plan,
                    "plan_sha256": _plan_sha256(plan),
                    "allocation_attempts": store.allocation_attempts,
                    "collection": store._collection(),
                    "content_integrity": store.content_integrity,
                },
            )

        results_path = output_dir / "results.jsonl"
        if results_path.exists():
            with results_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        raise ValueError(
                            f"blank line in existing results.jsonl at line {line_number}"
                        )
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid existing results.jsonl line {line_number}: {exc}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise ValueError(f"results.jsonl line {line_number} must be a JSON object")
                    store._accept(row, source=f"results.jsonl line {line_number}")

        # Repair stale summary/manifest metadata after a crash and canonicalize
        # any identical duplicate lines left by an older append-only writer.
        store.persist()
        return store

    @property
    def rows(self) -> list[dict[str, Any]]:
        return sorted(self._rows.values(), key=_row_sort_key)

    @property
    def completed_task_ids(self) -> set[str]:
        return set(self._rows)

    def add(self, row: dict[str, Any]) -> bool:
        if self.content_integrity.get("status") != "verified":
            raise ValueError("cannot add a result after content-integrity failure")
        added = self._accept(row, source="new result")
        if added:
            self.persist()
        return added

    def mark_integrity_failure(self, reason: str) -> None:
        failures = self.content_integrity.setdefault("failures", [])
        if not isinstance(failures, list):
            failures = []
            self.content_integrity["failures"] = failures
        if reason not in failures:
            failures.append(reason)
        self.content_integrity["status"] = "failed"
        self.persist()

    def _accept(self, row: dict[str, Any], *, source: str) -> bool:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in self.expected:
            raise ValueError(f"{source} has unknown or missing task_id {task_id!r}")
        for key, expected_value in self.expected[task_id].items():
            if row.get(key) != expected_value:
                raise ValueError(f"{source} field {key!r} does not match task {task_id!r}")
        self._validate_row_schema(row, source=source)
        existing = self._rows.get(task_id)
        if existing is not None:
            if existing != row:
                raise ValueError(f"conflicting duplicate result for task {task_id!r}")
            return False
        self._rows[task_id] = row
        return True

    def _validate_row_schema(self, row: dict[str, Any], *, source: str) -> None:
        required = {
            "allocation_attempt_id",
            "task_status",
            "session_id",
            "session_status",
            "valid",
            "reward",
            "reported_reward",
            "harbor_outcome_reward",
            "eval_only",
            "actor_invoked",
            "forced_route_acknowledgement",
            "forced_candidate_model",
            "router_action_valid",
            "router_submitted",
            "termination_reason",
            "pool_status",
            "pool_attempted",
            "pool_return_code",
            "pool_timed_out",
            "pool_failure_kind",
            "pool_duration_ms",
            "e2e_ms",
            "run_ms",
            "eval_ms",
            "integrity_errors",
        }
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"{source} is missing authoritative fields: {missing}")
        attempt_id = row.get("allocation_attempt_id")
        if not isinstance(attempt_id, str) or not re.fullmatch(r"[0-9a-f]{64}", attempt_id):
            raise ValueError(f"{source} has invalid allocation_attempt_id")
        known_attempt_ids = {
            item.get("allocation_attempt_id")
            for item in self.allocation_attempts
            if isinstance(item, dict)
        }
        if known_attempt_ids and attempt_id not in known_attempt_ids:
            raise ValueError(f"{source} belongs to an unknown allocation attempt")
        if row.get("task_status") != "completed" or row.get("session_status") != "COMPLETED":
            raise ValueError(f"{source} is not a completed task/session outcome")
        if row.get("valid") is not True:
            raise ValueError(f"{source} is not an authoritative valid outcome")
        reward = _finite_reward(row.get("reward"))
        reported = _finite_reward(row.get("reported_reward"))
        harbor = _finite_reward(row.get("harbor_outcome_reward"))
        if reward is None or reported is None or harbor is None or reward != reported:
            raise ValueError(f"{source} has missing, non-finite, or inconsistent reward")
        if (
            row.get("eval_only") is not True
            or row.get("actor_invoked") is not False
            or row.get("forced_route_acknowledgement") != EVAL_ONLY_ACK
            or row.get("forced_candidate_model") != row.get("candidate_model")
            or row.get("router_action_valid") is not True
            or row.get("router_submitted") is not True
            or row.get("termination_reason") != "m0_auto_submit"
            or row.get("integrity_errors") != []
        ):
            raise ValueError(f"{source} has invalid forced-route provenance markers")
        if row.get("pool_attempted") is not True:
            raise ValueError(f"{source} has no explicit candidate-call attempt")
        pool_status = row.get("pool_status")
        if pool_status == "completed":
            if (
                row.get("pool_return_code") != 0
                or row.get("pool_timed_out") is not False
                or row.get("pool_failure_kind") is not None
            ):
                raise ValueError(f"{source} has inconsistent completed-call status")
        elif pool_status in {"timeout", "timed_out"}:
            if (
                row.get("pool_timed_out") is not True
                or row.get("pool_failure_kind") != "timeout"
            ):
                raise ValueError(f"{source} has ambiguous candidate-timeout status")
        else:
            raise ValueError(f"{source} has unsupported pool status {pool_status!r}")
        duration = _finite_reward(row.get("pool_duration_ms"))
        if duration is None or duration < 0:
            raise ValueError(f"{source} has invalid pool duration")
        for field in ("e2e_ms", "run_ms", "eval_ms"):
            value = row.get(field)
            if value is not None and (_finite_reward(value) is None or float(value) < 0):
                raise ValueError(f"{source} has invalid {field}")

    def _collection(self) -> dict[str, Any]:
        expected_count = len(self.expected)
        collected_count = len(self._rows)
        return {
            "status": "complete" if collected_count == expected_count else "partial",
            "result_set_complete": collected_count == expected_count,
            "expected_result_count": expected_count,
            "collected_result_count": collected_count,
            "missing_result_count": expected_count - collected_count,
            "valid_result_count": sum(row.get("valid") is True for row in self._rows.values()),
            "invalid_result_count": sum(
                row.get("valid") is not True for row in self._rows.values()
            ),
        }

    def persist(self) -> None:
        rows = self.rows
        collection = self._collection()
        manifest = {
            **self.plan,
            "plan_sha256": _plan_sha256(self.plan),
            "allocation_attempts": self.allocation_attempts,
            "collection": collection,
            "content_integrity": self.content_integrity,
            "publication": {
                "status": "pending_teardown",
                "allocation_attempt_id": self.current_allocation_attempt_id,
            },
        }
        collected_only = summarize(
            rows,
            self.candidates,
            expected_rows=self.expected.values(),
        )
        integrity_verified = self.content_integrity.get("status") == "verified"
        summary = {
            **manifest,
            "collected_only": collected_only,
            # The evaluator is not the lifecycle owner. Only the parent
            # launcher may publish after every service has cleanly torn down
            # and post-teardown identities have been reverified.
            "final_metrics": None,
            "final_metrics_status": (
                "pending_teardown" if integrity_verified else "withheld_integrity"
            ),
        }
        # Results are authoritative.  Replace them first; a crash before the
        # metadata replacements is repaired by open() on the next invocation.
        write_jsonl(self.output_dir / "results.jsonl", rows)
        write_json(self.output_dir / "summary.json", summary)
        write_json(self.output_dir / "manifest.json", manifest)


async def _async_main_locked(args: argparse.Namespace) -> int:
    token = os.environ.get(_CONTROL_TOKEN_ENV, "").strip()
    if not token:
        raise ValueError(f"{_CONTROL_TOKEN_ENV} is required")
    semantic_identity = load_semantic_identity(args.semantic_identity)
    verify_semantic_identity(semantic_identity)
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
    candidates = selected_candidates(
        include_qwen35_baseline=args.include_qwen35_baseline
    )
    validate_candidates(config, candidates)
    timeout_contract = validate_timeout_contract(config, args)
    data_sha = sha256_file(args.data)
    raw_config_sha = sha256_file(args.polar_config)
    semantic_config = build_semantic_config_manifest(args.polar_config)
    implementation_manifest = build_implementation_manifest()
    task_asset_manifest = build_task_asset_manifest(items)
    work_items = make_work_items(
        items,
        candidates=candidates,
        replicates=args.replicates,
        seed=args.seed,
    )
    plan = {
        "schema_version": 6,
        "eval_only": True,
        "actor_training": False,
        "actor_invoked": False,
        "acknowledgement": EVAL_ONLY_ACK,
        "run_id": args.run_id,
        "data_sha256": data_sha,
        "semantic_config": semantic_config,
        "implementation_manifest": implementation_manifest,
        "task_asset_manifest": task_asset_manifest,
        "semantic_identity": semantic_identity["semantic"],
        "semantic_identity_sha256": semantic_identity["semantic_sha256"],
        "start_index": args.start_index,
        "max_tasks": args.max_tasks,
        "replicates": args.replicates,
        "seed": args.seed,
        "forward_seed_to_pool": args.forward_seed_to_pool,
        "include_qwen35_baseline": args.include_qwen35_baseline,
        "max_concurrency": args.max_concurrency,
        "retry_policy": {
            "max_paid_attempts_per_work": args.max_paid_attempts_per_work,
            "allow_ambiguous_paid_retry": args.allow_ambiguous_paid_retry,
            "control_plane_retry_attempts": args.control_plane_retry_attempts,
            "control_plane_retry_backoff_seconds": (
                args.control_plane_retry_backoff_seconds
            ),
        },
        **timeout_contract,
        "dataset_indices": [item.dataset_index for item in items],
        "dataset_rows": [
            {
                "dataset_index": item.dataset_index,
                "dataset_row_sha256": dataset_row_sha256(item),
            }
            for item in items
        ],
        "candidates": [
            {
                "pool_model": candidate.pool_model,
                "endpoint_model": candidate.endpoint_model,
                "label": candidate.label,
            }
            for candidate in candidates
        ],
        "expected_result_count": len(work_items),
    }
    plan_fingerprint = _plan_sha256(plan)
    allocation_attempt = {
        "allocation_attempt_id": args.allocation_attempt_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
        "data_path": str(args.data.resolve()),
        "polar_config_path": str(args.polar_config.resolve()),
        "polar_config_sha256": raw_config_sha,
        "semantic_identity_path": str(args.semantic_identity.resolve()),
        "semantic_identity_file_sha256": sha256_file(args.semantic_identity),
        "rollout_url": config.rollout_server_url,
        "control_plane_retry_attempts": args.control_plane_retry_attempts,
        "control_plane_retry_backoff_seconds": args.control_plane_retry_backoff_seconds,
        "teardown_verification": {"status": "pending", "reason": None},
    }

    payloads = [
        build_payload(
            work,
            args=config_args,
            config=config,
            run_id=args.run_id,
            data_sha256=data_sha,
            plan_sha256=plan_fingerprint,
            forward_seed_to_pool=args.forward_seed_to_pool,
        )
        for work in work_items
    ]
    expected = {
        str(payload["task_id"]): _expected_result_fields(
            work,
            str(payload["task_id"]),
            plan_sha256=plan_fingerprint,
        )
        for work, payload in zip(work_items, payloads, strict=True)
    }
    store = ResultStore.open(
        args.output_dir,
        plan=plan,
        expected=expected,
        resume=args.resume,
        allocation_attempt=allocation_attempt,
        candidates=candidates,
    )
    attempt_ledger = AttemptLedger(
        args.output_dir / "attempt_ledger.json",
        plan_sha256=plan_fingerprint,
        work_ids=(
            str(fields["forced_eval_work_sha256"])
            for fields in expected.values()
        ),
        max_paid_attempts=args.max_paid_attempts_per_work,
    )
    for completed_task_id in store.completed_task_ids:
        attempt_ledger.mark_completed(
            str(expected[completed_task_id]["forced_eval_work_sha256"]),
            task_id=completed_task_id,
        )
    pending = [
        (work, payload)
        for work, payload in zip(work_items, payloads, strict=True)
        if str(payload["task_id"]) not in store.completed_task_ids
    ]
    timeout = httpx.Timeout(args.request_timeout)
    limits = httpx.Limits(
        max_connections=args.max_concurrency,
        max_keepalive_connections=args.max_concurrency,
    )
    semaphore = asyncio.Semaphore(args.max_concurrency)
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    try:
        async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
            tasks = [
                asyncio.create_task(
                    submit_one(
                        client,
                        semaphore,
                        rollout_url=config.rollout_server_url,
                        token=token,
                        poll_seconds=args.poll_seconds,
                        retry_attempts=args.control_plane_retry_attempts,
                        retry_backoff_seconds=args.control_plane_retry_backoff_seconds,
                        work=work,
                        payload=payload,
                        attempt_ledger=attempt_ledger,
                        allocation_attempt_id=args.allocation_attempt_id,
                        allow_ambiguous_paid_retry=args.allow_ambiguous_paid_retry,
                    )
                )
                for work, payload in pending
            ]
            for completed in asyncio.as_completed(tasks):
                try:
                    row = await completed
                except (PendingWorkError, TaskIdentityError) as exc:
                    print(
                        f"forced-route work remains pending: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                store.add(row)
                attempt_ledger.mark_completed(
                    str(row["forced_eval_work_sha256"]),
                    task_id=str(row["task_id"]),
                )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    try:
        if sha256_file(args.data) != data_sha:
            raise ValueError("evaluation dataset changed during evaluation")
        if build_semantic_config_manifest(args.polar_config) != semantic_config:
            raise ValueError("rendered semantic config changed during evaluation")
        if build_implementation_manifest() != implementation_manifest:
            raise ValueError("forced-eval implementation changed during evaluation")
        if build_task_asset_manifest(items) != task_asset_manifest:
            raise ValueError("task SIF or verifier content changed during evaluation")
        verify_semantic_identity(semantic_identity)
        if sha256_file(args.semantic_identity) != allocation_attempt[
            "semantic_identity_file_sha256"
        ]:
            raise ValueError("semantic identity manifest changed during evaluation")
    except (OSError, ValueError) as exc:
        reason = f"pre_final_identity_verification: {exc}"
        store.mark_integrity_failure(reason)
        print(f"forced-route integrity failure: {reason}", file=sys.stderr)
        return 2
    rows = store.rows
    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"collected_only": summary["collected_only"]}, indent=2, sort_keys=True))
    print(f"final_metrics_status={summary['final_metrics_status']}", flush=True)
    collection_complete = len(rows) == len(work_items)
    if not collection_complete:
        return 3
    return 0 if all(row.get("valid") is True for row in rows) else 2


async def async_main(args: argparse.Namespace) -> int:
    with OutputDirectoryLock(args.output_dir):
        return await _async_main_locked(args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"forced-route eval error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
