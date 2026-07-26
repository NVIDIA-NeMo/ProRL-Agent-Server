#!/usr/bin/env python3
"""Build a strict matched outcome ledger for a frozen SPilot Router.

The evaluator deliberately separates expensive candidate execution from the
learned routing decision:

* ``collect`` queries one frozen Router checkpoint on each fixed task/replicate
  cell under both the production slot assignment and its counterfactual swap.
* ``finalize`` joins those immutable decisions to a *published* paired
  ``forced_route_eval.py`` ledger for GPT-5.5 and Qwen3.6-27B.

The Router arm is therefore an exact one-call potential-outcome replay: it
selects one of the two already-measured candidate outcomes without rerunning a
stochastic coding agent.  It is not a full two-call SPilot episode and the
output schema labels that limitation explicitly.  Final metrics are withheld
unless every Router decision and every forced candidate outcome is present,
valid, content-matched, and provenance-verified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlparse

import httpx
import yaml

from polar.agent.presets.spilot_router_runner import (
    Candidate,
    GatewayInfrastructureError,
    RouterProtocolError,
    _openai_chat_payload,
    build_initial_router_messages,
    parse_router_action,
)
from slime_bridge._messages import prompt_to_instruction_text


SCHEMA_VERSION = 1
FORCED_LEDGER_SCHEMA_VERSION = 6
ACKNOWLEDGEMENT = "I_UNDERSTAND_THIS_IS_A_ONE_CALL_POTENTIAL_OUTCOME_REPLAY"
ACK_ENV = "SPILOT_MATCHED_ROUTER_REPLAY_ACK"
PRIMARY_MODELS = ("pool/qwen3.6-27b", "pool/gpt-5.5")
PERMUTATIONS = ("production", "counterfactual_swap")
MAX_RAW_RESPONSE_BYTES = 256 * 1024
_SLOT_RE = re.compile(r"^M(?:0|[1-9][0-9]*)$")
_DYNAMIC_FORCED_MANIFEST_KEYS = {
    "plan_sha256",
    "allocation_attempts",
    "collection",
    "content_integrity",
    "publication",
}


class LedgerError(ValueError):
    """The strict replay contract cannot be proven."""


@dataclass(frozen=True, slots=True)
class DatasetCell:
    dataset_index: int
    prompt: object
    metadata: dict[str, Any]
    instruction: str
    dataset_row_sha256: str


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    path: Path
    file_sha256: str
    semantic: dict[str, Any]
    semantic_sha256: str
    configured_slots: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    shuffle_slots: bool
    router_max_tokens: int
    router_model_kwargs: dict[str, Any]
    cost_penalty_lambda: float
    cost_normalizer: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise LedgerError(f"required file does not exist: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise LedgerError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LedgerError(f"required JSON file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LedgerError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LedgerError(f"JSON document must be an object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise LedgerError(f"required JSONL file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise LedgerError(f"blank JSONL record at {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise LedgerError(f"non-object JSONL record at {path}:{line_number}")
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise LedgerError(f"invalid JSONL at {path}:{exc.lineno}: {exc.msg}") from exc
    return rows


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            stream.write("\n")
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
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_number(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or (nonnegative and parsed < 0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise LedgerError(f"{name} must be {qualifier}")
    return parsed


def pair_seed(seed: int, dataset_index: int, replicate: int) -> int:
    material = f"{seed}\0{dataset_index}\0{replicate}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def load_dataset_slice(
    path: Path,
    *,
    start_index: int,
    max_tasks: int,
) -> list[DatasetCell]:
    if not path.is_file():
        raise LedgerError(f"evaluation dataset does not exist: {path}")
    cells: list[DatasetCell] = []
    with path.open(encoding="utf-8") as stream:
        for dataset_index, line in enumerate(stream):
            if dataset_index < start_index:
                continue
            if len(cells) >= max_tasks:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(
                    f"invalid JSON at dataset row {dataset_index}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict) or "prompt" not in row:
                raise LedgerError(
                    f"dataset row {dataset_index} must be an object with prompt"
                )
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                raise LedgerError(f"dataset row {dataset_index} metadata must be an object")
            instruction = prompt_to_instruction_text(row["prompt"])
            if not instruction:
                raise LedgerError(f"dataset row {dataset_index} has an empty instruction")
            row_identity = {
                "dataset_index": dataset_index,
                "prompt": row["prompt"],
                "metadata": metadata,
            }
            cells.append(
                DatasetCell(
                    dataset_index=dataset_index,
                    prompt=row["prompt"],
                    metadata=dict(metadata),
                    instruction=instruction,
                    dataset_row_sha256=canonical_sha256(row_identity),
                )
            )
    if len(cells) != max_tasks:
        raise LedgerError(
            f"requested {max_tasks} tasks at start index {start_index}, found {len(cells)}"
        )
    return cells


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LedgerError(f"{name} must be a mapping")
    return dict(value)


def _reject_credential_fields(value: object, *, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if (
                normalized in {"api_key", "authorization", "password", "secret", "token"}
                or normalized.endswith("_api_key")
                or normalized.endswith("_password")
                or normalized.endswith("_secret")
                or normalized.endswith("_token")
            ):
                raise LedgerError(f"{path} contains credential-like field {key!r}")
            _reject_credential_fields(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_credential_fields(child, path=f"{path}[{index}]")


def load_policy_snapshot(path: Path) -> PolicySnapshot:
    if not path.is_file():
        raise LedgerError(f"rendered policy config does not exist: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise LedgerError(f"cannot load rendered policy config {path}: {exc}") from exc
    root = _mapping(document, name="rendered policy config")
    if root.get("polar_instruction_template") not in (None, ""):
        raise LedgerError(
            "matched replay currently requires the production config to use the "
            "dataset instruction verbatim (no polar_instruction_template)"
        )
    task_template = _mapping(root.get("polar_task_template"), name="polar_task_template")
    agent = _mapping(task_template.get("agent"), name="polar_task_template.agent")
    if agent.get("harness") != "spilot_router":
        raise LedgerError("policy config agent.harness must be spilot_router")
    settings = _mapping(agent.get("settings"), name="agent.settings")
    raw_pool = _mapping(settings.get("model_pool"), name="agent.settings.model_pool")
    slots = tuple(sorted(raw_pool, key=lambda value: int(value[1:]) if _SLOT_RE.fullmatch(value) else -1))
    if len(slots) != 2 or any(not _SLOT_RE.fullmatch(slot) for slot in slots):
        raise LedgerError("matched replay requires exactly two logical M<number> slots")
    candidates: list[Candidate] = []
    for slot in slots:
        raw = _mapping(raw_pool[slot], name=f"model_pool.{slot}")
        unknown = set(raw).difference(
            {"model", "card", "description", "cost_weight", "model_kwargs"}
        )
        if unknown:
            raise LedgerError(
                f"model_pool.{slot} has unknown fields: {', '.join(sorted(unknown))}"
            )
        model = raw.get("model")
        if not isinstance(model, str) or not model.strip():
            raise LedgerError(f"model_pool.{slot}.model must be a non-empty string")
        card = raw.get("card", raw.get("description", {"name": model}))
        try:
            canonical_json(card)
        except (TypeError, ValueError) as exc:
            raise LedgerError(f"model_pool.{slot}.card must be JSON-serializable") from exc
        cost = _finite_number(
            raw.get("cost_weight", 1.0),
            name=f"model_pool.{slot}.cost_weight",
            nonnegative=True,
        )
        model_kwargs = _mapping(
            raw.get("model_kwargs", {}), name=f"model_pool.{slot}.model_kwargs"
        )
        _reject_credential_fields(model_kwargs, path=f"model_pool.{slot}.model_kwargs")
        candidates.append(
            Candidate(
                slot=slot,
                model=model.strip(),
                card=card,
                cost_weight=cost,
                model_kwargs=model_kwargs,
            )
        )
    if {candidate.model for candidate in candidates} != set(PRIMARY_MODELS):
        raise LedgerError(
            "matched replay requires exactly pool/qwen3.6-27b and pool/gpt-5.5"
        )
    shuffle_slots = settings.get("shuffle_slots", True)
    if not isinstance(shuffle_slots, bool):
        raise LedgerError("agent.settings.shuffle_slots must be boolean")
    router_max_tokens = settings.get("router_max_tokens", 192)
    if (
        isinstance(router_max_tokens, bool)
        or not isinstance(router_max_tokens, int)
        or not 16 <= router_max_tokens <= 4096
    ):
        raise LedgerError("agent.settings.router_max_tokens must be an integer in [16, 4096]")
    router_model_kwargs = _mapping(
        settings.get("router_model_kwargs", {}),
        name="agent.settings.router_model_kwargs",
    )
    forbidden = {"model", "messages", "stream", "seed"}.intersection(router_model_kwargs)
    if forbidden:
        raise LedgerError(
            "router_model_kwargs may not preconfigure replay-owned fields: "
            + ", ".join(sorted(forbidden))
        )
    _reject_credential_fields(router_model_kwargs, path="agent.settings.router_model_kwargs")
    evaluator = _mapping(task_template.get("evaluator"), name="polar_task_template.evaluator")
    if evaluator.get("strategy") != "spilot_harbor":
        raise LedgerError("policy config evaluator.strategy must be spilot_harbor")
    evaluator_config = _mapping(evaluator.get("config"), name="evaluator.config")
    if evaluator_config.get("require_valid_action", True) is not True:
        raise LedgerError("matched replay requires evaluator.config.require_valid_action=true")
    max_pool_calls = settings.get("max_pool_calls", 2)
    if isinstance(max_pool_calls, bool) or max_pool_calls not in {1, 2}:
        raise LedgerError("agent.settings.max_pool_calls must be 1 or 2")
    cost_lambda = _finite_number(
        evaluator_config.get("cost_penalty_lambda", 0.0),
        name="evaluator.config.cost_penalty_lambda",
        nonnegative=True,
    )
    cost_normalizer = _finite_number(
        evaluator_config.get("cost_normalizer", 1.0),
        name="evaluator.config.cost_normalizer",
        nonnegative=True,
    )
    if cost_normalizer <= 0:
        raise LedgerError("evaluator.config.cost_normalizer must be greater than zero")
    semantic = {
        "harness": "spilot_router",
        "model_pool": {
            candidate.slot: candidate.public_metadata() for candidate in candidates
        },
        "shuffle_slots": shuffle_slots,
        "shuffle_seed": settings.get("shuffle_seed", 0),
        "max_pool_calls": max_pool_calls,
        "router_max_tokens": router_max_tokens,
        "router_model_kwargs": router_model_kwargs,
        "evaluator": {
            "strategy": "spilot_harbor",
            "require_valid_action": evaluator_config.get("require_valid_action", True),
            "cost_penalty_lambda": cost_lambda,
            "cost_normalizer": cost_normalizer,
        },
        "instruction_template": None,
    }
    return PolicySnapshot(
        path=path.resolve(),
        file_sha256=sha256_file(path),
        semantic=semantic,
        semantic_sha256=canonical_sha256(semantic),
        configured_slots=slots,
        candidates=tuple(candidates),
        shuffle_slots=shuffle_slots,
        router_max_tokens=router_max_tokens,
        router_model_kwargs=router_model_kwargs,
        cost_penalty_lambda=cost_lambda,
        cost_normalizer=cost_normalizer,
    )


def load_actor_identity(
    *,
    ready_path: Path,
    checkpoint_manifest_path: Path,
) -> dict[str, Any]:
    ready = read_json_object(ready_path)
    if ready.get("schema_version") != 1:
        raise LedgerError("Router ready document must use schema_version=1")
    base_url = ready.get("base_url")
    model_id = ready.get("model_id")
    if not isinstance(base_url, str) or not base_url:
        raise LedgerError("Router ready document has no base_url")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LedgerError("Router ready base_url must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise LedgerError("Router ready base_url must not contain a query or fragment")
    if not isinstance(model_id, str) or not model_id:
        raise LedgerError("Router ready document has no model_id")

    server_config_path = ready_path.with_name("config.json")
    server_config = read_json_object(server_config_path)
    if server_config.get("model_id") != model_id:
        raise LedgerError("Router ready and server config model identities differ")
    checkpoint_manifest = read_json_object(checkpoint_manifest_path)
    model_path = server_config.get("model_path")
    if not isinstance(model_path, str) or Path(model_path).resolve() != checkpoint_manifest_path.parent.resolve():
        raise LedgerError(
            "Router server model_path does not match the checkpoint export manifest"
        )
    checkpoint_config_path = checkpoint_manifest_path.parent / "config.json"
    checkpoint_config_sha256 = sha256_file(checkpoint_config_path)
    return {
        "ready_path": str(ready_path.resolve()),
        "ready_sha256": sha256_file(ready_path),
        "ready": ready,
        "server_config_path": str(server_config_path.resolve()),
        "server_config_sha256": sha256_file(server_config_path),
        "server_config": server_config,
        "checkpoint_manifest_path": str(checkpoint_manifest_path.resolve()),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "checkpoint_manifest": checkpoint_manifest,
        "checkpoint_config_path": str(checkpoint_config_path.resolve()),
        "checkpoint_config_sha256": checkpoint_config_sha256,
        "request_model": model_id,
        "base_url": base_url.rstrip("/"),
    }


def validate_actor_identity(
    identity: Mapping[str, Any],
    *,
    allow_ephemeral_ready_missing: bool = False,
) -> None:
    for prefix in (
        "ready",
        "server_config",
        "checkpoint_manifest",
        "checkpoint_config",
    ):
        path = Path(str(identity.get(f"{prefix}_path", "")))
        expected = identity.get(f"{prefix}_sha256")
        if prefix == "ready" and allow_ephemeral_ready_missing and not path.exists():
            # The managed SGLang lifecycle intentionally deletes ready.json and
            # the credential at teardown. The collection manifest already
            # embeds and hashes the ready document; persistent server/checkpoint
            # identities below must still remain byte-identical.
            continue
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise LedgerError(f"frozen actor identity changed: {prefix}")


def _candidate_assignment(
    policy: PolicySnapshot,
    *,
    sampling_seed: int,
    permutation: str,
) -> tuple[Candidate, ...]:
    if permutation not in PERMUTATIONS:
        raise LedgerError(f"unknown Router permutation: {permutation}")
    assigned = list(policy.candidates)
    if policy.shuffle_slots:
        random.Random(sampling_seed).shuffle(assigned)
    if permutation == "counterfactual_swap":
        assigned.reverse()
    return tuple(
        Candidate(
            slot=slot,
            model=candidate.model,
            card=candidate.card,
            cost_weight=candidate.cost_weight,
            model_kwargs=dict(candidate.model_kwargs),
        )
        for slot, candidate in zip(policy.configured_slots, assigned, strict=True)
    )


def slot_mapping(candidates: Iterable[Candidate]) -> tuple[dict[str, Any], str]:
    mapping = {
        candidate.slot: candidate.public_metadata() for candidate in candidates
    }
    encoded = json.dumps(mapping, separators=(",", ":"), sort_keys=True).encode()
    return mapping, hashlib.sha256(encoded).hexdigest()


def build_router_request(
    *,
    cell: DatasetCell,
    policy: PolicySnapshot,
    request_model: str,
    sampling_seed: int,
    permutation: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = _candidate_assignment(
        policy,
        sampling_seed=sampling_seed,
        permutation=permutation,
    )
    messages = build_initial_router_messages(
        # Replays target checkpoints trained on the historical M0/M1 protocol;
        # the byte-exact prompt requires the anonymous label mode explicitly.
        cell.instruction,
        candidates,
        label_mode="anonymous",
    )
    model_kwargs = dict(policy.router_model_kwargs)
    model_kwargs["seed"] = sampling_seed
    model_kwargs.setdefault("max_tokens", policy.router_max_tokens)
    try:
        payload = _openai_chat_payload(
            model=request_model,
            messages=messages,
            model_kwargs=model_kwargs,
        )
    except GatewayInfrastructureError as exc:
        raise LedgerError(f"invalid frozen Router request kwargs: {exc}") from exc
    mapping, mapping_fingerprint = slot_mapping(candidates)
    provenance = {
        "permutation": permutation,
        "sampling_seed": sampling_seed,
        "slot_mapping": mapping,
        "slot_mapping_fingerprint": mapping_fingerprint,
        "messages_sha256": canonical_sha256(messages),
        "model_kwargs": model_kwargs,
        "request_payload_sha256": canonical_sha256(payload),
    }
    return payload, provenance


def _response_request_ids(response: httpx.Response) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "x-request-id",
        "x-sglang-request-id",
        "request-id",
        "traceparent",
    ):
        value = response.headers.get(name)
        if value:
            result[name] = value[:512]
    return result


def parse_router_response(
    response: httpx.Response,
    *,
    allowed_slots: set[str],
) -> dict[str, Any]:
    response.raise_for_status()
    raw_bytes = response.content
    if len(raw_bytes) > MAX_RAW_RESPONSE_BYTES:
        raise LedgerError(
            f"Router response exceeds {MAX_RAW_RESPONSE_BYTES} byte audit limit"
        )
    try:
        body = response.json()
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise LedgerError(f"Router response is not JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise LedgerError("Router response must be a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise LedgerError("Router response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise LedgerError("Router choice must be an object")
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise LedgerError("Router choice must contain string message.content")
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        raise LedgerError(f"Router choice did not finish with stop: {finish_reason!r}")
    content = message["content"]
    try:
        action = parse_router_action(
            content,
            expected="ROUTE",
            allowed_slots=allowed_slots,
        )
    except RouterProtocolError as exc:
        raise LedgerError(f"invalid frozen Router action: {exc}") from exc
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        raise LedgerError("Router response usage must be an object when present")
    sanitized_usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LedgerError(f"Router response usage.{key} is invalid")
            sanitized_usage[key] = value
    return {
        "action": action,
        "content": content,
        "finish_reason": finish_reason,
        "usage": sanitized_usage,
        "response_id": body.get("id"),
        "response_model": body.get("model"),
        "raw_response": body,
        "response_sha256": canonical_sha256(body),
    }


RequestFunction = Callable[
    [str, dict[str, str], dict[str, Any], float],
    httpx.Response,
]


def _default_request(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> httpx.Response:
    # Deliberately no automatic retry: a response-lost retry would resample the
    # stochastic policy and break the one-decision-per-cell ledger.
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        return client.post(url, headers=headers, json=payload)


def collect_router_decisions(
    *,
    cells: list[DatasetCell],
    replicates: int,
    base_seed: int,
    policy: PolicySnapshot,
    actor_identity: Mapping[str, Any],
    api_key: str,
    collection_plan_sha256: str,
    request_timeout: float,
    request_fn: RequestFunction = _default_request,
) -> list[dict[str, Any]]:
    if not api_key:
        raise LedgerError("Router API key file is empty")
    url = f"{actor_identity['base_url']}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    decisions: list[dict[str, Any]] = []
    for replicate in range(replicates):
        for cell in cells:
            sampling_seed = pair_seed(base_seed, cell.dataset_index, replicate)
            for permutation in PERMUTATIONS:
                payload, request_provenance = build_router_request(
                    cell=cell,
                    policy=policy,
                    request_model=str(actor_identity["request_model"]),
                    sampling_seed=sampling_seed,
                    permutation=permutation,
                )
                started = time.monotonic()
                row: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "collection_plan_sha256": collection_plan_sha256,
                    "dataset_index": cell.dataset_index,
                    "dataset_row_sha256": cell.dataset_row_sha256,
                    "replicate": replicate,
                    "pair_seed": sampling_seed,
                    **request_provenance,
                    "valid": False,
                    "error": None,
                    "selected_slot": None,
                    "selected_model": None,
                    "request_url": url,
                    "request_model": actor_identity["request_model"],
                }
                response: httpx.Response | None = None
                try:
                    response = request_fn(url, headers, payload, request_timeout)
                    row["http_status"] = response.status_code
                    row["response_headers"] = _response_request_ids(response)
                    parsed = parse_router_response(
                        response,
                        allowed_slots=set(policy.configured_slots),
                    )
                    selected_slot = parsed["action"]["model_slot"]
                    selected = request_provenance["slot_mapping"][selected_slot]
                    row.update(
                        {
                            "valid": True,
                            "selected_slot": selected_slot,
                            "selected_model": selected["model"],
                            **parsed,
                        }
                    )
                except (httpx.HTTPError, LedgerError, OSError, ValueError) as exc:
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
                    if response is not None:
                        row["response_excerpt"] = response.text[:4000]
                row["latency_ms"] = int(round((time.monotonic() - started) * 1000))
                row["decision_sha256"] = canonical_sha256(row)
                decisions.append(row)
    return decisions


def make_collection_plan(
    *,
    data_path: Path,
    cells: list[DatasetCell],
    start_index: int,
    max_tasks: int,
    replicates: int,
    seed: int,
    policy: PolicySnapshot,
    actor_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "matched_one_call_potential_outcome_replay",
        "limitations": {
            "full_agent_rerun": False,
            "second_router_action_evaluated": False,
            "candidate_outcomes_replayed_from_forced_ledger": True,
        },
        "acknowledgement": ACKNOWLEDGEMENT,
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "start_index": start_index,
        "max_tasks": max_tasks,
        "replicates": replicates,
        "seed": seed,
        "dataset_rows": [
            {
                "dataset_index": cell.dataset_index,
                "dataset_row_sha256": cell.dataset_row_sha256,
                "instruction_sha256": hashlib.sha256(
                    cell.instruction.encode("utf-8")
                ).hexdigest(),
            }
            for cell in cells
        ],
        "permutations": list(PERMUTATIONS),
        "expected_decision_count": len(cells) * replicates * len(PERMUTATIONS),
        "policy": {
            "path": str(policy.path),
            "file_sha256": policy.file_sha256,
            "semantic": policy.semantic,
            "semantic_sha256": policy.semantic_sha256,
        },
        "actor": dict(actor_identity),
    }


def _expected_decision_request(
    *,
    plan: Mapping[str, Any],
    cell: DatasetCell,
    replicate: int,
    permutation: str,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    policy_container = plan.get("policy")
    if not isinstance(policy_container, dict):
        raise LedgerError("collection plan policy identity is malformed")
    semantic = policy_container.get("semantic")
    if not isinstance(semantic, dict):
        raise LedgerError("collection plan policy semantic identity is malformed")
    raw_pool = semantic.get("model_pool")
    if not isinstance(raw_pool, dict):
        raise LedgerError("collection plan policy model_pool is malformed")
    slots = tuple(sorted(raw_pool, key=lambda value: int(value[1:])))
    if len(slots) != 2 or any(not _SLOT_RE.fullmatch(slot) for slot in slots):
        raise LedgerError("collection plan policy slots are malformed")
    configured: list[Candidate] = []
    for slot in slots:
        metadata = raw_pool.get(slot)
        if not isinstance(metadata, dict):
            raise LedgerError("collection plan candidate metadata is malformed")
        model = metadata.get("model")
        cost = metadata.get("cost_weight")
        if not isinstance(model, str):
            raise LedgerError("collection plan candidate model is malformed")
        configured.append(
            Candidate(
                slot=slot,
                model=model,
                card=metadata.get("card"),
                cost_weight=_finite_number(
                    cost,
                    name=f"collection plan {slot} cost_weight",
                    nonnegative=True,
                ),
            )
        )
    expected_seed = pair_seed(
        int(plan["seed"]),
        cell.dataset_index,
        replicate,
    )
    assigned = list(configured)
    if semantic.get("shuffle_slots") is True:
        random.Random(expected_seed).shuffle(assigned)
    elif semantic.get("shuffle_slots") is not False:
        raise LedgerError("collection plan shuffle_slots is malformed")
    if permutation == "counterfactual_swap":
        assigned.reverse()
    elif permutation != "production":
        raise LedgerError(f"unknown Router permutation: {permutation}")
    candidates = tuple(
        Candidate(
            slot=slot,
            model=candidate.model,
            card=candidate.card,
            cost_weight=candidate.cost_weight,
        )
        for slot, candidate in zip(slots, assigned, strict=True)
    )
    mapping, mapping_fingerprint = slot_mapping(candidates)
    messages = build_initial_router_messages(
        # Replays target checkpoints trained on the historical M0/M1 protocol;
        # the byte-exact prompt requires the anonymous label mode explicitly.
        cell.instruction,
        candidates,
        label_mode="anonymous",
    )
    raw_kwargs = semantic.get("router_model_kwargs")
    if not isinstance(raw_kwargs, dict):
        raise LedgerError("collection plan router_model_kwargs is malformed")
    kwargs = dict(raw_kwargs)
    kwargs["seed"] = expected_seed
    max_tokens = semantic.get("router_max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise LedgerError("collection plan router_max_tokens is malformed")
    kwargs.setdefault("max_tokens", max_tokens)
    actor = plan.get("actor")
    if not isinstance(actor, dict):
        raise LedgerError("collection plan actor identity is malformed")
    request_model = actor.get("request_model")
    if not isinstance(request_model, str):
        raise LedgerError("collection plan actor request model is malformed")
    try:
        payload = _openai_chat_payload(
            model=request_model,
            messages=messages,
            model_kwargs=kwargs,
        )
    except GatewayInfrastructureError as exc:
        raise LedgerError(f"collection plan cannot rebuild Router request: {exc}") from exc
    provenance = {
        "permutation": permutation,
        "sampling_seed": expected_seed,
        "slot_mapping": mapping,
        "slot_mapping_fingerprint": mapping_fingerprint,
        "messages_sha256": canonical_sha256(messages),
        "model_kwargs": kwargs,
        "request_payload_sha256": canonical_sha256(payload),
    }
    return expected_seed, payload, provenance


def _validate_decision_semantics(
    row: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    cell: DatasetCell,
) -> None:
    replicate = row.get("replicate")
    permutation = row.get("permutation")
    if isinstance(replicate, bool) or not isinstance(replicate, int):
        raise LedgerError("Router decision replicate is invalid")
    if not isinstance(permutation, str):
        raise LedgerError("Router decision permutation is invalid")
    expected_seed, payload, provenance = _expected_decision_request(
        plan=plan,
        cell=cell,
        replicate=replicate,
        permutation=permutation,
    )
    if row.get("dataset_row_sha256") != cell.dataset_row_sha256:
        raise LedgerError("Router decision dataset row identity is invalid")
    if row.get("pair_seed") != expected_seed:
        raise LedgerError("Router decision pair seed is invalid")
    for field, expected in provenance.items():
        if row.get(field) != expected:
            raise LedgerError(f"Router decision {field} provenance is invalid")
    actor = plan["actor"]
    expected_url = f"{str(actor['base_url']).rstrip('/')}/chat/completions"
    if row.get("request_url") != expected_url:
        raise LedgerError("Router decision request URL is invalid")
    if row.get("request_model") != actor.get("request_model"):
        raise LedgerError("Router decision request model is invalid")
    if canonical_sha256(payload) != row.get("request_payload_sha256"):
        raise LedgerError("Router decision request payload hash is invalid")
    latency = row.get("latency_ms")
    if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
        raise LedgerError("Router decision latency is invalid")
    if row.get("valid") is not True:
        return
    if row.get("http_status") != 200:
        raise LedgerError("valid Router decision does not have HTTP 200")
    raw_response = row.get("raw_response")
    if not isinstance(raw_response, dict):
        raise LedgerError("valid Router decision has no raw response")
    if canonical_sha256(raw_response) != row.get("response_sha256"):
        raise LedgerError("Router decision response hash is invalid")
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise LedgerError("Router decision raw response choices are invalid")
    choice = choices[0]
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or content != row.get("content"):
        raise LedgerError("Router decision response content is invalid")
    try:
        action = parse_router_action(
            content,
            expected="ROUTE",
            allowed_slots=set(provenance["slot_mapping"]),
        )
    except RouterProtocolError as exc:
        raise LedgerError(f"Router decision action is invalid: {exc}") from exc
    selected_slot = action["model_slot"]
    selected_model = provenance["slot_mapping"][selected_slot]["model"]
    if row.get("action") != action:
        raise LedgerError("Router decision parsed action is invalid")
    if row.get("selected_slot") != selected_slot:
        raise LedgerError("Router decision selected slot is invalid")
    if row.get("selected_model") != selected_model:
        raise LedgerError("Router decision selected model is invalid")
    if row.get("finish_reason") != "stop" or choice.get("finish_reason") != "stop":
        raise LedgerError("Router decision finish reason is invalid")
    if row.get("response_id") != raw_response.get("id"):
        raise LedgerError("Router decision response id is invalid")
    if row.get("response_model") != actor.get("request_model"):
        raise LedgerError("Router response model does not match frozen actor")
    if raw_response.get("model") != row.get("response_model"):
        raise LedgerError("Router decision raw response model is invalid")
    raw_usage = raw_response.get("usage", {})
    if not isinstance(raw_usage, dict):
        raise LedgerError("Router decision raw response usage is invalid")
    expected_usage = {
        key: raw_usage[key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if key in raw_usage
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in expected_usage.values()
    ):
        raise LedgerError("Router decision raw response token counts are invalid")
    if row.get("usage") != expected_usage:
        raise LedgerError("Router decision usage is invalid")


def validate_collection_output(
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = output_dir / "collection_manifest.json"
    decisions_path = output_dir / "decisions.jsonl"
    manifest_file_sha256 = sha256_file(manifest_path)
    manifest = read_json_object(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise LedgerError("collection manifest schema version is unsupported")
    plan = manifest.get("plan")
    if not isinstance(plan, dict):
        raise LedgerError("collection manifest has no immutable plan")
    plan_sha = manifest.get("collection_plan_sha256")
    if not isinstance(plan_sha, str) or canonical_sha256(plan) != plan_sha:
        raise LedgerError("collection plan hash does not match its content")
    decisions_file_sha256 = sha256_file(decisions_path)
    if manifest.get("decisions_sha256") != decisions_file_sha256:
        raise LedgerError("Router decisions file hash does not match collection manifest")
    decisions = read_jsonl(decisions_path)
    expected_count = plan.get("expected_decision_count")
    if len(decisions) != expected_count:
        raise LedgerError("Router decision count does not match immutable plan")
    data_path = Path(str(plan.get("data_path", "")))
    if sha256_file(data_path) != plan.get("data_sha256"):
        raise LedgerError("Router collection dataset content changed")
    cells = load_dataset_slice(
        data_path,
        start_index=int(plan.get("start_index", -1)),
        max_tasks=int(plan.get("max_tasks", 0)),
    )
    cells_by_index = {cell.dataset_index: cell for cell in cells}
    expected_dataset_rows = [
        {
            "dataset_index": cell.dataset_index,
            "dataset_row_sha256": cell.dataset_row_sha256,
            "instruction_sha256": hashlib.sha256(
                cell.instruction.encode("utf-8")
            ).hexdigest(),
        }
        for cell in cells
    ]
    if plan.get("dataset_rows") != expected_dataset_rows:
        raise LedgerError("Router collection plan dataset row identities are invalid")
    policy_identity = plan.get("policy")
    if not isinstance(policy_identity, dict) or not isinstance(
        policy_identity.get("semantic"), dict
    ):
        raise LedgerError("Router collection plan policy identity is malformed")
    if canonical_sha256(policy_identity["semantic"]) != policy_identity.get(
        "semantic_sha256"
    ):
        raise LedgerError("Router collection policy semantic hash is invalid")
    expected_keys: set[tuple[int, int, str]] = set()
    for dataset_row in plan.get("dataset_rows", []):
        if not isinstance(dataset_row, dict):
            raise LedgerError("collection plan dataset_rows is malformed")
        dataset_index = dataset_row.get("dataset_index")
        if isinstance(dataset_index, bool) or not isinstance(dataset_index, int):
            raise LedgerError("collection plan has invalid dataset_index")
        for replicate in range(int(plan.get("replicates", 0))):
            for permutation in PERMUTATIONS:
                expected_keys.add((dataset_index, replicate, permutation))
    observed: set[tuple[int, int, str]] = set()
    for row in decisions:
        stored_hash = row.get("decision_sha256")
        unhashed = {key: value for key, value in row.items() if key != "decision_sha256"}
        if not isinstance(stored_hash, str) or canonical_sha256(unhashed) != stored_hash:
            raise LedgerError("Router decision row hash is invalid")
        if row.get("collection_plan_sha256") != plan_sha:
            raise LedgerError("Router decision belongs to another collection plan")
        key = (row.get("dataset_index"), row.get("replicate"), row.get("permutation"))
        if key in observed:
            raise LedgerError(f"duplicate Router decision cell: {key}")
        observed.add(key)  # type: ignore[arg-type]
        dataset_index = row.get("dataset_index")
        if dataset_index not in cells_by_index:
            raise LedgerError("Router decision has an unexpected dataset index")
        _validate_decision_semantics(
            row,
            plan=plan,
            cell=cells_by_index[dataset_index],
        )
    if observed != expected_keys:
        raise LedgerError("Router decision cells do not exactly match immutable plan")
    collection = manifest.get("collection")
    if not isinstance(collection, dict):
        raise LedgerError("collection manifest has no collection state")
    valid_count = sum(row.get("valid") is True for row in decisions)
    if collection != {
        "status": "complete" if valid_count == len(decisions) else "invalid",
        "expected_decision_count": len(decisions),
        "collected_decision_count": len(decisions),
        "valid_decision_count": valid_count,
        "invalid_decision_count": len(decisions) - valid_count,
    }:
        raise LedgerError("collection manifest state does not match Router decisions")
    if sha256_file(manifest_path) != manifest_file_sha256:
        raise LedgerError("collection manifest changed while it was being validated")
    if sha256_file(decisions_path) != decisions_file_sha256:
        raise LedgerError("Router decisions changed while they were being validated")
    if sha256_file(data_path) != plan.get("data_sha256"):
        raise LedgerError("Router collection dataset changed while it was being validated")
    return manifest, decisions


def _validate_source_files(plan: Mapping[str, Any], data_path: Path, policy_path: Path) -> None:
    if Path(str(plan.get("data_path", ""))).resolve() != data_path.resolve():
        raise LedgerError("finalize data path differs from Router collection")
    if sha256_file(data_path) != plan.get("data_sha256"):
        raise LedgerError("evaluation dataset changed after Router collection")
    policy = plan.get("policy")
    if not isinstance(policy, dict):
        raise LedgerError("collection plan policy identity is malformed")
    if Path(str(policy.get("path", ""))).resolve() != policy_path.resolve():
        raise LedgerError("finalize policy path differs from Router collection")
    if sha256_file(policy_path) != policy.get("file_sha256"):
        raise LedgerError("policy config changed after Router collection")
    actor = plan.get("actor")
    if not isinstance(actor, dict):
        raise LedgerError("collection plan actor identity is malformed")
    validate_actor_identity(actor, allow_ephemeral_ready_missing=True)


def _forced_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in _DYNAMIC_FORCED_MANIFEST_KEYS
    }


def load_forced_ledger(
    forced_output_dir: Path,
    *,
    collection_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = forced_output_dir / "manifest.json"
    summary_path = forced_output_dir / "summary.json"
    results_path = forced_output_dir / "results.jsonl"
    source_hashes = {
        "manifest_sha256": sha256_file(manifest_path),
        "summary_sha256": sha256_file(summary_path),
        "results_sha256": sha256_file(results_path),
    }
    manifest = read_json_object(manifest_path)
    summary = read_json_object(summary_path)
    rows = read_jsonl(results_path)

    if manifest.get("schema_version") != FORCED_LEDGER_SCHEMA_VERSION:
        raise LedgerError(
            f"forced ledger must use schema_version={FORCED_LEDGER_SCHEMA_VERSION}"
        )
    plan_sha = manifest.get("plan_sha256")
    if not isinstance(plan_sha, str) or canonical_sha256(_forced_plan(manifest)) != plan_sha:
        raise LedgerError("forced ledger plan hash does not match manifest content")
    if summary.get("plan_sha256") != plan_sha:
        raise LedgerError("forced manifest and summary plan identities differ")
    if manifest.get("collection") != summary.get("collection"):
        raise LedgerError("forced manifest and summary collection states differ")
    collection = manifest.get("collection")
    if not isinstance(collection, dict) or collection.get("result_set_complete") is not True:
        raise LedgerError("forced ledger collection is not complete")
    if collection.get("status") != "complete" or collection.get("invalid_result_count") != 0:
        raise LedgerError("forced ledger contains missing or invalid outcomes")
    if manifest.get("content_integrity") != {"status": "verified", "failures": []}:
        raise LedgerError("forced ledger content integrity is not verified")
    publication = manifest.get("publication")
    if not isinstance(publication, dict) or publication.get("status") != "teardown_verified":
        raise LedgerError("forced ledger parent teardown is not verified")
    if summary.get("final_metrics_status") != "published":
        raise LedgerError("forced ledger final metrics are not published")
    summary_publication = summary.get("publication")
    if not isinstance(summary_publication, dict) or summary_publication.get("status") != "published":
        raise LedgerError("forced ledger summary is not at its publication boundary")
    attempts = manifest.get("allocation_attempts")
    if not isinstance(attempts, list) or not attempts:
        raise LedgerError("forced ledger has no allocation attempt provenance")
    if any(
        not isinstance(attempt, dict)
        or not isinstance(attempt.get("teardown_verification"), dict)
        or attempt["teardown_verification"].get("status") != "verified"
        for attempt in attempts
    ):
        raise LedgerError("forced ledger has an unverified allocation attempt")

    if manifest.get("eval_only") is not True or manifest.get("actor_invoked") is not False:
        raise LedgerError("forced source is not an actor-free eval-only ledger")
    if manifest.get("include_qwen35_baseline") is not False:
        raise LedgerError("matched replay requires the exact two-candidate forced ledger")
    if manifest.get("forward_seed_to_pool") is not True:
        raise LedgerError("forced source must forward the matched pair seed to candidates")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or {
        item.get("pool_model") for item in candidates if isinstance(item, dict)
    } != set(PRIMARY_MODELS) or len(candidates) != 2:
        raise LedgerError("forced ledger candidate set is not the required GPT/Qwen pair")
    endpoint_by_model = {
        str(item["pool_model"]): str(item["endpoint_model"])
        for item in candidates
        if isinstance(item, dict)
        and isinstance(item.get("pool_model"), str)
        and isinstance(item.get("endpoint_model"), str)
    }
    if set(endpoint_by_model) != set(PRIMARY_MODELS):
        raise LedgerError("forced ledger endpoint identities are malformed")

    expected_dataset_rows = collection_plan.get("dataset_rows")
    if not isinstance(expected_dataset_rows, list):
        raise LedgerError("Router collection dataset_rows is malformed")
    expected_indices = [row["dataset_index"] for row in expected_dataset_rows]
    expected_hashes = {
        row["dataset_index"]: row["dataset_row_sha256"] for row in expected_dataset_rows
    }
    if manifest.get("data_sha256") != collection_plan.get("data_sha256"):
        raise LedgerError("forced and Router ledgers use different dataset bytes")
    if manifest.get("start_index") != collection_plan.get("start_index"):
        raise LedgerError("forced and Router ledgers use different start indices")
    if manifest.get("max_tasks") != collection_plan.get("max_tasks"):
        raise LedgerError("forced and Router ledgers use different slice lengths")
    if manifest.get("replicates") != collection_plan.get("replicates"):
        raise LedgerError("forced and Router ledgers use different replicate counts")
    if manifest.get("seed") != collection_plan.get("seed"):
        raise LedgerError("forced and Router ledgers use different base seeds")
    if manifest.get("dataset_indices") != expected_indices:
        raise LedgerError("forced and Router ledgers use different dataset indices")
    forced_dataset_rows = manifest.get("dataset_rows")
    if not isinstance(forced_dataset_rows, list) or {
        row.get("dataset_index"): row.get("dataset_row_sha256")
        for row in forced_dataset_rows
        if isinstance(row, dict)
    } != expected_hashes:
        raise LedgerError("forced and Router dataset row identities differ")

    expected_keys = {
        (dataset_index, replicate, model)
        for dataset_index in expected_indices
        for replicate in range(int(collection_plan["replicates"]))
        for model in PRIMARY_MODELS
    }
    if manifest.get("expected_result_count") != len(expected_keys):
        raise LedgerError("forced expected-result denominator is inconsistent")
    if collection.get("expected_result_count") != len(expected_keys):
        raise LedgerError("forced collection denominator is inconsistent")
    observed: set[tuple[int, int, str]] = set()
    for row in rows:
        key = (row.get("dataset_index"), row.get("replicate"), row.get("candidate_model"))
        if key in observed:
            raise LedgerError(f"duplicate forced outcome cell: {key}")
        observed.add(key)  # type: ignore[arg-type]
        dataset_index, replicate, model = key
        if key not in expected_keys:
            raise LedgerError(f"unexpected forced outcome cell: {key}")
        expected_seed = pair_seed(
            int(collection_plan["seed"]), int(dataset_index), int(replicate)
        )
        if row.get("pair_seed") != expected_seed:
            raise LedgerError(f"forced outcome has wrong pair seed: {key}")
        if row.get("dataset_row_sha256") != expected_hashes[dataset_index]:
            raise LedgerError(f"forced outcome has wrong dataset row identity: {key}")
        if row.get("forced_eval_plan_sha256") != plan_sha:
            raise LedgerError(f"forced outcome belongs to another plan: {key}")
        if (
            row.get("valid") is not True
            or row.get("task_status") != "completed"
            or row.get("session_status") != "COMPLETED"
            or row.get("actor_invoked") is not False
            or row.get("eval_only") is not True
        ):
            raise LedgerError(f"forced outcome is not authoritative and valid: {key}")
        pool_status = row.get("pool_status")
        if pool_status == "completed":
            if (
                row.get("pool_return_code") != 0
                or row.get("pool_timed_out") is not False
                or row.get("pool_failure_kind") is not None
            ):
                raise LedgerError(f"forced completed-call status is inconsistent: {key}")
        elif pool_status in {"timeout", "timed_out"}:
            if (
                row.get("pool_timed_out") is not True
                or row.get("pool_failure_kind") != "timeout"
            ):
                raise LedgerError(f"forced timeout status is inconsistent: {key}")
        else:
            raise LedgerError(f"forced outcome has unsupported pool status: {key}")
        reward = _finite_number(row.get("reward"), name=f"forced reward {key}")
        raw = _finite_number(
            row.get("harbor_outcome_reward"), name=f"forced raw outcome {key}"
        )
        reported = _finite_number(
            row.get("reported_reward"), name=f"forced reported reward {key}"
        )
        if reward != raw or reward != reported:
            raise LedgerError(f"forced outcome reward fields disagree: {key}")
        if not 0.0 <= raw <= 1.0:
            raise LedgerError(f"forced accuracy outcome lies outside [0, 1]: {key}")
        expected_endpoint = endpoint_by_model[model]
        if row.get("candidate_endpoint_model") != expected_endpoint:
            raise LedgerError(f"forced outcome has wrong endpoint identity: {key}")
        expected_work_sha = canonical_sha256(
            {
                "plan_sha256": plan_sha,
                "dataset_index": dataset_index,
                "dataset_row_sha256": expected_hashes[dataset_index],
                "replicate": replicate,
                "pair_seed": expected_seed,
                "candidate_model": model,
                "candidate_endpoint_model": expected_endpoint,
            }
        )
        if row.get("forced_eval_work_sha256") != expected_work_sha:
            raise LedgerError(f"forced outcome has wrong work identity: {key}")
    if observed != expected_keys:
        raise LedgerError("forced result cells do not exactly match strict denominator")
    if sha256_file(manifest_path) != source_hashes["manifest_sha256"]:
        raise LedgerError("forced manifest changed while it was being validated")
    if sha256_file(summary_path) != source_hashes["summary_sha256"]:
        raise LedgerError("forced summary changed while it was being validated")
    if sha256_file(results_path) != source_hashes["results_sha256"]:
        raise LedgerError("forced results changed while they were being validated")
    source = {
        "output_dir": str(forced_output_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": source_hashes["manifest_sha256"],
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": source_hashes["summary_sha256"],
        "results_path": str(results_path.resolve()),
        "results_sha256": source_hashes["results_sha256"],
        "run_id": manifest.get("run_id"),
        "plan_sha256": plan_sha,
        "semantic_config_sha256": (
            manifest.get("semantic_config", {}).get("sha256")
            if isinstance(manifest.get("semantic_config"), dict)
            else None
        ),
        "semantic_identity_sha256": manifest.get("semantic_identity_sha256"),
        "implementation_sha256": (
            manifest.get("implementation_manifest", {}).get("sha256")
            if isinstance(manifest.get("implementation_manifest"), dict)
            else None
        ),
        "task_assets_sha256": (
            manifest.get("task_asset_manifest", {}).get("sha256")
            if isinstance(manifest.get("task_asset_manifest"), dict)
            else None
        ),
        "allocation_attempts": attempts,
    }
    return manifest, rows, source


def shaped_one_call_outcome(
    accuracy: float,
    *,
    cost: float,
    cost_penalty_lambda: float,
    cost_normalizer: float,
) -> dict[str, float]:
    applied_fraction = 0.0
    if accuracy > 0.0 and cost_penalty_lambda > 0.0:
        applied_fraction = min(1.0, cost_penalty_lambda * cost / cost_normalizer)
    reward = accuracy * (1.0 - applied_fraction)
    return {
        "accuracy_outcome": accuracy,
        "cost": cost,
        "applied_cost_penalty_fraction": applied_fraction,
        "cost_penalty_delta": accuracy - reward,
        "reward": reward,
    }


def build_replay_rows(
    *,
    collection_plan: Mapping[str, Any],
    decisions: list[dict[str, Any]],
    forced_rows: list[dict[str, Any]],
    policy: PolicySnapshot,
    forced_plan_sha256: str,
) -> list[dict[str, Any]]:
    if any(row.get("valid") is not True for row in decisions):
        raise LedgerError("final metrics withheld: at least one Router decision is invalid")
    by_forced = {
        (int(row["dataset_index"]), int(row["replicate"]), str(row["candidate_model"])): row
        for row in forced_rows
    }
    by_decision = {
        (int(row["dataset_index"]), int(row["replicate"]), str(row["permutation"])): row
        for row in decisions
    }
    costs = {candidate.model: candidate.cost_weight for candidate in policy.candidates}
    output: list[dict[str, Any]] = []
    for dataset_row in collection_plan["dataset_rows"]:
        dataset_index = int(dataset_row["dataset_index"])
        for replicate in range(int(collection_plan["replicates"])):
            pair = pair_seed(int(collection_plan["seed"]), dataset_index, replicate)
            forced_arms: dict[str, Any] = {}
            for model in PRIMARY_MODELS:
                source = by_forced[(dataset_index, replicate, model)]
                accuracy = float(source["harbor_outcome_reward"])
                forced_arms[model] = {
                    **shaped_one_call_outcome(
                        accuracy,
                        cost=costs[model],
                        cost_penalty_lambda=policy.cost_penalty_lambda,
                        cost_normalizer=policy.cost_normalizer,
                    ),
                    "source": {
                        "task_id": source.get("task_id"),
                        "session_id": source.get("session_id"),
                        "forced_eval_work_sha256": source["forced_eval_work_sha256"],
                        "forced_eval_plan_sha256": forced_plan_sha256,
                        "allocation_attempt_id": source.get("allocation_attempt_id"),
                    },
                }
            router_arms: dict[str, Any] = {}
            for permutation in PERMUTATIONS:
                decision = by_decision[(dataset_index, replicate, permutation)]
                selected_model = str(decision["selected_model"])
                selected_outcome = forced_arms[selected_model]
                router_arms[permutation] = {
                    "selected_slot": decision["selected_slot"],
                    "selected_model": selected_model,
                    "slot_mapping_fingerprint": decision["slot_mapping_fingerprint"],
                    "decision_sha256": decision["decision_sha256"],
                    "replayed_forced_eval_work_sha256": selected_outcome["source"][
                        "forced_eval_work_sha256"
                    ],
                    **{
                        key: selected_outcome[key]
                        for key in (
                            "accuracy_outcome",
                            "cost",
                            "applied_cost_penalty_fraction",
                            "cost_penalty_delta",
                            "reward",
                        )
                    },
                }
            row = {
                "schema_version": SCHEMA_VERSION,
                "method": "matched_one_call_potential_outcome_replay",
                "dataset_index": dataset_index,
                "dataset_row_sha256": dataset_row["dataset_row_sha256"],
                "replicate": replicate,
                "pair_seed": pair,
                "forced_arms": forced_arms,
                "router_arms": router_arms,
            }
            row["replay_row_sha256"] = canonical_sha256(row)
            output.append(row)
    return output


def _metric(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": statistics.fmean(values) if values else None,
    }


def _arm_metrics(rows: list[dict[str, Any]], accessor: Callable[[dict[str, Any]], Mapping[str, Any]]) -> dict[str, Any]:
    return {
        metric: _metric([float(accessor(row)[metric]) for row in rows])
        for metric in (
            "accuracy_outcome",
            "cost",
            "cost_penalty_delta",
            "reward",
        )
    }


def _paired_delta(
    rows: list[dict[str, Any]],
    *,
    first_name: str,
    second_name: str,
    first: Callable[[dict[str, Any]], Mapping[str, Any]],
    second: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "delta_definition": f"{second_name} - {first_name}",
        "count": len(rows),
    }
    for metric in ("accuracy_outcome", "cost", "reward"):
        deltas = [float(second(row)[metric]) - float(first(row)[metric]) for row in rows]
        result[metric] = {
            **_metric(deltas),
            "second_wins": sum(delta > 0 for delta in deltas),
            "first_wins": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
        }
    return result


def summarize_replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def qwen(row: dict[str, Any]) -> Mapping[str, Any]:
        return row["forced_arms"]["pool/qwen3.6-27b"]

    def gpt(row: dict[str, Any]) -> Mapping[str, Any]:
        return row["forced_arms"]["pool/gpt-5.5"]

    def production(row: dict[str, Any]) -> Mapping[str, Any]:
        return row["router_arms"]["production"]

    def swapped(row: dict[str, Any]) -> Mapping[str, Any]:
        return row["router_arms"]["counterfactual_swap"]

    semantic_agreement = sum(
        production(row)["selected_model"] == swapped(row)["selected_model"] for row in rows
    )
    slot_agreement = sum(
        production(row)["selected_slot"] == swapped(row)["selected_slot"] for row in rows
    )
    return {
        "method": "matched_one_call_potential_outcome_replay",
        "denominator": {
            "expected_cell_count": len(rows),
            "complete_cell_count": len(rows),
            "missing_cell_count": 0,
            "strict_complete_case": True,
        },
        "arms": {
            "forced_qwen3.6_27b": _arm_metrics(rows, qwen),
            "forced_gpt_5.5": _arm_metrics(rows, gpt),
            "router_production_one_call_replay": {
                **_arm_metrics(rows, production),
                "selected_model_counts": _counts(
                    production(row)["selected_model"] for row in rows
                ),
                "selected_slot_counts": _counts(
                    production(row)["selected_slot"] for row in rows
                ),
            },
        },
        "slot_permutation_diagnostic": {
            "counterfactual_arm": _arm_metrics(rows, swapped),
            "counterfactual_selected_model_counts": _counts(
                swapped(row)["selected_model"] for row in rows
            ),
            "counterfactual_selected_slot_counts": _counts(
                swapped(row)["selected_slot"] for row in rows
            ),
            "comparable_count": len(rows),
            "semantic_selection_agreement_count": semantic_agreement,
            "semantic_selection_agreement_fraction": (
                semantic_agreement / len(rows) if rows else None
            ),
            "presented_slot_agreement_count": slot_agreement,
            "presented_slot_agreement_fraction": slot_agreement / len(rows) if rows else None,
            "production_vs_counterfactual": _paired_delta(
                rows,
                first_name="router_production_one_call_replay",
                second_name="router_counterfactual_swap_one_call_replay",
                first=production,
                second=swapped,
            ),
        },
        "paired_deltas": {
            "gpt_minus_qwen": _paired_delta(
                rows,
                first_name="forced_qwen3.6_27b",
                second_name="forced_gpt_5.5",
                first=qwen,
                second=gpt,
            ),
            "router_minus_qwen": _paired_delta(
                rows,
                first_name="forced_qwen3.6_27b",
                second_name="router_production_one_call_replay",
                first=qwen,
                second=production,
            ),
            "router_minus_gpt": _paired_delta(
                rows,
                first_name="forced_gpt_5.5",
                second_name="router_production_one_call_replay",
                first=gpt,
                second=production,
            ),
        },
        "interpretation_guardrail": (
            "Descriptive matched outcomes only. This ledger evaluates the frozen "
            "Router's first decision by replaying one-call candidate outcomes; it does "
            "not evaluate SPilot's optional second VERIFY/SUBMIT action and does not by "
            "itself establish statistical superiority."
        ),
    }


def _counts(values: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _collection_state(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    valid_count = sum(row.get("valid") is True for row in decisions)
    return {
        "status": "complete" if valid_count == len(decisions) else "invalid",
        "expected_decision_count": len(decisions),
        "collected_decision_count": len(decisions),
        "valid_decision_count": valid_count,
        "invalid_decision_count": len(decisions) - valid_count,
    }


def run_collect(
    args: argparse.Namespace,
    *,
    request_fn: RequestFunction = _default_request,
) -> int:
    if args.output_dir.exists():
        raise LedgerError(f"fresh collection output directory already exists: {args.output_dir}")
    cells = load_dataset_slice(
        args.data,
        start_index=args.start_index,
        max_tasks=args.max_tasks,
    )
    policy = load_policy_snapshot(args.policy_config)
    actor_identity = load_actor_identity(
        ready_path=args.router_ready_json,
        checkpoint_manifest_path=args.checkpoint_manifest,
    )
    api_key = args.router_api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise LedgerError("Router API key file is empty")
    plan = make_collection_plan(
        data_path=args.data,
        cells=cells,
        start_index=args.start_index,
        max_tasks=args.max_tasks,
        replicates=args.replicates,
        seed=args.seed,
        policy=policy,
        actor_identity=actor_identity,
    )
    plan_sha = canonical_sha256(plan)
    decisions = collect_router_decisions(
        cells=cells,
        replicates=args.replicates,
        base_seed=args.seed,
        policy=policy,
        actor_identity=actor_identity,
        api_key=api_key,
        collection_plan_sha256=plan_sha,
        request_timeout=args.request_timeout,
        request_fn=request_fn,
    )
    # Verify the actor and all public source bytes one more time before making
    # the decision ledger durable.
    validate_actor_identity(actor_identity)
    if sha256_file(args.data) != plan["data_sha256"]:
        raise LedgerError("evaluation dataset changed during Router collection")
    if sha256_file(args.policy_config) != plan["policy"]["file_sha256"]:
        raise LedgerError("policy config changed during Router collection")
    # Create the output only after all preflight and requests succeed. A bad
    # input path must not consume the fresh-output identity with an empty tree.
    if args.output_dir.exists():
        raise LedgerError(f"collection output appeared concurrently: {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(mode=0o700)
    decisions_path = args.output_dir / "decisions.jsonl"
    write_jsonl(decisions_path, decisions)
    state = _collection_state(decisions)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "plan": plan,
        "collection_plan_sha256": plan_sha,
        "decisions_path": str(decisions_path.resolve()),
        "decisions_sha256": sha256_file(decisions_path),
        "collection": state,
        "publication": {"status": "awaiting_forced_outcomes"},
    }
    write_json(args.output_dir / "collection_manifest.json", manifest)
    write_json(args.output_dir / "manifest.json", manifest)
    summary = {
        **manifest,
        "final_metrics": None,
        "final_metrics_status": (
            "awaiting_forced_outcomes"
            if state["invalid_decision_count"] == 0
            else "withheld_router_invalid"
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    # Never retain the credential or its digest in the output tree.
    return 0 if state["invalid_decision_count"] == 0 else 3


def run_finalize(args: argparse.Namespace) -> int:
    collection_manifest, decisions = validate_collection_output(args.output_dir)
    plan = collection_manifest["plan"]
    _validate_source_files(plan, args.data, args.policy_config)
    cells = load_dataset_slice(
        args.data,
        start_index=int(plan["start_index"]),
        max_tasks=int(plan["max_tasks"]),
    )
    if [
        (cell.dataset_index, cell.dataset_row_sha256) for cell in cells
    ] != [
        (row["dataset_index"], row["dataset_row_sha256"])
        for row in plan["dataset_rows"]
    ]:
        raise LedgerError("dataset row identities changed after Router collection")
    if any(row.get("valid") is not True for row in decisions):
        raise LedgerError("final metrics withheld: Router collection contains invalid decisions")
    policy = load_policy_snapshot(args.policy_config)
    if policy.semantic_sha256 != plan["policy"]["semantic_sha256"]:
        raise LedgerError("policy semantic identity changed after Router collection")
    forced_manifest, forced_rows, forced_source = load_forced_ledger(
        args.forced_output_dir,
        collection_plan=plan,
    )
    replay_rows = build_replay_rows(
        collection_plan=plan,
        decisions=decisions,
        forced_rows=forced_rows,
        policy=policy,
        forced_plan_sha256=str(forced_manifest["plan_sha256"]),
    )
    replay_path = args.output_dir / "replay_rows.jsonl"
    write_jsonl(replay_path, replay_rows)
    final_metrics = summarize_replay(replay_rows)
    # Recheck every source immediately before the publication boundary. The
    # summary is written last, so a failed recheck can never leave formally
    # published metrics behind.
    validate_collection_output(args.output_dir)
    if sha256_file(Path(forced_source["manifest_path"])) != forced_source["manifest_sha256"]:
        raise LedgerError("forced manifest changed during replay construction")
    if sha256_file(Path(forced_source["summary_path"])) != forced_source["summary_sha256"]:
        raise LedgerError("forced summary changed during replay construction")
    if sha256_file(Path(forced_source["results_path"])) != forced_source["results_sha256"]:
        raise LedgerError("forced results changed during replay construction")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "published_at_utc": utc_now(),
        "method": "matched_one_call_potential_outcome_replay",
        "collection_plan_sha256": collection_manifest["collection_plan_sha256"],
        "collection_manifest_path": str(
            (args.output_dir / "collection_manifest.json").resolve()
        ),
        "collection_manifest_sha256": sha256_file(
            args.output_dir / "collection_manifest.json"
        ),
        "decisions_path": str((args.output_dir / "decisions.jsonl").resolve()),
        "decisions_sha256": sha256_file(args.output_dir / "decisions.jsonl"),
        "replay_rows_path": str(replay_path.resolve()),
        "replay_rows_sha256": sha256_file(replay_path),
        "expected_cell_count": len(replay_rows),
        "complete_cell_count": len(replay_rows),
        "policy": plan["policy"],
        "actor": plan["actor"],
        "forced_source": forced_source,
        "cost_shaping": {
            "formula": (
                "reward = accuracy_outcome * "
                "(1 - min(1, cost_penalty_lambda * cost / cost_normalizer))"
            ),
            "cost_penalty_lambda": policy.cost_penalty_lambda,
            "cost_normalizer": policy.cost_normalizer,
        },
        "limitations": plan["limitations"],
        "publication": {
            "status": "replay_integrity_verified_awaiting_summary",
            "strict_denominator_complete": True,
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    summary = {
        **manifest,
        "publication": {
            "status": "published_strict_complete_replay",
            "strict_denominator_complete": True,
        },
        "final_metrics_status": "published_strict_complete_replay",
        "final_metrics": final_metrics,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(final_metrics, indent=2, sort_keys=True))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect",
        help="query the frozen Router under production and swapped slot mappings",
    )
    collect.add_argument("--data", type=Path, required=True)
    collect.add_argument("--policy-config", type=Path, required=True)
    collect.add_argument("--router-ready-json", type=Path, required=True)
    collect.add_argument("--router-api-key-file", type=Path, required=True)
    collect.add_argument("--checkpoint-manifest", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--start-index", type=int, default=0)
    collect.add_argument("--max-tasks", type=int, required=True)
    collect.add_argument("--replicates", type=int, default=1)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--request-timeout", type=float, default=180.0)
    collect.add_argument(
        "--acknowledge-one-call-replay",
        action="store_true",
        help="acknowledge that this replays first-action outcomes, not a full episode",
    )

    finalize = subparsers.add_parser(
        "finalize",
        help="join a complete Router collection to a published forced outcome ledger",
    )
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--forced-output-dir", type=Path, required=True)
    finalize.add_argument("--data", type=Path, required=True)
    finalize.add_argument("--policy-config", type=Path, required=True)
    finalize.add_argument(
        "--acknowledge-one-call-replay",
        action="store_true",
        help="acknowledge that this replays first-action outcomes, not a full episode",
    )

    args = parser.parse_args(argv)
    acknowledged = args.acknowledge_one_call_replay or os.environ.get(ACK_ENV) == ACKNOWLEDGEMENT
    if not acknowledged:
        parser.error(
            "explicit acknowledgement required: pass --acknowledge-one-call-replay "
            f"or set {ACK_ENV}={ACKNOWLEDGEMENT}"
        )
    if args.command == "collect":
        if args.start_index < 0:
            parser.error("--start-index must be non-negative")
        if args.max_tasks <= 0:
            parser.error("--max-tasks must be positive")
        if args.replicates <= 0:
            parser.error("--replicates must be positive")
        if args.seed < 0:
            parser.error("--seed must be non-negative")
        if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
            parser.error("--request-timeout must be positive and finite")
    return args


def main(argv: list[str] | None = None) -> int:
    os.umask(0o077)
    args = parse_args(argv)
    try:
        if args.command == "collect":
            return run_collect(args)
        if args.command == "finalize":
            return run_finalize(args)
        raise AssertionError(f"unknown command: {args.command}")
    except (LedgerError, OSError, UnicodeError, yaml.YAMLError) as exc:
        print(f"matched Router replay error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
