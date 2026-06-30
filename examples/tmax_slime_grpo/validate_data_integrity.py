#!/usr/bin/env python3
"""Validate disjoint TMax train/eval datasets and write a hash manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1


def canonical_path(path: str | Path) -> Path:
    """Resolve an input path even when its final component does not exist yet."""

    return Path(path).expanduser().resolve(strict=False)


def _same_path(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def require_distinct_dataset_paths(
    train_data: str | Path,
    eval_datasets: Sequence[tuple[str, str | Path]],
) -> tuple[Path, list[tuple[str, Path]]]:
    """Reject duplicate names and every canonical/symlink path alias."""

    train_path = canonical_path(train_data)
    resolved: list[tuple[str, Path]] = []
    names: set[str] = set()
    for name, raw_path in eval_datasets:
        if not name:
            raise SystemExit("Eval dataset name must be non-empty")
        if name in names:
            raise SystemExit(f"Duplicate eval dataset name: {name!r}")
        names.add(name)
        path = canonical_path(raw_path)
        for other_name, other_path in [("train", train_path), *resolved]:
            if _same_path(path, other_path):
                raise SystemExit(
                    "TMAX_TRAIN_DATA and eval dataset files must resolve to different "
                    f"files; {name}={path} aliases {other_name}={other_path}"
                )
        resolved.append((name, path))
    if not resolved:
        raise SystemExit("At least one eval dataset is required")
    return train_path, resolved


def require_distinct_paths(train_data: str | Path, eval_data: str | Path) -> tuple[Path, Path]:
    """Backward-compatible two-file path validator."""

    train_path, datasets = require_distinct_dataset_paths(
        train_data, [("eval", eval_data)]
    )
    return train_path, datasets[0][1]


def _read_task_file(path: Path) -> tuple[bytes, list[str], dict[str, str]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise SystemExit(f"Cannot read TMax data file {path}: {exc}") from exc
    if not payload:
        raise SystemExit(f"TMax data file is empty: {path}")

    task_names: list[str] = []
    prompt_hash_to_task: dict[str, str] = {}
    seen: set[str] = set()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"TMax data file is not UTF-8: {path}: {exc}") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            task_name = row["metadata"]["task_name"]
            if not isinstance(task_name, str) or not task_name:
                raise TypeError("metadata.task_name must be a non-empty string")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SystemExit(
                f"Invalid TMax data row in {path} at line {line_number}: {exc}"
            ) from exc
        if task_name in seen:
            raise SystemExit(f"Duplicate TMax task_name {task_name!r} in {path}")
        seen.add(task_name)
        task_names.append(task_name)
        prompt = row.get("prompt", [])
        if isinstance(prompt, list):
            content = "\n".join(
                str(message.get("content", ""))
                for message in prompt
                if isinstance(message, dict)
            )
        else:
            content = str(prompt)
        normalized = re.sub(r"\s+", " ", content).strip().casefold()
        if normalized:
            prompt_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if prompt_hash in prompt_hash_to_task:
                raise SystemExit(
                    f"Duplicate normalized prompt in {path}: "
                    f"{prompt_hash_to_task[prompt_hash]!r} and {task_name!r}"
                )
            prompt_hash_to_task[prompt_hash] = task_name
    if not task_names:
        raise SystemExit(f"TMax data file has no non-empty task rows: {path}")
    return payload, task_names, prompt_hash_to_task


def _file_record(
    *, name: str, path: Path, payload: bytes, task_names: list[str]
) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "row_count": len(task_names),
    }


def _reject_overlap(
    left_name: str,
    left_tasks: list[str],
    left_prompts: dict[str, str],
    right_name: str,
    right_tasks: list[str],
    right_prompts: dict[str, str],
) -> None:
    overlap = sorted(set(left_tasks).intersection(right_tasks))
    relation = "train/eval" if left_name == "train" else f"{left_name}/{right_name}"
    if overlap:
        preview = ", ".join(overlap[:10])
        suffix = " ..." if len(overlap) > 10 else ""
        raise SystemExit(
            f"TMax {relation} task_name overlap ({len(overlap)} task(s)): "
            f"{preview}{suffix}"
        )
    prompt_overlap = sorted(set(left_prompts).intersection(right_prompts))
    if prompt_overlap:
        examples = ", ".join(
            f"{left_prompts[digest]} == {right_prompts[digest]}"
            for digest in prompt_overlap[:10]
        )
        suffix = " ..." if len(prompt_overlap) > 10 else ""
        raise SystemExit(
            f"TMax {relation} normalized-prompt overlap "
            f"({len(prompt_overlap)} prompt(s)): {examples}{suffix}"
        )


def build_multi_manifest(
    *,
    train_data: str | Path,
    eval_datasets: Sequence[tuple[str, str | Path]],
) -> dict[str, Any]:
    """Build one manifest after enforcing pairwise task/prompt disjointness."""

    train_path, resolved = require_distinct_dataset_paths(train_data, eval_datasets)
    train_payload, train_names, train_prompt_hashes = _read_task_file(train_path)
    parsed: list[tuple[str, Path, bytes, list[str], dict[str, str]]] = []
    for name, path in resolved:
        payload, task_names, prompt_hashes = _read_task_file(path)
        _reject_overlap(
            "train",
            train_names,
            train_prompt_hashes,
            name,
            task_names,
            prompt_hashes,
        )
        for other_name, _, _, other_tasks, other_prompts in parsed:
            _reject_overlap(
                other_name,
                other_tasks,
                other_prompts,
                name,
                task_names,
                prompt_hashes,
            )
        parsed.append((name, path, payload, task_names, prompt_hashes))

    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "sha256",
        "train": _file_record(
            name="train",
            path=train_path,
            payload=train_payload,
            task_names=train_names,
        ),
        "datasets": [
            _file_record(name=name, path=path, payload=payload, task_names=task_names)
            for name, path, payload, task_names, _ in parsed
        ],
    }


def build_manifest(
    *,
    train_data: str | Path,
    eval_data: str | Path,
    eval_name: str,
) -> dict[str, Any]:
    """Backward-compatible single-eval manifest builder."""

    return build_multi_manifest(
        train_data=train_data,
        eval_datasets=[(eval_name, eval_data)],
    )


def eval_bundle_sha256_from_records(records: Iterable[dict[str, Any]]) -> str:
    """Hash the named eval-file hashes independent of paths and list ordering."""

    compact = sorted(
        ({"name": str(item["name"]), "sha256": str(item["sha256"])} for item in records),
        key=lambda item: item["name"],
    )
    payload = json.dumps(compact, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def eval_bundle_sha256_for_files(
    eval_datasets: Sequence[tuple[str, str | Path]],
) -> str:
    names: set[str] = set()
    records: list[dict[str, str]] = []
    for name, raw_path in eval_datasets:
        if not name or name in names:
            raise SystemExit(f"Invalid or duplicate eval dataset name: {name!r}")
        names.add(name)
        path = canonical_path(raw_path)
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise SystemExit(f"Cannot read eval data file {path}: {exc}") from exc
        records.append({"name": name, "sha256": digest})
    if not records:
        raise SystemExit("At least one eval dataset is required")
    return eval_bundle_sha256_from_records(records)


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    output = canonical_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(manifest_bytes(manifest))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def encode_manifest(manifest: dict[str, Any]) -> str:
    """Return an environment-safe immutable copy consumed by Polar workers."""

    return base64.b64encode(manifest_bytes(manifest)).decode("ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data")
    parser.add_argument("--eval-data")
    parser.add_argument("--eval-name", default="tmax_holdout")
    parser.add_argument(
        "--eval-dataset",
        nargs=2,
        action="append",
        default=[],
        metavar=("NAME", "PATH"),
        help="Named eval dataset; repeat for a multi-eval contract.",
    )
    parser.add_argument(
        "--bundle-sha256",
        nargs=2,
        action="append",
        default=[],
        metavar=("NAME", "PATH"),
        help="Print only the canonical hash of these named eval files.",
    )
    parser.add_argument("--manifest")
    parser.add_argument(
        "--check-paths-only",
        action="store_true",
        help="Only reject canonical path aliasing; files need not exist yet.",
    )
    parser.add_argument(
        "--print-base64",
        action="store_true",
        help="Print an environment-safe copy of the manifest to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bundle_sha256:
        if any(
            (
                args.train_data,
                args.eval_data,
                args.eval_dataset,
                args.manifest,
                args.check_paths_only,
                args.print_base64,
            )
        ):
            raise SystemExit("--bundle-sha256 cannot be combined with manifest options")
        print(eval_bundle_sha256_for_files(args.bundle_sha256))
        return 0

    if not args.train_data:
        raise SystemExit("--train-data is required")
    eval_datasets: list[tuple[str, str]] = []
    if args.eval_data:
        eval_datasets.append((args.eval_name, args.eval_data))
    eval_datasets.extend((name, path) for name, path in args.eval_dataset)
    train_path, resolved = require_distinct_dataset_paths(args.train_data, eval_datasets)
    if args.check_paths_only:
        if args.manifest or args.print_base64:
            raise SystemExit("--check-paths-only cannot write or print a manifest")
        details = "; ".join(f"{name}={path}" for name, path in resolved)
        print(f"Validated distinct TMax data paths: train={train_path}; {details}")
        return 0

    if not args.manifest:
        raise SystemExit("--manifest is required unless --check-paths-only is used")
    manifest = build_multi_manifest(train_data=train_path, eval_datasets=resolved)
    output = write_manifest(args.manifest, manifest)
    if args.print_base64:
        print(encode_manifest(manifest))
    else:
        eval_rows = ", ".join(
            f"{item['name']}={item['row_count']}" for item in manifest["datasets"]
        )
        print(
            f"Wrote TMax data integrity manifest: {output}; "
            f"train_rows={manifest['train']['row_count']}; eval_rows={eval_rows}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
