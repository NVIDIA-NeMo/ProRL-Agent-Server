#!/usr/bin/env python3
"""Validate the shard set referenced by a PyTorch distributed checkpoint.

PyTorch stores ``.metadata`` as a pickle. This validator deliberately scans
pickle opcodes instead of unpickling the file, so validation never executes
objects embedded in metadata. For every ``_StorageInfo`` record it extracts
the shard path, byte offset, and byte length, then verifies that the on-disk
``*.distcp`` set and exact sizes match the metadata contract.
"""

from __future__ import annotations

import argparse
import pickletools
from pathlib import Path
from typing import Final


class CheckpointValidationError(ValueError):
    """Raised when checkpoint metadata and shard files disagree."""


_MISSING: Final = object()
_STRING_OPS: Final = {
    "UNICODE",
    "SHORT_BINUNICODE",
    "BINUNICODE",
    "BINUNICODE8",
}
_INTEGER_OPS: Final = {
    "INT",
    "BININT",
    "BININT1",
    "BININT2",
    "LONG",
    "LONG1",
    "LONG4",
}
_STORAGE_FIELDS: Final = {"relative_path", "offset", "length"}


def _validate_storage_record(record: dict[str, object]) -> tuple[str, int]:
    shard_name = record["relative_path"]
    offset = record["offset"]
    length = record["length"]
    if not isinstance(shard_name, str):
        raise CheckpointValidationError("metadata storage path is not a string")
    relative_path = Path(shard_name)
    if (
        not shard_name.endswith(".distcp")
        or relative_path.is_absolute()
        or len(relative_path.parts) != 1
        or relative_path.name != shard_name
    ):
        raise CheckpointValidationError(
            f"metadata contains an unsafe shard path: {shard_name!r}"
        )
    if type(offset) is not int or offset < 0:  # bool is intentionally rejected
        raise CheckpointValidationError(
            f"metadata contains an invalid offset for {shard_name!r}: {offset!r}"
        )
    if type(length) is not int or length <= 0:  # bool is intentionally rejected
        raise CheckpointValidationError(
            f"metadata contains an invalid length for {shard_name!r}: {length!r}"
        )
    return shard_name, offset + length


def referenced_distcp_shard_sizes(metadata_path: Path) -> dict[str, int]:
    try:
        payload = metadata_path.read_bytes()
    except OSError as exc:
        raise CheckpointValidationError(
            f"cannot read metadata: {metadata_path}: {exc}"
        ) from exc

    memo: dict[int, object] = {}
    next_memo_index = 0
    last_scalar: object = _MISSING
    pending_field: str | None = None
    storage_record: dict[str, object] = {}
    expected_sizes: dict[str, int] = {}

    try:
        for opcode, argument, _position in pickletools.genops(payload):
            opcode_name = opcode.name
            if opcode_name == "MEMOIZE":
                memo[next_memo_index] = last_scalar
                next_memo_index += 1
                continue
            if opcode_name in {"BINPUT", "LONG_BINPUT", "PUT"}:
                memo_index = int(argument)
                memo[memo_index] = last_scalar
                next_memo_index = max(next_memo_index, memo_index + 1)
                continue

            scalar: object = _MISSING
            if opcode_name in {"BINGET", "LONG_BINGET", "GET"}:
                scalar = memo.get(int(argument), _MISSING)
            elif opcode_name in _STRING_OPS and isinstance(argument, str):
                scalar = argument
            elif opcode_name in _INTEGER_OPS and type(argument) is int:
                scalar = argument

            last_scalar = scalar
            if scalar is _MISSING:
                continue

            if pending_field is not None:
                storage_record[pending_field] = scalar
                pending_field = None
                if _STORAGE_FIELDS <= storage_record.keys():
                    shard_name, required_size = _validate_storage_record(storage_record)
                    expected_sizes[shard_name] = max(
                        expected_sizes.get(shard_name, 0), required_size
                    )
                    storage_record = {}
                continue

            if scalar == "relative_path":
                if storage_record:
                    raise CheckpointValidationError(
                        "metadata contains an incomplete storage record"
                    )
                pending_field = "relative_path"
            elif storage_record and scalar in {"offset", "length"}:
                pending_field = str(scalar)
    except CheckpointValidationError:
        raise
    except (IndexError, UnicodeDecodeError, ValueError) as exc:
        raise CheckpointValidationError(
            f"invalid checkpoint metadata pickle: {exc}"
        ) from exc

    if pending_field is not None or storage_record:
        raise CheckpointValidationError("metadata contains an incomplete storage record")
    if not expected_sizes:
        raise CheckpointValidationError(
            "metadata does not contain any complete .distcp storage records"
        )
    return expected_sizes


def validate_model_directory(model_dir: Path) -> set[str]:
    if not model_dir.is_dir():
        raise CheckpointValidationError(
            f"model checkpoint directory does not exist: {model_dir}"
        )

    metadata_path = model_dir / ".metadata"
    if (
        not metadata_path.is_file()
        or metadata_path.is_symlink()
        or metadata_path.stat().st_size <= 0
    ):
        raise CheckpointValidationError(
            f"metadata is missing, empty, or not a regular file: {metadata_path}"
        )

    expected_sizes = referenced_distcp_shard_sizes(metadata_path)
    referenced = set(expected_sizes)
    present = {path.name for path in model_dir.glob("*.distcp")}
    missing = sorted(referenced - present)
    unexpected = sorted(present - referenced)
    if missing:
        raise CheckpointValidationError(
            f"metadata references missing shard(s): {', '.join(missing)}"
        )
    if unexpected:
        raise CheckpointValidationError(
            f"checkpoint contains shard(s) absent from metadata: {', '.join(unexpected)}"
        )

    invalid = []
    for shard_name in sorted(referenced):
        path = model_dir / shard_name
        if not path.is_file() or path.is_symlink():
            invalid.append(f"{shard_name} (not a regular file)")
            continue
        actual_size = path.stat().st_size
        expected_size = expected_sizes[shard_name]
        if actual_size != expected_size:
            invalid.append(
                f"{shard_name} (expected {expected_size} bytes, found {actual_size})"
            )
    if invalid:
        raise CheckpointValidationError(
            f"empty or invalid shard(s): {', '.join(invalid)}"
        )
    return referenced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_model_directory(args.model_dir)
    except (CheckpointValidationError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
