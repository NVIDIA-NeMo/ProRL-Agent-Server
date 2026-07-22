#!/usr/bin/env python3
"""Prepare Polar JSONL from locally downloaded TMax Open-Instruct assets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-instruct-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def image_digest(image: str) -> str:
    digest = image.rsplit(":", 1)[-1].strip()
    if not digest or "/" in digest or digest in {".", ".."}:
        raise ValueError(f"invalid image reference: {image!r}")
    return digest


def ready_rows(root: Path) -> tuple[list[dict[str, object]], set[str]]:
    parquet = root / "data/train-00000-of-00001.parquet"
    image_dir = root / "enroot-images"
    if not parquet.is_file():
        raise SystemExit(f"TMax parquet is missing: {parquet}")
    if not image_dir.is_dir():
        raise SystemExit(f"TMax image directory is missing: {image_dir}")

    rows: list[dict[str, object]] = []
    image_names: set[str] = set()
    table = pq.read_table(parquet, columns=["messages", "env_config"])
    for source in table.to_pylist():
        env = source["env_config"]
        task_name = str(env["task_id"])
        digest = image_digest(str(env["image"]))
        image = image_dir / f"{digest}.sqsh"
        if not image.is_file() or image.stat().st_size == 0:
            continue
        prompt = next(
            (
                str(message["content"])
                for message in source["messages"]
                if message["role"] == "user"
            ),
            "",
        ).strip()
        if not prompt:
            raise SystemExit(f"TMax task {task_name} has no user instruction")
        rows.append(
            {
                "task_name": task_name,
                "prompt": prompt,
                "sif_path": str(image.resolve()),
            }
        )
        image_names.add(image.name)
    if not rows:
        raise SystemExit(f"No parquet rows have a local .sqsh under {image_dir}")
    return rows, image_names


def archive_task_names(archive: Path) -> set[str]:
    with tarfile.open(archive, "r:gz") as stream:
        return {
            PurePosixPath(member.name).parts[0]
            for member in stream.getmembers()
            if PurePosixPath(member.name).parts
        }


def materialize_tasks(archive: Path, destination: Path, selected: set[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] not in selected:
                continue
            if member.issym() or member.islnk():
                raise SystemExit(f"Refusing link in TMax task archive: {member.name}")
            target = (destination / Path(*parts)).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise SystemExit(f"Unsafe path in TMax task archive: {member.name}")
            stream.extract(member, destination)


def polar_rows(rows: list[dict[str, object]], tasks_dir: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        task_name = str(row["task_name"])
        task_dir = (tasks_dir / task_name).resolve()
        tests_dir = task_dir / "tests"
        if not (tests_dir / "test.sh").is_file():
            raise SystemExit(f"TMax verifier is missing: {tests_dir / 'test.sh'}")
        output.append(
            {
                "prompt": [{"role": "user", "content": row["prompt"]}],
                "label": "",
                "metadata": {
                    "task_name": task_name,
                    "task_dir": str(task_dir),
                    "tests_dir": str(tests_dir),
                    "sif_path": row["sif_path"],
                    "timeout_seconds": 840.0,
                    "agent_timeout": 600.0,
                    "verifier_timeout": 120.0,
                    "cpus": 1,
                    "memory_mb": 2048,
                    "allow_internet": True,
                    "workdir": "/root",
                },
            }
        )
    return output


def write_jsonl(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    root = Path(args.open_instruct_dir).expanduser().resolve()
    archive = root / "task-data.tar.gz"
    if not archive.is_file():
        raise SystemExit(f"TMax task archive is missing: {archive}")
    rows, image_names = ready_rows(root)
    selected = {str(row["task_name"]) for row in rows}
    missing_tasks = sorted(selected - archive_task_names(archive))
    if missing_tasks:
        raise SystemExit(
            f"TMax task archive is missing {len(missing_tasks)} selected task(s): "
            + ", ".join(missing_tasks[:10])
        )
    print(f"TMax ready rows={len(rows)} images={len(image_names)}")
    if args.check_only:
        return
    tasks_dir = Path(args.tasks_dir).expanduser().resolve()
    materialize_tasks(archive, tasks_dir, selected)
    write_jsonl(polar_rows(rows, tasks_dir), Path(args.output).expanduser().resolve())


if __name__ == "__main__":
    main()
