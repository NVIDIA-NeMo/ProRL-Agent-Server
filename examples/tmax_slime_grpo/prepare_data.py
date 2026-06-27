#!/usr/bin/env python3
"""Build Slime JSONL prompts from a local TMax-15K-Harbor export."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
TMAX_EXAMPLE_DIR = EXAMPLE_DIR.parent / "tmax-15k"
sys.path.insert(0, str(TMAX_EXAMPLE_DIR))

from dataset import TmaxTask, load_tasks, sif_filename_for  # noqa: E402

DEFAULT_DATA_ROOT = Path(
    "/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/data"
)
DEFAULT_DATASET_DIR = DEFAULT_DATA_ROOT / "tmax-15k"
DEFAULT_IMAGE_DIR = DEFAULT_DATA_ROOT / "tmax-15k-sif"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "tmax-15k-train.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("TMAX_DATASET_DIR", str(DEFAULT_DATASET_DIR)),
    )
    parser.add_argument(
        "--image-dir",
        default=os.environ.get("APPTAINER_IMAGE_DIR", str(DEFAULT_IMAGE_DIR)),
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("TMAX_TRAIN_DATA", str(DEFAULT_OUTPUT)),
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=int(os.environ.get("TMAX_MAX_TASKS", "-1")),
        help="Limit rows after image filtering. -1 selects every task.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Select an exact task name. Repeatable.",
    )
    parser.add_argument(
        "--only-ready",
        action="store_true",
        default=os.environ.get("TMAX_ONLY_READY", "0") == "1",
        help="Drop tasks without a matching SIF instead of failing.",
    )
    return parser.parse_args()


def image_path(task: TmaxTask, image_dir: Path) -> Path:
    return image_dir / sif_filename_for(task.name)


def row_for_task(task: TmaxTask, image_dir: Path) -> dict[str, object]:
    return {
        "prompt": [{"role": "user", "content": task.instruction}],
        "label": "",
        "metadata": {
            "task_name": task.name,
            "task_dir": str(task.task_dir.resolve()),
            "tests_dir": str(task.tests_dir.resolve()),
            "sif_path": str(image_path(task, image_dir).resolve()),
            "timeout_seconds": task.agent_timeout + task.verifier_timeout + 120.0,
            "agent_timeout": task.agent_timeout,
            "verifier_timeout": task.verifier_timeout,
            "cpus": task.cpus or 1,
            "memory_mb": task.memory_mb or 2048,
            "allow_internet": task.allow_internet,
            "workdir": task.workdir or "/root",
        },
    }


def select_tasks(args: argparse.Namespace) -> tuple[list[TmaxTask], int]:
    tasks = load_tasks(args.dataset_dir, names=args.task or None)
    image_dir = Path(args.image_dir).expanduser().resolve()

    if args.only_ready:
        selected = [task for task in tasks if image_path(task, image_dir).is_file()]
        missing_count = len(tasks) - len(selected)
        if args.max_tasks > 0:
            selected = selected[: args.max_tasks]
        return selected, missing_count

    selected = tasks[: args.max_tasks] if args.max_tasks > 0 else tasks
    missing = [task for task in selected if not image_path(task, image_dir).is_file()]
    if missing:
        preview = ", ".join(task.name for task in missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise SystemExit(
            f"Missing {len(missing)}/{len(selected)} selected SIF(s) in {image_dir}: "
            f"{preview}{suffix}\n"
            "Wait for the SIF build to finish, select a smaller --max-tasks slice, "
            "or pass --only-ready for a smoke run."
        )
    return selected, 0


def write_rows(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n"
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    tasks, skipped_missing = select_tasks(args)
    if not tasks:
        raise SystemExit("No TMax tasks with ready SIF images were selected.")

    rows = [row_for_task(task, image_dir) for task in tasks]
    write_rows(rows, output)
    print(
        f"Wrote {len(rows)} TMax training row(s) to {output}; "
        f"image_dir={image_dir}; skipped_missing={skipped_missing}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
