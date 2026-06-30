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
        "--start-index",
        type=int,
        default=int(os.environ.get("TMAX_START_INDEX", "0")),
        help=(
            "Skip this many tasks in the deterministic sorted task list before "
            "applying --max-tasks. This supports a stable train/holdout split."
        ),
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=int(os.environ.get("TMAX_MAX_TASKS", "-1")),
        help=(
            "Select this many tasks from the deterministic window beginning at "
            "--start-index before checking images. -1 selects the rest."
        ),
    )
    parser.add_argument(
        "--expected-total-tasks",
        type=int,
        default=0,
        help=(
            "When positive, require the source dataset to contain exactly this "
            "many valid tasks before selecting the requested window."
        ),
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
        help=(
            "Explicit partial-data smoke mode: drop missing images from the "
            "selected prefix instead of failing; later tasks never backfill it."
        ),
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help=("Validate an existing --output JSONL and all referenced SIFs without rewriting it."),
    )
    return parser.parse_args()


def image_path(task: TmaxTask, image_dir: Path) -> Path:
    return image_dir / sif_filename_for(task.name)


def image_is_ready(task: TmaxTask, image_dir: Path) -> bool:
    """Return whether a task has a non-empty local SIF regular file."""
    path = image_path(task, image_dir)
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


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


def load_task_prefix(args: argparse.Namespace) -> list[TmaxTask]:
    # Freeze the requested window before looking at image readiness. Filtering
    # all tasks first used to let a ready task after the requested prefix
    # silently replace a missing task inside it, changing the training set as
    # builds completed.
    start_index = int(getattr(args, "start_index", 0))
    if start_index < 0:
        raise SystemExit(f"--start-index must be non-negative, got {start_index}")
    if args.task and start_index:
        raise SystemExit("--start-index cannot be combined with explicit --task values")
    expected_total = int(getattr(args, "expected_total_tasks", 0))
    if expected_total < 0:
        raise SystemExit(
            f"--expected-total-tasks must be non-negative, got {expected_total}"
        )
    load_limit = (
        -1
        if expected_total
        else start_index + args.max_tasks
        if args.max_tasks > 0 and not args.task
        else -1
    )
    tasks = load_tasks(
        args.dataset_dir,
        max_tasks=load_limit,
        names=args.task or None,
    )
    if expected_total and len(tasks) != expected_total:
        raise SystemExit(
            f"Expected exactly {expected_total} valid TMax task(s) under "
            f"{args.dataset_dir}, found {len(tasks)}. Refusing to change the "
            "fixed train/holdout population."
        )
    if not args.task:
        tasks = tasks[start_index:]
    return tasks[: args.max_tasks] if args.max_tasks > 0 else tasks


def require_complete_prefix(args: argparse.Namespace, tasks: list[TmaxTask]) -> None:
    if args.task or args.max_tasks <= 0 or len(tasks) == args.max_tasks:
        return
    start_index = int(getattr(args, "start_index", 0))
    requested = (
        f"the first {args.max_tasks}"
        if start_index == 0
        else f"{args.max_tasks} beginning at deterministic index {start_index}"
    )
    raise SystemExit(
        f"Requested {requested} TMax task(s), but only {len(tasks)} valid "
        f"task(s) exist in that window under {args.dataset_dir}.\n"
        "Restore the dataset or explicitly pass --only-ready for a "
        "partial-data smoke run."
    )


def select_tasks(args: argparse.Namespace) -> tuple[list[TmaxTask], int]:
    selected = load_task_prefix(args)
    image_dir = Path(args.image_dir).expanduser().resolve()

    if not args.only_ready:
        require_complete_prefix(args, selected)

    if args.only_ready:
        ready = [task for task in selected if image_is_ready(task, image_dir)]
        return ready, len(selected) - len(ready)

    missing = [task for task in selected if not image_is_ready(task, image_dir)]
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
        temporary.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def validate_existing_output(
    output: Path,
    *,
    expected_tasks: list[TmaxTask],
    image_dir: Path,
    allow_partial: bool,
) -> int:
    """Validate an immutable prompt file before submitting another allocation."""
    if not output.is_file() or output.stat().st_size == 0:
        raise SystemExit(f"TMax training JSONL is missing or empty: {output}")

    row_count = 0
    invalid_rows: list[str] = []
    missing_images: list[str] = []
    task_names: set[str] = set()
    ordered_task_names: list[str] = []
    duplicate_names: list[str] = []
    mismatched_image_paths: list[str] = []
    expected_image_paths = {
        task.name: image_path(task, image_dir).resolve() for task in expected_tasks
    }
    with output.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                row = json.loads(line)
                metadata = row["metadata"]
                task_name = metadata["task_name"]
                sif_path = Path(metadata["sif_path"]).expanduser()
                if not isinstance(task_name, str) or not task_name:
                    raise TypeError("metadata.task_name must be a non-empty string")
                if task_name in task_names:
                    duplicate_names.append(task_name)
                task_names.add(task_name)
                ordered_task_names.append(task_name)
                expected_image_path = expected_image_paths.get(task_name)
                if expected_image_path is not None and sif_path.resolve() != expected_image_path:
                    mismatched_image_paths.append(
                        f"{task_name}: {sif_path} != {expected_image_path}"
                    )
                try:
                    image_ready = sif_path.is_file() and sif_path.stat().st_size > 0
                except OSError:
                    image_ready = False
                if not image_ready:
                    missing_images.append(f"{task_name} ({sif_path})")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                invalid_rows.append(f"line {line_number}: {exc}")

    if invalid_rows:
        preview = "; ".join(invalid_rows[:5])
        raise SystemExit(f"Invalid TMax training JSONL row(s) in {output}: {preview}")
    if row_count == 0:
        raise SystemExit(f"TMax training JSONL has no non-empty rows: {output}")
    if duplicate_names:
        preview = ", ".join(duplicate_names[:10])
        raise SystemExit(f"Duplicate TMax task_name value(s) in {output}: {preview}")
    expected_names = [task.name for task in expected_tasks]
    if allow_partial:
        expected_positions = {name: index for index, name in enumerate(expected_names)}
        unexpected = [name for name in ordered_task_names if name not in expected_positions]
        positions = [
            expected_positions[name] for name in ordered_task_names if name in expected_positions
        ]
        if unexpected or positions != sorted(positions):
            detail = f"unexpected={unexpected[:10]}, actual_order={ordered_task_names[:10]}"
            raise SystemExit(
                "Partial TMax training task_name values must be an ordered "
                f"subsequence of the selected prefix: {detail}"
            )
    elif ordered_task_names != expected_names:
        mismatch_index = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(ordered_task_names, expected_names, strict=False)
                )
                if actual != expected
            ),
            min(len(ordered_task_names), len(expected_names)),
        )
        actual = (
            ordered_task_names[mismatch_index]
            if mismatch_index < len(ordered_task_names)
            else "<missing>"
        )
        expected = (
            expected_names[mismatch_index] if mismatch_index < len(expected_names) else "<end>"
        )
        raise SystemExit(
            "Strict TMax training JSONL must match the deterministic selected "
            f"prefix exactly; row {mismatch_index + 1}: "
            f"expected {expected}, found {actual} "
            f"(expected_rows={len(expected_names)}, actual_rows={row_count})."
        )
    if mismatched_image_paths:
        preview = ", ".join(mismatched_image_paths[:10])
        raise SystemExit(
            f"TMax training JSONL references SIF path(s) outside the "
            f"configured task mapping: {preview}"
        )
    if missing_images:
        preview = ", ".join(missing_images[:10])
        suffix = " ..." if len(missing_images) > 10 else ""
        raise SystemExit(
            f"Missing or empty SIF(s) for {len(missing_images)}/{row_count} "
            f"TMax training row(s): {preview}{suffix}"
        )
    return row_count


def main() -> int:
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if args.validate_existing:
        expected_tasks = load_task_prefix(args)
        if not args.only_ready:
            require_complete_prefix(args, expected_tasks)
        row_count = validate_existing_output(
            output,
            expected_tasks=expected_tasks,
            image_dir=image_dir,
            allow_partial=args.only_ready,
        )
        print(
            f"Validated {row_count} TMax training row(s) in {output}; "
            "all referenced SIFs are non-empty"
        )
        return 0
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
