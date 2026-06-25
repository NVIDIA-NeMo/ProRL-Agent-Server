#!/usr/bin/env python3
"""Build runnable images for TMax-15K-Harbor tasks.

Two layers per task:

  1. ``polar-tmax15k-env:<task>``     — built straight from the task's
     ``environment/Dockerfile`` (self-contained; this is the TMax task sandbox).
  2. ``polar-tmax15k-runtime:<task>`` — the env image + Node.js (runtime/Dockerfile),
     which is what Polar launches. Harness CLIs install at task time.

The verifier (``tests/``) is deliberately NOT baked in — the harbor
evaluator injects it after the agent finishes, so the agent can't read the tests.

Usage:
    python build_images.py --dataset-dir <dir>                 # all tasks in dir
    python build_images.py --dataset-dir <dir> --max-tasks 10  # first 10
    python build_images.py --dataset-dir <dir> --task task_000123_ab12cd34
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dataset import env_image_for, load_tasks, runtime_image_for

EXAMPLE_DIR = Path(__file__).resolve().parent
IMAGE_LAYOUT_VERSION = "2"
IMAGE_VERSION_LABEL = "io.polar.tmax15k-image-version"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Exported Harbor dataset directory.")
    parser.add_argument("--task", action="append", default=[], help="Only build these task(s). Repeatable.")
    parser.add_argument("--max-tasks", type=int, default=-1, help="Max tasks to build. -1 = all.")
    parser.add_argument("--force", action="store_true", help="Rebuild existing images.")
    return parser.parse_args()


def run_command(command: list[str], cwd: str | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True, cwd=cwd)


def image_exists(image_ref: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def image_layout_version(image_ref: str) -> str | None:
    result = subprocess.run(
        [
            "docker", "image", "inspect", "--format",
            '{{ index .Config.Labels "' + IMAGE_VERSION_LABEL + '" }}',
            image_ref,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def main() -> int:
    args = parse_args()
    tasks = load_tasks(args.dataset_dir, max_tasks=args.max_tasks, names=args.task or None)

    runtime_dockerfile_dir = EXAMPLE_DIR / "runtime"
    if not (runtime_dockerfile_dir / "Dockerfile").is_file():
        raise SystemExit(f"No Dockerfile at {runtime_dockerfile_dir / 'Dockerfile'}")

    print(f"Building {len(tasks)} task image(s) ...")
    built = skipped = failed = 0
    for idx, task in enumerate(tasks, 1):
        env_image = env_image_for(task.name)
        runtime_image = runtime_image_for(task.name)

        if image_exists(runtime_image) and not args.force:
            if image_layout_version(runtime_image) == IMAGE_LAYOUT_VERSION:
                print(f"  [{idx}/{len(tasks)}] skip: {runtime_image}")
                skipped += 1
                continue

        try:
            # 1. Task sandbox straight from environment/Dockerfile.
            run_command(["docker", "build", "--tag", env_image, str(task.environment_dir)])
            # 2. Layer Node.js so the harness CLI can run inside the sandbox.
            run_command([
                "docker", "build",
                "--build-arg", f"BASE_IMAGE={env_image}",
                "--build-arg", f"POLAR_TMAX_IMAGE_VERSION={IMAGE_LAYOUT_VERSION}",
                "--tag", runtime_image,
                str(runtime_dockerfile_dir),
            ])
            built += 1
        except subprocess.CalledProcessError as exc:
            print(f"  [{idx}/{len(tasks)}] FAILED {task.name}: {exc}")
            failed += 1

    print(f"\nDone. built={built}  skipped={skipped}  failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
