#!/usr/bin/env python3
"""Snapshot built TMax-15K runtime images into Apptainer ``.sif`` files (Route A).

Docker-free Slurm flow: build the runtime images with ``build_images.py`` on a
docker-capable box, run this to convert each into a ``.sif``, copy the output dir
to the cluster, then submit with::

    --runtime-backend apptainer --apptainer-image-dir <dir>

Polar's ApptainerRuntime launches the ``.sif`` directly, so the Slurm nodes need
no docker (only ``apptainer``). This step reads the images out of the local docker
daemon (``docker-daemon://``), so it must run where the docker images live.

    uv run python examples/tmax-15k/prepare_apptainer_images.py \
        --dataset-dir ~/tmax15k --image-dir ~/tmax15k-sif --max-tasks 10
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dataset import load_tasks, runtime_image_for, sif_filename_for


def apptainer_binary() -> str:
    """Mirror Polar's ApptainerRuntime: ``POLAR_APPTAINER_BIN``, else apptainer, else singularity."""
    override = os.environ.get("POLAR_APPTAINER_BIN")
    if override:
        return override
    for candidate in ("apptainer", "singularity"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise SystemExit("No apptainer/singularity binary found (set POLAR_APPTAINER_BIN).")


def docker_image_exists(image_ref: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Exported Harbor dataset directory.")
    parser.add_argument("--image-dir", required=True, help="Output directory for the .sif files.")
    parser.add_argument("--max-tasks", type=int, default=-1, help="Max tasks to convert. -1 = all.")
    parser.add_argument("--task", action="append", default=[], help="Only convert these task(s). Repeatable.")
    parser.add_argument("--force", action="store_true", help="Rebuild .sif files that already exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = load_tasks(args.dataset_dir, max_tasks=args.max_tasks, names=args.task or None)
    binary = apptainer_binary()
    image_dir = Path(args.image_dir).expanduser()
    image_dir.mkdir(parents=True, exist_ok=True)

    built = skipped = failed = 0
    for task in tasks:
        docker_ref = runtime_image_for(task.name)
        sif = image_dir / sif_filename_for(task.name)
        if sif.is_file() and not args.force:
            print(f"skip   {task.name} -> {sif.name} (exists)")
            skipped += 1
            continue
        if not docker_image_exists(docker_ref):
            print(f"MISS   {task.name}: docker image {docker_ref} not found — run build_images.py first", file=sys.stderr)
            failed += 1
            continue
        # Build to a temp file, then atomic-rename, so an interrupted build never
        # leaves a half-written .sif that submit would treat as ready.
        tmp = sif.with_name(f".{sif.name}.tmp-{os.getpid()}")
        cmd = [binary, "build", "--force", str(tmp), f"docker-daemon://{docker_ref}"]
        print("+", " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True)
            tmp.replace(sif)
            built += 1
        except subprocess.CalledProcessError as exc:
            print(f"FAIL   {task.name}: {binary} build exit {exc.returncode}", file=sys.stderr)
            failed += 1
        finally:
            if tmp.exists():
                tmp.unlink()

    print(f"\nDone. built={built} skipped={skipped} failed={failed}  ->  {image_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
