#!/usr/bin/env python3
"""Prepare local Apptainer SIF images for the full SWE-Gym train dataset."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from sample_tasks import fetch_all_instances, registry_image_for_instance_id  # noqa: E402


DEFAULT_IMAGE_DIR = PROJECT_ROOT / "tmp" / "swegym_apptainer_images"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "tmp" / "apptainer_cache"
DEFAULT_TMP_DIR = PROJECT_ROOT / "tmp" / "apptainer_tmp"
DEFAULT_AGENT_CLI_DIR = PROJECT_ROOT / "tmp" / "swegym_agent_cli" / "opt_node"
NODE_VERSION = "22.11.0"
NODE_DIST_URL = (
    f"https://nodejs.org/dist/v{NODE_VERSION}/"
    f"node-v{NODE_VERSION}-linux-x64.tar.xz"
)
AGENT_CLI_PACKAGES = (
    "@openai/codex@latest",
    "@anthropic-ai/claude-code@latest",
    "@qwen-code/qwen-code@latest",
    "opencode-ai@latest",
    "@mariozechner/pi-coding-agent@latest",
)
REQUIRED_AGENT_BINS = (
    "node",
    "npm",
    "npx",
    "codex",
    "claude",
    "qwen",
    "opencode",
    "pi",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Only prepare the SIF image for this instance_id.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory for prepared .sif images.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Apptainer cache directory.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=DEFAULT_TMP_DIR,
        help="Apptainer temporary build directory.",
    )
    parser.add_argument(
        "--agent-cli-dir",
        type=Path,
        default=DEFAULT_AGENT_CLI_DIR,
        help="Host directory mounted as /opt/node in task containers.",
    )
    parser.add_argument(
        "--skip-cli",
        action="store_true",
        help="Only prepare SIF images; do not prepare the shared CLI directory.",
    )
    parser.add_argument(
        "--force-cli",
        action="store_true",
        help="Rebuild and re-extract the shared Node/agent CLI directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-pull SIF images even when the output file exists.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Number of concurrent apptainer pull jobs.",
    )
    parser.add_argument(
        "--refresh-dataset-cache",
        action="store_true",
        help="Refresh the cached dataset rows before preparing images.",
    )
    return parser.parse_args()


def select_instances(args: argparse.Namespace) -> list[dict[str, object]]:
    instances = fetch_all_instances(refresh=args.refresh_dataset_cache)
    if args.instance_id:
        wanted = set(args.instance_id)
        selected = [instance for instance in instances if str(instance.get("instance_id")) in wanted]
        missing = sorted(wanted - {str(instance.get("instance_id")) for instance in selected})
        if missing:
            raise SystemExit(f"Unknown instance_id(s): {', '.join(missing)}")
        return selected
    return instances


def sif_path_for_instance(instance_id: str, image_dir: Path) -> Path:
    if "/" in instance_id or "\0" in instance_id:
        raise ValueError(f"instance_id is not safe for a filename: {instance_id!r}")
    return image_dir / f"{instance_id}.sif"


def apptainer_registry_uri(image_ref: str) -> str:
    if image_ref.startswith(("docker://", "oras://", "library://", "shub://")):
        return image_ref
    return f"docker://{image_ref}"


def image_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def ensure_sif(
    instance: dict[str, object],
    *,
    image_dir: Path,
    force: bool,
    env: dict[str, str],
) -> tuple[str, str]:
    instance_id = str(instance["instance_id"])
    target = sif_path_for_instance(instance_id, image_dir)
    if image_ready(target) and not force:
        return ("skipped", str(target))

    image_ref = registry_image_for_instance_id(instance_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(
        f".{target.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    if tmp_path.exists():
        tmp_path.unlink()

    command = [
        _apptainer_binary(),
        "pull",
        "--force",
        str(tmp_path),
        apptainer_registry_uri(image_ref),
    ]
    print("+", " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, env=env)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return ("pulled", str(target))


def prepare_sifs(
    instances: list[dict[str, object]],
    *,
    image_dir: Path,
    cache_dir: Path,
    tmp_dir: Path,
    force: bool,
    jobs: int,
) -> None:
    env = {
        **os.environ,
        "APPTAINER_CACHEDIR": str(cache_dir.resolve()),
        "APPTAINER_TMPDIR": str(tmp_dir.resolve()),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    if jobs <= 1:
        for instance in instances:
            status, path = ensure_sif(
                instance, image_dir=image_dir, force=force, env=env
            )
            print(f"{status}: {path}", flush=True)
        return

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                ensure_sif, instance, image_dir=image_dir, force=force, env=env
            )
            for instance in instances
        ]
        for future in as_completed(futures):
            status, path = future.result()
            print(f"{status}: {path}", flush=True)


def _apptainer_binary() -> str:
    override = os.environ.get("POLAR_APPTAINER_BIN")
    if override:
        return override
    for candidate in ("/usr/bin/apptainer", "/bin/apptainer"):
        if Path(candidate).is_file():
            return candidate
    resolved = shutil.which("apptainer")
    if resolved:
        return resolved
    return "apptainer"


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, check=True, env=env)


def _agent_cli_missing_bins(agent_cli_dir: Path) -> list[str]:
    return [
        name for name in REQUIRED_AGENT_BINS
        if not (agent_cli_dir / "bin" / name).is_file()
    ]


def ensure_agent_cli_dir(agent_cli_dir: Path, *, force: bool) -> None:
    missing_bins = _agent_cli_missing_bins(agent_cli_dir)
    if not missing_bins and not force:
        print(f"Shared agent CLI directory already exists: {agent_cli_dir}")
        return

    agent_cli_dir = agent_cli_dir.resolve()
    agent_cli_dir.parent.mkdir(parents=True, exist_ok=True)
    if agent_cli_dir.exists():
        shutil.rmtree(agent_cli_dir)
    agent_cli_dir.mkdir(parents=True)

    run_command([
        "bash",
        "-c",
        (
            f"curl -fsSL {shlex.quote(NODE_DIST_URL)} | "
            f"tar -xJ --strip-components=1 -C {shlex.quote(str(agent_cli_dir))}"
        ),
    ])

    npm_bin = agent_cli_dir / "bin" / "npm"
    env = {
        **os.environ,
        "PATH": f"{agent_cli_dir / 'bin'}:{os.environ.get('PATH', '')}",
    }
    run_command(
        [
            str(npm_bin),
            "install",
            "-g",
            "--no-audit",
            "--no-fund",
            f"--prefix={agent_cli_dir}",
            *AGENT_CLI_PACKAGES,
        ],
        env=env,
    )

    missing_bins = _agent_cli_missing_bins(agent_cli_dir)
    if missing_bins:
        raise RuntimeError(
            "agent CLI setup did not create expected executable(s): "
            + ", ".join(missing_bins)
        )
    print(f"Prepared shared agent CLI directory: {agent_cli_dir}")


def main() -> int:
    args = parse_args()
    if shutil.which(_apptainer_binary()) is None and not Path(_apptainer_binary()).is_file():
        raise SystemExit("apptainer not found")

    if not args.skip_cli:
        ensure_agent_cli_dir(args.agent_cli_dir, force=args.force_cli)

    instances = select_instances(args)
    if not instances:
        raise SystemExit("No instances selected.")

    print(
        f"Preparing {len(instances)} Apptainer SIF image(s) with {max(args.jobs, 1)} pull job(s).",
        flush=True,
    )
    print(f"Image dir: {args.image_dir.resolve()}", flush=True)
    print(f"Cache dir: {args.cache_dir.resolve()}", flush=True)
    print(f"Tmp dir: {args.tmp_dir.resolve()}", flush=True)
    prepare_sifs(
        instances,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        tmp_dir=args.tmp_dir,
        force=args.force,
        jobs=max(args.jobs, 1),
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
