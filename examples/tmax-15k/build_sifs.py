#!/usr/bin/env python3
"""Build TMax-15K runtime SIF images with deterministic sharding.

This is the Slurm-friendly version of the TMax image preparation path.
By default it avoids Docker entirely: each task's constrained Dockerfile is
translated into a temporary Apptainer definition and built directly into a SIF.

Each selected task is handled independently, so callers can split the dataset
across many CPU jobs:

  python examples/tmax-15k/build_sifs.py \
    --dataset-dir ~/tmax15k --image-dir ~/tmax15k-sif \
    --num-shards 64 --shard-index 0 --jobs 1

The legacy Docker-daemon conversion path is still available with
``--builder docker-daemon`` for docker-capable machines.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shlex
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from build_images import IMAGE_LAYOUT_VERSION, IMAGE_VERSION_LABEL
from dataset import TmaxTask, load_tasks, runtime_image_for, sif_filename_for

EXAMPLE_DIR = Path(__file__).resolve().parent
RUNTIME_DOCKERFILE_DIR = EXAMPLE_DIR / "runtime"
DEFAULT_DATA_ROOT = Path("/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/data")
DEFAULT_DATASET_DIR = DEFAULT_DATA_ROOT / "tmax-15k"
DEFAULT_IMAGE_DIR = DEFAULT_DATA_ROOT / "tmax-15k-sif"
DIRECT_CONTEXT_DIR = "/opt/polar-tmax-context"


@dataclass(frozen=True)
class DockerfilePlan:
    base_image: str
    env: list[tuple[str, str]]
    steps: list[tuple[str, str, str]]


def apptainer_binary() -> str:
    override = os.environ.get("POLAR_APPTAINER_BIN")
    if override:
        return override
    for candidate in ("apptainer", "singularity"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise SystemExit("No apptainer/singularity binary found (set POLAR_APPTAINER_BIN).")


def docker_binary() -> str:
    override = os.environ.get("POLAR_DOCKER_BIN")
    if override:
        if shutil.which(override) or Path(override).is_file():
            return override
        raise SystemExit(f"POLAR_DOCKER_BIN is set but not executable/found: {override}")
    resolved = shutil.which("docker")
    if resolved:
        return resolved
    raise SystemExit(
        "Docker CLI not found in PATH. TMax SIF builds need Docker because each "
        "task starts from environment/Dockerfile and conversion reads docker-daemon://. "
        "Run on docker-capable nodes or set POLAR_DOCKER_BIN."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("TMAX_DATASET_DIR", str(DEFAULT_DATASET_DIR)),
        help="Exported Harbor dataset directory. Defaults to TMAX_DATASET_DIR or the local spilot data dir.",
    )
    parser.add_argument(
        "--image-dir",
        default=(
            os.environ.get("APPTAINER_IMAGE_DIR")
            or os.environ.get("POLAR_SIF_DIR")
            or str(DEFAULT_IMAGE_DIR)
        ),
        help="Output directory for .sif files. Defaults to APPTAINER_IMAGE_DIR/POLAR_SIF_DIR or the local spilot data dir.",
    )
    parser.add_argument("--task", action="append", default=[], help="Only build these task(s). Repeatable.")
    parser.add_argument("--max-tasks", type=int, default=-1, help="Max tasks before sharding. -1 = all.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split selected tasks into N shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index to build.")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent builds within this shard.")
    parser.add_argument(
        "--builder",
        choices=("direct-apptainer", "docker-daemon"),
        default=os.environ.get("TMAX_SIF_BUILDER", "direct-apptainer"),
        help="direct-apptainer builds SIFs without Docker. docker-daemon preserves the old Docker path.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild .sif files that already exist.")
    parser.add_argument("--force-docker", action="store_true", help="Rebuild docker runtime images too.")
    parser.add_argument(
        "--skip-docker-build",
        action="store_true",
        help="Only convert existing local docker runtime images to .sif.",
    )
    parser.add_argument(
        "--cache-root",
        default=(
            os.environ.get("POLAR_JOB_CACHE_ROOT")
            or f"/tmp/polar-tmax-sifbuild-{os.environ.get('USER') or os.getuid()}"
        ),
        help="Node-local scratch for Apptainer cache/tmp.",
    )
    parser.add_argument(
        "--mksquashfs-args",
        default=os.environ.get("POLAR_MKSQUASHFS_ARGS", "-processors 1 -mem 1024M"),
        help="Value passed to apptainer build --mksquashfs-args. Use '' to disable.",
    )
    parser.add_argument(
        "--apptainer-fakeroot",
        action="store_true",
        default=os.environ.get("TMAX_SIF_APPTAINER_FAKEROOT", "0") == "1",
        help="Pass --fakeroot to apptainer build for direct-apptainer definition builds.",
    )
    parser.add_argument(
        "--base-sif",
        default=os.environ.get("TMAX_SIF_BASE_SIF", ""),
        help="Optional local Ubuntu 22.04 SIF for direct-apptainer Bootstrap: localimage.",
    )
    return parser.parse_args()


def parse_env_payload(payload: str, *, dockerfile: Path) -> list[tuple[str, str]]:
    tokens = shlex.split(payload)
    if not tokens:
        raise ValueError(f"{dockerfile}: empty ENV instruction")

    if any("=" in token for token in tokens):
        pairs: list[tuple[str, str]] = []
        for token in tokens:
            if "=" not in token:
                raise ValueError(f"{dockerfile}: mixed ENV syntax is unsupported: {payload}")
            key, value = token.split("=", 1)
            pairs.append((key, value))
        return pairs

    if len(tokens) < 2:
        raise ValueError(f"{dockerfile}: unsupported ENV instruction: {payload}")
    return [(tokens[0], " ".join(tokens[1:]))]


def parse_task_dockerfile(task: TmaxTask) -> DockerfilePlan:
    dockerfile = task.environment_dir / "Dockerfile"
    base_image = ""
    env: list[tuple[str, str]] = []
    steps: list[tuple[str, str, str]] = []

    for line_no, raw in enumerate(dockerfile.read_text().splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            raise ValueError(f"{dockerfile}:{line_no}: line continuations are unsupported")
        if " " in stripped:
            instruction, payload = stripped.split(None, 1)
        else:
            instruction, payload = stripped, ""
        instruction = instruction.upper()

        if instruction == "FROM":
            base_image = payload.strip()
        elif instruction == "ENV":
            env.extend(parse_env_payload(payload, dockerfile=dockerfile))
        elif instruction == "COPY":
            tokens = shlex.split(payload)
            if len(tokens) != 2 or tokens[0].startswith("--"):
                raise ValueError(f"{dockerfile}:{line_no}: unsupported COPY instruction: {payload}")
            src, dst = tokens
            if not (task.environment_dir / src).exists():
                raise ValueError(f"{dockerfile}:{line_no}: COPY source not found: {src}")
            steps.append(("copy", src, dst))
        elif instruction == "RUN":
            steps.append(("run", payload, ""))
        else:
            raise ValueError(f"{dockerfile}:{line_no}: unsupported instruction: {instruction}")

    if base_image != "ubuntu:22.04":
        raise ValueError(f"{dockerfile}: expected FROM ubuntu:22.04, got {base_image!r}")
    return DockerfilePlan(base_image=base_image, env=env, steps=steps)


def runtime_env(task_env: list[tuple[str, str]]) -> list[tuple[str, str]]:
    merged: dict[str, str] = {}
    for key, value in task_env:
        merged[key] = value
    merged.setdefault("DEBIAN_FRONTEND", "noninteractive")
    merged["NPM_CONFIG_UPDATE_NOTIFIER"] = "false"
    return list(merged.items())


def shell_value(value: str) -> str:
    if "$" in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
        return f'"{escaped}"'
    return shlex.quote(value)


def export_line(key: str, value: str) -> str:
    return f"export {key}={shell_value(value)}"


def render_direct_definition(task: TmaxTask, definition_path: Path, *, base_sif: str = "") -> None:
    plan = parse_task_dockerfile(task)
    env = runtime_env(plan.env)
    if base_sif:
        bootstrap = "localimage"
        source = str(Path(base_sif).expanduser().resolve())
    else:
        bootstrap = "docker"
        source = plan.base_image

    lines = [
        f"Bootstrap: {bootstrap}",
        f"From: {source}",
        "",
        "%labels",
        f"    {IMAGE_VERSION_LABEL} {IMAGE_LAYOUT_VERSION}",
        "",
        "%files",
    ]
    for step in plan.steps:
        if step[0] == "copy":
            _, src, _ = step
            lines.append(f"    {task.environment_dir / src} {DIRECT_CONTEXT_DIR}/{src}")
    lines.extend(
        [
            "",
            "%environment",
        ]
    )
    lines.extend(f"    {export_line(key, value)}" for key, value in env)
    lines.extend(
        [
            "",
            "%post",
            "    set -eu",
            f"    CONTEXT={shlex.quote(DIRECT_CONTEXT_DIR)}",
            '    if [ -d "${CONTEXT}/environment" ] && [ -f "${CONTEXT}/environment/Dockerfile" ]; then',
            '        CONTEXT="${CONTEXT}/environment"',
            "    fi",
            "    copy_from_context() {",
            '        src="$1"',
            '        dst="$2"',
            '        mkdir -p "$(dirname "${dst}")"',
            '        cp -a "${CONTEXT}/${src}" "${dst}"',
            "    }",
        ]
    )
    lines.extend(f"    {export_line(key, value)}" for key, value in env)
    lines.extend(
        [
            '    mkdir -p /etc/apt/apt.conf.d',
            '    printf \'APT::Sandbox::User "root";\\n\' > /etc/apt/apt.conf.d/99polar-no-sandbox',
        ]
    )

    for step in plan.steps:
        if step[0] == "copy":
            _, src, dst = step
            lines.append(f"    copy_from_context {shlex.quote(src)} {shlex.quote(dst)}")
        elif step[0] == "run":
            _, command, _ = step
            lines.append(f"    {command}")
        else:
            raise AssertionError(f"unknown step kind: {step[0]}")

    lines.extend(
        [
            "    apt-get update",
            "    apt-get install -y --no-install-recommends ca-certificates curl gnupg git",
            "    apt-get purge -y nodejs npm libnode-dev libnode72 >/dev/null 2>&1 || true",
            "    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
            "    apt-get install -y --no-install-recommends nodejs",
            "    apt-get clean",
            "    rm -rf /var/lib/apt/lists/*",
            "    node --version | grep -q '^v22\\.'",
            "    mkdir -p /polar/session",
            f"    rm -rf {shlex.quote(DIRECT_CONTEXT_DIR)}",
            "",
        ]
    )

    definition_path.parent.mkdir(parents=True, exist_ok=True)
    definition_path.write_text("\n".join(lines))


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def image_exists(image_ref: str, *, docker_bin: str) -> bool:
    return subprocess.run(
        [docker_bin, "image", "inspect", image_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def image_layout_version(image_ref: str, *, docker_bin: str) -> str | None:
    result = subprocess.run(
        [
            docker_bin,
            "image",
            "inspect",
            "--format",
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


def docker_runtime_ready(image_ref: str, *, docker_bin: str) -> bool:
    return image_exists(image_ref, docker_bin=docker_bin) and (
        image_layout_version(image_ref, docker_bin=docker_bin) == IMAGE_LAYOUT_VERSION
    )


def build_docker_runtime(
    task: TmaxTask,
    *,
    docker_bin: str,
    force: bool,
    env: dict[str, str],
) -> None:
    runtime_image = runtime_image_for(task.name)
    if docker_runtime_ready(runtime_image, docker_bin=docker_bin) and not force:
        print(f"skip docker: {runtime_image}", flush=True)
        return

    from dataset import env_image_for

    env_image = env_image_for(task.name)
    run_command([docker_bin, "build", "--tag", env_image, str(task.environment_dir)], env=env)
    run_command(
        [
            docker_bin,
            "build",
            "--build-arg",
            f"BASE_IMAGE={env_image}",
            "--build-arg",
            f"POLAR_TMAX_IMAGE_VERSION={IMAGE_LAYOUT_VERSION}",
            "--tag",
            runtime_image,
            str(RUNTIME_DOCKERFILE_DIR),
        ],
        env=env,
    )


def sif_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def build_one(
    task: TmaxTask,
    *,
    image_dir: Path,
    binary: str,
    builder: str,
    definition_dir: Path,
    base_sif: str,
    docker_bin: str | None,
    env: dict[str, str],
    force: bool,
    force_docker: bool,
    skip_docker_build: bool,
    mksquashfs_args: str,
    apptainer_fakeroot: bool,
) -> tuple[str, str, str]:
    target = image_dir / sif_filename_for(task.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if sif_ready(target) and not force:
                return ("skip", task.name, str(target))

            tmp = target.with_name(
                f".{target.name}.tmp-{os.getpid()}-{threading.get_ident()}"
            )
            if tmp.exists():
                tmp.unlink()
            cmd = [binary, "build", "--force"]
            if apptainer_fakeroot and builder == "direct-apptainer":
                cmd.append("--fakeroot")
            if mksquashfs_args:
                cmd.extend(["--mksquashfs-args", mksquashfs_args])
            if builder == "direct-apptainer":
                definition_path = definition_dir / f"{task.name}.def"
                render_direct_definition(task, definition_path, base_sif=base_sif)
                cmd.extend([str(tmp), str(definition_path)])
            elif builder == "docker-daemon":
                if docker_bin is None:
                    raise RuntimeError("docker-daemon builder requires Docker")
                if not skip_docker_build:
                    build_docker_runtime(task, docker_bin=docker_bin, force=force_docker, env=env)
                docker_ref = runtime_image_for(task.name)
                if not image_exists(docker_ref, docker_bin=docker_bin):
                    raise RuntimeError(
                        f"docker image {docker_ref} not found; run without --skip-docker-build "
                        "or build it on this node first"
                    )
                cmd.extend([str(tmp), f"docker-daemon://{docker_ref}"])
            else:
                raise RuntimeError(f"unknown builder: {builder}")
            try:
                run_command(cmd, env=env)
                tmp.replace(target)
            finally:
                if tmp.exists():
                    tmp.unlink()
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return ("built", task.name, str(target))


def shard_tasks(tasks: list[TmaxTask], *, num_shards: int, shard_index: int) -> list[TmaxTask]:
    if num_shards < 1:
        raise SystemExit("ERROR: --num-shards must be >= 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit("ERROR: --shard-index must satisfy 0 <= shard_index < num_shards.")
    if num_shards == 1:
        return tasks
    return [task for index, task in enumerate(tasks) if index % num_shards == shard_index]


def main() -> int:
    args = parse_args()
    if not args.image_dir:
        raise SystemExit("ERROR: set --image-dir or APPTAINER_IMAGE_DIR/POLAR_SIF_DIR.")
    if not (RUNTIME_DOCKERFILE_DIR / "Dockerfile").is_file():
        raise SystemExit(f"No Dockerfile at {RUNTIME_DOCKERFILE_DIR / 'Dockerfile'}")
    if args.base_sif and not Path(args.base_sif).expanduser().is_file():
        raise SystemExit(f"TMAX_SIF_BASE_SIF/--base-sif not found: {args.base_sif}")

    binary = apptainer_binary()
    docker_bin = docker_binary() if args.builder == "docker-daemon" else None
    if args.builder == "direct-apptainer" and (args.force_docker or args.skip_docker_build):
        print(
            "Ignoring --force-docker/--skip-docker-build with --builder direct-apptainer.",
            file=sys.stderr,
            flush=True,
        )
    all_tasks = load_tasks(args.dataset_dir, max_tasks=args.max_tasks, names=args.task or None)
    tasks = shard_tasks(all_tasks, num_shards=args.num_shards, shard_index=args.shard_index)
    image_dir = Path(args.image_dir).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    apptainer_cache = cache_root / "apptainer-cache"
    apptainer_tmp = cache_root / "apptainer-tmp"
    definition_dir = cache_root / "definitions"
    apptainer_cache.mkdir(parents=True, exist_ok=True)
    apptainer_tmp.mkdir(parents=True, exist_ok=True)
    definition_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "APPTAINER_CACHEDIR": str(apptainer_cache),
        "SINGULARITY_CACHEDIR": str(apptainer_cache),
        "APPTAINER_TMPDIR": str(apptainer_tmp),
        "SINGULARITY_TMPDIR": str(apptainer_tmp),
        "TMPDIR": str(apptainer_tmp),
    }

    print(
        f"Building {len(tasks)}/{len(all_tasks)} TMax SIF(s) into {image_dir} "
        f"with {max(args.jobs, 1)} job(s); shard={args.shard_index}/{args.num_shards}; "
        f"builder={args.builder}; fakeroot={args.apptainer_fakeroot}; "
        f"base_sif={args.base_sif or 'docker://ubuntu:22.04'}; "
        f"mksquashfs_args={args.mksquashfs_args!r}",
        flush=True,
    )
    if not tasks:
        print("Shard is empty; nothing to do.")
        return 0

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as executor:
        futures = {
            executor.submit(
                build_one,
                task,
                image_dir=image_dir,
                binary=binary,
                builder=args.builder,
                definition_dir=definition_dir,
                base_sif=args.base_sif,
                docker_bin=docker_bin,
                env=env,
                force=args.force,
                force_docker=args.force_docker,
                skip_docker_build=args.skip_docker_build,
                mksquashfs_args=args.mksquashfs_args,
                apptainer_fakeroot=args.apptainer_fakeroot,
            ): task.name
            for task in tasks
        }
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                status, _, path = future.result()
                print(f"{status}: {task_name} -> {path}", flush=True)
            except Exception as exc:
                failures.append(task_name)
                print(f"FAILED: {task_name}: {exc}", file=sys.stderr, flush=True)

    if failures:
        print(f"\n{len(failures)} build(s) FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
