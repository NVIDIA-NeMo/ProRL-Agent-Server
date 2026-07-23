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
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from build_images import IMAGE_LAYOUT_VERSION, IMAGE_VERSION_LABEL
from dataset import TmaxTask, load_tasks, runtime_image_for, sanitize, sif_filename_for

EXAMPLE_DIR = Path(__file__).resolve().parent
RUNTIME_DOCKERFILE_DIR = EXAMPLE_DIR / "runtime"
DEFAULT_DATA_ROOT = Path("/lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/jiaruiy/spilot/data")
DEFAULT_DATASET_DIR = DEFAULT_DATA_ROOT / "tmax-15k"
DEFAULT_IMAGE_DIR = DEFAULT_DATA_ROOT / "tmax-15k-sif"
DIRECT_CONTEXT_DIR = "/opt/polar-tmax-context"
DIRECT_FINAL_PACK_ATTEMPTS = 3


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
    return not path.is_symlink() and path.is_file() and path.stat().st_size > 0


@contextmanager
def isolated_build_workspace(
    task_name: str,
    env: dict[str, str],
) -> Iterator[tuple[dict[str, str], Path, Path]]:
    """Give each Apptainer build an independent host temporary workspace.

    TMax Dockerfiles commonly use fixed helper names such as
    ``/tmp/post_install.sh``. Isolating all Apptainer/process temporary roots
    prevents concurrent builders from sharing scratch state while leaving the
    image's own ``/tmp`` semantics intact.
    """
    tmp_root_value = (
        env.get("APPTAINER_TMPDIR")
        or env.get("SINGULARITY_TMPDIR")
        or env.get("TMPDIR")
        or tempfile.gettempdir()
    )
    tmp_root = Path(tmp_root_value).expanduser()
    tmp_root.mkdir(parents=True, exist_ok=True)
    prefix = (sanitize(task_name) or "task")[:48]
    workspace = Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=tmp_root))
    apptainer_tmp = workspace / "apptainer-tmp"
    process_tmp = workspace / "process-tmp"
    container_tmp = workspace / "container-tmp"
    for path in (apptainer_tmp, process_tmp, container_tmp):
        path.mkdir()
    container_tmp.chmod(0o1777)
    sandbox = workspace / "rootfs"

    build_env = {
        **env,
        "APPTAINER_TMPDIR": str(apptainer_tmp),
        "SINGULARITY_TMPDIR": str(apptainer_tmp),
        "TMPDIR": str(process_tmp),
    }
    try:
        yield build_env, container_tmp, sandbox
    finally:
        try:
            shutil.rmtree(workspace)
        except FileNotFoundError:
            pass
        except OSError as exc:
            # A failed %post can leave unusual ownership/modes behind. Never
            # replace the primary build exception with a cleanup exception,
            # but make the leaked node-local scratch visible to operators.
            print(
                f"WARNING: failed to remove isolated build workspace {workspace}: {exc}",
                file=sys.stderr,
                flush=True,
            )


def merge_private_tmp(
    container_tmp: Path,
    sandbox: Path,
    *,
    env: dict[str, str],
) -> None:
    """Persist the build-time private ``/tmp`` into the sandbox rootfs.

    Apptainer bind mounts are intentionally absent from the completed image.
    Building to a sandbox first lets concurrent ``%post`` sections use private
    host ``/tmp`` directories, after which this merge restores Docker's layer
    semantics before the sandbox is squashed into the final SIF.
    """
    sandbox_tmp = sandbox / "tmp"
    try:
        sandbox_tmp_mode = sandbox_tmp.lstat().st_mode
    except FileNotFoundError:
        sandbox_tmp.mkdir(parents=True)
    else:
        if stat.S_ISLNK(sandbox_tmp_mode) or not stat.S_ISDIR(sandbox_tmp_mode):
            raise RuntimeError(
                f"refusing to merge private /tmp through non-directory sandbox path: {sandbox_tmp}"
            )
    run_command(["cp", "-a", f"{container_tmp}/.", str(sandbox_tmp)], env=env)


def pack_final_sif(
    command: list[str],
    *,
    env: dict[str, str],
    tmp: Path,
    attempts: int,
) -> None:
    """Run final image packaging without rebuilding the prepared sandbox.

    squashfs-tools 4.7.5 has a confirmed duplicate-checking race that can
    abort with ``BUG in get_virt_disk``. The sandbox is already complete at
    this point, so retry only this deterministic-input compression stage.
    """
    if attempts < 1:
        raise ValueError("final SIF packaging attempts must be positive")

    for attempt in range(1, attempts + 1):
        attempt_command = command
        if attempts > 1 and attempt == attempts:
            # The known 4.7.5 race lives only in duplicate checking. Preserve
            # normal deduplication for successful images, but make the final
            # retry deterministic instead of repeatedly exercising that path.
            attempt_command = _disable_mksquashfs_duplicates(command)
        tmp.unlink(missing_ok=True)
        try:
            run_command(attempt_command, env=env)
        except subprocess.CalledProcessError:
            tmp.unlink(missing_ok=True)
            if attempt >= attempts:
                raise
            delay_seconds = attempt
            print(
                f"WARNING: final SIF packaging failed (attempt {attempt}/{attempts}); "
                f"retrying in {delay_seconds}s: {tmp}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay_seconds)
            continue

        if not sif_ready(tmp):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"final SIF packaging returned success without a non-empty regular file: {tmp}"
            )
        return


def _disable_mksquashfs_duplicates(command: list[str]) -> list[str]:
    fallback = list(command)
    try:
        args_index = fallback.index("--mksquashfs-args") + 1
    except ValueError:
        # The final two arguments are always output SIF and input sandbox/image.
        fallback[-2:-2] = ["--mksquashfs-args", "-no-duplicates"]
        return fallback

    mksquashfs_args = shlex.split(fallback[args_index])
    if "-no-duplicates" not in mksquashfs_args:
        mksquashfs_args.append("-no-duplicates")
        fallback[args_index] = shlex.join(mksquashfs_args)
    return fallback


def pack_sandbox_sif_without_reimport(
    binary: str,
    sandbox: Path,
    tmp: Path,
    *,
    env: dict[str, str],
    mksquashfs_args: str,
) -> None:
    """Pack an already-built sandbox without Apptainer's tar reimport.

    Some security-oriented TMax tasks intentionally contain symlinks whose
    lexical target escapes the rootfs. They are inert inside the container
    mount namespace, and SquashFS represents them faithfully, but Apptainer's
    sandbox-to-SIF path first copies the rootfs through an archive extractor
    that rejects those links. Build the standard SIF system partition directly
    from SquashFS as a compatibility fallback.
    """

    squashfs = tmp.with_name(f"{tmp.name}.rootfs.squashfs")
    tmp.unlink(missing_ok=True)
    squashfs.unlink(missing_ok=True)
    try:
        command = ["mksquashfs", str(sandbox), str(squashfs), "-noappend"]
        command.extend(shlex.split(mksquashfs_args))
        run_command(command, env=env)
        run_command([binary, "sif", "new", str(tmp)], env=env)
        run_command(
            [
                binary,
                "sif",
                "add",
                str(tmp),
                str(squashfs),
                "--groupid",
                "1",
                "--datatype",
                "4",
                "--parttype",
                "2",
                "--partfs",
                "1",
                "--partarch",
                "2",
            ],
            env=env,
        )
        if not sif_ready(tmp):
            raise RuntimeError(
                f"direct SquashFS packaging returned no non-empty SIF: {tmp}"
            )
    finally:
        squashfs.unlink(missing_ok=True)


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

            with isolated_build_workspace(task.name, env) as (
                build_env,
                container_tmp,
                sandbox,
            ):
                tmp = target.with_name(
                    f".{target.name}.tmp-{os.getpid()}-{threading.get_ident()}"
                )
                if tmp.exists():
                    tmp.unlink()
                if builder == "direct-apptainer":
                    definition_path = definition_dir / f"{task.name}.def"
                    render_direct_definition(task, definition_path, base_sif=base_sif)
                    sandbox_cmd = [binary, "build", "--force", "--sandbox"]
                    if apptainer_fakeroot:
                        sandbox_cmd.append("--fakeroot")
                    sandbox_cmd.extend(
                        [
                            "--bind",
                            f"{container_tmp}:/tmp",
                            str(sandbox),
                            str(definition_path),
                        ]
                    )
                    run_command(sandbox_cmd, env=build_env)
                    merge_private_tmp(container_tmp, sandbox, env=build_env)

                    cmd = [binary, "build", "--force"]
                    if apptainer_fakeroot:
                        cmd.append("--fakeroot")
                    if mksquashfs_args:
                        cmd.extend(["--mksquashfs-args", mksquashfs_args])
                    cmd.extend([str(tmp), str(sandbox)])
                elif builder == "docker-daemon":
                    cmd = [binary, "build", "--force"]
                    if mksquashfs_args:
                        cmd.extend(["--mksquashfs-args", mksquashfs_args])
                    if docker_bin is None:
                        raise RuntimeError("docker-daemon builder requires Docker")
                    if not skip_docker_build:
                        build_docker_runtime(
                            task,
                            docker_bin=docker_bin,
                            force=force_docker,
                            env=build_env,
                        )
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
                    try:
                        pack_final_sif(
                            cmd,
                            env=build_env,
                            tmp=tmp,
                            attempts=(
                                DIRECT_FINAL_PACK_ATTEMPTS
                                if builder == "direct-apptainer"
                                else 1
                            ),
                        )
                    except subprocess.CalledProcessError:
                        if builder != "direct-apptainer":
                            raise
                        print(
                            "WARNING: Apptainer sandbox reimport failed; "
                            f"packing SquashFS system partition directly: {task.name}",
                            file=sys.stderr,
                            flush=True,
                        )
                        pack_sandbox_sif_without_reimport(
                            binary,
                            sandbox,
                            tmp,
                            env=build_env,
                            mksquashfs_args=mksquashfs_args,
                        )
                    tmp.replace(target)
                finally:
                    tmp.unlink(missing_ok=True)
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
