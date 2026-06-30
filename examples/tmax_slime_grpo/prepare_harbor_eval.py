#!/usr/bin/env python3
"""Build fixed Slime eval JSONL from an exported Harbor task dataset.

The generated rows use each task's official ``tests/test.sh`` and a local
Apptainer-compatible image.  Bare ``.sqsh`` images are accepted intentionally:
Apptainer supports raw SquashFS containers, so an existing enroot image does
not need to be duplicated as a SIF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any, NamedTuple


_RUNTIME_SEMANTICS_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--dataset-revision",
        default=os.environ.get("TMAX_HARBOR_EVAL_REVISION", ""),
    )
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument(
        "--agent-timeout-cap",
        type=float,
        default=float(os.environ.get("TMAX_HARBOR_EVAL_AGENT_TIMEOUT_CAP", "900")),
    )
    parser.add_argument(
        "--verifier-timeout-cap",
        type=float,
        default=float(os.environ.get("TMAX_HARBOR_EVAL_VERIFIER_TIMEOUT_CAP", "600")),
    )
    parser.add_argument(
        "--timeout-overhead",
        type=float,
        default=float(os.environ.get("TMAX_HARBOR_EVAL_TIMEOUT_OVERHEAD", "120")),
    )
    parser.add_argument(
        "--agent-step-limit",
        type=int,
        default=int(
            os.environ.get(
                "TMAX_EXTERNAL_EVAL_AGENT_STEP_LIMIT",
                os.environ.get("TMAX_HARBOR_EVAL_AGENT_STEP_LIMIT", "64"),
            )
        ),
    )
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args()


def _positive(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and parsed > 0 else default


def _capped(value: Any, default: float, cap: float) -> float:
    parsed = _positive(value, default)
    return min(parsed, cap) if cap > 0 else parsed


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resource_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    field: str,
    task_dir: Path,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise SystemExit(f"Invalid {field} for Harbor task {task_dir.name!r}: {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid {field} for Harbor task {task_dir.name!r}: {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise SystemExit(f"Invalid {field} for Harbor task {task_dir.name!r}: {value!r}")
    if parsed < minimum:
        raise SystemExit(
            f"Invalid {field} for Harbor task {task_dir.name!r}: "
            f"require >= {minimum}, got {parsed}"
        )
    return parsed


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_REFERENCE_RE = re.compile(
    r"(?<!\\)\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|"
    r"(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)


class _DockerCommand(NamedTuple):
    argv: tuple[str, ...] | None
    shell: str | None


class _DockerRuntimeMetadata(NamedTuple):
    workdir: str
    env: dict[str, str]
    entrypoint: _DockerCommand | None
    cmd: _DockerCommand | None
    shell: tuple[str, ...]


def _logical_dockerfile_lines(content: str) -> list[str]:
    logical_lines: list[str] = []
    pending = ""
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not pending and (not stripped or stripped.startswith("#")):
            continue
        continued = bool(re.search(r"(?<!\\)(?:\\\\)*\\\s*$", raw_line))
        fragment = re.sub(r"\\\s*$", "", raw_line).strip() if continued else stripped
        pending = f"{pending} {fragment}".strip()
        if not continued:
            if pending:
                logical_lines.append(pending)
            pending = ""
    if pending:
        raise SystemExit("Dockerfile ends with an incomplete line continuation")
    return logical_lines


def _expand_docker_value(value: str, variables: dict[str, str]) -> str:
    expanded = _ENV_REFERENCE_RE.sub(
        lambda match: variables.get(match.group("braced") or match.group("plain"), ""),
        value,
    )
    return expanded.replace(r"\$", "$")


def _parse_env_instruction(argument: str, *, task_dir: Path) -> list[tuple[str, str]]:
    try:
        tokens = shlex.split(argument, posix=True)
    except ValueError as exc:
        raise SystemExit(f"Invalid ENV in {task_dir / 'environment/Dockerfile'}: {exc}") from exc
    if not tokens:
        raise SystemExit(f"Empty ENV in {task_dir / 'environment/Dockerfile'}")
    if "=" not in tokens[0]:
        key, separator, value = argument.partition(" ")
        if not separator:
            raise SystemExit(f"Invalid legacy ENV in {task_dir / 'environment/Dockerfile'}")
        pairs = [(key.strip(), value.strip().strip("\"'"))]
    else:
        if any("=" not in token for token in tokens):
            raise SystemExit(f"Mixed ENV syntax in {task_dir / 'environment/Dockerfile'}")
        pairs = [tuple(token.split("=", 1)) for token in tokens]
    for key, _value in pairs:
        if not _ENV_NAME_RE.fullmatch(key):
            raise SystemExit(
                f"Invalid Docker ENV name {key!r} in {task_dir / 'environment/Dockerfile'}"
            )
    return pairs


def _parse_docker_command(
    argument: str, *, directive: str, task_dir: Path
) -> _DockerCommand:
    stripped = argument.strip()
    if not stripped:
        raise SystemExit(
            f"Empty {directive} in {task_dir / 'environment/Dockerfile'}"
        )
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Invalid JSON {directive} in {task_dir / 'environment/Dockerfile'}: {exc}"
            ) from exc
        if not isinstance(parsed, list) or not parsed or not all(
            isinstance(value, str) and value for value in parsed
        ):
            raise SystemExit(
                f"JSON {directive} must be a non-empty string array in "
                f"{task_dir / 'environment/Dockerfile'}"
            )
        return _DockerCommand(argv=tuple(parsed), shell=None)
    return _DockerCommand(argv=None, shell=stripped)


def _docker_runtime_metadata(task_dir: Path) -> _DockerRuntimeMetadata:
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return _DockerRuntimeMetadata(
            workdir="/root",
            env={},
            entrypoint=None,
            cmd=None,
            shell=("/bin/sh", "-c"),
        )

    workdir = "/root"
    environment: dict[str, str] = {}
    build_args: dict[str, str] = {}
    entrypoint: _DockerCommand | None = None
    cmd: _DockerCommand | None = None
    shell = ("/bin/sh", "-c")
    user = "root"
    seen_from = False

    content = dockerfile.read_text(encoding="utf-8", errors="replace")
    for line in _logical_dockerfile_lines(content):
        match = re.match(r"^([A-Za-z]+)(?:\s+(.*))?$", line, flags=re.DOTALL)
        if match is None:
            raise SystemExit(f"Cannot parse Dockerfile line in {dockerfile}: {line!r}")
        directive = match.group(1).upper()
        argument = (match.group(2) or "").strip()
        if directive == "FROM":
            seen_from = True
            # Runtime config belongs to the final image stage. The audited TB
            # Terminal-Bench 2.x multi-stage tasks can use an independent final
            # base image.
            workdir = "/root"
            environment = {}
            entrypoint = None
            cmd = None
            shell = ("/bin/sh", "-c")
            user = "root"
            continue
        if directive == "ARG":
            key, separator, value = argument.partition("=")
            key = key.strip()
            if not _ENV_NAME_RE.fullmatch(key):
                raise SystemExit(f"Invalid ARG {argument!r} in {dockerfile}")
            if separator:
                build_args[key] = _expand_docker_value(
                    value.strip(), {**build_args, **environment}
                )
            continue
        if not seen_from:
            # Parser directives and global ARG are the only meaningful
            # pre-FROM instructions for this exported task format.
            if directive != "ARG":
                raise SystemExit(f"Unsupported pre-FROM {directive} in {dockerfile}")
            continue
        variables = {**build_args, **environment}
        if directive == "ENV":
            for key, value in _parse_env_instruction(argument, task_dir=task_dir):
                environment[key] = _expand_docker_value(value, variables)
                variables[key] = environment[key]
        elif directive == "WORKDIR":
            value = _expand_docker_value(argument.strip().strip("\"'"), variables)
            if not value:
                raise SystemExit(f"Empty expanded WORKDIR in {dockerfile}")
            workdir = (
                posixpath.normpath(value)
                if value.startswith("/")
                else posixpath.normpath(posixpath.join(workdir, value))
            )
        elif directive == "ENTRYPOINT":
            entrypoint = _parse_docker_command(
                argument, directive=directive, task_dir=task_dir
            )
        elif directive == "CMD":
            cmd = _parse_docker_command(argument, directive=directive, task_dir=task_dir)
        elif directive == "SHELL":
            parsed_shell = _parse_docker_command(
                argument, directive=directive, task_dir=task_dir
            )
            if parsed_shell.argv is None or len(parsed_shell.argv) < 2:
                raise SystemExit(f"Docker SHELL must be a JSON string array in {dockerfile}")
            shell = parsed_shell.argv
        elif directive == "USER":
            user = _expand_docker_value(argument, variables)
        elif directive == "VOLUME":
            raise SystemExit(
                f"Unsupported Docker VOLUME runtime semantics in Harbor task "
                f"{task_dir.name!r}: {argument}"
            )

    if not seen_from:
        raise SystemExit(f"Dockerfile has no FROM instruction: {dockerfile}")
    if user not in {"root", "0", "0:0", "root:root"}:
        raise SystemExit(
            f"Unsupported non-root Docker USER {user!r} in Harbor task {task_dir.name!r}"
        )
    return _DockerRuntimeMetadata(
        workdir=workdir,
        env=environment,
        entrypoint=entrypoint,
        cmd=cmd,
        shell=shell,
    )


def _squashfs_environment(image: Path) -> dict[str, str] | None:
    try:
        magic = image.open("rb").read(4)
    except OSError as exc:
        raise SystemExit(f"Cannot inspect Harbor image {image}: {exc}") from exc
    if image.suffix != ".sqsh" or magic != b"hsqs":
        return None
    unsquashfs = shutil.which("unsquashfs")
    if unsquashfs is None:
        raise SystemExit(
            f"unsquashfs is required to verify OCI ENV metadata in {image}"
        )
    result = subprocess.run(
        [unsquashfs, "-cat", str(image), "etc/environment"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Cannot read /etc/environment from Harbor image {image}: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    environment: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not _ENV_NAME_RE.fullmatch(key):
            raise SystemExit(f"Invalid /etc/environment entry in {image}: {line!r}")
        environment[key] = value
    return environment


def _runtime_environment(
    *,
    docker_metadata: _DockerRuntimeMetadata,
    task_environment: dict[str, Any],
    image: Path,
    task_dir: Path,
) -> dict[str, str]:
    runtime_env = dict(docker_metadata.env)
    if runtime_env:
        image_env = _squashfs_environment(image)
        if image_env is not None:
            missing = sorted(set(runtime_env) - set(image_env))
            if missing:
                raise SystemExit(
                    f"Harbor image {image} is missing Dockerfile ENV key(s) for "
                    f"{task_dir.name!r}: {', '.join(missing)}"
                )
            runtime_env = {key: image_env[key] for key in runtime_env}

    configured_env = task_environment.get("env", {})
    if configured_env is None:
        configured_env = {}
    if not isinstance(configured_env, dict) or not all(
        isinstance(key, str)
        and _ENV_NAME_RE.fullmatch(key)
        and isinstance(value, (str, int, float, bool))
        for key, value in configured_env.items()
    ):
        raise SystemExit(
            f"Invalid environment.env mapping for Harbor task {task_dir.name!r}"
        )
    runtime_env.update({key: str(value) for key, value in configured_env.items()})
    return runtime_env


def _render_docker_command(
    command: _DockerCommand, shell: tuple[str, ...]
) -> list[str]:
    if command.argv is not None:
        return list(command.argv)
    assert command.shell is not None
    return [*shell, command.shell]


def _runtime_init_command(metadata: _DockerRuntimeMetadata) -> str | None:
    entrypoint = metadata.entrypoint
    cmd = metadata.cmd
    if entrypoint is None and cmd is None:
        return None
    if entrypoint is not None and entrypoint.shell is not None:
        argv = _render_docker_command(entrypoint, metadata.shell)
    elif entrypoint is not None:
        assert entrypoint.argv is not None
        argv = list(entrypoint.argv)
        if cmd is not None:
            argv.extend(_render_docker_command(cmd, metadata.shell))
    else:
        assert cmd is not None
        argv = _render_docker_command(cmd, metadata.shell)
    command = shlex.join(argv)
    return (
        "mkdir -p /polar/session/logs && "
        f"({command} >>/polar/session/logs/container-init.log 2>&1 &)"
    )


def _command_metadata(command: _DockerCommand | None) -> dict[str, Any] | None:
    if command is None:
        return None
    if command.argv is not None:
        return {"form": "exec", "argv": list(command.argv)}
    return {"form": "shell", "command": command.shell}


def _reject_unsupported_process_env(
    section: dict[str, Any], *, section_name: str, task_dir: Path
) -> None:
    process_env = section.get("env", {})
    if process_env in (None, {}):
        return
    raise SystemExit(
        f"Unsupported non-empty {section_name}.env for Harbor task "
        f"{task_dir.name!r}; process-scoped env must be modeled explicitly"
    )


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_image(task_dir: Path, image_dir: Path, metadata: dict[str, Any]) -> Path:
    environment = metadata.get("environment", {})
    docker_image = environment.get("docker_image") if isinstance(environment, dict) else None
    candidates: list[Path] = []
    if isinstance(docker_image, str) and ":" in docker_image:
        tag = docker_image.rsplit(":", 1)[1]
        candidates.append(image_dir / f"{task_dir.name}+{tag}.sqsh")
    candidates.extend(sorted(image_dir.glob(f"{task_dir.name}+*.sqsh")))
    candidates.extend(
        path
        for suffix in (".sif", ".sqsh")
        if (path := image_dir / f"{task_dir.name}{suffix}").is_file()
    )

    ready: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            if resolved.is_file() and resolved.stat().st_size > 0:
                ready.append(resolved)
        except OSError:
            pass
    if len(ready) != 1:
        rendered = ", ".join(str(path) for path in ready) or "none"
        raise SystemExit(
            f"Expected exactly one ready image for Harbor task {task_dir.name!r}; "
            f"found {rendered} under {image_dir}"
        )
    return ready[0]


def _task_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks_dir = Path(args.tasks_dir).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    if not tasks_dir.is_dir():
        raise SystemExit(f"Harbor tasks directory does not exist: {tasks_dir}")
    if not image_dir.is_dir():
        raise SystemExit(f"Harbor image directory does not exist: {image_dir}")
    if not args.dataset_name.strip():
        raise SystemExit("--dataset-name must be non-empty")
    if not args.dataset_revision.strip() or args.dataset_revision.strip().lower() == "unknown":
        raise SystemExit("--dataset-revision must pin a concrete Harbor dataset revision")
    if args.max_tasks == 0 or args.max_tasks < -1:
        raise SystemExit("--max-tasks must be -1 or a positive integer")
    if args.agent_step_limit <= 0:
        raise SystemExit("--agent-step-limit must be positive")
    for name in ("agent_timeout_cap", "verifier_timeout_cap", "timeout_overhead"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and non-negative")

    task_dirs = sorted({path.parent for path in tasks_dir.rglob("task.toml")})
    available_task_count = len(task_dirs)
    if args.max_tasks > 0:
        task_dirs = task_dirs[: args.max_tasks]
    if not task_dirs:
        raise SystemExit(f"No Harbor task.toml files found under {tasks_dir}")
    if args.max_tasks > 0 and available_task_count < args.max_tasks:
        raise SystemExit(
            f"Requested {args.max_tasks} Harbor eval tasks, but only "
            f"{available_task_count} were found under {tasks_dir}"
        )

    task_names = [task_dir.name for task_dir in task_dirs]
    duplicate_names = sorted(name for name in set(task_names) if task_names.count(name) > 1)
    if duplicate_names:
        raise SystemExit(
            "Harbor eval task names must be unique; duplicate(s): " + ", ".join(duplicate_names)
        )

    dataset_manifest = tasks_dir / "dataset.toml"
    dataset_manifest_sha256 = (
        _sha256_file(dataset_manifest) if dataset_manifest.is_file() else None
    )

    rows: list[dict[str, Any]] = []
    for task_dir in task_dirs:
        tests_dir = task_dir / "tests"
        instruction_path = task_dir / "instruction.md"
        if not instruction_path.is_file() or not (tests_dir / "test.sh").is_file():
            raise SystemExit(
                f"Incomplete Harbor task {task_dir}: require instruction.md and tests/test.sh"
            )
        task_toml = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        agent = task_toml.get("agent", {})
        verifier = task_toml.get("verifier", {})
        environment = task_toml.get("environment", {})
        if (
            not isinstance(agent, dict)
            or not isinstance(verifier, dict)
            or not isinstance(environment, dict)
        ):
            raise SystemExit(f"Invalid Harbor task metadata sections in {task_dir / 'task.toml'}")
        _reject_unsupported_process_env(
            agent, section_name="agent", task_dir=task_dir
        )
        _reject_unsupported_process_env(
            verifier, section_name="verifier", task_dir=task_dir
        )

        source_agent_timeout = _positive(agent.get("timeout_sec"), 600.0)
        source_verifier_timeout = _positive(verifier.get("timeout_sec"), 120.0)
        agent_timeout = _capped(source_agent_timeout, 600.0, args.agent_timeout_cap)
        verifier_timeout = _capped(source_verifier_timeout, 120.0, args.verifier_timeout_cap)
        cpus = _resource_int(
            environment.get("cpus"),
            default=1,
            minimum=1,
            field="environment.cpus",
            task_dir=task_dir,
        )
        memory_mb = _resource_int(
            environment.get("memory_mb"),
            default=2048,
            minimum=1,
            field="environment.memory_mb",
            task_dir=task_dir,
        )
        storage_mb = _resource_int(
            environment.get("storage_mb"),
            default=0,
            minimum=0,
            field="environment.storage_mb",
            task_dir=task_dir,
        )
        allow_internet = environment.get("allow_internet", True)
        if not isinstance(allow_internet, bool):
            raise SystemExit(
                f"Invalid environment.allow_internet for Harbor task "
                f"{task_dir.name!r}: {allow_internet!r}"
            )
        image = _resolve_image(task_dir, image_dir, task_toml)
        docker_metadata = _docker_runtime_metadata(task_dir)
        runtime_env = _runtime_environment(
            docker_metadata=docker_metadata,
            task_environment=environment,
            image=image,
            task_dir=task_dir,
        )
        runtime_init_command = _runtime_init_command(docker_metadata)
        image_stat = image.stat()
        dockerfile = task_dir / "environment" / "Dockerfile"
        rows.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": instruction_path.read_text(encoding="utf-8").strip(),
                    }
                ],
                "label": "",
                "metadata": {
                    "task_name": f"{args.dataset_name}/{task_dir.name}",
                    "source_dataset": args.dataset_name,
                    "source_dataset_revision": args.dataset_revision,
                    "source_dataset_manifest_sha256": dataset_manifest_sha256,
                    "task_dir": str(task_dir.resolve()),
                    "tests_dir": str(tests_dir.resolve()),
                    "sif_path": str(image),
                    "source_docker_image": environment.get("docker_image"),
                    "image_size_bytes": image_stat.st_size,
                    "image_mtime_ns": image_stat.st_mtime_ns,
                    "instruction_sha256": _sha256_file(instruction_path),
                    "task_toml_sha256": _sha256_file(task_dir / "task.toml"),
                    "tests_tree_sha256": _tree_sha256(tests_dir),
                    "dockerfile_sha256": (
                        _sha256_file(dockerfile) if dockerfile.is_file() else None
                    ),
                    "runtime_semantics_version": _RUNTIME_SEMANTICS_VERSION,
                    "timeout_seconds": agent_timeout + verifier_timeout + args.timeout_overhead,
                    "agent_timeout": agent_timeout,
                    "verifier_timeout": verifier_timeout,
                    "source_agent_timeout": source_agent_timeout,
                    "source_verifier_timeout": source_verifier_timeout,
                    "agent_timeout_cap": args.agent_timeout_cap,
                    "verifier_timeout_cap": args.verifier_timeout_cap,
                    "timeout_overhead": args.timeout_overhead,
                    "agent_step_limit": args.agent_step_limit,
                    "cpus": cpus,
                    "memory_mb": memory_mb,
                    "storage_mb": storage_mb,
                    "allow_internet": allow_internet,
                    "workdir": docker_metadata.workdir,
                    "runtime_env": runtime_env,
                    "runtime_init_command": runtime_init_command,
                    "docker_entrypoint": _command_metadata(
                        docker_metadata.entrypoint
                    ),
                    "docker_cmd": _command_metadata(docker_metadata.cmd),
                    "docker_shell": list(docker_metadata.shell),
                },
            }
        )
    return rows


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read existing Harbor eval JSONL {path}: {exc}") from exc


def validate_existing_output(output: Path, expected: list[dict[str, Any]]) -> None:
    if _read_rows(output) != expected:
        raise SystemExit(
            f"Existing Harbor eval JSONL does not match the selected immutable task set: {output}"
        )


def main() -> int:
    args = parse_args()
    rows = _task_rows(args)
    output = Path(args.output).expanduser().resolve()
    if args.validate_existing:
        validate_existing_output(output, rows)
        print(f"Validated Harbor eval JSONL: {output}; rows={len(rows)}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote Harbor eval JSONL: {output}; rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
