"""Resolve and enumerate a locally-exported ``tmax/TMax-15K-Harbor`` dataset.

Harbor hub serves task *directories*, not prebuilt images. Pull them once with
the Harbor CLI (a TMax dependency — ``uv pip install harbor`` into this env, or
use the tmax checkout's environment):

    harbor download 'tmax/TMax-15K-Harbor@latest' --export --output-dir <dir>

Each task directory holds ``instruction.md``, ``task.toml``, ``environment/``
(a self-contained ``Dockerfile``) and ``tests/`` (``test.sh`` + ``test_final_state.py``
— the programmatic verifier). This module finds those directories under
``--dataset-dir`` and parses the per-task metadata that ``build_images.py`` and
``submit_tmax_tasks.py`` need. It does not import ``harbor``.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

HUB_DATASET = "tmax/TMax-15K-Harbor@latest"
HUB_EXPORT_DIRNAME = "TMax-15K-Harbor"
IMAGE_PREFIX = "polar-tmax15k"

# Coding-agent harnesses install at task time via the INIT prepare step (the
# Node CLIs globally, hermes and mini-swe-agent from PyPI). See HARNESS_INSTALL
# in submit_tmax_tasks.py.
SUPPORTED_HARNESSES = (
    "codex",
    "claude_code",
    "opencode",
    "qwen_code",
    "pi",
    "hermes",
    "mini_swe_agent",
)


@dataclass(frozen=True)
class TmaxTask:
    """One TMax/Harbor task directory, with the fields Polar needs."""

    name: str
    task_dir: Path
    instruction: str
    environment_dir: Path
    tests_dir: Path
    agent_timeout: float
    verifier_timeout: float
    cpus: int | None
    memory_mb: int | None
    allow_internet: bool
    workdir: str | None


def sanitize(name: str) -> str:
    normalized = name.strip().lower().replace("__", "--")
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.strip("-")


def env_image_for(name: str) -> str:
    """Tag for the image built straight from a task's ``environment/`` Dockerfile."""
    return f"{IMAGE_PREFIX}-env:{sanitize(name)}"


def runtime_image_for(name: str) -> str:
    """Tag for the runnable image (env image + Node layer) Polar launches."""
    return f"{IMAGE_PREFIX}-runtime:{sanitize(name)}"


def sif_filename_for(name: str) -> str:
    """Filename for the Apptainer ``.sif`` snapshot of a task's runtime image
    (docker-free Slurm flow). Matches the tag in :func:`runtime_image_for`."""
    return f"{sanitize(name)}.sif"


def _coerce(value: object, caster: type, default: object) -> object:
    try:
        return caster(value)  # type: ignore[call-arg]
    except (TypeError, ValueError):
        return default


def _load_task(task_dir: Path) -> TmaxTask:
    toml_path = task_dir / "task.toml"
    tests_dir = task_dir / "tests"
    environment_dir = task_dir / "environment"
    required = (toml_path, tests_dir / "test.sh", environment_dir / "Dockerfile")
    missing = [str(path.relative_to(task_dir)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            f"Invalid TMax task directory {task_dir}: missing required file(s): "
            f"{', '.join(missing)}"
        )

    meta = tomllib.loads(toml_path.read_text())
    agent = meta.get("agent", {})
    verifier = meta.get("verifier", {})
    env = meta.get("environment", {})
    instruction_path = task_dir / "instruction.md"
    instruction = (instruction_path.read_text() if instruction_path.is_file() else "").strip()

    return TmaxTask(
        name=task_dir.name,
        task_dir=task_dir,
        instruction=instruction,
        environment_dir=environment_dir,
        tests_dir=tests_dir,
        agent_timeout=_coerce(agent.get("timeout_sec"), float, 600.0),
        verifier_timeout=_coerce(verifier.get("timeout_sec"), float, 120.0),
        cpus=_coerce(env.get("cpus"), int, None) if env.get("cpus") is not None else None,
        memory_mb=(_coerce(env.get("memory_mb"), int, None) if env.get("memory_mb") is not None else None),
        allow_internet=bool(env.get("allow_internet", True)),
        workdir=str(env["workdir"]) if env.get("workdir") else None,
    )


def find_dataset_dir(dataset_dir: str | Path) -> Path:
    path = Path(dataset_dir).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(
            f"Dataset dir not found: {path}\n"
            f"Pull it first with the Harbor CLI:\n"
            f"  harbor download '{HUB_DATASET}' --export --output-dir {path}"
        )
    return path


def load_tasks(
    dataset_dir: str | Path,
    *,
    max_tasks: int = -1,
    names: list[str] | None = None,
) -> list[TmaxTask]:
    """Enumerate tasks under *dataset_dir* (any depth — robust to export nesting)."""
    root = find_dataset_dir(dataset_dir)
    # Harbor's canonical export has one known wrapper directory with every task
    # directly beneath it. Prefer that bounded scan: a recursive walk also
    # descends into all 15K environment/test trees and can generate gigabytes of
    # unnecessary Lustre metadata traffic. Keep rglob as the compatibility
    # fallback for non-standard exports and test fixtures.
    canonical_root = (
        root if root.name == HUB_EXPORT_DIRNAME else root / HUB_EXPORT_DIRNAME
    )
    canonical_paths = (
        list(canonical_root.glob("*/task.toml"))
        if canonical_root.is_dir()
        else []
    )
    task_paths = canonical_paths or list(root.rglob("task.toml"))
    task_dirs = sorted({path.parent for path in task_paths})
    tasks: list[TmaxTask] = []
    # Keep selection deterministic while avoiding a full metadata/content read
    # when callers request only a prefix. SIF array workers used to parse all
    # ~15K TOMLs and instructions independently before slicing to 1K tasks,
    # causing minutes of duplicated Lustre small-file traffic per worker.
    for task_dir in task_dirs:
        task = _load_task(task_dir)
        tasks.append(task)
        if not names and max_tasks > 0 and len(tasks) >= max_tasks:
            break
    if not tasks:
        raise SystemExit(
            f"No tasks found under {root}. Expected per-task dirs with "
            f"task.toml + tests/test.sh + environment/Dockerfile."
        )
    if names:
        wanted = set(names)
        selected = [t for t in tasks if t.name in wanted]
        missing = sorted(wanted - {t.name for t in selected})
        if missing:
            raise SystemExit(f"Unknown task(s): {', '.join(missing)}")
        return selected
    return tasks
