#!/usr/bin/env python3
"""Run the paired SPilot pool evaluation in a services-only allocation.

This launcher starts Polar rollout, one Polar gateway, a tokenizer-only Qwen
service, and the sandbox UDS bridge.  It never starts Ray, Slime, SGLang, model
weights, or a Router actor.  The benchmark runs in the same allocation and
process tree, so its loopback URLs and control-plane credential never need to
cross a cluster boundary.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import fcntl
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from types import FrameType
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

import yaml

from polar.config import TopologyConfig


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
CONTROL_TOKEN_ENV = "POLAR_CONTROL_PLANE_TOKEN"
NVIDIA_KEY_ENV = "POLAR_NVIDIA_API_KEY"
CONTROL_TOKEN_RE = re.compile(r"^[0-9A-Za-z_-]{32,128}$")
TEMPLATE_VARIABLE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
DEFAULT_POOL_BASE_URL = "https://integrate.api.nvidia.com/v1"
GATEWAY_NODE_ID = "localhost-node-01"
FORCED_EVAL_ROUTER_TIMEOUT_SECONDS = 180
FORCED_EVAL_RESERVE_EVALUATOR_SECONDS = 300
FORCED_EVAL_DEADLINE_MARGIN_SECONDS = 5
FORCED_EVAL_ADMISSION_WAIT_SECONDS = 300
DEFAULT_RUNNER_TOTAL_TIMEOUT_SECONDS = 3000
DEFAULT_FORCED_EVAL_CANDIDATE_COUNT = 2
DEFAULT_SLURM_MARGIN_SECONDS = 1800
OWNER_FD_ENV = "SPILOT_FORCED_EVAL_OWNER_FD"
REQUIRED_ISOLATION_VARIABLES = (
    "POLAR_APPTAINER_NO_INSTANCE",
    "POLAR_APPTAINER_NO_MOUNT_HOSTFS",
    "POLAR_APPTAINER_NO_MOUNT_TMP",
    "POLAR_APPTAINER_ISOLATE_PID",
    "POLAR_APPTAINER_ISOLATE_IPC",
)
SNAPSHOT_TREES = ("src", "examples/spilot_router_slime_grpo")
TOKENIZER_ASSETS = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
DEPENDENCY_DISTRIBUTIONS = (
    "fastapi",
    "httpx",
    "pydantic",
    "PyYAML",
    "transformers",
    "uvicorn",
)


class LauncherError(RuntimeError):
    """Expected launcher failure with a concise user-facing message."""


class GatewayAdmissionFatalError(LauncherError):
    """The gateway retained an unreaped paid episode and is poisoned."""


def required_isolation_environment() -> dict[str, str]:
    values = {name: os.environ.get(name, "1") for name in REQUIRED_ISOLATION_VARIABLES}
    invalid = {name: value for name, value in values.items() if value != "1"}
    if invalid:
        rendered = ", ".join(f"{name}={value!r}" for name, value in sorted(invalid.items()))
        raise LauncherError(f"forced evaluation requires isolation variables exact=1: {rendered}")
    return values


class AllocationOwnerLease:
    """Allocation-wide exclusive owner held across every evaluator pass."""

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / ".forced-eval.owner.lock"
        self._stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._stream.close()
            raise LauncherError(
                f"another allocation owns this forced-eval output: {self.path}"
            ) from exc
        self._stream.seek(0)
        self._stream.truncate()
        self._stream.write(
            f"pid={os.getpid()} slurm_job_id={os.environ.get('SLURM_JOB_ID', '')}\n"
        )
        self._stream.flush()
        os.fsync(self._stream.fileno())

    @property
    def fd(self) -> int:
        return self._stream.fileno()

    def close(self) -> None:
        if self._stream.closed:
            return
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()


@dataclass(frozen=True, slots=True)
class DatasetPreflight:
    needs_internet: bool
    max_dataset_task_timeout_seconds: int
    max_verifier_timeout_seconds: int
    task_timeout_floor_seconds: int
    outer_task_envelope_seconds: int


def required_slurm_walltime_seconds(
    *,
    max_tasks: int,
    replicates: int,
    candidate_count: int = DEFAULT_FORCED_EVAL_CANDIDATE_COUNT,
    max_concurrency: int,
    outer_task_envelope_seconds: int,
    max_paid_attempts_per_work: int,
    margin_seconds: int,
) -> tuple[int, int, int]:
    """Return (required seconds, work item count, worst-case waves)."""

    if candidate_count < 2:
        raise LauncherError("forced evaluation requires at least two candidates")
    work_item_count = max_tasks * candidate_count * replicates * max_paid_attempts_per_work
    waves = (work_item_count + max_concurrency - 1) // max_concurrency
    required = waves * outer_task_envelope_seconds + margin_seconds
    return required, work_item_count, waves


def required_resume_walltime_seconds(
    *,
    remaining_paid_attempt_count: int,
    max_concurrency: int,
    outer_task_envelope_seconds: int,
    shutdown_margin_seconds: int = 300,
) -> int:
    if remaining_paid_attempt_count <= 0:
        return 0
    waves = (remaining_paid_attempt_count + max_concurrency - 1) // max_concurrency
    return waves * outer_task_envelope_seconds + shutdown_margin_seconds


def remaining_allocation_seconds(
    *,
    allocated_seconds: int | None,
    launcher_started_monotonic: float,
) -> float | None:
    end_time = os.environ.get("SLURM_JOB_END_TIME", "").strip()
    if end_time.isdigit():
        return max(0.0, float(end_time) - time.time())
    if allocated_seconds is None:
        return None
    return max(0.0, allocated_seconds - (time.monotonic() - launcher_started_monotonic))


def partial_result_count(output_dir: Path) -> int:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise LauncherError("partial evaluator exit did not leave summary.json")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LauncherError(f"partial evaluator summary is invalid: {exc}") from exc
    collection = summary.get("collection") if isinstance(summary, dict) else None
    missing = collection.get("missing_result_count") if isinstance(collection, dict) else None
    if isinstance(missing, bool) or not isinstance(missing, int) or missing <= 0:
        raise LauncherError("partial evaluator summary has no positive missing_result_count")
    return missing


def remaining_paid_attempt_budget(
    output_dir: Path,
    *,
    missing_result_count: int,
    max_paid_attempts_per_work: int,
) -> int:
    ledger_path = output_dir / "attempt_ledger.json"
    if not ledger_path.is_file():
        return missing_result_count * max_paid_attempts_per_work
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LauncherError(f"attempt ledger is invalid: {exc}") from exc
    works = ledger.get("works") if isinstance(ledger, dict) else None
    maximum = ledger.get("max_paid_attempts_per_work") if isinstance(ledger, dict) else None
    if (
        not isinstance(ledger, dict)
        or ledger.get("schema_version") != 2
        or not isinstance(works, dict)
        or maximum != max_paid_attempts_per_work
    ):
        raise LauncherError("attempt ledger retry policy is inconsistent")
    remaining = 0
    incomplete = 0
    for work in works.values():
        if not isinstance(work, dict):
            raise LauncherError("attempt ledger work entry is invalid")
        if work.get("completed") is True:
            continue
        attempts = work.get("paid_attempts")
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 0
            or attempts > max_paid_attempts_per_work
        ):
            raise LauncherError("attempt ledger paid_attempts is invalid")
        incomplete += 1
        future_attempts = max(0, max_paid_attempts_per_work - attempts)
        attempt_rows = work.get("attempts")
        if not isinstance(attempt_rows, list):
            raise LauncherError("attempt ledger attempts are invalid")
        outstanding = sum(
            isinstance(attempt, dict)
            and attempt.get("state")
            in {"reserved_not_sent", "submit_started_ambiguous", "accepted_inflight"}
            for attempt in attempt_rows
        )
        if outstanding > 1:
            raise LauncherError("attempt ledger has multiple outstanding reservations")
        # Reconciliation/polling can consume a full outer envelope even when
        # the paid budget is already exhausted (e.g. ACK loss at cap=1).
        remaining += future_attempts + outstanding
    if incomplete != missing_result_count:
        raise LauncherError("attempt ledger incomplete count does not match summary")
    return remaining


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--service-dir", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the same immutable benchmark plan in a fresh service allocation",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, required=True)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument(
        "--rollout-port",
        type=int,
        help="Allocation-local rollout port; otherwise choose a free loopback port",
    )
    parser.add_argument(
        "--gateway-port",
        type=int,
        help="Allocation-local gateway port; otherwise choose a free loopback port",
    )
    parser.add_argument(
        "--tokenizer-port",
        type=int,
        help="Allocation-local tokenizer port; otherwise choose a free loopback port",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Local Qwen3.5 tokenizer assets; defaults below the data root",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--control-plane-retry-attempts", type=int, default=5)
    parser.add_argument("--control-plane-retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--max-paid-attempts-per-work", type=int, default=1)
    parser.add_argument("--allow-ambiguous-paid-retry", action="store_true")
    parser.add_argument("--evaluator-max-passes", type=int, default=3)
    parser.add_argument("--evaluator-resume-backoff-seconds", type=float, default=10.0)
    parser.add_argument(
        "--pool-timeout-seconds",
        type=int,
        default=1200,
        help="Hard timeout for the one forced candidate episode",
    )
    parser.add_argument(
        "--runner-total-timeout-seconds",
        type=int,
        help=(
            "Runner envelope; defaults to at least pool + Router + evaluator reserve "
            "+ deadline margin"
        ),
    )
    parser.add_argument("--forward-seed-to-pool", action="store_true")
    parser.add_argument("--include-qwen35-baseline", action="store_true")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="SPilot data root; defaults to POLAR_DATA_ROOT or ../../data from the repo",
    )
    parser.add_argument(
        "--pool-base-url",
        default=os.environ.get(
            "POLAR_MODEL_POOL_BASE_URL",
            os.environ.get("NVIDIA_BASE_URL", DEFAULT_POOL_BASE_URL),
        ),
    )
    parser.add_argument(
        "--agent-timeout-seconds",
        type=int,
        default=3300,
        help="Hard agent timeout in the rendered allocation config",
    )
    parser.add_argument(
        "--slurm-walltime-seconds",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--slurm-margin-seconds",
        type=int,
        default=DEFAULT_SLURM_MARGIN_SECONDS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render and validate fresh allocation files without starting services",
    )
    parser.add_argument(
        "--i-understand-eval-only",
        action="store_true",
        help="Required acknowledgement that this bypasses the trainable Router",
    )
    args = parser.parse_args(argv)
    if not args.i_understand_eval_only:
        parser.error("--i-understand-eval-only is required")
    for name in ("start_index", "seed"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in (
        "max_tasks",
        "replicates",
        "max_concurrency",
        "agent_timeout_seconds",
        "pool_timeout_seconds",
        "control_plane_retry_attempts",
        "max_paid_attempts_per_work",
        "evaluator_max_passes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.slurm_margin_seconds < 0:
        parser.error("--slurm-margin-seconds must be non-negative")
    if args.slurm_walltime_seconds is not None and args.slurm_walltime_seconds <= 0:
        parser.error("--slurm-walltime-seconds must be positive")
    if (
        args.poll_seconds <= 0
        or args.request_timeout <= 0
        or args.control_plane_retry_backoff_seconds <= 0
        or args.evaluator_resume_backoff_seconds <= 0
    ):
        parser.error("poll and request timeouts must be positive")
    for name in ("rollout_port", "gateway_port", "tokenizer_port"):
        value = getattr(args, name)
        if value is not None and not 1 <= value <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be between 1 and 65535")
    configured_ports = [
        value
        for value in (args.rollout_port, args.gateway_port, args.tokenizer_port)
        if value is not None
    ]
    if len(configured_ports) != len(set(configured_ports)):
        parser.error("rollout, gateway, and tokenizer ports must differ")
    minimum_runner_total = (
        args.pool_timeout_seconds
        + FORCED_EVAL_ROUTER_TIMEOUT_SECONDS
        + FORCED_EVAL_RESERVE_EVALUATOR_SECONDS
        + FORCED_EVAL_DEADLINE_MARGIN_SECONDS
        + FORCED_EVAL_ADMISSION_WAIT_SECONDS
    )
    if args.runner_total_timeout_seconds is None:
        args.runner_total_timeout_seconds = max(
            DEFAULT_RUNNER_TOTAL_TIMEOUT_SECONDS,
            minimum_runner_total,
        )
    elif args.runner_total_timeout_seconds <= 0:
        parser.error("--runner-total-timeout-seconds must be positive")
    if args.runner_total_timeout_seconds < minimum_runner_total:
        parser.error(
            "--runner-total-timeout-seconds must cover pool + Router + evaluator "
            f"reserve + deadline margin ({args.runner_total_timeout_seconds} < "
            f"{minimum_runner_total})"
        )
    if args.agent_timeout_seconds < args.runner_total_timeout_seconds:
        parser.error(
            "--agent-timeout-seconds must be at least the runner total timeout "
            f"({args.agent_timeout_seconds} < {args.runner_total_timeout_seconds})"
        )
    if not args.dry_run and not os.environ.get("SLURM_JOB_ID"):
        parser.error("run inside a Slurm allocation (or use --dry-run)")
    return args


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise LauncherError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LauncherError(f"identity file does not exist: {path}")
    return {"size": path.stat().st_size, "sha256": _sha256_file(path)}


def _tree_paths(root: Path) -> list[Path]:
    ignored_names = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}
    return [
        path
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if not any(part in ignored_names for part in path.relative_to(root).parts)
        and path.suffix not in {".pyc", ".pyo"}
    ]


def _tree_identity(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise LauncherError(f"identity tree does not exist: {root}")
    paths_before = _tree_paths(root)
    entries: list[dict[str, Any]] = []
    total_size = 0
    for path in paths_before:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            entries.append({"path": relative, "type": "symlink", "target": os.readlink(path)})
        elif path.is_dir():
            continue
        elif path.is_file():
            identity = _file_identity(path)
            total_size += int(identity["size"])
            entries.append({"path": relative, "type": "file", **identity})
        else:
            raise LauncherError(f"unsupported identity-tree entry: {path}")
    paths_after = _tree_paths(root)
    before_names = [path.relative_to(root).as_posix() for path in paths_before]
    after_names = [path.relative_to(root).as_posix() for path in paths_after]
    if before_names != after_names:
        raise LauncherError(f"content tree changed while hashing: {root}")
    return {
        "sha256": _canonical_sha256(entries),
        "entry_count": len(entries),
        "total_file_bytes": total_size,
    }


def source_snapshot_identity(root: Path) -> dict[str, Any]:
    trees = [{"path": relative, **_tree_identity(root / relative)} for relative in SNAPSHOT_TREES]
    return {"trees": trees, "sha256": _canonical_sha256(trees)}


def _snapshot_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}
    return {name for name in names if name in ignored or name.endswith((".pyc", ".pyo"))}


def create_source_snapshot(repo_root: Path, snapshot_root: Path) -> dict[str, Any]:
    """Copy and verify the exact source tree consumed by every child service."""

    before = source_snapshot_identity(repo_root)
    snapshot_root.mkdir(mode=0o700)
    for relative in SNAPSHOT_TREES:
        source = repo_root / relative
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, symlinks=True, ignore=_snapshot_ignore)
    after = source_snapshot_identity(repo_root)
    snapshot = source_snapshot_identity(snapshot_root)
    if before != after:
        raise LauncherError("live source changed while constructing the forced-eval snapshot")
    if before != snapshot:
        raise LauncherError("forced-eval source snapshot does not match its live source")
    for path in sorted(snapshot_root.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        path.chmod(0o555 if path.is_dir() else 0o444)
    snapshot_root.chmod(0o555)
    return snapshot


def _semantic_topology(topology: dict[str, Any]) -> dict[str, Any]:
    gateway = topology["gateway"]
    nodes = gateway["nodes"]
    semantic_nodes: list[dict[str, Any]] = []
    for node in nodes:
        semantic_nodes.append(
            {
                "id": node["id"],
                "model_served": node["model_served"],
                "worker_caps": {
                    "init": node["max_init_workers"],
                    "run": node["max_run_workers"],
                    "postrun": node["max_postrun_workers"],
                },
                "inference_engine": node["inference"]["engine"],
                "model_pool": [
                    {
                        "alias": candidate["alias"],
                        "model": candidate["model"],
                        "base_url": candidate["base_url"],
                        "api_key_env": candidate["api_key_env"],
                        "max_concurrency": candidate["max_concurrency"],
                        "max_active_episodes": candidate["max_active_episodes"],
                    }
                    for candidate in node["model_pool"]
                ],
            }
        )
    return {
        "rollout": {
            "http_max_connections": topology["rollout"]["http_max_connections"],
            "http_max_keepalive_connections": topology["rollout"][
                "http_max_keepalive_connections"
            ],
            "cleanup_max_concurrency": topology["rollout"]["cleanup_max_concurrency"],
        },
        "gateway": {
            "heartbeat_interval_seconds": gateway["heartbeat_interval_seconds"],
            "completion_persistence": gateway["completion_persistence"],
            "nodes": semantic_nodes,
        },
    }


def _dependency_identity() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in DEPENDENCY_DISTRIBUTIONS:
        try:
            result[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def _benchmark_asset_identity(
    data_path: Path,
    *,
    start_index: int,
    max_tasks: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    file_cache: dict[Path, dict[str, Any]] = {}
    tree_cache: dict[Path, dict[str, Any]] = {}
    with data_path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index < start_index:
                continue
            if len(rows) >= max_tasks:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LauncherError(f"invalid JSON at dataset row {index}: {exc}") from exc
            metadata = row.get("metadata") if isinstance(row, dict) else None
            if not isinstance(row, dict) or not isinstance(metadata, dict):
                raise LauncherError(f"dataset row {index} has no metadata object")
            sif_path = _absolute(Path(str(metadata.get("sif_path", ""))))
            tests_path = _absolute(Path(str(metadata.get("tests_dir", ""))))
            if sif_path not in file_cache:
                file_cache[sif_path] = _file_identity(sif_path)
            if tests_path not in tree_cache:
                tree_cache[tests_path] = _tree_identity(tests_path)
            rows.append(
                {
                    "dataset_index": index,
                    "dataset_row_sha256": _canonical_sha256(row),
                    "sif": file_cache[sif_path],
                    "verifier_tests": tree_cache[tests_path],
                }
            )
            verification_rows.append(
                {
                    "dataset_index": index,
                    "sif_path": str(sif_path),
                    "tests_path": str(tests_path),
                }
            )
    if len(rows) != max_tasks:
        raise LauncherError(
            f"requested {max_tasks} benchmark identities at {start_index}, found {len(rows)}"
        )
    semantic = {
        "dataset_file": _file_identity(data_path),
        "rows": rows,
        "sha256": _canonical_sha256(rows),
    }
    verification = {
        "dataset_path": str(data_path),
        "rows": verification_rows,
    }
    return semantic, verification


def build_semantic_identity(
    *,
    topology: dict[str, Any],
    topology_path: Path,
    polar_config_path: Path,
    snapshot_root: Path,
    snapshot_identity: dict[str, Any],
    tokenizer_path: Path,
    data_root: Path,
    data_path: Path,
    start_index: int,
    max_tasks: int,
) -> dict[str, Any]:
    mini_swe_root = data_root / "mini_swe_agent_runtime"
    agent_cli_root = data_root / "agent_cli" / "opt_node"
    train_container = _absolute(
        Path(
            os.environ.get(
                "POLR_TRAIN_SQSH",
                str(data_root / "container" / "flappydora-ubuntu22.04-cuda13.3.sqsh"),
            )
        )
    )
    python_executable = Path(sys.executable).resolve()
    tokenizer_files = {
        name: _file_identity(tokenizer_path / name) for name in TOKENIZER_ASSETS
    }
    runtime_trees = {
        "mini_swe_agent_runtime": _tree_identity(mini_swe_root),
        "agent_cli_opt_node": _tree_identity(agent_cli_root),
    }
    executable_files: dict[str, dict[str, Any]] = {
        "train_container": _file_identity(train_container),
        "python_executable": _file_identity(python_executable),
    }
    apptainer_path = Path(os.environ.get("POLAR_APPTAINER_BIN", "/usr/bin/apptainer"))
    if apptainer_path.is_file():
        executable_files["apptainer_executable"] = _file_identity(apptainer_path)
    benchmark_assets, benchmark_verification = _benchmark_asset_identity(
        data_path,
        start_index=start_index,
        max_tasks=max_tasks,
    )
    semantic = {
        "schema_version": 1,
        "topology": _semantic_topology(topology),
        "source_snapshot": snapshot_identity,
        "tokenizer": {"assets": tokenizer_files},
        "runtime_trees": runtime_trees,
        "runtime_files": executable_files,
        "python": {
            "implementation": sys.implementation.name,
            "version": list(sys.version_info[:3]),
        },
        "dependencies": _dependency_identity(),
        "runtime_isolation": required_isolation_environment(),
        "benchmark_assets": benchmark_assets,
    }
    verification = {
        "source_snapshot_root": str(snapshot_root),
        "tokenizer_path": str(tokenizer_path),
        "mini_swe_agent_runtime_path": str(mini_swe_root),
        "agent_cli_opt_node_path": str(agent_cli_root),
        "train_container_path": str(train_container),
        "python_executable_path": str(python_executable),
        "apptainer_executable_path": (
            str(apptainer_path) if "apptainer_executable" in executable_files else None
        ),
        "topology_path": str(topology_path),
        "topology_file": _file_identity(topology_path),
        "polar_config_path": str(polar_config_path),
        "polar_config_file": _file_identity(polar_config_path),
        "benchmark_assets": benchmark_verification,
    }
    return {
        "schema_version": 1,
        "semantic": semantic,
        "semantic_sha256": _canonical_sha256(semantic),
        "verification": verification,
    }


def verify_semantic_identity(document: dict[str, Any]) -> None:
    semantic = document.get("semantic")
    verification = document.get("verification")
    if not isinstance(semantic, dict) or not isinstance(verification, dict):
        raise LauncherError("semantic identity is missing semantic or verification data")
    if document.get("semantic_sha256") != _canonical_sha256(semantic):
        raise LauncherError("semantic identity digest does not match its document")
    snapshot_root = Path(str(verification["source_snapshot_root"]))
    if source_snapshot_identity(snapshot_root) != semantic.get("source_snapshot"):
        raise LauncherError("source snapshot content identity changed")
    tokenizer_path = Path(str(verification["tokenizer_path"]))
    tokenizer = semantic.get("tokenizer")
    expected_tokenizer = tokenizer.get("assets") if isinstance(tokenizer, dict) else None
    actual_tokenizer = {name: _file_identity(tokenizer_path / name) for name in TOKENIZER_ASSETS}
    if actual_tokenizer != expected_tokenizer:
        raise LauncherError("tokenizer content identity changed")
    expected_trees = semantic.get("runtime_trees")
    actual_trees = {
        "mini_swe_agent_runtime": _tree_identity(
            Path(str(verification["mini_swe_agent_runtime_path"]))
        ),
        "agent_cli_opt_node": _tree_identity(Path(str(verification["agent_cli_opt_node_path"]))),
    }
    if actual_trees != expected_trees:
        raise LauncherError("mini-SWE or agent-cli runtime identity changed")
    expected_files = semantic.get("runtime_files")
    if not isinstance(expected_files, dict):
        raise LauncherError("semantic runtime file identity is invalid")
    actual_files = {
        "train_container": _file_identity(Path(str(verification["train_container_path"]))),
        "python_executable": _file_identity(Path(str(verification["python_executable_path"]))),
    }
    apptainer = verification.get("apptainer_executable_path")
    if apptainer is not None:
        actual_files["apptainer_executable"] = _file_identity(Path(str(apptainer)))
    if actual_files != expected_files:
        raise LauncherError("container or executable identity changed")
    if _dependency_identity() != semantic.get("dependencies"):
        raise LauncherError("Python dependency identity changed")
    if required_isolation_environment() != semantic.get("runtime_isolation"):
        raise LauncherError("runtime isolation environment changed")
    expected_assets = semantic.get("benchmark_assets")
    asset_verification = verification.get("benchmark_assets")
    if not isinstance(expected_assets, dict) or not isinstance(asset_verification, dict):
        raise LauncherError("benchmark asset identity is invalid")
    dataset_path = Path(str(asset_verification["dataset_path"]))
    if _file_identity(dataset_path) != expected_assets.get("dataset_file"):
        raise LauncherError("benchmark dataset identity changed")
    expected_rows = expected_assets.get("rows")
    verification_rows = asset_verification.get("rows")
    if not isinstance(expected_rows, list) or not isinstance(verification_rows, list):
        raise LauncherError("benchmark row identity is invalid")
    if len(expected_rows) != len(verification_rows):
        raise LauncherError("benchmark verification row count changed")
    for expected, resource in zip(expected_rows, verification_rows, strict=True):
        if not isinstance(expected, dict) or not isinstance(resource, dict):
            raise LauncherError("benchmark verification row is invalid")
        if _file_identity(Path(str(resource["sif_path"]))) != expected.get("sif"):
            raise LauncherError("benchmark SIF identity changed")
        if _tree_identity(Path(str(resource["tests_path"]))) != expected.get(
            "verifier_tests"
        ):
            raise LauncherError("benchmark verifier identity changed")
    for name in ("topology", "polar_config"):
        path = Path(str(verification[f"{name}_path"]))
        if _file_identity(path) != verification.get(f"{name}_file"):
            raise LauncherError(f"rendered {name} content changed")


def resolve_data_root(value: Path | None) -> Path:
    if value is not None:
        return _absolute(value)
    configured = os.environ.get("POLAR_DATA_ROOT")
    if configured:
        return _absolute(Path(configured))
    candidate = REPO_ROOT.parent.parent / "data"
    if candidate.is_dir():
        return candidate.resolve()
    raise LauncherError("cannot infer data root; pass --data-root or set POLAR_DATA_ROOT")


def _positive_finite_metadata_number(
    metadata: dict[str, Any],
    field: str,
    *,
    dataset_index: int,
) -> float:
    value = metadata.get(field)
    if isinstance(value, bool):
        raise LauncherError(
            f"dataset row {dataset_index} metadata.{field} must be positive and finite"
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LauncherError(
            f"dataset row {dataset_index} metadata.{field} must be positive and finite"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise LauncherError(
            f"dataset row {dataset_index} metadata.{field} must be positive and finite"
        )
    return parsed


def preflight_eval_slice(
    path: Path,
    *,
    start_index: int,
    max_tasks: int,
    agent_timeout_seconds: int,
) -> DatasetPreflight:
    """Validate the paid slice and derive its actual timeout envelope."""

    if not path.is_file():
        raise LauncherError(f"evaluation data does not exist: {path}")
    selected = 0
    needs_internet = False
    dataset_timeouts: list[float] = []
    verifier_timeouts: list[float] = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index < start_index:
                continue
            if selected >= max_tasks:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LauncherError(f"invalid JSON at dataset row {index}: {exc}") from exc
            metadata = row.get("metadata") if isinstance(row, dict) else None
            if not isinstance(row, dict) or "prompt" not in row or not isinstance(metadata, dict):
                raise LauncherError(f"dataset row {index} needs prompt and metadata")
            for field in ("sif_path", "tests_dir", "workdir"):
                if not str(metadata.get(field, "")).strip():
                    raise LauncherError(f"dataset row {index} metadata.{field} is required")
            sif_path = _absolute(Path(str(metadata["sif_path"])))
            tests_dir = _absolute(Path(str(metadata["tests_dir"])))
            if not sif_path.is_file():
                raise LauncherError(f"dataset row {index} SIF does not exist: {sif_path}")
            if not tests_dir.is_dir():
                raise LauncherError(f"dataset row {index} tests dir does not exist: {tests_dir}")
            dataset_timeouts.append(
                _positive_finite_metadata_number(
                    metadata,
                    "timeout_seconds",
                    dataset_index=index,
                )
            )
            verifier_timeouts.append(
                _positive_finite_metadata_number(
                    metadata,
                    "verifier_timeout",
                    dataset_index=index,
                )
            )
            needs_internet = needs_internet or bool(metadata.get("allow_internet", True))
            selected += 1
    if selected != max_tasks:
        raise LauncherError(
            f"requested {max_tasks} tasks at start index {start_index}, found {selected}"
        )
    max_dataset_timeout = math.ceil(max(dataset_timeouts))
    max_verifier_timeout = math.ceil(max(verifier_timeouts))
    task_timeout_floor = agent_timeout_seconds + max_verifier_timeout
    return DatasetPreflight(
        needs_internet=needs_internet,
        max_dataset_task_timeout_seconds=max_dataset_timeout,
        max_verifier_timeout_seconds=max_verifier_timeout,
        task_timeout_floor_seconds=task_timeout_floor,
        outer_task_envelope_seconds=max(max_dataset_timeout, task_timeout_floor),
    )


def parse_proxy_target(proxy_url: str) -> str:
    parsed = urlsplit(proxy_url if "://" in proxy_url else f"http://{proxy_url}")
    if not parsed.hostname:
        raise LauncherError("HTTP proxy URL has no hostname")
    if parsed.username is not None or parsed.password is not None:
        raise LauncherError("credential-bearing HTTP proxy URLs are not supported")
    try:
        port = parsed.port
    except ValueError as exc:
        raise LauncherError(f"invalid HTTP proxy port: {exc}") from exc
    if port is None:
        port = 443 if parsed.scheme.lower() == "https" else 80
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"{host}:{port}"


def validate_pool_base_url(value: str) -> str:
    if not value or value != value.strip():
        raise LauncherError("model-pool base URL must be non-empty without whitespace")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LauncherError("model-pool base URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise LauncherError("credential-bearing model-pool base URLs are not supported")
    if parsed.query or parsed.fragment:
        raise LauncherError("model-pool base URL must not contain query credentials or fragments")
    return value.rstrip("/")


def reserve_loopback_ports(count: int = 2) -> list[int]:
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            sockets.append(listener)
        return [int(listener.getsockname()[1]) for listener in sockets]
    finally:
        for listener in sockets:
            listener.close()


def build_topology(
    *,
    rollout_port: int,
    gateway_port: int,
    tokenizer_port: int,
    service_dir: Path,
    pool_base_url: str,
    max_concurrency: int,
    include_qwen35_baseline: bool = False,
) -> dict[str, Any]:
    return {
        "rollout": {
            "host": "127.0.0.1",
            "port": rollout_port,
            "public_url": f"http://127.0.0.1:{rollout_port}",
            "save_dir": str(service_dir / "rollout_results"),
            "http_max_connections": max(32, max_concurrency * 4),
            "http_max_keepalive_connections": max(16, max_concurrency * 2),
            "cleanup_max_concurrency": max(16, max_concurrency * 2),
        },
        "gateway": {
            "heartbeat_interval_seconds": 10,
            "completion_persistence": {"enabled": False},
            "nodes": [
                {
                    "id": GATEWAY_NODE_ID,
                    "host": "127.0.0.1",
                    "port": gateway_port,
                    "public_url": f"http://127.0.0.1:{gateway_port}",
                    "max_init_workers": max_concurrency,
                    "max_run_workers": max_concurrency,
                    "max_postrun_workers": max_concurrency,
                    "model_served": "eval-only/forced-route-no-actor",
                    "inference": {
                        "engine": "sglang",
                        "base_url": f"http://127.0.0.1:{tokenizer_port}",
                    },
                    "model_pool": [
                        {
                            "alias": "pool/qwen3.6-27b",
                            "model": "nvidia/qwen/qwen3.6-27b",
                            "base_url": pool_base_url,
                            "api_key_env": NVIDIA_KEY_ENV,
                            "max_concurrency": max_concurrency,
                            "max_active_episodes": max_concurrency,
                        },
                        {
                            "alias": "pool/gpt-5.5",
                            "model": "openai/openai/gpt-5.5",
                            "base_url": pool_base_url,
                            "api_key_env": NVIDIA_KEY_ENV,
                            "max_concurrency": max_concurrency,
                            "max_active_episodes": max_concurrency,
                        },
                        *(
                            [
                                {
                                    "alias": "pool/qwen3.5-9b-baseline",
                                    "model": "nvidia/qwen/qwen3.5-9b",
                                    "base_url": pool_base_url,
                                    "api_key_env": NVIDIA_KEY_ENV,
                                    "max_concurrency": max_concurrency,
                                    "max_active_episodes": max_concurrency,
                                }
                            ]
                            if include_qwen35_baseline
                            else []
                        ),
                    ],
                }
            ],
        },
    }


def render_polar_config(
    *,
    data_root: Path,
    rollout_port: int,
    gateway_port: int,
    uds_root: Path,
    proxy_url: str,
    agent_timeout_seconds: int,
    pool_timeout_seconds: int,
    runner_total_timeout_seconds: int,
    task_timeout_floor_seconds: int,
    outer_task_envelope_seconds: int,
    template_path: Path | None = None,
    include_qwen35_baseline: bool = False,
) -> dict[str, Any]:
    gateway_uds_dir = uds_root / "gateway"
    proxy_uds_dir = uds_root / "proxy"
    runtime_dir = data_root / "mini_swe_agent_runtime"
    values = {
        "POLAR_ROLLOUT_URL": f"http://127.0.0.1:{rollout_port}",
        "POLAR_GATEWAY_URL": f"http://127.0.0.1:{gateway_port}",
        "POLAR_MAX_ASYNC_LEVEL": "1",
        "POLAR_FULLY_ASYNC": "false",
        "POLAR_REQUEST_TIMEOUT": str(outer_task_envelope_seconds + 600),
        "POLAR_TASK_TIMEOUT_FLOOR_SECONDS": str(task_timeout_floor_seconds),
        "TMAX_TRAIN_AGENT_TIMEOUT_SECONDS": str(agent_timeout_seconds),
        "POLAR_CALLBACK_HOST": "127.0.0.1",
        "POLAR_MIN_COMPLETE_ACCEPT_FRACTION": "0",
        "POLAR_EARLY_STOP_GRACE_SESSIONS": "0",
        # Forced-route evaluation never hands samples to an optimizer.  Keep
        # the training-only provider-health gate disabled in this renderer.
        "POLAR_CANDIDATE_POOL_HEALTH_GATE_ENABLED": "false",
        "POLAR_CANDIDATE_POOL_HEALTH_MIN_OBSERVED_SESSIONS": "16",
        "POLAR_CANDIDATE_POOL_HEALTH_MIN_COMPLETION_FRACTION": "0.1",
        "TMAX_TRAIN_PACK_LENGTH": "67584",
        "TMAX_ALLOW_SINGLE_SAMPLE_OVER_TOKEN_CAP": "1",
        "AGENT_CLI_DIR": str(data_root / "agent_cli" / "opt_node"),
        "APPTAINER_IMAGE_DIR": str(data_root / "tmax-15k-sif"),
        "POLAR_SANDBOX_NETWORK": "none",
        "POLAR_AGENT_PATH": (
            "/opt/polar-mini-swe-agent/bin:/opt/node/bin:/usr/local/sbin:/usr/local/bin:"
            "/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "POLAR_SANDBOX_GATEWAY_UDS": "/polar/gateway/gateway.sock",
        "POLAR_SANDBOX_HTTP_PROXY_UDS": ("/polar/proxy/proxy.sock" if proxy_url else ""),
        "POLAR_SANDBOX_HTTP_PROXY_PORT": "28100",
        "POLAR_APT_HTTP_SOURCE_POLICY": "https",
        "http_proxy": proxy_url,
        "https_proxy": proxy_url,
        "HTTP_PROXY": proxy_url,
        "HTTPS_PROXY": proxy_url,
        "no_proxy": "127.0.0.1,localhost",
        "NO_PROXY": "127.0.0.1,localhost",
        "POLAR_GATEWAY_UDS_DIR": str(gateway_uds_dir),
        "POLAR_AGENT_RUNTIME_VOLUME": (f"        - {runtime_dir}:/opt/polar-mini-swe-agent:ro"),
        "POLAR_INTERNET_RUNTIME_VOLUME": (
            f'    internet_volumes:\n      - "{proxy_uds_dir}:/polar/proxy:ro"'
            if proxy_url
            else ""
        ),
        "POLAR_AGENT_MODEL_NAME": "eval-only/forced-route-no-actor",
        "POLAR_AGENT_TEMPERATURE": "1.0",
        "POLAR_AGENT_TOP_P": "1.0",
        # Paid forced evaluation requires the same per-episode lease and
        # one-shot model-call capability as training. The topology caps each
        # candidate at max_concurrency; this positive budget only bounds
        # queueing under service jitter.
        "SPILOT_EPISODE_ADMISSION_ENABLED": "true",
        "SPILOT_EPISODE_ADMISSION_WAIT_BUDGET_SECONDS": str(
            FORCED_EVAL_ADMISSION_WAIT_SECONDS
        ),
        # Forced-route metrics report actual calls and cost but do not apply
        # the training experiment's optional cost-shaped reward.
        "SPILOT_QWEN_COST_WEIGHT": "1.0",
        "SPILOT_GPT_COST_WEIGHT": "1.0",
        "SPILOT_COST_PENALTY_LAMBDA": "0.0",
        "SPILOT_COST_NORMALIZER": "1.0",
    }
    template = (template_path or EXAMPLE_DIR / "polar_config.yaml").read_text(encoding="utf-8")
    missing = sorted(set(TEMPLATE_VARIABLE_RE.findall(template)) - values.keys())
    if missing:
        raise LauncherError(f"launcher has no values for template variables: {missing}")
    rendered = TEMPLATE_VARIABLE_RE.sub(lambda match: values[match.group(1)], template)
    unresolved = sorted(set(TEMPLATE_VARIABLE_RE.findall(rendered)))
    if unresolved:
        raise LauncherError(f"unresolved template variables: {unresolved}")
    document = yaml.safe_load(rendered)
    if not isinstance(document, dict):
        raise LauncherError("rendered Polar config is not a mapping")
    try:
        settings = document["polar_task_template"]["agent"]["settings"]
    except (KeyError, TypeError) as exc:
        raise LauncherError("rendered Polar config has no SPilot agent settings") from exc
    if not isinstance(settings, dict):
        raise LauncherError("rendered SPilot agent settings are not a mapping")
    model_pool = settings.get("model_pool")
    if not isinstance(model_pool, dict):
        raise LauncherError("rendered SPilot model_pool is not a mapping")
    if include_qwen35_baseline:
        if "M2" in model_pool:
            raise LauncherError("rendered SPilot model_pool already defines M2")
        model_pool["M2"] = {
            "model": "pool/qwen3.5-9b-baseline",
            "card": {
                "name": "Qwen3.5-9B baseline",
                "provider": "NVIDIA",
                "role": "direct-model baseline coding agent",
            },
            "cost_weight": 1.0,
            "model_kwargs": {
                "max_tokens": 16_384,
                "temperature": 1.0,
                "top_p": 1.0,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            },
        }
    settings["max_pool_calls"] = 1
    settings["pool_timeout_seconds"] = pool_timeout_seconds
    settings["total_timeout_seconds"] = runner_total_timeout_seconds
    minimum_total = (
        pool_timeout_seconds
        + int(settings.get("router_timeout_seconds", 0))
        + int(settings.get("reserve_evaluator_seconds", 0))
        + int(settings.get("deadline_margin_seconds", 0))
        + int(settings.get("pool_episode_admission_wait_budget_seconds", 0))
    )
    if runner_total_timeout_seconds < minimum_total:
        raise LauncherError(
            "runner total timeout does not cover pool + Router + evaluator reserve "
            f"+ deadline margin ({runner_total_timeout_seconds} < {minimum_total})"
        )
    if agent_timeout_seconds < runner_total_timeout_seconds:
        raise LauncherError(
            "outer agent timeout does not cover runner total timeout "
            f"({agent_timeout_seconds} < {runner_total_timeout_seconds})"
        )
    return document


def _write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _set_attempt_teardown(
    document: dict[str, Any],
    *,
    allocation_attempt_id: str | None,
    status: str,
    reason: str | None,
) -> str | None:
    attempts = document.get("allocation_attempts")
    if not isinstance(attempts, list):
        return None
    selected: dict[str, Any] | None = None
    if allocation_attempt_id is not None:
        selected = next(
            (
                item
                for item in attempts
                if isinstance(item, dict)
                and item.get("allocation_attempt_id") == allocation_attempt_id
            ),
            None,
        )
    else:
        selected = next(
            (
                item
                for item in reversed(attempts)
                if isinstance(item, dict)
                and isinstance(item.get("teardown_verification"), dict)
                and item["teardown_verification"].get("status") == "pending"
            ),
            None,
        )
    if selected is None:
        return None
    current = selected.get("teardown_verification")
    if not isinstance(current, dict):
        raise LauncherError("allocation attempt has invalid teardown state")
    if current.get("status") == "failed" and status == "verified":
        return str(selected.get("allocation_attempt_id"))
    selected["teardown_verification"] = {"status": status, "reason": reason}
    return str(selected.get("allocation_attempt_id"))


def invalidate_benchmark_output(
    output_dir: Path,
    reason: str,
    *,
    allocation_attempt_id: str | None = None,
) -> None:
    """Withhold formal metrics after a post-evaluator identity failure."""

    for name in ("manifest.json", "summary.json"):
        path = output_dir / name
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LauncherError(f"cannot invalidate malformed {name}: {exc}") from exc
        if not isinstance(document, dict):
            raise LauncherError(f"cannot invalidate non-object {name}")
        integrity = document.setdefault("content_integrity", {})
        if not isinstance(integrity, dict):
            integrity = {}
            document["content_integrity"] = integrity
        failures = integrity.setdefault("failures", [])
        if not isinstance(failures, list):
            failures = []
            integrity["failures"] = failures
        if reason not in failures:
            failures.append(reason)
        integrity["status"] = "failed"
        selected_attempt_id = _set_attempt_teardown(
            document,
            allocation_attempt_id=allocation_attempt_id,
            status="failed",
            reason=reason,
        )
        document["publication"] = {
            "status": "failed",
            "allocation_attempt_id": selected_attempt_id,
        }
        if name == "summary.json":
            document["final_metrics"] = None
            document["final_metrics_status"] = "withheld_integrity"
        _write_json(path, document)


def finalize_benchmark_output(
    output_dir: Path,
    *,
    allocation_attempt_id: str,
) -> None:
    """Atomically publish summary metrics only after parent-owned teardown."""

    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise LauncherError("evaluator did not leave manifest.json and summary.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LauncherError(f"cannot finalize malformed evaluator output: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise LauncherError("cannot finalize non-object evaluator output")
    if manifest.get("plan_sha256") != summary.get("plan_sha256"):
        raise LauncherError("manifest and summary plan identities differ")
    if manifest.get("collection") != summary.get("collection"):
        raise LauncherError("manifest and summary collection states differ")

    selected = _set_attempt_teardown(
        manifest,
        allocation_attempt_id=allocation_attempt_id,
        status="verified",
        reason=None,
    )
    if selected != allocation_attempt_id:
        raise LauncherError("current allocation attempt is absent from evaluator manifest")
    attempts = manifest.get("allocation_attempts")
    current = next(
        item
        for item in attempts
        if isinstance(item, dict) and item.get("allocation_attempt_id") == allocation_attempt_id
    )
    current_teardown = current.get("teardown_verification", {})
    if current_teardown.get("status") != "verified":
        raise LauncherError("failed allocation attempt cannot be teardown-verified")
    manifest["publication"] = {
        "status": "teardown_verified",
        "allocation_attempt_id": allocation_attempt_id,
    }
    # Persist proof first. A crash before the final summary replacement leaves
    # final_metrics pending/withheld and therefore cannot publish early.
    _write_json(manifest_path, manifest)

    summary["allocation_attempts"] = attempts
    summary["publication"] = dict(manifest["publication"])
    integrity = summary.get("content_integrity")
    collection = summary.get("collection")
    all_attempts_verified = all(
        isinstance(item, dict)
        and isinstance(item.get("teardown_verification"), dict)
        and item["teardown_verification"].get("status") == "verified"
        for item in attempts
    )
    if not isinstance(integrity, dict) or integrity.get("status") != "verified":
        summary["final_metrics"] = None
        summary["final_metrics_status"] = "withheld_integrity"
    elif not isinstance(collection, dict) or collection.get("result_set_complete") is not True:
        summary["final_metrics"] = None
        summary["final_metrics_status"] = "withheld_incomplete"
    elif collection.get("invalid_result_count") != 0:
        summary["final_metrics"] = None
        summary["final_metrics_status"] = "withheld_invalid_results"
    elif not all_attempts_verified:
        summary["final_metrics"] = None
        summary["final_metrics_status"] = "withheld_unverified_attempts"
    else:
        collected = summary.get("collected_only")
        if not isinstance(collected, dict):
            raise LauncherError("complete evaluator summary has no collected metrics")
        summary["final_metrics"] = collected
        summary["final_metrics_status"] = "published"
        summary["publication"] = {
            "status": "published",
            "allocation_attempt_id": allocation_attempt_id,
        }
    # This single atomic replacement is the formal publication boundary.
    _write_json(summary_path, summary)


def _http_json(url: str, *, timeout: float = 2.0) -> Any:
    opener = build_opener(ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return json.load(response)


def check_gateway_health(url: str) -> dict[str, Any]:
    document: Any
    try:
        document = _http_json(url)
    except HTTPError as exc:
        try:
            document = json.loads(exc.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise
        admission = (
            document.get("model_pool_episode_admission_health")
            if isinstance(document, dict)
            else None
        )
        if isinstance(admission, dict) and admission.get("fatal_retained") is True:
            raise GatewayAdmissionFatalError(
                "gateway retained an unreaped model-pool episode"
            ) from exc
        raise
    if not isinstance(document, dict):
        raise LauncherError("gateway health response is not a JSON object")
    admission = document.get("model_pool_episode_admission_health")
    if not isinstance(admission, dict):
        raise LauncherError("gateway health omitted episode-admission state")
    if admission.get("fatal_retained") is True or admission.get("healthy") is not True:
        raise GatewayAdmissionFatalError("gateway episode-admission state is unhealthy")
    if document.get("status") != "ok":
        raise LauncherError("gateway health status is not ok")
    return document


def wait_http(
    name: str,
    url: str,
    process: subprocess.Popen[bytes],
    *,
    predicate=lambda _: True,
    timeout: float = 60.0,
) -> Any:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise LauncherError(f"{name} exited before readiness (rc={return_code})")
        try:
            document = _http_json(url)
            if predicate(document):
                return document
            last_error = f"readiness predicate rejected {document!r}"
        except Exception as exc:  # service may not have bound its socket yet
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.2)
    raise LauncherError(f"timed out waiting for {name}: {last_error}")


def wait_uds(
    process: subprocess.Popen[bytes],
    *,
    ready_file: Path,
    sockets: list[Path],
    timeout: float = 20.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise LauncherError(f"UDS tunnel exited before readiness (rc={return_code})")
        if ready_file.is_file() and all(
            path.exists() and stat.S_ISSOCK(path.stat().st_mode) for path in sockets
        ):
            return
        time.sleep(0.1)
    raise LauncherError("timed out waiting for sandbox UDS tunnel")


def scoped_environment(
    *,
    source_root: Path = REPO_ROOT,
    control_token: str | None = None,
    nvidia_key: str | None = None,
    owner_fd: int | None = None,
):
    environment = dict(os.environ)
    environment.pop(CONTROL_TOKEN_ENV, None)
    environment.pop(NVIDIA_KEY_ENV, None)
    environment.pop("NVIDIA_API_KEY", None)
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(source_root / "src") + (
        f":{current_pythonpath}" if current_pythonpath else ""
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.update(required_isolation_environment())
    # Rollout, gateway, evaluator, and their heartbeats communicate only over
    # allocation-local loopback.  A cluster-wide HTTP proxy must never capture
    # those requests (including the tokenizer-only inference sentinel).
    environment["no_proxy"] = "127.0.0.1,localhost"
    environment["NO_PROXY"] = environment["no_proxy"]
    if control_token is not None:
        environment[CONTROL_TOKEN_ENV] = control_token
    if nvidia_key is not None:
        environment[NVIDIA_KEY_ENV] = nvidia_key
    if owner_fd is not None:
        environment[OWNER_FD_ENV] = str(owner_fd)
    return environment


def terminate_processes(
    processes: list[subprocess.Popen[bytes]],
    *,
    grace_seconds: float = 30.0,
) -> dict[int, dict[str, int | bool]]:
    outcomes: dict[int, dict[str, int | bool]] = {
        id(process): {"was_running": process.poll() is None, "killed": False}
        for process in processes
    }
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + grace_seconds
    for process in reversed(processes):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            outcomes[id(process)]["killed"] = True
            process.kill()
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                outcomes[id(process)]["killed"] = True
                process.kill()
                process.wait(timeout=5.0)
        outcomes[id(process)]["return_code"] = int(process.returncode or 0)
    return outcomes


def teardown_process_tree(
    processes: list[subprocess.Popen[bytes]],
    services: dict[str, subprocess.Popen[bytes]],
) -> dict[int, dict[str, int | bool]]:
    """Stop submitters, then give gateway containment its full shutdown bound."""

    outcomes: dict[int, dict[str, int | bool]] = {}
    service_ids = {id(process) for process in services.values()}
    evaluators = [process for process in processes if id(process) not in service_ids]
    outcomes.update(terminate_processes(evaluators, grace_seconds=15.0))
    gateway = services.get("gateway")
    if gateway is not None:
        # Uvicorn may spend 60s draining HTTP before Dispatcher/runtime use a
        # second ~60s containment-proof window. Keep a 30s parent margin so the
        # launcher observes the child's clean/non-clean exit instead of racing
        # it with SIGKILL at either internal deadline.
        outcomes.update(terminate_processes([gateway], grace_seconds=150.0))
    sidecars = [process for name, process in services.items() if name != "gateway"]
    outcomes.update(terminate_processes(sidecars, grace_seconds=30.0))
    return outcomes


def service_teardown_failure(
    services: dict[str, subprocess.Popen[bytes]],
    outcomes: dict[int, dict[str, int | bool]],
) -> str | None:
    failures: list[str] = []
    for name, process in services.items():
        outcome = outcomes.get(id(process))
        if outcome is None:
            failures.append(f"{name}: missing teardown outcome")
            continue
        return_code = int(outcome.get("return_code", process.returncode or 0))
        killed = outcome.get("killed") is True
        stopped_before_teardown = outcome.get("was_running") is False
        allowed_codes = {0, -signal.SIGTERM} if name == "tokenizer" else {0}
        if killed or stopped_before_teardown or return_code not in allowed_codes:
            failures.append(
                f"{name}: rc={return_code} killed={killed} "
                f"stopped_before_teardown={stopped_before_teardown}"
            )
    return "; ".join(failures) if failures else None


def run_evaluator_passes(
    *,
    evaluator_command: list[str],
    evaluator_environment: dict[str, str],
    processes: list[subprocess.Popen[bytes]],
    services: tuple[tuple[str, subprocess.Popen[bytes]], ...],
    output_dir: Path,
    max_passes: int,
    resume_backoff_seconds: float,
    max_concurrency: int,
    outer_task_envelope_seconds: int,
    max_paid_attempts_per_work: int,
    allocated_slurm_seconds: int | None,
    launcher_started_monotonic: float,
    owner_fd: int,
    gateway_health_url: str | None = None,
) -> int:
    current_command = list(evaluator_command)
    evaluator_environment = dict(evaluator_environment)
    evaluator_environment[OWNER_FD_ENV] = str(owner_fd)
    consecutive_health_failures = 0

    def poll_gateway_health() -> None:
        nonlocal consecutive_health_failures
        if gateway_health_url is None:
            return
        try:
            check_gateway_health(gateway_health_url)
        except GatewayAdmissionFatalError as exc:
            reason = f"gateway_admission_fatal: {exc}"
            invalidate_benchmark_output(output_dir, reason)
            raise
        except Exception as exc:
            consecutive_health_failures += 1
            if consecutive_health_failures >= 3:
                reason = (
                    "gateway_health_unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )
                invalidate_benchmark_output(output_dir, reason)
                raise LauncherError(reason) from exc
        else:
            consecutive_health_failures = 0

    for pass_index in range(1, max_passes + 1):
        evaluator = subprocess.Popen(
            current_command,
            env=evaluator_environment,
            pass_fds=(owner_fd,),
        )
        processes.append(evaluator)
        while evaluator.poll() is None:
            for name, process in services:
                if process.poll() is not None:
                    reason = (
                        f"service_exit_during_benchmark: {name} rc={process.returncode}"
                    )
                    invalidate_benchmark_output(output_dir, reason)
                    raise LauncherError(reason)
            poll_gateway_health()
            time.sleep(1.0)
        if gateway_health_url is not None:
            for _ in range(3):
                poll_gateway_health()
                if consecutive_health_failures == 0:
                    break
                time.sleep(0.2)
        return_code = int(evaluator.returncode or 0)
        if return_code != 3 or pass_index >= max_passes:
            return return_code
        missing = partial_result_count(output_dir)
        remaining_paid = remaining_paid_attempt_budget(
            output_dir,
            missing_result_count=missing,
            max_paid_attempts_per_work=max_paid_attempts_per_work,
        )
        required = required_resume_walltime_seconds(
            remaining_paid_attempt_count=remaining_paid,
            max_concurrency=max_concurrency,
            outer_task_envelope_seconds=outer_task_envelope_seconds,
        )
        remaining = remaining_allocation_seconds(
            allocated_seconds=allocated_slurm_seconds,
            launcher_started_monotonic=launcher_started_monotonic,
        )
        if remaining is None:
            print(
                "Not restarting partial evaluator: remaining Slurm walltime is unknown",
                file=sys.stderr,
                flush=True,
            )
            return 3
        if remaining < (required + resume_backoff_seconds):
            print(
                "Not restarting partial evaluator: allocation has "
                f"{remaining:.0f}s remaining but {required}s is required",
                file=sys.stderr,
                flush=True,
            )
            return 3
        print(
            f"Evaluator pass {pass_index} left {missing} results pending; "
            "resuming against the same live services",
            flush=True,
        )
        time.sleep(resume_backoff_seconds)
        if "--resume" not in current_command:
            current_command.append("--resume")
    return 3


def main(argv: list[str] | None = None) -> int:
    launcher_started_monotonic = time.monotonic()
    args = parse_args(argv)
    isolation_environment = required_isolation_environment()
    data = _absolute(args.data)
    output_dir = _absolute(args.output_dir)
    if args.service_dir is not None:
        service_dir = _absolute(args.service_dir)
    elif args.resume:
        attempt = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        service_dir = output_dir.with_name(f"{output_dir.name}.service.resume.{attempt}")
    else:
        service_dir = output_dir.with_name(f"{output_dir.name}.service")
    if args.resume:
        if not output_dir.is_dir() or not (output_dir / "manifest.json").is_file():
            raise LauncherError("--resume requires an existing output manifest")
    elif output_dir.exists():
        raise LauncherError(f"output directory already exists: {output_dir}")
    if service_dir.exists():
        raise LauncherError(f"service directory already exists: {service_dir}")
    if (
        output_dir == service_dir
        or service_dir in output_dir.parents
        or output_dir in service_dir.parents
    ):
        raise LauncherError("service directory must be separate from the benchmark output")

    data_root = resolve_data_root(args.data_root)
    tokenizer_path = _absolute(
        args.tokenizer_path
        if args.tokenizer_path is not None
        else data_root / "checkpoints" / "Qwen3.5-9B"
    )
    for path in (
        data_root / "agent_cli" / "opt_node",
        data_root / "mini_swe_agent_runtime",
        data_root / "tmax-15k-sif",
    ):
        if not path.is_dir():
            raise LauncherError(f"required runtime directory does not exist: {path}")
    for tokenizer_asset in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if not (tokenizer_path / tokenizer_asset).is_file():
            raise LauncherError(
                f"required tokenizer asset does not exist: {tokenizer_path / tokenizer_asset}"
            )
    dataset_preflight = preflight_eval_slice(
        data,
        start_index=args.start_index,
        max_tasks=args.max_tasks,
        agent_timeout_seconds=args.agent_timeout_seconds,
    )
    args.pool_base_url = validate_pool_base_url(args.pool_base_url)
    candidate_count = DEFAULT_FORCED_EVAL_CANDIDATE_COUNT + int(
        args.include_qwen35_baseline
    )
    (
        args.required_slurm_walltime_seconds,
        args.forced_eval_work_item_count,
        args.forced_eval_waves,
    ) = required_slurm_walltime_seconds(
        max_tasks=args.max_tasks,
        replicates=args.replicates,
        candidate_count=candidate_count,
        max_concurrency=args.max_concurrency,
        outer_task_envelope_seconds=dataset_preflight.outer_task_envelope_seconds,
        margin_seconds=args.slurm_margin_seconds,
        max_paid_attempts_per_work=args.max_paid_attempts_per_work,
    )
    if (
        args.slurm_walltime_seconds is not None
        and args.slurm_walltime_seconds < args.required_slurm_walltime_seconds
    ):
        raise LauncherError(
            "Slurm walltime cannot cover the selected dataset's worst-case waves "
            f"plus margin ({args.slurm_walltime_seconds} < "
            f"{args.required_slurm_walltime_seconds} seconds)"
        )
    proxy_url = (
        os.environ.get("http_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTPS_PROXY")
        or ""
    ).strip()
    if dataset_preflight.needs_internet and not proxy_url:
        raise LauncherError("selected tasks allow internet but no HTTP proxy is configured")
    proxy_target = parse_proxy_target(proxy_url) if proxy_url else ""

    service_dir.parent.mkdir(parents=True, exist_ok=True)
    service_dir.mkdir(mode=0o700)
    snapshot_root = service_dir / "source_snapshot"
    snapshot_identity = create_source_snapshot(REPO_ROOT, snapshot_root)
    snapshot_example_dir = snapshot_root / "examples" / "spilot_router_slime_grpo"
    if args.rollout_port is None and args.gateway_port is None and args.tokenizer_port is None:
        if args.dry_run:
            rollout_port, gateway_port, tokenizer_port = 18080, 18100, 18200
        else:
            rollout_port, gateway_port, tokenizer_port = reserve_loopback_ports(3)
    else:
        requested = [args.rollout_port, args.gateway_port, args.tokenizer_port]
        explicit_ports = {value for value in requested if value is not None}
        chosen: list[int] = []
        for value in requested:
            if value is None:
                candidate = reserve_loopback_ports(1)[0]
                while candidate in chosen or candidate in explicit_ports:
                    candidate = reserve_loopback_ports(1)[0]
                chosen.append(candidate)
            else:
                chosen.append(value)
        rollout_port, gateway_port, tokenizer_port = chosen
    if len({rollout_port, gateway_port, tokenizer_port}) != 3:
        raise LauncherError("rollout, gateway, and tokenizer ports must differ")
    uds_root = Path(
        f"/tmp/polar-forced-eval-{os.environ.get('SLURM_JOB_ID', 'dry')}-{os.getpid()}"
    )
    gateway_uds = uds_root / "gateway" / "gateway.sock"
    proxy_uds = uds_root / "proxy" / "proxy.sock"
    ready_file = uds_root / "ready"
    uds_root.mkdir(mode=0o700, exist_ok=True)
    uds_root.chmod(0o700)
    for directory in (gateway_uds.parent, proxy_uds.parent):
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)

    topology = build_topology(
        rollout_port=rollout_port,
        gateway_port=gateway_port,
        tokenizer_port=tokenizer_port,
        service_dir=service_dir,
        pool_base_url=args.pool_base_url,
        max_concurrency=args.max_concurrency,
        include_qwen35_baseline=args.include_qwen35_baseline,
    )
    TopologyConfig.model_validate(topology)
    polar_config = render_polar_config(
        data_root=data_root,
        rollout_port=rollout_port,
        gateway_port=gateway_port,
        uds_root=uds_root,
        proxy_url=proxy_url,
        agent_timeout_seconds=args.agent_timeout_seconds,
        pool_timeout_seconds=args.pool_timeout_seconds,
        runner_total_timeout_seconds=args.runner_total_timeout_seconds,
        task_timeout_floor_seconds=dataset_preflight.task_timeout_floor_seconds,
        outer_task_envelope_seconds=dataset_preflight.outer_task_envelope_seconds,
        template_path=snapshot_example_dir / "polar_config.yaml",
        include_qwen35_baseline=args.include_qwen35_baseline,
    )
    topology_path = service_dir / "topology.yaml"
    polar_config_path = service_dir / "polar_config.yaml"
    _write_yaml(topology_path, topology)
    _write_yaml(polar_config_path, polar_config)
    semantic_identity_path = service_dir / "semantic_identity.json"
    semantic_identity = build_semantic_identity(
        topology=topology,
        topology_path=topology_path,
        polar_config_path=polar_config_path,
        snapshot_root=snapshot_root,
        snapshot_identity=snapshot_identity,
        tokenizer_path=tokenizer_path,
        data_root=data_root,
        data_path=data,
        start_index=args.start_index,
        max_tasks=args.max_tasks,
    )
    verify_semantic_identity(semantic_identity)
    _write_json(semantic_identity_path, semantic_identity)
    semantic_identity_file = _file_identity(semantic_identity_path)
    allocation_attempt_id = _canonical_sha256(
        {
            "run_id": args.run_id,
            "service_dir": str(service_dir),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": socket.gethostname(),
            "semantic_identity_sha256": semantic_identity["semantic_sha256"],
        }
    )
    evaluator_command = [
        sys.executable,
        str(snapshot_example_dir / "forced_route_eval.py"),
        "--i-understand-eval-only",
        "--data",
        str(data),
        "--polar-config",
        str(polar_config_path),
        "--semantic-identity",
        str(semantic_identity_path),
        "--rollout-url",
        f"http://127.0.0.1:{rollout_port}",
        "--output-dir",
        str(output_dir),
        "--run-id",
        args.run_id,
        "--allocation-attempt-id",
        allocation_attempt_id,
        "--start-index",
        str(args.start_index),
        "--max-tasks",
        str(args.max_tasks),
        "--replicates",
        str(args.replicates),
        "--seed",
        str(args.seed),
        "--max-concurrency",
        str(args.max_concurrency),
        "--poll-seconds",
        str(args.poll_seconds),
        "--request-timeout",
        str(args.request_timeout),
        "--control-plane-retry-attempts",
        str(args.control_plane_retry_attempts),
        "--control-plane-retry-backoff-seconds",
        str(args.control_plane_retry_backoff_seconds),
        "--max-paid-attempts-per-work",
        str(args.max_paid_attempts_per_work),
        "--pool-timeout-seconds",
        str(args.pool_timeout_seconds),
        "--runner-total-timeout-seconds",
        str(args.runner_total_timeout_seconds),
    ]
    if args.forward_seed_to_pool:
        evaluator_command.append("--forward-seed-to-pool")
    if args.allow_ambiguous_paid_retry:
        evaluator_command.append("--allow-ambiguous-paid-retry")
    if args.include_qwen35_baseline:
        evaluator_command.append("--include-qwen35-baseline")
    if args.resume:
        evaluator_command.append("--resume")
    _write_json(
        service_dir / "launcher.json",
        {
            "schema_version": 1,
            "eval_only": True,
            "actor_training": False,
            "actor_invoked": False,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": socket.gethostname(),
            "rollout_url": f"http://127.0.0.1:{rollout_port}",
            "gateway_url": f"http://127.0.0.1:{gateway_port}",
            "tokenizer_url": f"http://127.0.0.1:{tokenizer_port}",
            "tokenizer_path": str(tokenizer_path),
            "source_snapshot_path": str(snapshot_root),
            "source_snapshot_sha256": snapshot_identity["sha256"],
            "semantic_identity_path": str(semantic_identity_path),
            "semantic_identity_sha256": semantic_identity["semantic_sha256"],
            "allocation_attempt_id": allocation_attempt_id,
            "topology_path": str(topology_path),
            "polar_config_path": str(polar_config_path),
            "output_dir": str(output_dir),
            "resume": args.resume,
            "control_token_persisted": False,
            "model_credential_persisted": False,
            "runtime_isolation": isolation_environment,
            "pool_timeout_seconds": args.pool_timeout_seconds,
            "runner_total_timeout_seconds": args.runner_total_timeout_seconds,
            "router_timeout_seconds": FORCED_EVAL_ROUTER_TIMEOUT_SECONDS,
            "reserve_evaluator_seconds": FORCED_EVAL_RESERVE_EVALUATOR_SECONDS,
            "deadline_margin_seconds": FORCED_EVAL_DEADLINE_MARGIN_SECONDS,
            "outer_agent_timeout_seconds": args.agent_timeout_seconds,
            "control_plane_retry_attempts": args.control_plane_retry_attempts,
            "control_plane_retry_backoff_seconds": (args.control_plane_retry_backoff_seconds),
            "max_paid_attempts_per_work": args.max_paid_attempts_per_work,
            "allow_ambiguous_paid_retry": args.allow_ambiguous_paid_retry,
            "include_qwen35_baseline": args.include_qwen35_baseline,
            "forced_eval_candidate_count": candidate_count,
            "evaluator_max_passes": args.evaluator_max_passes,
            "evaluator_resume_backoff_seconds": args.evaluator_resume_backoff_seconds,
            "max_dataset_task_timeout_seconds": (
                dataset_preflight.max_dataset_task_timeout_seconds
            ),
            "max_verifier_timeout_seconds": (dataset_preflight.max_verifier_timeout_seconds),
            "task_timeout_floor_seconds": dataset_preflight.task_timeout_floor_seconds,
            "outer_task_envelope_seconds": (dataset_preflight.outer_task_envelope_seconds),
            "forced_eval_work_item_count": (
                args.max_tasks * candidate_count * args.replicates
            ),
            "max_paid_attempt_count": args.forced_eval_work_item_count,
            "forced_eval_waves": args.forced_eval_waves,
            "slurm_margin_seconds": args.slurm_margin_seconds,
            "required_slurm_walltime_seconds": args.required_slurm_walltime_seconds,
            "allocated_slurm_walltime_seconds": args.slurm_walltime_seconds,
            "evaluator_command": evaluator_command,
        },
    )
    if args.dry_run:
        for directory in (gateway_uds.parent, proxy_uds.parent, uds_root):
            directory.rmdir()
        print(f"Rendered services-only plan: {service_dir}")
        return 0

    nvidia_key = (
        os.environ.get(NVIDIA_KEY_ENV, "").strip() or os.environ.get("NVIDIA_API_KEY", "").strip()
    )
    if not nvidia_key:
        raise LauncherError(f"{NVIDIA_KEY_ENV} or NVIDIA_API_KEY is required")
    control_token = os.environ.get(CONTROL_TOKEN_ENV, "").strip()
    if not control_token:
        control_token = secrets.token_urlsafe(48)
    if not CONTROL_TOKEN_RE.fullmatch(control_token):
        raise LauncherError(f"{CONTROL_TOKEN_ENV} must be a 32-128 character opaque token")

    owner_lease = AllocationOwnerLease(output_dir)
    processes: list[subprocess.Popen[bytes]] = []
    service_processes: dict[str, subprocess.Popen[bytes]] = {}
    previous_handlers: dict[int, Any] = {}
    evaluator_invoked = False
    evaluator_return_code: int | None = None

    def interrupted(signum: int, _frame: FrameType | None) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, interrupted)

    try:
        with ExitStack() as stack:
            tokenizer_log = stack.enter_context((service_dir / "tokenizer.log").open("wb"))
            rollout_log = stack.enter_context((service_dir / "rollout.log").open("wb"))
            gateway_log = stack.enter_context((service_dir / "gateway.log").open("wb"))
            tunnel_log = stack.enter_context((service_dir / "uds_tunnel.log").open("wb"))
            tokenizer = subprocess.Popen(
                [
                    sys.executable,
                    str(snapshot_example_dir / "serve_tokenizer.py"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(tokenizer_port),
                    "--tokenizer-path",
                    str(tokenizer_path),
                ],
                stdout=tokenizer_log,
                stderr=subprocess.STDOUT,
                env=scoped_environment(
                    source_root=snapshot_root,
                    owner_fd=owner_lease.fd,
                ),
                pass_fds=(owner_lease.fd,),
            )
            processes.append(tokenizer)
            service_processes["tokenizer"] = tokenizer
            wait_http(
                "tokenizer-only service",
                f"http://127.0.0.1:{tokenizer_port}/health",
                tokenizer,
                timeout=180.0,
            )

            rollout = subprocess.Popen(
                [sys.executable, "-m", "polar.cli", "serve_rollout", "-c", str(topology_path)],
                stdout=rollout_log,
                stderr=subprocess.STDOUT,
                env=scoped_environment(
                    source_root=snapshot_root,
                    control_token=control_token,
                    owner_fd=owner_lease.fd,
                ),
                pass_fds=(owner_lease.fd,),
            )
            processes.append(rollout)
            service_processes["rollout"] = rollout
            wait_http("Polar rollout", f"http://127.0.0.1:{rollout_port}/health", rollout)

            gateway = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "polar.cli",
                    "serve_gateway",
                    "-c",
                    str(topology_path),
                    "--node-id",
                    GATEWAY_NODE_ID,
                ],
                stdout=gateway_log,
                stderr=subprocess.STDOUT,
                env=scoped_environment(
                    source_root=snapshot_root,
                    control_token=control_token,
                    nvidia_key=nvidia_key,
                    owner_fd=owner_lease.fd,
                ),
                pass_fds=(owner_lease.fd,),
            )
            processes.append(gateway)
            service_processes["gateway"] = gateway
            wait_http("Polar gateway", f"http://127.0.0.1:{gateway_port}/health", gateway)
            wait_http(
                "gateway registration",
                f"http://127.0.0.1:{rollout_port}/nodes",
                rollout,
                predicate=lambda document: (
                    isinstance(document, list)
                    and len(document) == 1
                    and isinstance(document[0], dict)
                    and document[0].get("node_id") == GATEWAY_NODE_ID
                    and document[0].get("healthy") is True
                    and document[0].get("gateway_url") == f"http://127.0.0.1:{gateway_port}"
                ),
            )

            mappings = [f"{gateway_uds}=127.0.0.1:{gateway_port}"]
            expected_sockets = [gateway_uds]
            if proxy_url:
                mappings.append(f"{proxy_uds}={proxy_target}")
                expected_sockets.append(proxy_uds)
            tunnel = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "polar.runtime.uds_tunnel",
                    "--ready-file",
                    str(ready_file),
                    "--socket-mode",
                    "0600",
                    *mappings,
                ],
                stdout=tunnel_log,
                stderr=subprocess.STDOUT,
                env=scoped_environment(
                    source_root=snapshot_root,
                    owner_fd=owner_lease.fd,
                ),
                pass_fds=(owner_lease.fd,),
            )
            processes.append(tunnel)
            service_processes["UDS"] = tunnel
            wait_uds(tunnel, ready_file=ready_file, sockets=expected_sockets)

            print(
                "Services ready on allocation-local loopback; starting paired forced-route eval",
                flush=True,
            )
            evaluator_invoked = True
            evaluator_return_code = run_evaluator_passes(
                evaluator_command=evaluator_command,
                evaluator_environment=scoped_environment(
                    source_root=snapshot_root,
                    control_token=control_token,
                ),
                processes=processes,
                services=(
                    ("tokenizer", tokenizer),
                    ("rollout", rollout),
                    ("gateway", gateway),
                    ("UDS", tunnel),
                ),
                output_dir=output_dir,
                max_passes=args.evaluator_max_passes,
                resume_backoff_seconds=args.evaluator_resume_backoff_seconds,
                max_concurrency=args.max_concurrency,
                outer_task_envelope_seconds=(dataset_preflight.outer_task_envelope_seconds),
                max_paid_attempts_per_work=args.max_paid_attempts_per_work,
                allocated_slurm_seconds=args.slurm_walltime_seconds,
                launcher_started_monotonic=launcher_started_monotonic,
                owner_fd=owner_lease.fd,
                gateway_health_url=f"http://127.0.0.1:{gateway_port}/health",
            )
    finally:
        teardown_outcomes = teardown_process_tree(processes, service_processes)
        integrity_failure: str | None = None
        try:
            teardown_failure = service_teardown_failure(
                service_processes,
                teardown_outcomes,
            )
            if teardown_failure is not None:
                raise LauncherError(f"service teardown failed: {teardown_failure}")
            if _file_identity(semantic_identity_path) != semantic_identity_file:
                raise LauncherError("semantic identity manifest changed during evaluation")
            verify_semantic_identity(semantic_identity)
        except (LauncherError, OSError, ValueError) as exc:
            integrity_failure = f"post_teardown_identity_verification: {exc}"
            try:
                invalidate_benchmark_output(output_dir, integrity_failure)
            except (LauncherError, OSError, ValueError) as invalidate_exc:
                integrity_failure = f"{integrity_failure}; invalidation failed: {invalidate_exc}"
        if (
            integrity_failure is None
            and evaluator_invoked
            and (output_dir / "manifest.json").is_file()
            and (output_dir / "summary.json").is_file()
        ):
            try:
                finalize_benchmark_output(
                    output_dir,
                    allocation_attempt_id=allocation_attempt_id,
                )
            except (LauncherError, OSError, ValueError) as exc:
                integrity_failure = f"post_teardown_publication: {exc}"
                try:
                    invalidate_benchmark_output(
                        output_dir,
                        integrity_failure,
                        allocation_attempt_id=allocation_attempt_id,
                    )
                except (LauncherError, OSError, ValueError) as invalidate_exc:
                    integrity_failure = (
                        f"{integrity_failure}; invalidation failed: {invalidate_exc}"
                    )
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        try:
            for path in (gateway_uds, proxy_uds, ready_file):
                path.unlink(missing_ok=True)
            for directory in (gateway_uds.parent, proxy_uds.parent, uds_root):
                directory.rmdir()
        except OSError:
            pass
        owner_lease.close()
        if integrity_failure is not None:
            raise LauncherError(integrity_failure)
    if evaluator_return_code is None:
        raise LauncherError("evaluator did not return a status")
    return evaluator_return_code


if __name__ == "__main__":
    try:
        os.umask(0o077)
        raise SystemExit(main())
    except KeyboardInterrupt as exc:
        print(f"services-only eval interrupted: {exc}", file=sys.stderr)
        raise SystemExit(130) from exc
    except (LauncherError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"services-only eval error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
