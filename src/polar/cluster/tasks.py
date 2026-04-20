"""Build and submit Polar task payloads on SLURM.

Called inside the SLURM job to construct task JSON with correct absolute SIF
image paths and submit them through the Polar CLI.

Usage (from sbatch script)::

    python -m polar.cluster.tasks --example calculator --harness opencode \\
        --topology /path/to/topology.yaml --sif-dir /lustre/.../sif_images
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True))


def _submit_task(request_path: Path, topology_path: str) -> dict[str, Any]:
    """Submit a task JSON via ``polar submit`` and return the response."""
    command = [
        sys.executable, "-m", "polar.cli",
        "submit", str(request_path),
        "-c", topology_path,
        "--json",
    ]
    print(f"[tasks] Running: {' '.join(command)}")
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _summarize_result(response: dict[str, Any]) -> dict[str, Any]:
    sessions = response.get("results") or []
    completed = sum(1 for s in sessions if s.get("status") == "COMPLETED")
    reward_one = 0
    for s in sessions:
        traces = (s.get("trajectory") or {}).get("traces") or []
        if traces and traces[-1].get("reward") == 1.0:
            reward_one += 1
    return {
        "total_sessions": len(sessions),
        "completed_sessions": completed,
        "reward_one_sessions": reward_one,
    }


def _sanitize_instance_id(instance_id: str) -> str:
    normalized = instance_id.strip().lower()
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", normalized.replace("__", "--"))
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.strip("-")


# ── Calculator ────────────────────────────────────────────────────────────────

CALCULATOR_INSTRUCTION = """\
Write a Python calculator with no extra imports. Support arithmetic expressions over integers and
parentheses. Save it as `calculator.py`.

Expose a `Calculator` class that can be called with a string expression.

Example:

from calculator import Calculator
cal = Calculator()
print(cal("4*3-3"))  # should print 9"""

CALCULATOR_TEST = """\
from calculator import Calculator

cal = Calculator()

assert cal("4*3-3") == 9
assert cal("(2+3)*4") == 20
assert cal("10/2+7") == 12
assert cal("18-(3*4)") == 6
assert cal(" 8 + 2 * 5 ") == 18

print("calculator tests passed")
"""


def build_calculator_task(
    harness: str,
    sif_dir: str,
    output_dir: str,
    *,
    agent_model: str = "openai/gpt-4o",
    num_rollouts: int = 4,
    timeout_seconds: float = 900.0,
    batch_id: str = "",
) -> dict[str, Any]:
    """Build a calculator task payload."""
    sif_path = os.path.join(sif_dir, f"calculator-{harness}.sif")
    if not os.path.isfile(sif_path):
        raise FileNotFoundError(
            f"Calculator SIF not found: {sif_path}\n"
            f"Build it with: polar cluster build-sif --example calculator --harness {harness}"
        )

    test_dir = Path(output_dir) / "assets"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "test_calculator.py"
    test_file.write_text(CALCULATOR_TEST)

    return {
        "task_id": f"calculator-{harness}-slurm-{batch_id}",
        "instruction": CALCULATOR_INSTRUCTION,
        "num_rollouts": num_rollouts,
        "timeout_seconds": timeout_seconds,
        "runtime": {
            "backend": "apptainer",
            "image": sif_path,
            "prepare": [
                {
                    "type": "exec",
                    "command": (
                        "mkdir -p /polar/session/workspace /polar/session/logs/agent && "
                        "cd /polar/session/workspace && git init && "
                        "git config user.email 'polar@test' && "
                        "git config user.name 'Polar'"
                    ),
                },
                {
                    "type": "upload_file",
                    "source": str(test_file.resolve()),
                    "target": "/polar/session/workspace/test_calculator.py",
                },
                {
                    "type": "exec",
                    "command": "cd /polar/session/workspace && git add -A && git commit -m 'initial'",
                },
            ],
            "env": {},
            "network": "host",
            "workdir": "/polar/session/workspace",
            # swe_agent's swerex does chown inside the container; Apptainer
            # needs --fakeroot to support ownership changes on overlayFS.
            **({"kwargs": {"fakeroot": True}} if harness == "swe_agent" else {}),
        },
        "agent": {
            "harness": harness,
            "model_name": agent_model,
            "settings": {},
            "env": {},
        },
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {
            "strategy": "swegym_git_diff",
            "config": {
                "repo_dir": "/polar/session/workspace",
                "patch_command": (
                    "cd /polar/session/workspace && git add -A && git diff --cached --binary"
                ),
                "test_command": (
                    "cd /polar/session/workspace && python3 test_calculator.py && "
                    "echo 'PASSED test_calculator'"
                ),
                "test_timeout": 60.0,
                "expected_output_json": {"test_calculator": "PASSED"},
            },
            "refresh_runtime": False,
        },
    }


# ── SWE-Gym ──────────────────────────────────────────────────────────────────

SWEGYM_SAMPLE = [
    {"instance_id": "getmoto__moto-7365", "repo": "getmoto/moto"},
    {"instance_id": "python__mypy-10392", "repo": "python/mypy"},
    {"instance_id": "conan-io__conan-13721", "repo": "conan-io/conan"},
    {"instance_id": "iterative__dvc-1809", "repo": "iterative/dvc"},
    {"instance_id": "dask__dask-10441", "repo": "dask/dask"},
    {"instance_id": "pydantic__pydantic-8072", "repo": "pydantic/pydantic"},
    {"instance_id": "pandas-dev__pandas-58335", "repo": "pandas-dev/pandas"},
    {"instance_id": "facebookresearch__hydra-1783", "repo": "facebookresearch/hydra"},
    {"instance_id": "bokeh__bokeh-13636", "repo": "bokeh/bokeh"},
    {"instance_id": "Project-MONAI__MONAI-2238", "repo": "Project-MONAI/MONAI"},
]

SWEGYM_PREPARE = (
    "rm -rf /polar/session/workspace && "
    "mkdir -p /polar/session/logs/agent /polar/session/workspace /root/.venv/bin && "
    "cp -a /testbed/. /polar/session/workspace/ && "
    # swerex's shutil.copytree fails on dangling symlinks (e.g. bokeh repo)
    "find /polar/session/workspace -xtype l -delete 2>/dev/null; "
    "ln -sf /opt/miniconda3/envs/testbed/bin/python /root/.venv/bin/python && "
    "ln -sf /opt/miniconda3/envs/testbed/bin/python /root/.venv/bin/python3 && "
    "git config --global core.pager '' && "
    "cd /polar/session/workspace && git reset --hard"
)


def build_swegym_task(
    harness: str,
    sif_dir: str,
    instance: dict[str, Any],
    *,
    agent_model: str = "openai/gpt-4o",
    num_rollouts: int = 4,
    timeout_seconds: float = 900.0,
    batch_id: str = "",
) -> dict[str, Any]:
    """Build a SWE-Gym task payload for a single instance."""
    instance_id = instance["instance_id"]
    sif_name = f"swegym-{harness}-{_sanitize_instance_id(instance_id)}.sif"
    sif_path = os.path.join(sif_dir, sif_name)

    if not os.path.isfile(sif_path):
        raise FileNotFoundError(f"SWE-Gym SIF not found: {sif_path}")

    agent_settings: dict[str, Any] = {}
    agent_env: dict[str, str] = {}
    if harness == "swe_agent":
        agent_settings = {
            "repo_path": "/polar/session/workspace",
            "shell_preamble": (
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda activate polar-sweagent && "
                "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH"
            ),
        }
    elif harness in ("openhands_sdk", "openhands"):
        agent_env = {"WORKSPACE_BASE": "/polar/session/workspace"}

    return {
        "task_id": f"swegym-{harness}-{_sanitize_instance_id(instance_id)}-{batch_id}",
        "instruction": str(instance.get("problem_statement", "")).strip(),
        "num_rollouts": num_rollouts,
        "timeout_seconds": timeout_seconds,
        "runtime": {
            "backend": "apptainer",
            "image": sif_path,
            "prepare": [{"type": "exec", "command": SWEGYM_PREPARE}],
            "env": {},
            "network": "host",
            "workdir": "/polar/session/workspace",
            **({"kwargs": {"fakeroot": True}} if harness == "swe_agent" else {}),
        },
        "agent": {
            "harness": harness,
            "model_name": agent_model,
            "settings": agent_settings,
            "env": agent_env,
        },
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {
            "strategy": "swegym_git_diff",
            "config": {
                "repo_dir": "/testbed",
                "patch_command": "cd /polar/session/workspace && git add -A && git diff --cached --binary --submodule=diff",
                "instance": instance,
            },
            "refresh_runtime": False,
        },
    }


# ── SWE-bench Verified ────────────────────────────────────────────────────────

SWEBENCH_PREPARE_BASE = (
    "rm -rf /polar/session/workspace && "
    "mkdir -p /polar/session/logs/agent /polar/session/workspace /root/.venv/bin && "
    "cp -a /testbed/. /polar/session/workspace/ && "
    "find /polar/session/workspace -xtype l -delete 2>/dev/null; "
    "ln -sf /opt/miniconda3/envs/testbed/bin/python /root/.venv/bin/python && "
    "ln -sf /opt/miniconda3/envs/testbed/bin/python /root/.venv/bin/python3 && "
    "git config --global core.pager '' && "
    "cd /polar/session/workspace && git reset --hard; true"
)


def build_swebench_task(
    harness: str,
    sif_dir: str,
    instance: dict[str, Any],
    *,
    agent_model: str = "openai/gpt-4o",
    num_rollouts: int = 1,
    timeout_seconds: float = 3600.0,
    batch_id: str = "",
) -> dict[str, Any]:
    """Build a SWE-bench Verified task payload for a single instance."""
    instance_id = instance["instance_id"]
    sif_name = f"swebench-{harness}-{_sanitize_instance_id(instance_id)}.sif"
    sif_path = os.path.join(sif_dir, sif_name)

    if not os.path.isfile(sif_path):
        raise FileNotFoundError(f"SWE-bench SIF not found: {sif_path}")

    runtime_env: dict[str, str] = {}
    if harness == "opencode":
        runtime_env["OPENCODE_FAKE_VCS"] = "git"

    exclude_patterns: list[str] = []
    if harness == "claude_code":
        exclude_patterns.extend([".claude/**", "**/.claude/**"])

    agent_settings: dict[str, Any] = {}
    agent_env: dict[str, str] = {}
    if harness == "swe_agent":
        agent_settings = {
            "repo_path": "/polar/session/workspace",
            "shell_preamble": (
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda activate polar-sweagent && "
                "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH"
            ),
        }
    elif harness in ("openhands_sdk", "openhands"):
        agent_env = {"WORKSPACE_BASE": "/polar/session/workspace"}

    runtime_kwargs: dict[str, Any] = {}
    if harness == "swe_agent":
        runtime_kwargs["fakeroot"] = True

    return {
        "task_id": f"swebench-{harness}-{_sanitize_instance_id(instance_id)}-{batch_id}",
        "instruction": str(instance.get("problem_statement", "")).strip(),
        "num_rollouts": num_rollouts,
        "timeout_seconds": timeout_seconds,
        "runtime": {
            "backend": "apptainer",
            "image": sif_path,
            "prepare": [{"type": "exec", "command": SWEBENCH_PREPARE_BASE}],
            "env": runtime_env,
            "network": "host",
            "workdir": "/polar/session/workspace",
            **({"kwargs": runtime_kwargs} if runtime_kwargs else {}),
        },
        "agent": {
            "harness": harness,
            "model_name": agent_model,
            "settings": agent_settings,
            "env": agent_env,
        },
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {
            "strategy": "swegym_git_diff",
            "config": {
                "repo_dir": "/testbed",
                "patch_command": (
                    "cd /polar/session/workspace && "
                    "git add -A && git diff --cached --binary"
                ),
                "instance": instance,
                **({"exclude_patterns": exclude_patterns} if exclude_patterns else {}),
            },
            "refresh_runtime": False,
        },
    }


# ── Runner functions ──────────────────────────────────────────────────────────

def run_calculator(args: argparse.Namespace) -> int:
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir)

    payload = build_calculator_task(
        args.harness,
        args.sif_dir,
        args.output_dir,
        agent_model=args.agent_model,
        num_rollouts=args.num_rollouts,
        timeout_seconds=args.timeout_seconds,
        batch_id=batch_id,
    )
    request_path = output_dir / "request.json"
    response_path = output_dir / "response.json"
    _write_json(request_path, payload)
    print(f"[calculator] Wrote request to {request_path}")

    if args.dry_run:
        print("[calculator] Dry run — not submitting.")
        return 0

    result = _submit_task(request_path, args.topology)
    _write_json(response_path, result)
    summary = _summarize_result(result)
    print(f"[calculator] Done: {summary['reward_one_sessions']}/{summary['total_sessions']} reward=1.0")
    print(f"[calculator] Response: {response_path}")
    return 0


def run_swegym(args: argparse.Namespace) -> int:
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir)

    instances = SWEGYM_SAMPLE
    if args.instance_id:
        wanted = set(args.instance_id)
        instances = [i for i in instances if i["instance_id"] in wanted]
        missing = wanted - {i["instance_id"] for i in instances}
        if missing:
            print(f"[swegym] WARNING: Unknown instance_ids: {missing}")
    instances = instances[: args.max_tasks]

    if not instances:
        print("[swegym] No instances selected.")
        return 1

    # Try to load full instance data from cache
    cache_path = Path.home() / ".cache" / "polar" / "swegym_sample_10.json"
    full_instances: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        full_instances = {str(i.get("instance_id")): i for i in cached}

    manifest = {
        "batch_id": batch_id,
        "harness": args.harness,
        "model_name": args.model_name,
        "num_rollouts": args.num_rollouts,
        "tasks": [i["instance_id"] for i in instances],
    }
    _write_json(output_dir / "manifest.json", manifest)

    summaries: list[dict[str, Any]] = []
    for instance_meta in instances:
        instance_id = instance_meta["instance_id"]
        instance = {**instance_meta}
        if instance_id in full_instances:
            instance = full_instances[instance_id]

        if not instance.get("problem_statement"):
            print(
                f"[swegym] WARNING: No problem_statement for {instance_id}. "
                f"Run: python examples/swegym/sample_tasks.py to populate cache."
            )
            continue

        task_dir = output_dir / _sanitize_instance_id(instance_id)
        request_path = task_dir / "request.json"
        response_path = task_dir / "response.json"

        try:
            payload = build_swegym_task(
                args.harness,
                args.sif_dir,
                instance,
                agent_model=args.agent_model,
                num_rollouts=args.num_rollouts,
                timeout_seconds=args.timeout_seconds,
                batch_id=batch_id,
            )
        except FileNotFoundError as e:
            print(f"[swegym] Skipping {instance_id}: {e}")
            continue

        _write_json(request_path, payload)
        print(f"[swegym] [{instance_id}] Wrote request to {request_path}")

        if args.dry_run:
            summaries.append({"instance_id": instance_id, "dry_run": True})
            continue

        try:
            result = _submit_task(request_path, args.topology)
            _write_json(response_path, result)
            summary = {
                "instance_id": instance_id,
                "task_id": payload["task_id"],
                **_summarize_result(result),
            }
            summaries.append(summary)
            print(
                f"[swegym] [{instance_id}] Done: "
                f"reward_1={summary['reward_one_sessions']}/{summary['total_sessions']}"
            )
        except subprocess.CalledProcessError as e:
            print(f"[swegym] [{instance_id}] FAILED: {e}")
            if e.stderr:
                print(f"  stderr: {e.stderr[:500]}")
            summaries.append({"instance_id": instance_id, "error": str(e)})

    _write_json(output_dir / "summary.json", summaries)
    print(f"[swegym] Batch summary: {output_dir / 'summary.json'}")
    return 0


def run_swebench(args: argparse.Namespace) -> int:
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir)

    cache_path = Path.home() / ".cache" / "polar" / "swebench_verified.json"
    if not cache_path.exists():
        print(
            f"[swebench] ERROR: Dataset cache not found at {cache_path}\n"
            f"  Populate it with: python -c \""
            f"from examples.swebench_verified.dataset import load_swebench_verified; "
            f"load_swebench_verified()\""
        )
        return 1

    all_instances = json.loads(cache_path.read_text())
    instances_by_id: dict[str, dict[str, Any]] = {
        str(i["instance_id"]): i for i in all_instances
    }

    if args.instance_id:
        wanted = set(args.instance_id)
        instances = [instances_by_id[iid] for iid in wanted if iid in instances_by_id]
        missing = wanted - {str(i["instance_id"]) for i in instances}
        if missing:
            print(f"[swebench] WARNING: Unknown instance_ids: {missing}")
    else:
        instances = all_instances
    instances = instances[: args.max_tasks]

    if not instances:
        print("[swebench] No instances selected.")
        return 1

    manifest = {
        "batch_id": batch_id,
        "harness": args.harness,
        "num_rollouts": args.num_rollouts,
        "tasks": [str(i["instance_id"]) for i in instances],
    }
    _write_json(output_dir / "manifest.json", manifest)

    summaries: list[dict[str, Any]] = []
    for instance in instances:
        instance_id = str(instance["instance_id"])
        task_dir = output_dir / _sanitize_instance_id(instance_id)
        request_path = task_dir / "request.json"
        response_path = task_dir / "response.json"

        if not instance.get("problem_statement"):
            print(f"[swebench] WARNING: No problem_statement for {instance_id}, skipping.")
            continue

        try:
            payload = build_swebench_task(
                args.harness,
                args.sif_dir,
                instance,
                agent_model=args.agent_model,
                num_rollouts=args.num_rollouts,
                timeout_seconds=args.timeout_seconds,
                batch_id=batch_id,
            )
        except FileNotFoundError as e:
            print(f"[swebench] Skipping {instance_id}: {e}")
            continue

        _write_json(request_path, payload)
        print(f"[swebench] [{instance_id}] Wrote request to {request_path}")

        if args.dry_run:
            summaries.append({"instance_id": instance_id, "dry_run": True})
            continue

        try:
            result = _submit_task(request_path, args.topology)
            _write_json(response_path, result)
            summary = {
                "instance_id": instance_id,
                "task_id": payload["task_id"],
                **_summarize_result(result),
            }
            summaries.append(summary)
            print(
                f"[swebench] [{instance_id}] Done: "
                f"reward_1={summary['reward_one_sessions']}/{summary['total_sessions']}"
            )
        except subprocess.CalledProcessError as e:
            print(f"[swebench] [{instance_id}] FAILED: {e}")
            if e.stderr:
                print(f"  stderr: {e.stderr[:500]}")
            summaries.append({"instance_id": instance_id, "error": str(e)})

    _write_json(output_dir / "summary.json", summaries)
    print(f"[swebench] Batch summary: {output_dir / 'summary.json'}")
    return 0


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and submit Polar tasks on SLURM")
    parser.add_argument("--example", required=True, choices=["calculator", "swegym", "swebench_verified"])
    parser.add_argument("--harness", default="opencode")
    parser.add_argument("--topology", required=True, help="Path to topology.yaml")
    parser.add_argument("--sif-dir", required=True, help="Directory containing SIF images")
    parser.add_argument("--output-dir", default="./task_outputs")
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-27B"),
    )
    parser.add_argument(
        "--agent-model",
        default=os.environ.get("AGENT_MODEL", "openai/gpt-4o"),
    )
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.example == "calculator":
        return run_calculator(args)
    elif args.example == "swegym":
        return run_swegym(args)
    elif args.example == "swebench_verified":
        return run_swebench(args)
    else:
        print(f"Unknown example: {args.example}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
