#!/usr/bin/env python3
"""Submit SWE-bench Verified tasks through the Polar rollout server.

Usage:
    python submit_swebench_tasks.py --harness opencode
    python submit_swebench_tasks.py --harness codex --max-tasks 50 --num-samples 4
    python submit_swebench_tasks.py --harness claude_code --instance-id django__django-15098
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dataset import (
    DATASET_NAME,
    SUPPORTED_HARNESSES,
    load_swebench_verified,
    runtime_image_for_instance,
    sanitize_instance_id,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_TOPOLOGY = EXAMPLE_DIR / "topology.yaml"

HARNESS_NPM_PACKAGE: dict[str, str] = {
    "codex": "@openai/codex@latest",
    "opencode": "opencode-ai@latest",
    "claude_code": "@anthropic-ai/claude-code@latest",
    "gemini_cli": "@google/gemini-cli@latest",
    "qwen_code": "@qwen-code/qwen-code@latest",
}

_PREPARE_BASE = (
    "rm -rf /polar/session/workspace && "
    "mkdir -p /polar/session/logs/agent /polar/session/workspace \"$HOME/.venv/bin\" && "
    "cp -a /testbed/. /polar/session/workspace/ && "
    "ln -sf /opt/miniconda3/envs/testbed/bin/python \"$HOME/.venv/bin/python\" && "
    "ln -sf /opt/miniconda3/envs/testbed/bin/python \"$HOME/.venv/bin/python3\" && "
    "git config --global core.pager '' && "
    "cd /polar/session/workspace && git reset --hard; true"
)


def prepare_command_for_harness(harness: str) -> str:
    if harness in HARNESS_NPM_PACKAGE:
        pkg = HARNESS_NPM_PACKAGE[harness]
        return f"npm install -g {pkg} && {_PREPARE_BASE}"
    if harness == "swe_agent":
        return (
            "source /opt/miniconda3/etc/profile.d/conda.sh && "
            "conda activate polar-sweagent && "
            f"{_PREPARE_BASE}"
        )
    if harness == "openhands_sdk":
        return _PREPARE_BASE
    raise ValueError(f"Unknown harness: {harness}")


def runtime_env_for_harness(harness: str) -> dict[str, str]:
    env: dict[str, str] = {}
    if harness == "opencode":
        env["OPENCODE_FAKE_VCS"] = "git"
    return env


def evaluator_exclude_patterns_for_harness(harness: str) -> list[str]:
    patterns: list[str] = []
    if harness == "claude_code":
        patterns.extend([".claude/**", "**/.claude/**"])
    return patterns



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, choices=SUPPORTED_HARNESSES)
    parser.add_argument(
        "--topology",
        default=os.environ.get("POLAR_TOPOLOGY", str(DEFAULT_TOPOLOGY)),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=int(os.environ.get("NUM_SAMPLES", "1")),
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=int(os.environ.get("MAX_TASKS", "-1")),
        help="Maximum tasks to submit. -1 = all 500.",
    )
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--runtime-backend",
        choices=["docker", "apptainer"],
        default=os.environ.get("RUNTIME_BACKEND", "apptainer"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=int(os.environ.get("MAX_CONCURRENT", "32")),
        help="Max parallel task submissions.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True))


def docker_image_exists(image_ref: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def runtime_image_for_backend(image: str, backend: str) -> str:
    if backend != "apptainer":
        return image
    if image.startswith(("docker-daemon:", "docker://", "oras://")):
        return image
    return f"docker-daemon:{image}"


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------

def select_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    instances = load_swebench_verified()
    if args.instance_id:
        wanted = set(args.instance_id)
        selected = [i for i in instances if str(i.get("instance_id")) in wanted]
        missing = sorted(wanted - {str(i.get("instance_id")) for i in selected})
        if missing:
            raise SystemExit(f"Unknown instance_id(s): {', '.join(missing)}")
        return selected
    if args.max_tasks > 0:
        return instances[: args.max_tasks]
    return instances


def agent_settings_for_harness(harness: str) -> dict[str, Any]:
    if harness == "swe_agent":
        return {
            "repo_path": "/polar/session/workspace",
            "shell_preamble": (
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda activate polar-sweagent && "
                "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH"
            ),
        }
    return {}


def agent_env_for_harness(harness: str) -> dict[str, str]:
    if harness in ("openhands_sdk", "openhands"):
        return {"WORKSPACE_BASE": "/polar/session/workspace"}
    return {}


def runtime_kwargs_for_harness(harness: str) -> dict[str, Any]:
    if harness == "swe_agent":
        return {"fakeroot": True}
    return {}


def build_task_request(
    args: argparse.Namespace,
    *,
    instance: dict[str, Any],
    batch_id: str,
) -> dict[str, Any]:
    instance_id = str(instance["instance_id"])
    image = runtime_image_for_instance(instance_id)
    kwargs = runtime_kwargs_for_harness(args.harness)
    return {
        "task_id": f"swebench-{args.harness}-{sanitize_instance_id(instance_id)}-{batch_id}",
        "instruction": str(instance["problem_statement"]).strip(),
        "num_samples": args.num_samples,
        "timeout_seconds": args.timeout_seconds,
        "runtime": {
            "backend": args.runtime_backend,
            "image": runtime_image_for_backend(image, args.runtime_backend),
            "prepare": [{"type": "exec", "command": prepare_command_for_harness(args.harness)}],
            "env": runtime_env_for_harness(args.harness),
            "network": "host",
            "workdir": "/polar/session/workspace",
            **({"kwargs": kwargs} if kwargs else {}),
        },
        "agent": {
            "harness": args.harness,
            "settings": agent_settings_for_harness(args.harness),
            "env": agent_env_for_harness(args.harness),
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
                "exclude_patterns": evaluator_exclude_patterns_for_harness(args.harness),
            },
            "refresh_runtime": True,
        },
    }


# ---------------------------------------------------------------------------
# Submission & result handling
# ---------------------------------------------------------------------------

def submit_task_file(
    request_path: Path,
    *,
    topology: str | None,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "polar.cli", "submit", str(request_path), "--json"]
    if topology:
        command.extend(["-c", topology])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def summarize_result(response: dict[str, Any]) -> dict[str, Any]:
    sessions = response.get("results") or []
    reward_one = completed = session_errors = trajectory_errors = 0
    for session in sessions:
        if session.get("status") == "COMPLETED":
            completed += 1
        if session.get("error"):
            session_errors += 1
        traj = session.get("trajectory") or {}
        if traj.get("status") == "ERROR" or traj.get("error"):
            trajectory_errors += 1
        traces = traj.get("traces") or []
        if traces and traces[-1].get("reward") == 1.0:
            reward_one += 1
    return {
        "total_sessions": len(sessions),
        "completed_sessions": completed,
        "reward_one_sessions": reward_one,
        "session_errors": session_errors,
        "trajectory_errors": trajectory_errors,
    }


def print_reward_summary(summaries: list[dict[str, Any]]) -> None:
    total_tasks = len(summaries)
    resolved = sum(1 for s in summaries if s.get("reward_one_sessions", 0) > 0)
    total_sessions = sum(s.get("total_sessions", 0) for s in summaries)
    total_reward_one = sum(s.get("reward_one_sessions", 0) for s in summaries)
    total_errors = sum(s.get("session_errors", 0) + s.get("trajectory_errors", 0) for s in summaries)

    print("\n" + "=" * 72)
    print("  SWE-bench Verified — Reward Summary")
    print("=" * 72)
    print(f"  Tasks submitted:       {total_tasks}")
    print(
        f"  Tasks resolved (≥1):   {resolved}/{total_tasks}"
        f"  ({100 * resolved / max(total_tasks, 1):.1f}%)"
    )
    print(f"  Total sessions:        {total_sessions}")
    print(
        f"  Sessions reward=1:     {total_reward_one}/{total_sessions}"
        f"  ({100 * total_reward_one / max(total_sessions, 1):.1f}%)"
    )
    if total_errors:
        print(f"  Errors:                {total_errors}")
    print("=" * 72)

    print(f"\n  {'Instance ID':<45} {'Resolved':>10} {'Sessions':>10}")
    print("  " + "-" * 65)
    for s in sorted(summaries, key=lambda x: x.get("instance_id", "")):
        iid = s.get("instance_id", "???")
        r1 = s.get("reward_one_sessions", 0)
        total = s.get("total_sessions", 0)
        print(f"  {iid:<45} {r1:>5}/{total:<4} {total:>10}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    instances = select_instances(args)
    if not instances:
        raise SystemExit("No instances selected.")
    print(f"Selected {len(instances)} SWE-bench Verified instance(s).")

    ready: list[dict[str, Any]] = []
    missing_images: list[str] = []
    for instance in instances:
        image_ref = runtime_image_for_instance(str(instance["instance_id"]))
        if docker_image_exists(image_ref):
            ready.append(instance)
        else:
            missing_images.append(image_ref)

    if args.instance_id and missing_images:
        raise SystemExit(
            f"Missing runtime image(s):\n"
            + "\n".join(f"  - {img}" for img in missing_images)
            + "\nRun: python build_images.py"
        )
    if not ready:
        raise SystemExit(
            "No runtime images found.\n"
            "Run: python build_images.py"
        )
    if missing_images:
        print(f"Skipping {len(missing_images)} instance(s) with missing images.")
    instances = ready

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else EXAMPLE_DIR / args.harness / "batches" / batch_id
    )
    manifest = {
        "dataset": DATASET_NAME,
        "batch_id": batch_id,
        "harness": args.harness,
        "num_samples": args.num_samples,
        "total_tasks": len(instances),
    }
    write_json(output_dir / "manifest.json", manifest)

    prepared: list[tuple[str, dict[str, Any], Path, Path]] = []
    for instance in instances:
        iid = str(instance["instance_id"])
        task_dir = output_dir / sanitize_instance_id(iid)
        req_path = task_dir / "request.json"
        resp_path = task_dir / "response.json"
        payload = build_task_request(args, instance=instance, batch_id=batch_id)
        write_json(req_path, payload)
        prepared.append((iid, payload, req_path, resp_path))
    print(f"Wrote {len(prepared)} request file(s) under {output_dir}")

    max_workers = min(len(prepared), args.max_concurrent)
    print(f"Submitting {len(prepared)} task(s) (max_concurrent={max_workers}) ...")

    summaries: list[dict[str, Any]] = []

    def _submit_one(item: tuple[str, dict[str, Any], Path, Path]) -> dict[str, Any]:
        iid, payload, req_path, resp_path = item
        result = submit_task_file(req_path, topology=args.topology)
        write_json(resp_path, result)
        summary = {
            "instance_id": iid,
            "task_id": payload["task_id"],
            "response_path": str(resp_path),
            **summarize_result(result),
        }
        print(
            f"  [{iid}] reward_1={summary['reward_one_sessions']}"
            f"/{summary['total_sessions']}"
        )
        return summary

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_submit_one, item): item for item in prepared}
        for future in as_completed(futures):
            try:
                summaries.append(future.result())
            except Exception as exc:
                iid = futures[future][0]
                print(f"  [{iid}] FAILED: {exc}")
                summaries.append({
                    "instance_id": iid,
                    "task_id": futures[future][1]["task_id"],
                    "error": str(exc),
                    "total_sessions": 0,
                    "completed_sessions": 0,
                    "reward_one_sessions": 0,
                    "session_errors": 1,
                    "trajectory_errors": 0,
                })

    write_json(output_dir / "summary.json", summaries)
    print_reward_summary(summaries)
    print(f"Batch summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
