#!/usr/bin/env python3
"""Submit TMax-15K-Harbor tasks to the Polar rollout server.

Each task runs a coding-agent harness inside the task's container, then the
``harbor`` evaluator injects the task's ``tests/`` and runs ``test.sh`` —
reproducing TMax's reward (1.0 = ``test_final_state.py`` passed, else 0.0).

    uv run python examples/tmax-15k/submit_tmax_tasks.py --dataset-dir <dir> --harness codex --max-tasks 10
    uv run python examples/tmax-15k/submit_tmax_tasks.py --dataset-dir <dir> --harness claude_code --num-samples 4
    uv run python examples/tmax-15k/submit_tmax_tasks.py --dataset-dir <dir> --harness codex --task task_000123_ab12cd34
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from dataset import (
    SUPPORTED_HARNESSES,
    TmaxTask,
    load_tasks,
    runtime_image_for,
    sanitize,
    sif_filename_for,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_TOPOLOGY = EXAMPLE_DIR / "topology.vllm.yaml"
POLL_INTERVAL_SECONDS = 15.0

# Per-harness INIT install command. The Node CLIs install globally. hermes and
# mini-swe-agent are PyPI packages that need Python >=3.11, but TMax task images
# ship whatever Python they were built with (this set includes 3.10), so installing
# against the image's system Python fails. We install them with uv against a managed
# 3.12 interpreter; `uv tool install` drops the entry point in $HOME/.local/bin,
# which both presets' PATH already includes.
# codex must match presets/codex.py DEFAULT_CODEX_VERSION (the preset hard-fails
# on a version mismatch). Bump versions intentionally.
HARNESS_INSTALL: dict[str, str] = {
    "codex": "npm install -g @openai/codex@0.125.0",
    "claude_code": "npm install -g @anthropic-ai/claude-code@2.1.111",
    "opencode": "npm install -g opencode-ai@1.4.6",
    "qwen_code": "npm install -g @qwen-code/qwen-code@0.14.5",
    "pi": "npm install -g @mariozechner/pi-coding-agent@0.67.68",
    "hermes": (
        "curl -LsSf https://astral.sh/uv/install.sh | sh "
        '&& export PATH="$HOME/.local/bin:$PATH" '
        "&& uv tool install --python 3.12 hermes-agent==0.15.1"
    ),
    "mini_swe_agent": (
        "curl -LsSf https://astral.sh/uv/install.sh | sh "
        '&& export PATH="$HOME/.local/bin:$PATH" '
        "&& uv tool install --python 3.12 mini-swe-agent==2.4.2"
    ),
}


def model_name_for(harness: str, model_name: str) -> str:
    """The pi preset requires ``provider/model`` form; the gateway rewrites the
    model id to the served model, so the provider prefix is all that matters."""
    if harness == "pi" and "/" not in model_name:
        return f"openai/{model_name}"
    return model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Exported Harbor dataset directory.")
    parser.add_argument("--harness", required=True, choices=SUPPORTED_HARNESSES)
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per task (pass@k).")
    parser.add_argument("--max-tasks", type=int, default=-1, help="Max tasks to submit. -1 = all.")
    parser.add_argument("--task", action="append", default=[], help="Only submit these task(s). Repeatable.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=-1.0,
        help="Session budget per sample. -1 = derive from task.toml (agent + verifier + 120s).",
    )
    parser.add_argument(
        "--workdir",
        default="/root",
        help="Agent working dir in the container. TMax instructions use absolute paths; "
        "pass '/' to match Harbor exactly for tasks with relative-path instructions.",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-5.4",
        help="Model name the harness sends; the gateway rewrites it to the served model.",
    )
    parser.add_argument("--runtime-backend", choices=["docker", "apptainer"], default="docker")
    parser.add_argument(
        "--apptainer-image-dir",
        default=None,
        help="Dir of prebuilt .sif files for a docker-free Slurm flow. With "
        "--runtime-backend apptainer, launch <dir>/<task>.sif directly instead of "
        "reading from a local docker daemon. Build them with prepare_apptainer_images.py.",
    )
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    return parser.parse_args()


def resolve_runtime_image(args: argparse.Namespace, task: TmaxTask) -> str:
    """The image reference Polar's runtime launches for *task*.

    ``.sif`` path (docker-free apptainer) > ``docker-daemon:`` (apptainer reading
    the local docker daemon) > the plain docker tag.
    """
    if args.runtime_backend == "apptainer" and args.apptainer_image_dir:
        return str(Path(args.apptainer_image_dir).expanduser() / sif_filename_for(task.name))
    image = runtime_image_for(task.name)
    if args.runtime_backend == "apptainer" and not image.startswith(("docker-daemon:", "docker://", "oras://")):
        return f"docker-daemon:{image}"
    return image


def image_available(args: argparse.Namespace, task: TmaxTask) -> bool:
    """Whether the launchable image exists — a ``.sif`` for docker-free apptainer,
    otherwise a local docker image (so this works on nodes without docker)."""
    if args.runtime_backend == "apptainer" and args.apptainer_image_dir:
        return (Path(args.apptainer_image_dir).expanduser() / sif_filename_for(task.name)).is_file()
    return docker_image_exists(runtime_image_for(task.name))


def docker_image_exists(image_ref: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def session_timeout(args: argparse.Namespace, task: TmaxTask) -> float:
    if args.timeout_seconds > 0:
        return args.timeout_seconds
    return task.agent_timeout + task.verifier_timeout + 120.0


def build_task_request(args: argparse.Namespace, task: TmaxTask, batch_id: str) -> dict[str, Any]:
    image = resolve_runtime_image(args, task)
    return {
        "task_id": f"tmax15k-{args.harness}-{sanitize(task.name)}-{batch_id}",
        "instruction": task.instruction,
        "num_samples": args.num_samples,
        "timeout_seconds": session_timeout(args, task),
        "runtime": {
            "backend": args.runtime_backend,
            "image": image,
            "prepare": [{"type": "exec", "command": HARNESS_INSTALL[args.harness]}],
            "env": {"HOME": args.workdir} if args.workdir == "/root" else {},
            "network": "host",
            "workdir": task.workdir or args.workdir,
        },
        "agent": {"harness": args.harness, "model_name": model_name_for(args.harness, args.model_name)},
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {
            "strategy": "harbor",
            "config": {
                "tests_dir": str(task.tests_dir.resolve()),
                "verifier_timeout": task.verifier_timeout,
            },
            "refresh_runtime": False,
        },
    }


def task_stats(result: dict[str, Any]) -> tuple[int, int]:
    """Return (sessions with reward==1, total sessions) for one finished task."""
    sessions = result.get("results") or []
    reward_one = 0
    for session in sessions:
        traces = (session.get("trajectory") or {}).get("traces") or []
        if traces and traces[-1].get("reward") == 1.0:
            reward_one += 1
    return reward_one, len(sessions)


def print_summary(stats: dict[str, tuple[int, int]], elapsed: float) -> None:
    total_tasks = len(stats)
    resolved = sum(1 for r1, _ in stats.values() if r1 > 0)
    total_sessions = sum(total for _, total in stats.values())
    reward_one = sum(r1 for r1, _ in stats.values())

    print("\n" + "=" * 72)
    print("  TMax-15K-Harbor — Reward Summary")
    print("=" * 72)
    print(f"  Tasks resolved (>=1):  {resolved}/{total_tasks}  ({100 * resolved / max(total_tasks, 1):.1f}%)")
    print(f"  Sessions reward=1:     {reward_one}/{total_sessions}  ({100 * reward_one / max(total_sessions, 1):.1f}%)")
    print(f"  Wall time:             {elapsed:.0f}s")
    print("=" * 72)
    for name in sorted(stats):
        r1, total = stats[name]
        print(f"  {name:<45} {f'{r1}/{total}':>12}")
    print(f"\n  Per-session detail: polar dashboard -c {DEFAULT_TOPOLOGY}")


def main() -> int:
    args = parse_args()
    batch_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tasks = load_tasks(args.dataset_dir, max_tasks=args.max_tasks, names=args.task or None)

    ready, missing = [], []
    for task in tasks:
        (ready if image_available(args, task) else missing).append(task)
    if not ready:
        how = (
            "prepare_apptainer_images.py and copy the .sif dir to this node"
            if args.runtime_backend == "apptainer" and args.apptainer_image_dir
            else "build_images.py --dataset-dir ..."
        )
        raise SystemExit(f"No runtime images found. Build them with: python {how}")
    if missing:
        names = ", ".join(t.name for t in missing)
        print(f"Skipping {len(missing)} task(s) with missing images (build them first): {names}")
    tasks = ready

    from polar.config import TopologyConfig

    rollout_url = TopologyConfig.load(args.topology).rollout.public_url
    print(f"Submitting {len(tasks)} task(s) to {rollout_url} "
          f"(harness={args.harness}, samples={args.num_samples}, backend={args.runtime_backend})")

    timeout = httpx.Timeout(None, connect=30.0)
    with httpx.Client(base_url=rollout_url, timeout=timeout) as client:
        task_ids: dict[str, str] = {}  # task name -> rollout task_id
        for task in tasks:
            payload = build_task_request(args, task, batch_id)
            resp = client.post("/rollout/task/submit", json=payload)
            resp.raise_for_status()
            task_ids[task.name] = resp.json()["task_id"]

        print(f"Polling every {POLL_INTERVAL_SECONDS:.0f}s (watch live in the dashboard) ...")
        t0 = time.monotonic()
        stats: dict[str, tuple[int, int]] = {}
        while len(stats) < len(task_ids):
            time.sleep(POLL_INTERVAL_SECONDS)
            for name, tid in task_ids.items():
                if name in stats:
                    continue
                status = client.get(f"/rollout/task/{tid}").json()
                if status["status"] != "running":
                    r1, total = task_stats(status)
                    stats[name] = (r1, total)
                    print(f"  [{time.monotonic() - t0:>5.0f}s] {name:<45} resolved={r1}/{total}  "
                          f"({len(stats)}/{len(task_ids)} done)")
        elapsed = time.monotonic() - t0

    print_summary(stats, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
