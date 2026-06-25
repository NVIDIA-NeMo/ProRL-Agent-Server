#!/usr/bin/env python3
"""Run the calculator demo across every harness and print a comparison table.

Each harness gets a tiny `calculator.py` with parser stubs, edits it, and the
evaluator runs `python3 test_calculator.py`. All harnesses are submitted at
once; live progress and per-session detail are visible in the dashboard
(`polar dashboard -c examples/calculator/topology.vllm.yaml`).

    uv run python examples/calculator/run.py                 # docker (default)
    uv run python examples/calculator/run.py --backend apptainer
    uv run python examples/calculator/run.py --harness codex # Codex-only smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = EXAMPLE_DIR / "assets"
TEST_FILE = ASSETS_DIR / "test_calculator.py"
STARTER_FILE = ASSETS_DIR / "calculator.py"
DEFAULT_TOPOLOGY = EXAMPLE_DIR / "topology.vllm.yaml"
RUNTIME_IMAGE = "polar-localhost-calculator:latest"
NUM_SAMPLES = 4
# Generous budget: INIT install (npm / pip / venv) shares the per-task budget
# with the agent run and evaluation.
TIMEOUT_SECONDS = 1200.0
POLL_INTERVAL_SECONDS = 10.0
CODEX_VERSION = "0.125.0"
CODEX_REASONING_EFFORT = "xhigh"

HARNESSES = (
    "claude_code",
    "codex",
    "gemini_cli",
    "opencode",
    "pi",
    "qwen_code",
    "openhands_sdk",
    "openclaw",
    "hermes",
    "mini_swe_agent",
)

INSTRUCTION = """\
`calculator.py` has a `Calculator` class with a tokenizer and three stub methods.
Each stub is marked with a `# TODO` comment and returns `0`.

Implement the three methods to build a recursive-descent expression parser:

1. `_parse_expr`  — handle `+` and `-` by calling `_parse_term`
2. `_parse_term`  — handle `*` and `/` (integer division) by calling `_parse_factor`
3. `_parse_factor` — handle integer literals and parenthesized sub-expressions

Also fix `__call__` to return the parsed value instead of `0`.

Requirements:
- Work only in `/polar/session/workspace/calculator.py`.
- Keep the existing file structure, `_tokenize`, `_peek`, and `_consume` as-is.
- Do not add imports.
- Use `//` for division (integer division).
- You must make actual edits. An empty git diff fails the task.

After editing, run `python3 test_calculator.py` to test.
"""

# Per-harness INIT install command. npm CLIs install globally into
# ~/.local/bin; the Python agents install via pip (hermes and mini-swe-agent from
# PyPI into ~/.local, openhands-sdk into ~/.venv where its harness looks for the
# interpreter). The 3.12 runtime image satisfies their Python >=3.11 floor, so a
# plain pip install works here (the tmax example needs uv for its 3.10 images).
# Pinned versions keep the quickstart stable. Bump intentionally.
HARNESS_INSTALL: dict[str, str] = {
    "claude_code": "npm install -g @anthropic-ai/claude-code@2.1.111",
    "codex": f"npm install -g @openai/codex@{CODEX_VERSION}",
    "gemini_cli": "npm install -g @google/gemini-cli@0.38.1",
    "opencode": "npm install -g opencode-ai@1.4.6",
    "pi": "npm install -g @mariozechner/pi-coding-agent@0.67.68",
    "qwen_code": "npm install -g @qwen-code/qwen-code@0.14.5",
    "openclaw": "npm install -g openclaw@2026.5.27",
    "hermes": "python3 -m pip install --user --quiet hermes-agent==0.15.1",
    "mini_swe_agent": "python3 -m pip install --user --quiet mini-swe-agent==2.4.2",
    # Pin sdk + tools to the same version. Unpinned, pip resolves a mismatched
    # pair (sdk 1.17 + tools 1.24) whose imports break; the latest 1.24 needs
    # Python 3.13 (lmnr dep conflict on 3.12), so pin to 1.17.0 for this image.
    "openhands_sdk": (
        "python3 -m venv $HOME/.venv && "
        "$HOME/.venv/bin/pip install --quiet "
        "openhands-sdk==1.17.0 openhands-tools==1.17.0"
    ),
}

# Model name the harness CLI sends; the gateway rewrites it to the served model.
HARNESS_MODEL: dict[str, str] = {
    "claude_code": "claude-opus-4-5",
    "codex": "openai/gpt-5.5",
    "gemini_cli": "gemini-2.5-flash-lite",
    "opencode": "openai/gpt-5.4",
    "pi": "openai/gpt-5.4",
    "qwen_code": "qwen3-coder-plus",
    "openhands_sdk": "openai/gpt-5.4",
    "openclaw": "openai/gpt-5.4",
    "hermes": "openai/gpt-5.4",
    "mini_swe_agent": "openai/gpt-5.4",
}

# INIT stage: install the harness CLI, then set up a clean git workspace.
_WORKSPACE_PREPARE = (
    "rm -rf /polar/session/workspace && "
    "mkdir -p /polar/session/workspace /polar/session/logs/agent && "
    "cd /polar/session/workspace && "
    "git init -q && "
    "git config user.email 'polar@test' && "
    "git config user.name 'Polar'"
)

# Config/cache dirs that can leak into the workspace git diff.
_EVAL_EXCLUDES: dict[str, list[str]] = {
    "claude_code": [".claude/**", "**/.claude/**"],
    "codex": [".codex/**", "**/.codex/**"],
    "gemini_cli": [".gemini/**", "**/.gemini/**"],
    "opencode": [".opencode/**", "**/.opencode/**", ".config/opencode/**"],
    "pi": [".pi/**", "**/.pi/**"],
    "qwen_code": [".qwen/**", "**/.qwen/**"],
    "openclaw": [".openclaw/**", "**/.openclaw/**"],
    "hermes": [".hermes/**", "**/.hermes/**"],
    "openhands_sdk": [".openhands/**", "**/.openhands/**"],
    "mini_swe_agent": [".mini-swe-agent/**", "**/.mini-swe-agent/**", ".config/mini-swe-agent/**"],
}
_COMMON_EXCLUDES = [
    "node_modules/**",
    "**/node_modules/**",
    ".cache/**",
    "**/.cache/**",
    ".venv/**",
    "**/.venv/**",
]


def runtime_image_for_backend(backend: str) -> str:
    if backend == "apptainer":
        return f"docker-daemon:{RUNTIME_IMAGE}"
    return RUNTIME_IMAGE


def build_task_payload(harness: str, batch_id: str, backend: str) -> dict[str, Any]:
    agent: dict[str, Any] = {"harness": harness, "model_name": HARNESS_MODEL[harness]}
    if harness == "codex":
        agent["settings"] = {
            "version": CODEX_VERSION,
            "reasoning_effort": CODEX_REASONING_EFFORT,
        }

    return {
        "task_id": f"calculator-{harness}-{batch_id}",
        "instruction": INSTRUCTION,
        "num_samples": NUM_SAMPLES,
        "timeout_seconds": TIMEOUT_SECONDS,
        "runtime": {
            "backend": backend,
            "image": runtime_image_for_backend(backend),
            "prepare": [
                {"type": "exec", "command": f"{HARNESS_INSTALL[harness]} && {_WORKSPACE_PREPARE}"},
                {
                    "type": "upload_file",
                    "source": str(TEST_FILE),
                    "target": "/polar/session/workspace/test_calculator.py",
                },
                {
                    "type": "upload_file",
                    "source": str(STARTER_FILE),
                    "target": "/polar/session/workspace/calculator.py",
                },
                {
                    "type": "exec",
                    "command": "cd /polar/session/workspace && git add -A && git commit -qm 'initial'",
                },
            ],
            "network": "host",
            "workdir": "/polar/session/workspace",
        },
        "agent": agent,
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {
            "strategy": "test_on_output",
            "config": {
                "repo_dir": "/polar/session/workspace",
                "patch_command": "cd /polar/session/workspace && git add -A && git diff --cached --binary",
                "test_command": "cd /polar/session/workspace && python3 test_calculator.py && echo 'PASSED test_calculator'",
                "test_timeout": 60.0,
                "expected_output_json": {"test_calculator": "PASSED"},
                "exclude_patterns": [*_COMMON_EXCLUDES, *_EVAL_EXCLUDES[harness]],
            },
            "refresh_runtime": True,
        },
    }


def session_reward(session: dict[str, Any]) -> float | None:
    traces = (session.get("trajectory") or {}).get("traces") or []
    reward = traces[-1].get("reward") if traces else None
    return float(reward) if isinstance(reward, (int, float)) else None


def print_comparison(finished: dict[str, dict[str, Any]], elapsed: float) -> None:
    header = f"{'Harness':<16} {'Reward':>8}  {'Done':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for harness, result in finished.items():
        sessions = result.get("results") or []
        rewards = [r for r in (session_reward(s) for s in sessions) if r is not None]
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        done = sum(1 for s in sessions if s.get("status") == "COMPLETED")
        print(f"{harness:<16} {mean:>8.3f}  {done:>2}/{len(sessions):<2}")
    print("=" * len(header))
    print(f"Wall time: {elapsed:.0f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["docker", "apptainer"], default="docker")
    parser.add_argument(
        "--harness",
        action="append",
        choices=HARNESSES,
        help="Run only this harness. Repeat to select more than one.",
    )
    args = parser.parse_args()
    backend = args.backend
    selected_harnesses = tuple(args.harness or HARNESSES)

    from polar.config import TopologyConfig

    rollout_url = TopologyConfig.load(DEFAULT_TOPOLOGY).rollout.public_url
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    print(f"Submitting {len(selected_harnesses)} harnesses to {rollout_url} (backend={backend})")
    timeout = httpx.Timeout(None, connect=30.0)
    with httpx.Client(base_url=rollout_url, timeout=timeout) as client:
        task_ids: dict[str, str] = {}
        for harness in selected_harnesses:
            payload = build_task_payload(harness, batch_id, backend)
            resp = client.post("/rollout/task/submit", json=payload)
            resp.raise_for_status()
            task_ids[harness] = resp.json()["task_id"]
            print(f"  {harness:<16} -> {task_ids[harness]}")

        print(f"\nPolling every {POLL_INTERVAL_SECONDS:.0f}s (watch live in the dashboard) ...")
        t0 = time.monotonic()
        finished: dict[str, dict[str, Any]] = {}
        while len(finished) < len(selected_harnesses):
            time.sleep(POLL_INTERVAL_SECONDS)
            for harness, tid in task_ids.items():
                if harness in finished:
                    continue
                status = client.get(f"/rollout/task/{tid}").json()
                if status["status"] != "running":
                    finished[harness] = status
                    print(f"  [{time.monotonic() - t0:>5.0f}s] {harness} done")
        elapsed = time.monotonic() - t0

    print_comparison(finished, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
