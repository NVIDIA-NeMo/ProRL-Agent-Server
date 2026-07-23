#!/usr/bin/env python3
"""Evaluate an Inference Hub Codex model on a fixed Skill2Env sample."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shlex
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


DEFAULT_SKILL_DIR = Path("/home/haozh/.codex/skills/inference-hub")


def private_config(skill_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in (skill_dir / "credentials.env").read_text().splitlines():
        if raw.strip() and not raw.lstrip().startswith("#") and "=" in raw:
            key, value = raw.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def smoke(config: dict[str, str]) -> None:
    body = json.dumps({
        "model": config["OPENAI_MODEL"],
        "input": "Reply with exactly OK",
        "max_output_tokens": 16,
    }).encode()
    request = urllib.request.Request(
        config["OPENAI_BASE_URL"].rstrip("/") + "/responses",
        data=body,
        headers={
            "Authorization": "Bearer " + config["OPENAI_API_KEY"],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status // 100 != 2:
            raise RuntimeError(f"Inference Hub smoke test returned HTTP {response.status}")


def load_rows(manifest: Path) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    seen: set[str] = set()
    for row in rows:
        metadata = row["metadata"]
        task = metadata["task_name"].rsplit("/", 1)[-1]
        if task in seen:
            raise SystemExit(f"duplicate task id in manifest: {task}")
        seen.add(task)
        metadata["task_id"] = task
        for key in ("task_dir", "tests_dir", "sif_path", "workdir"):
            if not metadata.get(key):
                raise SystemExit(f"task {task} is missing metadata.{key}")
    return rows


def install_codex(apptainer: str, sif: Path, cli_dir: Path, version: str) -> None:
    cli_dir.mkdir(parents=True, exist_ok=True)
    binary = cli_dir / "node_modules" / ".bin" / "codex"
    if binary.is_file():
        return
    subprocess.run(
        [
            apptainer, "exec", "--cleanenv", "--bind", f"{cli_dir}:/opt/codex",
            str(sif), "bash", "-lc",
            f"npm install --prefix /opt/codex @openai/codex@{shlex.quote(version)}",
        ],
        check=True,
        timeout=600,
    )


def command_env(config: dict[str, str], metadata: dict) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        env.pop(key, None)
        env[f"APPTAINERENV_{key}"] = config[key]
    env["APPTAINERENV_CODEX_HOME"] = "/polar/session/.codex"
    trusted_path = "/opt/codex/node_modules/.bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    runtime_env = dict(metadata.get("runtime_env") or {})
    if "PATH" in runtime_env:
        parts = runtime_env["PATH"].split(":")
        runtime_env["PATH"] = ":".join(trusted_path if part == "" else part for part in parts)
    else:
        runtime_env["PATH"] = trusted_path
    for key, value in runtime_env.items():
        env[f"APPTAINERENV_{key}"] = str(value)
    return env


def runtime_prefix(
    apptainer: str,
    sif: Path,
    overlay: Path,
    session: Path,
    logs: Path,
    cli_dir: Path,
) -> list[str]:
    return [
        apptainer, "exec", "--overlay", str(overlay),
        "--bind", f"{session}:/polar/session",
        "--bind", f"{logs}:/logs",
        "--bind", f"{cli_dir}:/opt/codex:ro",
        str(sif),
    ]


def exec_runtime(
    prefix: list[str], command: str, env: dict[str, str], timeout: float, log=None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*prefix, "bash", "-lc", command],
        env=env,
        stdout=log or subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )


def inject_tests(prefix: list[str], tests: Path, env: dict[str, str]) -> None:
    made = exec_runtime(prefix, "mkdir -p /tests", env, 30)
    if made.returncode:
        raise RuntimeError("failed to create /tests")
    producer = subprocess.Popen(
        ["tar", "-cf", "-", "-C", str(tests), "."], stdout=subprocess.PIPE
    )
    consumer = subprocess.run(
        [*prefix, "tar", "-xf", "-", "-C", "/tests"],
        env=env,
        stdin=producer.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    assert producer.stdout is not None
    producer.stdout.close()
    producer.wait(timeout=30)
    if producer.returncode or consumer.returncode:
        raise RuntimeError("failed to inject verifier tests")


def redact(value: str, config: dict[str, str]) -> str:
    secret = config.get("OPENAI_API_KEY")
    return value.replace(secret, "[REDACTED]") if secret else value


def parse_reward(raw_text: str, raw_json: str) -> tuple[float | None, dict | None]:
    text = raw_text.strip()
    if text:
        try:
            return max(0.0, min(1.0, float(text))), None
        except ValueError:
            pass
    if raw_json.strip():
        try:
            scoring = json.loads(raw_json)
            if isinstance(scoring, (int, float)):
                return max(0.0, min(1.0, float(scoring))), None
            if isinstance(scoring, dict) and scoring:
                reward = sum(float(value) for value in scoring.values()) / len(scoring)
                return max(0.0, min(1.0, reward)), scoring
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return None, None


def run_task(
    row: dict,
    args: argparse.Namespace,
    config: dict[str, str],
    apptainer: str,
) -> dict:
    started = time.time()
    metadata = row["metadata"]
    task = metadata["task_id"]
    out = args.output_dir / "tasks" / task
    overlay, session, logs = out / "overlay", out / "session", out / "logs"
    for path in (overlay, session, logs):
        path.mkdir(parents=True, exist_ok=True)
    env = command_env(config, metadata)
    prefix = runtime_prefix(
        apptainer, Path(metadata["sif_path"]), overlay, session, logs, args.cli_dir
    )
    status, reward, error = "error", None, None
    agent_rc = verifier_rc = None
    verifier_scores = None
    try:
        setup = (
            "mkdir -p /polar/session/.codex /polar/session/logs /logs/agent && "
            "printf '{\"OPENAI_API_KEY\": \"%s\"}' \"$OPENAI_API_KEY\" "
            "> /polar/session/.codex/auth.json && "
            "printf 'openai_base_url = \"%s\"\\n' \"$OPENAI_BASE_URL\" "
            "> /polar/session/.codex/config.toml"
        )
        configured = exec_runtime(prefix, setup, env, 60)
        if configured.returncode:
            raise RuntimeError("Codex configuration failed: " + redact((configured.stdout or "")[-1000:], config))
        runtime_init = metadata.get("runtime_init_command")
        if runtime_init:
            initialized = exec_runtime(prefix, runtime_init, env, 60)
            if initialized.returncode:
                raise RuntimeError("runtime init failed: " + redact((initialized.stdout or "")[-1000:], config))
        instruction = row["prompt"][0]["content"].strip()
        command = (
            f"cd {shlex.quote(metadata['workdir'])} && "
            "/opt/codex/node_modules/.bin/codex exec "
            "--dangerously-bypass-approvals-and-sandbox "
            "--skip-git-repo-check --json --enable unified_exec "
            f"--model {shlex.quote(config['OPENAI_MODEL'])} "
            "-c model_reasoning_effort=xhigh -- " + shlex.quote(instruction)
        )
        with (logs / "codex.log").open("w") as log:
            agent = exec_runtime(prefix, command, env, float(metadata["agent_timeout"]) + 60, log)
        agent_rc = agent.returncode
        inject_tests(prefix, Path(metadata["tests_dir"]), env)
        with (logs / "verifier.log").open("w") as log:
            verifier = exec_runtime(
                prefix, "bash /tests/test.sh", env,
                float(metadata["verifier_timeout"]) + 60, log,
            )
        verifier_rc = verifier.returncode
        reward_text = exec_runtime(
            prefix, "test -f /logs/verifier/reward.txt && cat /logs/verifier/reward.txt", env, 30
        )
        reward_json = exec_runtime(
            prefix, "test -f /logs/verifier/reward.json && cat /logs/verifier/reward.json", env, 30
        )
        reward, verifier_scores = parse_reward(
            reward_text.stdout or "", reward_json.stdout or ""
        )
        status = "completed" if reward is not None else "error"
        if reward is None:
            error = "verifier produced no valid reward"
    except subprocess.TimeoutExpired as exc:
        error = f"timeout: {exc.cmd[-1] if isinstance(exc.cmd, list) else exc.cmd}"
    except Exception as exc:
        error = redact(str(exc), config)
    result = {
        "task": task,
        "status": status,
        "reward": reward,
        "agent_returncode": agent_rc,
        "verifier_returncode": verifier_rc,
        "verifier_scores": verifier_scores,
        "elapsed_seconds": round(time.time() - started, 1),
        "error": error,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (session / ".codex" / "auth.json").unlink(missing_ok=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cli-dir", type=Path, required=True)
    parser.add_argument("--skill-dir", type=Path, default=DEFAULT_SKILL_DIR)
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--codex-version", default="0.125.0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = private_config(args.skill_dir)
    smoke(config)
    apptainer = shutil.which("apptainer") or shutil.which("singularity")
    if not apptainer:
        raise SystemExit("apptainer/singularity not found")
    rows = load_rows(args.manifest)
    if args.task_id:
        by_id = {row["metadata"]["task_id"]: row for row in rows}
        missing = [task_id for task_id in args.task_id if task_id not in by_id]
        if missing:
            raise SystemExit(f"task ids not found in manifest: {missing}")
        rows = [by_id[task_id] for task_id in args.task_id]
        args.num_tasks = len(rows)
    if len(rows) < args.num_tasks:
        raise SystemExit(f"only {len(rows)} runnable manifest rows")
    selected = rows if args.task_id else random.Random(args.seed).sample(rows, args.num_tasks)
    selected_ids = [row["metadata"]["task_id"] for row in selected]
    (args.output_dir / "selected_tasks.txt").write_text("\n".join(selected_ids) + "\n")
    install_codex(
        apptainer, Path(selected[0]["metadata"]["sif_path"]),
        args.cli_dir, args.codex_version,
    )
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_task, row, args, config, apptainer) for row in selected]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    results.sort(key=lambda item: item["task"])
    completed = [item for item in results if item["reward"] is not None]
    correct = sum(item["reward"] == 1.0 for item in completed)
    summary = {
        "requested": args.num_tasks,
        "completed": len(completed),
        "correct": correct,
        "accuracy_all_requested": correct / args.num_tasks,
        "accuracy_completed": correct / len(completed) if completed else None,
        "seed": args.seed,
        "model": config["OPENAI_MODEL"],
        "manifest": str(args.manifest.resolve()),
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if len(completed) == args.num_tasks else 2


if __name__ == "__main__":
    raise SystemExit(main())
