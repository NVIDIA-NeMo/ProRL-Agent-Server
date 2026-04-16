"""Command-line interface for Polar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse

import httpx
import yaml

from polar.config import TopologyConfig
from polar.gateway.server import serve as serve_gateway
from polar.rollout.server import serve as serve_rollout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polar",
        description="Run Polar services and interact with a Polar topology.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rollout_parser = subparsers.add_parser(
        "serve_rollout",
        help="Start the rollout service.",
    )
    rollout_parser.add_argument(
        "-c",
        "--config",
        default="topology.yaml",
        help="Path to topology.yaml",
    )

    gateway_parser = subparsers.add_parser(
        "serve_gateway",
        help="Start one gateway node from topology.yaml.",
    )
    gateway_parser.add_argument(
        "-c",
        "--config",
        default="topology.yaml",
        help="Path to topology.yaml",
    )
    gateway_parser.add_argument(
        "--node-id",
        help="Gateway node id to launch. Required when topology has more than one node.",
    )

    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit a task file to the rollout service.",
    )
    submit_parser.add_argument("task_file", help="Path to a task JSON or YAML file")
    submit_parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Optional topology.yaml used to discover the rollout URL.",
    )
    submit_parser.add_argument(
        "--rollout-url",
        default=None,
        help="Override the rollout server URL.",
    )
    submit_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw response JSON.",
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Show rollout health and gateway topology.",
    )
    status_parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Optional topology.yaml used to discover the rollout URL.",
    )
    status_parser.add_argument(
        "--rollout-url",
        default=None,
        help="Override the rollout server URL.",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw response JSON.",
    )

    # ── cluster subcommands ─────────────────────────────────────────────────
    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Cluster deployment operations (launch, setup, sync, build-sif, status, train).",
    )
    cluster_sub = cluster_parser.add_subparsers(
        dest="cluster_command",
        required=True,
    )

    # polar cluster launch
    launch_p = cluster_sub.add_parser("launch", help="Sync code and submit a cluster job.")
    launch_p.add_argument("-c", "--config", required=True, help="Path to cluster.yaml")
    launch_p.add_argument("--example", default=None, help="Override task.example")
    launch_p.add_argument("--harness", default=None, help="Override task.harness")
    launch_p.add_argument("--model", default=None, help="Override model.name")
    launch_p.add_argument("--nodes", type=int, default=None, help="Override resources.nodes")
    launch_p.add_argument("--gpus", type=int, default=None, help="Override resources.gpus_per_node")
    launch_p.add_argument("--time", default=None, help="Override resources.time (HH:MM:SS)")
    launch_p.add_argument("--num-rollouts", type=int, default=None)
    launch_p.add_argument("--timeout-seconds", type=float, default=None)
    launch_p.add_argument(
        "--instance-id", action="append", default=None,
        help="SWE-Gym instance ID (repeatable; defaults to sample 10)",
    )
    launch_p.add_argument("--no-sync", action="store_true", help="Skip rsync to cluster")
    launch_p.add_argument("--dry-run", action="store_true", help="Print sbatch command only")

    # polar cluster setup
    setup_p = cluster_sub.add_parser("setup", help="One-time cluster environment setup.")
    setup_p.add_argument("-c", "--config", required=True, help="Path to cluster.yaml")

    # polar cluster status
    cstatus_p = cluster_sub.add_parser("status", help="Check SLURM job status.")
    cstatus_p.add_argument("-c", "--config", required=True, help="Path to cluster.yaml")
    cstatus_p.add_argument("--job-id", default=None, help="Specific job ID to query")

    # polar cluster sync
    sync_p = cluster_sub.add_parser("sync", help="Sync code/results from cluster.")
    sync_p.add_argument("-c", "--config", required=True, help="Path to cluster.yaml")
    sync_p.add_argument("--job-id", default=None, help="Sync specific job results")
    sync_p.add_argument("--code-only", action="store_true")
    sync_p.add_argument("--results-only", action="store_true")
    sync_p.add_argument("--dry-run", action="store_true")

    # polar cluster build-sif
    sif_p = cluster_sub.add_parser("build-sif", help="Build Apptainer SIF images.")
    sif_p.add_argument("-c", "--config", required=True, help="Path to cluster.yaml")
    sif_p.add_argument("--example", required=True, help="Example name (calculator, swegym, swebench_verified, train)")
    sif_p.add_argument("--harness", default=None, help="Comma-separated harness names (required except for --example train)")
    sif_p.add_argument("--force", action="store_true", help="Rebuild even if SIF exists")
    sif_p.add_argument(
        "--instance-id", action="append", default=None,
        help="SWE-Gym instance ID (repeatable; defaults to sample 10)",
    )

    # polar cluster serve
    serve_p = cluster_sub.add_parser("serve", help="Start services (vLLM + rollout + gateway).")
    serve_p.add_argument("-c", "--config", required=True, help="Path to cluster.yaml")
    serve_p.add_argument("--model", default=None, help="Override model.name")
    serve_p.add_argument("--nodes", type=int, default=None, help="Override resources.nodes")
    serve_p.add_argument("--gpus", type=int, default=None, help="Override resources.gpus_per_node")
    serve_p.add_argument("--time", default=None, help="Override resources.time (HH:MM:SS)")
    serve_p.add_argument("--no-sync", action="store_true", help="Skip rsync to cluster")
    serve_p.add_argument("--no-wait", action="store_true", help="Don't wait for services to be ready")
    serve_p.add_argument("--wait-timeout", type=int, default=600, help="Seconds to wait for readiness (default: 600)")
    serve_p.add_argument("--dry-run", action="store_true", help="Print sbatch command only")

    # polar cluster submit-task
    submit_task_p = cluster_sub.add_parser("submit-task", help="Submit tasks to a running serve job.")
    submit_task_p.add_argument("-c", "--config", required=True, help="Path to cluster.yaml")
    submit_task_p.add_argument("--job-id", required=True, help="SLURM job ID of the serve job")
    submit_task_p.add_argument("--example", default=None, help="Override task.example")
    submit_task_p.add_argument("--harness", default=None, help="Override task.harness")
    submit_task_p.add_argument("--num-rollouts", type=int, default=None)
    submit_task_p.add_argument("--timeout-seconds", type=float, default=None)
    submit_task_p.add_argument(
        "--instance-id", action="append", default=None,
        help="SWE-Gym instance ID (repeatable; defaults to sample 10)",
    )

    # polar cluster train
    train_p = cluster_sub.add_parser("train", help="Submit a distributed RL training job.")
    train_p.add_argument("-c", "--config", required=True, help="Path to cluster.yaml")
    train_p.add_argument("--polar-config", default=None, help="Path to polar_config.yaml (bridge config)")
    train_p.add_argument("--prompt-data", default=None, help="Path to JSONL training data")
    train_p.add_argument("--hf-checkpoint", default=None, help="HuggingFace model checkpoint")
    train_p.add_argument("--num-rollouts", type=int, default=None, help="Number of training steps")
    train_p.add_argument("--rollout-batch-size", type=int, default=None)
    train_p.add_argument("--n-samples-per-prompt", type=int, default=None)
    train_p.add_argument("--global-batch-size", type=int, default=None)
    train_p.add_argument("--actor-gpus", type=int, default=None)
    train_p.add_argument("--rollout-gpus", type=int, default=None)
    train_p.add_argument("--tp-size", type=int, default=None)
    train_p.add_argument("--nodes", type=int, default=None, help="Override resources.nodes")
    train_p.add_argument("--gpus", type=int, default=None, help="Override resources.gpus_per_node")
    train_p.add_argument("--time", default=None, help="Override resources.time (HH:MM:SS)")
    train_p.add_argument("--no-sync", action="store_true", help="Skip rsync to cluster")
    train_p.add_argument("--no-wait", action="store_true", help="Don't wait for training to complete")
    train_p.add_argument("--wait-timeout", type=int, default=3600, help="Seconds to wait (default: 3600)")
    train_p.add_argument("--dry-run", action="store_true", help="Print sbatch command only")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "serve_rollout":
            serve_rollout(args.config)
            return 0
        if args.command == "serve_gateway":
            serve_gateway(args.config, node_id=args.node_id)
            return 0
        if args.command == "submit":
            return _handle_submit(args)
        if args.command == "status":
            return _handle_status(args)
        if args.command == "cluster":
            return _handle_cluster(args)
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.strip()
        if body:
            print(
                f"error: rollout server returned {exc.response.status_code}: {body}",
                file=sys.stderr,
            )
        else:
            print(
                f"error: rollout server returned {exc.response.status_code}",
                file=sys.stderr,
            )
        return 1
    except httpx.HTTPError as exc:
        print(f"error: could not reach the rollout service: {exc}", file=sys.stderr)
        return 1
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"error: command failed with exit code {exc.returncode}", file=sys.stderr)
        if exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def _handle_cluster(args: argparse.Namespace) -> int:
    # Lazy imports — avoid loading cluster modules for non-cluster commands.
    from polar.cluster.config import ClusterConfig
    from polar.cluster.backend import get_backend

    config = ClusterConfig.load(args.config)
    overrides = _build_cluster_overrides(args)
    if overrides:
        config = config.apply_overrides(overrides)

    repo_root = Path.cwd()
    backend = get_backend(config)

    cmd = args.cluster_command
    if cmd == "launch":
        backend.launch(repo_root, dry_run=args.dry_run, no_sync=args.no_sync)
        return 0
    if cmd == "setup":
        backend.setup(repo_root)
        return 0
    if cmd == "status":
        result = backend.status(job_id=args.job_id)
        jobs = result.get("jobs", [])
        if not jobs:
            print("No jobs found.")
        else:
            print(f"{'JOB_ID':<12} {'NAME':<30} {'STATE':<12} {'TIME':<10} {'NODES'}")
            for j in jobs:
                print(f"{j.get('job_id',''):<12} {j.get('name',''):<30} {j.get('state',''):<12} {j.get('time',''):<10} {j.get('nodes','')}")
        return 0
    if cmd == "sync":
        backend.sync(
            repo_root,
            job_id=args.job_id,
            code_only=args.code_only,
            results_only=args.results_only,
            dry_run=args.dry_run,
        )
        return 0
    if cmd == "build-sif":
        if args.example != "train" and not args.harness:
            print("error: --harness is required for non-train examples", file=sys.stderr)
            return 1
        harnesses = [h.strip() for h in args.harness.split(",")] if args.harness else []
        results = backend.build_sif(
            repo_root, args.example, harnesses,
            force=args.force,
            instance_ids=getattr(args, "instance_id", None),
        )
        for key, sif_path in results.items():
            print(f"  {key}: {sif_path}")
        return 0
    if cmd == "serve":
        result = backend.serve(
            repo_root,
            dry_run=args.dry_run,
            no_sync=args.no_sync,
            wait=not args.no_wait,
            wait_timeout=args.wait_timeout,
        )
        if result:
            print(f"\n[cluster] Services ready.")
            print(f"[cluster] Job ID: {result['job_id']}")
            print(f"[cluster] Topology: {result['topology']}")
            print(f"\n[cluster] Submit tasks with:")
            print(f"  python -m polar.cli cluster submit-task -c {args.config} \\")
            print(f"      --job-id {result['job_id']} --example calculator --harness opencode")
        return 0
    if cmd == "submit-task":
        return backend.submit_task(
            repo_root,
            job_id=args.job_id,
            example=getattr(args, "example", None),
            harness=getattr(args, "harness", None),
        )
    if cmd == "train":
        result = backend.train(
            repo_root,
            dry_run=args.dry_run,
            no_sync=args.no_sync,
            wait=not args.no_wait,
            wait_timeout=args.wait_timeout,
        )
        if result:
            print(f"\n[cluster] Training job info:")
            for k, v in result.items():
                print(f"  {k}: {v}")
        return 0

    print(f"Unknown cluster command: {cmd}", file=sys.stderr)
    return 2


def _build_cluster_overrides(args: argparse.Namespace) -> dict:
    """Extract CLI flag overrides into a nested dict for ``ClusterConfig.apply_overrides``."""
    overrides: dict = {}
    if getattr(args, "example", None):
        overrides.setdefault("task", {})["example"] = args.example
    if getattr(args, "harness", None) and args.cluster_command == "launch":
        overrides.setdefault("task", {})["harness"] = args.harness
    if getattr(args, "model", None):
        overrides.setdefault("model", {})["name"] = args.model
    if getattr(args, "nodes", None) is not None:
        overrides.setdefault("resources", {})["nodes"] = args.nodes
    if getattr(args, "gpus", None) is not None:
        overrides.setdefault("resources", {})["gpus_per_node"] = args.gpus
    if getattr(args, "time", None):
        overrides.setdefault("resources", {})["time"] = args.time
    if getattr(args, "num_rollouts", None) is not None:
        if getattr(args, "cluster_command", None) == "train":
            overrides.setdefault("train", {})["num_rollouts"] = args.num_rollouts
        else:
            overrides.setdefault("task", {})["num_rollouts"] = args.num_rollouts
    if getattr(args, "timeout_seconds", None) is not None:
        overrides.setdefault("task", {})["timeout_seconds"] = args.timeout_seconds
    if getattr(args, "instance_id", None):
        overrides.setdefault("task", {})["instance_ids"] = args.instance_id
    # Train-specific overrides
    if getattr(args, "polar_config", None):
        overrides.setdefault("train", {})["polar_config"] = args.polar_config
    if getattr(args, "prompt_data", None):
        overrides.setdefault("train", {})["prompt_data"] = args.prompt_data
    if getattr(args, "hf_checkpoint", None):
        overrides.setdefault("train", {})["hf_checkpoint"] = args.hf_checkpoint
    if getattr(args, "rollout_batch_size", None) is not None:
        overrides.setdefault("train", {})["rollout_batch_size"] = args.rollout_batch_size
    if getattr(args, "n_samples_per_prompt", None) is not None:
        overrides.setdefault("train", {})["n_samples_per_prompt"] = args.n_samples_per_prompt
    if getattr(args, "global_batch_size", None) is not None:
        overrides.setdefault("train", {})["global_batch_size"] = args.global_batch_size
    if getattr(args, "actor_gpus", None) is not None:
        overrides.setdefault("train", {})["actor_gpus"] = args.actor_gpus
    if getattr(args, "rollout_gpus", None) is not None:
        overrides.setdefault("train", {})["rollout_gpus"] = args.rollout_gpus
    if getattr(args, "tp_size", None) is not None:
        overrides.setdefault("train", {})["tp_size"] = args.tp_size
    return overrides


def _handle_submit(args: argparse.Namespace) -> int:
    rollout_url = _resolve_rollout_url(args.config, args.rollout_url)
    payload = _load_structured_file(args.task_file)
    timeout = httpx.Timeout(None, connect=30.0)
    with httpx.Client(base_url=rollout_url, timeout=timeout) as client:
        response = client.post("/rollout/task", json=payload)
        response.raise_for_status()
        result = response.json()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    sessions = result.get("results") or []
    completed = sum(1 for session in sessions if session.get("status") == "COMPLETED")
    rewarded = sum(
        1
        for session in sessions
        if ((session.get("trajectory") or {}).get("traces") or [{}])[-1].get("reward") == 1.0
    )
    print(f"Task: {result.get('task_id')}")
    print(f"Status: {result.get('status')}")
    print(f"Completed sessions: {completed}/{len(sessions)}")
    print(f"Reward=1.0 sessions: {rewarded}")
    result_paths = result.get("result_paths") or []
    if result_paths:
        print("Result paths:")
        for path in result_paths:
            print(f"  {path}")
    return 0


def _handle_status(args: argparse.Namespace) -> int:
    rollout_url = _resolve_rollout_url(args.config, args.rollout_url)
    timeout = httpx.Timeout(10.0, connect=5.0)
    with httpx.Client(base_url=rollout_url, timeout=timeout) as client:
        health = client.get("/health").json()
        status = client.get("/rollout/status").json()

    gateway_health_by_node: dict[str, dict[str, Any]] = {}
    if args.config:
        topology = TopologyConfig.load(args.config)
        for node in topology.gateway.nodes:
            try:
                with httpx.Client(base_url=node.public_url, timeout=timeout) as client:
                    gateway_health_by_node[node.id] = client.get("/health").json()
            except httpx.HTTPError:
                gateway_health_by_node[node.id] = {
                    "status": "unreachable",
                    "gateway_url": node.public_url,
                }

    if args.json:
        print(
            json.dumps(
                {
                    "rollout_url": rollout_url,
                    "health": health,
                    "status": status,
                    "gateway_health": gateway_health_by_node,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    topology = TopologyConfig.load(args.config) if args.config else None
    node_rows = ((status.get("nodes") or {}).get("nodes")) or []

    overview_lines = [
        f"Rollout: {_endpoint_label('rollout', rollout_url)}",
        f"Health: {str(health.get('status', 'unknown')).upper()}",
        f"Registered Nodes: {health.get('nodes', 0)}",
        f"Running Sessions: {_format_count((status.get('pipeline') or {}).get('pending_sessions', 0), 'session')}",
    ]
    print(_boxed_section("Polar Status", overview_lines))

    if topology is not None:
        print()
        print(_boxed_section("Topology", _build_topology_lines(topology, node_rows)))

    if node_rows:
        print()
        print(_boxed_section("Live Load", _build_live_load_lines(node_rows, gateway_health_by_node)))
    return 0


def _build_topology_lines(
    topology: TopologyConfig,
    node_rows: list[dict[str, Any]],
) -> list[str]:
    node_status_by_id = {str(node.get("node_id")): node for node in node_rows}
    lines = [f"{_endpoint_label('rollout', topology.rollout.public_url)}"]
    last_index = len(topology.gateway.nodes) - 1
    for index, node in enumerate(topology.gateway.nodes):
        branch = "└──" if index == last_index else "├──"
        status = node_status_by_id.get(node.id, {})
        badge = _node_badge(
            healthy=bool(status.get("healthy", False)),
            draining=bool(status.get("draining", False)),
            reachable=bool(status),
        )
        gateway_label = _gateway_label(node.id)
        inference_label = _endpoint_label("inference", node.sglang_base_url)
        lines.append(f"{branch} {gateway_label} [{badge}] ── {inference_label}")
    return lines


def _build_live_load_lines(
    node_rows: list[dict[str, Any]],
    gateway_health_by_node: dict[str, dict[str, Any]],
) -> list[str]:
    gateway_labels = {
        str(node.get("node_id")): _gateway_label(str(node.get("node_id")))
        for node in node_rows
    }
    name_width = max((len(label) for label in gateway_labels.values()), default=0)
    lines: list[str] = []
    for node in node_rows:
        node_id = str(node.get("node_id"))
        metrics = node.get("metrics") or {}
        node_health = gateway_health_by_node.get(node_id, {})
        status_counts = node_health.get("active_status_counts") or {}
        compact_status = ""
        if status_counts:
            compact = ", ".join(
                f"{name.lower()}={count}"
                for name, count in sorted(status_counts.items())
            )
            compact_status = f"  statuses[{compact}]"
        lines.append(
            f"{gateway_labels[node_id].ljust(name_width)}  "
            f"init {metrics.get('init_inflight', 0)}/{node.get('max_init_workers')}  "
            f"runtime_pool {metrics.get('ready_depth', 0)}  "
            f"run {metrics.get('run_inflight', 0)}/{node.get('max_run_workers')}  "
            f"postrun {metrics.get('postrun_inflight', 0)}/{node.get('max_postrun_workers')}"
            f"{compact_status}"
        )
    return lines


def _boxed_section(title: str, lines: list[str]) -> str:
    content = lines or ["(empty)"]
    width = max(len(title), *(len(line) for line in content))
    top = f"┌─ {title} " + "─" * max(0, width - len(title) - 1) + "┐"
    body = [f"│ {line.ljust(width)} │" for line in content]
    bottom = "└" + "─" * (width + 2) + "┘"
    return "\n".join([top, *body, bottom])


def _endpoint_label(prefix: str, url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or parsed.netloc or url
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{prefix}_{host}{port}"


def _format_count(count: object, noun: str) -> str:
    value = int(count or 0)
    suffix = noun if value == 1 else f"{noun}s"
    return f"{value} {suffix}"


def _gateway_label(node_id: str) -> str:
    return f"gateway_{node_id}"


def _node_badge(*, healthy: bool, draining: bool, reachable: bool) -> str:
    if not reachable:
        return "DOWN"
    if draining:
        return "DRAIN"
    if healthy:
        return "UP"
    return "DOWN"


def _resolve_rollout_url(config_path: str | None, explicit_url: str | None) -> str:
    if explicit_url:
        return explicit_url.rstrip("/")
    if config_path:
        topology = TopologyConfig.load(config_path)
        return topology.rollout.public_url
    return "http://127.0.0.1:8080"


def _load_structured_file(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    raw = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw)
    else:
        payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Task file {path} must contain a top-level mapping")
    return payload


if __name__ == "__main__":
    sys.exit(main())
