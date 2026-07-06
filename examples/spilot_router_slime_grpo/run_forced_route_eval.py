#!/usr/bin/env python3
"""Run the paired SPilot pool evaluation in a services-only allocation.

This launcher deliberately starts only Polar rollout, one Polar gateway, and
the sandbox UDS bridge.  It never starts Ray, Slime, SGLang, or a Router actor.
The benchmark runs in the same allocation and process tree, so its loopback
URLs and control-plane credential never need to cross a cluster boundary.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import time
from types import FrameType
from typing import Any
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


class LauncherError(RuntimeError):
    """Expected launcher failure with a concise user-facing message."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--service-dir", type=Path)
    parser.add_argument("--run-id", required=True)
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
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--forward-seed-to-pool", action="store_true")
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
    for name in ("max_tasks", "replicates", "max_concurrency", "agent_timeout_seconds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.poll_seconds <= 0 or args.request_timeout <= 0:
        parser.error("poll and request timeouts must be positive")
    for name in ("rollout_port", "gateway_port"):
        value = getattr(args, name)
        if value is not None and not 1 <= value <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be between 1 and 65535")
    if args.rollout_port is not None and args.rollout_port == args.gateway_port:
        parser.error("rollout and gateway ports must differ")
    if not args.dry_run and not os.environ.get("SLURM_JOB_ID"):
        parser.error("run inside a Slurm allocation (or use --dry-run)")
    return args


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve()


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


def preflight_eval_slice(path: Path, *, start_index: int, max_tasks: int) -> bool:
    """Validate the paid slice and return whether any row requests internet."""

    if not path.is_file():
        raise LauncherError(f"evaluation data does not exist: {path}")
    selected = 0
    needs_internet = False
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
            needs_internet = needs_internet or bool(metadata.get("allow_internet", True))
            selected += 1
    if selected != max_tasks:
        raise LauncherError(
            f"requested {max_tasks} tasks at start index {start_index}, found {selected}"
        )
    return needs_internet


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
    service_dir: Path,
    pool_base_url: str,
    max_concurrency: int,
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
                        "base_url": "http://127.0.0.1:9",
                    },
                    "model_pool": [
                        {
                            "alias": "pool/qwen3.6-27b",
                            "model": "nvidia/qwen/qwen3.6-27b",
                            "base_url": pool_base_url,
                            "api_key_env": NVIDIA_KEY_ENV,
                            "max_concurrency": max_concurrency,
                        },
                        {
                            "alias": "pool/gpt-5.5",
                            "model": "openai/openai/gpt-5.5",
                            "base_url": pool_base_url,
                            "api_key_env": NVIDIA_KEY_ENV,
                            "max_concurrency": max_concurrency,
                        },
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
) -> dict[str, Any]:
    gateway_uds_dir = uds_root / "gateway"
    proxy_uds_dir = uds_root / "proxy"
    runtime_dir = data_root / "mini_swe_agent_runtime"
    values = {
        "POLAR_ROLLOUT_URL": f"http://127.0.0.1:{rollout_port}",
        "POLAR_GATEWAY_URL": f"http://127.0.0.1:{gateway_port}",
        "POLAR_MAX_ASYNC_LEVEL": "1",
        "POLAR_FULLY_ASYNC": "false",
        "POLAR_REQUEST_TIMEOUT": str(agent_timeout_seconds + 1800),
        "POLAR_TASK_TIMEOUT_FLOOR_SECONDS": str(agent_timeout_seconds + 1200),
        "TMAX_TRAIN_AGENT_TIMEOUT_SECONDS": str(agent_timeout_seconds),
        "POLAR_CALLBACK_HOST": "127.0.0.1",
        "POLAR_MIN_COMPLETE_ACCEPT_FRACTION": "0",
        "POLAR_EARLY_STOP_GRACE_SESSIONS": "0",
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
        "POLAR_SANDBOX_HTTP_PROXY_UDS": (
            "/polar/proxy/proxy.sock" if proxy_url else ""
        ),
        "POLAR_SANDBOX_HTTP_PROXY_PORT": "28100",
        "POLAR_APT_HTTP_SOURCE_POLICY": "https",
        "http_proxy": proxy_url,
        "https_proxy": proxy_url,
        "HTTP_PROXY": proxy_url,
        "HTTPS_PROXY": proxy_url,
        "no_proxy": "127.0.0.1,localhost",
        "NO_PROXY": "127.0.0.1,localhost",
        "POLAR_GATEWAY_UDS_DIR": str(gateway_uds_dir),
        "POLAR_AGENT_RUNTIME_VOLUME": (
            f"        - {runtime_dir}:/opt/polar-mini-swe-agent:ro"
        ),
        "POLAR_INTERNET_RUNTIME_VOLUME": (
            "    internet_volumes:\n"
            f"      - \"{proxy_uds_dir}:/polar/proxy:ro\""
            if proxy_url
            else ""
        ),
        "POLAR_AGENT_MODEL_NAME": "eval-only/forced-route-no-actor",
        "POLAR_AGENT_TEMPERATURE": "1.0",
        "POLAR_AGENT_TOP_P": "1.0",
    }
    template = (EXAMPLE_DIR / "polar_config.yaml").read_text(encoding="utf-8")
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
    return document


def _write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _http_json(url: str, *, timeout: float = 2.0) -> Any:
    opener = build_opener(ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return json.load(response)


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


def scoped_environment(*, control_token: str | None = None, nvidia_key: str | None = None):
    environment = dict(os.environ)
    environment.pop(CONTROL_TOKEN_ENV, None)
    environment.pop(NVIDIA_KEY_ENV, None)
    environment.pop("NVIDIA_API_KEY", None)
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPO_ROOT / "src") + (
        f":{current_pythonpath}" if current_pythonpath else ""
    )
    if control_token is not None:
        environment[CONTROL_TOKEN_ENV] = control_token
    if nvidia_key is not None:
        environment[NVIDIA_KEY_ENV] = nvidia_key
    return environment


def terminate_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10.0
    for process in reversed(processes):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
    for process in reversed(processes):
        if process.poll() is None:
            process.wait(timeout=5.0)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = _absolute(args.data)
    output_dir = _absolute(args.output_dir)
    service_dir = _absolute(
        args.service_dir
        if args.service_dir is not None
        else output_dir.with_name(f"{output_dir.name}.service")
    )
    if output_dir.exists():
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
    for path in (
        data_root / "agent_cli" / "opt_node",
        data_root / "mini_swe_agent_runtime",
        data_root / "tmax-15k-sif",
    ):
        if not path.is_dir():
            raise LauncherError(f"required runtime directory does not exist: {path}")
    needs_internet = preflight_eval_slice(
        data,
        start_index=args.start_index,
        max_tasks=args.max_tasks,
    )
    proxy_url = (
        os.environ.get("http_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTPS_PROXY")
        or ""
    ).strip()
    if needs_internet and not proxy_url:
        raise LauncherError("selected tasks allow internet but no HTTP proxy is configured")
    proxy_target = parse_proxy_target(proxy_url) if proxy_url else ""

    service_dir.parent.mkdir(parents=True, exist_ok=True)
    service_dir.mkdir(mode=0o700)
    if args.rollout_port is None and args.gateway_port is None:
        if args.dry_run:
            rollout_port, gateway_port = 18080, 18100
        else:
            rollout_port, gateway_port = reserve_loopback_ports()
    elif args.rollout_port is None:
        rollout_port = reserve_loopback_ports(1)[0]
        gateway_port = args.gateway_port
    elif args.gateway_port is None:
        rollout_port = args.rollout_port
        gateway_port = reserve_loopback_ports(1)[0]
    else:
        rollout_port, gateway_port = args.rollout_port, args.gateway_port
    if rollout_port == gateway_port:
        raise LauncherError("rollout and gateway ports must differ")
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
        service_dir=service_dir,
        pool_base_url=args.pool_base_url,
        max_concurrency=args.max_concurrency,
    )
    TopologyConfig.model_validate(topology)
    polar_config = render_polar_config(
        data_root=data_root,
        rollout_port=rollout_port,
        gateway_port=gateway_port,
        uds_root=uds_root,
        proxy_url=proxy_url,
        agent_timeout_seconds=args.agent_timeout_seconds,
    )
    topology_path = service_dir / "topology.yaml"
    polar_config_path = service_dir / "polar_config.yaml"
    _write_yaml(topology_path, topology)
    _write_yaml(polar_config_path, polar_config)
    evaluator_command = [
        sys.executable,
        str(EXAMPLE_DIR / "forced_route_eval.py"),
        "--i-understand-eval-only",
        "--data",
        str(data),
        "--polar-config",
        str(polar_config_path),
        "--rollout-url",
        f"http://127.0.0.1:{rollout_port}",
        "--output-dir",
        str(output_dir),
        "--run-id",
        args.run_id,
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
    ]
    if args.forward_seed_to_pool:
        evaluator_command.append("--forward-seed-to-pool")
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
            "topology_path": str(topology_path),
            "polar_config_path": str(polar_config_path),
            "output_dir": str(output_dir),
            "control_token_persisted": False,
            "model_credential_persisted": False,
            "evaluator_command": evaluator_command,
        },
    )
    if args.dry_run:
        for directory in (gateway_uds.parent, proxy_uds.parent, uds_root):
            directory.rmdir()
        print(f"Rendered services-only plan: {service_dir}")
        return 0

    nvidia_key = (
        os.environ.get(NVIDIA_KEY_ENV, "").strip()
        or os.environ.get("NVIDIA_API_KEY", "").strip()
    )
    if not nvidia_key:
        raise LauncherError(f"{NVIDIA_KEY_ENV} or NVIDIA_API_KEY is required")
    control_token = os.environ.get(CONTROL_TOKEN_ENV, "").strip()
    if not control_token:
        control_token = secrets.token_urlsafe(48)
    if not CONTROL_TOKEN_RE.fullmatch(control_token):
        raise LauncherError(f"{CONTROL_TOKEN_ENV} must be a 32-128 character opaque token")

    processes: list[subprocess.Popen[bytes]] = []
    previous_handlers: dict[int, Any] = {}

    def interrupted(signum: int, _frame: FrameType | None) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, interrupted)

    try:
        with ExitStack() as stack:
            rollout_log = stack.enter_context((service_dir / "rollout.log").open("wb"))
            gateway_log = stack.enter_context((service_dir / "gateway.log").open("wb"))
            tunnel_log = stack.enter_context((service_dir / "uds_tunnel.log").open("wb"))
            rollout = subprocess.Popen(
                [sys.executable, "-m", "polar.cli", "serve_rollout", "-c", str(topology_path)],
                stdout=rollout_log,
                stderr=subprocess.STDOUT,
                env=scoped_environment(control_token=control_token),
            )
            processes.append(rollout)
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
                env=scoped_environment(control_token=control_token, nvidia_key=nvidia_key),
            )
            processes.append(gateway)
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
                    and document[0].get("gateway_url")
                    == f"http://127.0.0.1:{gateway_port}"
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
                env=scoped_environment(),
            )
            processes.append(tunnel)
            wait_uds(tunnel, ready_file=ready_file, sockets=expected_sockets)

            print(
                "Services ready on allocation-local loopback; starting paired forced-route eval",
                flush=True,
            )
            evaluator = subprocess.Popen(
                evaluator_command,
                env=scoped_environment(control_token=control_token),
            )
            processes.append(evaluator)
            while evaluator.poll() is None:
                for name, process in (("rollout", rollout), ("gateway", gateway), ("UDS", tunnel)):
                    if process.poll() is not None:
                        raise LauncherError(
                            f"{name} service exited during benchmark (rc={process.returncode})"
                        )
                time.sleep(1.0)
            return int(evaluator.returncode or 0)
    finally:
        terminate_processes(processes)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        try:
            for path in (gateway_uds, proxy_uds, ready_file):
                path.unlink(missing_ok=True)
            for directory in (gateway_uds.parent, proxy_uds.parent, uds_root):
                directory.rmdir()
        except OSError:
            pass


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
