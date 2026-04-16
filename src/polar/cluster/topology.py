"""Generate topology.yaml for a Polar SLURM job.

Called inside the SLURM job after hostname discovery. Reads
``SLURM_JOB_NODELIST`` and environment variables to produce a complete
``topology.yaml`` consumed by ``polar serve_rollout`` / ``polar serve_gateway``.

Usage (from sbatch script)::

    python -m polar.cluster.topology --output /path/to/topology.yaml
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

import yaml


def discover_hostnames() -> list[str]:
    """Return the SLURM-allocated hostnames, or ``[localhost]`` for local testing."""
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        hostname = socket.gethostname()
        print(f"[topology] No SLURM_JOB_NODELIST; using local hostname: {hostname}")
        return [hostname]

    result = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        capture_output=True,
        text=True,
        check=True,
    )
    hostnames = [h.strip() for h in result.stdout.strip().split("\n") if h.strip()]
    if not hostnames:
        raise RuntimeError(f"scontrol returned no hostnames for {nodelist}")
    print(f"[topology] Discovered {len(hostnames)} node(s): {hostnames}")
    return hostnames


def build_topology(
    hostnames: list[str],
    *,
    vllm_port: int | None = None,
    sglang_base_url: str | None = None,
    rollout_port: int | None = None,
    gateway_base_port: int | None = None,
    model_name: str | None = None,
    default_sif: str | None = None,
    max_init_workers: int | None = None,
    max_run_workers: int | None = None,
    max_postrun_workers: int | None = None,
    ready_buffer_target: int | None = None,
    save_dir: str | None = None,
    vllm_timeout: int | None = None,
) -> dict:
    """Build the topology dict from discovered hostnames and configuration.

    Each parameter falls back to the corresponding environment variable and
    then to a hardcoded default — matching the contract of the old shell-based
    ``generate_topology.py``.
    """
    _vllm_port = vllm_port or int(os.environ.get("VLLM_PORT", "8000"))
    _rollout_port = rollout_port or int(os.environ.get("ROLLOUT_PORT", "8080"))
    _gw_base = gateway_base_port or int(os.environ.get("GATEWAY_BASE_PORT", "8100"))
    _model = model_name or os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-27B")
    _sif = default_sif or os.environ.get("DEFAULT_SIF_IMAGE", "")
    _init = max_init_workers or int(os.environ.get("MAX_INIT_WORKERS", "8"))
    _run = max_run_workers or int(os.environ.get("MAX_RUN_WORKERS", "4"))
    _post = max_postrun_workers or int(os.environ.get("MAX_POSTRUN_WORKERS", "4"))
    _buf = ready_buffer_target or int(os.environ.get("READY_BUFFER_TARGET", "4"))
    _save = save_dir or os.environ.get("SAVE_DIR", "./rollout_results")
    _timeout = vllm_timeout or int(os.environ.get("VLLM_TIMEOUT", "300"))

    vllm_host = hostnames[0]
    rollout_host = hostnames[0]

    gateway_nodes = []
    for i, hostname in enumerate(hostnames):
        port = _gw_base + i
        node: dict = {
            "id": f"node-{i:02d}",
            "host": "0.0.0.0",
            "port": port,
            "public_url": f"http://{hostname}:{port}",
            "max_init_workers": _init,
            "max_run_workers": _run,
            "max_postrun_workers": _post,
            "ready_buffer_target": _buf,
            "model_served": _model,
        }
        if sglang_base_url:
            node["sglang"] = {
                "base_url": sglang_base_url,
                "timeout": _timeout,
            }
        else:
            node["vllm"] = {
                "base_url": f"http://{vllm_host}:{_vllm_port}",
                "timeout": _timeout,
            }
        if _sif:
            runtime_cfg: dict = {
                "backend": "apptainer",
                "image": _sif,
                "network": "host",
            }
            # swe_agent's swerex needs chown inside the container, which
            # requires fakeroot in Apptainer.
            if os.environ.get("RUNTIME_FAKEROOT", "").lower() in ("1", "true", "yes"):
                runtime_cfg["kwargs"] = {"fakeroot": True}
            node["default_runtime"] = runtime_cfg
        gateway_nodes.append(node)

    return {
        "rollout": {
            "host": "0.0.0.0",
            "port": _rollout_port,
            "public_url": f"http://{rollout_host}:{_rollout_port}",
            "save_dir": _save,
            "dispatch_poll_interval_seconds": 1.0,
            "callback_grace_seconds": 10.0,
        },
        "gateway": {
            "heartbeat_interval_seconds": 15,
            "rollout_server_url": f"http://{rollout_host}:{_rollout_port}",
            "nodes": gateway_nodes,
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Polar topology.yaml for SLURM jobs",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for topology.yaml (default: stdout)",
    )
    parser.add_argument(
        "--default-sif",
        default=None,
        help="Default SIF image path for agent containers",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Rollout results save directory",
    )
    parser.add_argument(
        "--sglang-base-url",
        default=None,
        help="SGLang router URL (use sglang instead of vllm backend)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    hostnames = discover_hostnames()
    topology = build_topology(
        hostnames,
        default_sif=args.default_sif or None,
        save_dir=args.save_dir or None,
        sglang_base_url=args.sglang_base_url or None,
    )
    output = yaml.dump(topology, default_flow_style=False, sort_keys=False)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)
        print(f"[topology] Wrote topology to {args.output}")
    else:
        print(output)

    nodes = topology["gateway"]["nodes"]
    print("[topology] Summary:")
    print(f"  Rollout: {topology['rollout']['public_url']}")
    if "vllm" in nodes[0]:
        print(f"  vLLM:    {nodes[0]['vllm']['base_url']}")
    elif "sglang" in nodes[0]:
        print(f"  SGLang:  {nodes[0]['sglang']['base_url']}")
    for node in nodes:
        print(f"  Gateway: {node['id']} @ {node['public_url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
