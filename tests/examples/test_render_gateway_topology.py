from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from polar.config import TopologyConfig


ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / "examples" / "swegym_slime_grpo" / "render_gateway_topology.py"


def _prototype() -> dict:
    return {
        "rollout": {
            "host": "0.0.0.0",
            "port": 18080,
            "public_url": "http://10.0.0.1:18080",
        },
        "gateway": {
            "completion_persistence": {
                "queue_size": 8192,
                "write_workers": 4,
            },
            "nodes": [
                {
                    "id": "prototype",
                    "host": "0.0.0.0",
                    "port": 18100,
                    "public_url": "http://10.0.0.1:18100",
                    "max_init_workers": 12,
                    "max_run_workers": 96,
                    "max_postrun_workers": 24,
                }
            ],
        },
    }


def test_renderer_atomically_expands_four_slurm_gateways(tmp_path: Path) -> None:
    source = tmp_path / "prototype.yaml"
    output = tmp_path / "topology.yaml"
    source.write_text(yaml.safe_dump(_prototype(), sort_keys=False))
    command = [
        sys.executable,
        str(RENDERER),
        "--input",
        str(source),
        "--output",
        str(output),
    ]
    for host in ("node-a", "node-b", "node-c", "node-d"):
        command.extend(("--gateway-host", host))

    result = subprocess.run(command, text=True, capture_output=True, check=True)

    topology = TopologyConfig.load(output)
    assert [node.id for node in topology.gateway.nodes] == [
        "slurm-rank-0",
        "slurm-rank-1",
        "slurm-rank-2",
        "slurm-rank-3",
    ]
    assert [node.public_url for node in topology.gateway.nodes] == [
        "http://node-a:18100",
        "http://node-b:18100",
        "http://node-c:18100",
        "http://node-d:18100",
    ]
    assert all(node.max_init_workers == 12 for node in topology.gateway.nodes)
    assert all(node.max_run_workers == 96 for node in topology.gateway.nodes)
    assert all(node.max_postrun_workers == 24 for node in topology.gateway.nodes)
    assert result.stdout.count("[polar topology]") == 4
    assert "node_id=slurm-rank-1 host=node-b bind_host=0.0.0.0" in result.stdout
    assert "quotas=12/96/24 completion=8192/4" in result.stdout
    assert not list(tmp_path.glob(".topology.yaml.*.tmp"))


def test_renderer_rejects_duplicate_gateway_hosts(tmp_path: Path) -> None:
    source = tmp_path / "prototype.yaml"
    output = tmp_path / "topology.yaml"
    source.write_text(yaml.safe_dump(_prototype(), sort_keys=False))

    result = subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--input",
            str(source),
            "--output",
            str(output),
            "--gateway-host",
            "node-a",
            "--gateway-host",
            " node-a ",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "gateway hosts must be unique" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("host", ["", "   "])
def test_renderer_rejects_blank_gateway_host(tmp_path: Path, host: str) -> None:
    source = tmp_path / "prototype.yaml"
    output = tmp_path / "topology.yaml"
    source.write_text(yaml.safe_dump(_prototype(), sort_keys=False))

    result = subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--input",
            str(source),
            "--output",
            str(output),
            "--gateway-host",
            host,
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "gateway hosts must be non-empty" in result.stderr
