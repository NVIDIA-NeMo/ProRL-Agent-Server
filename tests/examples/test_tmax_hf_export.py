from __future__ import annotations

import json
import os
import pickle
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = ROOT / "examples" / "tmax_slime_grpo" / "export_hf_checkpoint.sh"


def _write_distcp_metadata(model_dir: Path, *shard_names: str) -> None:
    payload = {
        "storage_data": [
            {"relative_path": name, "offset": 0, "length": len(b"weights")}
            for name in shard_names
        ]
    }
    (model_dir / ".metadata").write_bytes(pickle.dumps(payload, protocol=4))


def _write_safetensors(
    path: Path, tensor_name: str, *, dtype: str = "BF16"
) -> None:
    element_size = {"BF16": 2, "F32": 4}[dtype]
    header = {
        tensor_name: {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [0, element_size],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"\0" * element_size
    )


def test_hf_export_worker_supports_nested_qwen_text_config(tmp_path: Path) -> None:
    input_dir = tmp_path / "checkpoint" / "iter_0000047"
    input_dir.mkdir(parents=True)
    (input_dir / "common.pt").write_bytes(b"common")
    _write_distcp_metadata(input_dir, "__0_0.distcp")
    (input_dir / "__0_0.distcp").write_bytes(b"weights")

    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"model_type": "qwen3_5_text", "vocab_size": 248320},
            }
        )
    )
    tensor_name = "model.embed_tokens.weight"
    lm_head_name = "lm_head.weight"
    shard_name = "model-00001-of-00002.safetensors"
    lm_head_shard_name = "model-00002-of-00002.safetensors"
    (origin / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    tensor_name: shard_name,
                    lm_head_name: lm_head_shard_name,
                }
            }
        )
    )
    _write_safetensors(origin / shard_name, tensor_name)
    _write_safetensors(
        origin / lm_head_shard_name, lm_head_name, dtype="F32"
    )
    for asset in ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
        (origin / asset).write_text("{}" if asset.endswith(".json") else "{{ messages }}")

    slime_dir = tmp_path / "slime"
    tools_dir = slime_dir / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "convert_torch_dist_to_hf.py").write_text(
        """\
import argparse
import shutil

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--origin-hf-dir", required=True)
args, _unknown = parser.parse_known_args()
shutil.copytree(args.origin_hf_dir, args.output_dir)
"""
    )

    output_dir = tmp_path / "output" / "iter_0000047-bf16"
    env = os.environ.copy()
    env.update(
        RUN_ID="nested-qwen",
        POLAR_DATA_ROOT=str(tmp_path / "data"),
        TMAX_HF_EXPORT_CKPT_ROOT=str(input_dir.parent),
        TMAX_HF_EXPORT_OUTPUT_ROOT=str(output_dir.parent),
        SLIME_DIR=str(slime_dir),
        TMAX_HF_EXPORT_PYTHON=sys.executable,
        MEGATRON_DIR=str(tmp_path / "megatron"),
        TMAX_HF_EXPORT_ORIGIN=str(origin),
    )

    result = subprocess.run(
        ["bash", str(EXPORT_SCRIPT), "--worker", str(input_dir), str(output_dir), "47"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "origin model=qwen3_5 vocab_size=248320" in result.stdout
    manifest = json.loads((output_dir / ".export_complete.json").read_text())
    assert manifest["model_type"] == "qwen3_5"
    assert manifest["vocab_size"] == 248320
    assert manifest["dtype_counts"] == {"BF16": 1, "F32": 1}
