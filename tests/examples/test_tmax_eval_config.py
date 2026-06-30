from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from slime.utils.eval_config import EvalDatasetConfig

from slime_bridge import rollout as rollout_module
from slime_bridge.config import resolve_polar_slime_config


ROOT = Path(__file__).resolve().parents[2]
TMAX = ROOT / "examples" / "tmax_slime_grpo"


def _clean_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir()
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "USER": os.environ.get("USER", "test"),
        "POLAR_DATA_ROOT": str(tmp_path / "data"),
        "PYTHON": sys.executable,
    }


def _write_eval_row(path: Path, *, step_limit: int | None = None) -> None:
    metadata: dict[str, object] = {
        "agent_timeout": 900.0,
        "timeout_seconds": 1620.0,
    }
    if step_limit is not None:
        metadata["agent_step_limit"] = step_limit
    path.write_text(json.dumps({"prompt": "Fix the task", "metadata": metadata}) + "\n")


def _generate_default_config(tmp_path: Path) -> tuple[dict, Path, Path]:
    primary = tmp_path / "holdout.jsonl"
    external = tmp_path / "terminal-bench.jsonl"
    output = tmp_path / "eval-config.json"
    _write_eval_row(primary)
    # Prove the config-level Terminal-Bench setting replaces stale prepared
    # rows from before the default changed from 50 to 64.
    _write_eval_row(external, step_limit=50)
    env = _clean_env(tmp_path)
    env.update(PRIMARY=str(primary), EXTERNAL=str(external), OUTPUT=str(output))
    script = f"""
source {TMAX / 'env.cwdfw.sh'} >/dev/null
"$PYTHON" {TMAX / 'build_eval_config.py'} \
  --output "$OUTPUT" \
  --primary-name "$TMAX_EVAL_DATASET_NAME" \
  --primary-path "$PRIMARY" \
  --primary-samples "$TMAX_EVAL_SAMPLES_PER_PROMPT" \
  --primary-minimum "$TMAX_EVAL_MIN_VALID_SAMPLES" \
  --primary-temperature "$TMAX_EVAL_TEMPERATURE" \
  --primary-top-p "$TMAX_EVAL_TOP_P" \
  --primary-max-response-len "$TMAX_EVAL_MAX_RESPONSE_LEN" \
  --external-name "$TMAX_EXTERNAL_EVAL_DATASET_NAME" \
  --external-path "$EXTERNAL" \
  --external-samples "$TMAX_EXTERNAL_EVAL_SAMPLES_PER_PROMPT" \
  --external-minimum "$TMAX_EXTERNAL_EVAL_MIN_VALID_SAMPLES" \
  --external-temperature "$TMAX_EXTERNAL_EVAL_TEMPERATURE" \
  --external-top-p "$TMAX_EXTERNAL_EVAL_TOP_P" \
  --external-max-response-len "$TMAX_EXTERNAL_EVAL_MAX_RESPONSE_LEN" \
  --external-agent-step-limit "$TMAX_EXTERNAL_EVAL_AGENT_STEP_LIMIT" \
  --eval-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" \
  --sglang-context-length "$SGLANG_CONTEXT_LENGTH" \
  --model-max-context-length "$TMAX_MODEL_MAX_CONTEXT_LENGTH"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text()), primary, external


def _rollout_args() -> SimpleNamespace:
    return SimpleNamespace(
        polar_rollout_url="http://rollout:8080",
        polar_task_template={
            "timeout_seconds": "{sample.metadata.timeout_seconds}",
            "agent": {
                "harness": "mini_swe_agent",
                "model_name": "Qwen/Qwen3.5-9B",
                "settings": {"step_limit": 20},
            }
        },
        polar_task_id_template="eval-{rollout_id}-{sample.group_index}",
        polar_max_async_level=1,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        update_weights_interval=1,
        polar_min_complete_accept_fraction=0.5,
        polar_early_stop_grace_sessions=0,
        polar_train_agent_timeout=1200,
        hf_checkpoint="tokenizer",
    )


def test_tmax_defaults_generate_dataset_specific_eval_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, _primary_path, _external_path = _generate_default_config(tmp_path)
    primary_raw, external_raw = raw["eval"]["datasets"]

    assert primary_raw["name"] == "tmax_holdout"
    assert primary_raw["temperature"] == 0.2
    assert primary_raw["top_p"] == 1.0
    assert primary_raw["max_response_len"] == 16384
    assert "metadata_overrides" not in primary_raw

    assert external_raw["name"] == "terminal_bench_2_0"
    assert external_raw["n_samples_per_eval_prompt"] == 1
    assert external_raw["temperature"] == 0.7
    assert external_raw["top_p"] == 0.95
    assert external_raw["max_response_len"] == 16384
    assert external_raw["metadata_overrides"] == {"agent_step_limit": 64}

    class Sample:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    monkeypatch.setattr(rollout_module, "_load_sample_type", lambda: Sample)
    args = _rollout_args()
    bridge_config = resolve_polar_slime_config(args)
    dataset = EvalDatasetConfig(**external_raw)
    groups = rollout_module._load_eval_sample_groups(args, dataset)
    payload = rollout_module._build_task_payload(
        args=args,
        config=bridge_config,
        group=groups[0],
        rollout_id=0,
        task_position=0,
        eval_dataset_cfg=dataset,
        eval_dataset_name=dataset.name,
    )

    settings = payload["agent"]["settings"]
    assert bridge_config.train_agent_timeout == 1200.0
    assert payload["metadata"]["agent_timeout"] == 900.0
    assert settings["step_limit"] == 64
    assert settings["model_kwargs"] == {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 16384,
    }


def test_tmax_context_default_uses_qwen_native_window(
    tmp_path: Path,
) -> None:
    env = _clean_env(tmp_path)
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {TMAX / 'env.cwdfw.sh'} >/dev/null; "
            "printf '%s|%s' \"$SGLANG_CONTEXT_LENGTH\" \"$TMAX_MODEL_MAX_CONTEXT_LENGTH\"",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "262144|262144"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "SGLANG_CONTEXT_LENGTH": "18000",
                "ROLLOUT_MAX_RESPONSE_LEN": "4096",
                "POLAR_AGENT_MAX_TOKENS": "4096",
                "TMAX_EVAL_MAX_RESPONSE_LEN": "4096",
            },
            "TMAX_TRAIN_PACK_LENGTH=67584 exceeds SGLANG_CONTEXT_LENGTH=18000",
        ),
        (
            {"SGLANG_CONTEXT_LENGTH": "262145"},
            "exceeds Qwen3.5-9B native context 262144",
        ),
        (
            {"TMAX_EXTERNAL_EVAL_TOP_P": "1.1"},
            "TMAX_EXTERNAL_EVAL_TOP_P must be in (0, 1]",
        ),
    ],
)
def test_tmax_rejects_invalid_generation_limits(
    tmp_path: Path,
    overrides: dict[str, str],
    message: str,
) -> None:
    env = _clean_env(tmp_path)
    env.update(overrides)
    result = subprocess.run(
        ["bash", "-c", f"source {TMAX / 'env.cwdfw.sh'}"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert message in result.stderr
