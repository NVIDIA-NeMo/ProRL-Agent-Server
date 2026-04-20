"""Load and cache the SWE-bench Verified dataset (500 tasks)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DATASET_SPLIT = "test"
DEFAULT_CACHE_PATH = Path.home() / ".cache" / "polar" / "swebench_verified.json"
HARNESS_IMAGE_PREFIX = "polar-swebench"

SUPPORTED_HARNESSES = (
    "opencode", "codex", "claude_code",
    "gemini_cli", "qwen_code", "swe_agent", "openhands_sdk",
)


def sanitize_instance_id(instance_id: str) -> str:
    normalized = instance_id.strip().lower()
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", normalized.replace("__", "--"))
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.strip("-")


def _normalize_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Ensure FAIL_TO_PASS and PASS_TO_PASS are lists, not JSON strings."""
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = instance.get(key)
        if isinstance(value, str):
            try:
                instance[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
    return instance


def base_image_for_instance(instance: dict[str, Any]) -> str:
    """Derive the SWE-bench Docker image name via swebench library, with fallback."""
    try:
        try:
            from swegym.harness.test_spec import make_test_spec

            spec = make_test_spec(instance)
        except ModuleNotFoundError:
            from swebench.harness.test_spec.test_spec import make_test_spec

            spec = make_test_spec(instance, namespace="swebench")
        image_ref = spec.instance_image_key.replace("arm64", "x86_64")
        if "/" in image_ref:
            return image_ref
    except Exception:
        pass

    # Fallback: xingyaoww community mirror convention
    instance_id = instance["instance_id"]
    suffix = instance_id.replace("__", "_s_").lower()
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{suffix}:latest"


def runtime_image_for_instance(instance_id: str) -> str:
    return f"{HARNESS_IMAGE_PREFIX}-runtime:{sanitize_instance_id(instance_id)}"


def load_swebench_verified(
    *,
    refresh: bool = False,
    cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load SWE-bench Verified via HuggingFace ``datasets``, with local JSON cache."""
    cache_file = cache_path or DEFAULT_CACHE_PATH
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())

    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    instances = [_normalize_instance(dict(row)) for row in ds]

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(instances, indent=2, ensure_ascii=True, sort_keys=True))
    print(f"Cached {len(instances)} SWE-bench Verified instances to {cache_file}")
    return instances
