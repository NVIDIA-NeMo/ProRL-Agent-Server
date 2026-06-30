#!/usr/bin/env python3
"""Build the two-dataset TMax evaluation config with fail-closed limits."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def _positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _temperature(name: str, value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}")
    return value


def _top_p(name: str, value: float) -> float:
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def build_eval_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return a validated Slime multi-dataset evaluation configuration."""

    if not args.primary_name or not args.external_name:
        raise ValueError("eval dataset names must be non-empty")
    if args.primary_name == args.external_name:
        raise ValueError("primary and external eval dataset names must be distinct")
    if not args.primary_path or not args.external_path:
        raise ValueError("eval dataset paths must be non-empty")

    context_length = _positive("sglang_context_length", args.sglang_context_length)
    model_max_context = _positive("model_max_context_length", args.model_max_context_length)
    max_prompt_len = _positive("eval_max_prompt_len", args.eval_max_prompt_len)
    if context_length > model_max_context:
        raise ValueError(
            "sglang_context_length exceeds the model's native context window: "
            f"{context_length} > {model_max_context}"
        )

    def dataset(
        *,
        name: str,
        path: str,
        samples: int,
        minimum: int,
        temperature: float,
        top_p: float,
        max_response_len: int,
        agent_step_limit: int | None = None,
    ) -> dict[str, Any]:
        samples = _positive(f"{name}.n_samples_per_eval_prompt", samples)
        minimum = _positive(f"{name}.min_eval_samples", minimum)
        max_response_len = _positive(f"{name}.max_response_len", max_response_len)
        if max_prompt_len + max_response_len > context_length:
            raise ValueError(
                f"{name}: eval_max_prompt_len + max_response_len exceeds "
                f"sglang_context_length ({max_prompt_len} + {max_response_len} "
                f"> {context_length})"
            )
        result: dict[str, Any] = {
            "name": name,
            "path": path,
            "n_samples_per_eval_prompt": samples,
            "min_eval_samples": minimum,
            "temperature": _temperature(f"{name}.temperature", temperature),
            "top_p": _top_p(f"{name}.top_p", top_p),
            "max_response_len": max_response_len,
        }
        if agent_step_limit is not None:
            result["metadata_overrides"] = {
                "agent_step_limit": _positive(
                    f"{name}.agent_step_limit", agent_step_limit
                )
            }
        return result

    return {
        "eval": {
            "datasets": [
                dataset(
                    name=args.primary_name,
                    path=args.primary_path,
                    samples=args.primary_samples,
                    minimum=args.primary_minimum,
                    temperature=args.primary_temperature,
                    top_p=args.primary_top_p,
                    max_response_len=args.primary_max_response_len,
                ),
                dataset(
                    name=args.external_name,
                    path=args.external_path,
                    samples=args.external_samples,
                    minimum=args.external_minimum,
                    temperature=args.external_temperature,
                    top_p=args.external_top_p,
                    max_response_len=args.external_max_response_len,
                    agent_step_limit=args.external_agent_step_limit,
                ),
            ]
        }
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-name", required=True)
    parser.add_argument("--primary-path", required=True)
    parser.add_argument("--primary-samples", type=int, required=True)
    parser.add_argument("--primary-minimum", type=int, required=True)
    parser.add_argument("--primary-temperature", type=float, required=True)
    parser.add_argument("--primary-top-p", type=float, required=True)
    parser.add_argument("--primary-max-response-len", type=int, required=True)
    parser.add_argument("--external-name", required=True)
    parser.add_argument("--external-path", required=True)
    parser.add_argument("--external-samples", type=int, required=True)
    parser.add_argument("--external-minimum", type=int, required=True)
    parser.add_argument("--external-temperature", type=float, required=True)
    parser.add_argument("--external-top-p", type=float, required=True)
    parser.add_argument("--external-max-response-len", type=int, required=True)
    parser.add_argument("--external-agent-step-limit", type=int, required=True)
    parser.add_argument("--eval-max-prompt-len", type=int, required=True)
    parser.add_argument("--sglang-context-length", type=int, required=True)
    parser.add_argument("--model-max-context-length", type=int, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        config = build_eval_config(args)
    except ValueError as exc:
        raise SystemExit(f"ERROR: invalid TMax eval configuration: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(config, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
