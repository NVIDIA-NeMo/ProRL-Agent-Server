#!/usr/bin/env python3
"""Fail closed when allocation services would import outside PROJECT_ROOT."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any


class PythonProvenanceError(RuntimeError):
    """The selected interpreter resolved Polar from an unexpected source tree."""


def _module_path(module: ModuleType, *, name: str) -> Path:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise PythonProvenanceError(f"{name} does not have a concrete source file")
    return Path(raw_path).resolve()


def _require_below(path: Path, root: Path, *, name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PythonProvenanceError(
            f"{name} resolved outside the frozen project source: {path} (expected below {root})"
        ) from exc


def verify_python_provenance(project_root: Path, harness: str) -> dict[str, Any]:
    """Return source attestation after validating service and runner imports."""

    source_root = (project_root / "src").resolve()
    if not source_root.is_dir():
        raise PythonProvenanceError(f"project source directory is missing: {source_root}")

    polar = importlib.import_module("polar")
    polar_cli = importlib.import_module("polar.cli")
    polar_path = _module_path(polar, name="polar")
    cli_path = _module_path(polar_cli, name="polar.cli")
    _require_below(polar_path, source_root, name="polar")
    _require_below(cli_path, source_root, name="polar.cli")

    attestation: dict[str, Any] = {
        "schema_version": 1,
        "source_root": str(source_root),
        "polar": str(polar_path),
        "polar_cli": str(cli_path),
        "harness": harness,
    }
    if harness == "spilot_router":
        runner = importlib.import_module("polar.agent.presets.spilot_router_runner")
        runner_path = _module_path(runner, name="SPilot runner")
        _require_below(runner_path, source_root, name="SPilot runner")
        orchestrator = getattr(runner, "SpilotOrchestrator", None)
        router_completion = getattr(orchestrator, "_router_completion", None)
        if not callable(router_completion):
            raise PythonProvenanceError(
                "SPilot runner is missing callable SpilotOrchestrator._router_completion"
            )
        attestation.update(
            {
                "spilot_runner": str(runner_path),
                "spilot_runner_sha256": hashlib.sha256(runner_path.read_bytes()).hexdigest(),
            }
        )
    return attestation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--harness", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        attestation = verify_python_provenance(args.project_root, args.harness)
    except (ImportError, OSError, PythonProvenanceError) as exc:
        raise SystemExit(f"ERROR: Polar Python provenance check failed: {exc}") from exc
    print("[tmax provenance] " + json.dumps(attestation, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
