"""Local backend — run Polar services directly on the current machine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polar.cluster.backend import ClusterBackend


class LocalBackend(ClusterBackend):
    """Run Polar services locally (no cluster scheduler).

    For local development the recommended flow is the per-example
    ``submit_tasks.py`` scripts under ``examples/``.  This backend exists
    as a placeholder so the unified config schema works with
    ``backend: local`` and can be extended in the future.
    """

    def launch(self, repo_root: Path, *, dry_run: bool = False, no_sync: bool = False) -> str:
        raise NotImplementedError(
            "Local launch is not yet integrated into 'polar cluster launch'.\n"
            "Use the per-example submit scripts instead:\n"
            "  python examples/calculator/opencode/submit_tasks.py"
        )

    def setup(self, repo_root: Path) -> None:
        print("[local] No setup required for local backend.")

    def status(self, job_id: str | None = None) -> dict[str, Any]:
        return {"backend": "local", "status": "not implemented"}

    def sync(
        self,
        repo_root: Path,
        *,
        job_id: str | None = None,
        code_only: bool = False,
        results_only: bool = False,
        dry_run: bool = False,
    ) -> None:
        print("[local] No sync needed for local backend.")

    def build_sif(
        self,
        repo_root: Path,
        example: str,
        harnesses: list[str],
        *,
        force: bool = False,
        instance_ids: list[str] | None = None,
    ) -> dict[str, Path]:
        raise NotImplementedError(
            "Local SIF builds require Docker or Apptainer installed locally.\n"
            "Use 'docker build' in the example directory, or set backend: slurm\n"
            "to build on a cluster with Apptainer."
        )
