"""Abstract cluster backend and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from polar.cluster.config import ClusterConfig


class ClusterBackend(ABC):
    """Base class for deployment backends (local, SLURM, K8s, ...)."""

    def __init__(self, config: ClusterConfig) -> None:
        self.config = config

    @abstractmethod
    def launch(
        self,
        repo_root: Path,
        *,
        dry_run: bool = False,
        no_sync: bool = False,
    ) -> str:
        """Sync code and launch a job. Return a job identifier string."""

    @abstractmethod
    def setup(self, repo_root: Path) -> None:
        """One-time environment setup on the target cluster."""

    @abstractmethod
    def status(self, job_id: str | None = None) -> dict[str, Any]:
        """Query job / service status."""

    @abstractmethod
    def sync(
        self,
        repo_root: Path,
        *,
        job_id: str | None = None,
        code_only: bool = False,
        results_only: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Sync results (and optionally code) back from the cluster."""

    @abstractmethod
    def build_sif(
        self,
        repo_root: Path,
        example: str,
        harnesses: list[str],
        *,
        force: bool = False,
        instance_ids: list[str] | None = None,
    ) -> dict[str, Path]:
        """Build Apptainer SIF images. Return ``{key: sif_path}``.

        For calculator, *key* is the harness name.
        For swegym, *key* is ``harness/sanitized_instance_id``.
        When *instance_ids* is ``None`` for swegym, builds all sample instances.
        """


    # ── Optional two-phase methods (non-abstract) ─────────────────────────

    def serve(
        self,
        repo_root: Path,
        *,
        dry_run: bool = False,
        no_sync: bool = False,
        wait: bool = True,
        wait_timeout: int = 600,
    ) -> dict[str, str]:
        """Start services without submitting tasks. Return job info."""
        raise NotImplementedError(
            f"The {type(self).__name__} backend does not support 'serve'. "
            "Use 'polar cluster launch' for a combined workflow."
        )

    def submit_task(
        self,
        repo_root: Path,
        *,
        job_id: str,
        example: str | None = None,
        harness: str | None = None,
    ) -> int:
        """Submit tasks to a running service. Return exit code."""
        raise NotImplementedError(
            f"The {type(self).__name__} backend does not support 'submit-task'. "
            "Use 'polar cluster launch' for a combined workflow."
        )

    def train(
        self,
        repo_root: Path,
        *,
        dry_run: bool = False,
        no_sync: bool = False,
        wait: bool = True,
        wait_timeout: int = 3600,
    ) -> dict[str, str]:
        """Submit a training job. Return job info dict."""
        raise NotImplementedError(
            f"The {type(self).__name__} backend does not support 'train'."
        )


def get_backend(config: ClusterConfig) -> ClusterBackend:
    """Return the appropriate backend for *config.backend*."""
    if config.backend == "slurm":
        from polar.cluster.slurm import SlurmBackend

        return SlurmBackend(config)
    if config.backend == "local":
        from polar.cluster.local import LocalBackend

        return LocalBackend(config)
    if config.backend == "k8s":
        raise NotImplementedError(
            "Kubernetes backend is not yet implemented. "
            "Contributions welcome!"
        )
    raise ValueError(f"Unknown backend: {config.backend!r}")
