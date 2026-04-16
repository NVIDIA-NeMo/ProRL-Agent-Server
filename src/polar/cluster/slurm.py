"""SLURM cluster backend — rsync code, submit sbatch jobs, sync results."""

from __future__ import annotations

import importlib.resources
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from polar.cluster.backend import ClusterBackend
from polar.cluster.config import ClusterConfig

# Files and directories excluded from rsync to the cluster.
_RSYNC_EXCLUDES = [
    ".git",
    "__pycache__",
    "*.pyc",
    ".venv",
    "node_modules",
    "worklogs/",
    "results/",
    "rollout_results/",
    "*.egg-info",
]


class SlurmBackend(ClusterBackend):
    """Deploy Polar via SSH + sbatch on a SLURM cluster."""

    def __init__(self, config: ClusterConfig) -> None:
        super().__init__(config)
        slurm = config.slurm
        if not slurm.login_node:
            raise ValueError("slurm.login_node is required for the SLURM backend")
        if not slurm.account:
            raise ValueError("slurm.account is required for the SLURM backend")
        if not slurm.partition:
            raise ValueError("slurm.partition is required for the SLURM backend")
        if not config.paths.workspace:
            raise ValueError("paths.workspace is required for the SLURM backend")

    # ── Public API ────────────────────────────────────────────────────────────

    def launch(
        self,
        repo_root: Path,
        *,
        dry_run: bool = False,
        no_sync: bool = False,
    ) -> str:
        cfg = self.config
        self._print_summary()

        if not no_sync:
            self._sync_code_to_cluster(repo_root)

        # Pre-populate swegym sample cache so the job can find instance data
        if cfg.task.example == "swegym":
            self._sync_swegym_cache()
        if cfg.task.example == "swebench_verified":
            self._sync_swebench_cache()

        sbatch_cmd = self._build_sbatch_command()
        print(f"\n[cluster] sbatch command:\n  {sbatch_cmd}")

        if dry_run:
            print("\n[cluster] Dry run — not submitting.")
            return ""

        results_dir = cfg.paths.results
        self._ssh_run(f"mkdir -p '{results_dir}'")
        out = self._ssh_run(sbatch_cmd, capture=True)
        job_id = out.strip().split()[-1]

        login = cfg.slurm.login_node
        example, harness = cfg.task.example, cfg.task.harness
        print(f"\n[cluster] Job submitted: {job_id}")
        print(f"[cluster] Monitor:")
        print(f"  ssh {login} squeue -j {job_id}")
        print(f"  ssh {login} 'tail -f {results_dir}/polar-{example}-{harness}_{job_id}.out'")
        print(f"\n[cluster] Sync results after completion:")
        print(f"  polar cluster sync -c <config>")
        return job_id

    def setup(self, repo_root: Path) -> None:
        cfg = self.config
        self._sync_code_to_cluster(repo_root)
        print(f"\n[cluster] Running setup on {cfg.slurm.login_node}...")
        env_parts = [f"export POLAR_WORKSPACE='{cfg.paths.workspace}'"]
        if cfg.paths.apptainer_bin_dir:
            env_parts.append(f"export APPTAINER_BIN_DIR='{cfg.paths.apptainer_bin_dir}'")
        if cfg.paths.cuda_home:
            env_parts.append(f"export CUDA_HOME='{cfg.paths.cuda_home}'")
        env_str = " && ".join(env_parts)
        setup_script = f"{cfg.paths.code}/examples/slurm/setup_cluster.sh"
        self._ssh_run(f"{env_str} && cd '{cfg.paths.code}' && bash '{setup_script}'")
        print("[cluster] Setup complete.")

    def status(self, job_id: str | None = None) -> dict[str, Any]:
        if job_id:
            out = self._ssh_run(
                f"squeue -j {job_id} --format='%i %j %T %M %N' --noheader",
                capture=True,
            )
        else:
            out = self._ssh_run(
                "squeue -u $USER --format='%i %j %T %M %N' --noheader",
                capture=True,
            )
        jobs: list[dict[str, str]] = []
        for line in out.strip().splitlines():
            parts = line.split(None, 4)
            if len(parts) >= 3:
                jobs.append({
                    "job_id": parts[0],
                    "name": parts[1] if len(parts) > 1 else "",
                    "state": parts[2] if len(parts) > 2 else "",
                    "time": parts[3] if len(parts) > 3 else "",
                    "nodes": parts[4] if len(parts) > 4 else "",
                })
        return {"jobs": jobs}

    def sync(
        self,
        repo_root: Path,
        *,
        job_id: str | None = None,
        code_only: bool = False,
        results_only: bool = False,
        dry_run: bool = False,
    ) -> None:
        cfg = self.config
        login = cfg.slurm.login_node
        local = self._is_local()
        extra: list[str] = []
        if dry_run:
            extra.append("--dry-run")

        def _remote_path(p: str) -> str:
            return p if local else f"{login}:{p}"

        if not results_only:
            print(f"[cluster] Syncing code from {login}...")
            src = _remote_path(cfg.paths.code + "/")
            dst = str(repo_root) + "/"
            self._rsync(src, dst, extra_args=extra, exclude=_RSYNC_EXCLUDES)

        if not code_only:
            print(f"[cluster] Syncing results from {login}...")
            if job_id:
                pattern = f"*_{job_id}"
                src = _remote_path(f"{cfg.paths.results}/{pattern}/")
                dst_dir = repo_root / "results" / pattern
                dst_dir.mkdir(parents=True, exist_ok=True)
                self._rsync(src, str(dst_dir) + "/", extra_args=extra)
            else:
                src = _remote_path(cfg.paths.results + "/")
                dst_dir = repo_root / "results"
                dst_dir.mkdir(parents=True, exist_ok=True)
                self._rsync(src, str(dst_dir) + "/", extra_args=extra)

        print("[cluster] Sync complete.")

    def build_sif(
        self,
        repo_root: Path,
        example: str,
        harnesses: list[str],
        *,
        force: bool = False,
        instance_ids: list[str] | None = None,
    ) -> dict[str, Path]:
        cfg = self.config
        results: dict[str, Path] = {}

        if example == "train":
            sif_name = "train-slime-grpo.sif"
            def_content = _generate_train_def(cfg.paths.code)
            sif_path = self._build_single_sif(
                sif_name, def_content, force=force,
                srun_time="01:00:00", srun_mem="64G",
            )
            results["train"] = Path(sif_path)
            return results

        if example in ("swegym", "swebench_verified"):
            # Per-instance examples: one SIF per (harness, instance_id)
            if not instance_ids:
                if example == "swegym":
                    from polar.cluster.tasks import SWEGYM_SAMPLE
                    instance_ids = [i["instance_id"] for i in SWEGYM_SAMPLE]
                else:
                    raise ValueError(
                        f"--instance-id is required for {example} SIF builds."
                    )
            prefix = "swegym" if example == "swegym" else "swebench"
            for harness in harnesses:
                for instance_id in instance_ids:
                    sanitized = _sanitize_instance_id(instance_id)
                    sif_name = f"{prefix}-{harness}-{sanitized}.sif"
                    def_content = _generate_def_file(
                        repo_root, example, harness, instance_id=instance_id,
                    )
                    if def_content is None:
                        print(f"[cluster] WARNING: No .def for {example}/{harness}/{instance_id}")
                        continue
                    sif_path = self._build_single_sif(
                        sif_name, def_content, force=force,
                    )
                    key = f"{harness}/{sanitized}"
                    results[key] = Path(sif_path)
        else:
            # Calculator and other examples: one SIF per harness
            for harness in harnesses:
                sif_name = f"{example}-{harness}.sif"
                def_content = _generate_def_file(repo_root, example, harness)
                if def_content is None:
                    print(f"[cluster] WARNING: Cannot generate .def for {example}/{harness}, skipping")
                    continue
                sif_path = self._build_single_sif(
                    sif_name, def_content, force=force,
                )
                results[harness] = Path(sif_path)

        return results

    def serve(
        self,
        repo_root: Path,
        *,
        dry_run: bool = False,
        no_sync: bool = False,
        wait: bool = True,
        wait_timeout: int = 600,
    ) -> dict[str, str]:
        """Submit a serve-only sbatch job. Return {job_id, topology}."""
        cfg = self.config
        self._print_serve_summary()

        if not no_sync:
            self._sync_code_to_cluster(repo_root)

        sbatch_cmd = self._build_serve_sbatch_command()
        print(f"\n[cluster] sbatch command:\n  {sbatch_cmd}")

        if dry_run:
            print("\n[cluster] Dry run — not submitting.")
            return {}

        results_dir = cfg.paths.results
        self._ssh_run(f"mkdir -p '{results_dir}'")
        out = self._ssh_run(sbatch_cmd, capture=True)
        job_id = out.strip().split()[-1]

        job_dir = f"{results_dir}/polar-serve_{job_id}"
        sentinel = f"{job_dir}/.services_ready"

        print(f"\n[cluster] Job submitted: {job_id}")

        if not wait:
            print(f"[cluster] Not waiting. Check readiness with:")
            print(f"  polar cluster status -c <config> --job-id {job_id}")
            return {"job_id": job_id, "topology": f"{job_dir}/topology.yaml"}

        print(f"[cluster] Waiting for services to be ready (timeout: {wait_timeout}s)...")
        poll_interval = 10
        for attempt in range(wait_timeout // poll_interval):
            content = self._ssh_run(
                f"cat '{sentinel}' 2>/dev/null || true",
                capture=True,
            )
            if "TOPOLOGY=" in content:
                topology = ""
                for line in content.strip().splitlines():
                    if line.startswith("TOPOLOGY="):
                        topology = line.split("=", 1)[1]
                        break
                return {"job_id": job_id, "topology": topology}

            # Check job is still alive
            state = self._get_job_state(job_id)
            if state in ("FAILED", "CANCELLED", "TIMEOUT", "COMPLETED", ""):
                raise RuntimeError(
                    f"Serve job {job_id} entered state '{state}' before services were ready. "
                    f"Check logs: {job_dir}/logs/"
                )
            elapsed = (attempt + 1) * poll_interval
            if elapsed % 30 == 0:
                print(f"[cluster] Still waiting... ({elapsed}s, job state: {state})")
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Services not ready after {wait_timeout}s. "
            f"Check job logs: {job_dir}/logs/"
        )

    def submit_task(
        self,
        repo_root: Path,
        *,
        job_id: str,
        example: str | None = None,
        harness: str | None = None,
    ) -> int:
        """Submit tasks to a running serve job. Return exit code."""
        cfg = self.config
        example = example or cfg.task.example
        harness = harness or cfg.task.harness

        # Discover topology from job ID
        topology_path = self._find_topology(job_id)
        job_dir = str(Path(topology_path).parent)

        print(f"[cluster] Submitting tasks to job {job_id}")
        print(f"[cluster]   Topology: {topology_path}")
        print(f"[cluster]   Example:  {example}")
        print(f"[cluster]   Harness:  {harness}")

        # Build instance-id args
        instance_id_args = ""
        if cfg.task.instance_ids:
            for iid in cfg.task.instance_ids:
                instance_id_args += f" --instance-id {iid}"

        env_setup = (
            f"export POLAR_WORKSPACE='{cfg.paths.workspace}' && "
            f"source '{cfg.paths.code}/src/polar/cluster/templates/env.sh'"
        )
        task_cmd = (
            f"{env_setup} && "
            f"python -m polar.cluster.tasks "
            f"--example {example} --harness {harness} "
            f"--topology {topology_path} "
            f"--sif-dir {cfg.paths.sif_dir} "
            f"--output-dir {job_dir}/tasks "
            f"--num-rollouts {cfg.task.num_rollouts} "
            f"--timeout-seconds {cfg.task.timeout_seconds}"
            f"{instance_id_args}"
        )

        try:
            self._ssh_run(task_cmd)
            print("[cluster] Task submission complete.")
            return 0
        except subprocess.CalledProcessError as exc:
            print(f"[cluster] Task submission failed (exit code {exc.returncode})")
            return exc.returncode

    def train(
        self,
        repo_root: Path,
        *,
        dry_run: bool = False,
        no_sync: bool = False,
        wait: bool = True,
        wait_timeout: int = 3600,
    ) -> dict[str, str]:
        """Submit a training sbatch job. Return job info dict."""
        cfg = self.config
        self._print_train_summary()

        if not no_sync:
            self._sync_code_to_cluster(repo_root)

        sbatch_cmd = self._build_train_sbatch_command()
        print(f"\n[cluster] sbatch command:\n  {sbatch_cmd}")

        if dry_run:
            print("\n[cluster] Dry run — not submitting.")
            return {}

        results_dir = cfg.paths.results
        self._ssh_run(f"mkdir -p '{results_dir}'")
        out = self._ssh_run(sbatch_cmd, capture=True)
        job_id = out.strip().split()[-1]

        job_dir = f"{results_dir}/polar-train_{job_id}"
        login = cfg.slurm.login_node

        print(f"\n[cluster] Training job submitted: {job_id}")
        print(f"[cluster] Monitor:")
        print(f"  ssh {login} squeue -j {job_id}")
        print(f"  ssh {login} 'tail -f {results_dir}/polar-train_{job_id}.out'")

        if not wait:
            return {"job_id": job_id, "job_dir": job_dir}

        print(f"[cluster] Waiting for training to complete (timeout: {wait_timeout}s)...")
        poll_interval = 30
        for attempt in range(wait_timeout // poll_interval):
            state = self._get_job_state(job_id)
            if state == "COMPLETED":
                print(f"[cluster] Training job {job_id} completed successfully.")
                return {"job_id": job_id, "job_dir": job_dir, "state": "COMPLETED"}
            if state in ("FAILED", "CANCELLED", "TIMEOUT", ""):
                raise RuntimeError(
                    f"Training job {job_id} entered state '{state}'. "
                    f"Check logs: ssh {login} 'tail -100 {results_dir}/polar-train_{job_id}.out'"
                )
            elapsed = (attempt + 1) * poll_interval
            if elapsed % 120 == 0:
                print(f"[cluster] Training still running... ({elapsed}s, state: {state})")
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Training job {job_id} not completed after {wait_timeout}s. "
            f"Job may still be running. Check: ssh {login} squeue -j {job_id}"
        )

    def _build_single_sif(
        self,
        sif_name: str,
        def_content: str,
        *,
        force: bool = False,
        srun_time: str = "00:30:00",
        srun_mem: str = "32G",
    ) -> str:
        """Build a single SIF image on the cluster and return its path."""
        cfg = self.config
        sif_path = f"{cfg.paths.sif_dir}/{sif_name}"

        # Check if SIF already exists (skip unless --force)
        if not force:
            try:
                self._ssh_run(f"test -f '{sif_path}'", check=True)
                print(f"[cluster] SIF exists, skipping: {sif_name}")
                return sif_path
            except subprocess.CalledProcessError:
                pass  # file doesn't exist, proceed

        print(f"[cluster] Building SIF: {sif_name}")
        def_dir = f"{cfg.paths.polar_root}/tmp_defs"
        remote_def = f"{def_dir}/{sif_name}.def"
        self._ssh_run(f"mkdir -p '{def_dir}'")
        self._ssh_run(f"cat > '{remote_def}' << 'POLAREOF'\n{def_content}\nPOLAREOF")
        self._ssh_run(f"mkdir -p '{cfg.paths.sif_dir}'")

        force_flag = "--force" if force else ""
        cache_dir = f"{cfg.paths.polar_root}/apptainer_cache"
        path_prefix = ""
        if cfg.paths.apptainer_bin_dir:
            path_prefix = f"export PATH='{cfg.paths.apptainer_bin_dir}':$PATH && "
        build_cmd = (
            f"{path_prefix}"
            f"export APPTAINER_CACHEDIR='{cache_dir}' && mkdir -p '{cache_dir}' && "
            f"apptainer build {force_flag} '{sif_path}' '{remote_def}'"
        )
        account = cfg.slurm.account
        try:
            self._ssh_run(
                f"srun --account={account} --partition=cpu_short --time={srun_time} "
                f"--cpus-per-task=8 --mem={srun_mem} bash -c {shlex.quote(build_cmd)}"
            )
        except subprocess.CalledProcessError:
            print(f"[cluster] srun failed, trying direct build...")
            self._ssh_run(build_cmd)

        self._ssh_run(f"rm -f '{remote_def}'")
        print(f"[cluster] Built: {sif_path}")
        return sif_path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_local(self) -> bool:
        """Return True if we're already on the login node (sbatch available)."""
        if not hasattr(self, "_local_cache"):
            import shutil
            self._local_cache = shutil.which("sbatch") is not None
        return self._local_cache

    def _ssh_run(
        self,
        command: str,
        *,
        capture: bool = False,
        check: bool = True,
    ) -> str:
        if self._is_local():
            cmd = ["bash", "-l", "-c", command]
        else:
            login = self.config.slurm.login_node
            cmd = ["ssh", "-o", "ConnectTimeout=10", login, command]
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            check=check,
        )
        if capture:
            return result.stdout
        return ""

    def _rsync(
        self,
        src: str,
        dst: str,
        *,
        exclude: list[str] | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        cmd = ["rsync", "-avz", "--delete"]
        for pattern in exclude or []:
            cmd.extend(["--exclude", pattern])
        cmd.extend(extra_args or [])
        cmd.extend([src, dst])
        subprocess.run(cmd, check=True)

    def _sync_code_to_cluster(self, repo_root: Path) -> None:
        cfg = self.config
        login = cfg.slurm.login_node
        print(f"\n[cluster] Syncing code to {login}:{cfg.paths.code}/ ...")
        self._ssh_run(f"mkdir -p '{cfg.paths.code}'")
        if self._is_local():
            src = str(repo_root) + "/"
            dst = cfg.paths.code + "/"
        else:
            src = str(repo_root) + "/"
            dst = f"{login}:{cfg.paths.code}/"
        self._rsync(src, dst, exclude=_RSYNC_EXCLUDES)
        print("[cluster] Sync complete.")

    def _sync_swegym_cache(self) -> None:
        """Ensure the SWE-Gym sample instance cache exists on the cluster."""
        cache_file = Path.home() / ".cache" / "polar" / "swegym_sample_10.json"
        if not cache_file.exists():
            try:
                from examples.swegym.sample_tasks import fetch_sample_instances
                print("[cluster] Fetching SWE-Gym sample data from HuggingFace...")
                fetch_sample_instances()
            except Exception as exc:
                print(f"[cluster] WARNING: Could not fetch SWE-Gym sample data: {exc}")
                return
        if cache_file.exists():
            if self._is_local():
                # Already on the login node — cache is in place, nothing to sync.
                print("[cluster] SWE-Gym sample cache already present.")
            else:
                login = self.config.slurm.login_node
                self._ssh_run("mkdir -p ~/.cache/polar/")
                self._rsync(str(cache_file), f"{login}:~/.cache/polar/swegym_sample_10.json")
                print("[cluster] SWE-Gym sample cache synced.")

    def _sync_swebench_cache(self) -> None:
        """Ensure the SWE-bench Verified dataset cache exists on the cluster."""
        cache_file = Path.home() / ".cache" / "polar" / "swebench_verified.json"
        if not cache_file.exists():
            try:
                import importlib
                spec = importlib.util.spec_from_file_location(
                    "dataset",
                    Path(__file__).resolve().parents[2] / "examples" / "swebench_verified" / "dataset.py",
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                print("[cluster] Fetching SWE-bench Verified dataset from HuggingFace...")
                mod.load_swebench_verified()
            except Exception as exc:
                print(f"[cluster] WARNING: Could not fetch SWE-bench Verified data: {exc}")
                return
        if cache_file.exists():
            if self._is_local():
                print("[cluster] SWE-bench Verified cache already present.")
            else:
                login = self.config.slurm.login_node
                self._ssh_run("mkdir -p ~/.cache/polar/")
                self._rsync(str(cache_file), f"{login}:~/.cache/polar/swebench_verified.json")
                print("[cluster] SWE-bench Verified cache synced.")

    def _build_sbatch_command(self) -> str:
        cfg = self.config
        export_vars = cfg.sbatch_export_vars()
        export_str = "ALL," + ",".join(f"{k}={v}" for k, v in export_vars.items())

        example = cfg.task.example
        harness = cfg.task.harness
        results_dir = cfg.paths.results
        job_name = f"polar-{example}-{harness}"
        sbatch_path = f"{cfg.paths.code}/src/polar/cluster/templates/polar_slurm.sbatch"

        parts = [
            "sbatch",
            f"--account={cfg.slurm.account}",
            f"--partition={cfg.slurm.partition}",
            f"--nodes={cfg.resources.nodes}",
            f"--gres=gpu:{cfg.resources.gpus_per_node}",
            f"--cpus-per-task={cfg.resources.cpus_per_task}",
            f"--mem={cfg.resources.mem}",
            f"--time={cfg.resources.time}",
            f"--job-name={job_name}",
            f"--output={results_dir}/{job_name}_%j.out",
            f"--error={results_dir}/{job_name}_%j.err",
            f"--export={export_str}",
            sbatch_path,
        ]
        return " ".join(parts)

    def _build_serve_sbatch_command(self) -> str:
        cfg = self.config
        export_vars = cfg.sbatch_serve_export_vars()
        export_str = "ALL," + ",".join(f"{k}={v}" for k, v in export_vars.items())

        results_dir = cfg.paths.results
        job_name = "polar-serve"
        sbatch_path = f"{cfg.paths.code}/src/polar/cluster/templates/polar_slurm_serve.sbatch"

        parts = [
            "sbatch",
            f"--account={cfg.slurm.account}",
            f"--partition={cfg.slurm.partition}",
            f"--nodes={cfg.resources.nodes}",
            f"--gres=gpu:{cfg.resources.gpus_per_node}",
            f"--cpus-per-task={cfg.resources.cpus_per_task}",
            f"--mem={cfg.resources.mem}",
            f"--time={cfg.resources.time}",
            f"--job-name={job_name}",
            f"--output={results_dir}/{job_name}_%j.out",
            f"--error={results_dir}/{job_name}_%j.err",
            f"--export={export_str}",
            sbatch_path,
        ]
        return " ".join(parts)

    def _build_train_sbatch_command(self) -> str:
        cfg = self.config
        export_vars = cfg.sbatch_train_export_vars()
        export_str = "ALL," + ",".join(f"{k}={v}" for k, v in export_vars.items())

        results_dir = cfg.paths.results
        job_name = "polar-train"
        sbatch_path = f"{cfg.paths.code}/src/polar/cluster/templates/polar_slurm_train.sbatch"

        parts = [
            "sbatch",
            f"--account={cfg.slurm.account}",
            f"--partition={cfg.slurm.partition}",
            f"--nodes={cfg.resources.nodes}",
            f"--gres=gpu:{cfg.resources.gpus_per_node}",
            f"--cpus-per-task={cfg.resources.cpus_per_task}",
            f"--mem={cfg.resources.mem}",
            f"--time={cfg.resources.time}",
            f"--job-name={job_name}",
            f"--output={results_dir}/{job_name}_%j.out",
            f"--error={results_dir}/{job_name}_%j.err",
            # Quote the export string — values like MODEL_ARGS contain spaces
            f"--export={shlex.quote(export_str)}",
            sbatch_path,
        ]
        return " ".join(parts)

    def _get_job_state(self, job_id: str) -> str:
        """Query SLURM for the current state of a job."""
        out = self._ssh_run(
            f"squeue -j {job_id} --format='%T' --noheader 2>/dev/null || true",
            capture=True,
        )
        return out.strip()

    def _find_topology(self, job_id: str) -> str:
        """Discover the topology.yaml path for a running serve job."""
        cfg = self.config
        results_dir = cfg.paths.results

        # Try the sentinel file first (written by polar_slurm_serve.sbatch)
        sentinel = f"{results_dir}/polar-serve_{job_id}/.services_ready"
        content = self._ssh_run(
            f"cat '{sentinel}' 2>/dev/null || true",
            capture=True,
        )
        if "TOPOLOGY=" in content:
            for line in content.strip().splitlines():
                if line.startswith("TOPOLOGY="):
                    return line.split("=", 1)[1]

        # Fallback: search for topology.yaml matching the job ID
        out = self._ssh_run(
            f"ls '{results_dir}'/*_{job_id}/topology.yaml 2>/dev/null || true",
            capture=True,
        )
        path = out.strip().splitlines()[0] if out.strip() else ""
        if path:
            return path

        raise FileNotFoundError(
            f"Cannot find topology for job {job_id}. "
            f"Is the serve job running? Check: polar cluster status -c <config> --job-id {job_id}"
        )

    def _print_serve_summary(self) -> None:
        cfg = self.config
        lines = [
            "=" * 65,
            "Polar SLURM Serve (services only)",
            "=" * 65,
            f"  Model:      {cfg.model.name}",
            f"  TP size:    {cfg.model.tensor_parallel_size}",
            "  " + "-" * 60,
            f"  Login node: {cfg.slurm.login_node}",
            f"  Account:    {cfg.slurm.account}",
            f"  Partition:  {cfg.slurm.partition}",
            f"  Nodes:      {cfg.resources.nodes}",
            f"  GPUs/node:  {cfg.resources.gpus_per_node}",
            f"  Time limit: {cfg.resources.time}",
            f"  Workspace:  {cfg.paths.workspace}",
            "=" * 65,
        ]
        print("\n".join(lines))

    def _print_train_summary(self) -> None:
        cfg = self.config
        t = cfg.train
        lines = [
            "=" * 65,
            "Polar SLURM Training Job (Slime + Megatron GRPO)",
            "=" * 65,
            f"  HF checkpoint: {t.hf_checkpoint}",
            f"  Actor GPUs:    {t.actor_gpus} (TP={t.tp_size})",
            f"  Rollout GPUs:  {t.rollout_gpus}",
            f"  Num rollouts:  {t.num_rollouts}",
            f"  Batch:         {t.rollout_batch_size} prompts x {t.n_samples_per_prompt} samples",
            f"  Global batch:  {t.global_batch_size}",
            "  " + "-" * 60,
            f"  Login node:    {cfg.slurm.login_node}",
            f"  Account:       {cfg.slurm.account}",
            f"  Partition:     {cfg.slurm.partition}",
            f"  Nodes:         {cfg.resources.nodes}",
            f"  GPUs/node:     {cfg.resources.gpus_per_node}",
            f"  Time limit:    {cfg.resources.time}",
            f"  Workspace:     {cfg.paths.workspace}",
            "=" * 65,
        ]
        print("\n".join(lines))

    def _print_summary(self) -> None:
        cfg = self.config
        lines = [
            "=" * 65,
            "Polar SLURM Job Submission",
            "=" * 65,
            f"  Example:    {cfg.task.example}",
            f"  Harness:    {cfg.task.harness}",
            f"  Model:      {cfg.model.name}",
            f"  TP size:    {cfg.model.tensor_parallel_size}",
            f"  Rollouts:   {cfg.task.num_rollouts}",
            f"  Timeout:    {cfg.task.timeout_seconds}s",
            "  " + "-" * 60,
            f"  Login node: {cfg.slurm.login_node}",
            f"  Account:    {cfg.slurm.account}",
            f"  Partition:  {cfg.slurm.partition}",
            f"  Nodes:      {cfg.resources.nodes}",
            f"  GPUs/node:  {cfg.resources.gpus_per_node}",
            f"  Time limit: {cfg.resources.time}",
            f"  Workspace:  {cfg.paths.workspace}",
            "=" * 65,
        ]
        print("\n".join(lines))


# ── SIF definition file generation ────────────────────────────────────────────


def _sanitize_instance_id(instance_id: str) -> str:
    """Normalize instance ID for use in filenames."""
    normalized = instance_id.strip().lower()
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", normalized.replace("__", "--"))
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.strip("-")


# Maps harness name to (base_image, install_commands)
_CALCULATOR_HARNESS_DEFS: dict[str, tuple[str, list[str]]] = {
    "opencode": (
        "node:22-bookworm-slim",
        [
            "apt-get update && apt-get install -y --no-install-recommends bash ca-certificates curl git python-is-python3 python3 && rm -rf /var/lib/apt/lists/*",
            "npm install -g opencode-ai@latest",
        ],
    ),
    "codex": (
        "node:22-bookworm-slim",
        [
            "apt-get update && apt-get install -y --no-install-recommends bash ca-certificates curl git python-is-python3 python3 && rm -rf /var/lib/apt/lists/*",
            "npm install -g @openai/codex@latest",
        ],
    ),
    "claude_code": (
        "node:22-bookworm-slim",
        [
            "apt-get update && apt-get install -y --no-install-recommends bash ca-certificates curl git python-is-python3 python3 && rm -rf /var/lib/apt/lists/*",
            "npm install -g @anthropic-ai/claude-code@latest",
        ],
    ),
    "gemini_cli": (
        "node:22-bookworm-slim",
        [
            "apt-get update && apt-get install -y --no-install-recommends bash ca-certificates curl git python-is-python3 python3 && rm -rf /var/lib/apt/lists/*",
            "npm install -g @google/gemini-cli@latest",
        ],
    ),
    "qwen_code": (
        "node:22-bookworm-slim",
        [
            "apt-get update && apt-get install -y --no-install-recommends bash ca-certificates curl git python-is-python3 python3 && rm -rf /var/lib/apt/lists/*",
            "npm install -g @qwen-code/qwen-code@latest",
        ],
    ),
    "swe_agent": (
        "python:3.12-slim",
        [
            "apt-get update && apt-get install -y --no-install-recommends bash ca-certificates curl git build-essential && rm -rf /var/lib/apt/lists/*",
            "pip install --no-cache-dir 'sweagent[all] @ git+https://github.com/SWE-agent/SWE-agent.git'",
            "pip install --no-cache-dir tree-sitter==0.21.3 tree-sitter-languages",
            "SITE=$(python -c 'import site; print(site.getsitepackages()[0])') && "
            "git clone --depth 1 https://github.com/SWE-agent/SWE-agent.git /tmp/swe-agent-src && "
            "cp -r /tmp/swe-agent-src/config $SITE/config && "
            "cp -r /tmp/swe-agent-src/tools $SITE/tools && "
            "mkdir -p $SITE/trajectories && "
            "rm -rf /tmp/swe-agent-src",
        ],
    ),
    "openhands_sdk": (
        "python:3.12-slim",
        [
            "apt-get update && apt-get install -y --no-install-recommends bash ca-certificates curl git && rm -rf /var/lib/apt/lists/*",
            "pip install --no-cache-dir openhands-sdk openhands-tools",
        ],
    ),
}


def _generate_def_file(
    repo_root: Path,
    example: str,
    harness: str,
    instance_id: str | None = None,
) -> str | None:
    """Generate an Apptainer ``.def`` file for building a harness SIF."""
    if example == "calculator":
        spec = _CALCULATOR_HARNESS_DEFS.get(harness)
        if spec is None:
            return None
        base_image, commands = spec
        post = "\n    ".join(commands)
        return (
            f"Bootstrap: docker\n"
            f"From: {base_image}\n"
            f"\n"
            f"%post\n"
            f"    {post}\n"
            f"    mkdir -p /polar/session/workspace /polar/session/logs/agent\n"
            f"\n"
            f"%environment\n"
            f"    export DEBIAN_FRONTEND=noninteractive\n"
            f"\n"
            f"%labels\n"
            f"    io.polar.example {example}\n"
            f"    io.polar.harness {harness}\n"
        )
    if example == "swegym":
        return _generate_swegym_def(harness, instance_id)
    if example == "swebench_verified":
        return _generate_swebench_def(harness, instance_id)
    return None


def _swegym_base_image(instance_id: str) -> str:
    """Derive the SWE-Gym eval base image from an instance ID."""
    suffix = instance_id.replace("__", "_s_").lower()
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{suffix}:latest"


# SWE-Gym harness install commands (layered on top of per-instance base image).
# Base images already have conda + testbed env; we add the agent harness tools.
_SWEGYM_HARNESS_DEFS: dict[str, list[str]] = {
    "swe_agent": [
        "/opt/miniconda3/bin/conda create -y -n polar-sweagent python=3.11 pip",
        "/opt/miniconda3/envs/polar-sweagent/bin/python -m pip install --no-cache-dir "
        "'git+https://github.com/SWE-agent/SWE-agent.git'",
        "/opt/miniconda3/envs/polar-sweagent/bin/python -m pip install --no-cache-dir "
        "tree-sitter==0.21.3 tree-sitter-languages",
        "SITE=$(/opt/miniconda3/envs/polar-sweagent/bin/python -c "
        "\"import site; print(site.getsitepackages()[0])\") && "
        "git clone --depth 1 https://github.com/SWE-agent/SWE-agent.git /tmp/swe-agent-src && "
        "cp -r /tmp/swe-agent-src/config $SITE/config && "
        "cp -r /tmp/swe-agent-src/tools $SITE/tools && "
        "mkdir -p $SITE/trajectories && "
        "/opt/miniconda3/bin/conda clean -afy && "
        "rm -rf /tmp/swe-agent-src",
    ],
}


def _generate_swegym_def(
    harness: str,
    instance_id: str | None,
) -> str | None:
    """Generate an Apptainer .def for a SWE-Gym per-instance SIF."""
    if instance_id is None:
        return None
    spec = _SWEGYM_HARNESS_DEFS.get(harness)
    if spec is None:
        return None
    base_image = _swegym_base_image(instance_id)
    post = "\n    ".join(spec)
    return (
        f"Bootstrap: docker\n"
        f"From: {base_image}\n"
        f"\n"
        f"%post\n"
        f"    {post}\n"
        f"    mkdir -p /polar/session/workspace /polar/session/logs/agent\n"
        f"\n"
        f"%environment\n"
        f"    export DEBIAN_FRONTEND=noninteractive\n"
        f"    export PATH=/opt/miniconda3/envs/testbed/bin:"
        f"/opt/miniconda3/envs/polar-sweagent/bin:$PATH\n"
        f"\n"
        f"%labels\n"
        f"    io.polar.example swegym\n"
        f"    io.polar.harness {harness}\n"
        f"    io.polar.instance_id {instance_id}\n"
    )


# ── SWE-bench Verified SIF definitions ──────────────────────────────────────

_SWEBENCH_NODE_INSTALL = (
    "apt-get update && "
    "apt-get install -y --no-install-recommends ca-certificates curl gnupg && "
    "curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && "
    "apt-get install -y --no-install-recommends nodejs && "
    "apt-get clean && rm -rf /var/lib/apt/lists/*"
)

_SWEBENCH_HARNESS_DEFS: dict[str, list[str]] = {
    "opencode": [
        _SWEBENCH_NODE_INSTALL,
        "npm install -g opencode-ai@latest",
    ],
    "codex": [
        _SWEBENCH_NODE_INSTALL,
        "npm install -g @openai/codex@latest",
    ],
    "claude_code": [
        _SWEBENCH_NODE_INSTALL,
        "npm install -g @anthropic-ai/claude-code@latest",
    ],
}


def _swebench_base_image(instance_id: str) -> str:
    """Derive the SWE-bench eval base image from an instance ID."""
    suffix = instance_id.replace("__", "_s_").lower()
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{suffix}:latest"


def _generate_swebench_def(
    harness: str,
    instance_id: str | None,
) -> str | None:
    """Generate an Apptainer .def for a SWE-bench Verified per-instance SIF."""
    if instance_id is None:
        return None
    spec = _SWEBENCH_HARNESS_DEFS.get(harness)
    if spec is None:
        return None
    base_image = _swebench_base_image(instance_id)
    post = "\n    ".join(spec)
    return (
        f"Bootstrap: docker\n"
        f"From: {base_image}\n"
        f"\n"
        f"%post\n"
        f"    {post}\n"
        f"    mkdir -p /polar/session/workspace /polar/session/logs/agent\n"
        f"\n"
        f"%environment\n"
        f"    export DEBIAN_FRONTEND=noninteractive\n"
        f"    export PATH=/opt/miniconda3/envs/testbed/bin:$PATH\n"
        f"\n"
        f"%labels\n"
        f"    io.polar.example swebench_verified\n"
        f"    io.polar.harness {harness}\n"
        f"    io.polar.instance_id {instance_id}\n"
    )


def _generate_train_def(code_path: str) -> str:
    """Generate an Apptainer .def for the Slime+Megatron GRPO training SIF.

    Based on NGC PyTorch with Slime, Megatron-LM, and Polar patches installed.
    """
    return (
        "Bootstrap: docker\n"
        "From: nvcr.io/nvidia/pytorch:24.12-py3\n"
        "\n"
        "%files\n"
        f"    {code_path} /opt/polar\n"
        "\n"
        "%post\n"
        "    # Freeze NGC's PyTorch version and constrain transformers.\n"
        "    # transformers >=4.51 adds Qwen3 support; >=4.53 imports\n"
        "    # torch._dynamo.TransformGetItemToIndex which doesn't exist in\n"
        "    # NGC's torch 2.6.0a0.  So we pin to [4.51, 4.53).\n"
        "    TORCH_VER=$(pip show torch 2>/dev/null | grep '^Version' | awk '{print $2}')\n"
        '    echo "torch==${TORCH_VER}" > /tmp/torch_constraint.txt\n'
        '    echo "transformers>=4.51.0,<4.53.0" >> /tmp/torch_constraint.txt\n'
        "\n"
        "    # Install Slime (training framework) — constrain torch to NGC version\n"
        "    git clone https://github.com/THUDM/slime.git /opt/slime\n"
        "    cd /opt/slime && pip install -e . -c /tmp/torch_constraint.txt\n"
        "\n"
        "    # Install mbridge (model bridge for HF↔Megatron weight conversion)\n"
        "    pip install mbridge -c /tmp/torch_constraint.txt\n"
        "\n"
        "    # Install Megatron-LM at the commit Slime officially ships with.\n"
        "    # Commit 3714d81d is what Slime's Docker image uses; newer commits\n"
        "    # pass a 'config' kwarg to model_provider which Slime doesn't accept.\n"
        "    # --no-deps: megatron-core requires torch>=2.6.0 but NGC's\n"
        "    # 2.6.0a0 pre-release doesn't satisfy that constraint.\n"
        "    git clone https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM\n"
        "    cd /opt/Megatron-LM && git checkout 3714d81d\n"
        "    pip install -e . --no-deps\n"
        "\n"
        "    # Install SGLang without its dependency tree.  sglang requires\n"
        "    # torch>=2.9 but NGC ships 2.6.0a0 — pip can't resolve that,\n"
        "    # so --no-deps is mandatory.  Then install the critical runtime\n"
        "    # deps separately (they don't drag in torch).\n"
        "    pip install sglang sglang-router --no-deps\n"
        "    pip install openai pybase64 partial_json_parser interegular outlines\n"
        "\n"
        "    # Install Polar\n"
        "    pip install -e /opt/polar\n"
        "\n"
        "    # Apply SGLang patch (adds token IDs to logprobs for serving).\n"
        "    # Non-fatal: the training workflow uses SGLang through Slime's\n"
        "    # internal APIs, not the OpenAI chat endpoint the patch modifies.\n"
        "    bash /opt/polar/scripts/patch/patch_sglang.sh || echo 'WARNING: SGLang patch skipped (version mismatch)'\n"
        "\n"
        "    # Apply Slime patch (adds external advantage estimator)\n"
        "    bash /opt/polar/scripts/patch/patch_slime.sh\n"
        "\n"
        "%environment\n"
        '    export PYTHONPATH="/opt/Megatron-LM:/opt/polar/src:${PYTHONPATH:-}"\n'
        "    export CUDA_DEVICE_MAX_CONNECTIONS=1\n"
        "    # Prevent user-site packages (~/.local) from shadowing container packages\n"
        "    export PYTHONNOUSERSITE=1\n"
        "    # CUDA forward-compat: allow newer toolkit to work with older drivers\n"
        '    export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"\n'
        "\n"
        "%labels\n"
        "    io.polar.example train\n"
        "    io.polar.framework slime-grpo\n"
    )
