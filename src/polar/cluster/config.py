"""Unified cluster configuration model.

Parses a ``cluster.yaml`` file into a typed Pydantic model. The config is
backend-agnostic: the same YAML schema works for local, SLURM, and (future)
K8s backends — only the ``backend`` field and its corresponding connection
section differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, model_validator


class SlurmConnection(BaseModel):
    """SLURM-specific connection details (only required when ``backend: slurm``)."""

    login_node: str = ""
    account: str = ""
    partition: str = ""


class ClusterPaths(BaseModel):
    """Filesystem paths on the target cluster / machine."""

    workspace: str = ""
    polar_root: str = ""
    code: str = ""
    sif_dir: str = ""
    results: str = ""
    venv: str = ""
    apptainer_bin_dir: str = ""
    cuda_home: str = ""

    @model_validator(mode="after")
    def _derive_paths(self) -> "ClusterPaths":
        if self.workspace:
            if not self.polar_root:
                self.polar_root = f"{self.workspace}/polar"
            if not self.code:
                self.code = f"{self.polar_root}/ProRL-Agent-Server"
            if not self.sif_dir:
                self.sif_dir = f"{self.polar_root}/sif_images"
            if not self.results:
                self.results = f"{self.polar_root}/results"
            if not self.venv:
                self.venv = f"{self.polar_root}/.venv"
        return self


class ModelConfig(BaseModel):
    """LLM model configuration for vLLM."""

    name: str = "Qwen/Qwen3.5-27B"
    tensor_parallel_size: int = 8
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 64
    tool_call_parser: str = "qwen3_xml"


class TaskConfig(BaseModel):
    """Default task / example settings."""

    example: str = "calculator"
    harness: str = "opencode"
    num_rollouts: int = 4
    timeout_seconds: float = 900.0
    instance_ids: list[str] = []


class ResourceConfig(BaseModel):
    """Compute resource allocation."""

    nodes: int = 1
    gpus_per_node: int = 8
    cpus_per_task: int = 64
    mem: str = "512G"
    time: str = "04:00:00"


class PortConfig(BaseModel):
    """Service port assignments."""

    vllm: int = 18000
    rollout: int = 18080
    gateway_base: int = 18100


class GatewayTuning(BaseModel):
    """Gateway worker pool sizing."""

    max_init_workers: int = 8
    max_run_workers: int = 4
    max_postrun_workers: int = 4
    ready_buffer_target: int = 4


class TrainConfig(BaseModel):
    """RL training configuration for Slime + Megatron GRPO."""

    polar_config: str = ""
    prompt_data: str = ""
    hf_checkpoint: str = "Qwen/Qwen3-4B"
    torch_dist_dir: str = ""
    save_dir: str = ""
    model_args: list[str] = [
        "--swiglu",
        "--num-layers", "36",
        "--hidden-size", "2560",
        "--ffn-hidden-size", "9728",
        "--num-attention-heads", "32",
        "--group-query-attention",
        "--num-query-groups", "8",
        "--use-rotary-position-embeddings",
        "--disable-bias-linear",
        "--normalization", "RMSNorm",
        "--norm-epsilon", "1e-6",
        "--rotary-base", "1000000",
        "--vocab-size", "151936",
        "--kv-channels", "128",
        "--qk-layernorm",
    ]
    num_rollouts: int = 5
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 16
    global_batch_size: int = 32
    actor_gpus: int = 4
    rollout_gpus: int = 4
    tp_size: int = 2
    sglang_router_port: int = 9000
    ray_port: int = 6379
    ray_dashboard_port: int = 8265
    extra_args: list[str] = []
    wandb_project: str = ""
    wandb_exp_name: str = ""
    wandb_group: str = ""


class ClusterConfig(BaseModel):
    """Top-level cluster configuration.

    The ``backend`` field selects which deployment backend to use:
    ``"local"``, ``"slurm"``, or ``"k8s"`` (future).
    """

    backend: Literal["local", "slurm", "k8s"] = "slurm"
    slurm: SlurmConnection = SlurmConnection()
    paths: ClusterPaths = ClusterPaths()
    model: ModelConfig = ModelConfig()
    task: TaskConfig = TaskConfig()
    resources: ResourceConfig = ResourceConfig()
    ports: PortConfig = PortConfig()
    gateway: GatewayTuning = GatewayTuning()
    train: TrainConfig = TrainConfig()

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path) -> "ClusterConfig":
        """Load configuration from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Cluster config not found: {p}")
        with p.open() as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Cluster config must be a YAML mapping: {p}")
        # Support legacy configs that use 'cluster' section for slurm fields
        if "cluster" in raw and "slurm" not in raw:
            raw["slurm"] = raw.pop("cluster")
        return cls.model_validate(raw)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def sbatch_export_vars(self) -> dict[str, str]:
        """Build the flat env-var dict passed to ``sbatch --export``."""
        v: dict[str, str] = {
            "POLAR_CODE": self.paths.code,
            "POLAR_WORKSPACE": self.paths.workspace,
            "EXAMPLE": self.task.example,
            "HARNESS": self.task.harness,
            "MODEL_NAME": self.model.name,
            "MODEL_PATH": self.model.name,
            "TENSOR_PARALLEL_SIZE": str(self.model.tensor_parallel_size),
            "NUM_ROLLOUTS": str(self.task.num_rollouts),
            "TIMEOUT_SECONDS": str(self.task.timeout_seconds),
            "VLLM_PORT": str(self.ports.vllm),
            "ROLLOUT_PORT": str(self.ports.rollout),
            "GATEWAY_BASE_PORT": str(self.ports.gateway_base),
            "MAX_INIT_WORKERS": str(self.gateway.max_init_workers),
            "MAX_RUN_WORKERS": str(self.gateway.max_run_workers),
            "MAX_POSTRUN_WORKERS": str(self.gateway.max_postrun_workers),
            "READY_BUFFER_TARGET": str(self.gateway.ready_buffer_target),
            "GPU_MEMORY_UTILIZATION": str(self.model.gpu_memory_utilization),
            "MAX_MODEL_LEN": str(self.model.max_model_len),
            "MAX_NUM_SEQS": str(self.model.max_num_seqs),
            "TOOL_CALL_PARSER": self.model.tool_call_parser,
        }
        if self.task.instance_ids:
            v["INSTANCE_IDS"] = ",".join(self.task.instance_ids)
        if self.paths.apptainer_bin_dir:
            v["APPTAINER_BIN_DIR"] = self.paths.apptainer_bin_dir
        if self.paths.cuda_home:
            v["CUDA_HOME"] = self.paths.cuda_home
        # swe_agent's swerex needs --fakeroot in Apptainer for chown support
        if self.task.harness == "swe_agent":
            v["RUNTIME_FAKEROOT"] = "true"
        return v

    def sbatch_serve_export_vars(self) -> dict[str, str]:
        """Build env-var dict for serve-only sbatch (no task-specific vars)."""
        v: dict[str, str] = {
            "POLAR_CODE": self.paths.code,
            "POLAR_WORKSPACE": self.paths.workspace,
            "MODEL_NAME": self.model.name,
            "MODEL_PATH": self.model.name,
            "TENSOR_PARALLEL_SIZE": str(self.model.tensor_parallel_size),
            "VLLM_PORT": str(self.ports.vllm),
            "ROLLOUT_PORT": str(self.ports.rollout),
            "GATEWAY_BASE_PORT": str(self.ports.gateway_base),
            "MAX_INIT_WORKERS": str(self.gateway.max_init_workers),
            "MAX_RUN_WORKERS": str(self.gateway.max_run_workers),
            "MAX_POSTRUN_WORKERS": str(self.gateway.max_postrun_workers),
            "READY_BUFFER_TARGET": str(self.gateway.ready_buffer_target),
            "GPU_MEMORY_UTILIZATION": str(self.model.gpu_memory_utilization),
            "MAX_MODEL_LEN": str(self.model.max_model_len),
            "MAX_NUM_SEQS": str(self.model.max_num_seqs),
            "TOOL_CALL_PARSER": self.model.tool_call_parser,
        }
        if self.paths.apptainer_bin_dir:
            v["APPTAINER_BIN_DIR"] = self.paths.apptainer_bin_dir
        if self.paths.cuda_home:
            v["CUDA_HOME"] = self.paths.cuda_home
        return v

    def sbatch_train_export_vars(self) -> dict[str, str]:
        """Build env-var dict for the training sbatch job."""
        t = self.train
        v: dict[str, str] = {
            "POLAR_CODE": self.paths.code,
            "POLAR_WORKSPACE": self.paths.workspace,
            "POLAR_CONFIG_PATH": t.polar_config,
            "PROMPT_DATA": t.prompt_data,
            "HF_CHECKPOINT": t.hf_checkpoint,
            "MODEL_NAME": t.hf_checkpoint,
            "TRAIN_NUM_ROLLOUTS": str(t.num_rollouts),
            "ROLLOUT_BATCH_SIZE": str(t.rollout_batch_size),
            "N_SAMPLES_PER_PROMPT": str(t.n_samples_per_prompt),
            "GLOBAL_BATCH_SIZE": str(t.global_batch_size),
            "ACTOR_GPUS": str(t.actor_gpus),
            "ROLLOUT_GPUS": str(t.rollout_gpus),
            "TP_SIZE": str(t.tp_size),
            "SGLANG_ROUTER_PORT": str(t.sglang_router_port),
            "RAY_PORT": str(t.ray_port),
            "RAY_DASHBOARD_PORT": str(t.ray_dashboard_port),
            "ROLLOUT_PORT": str(self.ports.rollout),
            "GATEWAY_BASE_PORT": str(self.ports.gateway_base),
            "MAX_INIT_WORKERS": str(self.gateway.max_init_workers),
            "MAX_RUN_WORKERS": str(self.gateway.max_run_workers),
            "MAX_POSTRUN_WORKERS": str(self.gateway.max_postrun_workers),
            "READY_BUFFER_TARGET": str(self.gateway.ready_buffer_target),
        }
        if t.torch_dist_dir:
            v["TORCH_DIST_DIR"] = t.torch_dist_dir
        if t.save_dir:
            v["TRAIN_SAVE_DIR"] = t.save_dir
        if t.model_args:
            v["MODEL_ARGS"] = " ".join(t.model_args)
        if t.extra_args:
            v["EXTRA_TRAIN_ARGS"] = " ".join(t.extra_args)
        if t.wandb_project:
            v["WANDB_PROJECT"] = t.wandb_project
        if t.wandb_exp_name:
            v["WANDB_EXP_NAME"] = t.wandb_exp_name
        if t.wandb_group:
            v["WANDB_GROUP"] = t.wandb_group
        if self.paths.apptainer_bin_dir:
            v["APPTAINER_BIN_DIR"] = self.paths.apptainer_bin_dir
        if self.paths.cuda_home:
            v["CUDA_HOME"] = self.paths.cuda_home
        return v

    def apply_overrides(self, overrides: dict[str, Any]) -> "ClusterConfig":
        """Return a new config with *overrides* deep-merged on top."""
        data = self.model_dump()
        _deep_merge(data, overrides)
        return ClusterConfig.model_validate(data)


def _deep_merge(base: dict, overlay: dict) -> None:
    """Recursively merge *overlay* into *base* in place."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
