from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import yaml
import pytest

from polar.config import TopologyConfig
from slime_bridge.config import resolve_polar_slime_config


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "spilot_router_slime_grpo"
SCRIPT = EXAMPLE / "run_forced_route_eval.py"
RUN_SCRIPT = EXAMPLE / "run_forced_route_eval.sh"
SUBMIT_SCRIPT = EXAMPLE / "submit_forced_route_eval.sh"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("spilot_forced_eval_launcher", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _assets(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    for relative in ("agent_cli/opt_node", "mini_swe_agent_runtime", "tmax-15k-sif"):
        (data_root / relative).mkdir(parents=True)
    task_dir = tmp_path / "task"
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True)
    sif = data_root / "tmax-15k-sif" / "task.sif"
    sif.write_bytes(b"test-sif")
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "fix it"}],
                "metadata": {
                    "task_name": "task",
                    "sif_path": str(sif),
                    "tests_dir": str(tests_dir),
                    "workdir": "/root",
                    "timeout_seconds": 840,
                    "agent_timeout": 600,
                    "verifier_timeout": 120,
                    "allow_internet": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return data_root, dataset


def test_services_only_topology_is_loopback_zero_actor_and_exact_pool(tmp_path: Path) -> None:
    document = module.build_topology(
        rollout_port=18080,
        gateway_port=18100,
        service_dir=tmp_path,
        pool_base_url="https://inference.example/v1",
        max_concurrency=4,
    )
    topology = TopologyConfig.model_validate(document)
    node = topology.gateway.nodes[0]

    assert topology.rollout.host == "127.0.0.1"
    assert topology.rollout.public_url == "http://127.0.0.1:18080"
    assert node.host == "127.0.0.1"
    assert node.public_url == "http://127.0.0.1:18100"
    assert node.model_served == "eval-only/forced-route-no-actor"
    assert node.inference_base_url == "http://127.0.0.1:9"
    assert topology.gateway.completion_persistence.enabled is False
    assert (node.max_init_workers, node.max_run_workers, node.max_postrun_workers) == (4, 4, 4)
    assert [(candidate.alias, candidate.model) for candidate in node.model_pool] == [
        ("pool/qwen3.6-27b", "nvidia/qwen/qwen3.6-27b"),
        ("pool/gpt-5.5", "openai/openai/gpt-5.5"),
    ]
    assert {candidate.api_key_env for candidate in node.model_pool} == {
        "POLAR_NVIDIA_API_KEY"
    }


def test_rendered_config_reuses_current_pool_parity_and_job_local_uds(tmp_path: Path) -> None:
    data_root, _ = _assets(tmp_path)
    uds_root = Path("/tmp/polar-forced-test")
    document = module.render_polar_config(
        data_root=data_root,
        rollout_port=18080,
        gateway_port=18100,
        uds_root=uds_root,
        proxy_url="http://proxy.example:3128",
        agent_timeout_seconds=3300,
    )
    args = SimpleNamespace(**document, rollout_batch_size=4, n_samples_per_prompt=1)
    config = resolve_polar_slime_config(args)
    task = config.task_template

    assert config.rollout_server_url == "http://127.0.0.1:18080"
    assert config.request_timeout == 5100
    assert config.task_timeout_floor == 4500
    assert config.eval_agent_timeout == 3300
    assert task["runtime"]["network"] == "none"
    assert f"{uds_root}/gateway:/polar/gateway:ro" in task["runtime"]["kwargs"]["volumes"]
    assert task["runtime"]["internet_volumes"] == [
        f"{uds_root}/proxy:/polar/proxy:ro"
    ]
    assert task["agent"]["model_name"] == "eval-only/forced-route-no-actor"
    settings = task["agent"]["settings"]
    assert settings["model_pool"]["M0"]["model_kwargs"] == {
        "max_tokens": 16_384,
        "temperature": 1.0,
        "top_p": 1.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    assert settings["model_pool"]["M1"]["model_kwargs"] == {
        "max_completion_tokens": 16_384
    }
    assert settings["pool_step_limit"] == 64
    assert settings["pool_model_retry_attempts"] == 5


def test_dry_run_writes_auditable_secret_free_plan(tmp_path: Path, monkeypatch) -> None:
    data_root, dataset = _assets(tmp_path)
    output_dir = tmp_path / "benchmark"
    service_dir = tmp_path / "services"
    key_sentinel = "nvapi-secret-sentinel"
    token_sentinel = "control-secret-sentinel-0123456789abcdef"
    monkeypatch.setenv("POLAR_NVIDIA_API_KEY", key_sentinel)
    monkeypatch.setenv("POLAR_CONTROL_PLANE_TOKEN", token_sentinel)
    monkeypatch.setenv("http_proxy", "http://proxy.example:3128")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    result = module.main(
        [
            "--dry-run",
            "--i-understand-eval-only",
            "--run-id",
            "paired-dry-run",
            "--data",
            str(dataset),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--service-dir",
            str(service_dir),
            "--max-tasks",
            "1",
        ]
    )

    assert result == 0
    assert not output_dir.exists()
    assert {path.name for path in service_dir.iterdir()} == {
        "launcher.json",
        "polar_config.yaml",
        "topology.yaml",
    }
    persisted = "\n".join(path.read_text() for path in service_dir.iterdir())
    assert key_sentinel not in persisted
    assert token_sentinel not in persisted
    assert re.search(r"nvapi-[A-Za-z0-9_-]+", persisted) is None
    manifest = json.loads((service_dir / "launcher.json").read_text())
    assert manifest["actor_training"] is False
    assert manifest["actor_invoked"] is False
    assert manifest["control_token_persisted"] is False
    assert manifest["model_credential_persisted"] is False
    assert manifest["rollout_url"] == "http://127.0.0.1:18080"
    TopologyConfig.load(service_dir / "topology.yaml")
    assert isinstance(yaml.safe_load((service_dir / "polar_config.yaml").read_text()), dict)


def test_proxy_target_parser_handles_default_and_ipv6_ports() -> None:
    assert module.parse_proxy_target("http://cache.example:3128") == "cache.example:3128"
    assert module.parse_proxy_target("https://cache.example") == "cache.example:443"
    assert module.parse_proxy_target("http://[::1]:8080") == "[::1]:8080"
    with pytest.raises(module.LauncherError, match="credential-bearing"):
        module.parse_proxy_target("http://user:secret@cache.example:3128")


def test_job_local_uds_paths_fit_linux_socket_limit() -> None:
    root = Path("/tmp/polar-forced-eval-12345678901234567890-1234567890")
    assert len(os.fsencode(root / "gateway" / "gateway.sock")) <= 107
    assert len(os.fsencode(root / "proxy" / "proxy.sock")) <= 107


def test_registration_readiness_accepts_nodes_list(monkeypatch) -> None:
    responses = iter(
        [
            [{"node_id": "localhost-node-01", "healthy": False}],
            [
                {
                    "node_id": "localhost-node-01",
                    "healthy": True,
                    "gateway_url": "http://127.0.0.1:18100",
                }
            ],
        ]
    )

    class Process:
        def poll(self):
            return None

    monkeypatch.setattr(module, "_http_json", lambda _url: next(responses))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    result = module.wait_http(
        "gateway registration",
        "http://127.0.0.1:18080/nodes",
        Process(),
        predicate=lambda document: (
            isinstance(document, list)
            and len(document) == 1
            and document[0].get("healthy") is True
        ),
        timeout=1,
    )

    assert result[0]["healthy"] is True


def test_slurm_entrypoint_is_one_node_and_has_no_training_stack() -> None:
    run_text = RUN_SCRIPT.read_text()
    submit_text = SUBMIT_SCRIPT.read_text()
    for script in (RUN_SCRIPT, SUBMIT_SCRIPT):
        completed = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    assert "#SBATCH --nodes=1" in run_text
    assert "#SBATCH --ntasks=1" in run_text
    assert "--container-image=" in run_text
    assert "--container-mounts=" in run_text
    assert 'source "${CREDENTIAL_PATH}"' in run_text
    assert 'rm -f -- "${CREDENTIAL_PATH}"' in run_text
    assert "export SLURM_EXPORT_ENV=ALL" in run_text
    assert "trap cleanup EXIT" in run_text
    assert 'run_child "${SRUN_BIN}"' in run_text
    assert "printf 'export SRUN_BIN=%q" in submit_text
    assert 'FORCED_EVAL_GPUS="${FORCED_EVAL_GPUS:-0}"' in submit_text
    assert 'SBATCH_ARGS+=(--gres="gpu:${FORCED_EVAL_GPUS}")' in submit_text
    assert "--gpus" not in run_text
    assert "--gres" not in run_text
    combined = run_text.lower() + submit_text.lower()
    assert "ray start" not in combined
    assert "train_async" not in combined
    assert "serve_sglang" not in combined
    assert "megatron" not in combined
    assert "checkpoint" not in combined


def test_submit_dry_run_is_secret_free_cpu_default_with_explicit_gpu_fallback(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text("{}\n")
    secret = "nvapi-submit-secret-sentinel"

    def invoke(output: Path, gpus: int) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment.update(
            {
                "POLAR_NVIDIA_API_KEY": secret,
                "SUBMIT_DRY_RUN": "1",
                "FORCED_EVAL_GPUS": str(gpus),
            }
        )
        return subprocess.run(
            [
                "bash",
                str(SUBMIT_SCRIPT),
                "--i-understand-eval-only",
                "--run-id",
                f"dry-{gpus}",
                "--data",
                str(dataset),
                "--output-dir",
                str(output),
                "--max-tasks",
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    cpu = invoke(tmp_path / "cpu", 0)
    assert cpu.returncode == 0, cpu.stderr
    assert secret not in cpu.stdout + cpu.stderr
    assert "--gres" not in cpu.stdout
    assert "POLAR_FORCED_EVAL_ENV_FILE=" in cpu.stdout
    assert not (tmp_path / "cpu.submit").exists()

    fallback = invoke(tmp_path / "gpu", 1)
    assert fallback.returncode == 0, fallback.stderr
    assert secret not in fallback.stdout + fallback.stderr
    assert "--gres=gpu:1" in fallback.stdout
    assert not (tmp_path / "gpu.submit").exists()


def test_allocation_entrypoint_sources_mode_600_envelope_then_deletes_it(
    tmp_path: Path,
) -> None:
    credential = tmp_path / "credentials.env"
    secret = "nvapi-entrypoint-secret-sentinel"
    credential.write_text(f"export POLAR_NVIDIA_API_KEY={secret}\n")
    credential.chmod(0o600)
    data_root = tmp_path / "data"
    data_root.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NUM_NODES": "1",
            "POLAR_FORCED_EVAL_ENV_FILE": str(credential),
            "POLAR_DATA_ROOT": str(data_root),
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--i-understand-eval-only"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "Pyxis image does not exist" in completed.stderr
    assert secret not in completed.stdout + completed.stderr
    assert not credential.exists()


def test_submitter_persists_only_private_envelope_until_allocation_starts(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text("{}\n")
    output = tmp_path / "submitted"
    args_capture = tmp_path / "sbatch.args"
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" >\"$FAKE_SBATCH_ARGS\"\nprintf '424242\\n'\n"
    )
    fake_sbatch.chmod(0o700)
    secret = "nvapi-private-envelope-sentinel"
    environment = dict(os.environ)
    environment.update(
        {
            "POLAR_NVIDIA_API_KEY": secret,
            "POLAR_DATA_ROOT": str(tmp_path / "data"),
            "SBATCH_BIN": str(fake_sbatch),
            "FAKE_SBATCH_ARGS": str(args_capture),
        }
    )

    completed = subprocess.run(
        [
            "bash",
            str(SUBMIT_SCRIPT),
            "--i-understand-eval-only",
            "--run-id",
            "submitted",
            "--data",
            str(dataset),
            "--output-dir",
            str(output),
            "--max-tasks",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert secret not in completed.stdout + completed.stderr
    credential = tmp_path / "submitted.submit" / "credentials.env"
    assert stat.S_IMODE(credential.stat().st_mode) == 0o600
    assert secret in credential.read_text()
    assert secret not in args_capture.read_text()
    assert "POLAR_FORCED_EVAL_ENV_FILE=" in args_capture.read_text()
    assert "SPILOT_FORCED_EVAL_PROJECT_ROOT=" in credential.read_text()
    assert (tmp_path / "submitted.submit" / "job_id").read_text().strip() == "424242"


def test_spooled_allocation_script_uses_submitted_project_root(tmp_path: Path) -> None:
    capture = tmp_path / "srun.args"
    fake_srun = tmp_path / "srun"
    fake_srun.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" >\"$FAKE_SRUN_ARGS\"\n"
    )
    fake_srun.chmod(0o700)
    spooled_script = tmp_path / "slurm-spool" / "script"
    spooled_script.parent.mkdir()
    spooled_script.write_text(RUN_SCRIPT.read_text())
    spooled_script.chmod(0o700)
    credential = tmp_path / "credentials.env"
    credential.write_text(
        "export POLAR_NVIDIA_API_KEY=nvapi-spool-sentinel\n"
        f"export SPILOT_FORCED_EVAL_PROJECT_ROOT={ROOT}\n"
        f"export SRUN_BIN={fake_srun}\n"
    )
    credential.chmod(0o600)
    train_image = tmp_path / "train.sqsh"
    train_image.write_bytes(b"test")
    environment = dict(os.environ)
    environment.pop("POLAR_DATA_ROOT", None)
    environment.update(
        {
            "SLURM_JOB_ID": "23456",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_CPUS_PER_TASK": "2",
            "POLAR_FORCED_EVAL_ENV_FILE": str(credential),
            "POLR_TRAIN_SQSH": str(train_image),
            "TMAX_SIF_PYTHON_BIN": "/bin/true",
            "FAKE_SRUN_ARGS": str(capture),
        }
    )

    completed = subprocess.run(
        ["bash", str(spooled_script), "--i-understand-eval-only"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = capture.read_text().splitlines()
    assert f"--container-workdir={ROOT}" in arguments
    assert str(EXAMPLE / "run_forced_route_eval.sh") in arguments
    assert str(spooled_script) not in arguments
    assert not credential.exists()


def test_allocation_reexports_sourced_key_to_pyxis_step_without_control_token(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "srun.env"
    fake_srun = tmp_path / "srun"
    fake_srun.write_text(
        "#!/bin/sh\n"
        "printf '%s|%s|%s\\n' \"${POLAR_NVIDIA_API_KEY:-}\" "
        "\"${POLAR_CONTROL_PLANE_TOKEN:-}\" \"${SLURM_EXPORT_ENV:-}\" "
        ">\"$FAKE_SRUN_ENV\"\n"
    )
    fake_srun.chmod(0o700)
    credential = tmp_path / "credentials.env"
    secret = "nvapi-step-secret-sentinel"
    credential.write_text(
        f"export POLAR_NVIDIA_API_KEY={secret}\nexport SRUN_BIN={fake_srun}\n"
    )
    credential.chmod(0o600)
    train_image = tmp_path / "train.sqsh"
    train_image.write_bytes(b"test")
    data_root = tmp_path / "data"
    data_root.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_CPUS_PER_TASK": "2",
            "POLAR_FORCED_EVAL_ENV_FILE": str(credential),
            "POLAR_DATA_ROOT": str(data_root),
            "POLR_TRAIN_SQSH": str(train_image),
            "TMAX_SIF_PYTHON_BIN": "/bin/true",
            "FAKE_SRUN_ENV": str(capture),
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--i-understand-eval-only"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert secret not in completed.stdout + completed.stderr
    assert capture.read_text().strip() == f"{secret}||ALL"
    assert not credential.exists()


def test_inner_pyxis_entrypoint_does_not_require_host_srun_path(tmp_path: Path) -> None:
    train_image = tmp_path / "train.sqsh"
    train_image.write_bytes(b"test")
    data_root = tmp_path / "data"
    data_root.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "SLURM_JOB_ID": "54321",
            "SLURM_JOB_NUM_NODES": "1",
            "SPILOT_FORCED_EVAL_IN_CONTAINER": "1",
            "POLAR_NVIDIA_API_KEY": "nvapi-inner-sentinel",
            "POLAR_DATA_ROOT": str(data_root),
            "POLR_TRAIN_SQSH": str(train_image),
            "TMAX_SIF_PYTHON_BIN": "/bin/true",
            "SRUN_BIN": "/host-only/path/to/srun",
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--i-understand-eval-only"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


def test_child_environments_scope_each_credential(monkeypatch) -> None:
    monkeypatch.setenv("POLAR_NVIDIA_API_KEY", "inherited-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "inherited-fallback")
    monkeypatch.setenv("POLAR_CONTROL_PLANE_TOKEN", "inherited-token")

    rollout = module.scoped_environment(control_token="control")
    gateway = module.scoped_environment(control_token="control", nvidia_key="key")
    tunnel = module.scoped_environment()

    assert rollout["POLAR_CONTROL_PLANE_TOKEN"] == "control"
    assert "POLAR_NVIDIA_API_KEY" not in rollout
    assert gateway["POLAR_CONTROL_PLANE_TOKEN"] == "control"
    assert gateway["POLAR_NVIDIA_API_KEY"] == "key"
    assert "NVIDIA_API_KEY" not in gateway
    assert "POLAR_CONTROL_PLANE_TOKEN" not in tunnel
    assert "POLAR_NVIDIA_API_KEY" not in tunnel
