from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import signal
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
    (data_root / "agent_cli" / "opt_node" / "node").write_bytes(b"test-node-runtime")
    (data_root / "mini_swe_agent_runtime" / "VERSION").write_text("test-runtime\n")
    container_dir = data_root / "container"
    container_dir.mkdir()
    (container_dir / "flappydora-ubuntu22.04-cuda13.3.sqsh").write_bytes(
        b"test-train-container"
    )
    tokenizer_dir = data_root / "checkpoints" / "Qwen3.5-9B"
    tokenizer_dir.mkdir(parents=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (tokenizer_dir / name).write_text("{}" if name.endswith(".json") else "test")
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


def _timing_only_dataset(path: Path, *, count: int = 1) -> None:
    row = json.dumps(
        {
            "metadata": {
                "timeout_seconds": 840,
                "verifier_timeout": 120,
            }
        }
    )
    path.write_text((row + "\n") * count)


def test_services_only_topology_is_loopback_zero_actor_and_exact_pool(tmp_path: Path) -> None:
    document = module.build_topology(
        rollout_port=18080,
        gateway_port=18100,
        tokenizer_port=18200,
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
    assert node.inference_base_url == "http://127.0.0.1:18200"
    assert topology.gateway.completion_persistence.enabled is False
    assert (node.max_init_workers, node.max_run_workers, node.max_postrun_workers) == (4, 4, 4)
    assert [(candidate.alias, candidate.model) for candidate in node.model_pool] == [
        ("pool/qwen3.6-27b", "nvidia/qwen/qwen3.6-27b"),
        ("pool/gpt-5.5", "openai/openai/gpt-5.5"),
    ]
    assert {candidate.api_key_env for candidate in node.model_pool} == {"POLAR_NVIDIA_API_KEY"}
    assert {candidate.max_active_episodes for candidate in node.model_pool} == {4}

    with_baseline = module.build_topology(
        rollout_port=18080,
        gateway_port=18100,
        tokenizer_port=18200,
        service_dir=tmp_path,
        pool_base_url="https://inference.example/v1",
        max_concurrency=4,
        include_qwen35_baseline=True,
    )
    baseline_node = TopologyConfig.model_validate(with_baseline).gateway.nodes[0]
    assert [(candidate.alias, candidate.model) for candidate in baseline_node.model_pool] == [
        ("pool/qwen3.6-27b", "nvidia/qwen/qwen3.6-27b"),
        ("pool/gpt-5.5", "openai/openai/gpt-5.5"),
        ("pool/qwen3.5-9b-baseline", "nvidia/qwen/qwen3.5-9b"),
    ]
    training_topology = yaml.safe_load((EXAMPLE / "topology.yaml").read_text())
    assert "pool/qwen3.5-9b-baseline" not in json.dumps(training_topology)


def test_protected_runtime_contract_requires_persistent_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in module.REQUIRED_PROTECTED_RUNTIME_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    contract = module.required_isolation_environment()

    assert contract == {
        name: "1" for name in module.REQUIRED_PROTECTED_RUNTIME_VARIABLES
    }
    assert contract["POLAR_APPTAINER_PERSISTENT_BROKER"] == "1"
    assert contract["POLAR_APPTAINER_CLEANENV"] == "1"

    monkeypatch.setenv("POLAR_APPTAINER_PERSISTENT_BROKER", "0")
    with pytest.raises(
        module.LauncherError,
        match=r"POLAR_APPTAINER_PERSISTENT_BROKER='0'",
    ):
        module.required_isolation_environment()


def test_rendered_config_reuses_current_pool_parity_and_job_local_uds(tmp_path: Path) -> None:
    data_root, dataset = _assets(tmp_path)
    uds_root = Path("/tmp/polar-forced-test")
    document = module.render_polar_config(
        data_root=data_root,
        rollout_port=18080,
        gateway_port=18100,
        uds_root=uds_root,
        proxy_url="http://proxy.example:3128",
        agent_timeout_seconds=3300,
        pool_timeout_seconds=1200,
        runner_total_timeout_seconds=3000,
        task_timeout_floor_seconds=3420,
        outer_task_envelope_seconds=3420,
    )
    args = SimpleNamespace(**document, rollout_batch_size=4, n_samples_per_prompt=1)
    config = resolve_polar_slime_config(args)
    task = config.task_template

    assert config.rollout_server_url == "http://127.0.0.1:18080"
    assert config.request_timeout == 4020
    assert config.task_timeout_floor == 3420
    assert config.eval_agent_timeout == 3300
    assert task["runtime"]["network"] == "none"
    assert f"{uds_root}/gateway:/polar/gateway:ro" in task["runtime"]["kwargs"]["volumes"]
    assert task["runtime"]["internet_volumes"] == [f"{uds_root}/proxy:/polar/proxy:ro"]
    assert task["agent"]["model_name"] == "eval-only/forced-route-no-actor"
    settings = task["agent"]["settings"]
    assert settings["model_pool"]["M0"]["model_kwargs"] == {
        "max_tokens": 16_384,
        "temperature": 1.0,
        "top_p": 1.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    assert settings["model_pool"]["M1"]["model_kwargs"] == {"max_completion_tokens": 16_384}
    assert settings["pool_step_limit"] == 64
    assert settings["pool_model_retry_attempts"] == 5
    assert settings["pool_episode_admission_enabled"] is True
    assert settings["pool_episode_admission_wait_budget_seconds"] == 300
    assert settings["max_pool_calls"] == 1
    assert settings["pool_timeout_seconds"] == 1200
    assert settings["total_timeout_seconds"] == 3000

    baseline_document = module.render_polar_config(
        data_root=data_root,
        rollout_port=18080,
        gateway_port=18100,
        uds_root=uds_root,
        proxy_url="http://proxy.example:3128",
        agent_timeout_seconds=3300,
        pool_timeout_seconds=1200,
        runner_total_timeout_seconds=3000,
        task_timeout_floor_seconds=3420,
        outer_task_envelope_seconds=3420,
        include_qwen35_baseline=True,
    )
    baseline_settings = baseline_document["polar_task_template"]["agent"]["settings"]
    assert baseline_settings["model_pool"]["M2"]["model"] == "pool/qwen3.5-9b-baseline"
    assert baseline_settings["model_pool"]["M2"]["model_kwargs"] == (
        baseline_settings["model_pool"]["M0"]["model_kwargs"]
    )


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
        "semantic_identity.json",
        "source_snapshot",
        "topology.yaml",
    }
    persisted = "\n".join(
        path.read_text() for path in service_dir.iterdir() if path.is_file()
    )
    assert key_sentinel not in persisted
    assert token_sentinel not in persisted
    assert re.search(r"nvapi-[A-Za-z0-9_-]+", persisted) is None
    manifest = json.loads((service_dir / "launcher.json").read_text())
    assert manifest["actor_training"] is False
    assert manifest["actor_invoked"] is False
    assert manifest["control_token_persisted"] is False
    assert manifest["model_credential_persisted"] is False
    assert manifest["rollout_url"] == "http://127.0.0.1:18080"
    assert manifest["gateway_url"] == "http://127.0.0.1:18100"
    assert manifest["tokenizer_url"] == "http://127.0.0.1:18200"
    assert manifest["source_snapshot_sha256"]
    assert manifest["semantic_identity_sha256"]
    assert manifest["runtime_isolation"]["POLAR_APPTAINER_PERSISTENT_BROKER"] == "1"
    assert manifest["runtime_isolation"]["POLAR_APPTAINER_CLEANENV"] == "1"
    assert manifest["pool_timeout_seconds"] == 1200
    assert manifest["runner_total_timeout_seconds"] == 3000
    assert manifest["outer_agent_timeout_seconds"] == 3300
    assert manifest["forced_eval_work_item_count"] == 2
    assert manifest["max_paid_attempt_count"] == 2
    assert manifest["forced_eval_waves"] == 1
    assert manifest["max_dataset_task_timeout_seconds"] == 840
    assert manifest["max_verifier_timeout_seconds"] == 120
    assert manifest["task_timeout_floor_seconds"] == 3420
    assert manifest["outer_task_envelope_seconds"] == 3420
    assert manifest["required_slurm_walltime_seconds"] == 5220
    assert manifest["allocated_slurm_walltime_seconds"] is None
    assert manifest["evaluator_max_passes"] == 3
    assert "--pool-timeout-seconds" in manifest["evaluator_command"]
    assert "--runner-total-timeout-seconds" in manifest["evaluator_command"]
    assert "--semantic-identity" in manifest["evaluator_command"]
    assert str(service_dir / "source_snapshot") in manifest["evaluator_command"][1]
    identity = json.loads((service_dir / "semantic_identity.json").read_text())
    assert identity["semantic"]["runtime_isolation"] == manifest["runtime_isolation"]
    assert identity["semantic"]["topology"]["gateway"]["nodes"][0]["model_pool"][0][
        "max_active_episodes"
    ] == 4
    assert key_sentinel not in json.dumps(identity)
    assert token_sentinel not in json.dumps(identity)
    TopologyConfig.load(service_dir / "topology.yaml")
    assert isinstance(yaml.safe_load((service_dir / "polar_config.yaml").read_text()), dict)


def test_launcher_dry_resume_uses_existing_output_and_fresh_service(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root, dataset = _assets(tmp_path)
    output = tmp_path / "existing-output"
    output.mkdir()
    (output / "manifest.json").write_text("{}\n")
    service = tmp_path / "fresh-service"
    monkeypatch.setenv("http_proxy", "http://proxy.example:3128")

    result = module.main(
        [
            "--dry-run",
            "--resume",
            "--i-understand-eval-only",
            "--run-id",
            "resume-run",
            "--data",
            str(dataset),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output),
            "--service-dir",
            str(service),
            "--max-tasks",
            "1",
        ]
    )

    assert result == 0
    launcher = json.loads((service / "launcher.json").read_text())
    assert launcher["resume"] is True
    assert "--resume" in launcher["evaluator_command"]


def test_dry_run_optional_baseline_is_in_plan_and_walltime_matrix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root, dataset = _assets(tmp_path)
    output = tmp_path / "baseline-output"
    service = tmp_path / "baseline-service"
    monkeypatch.setenv("http_proxy", "http://proxy.example:3128")
    result = module.main(
        [
            "--dry-run",
            "--i-understand-eval-only",
            "--include-qwen35-baseline",
            "--run-id",
            "baseline-run",
            "--data",
            str(dataset),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output),
            "--service-dir",
            str(service),
            "--max-tasks",
            "1",
        ]
    )
    assert result == 0
    launcher = json.loads((service / "launcher.json").read_text())
    assert launcher["include_qwen35_baseline"] is True
    assert launcher["forced_eval_candidate_count"] == 3
    assert launcher["forced_eval_work_item_count"] == 3
    assert launcher["max_paid_attempt_count"] == 3
    assert "--include-qwen35-baseline" in launcher["evaluator_command"]
    identity = json.loads((service / "semantic_identity.json").read_text())
    pool = identity["semantic"]["topology"]["gateway"]["nodes"][0]["model_pool"]
    assert pool[2]["alias"] == "pool/qwen3.5-9b-baseline"


def test_semantic_identity_binds_snapshot_topology_tokenizer_and_runtimes(
    tmp_path: Path,
) -> None:
    data_root, dataset = _assets(tmp_path)
    service = tmp_path / "identity-service"
    service.mkdir()
    snapshot_root = service / "source_snapshot"
    snapshot = module.create_source_snapshot(ROOT, snapshot_root)
    assert snapshot == module.source_snapshot_identity(snapshot_root)
    assert snapshot_root.stat().st_mode & stat.S_IWUSR == 0

    tokenizer = data_root / "checkpoints" / "Qwen3.5-9B"
    topology_path = service / "topology.yaml"
    config_path = service / "polar_config.yaml"
    config_path.write_text("test: true\n")
    topology = module.build_topology(
        rollout_port=18080,
        gateway_port=18100,
        tokenizer_port=18200,
        service_dir=service,
        pool_base_url="https://inference.example/v1",
        max_concurrency=4,
    )
    module._write_yaml(topology_path, topology)

    identity = module.build_semantic_identity(
        topology=topology,
        topology_path=topology_path,
        polar_config_path=config_path,
        snapshot_root=snapshot_root,
        snapshot_identity=snapshot,
        tokenizer_path=tokenizer,
        data_root=data_root,
        data_path=dataset,
        start_index=0,
        max_tasks=1,
    )
    module.verify_semantic_identity(identity)

    task_sif = data_root / "tmax-15k-sif" / "task.sif"
    original_sif = task_sif.read_bytes()
    task_sif.write_bytes(b"changed-task-sif")
    with pytest.raises(module.LauncherError, match="SIF"):
        module.verify_semantic_identity(identity)
    task_sif.write_bytes(original_sif)

    tokenizer_asset = tokenizer / "tokenizer.json"
    original_tokenizer = tokenizer_asset.read_text()
    tokenizer_asset.write_text('{"changed": true}')
    with pytest.raises(module.LauncherError, match="tokenizer"):
        module.verify_semantic_identity(identity)
    tokenizer_changed = module.build_semantic_identity(
        topology=topology,
        topology_path=topology_path,
        polar_config_path=config_path,
        snapshot_root=snapshot_root,
        snapshot_identity=snapshot,
        tokenizer_path=tokenizer,
        data_root=data_root,
        data_path=dataset,
        start_index=0,
        max_tasks=1,
    )
    assert tokenizer_changed["semantic_sha256"] != identity["semantic_sha256"]
    tokenizer_asset.write_text(original_tokenizer)

    runtime_file = data_root / "mini_swe_agent_runtime" / "VERSION"
    runtime_file.write_text("changed-runtime\n")
    runtime_changed = module.build_semantic_identity(
        topology=topology,
        topology_path=topology_path,
        polar_config_path=config_path,
        snapshot_root=snapshot_root,
        snapshot_identity=snapshot,
        tokenizer_path=tokenizer,
        data_root=data_root,
        data_path=dataset,
        start_index=0,
        max_tasks=1,
    )
    assert runtime_changed["semantic_sha256"] != identity["semantic_sha256"]

    topology_changed = module.build_topology(
        rollout_port=18080,
        gateway_port=18100,
        tokenizer_port=18200,
        service_dir=service,
        pool_base_url="https://other-endpoint.example/v1",
        max_concurrency=3,
    )
    module._write_yaml(topology_path, topology_changed)
    topology_identity = module.build_semantic_identity(
        topology=topology_changed,
        topology_path=topology_path,
        polar_config_path=config_path,
        snapshot_root=snapshot_root,
        snapshot_identity=snapshot,
        tokenizer_path=tokenizer,
        data_root=data_root,
        data_path=dataset,
        start_index=0,
        max_tasks=1,
    )
    assert topology_identity["semantic_sha256"] != runtime_changed["semantic_sha256"]
    pool = topology_identity["semantic"]["topology"]["gateway"]["nodes"][0]["model_pool"]
    assert {candidate["max_active_episodes"] for candidate in pool} == {3}


def test_gateway_admission_fatal_withholds_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "fatal-output"
    output.mkdir()
    base = {
        "content_integrity": {"status": "verified", "failures": []},
        "final_metrics": {"candidate_metrics": {}},
        "final_metrics_status": "published",
    }
    (output / "manifest.json").write_text(json.dumps(base))
    (output / "summary.json").write_text(json.dumps(base))

    monkeypatch.setattr(
        module,
        "check_gateway_health",
        lambda _url: (_ for _ in ()).throw(
            module.GatewayAdmissionFatalError("retained lease")
        ),
    )

    class Process:
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    owner_fd = os.open(tmp_path / "owner-health.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with pytest.raises(module.GatewayAdmissionFatalError, match="retained lease"):
            module.run_evaluator_passes(
                evaluator_command=["python", "forced_route_eval.py"],
                evaluator_environment={},
                processes=[],
                services=(),
                output_dir=output,
                max_passes=1,
                resume_backoff_seconds=1,
                max_concurrency=1,
                outer_task_envelope_seconds=100,
                max_paid_attempts_per_work=1,
                allocated_slurm_seconds=1000,
                launcher_started_monotonic=module.time.monotonic(),
                owner_fd=owner_fd,
                gateway_health_url="http://127.0.0.1:1/health",
            )
    finally:
        os.close(owner_fd)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["content_integrity"]["status"] == "failed"
    assert summary["final_metrics"] is None
    assert summary["final_metrics_status"] == "withheld_integrity"


def test_gateway_teardown_nonzero_or_kill_is_an_integrity_failure() -> None:
    gateway = SimpleNamespace(returncode=2)
    outcomes = {id(gateway): {"return_code": 2, "killed": False}}
    assert "gateway: rc=2" in module.service_teardown_failure(
        {"gateway": gateway}, outcomes
    )
    outcomes[id(gateway)] = {"return_code": 0, "killed": True}
    assert "killed=True" in module.service_teardown_failure({"gateway": gateway}, outcomes)

    tokenizer = SimpleNamespace(returncode=-signal.SIGTERM)
    assert (
        module.service_teardown_failure(
            {"tokenizer": tokenizer},
            {id(tokenizer): {"return_code": -signal.SIGTERM, "killed": False}},
        )
        is None
    )


def test_gateway_teardown_gets_margin_beyond_internal_sixty_second_bound() -> None:
    class SlowGateway:
        returncode = None

        def __init__(self) -> None:
            self.wait_timeouts: list[float] = []
            self.killed = False

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            pass

        def wait(self, timeout: float) -> int:
            self.wait_timeouts.append(timeout)
            self.returncode = 0
            return 0

        def kill(self) -> None:
            self.killed = True
            self.returncode = -signal.SIGKILL

    gateway = SlowGateway()
    outcomes = module.teardown_process_tree([gateway], {"gateway": gateway})
    assert gateway.wait_timeouts and gateway.wait_timeouts[0] > 80
    assert gateway.killed is False
    assert module.service_teardown_failure({"gateway": gateway}, outcomes) is None


def test_proxy_target_parser_handles_default_and_ipv6_ports() -> None:
    assert module.parse_proxy_target("http://cache.example:3128") == "cache.example:3128"
    assert module.parse_proxy_target("https://cache.example") == "cache.example:443"
    assert module.parse_proxy_target("http://[::1]:8080") == "[::1]:8080"
    with pytest.raises(module.LauncherError, match="credential-bearing"):
        module.parse_proxy_target("http://user:secret@cache.example:3128")


def test_forced_eval_pool_timeout_is_explicit_and_outer_budget_is_fail_closed() -> None:
    common = [
        "--dry-run",
        "--i-understand-eval-only",
        "--run-id",
        "timeout-check",
        "--data",
        "/tmp/eval.jsonl",
        "--output-dir",
        "/tmp/out",
        "--max-tasks",
        "1",
        "--pool-timeout-seconds",
        "1800",
    ]
    parsed = module.parse_args(common)
    assert parsed.pool_timeout_seconds == 1800
    assert parsed.runner_total_timeout_seconds == 3000
    assert parsed.agent_timeout_seconds == 3300

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                *common,
                "--runner-total-timeout-seconds",
                "2200",
            ]
        )
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                *common,
                "--agent-timeout-seconds",
                "2500",
            ]
        )


def test_slurm_walltime_budget_uses_full_work_matrix_and_outer_envelope(
    tmp_path: Path,
) -> None:
    required, work_items, waves = module.required_slurm_walltime_seconds(
        max_tasks=32,
        replicates=1,
        candidate_count=2,
        max_concurrency=4,
        outer_task_envelope_seconds=4500,
        max_paid_attempts_per_work=2,
        margin_seconds=1800,
    )
    assert work_items == 128
    assert waves == 32
    assert required == 145_800

    baseline_required, baseline_work_items, baseline_waves = (
        module.required_slurm_walltime_seconds(
            max_tasks=32,
            replicates=1,
            candidate_count=3,
            max_concurrency=4,
            outer_task_envelope_seconds=3420,
            max_paid_attempts_per_work=1,
            margin_seconds=1800,
        )
    )
    assert baseline_work_items == 96
    assert baseline_waves == 24
    assert baseline_required == 83_880

    data_root, dataset = _assets(tmp_path)
    row = json.loads(dataset.read_text())
    row["metadata"]["timeout_seconds"] = 9000
    row["metadata"]["verifier_timeout"] = 600
    dataset.write_text(dataset.read_text() + json.dumps(row) + "\n")
    preflight = module.preflight_eval_slice(
        dataset,
        start_index=1,
        max_tasks=1,
        agent_timeout_seconds=3300,
    )
    assert preflight.max_dataset_task_timeout_seconds == 9000
    assert preflight.max_verifier_timeout_seconds == 600
    assert preflight.task_timeout_floor_seconds == 3900
    assert preflight.outer_task_envelope_seconds == 9000

    with pytest.raises(module.LauncherError, match="selected dataset"):
        module.main(
            [
                "--dry-run",
                "--i-understand-eval-only",
                "--run-id",
                "walltime-check",
                "--data",
                str(dataset),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(tmp_path / "out"),
                "--start-index",
                "1",
                "--max-tasks",
                "1",
                "--max-concurrency",
                "4",
                "--slurm-walltime-seconds",
                "10799",
            ]
        )


def test_partial_evaluator_auto_resumes_with_walltime_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "summary.json").write_text(json.dumps({"collection": {"missing_result_count": 2}}))
    return_codes = iter((3, 0))
    commands: list[list[str]] = []

    class Process:
        def __init__(self, return_code: int | None):
            self.returncode = return_code

        def poll(self):
            return self.returncode

    def popen(command, **_kwargs):
        commands.append(list(command))
        return Process(next(return_codes))

    service = Process(None)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    processes = []
    owner_fd = os.open(tmp_path / "owner.lock", os.O_CREAT | os.O_RDWR, 0o600)
    result = module.run_evaluator_passes(
        evaluator_command=["python", "forced_route_eval.py"],
        evaluator_environment={},
        processes=processes,
        services=(("service", service),),
        output_dir=output,
        max_passes=3,
        resume_backoff_seconds=1,
        max_concurrency=2,
        outer_task_envelope_seconds=100,
        max_paid_attempts_per_work=2,
        allocated_slurm_seconds=10_000,
        launcher_started_monotonic=module.time.monotonic(),
        owner_fd=owner_fd,
    )
    assert result == 0
    assert len(processes) == 2
    assert "--resume" not in commands[0]
    assert "--resume" in commands[1]

    commands.clear()
    return_codes = iter((3,))
    result = module.run_evaluator_passes(
        evaluator_command=["python", "forced_route_eval.py"],
        evaluator_environment={},
        processes=[],
        services=(("service", service),),
        output_dir=output,
        max_passes=3,
        resume_backoff_seconds=1,
        max_concurrency=2,
        outer_task_envelope_seconds=100,
        max_paid_attempts_per_work=2,
        allocated_slurm_seconds=100,
        launcher_started_monotonic=module.time.monotonic(),
        owner_fd=owner_fd,
    )
    assert result == 3
    assert len(commands) == 1
    os.close(owner_fd)


def test_allocation_owner_lease_is_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "owned-output"
    first = module.AllocationOwnerLease(output)
    try:
        with pytest.raises(module.LauncherError, match="another allocation owns"):
            module.AllocationOwnerLease(output)
    finally:
        first.close()
    second = module.AllocationOwnerLease(output)
    second.close()


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
    launcher_text = SCRIPT.read_text()
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
    assert (
        'POLAR_APPTAINER_PERSISTENT_BROKER="${POLAR_APPTAINER_PERSISTENT_BROKER:-1}"'
        in run_text
    )
    assert "POLAR_APPTAINER_PERSISTENT_BROKER:-0" not in run_text
    assert "--gpus" not in run_text
    assert "--gres" not in run_text
    combined = run_text.lower() + submit_text.lower()
    assert "ray start" not in combined
    assert "train_async" not in combined
    assert "serve_sglang" not in combined
    assert "megatron" not in combined
    assert "checkpoint" not in combined
    assert 'str(snapshot_example_dir / "serve_tokenizer.py")' in launcher_text
    assert "source_root=snapshot_root" in launcher_text


def test_submit_dry_run_is_secret_free_cpu_default_with_explicit_gpu_fallback(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "eval.jsonl"
    _timing_only_dataset(dataset)
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
    assert "--time=01:27:00" in cpu.stdout
    assert "POLAR_FORCED_EVAL_ENV_FILE=" in cpu.stdout
    assert not (tmp_path / "cpu.submit").exists()

    fallback = invoke(tmp_path / "gpu", 1)
    assert fallback.returncode == 0, fallback.stderr
    assert secret not in fallback.stdout + fallback.stderr
    assert "--gres=gpu:1" in fallback.stdout
    assert not (tmp_path / "gpu.submit").exists()


def test_submitter_rejects_insufficient_explicit_walltime_before_submit(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "eval.jsonl"
    _timing_only_dataset(dataset, count=32)
    output = tmp_path / "insufficient"
    environment = dict(os.environ)
    environment.update(
        {
            "POLAR_NVIDIA_API_KEY": "nvapi-walltime-test",
            "SUBMIT_DRY_RUN": "1",
            "WALL_TIME": "12:00:00",
        }
    )
    completed = subprocess.run(
        [
            "bash",
            str(SUBMIT_SCRIPT),
            "--i-understand-eval-only",
            "--run-id",
            "walltime-rejected",
            "--data",
            str(dataset),
            "--output-dir",
            str(output),
            "--max-tasks",
            "32",
            "--replicates",
            "1",
            "--max-concurrency",
            "4",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "requires at least 56520s" in completed.stderr
    assert "64 work items * 1 max paid attempts" in completed.stderr

    environment["WALL_TIME"] = "15:42:00"
    baseline = subprocess.run(
        [
            "bash",
            str(SUBMIT_SCRIPT),
            "--i-understand-eval-only",
            "--include-qwen35-baseline",
            "--run-id",
            "baseline-walltime-rejected",
            "--data",
            str(dataset),
            "--output-dir",
            str(tmp_path / "baseline-insufficient"),
            "--max-tasks",
            "32",
            "--replicates",
            "1",
            "--max-concurrency",
            "4",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert baseline.returncode == 2
    assert "requires at least 83880s" in baseline.stderr
    assert "96 work items * 1 max paid attempts" in baseline.stderr
    assert not output.with_name(f"{output.name}.submit").exists()


def test_submitter_explicit_resume_keeps_output_and_allocates_fresh_service(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "eval.jsonl"
    _timing_only_dataset(dataset)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "manifest.json").write_text("{}\n")
    environment = dict(os.environ)
    environment.update(
        {
            "POLAR_NVIDIA_API_KEY": "nvapi-resume-test",
            "SUBMIT_DRY_RUN": "1",
        }
    )

    completed = subprocess.run(
        [
            "bash",
            str(SUBMIT_SCRIPT),
            "--resume",
            "--i-understand-eval-only",
            "--run-id",
            "resume-run",
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
    assert "--resume" in completed.stdout
    assert "--service-dir" in completed.stdout
    assert output.is_dir()
    assert not list(tmp_path.glob("existing.service.resume.*.submit"))


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
    _timing_only_dataset(dataset)
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
    fake_srun.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" >"$FAKE_SRUN_ARGS"\n')
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
        '"${POLAR_CONTROL_PLANE_TOKEN:-}" "${SLURM_EXPORT_ENV:-}" '
        '>"$FAKE_SRUN_ENV"\n'
    )
    fake_srun.chmod(0o700)
    credential = tmp_path / "credentials.env"
    secret = "nvapi-step-secret-sentinel"
    credential.write_text(f"export POLAR_NVIDIA_API_KEY={secret}\nexport SRUN_BIN={fake_srun}\n")
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


def test_inner_entrypoint_enables_persistent_broker_and_rejects_opt_out(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "protected-runtime.env"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s|%s|%s\\n' "
        '"${POLAR_APPTAINER_PERSISTENT_BROKER:-}" '
        '"${POLAR_APPTAINER_NO_INSTANCE:-}" '
        '"${POLAR_APPTAINER_CLEANENV:-}" >"${FAKE_PYTHON_ENV}"\n'
    )
    fake_python.chmod(0o700)
    train_image = tmp_path / "train.sqsh"
    train_image.write_bytes(b"test")
    data_root = tmp_path / "data"
    data_root.mkdir()
    environment = dict(os.environ)
    environment.pop("POLAR_CONTROL_PLANE_TOKEN", None)
    for name in module.REQUIRED_PROTECTED_RUNTIME_VARIABLES:
        environment.pop(name, None)
    environment.update(
        {
            "SLURM_JOB_ID": "5432101",
            "SLURM_JOB_NUM_NODES": "1",
            "SPILOT_FORCED_EVAL_IN_CONTAINER": "1",
            "POLAR_NVIDIA_API_KEY": "nvapi-protected-runtime-sentinel",
            "POLAR_DATA_ROOT": str(data_root),
            "POLR_TRAIN_SQSH": str(train_image),
            "TMAX_SIF_PYTHON_BIN": str(fake_python),
            "FAKE_PYTHON_ENV": str(capture),
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
    assert capture.read_text().strip() == "1|1|1"

    capture.unlink()
    environment["SLURM_JOB_ID"] = "5432102"
    environment["POLAR_APPTAINER_PERSISTENT_BROKER"] = "0"
    rejected = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--i-understand-eval-only"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert rejected.returncode == 2
    assert (
        "POLAR_APPTAINER_PERSISTENT_BROKER must be exactly 1 for protected forced evaluation"
    ) in rejected.stderr
    assert not capture.exists()


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
    for environment in (rollout, gateway, tunnel):
        assert environment["no_proxy"] == "127.0.0.1,localhost"
        assert environment["NO_PROXY"] == "127.0.0.1,localhost"
