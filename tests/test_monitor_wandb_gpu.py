from argparse import Namespace
import importlib.util
from pathlib import Path
import threading

import pytest


_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "monitor_wandb_gpu.py"
_SPEC = importlib.util.spec_from_file_location("monitor_wandb_gpu", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
monitor_wandb_gpu = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(monitor_wandb_gpu)


def _row(gpu: int = 0) -> dict[str, float | int]:
    return {
        "gpu": gpu,
        "util_gpu_pct": 75.0,
        "memory_used_mb": 2048.0,
        "power_draw_w": 350.0,
    }


@pytest.mark.unit
def test_gpu_metrics_carry_train_step_and_keep_wall_time_for_audit():
    first = monitor_wandb_gpu._build_wandb_metrics(
        [_row()],
        prefix="polar_system/actor_node_0",
        wall_time_unix_s=1_800_000_000.25,
        elapsed_s=120.0,
        sample_index=12,
        train_step=7,
        train_gpus={0},
        rollout_gpus=set(),
    )
    restarted = monitor_wandb_gpu._build_wandb_metrics(
        [_row()],
        prefix="polar_system/actor_node_0",
        wall_time_unix_s=1_800_000_300.5,
        elapsed_s=0.2,
        sample_index=0,
        train_step=9,
        train_gpus={0},
        rollout_gpus=set(),
    )

    axis = "polar_system/actor_node_0/gpu_monitor/wall_time_unix_s"
    assert first[axis] < restarted[axis]
    assert "train/step" not in first
    assert "train/step" not in restarted
    assert first["polar_system/actor_node_0/train_step"] == 7
    assert restarted["polar_system/actor_node_0/train_step"] == 9
    assert restarted["polar_system/actor_node_0/gpu_monitor/sample_index"] == 0
    assert restarted["polar_system/actor_node_0/gpu_monitor/elapsed_s"] == 0.2


@pytest.mark.unit
def test_wandb_gpu_metrics_are_bound_to_train_step(monkeypatch, tmp_path):
    import wandb

    definitions = []
    init_kwargs = {}
    expected_run = object()
    monkeypatch.setattr(wandb, "Settings", lambda **kwargs: kwargs)

    def init(**kwargs):
        init_kwargs.update(kwargs)
        return expected_run

    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(
        wandb,
        "define_metric",
        lambda name, **kwargs: definitions.append((name, kwargs.get("step_metric"))),
    )
    args = Namespace(
        wandb_mode="shared",
        wandb_label="gpu-monitor",
        wandb_dir=str(tmp_path),
        wandb_project="test-project",
        wandb_entity="",
        wandb_run_id="resumed-run",
        wandb_group="test-group",
        metric_prefix="polar_system/rollout_node_1",
        wandb_finish_timeout_s=11.0,
    )

    run = monitor_wandb_gpu._init_wandb(args)

    assert run is expected_run
    assert init_kwargs["group"] == "test-group"
    assert definitions == [
        ("polar_system/rollout_node_1/train_step", None),
        (
            "polar_system/rollout_node_1/*",
            "polar_system/rollout_node_1/train_step",
        ),
    ]


@pytest.mark.unit
def test_gpu_sidecar_defines_every_concrete_metric_exactly_once(monkeypatch):
    import wandb

    definitions = []
    monkeypatch.setattr(
        wandb,
        "define_metric",
        lambda name, **kwargs: definitions.append((name, kwargs.get("step_metric"))),
    )
    metrics = {
        "polar_tmax_system/node_0/train_step": 4,
        "polar_tmax_system/node_0/gpu_all/mean_util_pct": 75.0,
        "polar_tmax_system/node_0/gpu_monitor/wall_time_unix_s": 123.0,
    }
    cache = set()

    monitor_wandb_gpu._define_exact_wandb_axes(
        metrics,
        step_metric="polar_tmax_system/node_0/train_step",
        defined_metrics=cache,
    )
    monitor_wandb_gpu._define_exact_wandb_axes(
        metrics,
        step_metric="polar_tmax_system/node_0/train_step",
        defined_metrics=cache,
    )

    assert definitions == [
        (
            "polar_tmax_system/node_0/gpu_all/mean_util_pct",
            "polar_tmax_system/node_0/train_step",
        ),
        (
            "polar_tmax_system/node_0/gpu_monitor/wall_time_unix_s",
            "polar_tmax_system/node_0/train_step",
        ),
    ]


@pytest.mark.unit
def test_gpu_sidecar_wandb_record_never_publishes_canonical_train_step(
    monkeypatch,
    tmp_path,
):
    import wandb
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    monkeypatch.setenv("WANDB_SILENT", "true")
    prefix = "polar_tmax_system/node_2"
    step_metric = f"{prefix}/train_step"
    run = wandb.init(
        project="axis-protobuf-test",
        mode="offline",
        dir=str(tmp_path),
        settings=wandb.Settings(console="off", x_disable_stats=True),
    )
    wandb.define_metric(step_metric)
    wandb.define_metric(f"{prefix}/*", step_metric=step_metric)
    metrics = monitor_wandb_gpu._build_wandb_metrics(
        [_row()],
        prefix=prefix,
        wall_time_unix_s=1_800_000_000.25,
        elapsed_s=1.0,
        sample_index=0,
        train_step=7,
        train_gpus={0},
        rollout_gpus=set(),
    )
    monitor_wandb_gpu._define_exact_wandb_axes(
        metrics,
        step_metric=step_metric,
        defined_metrics=set(),
    )
    run.log(metrics)
    run.finish()

    wandb_file = next(tmp_path.glob("wandb/*/run-*.wandb"))
    datastore = DataStore()
    datastore.open_for_scan(str(wandb_file))
    history_keys = set()
    definitions = {}
    while (record_bytes := datastore.scan_data()) is not None:
        record = wandb_internal_pb2.Record()
        record.ParseFromString(record_bytes)
        record_type = record.WhichOneof("record_type")
        if record_type == "history":
            history_keys.update(
                "/".join(item.nested_key) if item.nested_key else item.key
                for item in record.history.item
            )
        elif record_type == "metric" and record.metric.name:
            definitions[record.metric.name] = record.metric.step_metric

    assert "train/step" not in history_keys
    assert step_metric in history_keys
    assert f"{prefix}/gpu_all/mean_util_pct" in history_keys
    assert definitions[f"{prefix}/gpu_all/mean_util_pct"] == step_metric


@pytest.mark.unit
def test_read_train_step_is_monotonic_and_tolerates_partial_file(tmp_path):
    progress = tmp_path / "train.step"

    assert monitor_wandb_gpu._read_train_step(progress, default=3) == 3
    progress.write_text("8\n")
    assert monitor_wandb_gpu._read_train_step(progress, default=3) == 8
    progress.write_text("partial")
    assert monitor_wandb_gpu._read_train_step(progress, default=8) == 8
    progress.write_text("2\n")
    assert monitor_wandb_gpu._read_train_step(progress, default=8) == 8


@pytest.mark.unit
def test_wandb_finish_has_hard_timeout():
    blocked = threading.Event()

    class Run:
        def finish(self, *, exit_code):
            blocked.wait()

    assert not monitor_wandb_gpu._finish_wandb_with_timeout(Run(), timeout_s=0.01)
    blocked.set()


@pytest.mark.unit
def test_shared_launcher_wires_trainer_progress_into_every_gpu_sidecar():
    launcher = (_SCRIPT_PATH.parents[1] / "examples" / "swegym_slime_grpo" / "run.sh").read_text()

    assert (
        'SLIME_TRAIN_PROGRESS_FILE="${SLIME_TRAIN_PROGRESS_FILE:-${SAVE_DIR}/train_progress.step}"'
        in launcher
    )
    assert '--train-progress-file "$SLIME_TRAIN_PROGRESS_FILE"' in launcher
    assert '\\"SLIME_TRAIN_PROGRESS_FILE\\": \\"${SLIME_TRAIN_PROGRESS_FILE}\\"' in launcher
    assert '--train-env-vars "$TRAIN_PROGRESS_ENV_JSON"' in launcher
