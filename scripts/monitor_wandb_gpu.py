#!/usr/bin/env python3
"""Sample host GPU utilization and optionally log stable metrics to W&B.

This is a sidecar monitor for long Ray/Slime jobs. W&B's built-in system
metrics can create many host/process-scoped GPU labels when multiple Ray actors
attach to the same run. This script logs one stable metric family instead:

    polar_system/gpu_0/util_pct
    polar_system/gpu_train/mean_util_pct
    polar_system/gpu_rollout/mean_util_pct

Raw samples are always written to CSV under tmp/gpu_monitor by default.
W&B charts sample the trainer's shared ``train/step`` progress file, but write
it to a per-node namespaced axis.  This keeps the canonical trainer axis
monotonic when several sidecars publish concurrently. CSV retains wall time
(and the sampled train step) for high-resolution offline analysis.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "tmp" / "gpu_monitor"
QUERY = "timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
CSV_FIELDS = [
    "sample_time",
    "timestamp",
    "gpu",
    "util_gpu_pct",
    "util_mem_pct",
    "memory_used_mb",
    "memory_total_mb",
    "power_draw_w",
    "train_step",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-s", type=float, default=10.0, help="sampling interval")
    parser.add_argument(
        "--samples", type=int, default=0, help="number of samples; 0 means run until stopped"
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="CSV path for raw samples; defaults to tmp/gpu_monitor/<timestamp>_gpu.csv",
    )
    parser.add_argument("--train-gpus", default="0,1,2,3", help="comma-separated trainer GPU ids")
    parser.add_argument(
        "--rollout-gpus", default="4,5,6,7", help="comma-separated rollout GPU ids"
    )
    parser.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "polar-swegym-grpo")
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID", ""))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP", ""))
    parser.add_argument("--wandb-dir", default=str(ROOT / "logs"))
    parser.add_argument("--wandb-label", default="gpu-monitor")
    parser.add_argument(
        "--wandb-mode",
        default="shared",
        choices=["online", "shared", "offline", "disabled"],
        help="use 'shared' when attaching to an active multi-process run",
    )
    parser.add_argument("--metric-prefix", default="polar_system")
    parser.add_argument(
        "--static-metric",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="stable run-level metric added to every W&B sample; repeatable",
    )
    parser.add_argument(
        "--one-shot-static",
        action="store_true",
        help=(
            "publish static metrics once without sampling GPUs; intended for "
            "terminal watcher events after an allocation has exited"
        ),
    )
    parser.add_argument(
        "--train-progress-file",
        default=os.environ.get("SLIME_TRAIN_PROGRESS_FILE", ""),
        help="shared file containing the latest completed train/step",
    )
    parser.add_argument(
        "--wandb-finish-timeout-s",
        type=float,
        default=15.0,
        help="hard timeout for the sidecar's non-primary W&B finish",
    )
    parser.add_argument("--no-wandb", action="store_true", help="only write CSV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.interval_s <= 0:
        raise SystemExit("--interval-s must be greater than 0")
    if args.wandb_finish_timeout_s <= 0:
        raise SystemExit("--wandb-finish-timeout-s must be greater than 0")
    static_metrics = _parse_static_metrics(args.static_metric)
    train_progress_file = Path(args.train_progress_file) if args.train_progress_file else None
    if args.one_shot_static:
        if not static_metrics:
            raise SystemExit("--one-shot-static requires at least one --static-metric")
        if args.no_wandb or not args.wandb_run_id:
            raise SystemExit("--one-shot-static requires an attached W&B run")
        wandb_run = _init_wandb(args)
        step_metric = f"{args.metric_prefix}/train_step"
        metrics: dict[str, float | int] = {
            step_metric: _read_train_step(train_progress_file, default=0),
            **static_metrics,
        }
        _define_exact_wandb_axes(
            metrics,
            step_metric=step_metric,
            defined_metrics=set(),
        )
        wandb_run.log(metrics)
        if not _finish_wandb_with_timeout(
            wandb_run,
            timeout_s=args.wandb_finish_timeout_s,
        ):
            print(
                f"W&B one-shot finish exceeded {args.wandb_finish_timeout_s:g}s; forcing process exit",
                file=sys.stderr,
                flush=True,
            )
            # The watcher uses this process status as the publication
            # acknowledgement.  A timed-out finish is not proof that W&B
            # committed the metric, so fail rather than printing a false
            # success while still using os._exit to preserve the hard bound.
            os._exit(75)
        print("published one-shot W&B metrics")
        return 0

    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi not found")

    train_gpus = _parse_gpu_ids(args.train_gpus)
    rollout_gpus = _parse_gpu_ids(args.rollout_gpus)
    csv_path = Path(args.out_csv) if args.out_csv else _default_csv_path()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    defined_metric_axes: set[str] = set()
    if not args.no_wandb and args.wandb_run_id:
        wandb_run = _init_wandb(args)

    stop = False

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    start_monotonic = time.monotonic()
    sample_index = 0
    train_step = 0
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if fh.tell() == 0:
            writer.writeheader()

        while not stop:
            rows = _sample_nvidia_smi()
            wall_time_unix_s = time.time()
            train_step = _read_train_step(train_progress_file, default=train_step)
            for row in rows:
                row["sample_time"] = wall_time_unix_s
                row["train_step"] = train_step
                writer.writerow({key: row.get(key) for key in CSV_FIELDS})
            fh.flush()

            if wandb_run is not None and rows:
                metrics = _build_wandb_metrics(
                    rows,
                    prefix=args.metric_prefix,
                    wall_time_unix_s=wall_time_unix_s,
                    elapsed_s=time.monotonic() - start_monotonic,
                    sample_index=sample_index,
                    train_step=train_step,
                    train_gpus=train_gpus,
                    rollout_gpus=rollout_gpus,
                )
                metrics.update(static_metrics)
                _define_exact_wandb_axes(
                    metrics,
                    step_metric=f"{args.metric_prefix}/train_step",
                    defined_metrics=defined_metric_axes,
                )
                wandb_run.log(metrics)

            sample_index += 1
            if args.samples and sample_index >= args.samples:
                break
            _sleep_interruptibly(args.interval_s, lambda: stop)

    print(f"wrote {csv_path}")
    sys.stdout.flush()
    if wandb_run is not None and not _finish_wandb_with_timeout(
        wandb_run,
        timeout_s=args.wandb_finish_timeout_s,
    ):
        print(
            f"W&B sidecar finish exceeded {args.wandb_finish_timeout_s:g}s; forcing process exit",
            file=sys.stderr,
            flush=True,
        )
        # W&B's atexit hook calls finish again. A normal interpreter exit can
        # therefore hang the Slurm allocation after our own timeout expires.
        os._exit(0)
    return 0


def _init_wandb(args: argparse.Namespace) -> Any:
    import wandb

    settings = wandb.Settings(
        mode=args.wandb_mode,
        x_label=args.wandb_label,
        x_disable_stats=True,
        x_primary=False,
        x_update_finish_state=False,
        finish_timeout=args.wandb_finish_timeout_s,
        console="off",
        root_dir=args.wandb_dir,
    )
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        id=args.wandb_run_id,
        group=args.wandb_group or None,
        resume="allow",
        dir=args.wandb_dir,
        settings=settings,
    )
    step_metric = f"{args.metric_prefix}/train_step"
    wandb.define_metric(step_metric)
    wandb.define_metric(f"{args.metric_prefix}/*", step_metric=step_metric)
    return run


def _define_exact_wandb_axes(
    metrics: dict[str, float | int],
    *,
    step_metric: str,
    defined_metrics: set[str],
) -> None:
    """Define each sidecar value explicitly before publishing it."""

    import wandb

    for metric_name in metrics:
        if metric_name == step_metric or metric_name in defined_metrics:
            continue
        wandb.define_metric(metric_name, step_metric=step_metric)
        defined_metrics.add(metric_name)


def _finish_wandb_with_timeout(run: Any, *, timeout_s: float) -> bool:
    """Finish a non-primary W&B attachment without holding the allocation."""

    done = threading.Event()

    def finish() -> None:
        try:
            run.finish(exit_code=0)
        finally:
            done.set()

    thread = threading.Thread(target=finish, name="wandb-sidecar-finish", daemon=True)
    thread.start()
    return done.wait(timeout_s)


def _read_train_step(path: Path | None, *, default: int) -> int:
    if path is None:
        return default
    try:
        step = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return default
    # A stale/replaced progress file must not make a resumed W&B series move
    # backwards. Atomic trainer writes ensure readers never see partial text.
    return max(default, step)


def _parse_static_metrics(values: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw in values:
        name, separator, value_text = raw.partition("=")
        name = name.strip()
        if not separator or not name or any(character.isspace() for character in name):
            raise SystemExit(f"invalid --static-metric {raw!r}; expected NAME=VALUE")
        try:
            value = float(value_text)
        except ValueError as exc:
            raise SystemExit(
                f"invalid --static-metric {raw!r}; VALUE must be numeric"
            ) from exc
        if not math.isfinite(value):
            raise SystemExit(
                f"invalid --static-metric {raw!r}; VALUE must be finite"
            )
        if name in metrics:
            raise SystemExit(f"duplicate --static-metric name: {name}")
        metrics[name] = value
    return metrics


def _sample_nvidia_smi() -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"nvidia-smi exited {proc.returncode}")
    rows = []
    for line in proc.stdout.splitlines():
        parsed = _parse_nvidia_smi_line(line)
        if parsed is not None:
            rows.append(parsed)
    return rows


def _parse_nvidia_smi_line(line: str) -> dict[str, Any] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 7:
        return None
    try:
        return {
            "timestamp": parts[0],
            "gpu": int(parts[1]),
            "util_gpu_pct": float(parts[2]),
            "util_mem_pct": float(parts[3]),
            "memory_used_mb": float(parts[4]),
            "memory_total_mb": float(parts[5]),
            "power_draw_w": float(parts[6]),
        }
    except ValueError:
        return None


def _build_wandb_metrics(
    rows: list[dict[str, Any]],
    *,
    prefix: str,
    wall_time_unix_s: float,
    elapsed_s: float,
    sample_index: int,
    train_step: int,
    train_gpus: set[int],
    rollout_gpus: set[int],
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        f"{prefix}/train_step": train_step,
        f"{prefix}/gpu_monitor/wall_time_unix_s": wall_time_unix_s,
        f"{prefix}/gpu_monitor/elapsed_s": elapsed_s,
        f"{prefix}/gpu_monitor/sample_index": sample_index,
    }
    for row in rows:
        gpu = int(row["gpu"])
        base = f"{prefix}/gpu_{gpu}"
        metrics[f"{base}/util_pct"] = float(row["util_gpu_pct"])
        metrics[f"{base}/mem_used_gb"] = float(row["memory_used_mb"]) / 1024.0
        metrics[f"{base}/power_w"] = float(row["power_draw_w"])

    _add_group_metrics(metrics, prefix=prefix, name="gpu_all", rows=rows)
    _add_group_metrics(
        metrics,
        prefix=prefix,
        name="gpu_train",
        rows=[row for row in rows if int(row["gpu"]) in train_gpus],
    )
    _add_group_metrics(
        metrics,
        prefix=prefix,
        name="gpu_rollout",
        rows=[row for row in rows if int(row["gpu"]) in rollout_gpus],
    )
    return metrics


def _add_group_metrics(
    metrics: dict[str, float | int],
    *,
    prefix: str,
    name: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    util = [float(row["util_gpu_pct"]) for row in rows]
    mem = [float(row["memory_used_mb"]) / 1024.0 for row in rows]
    power = [float(row["power_draw_w"]) for row in rows]
    base = f"{prefix}/{name}"
    metrics[f"{base}/mean_util_pct"] = sum(util) / len(util)
    metrics[f"{base}/min_util_pct"] = min(util)
    metrics[f"{base}/max_util_pct"] = max(util)
    metrics[f"{base}/mean_mem_used_gb"] = sum(mem) / len(mem)
    metrics[f"{base}/mean_power_w"] = sum(power) / len(power)


def _parse_gpu_ids(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _default_csv_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUT_DIR / f"{timestamp}_gpu.csv"


def _sleep_interruptibly(seconds: float, should_stop: Any) -> None:
    deadline = time.monotonic() + seconds
    while not should_stop():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.5))


if __name__ == "__main__":
    raise SystemExit(main())
