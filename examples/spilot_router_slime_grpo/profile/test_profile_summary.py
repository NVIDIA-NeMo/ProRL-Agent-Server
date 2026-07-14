#!/usr/bin/env python3
"""Unit tests for profile_summary.py."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import profile_summary as ps  # noqa: E402


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


class ProfileSummaryTest(unittest.TestCase):
    def make_run(self, root: Path) -> Path:
        run = root / "profile-run"
        job = run / "job-123"
        job.mkdir(parents=True)
        _write(
            run / "submit" / "env-001.sh",
            "\n".join(
                [
                    "export PROFILE_ARM=async-a2",
                    "export TMAX_PROFILE_ARM=profile-async-a2",
                    "export TMAX_PROFILE_BATCH=batch-001",
                    "export POLAR_SUBMITTED_JOB_ID=123",
                    "export POLAR_FULLY_ASYNC=true",
                    "export POLAR_MAX_ASYNC_LEVEL=2",
                    "export POLAR_MULTI_GATEWAY=1",
                    "export POLAR_GATEWAY_COUNT_OVERRIDE=2",
                    "export ACTOR_NUM_NODES=1",
                    "export ACTOR_NUM_GPUS_PER_NODE=2",
                    "export ROLLOUT_NUM_GPUS=2",
                    "export RAY_NUM_NODES=2",
                    "export RAY_NUM_GPUS_PER_NODE=2",
                    # This must never be retained in the report.
                    "export WANDB_API_KEY=secret-value",
                ]
            )
            + "\n",
        )
        log = job / "wandb" / "files" / "output_pool0-test.log"
        lines = [
            # First step is warmup. Include a Ray ANSI prefix to exercise stripping.
            "\x1b[36m(Trainer pid=1)\x1b[0m train_metric_utils.py - perf 0: "
            "{'timing/train_wait_time': 100.0, 'timing/train_time': 20.0, "
            "'timing/step_time': 120.0, 'perf/wait_time_ratio': 0.8333333}",
            "rollout.py - perf 0: {'polar/accepted/group_count': 2.0, "
            "'polar/staleness/mean': 0.0, 'timing/service_time_max': 80.0}",
            "train_metric_utils.py - perf 1: {'timing/train_wait_time': 10.0, "
            "'timing/train_time': 10.0, 'timing/step_time': 20.0, "
            "'perf/wait_time_ratio': 0.5, 'perf/actor_train_tok_per_s': 1000.0}",
            "rollout.py - perf 1: {'polar/accepted/group_count': 2.0, "
            "'polar/candidate/accounted_sessions': 4.0, "
            "'polar/session_trainable_response_tokens/count': 4.0, "
            "'polar/session_trainable_response_tokens/mean': 100.0, "
            "'polar/staleness/mean': 1.0, 'timing/service_time_max': 11.0, "
            "'timing/service_window': 15.0, 'timing/pipeline_ms/rollout_collect': 5000.0, "
            "'polar/scheduler/completed_buffer': 1.0, "
            "'polar/scheduler/output_queue': 0.0, 'polar/scheduler/deferred_queue': 0.0}",
            "train_metric_utils.py - perf 2: {'timing/train_wait_time': 20.0, "
            "'timing/train_time': 10.0, 'timing/step_time': 30.0, "
            "'perf/wait_time_ratio': 0.6666666667, 'perf/actor_train_tok_per_s': 900.0}",
            "rollout.py - perf 2: {'polar/accepted/group_count': 2.0, "
            "'polar/session_outcome/accounted_count': 4.0, "
            "'polar/session_trainable_response_tokens/count': 4.0, "
            "'polar/session_trainable_response_tokens/mean': 150.0, "
            "'polar/staleness/mean': 2.0, 'timing/service_time_max': 12.0, "
            "'timing/service_window': 16.0, 'polar/scheduler/completed_buffer': 3.0, "
            "'polar/scheduler/output_queue': 0.0, 'polar/scheduler/deferred_queue': 0.0}",
        ]
        _write(log, "\n".join(lines) + "\n")

        header = (
            "sample_time,timestamp,gpu,util_gpu_pct,util_mem_pct,"
            "memory_used_mb,memory_total_mb,power_draw_w,train_step\n"
        )
        _write(
            job / "gpu_monitor" / "node_0.csv",
            header
            + "0,2026/07/14 00:00:00.000,0,5,0,100,1000,80,0\n"
            + "10,2026/07/14 00:00:10.000,0,60,0,200,1000,100,1\n"
            + "20,2026/07/14 00:00:20.000,0,80,0,300,1000,120,2\n",
        )
        # Exercise the alternate/newer header aliases requested by the profile tool.
        _write(
            job / "gpu_monitor" / "node_1.csv",
            "timestamp,index,utilization_gpu,memory_used_mb,memory_total_mb,power_w,temperature_c,step\n"
            + "0,0,2,100,1000,70,31,0\n"
            + "10,0,20,200,1000,90,32,1\n"
            + "20,0,40,300,1000,110,33,2\n",
        )
        return run

    def test_summarizes_steps_gpu_roles_and_throughput(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run(Path(temporary))
            report = ps.build_report([run], warmup_steps=1)
            self.assertEqual(len(report["jobs"]), 1)
            job = report["jobs"][0]

            self.assertEqual(job["job_id"], "123")
            self.assertEqual(job["config"]["mode"], "fully_async")
            self.assertEqual(job["config"]["label"], "profile-async-a2")
            self.assertEqual(job["config"]["profile_batch"], "batch-001")
            self.assertEqual(job["config"]["async_level"], 2)
            self.assertEqual(job["config"]["actor_gpus"], 2)
            self.assertEqual(job["config"]["rollout_gpus"], 2)
            self.assertEqual(job["config"]["allocated_gpus"], 4)
            self.assertEqual(job["config"]["gateway_count"], 2)
            self.assertNotIn("secret-value", json.dumps(job))

            self.assertEqual(job["warmup_step_ids"], [0])
            self.assertEqual(job["step_count_steady"], 2)
            metrics = job["steady_state"]["metrics"]
            self.assertAlmostEqual(metrics["step_time_s"]["mean"], 25.0)
            self.assertAlmostEqual(metrics["wait_ratio"]["mean"], 7.0 / 12.0)
            self.assertAlmostEqual(metrics["staleness_mean"]["mean"], 1.5)
            self.assertAlmostEqual(metrics["rollout_collect_s"]["mean"], 5.0)

            throughput = job["steady_state"]["throughput"]
            self.assertAlmostEqual(throughput["accepted_groups_per_s"], 4.0 / 50.0)
            self.assertAlmostEqual(throughput["accepted_sessions_per_s"], 8.0 / 50.0)
            self.assertAlmostEqual(throughput["accepted_trainable_tokens_per_s"], 1000.0 / 50.0)
            self.assertAlmostEqual(throughput["gpu_hours_estimate"], 4.0 * 50.0 / 3600.0)
            self.assertAlmostEqual(
                job["steady_state"]["queue_trend_per_step"]["queue_backlog_groups"],
                2.0,
            )

            self.assertEqual(job["gpu"]["steady_filter"], "train_step")
            self.assertAlmostEqual(
                job["gpu"]["roles"]["actor"]["utilization_gpu_pct"]["mean"], 70.0
            )
            self.assertAlmostEqual(
                job["gpu"]["roles"]["rollout"]["utilization_gpu_pct"]["mean"], 30.0
            )
            self.assertAlmostEqual(job["gpu"]["steady"]["utilization_gpu_pct"]["p50"], 50.0)

    def test_non_literal_perf_falls_back_to_scalar_pairs(self) -> None:
        parsed = ps._numeric_dict(
            "{'timing/train_wait_time': np.float64(4.0), "
            "'timing/train_time': 2.5, 'bad': nan"
        )
        self.assertEqual(parsed["timing/train_wait_time"], 4.0)
        self.assertEqual(parsed["timing/train_time"], 2.5)
        self.assertNotIn("bad", parsed)

    def test_missing_metrics_produce_nulls_and_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            empty_job = Path(temporary) / "job-999"
            empty_job.mkdir()
            summary = ps.summarize_job(empty_job, warmup_steps=1)
            self.assertEqual(summary["step_count_total"], 0)
            self.assertIsNone(
                summary["steady_state"]["metrics"]["step_time_s"]["mean"]
            )
            self.assertTrue(summary["warnings"])
            # Missing telemetry remains serializable and does not crash table rendering.
            json.dumps(summary)
            ps.render_table({"jobs": [summary]})

    def test_cli_json_stdout_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = self.make_run(Path(temporary))
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = ps.main([str(run), "--warmup-steps", "1", "--json", "-"])
            self.assertEqual(result, 0)
            decoded = json.loads(stdout.getvalue())
            self.assertEqual(decoded["jobs"][0]["job_id"], "123")
            self.assertIn("acc_s/GPUh", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
