from __future__ import annotations

from polar.rollout.models import SessionTiming
from polar.rollout.timer import StageTimer


def test_session_timing_remains_compatible_with_legacy_payload() -> None:
    timing = SessionTiming.model_validate(
        {
            "register_to_init_queue_ms": 1.0,
            "init_ms": 2.0,
            "run_ms": 3.0,
            "postrun_ms": 4.0,
        }
    )

    assert timing.run_ms == 3.0
    assert timing.container_start_ms == 0.0
    assert timing.runtime_validation_ms == 0.0
    assert timing.runtime_exec_ms_by_category == {}
    assert timing.mini_swe_command_count == 0


def test_stage_timer_exports_detailed_spans_and_sanitized_command_summaries() -> None:
    timer = StageTimer()
    timer._marks.update(  # type: ignore[attr-defined]
        {
            "dispatch_started": 1.0,
            "init_started": 2.0,
            "init_finished": 5.0,
            "runtime_validation_started": 2.0,
            "runtime_validation_finished": 3.0,
            "prepare_started": 3.0,
            "prepare_finished": 5.0,
            "run_started": 5.0,
            "run_finished": 10.0,
            "agent_setup_started": 5.0,
            "agent_setup_finished": 5.5,
            "agent_exec_started": 5.5,
            "agent_exec_finished": 9.0,
            "agent_postprocess_started": 9.0,
            "agent_postprocess_finished": 10.0,
            "postrun_started": 10.0,
            "postrun_finished": 12.0,
            "build_started": 10.0,
            "build_finished": 10.25,
            "eval_started": 10.25,
            "eval_finished": 12.0,
            "postrun_exec_started": 12.0,
            "postrun_exec_finished": 12.5,
            "runtime_stop_started": 12.5,
            "runtime_stop_finished": 13.0,
            "teardown_finished": 13.0,
            "return_finished": 13.0,
        }
    )
    timer.add_runtime_exec_summary(
        {
            "timeout_count": 1,
            "failure_count": 2,
            "ms_by_category": {
                "git_diff": 100.0,
                "test": 200.0,
                "raw-secret-command": 999.0,
            },
            "count_by_category": {
                "git_diff": 1,
                "test": 2,
                "raw-secret-command": 9,
            },
        }
    )
    timer.add_mini_swe_command_summary(
        {
            "timeout_count": 0,
            "failure_count": 1,
            "ms_by_category": {"git_diff": 25.0},
            "count_by_category": {"git_diff": 1},
        }
    )

    timing = timer.to_session_timing()

    assert timing.register_to_init_queue_ms == 1000.0
    assert timing.container_start_ms == 1000.0
    assert timing.container_start_ms == timing.runtime_validation_ms
    assert timing.runtime_validation_ms == 1000.0
    assert timing.prepare_ms == 2000.0
    assert timing.agent_setup_ms == 500.0
    assert timing.agent_exec_ms == 3500.0
    assert timing.agent_postprocess_ms == 1000.0
    assert timing.build_ms == 250.0
    assert timing.eval_ms == 1750.0
    assert timing.postrun_exec_ms == 500.0
    assert timing.runtime_stop_ms == 500.0
    assert timing.e2e_ms == 12000.0
    assert timing.runtime_exec_ms == 300.0
    assert timing.runtime_exec_count == 3
    assert timing.runtime_exec_timeout_count == 1
    assert timing.runtime_exec_failure_count == 2
    assert "raw-secret-command" not in timing.runtime_exec_ms_by_category
    assert timing.mini_swe_command_ms_by_category["git_diff"] == 25.0
