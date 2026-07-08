from __future__ import annotations

import json
import logging
import math
from types import SimpleNamespace

from slime_bridge.rollout import (
    _control_plane_headers,
    _log_trajectory_examples_to_wandb,
    _polar_extra_metrics,
    _prefix_eval_metrics,
    _redact_trajectory_value,
    _trajectory_example_records,
    log_rollout_trajectory_examples,
)


def test_control_plane_header_is_attached_without_exposing_token_in_payload(
    monkeypatch,
) -> None:
    monkeypatch.setenv("POLAR_CONTROL_PLANE_TOKEN", "trusted-control-token")

    assert _control_plane_headers() == {
        "Content-Type": "application/json",
        "X-Polar-Control-Token": "trusted-control-token",
    }


def _sample(
    session_id: str,
    reward: float,
    *,
    status: str = "COMPLETED",
    placeholder: bool = False,
    group_index: int = 0,
    rollout_id: int | None = None,
    parser_invalid: bool = False,
    trainable: bool = True,
    detailed_timing: bool = False,
    inference_timings: list[dict[str, float]] | None = None,
    trace_index: int = 0,
    early_stop_cancelled: bool = False,
    agent_timeout: bool = False,
    response_length: int = 1,
    loss_mask: list[int] | None = None,
    missing_loss_mask: bool = False,
    truncated: bool | None = None,
    spilot_router: dict | None = None,
) -> SimpleNamespace:
    polar = {
        "session_id": session_id,
        "session_status": status,
        "placeholder": placeholder,
        "timing": {
            "register_to_init_queue_ms": 1.0,
            "init_ms": 2.0,
            "run_ms": 3.0,
            "postrun_ms": 4.0,
        },
        "trace_index": trace_index,
    }
    if inference_timings is not None:
        polar["trace_metadata"] = {
            "completion_metadata": [
                {"inference_timings": inference_timings},
            ]
        }
    if early_stop_cancelled:
        polar["result_metadata"] = {
            "early_stop_cancelled": True,
            "fully_masked": True,
            "early_stop_elapsed_ms": 123.0,
        }
    if detailed_timing:
        polar["timing"].update(
            {
                "container_start_ms": 5.0,
                "runtime_validation_ms": 5.0,
                "prepare_ms": 6.0,
                "agent_setup_ms": 7.0,
                "agent_exec_ms": 8.0,
                "agent_postprocess_ms": 9.0,
                "build_ms": 10.0,
                "eval_ms": 11.0,
                "postrun_exec_ms": 12.0,
                "runtime_stop_ms": 13.0,
                "e2e_ms": 14.0,
                "runtime_exec_ms": 15.0,
                "runtime_exec_count": 2,
                "runtime_exec_timeout_count": 1,
                "runtime_exec_failure_count": 0,
                "runtime_exec_ms_by_category": {"git_diff": 4.0},
                "runtime_exec_count_by_category": {"git_diff": 1},
                "mini_swe_command_ms": 16.0,
                "mini_swe_command_count": 3,
                "mini_swe_command_timeout_count": 0,
                "mini_swe_command_failure_count": 1,
                "mini_swe_command_ms_by_category": {"git_diff": 5.0},
                "mini_swe_command_count_by_category": {"git_diff": 1},
            }
        )
    if parser_invalid:
        polar["training_filter"] = {"reason": "parser_invalid_tool_call"}
    if agent_timeout:
        polar["training_filter"] = {
            "masked": False,
            "trainable": True,
            "reason": "agent_timeout",
        }
    if spilot_router is not None:
        polar["trajectory_metadata"] = {"evaluation": {"spilot_router": spilot_router}}
    sample_is_truncated = agent_timeout if truncated is None else truncated
    return SimpleNamespace(
        reward={"score": reward},
        metadata={"polar": polar},
        group_index=group_index,
        rollout_id=rollout_id,
        index=rollout_id,
        response_length=response_length,
        loss_mask=(
            None
            if missing_loss_mask
            else (loss_mask if loss_mask is not None else ([1] if trainable else [0]))
        ),
        remove_sample=not trainable,
        status=SimpleNamespace(
            name="TRUNCATED" if sample_is_truncated else "COMPLETED",
            value="truncated" if sample_is_truncated else "completed",
        ),
    )


def test_polar_reward_mean_uses_unique_trainable_non_placeholder_sessions() -> None:
    samples = [
        _sample("completed-1", 1.0),
        _sample("completed-1", 1.0),
        _sample("completed-2", 0.0),
        _sample("timeout-1", 0.0, status="TIMEOUT", placeholder=True),
        _sample("empty-completed", 0.0, placeholder=True),
    ]
    metrics = _polar_extra_metrics(
        samples,
        rewards=[1.0, 1.0, 0.0, 0.0, 0.0],
        reward_key="score",
    )

    assert metrics["polar/reward_mean"] == 0.5
    assert metrics["polar/reward_mean_completed"] == 0.5
    assert metrics["polar/reward_mean_all_samples"] == 0.4
    assert metrics["polar/rollout_success_rate"] == 0.5
    assert metrics["polar/rollout_slot_completion_rate"] == 0.5


def test_polar_reward_mean_averages_traces_within_session_first() -> None:
    samples = [
        _sample("multi-trace", 1.0, trace_index=0),
        _sample("multi-trace", 0.0, trace_index=1),
        _sample("single-trace", 0.0),
        _sample("cancelled", 0.0, status="ERROR", placeholder=True, trainable=False),
    ]

    metrics = _polar_extra_metrics(
        samples,
        rewards=[1.0, 0.0, 0.0, 0.0],
        reward_key="score",
    )

    assert metrics["polar/reward_mean"] == 0.25
    assert metrics["polar/reward_mean_completed"] == 0.25
    assert metrics["polar/reward_mean_all_samples"] == 0.25


def test_spilot_router_metrics_count_each_session_once_with_fixed_slot_keys() -> None:
    route_submit = {
        "action_valid": True,
        "submitted": True,
        "admission_enabled": True,
        "admission_wait_ms": 0,
        "actions": [
            {"step": 0, "valid": True, "action": "ROUTE", "model_slot": "M0"},
            {"step": 1, "valid": True, "action": "SUBMIT"},
        ],
        "calls": [
            {
                "slot": "M0",
                "model": "vendor/model-a",
                "status": "completed",
                "duration_ms": 100,
                "admission_wait_ms": 0,
                "admission_local_cap": 1,
            }
        ],
        "total_cost": 1.0,
        "termination_reason": "router_submit",
        "slot_mapping": {
            "M0": {"model": "vendor/model-a"},
            "M1": {"model": "vendor/model-b"},
        },
    }
    route_verify = {
        "action_valid": True,
        "submitted": True,
        "admission_enabled": True,
        "admission_wait_ms": 3000,
        "admission_fatal_retained": True,
        "admission_node_healthy": False,
        "actions": [
            {"step": 0, "valid": True, "action": "ROUTE", "model_slot": "M1"},
            {"step": 1, "valid": True, "action": "VERIFY", "model_slot": "M0"},
        ],
        "calls": [
            {
                "slot": "M1",
                "model": "vendor/model-b",
                "status": "failed",
                "duration_ms": 200,
                "admission_wait_ms": 3000,
                "admission_local_cap": 4,
            },
            {
                "slot": "M0",
                "model": "vendor/model-a",
                "status": "timeout",
                "duration_ms": 1200,
                "admission_wait_ms": 1000,
                "admission_local_cap": 1,
            },
        ],
        "total_cost": 3.0,
        "termination_reason": "verify_auto_submit",
        "slot_mapping": {
            "M0": {"model": "vendor/model-a"},
            "M1": {"model": "vendor/model-b"},
        },
    }
    invalid = {
        "action_valid": False,
        "submitted": False,
        "actions": [{"step": 0, "valid": False}],
        "calls": [],
        "total_cost": 0.0,
        "termination_reason": "invalid_action_step_0",
        "slot_mapping": {},
    }
    samples = [
        _sample("submit", 1.0, spilot_router=route_submit, trace_index=0),
        # A second trace from the same trajectory must not duplicate Router metrics.
        _sample("submit", 1.0, spilot_router=route_submit, trace_index=1),
        _sample("verify", 0.0, spilot_router=route_verify),
        _sample("invalid", 0.0, spilot_router=invalid),
    ]

    metrics = _polar_extra_metrics(samples, rewards=[1.0, 1.0, 0.0, 0.0], reward_key="score")
    prefix = "polar/spilot_router"

    assert metrics[f"{prefix}/session_count"] == 3.0
    assert metrics[f"{prefix}/action_valid_count"] == 2.0
    assert metrics[f"{prefix}/action_valid_fraction"] == 2 / 3
    assert metrics[f"{prefix}/submitted_count"] == 2.0
    assert metrics[f"{prefix}/submitted_fraction"] == 2 / 3
    assert metrics[f"{prefix}/route_m0_count"] == 1.0
    assert metrics[f"{prefix}/route_m1_count"] == 1.0
    assert metrics[f"{prefix}/verify_m0_count"] == 1.0
    assert metrics[f"{prefix}/verify_m1_count"] == 0.0
    assert metrics[f"{prefix}/route_candidate_c0_count"] == 1.0
    assert metrics[f"{prefix}/route_candidate_c1_count"] == 1.0
    assert metrics[f"{prefix}/verify_candidate_c0_count"] == 1.0
    assert metrics[f"{prefix}/verify_candidate_c1_count"] == 0.0
    assert metrics[f"{prefix}/direct_submit_count"] == 1.0
    assert metrics[f"{prefix}/pool_call_count"] == 3.0
    assert metrics[f"{prefix}/pool_completed_count"] == 1.0
    assert metrics[f"{prefix}/pool_failed_count"] == 1.0
    assert metrics[f"{prefix}/pool_timeout_count"] == 1.0
    assert metrics[f"{prefix}/pool_completed_candidate_c0_count"] == 1.0
    assert metrics[f"{prefix}/pool_failed_candidate_c0_count"] == 0.0
    assert metrics[f"{prefix}/pool_timeout_candidate_c0_count"] == 1.0
    assert metrics[f"{prefix}/pool_completed_candidate_c1_count"] == 0.0
    assert metrics[f"{prefix}/pool_failed_candidate_c1_count"] == 1.0
    assert metrics[f"{prefix}/pool_timeout_candidate_c1_count"] == 0.0
    assert metrics[f"{prefix}/pool_duration_candidate_c0_count"] == 2.0
    assert metrics[f"{prefix}/pool_duration_candidate_c0_mean_ms"] == 650.0
    assert metrics[f"{prefix}/pool_duration_candidate_c0_max_ms"] == 1200.0
    assert metrics[f"{prefix}/pool_duration_candidate_c1_count"] == 1.0
    assert metrics[f"{prefix}/pool_duration_candidate_c1_mean_ms"] == 200.0
    assert metrics[f"{prefix}/pool_duration_candidate_c1_max_ms"] == 200.0
    assert metrics[f"{prefix}/admission_wait_candidate_c0_accounted_count"] == 2.0
    assert metrics[f"{prefix}/admission_wait_candidate_c0_mean_ms"] == 500.0
    assert metrics[f"{prefix}/admission_wait_candidate_c0_max_ms"] == 1000.0
    assert metrics[f"{prefix}/admission_wait_candidate_c1_accounted_count"] == 1.0
    assert metrics[f"{prefix}/admission_wait_candidate_c1_mean_ms"] == 3000.0
    assert metrics[f"{prefix}/admission_wait_candidate_c1_max_ms"] == 3000.0
    assert metrics[f"{prefix}/total_cost"] == 4.0
    assert metrics[f"{prefix}/admission_session_count"] == 2.0
    assert metrics[f"{prefix}/admission_wait_ms_total"] == 3000.0
    assert metrics[f"{prefix}/admission_wait_ms_mean"] == 1500.0
    assert metrics[f"{prefix}/admission_wait_ms_max"] == 3000.0
    assert metrics[f"{prefix}/admission_wait_accounted_session_count"] == 2.0
    assert metrics[f"{prefix}/admission_waited_session_count"] == 1.0
    assert metrics[f"{prefix}/admission_waited_session_fraction"] == 0.5
    assert metrics[f"{prefix}/admission_local_cap_observation_count"] == 3.0
    assert metrics[f"{prefix}/admission_local_cap_mean"] == 2.0
    assert metrics[f"{prefix}/admission_local_cap_min"] == 1.0
    assert metrics[f"{prefix}/admission_local_cap_max"] == 4.0
    assert metrics[f"{prefix}/admission_fatal_retained_session_count"] == 1.0
    assert metrics[f"{prefix}/admission_fatal_retained_session_fraction"] == 0.5
    # The first session has no node-health field. Missing telemetry is unknown,
    # not a healthy vote; only the explicit false value is accounted here.
    assert metrics[f"{prefix}/admission_node_health_accounted_session_count"] == 1.0
    assert metrics[f"{prefix}/admission_node_health_accounted_session_fraction"] == 0.5
    assert metrics[f"{prefix}/admission_node_healthy_session_count"] == 0.0
    assert metrics[f"{prefix}/admission_node_healthy_session_fraction"] == 0.0
    assert metrics[f"{prefix}/reward_accounted_session_count"] == 3.0
    assert metrics[f"{prefix}/reward_mean"] == 1 / 3
    assert metrics[f"{prefix}/reward_m0_mean"] == 1.0
    assert metrics[f"{prefix}/reward_m1_mean"] == 0.0
    assert metrics[f"{prefix}/reward_candidate_c0_mean"] == 1.0
    assert metrics[f"{prefix}/reward_candidate_c1_mean"] == 0.0
    assert not any("vendor/model" in key for key in metrics)


def test_spilot_router_candidate_metrics_are_stable_when_slots_shuffle() -> None:
    def metadata(
        *,
        route_slot: str,
        verify_slot: str,
        slot_mapping: dict[str, dict[str, str]],
    ) -> dict:
        return {
            "action_valid": True,
            "submitted": True,
            "actions": [
                {
                    "step": 0,
                    "valid": True,
                    "action": "ROUTE",
                    "model_slot": route_slot,
                },
                {
                    "step": 1,
                    "valid": True,
                    "action": "VERIFY",
                    "model_slot": verify_slot,
                },
            ],
            "calls": [],
            "total_cost": 0.0,
            "termination_reason": "verify_auto_submit",
            "slot_mapping": slot_mapping,
        }

    # In both sessions the initial candidate is model-z (stable C1), and the
    # verifier is model-a (stable C0), despite opposite M0/M1 assignments.
    samples = [
        _sample(
            "shuffle-a",
            1.0,
            spilot_router=metadata(
                route_slot="M0",
                verify_slot="M1",
                slot_mapping={
                    "M0": {"model": "pool/model-z"},
                    "M1": {"model": "pool/model-a"},
                },
            ),
        ),
        _sample(
            "shuffle-b",
            0.0,
            spilot_router=metadata(
                route_slot="M1",
                verify_slot="M0",
                slot_mapping={
                    "M0": {"model": "pool/model-a"},
                    "M1": {"model": "pool/model-z"},
                },
            ),
        ),
    ]

    metrics = _polar_extra_metrics(samples, rewards=[1.0, 0.0], reward_key="score")
    prefix = "polar/spilot_router"

    assert metrics[f"{prefix}/route_m0_count"] == 1.0
    assert metrics[f"{prefix}/route_m1_count"] == 1.0
    assert metrics[f"{prefix}/verify_m0_count"] == 1.0
    assert metrics[f"{prefix}/verify_m1_count"] == 1.0
    assert metrics[f"{prefix}/route_candidate_c0_count"] == 0.0
    assert metrics[f"{prefix}/route_candidate_c1_count"] == 2.0
    assert metrics[f"{prefix}/verify_candidate_c0_count"] == 2.0
    assert metrics[f"{prefix}/verify_candidate_c1_count"] == 0.0
    assert metrics[f"{prefix}/reward_candidate_c0_count"] == 0.0
    assert metrics[f"{prefix}/reward_candidate_c1_count"] == 2.0
    assert f"{prefix}/reward_candidate_c0_mean" not in metrics
    assert metrics[f"{prefix}/reward_candidate_c1_mean"] == 0.5
    assert not any("model-a" in key or "model-z" in key for key in metrics)


def test_spilot_router_attributes_admission_failures_without_fabricating_pool_calls() -> None:
    def failed_metadata(*, qwen_slot: str, wait_ms: int) -> dict:
        other_slot = "M1" if qwen_slot == "M0" else "M0"
        return {
            "action_valid": True,
            "submitted": False,
            "admission_enabled": True,
            "admission_wait_ms": wait_ms,
            "admission_failure": {
                "model": "pool/qwen",
                "attempt_id": "0:solve",
                "wait_ms": wait_ms,
                "error": "queue unavailable",
            },
            "actions": [
                {
                    "step": 0,
                    "valid": True,
                    "action": "ROUTE",
                    "model_slot": qwen_slot,
                }
            ],
            "calls": [],
            "total_cost": 0.0,
            "termination_reason": "infrastructure_error",
            "slot_mapping": {
                qwen_slot: {"model": "pool/qwen"},
                other_slot: {"model": "pool/gpt"},
            },
        }

    samples = [
        _sample(
            "admission-a",
            0.0,
            status="ERROR",
            placeholder=True,
            trainable=False,
            spilot_router=failed_metadata(qwen_slot="M0", wait_ms=1000),
        ),
        _sample(
            "admission-b",
            0.0,
            status="ERROR",
            placeholder=True,
            trainable=False,
            spilot_router=failed_metadata(qwen_slot="M1", wait_ms=3000),
        ),
    ]

    metrics = _polar_extra_metrics(samples, rewards=[0.0, 0.0], reward_key="score")
    prefix = "polar/spilot_router"

    # pool/gpt sorts to C0 and pool/qwen to C1, independently of M0/M1 shuffle.
    assert metrics[f"{prefix}/admission_failure_count"] == 2.0
    assert metrics[f"{prefix}/admission_failure_candidate_c0_count"] == 0.0
    assert metrics[f"{prefix}/admission_failure_candidate_c1_count"] == 2.0
    assert metrics[f"{prefix}/admission_failure_wait_candidate_c1_accounted_count"] == 2.0
    assert metrics[f"{prefix}/admission_failure_wait_candidate_c1_mean_ms"] == 2000.0
    assert metrics[f"{prefix}/admission_failure_wait_candidate_c1_max_ms"] == 3000.0
    assert metrics[f"{prefix}/pool_call_count"] == 0.0
    assert metrics[f"{prefix}/reward_accounted_session_count"] == 0.0


def test_spilot_router_does_not_attribute_inconsistent_call_model_and_slot() -> None:
    router_metadata = {
        "action_valid": True,
        "submitted": True,
        "actions": [
            {"step": 0, "valid": True, "action": "ROUTE", "model_slot": "M0"},
            {"step": 1, "valid": True, "action": "SUBMIT"},
        ],
        # M0 names model-a, but the call claims model-b. Aggregate failure
        # telemetry remains visible while candidate attribution fails closed.
        "calls": [
            {
                "slot": "M0",
                "model": "pool/model-b",
                "status": "completed",
                "duration_ms": 100,
            }
        ],
        "total_cost": 1.0,
        "termination_reason": "router_submit",
        "slot_mapping": {
            "M0": {"model": "pool/model-a"},
            "M1": {"model": "pool/model-b"},
        },
    }

    metrics = _polar_extra_metrics(
        [_sample("inconsistent-call", 1.0, spilot_router=router_metadata)],
        rewards=[1.0],
        reward_key="score",
    )
    prefix = "polar/spilot_router"

    assert metrics[f"{prefix}/pool_call_count"] == 1.0
    assert metrics[f"{prefix}/pool_completed_count"] == 1.0
    assert metrics[f"{prefix}/pool_unattributed_call_count"] == 1.0
    assert metrics[f"{prefix}/pool_completed_candidate_c0_count"] == 0.0
    assert metrics[f"{prefix}/pool_completed_candidate_c1_count"] == 0.0
    assert metrics[f"{prefix}/pool_duration_candidate_c0_count"] == 0.0
    assert metrics[f"{prefix}/pool_duration_candidate_c1_count"] == 0.0


def test_spilot_router_reward_metrics_omit_unsafe_unaccounted_session() -> None:
    router_metadata = {
        "action_valid": True,
        "submitted": True,
        "actions": [{"step": 0, "valid": True, "action": "ROUTE", "model_slot": "M0"}],
        "calls": [],
        # Invalid legacy telemetry must neither poison W&B with NaN nor count as cost.
        "total_cost": float("nan"),
        "termination_reason": "infrastructure_error",
        "slot_mapping": {"M0": {"model": "arbitrary/dynamic-model-id"}},
    }
    unaccounted = _sample(
        "unaccounted",
        1.0,
        status="ERROR",
        placeholder=True,
        trainable=False,
        spilot_router=router_metadata,
    )

    metrics = _polar_extra_metrics([unaccounted], rewards=[1.0], reward_key="score")
    prefix = "polar/spilot_router"

    assert metrics[f"{prefix}/session_count"] == 1.0
    assert metrics[f"{prefix}/total_cost"] == 0.0
    assert metrics[f"{prefix}/reward_accounted_session_count"] == 0.0
    assert f"{prefix}/reward_mean" not in metrics
    assert f"{prefix}/reward_m0_mean" not in metrics
    assert metrics[f"{prefix}/route_candidate_c0_count"] == 0.0
    assert metrics[f"{prefix}/route_candidate_c1_count"] == 0.0
    assert metrics[f"{prefix}/reward_candidate_c0_count"] == 0.0
    assert metrics[f"{prefix}/reward_candidate_c1_count"] == 0.0
    assert not any("dynamic-model-id" in key for key in metrics)


def test_polar_reward_mean_counts_trusted_model_failure_but_not_early_stop() -> None:
    completed = _sample("completed", 1.0)
    trusted_failure = _sample(
        "model-failure",
        1.0,
        status="ERROR",
        placeholder=True,
        trainable=False,
    )
    trusted_failure.metadata["polar"]["trajectory_metadata"] = {
        "evaluation": {
            "verifier_reward_accepted": True,
            "verifier_exit_code": 0,
        }
    }
    early_stop = _sample(
        "early-stop",
        0.0,
        status="ERROR",
        placeholder=True,
        trainable=False,
        early_stop_cancelled=True,
    )
    early_stop.metadata["polar"]["trajectory_metadata"] = {
        "evaluation": {
            "verifier_reward_accepted": True,
            "verifier_exit_code": 0,
        }
    }
    untrusted_infra_failure = _sample(
        "infra-failure",
        0.0,
        status="ERROR",
        placeholder=True,
        trainable=False,
    )

    metrics = _polar_extra_metrics(
        [completed, trusted_failure, early_stop, untrusted_infra_failure],
        rewards=[1.0, 1.0, 0.0, 0.0],
        reward_key="score",
    )

    assert metrics["polar/reward_mean"] == 0.5
    assert metrics["polar/reward_mean_completed"] == 1.0
    assert metrics["polar/reward_mean_all_samples"] == 0.5
    assert metrics["polar/reward_accounted_sessions"] == 2.0
    assert metrics["polar/reward_model_failure_sessions"] == 1.0


def test_polar_metrics_expose_parser_filter_and_real_grpo_signal() -> None:
    samples = [
        _sample("g0-a", 1.0, group_index=0, rollout_id=0),
        _sample("g0-b", 0.0, group_index=0, rollout_id=1),
        _sample("g1-a", 1.0, group_index=1, rollout_id=2),
        _sample("g1-b", 1.0, group_index=1, rollout_id=3),
        _sample(
            "g1-invalid",
            1.0,
            group_index=1,
            rollout_id=4,
            parser_invalid=True,
            trainable=False,
        ),
    ]

    metrics = _polar_extra_metrics(
        samples,
        rewards=[1.0, 0.0, 1.0, 1.0, 1.0],
        reward_key="score",
    )

    assert metrics["polar/training_filter/parser_invalid_trace_fraction"] == 0.2
    assert metrics["polar/training_filter/parser_invalid_session_fraction"] == 0.2
    assert metrics["polar/training_filter/trainable_trace_fraction"] == 0.8
    assert metrics["polar/reward_groups/count"] == 2.0
    assert metrics["polar/reward_groups/mixed_fraction"] == 0.5
    assert metrics["polar/reward_groups/trainable_fraction"] == 0.5
    # A stale positive scalar on the fully masked parser-invalid sample stays
    # visible only in the all-samples diagnostic, never primary quality.
    assert metrics["polar/reward_mean"] == 0.6
    assert metrics["polar/reward_mean_completed"] == 0.6
    assert metrics["polar/reward_mean_all_samples"] == 0.8


def test_polar_metrics_count_trainable_agent_timeout_as_model_failure() -> None:
    samples = [
        _sample("completed", 1.0, group_index=0, rollout_id=0),
        _sample(
            "agent-timeout",
            0.0,
            status="TIMEOUT",
            group_index=0,
            rollout_id=1,
            agent_timeout=True,
        ),
    ]

    metrics = _polar_extra_metrics(
        samples,
        rewards=[1.0, 0.0],
        reward_key="score",
    )

    assert metrics["polar/reward_mean_completed"] == 1.0
    assert metrics["polar/reward_mean"] == 0.5
    assert metrics["polar/reward_accounted_sessions"] == 2.0
    assert metrics["polar/reward_trainable_agent_timeout_sessions"] == 1.0
    assert metrics["polar/reward_model_failure_sessions"] == 1.0
    assert metrics["polar/training_filter/agent_timeout_trace_fraction"] == 0.5
    assert metrics["polar/training_filter/agent_timeout_session_fraction"] == 0.5
    assert metrics["polar/training_filter/trainable_trace_fraction"] == 1.0
    assert metrics["polar/reward_groups/mixed_fraction"] == 1.0
    assert metrics["polar/reward_groups/trainable_fraction"] == 1.0
    assert metrics["polar/rollout_slot_completion_rate"] == 1.0
    assert metrics["polar/rollout_success_rate"] == 0.5


def test_polar_metrics_report_session_trainability_and_terminal_failures() -> None:
    samples = [
        _sample("usable", 1.0, trace_index=0),
        _sample("usable", 1.0, trace_index=1),
        _sample("empty-completed", 0.0, placeholder=True, trainable=False),
        _sample("timeout", 0.0, status="TIMEOUT", trainable=False),
        _sample("error", 0.0, status="ERROR", trainable=False),
        _sample(
            "early-stop",
            0.0,
            status="ERROR",
            placeholder=True,
            trainable=False,
            early_stop_cancelled=True,
        ),
    ]

    metrics = _polar_extra_metrics(
        samples,
        rewards=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        reward_key="score",
    )

    assert metrics["polar/rollout_attempted_sessions"] == 4.0
    assert metrics["polar/rollout_attempted_session_fraction"] == 0.8
    assert metrics["polar/rollout_trainable_sessions"] == 1.0
    assert metrics["polar/rollout_trainable_session_fraction"] == 0.25
    assert metrics["polar/rollout_fully_masked_sessions"] == 3.0
    assert metrics["polar/rollout_fully_masked_session_fraction"] == 0.75
    assert metrics["polar/terminal_timeout_sessions"] == 1.0
    assert metrics["polar/terminal_error_sessions"] == 1.0
    assert metrics["polar/rollout_successful_sessions"] == 1.0
    assert metrics["polar/rollout_success_rate"] == 0.25


def test_polar_metrics_split_timeout_source_stage_and_trainability_by_session() -> None:
    exec_trace_0 = _sample(
        "agent-exec",
        0.0,
        status="TIMEOUT",
        agent_timeout=True,
        trace_index=0,
    )
    exec_trace_1 = _sample(
        "agent-exec",
        0.0,
        status="TIMEOUT",
        agent_timeout=True,
        trace_index=1,
    )
    postprocess = _sample("agent-postprocess", 0.0, status="TIMEOUT", trainable=False)
    session_timeout = _sample("session-exec", 0.0, status="TIMEOUT", trainable=False)
    for sample, timeout_source, timeout_stage in (
        (exec_trace_0, "agent", "exec"),
        (exec_trace_1, "agent", "exec"),
        (postprocess, "agent", "postprocess"),
        (session_timeout, "session", "exec"),
    ):
        sample.metadata["polar"]["trajectory_metadata"] = {
            "agent_result": {
                "status": "timeout",
                "timeout_source": timeout_source,
                "timeout_stage": timeout_stage,
            }
        }

    samples = [exec_trace_0, exec_trace_1, postprocess, session_timeout]
    metrics = _polar_extra_metrics(
        samples,
        rewards=[0.0] * len(samples),
        reward_key="score",
    )

    assert metrics["polar/rollout_attempted_sessions"] == 3.0
    assert metrics["polar/rollout_trainable_sessions"] == 1.0
    assert metrics["polar/rollout_fully_masked_sessions"] == 2.0
    assert metrics["polar/terminal_timeout_sessions"] == 3.0
    assert metrics["polar/timeout_agent_exec_sessions"] == 1.0
    assert metrics["polar/timeout_agent_postprocess_sessions"] == 1.0
    assert metrics["polar/timeout_trainable_sessions"] == 1.0
    assert metrics["polar/timeout_masked_sessions"] == 2.0
    assert metrics["polar/rollout_success_rate"] == 0.0


def test_polar_metrics_aggregate_response_lengths_by_session() -> None:
    samples = [
        _sample(
            "multi",
            1.0,
            trace_index=0,
            response_length=5,
            loss_mask=[1, 1, 0],
        ),
        _sample(
            "multi",
            1.0,
            trace_index=1,
            response_length=7,
            loss_mask=[1, 0, 1, 1],
        ),
        _sample("single", 0.0, response_length=9, loss_mask=[1]),
        _sample(
            "placeholder",
            0.0,
            placeholder=True,
            trainable=False,
            response_length=99,
            loss_mask=[0] * 99,
        ),
        # Intentional early-stop traces are excluded even if malformed input
        # carries real-looking tokens instead of the normal placeholder.
        _sample(
            "early-stop",
            0.0,
            early_stop_cancelled=True,
            response_length=50,
            loss_mask=[1] * 50,
        ),
    ]

    metrics = _polar_extra_metrics(
        samples,
        rewards=[1.0, 1.0, 0.0, 0.0, 0.0],
        reward_key="score",
    )

    assert metrics["polar/session_trainable_response_tokens/count"] == 2.0
    assert metrics["polar/session_trainable_response_tokens/mean"] == 3.0
    assert metrics["polar/session_trainable_response_tokens/median"] == 3.0
    assert metrics["polar/session_trainable_response_tokens/min"] == 1.0
    assert metrics["polar/session_trainable_response_tokens/max"] == 5.0
    assert metrics["polar/session_raw_response_tokens/mean"] == 10.5
    assert metrics["polar/session_raw_response_tokens/median"] == 10.5
    assert metrics["polar/session_raw_response_tokens/min"] == 9.0
    assert metrics["polar/session_raw_response_tokens/max"] == 12.0
    assert metrics["polar/session_trainable_response_tokens/by_status/completed_mean"] == 3.0


def test_polar_metrics_fallback_to_raw_length_when_loss_mask_is_missing() -> None:
    sample = _sample(
        "legacy",
        1.0,
        response_length=7,
        missing_loss_mask=True,
    )

    metrics = _polar_extra_metrics([sample], rewards=[1.0], reward_key="score")

    assert metrics["polar/session_trainable_response_tokens/count"] == 1.0
    assert metrics["polar/session_trainable_response_tokens/mean"] == 7.0
    assert metrics["polar/session_raw_response_tokens/mean"] == 7.0


def test_polar_metrics_decompose_trace_and_session_truncation_and_status() -> None:
    samples = [
        _sample("completed", 1.0, trace_index=0, truncated=True),
        _sample("completed", 1.0, trace_index=1),
        _sample("agent-timeout", 0.0, status="TIMEOUT", agent_timeout=True),
        _sample(
            "other-timeout",
            0.0,
            status="TIMEOUT",
            trainable=False,
            truncated=True,
        ),
        _sample("error", 0.0, status="ERROR"),
        _sample("unknown", 0.0, status="MYSTERY"),
        # Defensive malformed input: timeout metadata alone must not count as
        # truncation when the Slime sample status is COMPLETED.
        _sample(
            "agent-metadata-completed",
            0.0,
            status="COMPLETED",
            agent_timeout=True,
            truncated=False,
        ),
        _sample(
            "placeholder-error",
            0.0,
            status="ERROR",
            placeholder=True,
            trainable=False,
        ),
        _sample(
            "early-stop",
            0.0,
            status="ERROR",
            early_stop_cancelled=True,
            truncated=True,
        ),
    ]

    metrics = _polar_extra_metrics(
        samples,
        rewards=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        reward_key="score",
    )

    assert metrics["polar/trace_truncation/truncated_fraction"] == 3 / 7
    assert metrics["polar/trace_truncation/agent_timeout_fraction"] == 1 / 7
    assert metrics["polar/trace_truncation/non_agent_timeout_fraction"] == 2 / 7
    assert metrics["polar/trace_truncation/truncated_fraction"] == (
        metrics["polar/trace_truncation/agent_timeout_fraction"]
        + metrics["polar/trace_truncation/non_agent_timeout_fraction"]
    )
    assert metrics["polar/session_truncation/session_count"] == 6.0
    assert metrics["polar/session_truncation/truncated_count"] == 3.0
    assert metrics["polar/session_truncation/truncated_fraction"] == 3 / 6
    assert metrics["polar/session_truncation/agent_timeout_fraction"] == 1 / 6
    assert metrics["polar/session_truncation/non_agent_timeout_fraction"] == 2 / 6
    assert metrics["polar/session_truncation/truncated_fraction"] == (
        metrics["polar/session_truncation/agent_timeout_fraction"]
        + metrics["polar/session_truncation/non_agent_timeout_fraction"]
    )

    # Session fan-out is deduplicated, placeholders remain visible as status
    # failures, and intentional early-stop cancellation is absent.
    assert metrics["polar/session_status/completed_count"] == 2.0
    assert metrics["polar/session_status/timeout_count"] == 2.0
    assert metrics["polar/session_status/error_count"] == 2.0
    assert metrics["polar/session_status/unknown_count"] == 1.0
    assert metrics["polar/session_status/completed_fraction"] == 2 / 7
    assert metrics["polar/session_status/timeout_fraction"] == 2 / 7
    assert metrics["polar/session_status/error_fraction"] == 2 / 7
    assert metrics["polar/session_status/unknown_fraction"] == 1 / 7


def test_polar_metrics_report_accounted_outcomes_and_lengths() -> None:
    positive = _sample("positive", 1.0, response_length=8, loss_mask=[1] * 4)
    zero = _sample("zero", 0.0, response_length=6, loss_mask=[1] * 2)
    negative = _sample("negative", -1.0, response_length=10, loss_mask=[1] * 6)
    agent_timeout = _sample(
        "agent-timeout",
        1.0,
        status="TIMEOUT",
        agent_timeout=True,
        response_length=12,
        loss_mask=[1] * 3,
    )
    trusted_failure = _sample(
        "trusted-failure",
        1.0,
        status="ERROR",
        placeholder=True,
        trainable=False,
    )
    trusted_failure.metadata["polar"]["trajectory_metadata"] = {
        "evaluation": {
            "verifier_reward_accepted": True,
            "verifier_exit_code": 0,
        }
    }
    infra_failure = _sample(
        "infra-failure",
        0.0,
        status="ERROR",
        placeholder=True,
        trainable=False,
    )
    early_stop = _sample(
        "early-stop",
        0.0,
        status="ERROR",
        placeholder=True,
        trainable=False,
        early_stop_cancelled=True,
    )
    samples = [
        positive,
        zero,
        negative,
        agent_timeout,
        trusted_failure,
        infra_failure,
        early_stop,
    ]

    metrics = _polar_extra_metrics(
        samples,
        rewards=[1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0],
        reward_key="score",
    )

    assert metrics["polar/reward_accounted_sessions"] == 5.0
    assert metrics["polar/session_outcome/accounted_count"] == 5.0
    assert metrics["polar/session_outcome/unaccounted_count"] == 1.0
    assert metrics["polar/session_outcome/accounted_fraction"] == 5 / 6
    assert metrics["polar/session_outcome/unaccounted_fraction"] == 1 / 6
    assert metrics["polar/session_outcome/positive_count"] == 1.0
    assert metrics["polar/session_outcome/zero_count"] == 3.0
    assert metrics["polar/session_outcome/negative_count"] == 1.0
    assert metrics["polar/session_outcome/positive_fraction_of_accounted"] == 1 / 5
    assert metrics["polar/session_outcome/zero_fraction_of_accounted"] == 3 / 5
    assert metrics["polar/session_outcome/negative_fraction_of_accounted"] == 1 / 5
    assert metrics["polar/session_trainable_response_tokens/by_outcome/positive_mean"] == 4.0
    assert metrics["polar/session_trainable_response_tokens/by_outcome/zero_mean"] == 2.5
    assert metrics["polar/session_trainable_response_tokens/by_outcome/negative_mean"] == 6.0


def test_polar_metrics_omit_empty_length_and_truncation_distributions() -> None:
    placeholder = _sample(
        "infra-failure",
        0.0,
        status="ERROR",
        placeholder=True,
        trainable=False,
    )

    metrics = _polar_extra_metrics([placeholder], rewards=[0.0], reward_key="score")

    assert not any(key.startswith("polar/session_trainable_response_tokens/") for key in metrics)
    assert not any(key.startswith("polar/session_raw_response_tokens/") for key in metrics)
    assert not any(key.startswith("polar/trace_truncation/") for key in metrics)
    assert not any(key.startswith("polar/session_truncation/") for key in metrics)
    assert metrics["polar/session_status/error_count"] == 1.0
    assert metrics["polar/session_outcome/unaccounted_count"] == 1.0
    assert all(math.isfinite(value) for value in metrics.values())


def test_polar_metrics_fail_close_nonfinite_and_boolean_rewards() -> None:
    samples = [
        _sample("nan", float("nan")),
        _sample("positive-inf", float("inf")),
        _sample("negative-inf", float("-inf")),
        _sample("boolean", True),
    ]

    metrics = _polar_extra_metrics(
        samples,
        rewards=[float("nan"), float("inf"), float("-inf"), True],
        reward_key="score",
    )

    assert metrics["polar/reward_mean_all_samples"] == 0.0
    assert metrics["polar/reward_mean"] == 0.0
    assert metrics["polar/reward_mean_completed"] == 0.0
    assert metrics["polar/reward_accounted_sessions"] == 4.0
    assert all(math.isfinite(value) for value in metrics.values())


def test_polar_metrics_aggregate_detailed_pipeline_and_command_timings() -> None:
    samples = [
        _sample("s1", 1.0, detailed_timing=True),
        _sample("s1", 1.0, detailed_timing=True),
        _sample("s2", 0.0),
    ]

    metrics = _polar_extra_metrics(samples, rewards=[1.0, 1.0, 0.0], reward_key="score")

    assert metrics["timing/session_ms/container_start_mean"] == 2.5
    assert metrics["timing/session_ms/runtime_validation_mean"] == 2.5
    assert metrics["timing/session_ms/agent_exec_mean"] == 4.0
    assert metrics["timing/session_ms/e2e_mean"] == 7.0
    assert metrics["timing/runtime_exec/ms_per_session_mean"] == 7.5
    assert metrics["polar/runtime_exec/count_per_session_mean"] == 1.0
    assert metrics["timing/runtime_exec/ms_per_command_mean"] == 7.5
    assert metrics["timing/runtime_exec/git_diff_ms_per_session_mean"] == 2.0
    assert metrics["timing/runtime_exec/git_diff_ms_per_command_mean"] == 4.0
    assert metrics["timing/mini_swe_command/ms_per_session_mean"] == 8.0
    assert metrics["polar/mini_swe_command/git_diff_count_per_session_mean"] == 0.5
    assert metrics["timing/mini_swe_command/git_diff_ms_per_command_mean"] == 5.0


def test_polar_metrics_aggregate_sglang_inference_substages() -> None:
    samples = [
        _sample(
            "s1",
            1.0,
            inference_timings=[
                {
                    "e2e_ms": 100.0,
                    "queue_ms": 10.0,
                    "prefill_forward_ms": 20.0,
                    "decode_ms": 60.0,
                    "inference_service_ms": 80.0,
                    "forward_ms": 70.0,
                    # Legacy persisted traces may still carry SGLang's
                    # invalid non-streaming tokenizer-manager throughput.
                    "decode_throughput_tokens_per_s": 99_999_999.0,
                },
                {
                    "e2e_ms": 200.0,
                    "queue_ms": 30.0,
                    "prefill_forward_ms": 40.0,
                    "decode_ms": 120.0,
                    "inference_service_ms": 160.0,
                    "forward_ms": 140.0,
                },
            ],
            trace_index=0,
        ),
        # Same session/trace must not double-count merged completion metadata.
        _sample(
            "s1",
            1.0,
            inference_timings=[{"e2e_ms": 9999.0}],
            trace_index=0,
        ),
    ]

    metrics = _polar_extra_metrics(samples, rewards=[1.0, 1.0], reward_key="score")

    assert metrics["polar/inference/timed_completion_count"] == 2.0
    assert metrics["timing/inference/e2e_ms_mean"] == 150.0
    assert metrics["timing/inference/e2e_ms_p95"] == 200.0
    assert metrics["timing/inference/e2e_ms_max"] == 200.0
    assert metrics["timing/inference/queue_ms_mean"] == 20.0
    assert metrics["timing/inference/prefill_forward_ms_mean"] == 30.0
    assert metrics["timing/inference/decode_ms_mean"] == 90.0
    assert metrics["timing/inference/inference_service_ms_mean"] == 120.0
    assert metrics["timing/inference/forward_ms_mean"] == 105.0
    assert not any("decode_throughput" in key for key in metrics)


def test_polar_metrics_separate_intentional_straggler_cancellation() -> None:
    samples = [
        _sample("usable", 1.0),
        _sample(
            "cancelled",
            0.0,
            status="ERROR",
            placeholder=True,
            trainable=False,
            early_stop_cancelled=True,
        ),
    ]

    metrics = _polar_extra_metrics(samples, rewards=[1.0, 0.0], reward_key="score")

    assert metrics["polar/early_stop/cancelled_sessions"] == 1.0
    assert metrics["polar/early_stop/cancelled_session_fraction"] == 0.5
    assert metrics["polar/rollout_success_rate"] == 1.0
    assert metrics["polar/rollout_slot_completion_rate"] == 0.5
    assert metrics["timing/early_stop/elapsed_ms_mean"] == 123.0
    # The synthetic all-zero timing is excluded from completed stage means.
    assert metrics["timing/session_ms/run_mean"] == 3.0


def test_eval_metrics_use_quality_and_timing_namespaces() -> None:
    assert _prefix_eval_metrics(
        "swebench",
        {
            "polar/reward_mean": 0.5,
            "polar/resolved_rate": 0.25,
            "polar/session_trainable_response_tokens/mean": 42.0,
            "polar/trace_truncation/non_agent_timeout_fraction": 0.125,
            "timing/session_ms/e2e_mean": 123.0,
            "custom_count": 4.0,
        },
    ) == {
        "eval/swebench/reward_mean": 0.5,
        "eval/swebench/resolved_rate": 0.25,
        "eval/swebench/session_trainable_response_tokens/mean": 42.0,
        "eval/swebench/trace_truncation/non_agent_timeout_fraction": 0.125,
        "timing/eval/swebench/session_ms/e2e_mean": 123.0,
        "eval/swebench/custom_count": 4.0,
    }


def _artifact_sample(
    session_id: str = "session-1",
    *,
    rollout_id: int = 0,
    reward: float = 1.0,
    response_length: int = 3,
    response_content: str = "full model response",
) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=session_id,
        rollout_id=rollout_id,
        index=rollout_id,
        response_length=response_length,
        metadata={
            "polar": {
                "trace_index": 0,
                "task_id": f"task-{session_id}",
                "node_id": "node-1",
                "session_id": session_id,
                "session_status": "COMPLETED",
                "trace_debug": {
                    "finish_reason": "stop",
                    "response_messages": [{"role": "assistant", "content": response_content}],
                },
            }
        },
        prompt=[{"role": "user", "content": f"prompt for {session_id}"}],
        status=None,
        reward={"score": reward},
        remove_sample=False,
    )


def _example_args(*, use_wandb: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        global_batch_size=2,
        num_steps_per_rollout=2,
        use_wandb=use_wandb,
        debug_rollout_only=False,
    )


def test_post_train_hook_saves_two_full_jsonl_examples_at_exact_train_step(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("POLAR_ROLLOUT_EXAMPLES_DIR", str(tmp_path))
    monkeypatch.setenv("POLAR_ROLLOUT_EXAMPLE_INTERVAL", "10")
    monkeypatch.setenv("POLAR_ROLLOUT_EXAMPLE_COUNT", "2")
    monkeypatch.setenv("POLAR_ROLLOUT_EXAMPLES_WANDB", "0")
    samples = [
        _artifact_sample("high", rollout_id=0, reward=1.0, response_length=7),
        _artifact_sample("low", rollout_id=1, reward=0.0, response_length=9),
        # rollout_id=5 spans train/step 10 and 11. These belong only to step
        # 11 and therefore must not leak into the step-10 example file.
        _artifact_sample("later-a", rollout_id=2, reward=0.5, response_length=100),
        _artifact_sample("later-b", rollout_id=3, reward=0.5, response_length=100),
    ]

    assert log_rollout_trajectory_examples(5, _example_args(), samples, {}, 1.0) is False

    output = tmp_path / "trajectory_examples_step_000010.jsonl"
    assert output.is_file()
    assert output.stat().st_mode & 0o777 == 0o600
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 2
    assert {record["trajectory"]["session_id"] for record in records} == {"high", "low"}
    assert all(record["train_step"] == 10 for record in records)
    assert (
        records[0]["trajectory"]["traces"][0]["response_messages"][0]["content"]
        == "full model response"
    )
    assert not (tmp_path / "trajectory_examples_step_000011.jsonl").exists()


def test_trajectory_examples_redact_secrets_without_truncating() -> None:
    secret_tail = "TAIL-OF-FULL-TRAJECTORY"
    sample = _artifact_sample(
        response_content=(
            "Authorization: Bearer super-secret-value\n"
            "api_key='sk-1234567890abcdefghijklmnop'\n"
            f"{secret_tail}"
        )
    )

    records = _trajectory_example_records(10, [sample], 2)

    encoded = json.dumps(records, ensure_ascii=False)
    assert "super-secret-value" not in encoded
    assert "sk-1234567890abcdefghijklmnop" not in encoded
    assert "<redacted>" in encoded
    assert secret_tail in encoded
    assert records[0]["redaction_count"] >= 2


def test_trajectory_redaction_covers_mapping_keys_and_prefixed_env_names() -> None:
    payload = {
        "api_key": "bare-secret-value",
        "Authorization": "Bearer authorization-secret",
        "output": "\n".join(
            [
                "OPENAI_API_KEY=openai-secret",
                "WANDB_API_KEY=wandb-secret",
                "HF_TOKEN=hf-secret",
                "AWS_SECRET_ACCESS_KEY=aws-secret",
                "safe_tail=TAIL-OF-FULL-TRAJECTORY",
            ]
        ),
    }

    redacted, count = _redact_trajectory_value(payload)

    encoded = json.dumps(redacted)
    for secret in (
        "bare-secret-value",
        "authorization-secret",
        "openai-secret",
        "wandb-secret",
        "hf-secret",
        "aws-secret",
    ):
        assert secret not in encoded
    assert "TAIL-OF-FULL-TRAJECTORY" in encoded
    assert count == 6


def test_trajectory_redaction_preserves_safe_token_diagnostics() -> None:
    payload = {
        "token_clipping": {"dropped_response_tokens": 5},
        "access_token_count": 3,
        "private_key": "secret-key-material",
        "service_credentials": {"username": "user", "password": "secret"},
    }

    redacted, count = _redact_trajectory_value(payload)

    assert redacted["token_clipping"] == {"dropped_response_tokens": 5}
    assert redacted["access_token_count"] == 3
    assert redacted["private_key"] == "<redacted>"
    assert redacted["service_credentials"] == "<redacted>"
    assert count == 2


def test_wandb_table_contains_two_full_trajectories_on_train_step(monkeypatch) -> None:
    import wandb
    from slime.utils import wandb_utils

    tables = []
    logged = []
    defined = []

    class FakeTable:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            tables.append(self)

    monkeypatch.setattr(wandb, "run", SimpleNamespace())
    monkeypatch.setattr(wandb, "Table", FakeTable)
    monkeypatch.setattr(wandb, "log", logged.append)
    monkeypatch.setattr(
        wandb_utils,
        "define_logged_metric_axes",
        lambda metrics, *, step_metric: defined.append((metrics, step_metric)),
    )
    records = _trajectory_example_records(
        10,
        [
            _artifact_sample("raw-session-a", rollout_id=0, reward=1.0),
            _artifact_sample("raw-session-b", rollout_id=1, reward=0.0),
        ],
        2,
    )

    _log_trajectory_examples_to_wandb(
        SimpleNamespace(use_wandb=True),
        10,
        records,
    )

    assert len(tables) == 1
    assert len(tables[0].kwargs["data"]) == 2
    assert tables[0].kwargs["data"][0][0] == 10
    assert "full model response" in tables[0].kwargs["data"][0][-1]
    assert "raw-session-a" not in tables[0].kwargs["data"][0][1]
    assert logged == [
        {
            "train/step": 10,
            "examples/rollout_trajectories": tables[0],
        }
    ]
    assert defined[0][1] == "train/step"


def test_production_style_session_ids_keep_distinct_wandb_hashes() -> None:
    records = _trajectory_example_records(
        10,
        [
            _artifact_sample("sk-polar-session-one", rollout_id=0, reward=1.0),
            _artifact_sample("sk-polar-session-two", rollout_id=1, reward=0.0),
        ],
        2,
    )

    assert {record["trajectory"]["session_id"] for record in records} == {
        "sk-polar-session-one",
        "sk-polar-session-two",
    }
    assert len({record["session_hash"] for record in records}) == 2


def test_trajectory_example_telemetry_failures_never_fail_training(monkeypatch, caplog) -> None:
    import slime_bridge.rollout as rollout_module

    monkeypatch.setenv("POLAR_ROLLOUT_EXAMPLE_INTERVAL", "10")
    monkeypatch.setenv("POLAR_ROLLOUT_EXAMPLE_COUNT", "2")
    monkeypatch.setattr(
        rollout_module,
        "_persist_trajectory_example_records",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("read only")),
    )
    monkeypatch.setattr(
        rollout_module,
        "_log_trajectory_examples_to_wandb",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("service busy")),
    )
    samples = [
        _artifact_sample("a", rollout_id=0),
        _artifact_sample("b", rollout_id=1),
        _artifact_sample("c", rollout_id=2),
        _artifact_sample("d", rollout_id=3),
    ]

    with caplog.at_level(logging.WARNING, logger="slime_bridge.rollout"):
        result = log_rollout_trajectory_examples(5, _example_args(), samples, {}, 1.0)

    assert result is False
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert all(record.exc_info is None for record in warnings)
    assert "Failed to persist trajectory examples at train step 10" in warnings[0].getMessage()
    assert "Failed to log full-trajectory W&B Table at train step 10" in warnings[1].getMessage()


def test_debug_rollout_only_never_writes_post_train_examples(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("POLAR_ROLLOUT_EXAMPLES_DIR", str(tmp_path))
    monkeypatch.setenv("POLAR_ROLLOUT_EXAMPLE_INTERVAL", "10")
    args = _example_args()
    args.debug_rollout_only = True
    samples = [
        _artifact_sample("a", rollout_id=0),
        _artifact_sample("b", rollout_id=1),
        _artifact_sample("c", rollout_id=2),
        _artifact_sample("d", rollout_id=3),
    ]

    assert log_rollout_trajectory_examples(5, args, samples, {}, 1.0) is False
    assert list(tmp_path.iterdir()) == []
