from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from polar.agent.presets.spilot_forced_route_eval_runner import EVAL_ONLY_ACK
from slime_bridge.config import resolve_polar_slime_config


SCRIPT = (
    Path(__file__).parents[2]
    / "examples"
    / "spilot_router_slime_grpo"
    / "forced_route_eval.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("spilot_forced_route_eval_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _config() -> tuple[SimpleNamespace, object]:
    args = SimpleNamespace(
        polar_rollout_url="http://rollout:8080",
        polar_task_template={
            "timeout_seconds": "{sample.metadata.timeout_seconds}",
            "runtime": {
                "backend": "apptainer",
                "image": "{sample.metadata.sif_path}",
                "workdir": "{sample.metadata.workdir}",
            },
            "agent": {
                "harness": "spilot_router",
                "model_name": "eval-only/no-router",
                "settings": {
                    "model_pool": {
                        "M0": {"model": "pool/qwen3.6-27b", "card": {"name": "Qwen"}},
                        "M1": {"model": "pool/gpt-5.5", "card": {"name": "GPT"}},
                    },
                    "max_pool_calls": 2,
                },
            },
            "builder": {"strategy": "router_policy"},
            "evaluator": {
                "strategy": "spilot_harbor",
                "config": {
                    "tests_dir": "{sample.metadata.tests_dir}",
                    "verifier_timeout": "{sample.metadata.verifier_timeout}",
                },
            },
        },
        polar_task_id_template="unused-{sample.group_index}",
        polar_max_async_level=1,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        update_weights_interval=1,
        polar_eval_agent_timeout=600,
    )
    return args, resolve_polar_slime_config(args)


def _item(index: int = 4):
    return module.EvalItem(
        dataset_index=index,
        prompt=[{"role": "user", "content": "Fix the bug"}],
        metadata={
            "task_name": f"task-{index}",
            "timeout_seconds": 840,
            "agent_timeout": 600,
            "verifier_timeout": 120,
            "sif_path": f"/images/task-{index}.sif",
            "tests_dir": f"/tests/task-{index}",
            "workdir": "/root",
        },
    )


def test_cli_requires_explicit_eval_only_ack_and_positive_spend_guard() -> None:
    common = [
        "--data",
        "eval.jsonl",
        "--polar-config",
        "polar.yaml",
        "--output-dir",
        "out",
        "--run-id",
        "run-1",
        "--max-tasks",
        "1",
    ]
    with pytest.raises(SystemExit):
        module.parse_args(common)
    parsed = module.parse_args([*common, "--i-understand-eval-only"])
    assert parsed.max_tasks == 1


def test_work_matrix_pairs_identical_tasks_and_seeds_for_each_candidate() -> None:
    items = [_item(4), _item(5)]
    work = module.make_work_items(
        items,
        candidates=module.DEFAULT_CANDIDATES,
        replicates=2,
        seed=123,
    )
    assert len(work) == 8
    grouped: dict[tuple[int, int], list[object]] = {}
    for value in work:
        grouped.setdefault((value.item.dataset_index, value.replicate), []).append(value)
    assert set(grouped) == {(4, 0), (4, 1), (5, 0), (5, 1)}
    for pair in grouped.values():
        assert {item.candidate.pool_model for item in pair} == {
            "pool/qwen3.6-27b",
            "pool/gpt-5.5",
        }
        assert len({item.pair_seed for item in pair}) == 1


def test_payload_forces_one_candidate_auto_submit_and_nontrainable_builder() -> None:
    args, config = _config()
    work = module.WorkItem(
        item=_item(),
        candidate=module.DEFAULT_CANDIDATES[1],
        replicate=0,
        pair_seed=77,
    )
    payload = module.build_payload(
        work,
        args=args,
        config=config,
        run_id="paired-1",
        data_sha256="a" * 64,
        forward_seed_to_pool=False,
    )

    assert payload["num_samples"] == 1
    assert payload["early_stop_min_usable_sessions"] == 1
    assert payload["agent"]["harness"] == "spilot_router"
    assert payload["agent"]["model_name"] == "eval-only/forced-route-no-actor"
    assert payload["agent"]["settings"]["max_pool_calls"] == 1
    assert payload["agent"]["settings"]["forced_route_eval"] == {
        "enabled": True,
        "acknowledgement": EVAL_ONLY_ACK,
        "candidate_model": "pool/gpt-5.5",
    }
    assert payload["agent"]["env"]["SPILOT_FORCED_ROUTE_EVAL_ACK"] == EVAL_ONLY_ACK
    assert payload["builder"] == {
        "strategy": "polar.trajectory.builder.spilot_forced_eval:SpilotForcedEvalBuilder",
        "config": {"acknowledgement": EVAL_ONLY_ACK},
    }
    assert payload["evaluator"]["strategy"] == "spilot_harbor"
    assert payload["metadata"]["eval_seed"] == 77


def _task_status(*, traces: list[dict] | None = None, reward: float = 1.0) -> dict:
    return {
        "task_id": "task",
        "status": "completed",
        "results": [
            {
                "session_id": "session",
                "status": "COMPLETED",
                "trajectory": {
                    "status": "COMPLETED",
                    "traces": traces if traces is not None else [],
                    "metadata": {
                        "builder": "spilot_forced_eval",
                        "eval_only": True,
                        "evaluation": {
                            "reward": reward,
                            "harbor_outcome_reward": reward,
                            "spilot_router": {
                                "eval_only": True,
                                "actor_invoked": False,
                                "forced_route_acknowledgement": EVAL_ONLY_ACK,
                                "forced_candidate_model": "pool/qwen3.6-27b",
                                "action_valid": True,
                                "submitted": True,
                                "termination_reason": "m0_auto_submit",
                                "actions": [
                                    {"action": "ROUTE", "model_slot": "M0", "valid": True}
                                ],
                                "calls": [
                                    {
                                        "slot": "M0",
                                        "model": "pool/qwen3.6-27b",
                                        "status": "completed",
                                        "return_code": 0,
                                        "timed_out": False,
                                        "duration_ms": 100,
                                    }
                                ],
                            },
                        },
                    },
                },
                "timing": {"e2e_ms": 150, "run_ms": 100, "eval_ms": 50},
            }
        ],
    }


def test_result_parser_accepts_only_no_actor_trace_free_session() -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    row = module.result_row(work, "task", _task_status())
    assert row["valid"] is True
    assert row["reward"] == 1.0
    assert row["pool_duration_ms"] == 100

    contaminated = module.result_row(
        work,
        "task",
        _task_status(traces=[{"loss_mask": [1]}]),
    )
    assert contaminated["valid"] is False
    assert contaminated["reward"] == 0.0
    assert "forced evaluation emitted trainable traces" in contaminated["integrity_errors"]


@pytest.mark.asyncio
async def test_submitter_authenticates_polls_and_returns_audited_row() -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    payload = {"task_id": "forced-task"}
    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.method == "POST":
            assert request.headers["X-Polar-Control-Token"] == "control-token"
            return httpx.Response(200, json={"task_id": "forced-task", "status": "running"})
        polls += 1
        if polls == 1:
            return httpx.Response(200, json={"task_id": "forced-task", "status": "running"})
        return httpx.Response(200, json=_task_status())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await module.submit_one(
            client,
            asyncio.Semaphore(1),
            rollout_url="http://rollout",
            token="control-token",
            poll_seconds=0.001,
            work=work,
            payload=payload,
        )

    assert polls == 2
    assert row["valid"] is True
    assert row["reward"] == 1.0


def test_summary_uses_fixed_denominator_and_reports_paired_delta() -> None:
    rows = [
        {
            "dataset_index": 1,
            "replicate": 0,
            "candidate_model": "pool/qwen3.6-27b",
            "valid": True,
            "reward": 0.0,
            "e2e_ms": 100,
            "pool_status": "completed",
            "session_status": "COMPLETED",
        },
        {
            "dataset_index": 1,
            "replicate": 0,
            "candidate_model": "pool/gpt-5.5",
            "valid": True,
            "reward": 1.0,
            "e2e_ms": 200,
            "pool_status": "completed",
            "session_status": "COMPLETED",
        },
    ]
    summary = module.summarize(rows, module.DEFAULT_CANDIDATES)
    assert summary["candidate_metrics"]["pool/qwen3.6-27b"]["reward_mean"] == 0.0
    assert summary["candidate_metrics"]["pool/gpt-5.5"]["reward_mean"] == 1.0
    assert summary["paired"] == {
        "pair_count": 1,
        "delta_definition": "pool/gpt-5.5 - pool/qwen3.6-27b",
        "mean_reward_delta": 1.0,
        "second_wins": 1,
        "first_wins": 0,
        "ties": 0,
    }


def test_formal_training_template_does_not_enable_forced_eval() -> None:
    template = (
        Path(__file__).parents[2]
        / "examples"
        / "spilot_router_slime_grpo"
        / "polar_config.yaml"
    ).read_text(encoding="utf-8")
    assert "forced_route_eval" not in template
    assert "spilot_forced_eval" not in template
    assert 'strategy: "router_policy"' in template
