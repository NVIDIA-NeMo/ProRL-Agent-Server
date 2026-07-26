from __future__ import annotations

import asyncio
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from polar.agent.presets.spilot_forced_route_eval_runner import EVAL_ONLY_ACK
from slime_bridge.config import resolve_polar_slime_config


SCRIPT = (
    Path(__file__).parents[2] / "examples" / "spilot_router_slime_grpo" / "forced_route_eval.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("spilot_forced_route_eval_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()
PLAN_SHA256 = "b" * 64
ALLOCATION_ATTEMPT_ID = "a" * 64
IDENTITY = {
    "forced_eval_plan_sha256": PLAN_SHA256,
    "forced_eval_work_sha256": "c" * 64,
    "dataset_row_sha256": "d" * 64,
}


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
        "--semantic-identity",
        "identity.json",
        "--output-dir",
        "out",
        "--run-id",
        "run-1",
        "--allocation-attempt-id",
        ALLOCATION_ATTEMPT_ID,
        "--max-tasks",
        "1",
    ]
    with pytest.raises(SystemExit):
        module.parse_args(common)
    parsed = module.parse_args([*common, "--i-understand-eval-only"])
    assert parsed.max_tasks == 1
    assert parsed.pool_timeout_seconds == 1200
    assert parsed.runner_total_timeout_seconds == 3000
    assert parsed.max_paid_attempts_per_work == 1

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                *common,
                "--i-understand-eval-only",
                "--pool-timeout-seconds",
                "2800",
            ]
        )


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


def test_optional_qwen35_baseline_uses_identical_task_seed_matrix() -> None:
    candidates = module.selected_candidates(include_qwen35_baseline=True)
    work = module.make_work_items(
        [_item(4), _item(5)],
        candidates=candidates,
        replicates=2,
        seed=123,
    )
    assert len(work) == 12
    grouped: dict[tuple[int, int], list[object]] = {}
    for value in work:
        grouped.setdefault((value.item.dataset_index, value.replicate), []).append(value)
    for pair in grouped.values():
        assert {item.candidate.pool_model for item in pair} == {
            "pool/qwen3.6-27b",
            "pool/gpt-5.5",
            "pool/qwen3.5-9b-baseline",
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
        plan_sha256=PLAN_SHA256,
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
    assert payload["metadata"]["forced_eval_plan_sha256"] == PLAN_SHA256
    assert len(payload["metadata"]["forced_eval_work_sha256"]) == 64
    assert len(payload["metadata"]["dataset_row_sha256"]) == 64
    assert payload["metadata"]["forced_eval_work_sha256"][:24] in payload["task_id"]


def _task_status(
    *,
    traces: list[dict] | None = None,
    reward: float = 1.0,
    task_id: str = "task",
    identity: dict[str, str] | None = None,
) -> dict:
    identity = IDENTITY if identity is None else identity
    return {
        "task_id": task_id,
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
                        "task_metadata": dict(identity),
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
                                        "attempted": True,
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
    row = module.result_row(
        work,
        "task",
        _task_status(),
        expected_identity=IDENTITY,
    )
    assert row["valid"] is True
    assert row["reward"] == 1.0
    assert row["pool_duration_ms"] == 100

    with pytest.raises(module.TaskIdentityError, match="trainable traces"):
        module.result_row(
            work,
            "task",
            _task_status(traces=[{"loss_mask": [1]}]),
            expected_identity=IDENTITY,
        )


@pytest.mark.parametrize("session_status", ["ERROR", "TIMEOUT"])
def test_session_infrastructure_terminal_state_stays_pending(session_status: str) -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    status = _task_status()
    status["results"][0]["status"] = session_status

    with pytest.raises(module.PendingWorkError, match="not a benchmark outcome"):
        module.result_row(work, "task", status, expected_identity=IDENTITY)


@pytest.mark.parametrize("missing", ["acknowledgement", "call"])
def test_missing_forced_outcome_evidence_stays_pending(missing: str) -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    status = _task_status()
    router = status["results"][0]["trajectory"]["metadata"]["evaluation"]["spilot_router"]
    if missing == "acknowledgement":
        router.pop("forced_route_acknowledgement")
    else:
        router["calls"] = []

    with pytest.raises(module.PendingWorkError, match="missing"):
        module.result_row(work, "task", status, expected_identity=IDENTITY)


def test_failed_candidate_call_stays_pending() -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    status = _task_status(reward=0.0)
    call = status["results"][0]["trajectory"]["metadata"]["evaluation"]["spilot_router"][
        "calls"
    ][0]
    call.update(status="failed", return_code=1, timed_out=False, failure_kind="exit_code")

    with pytest.raises(module.PendingWorkError, match="not an attributable"):
        module.result_row(work, "task", status, expected_identity=IDENTITY)


def test_explicit_matching_candidate_timeout_is_a_benchmark_outcome() -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    status = _task_status(reward=0.0)
    call = status["results"][0]["trajectory"]["metadata"]["evaluation"]["spilot_router"][
        "calls"
    ][0]
    call.update(status="timeout", return_code=-1, timed_out=True, failure_kind="timeout")

    row = module.result_row(work, "task", status, expected_identity=IDENTITY)
    assert row["valid"] is True
    assert row["reward"] == 0.0
    assert row["pool_status"] == "timeout"


@pytest.mark.asyncio
async def test_submitter_authenticates_polls_and_returns_audited_row() -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    payload = {"task_id": "forced-task", "metadata": dict(IDENTITY)}
    polls = 0
    submitted = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls, submitted
        if request.method == "POST":
            assert request.headers["X-Polar-Control-Token"] == "control-token"
            submitted = True
            return httpx.Response(200, json={"task_id": "forced-task", "status": "running"})
        if not submitted:
            return httpx.Response(404, json={"detail": "Task not found"})
        polls += 1
        if polls == 1:
            return httpx.Response(200, json={"task_id": "forced-task", "status": "running"})
        return httpx.Response(200, json=_task_status(task_id="forced-task"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await module.submit_one(
            client,
            asyncio.Semaphore(1),
            rollout_url="http://rollout",
            token="control-token",
            poll_seconds=0.001,
            retry_attempts=3,
            retry_backoff_seconds=0.001,
            work=work,
            payload=payload,
        )

    assert polls == 2
    assert submitted is True
    assert row["valid"] is True
    assert row["reward"] == 1.0


@pytest.mark.asyncio
async def test_submitter_reattaches_to_completed_task_without_duplicate_post() -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    payload = {"task_id": "forced-task", "metadata": dict(IDENTITY)}
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=_task_status(task_id="forced-task"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await module.submit_one(
            client,
            asyncio.Semaphore(1),
            rollout_url="http://rollout",
            token="control-token",
            poll_seconds=0.001,
            retry_attempts=3,
            retry_backoff_seconds=0.001,
            work=work,
            payload=payload,
        )

    assert methods == ["GET"]
    assert row["valid"] is True


@pytest.mark.asyncio
async def test_submitter_recovers_post_ack_loss_without_duplicate_benchmark_row() -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    payload = {"task_id": "forced-task", "metadata": dict(IDENTITY)}
    accepted = False
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal accepted, posts
        if request.method == "POST":
            posts += 1
            accepted = True
            raise httpx.ReadError("response lost after server accept", request=request)
        if not accepted:
            return httpx.Response(404, json={"detail": "Task not found"})
        return httpx.Response(200, json=_task_status(task_id="forced-task"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await module.submit_one(
            client,
            asyncio.Semaphore(1),
            rollout_url="http://rollout",
            token="control-token",
            poll_seconds=0.001,
            retry_attempts=3,
            retry_backoff_seconds=0.001,
            work=work,
            payload=payload,
        )

    assert posts == 1
    assert row["valid"] is True


@pytest.mark.asyncio
async def test_local_transport_exhaustion_stays_pending_not_result() -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    payload = {"task_id": "forced-task", "metadata": dict(IDENTITY)}
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("control plane unavailable", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(module.PendingWorkError, match="remains pending"):
            await module.submit_one(
                client,
                asyncio.Semaphore(1),
                rollout_url="http://rollout",
                token="control-token",
                poll_seconds=0.001,
                retry_attempts=3,
                retry_backoff_seconds=0.001,
                work=work,
                payload=payload,
            )

    assert attempts == 3


@pytest.mark.asyncio
async def test_pipeline_failed_task_is_resubmitted_not_scored_zero() -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    payload = {"task_id": "forced-task", "metadata": dict(IDENTITY)}
    restarted = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal restarted
        if request.method == "GET" and not restarted:
            return httpx.Response(200, json={"task_id": "forced-task", "status": "failed"})
        if request.method == "POST":
            restarted = True
            return httpx.Response(200, json={"task_id": "forced-task", "status": "running"})
        return httpx.Response(200, json=_task_status(task_id="forced-task"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await module.submit_one(
            client,
            asyncio.Semaphore(1),
            rollout_url="http://rollout",
            token="control-token",
            poll_seconds=0.001,
            retry_attempts=3,
            retry_backoff_seconds=0.001,
            work=work,
            payload=payload,
        )

    assert restarted is True
    assert row["task_status"] == "completed"
    assert row["valid"] is True


@pytest.mark.asyncio
async def test_repeated_pipeline_failed_task_remains_pending(tmp_path: Path) -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    payload = {"task_id": "forced-task", "metadata": dict(IDENTITY)}
    posts = 0
    ledger = module.AttemptLedger(
        tmp_path / "attempts.json",
        plan_sha256=PLAN_SHA256,
        work_ids=[IDENTITY["forced_eval_work_sha256"]],
        max_paid_attempts=2,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "POST":
            posts += 1
        return httpx.Response(200, json={"task_id": "forced-task", "status": "failed"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(module.PendingWorkError, match="global paid-attempt budget"):
            await module.submit_one(
                client,
                asyncio.Semaphore(1),
                rollout_url="http://rollout",
                token="control-token",
                poll_seconds=0.001,
                retry_attempts=5,
                retry_backoff_seconds=0.001,
                work=work,
                payload=payload,
                attempt_ledger=ledger,
            )
    assert posts == 2


@pytest.mark.asyncio
async def test_evaluator_cancellation_is_not_result_and_can_resume() -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    payload = {"task_id": "forced-task", "metadata": dict(IDENTITY)}
    submitted = asyncio.Event()
    methods: list[str] = []

    def first_handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "GET" and not submitted.is_set():
            return httpx.Response(404, json={"detail": "Task not found"})
        if request.method == "POST":
            submitted.set()
            return httpx.Response(200, json={"task_id": "forced-task", "status": "running"})
        if request.method == "DELETE":
            return httpx.Response(200, json={"task_id": "forced-task", "status": "cancelled"})
        return httpx.Response(200, json={"task_id": "forced-task", "status": "running"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(first_handler)) as client:
        task = asyncio.create_task(
            module.submit_one(
                client,
                asyncio.Semaphore(1),
                rollout_url="http://rollout",
                token="control-token",
                poll_seconds=1.0,
                retry_attempts=3,
                retry_backoff_seconds=0.001,
                work=work,
                payload=payload,
            )
        )
        await submitted.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert "DELETE" in methods

    restarted = False

    def resume_handler(request: httpx.Request) -> httpx.Response:
        nonlocal restarted
        if request.method == "GET" and not restarted:
            return httpx.Response(200, json={"task_id": "forced-task", "status": "cancelled"})
        if request.method == "POST":
            restarted = True
            return httpx.Response(200, json={"task_id": "forced-task", "status": "running"})
        return httpx.Response(200, json=_task_status(task_id="forced-task"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(resume_handler)) as client:
        row = await module.submit_one(
            client,
            asyncio.Semaphore(1),
            rollout_url="http://rollout",
            token="control-token",
            poll_seconds=0.001,
            retry_attempts=3,
            retry_backoff_seconds=0.001,
            work=work,
            payload=payload,
        )
    assert restarted is True
    assert row["valid"] is True


@pytest.mark.asyncio
async def test_reattach_rejects_terminal_identity_collision() -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    payload = {"task_id": "forced-task", "metadata": dict(IDENTITY)}
    wrong_identity = {**IDENTITY, "forced_eval_plan_sha256": "e" * 64}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_task_status(task_id="forced-task", identity=wrong_identity),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(module.TaskIdentityError, match="immutable plan"):
            await module.submit_one(
                client,
                asyncio.Semaphore(1),
                rollout_url="http://rollout",
                token="control-token",
                poll_seconds=0.001,
                retry_attempts=3,
                retry_backoff_seconds=0.001,
                work=work,
                payload=payload,
            )


def _stored_row(
    work,
    task_id: str,
    *,
    plan_sha256: str,
    reward: float = 1.0,
    allocation_attempt_id: str = ALLOCATION_ATTEMPT_ID,
) -> dict:
    return {
        **module._expected_result_fields(
            work,
            task_id,
            plan_sha256=plan_sha256,
        ),
        "allocation_attempt_id": allocation_attempt_id,
        "task_status": "completed",
        "session_id": "session",
        "session_status": "COMPLETED",
        "valid": True,
        "reward": reward,
        "reported_reward": reward,
        "harbor_outcome_reward": reward,
        "eval_only": True,
        "actor_invoked": False,
        "forced_route_acknowledgement": EVAL_ONLY_ACK,
        "forced_candidate_model": work.candidate.pool_model,
        "router_action_valid": True,
        "router_submitted": True,
        "termination_reason": "m0_auto_submit",
        "pool_status": "completed",
        "pool_attempted": True,
        "pool_return_code": 0,
        "pool_timed_out": False,
        "pool_failure_kind": None,
        "pool_duration_ms": 100,
        "e2e_ms": 150,
        "run_ms": 100,
        "eval_ms": 50,
        "integrity_errors": [],
    }


def test_result_store_is_incremental_resume_safe_and_duplicate_idempotent(
    tmp_path: Path,
) -> None:
    qwen = module.WorkItem(_item(4), module.DEFAULT_CANDIDATES[0], 0, 77)
    gpt = module.WorkItem(_item(4), module.DEFAULT_CANDIDATES[1], 0, 77)
    plan = {
        "schema_version": 2,
        "run_id": "resume-test",
        "expected_result_count": 2,
    }
    plan_fingerprint = module._plan_sha256(plan)
    expected = {
        "task-qwen": module._expected_result_fields(
            qwen,
            "task-qwen",
            plan_sha256=plan_fingerprint,
        ),
        "task-gpt": module._expected_result_fields(
            gpt,
            "task-gpt",
            plan_sha256=plan_fingerprint,
        ),
    }
    output = tmp_path / "results"

    first = module.ResultStore.open(
        output,
        plan=plan,
        expected=expected,
        allocation_attempt={
            "allocation_attempt_id": ALLOCATION_ATTEMPT_ID,
            "job": "one",
            "rollout_url": "http://127.0.0.1:1",
            "teardown_verification": {"status": "pending", "reason": None},
        },
    )
    first.add(_stored_row(qwen, "task-qwen", plan_sha256=plan_fingerprint))
    partial_manifest = json.loads((output / "manifest.json").read_text())
    partial_summary = json.loads((output / "summary.json").read_text())
    assert partial_manifest["collection"] == partial_summary["collection"]
    assert partial_manifest["collection"]["status"] == "partial"
    assert partial_manifest["collection"]["collected_result_count"] == 1
    assert partial_manifest["collection"]["missing_result_count"] == 1
    assert "candidate_metrics" not in partial_summary
    assert "paired" not in partial_summary
    collected = partial_summary["collected_only"]
    assert collected["candidate_metrics"]["pool/qwen3.6-27b"]["missing_count"] == 0
    assert collected["candidate_metrics"]["pool/gpt-5.5"]["missing_count"] == 1
    assert collected["paired"]["expected_pair_count"] == 1
    assert collected["paired"]["missing_pair_count"] == 1
    assert partial_summary["final_metrics"] is None
    assert partial_summary["final_metrics_status"] == "pending_teardown"

    # Simulate interruption: a new process opens the same immutable plan and
    # sees only the missing task as pending.
    with pytest.raises(ValueError, match="pass --resume"):
        module.ResultStore.open(output, plan=plan, expected=expected)
    resumed = module.ResultStore.open(
        output,
        plan=plan,
        expected=expected,
        resume=True,
        allocation_attempt={
            "allocation_attempt_id": "e" * 64,
            "job": "two",
            "rollout_url": "http://127.0.0.1:2",
            "teardown_verification": {"status": "pending", "reason": None},
        },
    )
    assert resumed.completed_task_ids == {"task-qwen"}
    assert len(resumed.allocation_attempts) == 2
    resumed_manifest = json.loads((output / "manifest.json").read_text())
    assert resumed_manifest["plan_sha256"] == plan_fingerprint
    assert len(resumed_manifest["allocation_attempts"]) == 2
    resumed.add(
        _stored_row(
            gpt,
            "task-gpt",
            plan_sha256=plan_fingerprint,
            reward=0.0,
            allocation_attempt_id="e" * 64,
        )
    )
    complete = json.loads((output / "summary.json").read_text())
    assert complete["collection"]["status"] == "complete"
    assert complete["collection"]["result_set_complete"] is True
    assert complete["final_metrics"] is None
    assert complete["final_metrics_status"] == "pending_teardown"
    assert complete["publication"] == {
        "status": "pending_teardown",
        "allocation_attempt_id": "e" * 64,
    }

    # Identical legacy duplicate lines are canonicalized, not resubmitted or
    # double-counted. Conflicting duplicates remain a hard integrity error.
    lines = (output / "results.jsonl").read_text().splitlines()
    (output / "results.jsonl").write_text("\n".join([*lines, lines[0]]) + "\n")
    deduplicated = module.ResultStore.open(
        output,
        plan=plan,
        expected=expected,
        resume=True,
    )
    assert len(deduplicated.rows) == 2
    assert len((output / "results.jsonl").read_text().splitlines()) == 2

    conflicting = json.loads(lines[0])
    conflicting["reward"] = 0.5
    conflicting["reported_reward"] = 0.5
    conflicting["harbor_outcome_reward"] = 0.5
    with (output / "results.jsonl").open("a") as stream:
        stream.write(json.dumps(conflicting, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        module.ResultStore.open(output, plan=plan, expected=expected, resume=True)


def test_output_directory_lock_rejects_concurrent_resumer(tmp_path: Path) -> None:
    output = tmp_path / "locked"
    first = module.OutputDirectoryLock(output)
    try:
        with pytest.raises(ValueError, match="holds the output lock"):
            module.OutputDirectoryLock(output)
    finally:
        first.close()
    second = module.OutputDirectoryLock(output)
    second.close()


def test_paid_attempt_ledger_is_durable_global_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "attempt_ledger.json"
    first = module.AttemptLedger(
        path,
        plan_sha256="a" * 64,
        work_ids=["work-a"],
        max_paid_attempts=2,
    )
    first_attempt = first.reserve_paid_attempt(
        "work-a", task_id="task", reason="missing"
    )
    assert first_attempt == 1
    assert first.paid_attempts("work-a") == 0

    resumed = module.AttemptLedger(
        path,
        plan_sha256="a" * 64,
        work_ids=["work-a"],
        max_paid_attempts=2,
    )
    assert resumed.paid_attempts("work-a") == 0
    resumed.mark_submit_started("work-a", attempt_id=first_attempt)
    assert resumed.paid_attempts("work-a") == 1
    resumed.mark_terminal("work-a", outcome="failed")
    second_attempt = resumed.reserve_paid_attempt(
        "work-a", task_id="task", reason="failed"
    )
    assert second_attempt == 2
    resumed.mark_submit_started("work-a", attempt_id=second_attempt)
    assert resumed.paid_attempts("work-a") == 2
    resumed.mark_terminal("work-a", outcome="failed")
    with pytest.raises(module.PendingWorkError, match="global paid-attempt budget"):
        resumed.reserve_paid_attempt("work-a", task_id="task", reason="failed")

    with pytest.raises(ValueError, match="different semantic plan"):
        module.AttemptLedger(
            path,
            plan_sha256="b" * 64,
            work_ids=["work-a"],
            max_paid_attempts=2,
        )


def test_integrity_failure_withholds_complete_metrics_and_blocks_resume(
    tmp_path: Path,
) -> None:
    work = module.WorkItem(_item(4), module.DEFAULT_CANDIDATES[0], 0, 77)
    plan = {"schema_version": 5, "run_id": "integrity-test"}
    plan_sha = module._plan_sha256(plan)
    task_id = "integrity-task"
    expected = {
        task_id: module._expected_result_fields(work, task_id, plan_sha256=plan_sha)
    }
    output = tmp_path / "integrity-output"
    store = module.ResultStore.open(output, plan=plan, expected=expected)
    store.add(_stored_row(work, task_id, plan_sha256=plan_sha))
    before = json.loads((output / "summary.json").read_text())
    assert before["final_metrics"] is None
    assert before["final_metrics_status"] == "pending_teardown"

    store.mark_integrity_failure("snapshot changed")
    after = json.loads((output / "summary.json").read_text())
    assert after["content_integrity"]["status"] == "failed"
    assert after["final_metrics"] is None
    assert after["final_metrics_status"] == "withheld_integrity"
    with pytest.raises(ValueError, match="content-integrity"):
        module.ResultStore.open(output, plan=plan, expected=expected, resume=True)


def test_semantic_topology_cap_changes_plan_and_deterministic_task_id() -> None:
    work = module.WorkItem(_item(4), module.DEFAULT_CANDIDATES[0], 0, 77)
    plan_a = {
        "schema_version": 5,
        "semantic_identity": {"topology": {"max_active_episodes": 4}},
    }
    plan_b = {
        "schema_version": 5,
        "semantic_identity": {"topology": {"max_active_episodes": 3}},
    }
    hash_a = module._plan_sha256(plan_a)
    hash_b = module._plan_sha256(plan_b)
    assert hash_a != hash_b
    assert module.forced_task_id(work, run_id="cap", plan_sha256=hash_a) != (
        module.forced_task_id(work, run_id="cap", plan_sha256=hash_b)
    )


def test_active_allocation_owner_blocks_unrelated_resumer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "owned"
    output.mkdir()
    owner_path = output / ".forced-eval.owner.lock"
    owner = owner_path.open("a+")
    fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        monkeypatch.delenv("SPILOT_FORCED_EVAL_OWNER_FD", raising=False)
        with pytest.raises(ValueError, match="active allocation"):
            module.OutputDirectoryLock(output)
        monkeypatch.setenv("SPILOT_FORCED_EVAL_OWNER_FD", str(owner.fileno()))
        worker = module.OutputDirectoryLock(output)
        worker.close()
    finally:
        fcntl.flock(owner.fileno(), fcntl.LOCK_UN)
        owner.close()


def test_task_id_fingerprint_separates_plan_and_prompt_collisions() -> None:
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    changed_prompt = module.WorkItem(
        module.EvalItem(
            dataset_index=work.item.dataset_index,
            prompt=[{"role": "user", "content": "different prompt"}],
            metadata=work.item.metadata,
        ),
        work.candidate,
        work.replicate,
        work.pair_seed,
    )
    original = module.forced_task_id(
        work,
        run_id="same-run",
        plan_sha256="a" * 64,
    )
    different_plan = module.forced_task_id(
        work,
        run_id="same-run",
        plan_sha256="b" * 64,
    )
    different_prompt = module.forced_task_id(
        changed_prompt,
        run_id="same-run",
        plan_sha256="a" * 64,
    )
    assert len({original, different_plan, different_prompt}) == 3


def test_declared_implementation_manifest_changes_with_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "impl.py").write_text("VALUE = 1\n")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "worker.py").write_text("WORKER = 1\n")
    monkeypatch.setattr(module, "_IMPLEMENTATION_FILES", ("impl.py",))
    monkeypatch.setattr(module, "_IMPLEMENTATION_TREES", ("runtime",))

    first = module.build_implementation_manifest(tmp_path)
    work = module.WorkItem(_item(), module.DEFAULT_CANDIDATES[0], 0, 77)
    plan_v1 = {"schema_version": 3, "implementation": first}
    plan_sha_v1 = module._plan_sha256(plan_v1)
    task_id_v1 = module.forced_task_id(
        work,
        run_id="source-plan",
        plan_sha256=plan_sha_v1,
    )
    output = tmp_path / "source-output"
    module.ResultStore.open(
        output,
        plan=plan_v1,
        expected={
            task_id_v1: module._expected_result_fields(
                work,
                task_id_v1,
                plan_sha256=plan_sha_v1,
            )
        },
    )
    (runtime / "worker.py").write_text("WORKER = 2\n")
    second = module.build_implementation_manifest(tmp_path)
    assert first["sha256"] != second["sha256"]
    plan_v2 = {"schema_version": 3, "implementation": second}
    plan_sha_v2 = module._plan_sha256(plan_v2)
    task_id_v2 = module.forced_task_id(
        work,
        run_id="source-plan",
        plan_sha256=plan_sha_v2,
    )
    assert task_id_v1 != task_id_v2
    with pytest.raises(ValueError, match="different forced-eval plan"):
        module.ResultStore.open(
            output,
            plan=plan_v2,
            expected={
                task_id_v2: module._expected_result_fields(
                    work,
                    task_id_v2,
                    plan_sha256=plan_sha_v2,
                )
            },
            resume=True,
        )

    (tmp_path / "impl.py").unlink()
    with pytest.raises(ValueError, match="implementation file is missing"):
        module.build_implementation_manifest(tmp_path)


def test_semantic_config_hash_ignores_allocation_transport_only(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(
        "polar_rollout_url: http://127.0.0.1:18080\n"
        "socket: /tmp/polar-forced-eval-1-10/gateway/gateway.sock\n"
        "semantic:\n  pool_timeout_seconds: 1200\n"
    )
    second.write_text(
        "polar_rollout_url: http://127.0.0.1:28080\n"
        "socket: /tmp/polar-forced-eval-2-20/gateway/gateway.sock\n"
        "semantic:\n  pool_timeout_seconds: 1200\n"
    )
    assert (
        module.build_semantic_config_manifest(first)["sha256"]
        == module.build_semantic_config_manifest(second)["sha256"]
    )
    second.write_text(second.read_text().replace("1200", "1800"))
    assert (
        module.build_semantic_config_manifest(first)["sha256"]
        != module.build_semantic_config_manifest(second)["sha256"]
    )


def test_sif_and_verifier_tree_content_bind_plan_task_id_and_resume(
    tmp_path: Path,
) -> None:
    sif = tmp_path / "task.sif"
    sif.write_bytes(b"sif-v1")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    verifier = tests_dir / "test.sh"
    verifier.write_text("echo 1\n")
    item = module.EvalItem(
        dataset_index=4,
        prompt=[{"role": "user", "content": "fix it"}],
        metadata={
            "task_name": "task-4",
            "sif_path": str(sif),
            "tests_dir": str(tests_dir),
        },
    )
    work = module.WorkItem(item, module.DEFAULT_CANDIDATES[0], 0, 77)

    assets_v1 = module.build_task_asset_manifest([item])
    plan_v1 = {"schema_version": 3, "assets": assets_v1}
    plan_sha_v1 = module._plan_sha256(plan_v1)
    task_id_v1 = module.forced_task_id(
        work,
        run_id="asset-plan",
        plan_sha256=plan_sha_v1,
    )
    expected_v1 = {
        task_id_v1: module._expected_result_fields(
            work,
            task_id_v1,
            plan_sha256=plan_sha_v1,
        )
    }
    output = tmp_path / "output"
    module.ResultStore.open(output, plan=plan_v1, expected=expected_v1)

    sif.write_bytes(b"sif-v2")
    assets_sif_changed = module.build_task_asset_manifest([item])
    assert assets_v1["sha256"] != assets_sif_changed["sha256"]
    verifier.write_text("echo 0\n")
    assets_v2 = module.build_task_asset_manifest([item])
    assert assets_sif_changed["sha256"] != assets_v2["sha256"]
    plan_v2 = {"schema_version": 3, "assets": assets_v2}
    plan_sha_v2 = module._plan_sha256(plan_v2)
    task_id_v2 = module.forced_task_id(
        work,
        run_id="asset-plan",
        plan_sha256=plan_sha_v2,
    )
    assert task_id_v1 != task_id_v2
    with pytest.raises(ValueError, match="different forced-eval plan"):
        module.ResultStore.open(
            output,
            plan=plan_v2,
            expected={
                task_id_v2: module._expected_result_fields(
                    work,
                    task_id_v2,
                    plan_sha256=plan_sha_v2,
                )
            },
            resume=True,
        )


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
        "expected_pair_count": 1,
        "pair_count": 1,
        "missing_pair_count": 0,
        "delta_definition": "pool/gpt-5.5 - pool/qwen3.6-27b",
        "mean_reward_delta": 1.0,
        "second_wins": 1,
        "first_wins": 0,
        "ties": 0,
    }


def test_three_candidate_summary_preserves_pool_delta_and_adds_baseline_deltas() -> None:
    rewards = {
        "pool/qwen3.5-9b-baseline": 0.25,
        "pool/qwen3.6-27b": 0.5,
        "pool/gpt-5.5": 1.0,
    }
    rows = [
        {
            "dataset_index": 1,
            "replicate": 0,
            "candidate_model": model,
            "valid": True,
            "reward": reward,
            "e2e_ms": 100,
            "pool_status": "completed",
            "session_status": "COMPLETED",
        }
        for model, reward in rewards.items()
    ]
    summary = module.summarize(
        rows,
        module.selected_candidates(include_qwen35_baseline=True),
    )
    assert summary["paired"]["delta_definition"] == (
        "pool/gpt-5.5 - pool/qwen3.6-27b"
    )
    assert summary["paired"]["mean_reward_delta"] == 0.5
    comparisons = summary["paired_comparisons"]
    assert comparisons["gpt_vs_qwen35_baseline"]["delta_definition"] == (
        "pool/gpt-5.5 - pool/qwen3.5-9b-baseline"
    )
    assert comparisons["gpt_vs_qwen35_baseline"]["mean_reward_delta"] == 0.75
    assert comparisons["qwen36_vs_qwen35_baseline"]["delta_definition"] == (
        "pool/qwen3.6-27b - pool/qwen3.5-9b-baseline"
    )
    assert comparisons["qwen36_vs_qwen35_baseline"]["mean_reward_delta"] == 0.25
    assert summary["candidate_metrics"]["pool/qwen3.5-9b-baseline"]["endpoint_model"] == (
        "nvidia/qwen/qwen3.5-9b"
    )


def test_formal_training_template_does_not_enable_forced_eval() -> None:
    template = (
        Path(__file__).parents[2] / "examples" / "spilot_router_slime_grpo" / "polar_config.yaml"
    ).read_text(encoding="utf-8")
    assert "forced_route_eval" not in template
    assert "spilot_forced_eval" not in template
    assert 'strategy: "router_policy"' in template
