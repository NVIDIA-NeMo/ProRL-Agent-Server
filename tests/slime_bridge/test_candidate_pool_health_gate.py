from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime.utils.types import Sample
from slime_bridge import rollout as rollout_module
from slime_bridge.rollout import (
    AsyncPolarRolloutWorker,
    CandidatePoolHealthGateError,
    PolarLowCompleteAcceptFractionError,
    _CandidatePoolHealthAccumulator,
    _CandidatePoolHealthGateConfig,
    _CompletedGroup,
    _PendingGroup,
    _PartialRecoveryPlan,
    _enforce_candidate_pool_health_gate,
    _persist_candidate_pool_health_incident,
    generate_rollout_polar_async,
)


ALIASES = ("pool/a", "pool/b")


def _router_sample(
    *,
    reservation_id: int,
    index: int,
    candidate: str | None,
    statuses: tuple[str, ...] = (),
    reward: float = 0.0,
    swap_slots: bool = False,
    attempted: bool = True,
) -> Sample:
    mapping = (
        {"M0": {"model": ALIASES[1]}, "M1": {"model": ALIASES[0]}}
        if swap_slots
        else {"M0": {"model": ALIASES[0]}, "M1": {"model": ALIASES[1]}}
    )
    slot = next(
        (slot_name for slot_name, value in mapping.items() if value["model"] == candidate),
        "M0",
    )
    actions = (
        [{"action": "ROUTE", "model_slot": slot, "valid": True}]
        if candidate is not None
        else []
    )
    calls = [
        {
            "slot": slot,
            "model": candidate,
            "role": "solve" if call_index == 0 else "verify",
            "status": status,
            "attempted": attempted,
        }
        for call_index, status in enumerate(statuses)
    ]
    router = {
        "slot_mapping": mapping,
        "actions": actions,
        "calls": calls,
        "termination_reason": "router_submit",
    }
    return Sample(
        group_index=reservation_id,
        index=reservation_id * 1000 + index,
        rollout_id=reservation_id * 1000 + index,
        reward={"score": reward},
        response_length=1,
        loss_mask=[1],
        metadata={
            "polar": {
                "session_id": f"session-{reservation_id}-{index}",
                "session_status": "COMPLETED",
                "placeholder": False,
                "trajectory_metadata": {
                    "evaluation": {
                        "spilot_router": router,
                        "harbor_outcome_reward": reward,
                        "applied_cost_penalty": 0.0,
                    }
                },
            }
        },
        status=Sample.Status.COMPLETED,
    )


def _health_completion(
    reservation_id: int,
    specs: list[tuple[str | None, tuple[str, ...], float, bool]],
) -> _CompletedGroup:
    samples = [
        _router_sample(
            reservation_id=reservation_id,
            index=index,
            candidate=candidate,
            statuses=statuses,
            reward=reward,
            swap_slots=swap_slots,
        )
        for index, (candidate, statuses, reward, swap_slots) in enumerate(specs)
    ]
    return _CompletedGroup(
        group_id=reservation_id,
        group=[SimpleNamespace(group_index=reservation_id)],
        reservation_id=reservation_id,
        samples=samples,
        task_id=f"task-{reservation_id}",
        submitted_rollout_id=3,
        policy_version=3,
        session_count=len(samples),
    )


def _report(
    statuses: list[str],
    *,
    min_observed_sessions: int = 16,
    min_completion_fraction: float = 0.1,
) -> dict:
    completed = _health_completion(
        1,
        [(ALIASES[0], (status,), 0.0, False) for status in statuses],
    )
    accumulator = _CandidatePoolHealthAccumulator(ALIASES)
    accumulator.add(completed)
    return accumulator.report(
        _CandidatePoolHealthGateConfig(
            candidate_aliases=ALIASES,
            min_observed_sessions=min_observed_sessions,
            min_completion_fraction=min_completion_fraction,
        ),
        rollout_id=4,
        accepted_group_count=1,
    )


@pytest.mark.parametrize(
    ("statuses", "triggered"),
    [
        (["failed"] * 16, True),
        (["failed"] * 15, False),
        (["failed"] * 15 + ["completed"], True),
        (["failed"] * 14 + ["completed"] * 2, False),
        (["timeout"] * 16, True),
    ],
)
def test_candidate_pool_health_gate_uses_independent_session_denominator(
    statuses: list[str],
    triggered: bool,
) -> None:
    report = _report(statuses)

    assert report["triggered"] is triggered
    assert report["candidates"]["C1"]["observed_session_count"] == 0
    assert report["candidates"]["C1"]["insufficient_evidence"] is True


def test_candidate_pool_health_gate_does_not_double_count_solve_and_verify() -> None:
    completed = _health_completion(
        2,
        [(ALIASES[0], ("failed", "timeout"), 0.0, False)],
    )
    accumulator = _CandidatePoolHealthAccumulator(ALIASES)
    accumulator.add(completed)
    report = accumulator.report(
        _CandidatePoolHealthGateConfig(ALIASES, 2, 0.1),
        rollout_id=4,
        accepted_group_count=1,
    )

    c0 = report["candidates"]["C0"]
    assert c0["attempted_call_count"] == 2
    assert c0["observed_session_count"] == 1
    assert c0["unavailable_session_count"] == 1
    assert c0["eligible"] is False
    assert report["triggered"] is False


def test_candidate_pool_health_gate_is_stable_under_slot_randomization() -> None:
    completed = _health_completion(
        3,
        [
            (ALIASES[0], ("completed",), 1.0, False),
            (ALIASES[0], ("completed",), 1.0, True),
        ],
    )
    accumulator = _CandidatePoolHealthAccumulator(ALIASES)
    accumulator.add(completed)
    report = accumulator.report(
        _CandidatePoolHealthGateConfig(ALIASES, 2, 0.9),
        rollout_id=4,
        accepted_group_count=1,
    )

    assert report["triggered"] is False
    assert report["candidates"]["C0"]["available_session_count"] == 2
    assert report["candidates"]["C0"]["completion_fraction"] == 1.0


def test_candidate_pool_health_gate_skips_no_route_and_unattempted_calls() -> None:
    no_route = _router_sample(
        reservation_id=4,
        index=0,
        candidate=None,
    )
    skipped = _router_sample(
        reservation_id=4,
        index=1,
        candidate=ALIASES[0],
        statuses=("timeout",),
        attempted=False,
    )
    completed = _health_completion(4, [])
    completed.samples = [no_route, skipped]
    completed.session_count = 2
    accumulator = _CandidatePoolHealthAccumulator(ALIASES)
    accumulator.add(completed)
    report = accumulator.report(
        _CandidatePoolHealthGateConfig(ALIASES, 1, 0.5),
        rollout_id=4,
        accepted_group_count=1,
    )

    assert report["triggered"] is False
    assert report["candidates"]["C0"]["observed_session_count"] == 0
    assert report["candidates"]["C0"]["skipped_call_count"] == 1


def test_candidate_pool_health_gate_fails_closed_on_unattributed_cohort() -> None:
    completed = _health_completion(
        5,
        [(ALIASES[0], ("failed",), 0.0, False) for _ in range(16)],
    )
    for sample in completed.samples:
        router = sample.metadata["polar"]["trajectory_metadata"]["evaluation"][
            "spilot_router"
        ]
        router["calls"][0]["slot"] = "UNKNOWN"
    accumulator = _CandidatePoolHealthAccumulator(ALIASES)
    accumulator.add(completed)
    report = accumulator.report(
        _CandidatePoolHealthGateConfig(ALIASES, 16, 0.1),
        rollout_id=4,
        accepted_group_count=1,
    )

    assert report["triggered"] is True
    assert "global:unattributed_candidate_observations" in report["trigger_reasons"]


def test_candidate_pool_health_gate_attributes_verify_pre_call_infrastructure_failure() -> None:
    completed = _health_completion(
        6,
        [(ALIASES[0], ("completed",), 1.0, False) for _ in range(16)],
    )
    for sample in completed.samples:
        router = sample.metadata["polar"]["trajectory_metadata"]["evaluation"][
            "spilot_router"
        ]
        router["actions"].append(
            {"action": "VERIFY", "model_slot": "M1", "valid": True}
        )
        router["termination_reason"] = "infrastructure_error"
    accumulator = _CandidatePoolHealthAccumulator(ALIASES)
    accumulator.add(completed)
    report = accumulator.report(
        _CandidatePoolHealthGateConfig(ALIASES, 16, 0.1),
        rollout_id=4,
        accepted_group_count=0,
    )

    assert report["triggered"] is True
    assert report["candidates"]["C0"]["available_session_count"] == 16
    assert report["candidates"]["C1"]["unavailable_session_count"] == 16
    assert (
        report["candidates"]["C1"][
            "pre_call_infrastructure_failure_session_count"
        ]
        == 16
    )


def test_candidate_pool_health_incident_uses_run_state_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    blocked_save = tmp_path / "blocked-save"
    blocked_save.write_text("not a directory")
    state_file = tmp_path / "recovery.env"
    monkeypatch.setenv("TMAX_RUN_STATE_FILE", str(state_file))
    monkeypatch.setenv("SLURM_JOB_ID", "14000001")

    destination = _persist_candidate_pool_health_incident(
        SimpleNamespace(save=blocked_save),
        _report(["failed"] * 16),
    )

    assert destination.parent == Path(
        f"{state_file}.candidate_pool_health_incidents"
    )
    payload = json.loads(destination.read_text())
    assert payload["slurm_job_id"] == "14000001"
    assert payload["triggered"] is True


def test_candidate_pool_health_incident_records_wal_quarantine_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    completed = _health_completion(
        7,
        [(ALIASES[0], ("failed",), 0.0, False) for _ in range(16)],
    )
    accumulator = _CandidatePoolHealthAccumulator(ALIASES)
    accumulator.add(completed)

    class _BrokenStore:
        def quarantine(self, _reason: str) -> None:
            raise OSError("injected quarantine failure")

    monkeypatch.setattr(rollout_module, "stop_global_worker", lambda: None)
    with pytest.raises(CandidatePoolHealthGateError):
        _enforce_candidate_pool_health_gate(
            SimpleNamespace(save=tmp_path),
            accumulator,
            _CandidatePoolHealthGateConfig(ALIASES, 16, 0.1),
            rollout_id=9,
            accepted_group_count=0,
            partial_store=_BrokenStore(),
        )

    incident = json.loads(
        (
            tmp_path
            / "rollout"
            / "candidate_pool_health_incidents"
            / "rollout_0000009.json"
        ).read_text()
    )
    assert "global:partial_wal_quarantine_failed" in incident["trigger_reasons"]
    assert incident["partial_wal"] == {
        "present": True,
        "quarantine_error_type": "OSError",
        "quarantine_succeeded": False,
    }


class _GenerateWorker:
    def __init__(
        self,
        completions: list[_CompletedGroup],
        *,
        health_observations: list[_CompletedGroup] | None = None,
    ) -> None:
        self.config = SimpleNamespace(
            reward_key="score",
            candidate_pool_health_gate_enabled=True,
            candidate_pool_health_min_observed_sessions=16,
            candidate_pool_health_min_completion_fraction=0.1,
            task_template={
                "agent": {
                    "settings": {
                        "model_pool": {
                            "M0": {"model": ALIASES[0]},
                            "M1": {"model": ALIASES[1]},
                        }
                    }
                }
            },
        )
        self.completions = list(completions)
        self.health_observations = list(health_observations or [])
        self.requested: list[int] = []
        self.dynamic_filter_drops: list[int] = []
        self.consumed: list[tuple[list[int], str]] = []

    def set_rollout_context(self, _rollout_id: int) -> None:
        return None

    def request_groups(self, count: int) -> None:
        self.requested.append(count)

    def drain_completed(self, *, max_groups: int, rollout_id: int):
        del rollout_id
        drained = self.completions[:max_groups]
        del self.completions[:max_groups]
        return drained

    def drain_health_observations(self) -> list[_CompletedGroup]:
        drained = list(self.health_observations)
        self.health_observations.clear()
        return drained

    def queue_size(self) -> int:
        return len(self.completions)

    def mark_dynamic_filter_drop(self, completed, *, reason: str | None):
        del reason
        self.dynamic_filter_drops.append(completed.reservation_id)
        return self._consume_reservations(
            [completed.reservation_id],
            outcome="dynamic_filter",
        )

    def _consume_reservations(self, reservation_ids, *, outcome: str):
        self.consumed.append((list(reservation_ids), outcome))
        return {}

    def snapshot_metrics(self):
        return {}


def test_gate_sees_dynamic_filter_drops_and_never_commits_accepted_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    dropped = _health_completion(
        10,
        [(ALIASES[0], ("failed",), 0.0, False) for _ in range(15)]
        + [(None, (), 0.0, False)],
    )
    accepted = _health_completion(
        11,
        [(ALIASES[0], ("completed",), 0.0, False)]
        + [
            (ALIASES[1], ("completed",), float(index % 2), False)
            for index in range(15)
        ],
    )
    worker = _GenerateWorker([dropped, accepted])
    stopped: list[bool] = []
    monkeypatch.setattr(rollout_module, "get_global_async_worker", lambda *_args: worker)
    monkeypatch.setattr(rollout_module, "stop_global_worker", lambda: stopped.append(True))
    monkeypatch.setattr(rollout_module, "_current_ray_task_is_canceled", lambda: False)
    monkeypatch.setenv("SLURM_JOB_ID", "13960629")
    monkeypatch.setattr(
        rollout_module,
        "_load_rollout_train_output_type",
        lambda: pytest.fail("a rejected batch must not build a training output"),
    )

    with pytest.raises(CandidatePoolHealthGateError, match="C0"):
        generate_rollout_polar_async(
            SimpleNamespace(
                rollout_batch_size=1,
                reward_key="score",
                save=tmp_path,
                dynamic_sampling_filter_path=(
                    "slime.rollout.filter_hub.dynamic_sampling_filters."
                    "check_reward_nonzero_std"
                ),
            ),
            rollout_id=7,
            data_source=SimpleNamespace(),
        )

    assert stopped == [True]
    assert worker.dynamic_filter_drops == [10]
    assert worker.consumed == [([10], "dynamic_filter")]
    incident_path = (
        tmp_path
        / "rollout"
        / "candidate_pool_health_incidents"
        / "rollout_0000007.json"
    )
    incident = json.loads(incident_path.read_text())
    assert incident["triggered"] is True
    assert incident["slurm_job_id"] == "13960629"
    assert incident["decision_window_group_count"] == 2
    assert incident["candidates"]["C0"]["observed_session_count"] == 16
    assert incident["candidates"]["C1"]["available_session_count"] == 15


def test_gate_stops_worker_side_rejection_before_replacement_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    rejected = _health_completion(
        12,
        [(ALIASES[0], ("failed",), 0.0, False) for _ in range(16)],
    )
    worker = _GenerateWorker([], health_observations=[rejected])
    stopped: list[bool] = []
    monkeypatch.setattr(rollout_module, "get_global_async_worker", lambda *_args: worker)
    monkeypatch.setattr(rollout_module, "stop_global_worker", lambda: stopped.append(True))
    monkeypatch.setattr(rollout_module, "_current_ray_task_is_canceled", lambda: False)

    with pytest.raises(CandidatePoolHealthGateError, match="C0"):
        generate_rollout_polar_async(
            SimpleNamespace(rollout_batch_size=1, reward_key="score", save=tmp_path),
            rollout_id=7,
            data_source=SimpleNamespace(),
        )

    assert stopped == [True]
    assert worker.requested == [1]
    assert worker.dynamic_filter_drops == []


def test_low_complete_rejection_is_published_to_health_only_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        polar_rollout_url="http://rollout:8080",
        polar_task_template={
            "agent": {
                "harness": "codex",
                "settings": {
                    "model_pool": {
                        "M0": {"model": ALIASES[0]},
                        "M1": {"model": ALIASES[1]},
                    }
                },
            }
        },
        polar_max_async_level=1,
        rollout_batch_size=1,
        n_samples_per_prompt=16,
        update_weights_interval=1,
        polar_callback_host="127.0.0.1",
        polar_min_complete_accept_fraction=1.0,
        polar_candidate_pool_health_gate_enabled=True,
    )
    worker = AsyncPolarRolloutWorker(args, data_source=SimpleNamespace())
    samples = [
        _router_sample(
            reservation_id=13,
            index=index,
            candidate=ALIASES[0],
            statuses=("failed",),
        )
        for index in range(16)
    ]
    for sample in samples[1:]:
        sample.loss_mask = [0]
        sample.remove_sample = True
    task_result = SimpleNamespace(
        task_id="task-13",
        status="completed",
        results=[
            SimpleNamespace(
                session_id=f"session-13-{index}",
                status="COMPLETED" if index == 0 else "TIMEOUT",
            )
            for index in range(16)
        ],
    )
    pending = _PendingGroup(
        group_id=13,
        group=[SimpleNamespace(group_index=13) for _ in range(16)],
        reservation_id=13,
        submitted_rollout_id=7,
        policy_version=7,
        session_cost=16,
    )
    monkeypatch.setattr(
        rollout_module,
        "_build_task_payload",
        lambda **_kwargs: {"task_id": "task-13"},
    )

    async def return_result(_client, _payload):
        return task_result

    monkeypatch.setattr(worker, "_submit_with_callback", return_result)
    monkeypatch.setattr(
        rollout_module,
        "_convert_task_result_to_samples",
        lambda *_args, **_kwargs: samples,
    )

    with pytest.raises(PolarLowCompleteAcceptFractionError):
        asyncio.run(worker._submit_attempt(SimpleNamespace(), pending))

    observations = worker.drain_health_observations()
    assert len(observations) == 1
    assert observations[0].reservation_id == 13
    assert worker.output_queue.empty()


def test_gate_rejects_recovered_keep_before_mark_ready_or_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    recovered = _health_completion(
        20,
        [(ALIASES[0], ("failed",), 0.0, False) for _ in range(16)],
    )

    class _Store:
        def __init__(self) -> None:
            self.quarantined: list[str] = []

        def quarantine(self, reason: str) -> None:
            self.quarantined.append(reason)

        def mark_ready(self, _reservation_ids) -> None:
            pytest.fail("health gate must run before partial WAL mark_ready")

    store = _Store()
    worker = _GenerateWorker([])
    stopped: list[bool] = []
    monkeypatch.setattr(
        rollout_module,
        "_prepare_partial_recovery",
        lambda *_args, **_kwargs: _PartialRecoveryPlan(store=store, kept=[recovered]),
    )
    monkeypatch.setattr(rollout_module, "get_global_async_worker", lambda *_args: worker)
    monkeypatch.setattr(rollout_module, "stop_global_worker", lambda: stopped.append(True))
    monkeypatch.setattr(
        rollout_module,
        "_load_rollout_train_output_type",
        lambda: pytest.fail("a rejected batch must not build a training output"),
    )

    with pytest.raises(CandidatePoolHealthGateError):
        generate_rollout_polar_async(
            SimpleNamespace(rollout_batch_size=1, reward_key="score", save=tmp_path),
            rollout_id=8,
            data_source=SimpleNamespace(),
        )

    assert stopped == [True]
    assert worker.requested == []
    assert worker.consumed == []
    assert store.quarantined == [
        "candidate-pool health gate rejected the uncommitted decision window"
    ]
