from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "backfill_spilot_decomposed_metrics.py"
_SPEC = importlib.util.spec_from_file_location("backfill_spilot_decomposed_metrics", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
backfill = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = backfill
_SPEC.loader.exec_module(backfill)


def _sample(
    *,
    step: int,
    session_id: str,
    route_slot: str,
    raw_reward: float,
    shaped_reward: float,
    penalty: float,
    calls: list[dict],
    total_cost: float,
) -> SimpleNamespace:
    router = {
        "actions": [
            {"action": "ROUTE", "model_slot": route_slot, "valid": True},
            {"action": "SUBMIT", "valid": True},
        ],
        "calls": calls,
        "slot_mapping": {
            "M0": {"model": "pool/gpt"},
            "M1": {"model": "pool/qwen"},
        },
        "total_cost": total_cost,
    }
    return SimpleNamespace(
        reward={"score": shaped_reward},
        status=SimpleNamespace(name="COMPLETED"),
        remove_sample=False,
        loss_mask=[1],
        metadata={
            "polar": {
                "accepted_rollout_id": step,
                "session_id": session_id,
                "trajectory_metadata": {
                    "evaluation": {
                        "harbor_outcome_reward": raw_reward,
                        "reward": shaped_reward,
                        "applied_cost_penalty": penalty,
                        "total_cost": total_cost,
                        "spilot_router": router,
                    }
                },
            }
        },
    )


def _step_fixture(step: int = 0) -> tuple[list[SimpleNamespace], dict[str, float]]:
    c0 = _sample(
        step=step,
        session_id=f"c0-{step}",
        route_slot="M0",
        raw_reward=1.0,
        shaped_reward=0.9,
        penalty=0.1,
        calls=[
            {
                "model": "pool/gpt",
                "slot": "M0",
                "role": "solve",
                "cost": 15.0,
            },
            {
                "model": "pool/qwen",
                "slot": "M1",
                "role": "verify",
                "cost": 1.0,
            },
        ],
        total_cost=16.0,
    )
    c1 = _sample(
        step=step,
        session_id=f"c1-{step}",
        route_slot="M1",
        raw_reward=0.0,
        shaped_reward=0.0,
        penalty=0.0,
        calls=[
            {
                "model": "pool/qwen",
                "slot": "M1",
                "role": "solve",
                "cost": 1.0,
            }
        ],
        total_cost=1.0,
    )
    # Each accepted session has two trainable Router traces.  The script must
    # keep one vote per session while retaining trace_count for audit.
    samples = [c0, c0, c1, c1]
    metrics = {
        "polar/spilot_router/session_count": 2.0,
        "polar/spilot_router/reward_accounted_session_count": 2.0,
        "polar/spilot_router/reward_mean": 0.45,
        "polar/spilot_router/total_cost": 17.0,
        "polar/spilot_router/route_candidate_c0_count": 1.0,
        "polar/spilot_router/route_candidate_c1_count": 1.0,
    }
    return samples, metrics


@pytest.mark.unit
def test_reconstruct_step_deduplicates_traces_and_decomposes_reward_and_cost() -> None:
    samples, legacy = _step_fixture(step=7)

    row = backfill.reconstruct_step(7, samples, legacy)

    assert row["postrun_v1/rollout_step"] == 7
    assert row["postrun_v1/trace_count"] == 4
    assert row["postrun_v1/session_count"] == 2
    assert row["postrun_v1/accuracy_outcome"] == 0.5
    assert row["postrun_v1/accuracy_outcome_accounted_session_count"] == 2
    assert row["postrun_v1/accuracy_outcome_positive_count"] == 1
    assert row["postrun_v1/reward_shaped"] == 0.45
    assert row["postrun_v1/reward_accounted_session_count"] == 2.0
    assert row["postrun_v1/cost_penalty_fraction"] == 0.05
    assert row["postrun_v1/cost_penalty_contribution"] == 0.05
    assert row["postrun_v1/cost_penalty_contribution_total"] == 0.1
    assert row["postrun_v1/cost_total"] == 17.0
    assert row["postrun_v1/cost_accounted_session_count"] == 2
    assert row["postrun_v1/cost_per_session"] == 8.5
    assert row["postrun_v1/cost_per_session_std"] == 7.5
    assert row["postrun_v1/cost_per_session_median"] == 8.5
    assert row["postrun_v1/cost_per_session_min"] == 1.0
    assert row["postrun_v1/cost_per_session_max"] == 16.0
    assert row["postrun_v1/cost_candidate_c0_total"] == 15.0
    assert row["postrun_v1/cost_candidate_c1_total"] == 2.0
    assert row["postrun_v1/cost_unattributed_total"] == 0.0
    assert row["postrun_v1/cost_solve_total"] == 16.0
    assert row["postrun_v1/cost_verify_total"] == 1.0
    assert row["postrun_v1/cost_other_role_total"] == 0.0
    assert row["postrun_v1/route_candidate_c0_count"] == 1
    assert row["postrun_v1/route_candidate_c1_count"] == 1
    assert row["postrun_v1/accuracy_outcome_candidate_c0_accounted_session_count"] == 1
    assert row["postrun_v1/accuracy_outcome_candidate_c1_accounted_session_count"] == 1
    assert row["postrun_v1/accuracy_outcome_candidate_c0"] == 1.0
    assert row["postrun_v1/accuracy_outcome_candidate_c1"] == 0.0


@pytest.mark.unit
def test_reward_cohort_can_be_smaller_than_accuracy_and_cost_cohort() -> None:
    samples, legacy = _step_fixture(step=24)
    legacy["polar/spilot_router/reward_accounted_session_count"] = 1.0
    legacy["polar/spilot_router/reward_mean"] = 0.9

    row = backfill.reconstruct_step(24, samples, legacy)

    assert row["postrun_v1/accuracy_outcome_accounted_session_count"] == 2
    assert row["postrun_v1/cost_accounted_session_count"] == 2
    assert row["postrun_v1/reward_accounted_session_count"] == 1.0
    assert row["postrun_v1/accuracy_outcome"] == 0.5
    assert row["postrun_v1/reward_shaped"] == 0.9
    assert "postrun_v1/non_cost_reward_delta" not in row


@pytest.mark.unit
@pytest.mark.parametrize("invalid_count", [1.5, 3.0])
def test_reward_accounted_count_must_be_bounded_integer(invalid_count: float) -> None:
    samples, legacy = _step_fixture(step=24)
    legacy["polar/spilot_router/reward_accounted_session_count"] = invalid_count

    with pytest.raises(backfill.BackfillError, match="reward_accounted_session_count"):
        backfill.reconstruct_step(24, samples, legacy)


@pytest.mark.unit
def test_empty_candidate_cohort_has_count_but_omits_mean() -> None:
    samples, legacy = _step_fixture(step=25)
    only_c0 = samples[:2]
    legacy.update(
        {
            "polar/spilot_router/session_count": 1.0,
            "polar/spilot_router/reward_accounted_session_count": 1.0,
            "polar/spilot_router/reward_mean": 0.9,
            "polar/spilot_router/total_cost": 16.0,
            "polar/spilot_router/route_candidate_c0_count": 1.0,
            "polar/spilot_router/route_candidate_c1_count": 0.0,
        }
    )

    row = backfill.reconstruct_step(25, only_c0, legacy)

    assert row["postrun_v1/accuracy_outcome_candidate_c0_accounted_session_count"] == 1
    assert row["postrun_v1/accuracy_outcome_candidate_c1_accounted_session_count"] == 0
    assert row["postrun_v1/accuracy_outcome_candidate_c0"] == 1.0
    assert "postrun_v1/accuracy_outcome_candidate_c1" not in row


@pytest.mark.unit
def test_reconstruct_step_explicitly_excludes_failed_removed_placeholder() -> None:
    samples, legacy = _step_fixture(step=8)
    excluded = SimpleNamespace(
        status=SimpleNamespace(name="FAILED"),
        remove_sample=True,
        loss_mask=[0, 0],
        metadata={
            "polar": {
                "accepted_rollout_id": 8,
                "session_id": "excluded-8",
                "trajectory_metadata": {},
            }
        },
    )

    baseline = backfill.reconstruct_step(8, samples, legacy)
    with_excluded = backfill.reconstruct_step(8, [*samples, excluded, excluded], legacy)

    assert set(with_excluded) == set(baseline)
    assert with_excluded["postrun_v1/trace_count"] == 6
    assert with_excluded["postrun_v1/session_count"] == 2
    assert with_excluded["postrun_v1/excluded_session_count"] == 1
    assert with_excluded["postrun_v1/accuracy_outcome"] == baseline["postrun_v1/accuracy_outcome"]


@pytest.mark.unit
def test_reconstruct_step_rejects_conflicting_trace_metadata() -> None:
    samples, legacy = _step_fixture(step=2)
    conflicting = _sample(
        step=2,
        session_id="c0-2",
        route_slot="M0",
        raw_reward=0.0,
        shaped_reward=0.0,
        penalty=0.0,
        calls=[
            {
                "model": "pool/gpt",
                "slot": "M0",
                "role": "solve",
                "cost": 15.0,
            }
        ],
        total_cost=15.0,
    )

    with pytest.raises(backfill.BackfillError, match="conflicting trace-level"):
        backfill.reconstruct_step(2, [samples[0], conflicting, *samples[2:]], legacy)


@pytest.mark.unit
def test_reconstruct_rows_covers_zero_through_completed_step(tmp_path: Path) -> None:
    (tmp_path / "train_progress.step").write_text("1\n")
    expected_paths = []
    for step in range(2):
        pending, emitted = backfill.journal_paths(tmp_path, step)
        pending.parent.mkdir(parents=True, exist_ok=True)
        pending.write_bytes(b"trusted-test-placeholder")
        emitted.write_text(f"version=1\nrollout_id={step}\n")
        expected_paths.append(pending)

    loaded = []

    def loader(path: Path, step: int):
        loaded.append((path, step))
        return _step_fixture(step)

    completed, rows = backfill.reconstruct_rows(tmp_path, loader=loader)

    assert completed == 1
    assert [row["postrun_v1/rollout_step"] for row in rows] == [0, 1]
    assert loaded == list(zip(expected_paths, range(2), strict=True))


@pytest.mark.unit
def test_reconstruct_rows_requires_emitted_marker(tmp_path: Path) -> None:
    (tmp_path / "train_progress.step").write_text("0\n")
    pending, _ = backfill.journal_paths(tmp_path, 0)
    pending.parent.mkdir(parents=True)
    pending.write_bytes(b"trusted-test-placeholder")

    with pytest.raises(backfill.BackfillError, match="no emitted marker"):
        backfill.reconstruct_rows(tmp_path, loader=lambda *_args: _step_fixture(0))


@pytest.mark.unit
def test_reconstruct_rows_rejects_candidate_alias_pair_drift(tmp_path: Path) -> None:
    (tmp_path / "train_progress.step").write_text("1\n")
    for step in range(2):
        pending, emitted = backfill.journal_paths(tmp_path, step)
        pending.parent.mkdir(parents=True, exist_ok=True)
        pending.write_bytes(b"trusted-test-placeholder")
        emitted.write_text(f"version=1\nrollout_id={step}\n")

    def loader(_path: Path, step: int):
        samples, metrics = _step_fixture(step)
        samples = copy.deepcopy(samples)
        if step == 1:
            for sample in samples:
                evaluation = sample.metadata["polar"]["trajectory_metadata"]["evaluation"]
                evaluation["spilot_router"]["slot_mapping"]["M1"]["model"] = "pool/zeta"
        return samples, metrics

    with pytest.raises(backfill.BackfillError, match="alias pair drift"):
        backfill.reconstruct_rows(tmp_path, loader=loader)


@pytest.mark.unit
def test_default_cli_mode_is_dry_run_and_never_calls_commit(monkeypatch, tmp_path, capsys) -> None:
    rows = [{"postrun_v1/rollout_step": 0, "postrun_v1/value": 1.0}]
    monkeypatch.setattr(backfill, "reconstruct_rows", lambda *_args, **_kwargs: (0, rows))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run must not call commit_rows")

    monkeypatch.setattr(backfill, "commit_rows", forbidden)

    assert backfill.main(["--checkpoint-root", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert '"status": "dry-run-only"' in output
    assert "postrun_v1/rollout_step" in output


@pytest.mark.unit
def test_commit_cli_requires_explicit_environment_credential(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with pytest.raises(SystemExit):
        backfill.parse_args(
            [
                "--checkpoint-root",
                str(tmp_path),
                "--entity",
                "hwinf_dcm",
                "--project",
                "SPilot",
                "--run-id",
                "run-id",
                "--commit",
            ]
        )


class _RemoteRun:
    def __init__(self, rows=(), *, state="finished"):
        self.entity = "hwinf_dcm"
        self.project = "SPilot"
        self.id = "run-id"
        self.state = state
        self._rows = list(rows)

    def scan_history(self, **_kwargs):
        return iter(self._rows)


class _RemoteRunWithHistoryKeys(_RemoteRun):
    def __init__(self, rows=(), *, history_keys=(), state="finished"):
        super().__init__(rows=rows, state=state)
        self._attrs = {
            "historyKeys": {"keys": {key: {"typeCounts": []} for key in history_keys}}
        }


class _RemoteRunWithHistoryFallback(_RemoteRun):
    def __init__(self, rows, *, axis_count):
        super().__init__(rows=rows)
        self._attrs = {
            "historyKeys": {
                "keys": {
                    "postrun_v1/rollout_step": {
                        "typeCounts": [{"type": "number", "count": axis_count}]
                    }
                }
            }
        }
        self.history_kwargs = None

    def scan_history(self, **_kwargs):
        raise RuntimeError("Step column '_step' not found in schema")

    def history(self, **kwargs):
        self.history_kwargs = kwargs
        return list(self._rows)


class _Api:
    def __init__(self, run):
        self._run = run
        self.paths = []

    def run(self, path):
        self.paths.append(path)
        return self._run


@pytest.mark.unit
def test_preflight_pins_identity_and_rejects_live_writer() -> None:
    api = _Api(_RemoteRun(state="running"))

    with pytest.raises(backfill.BackfillError, match="only 'finished'"):
        backfill.preflight_remote_run(
            api,
            entity="hwinf_dcm",
            project="SPilot",
            run_id="run-id",
        )
    assert api.paths == ["hwinf_dcm/SPilot/run-id"]


@pytest.mark.unit
def test_preflight_rejects_other_terminal_states() -> None:
    for state in ("crashed", "failed", "killed", ""):
        with pytest.raises(backfill.BackfillError, match="only 'finished'"):
            backfill.preflight_remote_run(
                _Api(_RemoteRun(state=state)),
                entity="hwinf_dcm",
                project="SPilot",
                run_id="run-id",
            )


@pytest.mark.unit
def test_scan_and_compare_are_idempotent_but_reject_divergence() -> None:
    local = [
        {
            "postrun_v1/rollout_step": 0,
            "postrun_v1/accuracy_outcome": 0.75,
            "postrun_v1/cost_total": 10.0,
        },
        {
            "postrun_v1/rollout_step": 1,
            "postrun_v1/accuracy_outcome": 0.5,
            "postrun_v1/cost_total": 12.0,
        },
    ]
    metric_keys = ["postrun_v1/accuracy_outcome", "postrun_v1/cost_total"]
    remote = _RemoteRun(rows=[dict(local[0])])
    scanned = backfill.scan_remote_rows(
        remote,
        axis_key="postrun_v1/rollout_step",
        metric_keys=metric_keys,
    )

    missing = backfill.compare_remote_rows(
        local,
        scanned,
        axis_key="postrun_v1/rollout_step",
    )
    assert missing == [local[1]]

    divergent = {0: {**local[0], "postrun_v1/cost_total": 999.0}}
    with pytest.raises(backfill.BackfillError, match="differs at"):
        backfill.compare_remote_rows(
            local,
            divergent,
            axis_key="postrun_v1/rollout_step",
        )


@pytest.mark.unit
def test_scan_remote_rows_rejects_partial_and_duplicate_axis_rows() -> None:
    axis = "postrun_v1/rollout_step"
    metric_keys = ["postrun_v1/value"]
    with pytest.raises(backfill.BackfillError, match="partial"):
        backfill.scan_remote_rows(
            _RemoteRun(rows=[{axis: 0}]),
            axis_key=axis,
            metric_keys=metric_keys,
        )
    with pytest.raises(backfill.BackfillError, match="duplicate"):
        backfill.scan_remote_rows(
            _RemoteRun(rows=[{axis: 0, metric_keys[0]: 1}, {axis: 0, metric_keys[0]: 1}]),
            axis_key=axis,
            metric_keys=metric_keys,
        )


@pytest.mark.unit
def test_scan_remote_rows_uses_count_checked_history_fallback() -> None:
    axis = "postrun_v1/rollout_step"
    metric = "postrun_v1/value"
    source = _RemoteRunWithHistoryFallback(
        [{axis: 0, metric: 1.0}, {axis: 1, metric: 2.0}],
        axis_count=2,
    )

    assert backfill.scan_remote_rows(
        source,
        axis_key=axis,
        metric_keys=[metric],
    ) == {
        0: {axis: 0, metric: 1.0},
        1: {axis: 1, metric: 2.0},
    }
    assert source.history_kwargs == {
        "keys": [axis, metric],
        "samples": 10_000,
        "x_axis": "_step",
        "pandas": False,
    }


@pytest.mark.unit
def test_scan_remote_rows_rejects_incomplete_history_fallback() -> None:
    axis = "postrun_v1/rollout_step"
    metric = "postrun_v1/value"
    source = _RemoteRunWithHistoryFallback([{axis: 0, metric: 1.0}], axis_count=2)

    with pytest.raises(backfill.BackfillError, match="fallback is incomplete"):
        backfill.scan_remote_rows(source, axis_key=axis, metric_keys=[metric])


@pytest.mark.unit
def test_scan_remote_rows_rejects_orphaned_metric_fallback_row() -> None:
    axis = "postrun_v1/rollout_step"
    metric = "postrun_v1/value"
    source = _RemoteRunWithHistoryFallback(
        [{axis: 0, metric: 1.0}, {metric: 9.0}],
        axis_count=1,
    )

    with pytest.raises(backfill.BackfillError, match="without business axis"):
        backfill.scan_remote_rows(source, axis_key=axis, metric_keys=[metric])


@pytest.mark.unit
def test_scan_remote_rows_checks_fallback_metric_counts() -> None:
    axis = "postrun_v1/rollout_step"
    metric = "postrun_v1/value"
    source = _RemoteRunWithHistoryFallback([{axis: 0, metric: 1.0}], axis_count=1)
    source._attrs["historyKeys"]["keys"][metric] = {
        "typeCounts": [{"type": "number", "count": 2}]
    }

    with pytest.raises(backfill.BackfillError, match="does not match the expected count"):
        backfill.scan_remote_rows(
            source,
            axis_key=axis,
            metric_keys=[metric],
            expected_keys_by_step={0: {metric}},
        )


@pytest.mark.unit
def test_remote_history_key_names_distinguishes_absent_metadata() -> None:
    axis = "postrun_v1/rollout_step"

    assert backfill._remote_history_key_names(_RemoteRun()) is None
    assert backfill._remote_history_key_names(
        _RemoteRunWithHistoryKeys(history_keys=[axis, "postrun_v1/value"])
    ) == {axis, "postrun_v1/value"}


@pytest.mark.unit
def test_commit_skips_broken_scan_for_authoritatively_new_namespace(
    monkeypatch, tmp_path: Path
) -> None:
    axis = "postrun_v1/rollout_step"
    row = {axis: 0, "postrun_v1/accuracy_outcome": 0.75}
    api = _Api(_RemoteRunWithHistoryKeys(history_keys=["legacy/value"]))

    class _CommitWandb:
        @staticmethod
        def Api():
            return api

    appended = []
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setattr(
        backfill,
        "scan_remote_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an absent namespace must not call W&B scan_history")
        ),
    )
    monkeypatch.setattr(
        backfill,
        "append_missing_rows",
        lambda _wandb, rows, **_kwargs: appended.extend(rows),
    )
    monkeypatch.setattr(backfill, "wait_for_readback", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backfill, "_assert_progress_unchanged", lambda *_args: None)

    written = backfill.commit_rows(
        [row],
        checkpoint_root=tmp_path,
        completed_step=0,
        entity="hwinf_dcm",
        project="SPilot",
        run_id="run-id",
        metric_prefix="postrun_v1",
        wandb_dir=None,
        readback_timeout_s=1.0,
        readback_interval_s=0.01,
        wandb_module=_CommitWandb(),
    )

    assert written == 1
    assert appended == [row]


@pytest.mark.unit
def test_commit_rejects_orphaned_namespace_without_axis(monkeypatch, tmp_path: Path) -> None:
    axis = "postrun_v1/rollout_step"
    row = {axis: 0, "postrun_v1/accuracy_outcome": 0.75}
    api = _Api(_RemoteRunWithHistoryKeys(history_keys=["postrun_v1/reward_shaped"]))

    class _CommitWandb:
        @staticmethod
        def Api():
            return api

    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setattr(backfill, "_assert_progress_unchanged", lambda *_args: None)

    with pytest.raises(backfill.BackfillError, match="has keys but no business axis"):
        backfill.commit_rows(
            [row],
            checkpoint_root=tmp_path,
            completed_step=0,
            entity="hwinf_dcm",
            project="SPilot",
            run_id="run-id",
            metric_prefix="postrun_v1",
            wandb_dir=None,
            readback_timeout_s=1.0,
            readback_interval_s=0.01,
            wandb_module=_CommitWandb(),
        )


@pytest.mark.unit
def test_commit_without_history_metadata_still_scans_existing_rows(
    monkeypatch, tmp_path: Path
) -> None:
    axis = "postrun_v1/rollout_step"
    row = {axis: 0, "postrun_v1/accuracy_outcome": 0.75}
    api = _Api(_RemoteRun(rows=[row]))

    class _CommitWandb:
        @staticmethod
        def Api():
            return api

    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setattr(
        backfill,
        "append_missing_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an existing exact row must not be appended again")
        ),
    )
    monkeypatch.setattr(backfill, "wait_for_readback", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backfill, "_assert_progress_unchanged", lambda *_args: None)

    written = backfill.commit_rows(
        [row],
        checkpoint_root=tmp_path,
        completed_step=0,
        entity="hwinf_dcm",
        project="SPilot",
        run_id="run-id",
        metric_prefix="postrun_v1",
        wandb_dir=None,
        readback_timeout_s=1.0,
        readback_interval_s=0.01,
        wandb_module=_CommitWandb(),
    )

    assert written == 0


@pytest.mark.unit
def test_commit_with_known_axis_keeps_partial_row_validation(
    monkeypatch, tmp_path: Path
) -> None:
    axis = "postrun_v1/rollout_step"
    row = {axis: 0, "postrun_v1/accuracy_outcome": 0.75}
    api = _Api(_RemoteRunWithHistoryKeys(rows=[{axis: 0}], history_keys=[axis]))

    class _CommitWandb:
        @staticmethod
        def Api():
            return api

    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setattr(backfill, "_assert_progress_unchanged", lambda *_args: None)

    with pytest.raises(backfill.BackfillError, match="partial"):
        backfill.commit_rows(
            [row],
            checkpoint_root=tmp_path,
            completed_step=0,
            entity="hwinf_dcm",
            project="SPilot",
            run_id="run-id",
            metric_prefix="postrun_v1",
            wandb_dir=None,
            readback_timeout_s=1.0,
            readback_interval_s=0.01,
            wandb_module=_CommitWandb(),
        )


@pytest.mark.unit
def test_remote_readback_supports_legitimate_optional_candidate_mean() -> None:
    axis = "postrun_v1/rollout_step"
    count = "postrun_v1/accuracy_outcome_candidate_c1_accounted_session_count"
    mean = "postrun_v1/accuracy_outcome_candidate_c1"
    local = [
        {axis: 0, count: 1, mean: 0.75},
        {axis: 1, count: 0},
    ]
    metric_keys = backfill._metric_keys(local, axis_key=axis)
    expected = backfill._expected_keys_by_step(local, axis_key=axis)

    scanned = backfill.scan_remote_rows(
        _RemoteRun(rows=local),
        axis_key=axis,
        metric_keys=metric_keys,
        expected_keys_by_step=expected,
    )

    assert backfill.compare_remote_rows(local, scanned, axis_key=axis) == []


class _LoggingRun:
    entity = "hwinf_dcm"
    project = "SPilot"
    id = "run-id"

    def __init__(self):
        self.definitions = []
        self.logged = []
        self.finished = []

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def log(self, data, **kwargs):
        assert kwargs == {}, "SDK step= must never be supplied"
        self.logged.append(dict(data))

    def finish(self, **kwargs):
        self.finished.append(kwargs)


class _Wandb:
    def __init__(self):
        self.run = _LoggingRun()
        self.init_kwargs = None

    @staticmethod
    def Settings(**kwargs):
        return kwargs

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


@pytest.mark.unit
def test_append_resumes_exact_run_and_never_sets_sdk_step(tmp_path: Path) -> None:
    wandb = _Wandb()
    row = {
        "postrun_v1/rollout_step": 4,
        "postrun_v1/accuracy_outcome": 0.75,
    }

    backfill.append_missing_rows(
        wandb,
        [row],
        entity="hwinf_dcm",
        project="SPilot",
        run_id="run-id",
        axis_key="postrun_v1/rollout_step",
        metric_keys=["postrun_v1/accuracy_outcome"],
        wandb_dir=tmp_path,
    )

    assert wandb.init_kwargs["entity"] == "hwinf_dcm"
    assert wandb.init_kwargs["project"] == "SPilot"
    assert wandb.init_kwargs["id"] == "run-id"
    assert wandb.init_kwargs["resume"] == "must"
    assert wandb.run.definitions == [
        ("postrun_v1/rollout_step", {"summary": "none"}),
        (
            "postrun_v1/accuracy_outcome",
            {"step_metric": "postrun_v1/rollout_step", "summary": "none"},
        ),
    ]
    assert wandb.run.logged == [row]
    assert wandb.run.finished == [{"exit_code": 0}]


@pytest.mark.unit
def test_readback_requires_every_exact_row() -> None:
    row = {
        "postrun_v1/rollout_step": 0,
        "postrun_v1/accuracy_outcome": 0.75,
    }
    api = _Api(_RemoteRun(rows=[row]))

    backfill.wait_for_readback(
        api,
        [row],
        entity="hwinf_dcm",
        project="SPilot",
        run_id="run-id",
        axis_key="postrun_v1/rollout_step",
        metric_keys=["postrun_v1/accuracy_outcome"],
        timeout_s=1.0,
        interval_s=0.01,
        sleep=lambda _seconds: None,
    )
    assert api.paths == ["hwinf_dcm/SPilot/run-id"]


@pytest.mark.unit
def test_readback_refreshes_cached_remote_metadata() -> None:
    axis = "postrun_v1/rollout_step"
    metric = "postrun_v1/accuracy_outcome"
    row = {axis: 0, metric: 0.75}

    class _RefreshableRemoteRun(_RemoteRun):
        def __init__(self):
            super().__init__(rows=[row])
            self.loads = []

        def load(self, *, force):
            self.loads.append(force)
            return self

    remote = _RefreshableRemoteRun()
    backfill.wait_for_readback(
        _Api(remote),
        [row],
        entity="hwinf_dcm",
        project="SPilot",
        run_id="run-id",
        axis_key=axis,
        metric_keys=[metric],
        timeout_s=1.0,
        interval_s=0.01,
        sleep=lambda _seconds: None,
    )

    assert remote.loads == [True]
