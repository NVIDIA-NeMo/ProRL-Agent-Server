from types import SimpleNamespace

import pytest

from slime_bridge.eval_logging import (
    AGGREGATE_REWARD_METRIC,
    add_weighted_eval_metric,
)


def _dataset(rewards: list[float], *, valid_count: int | None = None) -> dict:
    return {
        "rewards": rewards,
        "valid_count": len(rewards) if valid_count is None else valid_count,
    }


def test_weighted_eval_metric_uses_full_tmax_and_terminal_bench_counts() -> None:
    metrics = {
        "eval/tmax_holdout/reward_mean": 0.25,
        "eval/terminal_bench_2_1/reward_mean": 0.5,
    }
    data = {
        "tmax_holdout": _dataset([0.25] * 100),
        "terminal_bench_2_1": _dataset([0.5] * 89),
    }

    skip_default_logging = add_weighted_eval_metric(
        0, SimpleNamespace(), data, metrics
    )

    assert skip_default_logging is False
    assert metrics[AGGREGATE_REWARD_METRIC] == pytest.approx(
        (100 * 0.25 + 89 * 0.5) / 189
    )
    assert metrics["eval/tmax_holdout/reward_mean"] == 0.25
    assert metrics["eval/terminal_bench_2_1/reward_mean"] == 0.5


def test_weighted_eval_metric_uses_actual_valid_samples_after_errors() -> None:
    metrics = {}
    data = {
        "tmax_holdout": _dataset([1.0, 0.0]),
        "terminal_bench_2_1": _dataset([1.0]),
    }

    add_weighted_eval_metric(7, SimpleNamespace(), data, metrics)

    # Dynamic 2:1 weighting, not the configured maximum 100:89 weighting.
    assert metrics[AGGREGATE_REWARD_METRIC] == pytest.approx(2 / 3)


def test_weighted_eval_metric_counts_zero_filled_errors_in_denominator() -> None:
    metrics = {}
    data = {
        "tmax_holdout": {
            "rewards": [1.0, 0.0],
            "valid_count": 1,
            "error_count": 1,
            "accounted_count": 2,
        },
        "terminal_bench_2_1": {
            "rewards": [1.0],
            "valid_count": 1,
            "error_count": 0,
            "accounted_count": 1,
        },
    }

    add_weighted_eval_metric(7, SimpleNamespace(), data, metrics)

    assert metrics[AGGREGATE_REWARD_METRIC] == pytest.approx(2 / 3)


def test_weighted_eval_metric_rejects_inconsistent_valid_count() -> None:
    with pytest.raises(ValueError, match="valid_count=2 does not match"):
        add_weighted_eval_metric(
            0,
            SimpleNamespace(),
            {"tmax_holdout": _dataset([1.0], valid_count=2)},
            {},
        )


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), True, "1"])
def test_weighted_eval_metric_rejects_invalid_rewards(reward: object) -> None:
    error_type = TypeError if isinstance(reward, (bool, str)) else ValueError
    with pytest.raises(error_type):
        add_weighted_eval_metric(
            0,
            SimpleNamespace(),
            {"tmax_holdout": _dataset([reward])},  # type: ignore[list-item]
            {},
        )


def test_weighted_eval_metric_does_not_publish_non_finite_empty_mean() -> None:
    metrics = {AGGREGATE_REWARD_METRIC: 123.0}

    assert (
        add_weighted_eval_metric(
            0,
            SimpleNamespace(),
            {
                "tmax_holdout": _dataset([]),
                "terminal_bench_2_1": _dataset([]),
            },
            metrics,
        )
        is False
    )
    assert AGGREGATE_REWARD_METRIC not in metrics


def test_slime_default_logger_keeps_all_eval_namespaces_on_one_train_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from slime.ray import rollout as slime_rollout

    published: list[tuple[dict, str]] = []
    monkeypatch.setattr(
        slime_rollout.logging_utils,
        "log",
        lambda _args, metrics, *, step_key: published.append(
            (dict(metrics), step_key)
        ),
    )
    args = SimpleNamespace(
        custom_eval_rollout_log_function_path=(
            "slime_bridge.eval_logging.add_weighted_eval_metric"
        ),
        log_passrate=False,
        wandb_always_use_train_step=True,
        rollout_batch_size=24,
        n_samples_per_prompt=8,
        global_batch_size=64,
    )
    data = {
        "tmax_holdout": {
            **_dataset([1.0, 0.0]),
            "samples": [],
            "truncated": [],
            "error_count": 0,
        },
        "terminal_bench_2_1": {
            **_dataset([1.0]),
            "samples": [],
            "truncated": [],
            "error_count": 0,
        },
    }

    result = slime_rollout._log_eval_rollout_data(
        rollout_id=2,
        args=args,
        data=data,
        extra_metrics={
            "eval/tmax_holdout/reward_mean": 0.5,
            "eval/terminal_bench_2_1/reward_mean": 1.0,
        },
        completed_train_batch=True,
    )

    assert len(published) == 1
    metrics, step_key = published[0]
    assert result == metrics
    assert metrics["eval/tmax_holdout/reward_mean"] == 0.5
    assert metrics["eval/terminal_bench_2_1/reward_mean"] == 1.0
    assert metrics[AGGREGATE_REWARD_METRIC] == pytest.approx(2 / 3)
    assert metrics["eval/train_step"] == 8
    assert "train/step" not in metrics
    assert step_key == "eval/train_step"
