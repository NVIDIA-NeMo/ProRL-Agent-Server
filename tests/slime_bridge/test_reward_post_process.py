from __future__ import annotations

import statistics
from types import SimpleNamespace
from typing import Any

import pytest

from slime_bridge.reward_post_process import post_process_rewards


class FakeSample:
    def __init__(
        self,
        *,
        group_index: int = 0,
        group_id: int,
        reward: Any,
        status: str = "COMPLETED",
        loss_mask: list[int] | None = None,
        remove_sample: bool = False,
        training_filter: dict | None = None,
        reward_components: dict[str, Any] | None = None,
    ) -> None:
        self.group_index = group_index
        self.group_id = group_id
        self.index = group_id
        self.reward = {"score": reward}
        if reward_components is not None:
            self.reward.update(reward_components)
        self.status = status
        self.loss_mask = [1] if loss_mask is None else loss_mask
        self.response_length = len(self.loss_mask)
        self.remove_sample = remove_sample
        self.metadata = {
            "polar": {"training_filter": training_filter} if training_filter is not None else {}
        }

    def get_reward_value(self, args) -> Any:
        return self.reward[args.reward_key]


def _args(**overrides):
    defaults = {
        "reward_key": "score",
        "rewards_normalization": True,
        "advantage_estimator": "grpo",
        "grpo_std_normalization": False,
        "dvao_reward_keys": None,
        "gdpo_reward_keys": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _production_grpo_args(**overrides):
    """Match the production GRPO reward-normalization switches."""
    return _args(grpo_std_normalization=True, **overrides)


def test_dynamic_trace_loo_keeps_per_trace_rewards() -> None:
    samples = [
        FakeSample(group_id=10, reward=2.0),
        FakeSample(group_id=10, reward=4.0),
        FakeSample(group_id=20, reward=10.0),
    ]

    raw, rewards = post_process_rewards(_args(), samples)

    assert raw == [2.0, 4.0, 10.0]
    assert rewards == [-8.0, -6.0, 7.0]


def test_gdpo_is_invariant_to_reward_component_scale() -> None:
    def advantages(cost_scale: float) -> list[float]:
        reward_vectors = [
            (1.0, 0.0),
            (0.0, 0.0),
            (1.0, cost_scale),
            (0.0, cost_scale),
        ]
        samples = [
            FakeSample(
                group_id=trajectory_id,
                reward=0.0,
                reward_components={
                    "accuracy": accuracy,
                    "cost": cost,
                },
            )
            for trajectory_id, (accuracy, cost) in enumerate(reward_vectors)
        ]
        _raw, normalized = post_process_rewards(
            _args(gdpo_reward_keys=["accuracy", "cost"]),
            samples,
        )
        return normalized

    assert advantages(1.0) == pytest.approx(advantages(100.0), abs=2e-4)


def test_dvao_matches_paper_formula_for_two_rewards() -> None:
    reward_vectors = [
        (1.0, 0.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]
    samples = [
        FakeSample(
            group_id=trajectory_id,
            reward=0.0,
            reward_components={
                "reward_1": reward_1,
                "reward_2": reward_2,
            },
        )
        for trajectory_id, (reward_1, reward_2) in enumerate(reward_vectors)
    ]

    raw, advantages = post_process_rewards(
        _args(dvao_reward_keys=["reward_1", "reward_2"]),
        samples,
    )

    assert raw == [0.5, 0.0, 1.0, 0.5]
    assert advantages == pytest.approx([0.0, -1.0, 1.0, 0.0])


def test_dvao_zero_variance_component_has_zero_weight() -> None:
    samples = [
        FakeSample(
            group_id=0,
            reward=0.0,
            reward_components={"reward_1": 0.0, "reward_2": 1.0},
        ),
        FakeSample(
            group_id=1,
            reward=0.0,
            reward_components={"reward_1": 1.0, "reward_2": 1.0},
        ),
    ]

    _raw, advantages = post_process_rewards(
        _args(dvao_reward_keys=["reward_1", "reward_2"]),
        samples,
    )

    assert advantages == pytest.approx([-1.0, 1.0])


def test_dvao_uses_trajectory_means_for_fanout_group_statistics() -> None:
    samples = [
        FakeSample(
            group_id=0,
            reward=0.0,
            reward_components={"reward_1": 0.0, "reward_2": 0.0},
        ),
        FakeSample(
            group_id=0,
            reward=0.0,
            reward_components={"reward_1": 1.0, "reward_2": 0.0},
        ),
        FakeSample(
            group_id=1,
            reward=0.0,
            reward_components={"reward_1": 1.0, "reward_2": 1.0},
        ),
    ]

    _raw, advantages = post_process_rewards(
        _args(dvao_reward_keys=["reward_1", "reward_2"]),
        samples,
    )

    # Trajectory vectors are (0.5, 0.0) and (1.0, 1.0). The two traces in
    # trajectory 0 retain distinct signals whose mean is its exact
    # trajectory-level DVAO advantage.
    assert advantages == pytest.approx([-5.0 / 3.0, -1.0 / 3.0, 1.0])
    assert statistics.fmean(advantages[:2]) == pytest.approx(-1.0)


def test_dvao_missing_trainable_reward_fails_fast() -> None:
    samples = [
        FakeSample(
            group_id=0,
            reward=0.0,
            reward_components={"reward_1": 1.0},
        ),
        FakeSample(
            group_id=1,
            reward=0.0,
            reward_components={"reward_1": 0.0, "reward_2": 1.0},
        ),
    ]

    with pytest.raises(ValueError, match="reward_2.*missing"):
        post_process_rewards(
            _args(dvao_reward_keys=["reward_1", "reward_2"]),
            samples,
        )


def test_dvao_failed_sample_does_not_require_components() -> None:
    samples = [
        FakeSample(group_id=0, reward=1.0, status="FAILED"),
        FakeSample(
            group_id=1,
            reward=0.0,
            reward_components={"reward_1": 0.0, "reward_2": 1.0},
        ),
    ]

    raw, advantages = post_process_rewards(
        _args(dvao_reward_keys=["reward_1", "reward_2"]),
        samples,
    )

    assert raw == [0.0, 0.5]
    assert advantages == [0.0, 0.0]


def test_production_grpo_single_winner_uses_common_group_std() -> None:
    raw_rewards = [1.0] + [0.0] * 7
    samples = [
        FakeSample(group_id=trajectory_id, reward=reward)
        for trajectory_id, reward in enumerate(raw_rewards)
    ]

    raw, rewards = post_process_rewards(_production_grpo_args(), samples)

    scale = statistics.stdev(raw_rewards) + 1e-6
    assert raw == raw_rewards
    assert rewards[0] == pytest.approx(1.0 / scale)
    assert rewards[1:] == pytest.approx([-1.0 / 7.0 / scale] * 7)
    assert max(abs(value) for value in rewards) < 3.0


def test_production_grpo_single_loser_uses_common_group_std() -> None:
    raw_rewards = [0.0] + [1.0] * 7
    samples = [
        FakeSample(group_id=trajectory_id, reward=reward)
        for trajectory_id, reward in enumerate(raw_rewards)
    ]

    raw, rewards = post_process_rewards(_production_grpo_args(), samples)

    scale = statistics.stdev(raw_rewards) + 1e-6
    assert raw == raw_rewards
    assert rewards[0] == pytest.approx(-1.0 / scale)
    assert rewards[1:] == pytest.approx([1.0 / 7.0 / scale] * 7)
    assert max(abs(value) for value in rewards) < 3.0


def test_production_grpo_identical_reward_group_has_zero_advantages() -> None:
    samples = [FakeSample(group_id=trajectory_id, reward=1.0) for trajectory_id in range(8)]

    raw, rewards = post_process_rewards(_production_grpo_args(), samples)

    assert raw == [1.0] * 8
    assert rewards == [0.0] * 8


def test_production_grpo_multi_trace_trajectory_uses_trajectory_means_for_common_std() -> None:
    samples = [
        FakeSample(group_id=10, reward=2.0),
        FakeSample(group_id=10, reward=4.0),
        FakeSample(group_id=20, reward=10.0),
        FakeSample(group_id=30, reward=6.0),
    ]

    raw, rewards = post_process_rewards(_production_grpo_args(), samples)

    # Trajectory means are [3, 10, 6]. The shared scale is computed once from
    # those three exchangeable units, while each trace retains its own reward
    # against its trajectory's leave-one-out baseline.
    scale = statistics.stdev([3.0, 10.0, 6.0]) + 1e-6
    assert raw == [2.0, 4.0, 10.0, 6.0]
    assert rewards == pytest.approx(
        [
            (2.0 - 8.0) / scale,
            (4.0 - 8.0) / scale,
            (10.0 - 4.5) / scale,
            (6.0 - 6.5) / scale,
        ]
    )


def test_failed_trajectory_is_excluded_from_other_baselines() -> None:
    samples = [
        FakeSample(group_id=1, reward=2.0),
        FakeSample(group_id=2, reward=10.0, status="FAILED"),
        FakeSample(group_id=3, reward=6.0),
    ]

    raw, rewards = post_process_rewards(_args(), samples)

    assert raw == [2.0, 0.0, 6.0]
    assert rewards == [-4.0, 0.0, 4.0]


def test_removed_sample_cannot_replay_a_positive_reward() -> None:
    samples = [
        FakeSample(group_id=1, reward=1.0, remove_sample=True),
        FakeSample(group_id=2, reward=0.0),
    ]

    raw, rewards = post_process_rewards(_production_grpo_args(), samples)

    assert raw == [0.0, 0.0]
    assert rewards == [0.0, 0.0]


def test_aligned_zero_reward_policy_failure_gets_negative_loo_advantage() -> None:
    samples = [
        # Aligned parser/model failure: reward is zero but the adapter keeps
        # trainable tokens and does not mark the sample for removal.
        FakeSample(group_id=1, reward=0.0, loss_mask=[1], remove_sample=False),
        FakeSample(group_id=2, reward=1.0),
    ]

    raw, rewards = post_process_rewards(_args(), samples)

    assert raw == [0.0, 1.0]
    assert rewards == [-1.0, 1.0]


def test_agent_timeout_fail_closes_stale_reward_but_keeps_negative_advantage() -> None:
    samples = [
        FakeSample(
            group_id=1,
            reward=1.0,
            status="TRUNCATED",
            loss_mask=[1],
            training_filter={
                "masked": False,
                "trainable": True,
                "reason": "agent_timeout",
            },
        ),
        FakeSample(group_id=2, reward=1.0),
    ]

    raw, rewards = post_process_rewards(_args(), samples)

    assert raw == [0.0, 1.0]
    assert rewards == [-1.0, 1.0]


@pytest.mark.parametrize(
    "malformed_reward",
    [float("nan"), float("inf"), float("-inf"), True, False, "not-a-number"],
)
def test_training_boundary_fail_closes_nonfinite_or_boolean_reward(
    malformed_reward: Any,
) -> None:
    samples = [
        FakeSample(group_id=1, reward=malformed_reward),
        FakeSample(group_id=2, reward=1.0),
    ]

    raw, rewards = post_process_rewards(_args(), samples)

    assert raw == [0.0, 1.0]
    assert rewards == [-1.0, 1.0]


def test_single_valid_trajectory_uses_zero_baseline() -> None:
    samples = [
        FakeSample(group_id=1, reward=2.0),
        FakeSample(group_id=1, reward=4.0),
    ]

    _, rewards = post_process_rewards(_args(), samples)

    assert rewards == [2.0, 4.0]


def test_fully_masked_trajectory_does_not_enter_baseline() -> None:
    samples = [
        FakeSample(group_id=1, reward=2.0, loss_mask=[0], remove_sample=True),
        FakeSample(group_id=2, reward=5.0),
    ]

    _, rewards = post_process_rewards(_args(), samples)

    assert rewards == [0.0, 5.0]


def test_removed_trace_with_valid_sibling_keeps_zero_advantage() -> None:
    samples = [
        # The first three traces belong to one trajectory.  Both an explicitly
        # removed trace and a fully-masked trace must stay at zero while their
        # valid sibling participates normally.
        FakeSample(group_id=1, reward=1.0, loss_mask=[0], remove_sample=True),
        FakeSample(group_id=1, reward=1.0, loss_mask=[0]),
        FakeSample(group_id=1, reward=1.0),
        FakeSample(group_id=2, reward=1.0),
        FakeSample(group_id=3, reward=0.0),
    ]

    raw, rewards = post_process_rewards(_production_grpo_args(), samples)

    scale = statistics.stdev([1.0, 1.0, 0.0]) + 1e-6
    assert raw == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert rewards == pytest.approx(
        [
            0.0,
            0.0,
            (1.0 - 0.5) / scale,
            (1.0 - 0.5) / scale,
            (0.0 - 1.0) / scale,
        ]
    )


def test_disabled_normalization_returns_raw_rewards() -> None:
    samples = [
        FakeSample(group_id=1, reward=2.0),
        FakeSample(group_id=2, reward=5.0),
    ]

    raw, rewards = post_process_rewards(_args(rewards_normalization=False), samples)

    assert raw == [2.0, 5.0]
    assert rewards == [2.0, 5.0]
