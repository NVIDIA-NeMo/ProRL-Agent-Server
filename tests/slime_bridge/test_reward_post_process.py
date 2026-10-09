from __future__ import annotations

from types import SimpleNamespace

from slime_bridge.reward_post_process import _trajectory_key, post_process_rewards


class FakeSample:
    def __init__(
        self,
        *,
        group_index: int = 0,
        rollout_id: int,
        reward: float,
        status: str = "COMPLETED",
        loss_mask: list[int] | None = None,
        remove_sample: bool = False,
    ) -> None:
        self.group_index = group_index
        self.rollout_id = rollout_id
        self.index = rollout_id
        self.reward = {"score": reward}
        self.status = status
        self.loss_mask = [1] if loss_mask is None else loss_mask
        self.response_length = len(self.loss_mask)
        self.remove_sample = remove_sample

    def get_reward_value(self, args) -> float:
        return float(self.reward[args.reward_key])


def _args(**overrides):
    defaults = {
        "reward_key": "score",
        "rewards_normalization": True,
        "advantage_estimator": "grpo",
        "grpo_std_normalization": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_trajectory_key_supports_slime_v030_group_id() -> None:
    sample = SimpleNamespace(group_index=2, group_id=7, index=99)

    assert _trajectory_key(sample, 0) == (2, (2, 7))


def test_dynamic_trace_loo_keeps_per_trace_rewards() -> None:
    samples = [
        FakeSample(rollout_id=10, reward=2.0),
        FakeSample(rollout_id=10, reward=4.0),
        FakeSample(rollout_id=20, reward=10.0),
    ]

    raw, rewards = post_process_rewards(_args(), samples)

    assert raw == [2.0, 4.0, 10.0]
    assert rewards == [-8.0, -6.0, 7.0]


def test_failed_trajectory_is_excluded_from_other_baselines() -> None:
    samples = [
        FakeSample(rollout_id=1, reward=2.0),
        FakeSample(rollout_id=2, reward=10.0, status="FAILED"),
        FakeSample(rollout_id=3, reward=6.0),
    ]

    _, rewards = post_process_rewards(_args(), samples)

    assert rewards == [-4.0, 0.0, 4.0]


def test_single_valid_trajectory_uses_zero_baseline() -> None:
    samples = [
        FakeSample(rollout_id=1, reward=2.0),
        FakeSample(rollout_id=1, reward=4.0),
    ]

    _, rewards = post_process_rewards(_args(), samples)

    assert rewards == [2.0, 4.0]


def test_fully_masked_trajectory_does_not_enter_baseline() -> None:
    samples = [
        FakeSample(rollout_id=1, reward=2.0, loss_mask=[0], remove_sample=True),
        FakeSample(rollout_id=2, reward=5.0),
    ]

    _, rewards = post_process_rewards(_args(), samples)

    assert rewards == [0.0, 5.0]


def test_disabled_normalization_returns_raw_rewards() -> None:
    samples = [
        FakeSample(rollout_id=1, reward=2.0),
        FakeSample(rollout_id=2, reward=5.0),
    ]

    raw, rewards = post_process_rewards(_args(rewards_normalization=False), samples)

    assert raw == [2.0, 5.0]
    assert rewards == [2.0, 5.0]
