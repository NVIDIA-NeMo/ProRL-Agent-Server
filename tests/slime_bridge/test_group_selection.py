from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from slime_bridge.group_selection import select_training_groups


def _sample(group_index: int, traj_id: int, accuracy: float, action: str | None) -> Any:
    trace_metadata = {"controller_actual_action": action} if action is not None else {}
    return SimpleNamespace(
        group_index=group_index,
        rollout_id=traj_id,
        index=traj_id,
        loss_mask=[1],
        response_length=1,
        remove_sample=False,
        metadata={"polar": {"trace_metadata": trace_metadata}},
        get_reward_value=lambda _args, a=accuracy: a,
    )


def _group(group_index: int, accuracies: list[float], actions: list[str | None] | None = None):
    actions = actions or [None] * len(accuracies)
    return [
        _sample(group_index, traj_id, accuracy, action)
        for traj_id, (accuracy, action) in enumerate(zip(accuracies, actions))
    ]


def _args(**overrides: Any) -> Any:
    defaults = {
        "reward_key": "score",
        "polar_drop_all_wrong_groups": False,
        "polar_drop_all_keep_groups": False,
        "polar_balance_all_correct_groups": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _masked(group: list[Any]) -> bool:
    """A group excluded from training is fully masked via remove_sample."""
    return all(sample.remove_sample for sample in group)


def _num_trainable(groups: list[list[Any]]) -> int:
    return sum(1 for group in groups if not _masked(group))


def test_all_knobs_off_is_passthrough() -> None:
    groups = [_group(0, [1.0, 0.0])]
    selected, metrics = select_training_groups(_args(), groups)
    assert selected is groups
    assert metrics == {}
    assert _num_trainable(selected) == 1


def test_drop_all_wrong_groups() -> None:
    groups = [
        _group(0, [1.0, 0.0]),  # mixed
        _group(1, [0.0, 0.0]),  # all-wrong
        _group(2, [1.0, 1.0]),  # all-correct
    ]
    selected, metrics = select_training_groups(_args(polar_drop_all_wrong_groups=True), groups)
    # Batch size is preserved; the all-wrong group is masked, not dropped.
    assert selected is groups
    assert _num_trainable(selected) == 2
    assert _masked(groups[1]) and not _masked(groups[0]) and not _masked(groups[2])
    assert metrics["polar/group_selection/dropped_all_wrong"] == 1.0
    assert metrics["polar/group_selection/masked"] == 1.0


def test_drop_all_keep_groups() -> None:
    groups = [
        _group(0, [1.0, 0.0], actions=["keep", "keep"]),
        _group(1, [1.0, 0.0], actions=["escalate", "keep"]),
    ]
    selected, metrics = select_training_groups(_args(polar_drop_all_keep_groups=True), groups)
    assert selected is groups
    assert _num_trainable(selected) == 1
    assert _masked(groups[0]) and not _masked(groups[1])
    assert metrics["polar/group_selection/dropped_all_keep"] == 1.0


def test_balance_all_correct_subsamples_to_mixed_count() -> None:
    groups = [_group(i, [1.0, 1.0]) for i in range(5)] + [
        _group(10, [1.0, 0.0]),
        _group(11, [1.0, 0.0]),
    ]
    selected, metrics = select_training_groups(
        _args(polar_balance_all_correct_groups=True), groups, rollout_id=0
    )
    assert selected is groups
    assert _num_trainable(selected) == 4  # 2 mixed + 2 sampled all-correct
    assert metrics["polar/group_selection/subsampled_all_correct"] == 3.0
    assert metrics["polar/group_selection/masked"] == 3.0


def test_balance_is_noop_when_all_correct_not_dominant() -> None:
    groups = [_group(0, [1.0, 1.0]), _group(1, [1.0, 0.0]), _group(2, [1.0, 0.0])]
    selected, _metrics = select_training_groups(
        _args(polar_balance_all_correct_groups=True), groups, rollout_id=0
    )
    assert selected is groups
    assert _num_trainable(selected) == 3  # nothing masked


def test_empty_batch_falls_back_to_original() -> None:
    groups = [_group(0, [0.0, 0.0]), _group(1, [0.0, 0.0])]  # both all-wrong
    selected, metrics = select_training_groups(_args(polar_drop_all_wrong_groups=True), groups)
    assert selected == groups
    assert _num_trainable(selected) == 2  # fallback keeps everything trainable
    assert metrics["polar/group_selection/empty_fallback"] == 1.0


def test_full_pipeline_drop_then_balance() -> None:
    groups = (
        [_group(0, [0.0, 0.0])]  # all-wrong -> masked by D
        + [_group(1, [1.0, 0.0], actions=["keep", "keep"])]  # mixed, no switch -> masked by B
        + [_group(2, [1.0, 0.0], actions=["escalate", "keep"])]  # mixed, switch -> kept
        + [_group(3 + i, [1.0, 1.0]) for i in range(4)]  # 4 all-correct
    )
    selected, _metrics = select_training_groups(
        _args(
            polar_drop_all_wrong_groups=True,
            polar_drop_all_keep_groups=True,
            polar_balance_all_correct_groups=True,
        ),
        groups,
        rollout_id=0,
    )
    # D masks group0, B masks group1 -> 1 mixed + 4 all-correct; C downsamples
    # all-correct to the mixed count (1) -> 1 mixed + 1 all-correct trainable.
    assert selected is groups  # batch size preserved
    assert _num_trainable(selected) == 2
