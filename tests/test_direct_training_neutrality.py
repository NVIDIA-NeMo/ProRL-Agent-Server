"""Regression guard: router-era changes must not alter direct TMax training.

These checks freeze the load-bearing facts from the 2026-07-21 direct-training
neutrality audit so a future edit that silently perturbs the direct
(fully_async, vanillux2, no router/pool) path fails here instead of in a run.

Run via ``scripts/verify_direct_training_neutral.sh`` (which also probes the
real portable runtime), or directly: ``pytest tests/test_direct_training_neutrality.py``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from slime_bridge.reward_post_process import post_process_rewards

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_task_template(config_relpath: str) -> dict:
    """Load a launcher polar_config's task template with placeholders neutralized."""
    text = (_EXAMPLES / config_relpath).read_text()
    # The two runtime-volume placeholders occupy bare lines (rendered to real
    # volume entries or empty at submit time); drop them. Every other ${VAR}
    # sits inside a quoted string, so replacing it with a scalar is harmless
    # and never touches the literal strategy fields under test.
    text = text.replace("${POLAR_AGENT_RUNTIME_VOLUME}", "")
    text = text.replace("${POLAR_INTERNET_RUNTIME_VOLUME}", "")
    text = re.sub(r"\$\{[^}]+\}", "x", text)
    return yaml.safe_load(text)["polar_task_template"]


def _sample(group_index: int, rollout_id: int, reward: object) -> SimpleNamespace:
    return SimpleNamespace(
        group_index=group_index,
        rollout_id=rollout_id,
        index=rollout_id,
        reward={"score": reward},
        status="COMPLETED",
        loss_mask=[1],
        response_length=1,
        remove_sample=False,
        metadata={"polar": {}},
        get_reward_value=lambda args, r=reward: r,
    )


def _grpo_args(**overrides) -> SimpleNamespace:
    base = {
        "reward_key": "score",
        "rewards_normalization": True,
        "advantage_estimator": "grpo",
        "grpo_std_normalization": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# One prompt group, three trajectories (rollout_ids 10/11/12) of two traces each.
# Exercises the leave-one-trajectory-out baseline and per-trace broadcast.
_TRAJECTORIES = [(10, 1.0), (11, 0.0), (12, 0.5)]


def _group_samples() -> list[SimpleNamespace]:
    samples: list[SimpleNamespace] = []
    for rollout_id, reward in _TRAJECTORIES:
        samples.append(_sample(0, rollout_id, reward))
        samples.append(_sample(0, rollout_id, reward))
    return samples


def test_loo_advantages_match_frozen_golden_mainline_config() -> None:
    # GRPO_STD_NORMALIZATION=0 is the mainline direct-training config. Any
    # change to the leave-one-out baseline / broadcast math trips this.
    raw, adv = post_process_rewards(_grpo_args(), _group_samples())
    assert raw == [1.0, 1.0, 0.0, 0.0, 0.5, 0.5]
    assert adv == [0.75, 0.75, -0.75, -0.75, 0.0, 0.0]


def test_loo_advantages_match_frozen_golden_with_std_normalization() -> None:
    _raw, adv = post_process_rewards(
        _grpo_args(grpo_std_normalization=True), _group_samples()
    )
    expected = [1.499997, 1.499997, -1.499997, -1.499997, 0.0, 0.0]
    assert all(math.isclose(a, e, abs_tol=1e-5) for a, e in zip(adv, expected)), adv


def test_reward_boundary_is_fail_closed_for_malformed_values() -> None:
    # Direct-path reward hardening: malformed rewards coerce to 0.0 rather than
    # propagating (float(True)=1.0 or NaN into advantages) — semantics for
    # well-formed numeric rewards are unchanged (covered by the golden tests).
    for bad in (float("nan"), float("inf"), True, "oops"):
        raw, _adv = post_process_rewards(
            _grpo_args(rewards_normalization=False),
            [_sample(0, 1, bad), _sample(0, 2, 1.0)],
        )
        assert raw[0] == 0.0, bad
        assert raw[1] == 1.0


def test_non_grpo_estimator_leaves_rewards_untouched() -> None:
    # The direct path must not gain advantage normalization it did not have.
    raw, adv = post_process_rewards(
        _grpo_args(advantage_estimator="ppo"), _group_samples()
    )
    assert raw == adv == [1.0, 1.0, 0.0, 0.0, 0.5, 0.5]


def test_direct_launcher_selects_non_router_strategies() -> None:
    # The load-bearing check: the DIRECT tmax launcher must actually SELECT the
    # non-router builder/evaluator, so router reward shaping (spilot_harbor) and
    # the router-policy trajectory builder cannot enter a direct run. A config
    # edit that pointed the direct path at the router strategies trips this.
    direct = _load_task_template("tmax_slime_grpo/polar_config.yaml")
    assert direct["builder"]["strategy"] == "prefix_merging"
    assert direct["evaluator"]["strategy"] == "harbor"
    assert direct["builder"]["strategy"] != "router_policy"
    assert direct["evaluator"]["strategy"] != "spilot_harbor"


def test_router_launcher_confines_router_strategies_to_the_router_config() -> None:
    # The router strategies are selected ONLY by the router launcher — proving
    # they are opt-in and confined, not a shared default the direct path shares.
    router = _load_task_template("spilot_router_slime_grpo/polar_config.yaml")
    assert router["builder"]["strategy"] == "router_policy"
    assert router["evaluator"]["strategy"] == "spilot_harbor"
    assert router["agent"]["harness"] == "spilot_router"
