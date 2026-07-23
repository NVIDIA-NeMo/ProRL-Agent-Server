from __future__ import annotations

from types import SimpleNamespace

from slime_bridge.dynamic_filters import _group_lacks_routing_switch


def _sample(action: str | None) -> SimpleNamespace:
    trace_metadata = {"controller_actual_action": action} if action is not None else {}
    return SimpleNamespace(metadata={"polar": {"trace_metadata": trace_metadata}})


def test_group_of_only_keeps_lacks_a_switch() -> None:
    assert _group_lacks_routing_switch([_sample("keep"), _sample("keep")]) is True


def test_group_with_an_escalate_has_a_switch() -> None:
    assert _group_lacks_routing_switch([_sample("keep"), _sample("escalate")]) is False


def test_group_with_a_deescalate_has_a_switch() -> None:
    assert _group_lacks_routing_switch([_sample("deescalate"), _sample("keep")]) is False


def test_group_without_stamped_actions_is_indeterminate() -> None:
    assert _group_lacks_routing_switch([_sample(None), _sample(None)]) is False
