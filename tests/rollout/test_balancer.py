from __future__ import annotations

from datetime import datetime, timedelta, timezone

from polar.rollout.balancer import NodeScheduler
from polar.rollout.models import NodeRegistrationRequest, NodeStageMetrics


def _register(
    scheduler: NodeScheduler,
    node_id: str,
    *,
    max_init_workers: int = 4,
    max_run_workers: int = 2,
    max_postrun_workers: int = 2,
) -> None:
    scheduler.register_node(
        NodeRegistrationRequest(
            node_id=node_id,
            gateway_url=f"http://127.0.0.1/{node_id}",
            max_init_workers=max_init_workers,
            max_run_workers=max_run_workers,
            max_postrun_workers=max_postrun_workers,
            heartbeat_interval_seconds=30,
        )
    )


def test_acquire_prefers_lower_run_pressure_before_init_pressure() -> None:
    scheduler = NodeScheduler()
    _register(scheduler, "busy-run", max_run_workers=2)
    _register(scheduler, "free-run", max_run_workers=2)
    scheduler.heartbeat(
        "busy-run",
        metrics=NodeStageMetrics(run_inflight=1, init_queue_depth=0),
    )
    scheduler.heartbeat(
        "free-run",
        metrics=NodeStageMetrics(run_inflight=0, init_queue_depth=3),
    )

    selected = scheduler.acquire_node()

    assert selected is not None
    assert selected.node_id == "free-run"
    assert selected.dispatch_reservations == 1


def test_release_reservation_decrements_dispatch_pressure() -> None:
    scheduler = NodeScheduler()
    _register(scheduler, "node-a")

    assert scheduler.acquire_node().dispatch_reservations == 1
    assert scheduler.release_reservation("node-a").dispatch_reservations == 0


def test_draining_node_does_not_receive_new_sessions() -> None:
    scheduler = NodeScheduler()
    _register(scheduler, "node-a")
    _register(scheduler, "node-b")

    scheduler.acquire_node()
    drained = scheduler.drain_node("node-a")
    selected = scheduler.acquire_node()

    assert drained.draining is True
    assert selected is not None
    assert selected.node_id == "node-b"


def test_stale_nodes_become_ineligible() -> None:
    scheduler = NodeScheduler(stale_factor=2.5)
    _register(scheduler, "node-a")
    scheduler._nodes["node-a"].last_heartbeat = datetime.now(timezone.utc) - timedelta(minutes=10)

    assert scheduler.acquire_node() is None
    assert scheduler.get_node("node-a").healthy is False


def test_postrun_backlog_blocks_admission() -> None:
    scheduler = NodeScheduler()
    _register(scheduler, "node-a", max_postrun_workers=2)
    scheduler.heartbeat(
        "node-a",
        metrics=NodeStageMetrics(postrun_queue_depth=4),
    )

    assert scheduler.acquire_node() is None


def test_four_gateway_fleet_balances_initial_dispatch_reservations() -> None:
    scheduler = NodeScheduler()
    for index in range(4):
        _register(
            scheduler,
            f"slurm-rank-{index}",
            max_init_workers=12,
            max_run_workers=96,
            max_postrun_workers=24,
        )

    selected = [scheduler.acquire_node() for _ in range(40)]

    assert all(node is not None for node in selected)
    counts = {
        node_id: sum(node is not None and node.node_id == node_id for node in selected)
        for node_id in (f"slurm-rank-{index}" for index in range(4))
    }
    assert counts == {
        "slurm-rank-0": 10,
        "slurm-rank-1": 10,
        "slurm-rank-2": 10,
        "slurm-rank-3": 10,
    }
