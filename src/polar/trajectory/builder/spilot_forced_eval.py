"""Non-trainable trajectory builder for forced-route evaluation sessions."""

from __future__ import annotations

from polar.agent.presets.spilot_forced_route_eval_runner import EVAL_ONLY_ACK
from polar.trajectory.builder.base import BaseTrajectoryBuilder
from polar.trajectory.models import CompletionSession, Trajectory


class SpilotForcedEvalBuilder(BaseTrajectoryBuilder):
    """Mark an eval session completed while deliberately emitting no traces.

    Forced candidate calls exercise the mutable task workspace and Harbor
    evaluator, but they must never become actor training data.  Construction
    therefore requires an explicit acknowledgement, and the resulting
    trajectory always has an empty trace list.
    """

    def __init__(self, *, acknowledgement: str) -> None:
        if acknowledgement != EVAL_ONLY_ACK:
            raise ValueError("spilot forced-eval builder acknowledgement is missing")

    async def build(self, session: CompletionSession) -> Trajectory:
        return Trajectory(
            status="COMPLETED",
            metadata={
                "builder": "spilot_forced_eval",
                "eval_only": True,
                "trainable": False,
                "session_id": session.session_id,
                "task_id": session.task_id,
                "task_metadata": dict(session.metadata),
                "raw_record_count": len(session.completions),
                "record_count": 0,
                "trace_count": 0,
                **{
                    key: session.metadata[key]
                    for key in ("group_id", "policy_version", "rollout_step")
                    if key in session.metadata
                },
            },
            traces=[],
        )


__all__ = ["SpilotForcedEvalBuilder"]
