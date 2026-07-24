"""Trajectory builder for SPilot router-policy completions.

The gateway can serve both the trainable router and frozen pool models during
one rollout.  Only completions that the gateway classifies from the reserved
``router/policy`` request alias may contribute loss-bearing tokens. Filtering before
prefix reconstruction is defense in depth: pool responses are excluded even
if a future gateway change accidentally persists them in the session store.
The gateway separately authenticates the reserved Router alias with a
host-issued, Router-scoped capability; the metadata check here remains a
second provenance boundary at training-data construction time.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from polar.trajectory.builder.base import BaseTrajectoryBuilder
from polar.trajectory.builder.prefix_merging import PrefixMergingBuilder
from polar.trajectory.models import CompletionSession, Trace, Trajectory


class RouterPolicyBuilder(BaseTrajectoryBuilder):
    """Filter to trusted router completions, then prefix-merge their turns.

    Parameters
    ----------
    trusted_roles:
        Exact values accepted from ``CompletionRecord.metadata.completion_role``.
        The secure default accepts only the provenance value written by the
        SPilot gateway.  This is configurable for migrations and tests, but an
        empty allowlist is rejected.
    end_of_turn_token_id:
        Optional explicit EOT id forwarded to :class:`PrefixMergingBuilder`.
    tokenizer_name_or_path:
        Optional tokenizer used to strictly reconstruct token IDs when an older
        SGLang OpenAI endpoint returns token strings and logprobs without IDs.
    """

    def __init__(
        self,
        *,
        trusted_roles: Iterable[str] = ("router_policy",),
        end_of_turn_token_id: int | None = None,
        tokenizer_name_or_path: str | None = None,
    ) -> None:
        if isinstance(trusted_roles, str):
            trusted_roles = (trusted_roles,)
        normalized = frozenset(
            role.strip()
            for role in trusted_roles
            if isinstance(role, str) and role.strip()
        )
        if not normalized:
            raise ValueError("router_policy trusted_roles must not be empty")
        self._trusted_roles = normalized
        self._delegate = PrefixMergingBuilder(
            end_of_turn_token_id=end_of_turn_token_id,
            tokenizer_name_or_path=tokenizer_name_or_path,
        )

    async def build(self, session: CompletionSession) -> Trajectory:
        role_counts: Counter[str] = Counter()
        trusted = []
        for completion in session.completions:
            role = completion.metadata.get("completion_role")
            role_label = role if isinstance(role, str) else "<missing>"
            role_counts[role_label] += 1
            if role in self._trusted_roles:
                trusted.append(completion)

        filtered_session = session.model_copy(
            update={
                "completion_count": len(trusted),
                "completions": trusted,
            }
        )
        trajectory = await self._delegate.build(filtered_session)
        _annotate_controller_actions(trajectory.traces)
        metadata = dict(trajectory.metadata)
        metadata.update(
            {
                "builder": "router_policy",
                "raw_record_count": len(session.completions),
                "record_count": len(trusted),
                "excluded_record_count": len(session.completions) - len(trusted),
                "completion_role_counts": dict(sorted(role_counts.items())),
                "trusted_completion_roles": sorted(self._trusted_roles),
            }
        )
        error = trajectory.error
        if not trusted:
            error = "no trusted router-policy completions"
        return trajectory.model_copy(update={"metadata": metadata, "error": error})


def _annotate_controller_actions(traces: list[Trace]) -> None:
    """Stamp each controller trace with its realized routing action.

    ``controller_worker_before`` (set by the gateway from the controller's
    request) is the worker active when the controller was consulted for a turn.
    A switch confirmed by that turn's decision appears as a different worker on
    the next trace, so the realized action of turn ``i`` is read from the
    transition into turn ``i + 1``. The final turn has no successor and is
    recorded as a no-switch ``keep``. Traces are left untouched when none carry
    a worker, so the annotation is inert for non-controller trajectories.
    """
    workers = [
        trace.metadata.get("controller_worker_before")
        if isinstance(trace.metadata, dict)
        else None
        for trace in traces
    ]
    if not any(worker in ("small", "large") for worker in workers):
        return
    for position, trace in enumerate(traces):
        current = workers[position]
        if current not in ("small", "large") or not isinstance(trace.metadata, dict):
            continue
        following = workers[position + 1] if position + 1 < len(workers) else None
        if current == "small" and following == "large":
            action = "escalate"
        elif current == "large" and following == "small":
            action = "deescalate"
        else:
            action = "keep"
        trace.metadata["controller_actual_action"] = action
