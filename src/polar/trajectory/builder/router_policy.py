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
from polar.trajectory.models import CompletionSession, Trajectory


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
    """

    def __init__(
        self,
        *,
        trusted_roles: Iterable[str] = ("router_policy",),
        end_of_turn_token_id: int | None = None,
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
