from __future__ import annotations

import pytest

from polar.trajectory.builder.router_policy import RouterPolicyBuilder
from polar.trajectory.models import CompletionRecord, CompletionSession


def _completion(
    completion_id: str,
    *,
    role: str | None,
    prompt_ids: list[int],
    response_ids: list[int],
) -> CompletionRecord:
    metadata = {} if role is None else {"completion_role": role}
    return CompletionRecord(
        completion_id=completion_id,
        request={"messages": [{"role": "user", "content": completion_id}]},
        response={
            "choices": [
                {
                    "prompt_token_ids": prompt_ids,
                    "token_ids": response_ids,
                    "message": {"role": "assistant", "content": completion_id},
                    "finish_reason": "length",
                    "logprobs": {
                        "content": [
                            {"token_id": token_id, "logprob": -0.1}
                            for token_id in response_ids
                        ]
                    },
                }
            ]
        },
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_router_policy_excludes_pool_and_unlabelled_completions() -> None:
    session = CompletionSession(
        session_id="session-1",
        completions=[
            _completion(
                "router",
                role="router_policy",
                prompt_ids=[1, 2],
                response_ids=[10, 11],
            ),
            _completion(
                "pool",
                role="model_pool",
                prompt_ids=[3, 4],
                response_ids=[20, 21],
            ),
            _completion(
                "legacy",
                role=None,
                prompt_ids=[5, 6],
                response_ids=[30],
            ),
        ],
    )

    trajectory = await RouterPolicyBuilder().build(session)

    assert trajectory.status == "COMPLETED"
    assert trajectory.error is None
    assert len(trajectory.traces) == 1
    assert trajectory.traces[0].response_ids == [10, 11]
    assert trajectory.traces[0].loss_mask == [1, 1]
    assert trajectory.metadata["record_count"] == 1
    assert trajectory.metadata["raw_record_count"] == 3
    assert trajectory.metadata["excluded_record_count"] == 2
    assert trajectory.metadata["completion_role_counts"] == {
        "<missing>": 1,
        "model_pool": 1,
        "router_policy": 1,
    }


@pytest.mark.asyncio
async def test_router_policy_fails_closed_without_trusted_completions() -> None:
    session = CompletionSession(
        session_id="session-2",
        completions=[
            _completion(
                "pool",
                role="model_pool",
                prompt_ids=[1],
                response_ids=[20],
            )
        ],
    )

    trajectory = await RouterPolicyBuilder().build(session)

    assert trajectory.status == "ERROR"
    assert trajectory.error == "no trusted router-policy completions"
    assert trajectory.traces == []
    assert trajectory.metadata["excluded_record_count"] == 1


@pytest.mark.asyncio
async def test_router_policy_allows_explicit_migration_role() -> None:
    session = CompletionSession(
        session_id="session-3",
        completions=[
            _completion(
                "migration",
                role="policy_v2",
                prompt_ids=[1],
                response_ids=[12],
            )
        ],
    )

    trajectory = await RouterPolicyBuilder(trusted_roles=["policy_v2"]).build(session)

    assert trajectory.status == "COMPLETED"
    assert trajectory.traces[0].response_ids == [12]


def test_router_policy_rejects_empty_role_allowlist() -> None:
    with pytest.raises(ValueError, match="trusted_roles must not be empty"):
        RouterPolicyBuilder(trusted_roles=[])
