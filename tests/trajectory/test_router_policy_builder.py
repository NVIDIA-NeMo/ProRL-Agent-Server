from __future__ import annotations

from collections import UserDict

import pytest

from polar.trajectory.builder.router_policy import RouterPolicyBuilder
from polar.trajectory.models import CompletionRecord, CompletionSession


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> UserDict[str, list[int]]:
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        return UserDict({"input_ids": [1, 2, 3]})


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
async def test_router_policy_annotates_realized_routing_actions() -> None:
    def controller_turn(
        completion_id: str,
        worker: str,
        prompt_ids: list[int],
        response_ids: list[int],
    ) -> CompletionRecord:
        record = _completion(
            completion_id,
            role="router_policy",
            prompt_ids=prompt_ids,
            response_ids=response_ids,
        )
        record.metadata["controller_worker_before"] = worker
        return record

    # Distinct, non-prefix prompts keep every turn its own trace, in order.
    session = CompletionSession(
        session_id="session-actions",
        completions=[
            controller_turn("t0", "small", [1, 2], [10]),
            controller_turn("t1", "large", [3, 4], [11]),
            controller_turn("t2", "small", [5, 6], [12]),
            controller_turn("t3", "small", [7, 8], [13]),
        ],
    )

    trajectory = await RouterPolicyBuilder().build(session)

    actions = [trace.metadata.get("controller_actual_action") for trace in trajectory.traces]
    assert actions == ["escalate", "deescalate", "keep", "keep"]


@pytest.mark.asyncio
async def test_router_policy_leaves_actions_unset_without_workers() -> None:
    session = CompletionSession(
        session_id="session-no-workers",
        completions=[
            _completion("t0", role="router_policy", prompt_ids=[1, 2], response_ids=[10]),
        ],
    )

    trajectory = await RouterPolicyBuilder().build(session)

    assert "controller_actual_action" not in trajectory.traces[0].metadata


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


@pytest.mark.asyncio
async def test_router_policy_reconstructs_missing_sglang_token_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "polar.trajectory.builder.prefix_merging._load_tokenizer",
        lambda *_args, **_kwargs: _Tokenizer(),
    )
    completion = CompletionRecord(
        completion_id="router",
        request={
            "messages": [{"role": "user", "content": "route"}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        response={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "{}"},
                    "finish_reason": "length",
                    "logprobs": {
                        "content": [
                            {"token": "{", "logprob": -0.1},
                            {"token": "}", "logprob": -0.2},
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
        metadata={"completion_role": "router_policy"},
    )

    trajectory = await RouterPolicyBuilder(
        tokenizer_name_or_path="/model",
    ).build(CompletionSession(session_id="session-4", completions=[completion]))

    trace = trajectory.traces[0]
    assert trace.prompt_ids == [1, 2, 3]
    assert trace.response_ids == [ord("{"), ord("}")]
    assert trace.response_logprobs == [-0.1, -0.2]
    assert trace.loss_mask == [1, 1]


@pytest.mark.asyncio
async def test_router_policy_reconstruction_fails_closed_on_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MismatchedTokenizer(_Tokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return [1]

    monkeypatch.setattr(
        "polar.trajectory.builder.prefix_merging._load_tokenizer",
        lambda *_args, **_kwargs: MismatchedTokenizer(),
    )
    completion = CompletionRecord(
        completion_id="router",
        request={"messages": [{"role": "user", "content": "route"}]},
        response={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "{}"},
                    "finish_reason": "length",
                    "logprobs": {
                        "content": [
                            {"token": "{", "logprob": -0.1},
                            {"token": "}", "logprob": -0.2},
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
        metadata={"completion_role": "router_policy"},
    )

    trajectory = await RouterPolicyBuilder(
        tokenizer_name_or_path="/model",
    ).build(CompletionSession(session_id="session-5", completions=[completion]))

    assert trajectory.traces[0].response_ids == []
    assert trajectory.traces[0].loss_mask == []
