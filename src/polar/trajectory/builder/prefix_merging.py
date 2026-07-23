"""Prefix-merging trajectory builder.

Reconstructs a single token-level training trace out of the many independent
LLM completions an agent emits during one rollout.  A harness (claude_code,
codex, pi, ...) drives the agent and each turn hits the gateway as a separate
completion request; this builder stitches those completions back into the
``prompt + response_1 + interstitial + response_2 + ...`` stream that an RL
trainer needs, without introducing tokenization drift.

Design in two stages:

1. **Grouping** — route each completion to the chain it append-extends, tested
   purely on tokens: a completion joins the chain whose last prompt is a prefix
   of it (``C_k.prompt_ids`` is a prefix of ``C_{k+1}.prompt_ids``).  This routes
   correctly even when parallel agents / sub-agents interleave (each has a
   distinct prompt prefix), and is robust to BPE re-tokenization because it
   compares only server-tokenized prompts, whose shared prefix is stable across
   the special-token generation-prompt boundary.  We never compare the *sampled*
   ``response_ids`` (those can re-tokenize in the next prompt, e.g.
   ``[fish, ing]`` → ``[fishing]``); a completion that extends no open chain
   starts a fresh one.

2. **Finalization** — walk each chain and build a merged token stream:

   - Assistant bodies come from the **raw** ``response_ids`` actually sampled
     by the model.  Their logprobs are real and we never decode→re-encode,
     so BPE non-canonicality cannot bite.
   - Interstitials (tool results, chat-template glue, intermediate user
     turns) come from ``C_{i+1}.prompt_ids`` — the server's **canonical**
     tokenization.  The boundary between "canonical copy of the previous
     assistant body" and the actual interstitial is the first end-of-turn
     token (``<|im_end|>`` on Qwen / ChatML; auto-detected or configurable).
   - Interstitial slots get synthesized logprobs and a zero ``loss_mask``;
     sampled assistant slots keep their real logprobs and a one ``loss_mask``.

"""

from __future__ import annotations

import logging
from copy import deepcopy
from functools import lru_cache
from typing import Any

from polar.trajectory.builder.base import BaseTrajectoryBuilder
from polar.trajectory.builder.record_utils import build_trace_from_completion
from polar.trajectory.models import CompletionRecord, CompletionSession, Trace, Trajectory

logger = logging.getLogger(__name__)

# finish_reasons where the model emitted the natural end-of-turn token itself.
_NATURAL_STOP_REASONS = frozenset({"stop", "tool_calls", "stop_sequence"})


@lru_cache(maxsize=4)
def _load_tokenizer(name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=True,
    )


class PrefixMergingBuilder(BaseTrajectoryBuilder):
    """Rebuild a chain's merged token stream using raw + canonical-interstitial.

    Parameters
    ----------
    end_of_turn_token_id:
        Explicit end-of-turn (EOT) token id used to locate the
        canonical-tail split between the prior assistant body and the
        interstitial.  When None (default), the builder auto-detects it
        from the last token of the first completion with a natural stop
        reason.  For Qwen / ChatML templates this is the
        ``<|im_end|>`` token id.
    """

    def __init__(
        self,
        *,
        end_of_turn_token_id: int | None = None,
        tokenizer_name_or_path: str | None = None,
    ) -> None:
        self._configured_eot_id = end_of_turn_token_id
        self._tokenizer = None
        if tokenizer_name_or_path:
            self._tokenizer = _load_tokenizer(tokenizer_name_or_path)

    async def build(self, session: CompletionSession) -> Trajectory:
        if not session.completions:
            return Trajectory(
                status="ERROR",
                metadata={
                    "builder": "prefix_merging",
                    "session_id": session.session_id,
                    "task_metadata": dict(session.metadata),
                    "record_count": 0,
                    **_top_level_scheduler_metadata(session.metadata),
                },
                traces=[],
                error="no completions",
            )

        chains: list[list[CompletionRecord]] = []
        chain_tips: list[list[int]] = []  # last completion's prompt_ids, per chain

        for completion in session.completions:
            prompt_ids = build_trace_from_completion(
                completion,
                tokenizer=self._tokenizer,
            ).prompt_ids
            chain_idx = self._find_extendable_chain(prompt_ids, chain_tips)
            if chain_idx is None:
                chain_idx = len(chains)
                chains.append([])
                chain_tips.append([])
            chains[chain_idx].append(completion)
            chain_tips[chain_idx] = prompt_ids

        stats: dict[str, int] = {
            "chains_total": len(chains),
            "chains_reconstructed_full": 0,
            "chains_reconstructed_truncated": 0,
            "completions_total": len(session.completions),
            "completions_merged": 0,
        }
        final_traces = [self._finalize_chain(chain, stats) for chain in chains]

        return Trajectory(
            status="COMPLETED",
            metadata={
                "builder": "prefix_merging",
                "session_id": session.session_id,
                "task_id": session.task_id,
                "api_type": session.api_type,
                "model_requested": session.model_requested,
                "model_used": session.model_used,
                "record_count": len(session.completions),
                "task_metadata": dict(session.metadata),
                "trace_count": len(chains),
                "reconstruction_stats": stats,
                **_top_level_scheduler_metadata(session.metadata),
            },
            traces=final_traces,
        )

    # ------------------------------------------------------------------
    # Chain finalization
    # ------------------------------------------------------------------

    def _finalize_chain(
        self,
        chain: list[CompletionRecord],
        stats: dict[str, int],
    ) -> Trace:
        # Everything in C_1.prompt_ids is the non-trainable
        # prompt; C_1.response_ids plus every subsequent raw response +
        # canonical interstitial becomes the trainable response.  No role-shape
        # constraint on the initial conversation — a harness preamble like
        # codex's [system, user, user, assistant, tool, ...] is treated as
        # static context.
        first_trace = build_trace_from_completion(
            chain[0],
            tokenizer=self._tokenizer,
        )
        eot_id = self._resolve_eot_id(chain)

        prompt_ids = list(first_trace.prompt_ids)
        stream_ids: list[int] = list(prompt_ids)
        response_slots: list[float | None] = []
        loss_mask: list[int] = []
        response_messages: list[dict[str, Any]] = []

        # Track the canonical prompt_ids of the most recently merged
        # completion — used for the canonical-vs-canonical prefix check.
        prev_prompt_ids: list[int] = list(first_trace.prompt_ids)
        prev_raw_response: list[int] = list(first_trace.response_ids)

        # Running count of messages consumed = prompt_messages + all response_messages emitted.
        msg_acc = len(first_trace.prompt_messages)

        self._append_response_tokens(first_trace, stream_ids, response_slots, loss_mask)
        response_messages.extend(deepcopy(m) for m in first_trace.response_messages)
        msg_acc += len(first_trace.response_messages)
        kept = 1

        for i in range(1, len(chain)):
            Ci_trace = build_trace_from_completion(
                chain[i],
                tokenizer=self._tokenizer,
            )
            Ci_prompt_ids = list(Ci_trace.prompt_ids)

            # Canonical-vs-canonical prefix check: both sides are server-side
            # tokenizations of the same message prefix — matches reliably
            # unless the harness rewrote prior messages.
            if (
                len(Ci_prompt_ids) < len(prev_prompt_ids)
                or Ci_prompt_ids[: len(prev_prompt_ids)] != prev_prompt_ids
            ):
                logger.debug(
                    "prefix_merging: canonical prefix break at step %d/%d",
                    i,
                    len(chain),
                )
                break

            # canonical_tail = canonical tokens for [prev assistant msg + new interstitials].
            canonical_tail = Ci_prompt_ids[len(prev_prompt_ids):]
            interstitial = self._slice_interstitial(
                canonical_tail=canonical_tail,
                prev_raw_response=prev_raw_response,
                eot_id=eot_id,
            )
            if interstitial is None:
                logger.debug(
                    "prefix_merging: interstitial split failed at step %d/%d "
                    "(eot_id=%r, tail_len=%d)",
                    i,
                    len(chain),
                    eot_id,
                    len(canonical_tail),
                )
                break

            if interstitial:
                stream_ids.extend(interstitial)
                response_slots.extend([None] * len(interstitial))
                loss_mask.extend([0] * len(interstitial))

            # Message-level interstitial bookkeeping.
            if len(Ci_trace.prompt_messages) > msg_acc:
                interstitial_msgs = Ci_trace.prompt_messages[msg_acc:]
                response_messages.extend(deepcopy(m) for m in interstitial_msgs)
                msg_acc += len(interstitial_msgs)

            self._append_response_tokens(Ci_trace, stream_ids, response_slots, loss_mask)
            response_messages.extend(deepcopy(m) for m in Ci_trace.response_messages)
            msg_acc += len(Ci_trace.response_messages)

            prev_prompt_ids = Ci_prompt_ids
            prev_raw_response = list(Ci_trace.response_ids)
            kept += 1

        stats["completions_merged"] += kept
        if kept == len(chain):
            stats["chains_reconstructed_full"] += 1
        else:
            stats["chains_reconstructed_truncated"] += 1

        response_ids = stream_ids[len(prompt_ids):]
        response_logprobs = self._finalize_logprobs(response_slots, loss_mask)
        last_kept_trace = build_trace_from_completion(
            chain[kept - 1],
            tokenizer=self._tokenizer,
        )

        return Trace(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            loss_mask=loss_mask,
            prompt_messages=[deepcopy(m) for m in first_trace.prompt_messages],
            response_messages=response_messages,
            tools=deepcopy(first_trace.tools),
            finish_reason=last_kept_trace.finish_reason,
            response_logprobs=response_logprobs,
            metadata=self._chain_metadata(chain[:kept]),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_eot_id(self, chain: list[CompletionRecord]) -> int | None:
        """Return configured EOT id, else auto-detect from the chain.

        Auto-detection uses the last token of the first completion whose
        ``finish_reason`` indicates the model emitted the natural stop
        marker itself (stop / tool_calls / stop_sequence).
        """
        if self._configured_eot_id is not None:
            return self._configured_eot_id
        for completion in chain:
            trace = build_trace_from_completion(
                completion,
                tokenizer=self._tokenizer,
            )
            if (
                trace.finish_reason in _NATURAL_STOP_REASONS
                and trace.response_ids
            ):
                return trace.response_ids[-1]
        return None

    @staticmethod
    def _slice_interstitial(
        *,
        canonical_tail: list[int],
        prev_raw_response: list[int],
        eot_id: int | None,
    ) -> list[int] | None:
        """Extract the canonical interstitial from C_{i+1}'s prompt tail.

        ``canonical_tail`` = canonical tokens for [prev assistant msg +
        harness-inserted messages + generation-prompt glue].  The first
        occurrence of ``eot_id`` marks the end of the prev assistant
        body; everything after is interstitial.

        If ``prev_raw_response`` already ends with ``eot_id`` (natural
        stop / tool_calls), skip it in the canonical tail to avoid
        duplication; otherwise (truncation) include it so the stream
        still closes the assistant turn.

        Returns None if ``eot_id`` is unknown or not present — caller
        should treat this as a break.
        """
        if eot_id is None:
            return None
        try:
            k = canonical_tail.index(eot_id)
        except ValueError:
            return None
        if prev_raw_response and prev_raw_response[-1] == eot_id:
            return canonical_tail[k + 1 :]
        return canonical_tail[k:]

    @staticmethod
    def _append_response_tokens(
        trace: Trace,
        stream_ids: list[int],
        response_slots: list[float | None],
        loss_mask: list[int],
    ) -> None:
        """Append a completion's response_ids and parallel logprob slots."""
        response_ids = list(trace.response_ids)
        stream_ids.extend(response_ids)
        trace_loss_mask = list(trace.loss_mask) or [1] * len(response_ids)
        if len(trace_loss_mask) != len(response_ids):
            raise ValueError("trace loss_mask length must match response_ids length")
        loss_mask.extend(trace_loss_mask)
        logprobs = trace.response_logprobs or []
        for pos in range(len(response_ids)):
            value = logprobs[pos] if pos < len(logprobs) else None
            response_slots.append(float(value) if isinstance(value, (int, float)) else None)

    @staticmethod
    def _finalize_logprobs(
        slots: list[float | None],
        loss_mask: list[int],
    ) -> list[float] | None:
        if len(slots) != len(loss_mask):
            raise ValueError("logprob slots length must match loss_mask length")
        if not any(slot is not None for slot in slots):
            return None
        if any(mask and slot is None for slot, mask in zip(slots, loss_mask)):
            return None
        # Interstitial slots (tool results, chat glue) get 0.0; loss_mask=0
        # makes the trainer ignore them.
        return [slot if slot is not None else 0.0 for slot in slots]

    @staticmethod
    def _chain_metadata(chain: list[CompletionRecord]) -> dict[str, Any]:
        completion_metadata = [dict(completion.metadata) for completion in chain]
        merged = dict(completion_metadata[0]) if completion_metadata else {}
        merged["completion_metadata"] = completion_metadata
        return merged

    @staticmethod
    def _find_extendable_chain(
        prompt_ids: list[int],
        chain_tips: list[list[int]],
    ) -> int | None:
        """Return the open chain this completion append-extends, else None.

        A completion continues a chain iff its prompt begins with that chain's
        last prompt (``tip`` is a token-prefix of ``prompt_ids``).  This routes
        completions to the right chain even when parallel agents / sub-agents
        interleave — each conversation has a distinct prompt prefix — and
        tolerates the just-finished turn being re-serialized in history (tool-call
        argument reformatting, whitespace), since that divergence falls *after*
        the prompt.  The compared prefix is two server-side tokenizations of the
        same text, so BPE re-tokenization of the sampled response never enters
        the decision.  On overlap the longest matching tip wins (most advanced
        chain).
        """
        best_idx: int | None = None
        best_len = -1
        for idx, tip in enumerate(chain_tips):
            n = len(tip)
            if n > best_len and 0 < n <= len(prompt_ids) and prompt_ids[:n] == tip:
                best_idx, best_len = idx, n
        return best_idx


def _top_level_scheduler_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = {"group_id", "policy_version", "rollout_step"}
    return {key: metadata[key] for key in keys if key in metadata}
