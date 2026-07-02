"""Convert Polar rollout results into Slime samples.

Every trace in ``Trajectory.traces`` becomes one Slime ``Sample``.  All
samples produced from the same session share ``Sample.rollout_id`` so Slime's
loss reducer counts the trajectory once even when it fans out into
multiple trace samples.  Builders own trace curation and per-token loss masks
— the adapter does not infer trainable positions from bridge details. Traces
that lack training tokens are dropped and represented as fully masked samples
so callers can keep the rest of the group trainable.
"""

from __future__ import annotations

from copy import deepcopy
import logging
import math
from typing import Any, TYPE_CHECKING

from polar.trajectory.training_filter import parser_invalid_tool_call_reason
from slime_bridge._messages import messages_to_text

if TYPE_CHECKING:
    from polar.rollout.models import SessionResult
    from polar.trajectory.models import Trace

logger = logging.getLogger(__name__)

_TRAINABLE_AGENT_TIMEOUT_REASON = "agent_timeout"


class RolloutLogprobError(ValueError):
    """Raised when a trainable Polar trace lacks aligned rollout logprobs."""


def session_result_to_samples(
    result: "SessionResult",
    group_index: int,
    *,
    trajectory_index: int,
    reward_key: str = "score",
    max_tokens: int | None = None,
) -> list[Any]:
    """Convert one Polar session result into Slime samples — one per trace.

    Every usable trace becomes an independent Sample sharing the same
    ``rollout_id`` key. Slime's loss reducer then averages all trace
    contributions as one trajectory, while the reward post-processor can still
    assign each trace its own advantage.

    Traces with empty tokens are dropped (logged). If a trace exceeds
    ``max_tokens``, the adapter keeps the longest causal prefix that contains
    the complete prompt and at least one response token. This preserves the
    exact rollout context and aligned old-policy logprobs for every retained
    trainable token. A trace whose prompt alone fills the budget cannot be
    clipped without changing the policy conditioning, so it is dropped. If
    *all* traces are dropped we emit a single zero-gradient placeholder so
    Slime's flattener doesn't crash on an empty list and the rest of the group
    can still train.
    """
    Sample = _load_sample_type()
    traces = result.trajectory.traces
    samples: list[Any] = []
    for trace_index, trace in enumerate(traces):
        try:
            sample = _build_sample(
                Sample=Sample,
                result=result,
                trace=trace,
                trace_index=trace_index,
                group_index=group_index,
                index=trajectory_index,
                reward_key=reward_key,
                max_tokens=max_tokens,
            )
        except RolloutLogprobError:
            # A malformed tool action is useful negative signal only while its
            # sampled tokens and old-policy log-probabilities align. Drop an
            # unaligned parser trace without terminating the whole batch.
            if parser_invalid_tool_call_reason(trace) is None:
                raise
            logger.warning(
                "Dropping unaligned parser-invalid trace %d from session %s",
                trace_index,
                result.session_id,
                exc_info=True,
            )
            sample = None
        if sample is not None:
            samples.append(sample)

    if samples:
        return samples

    logger.warning(
        "Session %s: no usable trace (traces=%d, max_tokens=%s); emitting dummy placeholder",
        result.session_id,
        len(traces),
        max_tokens,
    )
    return [
        _build_dummy_sample(
            Sample=Sample,
            result=result,
            group_index=group_index,
            index=trajectory_index,
            reward_key=reward_key,
        )
    ]


def session_result_to_placeholder(
    result: "SessionResult",
    group_index: int,
    *,
    trajectory_index: int,
    reward_key: str = "score",
    conversion_error: str | None = None,
) -> Any:
    """Create one zero-gradient sample after an isolated conversion failure."""

    sample = _build_dummy_sample(
        Sample=_load_sample_type(),
        result=result,
        group_index=group_index,
        index=trajectory_index,
        reward_key=reward_key,
    )
    if conversion_error:
        polar_metadata = sample.metadata["polar"]
        polar_metadata["sample_conversion_error"] = conversion_error
        polar_metadata["training_filter"] = {
            "masked": True,
            "reason": "sample_conversion_error",
            "detail": conversion_error,
        }
    return sample


def _build_sample(
    *,
    Sample: Any,
    result: "SessionResult",
    trace: "Trace",
    trace_index: int,
    group_index: int,
    index: int,
    reward_key: str,
    max_tokens: int | None = None,
) -> Any | None:
    prompt_ids = list(trace.prompt_ids)
    response_ids = list(trace.response_ids)

    if not prompt_ids or not response_ids:
        logger.warning(
            "Dropping trace %d from session %s: missing tokens (prompt=%d, response=%d)",
            trace_index,
            result.session_id,
            len(prompt_ids),
            len(response_ids),
        )
        return None

    prompt_messages = deepcopy(trace.prompt_messages)
    response_messages = deepcopy(trace.response_messages)
    response_text = messages_to_text(response_messages)

    status = _sample_status(Sample, result, trace)
    parser_invalid_reason = parser_invalid_tool_call_reason(trace)
    trainable_agent_timeout = _is_trainable_agent_timeout(result, trace)
    # A task-level evaluator result is not a valid policy reward when the
    # execution failed. An agent-budget timeout is an aligned sampled policy
    # action: keep it trainable, but fail-close its scalar reward to zero even
    # if a stale evaluator artifact carries a positive value.
    invalid_execution = status in (Sample.Status.ABORTED, Sample.Status.FAILED)
    reward_value = (
        0.0
        if parser_invalid_reason is not None or trainable_agent_timeout or invalid_execution
        else _reward_value(trace)
    )

    trainable = not invalid_execution
    loss_mask = _loss_mask_from_trace(
        trace,
        len(response_ids),
        require_loss_mask=trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )
    if status in (Sample.Status.ABORTED, Sample.Status.FAILED):
        loss_mask = [0] * len(response_ids)
    response_log_probs = _extract_rollout_log_probs(
        trace,
        response_len=len(response_ids),
        loss_mask=loss_mask,
        require_trainable_logprobs=trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )

    clipping_metadata = _clip_response_to_causal_prefix(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        loss_mask=loss_mask,
        response_log_probs=response_log_probs,
        max_tokens=max_tokens,
        session_id=result.session_id,
        trace_index=trace_index,
    )
    if clipping_metadata is None:
        return None
    if trainable and not any(loss_mask):
        logger.warning(
            "Dropping trace %d from session %s: retained causal prefix has zero trainable tokens",
            trace_index,
            result.session_id,
        )
        return None
    if clipping_metadata and status == Sample.Status.COMPLETED:
        status = Sample.Status.TRUNCATED

    prompt_value = prompt_messages if prompt_messages else ""

    polar_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_metadata": deepcopy(getattr(result, "metadata", {}) or {}),
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": trace_index,
        "trace_metadata": deepcopy(getattr(trace, "metadata", {}) or {}),
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
        # Preserved for the longest-trace wandb artifact dump; training reads
        # tokens+logprobs, not these.
        "trace_debug": {
            "finish_reason": trace.finish_reason,
            "response_messages": deepcopy(response_messages),
        },
    }
    if clipping_metadata:
        polar_metadata["token_clipping"] = clipping_metadata
    if parser_invalid_reason is not None or trainable_agent_timeout or invalid_execution:
        source_filter = trace.metadata.get("training_filter")
        training_filter = dict(source_filter) if isinstance(source_filter, dict) else {}
        if invalid_execution:
            training_filter.update(
                {
                    "masked": True,
                    "reason": (
                        "session_timeout" if status == Sample.Status.ABORTED else "session_error"
                    ),
                    "detail": (result.error or result.trajectory.error or str(result.status)),
                }
            )
        elif trainable_agent_timeout:
            training_filter.update(
                {
                    "masked": False,
                    "trainable": True,
                    "reason": _TRAINABLE_AGENT_TIMEOUT_REASON,
                    "detail": (
                        result.error or result.trajectory.error or "agent execution timeout"
                    ),
                }
            )
        else:
            training_filter.update(
                {
                    "masked": False,
                    "trainable": True,
                    "reason": "parser_invalid_tool_call",
                    "detail": parser_invalid_reason,
                }
            )
        training_filter.setdefault("original_reward", trace.reward)
        polar_metadata["training_filter"] = training_filter
    polar_metadata.update(_scheduler_metadata(result, trace))

    return Sample(
        group_index=group_index,
        index=index,
        prompt=prompt_value,
        tokens=prompt_ids + response_ids,
        response=response_text,
        response_length=len(response_ids),
        rollout_id=index,
        reward={reward_key: reward_value},
        loss_mask=loss_mask,
        rollout_log_probs=response_log_probs,
        status=status,
        session_id=result.session_id,
        metadata={"polar": polar_metadata},
        remove_sample=invalid_execution,
    )


def _clip_response_to_causal_prefix(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    loss_mask: list[int],
    response_log_probs: list[float],
    max_tokens: int | None,
    session_id: str,
    trace_index: int,
) -> dict[str, int | str] | None:
    """Fit a trace without changing any retained token's rollout context.

    Causal language-model losses for a response prefix depend only on the full
    prompt and earlier response tokens. Keeping ``prompt + response[:N]`` is
    therefore exact. Left-trimming the prompt is not: it changes both the
    current-policy logits and the meaning of the recorded rollout logprobs,
    which can corrupt TIS ratios. When the prompt leaves no response budget we
    deliberately drop the trace instead.
    """
    if max_tokens is None:
        return {}
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive when provided")

    total_len = len(prompt_ids) + len(response_ids)
    if total_len <= max_tokens:
        return {}

    response_budget = max_tokens - len(prompt_ids)
    if response_budget <= 0:
        logger.warning(
            "Dropping trace %d from session %s: prompt_len=%d leaves no exact "
            "response budget under max_tokens=%d",
            trace_index,
            session_id,
            len(prompt_ids),
            max_tokens,
        )
        return None

    original_response_len = len(response_ids)
    del response_ids[response_budget:]
    del loss_mask[response_budget:]
    del response_log_probs[response_budget:]
    dropped_response_tokens = original_response_len - len(response_ids)
    logger.warning(
        "Causal-prefix clipping trace %d from session %s: total_len=%d > "
        "max_tokens=%d (prompt=%d, response %d -> %d)",
        trace_index,
        session_id,
        total_len,
        max_tokens,
        len(prompt_ids),
        original_response_len,
        len(response_ids),
    )
    return {
        "strategy": "causal_prefix",
        "original_total_tokens": total_len,
        "original_prompt_tokens": len(prompt_ids),
        "original_response_tokens": original_response_len,
        "kept_prompt_tokens": len(prompt_ids),
        "kept_response_tokens": len(response_ids),
        "dropped_response_tokens": dropped_response_tokens,
        "max_tokens": max_tokens,
    }


def _build_dummy_sample(
    *,
    Sample: Any,
    result: "SessionResult",
    group_index: int,
    index: int,
    reward_key: str,
) -> Any:
    """Fully masked placeholder for a session with no usable trace.

    This carries no policy, TIS, or KL contribution. It lets the scheduler
    accept a partially usable group while still surfacing empty sessions in
    Polar metrics.
    """
    polar_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_metadata": deepcopy(getattr(result, "metadata", {}) or {}),
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": -1,
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
        "placeholder": True,
    }
    polar_metadata.update(_scheduler_metadata(result, None))
    return Sample(
        group_index=group_index,
        index=index,
        prompt="",
        tokens=[0, 0],
        response="",
        response_length=1,
        rollout_id=index,
        reward={reward_key: 0.0},
        loss_mask=[0],
        rollout_log_probs=[0.0],
        status=Sample.Status.ABORTED,
        remove_sample=True,
        session_id=result.session_id,
        metadata={"polar": polar_metadata},
    )


def _reward_value(trace: "Trace") -> float:
    """Read the reward the evaluator already placed on the trace.

    Reward assignment is the evaluator's job (including any broadcasting
    from session-level outcomes). slime_bridge just consumes what's there.
    """
    return float(trace.reward) if trace.reward is not None else 0.0


def _scheduler_metadata(result: "SessionResult", trace: "Trace | None") -> dict[str, Any]:
    keys = {"group_id", "policy_version", "rollout_step"}
    merged: dict[str, Any] = {}
    for source in (
        getattr(result, "metadata", None),
        getattr(result.trajectory, "metadata", None),
        getattr(trace, "metadata", None) if trace is not None else None,
    ):
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source:
                merged[key] = source[key]
    return merged


def _sample_status(Sample: Any, result: "SessionResult", trace: "Trace") -> Any:
    trajectory_status = result.trajectory.status
    if _is_trainable_agent_timeout(result, trace):
        return Sample.Status.TRUNCATED
    if trajectory_status == "TIMEOUT" or result.status == "TIMEOUT":
        return Sample.Status.ABORTED
    if (
        trajectory_status == "ERROR"
        or result.status == "ERROR"
        or result.error
        or result.trajectory.error
    ):
        return Sample.Status.FAILED
    if trace.finish_reason == "length":
        return Sample.Status.TRUNCATED
    return Sample.Status.COMPLETED


def _is_trainable_agent_timeout(result: "SessionResult", trace: "Trace") -> bool:
    trajectory = result.trajectory
    if trajectory.status != "TIMEOUT" or result.status != "TIMEOUT":
        return False
    agent_result = trajectory.metadata.get("agent_result")
    if not isinstance(agent_result, dict):
        return False
    if agent_result.get("status") != "timeout":
        return False
    if agent_result.get("timeout_source") != "agent":
        return False
    if agent_result.get("timeout_stage") != "exec":
        return False
    training_filter = trace.metadata.get("training_filter")
    if not isinstance(training_filter, dict):
        return False
    if training_filter.get("reason") != _TRAINABLE_AGENT_TIMEOUT_REASON:
        return False
    if training_filter.get("trainable") is not True:
        return False
    if training_filter.get("masked") is True:
        return False
    evaluation = trajectory.metadata.get("evaluation")
    return not (isinstance(evaluation, dict) and evaluation.get("verifier_timeout") is True)


def _extract_rollout_log_probs(
    trace: "Trace",
    *,
    response_len: int,
    loss_mask: list[int],
    require_trainable_logprobs: bool,
    session_id: str,
    trace_index: int,
) -> list[float]:
    logprobs = trace.response_logprobs
    if not logprobs:
        if require_trainable_logprobs and any(loss_mask):
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: missing rollout_log_probs "
                "for trainable response tokens"
            )
        return [0.0] * response_len

    if len(logprobs) != response_len:
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: rollout_log_probs length "
            f"{len(logprobs)} != response length {response_len}"
        )

    # response_logprobs is one float per response token (interstitials are 0.0,
    # masked out by loss_mask); the builder guarantees trainable tokens carry
    # their real sampled logprob.
    parsed_logprobs = [float(value) for value in logprobs]
    if not all(math.isfinite(value) for value in parsed_logprobs):
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: non-finite rollout_log_probs"
        )
    return parsed_logprobs


def _loss_mask_from_trace(
    trace: "Trace",
    response_len: int,
    *,
    require_loss_mask: bool,
    session_id: str,
    trace_index: int,
) -> list[int]:
    """Read and validate the builder-assigned per-response-token loss mask."""
    mask = list(trace.loss_mask)
    if not mask:
        if require_loss_mask:
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: missing loss_mask"
            )
        return [0] * response_len
    if len(mask) != response_len:
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: loss_mask length "
            f"{len(mask)} != response length {response_len}"
        )
    return [1 if int(value) else 0 for value in mask]


def _load_sample_type() -> Any:
    try:
        from slime.utils.types import Sample
    except ImportError as exc:
        raise ImportError(
            "Slime is required to convert Polar rollouts into training samples. "
            "Ensure the Slime package is installed in the current environment."
        ) from exc
    return Sample
