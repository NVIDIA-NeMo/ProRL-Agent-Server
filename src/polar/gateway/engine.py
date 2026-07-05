"""Inference backend strategies for the Polar gateway.

The gateway speaks the OpenAI Chat Completions API to a local inference server.
Two backends are supported, and they differ only in:

  1. the request params that make them emit the token ids + per-token logprobs
     Polar needs for training, and
  2. the exact shape of those fields in the response.

The base implements the canonical contract -- request ``logprobs`` (the one
training param every backend needs) and a pass-through response. A backend
overrides only what it does differently, via two hooks: ``prepare_request``
(extra request params) and ``normalize_response`` (response canonicalization).
Everything downstream -- storage, trace builder, transforms, slime adapter --
then sees one shape. The canonical shape is Polar's training output:

  - prompt token ids:   ``choice.input_token_ids`` (or ``response.prompt_token_ids``)
  - response token ids: ``choice.token_ids``       (or ``logprobs.content[].token_id``)
  - per-token logprobs: ``choice.logprobs.content[]`` with ``{token, token_id, logprob, ...}``

SGLang reaches this shape via source-supported prompt-token/meta-info extensions
plus a light response normalization. vLLM reaches it natively via the
``return_token_ids`` request flag plus a light response rename.
"""

from __future__ import annotations

from abc import ABC
import math
from typing import Any


# Private hand-off from the inference adapter to the gateway storage layer.
# The server removes this key before persisting/returning the OpenAI response
# and stores only its sanitized, low-cardinality values in completion metadata.
POLAR_INFERENCE_TIMINGS_KEY = "_polar_inference_timings"

_SGLANG_DURATION_SECONDS_FIELDS = {
    "e2e_latency": "e2e_ms",
    "queue_time": "queue_ms",
    "forward_duration": "forward_ms",
    "prefill_forward_duration": "prefill_forward_ms",
    "decode_forward_duration": "decode_forward_ms",
    # PD-disaggregated SGLang versions expose some or all of these fields.
    "pd_prefill_bootstrap_queue_duration": "pd_prefill_bootstrap_queue_ms",
    "pd_prefill_bootstrap_duration": "pd_prefill_bootstrap_ms",
    "pd_prefill_alloc_wait_duration": "pd_prefill_alloc_wait_ms",
    "pd_prefill_forward_duration": "pd_prefill_forward_ms",
    "pd_prefill_transfer_queue_duration": "pd_prefill_transfer_queue_ms",
    "pd_decode_prealloc_duration": "pd_decode_prealloc_ms",
    "pd_decode_bootstrap_duration": "pd_decode_bootstrap_ms",
    "pd_decode_alloc_wait_duration": "pd_decode_alloc_wait_ms",
    "pd_decode_transfer_duration": "pd_decode_transfer_ms",
    "pd_decode_forward_duration": "pd_decode_forward_ms",
}

_SGLANG_SCALAR_FIELDS = {
    # Do not retain SGLang 0.5.13's ``decode_throughput`` here. Polar's
    # gateway deliberately issues non-streaming requests, and that SGLang
    # path stamps ``first_token_time`` only when the final response reaches
    # the tokenizer manager. The resulting near-zero decode interval produces
    # impossible values (observed at 10^7--10^8 token/s). Explicit scheduler
    # duration/timestamp fields above remain trustworthy and are kept.
    "pd_transfer_speed_gb_s": "pd_transfer_speed_gb_s",
    "pd_transfer_total_mb": "pd_transfer_total_mb",
    "pd_prefill_retry_count": "pd_prefill_retry_count",
    "num_running_reqs": "num_running_reqs",
    "num_waiting_reqs": "num_waiting_reqs",
    "num_retractions": "num_retractions",
    "prompt_tokens": "prompt_tokens",
    "completion_tokens": "completion_tokens",
}

_POLAR_INFERENCE_TIMING_FIELDS = frozenset(
    {
        *_SGLANG_DURATION_SECONDS_FIELDS.values(),
        *_SGLANG_SCALAR_FIELDS.values(),
        "api_dispatch_ms",
        "request_to_forward_ms",
        "prefill_forward_ms",
        "decode_ms",
        "inference_service_ms",
    }
)


class InferenceEngine(ABC):
    """Strategy for one OpenAI-compatible inference backend.

    The base encodes the canonical contract: request ``logprobs`` (the one
    training param every backend needs) and pass the response through
    unchanged. A backend overrides only what it does differently.
    """

    name: str

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Inject the request params this backend needs to emit training signals.

        ``logprobs`` is universal; subclasses add backend-specific params (e.g.
        token-id flags) on top via ``super().prepare_request(...)``.
        """
        request["logprobs"] = True
        return request

    def normalize_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize the backend's response (in place) and return it.

        The default is a pass-through; a backend that differs from Polar's
        training shape overrides this.
        """
        return response

    @staticmethod
    def _stamp_token_ids_onto_logprobs(choice: dict[str, Any]) -> None:
        """Copy sampled token IDs onto OpenAI chat logprob entries when aligned."""

        token_ids = choice.get("token_ids")
        logprobs = choice.get("logprobs")
        if not isinstance(token_ids, list) or not isinstance(logprobs, dict):
            return
        content = logprobs.get("content")
        if not isinstance(content, list) or len(content) != len(token_ids):
            return
        for entry, token_id in zip(content, token_ids):
            if isinstance(entry, dict):
                entry.setdefault("token_id", token_id)


class OpenAICompatibleEngine(InferenceEngine):
    """Plain OpenAI-compatible inference used by frozen model-pool entries.

    Unlike the trainable SGLang/vLLM engines, this strategy intentionally does
    not request token IDs, logprobs, or backend metadata.  Pool responses are
    observations for the router rather than policy samples and therefore must
    never acquire Polar's training-only request extensions.
    """

    name = "openai_compatible"

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        for training_field in (
            "logprobs",
            "top_logprobs",
            "return_prompt_token_ids",
            "return_meta_info",
            "return_token_ids",
        ):
            request.pop(training_field, None)
        return request


class SGLangEngine(InferenceEngine):
    """SGLang via its source-supported OpenAI-compatible extensions."""

    name = "sglang"

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request = super().prepare_request(request)  # logprobs=True
        request["return_prompt_token_ids"] = True
        request["return_meta_info"] = True
        return request

    def normalize_response(self, response: dict[str, Any]) -> dict[str, Any]:
        # Never trust or preserve a backend-provided value under Polar's
        # private hand-off key; only values derived below from whitelisted
        # SGLang meta_info fields may enter completion metadata.
        response.pop(POLAR_INFERENCE_TIMINGS_KEY, None)
        choices = response.get("choices")
        if not isinstance(choices, list):
            return response
        inference_timings: list[dict[str, float]] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            self._canonicalize_prompt_token_ids(choice)
            self._canonicalize_response_token_ids(choice)
            self._stamp_token_ids_onto_logprobs(choice)
            timing = self._extract_inference_timing(choice.get("meta_info"))
            if timing:
                inference_timings.append(timing)
            # return_meta_info is requested as an internal bridge to recover
            # token IDs and timing from source SGLang. Do not expose or store
            # the duplicate raw meta payload downstream.
            choice.pop("meta_info", None)
        if inference_timings:
            response[POLAR_INFERENCE_TIMINGS_KEY] = inference_timings
        return response

    @staticmethod
    def _extract_inference_timing(meta_info: Any) -> dict[str, float]:
        """Keep finite numeric timings and derive stable request sub-stages.

        SGLang 0.5.13 exposes seconds plus wall-clock timestamps.  Newer PD
        builds also expose explicit duration fields.  Convert durations to
        milliseconds here and never retain request IDs, raw metadata, or
        timestamps in traces/W&B.
        """

        if not isinstance(meta_info, dict):
            return {}

        timing: dict[str, float] = {}
        for source, target in _SGLANG_DURATION_SECONDS_FIELDS.items():
            value = _nonnegative_finite_float(meta_info.get(source))
            if value is not None:
                timing[target] = value * 1000.0
        for source, target in _SGLANG_SCALAR_FIELDS.items():
            value = _nonnegative_finite_float(meta_info.get(source))
            if value is not None:
                timing[target] = value

        request_received = _finite_float(meta_info.get("request_received_ts"))
        dispatch_finished = _finite_float(
            meta_info.get("api_server_dispatch_finish_ts")
        )
        request_finished = _finite_float(meta_info.get("request_finished_ts"))
        forward_entry = _finite_float(meta_info.get("forward_entry_time"))
        prefill_finished = _finite_float(meta_info.get("prefill_finished_time"))

        _set_elapsed_ms(
            timing,
            "api_dispatch_ms",
            request_received,
            dispatch_finished,
        )
        _set_elapsed_ms(
            timing,
            "request_to_forward_ms",
            request_received,
            forward_entry,
        )
        _set_elapsed_ms(
            timing,
            "prefill_forward_ms",
            forward_entry,
            prefill_finished,
        )
        _set_elapsed_ms(
            timing,
            "decode_ms",
            prefill_finished,
            request_finished,
        )
        _set_elapsed_ms(
            timing,
            "inference_service_ms",
            forward_entry,
            request_finished,
        )
        return timing

    @staticmethod
    def _canonicalize_prompt_token_ids(choice: dict[str, Any]) -> None:
        if choice.get("input_token_ids") is not None:
            return
        prompt_token_ids = choice.get("prompt_token_ids")
        if isinstance(prompt_token_ids, list):
            choice["input_token_ids"] = list(prompt_token_ids)

    @staticmethod
    def _canonicalize_response_token_ids(choice: dict[str, Any]) -> None:
        meta_info = choice.get("meta_info")
        if not isinstance(meta_info, dict):
            return
        output_logprobs = meta_info.get("output_token_logprobs")
        if not isinstance(output_logprobs, list) or not output_logprobs:
            return

        token_ids: list[int] = []
        canonical_logprobs: list[dict[str, Any]] = []
        for item in output_logprobs:
            token_id = None
            logprob = None
            token = None
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                logprob = item[0]
                token_id = item[1]
                if len(item) >= 3:
                    token = item[2]
            elif isinstance(item, dict):
                token_id = item.get("token_id")
                logprob = item.get("logprob", item.get("token_logprob"))
                token = item.get("token")
            if token_id is None or logprob is None:
                return
            try:
                canonical_token_id = int(token_id)
                canonical_logprob = float(logprob)
            except (TypeError, ValueError):
                return

            token_ids.append(canonical_token_id)
            canonical_entry: dict[str, Any] = {
                "token_id": canonical_token_id,
                "logprob": canonical_logprob,
            }
            if token is not None:
                canonical_entry["token"] = str(token)
            canonical_logprobs.append(canonical_entry)

        choice["token_ids"] = token_ids

        # Some SGLang versions return a truncated OpenAI-style
        # ``logprobs.content`` while ``meta_info.output_token_logprobs`` still
        # contains every sampled token. Rebuild the canonical list from the
        # paired meta entries instead of keeping token IDs and logprobs with
        # different lengths (which would silently disable TIS downstream).
        logprobs = choice.get("logprobs")
        if not isinstance(logprobs, dict):
            logprobs = {}
            choice["logprobs"] = logprobs
        content = logprobs.get("content")
        if isinstance(content, list) and len(content) == len(canonical_logprobs):
            merged_content: list[dict[str, Any]] = []
            for existing, canonical in zip(content, canonical_logprobs):
                merged = dict(existing) if isinstance(existing, dict) else {}
                merged.update(canonical)
                merged_content.append(merged)
            logprobs["content"] = merged_content
        else:
            logprobs["content"] = canonical_logprobs


class VLLMEngine(InferenceEngine):
    """vLLM via its native OpenAI-compatible server.

    ``return_token_ids`` makes vLLM emit ``response.prompt_token_ids`` and
    ``choice.token_ids``. `top_logprobs`` must be set (not None) for vLLM
    to populate ``logprobs.content[]`` given ``logprobs=True``; 0 returns just
    the sampled token's logprob, which is all training needs.
    """

    name = "vllm"

    def prepare_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request = super().prepare_request(request)  # logprobs=True
        request["return_token_ids"] = True
        request.setdefault("top_logprobs", 0)
        # vLLM reads input reasoning from `reasoning`, not Polar's canonical
        # `reasoning_content`; rename it so prior turns' interleaved thinking
        # survives templating (else they render an empty `<think></think>`).
        for message in request.get("messages") or []:
            if isinstance(message, dict) and message.get("reasoning_content") is not None:
                message["reasoning"] = message.pop("reasoning_content")
        return request

    def normalize_response(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices")
        if not isinstance(choices, list):
            return response
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            self._canonicalize_reasoning(choice.get("message"))
            self._stamp_token_ids_onto_logprobs(choice)
        return response

    @staticmethod
    def _canonicalize_reasoning(message: Any) -> None:
        """vLLM names the field ``reasoning``; Polar's canonical field is ``reasoning_content``."""
        if not isinstance(message, dict):
            return
        if message.get("reasoning_content") is None and message.get("reasoning") is not None:
            message["reasoning_content"] = message.pop("reasoning")

    @staticmethod
    def _stamp_token_ids_onto_logprobs(choice: dict[str, Any]) -> None:
        """Parity with SGLang: copy token_id onto each logprob entry.

        vLLM builds ``logprobs.content`` and ``choice.token_ids`` from the same
        ``output.token_ids``, so they align; guard on equal length regardless.
        Not load-bearing for training (which reads ``choice.token_ids`` and the
        per-entry ``logprob``) -- it keeps stored traces one shape across engines.
        """
        InferenceEngine._stamp_token_ids_onto_logprobs(choice)


_ENGINES: dict[str, type[InferenceEngine]] = {
    SGLangEngine.name: SGLangEngine,
    VLLMEngine.name: VLLMEngine,
}


def get_engine(name: str) -> InferenceEngine:
    """Return the inference engine strategy for ``name`` (``sglang`` | ``vllm``)."""
    try:
        engine_cls = _ENGINES[name]
    except KeyError:
        supported = ", ".join(sorted(_ENGINES))
        raise ValueError(
            f"Unknown inference engine {name!r}; supported: {supported}"
        ) from None
    return engine_cls()


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _nonnegative_finite_float(value: Any) -> float | None:
    parsed = _finite_float(value)
    if parsed is None or parsed < 0.0:
        return None
    return parsed


def _set_elapsed_ms(
    timing: dict[str, float],
    key: str,
    started: float | None,
    finished: float | None,
) -> None:
    if key in timing or started is None or finished is None or finished < started:
        return
    timing[key] = (finished - started) * 1000.0


def sanitize_inference_timings(value: Any) -> list[dict[str, float]]:
    """Revalidate the private engine hand-off at the storage trust boundary."""

    if not isinstance(value, list):
        return []
    sanitized: list[dict[str, float]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        clean: dict[str, float] = {}
        for key in _POLAR_INFERENCE_TIMING_FIELDS:
            parsed = _nonnegative_finite_float(item.get(key))
            if parsed is not None:
                clean[key] = parsed
        if clean:
            sanitized.append(clean)
    return sanitized
