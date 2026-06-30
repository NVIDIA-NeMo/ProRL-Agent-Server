from __future__ import annotations

import pytest

from polar.gateway.engine import (
    POLAR_INFERENCE_TIMINGS_KEY,
    SGLangEngine,
    VLLMEngine,
    get_engine,
)


def test_get_engine_returns_the_right_strategy() -> None:
    assert isinstance(get_engine("sglang"), SGLangEngine)
    assert isinstance(get_engine("vllm"), VLLMEngine)


def test_get_engine_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown inference engine"):
        get_engine("tgi")


def test_sglang_engine_requests_source_token_extensions() -> None:
    engine = SGLangEngine()
    request = {"messages": []}
    out = engine.prepare_request(request)
    assert out is request and out["logprobs"] is True
    assert out["return_prompt_token_ids"] is True
    assert out["return_meta_info"] is True


def test_sglang_normalize_source_response_to_training_shape() -> None:
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hi"},
                "prompt_token_ids": [1, 2],
                "logprobs": {
                    "content": [
                        {"token": "h", "logprob": -0.1},
                        {"token": "i", "logprob": -0.2},
                    ]
                },
                "meta_info": {
                    "output_token_logprobs": [
                        [-0.1, 10, "h"],
                        [-0.2, 11, "i"],
                    ]
                },
            }
        ]
    }

    out = SGLangEngine().normalize_response(response)
    choice = out["choices"][0]

    assert choice["input_token_ids"] == [1, 2]
    assert choice["token_ids"] == [10, 11]
    assert [entry["token_id"] for entry in choice["logprobs"]["content"]] == [10, 11]
    assert "meta_info" not in choice


def test_sglang_normalize_preserves_sanitized_inference_timing() -> None:
    response = {
        "choices": [
            {
                "meta_info": {
                    "id": "must-not-be-copied",
                    "e2e_latency": 2.0,
                    "queue_time": 0.25,
                    # SGLang 0.5.13 computes this from tokenizer-manager
                    # delivery timestamps, which are invalid for Polar's
                    # forced non-streaming requests and must be discarded.
                    "decode_throughput": 10.0,
                    "request_received_ts": 100.0,
                    "api_server_dispatch_finish_ts": 100.1,
                    "forward_entry_time": 100.35,
                    "prefill_finished_time": 100.55,
                    "request_finished_ts": 102.0,
                    "num_running_reqs": 7,
                    "bad": float("nan"),
                }
            }
        ]
    }

    out = SGLangEngine().normalize_response(response)

    assert "meta_info" not in out["choices"][0]
    assert out[POLAR_INFERENCE_TIMINGS_KEY] == [
        {
            "e2e_ms": 2000.0,
            "queue_ms": 250.0,
            "num_running_reqs": 7.0,
            "api_dispatch_ms": pytest.approx(100.0),
            "request_to_forward_ms": pytest.approx(350.0),
            "prefill_forward_ms": pytest.approx(200.0),
            "decode_ms": pytest.approx(1450.0),
            "inference_service_ms": pytest.approx(1650.0),
        }
    ]


def test_sglang_keeps_explicit_forward_durations_but_drops_nonstreaming_throughput() -> None:
    response = {
        "choices": [
            {
                "meta_info": {
                    "forward_duration": 0.8,
                    "prefill_forward_duration": 0.2,
                    "decode_forward_duration": 0.6,
                    "decode_throughput": 99_999_999.0,
                }
            }
        ]
    }

    out = SGLangEngine().normalize_response(response)

    assert out[POLAR_INFERENCE_TIMINGS_KEY] == [
        {
            "forward_ms": pytest.approx(800.0),
            "prefill_forward_ms": pytest.approx(200.0),
            "decode_forward_ms": pytest.approx(600.0),
        }
    ]


def test_sglang_normalize_rejects_invalid_timing_values() -> None:
    response = {
        "choices": [
            {
                "meta_info": {
                    "e2e_latency": -1,
                    "queue_time": "not-a-number",
                    "decode_throughput": float("inf"),
                }
            }
        ]
    }

    out = SGLangEngine().normalize_response(response)

    assert POLAR_INFERENCE_TIMINGS_KEY not in out


def test_sglang_normalize_drops_spoofed_private_timing_without_choices() -> None:
    response = {
        "not_choices": [],
        POLAR_INFERENCE_TIMINGS_KEY: [{"secret": "must-not-be-persisted"}],
    }

    out = SGLangEngine().normalize_response(response)

    assert POLAR_INFERENCE_TIMINGS_KEY not in out


def test_vllm_prepare_request_requests_token_ids_and_logprobs() -> None:
    out = VLLMEngine().prepare_request({"messages": [], "logprobs": True})
    assert out["logprobs"] is True
    assert out["return_token_ids"] is True
    assert out["top_logprobs"] == 0


def test_vllm_prepare_request_keeps_explicit_top_logprobs() -> None:
    out = VLLMEngine().prepare_request({"logprobs": True, "top_logprobs": 5})
    assert out["top_logprobs"] == 5


def test_vllm_prepare_request_forces_logprobs_when_absent() -> None:
    out = VLLMEngine().prepare_request({"messages": []})
    assert out["logprobs"] is True
    assert out["return_token_ids"] is True
    assert out["top_logprobs"] == 0


def test_vllm_normalize_renames_reasoning_to_reasoning_content() -> None:
    response = {
        "choices": [
            {"message": {"role": "assistant", "content": "a", "reasoning": "because"}}
        ]
    }
    message = VLLMEngine().normalize_response(response)["choices"][0]["message"]
    assert message["reasoning_content"] == "because"
    assert "reasoning" not in message


def test_vllm_normalize_keeps_existing_reasoning_content() -> None:
    response = {"choices": [{"message": {"reasoning": "new", "reasoning_content": "kept"}}]}
    message = VLLMEngine().normalize_response(response)["choices"][0]["message"]
    assert message["reasoning_content"] == "kept"


def test_vllm_normalize_without_reasoning_is_noop() -> None:
    response = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    out = VLLMEngine().normalize_response(response)
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "hi"}


def test_vllm_normalize_stamps_token_ids_onto_logprobs() -> None:
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hi"},
                "token_ids": [10, 11],
                "logprobs": {
                    "content": [
                        {"token": "h", "logprob": -0.1},
                        {"token": "i", "logprob": -0.2},
                    ]
                },
            }
        ]
    }
    content = VLLMEngine().normalize_response(response)["choices"][0]["logprobs"]["content"]
    assert [entry["token_id"] for entry in content] == [10, 11]


def test_vllm_normalize_skips_token_id_stamp_on_length_mismatch() -> None:
    response = {
        "choices": [
            {
                "token_ids": [10, 11, 12],
                "logprobs": {"content": [{"token": "h", "logprob": -0.1}]},
            }
        ]
    }
    content = VLLMEngine().normalize_response(response)["choices"][0]["logprobs"]["content"]
    assert "token_id" not in content[0]
