from polar.gateway.transform.anthropic import AnthropicStreamState, AnthropicTransformer


def test_stream_state_output_tokens_not_reset_by_zero_usage_chunk():
    state = AnthropicStreamState(
        "claude-3", AnthropicTransformer.FINISH_TO_STOP_REASON
    )
    state.process_chunk(
        {"usage": {"completion_tokens": 5}, "choices": [{"delta": {"content": "hello"}}]}
    )
    assert state.output_tokens == 5
    state.process_chunk({"usage": {"completion_tokens": 0}, "choices": []})
    assert state.output_tokens == 5
    final = state.finalize()
    message_delta = next(e for e in final if e["type"] == "message_delta")
    assert message_delta["usage"]["output_tokens"] == 5


def test_stream_state_output_tokens_updates_with_larger_usage_chunk():
    state = AnthropicStreamState(
        "claude-3", AnthropicTransformer.FINISH_TO_STOP_REASON
    )
    state.process_chunk(
        {"usage": {"completion_tokens": 3}, "choices": [{"delta": {"content": "hi"}}]}
    )
    assert state.output_tokens == 3


def test_stream_state_reports_input_and_cached_tokens_in_message_delta():
    state = AnthropicStreamState(
        "claude-3", AnthropicTransformer.FINISH_TO_STOP_REASON
    )
    state.process_chunk(
        {"choices": [{"delta": {"content": "hi"}}]}, is_first=True
    )
    state.process_chunk(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
            "choices": [],
        }
    )
    final = state.finalize()
    message_delta = next(e for e in final if e["type"] == "message_delta")
    assert message_delta["usage"] == {
        "input_tokens": 60,
        "output_tokens": 7,
        "cache_read_input_tokens": 40,
    }
