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
