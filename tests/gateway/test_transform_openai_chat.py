from __future__ import annotations

from polar.gateway.transform.openai_chat import OpenAIChatTransformer

IMAGE_URL = "data:image/png;base64,abc123"


def test_openai_chat_request_preserves_fields_and_image_content() -> None:
    transformer = OpenAIChatTransformer()

    body = {
        "_polar_model_served": "Qwen/Qwen3.5-4B",
        "model": "client-visible-model",
        "messages": [
            {"role": "developer", "content": "Use short answers."},
            {"role": "system", "content": "Be precise."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Count stars."},
                    {"type": "image_url", "image_url": {"url": IMAGE_URL, "detail": "low"}},
                ],
            },
        ],
        "max_tokens": 64,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": True,
        "stop": ["END"],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write_answer",
                    "description": "Write the answer",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "tool_choice": "auto",
        "chat_template_kwargs": {"foo": "bar"},
    }

    transformed = transformer.transform_request(body)

    assert "_polar_model_served" not in transformed
    assert transformed["model"] == "client-visible-model"
    # `developer` role is normalized to `system` and merged with adjacent systems.
    assert transformed["messages"] == [
        {"role": "system", "content": "Use short answers.\n\nBe precise."},
        body["messages"][2],
    ]
    assert transformed["max_tokens"] == 64
    assert transformed["temperature"] == 0.2
    assert transformed["top_p"] == 0.9
    assert transformed["stream"] is True
    assert transformed["stop"] == ["END"]
    assert transformed["tools"] == body["tools"]
    assert transformed["tool_choice"] == "auto"
    assert transformed["chat_template_kwargs"] == {"foo": "bar", "enable_thinking": False}


def test_openai_chat_response_and_stream_preserve_requested_model() -> None:
    transformer = OpenAIChatTransformer()
    upstream = {
        "id": "chatcmpl-1",
        "model": "served-model",
        "choices": [{"message": {"content": "Done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }

    response = transformer.transform_response(upstream, {"model": "requested-model"})
    stream_chunk = transformer.transform_stream_chunk(
        {
            "id": "chatcmpl-1",
            "model": "served-model",
            "choices": [{"delta": {"content": "D"}}],
        },
        {"model": "requested-model"},
    )

    assert response == {**upstream, "model": "requested-model"}
    assert stream_chunk["model"] == "requested-model"
    assert stream_chunk["choices"][0]["delta"]["content"] == "D"


def test_openai_chat_request_aliases_max_completion_tokens() -> None:
    transformer = OpenAIChatTransformer()

    transformed = transformer.transform_request(
        {
            "messages": [{"role": "user", "content": "answer as JSON"}],
            "max_completion_tokens": 32,
            "response_format": {"type": "json_object"},
        }
    )

    assert transformed["max_completion_tokens"] == 32
    assert transformed["max_tokens"] == 32
    assert transformed["response_format"] == {"type": "json_object"}


def test_openai_chat_merges_developer_role_for_non_qwen_models() -> None:
    transformer = OpenAIChatTransformer()

    transformed = transformer.transform_request(
        {
            "_polar_model_served": "MiniMax-M2.5",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "developer", "content": "Use short answers."},
                {"role": "system", "content": [{"type": "text", "text": "Be precise."}]},
            ],
        }
    )

    assert transformed["messages"] == [
        {"role": "system", "content": "Use short answers.\n\nBe precise."},
        {"role": "user", "content": "hi"},
    ]
    assert "chat_template_kwargs" not in transformed


def test_qwen35_explicit_thinking_request_is_not_disabled() -> None:
    transformer = OpenAIChatTransformer()

    transformed = transformer.transform_request(
        {
            "_polar_model_served": "Qwen/Qwen3.5-9B",
            "messages": [{"role": "user", "content": "use bash"}],
            "chat_template_kwargs": {"enable_thinking": True},
        }
    )

    assert transformed["chat_template_kwargs"]["enable_thinking"] is True


def test_openai_chat_preserves_tool_turns_and_reasoning_content() -> None:
    transformer = OpenAIChatTransformer()
    body = {
        "messages": [
            {"role": "system", "content": "Use tools when needed."},
            {"role": "user", "content": "lookup x"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I need the lookup tool.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
            {"role": "user", "content": "summarize"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                    "strict": True,
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
    }

    transformed = transformer.transform_request(body)

    assert transformed["messages"] == body["messages"]
    assert transformed["tools"] == body["tools"]
    assert transformed["tool_choice"] == body["tool_choice"]
