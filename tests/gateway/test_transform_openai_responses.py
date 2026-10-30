from __future__ import annotations

from polar.gateway.transform.reasoning import encrypt_reasoning
from polar.gateway.transform.openai_responses import OpenAIResponsesTransformer

IMAGE_URL = "data:image/png;base64,abc123"


def test_responses_request_maps_all_fields_and_image_input_to_chat() -> None:
    transformer = OpenAIResponsesTransformer()

    transformed = transformer.transform_request(
        {
            "_polar_model_served": "Qwen/Qwen3.5-4B",
            "instructions": "You are a coding agent.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Count stars."},
                        {"type": "input_image", "image_url": IMAGE_URL, "detail": "low"},
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "shell",
                    "arguments": '{"cmd": "pytest"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "failed",
                },
                {"type": "input_text", "text": "Count again."},
                {"type": "input_image", "image_url": IMAGE_URL},
            ],
            "max_output_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_logprobs": 3,
            "parallel_tool_calls": False,
            "stream": True,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                        "required": ["count"],
                    },
                    "strict": True,
                }
            },
            "tool_choice": "auto",
            "tools": [
                {
                    "name": "shell",
                    "description": "Run a shell command",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
        }
    )

    assert transformed["messages"] == [
        {"role": "system", "content": "You are a coding agent."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Count stars."},
                {"type": "image_url", "image_url": {"url": IMAGE_URL, "detail": "low"}},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"cmd": "pytest"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "failed"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Count again."},
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            ],
        },
    ]
    assert transformed["max_tokens"] == 128
    assert transformed["temperature"] == 0.2
    assert transformed["top_p"] == 0.9
    assert transformed["top_logprobs"] == 3
    assert transformed["parallel_tool_calls"] is False
    assert transformed["stream"] is True
    assert transformed["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
            "strict": True,
        },
    }
    assert transformed["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a shell command",
                "parameters": {"type": "object"},
                "strict": True,
            },
        }
    ]
    assert transformed["tool_choice"] == "auto"
    assert transformed["chat_template_kwargs"]["enable_thinking"] is False


def test_responses_request_moves_image_function_output_to_user_message() -> None:
    transformer = OpenAIResponsesTransformer()

    transformed = transformer.transform_request(
        {
            "input": [
                {"type": "message", "role": "user", "content": "Look at the image."},
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "view_image",
                    "arguments": '{"path": "/polar/session/workspace/polar_stars.png"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": {
                        "body": [
                            {"type": "input_image", "image_url": IMAGE_URL, "detail": "high"}
                        ],
                        "success": True,
                    },
                },
            ],
        }
    )

    assert transformed["messages"] == [
        {"role": "user", "content": "Look at the image."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "view_image",
                        "arguments": '{"path": "/polar/session/workspace/polar_stars.png"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": IMAGE_URL, "detail": "high"}}
            ],
        },
    ]


def test_responses_request_drops_tool_choice_when_tools_are_empty() -> None:
    transformer = OpenAIResponsesTransformer()

    transformed = transformer.transform_request(
        {
            "input": "hello",
            "tool_choice": "auto",
            "tools": [],
        }
    )

    assert transformed["messages"] == [{"role": "user", "content": "hello"}]
    assert "tool_choice" not in transformed
    assert "tools" not in transformed


def test_responses_request_converts_nested_function_schema_and_preserves_strict() -> None:
    transformer = OpenAIResponsesTransformer()

    transformed = transformer.transform_request(
        {
            "input": "use a tool",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Lookup data",
                    "input_schema": {
                        "jsonSchema": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                        }
                    },
                    "strict": True,
                },
                {"type": "web_search"},
            ],
            "tool_choice": "required",
        }
    )

    assert transformed["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup data",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
                "strict": True,
            },
        }
    ]
    assert transformed["tool_choice"] == "required"


def test_responses_request_normalizes_flat_function_tool_choice() -> None:
    transformer = OpenAIResponsesTransformer()

    transformed = transformer.transform_request(
        {
            "input": "use a tool",
            "tools": [{"type": "function", "name": "lookup", "parameters": {}}],
            "tool_choice": {"type": "function", "name": "lookup"},
        }
    )

    assert transformed["tool_choice"] == {
        "type": "function",
        "function": {"name": "lookup"},
    }


def test_responses_text_format_text_is_omitted_for_sglang() -> None:
    transformer = OpenAIResponsesTransformer()

    transformed = transformer.transform_request(
        {
            "input": "plain text",
            "text": {"format": {"type": "text"}},
        }
    )

    assert "response_format" not in transformed


def test_responses_reasoning_effort_none_does_not_enable_thinking() -> None:
    transformer = OpenAIResponsesTransformer()

    transformed = transformer.transform_request(
        {
            "_polar_model_served": "MiniMax-M2.5",
            "input": "answer directly",
            "reasoning": {"effort": "none"},
        }
    )

    assert "chat_template_kwargs" not in transformed


def test_responses_request_round_trips_local_shell_items() -> None:
    transformer = OpenAIResponsesTransformer()

    transformed = transformer.transform_request(
        {
            "input": [
                {"type": "message", "role": "user", "content": "run tests"},
                {
                    "type": "local_shell_call",
                    "call_id": "call-shell",
                    "action": {"commands": ["pytest tests/gateway -q"], "timeout_ms": 1000},
                    "status": "completed",
                },
                {
                    "type": "local_shell_call_output",
                    "id": "call-shell",
                    "output": "ok",
                    "status": "completed",
                },
            ],
        }
    )

    assert transformed["messages"] == [
        {"role": "user", "content": "run tests"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-shell",
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": '{"cmd": "pytest tests/gateway -q", "timeout_ms": 1000}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-shell", "content": "ok"},
    ]


def test_responses_response_emits_local_shell_call_for_shell_function() -> None:
    transformer = OpenAIResponsesTransformer()

    response = transformer.transform_response(
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-shell",
                                "function": {
                                    "name": "shell",
                                    "arguments": '{"cmd": "pytest tests/gateway -q"}',
                                },
                            }
                        ],
                    }
                }
            ],
        },
        {"model": "requested-model"},
    )

    shell_call = response["output"][0]
    assert shell_call["type"] == "local_shell_call"
    assert shell_call["call_id"] == "call-shell"
    assert shell_call["action"] == {"commands": ["pytest tests/gateway -q"]}


def test_responses_request_recovers_reasoning_from_encrypted_content_only() -> None:
    transformer = OpenAIResponsesTransformer()
    encrypted = encrypt_reasoning("Recovered private reasoning.")

    transformed = transformer.transform_request(
        {
            "input": [
                {"type": "message", "role": "user", "content": "continue"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "content": [],
                    "encrypted_content": encrypted,
                },
                {"type": "message", "role": "assistant", "content": "ok"},
            ]
        }
    )

    assert transformed["messages"][1] == {
        "role": "assistant",
        "content": "ok",
        "reasoning_content": "Recovered private reasoning.",
    }


def test_responses_request_keeps_reasoning_with_each_tool_turn() -> None:
    transformer = OpenAIResponsesTransformer()

    transformed = transformer.transform_request(
        {
            "input": [
                {"type": "message", "role": "user", "content": "do two steps"},
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Plan first call."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call-a",
                    "name": "lookup",
                    "arguments": '{"step": 1}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-a",
                    "output": "first result",
                },
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Plan second call."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call-b",
                    "name": "lookup",
                    "arguments": '{"step": 2}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-b",
                    "output": "second result",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ]
        }
    )

    assert transformed["messages"] == [
        {"role": "user", "content": "do two steps"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"step": 1}'},
                }
            ],
            "reasoning_content": "Plan first call.",
        },
        {"role": "tool", "tool_call_id": "call-a", "content": "first result"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-b",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"step": 2}'},
                }
            ],
            "reasoning_content": "Plan second call.",
        },
        {"role": "tool", "tool_call_id": "call-b", "content": "second result"},
        {"role": "assistant", "content": "done"},
    ]


def test_responses_request_groups_parallel_function_calls_and_outputs() -> None:
    transformer = OpenAIResponsesTransformer()

    transformed = transformer.transform_request(
        {
            "input": [
                {"type": "message", "role": "user", "content": "parallel"},
                {
                    "type": "function_call",
                    "call_id": "call-a",
                    "name": "lookup",
                    "arguments": '{"q": "a"}',
                },
                {
                    "type": "function_call",
                    "call_id": "call-b",
                    "name": "lookup",
                    "arguments": '{"q": "b"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-a",
                    "output": {"body": "A"},
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-b",
                    "output": {"content": [{"type": "output_text", "text": "B"}]},
                },
            ]
        }
    )

    assert transformed["messages"] == [
        {"role": "user", "content": "parallel"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": "a"}'},
                },
                {
                    "id": "call-b",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": "b"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call-a", "content": "A"},
        {"role": "tool", "tool_call_id": "call-b", "content": "B"},
    ]


def test_responses_request_drops_server_side_tools() -> None:
    transformer = OpenAIResponsesTransformer()

    transformed = transformer.transform_request(
        {
            "input": "search the web",
            "tools": [
                {"type": "function", "name": "lookup", "parameters": {"type": "object"}},
                {"type": "web_search"},
                {"type": "file_search"},
                {"type": "computer_use_preview"},
                {"type": "mcp", "server_url": "https://example.test"},
                {"type": "code_interpreter"},
                {"type": "image_generation"},
                # Typed tool with a name should still be dropped — name must
                # not be enough to override the type filter.
                {"type": "computer_use", "name": "computer_use"},
            ],
            "tool_choice": "auto",
        }
    )

    # Only the custom function tool reaches SGLang.
    assert transformed["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert transformed["tool_choice"] == "auto"


def test_responses_response_maps_chat_result_back_to_response_shape() -> None:
    transformer = OpenAIResponsesTransformer()

    response = transformer.transform_response(
        {
            "id": "chatcmpl-1",
            "model": "served-model",
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "Done",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "lookup", "arguments": '{"q": "x"}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
            },
        },
        {"model": "requested-model"},
    )

    assert response == {
        "id": "chatcmpl-1",
        "object": "response",
        "created_at": 123,
        "status": "completed",
        "model": "requested-model",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Done"}],
            },
            {
                "type": "function_call",
                "id": response["output"][1]["id"],
                "call_id": "call-1",
                "name": "lookup",
                "arguments": '{"q": "x"}',
                "status": "completed",
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
    }


def test_responses_stream_state_emits_response_events_for_text_and_tools() -> None:
    transformer = OpenAIResponsesTransformer()
    state = transformer.create_stream_state({"model": "requested-model"})

    events = state.process_chunk(
        {
            "choices": [{"delta": {"content": "Hel"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        },
        is_first=True,
    )
    events.extend(
        state.process_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {"name": "lookup", "arguments": '{"q"'},
                                }
                            ]
                        }
                    }
                ]
            }
        )
    )
    events.extend(
        state.process_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "content": "lo",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ': "x"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }
        )
    )
    events.extend(state.finalize())

    event_types = [event["type"] for event in events]
    assert event_types[0] == "response.created"
    assert "response.output_text.delta" in event_types
    assert "response.function_call_arguments.delta" in event_types
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["model"] == "requested-model"
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Hello"
    assert events[-1]["response"]["output"][1]["arguments"] == '{"q": "x"}'


def test_responses_stream_state_emits_text_only_response() -> None:
    transformer = OpenAIResponsesTransformer()
    state = transformer.create_stream_state({"model": "requested-model"})

    events = state.process_chunk(
        {"choices": [{"delta": {"content": "Hel"}}]},
        is_first=True,
    )
    events.extend(
        state.process_chunk(
            {
                "choices": [{"delta": {"content": "lo."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        )
    )
    events.extend(state.finalize())

    types = [event["type"] for event in events]
    assert types[0] == "response.created"
    # Exactly one message item is opened and closed; no reasoning, no tool items.
    assert types.count("response.output_item.added") == 1
    assert types.count("response.output_item.done") == 1
    added_item_types = [
        event["item"]["type"]
        for event in events
        if event["type"] == "response.output_item.added"
    ]
    assert added_item_types == ["message"]
    assert "response.reasoning_summary_text.delta" not in types
    assert "response.function_call_arguments.delta" not in types

    completed = events[-1]
    assert completed["type"] == "response.completed"
    assert [item["type"] for item in completed["response"]["output"]] == ["message"]
    assert completed["response"]["output"][0]["content"][0]["text"] == "Hello."
    assert completed["response"]["usage"]["output_tokens"] == 2


def test_responses_stream_state_emits_reasoning_only_response() -> None:
    transformer = OpenAIResponsesTransformer()
    state = transformer.create_stream_state({"model": "requested-model"})

    events = state.process_chunk(
        {"choices": [{"delta": {"reasoning_content": "Think "}}]},
        is_first=True,
    )
    events.extend(
        state.process_chunk(
            {
                "choices": [
                    {"delta": {"reasoning_content": "harder."}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        )
    )
    events.extend(state.finalize())

    types = [event["type"] for event in events]
    assert types[0] == "response.created"
    # Reasoning summary lifecycle is complete: added → delta(s) → done.
    assert "response.reasoning_summary_part.added" in types
    assert "response.reasoning_summary_text.delta" in types
    assert "response.reasoning_summary_text.done" in types
    assert "response.reasoning_summary_part.done" in types
    # No text or tool events because the model never emitted content/tool_calls.
    assert "response.output_text.delta" not in types
    assert "response.function_call_arguments.delta" not in types

    added_item_types = [
        event["item"]["type"]
        for event in events
        if event["type"] == "response.output_item.added"
    ]
    assert added_item_types == ["reasoning"]

    completed = events[-1]
    assert completed["type"] == "response.completed"
    output = completed["response"]["output"]
    assert [item["type"] for item in output] == ["reasoning"]
    assert output[0]["summary"][0]["text"] == "Think harder."
    assert output[0]["content"][0]["text"] == "Think harder."
    assert output[0]["encrypted_content"]


def test_responses_stream_state_orders_reasoning_before_tool_without_text() -> None:
    transformer = OpenAIResponsesTransformer()
    state = transformer.create_stream_state({"model": "requested-model"})

    events = state.process_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "Need a lookup.",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
        is_first=True,
    )
    events.extend(state.finalize())

    added = [
        (event["output_index"], event["item"]["type"])
        for event in events
        if event["type"] == "response.output_item.added"
    ]
    done = [
        (event["output_index"], event["item"]["type"])
        for event in events
        if event["type"] == "response.output_item.done"
    ]

    assert added == [(0, "reasoning"), (1, "function_call")]
    assert done[:2] == [(0, "reasoning"), (1, "function_call")]
    assert events[-1]["response"]["output"][0]["type"] == "reasoning"
    assert events[-1]["response"]["output"][1]["type"] == "function_call"


def test_responses_request_merges_assistant_text_with_following_tool_call() -> None:
    """One model turn (text, then a tool call) must replay as ONE assistant message."""
    transformer = OpenAIResponsesTransformer()

    transformed = transformer.transform_request(
        {
            "input": [
                {"type": "message", "role": "user", "content": "find it"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me check."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call-a",
                    "name": "lookup",
                    "arguments": '{"q": 1}',
                },
                {"type": "function_call_output", "call_id": "call-a", "output": "result"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ]
        }
    )

    assert transformed["messages"] == [
        {"role": "user", "content": "find it"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": 1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-a", "content": "result"},
        {"role": "assistant", "content": "done"},
    ]


def test_responses_request_keeps_separate_text_only_assistant_turns() -> None:
    """Text-only assistant messages not followed by tool calls stay separate."""
    transformer = OpenAIResponsesTransformer()

    transformed = transformer.transform_request(
        {
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "message", "role": "assistant", "content": "first"},
                {"type": "message", "role": "assistant", "content": "second"},
            ]
        }
    )

    assert transformed["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
