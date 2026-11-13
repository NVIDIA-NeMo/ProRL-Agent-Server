from __future__ import annotations

from polar.gateway.transform.google import GoogleTransformer

IMAGE_B64 = "abc123"
IMAGE_URL = f"data:image/png;base64,{IMAGE_B64}"


def test_google_request_maps_all_fields_and_image_input_to_chat() -> None:
    transformer = GoogleTransformer()

    transformed = transformer.transform_request(
        {
            "_polar_model_served": "Qwen/Qwen3.5-4B",
            "_streaming": True,
            "config": {
                "systemInstruction": {"parts": [{"text": "Be direct."}]},
                "generationConfig": {
                    "maxOutputTokens": 128,
                    "temperature": 0.2,
                    "topP": 0.9,
                    "topK": 40,
                    "candidateCount": 2,
                    "presencePenalty": 0.1,
                    "frequencyPenalty": 0.2,
                    "seed": 123,
                    "logprobs": 4,
                    "stopSequences": ["END"],
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {"answer": {"type": "STRING"}},
                        "required": ["answer"],
                    },
                },
                "tools": [
                    {
                        "functionDeclarations": [
                            {
                                "name": "write_answer",
                                "description": "Write the answer",
                                "parameters": {"type": "object"},
                            }
                        ]
                    }
                ],
                "toolConfig": {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": ["write_answer"],
                    }
                },
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Count stars."},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": IMAGE_B64,
                            }
                        },
                        {
                            "fileData": {
                                "mimeType": "image/jpeg",
                                "fileUri": "https://example.test/star.jpg",
                            }
                        },
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {"text": "I will call a tool."},
                        {
                            "functionCall": {
                                "id": "call-1",
                                "name": "write_answer",
                                "args": {"answer": 2},
                            }
                        },
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "call-1",
                                "name": "write_answer",
                                "response": {"ok": True},
                            }
                        }
                    ],
                },
            ],
        }
    )

    assert transformed["messages"] == [
        {"role": "system", "content": "Be direct."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Count stars."},
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                {"type": "image_url", "image_url": {"url": "https://example.test/star.jpg"}},
            ],
        },
        {
            "role": "assistant",
            "content": "I will call a tool.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "write_answer",
                        "arguments": '{"answer": 2}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": '{"ok": true}'},
    ]
    assert transformed["max_tokens"] == 128
    assert transformed["temperature"] == 0.2
    assert transformed["top_p"] == 0.9
    assert transformed["top_k"] == 40
    assert transformed["n"] == 2
    assert transformed["presence_penalty"] == 0.1
    assert transformed["frequency_penalty"] == 0.2
    assert transformed["seed"] == 123
    assert transformed["top_logprobs"] == 4
    assert transformed["stop"] == ["END"]
    assert transformed["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "google_response",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    }
    assert transformed["stream"] is True
    assert transformed["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "write_answer",
                "description": "Write the answer",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert transformed["tool_choice"] == {
        "type": "function",
        "function": {"name": "write_answer"},
    }
    assert transformed["chat_template_kwargs"]["enable_thinking"] is False


def test_google_request_maps_tool_choice_modes() -> None:
    transformer = GoogleTransformer()
    base_body = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [
            {
                "functionDeclarations": [
                    {"name": "lookup", "parametersJsonSchema": {"type": "object"}}
                ]
            }
        ],
    }

    assert (
        transformer.transform_request(
            {
                **base_body,
                "toolConfig": {"functionCallingConfig": {"mode": "NONE"}},
            }
        )["tool_choice"]
        == "none"
    )
    assert (
        transformer.transform_request(
            {
                **base_body,
                "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            }
        )["tool_choice"]
        == "required"
    )
    assert transformer.transform_request(
        {
            **base_body,
            "toolConfig": {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": ["lookup"],
                }
            },
        }
    )["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}


def test_google_request_maps_system_instruction_and_system_content_role() -> None:
    transformer = GoogleTransformer()

    transformed = transformer.transform_request(
        {
            "systemInstruction": "Top-level system.",
            "contents": [
                {"role": "system", "parts": [{"text": "Inline system."}]},
                {"role": "user", "parts": [{"text": "Hi"}]},
            ],
        }
    )

    assert transformed["messages"] == [
        {"role": "system", "content": "Top-level system.\n\nInline system."},
        {"role": "user", "content": "Hi"},
    ]


def test_google_request_maps_multi_turn_reasoning_and_parallel_tools() -> None:
    transformer = GoogleTransformer()

    transformed = transformer.transform_request(
        {
            "contents": [
                {"role": "user", "parts": [{"text": "Plan and call tools."}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "thought": True,
                            "text": "Need two independent lookups.",
                            "thoughtSignature": "sig-1",
                        },
                        {
                            "functionCall": {
                                "id": "call-a",
                                "name": "lookup",
                                "args": {"q": "a"},
                            }
                        },
                        {
                            "functionCall": {
                                "id": "call-b",
                                "name": "lookup",
                                "args": {"q": "b"},
                            }
                        },
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "call-a",
                                "name": "lookup",
                                "response": {"text": "A"},
                            }
                        },
                        {
                            "functionResponse": {
                                "id": "call-b",
                                "name": "lookup",
                                "response": {"text": "B"},
                            }
                        },
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "thought": True,
                            "text": "Combine both results.",
                            "thoughtSignature": "sig-2",
                        },
                        {"text": "A and B"},
                    ],
                },
            ]
        }
    )

    assert transformed["messages"] == [
        {"role": "user", "content": "Plan and call tools."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Need two independent lookups.",
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
        {"role": "tool", "tool_call_id": "call-a", "content": '{"text": "A"}'},
        {"role": "tool", "tool_call_id": "call-b", "content": '{"text": "B"}'},
        {
            "role": "assistant",
            "content": "A and B",
            "reasoning_content": "Combine both results.",
        },
    ]


def test_google_response_maps_openai_content_tools_finish_and_usage_back() -> None:
    transformer = GoogleTransformer()

    response = transformer.transform_response(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": [
                            {"type": "text", "text": "There are two."},
                            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                        ],
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "write_answer", "arguments": '{"answer": 2}'},
                            }
                        ],
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        },
        {},
    )

    assert response == {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "There are two."},
                        {"inline_data": {"mime_type": "image/png", "data": IMAGE_B64}},
                        {
                            "functionCall": {
                                "name": "write_answer",
                                "args": {"answer": 2},
                                "id": "call-1",
                            }
                        },
                    ],
                    "role": "model",
                },
                "finishReason": "MAX_TOKENS",
                "index": 0,
                "safetyRatings": [],
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 4,
            "totalTokenCount": 14,
        },
        "functionCalls": [{"name": "write_answer", "args": {"answer": 2}, "id": "call-1"}],
    }


def test_google_stream_state_accumulates_tool_deltas_and_usage() -> None:
    transformer = GoogleTransformer()
    state = transformer.create_stream_state({})

    events = state.process_chunk(
        {
            "choices": [{"delta": {"content": "Hi"}}],
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
                                    "function": {"name": "write_answer", "arguments": '{"answer"'},
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
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ": 2}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }
        )
    )
    events.extend(state.finalize())

    assert events[0]["candidates"][0]["content"]["parts"] == [{"text": "Hi"}]
    assert events[0]["usageMetadata"] == {
        "promptTokenCount": 4,
        "candidatesTokenCount": 1,
        "totalTokenCount": 5,
    }
    assert events[-1]["candidates"][0]["finishReason"] == "STOP"
    assert events[-1]["functionCalls"] == [
        {"name": "write_answer", "args": {"answer": 2}, "id": "call-1"}
    ]


def test_google_stream_state_preserves_text_finish_reason() -> None:
    transformer = GoogleTransformer()
    state = transformer.create_stream_state({})

    events = state.process_chunk(
        {
            "choices": [
                {
                    "delta": {"content": "All done."},
                    "finish_reason": "stop",
                }
            ]
        }
    )

    assert events[0]["candidates"][0]["content"]["parts"] == [{"text": "All done."}]
    assert events[0]["candidates"][0]["finishReason"] == "STOP"
    assert state.finalize() == []


def test_google_stream_state_emits_finish_only_text_event() -> None:
    transformer = GoogleTransformer()
    state = transformer.create_stream_state({})

    events = state.process_chunk(
        {
            "choices": [{"delta": {}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }
    )

    assert events[0]["candidates"][0]["content"]["parts"] == []
    assert events[0]["candidates"][0]["finishReason"] == "MAX_TOKENS"
    assert events[0]["usageMetadata"] == {
        "promptTokenCount": 4,
        "candidatesTokenCount": 2,
        "totalTokenCount": 6,
    }


def test_google_response_maps_extended_finish_reasons() -> None:
    transformer = GoogleTransformer()

    stop_seq = transformer.transform_response(
        {
            "choices": [
                {"message": {"content": "x"}, "finish_reason": "stop_sequence"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        {},
    )
    assert stop_seq["candidates"][0]["finishReason"] == "STOP"

    unknown = transformer.transform_response(
        {
            "choices": [
                {"message": {"content": "x"}, "finish_reason": "weird_reason"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        {},
    )
    # Unknown finish reasons fall through to STOP rather than crashing.
    assert unknown["candidates"][0]["finishReason"] == "STOP"


def test_google_response_preserves_cached_usage_tokens() -> None:
    transformer = GoogleTransformer()

    response = transformer.transform_response(
        {
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 5},
            },
        },
        {},
    )

    assert response["usageMetadata"]["cachedContentTokenCount"] == 5


def test_google_request_drops_server_side_tools() -> None:
    transformer = GoogleTransformer()
    transformed = transformer.transform_request(
        {
            "contents": [{"role": "user", "parts": [{"text": "search"}]}],
            "tools": [
                {
                    "functionDeclarations": [
                        {"name": "lookup", "parameters": {"type": "object"}}
                    ]
                },
                {"googleSearch": {}},
                {"codeExecution": {}},
                {"urlContext": {}},
            ],
        }
    )

    # googleSearch, codeExecution, urlContext are server-side built-ins
    # without functionDeclarations; only the custom function survives.
    assert transformed["tools"] == [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]


def test_google_direct_stream_chunk_handles_reasoning_content() -> None:
    transformer = GoogleTransformer()

    transformed = transformer.transform_stream_chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_content": "Think first.",
                        "content": "Then answer.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
        {},
        is_first=True,
    )

    candidate = transformed["candidates"][0]
    assert candidate["content"]["parts"][0]["thought"] is True
    assert candidate["content"]["parts"][0]["text"] == "Think first."
    assert candidate["content"]["parts"][1] == {"text": "Then answer."}
    assert candidate["finishReason"] == "STOP"


_CACHED_USAGE = {
    "prompt_tokens": 10,
    "completion_tokens": 2,
    "total_tokens": 12,
    "prompt_tokens_details": {"cached_tokens": 5},
}


def test_google_direct_stream_chunk_preserves_cached_usage_tokens() -> None:
    transformer = GoogleTransformer()

    transformed = transformer.transform_stream_chunk(
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": _CACHED_USAGE,
        },
        {},
    )

    assert transformed["usageMetadata"]["cachedContentTokenCount"] == 5


def test_google_stream_state_preserves_cached_usage_tokens() -> None:
    transformer = GoogleTransformer()
    state = transformer.create_stream_state({})

    events = state.process_chunk(
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": _CACHED_USAGE,
        }
    )
    events.extend(state.finalize())

    usage_events = [event for event in events if "usageMetadata" in event]
    assert usage_events
    assert usage_events[-1]["usageMetadata"]["cachedContentTokenCount"] == 5
