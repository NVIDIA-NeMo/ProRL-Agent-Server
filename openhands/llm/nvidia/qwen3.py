import textwrap
from pydantic import SecretStr
import httpx
from transformers import AutoTokenizer
from typing import Any, Text
import re
import json
import uuid
from litellm import ModelResponse, ContextWindowExceededError

""" 
We modify the original chat template. The main issue for Qwen3 models is that the thinking content will be dropped from the content field under some conditions.
This create a mismatch if tokenizing the message in seperate turns compared with tokenizing all messages at once.
We modify the chat template to always include the thinking content in the content field. 
"""
qwen3_chat_template = chat_template_qwen3_thinking = (
    "{%- if tools %}"
    "{{'<|im_start|>system\\n' }}"
    "{%- if messages[0].role == 'system' %}"
    "{{ messages[0].content + '\\n\\n' }}"
    "{%- endif %}"
    "{{'# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\n"
    "You are provided with function signatures within <tools></tools> XML tags:\\n<tools>' }}"
    "{%- for tool in tools %}"
    "{{ '\\n' + (tool | tojson) }}"
    "{%- endfor %}"
    "{{ '\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:"
    "\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call><|im_end|>\\n' }}"
    "{%- else %}"
    "{%- if messages[0].role == 'system' %}"
    "{{'<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}"
    "{%- for message in messages[::-1] %}"
    "{%- set index = (messages|length - 1) - loop.index0 %}"
    "{%- if ns.multi_step_tool and message.role == 'user' and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}"
    "{%- set ns.multi_step_tool = false %}"
    "{%- set ns.last_query_index = index %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- for message in messages %}"
    "{%- if message.content is string %}"
    "{%- set content = message.content %}"
    "{%- else %}"
    "{%- set content = '' %}"
    "{%- endif %}"
    "{%- if (message.role == 'user') or (message.role == 'system' and not loop.first) %}"
    "{{'<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{% generation %}"
    "{%- set reasoning_content = '' %}"
    "{%- if message.reasoning_content is string %}"
    "{%- set reasoning_content = message.reasoning_content %}"
    "{%- else %}"
    "{%- if '</think>' in content %}"
    "{%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}"
    "{%- set content = content.split('</think>')[-1].lstrip('\\n') %}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- if loop.index0 > ns.last_query_index %}"
    "{%- if loop.last or (not loop.last and reasoning_content) %}"
    "{{ '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}"
    "{%- else %}"
    "{{ '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}" # modified to include thinking content
    "{%- endif %}"
    "{%- else %}"
    "{{ '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}" # modified to include thinking content
    "{%- endif %}"
    "{%- if message.tool_calls %}"
    "{%- for tool_call in message.tool_calls %}"
    "{%- if (loop.first and content) or (not loop.first) %}"
    "{{'\\n' }}"
    "{%- endif %}"
    "{%- if tool_call.function %}"
    "{%- set tool_call = tool_call.function %}"
    "{%- endif %}"
    "{{'<tool_call>\\n{\"name\": \"' + tool_call.name + '\", \"arguments\": ' + (tool_call.arguments if tool_call.arguments is string else (tool_call.arguments | tojson)) + '}\\n</tool_call>' }}"
    "{%- endfor %}"
    "{%- endif %}"
    "{{'<|im_end|>' }}"
    "{% endgeneration %}"
    "{{'\n' }}"
    "{%- elif message.role == 'tool' %}"
    "{%- if loop.first or (messages[loop.index0 - 1].role != 'tool') %}"
    "{{'<|im_start|>user' }}"
    "{%- endif %}"
    "{{'\\n<tool_response>\\n' + content + '\\n</tool_response>' }}"
    "{%- if loop.last or (messages[loop.index0 + 1].role != 'tool') %}"
    "{{'<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{'<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{'<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
)

def convert_messages_to_tokens(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    chat_template: str | None = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
    tools: list[dict] | None = None
    ) -> list[list[int]]:
    """
    Convert a list of messages to a list of tokens.
    """
    token_list = []
    for idx, message in enumerate(messages):
        if isinstance(message['content'], list):
            message['content'] = message['content'][0]['text']
        if message['role'] == 'system':
            # add_generation_prompt should be always False for system message
            token_ids = tokenizer.apply_chat_template(
                [message],
                add_generation_prompt=False,
                tokenize=True,
                enable_thinking=enable_thinking,
                tools=tools,
                chat_template=chat_template
                )
        elif idx == len(messages) - 1:
            # Only add generation prompt only for the last message
            # We do not pass tools to avoid system message added
            token_ids = tokenizer.apply_chat_template(
                [message],
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                enable_thinking=enable_thinking,
                chat_template=chat_template
                )
        else:
            # We do not pass tools to avoid system message added
            token_ids = tokenizer.apply_chat_template(
                [message],
                add_generation_prompt=False,
                tokenize=True,
                enable_thinking=enable_thinking,
                chat_template=chat_template
                )
        token_list.append(token_ids)
    return token_list

def parse_response_ids(
    response_ids: list[int],
    tokenizer: AutoTokenizer
) -> dict[str, Any]:
    if tokenizer.eos_token_id not in response_ids:
        format_error_type = 'length'
    else:
        format_error_type = None
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    split_idx = text.find("<tool_call>")
    # No tool calls
    if split_idx == -1 or format_error_type is not None:
        return {'content': text, 'tool_calls': [], 'format_error_type': format_error_type}
    
    content = text[:split_idx]
    tool_text = text[split_idx:]

    if tool_text.count("<tool_call>") != tool_text.count("</tool_call>"):
        format_error_type = 'tool_call'

    try:
        tool_call_strings = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", tool_text, re.DOTALL)
        tool_calls = [json.loads(tc) for tc in tool_call_strings]
    except Exception as e:
        tool_calls = []
        format_error_type = 'tool_call'
    
    format_tool_calls = []
    for tool_call in tool_calls:
        format_tool_calls.append(
            {
                'id': f'{uuid.uuid4()}',
                'type': 'function',
                'function': tool_call,
            }
        )
    return {'content': content, 'tool_calls': format_tool_calls, 'format_error_type': format_error_type}

def request_response_tokens(
    tokenizer: AutoTokenizer,
    base_url: str | None,
    timeout: int | None,
    top_p: float,
    seed: int | None,
    max_model_len: int,
    **kwargs,
) -> ModelResponse:

    input_ids = []
    #for message in kwargs['messages']:
    #    input_ids.extend(message)

    messages = kwargs['messages']
    pending_messages = []
    for message in reversed(messages):
        if message['output_ids'] is not None:
            if message['input_ids'] is not None:
                input_ids.extend(message['input_ids'])
            input_ids.extend(message['output_ids'])
            # we break here because all messages before this message have been processed
            break
        else:
            new_message = dict(message)
            new_message.pop('input_ids', None)
            new_message.pop('output_ids', None)
            pending_messages.append(new_message)

    chat_template_kwargs = kwargs.pop('chat_template_kwargs', {'enable_thinking': True})
    
    pending_messages_ids = convert_messages_to_tokens(
        list(reversed(pending_messages)), 
        tokenizer, 
        chat_template=qwen3_chat_template, # you might want to modify this if you are using a different model
        add_generation_prompt=True,
        enable_thinking=chat_template_kwargs['enable_thinking'],
        tools=kwargs.pop('tools', [])
    )
    for msg_ids in pending_messages_ids:
        input_ids.extend(msg_ids)

    kwargs.pop('messages')
    model = kwargs.pop('model', None)

    extra_body = kwargs.pop('extra_body', {})
    random_id = f"chatcmpl-{uuid.uuid4()}"

    if 'max_completion_tokens' in kwargs:
        kwargs['max_tokens'] = kwargs.pop('max_completion_tokens')
    # We ignore max_completion_tokens and instead use maximum available tokens left.
    kwargs['max_tokens'] = max_model_len - len(input_ids)

    if kwargs['max_tokens'] <= 0:
        raise ContextWindowExceededError(
            message=f'input length and `max_tokens` exceed context limit. input length: {len(input_ids)}, max_tokens: {kwargs["max_tokens"]}, max_model_len: {max_model_len}.',
            model=model,
            llm_provider='hosted_vllm',
        )
    
    resp = httpx.post(
        url=f'{base_url}/generate',
        json={
            'prompt_ids': input_ids,
            'top_p': top_p,
            'seed': seed,
            **kwargs,
        },
        timeout=timeout,
    )
    # Raise error if the request is not successful
    resp.raise_for_status()
    resp = resp.json()

    if 'response_ids' not in resp:
        raise ValueError(f'Response does not contain response_ids: {resp}')

    try:
        response_ids = resp['response_ids']
        if 'logprobs' in resp:
            logprobs = resp['logprobs']
        else:
            logprobs = None
        result =  parse_response_ids(response_ids, tokenizer)
        result = {
            'input_ids': input_ids,
            'output_ids': response_ids,
            'logprobs': logprobs,
            'message': result,
        }
        model_response = ModelResponse(
            model=model,
            id = random_id,
            choices=[result],
            role='assistant',
        )
        return model_response
    except Exception as e:
        raise ValueError(f'Failed to parse response: {e}')
