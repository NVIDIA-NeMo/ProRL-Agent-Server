import asyncio
import copy
import json
import queue
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
from pydantic import BaseModel
from transformers import AutoTokenizer

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.state.state import State
from openhands.events.action import (
    Action,
    AgentFinishAction,
)
from openhands.llm.nvidia.qwen3 import (
    convert_messages_to_tokens,
    qwen3_chat_template,
)
from openhands.nvidia.logger import nvidia_logger as logger
from openhands.nvidia.registry import JobDetails


def clear_queue(q: queue.Queue):
    """Clear all items from a queue and properly mark them as done.

    This function removes all items from the queue and calls task_done()
    for each item to maintain proper queue state for join() operations.
    """
    items_cleared = 0
    while not q.empty():
        try:
            q.get_nowait()
            try:
                q.task_done()  # Mark the task as done
                items_cleared += 1
            except ValueError as e:
                # task_done() called too many times - this shouldn't happen but handle gracefully
                logger.warning(
                    f'task_done() called too many times while clearing queue: {e}'
                )
                items_cleared += 1  # Still count the item as cleared
        except queue.Empty:
            # Queue became empty between qsize check and get_nowait
            break

    if items_cleared > 0:
        logger.debug(f'Cleared {items_cleared} items from queue')


def get_singularity_job_pids():
    all_pids = set()
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'openhands'], stdout=subprocess.PIPE, text=True, check=True
        )
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            if pid.strip():
                all_pids.add(pid.strip())

        result = subprocess.run(
            ['pgrep', '-f', 'openhands'],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            if pid.strip():
                all_pids.add(pid.strip())
    except subprocess.CalledProcessError:
        pass
    return all_pids


def kill_all_singularity_jobs(exclude_pids: set[str] | None = None):
    if exclude_pids is None:
        exclude_pids = set()
    try:
        # List all singularity-related processes
        result = subprocess.run(
            ['pgrep', '-f', 'apptainer'], stdout=subprocess.PIPE, text=True, check=True
        )
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            clean_pid = pid.strip()
            if clean_pid and clean_pid not in exclude_pids:
                subprocess.run(['kill', '-9', clean_pid])
                logger.info(f'Killed Singularity process with PID: {clean_pid}')
    except subprocess.CalledProcessError:
        logger.info('No Singularity processes found.')

    try:
        # List all openhands processes
        result = subprocess.run(
            ['pgrep', '-f', 'openhands'],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            clean_pid = pid.strip()
            if clean_pid and clean_pid not in exclude_pids:
                subprocess.run(['kill', '-9', clean_pid])
                logger.info(f'Killed Openhands process with PID: {clean_pid}')
    except subprocess.CalledProcessError:
        logger.info('No Openhands processes found.')


# Custom exceptions
class ServerNotRunningError(Exception):
    pass


class NoLLMServerError(Exception):
    pass


class JobTimeoutError(Exception):
    pass


class ProcessRequest(BaseModel):
    instance: dict[str, Any]
    sampling_params: dict[str, Any]
    job_id: str | None = None


class CancelRequest(BaseModel):
    job_id: str


class LLMServerRequest(BaseModel):
    address: str


async def cleanup_timed_out_job(server, job_id: str | None = None):
    """Clean up a job that has timed out or failed"""
    if job_id is None:
        return

    if not hasattr(server, '_job_details'):
        return

    if job_id in server._job_details:
        job_details = server._job_details[job_id]

        # Set timeout error flag
        job_details.timeout_error = True

        # Set the event to unblock any waiting threads
        if job_details.event is not None:
            job_details.event.set()

        # Remove from active jobs sets
        if job_id in server._active_init_jobs:
            server._active_init_jobs.remove(job_id)
        if job_id in server._active_run_jobs:
            server._active_run_jobs.remove(job_id)

        # Close runtime if it exists
        if job_details.runtime:
            try:
                job_details.runtime.close()
            except Exception:
                pass

        # Remove from job details
        if job_id in server._job_details:
            del server._job_details[job_id]


async def process_with_timeout(
    server,
    instance: pd.Series,
    sampling_params: dict[str, Any],
    timeout: float,
    thread_pool: ThreadPoolExecutor | None = None,
    job_id: str | None = None,
):
    if job_id is None:
        job_id = server.get_unique_id(instance)
    try:
        # Create a future for the process call

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            thread_pool,
            lambda: server.process(instance, sampling_params, job_id, timeout),
        )
        result = await future
        return result
    except asyncio.TimeoutError:
        # Clean up the timed-out job
        logger.error(f'Job {job_id} timed out after {timeout} seconds')
        await cleanup_timed_out_job(server, job_id)
        raise JobTimeoutError(f'Job {job_id} timed out after {timeout} seconds')
    except Exception as e:
        # Clean up on any other error
        logger.error(f'Job {job_id} failed with error: {str(e)}')
        await cleanup_timed_out_job(server, job_id)
        raise


def is_last_action_finish(state: State) -> bool:
    if state and state.history:
        last_action = next(
            (event for event in reversed(state.history) if isinstance(event, Action)),
            None,
        )
        if isinstance(last_action, AgentFinishAction):
            return True
    return False


def process_messages_from_agent_state(
    agent: CodeActAgent,
    state: State,
    job_details: JobDetails | None = None,
) -> dict[str, Any]:
    """
    Process the messages from the agent state. We reuse logic from CodeActAgent to process state.history into litellm messages.
    We then format the messages for the LLM back to huggingface format (recognized by hf tokenizer).
    Currently Qwen3 models expect the assistant message to have a <think> tag and a </think> tag, but sometimes the agent does not end properly reasoning properly.
    We then format the message to be consistent with the expected output from Qwen3 models.

    Example of Qwen3 model that did not end properly reasoning properly:
    <think>
    <tool_call>
    {tool_call}
    </tool_call>
    ...

    We obtains the following message: {'role': 'assistant', 'content': '<think>\n', 'tool_calls': [tool_call]}

    The above breaks the hf tokenizer because it does not recognize the <think> tag.
    We thus compress all tool calls into content and append a </think> tag: {'role': 'assistant', 'content': '<think>\n{tool_call}\n{tool_call}\n</tool_call>\n</think>'}
    """
    if job_details is not None:
        assert job_details.llm_config is not None, (
            'llm_config is required in job_details.'
        )
        token_level_generation = job_details.llm_config.token_level_generation
        enable_thinking = job_details.llm_config.enable_thinking
        if token_level_generation:
            model_name = job_details.llm_config.custom_tokenizer
            # Default to Qwen3-8B if custom_tokenizer is not set.
            if model_name is None:
                model_name = 'Qwen/Qwen3-4B-Instruct-2507'
            else:
                assert 'qwen3' in model_name.lower(), (
                    'token_level_generation is only supported for Qwen3 models.'
                )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            chat_template = qwen3_chat_template
    else:
        logger.warning(
            'No job_details provided in process_messages_from_agent_state. Assuming token_level_generation is False.'
        )
        token_level_generation = False

    initial_user_message = agent._get_initial_user_message(state.history)
    raw_messages = agent._get_messages(state.history, initial_user_message)

    if raw_messages[-1].role != 'assistant':
        raw_messages = raw_messages[:-1]

    # patch the last message if it is an AgentFinishAction
    if is_last_action_finish(state):
        tool_metadata = state.history[-1].tool_call_metadata
        if tool_metadata is not None:
            assistant_msg = getattr(tool_metadata.model_response.choices[0], 'message')
            raw_messages[-1].tool_calls = assistant_msg.tool_calls
    messages = agent.llm.format_messages_for_llm(raw_messages)

    while len(messages) > 0 and messages[-1]['role'] != 'assistant':
        messages = messages[:-1]

    from openhands.llm.llm_utils import check_tools

    tools = check_tools(agent.tools, agent.llm.config)

    new_messages = []
    for message in messages:
        new_message = {'role': message['role']}
        if isinstance(message['content'], str):
            new_message['content'] = message['content']
        elif isinstance(message['content'], list):
            if len(message['content']) == 0:
                new_message['content'] = ''  # empty string for default
            else:
                new_message['content'] = message['content'][0]['text']
        else:
            raise RuntimeError(
                f'Message content is not a string or list: {message["content"]}'
            )
        if 'tool_calls' in message:
            new_message['tool_calls'] = [
                tool_call['function'] for tool_call in message['tool_calls']
            ]

        # Update token_ids, currently this only works for Qwen3 models.
        if token_level_generation:
            input_ids = message.get('input_ids', None)
            output_ids = message.get('output_ids', None)
            logprobs = message.get('logprobs', None)
            if output_ids is not None:
                new_message['token_ids'] = output_ids
            else:
                new_message['token_ids'] = convert_messages_to_tokens(
                    [new_message],
                    tokenizer,
                    chat_template=chat_template,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                    tools=tools,
                )[0]
            new_message['input_ids'] = input_ids
            new_message['logprobs'] = logprobs

        # Handle the case where the agent did not end properly reasoning properly
        # This largely formats the message to be consistent with the expected output from Qwen3 models.
        if (
            job_details is not None
            and job_details.agent_config['ensure_thinking_end_properly']
            and message['role'] == 'assistant'
            and '<think>' in new_message['content']
            and '</think>' not in new_message['content']
        ):
            if 'tool_calls' in new_message:
                tool_calls_message = ''
                for tool_call in new_message['tool_calls']:
                    current_tool_call = []
                    current_tool_call.append(
                        '\n<tool_call>\n{"name": "'
                        + tool_call['name']
                        + '", "arguments": '
                    )
                    if isinstance(tool_call['arguments'], str):
                        current_tool_call.append(tool_call['arguments'])
                    else:
                        current_tool_call.append(json.dumps(tool_call['arguments']))
                    current_tool_call.append('}\n</tool_call>')
                    tool_calls_message += ''.join(current_tool_call)
                new_message['content'] = (
                    f'{new_message["content"]}\n{tool_calls_message}</think>'
                )
                new_message.pop('tool_calls')
            new_messages.append(new_message)
            return {
                'messages': new_messages,
                'tools': tools,
                'end_properly': False,
            }

        new_messages.append(new_message)
    return {
        'messages': new_messages,
        'tools': tools,
        'end_properly': not state.get_last_agent_format_error(),
    }


def get_messages_from_partial_result(job_details: JobDetails) -> dict[str, Any]:
    if job_details.agent is None or job_details.controller is None:
        return {'messages': [], 'tools': [], 'end_properly': True}
    controller = job_details.controller
    state = controller.get_state()
    assert state is not None, (
        'Error in get_messages_from_partial_result: state is None.'
    )
    return process_messages_from_agent_state(job_details.agent, state, job_details)


def get_instance_id(instance: dict) -> str:
    if 'instance_id' in instance:
        if instance['instance_id'] is not None:
            return instance['instance_id']
    data_source = instance.get('data_source', 'unknown')
    if 'extra_info' in instance:
        split = instance['extra_info'].get('split', 'unknown')
        index = instance['extra_info'].get('index', 'unknown')
        name = instance['extra_info'].get('name', 'unknown')
    else:
        split = 'unknown'
        index = 'unknown'
        name = 'unknown'
    return f'{data_source}_{name}_{split}_{index}'


def initialize_exception(job_details: JobDetails, e: Exception):
    tb = traceback.format_exc()
    instance_id = (
        job_details.instance.get('instance_id', None)
        if job_details.instance is not None
        else None
    )
    trajectory_id = (
        job_details.instance.get('trajectory_id', None)
        if job_details.instance is not None
        else None
    )
    return {
        'instance_id': instance_id,
        'trajectory_id': trajectory_id,
        'git_patch': None,
        'success': False,
        'error': f'Error in init: {str(e)}',
        'traceback': tb,
        'finish': False,
        'messages': [],
        'tools': [],
        'end_properly': False,
        'resolved': False,
        'critical_error': 'init',
    }


def run_exception(job_details: JobDetails, e: Exception):
    tb = traceback.format_exc()
    instance_id = (
        job_details.instance.get('instance_id', None)
        if job_details.instance is not None
        else None
    )
    trajectory_id = (
        job_details.instance.get('trajectory_id', None)
        if job_details.instance is not None
        else None
    )
    git_patch = (
        job_details.run_results.get('git_patch', None)
        if job_details.run_results is not None
        else None
    )
    success = (
        job_details.run_results.get('success', False)
        if job_details.run_results is not None
        else False
    )
    finish = (
        job_details.run_results.get('finish', False)
        if job_details.run_results is not None
        else False
    )
    messages = (
        job_details.run_results.get('messages', [])
        if job_details.run_results is not None
        else []
    )
    tools = (
        job_details.run_results.get('tools', [])
        if job_details.run_results is not None
        else []
    )
    end_properly = (
        job_details.run_results.get('end_properly', True)
        if job_details.run_results is not None
        else True
    )
    if len(messages) == 0:
        partial_result = get_messages_from_partial_result(job_details)
        messages = partial_result['messages']
        tools = partial_result['tools']
        end_properly = partial_result['end_properly']
    return {
        'instance_id': instance_id,
        'trajectory_id': trajectory_id,
        'git_patch': git_patch,
        'success': success,
        'error': f'Error in run agent: {str(e)}',
        'traceback': tb,
        'finish': finish,
        'messages': messages,
        'tools': tools,
        'end_properly': end_properly,
        'resolved': False,
        'critical_error': 'run',
    }


def eval_exception(job_details: JobDetails, e: Exception):
    tb = traceback.format_exc()
    instance_id = (
        job_details.instance.get('instance_id', None)
        if job_details.instance is not None
        else None
    )
    trajectory_id = (
        job_details.instance.get('trajectory_id', None)
        if job_details.instance is not None
        else None
    )
    git_patch = (
        job_details.run_results.get('git_patch', None)
        if job_details.run_results is not None
        else None
    )
    success = (
        job_details.run_results.get('success', False)
        if job_details.run_results is not None
        else False
    )
    finish = (
        job_details.run_results.get('finish', False)
        if job_details.run_results is not None
        else False
    )
    messages = (
        job_details.run_results.get('messages', [])
        if job_details.run_results is not None
        else []
    )
    tools = (
        job_details.run_results.get('tools', [])
        if job_details.run_results is not None
        else []
    )
    end_properly = (
        job_details.run_results.get('end_properly', True)
        if job_details.run_results is not None
        else True
    )
    if len(messages) == 0:
        partial_result = get_messages_from_partial_result(job_details)
        messages = partial_result['messages']
        tools = partial_result['tools']
        end_properly = partial_result['end_properly']
    return {
        'instance_id': instance_id,
        'trajectory_id': trajectory_id,
        'git_patch': git_patch,
        'success': success,
        'error': f'Error in eval: {str(e)}',
        'traceback': tb,
        'finish': finish,
        'messages': messages,
        'tools': tools,
        'end_properly': end_properly,
        'resolved': False,
        'critical_error': 'eval',
    }


def final_result(job_details: JobDetails):
    if job_details.results is None:
        if job_details.run_results is None:
            partial_result = get_messages_from_partial_result(job_details)
            messages = partial_result['messages']
            tools = partial_result['tools']
            end_properly = partial_result['end_properly']
            result = {
                'resolved': job_details.eval_results['resolved']
                if job_details.eval_results is not None
                else False,
                'critical_error': 'timeout' if job_details.timeout_error else None,
                'messages': messages,
                'tools': tools,
                'end_properly': end_properly,
            }
        else:
            result = {
                **job_details.run_results,
                'resolved': job_details.eval_results['resolved']
                if job_details.eval_results is not None
                else False,
                'critical_error': 'timeout' if job_details.timeout_error else None,
            }
    else:
        result = copy.deepcopy(job_details.results)
        if job_details.timeout_error:
            result['critical_error'] = 'timeout'

    if 'instance_id' not in result:
        instance_id = (
            job_details.instance.get('instance_id', None)
            if job_details.instance is not None
            else None
        )
        result['instance_id'] = instance_id

    if 'trajectory_id' not in result:
        trajectory_id = (
            job_details.instance.get('trajectory_id', None)
            if job_details.instance is not None
            else None
        )
        result['trajectory_id'] = trajectory_id

    # Determine if to filter the trajectory
    # Should filter if:
    # 1. Environment error (critical_error is not None)
    # 2. Length exceeded (normally AgentFormatError)
    def determine_filter(result: dict) -> bool:
        if 'critical_error' in result and result['critical_error'] is not None:
            return True
        if 'success' in result and result['success']:
            return False
        # Waive AgentToolCallError they should be trained
        if 'error' in result:
            for normal_error in ['AgentToolCallError']:
                if normal_error in result['error']:
                    return False
        return True

    result['filter'] = determine_filter(result)
    return result
