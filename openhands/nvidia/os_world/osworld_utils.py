# type: ignore
import time
import os
import pandas as pd
import numpy as np
import asyncio
from evaluation.utils.shared import (  # type: ignore
    EvalMetadata,
    get_default_sandbox_config_for_eval,
    update_llm_config_for_completions_logging,
    EvalException,
)
from pathlib import Path

from openhands.agenthub.gui_agent.osworld_agent import OSWorldAgent
from openhands.core.config.llm_config import LLMConfig
from openhands.runtime.base import Runtime
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config import (
    AgentConfig,
    OpenHandsConfig,
    SandboxConfig,
)
from openhands.core.setup import create_agent, create_controller, generate_sid
from openhands.storage import get_file_store
from openhands.events import EventStream
from openhands.runtime.impl.singularity.osworld_singularity_runtime import (
    OSWorldSingularityRuntime,
)
from openhands.controller.state.state import State
from openhands.events.action import CmdRunAction, IPythonRunCellAction, MessageAction
from openhands.events.observation import CmdOutputObservation
from openhands.nvidia.logger import nvidia_logger as logger
from openhands.core.logger import openhands_logger as openhands_logger
from evaluation.utils.shared import codeact_user_response, is_fatal_evaluation_error

from openhands.nvidia.utils import is_last_action_finish
import json
from openhands.nvidia.registry import JobDetails, _DEFAULT_AGENT_CONFIG
from openhands.nvidia.utils import get_instance_id
from openhands.nvidia.controller import run_controller_with_controller

from openhands.nvidia.os_world.controllers.setup import SetupController
from openhands.nvidia.os_world.evaluate import Evaluator



def get_config(
    instance: dict,
    metadata: EvalMetadata,
    agent_config: dict=_DEFAULT_AGENT_CONFIG,
) -> OpenHandsConfig:
    # Premade Singularity image for OSWorld

    sandbox_config = SandboxConfig(
        base_container_image='ubuntu:24.04',
        run_as_fakeroot=True,
    )

    config = OpenHandsConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        max_iterations=metadata.max_iterations,
        runtime='osworld',
        sandbox=sandbox_config,
        # do not mount workspace
        workspace_base=None,
        workspace_mount_path=None,
    )
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, get_instance_id(instance)
        )
    )

    # https://github.com/All-Hands-AI/OpenHands/blob/main/openhands/core/config/agent_config.py
    # Turn everything off for OSWorld
    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_editor=False,
        enable_cmd=False,
        enable_browsing=False,
        enable_llm_editor=False,
        enable_mcp=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
        enable_think=False, 
        enable_history_truncation=False, # turn off history truncation
        ensure_thinking_end_properly=agent_config['ensure_thinking_end_properly'], # set to true only if using text based server for training.
        action_timeout=30.0, # 30 seconds per action
        strict_loop_detector=agent_config['strict_loop_detector'], # set to true only if training
        enable_vision=False,
        enable_a11y_tree=True,
    )
    config.set_agent_config(agent_config)
    return config

def get_instruction(instance: pd.Series | dict, metadata: EvalMetadata, runtime: Runtime) -> MessageAction:
    """
    We keep all information here. screenshot and a11y tree will be processed in agent.
    """

    include_screenshot = True #runtime.config.agents['agent'].enable_vision
    include_a11y_tree = True #runtime.config.agents['agent'].enable_a11y_tree
    instruction = f"""Work on the following task accourding to the UI screenshot.

Instruction: {instance['instruction']}
"""
    
    if include_a11y_tree:
        accessibility_tree = runtime.get_vm_accessibility_tree()

    image_url = None
    if include_screenshot:
        image = runtime.get_vm_screenshot()
        if image:
            image_url = [f'data:image/png;base64,{image}']

    return MessageAction(content=instruction, image_urls=image_url, accessibility_tree=accessibility_tree)

def create_runtime(config: OpenHandsConfig, sid: str | None = None) -> Runtime:
    vm_image_path = os.getenv('OSWORLD_VM_IMAGE_PATH', './OS_images/Ubuntu.qcow2')
    assert Path(vm_image_path).exists(), f"ERROR: VM image not found at {vm_image_path}. Export OSWORLD_VM_IMAGE_PATH."
    logger.info(f"Using VM image path: {vm_image_path}")

    session_id = sid or generate_sid(config)

    file_store = get_file_store(config.file_store, config.file_store_path)
    event_stream = EventStream(session_id, file_store)
    runtime = OSWorldSingularityRuntime(
        config=config,
        event_stream=event_stream,
        sid=session_id,
        os_type='linux',
        vm_image_path=vm_image_path,
        attach_to_existing=False,
    )

    return runtime

async def initialize_agents(
        instance: dict,
        llm_config: LLMConfig | None = None,
        sid:str | None = None,
        eval_output_dir:str = "/root",
        git_commit:str = "9f93e8a1532d6e1da4ea702f3dbd31d0f6b2fb3a",
        dataset:str = "deepscaler",
        data_split:str = "train",
        agent_config: dict = dict(_DEFAULT_AGENT_CONFIG),
    ) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:

    if llm_config is None:
        raise ValueError('LLM config is None, cannot initialize.')

    metadata = EvalMetadata(
        agent_class="OSWorldAgent",
        llm_config=llm_config,
        agent_config=None,
        max_iterations=agent_config['max_iterations'],
        eval_output_dir=eval_output_dir,
        start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        git_commit=git_commit,
        dataset=dataset,
        data_split=data_split,
        details=None,
        condenser_config=NoOpCondenserConfig(),
    )


    config = get_config(instance, metadata, agent_config)

    runtime = create_runtime(config, sid=sid)

    await runtime.connect()
    logger.debug(f"Runtime connected {runtime.sid}")


    try:
        await initialize_runtime(runtime, instance, metadata)

    except Exception as e:
        logger.error(f"Error initializing runtime: {e}")
        raise e
    return runtime, metadata, config

async def initialize_runtime(runtime: Runtime, instance: dict, metadata: EvalMetadata):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    openhands_logger.info(f'{"-" * 50} BEGIN Runtime Initialization Fn {"-" * 50}')
    # create cache directory
    cache_dir = f"/tmp/osworld_cache_{runtime.sid}"
    os.makedirs(cache_dir, exist_ok=True)
    logger.debug(f"Created cache directory: {cache_dir}")

    runtime.setup_controller = SetupController(
        vm_ip="127.0.0.1",
        server_port=runtime._vm_server_port,
        chromium_port=runtime._chromium_port,
        cache_dir="/tmp/osworld_example",
        client_password="password",
        runtime=runtime  # Pass your runtime object here
    )

    await runtime.setup_controller.setup(instance['config'])
    openhands_logger.info(f'{"-" * 50} END Runtime Initialization Fn {"-" * 50}')

async def run_agent(
        job_details: JobDetails,
        sid: str | None = None,
    ) -> dict[str, object]:
    runtime = job_details.runtime
    metadata = job_details.metadata
    config = job_details.config
    instance = job_details.instance

    message_action = get_instruction(instance, metadata, runtime)
    try:
        agent = create_agent(config)
        job_details.agent = agent
        controller, initial_state = create_controller(
            agent=agent,
            runtime=runtime,
            config=config,
            replay_events=None,
        )
        job_details.controller = controller
        state: State | None = await run_controller_with_controller(
                config=config,
                initial_user_action=message_action,
                sid=sid,
                runtime=runtime,
                agent=agent,
                fake_user_response_fn=codeact_user_response,
                controller=controller,
                initial_state=initial_state,
            )
        # if fatal error, throw EvalError to log.
        if state is None:
            raise EvalException('Final state is None')
        if is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        logger.error(f"Error running agent: {e}")

    # get messages from agent history
    try:
        run_results = process_messages_from_agent_state(agent, state, job_details) # type: ignore
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        logger.error(f"Error while running, failed to retrieve agent messages: {e}")
        raise Exception(f"Failed to retrieve agent messages: {str(e)}")

    return {
        'success': not bool(state.last_error if state else True),
        'error': state.last_error if state and state.last_error else None,
        'finish': is_last_action_finish(state),
        **run_results,
    }

async def evaluate_agent(run_results: dict, instance: dict, runtime: Runtime):
    try:
        evaluator = Evaluator(instance, runtime.setup_controller)
        score = await evaluator.evaluate(run_results['messages'])
        if score > 0.99:
            return {'resolved': True, 'reward': score}
        return {'resolved': False, 'reward': score}
    except:
        return {'resolved': False, 'reward': 0}

def process_messages_from_agent_state(
    agent: OSWorldAgent,
    state: State,
    job_details: JobDetails | None = None,
) -> dict:
    """
    This has been modified for OSWorldAgent to account for vision input.
    We removed logic related to <think> and </think> tags.
    The content logic for assistant turns will always contain 3 items:
    - a text item with the instruction (this might have accessibility tree already embedded)
    - a image item with the screenshot
    - a text item with the accessibility tree
    
    logic for token_level_generation has not been checked or tested. Not supported for OSWorldAgent at the moment.
    """
    if job_details is not None:
        assert job_details.llm_config is not None, (
            'llm_config is required in job_details.'
        )
        token_level_generation = job_details.llm_config.token_level_generation
        assert token_level_generation is False, 'token_level_generation is not supported for OSWorldAgent at the moment.'
    else:
        logger.warning(
            'No job_details provided in process_messages_from_agent_state. Assuming token_level_generation is False.'
        )
        token_level_generation = False

    initial_user_message = agent._get_initial_user_message(state.history)
    messages = agent._get_messages_from_agent_state(state.history, initial_user_message)

    while len(messages) > 0 and messages[-1]['role'] != 'assistant':
        messages = messages[:-1]

    tools = agent.tools
    return {
        'problem_id': job_details.instance.get('id', None),
        'messages': messages,
        'tools': tools,
        'end_properly': not state.get_last_agent_format_error(),
    }

###############################################################################
# Begin of exception handling
# Used to override default process_messages_from_agent_state for OSWorldAgent
###############################################################################
def get_messages_from_partial_result(job_details: JobDetails) -> dict:
    if job_details.agent is None or job_details.controller is None:
        return {'messages': [], 'tools': [], 'end_properly': True}
    controller = job_details.controller
    state = controller.get_state()
    assert state is not None, (
        'Error in get_messages_from_partial_result: state is None.'
    )
    return process_messages_from_agent_state(job_details.agent, state, job_details)

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
###############################################################################
# End of exception handling
###############################################################################