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

from openhands.nvidia.utils import process_messages_from_agent_state, is_last_action_finish, get_messages_from_partial_result
import json
from openhands.nvidia.reward import Reward
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
    )
    config.set_agent_config(agent_config)
    return config

def get_instruction(instance: pd.Series | dict, metadata: EvalMetadata) -> MessageAction:

    def obtain_problem_statement(instance: dict) -> str:
        if isinstance(instance['prompt'], list):
            problem_statement = instance['prompt'][0]['content']
        elif isinstance(instance['prompt'], str):
            problem_statement = json.loads(instance['prompt'])[0]['content']
        else:
            raise ValueError(f'Invalid prompt type: {type(instance["prompt"])}')
        # Remove boxed instructions from problem statement
        if " Let's think step by step and output the final answer within \\boxed{}." in problem_statement:
            problem_statement = problem_statement.replace(" Let's think step by step and output the final answer within \\boxed{}.", "")
        return problem_statement

    instruction = f"""
Your task is to solve challenging math problems using the `execute_ipython_cell` tool, which gives you access to a full IPython environment. You are allowed and expected to use code to explore, solve, and verify your answers.

Environment:
- Libraries already imported: `math`, `cmath`, `numpy`, `sympy`, `scipy`
- You can also install additional libraries using `%pip install <library>` if necessary.

Instructions:
1. Read and understand the problem statement. Fist use the `think` tool to log down your thoughts and plan for solving the problem.
2. Plan your solution using a combination of reasoning and code. Always try to solve the problem using code. Also plan about how to verify your solution.
3. In subsequent steps after planning, use tools to execute your plan. Use `execute_ipython_cell` to run calculations, manipulate symbols, or perform verification.
4. Use code to verify your answer. If your answer is not correct, iterate your plan with the `think` tool and continue solving the problem until you are confident in your answer is correct.
5. Finally, when you are confident in your answer:
    - Only call the `finish` tool if you are confident in your answer and you have verified your answer with the `execute_ipython_cell` tool.
    - Terminate the conversation by calling the `finish` tool.
    - Put your final answer within \\boxed{{}} in the message with the `finish` tool.
    

Important Guidelines:
- Always first use the `think` tool to log down your thoughts and plan for solving the problem.
- Always try to use the `execute_ipython_cell` tool to solve the problem, especially for calculations, symbolic reasoning, or simulations.
- Always verify your answer with code and the `execute_ipython_cell` tool.
- Only use the `finish` tool is you are confident the answer is correct. If you think there is a mistake, iterate your plan with the `think` tool and continue solving the problem.
- Put your final answer within \\boxed{{}} in the message with the `finish` tool.

Now begin solving the following problem:
{obtain_problem_statement(instance)}
"""
    return MessageAction(content=instruction)

def create_runtime(config: OpenHandsConfig, sid: str | None = None) -> Runtime:
    vm_image_path = os.getenv('OSWORLD_VM_IMAGE_PATH', './OS_images/Ubuntu.qcow2')
    assert Path(vm_image_path).exists(), f"ERROR: VM image not found at {vm_image_path}"
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
        agent_class="CodeActAgent",
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

    message_action = get_instruction(instance, metadata)
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
        logger.error(f"Error running agent: {e}")

    # get messages from agent history
    try:
        run_results = process_messages_from_agent_state(agent, state, job_details) # type: ignore
    except Exception as e:
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
        score = await evaluator.evaluate()
        if score > 0.99:
            return {'resolved': True, 'reward': score}
        return {'resolved': False, 'reward': score}
    except:
        return {'resolved': False, 'reward': 0}
