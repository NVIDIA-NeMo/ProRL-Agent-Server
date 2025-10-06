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
)
from openhands.core.main import create_runtime
from openhands.core.setup import create_agent, create_controller
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


def get_config(
    instance: dict,
    metadata: EvalMetadata,
    agent_config: dict=_DEFAULT_AGENT_CONFIG,
) -> OpenHandsConfig:
    # Docker image from xingyaoww/od-eval-logic-reasoning:v1.0
    base_container_image = 'xingyaoww_od-eval-logic-reasoning'
    logger.debug(
        f'Using instance container image: {base_container_image}. '
        f'Please make sure this image exists. '
        f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
    )

    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.runtime_container_image = base_container_image
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False
    # Add platform to the sandbox config to solve issue 4401
    sandbox_config.platform = 'linux/amd64'

    # run as fakeroot
    # Currently set to False as some container require GLIBC_2.38
    sandbox_config.run_as_fakeroot = False

    # Disable browser, stops openhands from spawning 100+ threads
    sandbox_config.browsergym_eval_env = 'skip'

    config = OpenHandsConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        max_iterations=metadata.max_iterations,
        runtime='singularity',
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
    # We only enable jupyter ipython for now.
    agent_config = AgentConfig(
        enable_jupyter=True,
        enable_editor=False,
        enable_cmd=False,
        enable_browsing=False,
        enable_llm_editor=False,
        enable_mcp=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
        enable_think=True, # enable think tool now for instruct models
        enable_history_truncation=False, # turn off history truncation
        ensure_thinking_end_properly=agent_config['ensure_thinking_end_properly'], # set to true only if using text based server for training.
        action_timeout=30.0, # 30 seconds per action
        strict_loop_detector=agent_config['strict_loop_detector'], # set to true only if training
    )
    config.set_agent_config(agent_config)
    # Mount OpenHands source directory if provided via OVERWRITE_OPENHANDS_DIR
    mount_dir_env = os.environ.get("OVERWRITE_OPENHANDS_DIR", "").strip()
    if mount_dir_env:
        try:
            mount_path = Path(mount_dir_env).expanduser().resolve()
            if mount_path.exists() and mount_path.is_dir():
                config.sandbox.volumes = f"{mount_path}:/openhands/code:ro"
            else:
                logger.warning(
                    f"OVERWRITE_OPENHANDS_DIR is not a valid directory: {mount_dir_env}. Skipping mount."
                )
        except Exception as e:
            logger.error(
                f"Failed to set mount from OVERWRITE_OPENHANDS_DIR='{mount_dir_env}': {e}. Skipping mount."
            )
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
        initialize_runtime(runtime, instance, metadata)

    except Exception as e:
        logger.error(f"Error initializing runtime: {e}")
        raise e
    return runtime, metadata, config

def initialize_runtime(runtime: Runtime, instance: dict, metadata: EvalMetadata):
    """Initialize the runtime for the agent.

    This function is called before the runtime is used to run the agent.
    """
    openhands_logger.info(f'{"-" * 50} BEGIN Runtime Initialization Fn {"-" * 50}')
    obs: CmdOutputObservation

    # Set instance id
    action = CmdRunAction(command='mkdir -p /workspace')
    action.set_hard_timeout(5)
    openhands_logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0

    action = CmdRunAction(command='cd /workspace')
    action.set_hard_timeout(5)
    openhands_logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert obs.exit_code == 0

    """
    action = IPythonRunCellAction(code='%pip install numpy scipy sympy')
    action.set_hard_timeout(90)
    openhands_logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    """

    action = IPythonRunCellAction(code='import numpy, scipy, sympy, math, cmath')
    action.set_hard_timeout(10)
    openhands_logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)

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

async def evaluate_agent(reward: Reward, run_results: dict, instance: dict):
    try:
        response = json.loads(run_results['messages'][-1]['tool_calls'][0]['arguments'])
        response = response['message']
        if '\\boxed' not in response:
            return {'resolved': False, 'reward': 0}

        # Remote server requires <think> tag
        if '<think>' not in response:
            response = '<think>\nfake thought\n</think>\n' + response
        
        # Send to server for evaluation
        eval_results =  await reward.get_reward(instance, response)
        return eval_results
    except:
        return {'resolved': False, 'reward': 0}
