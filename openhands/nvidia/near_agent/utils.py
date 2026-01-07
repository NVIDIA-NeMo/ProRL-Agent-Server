# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Core utilities for NeAR agent initialization, execution, and evaluation."""

import os
import time
from typing import Any, Optional

from evaluation.utils.shared import (
    EvalMetadata,
    get_default_sandbox_config_for_eval,
    update_llm_config_for_completions_logging,
)
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, OpenHandsConfig
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.main import create_runtime
from openhands.core.setup import create_agent, create_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation
from openhands.nvidia.controller import run_controller_with_controller
from openhands.nvidia.logger import nvidia_logger as logger
from openhands.nvidia.registry import JobDetails
from openhands.nvidia.reward import Reward
from openhands.nvidia.utils import (
    is_last_action_finish,
    process_messages_from_agent_state,
)
from openhands.runtime.base import Runtime

from openhands.nvidia.near_agent.workspace_setup import setup_near_workspace

# Docker image configuration
DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'nvidia/')
NEAR_CONTAINER_IMAGE = f'{DOCKER_IMAGE_PREFIX}near-runtime:latest'

logger.info(f'Using NeAR container image: {NEAR_CONTAINER_IMAGE}')

# Default agent configuration for NeAR
_DEFAULT_NEAR_AGENT_CONFIG = {
    'max_iterations': 50,  # More iterations for research tasks
    'ensure_thinking_end_properly': False,
    'strict_loop_detector': False,
}


async def initialize_agents(
    instance: dict,
    llm_config: LLMConfig | None = None,
    sid: str | None = None,
    eval_output_dir: str = '/root',
    git_commit: str = 'main',
    dataset: str = 'near_research',
    data_split: str = 'train',
    agent_config: dict | None = None,
) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:
    """Initialize NeAR agent with enhanced tooling and workspace structure.

    This function:
    1. Creates metadata with NeAR-specific config
    2. Configures sandbox with NeAR container image
    3. Sets up AgentConfig with all required tools (browser, ipython, bash, editor, MCP)
    4. Creates runtime and connects
    5. Initializes workspace structure (/workspace/, /skills/)

    Args:
        instance: Task instance with problem description
        llm_config: LLM configuration
        sid: Session ID
        eval_output_dir: Directory for evaluation outputs
        git_commit: Git commit hash
        dataset: Dataset name
        data_split: Data split (train/test)
        agent_config: Agent configuration dict

    Returns:
        tuple[Runtime, EvalMetadata, OpenHandsConfig]

    Raises:
        ValueError: If llm_config is None
        RuntimeError: If workspace initialization fails
    """
    if llm_config is None:
        raise ValueError('LLM config is None, cannot initialize.')

    if agent_config is None:
        agent_config = dict(_DEFAULT_NEAR_AGENT_CONFIG)
    else:
        # Merge with defaults
        merged_config = dict(_DEFAULT_NEAR_AGENT_CONFIG)
        merged_config.update(agent_config)
        agent_config = merged_config

    # Create metadata
    metadata = EvalMetadata(
        agent_class='CodeActAgent',
        llm_config=llm_config,
        agent_config=None,
        max_iterations=agent_config['max_iterations'],
        eval_output_dir=eval_output_dir,
        start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        git_commit=git_commit,
        dataset=dataset,
        data_split=data_split,
        details={'mode': 'near_research'},
        condenser_config=NoOpCondenserConfig(),
    )

    # Configure sandbox with NeAR container
    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.runtime_container_image = NEAR_CONTAINER_IMAGE
    sandbox_config.enable_auto_lint = False  # Not needed for research
    sandbox_config.platform = 'linux/amd64'
    sandbox_config.remote_runtime_resource_factor = 2  # More resources for research
    sandbox_config.browsergym_eval_env = 'skip'
    sandbox_config.use_host_network = False
    sandbox_config.run_as_fakeroot = False

    # Mount user-provided skills if specified
    if 'custom_skills_dir' in instance and instance['custom_skills_dir']:
        from pathlib import Path

        custom_skills = Path(instance['custom_skills_dir']).resolve()
        if custom_skills.exists():
            sandbox_config.volumes = f'{custom_skills}:/workspace/custom_skills:ro'
            logger.info(f'Mounting custom skills from: {custom_skills}')

    # Create OpenHands config
    config = OpenHandsConfig(
        default_agent=metadata.agent_class,
        run_as_openhands=False,
        max_iterations=metadata.max_iterations,
        runtime='singularity',
        sandbox=sandbox_config,
        workspace_base=None,
        workspace_mount_path=None,
    )

    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config,
            metadata.eval_output_dir,
            instance.get('instance_id', 'near_task'),
        )
    )

    # Configure agent with ALL NeAR tools
    agent_cfg = AgentConfig(
        enable_cmd=True,  # Bash execution
        enable_jupyter=True,  # IPython (stateful)
        enable_browsing=True,  # Playwright browser
        enable_editor=True,  # File editor (str_replace)
        enable_llm_editor=False,  # Don't use LLM-based editor
        enable_mcp=True,  # Tavily web search via MCP
        enable_think=True,  # Enable thinking
        enable_finish=True,  # Enable finish action
        enable_history_truncation=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=True,
        ensure_thinking_end_properly=agent_config['ensure_thinking_end_properly'],
        action_timeout=60.0,  # Longer timeout for research tasks
        strict_loop_detector=agent_config['strict_loop_detector'],
        system_prompt_template='system_prompt_near',  # Custom NeAR prompt
    )
    config.set_agent_config(agent_cfg)

    # Create and connect runtime
    runtime = create_runtime(config, sid=sid)
    await runtime.connect()
    logger.debug(f'Runtime connected {runtime.sid}')

    try:
        # Initialize NeAR workspace structure
        setup_near_workspace(runtime, instance, metadata)
    except Exception as e:
        logger.error(f'Error initializing NeAR workspace: {e}')
        raise RuntimeError(f'Failed to initialize NeAR workspace: {str(e)}') from e

    return runtime, metadata, config


def get_near_instruction(instance: dict, metadata: EvalMetadata) -> MessageAction:
    """Generate instruction for NeAR research task.

    Args:
        instance: Task instance with 'problem_statement' or 'query'
        metadata: Evaluation metadata

    Returns:
        MessageAction with formatted instruction
    """
    problem = instance.get('problem_statement') or instance.get('query', '')

    instruction = f"""You are NeAR, a deep research agent with access to a complete workspace and tools.

<task>
{problem}
</task>

Your workspace is organized as:
- /workspace/notes.md - Store your research notes and findings here
- /workspace/assets/ - Resources and data files
- /workspace/mounted/ - User-provided filesystem mounts
- /workspace/outputs/ - Save final deliverables here
- /skills/ - Directory for custom skills (currently empty)

Available tools:
1. **bash** - Execute shell commands
2. **ipython** - Stateful Python REPL (variables persist across calls)
3. **browser** - Playwright-based web automation (navigate, extract content)
4. **str_replace_editor** - Create, view, and edit files with precise string replacement
5. **mcp__tavily__search** - Search the web using Tavily API

Approach:
1. Plan your research strategy
2. Use tools systematically to gather information
3. Document findings in /workspace/notes.md as you work
4. Save final outputs to /workspace/outputs/
5. When complete, use the finish action with your summary

Important notes:
- IPython state persists: variables, imports, and functions remain available across calls
- Browser sessions persist: navigation history maintained
- Use parallel actions when possible for efficiency
- Be systematic and thorough in your research
"""

    return MessageAction(content=instruction, source='user')


async def run_agent(
    job_details: JobDetails,
    sid: str | None = None,
) -> dict[str, object]:
    """Run NeAR agent on research task.

    This function:
    1. Creates CodeActAgent with NeAR tools
    2. Creates controller
    3. Runs agent loop
    4. Collects outputs from /workspace/outputs/
    5. Returns results dict with outputs, messages, success status

    Args:
        job_details: Job details with runtime, metadata, config, instance
        sid: Session ID

    Returns:
        dict with outputs, messages, success, error

    Raises:
        Exception: If agent execution fails
    """
    runtime = job_details.runtime
    metadata = job_details.metadata
    config = job_details.config
    instance = job_details.instance

    # Get initial instruction
    message_action = get_near_instruction(instance, metadata)

    try:
        # Create agent and controller
        agent = create_agent(config)
        job_details.agent = agent

        controller, initial_state = create_controller(
            agent=agent,
            runtime=runtime,
            config=config,
            replay_events=None,
        )
        job_details.controller = controller

        # Run agent loop
        from evaluation.benchmarks.swe_bench.run_infer import (
            codeact_user_response,
        )

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
        job_details.state = state

        # Collect outputs from workspace
        outputs = collect_near_outputs(runtime)

        # Check for errors
        if state is None:
            raise Exception('Final state is None')

    except Exception as e:
        logger.error(f'Error running NeAR agent: {e}')
        outputs = {}

    # Process messages from agent history
    try:
        run_results = process_messages_from_agent_state(agent, state, job_details)
    except Exception as e:
        logger.error(f'Failed to retrieve agent messages: {e}')
        raise Exception(f'Failed to retrieve agent messages: {str(e)}') from e

    return {
        'outputs': outputs,
        'success': not bool(state.last_error if state else True),
        'error': state.last_error if state and state.last_error else None,
        'finish': is_last_action_finish(state),
        **run_results,
    }


def collect_near_outputs(runtime: Runtime) -> dict[str, str]:
    """Collect all outputs from /workspace/outputs/ directory.

    Args:
        runtime: Runtime instance

    Returns:
        dict mapping filename to content
    """
    outputs = {}

    # List files in outputs directory
    action = CmdRunAction(command='ls -1 /workspace/outputs/ 2>/dev/null || echo ""')
    action.set_hard_timeout(5)
    obs = runtime.run_action(action)

    if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
        files = [f.strip() for f in obs.content.split('\n') if f.strip()]

        # Read each file
        for filename in files:
            read_action = CmdRunAction(command=f'cat /workspace/outputs/{filename}')
            read_action.set_hard_timeout(10)
            read_obs = runtime.run_action(read_action)

            if (
                isinstance(read_obs, CmdOutputObservation)
                and read_obs.exit_code == 0
            ):
                outputs[filename] = read_obs.content

    return outputs


async def evaluate_agent(
    run_results: dict | None,
    instance: dict,
    sid: str | None = None,
    allow_skip: bool = True,
    reward: Optional[Reward] = None,
    use_rler_reward: bool = False,
) -> dict[str, Any]:
    """Evaluate NeAR agent results.

    For research tasks, evaluation can be:
    1. RLER reward (dr-tulu methodology with rubrics, citations, format, search)
    2. Reward-based (using custom reward function)
    3. Output-based (checking for required outputs)
    4. Manual review (returning outputs for human evaluation)

    Args:
        run_results: Results from run_agent
        instance: Task instance
        sid: Session ID
        allow_skip: Whether to skip evaluation if no results
        reward: Optional reward function for automatic evaluation
        use_rler_reward: Use RLER (dr-tulu) reward calculation

    Returns:
        dict with evaluation results
    """
    if allow_skip and (run_results is None or not run_results.get('outputs')):
        return {'resolved': False, 'reason': 'No outputs generated'}

    # Option 1: RLER reward (dr-tulu methodology)
    if use_rler_reward or instance.get('use_rler_reward', False):
        try:
            from openhands.nvidia.near_agent.rler_reward import compute_rler_reward

            # Get the full response from messages
            messages = run_results.get('messages', [])
            response = _reconstruct_response_from_messages(messages)

            # Get ground truth (rubrics, expected content)
            ground_truth = instance.get('ground_truth', {})
            if not ground_truth:
                # Create default rubrics if none provided
                ground_truth = _create_default_rubrics(instance)

            # Get question
            question = instance.get('problem_statement') or instance.get('query', '')

            # Compute RLER reward
            rler_config = instance.get('rler_config', {})
            result = await compute_rler_reward(
                response=response,
                ground_truth=ground_truth,
                question=question,
                use_citation_reward=rler_config.get('use_citation_reward', True),
                use_likert_rubric=rler_config.get('use_likert_rubric', False),
                use_general_rubric=rler_config.get('use_general_rubric', False),
            )

            return {
                'resolved': result['reward'] > 0.5,
                'reward': result['reward'],
                'reward_breakdown': result['log_values'],
                'rubric_breakdown': result.get('rubric_breakdown', {}),
                'outputs': run_results.get('outputs', {}),
            }
        except Exception as e:
            logger.error(f'Error computing RLER reward: {e}')
            return {'resolved': False, 'error': str(e)}

    # Option 2: If reward function provided, use it
    if reward is not None:
        try:
            reward_score = reward.compute(run_results, instance)
            return {
                'resolved': reward_score > 0.5,
                'reward': reward_score,
                'outputs': run_results.get('outputs', {}),
            }
        except Exception as e:
            logger.error(f'Error computing reward: {e}')
            return {'resolved': False, 'error': str(e)}

    # Option 3: Check for required outputs
    required_outputs = instance.get('required_outputs', [])
    outputs = run_results.get('outputs', {})

    if required_outputs:
        has_all_outputs = all(req in outputs for req in required_outputs)
        return {
            'resolved': has_all_outputs,
            'outputs': outputs,
            'missing_outputs': [req for req in required_outputs if req not in outputs],
        }

    # Option 4: Default - return outputs for manual review
    return {
        'resolved': True,  # Assume success if agent finished
        'outputs': outputs,
        'manual_review_required': True,
    }


def _reconstruct_response_from_messages(messages: list) -> str:
    """Reconstruct full response from message history.

    Args:
        messages: List of message dicts

    Returns:
        Reconstructed response text
    """
    # Combine all assistant messages
    response_parts = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if content:
                response_parts.append(content)

    return '\n'.join(response_parts)


def _create_default_rubrics(instance: dict) -> dict:
    """Create default rubrics if none provided.

    Args:
        instance: Task instance

    Returns:
        Ground truth dict with default rubrics
    """
    # Default research quality rubrics
    default_rubrics = [
        {
            'description': 'The answer is comprehensive and covers all aspects of the question',
            'weight': 1.0,
        },
        {
            'description': 'The answer provides thorough analysis with sufficient depth',
            'weight': 1.0,
        },
        {
            'description': 'The answer is factually accurate and well-supported',
            'weight': 1.0,
        },
        {
            'description': 'The answer is well-organized and coherent',
            'weight': 0.5,
        },
    ]

    return {'rubrics': default_rubrics}
