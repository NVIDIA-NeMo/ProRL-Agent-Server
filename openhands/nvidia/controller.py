"""
This file is a adapted version of openhands/core/main.py.
The run_controller function is seperated to two functions such that the controller can be exposed.
"""

import json
import os

from openhands.controller.agent import Agent
from openhands.controller.agent_controller import AgentController
from openhands.controller.state.state import State
from openhands.core.config import (
    OpenHandsConfig,
)
from openhands.core.config.mcp_config import OpenHandsMCPConfigImpl
from openhands.core.logger import openhands_logger as logger
from openhands.core.loop import run_agent_until_done
from openhands.core.main import FakeUserResponseFunc, load_replay_log
from openhands.core.schema import AgentState
from openhands.core.setup import (
    create_agent,
    create_controller,
    create_memory,
    create_runtime,
    generate_sid,
    initialize_repository_for_runtime,
)
from openhands.events import EventSource, EventStreamSubscriber
from openhands.events.action import MessageAction, NullAction
from openhands.events.action.action import Action
from openhands.events.event import Event
from openhands.events.observation import AgentStateChangedObservation
from openhands.io import read_input
from openhands.mcp import add_mcp_tools_to_agent
from openhands.memory.memory import Memory
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync


async def run_controller_with_controller(
    config: OpenHandsConfig,
    initial_user_action: Action,
    sid: str | None = None,
    runtime: Runtime | None = None,
    agent: Agent | None = None,
    exit_on_message: bool = False,
    fake_user_response_fn: FakeUserResponseFunc | None = None,
    headless_mode: bool = True,
    memory: Memory | None = None,
    conversation_instructions: str | None = None,
    controller: AgentController | None = None,
    initial_state: State | None = None,
) -> State | None:
    """Main coroutine to run the agent controller with task input flexibility.

    It's only used when you launch openhands backend directly via cmdline.

    Args:
        config: The app config.
        initial_user_action: An Action object containing initial user input
        sid: (optional) The session id. IMPORTANT: please don't set this unless you know what you're doing.
            Set it to incompatible value will cause unexpected behavior on RemoteRuntime.
        runtime: (optional) A runtime for the agent to run on.
        agent: (optional) A agent to run.
        exit_on_message: quit if agent asks for a message from user (optional)
        fake_user_response_fn: An optional function that receives the current state
            (could be None) and returns a fake user response.
        headless_mode: Whether the agent is run in headless mode.

    Returns:
        The final state of the agent, or None if an error occurred.

    Raises:
        AssertionError: If initial_user_action is not an Action instance.
        Exception: Various exceptions may be raised during execution and will be logged.

    Notes:
        - State persistence: If config.file_store is set, the agent's state will be
          saved between sessions.
        - Trajectories: If config.trajectories_path is set, execution history will be
          saved as JSON for analysis.
        - Budget control: Execution is limited by config.max_iterations and
          config.max_budget_per_task.

    Example:
        >>> config = load_openhands_config()
        >>> action = MessageAction(content="Write a hello world program")
        >>> state = await run_controller(config=config, initial_user_action=action)
    """
    sid = sid or generate_sid(config)

    if agent is None:
        agent = create_agent(config)

    # when the runtime is created, it will be connected and clone the selected repository
    repo_directory = None
    if runtime is None:
        runtime = create_runtime(
            config,
            sid=sid,
            headless_mode=headless_mode,
            agent=agent,
        )
        # Connect to the runtime
        call_async_from_sync(runtime.connect)

        # Initialize repository if needed
        if config.sandbox.selected_repo:
            repo_directory = initialize_repository_for_runtime(
                runtime,
                selected_repository=config.sandbox.selected_repo,
            )

    event_stream = runtime.event_stream

    # when memory is created, it will load the microagents from the selected repository
    if memory is None:
        memory = create_memory(
            runtime=runtime,
            event_stream=event_stream,
            sid=sid,
            selected_repository=config.sandbox.selected_repo,
            repo_directory=repo_directory,
            conversation_instructions=conversation_instructions,
        )

    # Add MCP tools to the agent
    if agent.config.enable_mcp:
        # Add OpenHands' MCP server by default
        _, openhands_mcp_stdio_servers = (
            OpenHandsMCPConfigImpl.create_default_mcp_server_config(
                config.mcp_host, config, None
            )
        )
        config.mcp.stdio_servers.extend(openhands_mcp_stdio_servers)

        await add_mcp_tools_to_agent(agent, runtime, memory, config)

    replay_events: list[Event] | None = None
    if config.replay_trajectory_path:
        logger.info('Trajectory replay is enabled')
        assert isinstance(initial_user_action, NullAction)
        replay_events, initial_user_action = load_replay_log(
            config.replay_trajectory_path
        )

    if controller is None:
        controller, initial_state = create_controller(
            agent, runtime, config, replay_events=replay_events
        )

    assert isinstance(initial_user_action, Action), (
        f'initial user actions must be an Action, got {type(initial_user_action)}'
    )
    logger.debug(
        f'Agent Controller Initialized: Running agent {agent.name}, model '
        f'{agent.llm.config.model}, with actions: {initial_user_action}'
    )

    # start event is a MessageAction with the task, either resumed or new
    if initial_state is not None and initial_state.last_error:
        # we're resuming the previous session
        event_stream.add_event(
            MessageAction(
                content=(
                    "Let's get back on track. If you experienced errors before, do "
                    'NOT resume your task. Ask me about it.'
                ),
            ),
            EventSource.USER,
        )
    else:
        # init with the provided actions
        event_stream.add_event(initial_user_action, EventSource.USER)

    def on_event(event: Event) -> None:
        if isinstance(event, AgentStateChangedObservation):
            if event.agent_state == AgentState.AWAITING_USER_INPUT:
                if exit_on_message:
                    message = '/exit'
                elif fake_user_response_fn is None:
                    message = read_input(config.cli_multiline_input)
                else:
                    message = fake_user_response_fn(controller.get_state())
                action = MessageAction(content=message)
                event_stream.add_event(action, EventSource.USER)

    event_stream.subscribe(EventStreamSubscriber.MAIN, on_event, sid)

    end_states = [
        AgentState.FINISHED,
        AgentState.REJECTED,
        AgentState.ERROR,
        AgentState.PAUSED,
        AgentState.STOPPED,
    ]

    try:
        await run_agent_until_done(controller, runtime, memory, end_states)
    except Exception as e:
        logger.error(f'Exception in main loop: {e}')

    # save session when we're about to close
    if config.file_store is not None and config.file_store != 'memory':
        end_state = controller.get_state()
        # NOTE: the saved state does not include delegates events
        end_state.save_to_session(
            event_stream.sid, event_stream.file_store, event_stream.user_id
        )

    await controller.close(set_stop_state=False)

    state = controller.get_state()

    # save trajectories if applicable
    if config.save_trajectory_path is not None:
        # if save_trajectory_path is a folder, use session id as file name
        if os.path.isdir(config.save_trajectory_path):
            file_path = os.path.join(config.save_trajectory_path, sid + '.json')
        else:
            file_path = config.save_trajectory_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        histories = controller.get_trajectory(config.save_screenshots_in_trajectory)
        with open(file_path, 'w') as f:  # noqa: ASYNC101
            json.dump(histories, f, indent=4)

    return state
