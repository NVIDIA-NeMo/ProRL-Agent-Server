"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json

from litellm import (
    ModelResponse,
)

from openhands.agenthub.gui_agent.tools import (
    ClickTool,
    RightClickTool,
    MiddleClickTool,
    DoubleClickTool,
    TripleClickTool,
    MoveToTool,
    DragToTool,
    ScrollTool,
    HorizontalScrollTool,
    WriteTool,
    PressTool,
    HotkeyTool,
    FailTool,
    FinishTool,
    WaitTool,
)
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    AgentFinishAction,
    MessageAction,
)
from openhands.events.action.os import OSWorldInteractiveAction
from openhands.events.tool import ToolCallMetadata


def combine_thought(action: Action, thought: str) -> Action:
    if not hasattr(action, 'thought'):
        return action
    if thought and action.thought:
        action.thought = f'{thought}\n{action.thought}'
    elif thought:
        action.thought = thought
    return action


def response_to_actions(
    response: ModelResponse,
    mcp_tool_names: list[str] | None = None,
    timeout: float | None = None,
) -> Action:
    assert len(response.choices) == 1, 'Only one choice is supported for now'
    choice = response.choices[0]
    assistant_msg = choice.message
    if hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls:
        # Check if there's assistant_msg.content. If so, add it to the thought
        thought = ''
        if isinstance(assistant_msg.content, str):
            thought = assistant_msg.content
        elif isinstance(assistant_msg.content, list):
            for msg in assistant_msg.content:
                if msg['type'] == 'text':
                    thought += msg['text']

        # Process each tool call to OpenHands action
        tool_call = assistant_msg.tool_calls[0]
        action: Action
        logger.debug(f'Tool call in function_calling.py: {tool_call}')
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.decoder.JSONDecodeError as e:
            raise FunctionCallValidationError(
                f'Failed to parse tool call arguments: {tool_call.function.arguments}'
            ) from e

        # ================================================
        # ClickTool
        # ================================================

        if tool_call.function.name == ClickTool['function']['name']:
            if 'x' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "x" in tool call {tool_call.function.name}'
                )
            if 'y' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "y" in tool call {tool_call.function.name}'
                )
            action = OSWorldInteractiveAction(
                method='execute_agentic_action',
                params={
                    'action': {
                        'action_type': 'CLICK',
                        'parameters': arguments,
                    }
                },
            )
        # ================================================
        # RightClickTool
        # ================================================
        elif tool_call.function.name == RightClickTool['function']['name']:
            if 'x' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "x" in tool call {tool_call.function.name}'
                )
            if 'y' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "y" in tool call {tool_call.function.name}'
                )
            action = OSWorldInteractiveAction(
                method='execute_agentic_action',
                params={
                    'action': {
                        'action_type': 'RIGHT_CLICK',
                        'parameters': arguments,
                    }
                },
            )
        # ================================================
        # MiddleClickTool
        # ================================================
        elif tool_call.function.name == MiddleClickTool['function']['name']:
            if 'x' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "x" in tool call {tool_call.function.name}'
                )
            if 'y' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "y" in tool call {tool_call.function.name}'
                )
            action = OSWorldInteractiveAction(
                method='execute_agentic_action',
                params={
                    'action': {
                        'action_type': 'MIDDLE_CLICK',
                        'parameters': arguments,
                    }
                },
            )
        # ================================================
        # DoubleClickTool
        # ================================================
        elif tool_call.function.name == DoubleClickTool['function']['name']:
            if 'x' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "x" in tool call {tool_call.function.name}'
                )
            if 'y' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "y" in tool call {tool_call.function.name}'
                )
            action = OSWorldInteractiveAction(
                method='execute_agentic_action',
                params={
                    'action': {
                        'action_type': 'DOUBLE_CLICK',
                        'parameters': arguments,
                    }
                },
            )
        # ================================================
        # TripleClickTool
        # ================================================
        elif tool_call.function.name == TripleClickTool['function']['name']:
            if 'x' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "x" in tool call {tool_call.function.name}'
                )
            if 'y' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "y" in tool call {tool_call.function.name}'
                )
            action = OSWorldInteractiveAction(
                method='execute_agentic_action',
                params={
                    'action': {
                        'action_type': 'TRIPLE_CLICK',
                        'parameters': arguments,
                    }
                },
            )
        # ================================================
        # MoveToTool
        # ================================================
        elif tool_call.function.name == MoveToTool['function']['name']:
            if 'x' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "x" in tool call {tool_call.function.name}'
                )
            if 'y' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "y" in tool call {tool_call.function.name}'
                )
            action = OSWorldInteractiveAction(
                method='execute_agentic_action',
                params={
                    'action': {
                        'action_type': 'MOVE_TO',
                        'parameters': arguments,
                    }
                },
            )
        # ================================================
        # DragToTool
        # ================================================
        elif tool_call.function.name == DragToTool['function']['name']:
            if 'x' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "x" in tool call {tool_call.function.name}'
                )
            if 'y' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "y" in tool call {tool_call.function.name}'
                )
            action = OSWorldInteractiveAction(
                method='execute_agentic_action',
                params={
                    'action': {
                        'action_type': 'DRAG_TO',
                        'parameters': arguments,
                    }
                },
            )
        # ================================================
        # ScrollTool
        # ================================================
        elif tool_call.function.name == ScrollTool['function']['name']:
            if 'amount' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "amount" in tool call {tool_call.function.name}'
                )
            # Map vertical scroll amount to dy; dx defaults to 0
            action = OSWorldInteractiveAction(
                method='execute_agentic_action',
                params={
                    'action': {
                        'action_type': 'SCROLL',
                        'parameters': arguments,
                    }
                },
            )
        # ================================================
        # HorizontalScrollTool
        # ================================================
        elif tool_call.function.name == HorizontalScrollTool['function']['name']:
            if 'amount' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "amount" in tool call {tool_call.function.name}'
                )
            # Map vertical scroll amount to dy; dx defaults to 0
            action = OSWorldInteractiveAction(
                method='execute_agentic_action',
                params={
                    'action': {
                        'action_type': 'SCROLL',
                        'parameters': arguments,
                    }
                },
            )
        # ================================================
        # WriteTool
        # ================================================
        elif tool_call.function.name == WriteTool['function']['name']:
            if 'text' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "text" in tool call {tool_call.function.name}'
                )
            action = OSWorldInteractiveAction(
                method='execute_agentic_action',
                params={
                    'action': {
                        'action_type': 'TYPING',
                        'parameters': arguments,
                    }
                },
            )
        # ================================================
        # PressTool
        # ================================================
        elif tool_call.function.name == PressTool['function']['name']:
            if 'key' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "key" in tool call {tool_call.function.name}'
                )
            action = OSWorldInteractiveAction(
                method='execute_agentic_action',
                params={
                    'action': {
                        'action_type': 'PRESS',
                        'parameters': arguments,
                    }
                },
            )
        # ================================================
        # HotkeyTool
        # ================================================
        elif tool_call.function.name == HotkeyTool['function']['name']:
            if 'keys' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "keys" in tool call {tool_call.function.name}'
                )
            action = OSWorldInteractiveAction(
                method='execute_agentic_action',
                params={
                    'action': {
                        'action_type': 'HOTKEY',
                        'parameters': arguments,
                    }
                },
            )
        # ================================================
        # FailTool
        # ================================================
        elif tool_call.function.name == FailTool['function']['name']:
            action = AgentFinishAction(
                task_completed='false',
            )
        # ================================================
        # FinishTool
        # ================================================
        elif tool_call.function.name == FinishTool['function']['name']:
            action = AgentFinishAction(
                task_completed='true',
            )
        # ================================================
        # WaitTool
        # ================================================
        elif tool_call.function.name == WaitTool['function']['name']:
            if 'seconds' not in arguments:
                raise FunctionCallValidationError(
                    f'Missing required argument "seconds" in tool call {tool_call.function.name}'
                )
            action = OSWorldInteractiveAction(
                method='execute_agentic_action',
                params={
                    'action': {
                        'action_type': 'WAIT',
                        'parameters': arguments,
                    }
                },
            )
        else:
            raise FunctionCallNotExistsError(
                f'Tool {tool_call.function.name} is not registered. (arguments: {arguments}). Please check the tool name and retry with an existing tool.'
            )

           
        action = combine_thought(action, thought)
        # Add metadata for tool calling
        action.tool_call_metadata = ToolCallMetadata(
            tool_call_id=tool_call.id,
            function_name=tool_call.function.name,
            model_response=response,
            total_calls_in_response=len(assistant_msg.tool_calls),
        )
    else:
        action = MessageAction(
            content=str(assistant_msg.content) if assistant_msg.content else '',
            wait_for_response=True,
        )
        action.tool_call_metadata = ToolCallMetadata(
                tool_call_id='',
                function_name='',
                model_response=response,
                total_calls_in_response=0,
        )

    # Add response id to actions
    # This will ensure we can match both actions without tool calls (e.g. MessageAction)
    # and actions with tool calls (e.g. CmdRunAction, IPythonRunCellAction, etc.)
    # with the token usage data
    action.response_id = response.id
    if timeout:
        action.set_hard_timeout(timeout)
    return action
