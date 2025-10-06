import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.controller.agent import Agent


from datetime import timedelta

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from openhands.core.config.mcp_config import (
    MCPConfig,
    MCPSHTTPServerConfig,
    MCPSSEServerConfig,
)
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action.mcp import MCPAction
from openhands.events.observation.mcp import MCPObservation
from openhands.events.observation.observation import Observation
from openhands.mcp.client import MCPClient
from openhands.memory.memory import Memory
from openhands.runtime.base import Runtime


def convert_mcp_clients_to_tools(mcp_clients: list[MCPClient] | None) -> list[dict]:
    """
    Converts a list of MCPClient instances to ChatCompletionToolParam format
    that can be used by CodeActAgent.

    Args:
        mcp_clients: List of MCPClient instances or None

    Returns:
        List of dicts of tools ready to be used by CodeActAgent
    """
    if mcp_clients is None:
        logger.warning('mcp_clients is None, returning empty list')
        return []

    all_mcp_tools = []
    try:
        for client in mcp_clients:
            # Each MCPClient has an mcp_clients property that is a ToolCollection
            # The ToolCollection has a to_params method that converts tools to ChatCompletionToolParam format
            for tool in client.tools:
                mcp_tools = tool.to_param()
                all_mcp_tools.append(mcp_tools)
    except Exception as e:
        logger.error(f'Error in convert_mcp_clients_to_tools: {e}')
        return []
    return all_mcp_tools


async def create_mcp_clients(
    sse_servers: list[MCPSSEServerConfig],
    shttp_servers: list[MCPSHTTPServerConfig],
    conversation_id: str | None = None,
) -> list[MCPClient]:
    import sys

    # Skip MCP clients on Windows
    if sys.platform == 'win32':
        logger.info(
            'MCP functionality is disabled on Windows, skipping client creation'
        )
        return []

    servers: list[MCPSSEServerConfig | MCPSHTTPServerConfig] = sse_servers.copy()
    servers.extend(shttp_servers.copy())

    if not servers:
        return []

    mcp_clients = []

    for server in servers:
        is_sse = isinstance(server, MCPSSEServerConfig)
        connection_type = 'SSE' if is_sse else 'SHTTP'
        logger.info(
            f'Initializing MCP agent for {server} with {connection_type} connection...'
        )
        client = MCPClient()

        try:
            if is_sse:
                await client.connect_sse(
                    server.url,
                    api_key=server.api_key,
                    conversation_id=conversation_id,
                )
            else:
                await client.connect_shttp(
                    server.url,
                    api_key=server.api_key,
                    conversation_id=conversation_id,
                )

            # Only add the client to the list after a successful connection
            mcp_clients.append(client)

        except Exception as e:
            logger.error(f'Failed to connect to {server}: {str(e)}', exc_info=True)
            try:
                await client.disconnect()
            except Exception as disconnect_error:
                logger.error(
                    f'Error during disconnect after failed connection: {str(disconnect_error)}'
                )
    return mcp_clients


async def fetch_mcp_tools_from_config(
    mcp_config: MCPConfig, conversation_id: str | None = None
) -> list[dict]:
    """
    Retrieves the list of MCP tools from the MCP clients.

    Args:
        mcp_config: The MCP configuration
        conversation_id: Optional conversation ID to associate with the MCP clients

    Returns:
        A list of tool dictionaries. Returns an empty list if no connections could be established.
    """
    import sys

    # Skip MCP tools on Windows
    if sys.platform == 'win32':
        logger.info('MCP functionality is disabled on Windows, skipping tool fetching')
        return []

    mcp_clients = []
    mcp_tools = []
    try:
        logger.debug(f'Creating MCP clients with config: {mcp_config}')
        # Create clients - this will fetch tools but not maintain active connections
        mcp_clients = await create_mcp_clients(
            mcp_config.sse_servers, mcp_config.shttp_servers, conversation_id
        )

        if not mcp_clients:
            logger.debug('No MCP clients were successfully connected')
            return []

        # Convert tools to the format expected by the agent
        mcp_tools = convert_mcp_clients_to_tools(mcp_clients)

        # Always disconnect clients to clean up resources
        for mcp_client in mcp_clients:
            try:
                await mcp_client.disconnect()
            except Exception as disconnect_error:
                logger.error(f'Error disconnecting MCP client: {str(disconnect_error)}')

    except Exception as e:
        logger.error(f'Error fetching MCP tools: {str(e)}')
        return []

    logger.debug(f'MCP tools: {mcp_tools}')
    return mcp_tools


async def call_tool_mcp(mcp_clients: list[MCPClient], action: MCPAction) -> Observation:
    """
    Call a tool on an MCP server and return the observation.

    Args:
        mcp_clients: The list of MCP clients to execute the action on
        action: The MCP action to execute

    Returns:
        The observation from the MCP server
    """
    import sys

    from openhands.events.observation import ErrorObservation

    # Skip MCP tools on Windows
    if sys.platform == 'win32':
        logger.info('MCP functionality is disabled on Windows')
        return ErrorObservation('MCP functionality is not available on Windows')

    if not mcp_clients:
        raise ValueError('No MCP clients found')

    logger.debug(f'MCP action received: {action}')

    # Find the MCP client that has the matching tool name
    matching_client = None
    logger.debug(f'MCP clients: {mcp_clients}')
    logger.debug(f'MCP action name: {action.name}')

    for client in mcp_clients:
        logger.debug(f'MCP client tools: {client.tools}')
        if action.name in [tool.name for tool in client.tools]:
            matching_client = client
            break

    if matching_client is None:
        raise ValueError(f'No matching MCP agent found for tool name: {action.name}')

    logger.debug(f'Matching client: {matching_client}')

    # Call the tool - this will create a new connection internally
    response = await matching_client.call_tool(action.name, action.arguments)
    logger.debug(f'MCP response: {response}')

    return MCPObservation(
        content=json.dumps(response.model_dump(mode='json')),
        name=action.name,
        arguments=action.arguments,
    )


async def execute_mcp_action_from_config(
    mcp_config: MCPConfig, action: MCPAction, conversation_id: str | None = None
) -> Observation:
    """Connect to MCP servers one-by-one and execute the action on the first
    server that provides the requested tool, using ephemeral per-call sessions.
    All context managers are entered/exited in the same task to avoid anyio
    cancel-scope mismatches.
    """
    import asyncio

    # Lazy imports to avoid circular deps and optional runtime costs
    try:
        from anyio import ClosedResourceError, EndOfStream  # type: ignore
    except Exception:  # pragma: no cover - anyio always present in runtime
        ClosedResourceError = type('ClosedResourceError', (), {})  # type: ignore
        EndOfStream = type('EndOfStream', (), {})  # type: ignore
    from openhands.events.observation import ErrorObservation

    def _is_retryable(exc: BaseException) -> bool:
        """Classify transient/disconnect errors as retryable.

        - anyio ClosedResourceError / EndOfStream during SSE writes
        - asyncio.CancelledError (from cancel scopes/timeouts during I/O)
        - asyncio.TimeoutError
        - ExceptionGroup containing any of the above
        """
        # Handle exception groups (Python 3.11+)
        if hasattr(exc, 'exceptions') and isinstance(exc, BaseException):  # type: ignore[unreachable]
            try:
                for sub in exc.exceptions:  # type: ignore[attr-defined]
                    if _is_retryable(sub):
                        return True
            except Exception as iter_err:
                logger.debug(
                    f'Unexpected error while checking retryable sub-exception: {iter_err}'
                )

        if isinstance(exc, (asyncio.CancelledError, asyncio.TimeoutError)):
            return True
        # anyio types
        if isinstance(exc, (ClosedResourceError, EndOfStream)):
            return True

        # Fallback: string match for wrapped ClosedResourceError
        msg = str(exc)
        return 'ClosedResourceError' in msg or 'EndOfStream' in msg

    # Build prioritized lists: try SSE first (local router/external SSE), then SHTTP
    sse_servers = list(mcp_config.sse_servers)
    shttp_servers = list(getattr(mcp_config, 'shttp_servers', []) or [])
    last_error: BaseException | None = None
    last_error_retryable = False

    # Try SSE servers
    for server in sse_servers:
        try:
            headers = {}
            if server.api_key:
                headers = {
                    'Authorization': f'Bearer {server.api_key}',
                    's': server.api_key,
                    'X-Session-API-Key': server.api_key,
                }
            if conversation_id:
                headers['X-OpenHands-Conversation-ID'] = conversation_id

            # Ephemeral connection and session in a single task
            async with sse_client(
                url=server.url, headers=headers or None, timeout=30.0
            ) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=30),
                ) as session:
                    await session.initialize()
                    tools_resp = await session.list_tools()
                    tool_names = [t.name for t in tools_resp.tools]
                    if action.name not in tool_names:
                        continue
                    response = await session.call_tool(
                        name=action.name, arguments=action.arguments
                    )
                    return MCPObservation(
                        content=json.dumps(response.model_dump(mode='json')),
                        name=action.name,
                        arguments=action.arguments,
                    )
        except asyncio.CancelledError as e:
            last_error = e
            last_error_retryable = True
            continue
        except Exception as e:
            last_error = e
            last_error_retryable = _is_retryable(e)
            continue

    # Try SHTTP servers
    for server in shttp_servers:
        try:
            headers = {}
            if server.api_key:
                headers = {
                    'Authorization': f'Bearer {server.api_key}',
                    's': server.api_key,
                    'X-Session-API-Key': server.api_key,
                }
            if conversation_id:
                headers['X-OpenHands-Conversation-ID'] = conversation_id

            timeout_delta = timedelta(seconds=30)
            sse_read_timeout_delta = timedelta(seconds=300)

            async with streamablehttp_client(
                url=server.url,
                headers=headers if headers else None,
                timeout=timeout_delta,
                sse_read_timeout=sse_read_timeout_delta,
            ) as (read_stream, write_stream, _):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timeout_delta,
                ) as session:
                    await session.initialize()
                    tools_resp = await session.list_tools()
                    tool_names = [t.name for t in tools_resp.tools]
                    if action.name not in tool_names:
                        continue
                    response = await session.call_tool(
                        name=action.name, arguments=action.arguments
                    )
                    return MCPObservation(
                        content=json.dumps(response.model_dump(mode='json')),
                        name=action.name,
                        arguments=action.arguments,
                    )
        except asyncio.CancelledError as e:
            last_error = e
            last_error_retryable = True
            continue
        except Exception as e:
            last_error = e
            last_error_retryable = _is_retryable(e)
            continue

    if last_error is not None:
        # Return a structured observation so the agent can decide to retry.
        error_id = 'MCP_RETRYABLE_ERROR' if last_error_retryable else 'MCP_ERROR'
        message = (
            f'MCP endpoint error while executing tool {action.name}: '
            f'{type(last_error).__name__}: {last_error}. '
            f'{"This looks transient; you can retry." if last_error_retryable else ""}'
        ).strip()
        return ErrorObservation(content=message, error_id=error_id)

    return ErrorObservation(
        content=f'No MCP servers available to execute tool: {action.name}',
        error_id='MCP_NO_SERVER',
    )


async def add_mcp_tools_to_agent(
    agent: 'Agent', runtime: Runtime, memory: 'Memory', app_config: OpenHandsConfig
):
    """
    Add MCP tools to an agent.
    """
    import sys

    # Skip MCP tools on Windows
    if sys.platform == 'win32':
        logger.info('MCP functionality is disabled on Windows, skipping MCP tools')
        agent.set_mcp_tools([])
        return

    assert runtime.runtime_initialized, (
        'Runtime must be initialized before adding MCP tools'
    )

    extra_stdio_servers = []
    extra_shttp_servers: list[MCPSHTTPServerConfig] = []

    # Add microagent MCP tools if available
    microagent_mcp_configs = memory.get_microagent_mcp_tools()

    for micro_mcp_config in microagent_mcp_configs:
        if micro_mcp_config.sse_servers:
            logger.warning(
                'Microagent MCP config contains SSE servers, it is not yet supported.'
            )

        if micro_mcp_config.stdio_servers:
            for stdio_server in micro_mcp_config.stdio_servers:
                # Check if this stdio server is already in the config
                if stdio_server not in extra_stdio_servers:
                    extra_stdio_servers.append(stdio_server)
                    logger.info(f'Added microagent stdio server: {stdio_server.name}')

        # Also collect SHTTP servers from microagents for parity
        if (
            hasattr(micro_mcp_config, 'shttp_servers')
            and micro_mcp_config.shttp_servers
        ):
            for shttp_server in micro_mcp_config.shttp_servers:
                # Deduplicate by (url, api_key)
                if not any(
                    (s.url == shttp_server.url and s.api_key == shttp_server.api_key)
                    for s in extra_shttp_servers
                ):
                    extra_shttp_servers.append(shttp_server)
                    logger.info(
                        f'Added microagent SHTTP server: {getattr(shttp_server, "url", "<unknown>")}'
                    )

    # Also include stdio servers from app_config.mcp for parity
    if hasattr(app_config, 'mcp') and getattr(app_config, 'mcp') is not None:
        base_mcp: MCPConfig = app_config.mcp
        if base_mcp.stdio_servers:
            for stdio_server in base_mcp.stdio_servers:
                if stdio_server not in extra_stdio_servers:
                    extra_stdio_servers.append(stdio_server)
        if hasattr(base_mcp, 'shttp_servers') and base_mcp.shttp_servers:
            for shttp_server in base_mcp.shttp_servers:  # type: ignore[attr-defined]
                if not any(
                    (s.url == shttp_server.url and s.api_key == shttp_server.api_key)
                    for s in extra_shttp_servers
                ):
                    extra_shttp_servers.append(shttp_server)

    # Add the runtime as another MCP server (merges stdio servers and adds runtime SSE)
    updated_mcp_config = runtime.get_mcp_config(extra_stdio_servers)

    # Merge in extra SHTTP servers (if any) so downstream fetch includes them
    if extra_shttp_servers:
        # Ensure attribute exists
        current_shttp = list(getattr(updated_mcp_config, 'shttp_servers', []) or [])
        for server in extra_shttp_servers:
            if not any(
                (s.url == server.url and s.api_key == server.api_key)
                for s in current_shttp
            ):
                current_shttp.append(server)
        try:
            updated_mcp_config.shttp_servers = current_shttp  # type: ignore[attr-defined]
        except Exception:
            # If model is frozen or attribute missing, fall back to constructing a new config
            try:
                updated_mcp_config = MCPConfig(
                    sse_servers=updated_mcp_config.sse_servers,
                    stdio_servers=updated_mcp_config.stdio_servers,
                )
                updated_mcp_config.shttp_servers = current_shttp  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f'Failed to merge SHTTP servers into MCP config: {e}')

    # Fetch the MCP tools
    mcp_tools = await fetch_mcp_tools_from_config(updated_mcp_config)

    logger.info(
        f'Loaded {len(mcp_tools)} MCP tools: {[tool["function"]["name"] for tool in mcp_tools]}'
    )

    # Set the MCP tools on the agent
    agent.set_mcp_tools(mcp_tools)
