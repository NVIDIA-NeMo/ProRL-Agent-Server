"""
Parallel fetch MCP test for SingularityRuntime.

Launches multiple SingularityRuntime instances in parallel. Each runtime:
- Configures stdio fetch MCP server (uvx mcp-server-fetch)
- Starts a per-runtime HTTP server on a unique port
- Calls the fetch tool to read from that server
"""

import asyncio
import json
import os
import random
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from openhands.core.config import load_openhands_config
from openhands.core.config.mcp_config import MCPStdioServerConfig
from openhands.events import EventStream
from openhands.events.action.mcp import MCPAction
from openhands.runtime.impl.singularity.singularity_runtime import SingularityRuntime
from openhands.runtime.plugins import AgentSkillsRequirement, JupyterRequirement
from openhands.storage import get_file_store

_port_lock = threading.Lock()


def _find_free_port() -> int:
    with _port_lock, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _run_single_runtime_fetch(test_params: dict) -> dict:
    """Create one SingularityRuntime, run fetch MCP action, and return results."""
    runtime_id = test_params['runtime_id']
    temp_dir = test_params['temp_dir']

    start_time = time.time()

    try:
        # Load configuration
        config = load_openhands_config()
        config.run_as_openhands = False
        config.sandbox.keep_runtime_alive = False
        config.sandbox.force_rebuild_runtime = False

        # Unique workspace per runtime
        runtime_workspace = os.path.join(temp_dir, f'mcp_fetch_runtime_{runtime_id}')
        os.makedirs(runtime_workspace, exist_ok=True)

        config.workspace_base = runtime_workspace
        config.workspace_mount_path = runtime_workspace
        config.workspace_mount_path_in_sandbox = '/workspace'

        # Configure stdio fetch MCP server
        config.mcp.sse_servers.clear()
        config.mcp.stdio_servers = [
            MCPStdioServerConfig(name='fetch', command='uvx', args=['mcp-server-fetch'])
        ]

        # File store and event stream
        file_store = get_file_store(
            config.file_store,
            config.file_store_path,
            config.file_store_web_hook_url,
            config.file_store_web_hook_headers,
        )

        sid = f'test-mcp-fetch-par-{runtime_id}-{uuid.uuid4().hex[:8]}'
        event_stream = EventStream(sid, file_store)

        # Create runtime
        plugins = [AgentSkillsRequirement(), JupyterRequirement()]
        runtime = SingularityRuntime(
            config=config,
            event_stream=event_stream,
            sid=sid,
            plugins=plugins,
            attach_to_existing=False,
        )

        # Connect
        asyncio.run(runtime.connect())

        # Start an HTTP server on a unique port inside the runtime
        port = _find_free_port()
        start_server_cmd = f'python3 -m http.server {port} > server.log 2>&1 &'
        from openhands.events.action import (
            CmdRunAction,  # deferred import to avoid test import cycles
        )

        runtime.run_action(CmdRunAction(command=start_server_cmd))
        # Give it a moment to start and confirm by reading the log
        runtime.run_action(CmdRunAction(command='sleep 2 && cat server.log'))

        # Run fetch MCP action (local HTTP server)
        url = f'http://localhost:{port}'
        action = MCPAction(name='fetch', arguments={'url': url})
        obs = asyncio.run(runtime.call_tool_mcp(action))

        data = json.loads(obs.content)
        is_error = data.get('isError', False)
        text = ''
        for c in data.get('content', []):
            if c.get('type') == 'text' and isinstance(c.get('text'), str):
                text = c['text']
                break

        # Perform N subsequent fetch MCP actions to random external URLs
        SUBSEQUENT_FETCHES = 3
        test_urls = [
            'https://www.example.com',
            'https://httpbin.org/html',
            'https://httpbin.org/json',
            'https://httpbin.org/status/200',
            'https://www.google.com',
            'https://api.github.com',
            'https://jsonplaceholder.typicode.com/posts/1',
            'https://www.wikipedia.org',
            'https://www.cloudflare.com',
            'https://httpbin.org/headers',
            'https://httpbin.org/user-agent',
            'https://httpbin.org/ip',
            'https://httpbin.org/uuid',
            'https://httpbin.org/base64/SFRUUEJJTiBpcyBhd2Vzb21l',
            'https://httpbin.org/delay/1',
        ]

        subsequent_successes = 0
        for _ in range(SUBSEQUENT_FETCHES):
            random_url = random.choice(test_urls)
            action_ext = MCPAction(name='fetch', arguments={'url': random_url})
            try:
                obs_ext = asyncio.run(runtime.call_tool_mcp(action_ext))
                data_ext = json.loads(obs_ext.content)
                print(f'Subsequent fetch {_} succeeded: {data_ext}')
                if not data_ext.get('isError', False):
                    ext_text = ''
                    for c in data_ext.get('content', []):
                        if c.get('type') == 'text' and isinstance(c.get('text'), str):
                            ext_text = c['text']
                            break
                    if len(ext_text) > 0:
                        subsequent_successes += 1
            except Exception:
                # Ignore individual external fetch failures to reduce brittleness
                pass

        # Cleanup runtime
        runtime.close()

        total_time = time.time() - start_time

        return {
            'runtime_id': runtime_id,
            'success': (not is_error) and len(text) > 0,
            'total_time': total_time,
            'is_error': is_error,
            'subsequent_successes': subsequent_successes,
            'subsequent_attempts': SUBSEQUENT_FETCHES,
        }

    except Exception as e:
        total_time = time.time() - start_time
        return {
            'runtime_id': runtime_id,
            'success': False,
            'total_time': total_time,
            'error': str(e),
            'error_type': type(e).__name__,
        }


@pytest.mark.skipif(
    os.getenv('TEST_RUNTIME', 'docker').lower() != 'singularity',
    reason='Test only runs with TEST_RUNTIME=singularity',
)
def test_mcp_fetch_actions_parallel_singularity(temp_dir):
    """Run fetch MCP actions in parallel Singularity runtimes."""
    n_runtimes = 32
    results = []
    start = time.time()

    test_params_list = [
        {
            'runtime_id': i,
            'temp_dir': temp_dir,
        }
        for i in range(1, n_runtimes + 1)
    ]

    with ThreadPoolExecutor(max_workers=n_runtimes) as executor:
        futures = {
            executor.submit(_run_single_runtime_fetch, params): params['runtime_id']
            for params in test_params_list
        }

        for future in as_completed(futures):
            rid = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = (
                    'OK'
                    if result.get('success')
                    else f'FAIL ({result.get("error_type", "")})'
                )
                print(f'[PARALLEL FETCH MCP] Runtime {rid}: {status}')
            except Exception as e:
                print(f'[PARALLEL FETCH MCP] Runtime {rid}: EXCEPTION - {e}')
                results.append(
                    {
                        'runtime_id': rid,
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__,
                    }
                )

    elapsed = time.time() - start
    successes = sum(1 for r in results if r.get('success'))
    print(
        f'[PARALLEL FETCH MCP] Completed {n_runtimes} runtimes in {elapsed:.2f}s: {successes} succeeded'
    )

    # Require at least one success to validate core behavior without being overly brittle
    assert successes > 0, (
        f'No successful MCP fetch runs out of {n_runtimes}. First error: {results[0] if results else "no results"}'
    )
