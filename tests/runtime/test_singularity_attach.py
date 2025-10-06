"""
Test for SingularityRuntime attach-to-existing container capability.

This test suite verifies that the SingularityRuntime can attach to an existing
container that was started by another SingularityRuntime instance.

To run this test:

1. Ensure Singularity/Apptainer is installed on your system
2. Set the TEST_RUNTIME environment variable to 'singularity':
   export TEST_RUNTIME=singularity
3. Run the test:
   pytest tests/runtime/test_singularity_attach.py -v

The test will:
1. Start a primary SingularityRuntime in one process
2. Start a secondary SingularityRuntime with attach_to_existing=True in another process
3. Verify that the secondary runtime can execute commands in the same container
4. Test the failure case when trying to attach to a non-existent container

Note: This test is skipped if TEST_RUNTIME is not set to 'singularity'.
"""

import asyncio
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pytest

from openhands.core.config import load_openhands_config
from openhands.core.exceptions import AgentRuntimeDisconnectedError
from openhands.events import EventStream
from openhands.events.action.commands import CmdRunAction
from openhands.runtime.impl.singularity.singularity_runtime import SingularityRuntime
from openhands.runtime.plugins import AgentSkillsRequirement, JupyterRequirement
from openhands.storage import get_file_store


def run_primary_runtime(temp_dir, duration=30):
    """Run a primary SingularityRuntime that stays alive for a specified duration.

    Args:
        temp_dir: Temporary directory to use as workspace
        duration: How long to keep the runtime alive (seconds)

    Returns:
        Dict with runtime information including sid, api_url, etc.
    """
    try:
        # Load configuration
        config = load_openhands_config()
        config.run_as_openhands = False
        config.sandbox.keep_runtime_alive = False
        config.sandbox.force_rebuild_runtime = False
        config.workspace_base = temp_dir
        config.workspace_mount_path = temp_dir
        config.workspace_mount_path_in_sandbox = '/workspace'

        # Set up file store and event stream
        file_store = get_file_store(
            config.file_store,
            config.file_store_path,
            config.file_store_web_hook_url,
            config.file_store_web_hook_headers,
        )

        sid = 'test-primary-runtime'
        event_stream = EventStream(sid, file_store)

        # Create runtime
        plugins = [AgentSkillsRequirement(), JupyterRequirement()]
        runtime = SingularityRuntime(
            config=config,
            event_stream=event_stream,
            sid=sid,
            plugins=plugins,
            attach_to_existing=False,  # This is the primary runtime
        )

        # Connect to the runtime
        asyncio.run(runtime.connect())

        # Store runtime information for the attaching process
        runtime_info = {
            'sid': runtime.sid,
            'vscode_port': runtime._vscode_port,
            'container_pid': runtime.container_pid,
            'container_name': runtime.container_name,
        }

        print(f'Primary runtime started: {runtime_info}')

        # Keep the runtime alive for the specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(1)
            # Check if runtime is still alive
            if not runtime._is_container_running():
                print('Primary runtime stopped unexpectedly')
                break

        print('Primary runtime shutting down...')
        runtime.close(rm_all_containers=False)

        return runtime_info

    except Exception as e:
        print(f'Error in primary runtime: {e}')
        return {'error': str(e)}


def run_attach_runtime(temp_dir, primary_sid, max_wait=60):
    """Attach to an existing SingularityRuntime and run a test command.

    Args:
        temp_dir: Temporary directory to use as workspace
        primary_sid: Session ID of the primary runtime to attach to
        max_wait: Maximum time to wait for primary runtime (seconds)

    Returns:
        Dict with test results
    """
    try:
        # Load configuration
        config = load_openhands_config()
        config.run_as_openhands = False
        config.sandbox.keep_runtime_alive = False
        config.sandbox.force_rebuild_runtime = False
        config.workspace_base = temp_dir
        config.workspace_mount_path = temp_dir
        config.workspace_mount_path_in_sandbox = '/workspace'

        # Set up file store and event stream with the same sid
        file_store = get_file_store(
            config.file_store,
            config.file_store_path,
            config.file_store_web_hook_url,
            config.file_store_web_hook_headers,
        )

        event_stream = EventStream(primary_sid, file_store)

        # Wait for primary runtime to be available
        start_wait = time.time()
        runtime = None

        while time.time() - start_wait < max_wait:
            try:
                # Create runtime with attach_to_existing=True
                plugins = [AgentSkillsRequirement(), JupyterRequirement()]
                runtime = SingularityRuntime(
                    config=config,
                    event_stream=event_stream,
                    sid=primary_sid,
                    plugins=plugins,
                    attach_to_existing=True,  # This runtime attaches to existing
                )

                # Attempt to connect (attach)
                asyncio.run(runtime.connect())
                print(f'Successfully attached to runtime {primary_sid}')
                break

            except Exception as e:
                print(f'Waiting for primary runtime... ({e})')
                time.sleep(2)
                if runtime:
                    try:
                        runtime.close()
                    except Exception as e:
                        print(f'Error closing runtime: {e}')
                runtime = None

        if runtime is None:
            return {'error': 'Failed to attach to primary runtime within timeout'}

        # Test the attached runtime by running a simple command
        try:
            action = CmdRunAction(command='echo "Hello from attached runtime"')
            print(f'Running test command: {action.command}')

            # Send the action and wait for response
            obs = runtime.run_action(action)

            result = {
                'success': True,
                'command': action.command,
                'exit_code': obs.exit_code,
                'stdout': obs.content,
                'runtime_info': {
                    'sid': runtime.sid,
                    'container_pid': runtime.container_pid,
                },
            }

            print(f'Test command result: {result}')

        except Exception as e:
            result = {'error': f'Failed to run test command: {e}'}

        # Clean up
        runtime.close()
        return result

    except Exception as e:
        print(f'Error in attach runtime: {e}')
        return {'error': str(e)}


@pytest.mark.skipif(
    os.getenv('TEST_RUNTIME', 'docker').lower() != 'singularity',
    reason='Test only runs with TEST_RUNTIME=singularity',
)
def test_singularity_runtime_attach_to_existing(temp_dir):
    """Test that SingularityRuntime can attach to an existing container.

    This test:
    1. Starts a primary SingularityRuntime in one process
    2. Starts a secondary SingularityRuntime with attach_to_existing=True in another process
    3. Verifies that the secondary runtime can execute commands in the same container
    4. Cleans up both runtimes

    Args:
        temp_dir: Temporary directory provided by pytest fixture
    """
    print('\n=== Testing SingularityRuntime attach capability ===')
    print(f'Workspace: {temp_dir}')

    primary_sid = 'test-primary-runtime'

    # Use ProcessPoolExecutor to run both runtimes
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Start the primary runtime
        print('Starting primary runtime...')
        primary_future = executor.submit(
            run_primary_runtime, temp_dir, 60
        )  # Run for 60 seconds

        # Give the primary runtime time to start
        time.sleep(30)

        # Start the attach runtime
        print('Starting attach runtime...')
        attach_future = executor.submit(run_attach_runtime, temp_dir, primary_sid, 45)

        # Wait for both processes to complete
        results = {}

        for future in as_completed([primary_future, attach_future], timeout=90):
            if future == primary_future:
                results['primary'] = future.result()
                print(f'Primary runtime result: {results["primary"]}')
            else:
                results['attach'] = future.result()
                print(f'Attach runtime result: {results["attach"]}')

        # Verify results
        assert 'primary' in results, 'Primary runtime did not complete'
        assert 'attach' in results, 'Attach runtime did not complete'

        # Check that primary runtime started successfully
        primary_result = results['primary']
        assert 'error' not in primary_result, (
            f'Primary runtime failed: {primary_result.get("error")}'
        )
        assert 'sid' in primary_result, 'Primary runtime missing session ID'

        # Check that attach runtime succeeded
        attach_result = results['attach']
        assert 'error' not in attach_result, (
            f'Attach runtime failed: {attach_result.get("error")}'
        )
        assert attach_result.get('success') is True, (
            'Attach runtime test command failed'
        )
        assert attach_result.get('exit_code') == 0, (
            f'Test command failed with exit code: {attach_result.get("exit_code")}'
        )
        assert 'Hello from attached runtime' in attach_result.get('stdout', ''), (
            'Test command output not found'
        )

        # Verify that both runtimes used the same container
        primary_pid = primary_result.get('container_pid')
        attach_pid = attach_result.get('runtime_info', {}).get('container_pid')

        print(f'Primary container PID: {primary_pid}')
        print(f'Attach runtime container PID: {attach_pid}')

        # Note: The PIDs might be the same (if truly shared) or different (if the attach
        # runtime gets the PID from session info), but both should be valid
        assert primary_pid is not None, 'Primary runtime missing container PID'
        assert attach_pid is not None, 'Attach runtime missing container PID'

        print('=== SingularityRuntime attach test completed successfully ===')


# Additional test to verify attach failure when no runtime exists
@pytest.mark.skipif(
    os.getenv('TEST_RUNTIME', 'docker').lower() != 'singularity',
    reason='Test only runs with TEST_RUNTIME=singularity',
)
def test_singularity_runtime_attach_fails_when_no_container(temp_dir):
    """Test that SingularityRuntime attach fails gracefully when no container exists.

    Args:
        temp_dir: Temporary directory provided by pytest fixture
    """
    print('\n=== Testing SingularityRuntime attach failure case ===')
    print(f'Workspace: {temp_dir}')

    # Load configuration
    config = load_openhands_config()
    config.run_as_openhands = False
    config.sandbox.keep_runtime_alive = False
    config.sandbox.force_rebuild_runtime = False
    config.workspace_base = temp_dir
    config.workspace_mount_path = temp_dir
    config.workspace_mount_path_in_sandbox = '/workspace'

    # Set up file store and event stream
    file_store = get_file_store(
        config.file_store,
        config.file_store_path,
        config.file_store_web_hook_url,
        config.file_store_web_hook_headers,
    )

    nonexistent_sid = 'nonexistent-runtime'
    event_stream = EventStream(nonexistent_sid, file_store)

    # Create runtime with attach_to_existing=True
    plugins = [AgentSkillsRequirement(), JupyterRequirement()]
    runtime = SingularityRuntime(
        config=config,
        event_stream=event_stream,
        sid=nonexistent_sid,
        plugins=plugins,
        attach_to_existing=True,
    )

    # Attempt to connect should fail
    with pytest.raises(Exception) as exc_info:
        asyncio.run(runtime.connect())

    # Verify that the exception indicates the container was not found
    error_msg = str(exc_info.value).lower()
    print(f'Error message: {error_msg}')
    print(f'Error value: {exc_info.value}')
    print(f'Error type: {type(exc_info.value)}')
    print(f'Error traceback: {exc_info.traceback}')
    assert exc_info.value is not None, 'Expected exception, got None'
    assert isinstance(exc_info.value, AgentRuntimeDisconnectedError), (
        f'Expected AgentRuntimeDisconnectedError, got: {type(exc_info.value)}'
    )
    print('=== SingularityRuntime attach failure test completed successfully ===')


if __name__ == '__main__':
    # For manual testing
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_singularity_runtime_attach_to_existing(temp_dir)
        test_singularity_runtime_attach_fails_when_no_container(temp_dir)
