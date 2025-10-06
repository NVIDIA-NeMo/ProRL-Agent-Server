import os
import signal
import subprocess
import json
import time
from functools import lru_cache
from typing import Callable
from uuid import UUID
from pathlib import Path

import httpx
import tenacity

from openhands.core.config import OpenHandsConfig
from openhands.core.exceptions import (
    AgentRuntimeDisconnectedError,
    AgentRuntimeNotFoundError,
)
from openhands.core.logger import DEBUG, DEBUG_RUNTIME
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.utils import find_available_tcp_port
from openhands.runtime.utils.command import (
    DEFAULT_MAIN_MODULE,
    get_action_execution_server_startup_command,
)
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.shutdown_listener import add_shutdown_listener
from openhands.utils.tenacity_stop import stop_if_should_exit

CONTAINER_NAME_PREFIX = 'openhands-runtime-'

EXECUTION_SERVER_PORT_RANGE = (30000, 39999)
VSCODE_PORT_RANGE = (40000, 49999)
APP_PORT_RANGE_1 = (50000, 54999)
APP_PORT_RANGE_2 = (55000, 59999)


def kill_process_tree(pid):
    """Kill a specific process, not the entire process group."""
    try:
        # Kill the specific process with SIGTERM first
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)

        # Check if still alive, then force kill
        try:
            os.kill(pid, 0)  # Test if process exists
            os.kill(pid, signal.SIGKILL)
            logger.info(f'Force killed process {pid}')
        except ProcessLookupError:
            logger.info(f'Process {pid} already dead')
    except ProcessLookupError:
        logger.info(f'Process {pid} already dead')
    except Exception as e:
        logger.info(f'Failed to kill process {pid}: {e}')


def _is_retryablewait_until_alive_error(exception):
    if isinstance(exception, tenacity.RetryError):
        cause = exception.last_attempt.exception()
        return _is_retryablewait_until_alive_error(cause)

    return isinstance(
        exception,
        (
            ConnectionError,
            httpx.ConnectTimeout,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            httpx.HTTPStatusError,
            httpx.ReadTimeout,
        ),
    )


def stop_all_enroot_containers(prefix: str = CONTAINER_NAME_PREFIX):
    """Stop running enroot processes using tracked PIDs."""
    try:
        # Get all active container PIDs from the registry
        pids_to_kill = list(EnrootRuntime._active_container_pids.copy())

        if pids_to_kill:
            logger.info(f'Stopping {len(pids_to_kill)} tracked enroot container processes')

            for pid in pids_to_kill:
                try:
                    logger.info(f'Stopping enroot container process with PID: {pid}')
                    kill_process_tree(pid)
                    # Remove from active registry after successful kill attempt
                    EnrootRuntime._active_container_pids.discard(pid)
                except Exception as e:
                    logger.warning(f'Failed to stop process {pid}: {e}')
        else:
            logger.debug('No tracked enroot container processes to stop')

    except Exception as e:
        logger.warning(f'Failed to stop enroot containers: {e}')


class EnrootRuntime(ActionExecutionClient):
    """This runtime uses enroot to manage containers for action execution.

    When receive an event, it will send the event to runtime-client which run inside the enroot environment.

    Args:
        config (OpenHandsConfig): The application configuration.
        event_stream (EventStream): The event stream to subscribe to.
        sid (str, optional): The session ID. Defaults to 'default'.
        plugins (list[PluginRequirement] | None, optional): List of plugin requirements. Defaults to None.
        env_vars (dict[str, str] | None, optional): Environment variables to set. Defaults to None.
    """

    _shutdown_listener_id: UUID | None = None
    _active_container_pids: set[int] = set()  # Track all active container PIDs

    def __init__(
        self,
        config: OpenHandsConfig,
        event_stream: EventStream,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Callable | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
        main_module: str = DEFAULT_MAIN_MODULE,
    ):
        if not EnrootRuntime._shutdown_listener_id:
            EnrootRuntime._shutdown_listener_id = add_shutdown_listener(
                lambda: stop_all_enroot_containers(CONTAINER_NAME_PREFIX)
            )

        self.config = config
        self.status_callback = status_callback

        self._host_port = -1
        self._container_port = -1
        self._vscode_port = -1
        self._app_ports: list[int] = []

        # Check if enroot is available
        self._check_enroot_availability()

        self.api_url = f'{self.config.sandbox.local_runtime_url}:{self._container_port}'

        self.base_container_image = self.config.sandbox.base_container_image
        self.runtime_container_image = self.config.sandbox.runtime_container_image
        self.container_name = CONTAINER_NAME_PREFIX + self._get_enroot_image_name()
        self.container_process: subprocess.Popen[str] | None = None
        self.container_pid: int | None = None  # Store the actual container PID
        self.main_module = main_module

        super().__init__(
            config,
            event_stream,
            sid,
            plugins,
            env_vars,
            status_callback,
            attach_to_existing,
            headless_mode,
        )

        # Log runtime_extra_deps after base class initialization so self.sid is available
        if self.config.sandbox.runtime_extra_deps:
            self.log(
                'debug',
                f'Installing extra user-provided dependencies in the runtime image: {self.config.sandbox.runtime_extra_deps}',
            )

    @staticmethod
    def _check_enroot_availability():
        """Check if enroot is available on the system."""
        try:
            result = subprocess.run(
                ['enroot', 'version'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f'Enroot version: {result.stdout.strip()}')
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                'Enroot is not available on this system. Please install enroot to use EnrootRuntime.'
            ) from e

    def _run_enroot_command(self, cmd: list[str], check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run an enroot command and return the result."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                check=check
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f'Enroot command failed: {" ".join(cmd)}')
            logger.error(f'Error: {e.stderr}')
            raise

    def _container_exists(self) -> bool:
        """Check if the container exists."""
        try:
            result = self._run_enroot_command(['enroot', 'list'], check=False)
            if result.returncode == 0:
                containers = result.stdout.strip().split('\n')
                return self.container_name in containers
            return False
        except Exception:
            return False

    def _is_container_running(self) -> bool:
        """Check if the container is currently running."""
        # For enroot, we can check if our process is still alive
        return self.container_process is not None and self.container_process.poll() is None

    @property
    def action_execution_server_url(self):
        return self.api_url

    async def connect(self):
        self.send_status_message('STATUS$STARTING_RUNTIME')
        try:
            await call_sync_from_async(self._attach_to_container)
        except Exception as e:
            if self.attach_to_existing:
                self.log(
                    'warning',
                    f'Container {self.container_name} not found or not running.',
                )
                raise AgentRuntimeDisconnectedError from e

            self.maybe_prepare_runtime_container_image()
            self.log(
                'info', f'Starting runtime with image: {self.runtime_container_image}'
            )
            self.log(
                'info',
                f'Container started: {self.container_name}. VSCode URL: {self.vscode_url}',
            )

        if not self.attach_to_existing:
            self.log('info', f'Waiting for client to become ready at {self.api_url}...')
            self.send_status_message('STATUS$WAITING_FOR_CLIENT')

        await call_sync_from_async(self.init_container)

        await call_sync_from_async(self.wait_until_alive)

        if not self.attach_to_existing:
            self.log('info', 'Runtime is ready.')

        if not self.attach_to_existing:
            await call_sync_from_async(self.setup_initial_env)

        self.log(
            'debug',
            f'Container initialized with plugins: {[plugin.name for plugin in self.plugins]}. VSCode URL: {self.vscode_url}',
        )
        if not self.attach_to_existing:
            self.send_status_message(' ')
        self._runtime_initialized = True

    def maybe_prepare_runtime_container_image(self):
        """Prepare the runtime container image for enroot."""
        if self.runtime_container_image is None:
            if self.base_container_image is None:
                raise ValueError(
                    'Neither runtime container image nor base container image is set'
                )
            # For enroot, we'll use the base image directly
            # In a full implementation, you might want to build a custom image
            self.runtime_container_image = self.base_container_image
            self.send_status_message('STATUS$PREPARING_CONTAINER')

        # Import the image if it doesn't exist locally
        self._import_image_if_needed()

    def _import_image_if_needed(self):
        """Import the container image into enroot if not already available."""
        try:

            # Import the image
            logger.info(f'Importing image {self.runtime_container_image} into enroot...')
            filename = f'/tmp/{self._get_enroot_image_name()}.sqsh'
            if os.path.exists(filename):
                return
            import_cmd = ['enroot', 'import', '-o', filename, f'docker://{self.runtime_container_image}']
            self._run_enroot_command(import_cmd)
            logger.info(f'Successfully imported {self.runtime_container_image}')

        except Exception as e:
            logger.error(f'Failed to import image: {e}')
            raise

    def _get_enroot_image_name(self) -> str:
        """Get the enroot-formatted image name."""
        # Enroot converts docker image names to a specific format
        # e.g., "python:3.11" becomes "python_3.11"
        if self.runtime_container_image:
            return self.runtime_container_image.replace(':', '_').replace('/', '_')
        return ''

    def _process_volumes(self) -> list[str]:
        """Process volume mounts for enroot format.

        Returns:
            A list of volume mount arguments for enroot.
        """
        mount_args = []

        # Process volumes (comma-delimited)
        if self.config.sandbox.volumes is not None:
            # Handle multiple mounts with comma delimiter
            mounts = self.config.sandbox.volumes.split(',')

            for mount in mounts:
                parts = mount.split(':')
                if len(parts) >= 2:
                    host_path = os.path.abspath(parts[0])
                    container_path = parts[1]
                    # Enroot mount format: --mount host_path:container_path
                    mount_args.extend(['--mount', f'{host_path}:{container_path}'])
                    logger.debug(
                        f'Mount dir (sandbox.volumes): {host_path} to {container_path}'
                    )

        # Legacy mounting with workspace_* parameters
        elif (
            self.config.workspace_mount_path is not None
            and self.config.workspace_mount_path_in_sandbox is not None
        ):
            mount_args.extend([
                '--mount',
                f'{self.config.workspace_mount_path}:{self.config.workspace_mount_path_in_sandbox}'
            ])
            logger.debug(
                f'Mount dir (legacy): {self.config.workspace_mount_path}'
            )

        return mount_args

    def init_container(self):
        self.log('debug', 'Preparing to start enroot container...')
        self.send_status_message('STATUS$PREPARING_CONTAINER')

        self._host_port = self._find_available_port(EXECUTION_SERVER_PORT_RANGE)
        self._container_port = self._host_port
        # Use the configured vscode_port if provided, otherwise find an available port
        self._vscode_port = (
            self.config.sandbox.vscode_port
            or self._find_available_port(VSCODE_PORT_RANGE)
        )
        self._app_ports = [
            self._find_available_port(APP_PORT_RANGE_1),
            self._find_available_port(APP_PORT_RANGE_2),
        ]
        self.api_url = f'{self.config.sandbox.local_runtime_url}:{self._container_port}'

        # Prepare environment variables
        env_vars = {
            'port': str(self._container_port),
            'PYTHONUNBUFFERED': '1',
            'VSCODE_PORT': str(self._vscode_port),
            'APP_PORT_1': str(self._app_ports[0]),
            'APP_PORT_2': str(self._app_ports[1]),
            'PIP_BREAK_SYSTEM_PACKAGES': '1',
        }
        if self.config.debug or DEBUG:
            env_vars['DEBUG'] = 'true'
        # also update with runtime_startup_env_vars
        env_vars.update(self.config.sandbox.runtime_startup_env_vars)

        self.log('debug', f'Workspace Base: {self.config.workspace_base}')

        # Process volumes for mounting
        mount_args = self._process_volumes()

        self.log(
            'debug',
            f'Sandbox workspace: {self.config.workspace_mount_path_in_sandbox}',
        )

        # Get the startup command
        startup_command = self.get_action_execution_server_startup_command()

        try:
            # Create enroot container if it doesn't exist
            if not self._container_exists():
                self._create_enroot_container()

            # Build the enroot start command
            cmd = ['enroot', 'start', '--rw']

            # Add environment variables
            for key, value in env_vars.items():
                cmd.extend(['--env', f'{key}={value}'])

            # Add volume mounts
            cmd.extend(mount_args)

            # Add working directory
            # cmd.extend(['--pwd', '/openhands/code/'])

            # Add container name and command
            cmd.append(self.container_name)
            cmd.extend(startup_command)

            self.log('debug', f'Starting enroot container with command: {" ".join(cmd)}')

            # Start the container process
            self.container_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(os.environ, **env_vars)
            )

            # Save the container PID and add to active registry
            self.container_pid = self.container_process.pid
            EnrootRuntime._active_container_pids.add(self.container_pid)

            # Give the container a moment to start
            time.sleep(2)

            # Check if the process started successfully
            assert self.container_process is not None  # Help mypy understand this cannot be None
            if self.container_process.poll() is not None:
                # Process failed to start, remove from registry
                if self.container_pid in EnrootRuntime._active_container_pids:
                    EnrootRuntime._active_container_pids.remove(self.container_pid)
                stdout, stderr = self.container_process.communicate()
                raise RuntimeError(
                    f'Container failed to start. Return code: {self.container_process.returncode}\n'
                    f'Stdout: {stdout}\nStderr: {stderr}'
                )

            self.log('debug', f'Container started. Server url: {self.api_url}, PID: {self.container_pid}')
            self.send_status_message('STATUS$CONTAINER_STARTED')

        except Exception as e:
            self.log(
                'error',
                f'Error: Instance {self.container_name} FAILED to start container!\n',
            )
            self.log('error', str(e))
            self.close()
            raise e

    def _create_enroot_container(self):
        """Create an enroot container from the imported image."""
        try:
            image_name = f'/tmp/{self._get_enroot_image_name()}.sqsh'
            cmd = ['enroot', 'create', '--name', self.container_name, image_name]
            self._run_enroot_command(cmd)
            logger.debug(f'Created enroot container: {self.container_name}')
        except Exception as e:
            logger.error(f'Failed to create enroot container: {e}')
            raise

    def _attach_to_container(self):
        """Attach to an existing container."""
        if not self._container_exists():
            raise AgentRuntimeNotFoundError(f'Container {self.container_name} not found.')

        # For enroot, we need to determine the ports from environment or config
        # Since enroot doesn't have the same container inspection as Docker,
        # we'll use our configured ports
        self._host_port = self._find_available_port(EXECUTION_SERVER_PORT_RANGE)
        self._container_port = self._host_port
        self._vscode_port = (
            self.config.sandbox.vscode_port
            or self._find_available_port(VSCODE_PORT_RANGE)
        )
        self._app_ports = [
            self._find_available_port(APP_PORT_RANGE_1),
            self._find_available_port(APP_PORT_RANGE_2),
        ]

        self.api_url = f'{self.config.sandbox.local_runtime_url}:{self._container_port}'
        self.log(
            'debug',
            f'attached to container: {self.container_name} {self._container_port} {self.api_url}',
        )

    @tenacity.retry(
        stop=tenacity.stop_after_delay(120) | stop_if_should_exit(),
        retry=tenacity.retry_if_exception(_is_retryablewait_until_alive_error),
        reraise=True,
        wait=tenacity.wait_fixed(2),
    )
    def wait_until_alive(self):
        self.check_if_alive()

    def close(self, rm_all_containers: bool | None = None):
        """Closes the EnrootRuntime and associated objects

        Parameters:
        - rm_all_containers (bool): Whether to stop all container processes with the prefix
        """
        super().close()

        if rm_all_containers is None:
            rm_all_containers = self.config.sandbox.rm_all_containers

        if self.config.sandbox.keep_runtime_alive or self.attach_to_existing:
            return

        # # Stop the container process
        # if self.container_process and self.container_process.poll() is None:
        #     self.container_process.terminate()
        #     try:
        #         self.container_process.wait(timeout=10)
        #     except subprocess.TimeoutExpired:
        #         self.container_process.kill()
        #         self.container_process.wait()

        # Stop enroot processes (but keep containers for reuse)
        if rm_all_containers:
            stop_all_enroot_containers(CONTAINER_NAME_PREFIX)
        else:
            # Stop only this specific container's process using the stored PID
            if self.container_pid is not None:
                try:
                    logger.info(f'Stopping enroot container {self.container_name} with PID: {self.container_pid}')
                    kill_process_tree(self.container_pid)
                    EnrootRuntime._active_container_pids.discard(self.container_pid)
                    self.container_pid = None
                except Exception as e:
                    logger.warning(f'Failed to stop container process {self.container_name}: {e}')
            else:
                logger.debug(f'No PID stored for container {self.container_name}, nothing to stop')

    def _find_available_port(self, port_range, max_attempts=5):
        """Find an available port in the given range."""
        for _ in range(max_attempts):
            port = find_available_tcp_port(port_range[0], port_range[1])
            return port
        # If no port is found after max_attempts, return the last tried port
        return port

    @property
    def vscode_url(self) -> str | None:
        token = super().get_vscode_token()
        if not token:
            return None

        vscode_url = f'http://localhost:{self._vscode_port}/?tkn={token}&folder={self.config.workspace_mount_path_in_sandbox}'
        return vscode_url

    @property
    def web_hosts(self):
        hosts: dict[str, int] = {}

        host_addr = 'localhost'  # enroot typically runs on localhost
        for port in self._app_ports:
            hosts[f'http://{host_addr}:{port}'] = port

        return hosts

    def pause(self):
        """Pause the runtime by stopping the container process."""
        raise NotImplementedError('Pause is not implemented for EnrootRuntime')
        # if self.container_process and self.container_process.poll() is None:
        #     self.container_process.terminate()
        #     self.log('debug', f'Container {self.container_name} paused')

    def resume(self):
        """Resume the runtime by restarting the container."""
        raise NotImplementedError('Resume is not implemented for EnrootRuntime')
        # if not self._container_exists():
        #     raise RuntimeError('Container not found')

        # # Restart the container with the same configuration
        # self.init_container()
        # self.log('debug', f'Container {self.container_name} resumed')

        # # Wait for the container to be ready
        # self.wait_until_alive()

    @classmethod
    async def delete(cls, conversation_id: str):
        """Delete a container by conversation ID."""
        try:
            container_name = CONTAINER_NAME_PREFIX + conversation_id
            subprocess.run(
                ['enroot', 'remove', container_name],
                capture_output=True,
                check=False
            )
        except Exception as e:
            logger.warning(f'Failed to delete container {container_name}: {e}')

    def get_action_execution_server_startup_command(self):
        return get_action_execution_server_startup_command(
            server_port=self._container_port,
            plugins=self.plugins,
            app_config=self.config,
            main_module=self.main_module,
        )
