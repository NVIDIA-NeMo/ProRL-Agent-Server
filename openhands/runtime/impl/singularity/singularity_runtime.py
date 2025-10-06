import os
import stat
import signal
import subprocess
import json
import time
from functools import lru_cache
from typing import Callable, IO
from uuid import UUID
from pathlib import Path
import threading

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
from openhands.runtime.utils.loopback_ip_allocator import (
    allocate_loopback_ip,
    release_loopback_ip,
    get_loopback_ip,
)
from openhands.runtime.utils.command import (
    DEFAULT_MAIN_MODULE,
    get_action_execution_server_startup_command,
)
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.shutdown_listener import add_shutdown_listener
from openhands.runtime.utils.singularity_runtime_build import (
    build_runtime_image,
    SingularityRuntimeBuilder,
    get_runtime_image_repo,
)
from openhands.utils.tenacity_stop import stop_if_should_exit

CONTAINER_NAME_PREFIX = 'openhands-runtime-'

FILE_VIEWER_PORT_RANGE = (30000, 39999)
VSCODE_PORT_RANGE = (40000, 44999)
APP_PORT_RANGE_1 = (50000, 54999)
APP_PORT_RANGE_2 = (55000, 59999)
MCP_HTTP_PORT_RANGE = (60000, 64999)


def kill_process_tree(pid):
    """Kill a process group containing the container processes."""
    try:
        # Since we start the container with start_new_session=True,
        # we can kill the entire process group
        try:
            # Get the process group ID and kill the entire group
            pgid = os.getpgid(pid)
            logger.info(f'Killing process group {pgid} for process {pid}')

            # Try SIGTERM first for graceful shutdown
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(1)  # Give processes time to shut down gracefully

            # Check if any processes in the group are still alive
            try:
                os.killpg(pgid, 0)  # Test if process group exists
                # If we get here, some processes are still alive - force kill them
                logger.info(f'Force killing process group {pgid}')
                os.killpg(pgid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                logger.info(f'Process group {pgid} terminated successfully')

        except (OSError, ProcessLookupError) as e:
            # Process or process group doesn't exist, try individual process kill as fallback
            logger.info(f'Process group kill failed ({e}), trying individual process kill')
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)

                # Check if still alive, then force kill
                try:
                    os.kill(pid, 0)  # Test if process exists
                    os.kill(pid, signal.SIGKILL)
                    logger.info(f'Force killed individual process {pid}')
                except ProcessLookupError:
                    logger.info(f'Process {pid} terminated successfully')
            except ProcessLookupError:
                logger.info(f'Process {pid} already dead')

    except Exception as e:
        logger.warning(f'Failed to kill process tree {pid}: {e}')


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
            httpx.ConnectError,  # UDS not created yet
        ),
    )


def stop_all_singularity_containers(prefix: str = CONTAINER_NAME_PREFIX):
    """Stop running singularity processes using tracked PIDs."""
    try:
        # Get all active container PIDs from the registry
        pids_to_kill = list(SingularityRuntime._active_container_pids.copy())

        if pids_to_kill:
            logger.info(f'Stopping {len(pids_to_kill)} tracked singularity container processes')

            for pid in pids_to_kill:
                try:
                    logger.info(f'Stopping singularity container process with PID: {pid}')
                    kill_process_tree(pid)
                    # Remove from active registry after successful kill attempt
                    SingularityRuntime._active_container_pids.discard(pid)
                except Exception as e:
                    logger.warning(f'Failed to stop process {pid}: {e}')
        else:
            logger.debug('No tracked singularity container processes to stop')

        # Note: Session port information cleanup is now handled per-session

    except Exception as e:
        logger.warning(f'Failed to stop singularity containers: {e}')


class SingularityRuntime(ActionExecutionClient):
    """This runtime uses Singularity to manage containers for action execution.

    When receive an event, it will send the event to runtime-client which run inside the singularity environment.

    Args:
        config (OpenHandsConfig): The application configuration.
        event_stream (EventStream): The event stream to subscribe to.
        sid (str, optional): The session ID. Defaults to 'default'.
        plugins (list[PluginRequirement] | None, optional): List of plugin requirements. Defaults to None.
        env_vars (dict[str, str] | None, optional): Environment variables to set. Defaults to None.
    """

    _shutdown_listener_id: UUID | None = None
    _active_container_pids: set[int] = set()  # Keep for backward compatibility with shutdown
    _port_allocation_lock = threading.Lock()  # Lock for synchronizing port allocation
    _runtime_builder_lock = threading.Lock()  # Lock for synchronizing runtime builder
    _container_stdout: IO[str] | None = None
    _container_stderr: IO[str] | None = None

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
        if not SingularityRuntime._shutdown_listener_id:
            SingularityRuntime._shutdown_listener_id = add_shutdown_listener(
                lambda: stop_all_singularity_containers(CONTAINER_NAME_PREFIX)
            )

        self.config = config
        self.status_callback = status_callback

        self._vscode_port = -1
        self._file_viewer_port = -1
        self._app_ports: list[int] = []
        self._mcp_http_port = -1
        self._loopback_ip: str | None = None

        # Check if singularity is available
        self._check_singularity_availability()

        # Allocate unique loopback IP for this runtime instance
        try:
            if attach_to_existing:
                # Try to get existing IP first when attaching
                existing_ip = get_loopback_ip(sid)
                if existing_ip:
                    self._loopback_ip = existing_ip
                    logger.info(f'Using existing loopback IP {self._loopback_ip} for runtime session {sid}')
                else:
                    # No existing IP
                    raise RuntimeError(f'No existing loopback IP found for runtime session {sid}')
            else:
                # Always allocate new IP for new containers
                self._loopback_ip = allocate_loopback_ip(sid)
                logger.info(f'Allocated loopback IP {self._loopback_ip} for runtime session {sid}')
        except Exception as e:
            logger.warning(f'Failed to allocate loopback IP for session {sid}: {e}')
            self._loopback_ip = None

        self.base_container_image = self.config.sandbox.base_container_image
        self.runtime_container_image = self.config.sandbox.runtime_container_image
        self.container_name = CONTAINER_NAME_PREFIX + sid
        self.container_process: subprocess.Popen[str] | None = None
        self.container_pid: int | None = None  # Store the actual container PID
        self.main_module = main_module
        self.runtime_builder = SingularityRuntimeBuilder()
        self.headless_mode = headless_mode
        self._container_stdout = None
        self._container_stderr = None

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

    def _get_session_storage_path(self) -> str:
        """Get file path for storing session port information."""
        return f'runtime/singularity_sessions/{self.sid}.json'

    def _save_session_port_info(self, session_info: dict) -> None:
        """Save session port information to file store."""
        try:
            json_str = json.dumps(session_info)
            self.event_stream.file_store.write(self._get_session_storage_path(), json_str)
        except Exception as e:
            logger.warning(f'Failed to save session port info: {e}')

    def _load_session_port_info(self) -> dict | None:
        """Load session port information from file store."""
        try:
            json_str = self.event_stream.file_store.read(self._get_session_storage_path())
            return json.loads(json_str)
        except (FileNotFoundError, json.JSONDecodeError, Exception):
            return None

    def _delete_session_port_info(self) -> None:
        """Delete session port information from file store."""
        try:
            self.event_stream.file_store.delete(self._get_session_storage_path())
        except (FileNotFoundError, Exception):
            pass  # Ignore if file doesn't exist

    @staticmethod
    def _check_singularity_availability():
        """Check if Singularity is available on the system."""
        try:
            result = subprocess.run(
                ['singularity', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f'Singularity version: {result.stdout.strip()}')
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                'Singularity is not available on this system. Please install Singularity to use SingularityRuntime.'
            ) from e

    def _run_singularity_command(self, cmd: list[str], check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a singularity command and return the result."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                check=check
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f'Singularity command failed: {" ".join(cmd)}')
            logger.error(f'Error: {e.stderr}')
            raise

    def _image_exists(self) -> bool:
        """Check if the Singularity image file exists."""
        image_path = self._get_singularity_image_path()
        return os.path.exists(image_path)

    def _is_container_running(self) -> bool:
        """Check if the container is currently running."""
        # If we have a container_process (started by us), check if it's running
        if self.container_process is not None:
            return self.container_process.poll() is None

        # If we attached to an existing container, check by PID
        if self.container_pid is not None:
            try:
                # Check if process exists (os.kill with signal 0 doesn't kill, just checks existence)
                os.kill(self.container_pid, 0)
                return True
            except (OSError, ProcessLookupError):
                # Process doesn't exist
                return False

        # No container process or PID available
        return False

    async def connect(self):
        # log the headless mode
        self.send_status_message('STATUS$STARTING_RUNTIME')
        try:
            await call_sync_from_async(self._attach_to_container)
            if self.attach_to_existing:
                # We successfully attached to an existing container
                self.log('info', f'Successfully attached to existing container {self.container_name}')
        except Exception as e:
            logger.info(f'error: {e}')
            logger.info(f'connect to container: {self.sid} Unable to attach to existing container')
            if self.attach_to_existing:
                self.log(
                    'warning',
                    f'Container {self.container_name} not found or not running.',
                )
                raise AgentRuntimeDisconnectedError from e

        if not self._image_exists():
            self.log('info', f'Runtime Image {self.runtime_container_image} not found.')
            self.maybe_prepare_runtime_container_image()
            self.log(
                'info', f'Starting runtime with image: {self.runtime_container_image}'
            )

        if not self.attach_to_existing:
            self.log('info', f'Container Image: {self.runtime_container_image} is ready, launching runtime...')
            self.send_status_message('STATUS$WAITING_FOR_CLIENT')

        # Only call init_container if we're not attaching to an existing container
        if not self.attach_to_existing:
            await call_sync_from_async(self.init_container)

        await call_sync_from_async(self.wait_until_alive)

        if not self.attach_to_existing:
            self.log('info', 'Runtime is ready.')
        else:
            self.log('info', 'Attached to existing runtime.')

        if not self.attach_to_existing:
            await call_sync_from_async(self.setup_initial_env)

        self.log(
            'info',
            f'Container initialized with plugins: {[plugin.name for plugin in self.plugins]}. VSCode URL: {self.vscode_url} Headless mode: {self.headless_mode}',
        )
        if not self.attach_to_existing:
            self.send_status_message(' ')
        self._runtime_initialized = True

        self.log(
            'info',
            f'Container started: {self.container_name}. VSCode URL: {self.vscode_url}',
        )

    def maybe_prepare_runtime_container_image(self):
        """Prepare the runtime container image for Singularity."""
        if self.runtime_container_image is None:
            if self.base_container_image is None:
                raise ValueError(
                    'Neither runtime container image nor base container image is set'
                )
            self.send_status_message('STATUS$STARTING_CONTAINER')
            # add lock
            with SingularityRuntime._runtime_builder_lock:
                self.runtime_container_image = build_runtime_image(
                    self.base_container_image,
                    self.runtime_builder,
                    platform=self.config.sandbox.platform,
                    extra_deps=self.config.sandbox.runtime_extra_deps,
                    force_rebuild=self.config.sandbox.force_rebuild_runtime,
                    extra_build_args=self.config.sandbox.runtime_extra_build_args,
                )
        else:
            # It has runtime container image in the dockerhub, so we need to pull it
            # Pull the image if it doesn't exist locally
            self._pull_image_if_needed()

    def _pull_image_if_needed(self):
        """Pull the container image for Singularity if not already available."""
        try:
            image_path = self._get_singularity_image_path()

            if os.path.exists(image_path):
                logger.debug(f'Image {image_path} already available')
                return

            # Pull the image
            logger.info(f'Pulling image {self.runtime_container_image} for Singularity...')
            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            pull_cmd = [
                'singularity', 'pull', '--disable-cache',
                image_path,
                f'docker://{self.runtime_container_image}'
            ]
            self._run_singularity_command(pull_cmd)
            logger.info(f'Successfully pulled {self.runtime_container_image}')

        except Exception as e:
            logger.error(f'Failed to pull image: {e}')
            raise

    def _get_singularity_image_path(self) -> str:
        """Get the Singularity image file path."""
        if self.runtime_container_image:
            if not self.runtime_container_image.endswith('.sif'):
                image_name = self.runtime_container_image.replace(':', '_').replace('/', '_')
                image_repo = get_runtime_image_repo()
                os.makedirs(image_repo, exist_ok=True)
                return f'{image_repo}/{image_name}.sif'
            else:
                return self.runtime_container_image
        return ''

    def _process_volumes(self) -> list[str]:
        """Process volume mounts for Singularity format.

        Returns:
            A list of volume mount arguments for Singularity.
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
                    # Singularity mount format: --bind host_path:container_path
                    mount_args.extend(['--bind', f'{host_path}:{container_path}'])
                    logger.debug(
                        f'Mount dir (sandbox.volumes): {host_path} to {container_path}'
                    )

        # Legacy mounting with workspace_* parameters
        elif (
            self.config.workspace_mount_path is not None
            and self.config.workspace_mount_path_in_sandbox is not None
        ):
            mount_args.extend([
                '--bind',
                f'{self.config.workspace_mount_path}:{self.config.workspace_mount_path_in_sandbox}'
            ])
            logger.debug(
                f'Mount dir (legacy): {self.config.workspace_mount_path}'
            )

        return mount_args

    def init_container(self):
        self.log('debug', 'Preparing to start Singularity container...')
        self.send_status_message('STATUS$PREPARING_CONTAINER')

        # Use the configured vscode_port if provided, otherwise find an available port
        self._vscode_port = (
            self.config.sandbox.vscode_port
            or self._find_available_port(VSCODE_PORT_RANGE)
        )
        self._file_viewer_port = self._find_available_port(FILE_VIEWER_PORT_RANGE)

        self._app_ports = [
            self._find_available_port(APP_PORT_RANGE_1),
            self._find_available_port(APP_PORT_RANGE_2),
        ]
        self._mcp_http_port = self._find_available_port(MCP_HTTP_PORT_RANGE)

        # Prepare environment variables
        env_vars = {
            'PYTHONUNBUFFERED': '1',
            'VSCODE_PORT': str(self._vscode_port),
            'LOOPBACK_IP': self.loopback_ip,
            'FILE_VIEWER_PORT': str(self._file_viewer_port),
            'APP_PORT_1': str(self._app_ports[0]),
            'APP_PORT_2': str(self._app_ports[1]),
            'MCP_HTTP_PORT': str(self._mcp_http_port),
            'PIP_BREAK_SYSTEM_PACKAGES': '1',
            'OPENHANDS_SESSION_ID': self.sid,
            'OMP_NUM_THREADS': '4', # Limit threads at container startup
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
            # Get the image path
            image_path = self._get_singularity_image_path()
            if not os.path.exists(image_path):
                raise RuntimeError(f'Singularity image not found: {image_path}')

            # Build the singularity exec command
            cmd = ['singularity', 'run', '--pid', '--writable-tmpfs', '--no-mount', 'home,cwd,tmp', '--home', '/root', '--workdir', '/workspace']

            # add --fakeroot if configured
            if self.config.sandbox.run_as_fakeroot:
                cmd.extend(['--fakeroot'])

            # Add network isolation if configured
            if self.config.sandbox.isolate_network:
                cmd.extend(['--network', 'none'])

            # Add environment variables
            for key, value in env_vars.items():
                cmd.extend(['--env', f'{key}={value}'])

            # Add volume mounts
            cmd.extend(mount_args)

            # make /tmp/runtime directory if not exists
            if not os.path.exists('/tmp/runtime'):
                os.makedirs('/tmp/runtime')

            # Add UDS socket directory mount for communication
            cmd.extend(['--bind', '/tmp/runtime:/tmp/runtime'])

            cmd.extend(['--pwd', '/openhands/code'])

            # Add image path
            cmd.append(image_path)

            # Add startup command
            cmd.extend(startup_command)

            self.log('debug', f'Starting Singularity container with command: {" ".join(cmd)}')

            # Start the container process in a new process group for easier cleanup
            # Redirect stdout/stderr to per-runtime files to avoid PIPE backpressure
            log_dir = '/tmp/openhands_singularity_logs'
            os.makedirs(log_dir, exist_ok=True)
            stdout_path = os.path.join(log_dir, f'{self.sid}.out')
            stderr_path = os.path.join(log_dir, f'{self.sid}.err')
            self._container_stdout = open(stdout_path, 'w')
            self._container_stderr = open(stderr_path, 'w')

            self.container_process = subprocess.Popen(
                cmd,
                stdout=self._container_stdout,
                stderr=self._container_stderr,
                text=True,
                env=dict(os.environ, **env_vars),
                start_new_session=True  # Creates a new session and process group
            )

            assert self.container_process is not None

            # Save the container PID and add to active registry
            self.container_pid = self.container_process.pid
            SingularityRuntime._active_container_pids.add(self.container_pid)

            # Check if the process started successfully
            if self.container_process.poll() is not None:
                # Process failed to start, remove from registry
                if self.container_pid in SingularityRuntime._active_container_pids:
                    SingularityRuntime._active_container_pids.remove(self.container_pid)
                stdout, stderr = self.container_process.communicate()
                raise RuntimeError(
                    f'Process Poll: Container failed to start. Return code: {self.container_process.returncode}\n'
                    f'Stdout: {stdout}\nStderr: {stderr}'
                )

            # Store session information for later attachment
            session_info = {
                'pid': self.container_pid,
                'ports': {
                    'vscode_port': self._vscode_port,
                    'file_viewer_port': self._file_viewer_port,
                    'app_port_1': self._app_ports[0],
                    'app_port_2': self._app_ports[1],
                    'mcp_http_port': self._mcp_http_port,
                }
            }
            self._save_session_port_info(session_info)

            self.log('debug', f'Container started. PID: {self.container_pid}')
            self.send_status_message('STATUS$CONTAINER_STARTED')

        except Exception as e:
            self.log(
                'error',
                f'Error: Instance {self.container_name} FAILED to start container!\n',
            )
            self.log('error', str(e))
            self.close()
            raise e

    def _attach_to_container(self):
        """Attach to an existing container."""
        # Get port information from class-level session registry
        session_info = self._load_session_port_info()
        if not session_info:
            logger.info(f'attach to container: {self.sid} No session info found in file store')
            raise AgentRuntimeNotFoundError(f'Container {self.container_name} not found or not running.')

        self.container_pid = session_info['pid']
        port_info = session_info['ports']

        # Verify the process is still running
        if self.container_pid is not None:
            try:
                os.kill(self.container_pid, 0)  # Check if process exists
            except (OSError, ProcessLookupError):
                # Process doesn't exist anymore, clean up
                self._delete_session_port_info()
                SingularityRuntime._active_container_pids.discard(self.container_pid)
                raise AgentRuntimeNotFoundError(f'Container {self.container_name} process no longer running.')
        else:
            # If we don't have a PID, something went wrong
            raise AgentRuntimeNotFoundError(f'Container {self.container_name} has no valid PID.')

        # Set up port information from stored data
        self._vscode_port = port_info['vscode_port']
        self._file_viewer_port = port_info['file_viewer_port']
        self._app_ports = [port_info['app_port_1'], port_info['app_port_2']]
        self._mcp_http_port = port_info['mcp_http_port']
        self.log(
            'debug',
            f'Attached to container: {self.container_name} PID: {self.container_pid}',
        )

    @tenacity.retry(
        stop=tenacity.stop_after_delay(240) | stop_if_should_exit(),
        retry=tenacity.retry_if_exception(_is_retryablewait_until_alive_error),
        reraise=True,
        wait=tenacity.wait_fixed(1),
    )
    def wait_until_alive(self):
        if not self._is_container_running():
            # need to see the error message from the container
            if self.container_process is not None:
                stdout, stderr = self.container_process.communicate()
                logger.error(f'Container {self.container_name} is not running. Stdout: {stdout}, Stderr: {stderr}')
            else:
                logger.error(f'Container {self.container_name} is not running. No process information available.')
            raise AgentRuntimeDisconnectedError(
                f'Container {self.container_name} is not running.'
            )

        self.check_if_alive()

    def close(self, rm_all_containers: bool | None = None):
        """Closes the SingularityRuntime and associated objects

        Parameters:
        - rm_all_containers (bool): Whether to stop all container processes with the prefix
        """


        super().close()

        # Close container log file handles if open
        try:
            if self._container_stdout is not None:
                self._container_stdout.close()
                self._container_stdout = None
        except Exception as e:
            logger.warning(f'Failed to close container stdout for {self.container_name}: {e}')
        try:
            if self._container_stderr is not None:
                self._container_stderr.close()
                self._container_stderr = None
        except Exception as e:
            logger.warning(f'Failed to close container stderr for {self.container_name}: {e}')

        if rm_all_containers is None:
            rm_all_containers = self.config.sandbox.rm_all_containers

        if self.config.sandbox.keep_runtime_alive or self.attach_to_existing:
            # Don't clean up resources when keeping runtime alive or when we only attached
            return

        # Clean up session port information
        self._delete_session_port_info()

        # Release the allocated loopback IP (only when actually closing, not when attaching)
        if self._loopback_ip:
            try:
                release_loopback_ip(self.sid)
                logger.info(f'Released loopback IP {self._loopback_ip} for session {self.sid}')
            except Exception as e:
                logger.warning(f'Failed to release loopback IP {self._loopback_ip} for session {self.sid}: {e}')
            self._loopback_ip = None

        # clean up the socket file
        socket_file = f'/tmp/runtime/{self.sid}.sock'
        if os.path.exists(socket_file):
            os.remove(socket_file)

        # Stop Singularity processes
        if rm_all_containers:
            stop_all_singularity_containers(CONTAINER_NAME_PREFIX)
        else:
            # Stop only this specific container's process using the stored PID
            if self.container_pid is not None:
                try:
                    logger.info(f'Stopping Singularity container {self.container_name} with PID: {self.container_pid}')
                    kill_process_tree(self.container_pid)
                    SingularityRuntime._active_container_pids.discard(self.container_pid)
                    self.container_pid = None
                except Exception as e:
                    logger.warning(f'Failed to stop container process {self.container_name}: {e}')
            else:
                logger.debug(f'No PID stored for container {self.container_name}, nothing to stop')

    def _find_available_port(self, port_range, max_attempts=5):
        """Find an available port in the given range with proper synchronization."""
        with SingularityRuntime._port_allocation_lock:
            for _ in range(max_attempts):
                port = find_available_tcp_port(port_range[0], port_range[1])
                return port
            raise RuntimeError(f'Failed to find available port in range {port_range} after {max_attempts} attempts')

    @property
    def loopback_ip(self) -> str | None:
        """Get the unique loopback IP assigned to this runtime instance."""
        return self._loopback_ip

    @property
    def action_execution_server_url(self) -> str:
        """Get the action execution server URL assigned to this runtime instance."""
        return f'http://{self.loopback_ip}:{self._mcp_http_port}'

    @property
    def vscode_url(self) -> str | None:
        token = super().get_vscode_token()
        if not token:
            return None

        vscode_url = f'http://{self.loopback_ip}:{self._vscode_port}/?tkn={token}&folder={self.config.workspace_mount_path_in_sandbox}'
        return vscode_url

    @property
    def web_hosts(self):
        hosts: dict[str, int] = {}

        host_addr = self.loopback_ip  # Singularity typically runs on localhost
        for port in self._app_ports:
            hosts[f'http://{host_addr}:{port}'] = port

        return hosts

    def pause(self):
        """Pause the runtime by stopping the container process."""
        raise NotImplementedError('Pause is not implemented for SingularityRuntime')

    def resume(self):
        """Resume the runtime by restarting the container."""
        raise NotImplementedError('Resume is not implemented for SingularityRuntime')

    @classmethod
    async def delete(cls, conversation_id: str):
        """Delete container resources by conversation ID."""
        try:
            # For Singularity, we mainly need to clean up any running processes
            # and optionally remove the image file
            container_name = CONTAINER_NAME_PREFIX + conversation_id

            # Optionally clean up image files (commented out to preserve for reuse)
            image_repo = get_runtime_image_repo()
            image_path = f'{image_repo}/{container_name}.sif'
            if os.path.exists(image_path):
                logger.info(f'Removing image file {image_path}')
                # os.remove(image_path)

        except Exception as e:
            logger.warning(f'Failed to delete container {conversation_id}: {e}')

    def get_action_execution_server_startup_command(self):
        # Use dummy port since we communicate via UDS
        return get_action_execution_server_startup_command(
            server_port=0,  # Dummy port - UDS is used for actual communication
            plugins=self.plugins,
            app_config=self.config,
            main_module=self.main_module,
        )
