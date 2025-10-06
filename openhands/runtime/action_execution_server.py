"""
This is the main file for the runtime client.
It is responsible for executing actions received from OpenHands backend and producing observations.

NOTE: this will be executed inside the docker sandbox.
"""

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import shutil
import sys
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from zipfile import ZipFile

from anyio import ClosedResourceError, EndOfStream
from binaryornot.check import is_binary
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from mcpm import MCPRouter, RouterConfig
from mcpm.router.router import RouterSseTransport
from mcpm.router.router import logger as mcp_router_logger
from openhands_aci.editor.editor import OHEditor
from openhands_aci.editor.exceptions import ToolError
from openhands_aci.editor.results import ToolResult
from openhands_aci.utils.diff import get_diff
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from uvicorn.config import Config
from uvicorn.server import Server

from openhands.core.exceptions import BrowserUnavailableException
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
)
from openhands.events.event import FileEditSource, FileReadSource
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileEditObservation,
    FileReadObservation,
    FileWriteObservation,
    IPythonRunCellObservation,
    Observation,
)
from openhands.events.serialization import event_from_dict, event_to_dict
from openhands.runtime.browser import browse
from openhands.runtime.browser.browser_env import BrowserEnv
from openhands.runtime.file_viewer_server import start_file_viewer_server
from openhands.runtime.plugins import (
    ALL_PLUGINS,
    DirectJupyterPlugin,
    Plugin,
    VSCodePlugin,
)
from openhands.runtime.utils.async_bash import AsyncBashSession
from openhands.runtime.utils.bash import BashSession
from openhands.runtime.utils.efficient_bash import EfficientBashSession
from openhands.runtime.utils.files import insert_lines, read_lines
from openhands.runtime.utils.log_capture import capture_logs
from openhands.runtime.utils.memory_monitor import MemoryMonitor
from openhands.runtime.utils.runtime_init import init_user_and_working_directory
from openhands.runtime.utils.system_stats import get_system_stats
from openhands.utils.async_utils import call_sync_from_async, wait_all

# Set MCP router logger to the same level as the main logger
mcp_router_logger.setLevel(logger.getEffectiveLevel())


if sys.platform == 'win32':
    from openhands.runtime.utils.windows_bash import WindowsPowershellSession


class ActionRequest(BaseModel):
    action: dict


ROOT_GID = 0

SESSION_API_KEY = os.environ.get('SESSION_API_KEY')
api_key_header = APIKeyHeader(name='X-Session-API-Key', auto_error=False)


def verify_api_key(api_key: str = Depends(api_key_header)):
    if SESSION_API_KEY and api_key != SESSION_API_KEY:
        raise HTTPException(status_code=403, detail='Invalid API Key')
    return api_key


def _execute_file_editor(
    editor: OHEditor,
    command: str,
    path: str,
    file_text: str | None = None,
    view_range: list[int] | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | str | None = None,
    enable_linting: bool = False,
) -> tuple[str, tuple[str | None, str | None]]:
    """Execute file editor command and handle exceptions.

    Args:
        editor: The OHEditor instance
        command: Editor command to execute
        path: File path
        file_text: Optional file text content
        view_range: Optional view range tuple (start, end)
        old_str: Optional string to replace
        new_str: Optional replacement string
        insert_line: Optional line number for insertion (can be int or str)
        enable_linting: Whether to enable linting

    Returns:
        tuple: A tuple containing the output string and a tuple of old and new file content
    """
    result: ToolResult | None = None

    # Convert insert_line from string to int if needed
    if insert_line is not None and isinstance(insert_line, str):
        try:
            insert_line = int(insert_line)
        except ValueError:
            return (
                f"ERROR:\nInvalid insert_line value: '{insert_line}'. Expected an integer.",
                (None, None),
            )

    try:
        result = editor(
            command=command,
            path=path,
            file_text=file_text,
            view_range=view_range,
            old_str=old_str,
            new_str=new_str,
            insert_line=insert_line,
            enable_linting=enable_linting,
        )
    except ToolError as e:
        result = ToolResult(error=e.message)
    except TypeError as e:
        # Handle unexpected arguments or type errors
        return f'ERROR:\n{str(e)}', (None, None)

    if result.error:
        return f'ERROR:\n{result.error}', (None, None)

    if not result.output:
        logger.warning(f'No output from file_editor for {path}')
        return '', (None, None)

    return result.output, (result.old_content, result.new_content)


class ActionExecutor:
    """ActionExecutor is running inside docker sandbox.
    It is responsible for executing actions received from OpenHands backend and producing observations.
    """

    def __init__(
        self,
        plugins_to_load: list[Plugin],
        work_dir: str,
        username: str,
        user_id: int,
        browsergym_eval_env: str | None,
    ) -> None:
        self.plugins_to_load = plugins_to_load
        self._initial_cwd = work_dir
        self.username = username
        self.user_id = user_id
        _updated_user_id = init_user_and_working_directory(
            username=username, user_id=self.user_id, initial_cwd=work_dir
        )
        if _updated_user_id is not None:
            self.user_id = _updated_user_id

        self.bash_session: BashSession | EfficientBashSession | None = None  # type: ignore[name-defined]
        self.lock = asyncio.Lock()
        self.plugins: dict[str, Plugin] = {}
        self.file_editor = OHEditor(workspace_root=self._initial_cwd)
        self.browser: BrowserEnv | None = None
        self.browser_init_task: asyncio.Task | None = None
        self.browsergym_eval_env = browsergym_eval_env

        self.start_time = time.time()
        self.last_execution_time = self.start_time
        self._initialized = False

        self.max_memory_gb: int | None = None
        if _override_max_memory_gb := os.environ.get('RUNTIME_MAX_MEMORY_GB', None):
            self.max_memory_gb = int(_override_max_memory_gb)
            logger.info(
                f'Setting max memory to {self.max_memory_gb}GB (according to the RUNTIME_MAX_MEMORY_GB environment variable)'
            )
        else:
            logger.info('No max memory limit set, using all available system memory')

        self.memory_monitor = MemoryMonitor(
            enable=os.environ.get('RUNTIME_MEMORY_MONITOR', 'False').lower()
            in ['true', '1', 'yes']
        )
        self.memory_monitor.start_monitoring()

    @property
    def initial_cwd(self):
        return self._initial_cwd

    async def _init_browser_async(self):
        """Initialize the browser asynchronously."""
        if sys.platform == 'win32':
            logger.warning('Browser environment not supported on windows')
            return

        logger.debug('Initializing browser asynchronously')
        try:
            self.browser = BrowserEnv(self.browsergym_eval_env)
            logger.debug('Browser initialized asynchronously')
        except Exception as e:
            logger.error(f'Failed to initialize browser: {e}')
            self.browser = None

    async def _ensure_browser_ready(self):
        """Ensure the browser is ready for use."""

        if self.browser is None:
            if self.browser_init_task is None:
                # Start browser initialization if it hasn't been started
                self.browser_init_task = asyncio.create_task(self._init_browser_async())
            elif self.browser_init_task.done():
                # If the task is done but browser is still None, restart initialization
                self.browser_init_task = asyncio.create_task(self._init_browser_async())

            # Wait for browser to be initialized
            if self.browser_init_task:
                logger.debug('Waiting for browser to be ready...')
                await self.browser_init_task

            # Check if browser was successfully initialized
            if self.browser is None:
                raise BrowserUnavailableException('Browser initialization failed')

        # If we get here, the browser is ready
        logger.debug('Browser is ready')

    async def ainit(self):
        # bash needs to be initialized first
        logger.debug('Initializing bash session')
        if sys.platform == 'win32':
            self.bash_session = WindowsPowershellSession(  # type: ignore[name-defined]
                work_dir=self._initial_cwd,
                username=self.username,
                no_change_timeout_seconds=int(
                    os.environ.get('NO_CHANGE_TIMEOUT_SECONDS', 10)
                ),
                max_memory_mb=self.max_memory_gb * 1024 if self.max_memory_gb else None,
            )
        else:
            self.bash_session = EfficientBashSession(
                # self.bash_session = BashSession(
                work_dir=self._initial_cwd,
                username=self.username,
                no_change_timeout_seconds=int(
                    os.environ.get('NO_CHANGE_TIMEOUT_SECONDS', 10)
                ),
                max_memory_mb=self.max_memory_gb * 1024 if self.max_memory_gb else None,
            )
            self.bash_session.initialize()
        logger.debug('Bash session initialized')

        if self.browsergym_eval_env == 'skip':
            logger.debug('Skipping browser initialization')
            self.browser_init_task = None
        else:
            self.browser_init_task = asyncio.create_task(self._init_browser_async())
            logger.debug('Browser initialization started in background')

        await wait_all(
            (self._init_plugin(plugin) for plugin in self.plugins_to_load),
            timeout=int(os.environ.get('INIT_PLUGIN_TIMEOUT', '120')),
        )
        logger.debug('All plugins initialized')

        # This is a temporary workaround
        # TODO: refactor AgentSkills to be part of JupyterPlugin
        # AFTER ServerRuntime is deprecated
        logger.debug('Initializing AgentSkills')
        if 'agent_skills' in self.plugins and 'direct_jupyter' in self.plugins:
            obs = await self.run_ipython(
                IPythonRunCellAction(
                    code='from openhands.runtime.plugins.agent_skills.agentskills import *\n'
                )
            )
            logger.debug(f'AgentSkills initialized: {obs}')

        logger.debug('Initializing bash commands')
        await self._init_bash_commands()

        logger.debug('Runtime client initialized.')
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def _init_plugin(self, plugin: Plugin):
        assert self.bash_session is not None
        await plugin.initialize(self.username)
        self.plugins[plugin.name] = plugin
        logger.debug(f'Initializing plugin: {plugin.name}')

        if isinstance(plugin, DirectJupyterPlugin):
            # Escape backslashes in Windows path
            cwd = self.bash_session.cwd.replace('\\', '/')
            await self.run_ipython(
                IPythonRunCellAction(code=f'import os; os.chdir(r"{cwd}")')
            )

    async def _init_bash_commands(self):
        INIT_COMMANDS = []
        is_local_runtime = os.environ.get('LOCAL_RUNTIME_MODE') == '1'
        is_windows = sys.platform == 'win32'

        # Determine git config commands based on platform and runtime mode
        if is_local_runtime:
            if is_windows:
                # Windows, local - split into separate commands
                INIT_COMMANDS.append(
                    'git config --file ./.git_config user.name "openhands"'
                )
                INIT_COMMANDS.append(
                    'git config --file ./.git_config user.email "openhands@all-hands.dev"'
                )
                INIT_COMMANDS.append(
                    '$env:GIT_CONFIG = (Join-Path (Get-Location) ".git_config")'
                )
            else:
                # Linux/macOS, local
                base_git_config = (
                    'git config --file ./.git_config user.name "openhands" && '
                    'git config --file ./.git_config user.email "openhands@all-hands.dev" && '
                    'export GIT_CONFIG=$(pwd)/.git_config'
                )
                INIT_COMMANDS.append(base_git_config)
        else:
            # Non-local (implies Linux/macOS)
            base_git_config = (
                'git config --global user.name "openhands" && '
                'git config --global user.email "openhands@all-hands.dev"'
            )
            INIT_COMMANDS.append(base_git_config)

        # Determine no-pager command
        if is_windows:
            no_pager_cmd = 'function git { git.exe --no-pager $args }'
        else:
            no_pager_cmd = 'alias git="git --no-pager"'

        INIT_COMMANDS.append(no_pager_cmd)

        logger.info(f'Initializing by running {len(INIT_COMMANDS)} bash commands...')
        for command in INIT_COMMANDS:
            action = CmdRunAction(command=command)
            action.set_hard_timeout(300)
            logger.debug(f'Executing init command: {command}')
            obs = await self.run(action)
            assert isinstance(obs, CmdOutputObservation)
            logger.debug(
                f'Init command outputs (exit code: {obs.exit_code}): {obs.content}'
            )
            assert obs.exit_code == 0
        logger.debug('Bash init commands completed')

    async def run_action(self, action) -> Observation:
        async with self.lock:
            action_type = action.action
            observation = await getattr(self, action_type)(action)
            return observation

    async def run(
        self, action: CmdRunAction
    ) -> CmdOutputObservation | ErrorObservation:
        try:
            if action.is_static:
                path = action.cwd or self._initial_cwd
                result = await AsyncBashSession.execute(action.command, path)
                obs = CmdOutputObservation(
                    content=result.content,
                    exit_code=result.exit_code,
                    command=action.command,
                )
                return obs

            assert self.bash_session is not None
            if isinstance(self.bash_session, EfficientBashSession):
                obs = await self.bash_session.execute(action)  # type: ignore
            else:
                obs = await call_sync_from_async(self.bash_session.execute, action)
            return obs
        except Exception as e:
            logger.error(f'Error running command: {e}')
            return ErrorObservation(str(e))

    async def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        assert self.bash_session is not None
        if 'direct_jupyter' in self.plugins:
            _jupyter_plugin: DirectJupyterPlugin = self.plugins['direct_jupyter']  # type: ignore
            # This is used to make AgentSkills in Jupyter aware of the
            # current working directory in Bash
            jupyter_cwd = getattr(self, '_jupyter_cwd', None)
            if self.bash_session.cwd != jupyter_cwd:
                logger.debug(
                    f'{self.bash_session.cwd} != {jupyter_cwd} -> reset Jupyter PWD'
                )
                # escape windows paths
                cwd = self.bash_session.cwd.replace('\\', '/')
                reset_jupyter_cwd_code = f'import os; os.chdir("{cwd}")'
                _aux_action = IPythonRunCellAction(code=reset_jupyter_cwd_code)
                _reset_obs: IPythonRunCellObservation = await _jupyter_plugin.run(
                    _aux_action
                )
                logger.debug(
                    f'Changed working directory in IPython to: {self.bash_session.cwd}. Output: {_reset_obs}'
                )
                self._jupyter_cwd = self.bash_session.cwd

            obs: IPythonRunCellObservation = await _jupyter_plugin.run(action)
            obs.content = obs.content.rstrip()

            if action.include_extra:
                obs.content += (
                    f'\n[Jupyter current working directory: {self.bash_session.cwd}]'
                )
                obs.content += f'\n[Jupyter Python interpreter: {_jupyter_plugin.python_interpreter_path}]'
            return obs
        else:
            raise RuntimeError(
                'Direct JupyterRequirement not found. Unable to run IPython action.'
            )

    def _resolve_path(self, path: str, working_dir: str) -> str:
        filepath = Path(path)
        if not filepath.is_absolute():
            return str(Path(working_dir) / filepath)
        return str(filepath)

    async def read(self, action: FileReadAction) -> Observation:
        assert self.bash_session is not None

        # Cannot read binary files
        if is_binary(action.path):
            return ErrorObservation('ERROR_BINARY_FILE')

        if action.impl_source == FileReadSource.OH_ACI:
            result_str, _ = _execute_file_editor(
                self.file_editor,
                command='view',
                path=action.path,
                view_range=action.view_range,
            )

            return FileReadObservation(
                content=result_str,
                path=action.path,
                impl_source=FileReadSource.OH_ACI,
            )

        # NOTE: the client code is running inside the sandbox,
        # so there's no need to check permission
        working_dir = self.bash_session.cwd
        filepath = self._resolve_path(action.path, working_dir)
        try:
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                with open(filepath, 'rb') as file:  # noqa: ASYNC101
                    image_data = file.read()
                    encoded_image = base64.b64encode(image_data).decode('utf-8')
                    mime_type, _ = mimetypes.guess_type(filepath)
                    if mime_type is None:
                        mime_type = 'image/png'  # default to PNG if mime type cannot be determined
                    encoded_image = f'data:{mime_type};base64,{encoded_image}'

                return FileReadObservation(path=filepath, content=encoded_image)
            elif filepath.lower().endswith('.pdf'):
                with open(filepath, 'rb') as file:  # noqa: ASYNC101
                    pdf_data = file.read()
                    encoded_pdf = base64.b64encode(pdf_data).decode('utf-8')
                    encoded_pdf = f'data:application/pdf;base64,{encoded_pdf}'
                return FileReadObservation(path=filepath, content=encoded_pdf)
            elif filepath.lower().endswith(('.mp4', '.webm', '.ogg')):
                with open(filepath, 'rb') as file:  # noqa: ASYNC101
                    video_data = file.read()
                    encoded_video = base64.b64encode(video_data).decode('utf-8')
                    mime_type, _ = mimetypes.guess_type(filepath)
                    if mime_type is None:
                        mime_type = 'video/mp4'  # default to MP4 if MIME type cannot be determined
                    encoded_video = f'data:{mime_type};base64,{encoded_video}'

                return FileReadObservation(path=filepath, content=encoded_video)

            with open(filepath, 'r', encoding='utf-8') as file:  # noqa: ASYNC101
                lines = read_lines(file.readlines(), action.start, action.end)
        except FileNotFoundError:
            return ErrorObservation(
                f'File not found: {filepath}. Your current working directory is {working_dir}.'
            )
        except UnicodeDecodeError:
            return ErrorObservation(f'File could not be decoded as utf-8: {filepath}.')
        except IsADirectoryError:
            return ErrorObservation(
                f'Path is a directory: {filepath}. You can only read files'
            )

        code_view = ''.join(lines)
        return FileReadObservation(path=filepath, content=code_view)

    async def write(self, action: FileWriteAction) -> Observation:
        assert self.bash_session is not None
        working_dir = self.bash_session.cwd
        filepath = self._resolve_path(action.path, working_dir)

        insert = action.content.split('\n')
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        file_exists = os.path.exists(filepath)
        if file_exists:
            file_stat = os.stat(filepath)
        else:
            file_stat = None

        mode = 'w' if not file_exists else 'r+'
        try:
            with open(filepath, mode, encoding='utf-8') as file:  # noqa: ASYNC101
                if mode != 'w':
                    all_lines = file.readlines()
                    new_file = insert_lines(insert, all_lines, action.start, action.end)
                else:
                    new_file = [i + '\n' for i in insert]

                file.seek(0)
                file.writelines(new_file)
                file.truncate()

        except FileNotFoundError:
            return ErrorObservation(f'File not found: {filepath}')
        except IsADirectoryError:
            return ErrorObservation(
                f'Path is a directory: {filepath}. You can only write to files'
            )
        except UnicodeDecodeError:
            return ErrorObservation(f'File could not be decoded as utf-8: {filepath}')

        # Attempt to handle file permissions
        try:
            if file_exists:
                assert file_stat is not None
                # restore the original file permissions if the file already exists
                os.chmod(filepath, file_stat.st_mode)
                os.chown(filepath, file_stat.st_uid, file_stat.st_gid)
            else:
                # set the new file permissions if the file is new
                os.chmod(filepath, 0o664)
                os.chown(filepath, self.user_id, self.user_id)
        except PermissionError as e:
            return ErrorObservation(
                f'File {filepath} written, but failed to change ownership and permissions: {e}'
            )
        return FileWriteObservation(content='', path=filepath)

    async def edit(self, action: FileEditAction) -> Observation:
        assert action.impl_source == FileEditSource.OH_ACI
        result_str, (old_content, new_content) = _execute_file_editor(
            self.file_editor,
            command=action.command,
            path=action.path,
            file_text=action.file_text,
            old_str=action.old_str,
            new_str=action.new_str,
            insert_line=action.insert_line,
            enable_linting=False,
        )

        return FileEditObservation(
            content=result_str,
            path=action.path,
            old_content=action.old_str,
            new_content=action.new_str,
            impl_source=FileEditSource.OH_ACI,
            diff=get_diff(
                old_contents=old_content or '',
                new_contents=new_content or '',
                filepath=action.path,
            ),
        )

    async def browse(self, action: BrowseURLAction) -> Observation:
        if self.browsergym_eval_env == 'skip':
            return ErrorObservation('Browser functionality is disabled.')
        if self.browser is None:
            return ErrorObservation(
                'Browser functionality is not supported on Windows.'
            )
        await self._ensure_browser_ready()
        return await browse(action, self.browser, self.initial_cwd)

    async def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        if self.browsergym_eval_env == 'skip':
            return ErrorObservation('Browser functionality is disabled.')
        if self.browser is None:
            return ErrorObservation(
                'Browser functionality is not supported on Windows.'
            )
        await self._ensure_browser_ready()
        return await browse(action, self.browser, self.initial_cwd)

    def close(self):
        self.memory_monitor.stop_monitoring()
        if self.bash_session is not None:
            self.bash_session.close()
        if self.browser is not None:
            self.browser.close()


if __name__ == '__main__':
    logger.warning('Starting Action Execution Server')

    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int, help='Port to listen on')
    parser.add_argument('--working-dir', type=str, help='Working directory')
    parser.add_argument('--plugins', type=str, help='Plugins to initialize', nargs='+')
    parser.add_argument(
        '--username', type=str, help='User to run as', default='openhands'
    )
    parser.add_argument('--user-id', type=int, help='User ID to run as', default=1000)
    parser.add_argument(
        '--browsergym-eval-env',
        type=str,
        help='BrowserGym environment used for browser evaluation',
        default=None,
    )

    # example: python client.py 8000 --working-dir /workspace --plugins JupyterRequirement
    args = parser.parse_args()

    # Start the file viewer server in a separate thread
    logger.info('Starting file viewer server')
    _file_viewer_port_env = os.environ.get('FILE_VIEWER_PORT')
    if _file_viewer_port_env:
        _file_viewer_port = int(_file_viewer_port_env)
    else:
        raise RuntimeError('FILE_VIEWER_PORT environment variable is not set')
    server_url, _ = start_file_viewer_server(port=_file_viewer_port)
    logger.info(f'File viewer server started at {server_url}')

    plugins_to_load: list[Plugin] = []
    if args.plugins:
        for plugin in args.plugins:
            if plugin not in ALL_PLUGINS:
                raise ValueError(f'Plugin {plugin} not found')
            plugins_to_load.append(ALL_PLUGINS[plugin]())  # type: ignore

    client: ActionExecutor | None = None
    mcp_router: MCPRouter | None = None
    mcp_http_server: Server | None = None
    # Use a session-scoped MCP router profile file to avoid cross-runtime conflicts
    DEFAULT_MCP_ROUTER_PROFILE_PATH = os.path.join(
        os.path.dirname(__file__), 'mcp', 'config.json'
    )
    SESSION_ID_FOR_MCP = os.environ.get('OPENHANDS_SESSION_ID', 'default')
    MCP_ROUTER_PROFILE_PATH = os.path.join(
        '/tmp', f'openhands_mcp_config_{SESSION_ID_FOR_MCP}.json'
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global client, mcp_router, mcp_http_server
        logger.info('Initializing ActionExecutor...')
        client = ActionExecutor(
            plugins_to_load,
            work_dir=args.working_dir,
            username=args.username,
            user_id=args.user_id,
            browsergym_eval_env=args.browsergym_eval_env,
        )
        await client.ainit()
        logger.info('ActionExecutor initialized.')

        # Check if we're on Windows
        is_windows = sys.platform == 'win32'

        # Initialize MCP Router and start separate HTTP server (skip on Windows)
        if is_windows:
            logger.info('Skipping MCP Router initialization on Windows')
            mcp_router = None
            mcp_http_server = None
        else:
            logger.info('Initializing MCP Router...')
            # Ensure per-session MCP router profile exists; initialize from default if needed
            if not os.path.exists(MCP_ROUTER_PROFILE_PATH):
                try:
                    shutil.copy(
                        DEFAULT_MCP_ROUTER_PROFILE_PATH, MCP_ROUTER_PROFILE_PATH
                    )
                    logger.info(
                        f'Initialized MCP router profile for session at {MCP_ROUTER_PROFILE_PATH}'
                    )
                except Exception as e:
                    logger.error(
                        f'Failed to initialize MCP router profile: {e}', exc_info=True
                    )
                    raise
            mcp_router = MCPRouter(
                profile_path=MCP_ROUTER_PROFILE_PATH,
                router_config=RouterConfig(
                    api_key=SESSION_API_KEY,
                    auth_enabled=bool(SESSION_API_KEY),
                ),
            )
            # Ensure the router is fully initialized so aggregated_server has
            # the necessary initialization options in older mcpm versions.
            # Newer mcpm versions may not require or expose initialization_options,
            # but initialize_router() remains safe and idempotent.
            await mcp_router.initialize_router()
            allowed_origins = ['*']
            # Build SSE Starlette app manually to ensure proper ASGI callables
            api_key = (
                None
                if not mcp_router.router_config.auth_enabled
                else mcp_router.router_config.api_key
            )
            sse_transport = RouterSseTransport('/messages/', api_key=api_key)

            async def sse_asgi(scope, receive, send):
                try:
                    async with sse_transport.connect_sse(
                        scope,
                        receive,
                        send,
                    ) as (read_stream, write_stream):
                        # Wrapper try for generic exceptions only (no except* here)
                        try:
                            # Dedicated inner try that only uses except* for disconnection cases
                            try:
                                server = mcp_router.aggregated_server
                                # Backward/forward compatibility across mcpm versions:
                                # - Older versions require passing initialization_options
                                # - Newer versions may not expose it and accept (read, write)
                                try:
                                    init_opts = server.initialization_options  # type: ignore[attr-defined]
                                    await server.run(
                                        read_stream,
                                        write_stream,
                                        init_opts,
                                    )
                                except AttributeError:
                                    await server.run(
                                        read_stream,
                                        write_stream,
                                    )
                            except* (ClosedResourceError, EndOfStream):
                                # Client disconnected while server was running; benign
                                logger.debug(
                                    'SSE server run ended due to disconnect (ClosedResourceError/EndOfStream).'
                                )
                        except Exception as e:
                            # Log unexpected exceptions to aid debugging but prevent crashing the app
                            logger.error(
                                f'SSE server encountered an error: {e}', exc_info=True
                            )
                except* (ClosedResourceError, EndOfStream):
                    # Client disconnected; safe to ignore
                    logger.debug(
                        'SSE connection closed by client or ended (ClosedResourceError/EndOfStream).'
                    )

            middleware = []
            if allowed_origins is not None:
                middleware.append(
                    Middleware(
                        CORSMiddleware,
                        allow_origins=allowed_origins,
                        allow_methods=['*'],
                        allow_headers=['*'],
                    )
                )

            mcp_http_app = Starlette(
                debug=False,
                middleware=middleware,
                routes=[
                    Mount('/sse', app=sse_asgi),
                    Mount('/messages/', app=sse_transport.handle_post_message),
                ],
            )

            # MCP HTTP server configuration
            LOOPBACK_IP = os.environ.get('LOOPBACK_IP', '127.0.0.1')
            MCP_HTTP_PORT = int(os.environ.get('MCP_HTTP_PORT', '8080'))

            # Start MCP HTTP server in background
            mcp_http_config = Config(
                app=mcp_http_app,
                host=LOOPBACK_IP,
                port=MCP_HTTP_PORT,
                log_level='error',
            )
            mcp_http_server = Server(mcp_http_config)

            # Start the MCP HTTP server in a background task
            async def start_mcp_http_server():
                try:
                    if mcp_http_server is not None:
                        logger.info('MCP HTTP server task starting...')
                        await mcp_http_server.serve()
                except Exception as e:
                    logger.error(f'MCP HTTP server failed to start: {e}', exc_info=True)

            # Create task but don't await it - let it run in background
            mcp_http_task = asyncio.create_task(start_mcp_http_server())
            logger.info(
                f'Started MCP HTTP server task at http://{LOOPBACK_IP}:{MCP_HTTP_PORT}'
            )

            # Give the server a moment to start up
            await asyncio.sleep(0.1)
            logger.info('Lifespan initialization complete, yielding control...')

        yield

        # Clean up & release the resources
        logger.info('Shutting down MCP HTTP server...')
        mcp_http_task.cancel()
        if mcp_http_server:
            try:
                mcp_http_server.should_exit = True
                logger.info('MCP HTTP server shutdown successfully.')
            except Exception as e:
                logger.error(f'Error shutting down MCP HTTP server: {e}', exc_info=True)
        else:
            logger.info('MCP HTTP server instance not found for shutdown.')

        logger.info('Shutting down MCP Router...')
        if mcp_router:
            try:
                await mcp_router.shutdown()
                logger.info('MCP Router shutdown successfully.')
            except Exception as e:
                logger.error(f'Error shutting down MCP Router: {e}', exc_info=True)
        else:
            logger.info('MCP Router instance not found for shutdown.')

        logger.info('Closing ActionExecutor...')
        if client:
            try:
                client.close()
                logger.info('ActionExecutor closed successfully.')
            except Exception as e:
                logger.error(f'Error closing ActionExecutor: {e}', exc_info=True)
        else:
            logger.info('ActionExecutor instance not found for closing.')
        logger.info('Shutdown complete.')

    app = FastAPI(lifespan=lifespan)

    # TODO below 3 exception handlers were recommended by Sonnet.
    # Are these something we should keep?
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception('Unhandled exception occurred:')
        return JSONResponse(
            status_code=500,
            content={'detail': 'An unexpected error occurred. Please try again later.'},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.error(f'HTTP exception occurred: {exc.detail}')
        return JSONResponse(status_code=exc.status_code, content={'detail': exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        logger.error(f'Validation error occurred: {exc}')
        return JSONResponse(
            status_code=422,
            content={
                'detail': 'Invalid request parameters',
                'errors': str(exc.errors()),
            },
        )

    @app.middleware('http')
    async def authenticate_requests(request: Request, call_next):
        if request.url.path != '/alive' and request.url.path != '/server_info':
            try:
                verify_api_key(request.headers.get('X-Session-API-Key'))
            except HTTPException as e:
                return JSONResponse(
                    status_code=e.status_code, content={'detail': e.detail}
                )
        response = await call_next(request)
        return response

    @app.get('/server_info')
    async def get_server_info():
        assert client is not None
        current_time = time.time()
        uptime = current_time - client.start_time
        idle_time = current_time - client.last_execution_time

        response = {
            'uptime': uptime,
            'idle_time': idle_time,
            'resources': get_system_stats(),
        }
        logger.info('Server info endpoint response: %s', response)
        return response

    @app.post('/execute_action')
    async def execute_action(action_request: ActionRequest):
        assert client is not None
        try:
            action = event_from_dict(action_request.action)
            if not isinstance(action, Action):
                raise HTTPException(status_code=400, detail='Invalid action type')
            client.last_execution_time = time.time()
            observation = await client.run_action(action)
            return event_to_dict(observation)
        except Exception as e:
            logger.error(f'Error while running /execute_action: {str(e)}')
            raise HTTPException(
                status_code=500,
                detail=traceback.format_exc(),
            )

    @app.post('/update_mcp_server')
    async def update_mcp_server(request: Request):
        # Check if we're on Windows
        is_windows = sys.platform == 'win32'

        if is_windows:
            # On Windows, just return a success response without doing anything
            logger.info(
                'MCP server update request received on Windows - skipping as MCP is disabled'
            )
            return JSONResponse(
                status_code=200,
                content={
                    'detail': 'MCP server update skipped (MCP is disabled on Windows)',
                    'router_error_log': '',
                },
            )

        # Non-Windows implementation
        assert mcp_router is not None
        assert os.path.exists(MCP_ROUTER_PROFILE_PATH)

        # Use synchronous file operations outside of async function
        def read_profile():
            with open(MCP_ROUTER_PROFILE_PATH, 'r') as f:
                return json.load(f)

        current_profile = read_profile()
        assert 'default' in current_profile
        assert isinstance(current_profile['default'], list)

        # Get the request body
        mcp_tools_to_sync = await request.json()
        if not isinstance(mcp_tools_to_sync, list):
            raise HTTPException(
                status_code=400, detail='Request must be a list of MCP tools to sync'
            )

        logger.info(
            f'Updating MCP server to: {json.dumps(mcp_tools_to_sync, indent=2)}.\nPrevious profile: {json.dumps(current_profile, indent=2)}'
        )
        current_profile['default'] = mcp_tools_to_sync

        # Use synchronous file operations outside of async function
        def write_profile(profile):
            with open(MCP_ROUTER_PROFILE_PATH, 'w') as f:
                json.dump(profile, f)

        write_profile(current_profile)

        # Manually reload the profile and update the servers
        mcp_router.profile_manager.reload()
        servers_wait_for_update = mcp_router.get_unique_servers()
        async with capture_logs('mcpm.router.router') as log_capture:
            await mcp_router.update_servers(servers_wait_for_update)
        router_error_log = log_capture.getvalue()

        logger.info(
            f'MCP router updated successfully with unique servers: {servers_wait_for_update}'
        )
        if router_error_log:
            logger.warning(f'Some MCP servers failed to be added: {router_error_log}')

        return JSONResponse(
            status_code=200,
            content={
                'detail': 'MCP server updated successfully',
                'router_error_log': router_error_log,
            },
        )

    @app.post('/upload_file')
    async def upload_file(
        file: UploadFile, destination: str = '/', recursive: bool = False
    ):
        assert client is not None

        try:
            # Ensure the destination directory exists
            if not os.path.isabs(destination):
                raise HTTPException(
                    status_code=400, detail='Destination must be an absolute path'
                )

            full_dest_path = destination
            if not os.path.exists(full_dest_path):
                os.makedirs(full_dest_path, exist_ok=True)

            if recursive or file.filename.endswith('.zip'):
                # For recursive uploads, we expect a zip file
                if not file.filename.endswith('.zip'):
                    raise HTTPException(
                        status_code=400, detail='Recursive uploads must be zip files'
                    )

                zip_path = os.path.join(full_dest_path, file.filename)
                with open(zip_path, 'wb') as buffer:  # noqa: ASYNC101
                    shutil.copyfileobj(file.file, buffer)

                # Extract the zip file
                shutil.unpack_archive(zip_path, full_dest_path)
                os.remove(zip_path)  # Remove the zip file after extraction

                logger.debug(
                    f'Uploaded file {file.filename} and extracted to {destination}'
                )
            else:
                # For single file uploads
                file_path = os.path.join(full_dest_path, file.filename)
                with open(file_path, 'wb') as buffer:  # noqa: ASYNC101
                    shutil.copyfileobj(file.file, buffer)
                logger.debug(f'Uploaded file {file.filename} to {destination}')

            return JSONResponse(
                content={
                    'filename': file.filename,
                    'destination': destination,
                    'recursive': recursive,
                },
                status_code=200,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/download_files')
    def download_file(path: str):
        logger.debug('Downloading files')
        try:
            if not os.path.isabs(path):
                raise HTTPException(
                    status_code=400, detail='Path must be an absolute path'
                )

            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail='File not found')

            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                with ZipFile(temp_zip, 'w') as zipf:
                    for root, _, files in os.walk(path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zipf.write(
                                file_path, arcname=os.path.relpath(file_path, path)
                            )
                return FileResponse(
                    path=temp_zip.name,
                    media_type='application/zip',
                    filename=f'{os.path.basename(path)}.zip',
                    background=BackgroundTask(lambda: os.unlink(temp_zip.name)),
                )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/alive')
    async def alive():
        if client is None or not client.initialized:
            return {'status': 'not initialized'}
        return {'status': 'ok'}

    # ================================
    # VSCode-specific operations
    # ================================

    @app.get('/vscode/connection_token')
    async def get_vscode_connection_token():
        assert client is not None
        if 'vscode' in client.plugins:
            plugin: VSCodePlugin = client.plugins['vscode']  # type: ignore
            return {'token': plugin.vscode_connection_token}
        else:
            return {'token': None}

    # ================================
    # File-specific operations for UI
    # ================================

    @app.post('/list_files')
    async def list_files(request: Request):
        """List files in the specified path.

        This function retrieves a list of files from the agent's runtime file store,
        excluding certain system and hidden files/directories.

        To list files:
        ```sh
        curl -X POST -d '{"path": "/"}' http://localhost:3000/list_files
        ```

        Args:
            request (Request): The incoming request object.
            path (str, optional): The path to list files from. Defaults to '/'.

        Returns:
            list: A list of file names in the specified path.

        Raises:
            HTTPException: If there's an error listing the files.
        """
        assert client is not None

        # get request as dict
        request_dict = await request.json()
        path = request_dict.get('path', None)

        # Get the full path of the requested directory
        if path is None:
            full_path = client.initial_cwd
        elif os.path.isabs(path):
            full_path = path
        else:
            full_path = os.path.join(client.initial_cwd, path)

        if not os.path.exists(full_path):
            # if user just removed a folder, prevent server error 500 in UI
            return JSONResponse(content=[])

        try:
            # Check if the directory exists
            if not os.path.exists(full_path) or not os.path.isdir(full_path):
                return JSONResponse(content=[])

            entries = os.listdir(full_path)

            # Separate directories and files
            directories = []
            files = []
            for entry in entries:
                # Remove leading slash and any parent directory components
                entry_relative = entry.lstrip('/').split('/')[-1]

                # Construct the full path by joining the base path with the relative entry path
                full_entry_path = os.path.join(full_path, entry_relative)
                if os.path.exists(full_entry_path):
                    is_dir = os.path.isdir(full_entry_path)
                    if is_dir:
                        # add trailing slash to directories
                        # required by FE to differentiate directories and files
                        entry = entry.rstrip('/') + '/'
                        directories.append(entry)
                    else:
                        files.append(entry)

            # Sort directories and files separately
            directories.sort(key=lambda s: s.lower())
            files.sort(key=lambda s: s.lower())

            # Combine sorted directories and files
            sorted_entries = directories + files
            return JSONResponse(content=sorted_entries)

        except Exception as e:
            logger.error(f'Error listing files: {e}')
            return JSONResponse(content=[])

    logger.debug(f'Starting action execution API on port {args.port}')

    # Create UDS socket path using session_id
    session_id = os.environ.get('OPENHANDS_SESSION_ID', 'default')
    socket_dir = '/tmp/runtime'
    os.makedirs(socket_dir, exist_ok=True)
    socket_path = f'{socket_dir}/{session_id}.sock'

    # Remove existing socket file if it exists
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    logger.info(f'Starting action execution API on UDS socket: {socket_path}')

    # Run with UDS socket instead of TCP
    config = Config(app=app, uds=socket_path, log_level='error')
    server = Server(config)
    server.run()
