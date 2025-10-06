import asyncio
import os
import re
import sys
import time
import traceback
import io
import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64

# Import IPython components for magic command support
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action, IPythonRunCellAction
from openhands.events.observation import IPythonRunCellObservation
from openhands.runtime.plugins.requirement import Plugin, PluginRequirement
from openhands.utils.shutdown_listener import should_continue


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


@dataclass
class DirectJupyterRequirement(PluginRequirement):
    name: str = 'direct_jupyter'


class DirectIPythonExecutor:
    """Direct IPython executor with magic command support but no network communication."""

    def __init__(self, kernel_id: str) -> None:
        self.kernel_id = kernel_id
        self.shell: InteractiveShell | None = None
        self.initialized = False
        logger.info(f'Direct IPython executor created for {kernel_id}')

    async def initialize(self) -> None:
        """Initialize the IPython execution environment."""
        try:
            # Create IPython shell instance
            # Use get_ipython() first, and if it doesn't exist, create one
            self.shell = InteractiveShell.instance()
            if self.shell is None:
                # Create a new InteractiveShell instance
                self.shell = TerminalInteractiveShell.instance()

            # Configure the shell
            self.shell.colors = 'NoColor'  # Disable colored output
            self.shell.xmode = 'Plain'     # Simple exception format

            # Disable matplotlib interactive mode
            self.shell.run_cell("import matplotlib; matplotlib.use('Agg')")
            self.shell.run_cell("import matplotlib.pyplot as plt; plt.ioff()")

            # Pre-load common libraries
            common_imports = [
                "import sys",
                "import os",
                "import numpy as np",
                "import pandas as pd",
                "import matplotlib.pyplot as plt"
            ]

            for import_stmt in common_imports:
                try:
                    self.shell.run_cell(import_stmt)
                    logger.debug(f"Loaded: {import_stmt}")
                except Exception as e:
                    logger.warning(f"Failed to load {import_stmt}: {e}")

            self.initialized = True
            logger.info('Direct IPython executor initialized successfully')

        except Exception as e:
            logger.error(f'Failed to initialize Direct IPython executor: {e}')
            raise

    async def execute(self, code: str, timeout: int = 120) -> Dict[str, Any]:
        """Execute IPython code with magic command support and return structured output."""
        if not self.initialized or self.shell is None:
            raise RuntimeError('Executor not initialized')

        try:
            # Clean up any existing matplotlib figures
            plt.close('all')

            # Handle empty code
            code = code.strip()
            if not code:
                return {'text': '', 'images': []}

            # Use IPython's capture_output to capture stdout/stderr
            with capture_output() as captured:
                # Execute the code using IPython's run_cell
                # This handles both regular Python code and magic commands
                result = self.shell.run_cell(code)

            # Get captured output text
            output_text = ""
            if captured.stdout:
                output_text += captured.stdout
            if captured.stderr:
                if output_text:
                    output_text += "\n"
                output_text += captured.stderr

            # Handle execution result and errors
            if result.error_before_exec:
                output_text += f"\nError before execution: {result.error_before_exec}"
            elif result.error_in_exec:
                output_text += f"\nError during execution: {result.error_in_exec}"
            elif result.result is not None:
                # If there's a result and no other output, show the result
                if not output_text.strip():
                    output_text = str(result.result)

            # Capture any matplotlib figures
            image_outputs = []
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
                img_buffer.seek(0)
                img_data = base64.b64encode(img_buffer.read()).decode('utf-8')
                image_url = f'data:image/png;base64,{img_data}'
                image_outputs.append(image_url)
                img_buffer.close()

            # Close figures to free memory
            plt.close('all')

            # Clean ANSI escape sequences
            output_text = strip_ansi(output_text)

            if not output_text and not image_outputs:
                output_text = '[Code executed successfully with no output]'

            return {'text': output_text, 'images': image_outputs}

        except Exception as e:
            # Capture the full traceback
            error_traceback = traceback.format_exc()
            logger.error(f'Error executing code: {error_traceback}')
            return {'text': f'Error: {str(e)}\n{error_traceback}', 'images': []}

    async def shutdown_async(self) -> None:
        """Clean shutdown of executor resources."""
        try:
            plt.close('all')  # Close all matplotlib figures
            if self.shell:
                # Clear the namespace
                self.shell.reset(new_session=False)
            logger.info('Direct IPython executor shut down successfully')
        except Exception as e:
            logger.error(f'Error shutting down Direct IPython executor: {e}')


class DirectJupyterPlugin(Plugin):
    """Direct Jupyter plugin with IPython magic command support but no network overhead."""

    name: str = 'direct_jupyter'
    kernel_id: str
    python_interpreter_path: str

    async def initialize(
        self, username: str, kernel_id: str = 'openhands-default'
    ) -> None:
        """Initialize the direct jupyter plugin."""
        self.kernel_id = kernel_id
        is_local_runtime = os.environ.get('LOCAL_RUNTIME_MODE') == '1'

        # Set up environment similar to original plugin
        if not is_local_runtime:
            # Non-LocalRuntime - set up Python path and environment
            os.environ.setdefault('POETRY_VIRTUALENVS_PATH', '/openhands/poetry')
            python_path = os.environ.get('PYTHONPATH', '')
            if '/openhands/code' not in python_path:
                os.environ['PYTHONPATH'] = f'/openhands/code:{python_path}'
            os.environ.setdefault('MAMBA_ROOT_PREFIX', '/openhands/micromamba')
        else:
            # LocalRuntime
            code_repo_path = os.environ.get('OPENHANDS_REPO_PATH')
            if not code_repo_path:
                raise ValueError(
                    'OPENHANDS_REPO_PATH environment variable is not set. '
                    'This is required for the direct jupyter plugin to work with LocalRuntime.'
                )
            # Change to the code repo directory
            os.chdir(code_repo_path)

        logger.debug('Direct Jupyter plugin initialization started')

        # Initialize the direct IPython executor (no network communication)
        self.executor = DirectIPythonExecutor(self.kernel_id)
        await self.executor.initialize()

        # Get Python interpreter path
        _obs = await self.run(
            IPythonRunCellAction(code='import sys; print(sys.executable)')
        )
        self.python_interpreter_path = _obs.content.strip()

        logger.debug(f'Direct Jupyter plugin initialized with Python: {self.python_interpreter_path}')

    async def _run(self, action: Action) -> IPythonRunCellObservation:
        """Internal method to run a code cell in the direct IPython executor."""
        if not isinstance(action, IPythonRunCellAction):
            raise ValueError(
                f'Direct Jupyter plugin only supports IPythonRunCellAction, but got {action}'
            )

        if not hasattr(self, 'executor') or not self.executor.initialized:
            raise RuntimeError('Direct IPython executor not initialized')

        # Execute the code and get structured output
        timeout = action.timeout if action.timeout is not None else 120
        if isinstance(timeout, float):
            timeout = int(timeout)

        output = await self.executor.execute(action.code, timeout=timeout)

        # Extract text content and image URLs from the structured output
        text_content = output.get('text', '')
        image_urls = output.get('images', [])

        return IPythonRunCellObservation(
            content=text_content,
            code=action.code,
            image_urls=image_urls if image_urls else None,
        )

    async def run(self, action: Action) -> IPythonRunCellObservation:
        """Main interface for running actions."""
        obs = await self._run(action)
        return obs

    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        if hasattr(self, 'executor'):
            await self.executor.shutdown_async()
