import logging
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from openhands.sdk.workspace.models import CommandResult, FileOperationResult


_logger = logging.getLogger(__name__)


class RemoteWorkspaceMixin(BaseModel):
    """Mixin providing remote workspace operations.
    This allows the same code to be used for sync and async."""

    host: str = Field(description="The remote host URL for the workspace.")
    api_key: str | None = Field(
        default=None, description="API key for authenticating with the remote host."
    )
    working_dir: str = Field(
        description="The working directory for agent operations and tool execution."
    )
    init_directories: list[str] = Field(
        default_factory=list,
        description="Directories to create after workspace init (mkdir -p).",
    )
    init_commands: list[str] = Field(
        default_factory=list,
        description="Commands to run after workspace init.",
    )

    def model_post_init(self, context: Any) -> None:
        # Set up remote host
        self.host = self.host.rstrip("/")
        return super().model_post_init(context)

    def _run_init_hooks(self) -> None:
        """Run init directories and commands after workspace is ready.

        Creates directories specified in init_directories (using mkdir -p),
        then runs commands specified in init_commands sequentially.

        Raises:
            RuntimeError: If directory creation or command execution fails.
        """
        if not self.init_directories and not self.init_commands:
            return

        _logger.info("Running workspace initialization hooks...")

        for directory in self.init_directories:
            _logger.debug(f"Creating directory: {directory}")
            result = self.execute_command(f"mkdir -p {directory}", timeout=30.0)
            if result.exit_code != 0:
                raise RuntimeError(f"Failed to create '{directory}': {result.stderr}")

        if self.init_directories:
            _logger.info(f"Created {len(self.init_directories)} init directories")

        for command in self.init_commands:
            _logger.debug(f"Running init command: {command}")
            result = self.execute_command(command, timeout=300.0)
            if result.exit_code != 0:
                raise RuntimeError(f"Init command failed: {command}\n{result.stderr}")

        if self.init_commands:
            _logger.info(f"Completed {len(self.init_commands)} init commands")

    @property
    def _headers(self):
        headers = {}
        if self.api_key:
            headers["X-Session-API-Key"] = self.api_key
        return headers

    def _execute_command_generator(
        self,
        command: str,
        cwd: str | Path | None,
        timeout: float,
    ) -> Generator[dict[str, Any], httpx.Response, CommandResult]:
        """Execute a bash command on the remote system.

        This method starts a bash command via the remote agent server API,
        then polls for the output until the command completes.

        Args:
            command: The bash command to execute
            cwd: Working directory (optional)
            timeout: Timeout in seconds

        Returns:
            CommandResult: Result with stdout, stderr, exit_code, and other metadata
        """
        _logger.debug(f"Executing remote command: {command}")

        # Step 1: Start the bash command
        payload = {
            "command": command,
            "timeout": int(timeout),
        }
        if cwd is not None:
            payload["cwd"] = str(cwd)

        try:
            # Start the command
            response: httpx.Response = yield {
                "method": "POST",
                "url": f"{self.host}/api/bash/start_bash_command",
                "json": payload,
                "headers": self._headers,
                "timeout": timeout + 5.0,  # Add buffer to HTTP timeout
            }
            response.raise_for_status()
            bash_command = response.json()
            command_id = bash_command["id"]

            _logger.debug(f"Started command with ID: {command_id}")

            # Step 2: Poll for output until command completes
            start_time = time.time()
            stdout_parts = []
            stderr_parts = []
            exit_code = None

            while time.time() - start_time < timeout:
                # Search for all events
                response = yield {
                    "method": "GET",
                    "url": f"{self.host}/api/bash/bash_events/search",
                    "params": {
                        "command_id__eq": command_id,
                        "sort_order": "TIMESTAMP",
                        "limit": 100,
                    },
                    "headers": self._headers,
                    "timeout": timeout,
                }
                response.raise_for_status()
                search_result = response.json()

                # Filter for BashOutput events for this command
                for event in search_result.get("items", []):
                    if event.get("kind") == "BashOutput":
                        if event.get("stdout"):
                            stdout_parts.append(event["stdout"])
                        if event.get("stderr"):
                            stderr_parts.append(event["stderr"])
                        if event.get("exit_code") is not None:
                            exit_code = event["exit_code"]

                # If we have an exit code, the command is complete
                if exit_code is not None:
                    break

                # Wait a bit before polling again
                time.sleep(0.1)

            # If we timed out waiting for completion
            if exit_code is None:
                _logger.warning(f"Command timed out after {timeout} seconds: {command}")
                exit_code = -1
                stderr_parts.append(f"Command timed out after {timeout} seconds")

            # Combine all output parts
            stdout = "".join(stdout_parts)
            stderr = "".join(stderr_parts)

            return CommandResult(
                command=command,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timeout_occurred=exit_code == -1 and "timed out" in stderr,
            )

        except Exception as e:
            _logger.error(f"Remote command execution failed: {e}")
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"Remote execution error: {str(e)}",
                timeout_occurred=False,
            )

    def _file_upload_generator(
        self,
        source_path: str | Path,
        destination_path: str | Path,
    ) -> Generator[dict[str, Any], httpx.Response, FileOperationResult]:
        """Upload a file to the remote system.

        Reads the local file and sends it to the remote system via HTTP API.

        Args:
            source_path: Path to the local source file
            destination_path: Path where the file should be uploaded on remote system

        Returns:
            FileOperationResult: Result with success status and metadata
        """
        source = Path(source_path)
        destination = Path(destination_path)

        _logger.debug(f"Remote file upload: {source} -> {destination}")

        try:
            # Read the file content
            with open(source, "rb") as f:
                file_content = f.read()

            # Prepare the upload
            files = {"file": (source.name, file_content)}
            data = {"destination_path": str(destination)}

            # Make HTTP call
            response: httpx.Response = yield {
                "method": "POST",
                "url": f"{self.host}/api/file/upload/{destination}",
                "files": files,
                "data": data,
                "headers": self._headers,
                "timeout": 60.0,
            }
            response.raise_for_status()
            result_data = response.json()

            # Convert the API response to our model
            return FileOperationResult(
                success=result_data.get("success", True),
                source_path=str(source),
                destination_path=str(destination),
                file_size=result_data.get("file_size"),
                error=result_data.get("error"),
            )

        except Exception as e:
            _logger.error(f"Remote file upload failed: {e}")
            return FileOperationResult(
                success=False,
                source_path=str(source),
                destination_path=str(destination),
                error=str(e),
            )

    def _file_download_generator(
        self,
        source_path: str | Path,
        destination_path: str | Path,
    ) -> Generator[dict[str, Any], httpx.Response, FileOperationResult]:
        """Download a file from the remote system.

        Requests the file from the remote system via HTTP API and saves it locally.

        Args:
            source_path: Path to the source file on remote system
            destination_path: Path where the file should be saved locally

        Returns:
            FileOperationResult: Result with success status and metadata
        """
        source = Path(source_path)
        destination = Path(destination_path)

        _logger.debug(f"Remote file download: {source} -> {destination}")

        try:
            # Construct URL with path parameter (not query parameter)
            # Double slash ensures FastAPI extracts path with leading slash
            # for absolute path validation
            source_str = str(source)
            url = f"/api/file/download//{source_str.lstrip('/')}"

            # Make HTTP call
            response = yield {
                "method": "GET",
                "url": url,
                "headers": self._headers,
                "timeout": 60.0,
            }
            response.raise_for_status()

            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Write the file content
            with open(destination, "wb") as f:
                f.write(response.content)

            return FileOperationResult(
                success=True,
                source_path=str(source),
                destination_path=str(destination),
                file_size=len(response.content),
            )

        except Exception as e:
            _logger.error(f"Remote file download failed: {e}")
            return FileOperationResult(
                success=False,
                source_path=str(source),
                destination_path=str(destination),
                error=str(e),
            )
