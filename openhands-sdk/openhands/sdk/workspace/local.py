import shutil
from pathlib import Path
from typing import Any

from openhands.sdk.logger import get_logger
from openhands.sdk.utils.command import execute_command
from openhands.sdk.workspace.base import BaseWorkspace
from openhands.sdk.workspace.models import CommandResult, FileOperationResult


logger = get_logger(__name__)


class LocalWorkspace(BaseWorkspace):
    """Local workspace implementation that operates on the host filesystem.

    LocalWorkspace provides direct access to the local filesystem and command execution
    environment. It's suitable for development and testing scenarios where the agent
    should operate directly on the host system.

    When working_dir is None, the workspace can only be used with MCP tools
    (no filesystem-based tools like Terminal, FileEditor, Glob, Grep).

    Example:
        >>> # Full workspace with filesystem access
        >>> workspace = LocalWorkspace(working_dir="/path/to/project")
        >>> with workspace:
        ...     result = workspace.execute_command("ls -la")
        ...     content = workspace.read_file("README.md")

        >>> # MCP-only workspace without filesystem access
        >>> workspace = LocalWorkspace()  # No working_dir
    """

    def __init__(self, *, working_dir: str | Path | None = None, **kwargs: Any):
        # Accept Path in signature for ergonomics and type checkers,
        # but normalize to str for the underlying model field.
        super().__init__(
            working_dir=str(working_dir) if working_dir is not None else None, **kwargs
        )

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        """Execute a bash command locally.

        Uses the shared shell execution utility to run commands with proper
        timeout handling, output streaming, and error management.

        Args:
            command: The bash command to execute
            cwd: Working directory (optional)
            timeout: Timeout in seconds

        Returns:
            CommandResult: Result with stdout, stderr, exit_code, command, and
                timeout_occurred

        Raises:
            ValueError: If working_dir is None and cwd is not specified
        """
        effective_cwd = cwd if cwd is not None else self.working_dir
        if effective_cwd is None:
            raise ValueError(
                "Cannot execute command: working_dir is not set. "
                "LocalWorkspace was created without a working_dir, which means "
                "filesystem-based operations are disabled. Use MCP tools instead, "
                "or create LocalWorkspace with a working_dir."
            )
        logger.debug(f"Executing local bash command: {command} in {effective_cwd}")
        result = execute_command(
            command,
            cwd=str(effective_cwd),
            timeout=timeout,
            print_output=True,
        )
        return CommandResult(
            command=command,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timeout_occurred=result.returncode == -1,
        )

    def file_upload(
        self,
        source_path: str | Path,
        destination_path: str | Path,
    ) -> FileOperationResult:
        """Upload (copy) a file locally.

        For local systems, file upload is implemented as a file copy operation
        using shutil.copy2 to preserve metadata.

        Args:
            source_path: Path to the source file
            destination_path: Path where the file should be copied

        Returns:
            FileOperationResult: Result with success status and file information
        """
        source = Path(source_path)
        destination = Path(destination_path)

        logger.debug(f"Local file upload: {source} -> {destination}")

        try:
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file with metadata preservation
            shutil.copy2(source, destination)

            return FileOperationResult(
                success=True,
                source_path=str(source),
                destination_path=str(destination),
                file_size=destination.stat().st_size,
            )

        except Exception as e:
            logger.error(f"Local file upload failed: {e}")
            return FileOperationResult(
                success=False,
                source_path=str(source),
                destination_path=str(destination),
                error=str(e),
            )

    def file_download(
        self,
        source_path: str | Path,
        destination_path: str | Path,
    ) -> FileOperationResult:
        """Download (copy) a file locally.

        For local systems, file download is implemented as a file copy operation
        using shutil.copy2 to preserve metadata.

        Args:
            source_path: Path to the source file
            destination_path: Path where the file should be copied

        Returns:
            FileOperationResult: Result with success status and file information
        """
        source = Path(source_path)
        destination = Path(destination_path)

        logger.debug(f"Local file download: {source} -> {destination}")

        try:
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file with metadata preservation
            shutil.copy2(source, destination)

            return FileOperationResult(
                success=True,
                source_path=str(source),
                destination_path=str(destination),
                file_size=destination.stat().st_size,
            )

        except Exception as e:
            logger.error(f"Local file download failed: {e}")
            return FileOperationResult(
                success=False,
                source_path=str(source),
                destination_path=str(destination),
                error=str(e),
            )

    def upload_directory(
        self,
        source_dir: str | Path,
        destination_path: str | Path,
    ) -> str:
        """Upload a directory and all its contents locally.

        For local workspaces, this uses shutil.copytree for better performance
        than the base implementation which uploads files one at a time.

        Args:
            source_dir: Local path to the directory to upload
            destination_path: Destination path

        Returns:
            The destination path where files were uploaded

        Raises:
            ValueError: If source_dir is not a directory
            RuntimeError: If copy operation fails
        """
        source = Path(source_dir).resolve()
        dest = Path(destination_path).resolve()

        if not source.is_dir():
            raise ValueError(f"Source must be a directory: {source}")

        logger.debug(f"Local directory upload: {source} -> {dest}")

        try:
            # Remove destination if it exists to ensure clean copy
            if dest.exists():
                shutil.rmtree(dest)

            shutil.copytree(source, dest)

            # Count files for logging
            file_count = sum(1 for _ in dest.rglob("*") if _.is_file())
            logger.debug(f"Copied {file_count} files to {dest}")

            return str(dest)

        except Exception as e:
            logger.error(f"Local directory upload failed: {e}")
            raise RuntimeError(f"Failed to copy directory: {e}") from e

    def download_directory(
        self,
        source_path: str | Path,
        destination_dir: str | Path,
    ) -> str:
        """Download a directory and all its contents locally.

        For local workspaces, this uses shutil.copytree for better performance
        than the base implementation which downloads files one at a time.

        Args:
            source_path: Path to the source directory
            destination_dir: Local path where the directory should be downloaded

        Returns:
            The local destination path where files were downloaded

        Raises:
            RuntimeError: If copy operation fails
        """
        source = Path(source_path).resolve()
        dest = Path(destination_dir).resolve()

        logger.debug(f"Local directory download: {source} -> {dest}")

        try:
            # Remove destination if it exists to ensure clean copy
            if dest.exists():
                shutil.rmtree(dest)

            shutil.copytree(source, dest)

            # Count files for logging
            file_count = sum(1 for _ in dest.rglob("*") if _.is_file())
            logger.debug(f"Copied {file_count} files to {dest}")

            return str(dest)

        except Exception as e:
            logger.error(f"Local directory download failed: {e}")
            raise RuntimeError(f"Failed to copy directory: {e}") from e

    def upload_skill(self, source_dir: str | Path, mount_path: str) -> str:
        """Register a skill directory for local access.

        Local workspaces access skill files directly from the host, so we keep
        the source path as the sandbox path and avoid copying.
        """
        source = Path(source_dir).resolve()
        if not source.is_dir():
            raise ValueError(f"Skill source must be a directory: {source}")

        logger.debug(
            "LocalWorkspace: using skill source directory without upload: %s",
            source,
        )
        return str(source)

    def pause(self) -> None:
        """Pause the workspace (no-op for local workspaces).

        Local workspaces have nothing to pause since they operate directly
        on the host filesystem.
        """
        logger.debug("pause() called on LocalWorkspace - nothing to do")

    def resume(self) -> None:
        """Resume the workspace (no-op for local workspaces).

        Local workspaces have nothing to resume since they operate directly
        on the host filesystem.
        """
        logger.debug("resume() called on LocalWorkspace - nothing to do")
