"""Service which preloads chromium."""

from __future__ import annotations

import sys

from openhands.agent_server.config import get_default_config
from openhands.sdk.logger import get_logger


_logger = get_logger(__name__)


class ToolPreloadService:
    """Service which preloads tools / chromium reducing time to
    start first conversation"""

    running: bool = False

    async def start(self) -> bool:
        """Preload tools"""

        # Skip if already running
        if self.running:
            return True

        self.running = True
        try:
            if sys.platform == "win32":
                from openhands.tools.browser_use.impl_windows import (
                    WindowsBrowserToolExecutor as BrowserToolExecutor,
                )
            else:
                from openhands.tools.browser_use.impl import BrowserToolExecutor

            # Creating an instance here to preload chromium
            BrowserToolExecutor()

            _logger.debug(f"Loaded {BrowserToolExecutor}")
            return True
        except Exception:
            _logger.exception("Error preloading chromium")
            return False

    async def stop(self) -> None:
        """Stop the tool preload process."""
        self.running = False

    def is_running(self) -> bool:
        """Check if tool preload is running."""
        return self.running


_tool_preload_service: ToolPreloadService | None = None


def get_tool_preload_service() -> ToolPreloadService | None:
    """Get the tool preload service instance if preload is enabled."""
    global _tool_preload_service
    config = get_default_config()

    if not config.preload_tools:
        _logger.info("Tool preload is disabled in configuration")
        return None

    if _tool_preload_service is None:
        _tool_preload_service = ToolPreloadService()
    return _tool_preload_service
