"""HTTP client abstraction for OSWorld VM communication.

This module provides a unified HTTP client interface that works with both:
- Singularity runtime (direct HTTP to localhost)
- NVCF runtime (authenticated HTTPS to NVIDIA cloud)

This module is placed in openhands.runtime.utils to avoid circular imports
with openhands.nvidia, which has side effects on import (handler registration).
"""

from typing import Protocol, Optional, Dict
import requests

from openhands.core.logger import openhands_logger as logger


class OSWorldHttpClient(Protocol):
    """Protocol for HTTP communication with OSWorld VM.
    
    This abstraction allows controllers and getters to use the same code
    regardless of whether they're talking to a local VM or NVCF.
    """
    
    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a GET request to the VM server."""
        ...
    
    def post(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a POST request to the VM server."""
        ...
    
    def get_cdp_url(self) -> str:
        """Get the Chrome DevTools Protocol URL for Playwright."""
        ...
    
    def get_cdp_headers(self) -> Optional[Dict[str, str]]:
        """Get headers needed for CDP connection (None for direct connection)."""
        ...
    
    def get_vlc_url(self) -> str:
        """Get the VLC web interface base URL."""
        ...

    def update_launch_command(self, command: str) -> str:
        """Update the launch command to use the HTTP client."""
        ...


class DirectHttpClient:
    """Direct HTTP client for Singularity/local runtime.
    
    Makes direct HTTP requests to the VM server running on localhost.
    No special authentication or URL rewriting needed.
    """
    
    def __init__(self, base_url: str, chromium_port: int, vlc_port: int):
        """Initialize the direct HTTP client.
        
        Args:
            base_url: Base URL for the VM server (e.g., http://127.0.0.1:5000)
            chromium_port: Port for Chrome DevTools Protocol
            vlc_port: Port for VLC web interface
        """
        self.base_url = base_url.rstrip('/')
        self.chromium_port = chromium_port
        self.vlc_port = vlc_port
    
    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a GET request to the VM server."""
        url = self.base_url + endpoint
        return requests.get(url, **kwargs)
    
    def post(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a POST request to the VM server."""
        url = self.base_url + endpoint
        return requests.post(url, **kwargs)
    
    def get_cdp_url(self) -> str:
        """Get the Chrome DevTools Protocol URL."""
        return f"http://127.0.0.1:{self.chromium_port}"
    
    def get_cdp_headers(self) -> Optional[Dict[str, str]]:
        """No special headers needed for direct connection."""
        return None
    
    def get_vlc_url(self) -> str:
        """Get the VLC web interface URL."""
        return f"http://127.0.0.1:{self.vlc_port}"

    def update_launch_command(self, command: str) -> str:
        """Update the launch command to use the HTTP client."""
        return command

class NVCFHttpClient:
    """NVCF HTTP client with authentication and URL rewriting.
    
    Makes authenticated HTTPS requests to NVIDIA Cloud Functions.
    Handles WebSocket URL rewriting for Chrome DevTools Protocol.
    """
    
    NVCF_API_BASE = "https://grpc.nvcf.nvidia.com/api"
    NVCF_CHROME_BASE = "https://grpc.nvcf.nvidia.com/chrome"
    NVCF_VLC_BASE = "https://grpc.nvcf.nvidia.com/vlc"
    
    def __init__(self, api_key: str, function_id: str):
        """Initialize the NVCF HTTP client.
        
        Args:
            api_key: NGC API key for authentication
            function_id: NVCF function ID
        """
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Function-ID": function_id,
        }
        self._cached_cdp_url: Optional[str] = None
    
    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """Make an authenticated GET request to NVCF."""
        url = self.NVCF_API_BASE + endpoint
        # Merge auth headers with any provided headers
        headers = {**self.headers, **kwargs.pop('headers', {})}
        return requests.get(url, headers=headers, **kwargs)
    
    def post(self, endpoint: str, **kwargs) -> requests.Response:
        """Make an authenticated POST request to NVCF."""
        url = self.NVCF_API_BASE + endpoint
        # Merge auth headers with any provided headers
        headers = {**self.headers, **kwargs.pop('headers', {})}
        return requests.post(url, headers=headers, **kwargs)
    
    def get_cdp_url(self) -> str:
        """Get the Chrome DevTools Protocol URL with WebSocket rewriting.
        
        Playwright's connect_over_cdp() performs discovery by fetching /json/version.
        Chrome returns ws://localhost/... which doesn't work for NVCF.
        We fetch the discovery ourselves and rewrite the WebSocket URL.
        """
        try:
            # Fetch the WebSocket URL from Chrome's discovery endpoint
            response = requests.get(
                f"{self.NVCF_CHROME_BASE}/json/version",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            ws_url = data.get("webSocketDebuggerUrl", "")
            
            if not ws_url:
                logger.warning("No webSocketDebuggerUrl in Chrome discovery response")
                return self.NVCF_CHROME_BASE
            
            # Rewrite ws://localhost/... to wss://grpc.nvcf.nvidia.com/chrome/...
            rewritten_url = ws_url.replace(
                "ws://localhost/", 
                "wss://grpc.nvcf.nvidia.com/chrome/"
            )
            logger.debug(f"CDP URL rewritten: {ws_url} -> {rewritten_url}")
            return rewritten_url
            
        except Exception as e:
            logger.error(f"Failed to get CDP URL from NVCF: {e}")
            # Return base URL as fallback - Playwright will do its own discovery
            return self.NVCF_CHROME_BASE
    
    def get_cdp_headers(self) -> Dict[str, str]:
        """Get authentication headers for CDP connection."""
        return self.headers.copy()
    
    def get_vlc_url(self) -> str:
        """Get the VLC web interface URL through NVCF."""
        return self.NVCF_VLC_BASE

    def update_launch_command(self, command: str) -> str:
        """Update the launch command to use the HTTP client. NVCF will have different command for launching apps"""

        if command[0] == "google-chrome":
                      
            command = [ "google-chrome-wrapper",
                "--remote-debugging-port=9223",
                "--remote-debugging-address=127.0.0.1",
                "--remote-allow-origins=*",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-infobars",
                "--disable-session-crashed-bubble",
                "--disable-features=TranslateUI",
                "--start-maximized"
            ]

        return command
