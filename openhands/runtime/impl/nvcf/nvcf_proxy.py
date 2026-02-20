"""Local proxy for NVCF services that injects authentication headers.

This module provides a local HTTP/WebSocket proxy that forwards requests to NVCF
with the required Authorization and Function-ID headers. This is necessary for
services like Chrome DevTools Protocol and VLC that don't support custom headers.
"""

import asyncio
import socket
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
import urllib.request
import urllib.error
import ssl
import json

from openhands.core.logger import openhands_logger as logger


class NVCFProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler that forwards to NVCF with auth headers."""
    
    # Class-level configuration (set by NVCFLocalProxy)
    nvcf_base_url: str = ""
    nvcf_path_prefix: str = ""
    api_key: str = ""
    function_id: str = ""
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def _get_target_url(self, path: str) -> str:
        """Convert local path to NVCF target URL."""
        # Remove leading slash for clean join
        path = path.lstrip("/")
        # Build target URL: base + path_prefix + path
        prefix = self.nvcf_path_prefix.strip("/")
        if prefix:
            return f"{self.nvcf_base_url}/{prefix}/{path}"
        return f"{self.nvcf_base_url}/{path}"
    
    def _forward_request(self, method: str) -> None:
        """Forward request to NVCF."""
        target_url = self._get_target_url(self.path)
        
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None
        
        # Build headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Function-ID": self.function_id,
        }
        
        # Copy relevant headers from original request
        for header in ['Content-Type', 'Accept', 'User-Agent']:
            if header in self.headers:
                headers[header] = self.headers[header]
        
        try:
            # Create request
            req = urllib.request.Request(
                target_url,
                data=body,
                headers=headers,
                method=method,
            )
            
            # Create SSL context that doesn't verify (for simplicity)
            ctx = ssl.create_default_context()
            
            # Make request
            with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
                # Send response status
                self.send_response(response.status)
                
                # Forward response headers
                for header, value in response.headers.items():
                    if header.lower() not in ('transfer-encoding', 'connection'):
                        self.send_header(header, value)
                self.end_headers()
                
                # Forward response body
                self.wfile.write(response.read())
                
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            error_body = e.read() if e.fp else b''
            self.wfile.write(error_body or f"HTTP Error {e.code}: {e.reason}".encode())
        except urllib.error.URLError as e:
            self.send_response(502)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Proxy Error: {e.reason}".encode())
        except Exception as e:
            self.send_response(502)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Proxy Error: {e}".encode())
    
    def do_GET(self):
        self._forward_request("GET")
    
    def do_POST(self):
        self._forward_request("POST")
    
    def do_PUT(self):
        self._forward_request("PUT")
    
    def do_DELETE(self):
        self._forward_request("DELETE")
    
    def do_OPTIONS(self):
        self._forward_request("OPTIONS")
    
    def do_HEAD(self):
        self._forward_request("HEAD")


class NVCFLocalProxy:
    """Local proxy that forwards requests to NVCF with auth headers.
    
    Supports HTTP connections. Used for Chrome DevTools and VLC web interface
    access through NVCF.
    
    Note: This is a simple HTTP proxy. For full WebSocket support (needed for
    Chrome DevTools), you may need a more sophisticated solution.
    """
    
    def __init__(
        self,
        nvcf_base_url: str,
        nvcf_path_prefix: str,
        api_key: str,
        function_id: str,
        local_port: Optional[int] = None,
    ):
        """Initialize the NVCF local proxy.
        
        Args:
            nvcf_base_url: Base URL for NVCF (e.g., https://grpc.nvcf.nvidia.com)
            nvcf_path_prefix: Path prefix for the service (e.g., /chrome, /vlc)
            api_key: NGC API key for authentication
            function_id: NVCF function ID
            local_port: Local port to listen on (auto-assigned if None)
        """
        self.nvcf_base_url = nvcf_base_url.rstrip("/")
        self.nvcf_path_prefix = nvcf_path_prefix
        self.api_key = api_key
        self.function_id = function_id
        self.local_port = local_port or self._find_available_port()
        
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
    
    @staticmethod
    def _find_available_port() -> int:
        """Find an available port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    @property
    def local_url(self) -> str:
        """Get the local proxy URL."""
        return f"http://127.0.0.1:{self.local_port}"
    
    def _create_handler_class(self):
        """Create a handler class with configuration bound."""
        class ConfiguredHandler(NVCFProxyHandler):
            nvcf_base_url = self.nvcf_base_url
            nvcf_path_prefix = self.nvcf_path_prefix
            api_key = self.api_key
            function_id = self.function_id
        return ConfiguredHandler
    
    def _run_server(self) -> None:
        """Run the HTTP server in a thread."""
        handler_class = self._create_handler_class()
        self._server = HTTPServer(("127.0.0.1", self.local_port), handler_class)
        self._running = True
        self._server.serve_forever()
    
    def start(self) -> None:
        """Start the proxy server in a background thread."""
        if self._running:
            return
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        # Wait for server to be ready
        for _ in range(50):  # 5 seconds timeout
            time.sleep(0.1)
            if self._running:
                break
        logger.debug(f"NVCF proxy started on port {self.local_port}")
    
    def stop(self) -> None:
        """Stop the proxy server."""
        if not self._running:
            return
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.debug(f"NVCF proxy stopped on port {self.local_port}")


def find_available_port() -> int:
    """Find an available port on localhost."""
    return NVCFLocalProxy._find_available_port()
