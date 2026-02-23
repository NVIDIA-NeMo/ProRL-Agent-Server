"""OSWorld NVCF runtime: same API as OSWorldSingularityRuntime but dispatches to NVCF API."""

import os
import time
import threading
from typing import TYPE_CHECKING, Any, Optional

import httpx

from openhands.core.config import OpenHandsConfig
from openhands.core.exceptions import AgentRuntimeDisconnectedError
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.events.tool import ToolCallMetadata
from openhands.runtime.impl.nvcf.nvcf_runtime import NVCFRuntime
from openhands.runtime.impl.nvcf.nvcf_proxy import NVCFLocalProxy
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.utils.osworld_http_client import NVCFHttpClient

if TYPE_CHECKING:
    from openhands.events.observation import Observation

NVCF_API_BASE = "https://grpc.nvcf.nvidia.com/api"
NVCF_BASE_URL = "https://grpc.nvcf.nvidia.com"


class OSWorldNVCFRuntime(NVCFRuntime):
    """Runtime for OSWorld via NVCF. Same API as OSWorldSingularityRuntime; dispatches to NVCF."""

    def __init__(
        self,
        config: OpenHandsConfig,
        event_stream: EventStream,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Any | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
        user_id: str | None = None,
        git_provider_tokens: Any = None,
        nvcf_function_id: str | None = None,
        nvcf_version_id: str | None = None,
        nvcf_api_key: str | None = None,
        nvcf_org: str | None = None,
        nvcf_function_config: Any = None,
        nvcf_deployment_config: Any = None,
        undeploy_on_close: bool = True,
        os_type: str = 'linux',
        enable_chrome_proxy: bool = True,
        enable_vlc_proxy: bool = True,
    ):
        self.os_type = os_type.lower()
        self._nvcf_client: httpx.Client | None = None
        self.screen_size: tuple[int, int] = (1920, 1080)  # updated by get_vm_screen_size
        
        # Local proxy settings
        self._enable_chrome_proxy = enable_chrome_proxy
        self._enable_vlc_proxy = enable_vlc_proxy
        self._chrome_proxy: Optional[NVCFLocalProxy] = None
        self._vlc_proxy: Optional[NVCFLocalProxy] = None
        
        super().__init__(
            config=config,
            event_stream=event_stream,
            sid=sid,
            plugins=plugins,
            env_vars=env_vars,
            status_callback=status_callback,
            attach_to_existing=attach_to_existing,
            headless_mode=headless_mode,
            user_id=user_id,
            git_provider_tokens=git_provider_tokens,
            nvcf_function_id=nvcf_function_id,
            nvcf_version_id=nvcf_version_id,
            nvcf_api_key=nvcf_api_key,
            nvcf_org=nvcf_org,
            nvcf_function_config=nvcf_function_config,
            nvcf_deployment_config=nvcf_deployment_config,
            undeploy_on_close=undeploy_on_close,
        )

    async def connect(self) -> None:
        await super().connect()
        self.log("debug", "Connecting to OSWorld NVCF function...")
        headers = {
            "Authorization": f"Bearer {self._nvcf_api_key}",
            "Function-ID": self._nvcf_function_id,
        }
        self._nvcf_client = httpx.Client(
            base_url=NVCF_API_BASE,
            headers=headers,
            timeout=60.0,
        )
        # Verify endpoint and capture NVCF session headers
        r = self._nvcf_client.get("/screenshot", timeout=15.0)
        if r.status_code != 200:
            self._nvcf_client.close()
            self._nvcf_client = None
            raise AgentRuntimeDisconnectedError(
                f"NVCF function returned HTTP {r.status_code}"
            )
        # NVCF stateful functions return session routing headers — persist them
        self._nvcf_session_headers = {}
        self.log("info", f"NVCF init response headers: {dict(r.headers)}")
        for hdr in ("NVCF-REQID", "NVCF-SESSION-ID", "nvcf-reqid", "nvcf-session-id"):
            val = r.headers.get(hdr)
            if val:
                self._nvcf_client.headers[hdr] = val
                self._nvcf_session_headers[hdr] = val
                self.log("info", f"Captured NVCF session header: {hdr}={val}")
        self.log("info", f"NVCF session headers captured: {self._nvcf_session_headers}")
        self.log("info", f"OSWorld NVCF client ready: {self._nvcf_function_id}")

        # Start NVCF session keepalive to prevent idle timeout (~30-60s)
        self._keepalive_stop = threading.Event()
        self._keepalive_thread = threading.Thread(
            target=self._nvcf_keepalive_loop, daemon=True
        )
        self._keepalive_thread.start()

        # Start local proxies for Chrome DevTools and VLC
        self._start_local_proxies()
    
    def _nvcf_keepalive_loop(self) -> None:
        """Ping the NVCF function every 20s to prevent session idle timeout."""
        while not self._keepalive_stop.wait(20.0):
            try:
                r = self.http_client.get("/platform", timeout=10.0)
                if r.status_code != 200:
                    self.log("warning", f"Keepalive got HTTP {r.status_code}")
            except Exception as e:
                self.log("warning", f"Keepalive failed: {e}")

    def _start_local_proxies(self) -> None:
        """Start local proxies for Chrome DevTools and VLC web interface."""
        if self._enable_chrome_proxy:
            try:
                self._chrome_proxy = NVCFLocalProxy(
                    nvcf_base_url=NVCF_BASE_URL,
                    nvcf_path_prefix="/chrome",
                    api_key=self._nvcf_api_key,
                    function_id=self._nvcf_function_id,
                )
                self._chrome_proxy.start()
                self.log("info", f"Chrome DevTools proxy started at {self._chrome_proxy.local_url}")
            except Exception as e:
                self.log("warning", f"Failed to start Chrome proxy: {e}")
                self._chrome_proxy = None
        
        if self._enable_vlc_proxy:
            try:
                self._vlc_proxy = NVCFLocalProxy(
                    nvcf_base_url=NVCF_BASE_URL,
                    nvcf_path_prefix="/vlc",
                    api_key=self._nvcf_api_key,
                    function_id=self._nvcf_function_id,
                )
                self._vlc_proxy.start()
                self.log("info", f"VLC web interface proxy started at {self._vlc_proxy.local_url}")
            except Exception as e:
                self.log("warning", f"Failed to start VLC proxy: {e}")
                self._vlc_proxy = None
    
    def _stop_local_proxies(self) -> None:
        """Stop all local proxies."""
        if self._chrome_proxy:
            try:
                self._chrome_proxy.stop()
                self.log("debug", "Chrome DevTools proxy stopped")
            except Exception as e:
                self.log("warning", f"Failed to stop Chrome proxy: {e}")
            self._chrome_proxy = None
        
        if self._vlc_proxy:
            try:
                self._vlc_proxy.stop()
                self.log("debug", "VLC web interface proxy stopped")
            except Exception as e:
                self.log("warning", f"Failed to stop VLC proxy: {e}")
            self._vlc_proxy = None

    def check_if_alive(self) -> None:
        if not self._nvcf_client:
            raise AgentRuntimeDisconnectedError("OSWorld NVCF runtime is not connected.")
        r = self._nvcf_get("/screenshot", timeout=5.0)
        if r.status_code != 200:
            raise AgentRuntimeDisconnectedError("NVCF function is not responding")

    def close(self, rm_all_containers: bool | None = None) -> None:
        # Stop keepalive thread
        if hasattr(self, '_keepalive_stop'):
            self._keepalive_stop.set()

        # Stop local proxies first
        self._stop_local_proxies()

        # Clear shared http_client
        if hasattr(self, '_http_client'):
            self._http_client = None

        if self._nvcf_client:
            try:
                self._nvcf_client.close()
            except Exception as e:
                logger.warning(f"Failed to close NVCF client: {e}")
            self._nvcf_client = None
        super().close(rm_all_containers)

    # --- Properties (match OSWorldSingularityRuntime) ---
    @property
    def osworld_vm_url(self) -> str:
        return NVCF_API_BASE

    @property
    def vnc_url(self) -> str:
        # VNC is not currently proxied (would need WebSocket support for noVNC)
        return "vnc://nvcf-not-available"

    @property
    def chromium_devtools_url(self) -> str:
        """Get the Chrome DevTools URL (local proxy or placeholder)."""
        if self._chrome_proxy and self._chrome_proxy._running:
            return self._chrome_proxy.local_url
        return "http://nvcf-not-available"
    
    @property
    def chromium_port(self) -> int:
        """Get the local Chrome DevTools proxy port."""
        if self._chrome_proxy and self._chrome_proxy._running:
            return self._chrome_proxy.local_port
        return 9222  # Default fallback

    @property
    def vlc_url(self) -> str:
        """Get the VLC web interface URL (local proxy or placeholder)."""
        if self._vlc_proxy and self._vlc_proxy._running:
            return self._vlc_proxy.local_url
        return "http://nvcf-not-available"
    
    @property
    def vlc_port(self) -> int:
        """Get the local VLC proxy port."""
        if self._vlc_proxy and self._vlc_proxy._running:
            return self._vlc_proxy.local_port
        return 8080  # Default fallback
    
    @property
    def vm_ip(self) -> str:
        """Get the VM IP for setup controller compatibility.
        
        For NVCF, this returns localhost since we use local proxies.
        """
        return "127.0.0.1"
    
    @property
    def http_client(self):
        """Get the HTTP client for runtime-agnostic communication.

        Returns a shared NVCFHttpClient that handles NVCF authentication and URL rewriting.
        Reuses the same instance so NVCF session state is preserved across all callers.
        """
        if not hasattr(self, '_http_client') or self._http_client is None:
            self._http_client = NVCFHttpClient(
                api_key=self._nvcf_api_key,
                function_id=self._nvcf_function_id,
                session_headers=getattr(self, '_nvcf_session_headers', None),
            )
        return self._http_client

    # --- OSWorld API (NVCF HTTP) ---
    def _nvcf_get(self, endpoint: str, **kwargs):
        """GET via shared http_client (requests-based) to maintain NVCF session."""
        return self.http_client.get(endpoint, **kwargs)

    def _nvcf_post(self, endpoint: str, **kwargs):
        """POST via shared http_client (requests-based) to maintain NVCF session."""
        return self.http_client.post(endpoint, **kwargs)

    def get_vm_screenshot(self) -> bytes | None:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                r = self._nvcf_get("/screenshot", timeout=30.0)
                if r.status_code == 200:
                    return r.content
                body = r.text[:200] if r.text else ""
                self.log("warning",
                    f"Screenshot attempt {attempt + 1}/{max_retries} failed: "
                    f"HTTP {r.status_code} fn={self._nvcf_function_id} body={body}")
            except Exception as e:
                self.log("warning", f"Screenshot attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5.0)
        self.log("error", f"Failed to get VM screenshot after {max_retries} retries (fn={self._nvcf_function_id})")
        return None

    def get_vm_accessibility_tree(self) -> str | None:
        try:
            r = self._nvcf_get("/accessibility", timeout=30.0)
            if r.status_code != 200:
                return None
            try:
                return r.json().get("AT") or r.json().get("accessibility_tree") or r.text
            except Exception:
                return r.text
        except Exception as e:
            self.log("error", f"Failed to get VM accessibility tree: {e}")
            return None

    def _execute_pyautogui_command(self, pyautogui_command: str) -> dict:
        command = (
            "import pyautogui; import time; pyautogui.FAILSAFE = False; "
            + pyautogui_command
        )
        payload = {"command": ["python", "-c", command], "shell": False}
        max_retries = 5
        for attempt in range(max_retries):
            try:
                r = self._nvcf_post("/execute", json=payload, timeout=30.0)
                if r.status_code == 200:
                    return r.json()
                body = r.text[:200] if r.text else ""
                self.log("warning",
                    f"Execute attempt {attempt + 1}/{max_retries} failed: "
                    f"HTTP {r.status_code} fn={self._nvcf_function_id} body={body}")
            except Exception as e:
                self.log("warning", f"Execute attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5.0)
        return {"status": "error", "message": f"Failed after {max_retries} retries"}

    def _action_to_pyautogui_command(self, action_type: str, parameters: dict) -> str | None:
        import random
        move_mode = random.choice([
            "pyautogui.easeInQuad", "pyautogui.easeOutQuad", "pyautogui.easeInOutQuad",
            "pyautogui.easeInBounce", "pyautogui.easeInElastic",
        ])
        if action_type == "CLICK":
            x, y = parameters.get("x"), parameters.get("y")
            button = parameters.get("button", "left")
            num_clicks = parameters.get("clicks", 1)
            interval = parameters.get("interval", 0.0)
            duration = parameters.get("duration", 0.0)
            if x is not None and y is not None:
                return f"pyautogui.click(x={x}, y={y}, button='{button}', clicks={num_clicks}, interval={interval}, duration={duration})"
            return "pyautogui.click()"
        elif action_type == "DOUBLE_CLICK":
            x, y = parameters.get("x"), parameters.get("y")
            button = parameters.get("button", "left")
            interval = parameters.get("interval", 0.0)
            duration = parameters.get("duration", 0.0)
            if x is not None and y is not None:
                return f"pyautogui.doubleClick(x={x}, y={y}, button='{button}', interval={interval}, duration={duration})"
            return "pyautogui.doubleClick()"
        elif action_type == "TRIPLE_CLICK":
            x, y = parameters.get("x"), parameters.get("y")
            button = parameters.get("button", "left")
            interval = parameters.get("interval", 0.0)
            duration = parameters.get("duration", 0.0)
            if x is not None and y is not None:
                return f"pyautogui.tripleClick(x={x}, y={y}, button='{button}', interval={interval}, duration={duration})"
            return "pyautogui.tripleClick()"
        elif action_type == "RIGHT_CLICK":
            x, y = parameters.get("x"), parameters.get("y")
            interval = parameters.get("interval", 0.0)
            duration = parameters.get("duration", 0.0)
            if x is not None and y is not None:
                return f"pyautogui.rightClick(x={x}, y={y}, interval={interval}, duration={duration})"
            return "pyautogui.rightClick()"
        elif action_type == "MIDDLE_CLICK":
            x, y = parameters.get("x"), parameters.get("y")
            interval = parameters.get("interval", 0.0)
            duration = parameters.get("duration", 0.0)
            if x is not None and y is not None:
                return f"pyautogui.middleClick(x={x}, y={y}, interval={interval}, duration={duration})"
            return "pyautogui.click(button='middle')"
        elif action_type == "MOVE_TO":
            x, y = parameters.get("x"), parameters.get("y")
            duration = parameters.get("duration", 0.0)
            if x is not None and y is not None:
                return f"pyautogui.moveTo({x}, {y}, {duration}, {move_mode})"
            return "pyautogui.moveTo()"
        elif action_type == "DRAG_TO":
            x, y = parameters.get("x"), parameters.get("y")
            duration = parameters.get("duration", 0.0)
            button = parameters.get("button", "left")
            mouseDownUp = parameters.get("mouseDownUp", True)
            if x is not None and y is not None:
                return f"pyautogui.dragTo({x}, {y}, button='{button}', duration={duration}, mouseDownUp={mouseDownUp})"
            return None
        elif action_type == "SCROLL":
            x, y = parameters.get("x"), parameters.get("y")
            amount = parameters.get("amount", 1)
            return f"pyautogui.scroll({amount}, x={x}, y={y})"
        elif action_type == "HSCROLL":
            x, y = parameters.get("x"), parameters.get("y")
            amount = parameters.get("amount", 1)
            return f"pyautogui.hscroll({amount}, x={x}, y={y})"
        elif action_type == "TYPING":
            text = parameters.get("text", "")
            interval = parameters.get("interval", 0.0)
            return f"pyautogui.typewrite({repr(text)}, interval={interval})"
        elif action_type == "PRESS":
            key = parameters.get("key", "")
            presses = parameters.get("presses", 1)
            if isinstance(key, list):
                return f"pyautogui.hotkey({', '.join(repr(k) for k in key)})"
            if presses > 1:
                return f"pyautogui.press('{key}', presses={presses})"
            return f"pyautogui.press('{key}')"
        elif action_type == "HOTKEY":
            keys = parameters.get("keys", [])
            if isinstance(keys, list) and keys:
                return f"pyautogui.hotkey({', '.join(repr(k) for k in keys)})"
            return None
        elif action_type == "KEY_DOWN":
            return f"pyautogui.keyDown('{parameters.get('key', '')}')"
        elif action_type == "KEY_UP":
            return f"pyautogui.keyUp('{parameters.get('key', '')}')"
        elif action_type == "MOUSE_DOWN":
            return f"pyautogui.mouseDown(button='{parameters.get('button', 'left')}')"
        elif action_type == "MOUSE_UP":
            return f"pyautogui.mouseUp(button='{parameters.get('button', 'left')}')"
        elif action_type == "WAIT":
            return f"time.sleep({parameters.get('seconds', 1)})"
        return None

    def execute_vm_action(self, action_data: dict) -> dict:
        action_type = action_data.get("action_type")
        parameters = action_data.get("parameters", {})
        cmd = self._action_to_pyautogui_command(action_type, parameters)
        if cmd is None:
            return {"status": "error", "message": f"Unknown action type: {action_type}"}
        return self._execute_pyautogui_command(cmd)

    def run_action(self, action) -> "Observation":
        from openhands.events.action.os import OSWorldInteractiveAction
        if isinstance(action, OSWorldInteractiveAction):
            return self.osworld_interactive(action)
        return super().run_action(action)

    def osworld_interactive(self, action) -> "Observation":
        from openhands.events.observation import ErrorObservation
        method = action.method
        params = action.params or {}
        try:
            if method == "execute_action":
                return self._handle_execute_action(params)
            if method == "execute_agentic_action":
                return self._handle_execute_agentic_action(
                    params, action.tool_call_metadata, action.pause_time
                )
            if method == "get_screenshot":
                return self._handle_get_screenshot()
            if method == "get_accessibility_tree":
                return self._handle_get_accessibility_tree()
            if method == "get_terminal_output":
                return self._handle_get_terminal_output()
            if method == "get_file":
                return self._handle_get_file(params)
            if method == "execute_python_command":
                return self._handle_execute_python_command(params)
            if method == "run_python_script":
                return self._handle_run_python_script(params)
            if method == "run_bash_script":
                return self._handle_run_bash_script(params)
            if method == "start_recording":
                return self._handle_start_recording()
            if method == "end_recording":
                return self._handle_end_recording(params)
            if method == "get_vm_platform":
                return self._handle_get_vm_platform()
            if method == "get_vm_screen_size":
                return self._handle_get_vm_screen_size()
            if method == "get_vm_window_size":
                return self._handle_get_vm_window_size(params)
            if method == "get_vm_wallpaper":
                return self._handle_get_vm_wallpaper()
            if method == "get_vm_desktop_path":
                return self._handle_get_vm_desktop_path()
            if method == "get_vm_directory_tree":
                return self._handle_get_vm_directory_tree(params)
            return ErrorObservation(f"Unknown OSWorld method: {method}")
        except Exception as e:
            self.log("error", f"OSWorld action failed: {e}")
            return ErrorObservation(str(e))

    def _handle_execute_action(self, params: dict) -> "Observation":
        from openhands.events.observation import CmdOutputObservation
        action_data = params.get("action", params)
        result = self.execute_vm_action(action_data)
        if result.get("status") == "success":
            return CmdOutputObservation(
                content=result.get("output", "Action executed successfully"),
                command=str(action_data),
                exit_code=0,
            )
        return CmdOutputObservation(
            content=result.get("error", result.get("message", "Unknown error")),
            command=str(action_data),
            exit_code=1,
        )

    def _handle_execute_agentic_action(
        self, params: dict, tool_call_metadata: ToolCallMetadata | None, pause_time: float = 0.0
    ) -> "Observation":
        from openhands.events.observation import ErrorObservation
        from openhands.events.observation.osworld import OSWorldOutputObservation
        import base64
        action_data = params.get("action", params)
        if not getattr(self, "screen_size", None):
            self._handle_get_vm_screen_size()
        width, height = self.screen_size
        if "parameters" in action_data:
            p = action_data["parameters"]
            if "x" in p:
                p["x"] = int(p["x"] * width)
            if "y" in p:
                p["y"] = int(p["y"] * height)
        result = self.execute_vm_action(action_data)
        if result.get("status") != "success":
            return ErrorObservation(
                result.get("error", result.get("message", "Unknown error")),
                error_id=getattr(tool_call_metadata, "tool_call_id", None),
                name=getattr(tool_call_metadata, "function_name", None),
            )
        if pause_time > 0.5:
            time.sleep(pause_time)
        screenshot_b64 = None
        screenshot_bytes = self.get_vm_screenshot()
        if screenshot_bytes:
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        a11y = self.get_vm_accessibility_tree()
        return OSWorldOutputObservation(
            command=str(action_data),
            screenshot=screenshot_b64,
            accessibility_tree=a11y,
            tool_call_id=tool_call_metadata.tool_call_id if tool_call_metadata else None,
            name=tool_call_metadata.function_name if tool_call_metadata else None,
        )

    def _handle_get_screenshot(self) -> "Observation":
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        # NVCF may need a moment after connect; retry once to avoid transient failure
        screenshot_bytes = self.get_vm_screenshot()
        if not screenshot_bytes:
            time.sleep(2.0)
            screenshot_bytes = self.get_vm_screenshot()
        if screenshot_bytes:
            return CmdOutputObservation(
                content="Screenshot captured",
                command="get_screenshot",
                exit_code=0,
            )
        return ErrorObservation("Failed to capture screenshot")

    def _handle_get_accessibility_tree(self) -> "Observation":
        from openhands.events.observation import CmdOutputObservation
        # NVCF: GET /api/accessibility only
        at = self.get_vm_accessibility_tree()
        if at:
            return CmdOutputObservation(content=at, command="get_accessibility_tree", exit_code=0)
        return CmdOutputObservation(content="", command="get_accessibility_tree", exit_code=0)

    def _handle_get_terminal_output(self) -> "Observation":
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        for attempt in range(5):
            try:
                r = self._nvcf_get("/terminal", timeout=30.0)
                if r.status_code == 200:
                    output = r.json().get("output") or ""
                    return CmdOutputObservation(
                        content=output,
                        command="get_terminal_output",
                        exit_code=0,
                    )
                body = r.text[:200] if r.text else ""
                self.log("warning", f"Terminal output attempt {attempt + 1}/5: HTTP {r.status_code} body={body}")
            except Exception as e:
                self.log("warning", f"Terminal output attempt {attempt + 1}/5: {e}")
            if attempt < 4:
                time.sleep(5.0)
        return ErrorObservation("Failed to get terminal output after 5 retries")

    def _handle_get_file(self, params: dict) -> "Observation":
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        import base64
        file_path = params.get("file_path", "")
        if not file_path:
            return ErrorObservation("file_path parameter required")
        try:
            r = self._nvcf_post(
                "/file",
                data={"file_path": file_path},
                timeout=30.0,
            )
            if r.status_code == 200:
                content_b64 = base64.b64encode(r.content).decode("utf-8")
                return CmdOutputObservation(
                    content=f"base64:{content_b64}",
                    command=f"get_file {file_path}",
                    exit_code=0,
                )
            return ErrorObservation(f"Failed to get file: {r.status_code}")
        except Exception as e:
            return ErrorObservation(f"Failed to get file: {e}")

    def _execute_pyautogui_command(self, pyautogui_command: str) -> dict:
        """Execute a PyAutoGUI command string in the VM (same as singularity).

        Args:
            pyautogui_command: Raw PyAutoGUI command(s) to execute

        Returns:
            Response dictionary from OSWorld server (status, output, error, returncode or message).
        """
        wrapped = (
            "import pyautogui; import time; pyautogui.FAILSAFE = False; "
            f"{pyautogui_command}"
        )
        payload = {"command": ["python", "-c", wrapped], "shell": False}
        max_retries = 5
        for attempt in range(max_retries):
            try:
                r = self._nvcf_post("/execute", json=payload, timeout=30.0)
                if r.status_code == 200:
                    return r.json()
                body = r.text[:200] if r.text else ""
                self.log("warning",
                    f"Execute attempt {attempt + 1}/{max_retries} failed: "
                    f"HTTP {r.status_code} fn={self._nvcf_function_id} body={body}")
            except Exception as e:
                self.log("warning", f"Execute attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5.0)
        self.log("error", f"Failed to execute PyAutoGUI command after {max_retries} retries: {pyautogui_command[:100]}")
        return {"status": "error", "message": f"Failed after {max_retries} retries"}

    def _handle_execute_python_command(self, params: dict) -> "Observation":
        """Handle execute_python_command - raw Python command execution (PyAutoGUI-style, same as singularity)."""
        from openhands.events.observation import CmdOutputObservation

        command = params.get("command", "")
        if not command:
            return CmdOutputObservation(
                content="Error: command parameter required",
                command="execute_python_command",
                exit_code=1,
            )

        result = self._execute_pyautogui_command(command)

        if result.get("status") == "success":
            return CmdOutputObservation(
                content=result.get("output", ""),
                command=command,
                exit_code=result.get("returncode", 0),
            )
        else:
            error_msg = result.get("error", result.get("message", "Unknown error"))
            return CmdOutputObservation(
                content=f"Error: {error_msg}",
                command=command,
                exit_code=result.get("returncode", 1),
            )

    def _handle_run_python_script(self, params: dict) -> "Observation":
        from openhands.events.observation import CmdOutputObservation
        script = params.get("script", "")
        if not script:
            return CmdOutputObservation(content="script parameter required", command="run_python_script", exit_code=1)
        try:
            r = self._nvcf_post("/run_python", json={"code": script}, timeout=90.0)
            if r.status_code == 200:
                res = r.json()
                out, err = res.get("output", ""), res.get("error", "")
                return CmdOutputObservation(
                    content=f"Output:\n{out}" + (f"\nError:\n{err}" if err else ""),
                    command="run_python_script",
                    exit_code=res.get("returncode", 0),
                )
            return CmdOutputObservation(content=r.text or "Run failed", command="run_python_script", exit_code=1)
        except Exception as e:
            return CmdOutputObservation(content=str(e), command="run_python_script", exit_code=1)

    def _handle_run_bash_script(self, params: dict) -> "Observation":
        """Handle run_bash_script. Uses POST /api/run_bash_script."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation

        script = params.get("script", "")
        if not script:
            return ErrorObservation("script parameter required")
        timeout = params.get("timeout", 30)
        working_dir = params.get("working_dir")

        payload = {"script": script, "timeout": timeout}
        if working_dir is not None:
            payload["working_dir"] = working_dir

        for attempt in range(5):
            try:
                r = self._nvcf_post(
                    "/run_bash_script",
                    json=payload,
                    timeout=timeout + 10.0,
                )
                if r.status_code == 200:
                    result = r.json()
                    output = result.get("output", "")
                    error = result.get("error", "")
                    content = output
                    if error:
                        content = f"{content}\n{error}" if content else error
                    return CmdOutputObservation(
                        content=content,
                        command="run_bash_script",
                        exit_code=result.get("returncode", 0),
                    )
                if r.status_code in (404, 502, 503, 504):
                    body = r.text[:200] if r.text else ""
                    self.log("warning",
                        f"run_bash_script attempt {attempt + 1}/5: HTTP {r.status_code} body={body}")
                    if attempt < 4:
                        time.sleep(5.0)
                        continue
                try:
                    error_detail = r.json()
                    error_msg = error_detail.get("output", error_detail.get("message", "Unknown error"))
                except Exception:
                    error_msg = r.text or "Unknown error"
                return ErrorObservation(f"Failed to run bash script (HTTP {r.status_code}): {error_msg}")
            except Exception as e:
                self.log("warning", f"run_bash_script attempt {attempt + 1}/5: {e}")
                if attempt < 4:
                    time.sleep(5.0)
                    continue
                return ErrorObservation(f"Failed to run bash script: {e}")
        return ErrorObservation("Failed to run bash script after 5 retries")

    def _handle_start_recording(self) -> "Observation":
        """Handle start_recording. Uses POST /api/start_recording."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation

        try:
            r = self._nvcf_post("/start_recording", timeout=10.0)
            if r.status_code == 200:
                return CmdOutputObservation(
                    content="Recording started",
                    command="start_recording",
                    exit_code=0,
                )
            return ErrorObservation(f"Failed to start recording: {r.status_code}")
        except Exception as e:
            return ErrorObservation(f"Failed to start recording: {e}")

    def _handle_end_recording(self, params: dict) -> "Observation":
        """Handle end_recording. POST /api/end_recording returns video file (binary); return as base64."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        import base64

        try:
            r = self._nvcf_post("/end_recording", timeout=60.0)
            if r.status_code == 200:
                video_content = r.content
                content_b64 = base64.b64encode(video_content).decode("utf-8")
                return CmdOutputObservation(
                    content=f"base64:{content_b64}",
                    command="end_recording",
                    exit_code=0,
                )
            return ErrorObservation(f"Failed to end recording: {r.status_code}")
        except Exception as e:
            return ErrorObservation(f"Failed to end recording: {e}")

    def _handle_get_vm_platform(self) -> "Observation":
        """Handle get_vm_platform. Uses GET /api/platform (returns plain text e.g. Linux)."""
        from openhands.events.observation import CmdOutputObservation

        try:
            r = self._nvcf_get("/platform", timeout=30.0)
            if r.status_code == 200:
                platform_str = (r.text or "").strip()
                return CmdOutputObservation(
                    content=platform_str or "Unknown",
                    command="get_vm_platform",
                    exit_code=0,
                )
        except Exception:
            pass
        result = self._execute_pyautogui_command("import platform; print(platform.system())")
        if result.get("status") == "success":
            return CmdOutputObservation(
                content=result.get("output", "").strip() or "Unknown",
                command="get_vm_platform",
                exit_code=0,
            )
        return CmdOutputObservation(content="Unknown", command="get_vm_platform", exit_code=1)

    def _handle_get_vm_screen_size(self) -> "Observation":
        """Handle get_vm_screen_size. Uses POST /api/screen_size."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation

        try:
            if hasattr(self, "screen_size") and self.screen_size:
                width, height = self.screen_size
            else:
                r = self._nvcf_post("/screen_size", timeout=30.0)
                if r.status_code != 200:
                    return ErrorObservation(f"Failed to get screen size: {r.status_code}")
                size = r.json()
                width = size.get("width", 1920)
                height = size.get("height", 1080)
                self.screen_size = (width, height)
            return CmdOutputObservation(
                content=f"Width: {width}, Height: {height}",
                command="get_vm_screen_size",
                exit_code=0,
            )
        except Exception as e:
            return ErrorObservation(f"Failed to get screen size: {e}")

    def _handle_get_vm_window_size(self, params: dict) -> "Observation":
        from openhands.events.observation import CmdOutputObservation
        # NVCF has no /api/window_size; use /api/execute with wmctrl only
        app = params.get("app_class_name", "") or "window"
        try:
            payload = {"command": ["wmctrl", "-l", "-G"], "shell": False}
            r = self._nvcf_post("/execute", json=payload, timeout=30.0)
            if r.status_code == 200:
                res = r.json()
                out = (res.get("output") or "").strip()
                if out:
                    return CmdOutputObservation(
                        content=out[:2000],
                        command=f"get_vm_window_size {app}",
                        exit_code=0,
                    )
            return CmdOutputObservation(
                content="Window geometry not available (use get_vm_screen_size for display size).",
                command=f"get_vm_window_size {app}",
                exit_code=0,
            )
        except Exception:
            return CmdOutputObservation(
                content="Window geometry not available (use get_vm_screen_size for display size).",
                command=f"get_vm_window_size {app}",
                exit_code=0,
            )

    def _handle_get_vm_wallpaper(self) -> "Observation":
        """Handle get_vm_wallpaper. POST /api/wallpaper returns wallpaper image (binary); return as base64."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        import base64

        try:
            r = self._nvcf_post("/wallpaper", timeout=30.0)
            if r.status_code == 200:
                wallpaper_bytes = r.content
                content_b64 = base64.b64encode(wallpaper_bytes).decode("utf-8")
                return CmdOutputObservation(
                    content=f"base64:{content_b64}",
                    command="get_vm_wallpaper",
                    exit_code=0,
                )
            return ErrorObservation(f"Failed to get wallpaper: {r.status_code}")
        except Exception as e:
            return ErrorObservation(f"Failed to get wallpaper: {e}")

    def _handle_get_vm_desktop_path(self) -> "Observation":
        """Handle get_vm_desktop_path. Uses POST /api/desktop_path."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation

        try:
            r = self._nvcf_post("/desktop_path", timeout=30.0)
            if r.status_code == 200:
                desktop_path = r.json().get("desktop_path", "")
                return CmdOutputObservation(
                    content=desktop_path,
                    command="get_vm_desktop_path",
                    exit_code=0,
                )
            return ErrorObservation(f"Failed to get desktop path: {r.status_code}")
        except Exception as e:
            return ErrorObservation(f"Failed to get desktop path: {e}")

    def _handle_get_vm_directory_tree(self, params: dict) -> "Observation":
        """Handle get_vm_directory_tree. Uses POST /api/list_directory with JSON {path}."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        import json

        path = params.get("path", "")
        if not path:
            return ErrorObservation("path parameter required")
        try:
            r = self._nvcf_post(
                "/list_directory",
                json={"path": path},
                timeout=30.0,
            )
            if r.status_code == 200:
                directory_tree = r.json().get("directory_tree", {})
                content = json.dumps(directory_tree, indent=2)
                return CmdOutputObservation(
                    content=content,
                    command=f"get_vm_directory_tree {path}",
                    exit_code=0,
                )
            return ErrorObservation(f"Failed to get directory tree: {r.status_code}")
        except Exception as e:
            return ErrorObservation(f"Failed to get directory tree: {e}")

    def get_microagents_from_selected_repo(self, selected_repository: str | None):
        return []
