"""OSWorld Singularity Runtime for running QEMU VMs in Singularity containers.

This runtime extends SingularityRuntime to support OSWorld environments by:
1. Running QEMU VMs inside Singularity containers
2. Managing port mappings for parallel VM instances
3. Communicating with OSWorld server running inside the VM
"""

import os
import subprocess
import signal
import time
import json
import threading
from pathlib import Path
from turtle import window_width
from typing import TYPE_CHECKING, Callable

import httpx

from openhands.core.config import OpenHandsConfig

if TYPE_CHECKING:
    from openhands.events.observation import Observation
from openhands.core.exceptions import (
    AgentRuntimeDisconnectedError,
    AgentRuntimeNotFoundError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.runtime.impl.singularity.singularity_runtime import (
    SingularityRuntime,
    CONTAINER_NAME_PREFIX,
)
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.utils import find_available_tcp_port
from openhands.runtime.utils.command import DEFAULT_MAIN_MODULE

from openhands.events.tool import ToolCallMetadata

# Port ranges for OSWorld VM services
OSWORLD_VM_SERVER_PORT_RANGE = (15000, 19999)  # OSWorld Flask server inside VM (port 5000)
OSWORLD_VNC_PORT_RANGE = (18000, 22999)  # VNC server for VM display
OSWORLD_CHROMIUM_PORT_RANGE = (19000, 22999)  # Chrome DevTools Protocol (port 9222)
OSWORLD_VLC_PORT_RANGE = (20000, 22999)  # VLC web interface (port 8080)

OSWORLD_CONTAINER_NAME_PREFIX = 'openhands-osworld-runtime-'


class OSWorldSingularityRuntime(SingularityRuntime):
    """Runtime for OSWorld environments using QEMU VMs in Singularity containers.
    
    This runtime manages QEMU VMs inside Singularity containers, providing:
    - Port management for parallel VM instances
    - Communication with OSWorld server inside the VM
    - Support for both Linux and Windows VMs
    """
    
    _osworld_port_allocation_lock = threading.Lock()
    
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
        os_type: str = 'linux',  # 'linux' or 'windows'
        vm_image_path: str | None = None,
    ):
        """Initialize OSWorld Singularity Runtime.
        
        Args:
            config: OpenHands configuration
            event_stream: Event stream for communication
            sid: Session ID
            plugins: List of plugin requirements
            env_vars: Environment variables
            status_callback: Callback for status updates
            attach_to_existing: Whether to attach to existing container
            headless_mode: Whether to run in headless mode
            main_module: Main module to run
            os_type: Type of OS in VM ('linux' or 'windows')
            vm_image_path: Path to QCOW2 VM image file
        
        Note:
            Snapshot mode is ALWAYS enabled (disk writes are not saved).
            This protects the base QCOW2 image from modifications.
            Use QCOW2 backing files if you need per-session persistence.
        """
        self.os_type = os_type.lower()
        self.vm_image_path = vm_image_path or self._get_default_vm_image_path()
        # Permanently enable snapshot mode to protect base images
        self.snapshot_mode = True
        self.qemu_process: subprocess.Popen | None = None
        self.qemu_pid: int | None = None
        self._qemu_stdout = None
        self._qemu_stderr = None
        self._vm_server_port: int = -1
        self._vnc_port: int = -1
        self._chromium_port: int = -1
        self._vlc_port: int = -1
        
        # Override container name prefix for OSWorld
        self.container_name = OSWORLD_CONTAINER_NAME_PREFIX + sid
        
        # Call parent constructor
        super().__init__(
            config=config,
            event_stream=event_stream,
            sid=sid,
            plugins=plugins,
            env_vars=env_vars,
            status_callback=status_callback,
            attach_to_existing=attach_to_existing,
            headless_mode=headless_mode,
            main_module=main_module,
        )
        
    def _get_default_vm_image_path(self) -> str:
        """Get default VM image path based on OS type."""
        if self.os_type == 'linux':
            return '/OS_images/Ubuntu.qcow2'
        elif self.os_type == 'windows':
            return '/OS_images/Windows-10-x64.qcow2'
        else:
            raise ValueError(f'Unsupported OS type: {self.os_type}')
    
    def _get_singularity_image_path(self) -> str:
        """Get the Singularity image file path for OSWorld runtime."""
        # Use a specific OSWorld runtime image
        if self.runtime_container_image:
            if not self.runtime_container_image.endswith('.sif'):
                image_name = f'osworld_{self.runtime_container_image.replace(":", "_").replace("/", "_")}'
                from openhands.runtime.utils.singularity_runtime_build import get_runtime_image_repo
                image_repo = get_runtime_image_repo()
                os.makedirs(image_repo, exist_ok=True)
                return f'{image_repo}/{image_name}.sif'
            else:
                return self.runtime_container_image
        # Default to ubuntu base for OSWorld
        from openhands.runtime.utils.singularity_runtime_build import get_runtime_image_repo
        image_repo = get_runtime_image_repo()
        os.makedirs(image_repo, exist_ok=True)
        return f'{image_repo}/osworld_ubuntu_24_04.sif'
    
    def _allocate_osworld_ports(self) -> tuple[int, int, int, int]:
        """Allocate ports for OSWorld VM services.
        
        Returns:
            Tuple of (vm_server_port, vnc_port, chromium_port, vlc_port)
        """
        with OSWorldSingularityRuntime._osworld_port_allocation_lock:
            vm_server_port = find_available_tcp_port(
                OSWORLD_VM_SERVER_PORT_RANGE[0],
                OSWORLD_VM_SERVER_PORT_RANGE[1]
            )
            vnc_port = find_available_tcp_port(
                OSWORLD_VNC_PORT_RANGE[0],
                OSWORLD_VNC_PORT_RANGE[1]
            )
            chromium_port = find_available_tcp_port(
                OSWORLD_CHROMIUM_PORT_RANGE[0],
                OSWORLD_CHROMIUM_PORT_RANGE[1]
            )
            vlc_port = find_available_tcp_port(
                OSWORLD_VLC_PORT_RANGE[0],
                OSWORLD_VLC_PORT_RANGE[1]
            )
            return vm_server_port, vnc_port, chromium_port, vlc_port
    
    def _check_kvm_available(self) -> bool:
        """Check if KVM is available and accessible.
        
        Returns:
            True if /dev/kvm exists and has read/write permissions, False otherwise
        """
        kvm_device = '/dev/kvm'
        
        # Check if the device exists
        if not os.path.exists(kvm_device):
            self.log('debug', f'KVM device {kvm_device} does not exist')
            return False
        
        # Check if we have read and write permissions
        if not os.access(kvm_device, os.R_OK | os.W_OK):
            self.log('debug', f'KVM device {kvm_device} exists but lacks read/write permissions')
            return False
        
        self.log('debug', f'KVM device {kvm_device} is available and accessible')
        return True
    
    def _get_qemu_command(self) -> list[str]:
        """Build QEMU command based on OS type and configuration."""
        # Check if VM image exists
        if not os.path.exists(self.vm_image_path):
            raise FileNotFoundError(
                f'VM image not found: {self.vm_image_path}. '
                f'Please ensure the QCOW2 image file exists.'
            )
        
        # Get the VM image filename (it will be mounted at /OS_images/ inside container)
        vm_image_filename = os.path.basename(self.vm_image_path)
        vm_image_container_path = f'/OS_images/{vm_image_filename}'
        
        # Check if KVM is available
        kvm_available = self._check_kvm_available()
        if kvm_available:
            self.log('info', 'KVM is available, enabling hardware acceleration')
        else:
            self.log('warning', 'KVM is not available, running QEMU in emulation mode (slower)')
        
        # Log snapshot mode (always enabled)
        self.log('info', 'Snapshot mode ENABLED: disk changes will NOT be saved (protects base image)')
        
        # Base QEMU command
        cmd = [
            'qemu-system-x86_64',
            '-bios', '/usr/share/ovmf/OVMF.fd',
            '-machine', 'q35',
        ]
        
        # Add KVM flags only if available
        if kvm_available:
            cmd.extend(['-cpu', 'host', '-enable-kvm'])
        else:
            # Use generic CPU for emulation mode
            cmd.extend(['-cpu', 'qemu64'])
        
        # Continue with common flags
        # Port forwarding: VM:5000->host:vm_server_port, VM:9222->host:chromium_port, VM:8080->host:vlc_port
        portfwd = (
            f'user,id=net0,'
            f'hostfwd=tcp::{self._vm_server_port}-:5000,'
            f'hostfwd=tcp::{self._chromium_port}-:9222,'
            f'hostfwd=tcp::{self._vlc_port}-:8080'
        )
        
        cmd.extend([
            '-m', '2G',
            '-smp', '2',
            '-drive', f'file={vm_image_container_path},if=ide',
            '-netdev', portfwd,
            '-device', 'virtio-net-pci,netdev=net0',
            '-snapshot',  # Always use snapshot mode (non-persistent)
        ])
        
        # Add VNC on the allocated port (not the default 5900)
        # QEMU VNC uses display numbers: port = 5900 + display_number
        # Example: if _vnc_port=19242, display=13342, QEMU listens on 5900+13342=19242
        vnc_display = self._vnc_port - 5900
        cmd.extend(['-vnc', f'0.0.0.0:{vnc_display}'])  # VNC accessible from any IP
        
        # Note: Don't use -daemonize, we manage the process with Popen
        
        return cmd
    
    def maybe_prepare_runtime_container_image(self):
        """Prepare the OSWorld runtime container image."""
        # Use simple template for OSWorld
        if self.runtime_container_image is None:
            if self.base_container_image is None:
                # Default to Ubuntu 24.04 for OSWorld
                self.base_container_image = 'ubuntu:24.04'
            
            self.send_status_message('STATUS$STARTING_CONTAINER')
            
            with SingularityRuntime._runtime_builder_lock:
                from openhands.runtime.utils.singularity_runtime_build import (
                    build_runtime_image_from_template,
                )
                
                # Build OSWorld-specific image
                template_path = os.path.join(
                    os.path.dirname(__file__),
                    '../../utils/runtime_templates/osworld_singularity.j2'
                )
                
                self.runtime_container_image = build_runtime_image_from_template(
                    base_image=self.base_container_image,
                    template_path=template_path,
                    runtime_builder=self.runtime_builder,
                    platform=self.config.sandbox.platform,
                    extra_deps=self.config.sandbox.runtime_extra_deps,
                    force_rebuild=self.config.sandbox.force_rebuild_runtime,
                )
        else:
            # Pull the image if it doesn't exist locally
            self._pull_image_if_needed()
    
    def init_container(self):
        """Initialize the Singularity container and start QEMU VM."""
        self.log('debug', 'Preparing to start OSWorld Singularity container with QEMU VM...')
        self.send_status_message('STATUS$PREPARING_CONTAINER')
        
        # Allocate ports for OSWorld services
        self._vm_server_port, self._vnc_port, self._chromium_port, self._vlc_port = self._allocate_osworld_ports()
        
        self.log(
            'info',
            f'Allocated OSWorld ports - '
            f'VM Server: {self._vm_server_port}, '
            f'VNC: {self._vnc_port}, '
            f'Chromium DevTools: {self._chromium_port}, '
            f'VLC: {self._vlc_port}'
        )
        
        # Get the image path
        image_path = self._get_singularity_image_path()
        if not os.path.exists(image_path):
            raise RuntimeError(f'Singularity image not found: {image_path}')
        
        # Prepare environment variables for QEMU
        env_vars = {
            'VM_SERVER_PORT': str(self._vm_server_port),
            'VNC_PORT': str(self._vnc_port),
            'CHROMIUM_PORT': str(self._chromium_port),
            'VLC_PORT': str(self._vlc_port),
            'OS_TYPE': self.os_type,
        }
        
        # Get absolute path to VM image
        vm_image_abs_path = os.path.abspath(self.vm_image_path)
        vm_image_dir = os.path.dirname(vm_image_abs_path)
        
        # Build the singularity exec command to run QEMU
        cmd = [
            'singularity', 'exec',
            '--pid',
            '--writable-tmpfs',
            '--no-mount', 'cwd,tmp',
            '--home', '/root',
        ]
        
        # Add fakeroot if configured
        if self.config.sandbox.run_as_fakeroot:
            cmd.extend(['--fakeroot'])
        
        # Add environment variables
        for key, value in env_vars.items():
            cmd.extend(['--env', f'{key}={value}'])
        
        # Mount VM image directory (needs to be writable for QEMU to maintain disk state)
        cmd.extend(['--bind', f'{vm_image_dir}:/OS_images'])
        
        # Add image path
        cmd.append(image_path)
        
        # Add QEMU command
        qemu_cmd = self._get_qemu_command()
        cmd.extend(qemu_cmd)
        
        self.log('info', f'Starting QEMU VM with command: {" ".join(cmd)}')
        
        try:
            # Create log directory for QEMU output
            log_dir = '/tmp/openhands_osworld_logs'
            os.makedirs(log_dir, exist_ok=True)
            qemu_stdout_path = os.path.join(log_dir, f'{self.sid}_qemu.out')
            qemu_stderr_path = os.path.join(log_dir, f'{self.sid}_qemu.err')
            
            # Start QEMU in the container (non-blocking with Popen)
            self._qemu_stdout = open(qemu_stdout_path, 'w')
            self._qemu_stderr = open(qemu_stderr_path, 'w')
            
            self.qemu_process = subprocess.Popen(
                cmd,
                stdout=self._qemu_stdout,
                stderr=self._qemu_stderr,
                text=True,
                start_new_session=True  # Create new process group for easier cleanup
            )
            
            # Save the QEMU PID
            self.qemu_pid = self.qemu_process.pid
            
            # Check if QEMU process started successfully
            time.sleep(2)  # Give QEMU a moment to start
            if self.qemu_process.poll() is not None:
                # Process failed to start
                self._qemu_stdout.close()
                self._qemu_stderr.close()
                with open(qemu_stderr_path, 'r') as f:
                    error_output = f.read()
                raise RuntimeError(
                    f'QEMU failed to start. Return code: {self.qemu_process.returncode}\n'
                    f'Error: {error_output}'
                )
            
            self.log('info', f'QEMU VM started with PID: {self.qemu_pid}')
            self.log('debug', f'QEMU logs: stdout={qemu_stdout_path}, stderr={qemu_stderr_path}')
            
            # Wait for VM to boot and OSWorld server to be ready
            self._wait_for_vm_ready()
            
            # Store session information
            session_info = {
                'vm_server_port': self._vm_server_port,
                'vnc_port': self._vnc_port,
                'chromium_port': self._chromium_port,
                'vlc_port': self._vlc_port,
                'os_type': self.os_type,
                'vm_image_path': self.vm_image_path,
                'qemu_pid': self.qemu_pid,
            }
            self._save_session_port_info(session_info)
            
            self.log('info', 'OSWorld VM is ready')
            self.log('info', f'VM Services:')
            self.log('info', f'  • OSWorld API: {self.osworld_vm_url}')
            self.log('info', f'  • VNC: {self.vnc_url} (display :{self._vnc_port - 5900})')
            self.log('info', f'  • Chrome DevTools: {self.chromium_devtools_url}')
            self.log('info', f'  • VLC Web Interface: {self.vlc_url}')
            self.send_status_message('STATUS$CONTAINER_STARTED')
            
        except Exception as e:
            self.log('error', f'Error starting OSWorld runtime: {str(e)}')
            self.close()
            raise e
    
    def _wait_for_vm_ready(self, timeout: int = 300):
        """Wait for the VM to boot and OSWorld server to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        self.log('info', 'Waiting for OSWorld VM to boot...')
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to connect to OSWorld server
                response = httpx.get(
                    #f'http://localhost:{self._vm_server_port}/screenshot',
                    f'{self.osworld_vm_url}/terminal',
                    timeout=5.0
                )
                if response.status_code == 200:
                    self.log('info', 'OSWorld VM server is ready!')
                    return
            except Exception:
                pass
            
            # Check every 5 seconds
            time.sleep(5)
            self.log('debug', f'Still waiting for VM... ({int(time.time() - start_time)}s elapsed)')
        
        raise TimeoutError(
            f'OSWorld VM failed to become ready within {timeout} seconds. '
            f'VM Server port: {self._vm_server_port}'
        )
    
    def _is_container_running(self) -> bool:
        """Check if the QEMU VM is currently running."""
        # For OSWorld, we check the QEMU process instead of container process
        if self.qemu_process is not None:
            return self.qemu_process.poll() is None
        
        # If we attached to an existing QEMU, check by PID
        if self.qemu_pid is not None:
            try:
                os.kill(self.qemu_pid, 0)  # Check if process exists
                return True
            except (OSError, ProcessLookupError):
                return False
        
        return False
    
    def wait_until_alive(self):
        """Wait until the OSWorld VM is ready."""
        # For OSWorld, we only need to check if QEMU is running
        # The VM readiness check is already done in init_container
        if not self._is_container_running():
            raise AgentRuntimeDisconnectedError(
                f'QEMU VM for {self.container_name} is not running.'
            )
        # OSWorld VM is already validated in _wait_for_vm_ready()
        self.log('debug', 'OSWorld runtime is alive and ready')
    
    def _attach_to_container(self):
        """Attach to an existing OSWorld container."""
        # Get port information from session registry
        session_info = self._load_session_port_info()
        if not session_info:
            raise AgentRuntimeNotFoundError(
                f'OSWorld container {self.container_name} not found or not running.'
            )
        
        self._vm_server_port = session_info.get('vm_server_port', -1)
        self._vnc_port = session_info.get('vnc_port', -1)
        self._chromium_port = session_info.get('chromium_port', -1)
        self._vlc_port = session_info.get('vlc_port', -1)
        self.os_type = session_info.get('os_type', 'linux')
        self.vm_image_path = session_info.get('vm_image_path', self._get_default_vm_image_path())
        self.qemu_pid = session_info.get('qemu_pid')
        
        self.log(
            'debug',
            f'Attached to OSWorld container: {self.container_name} '
            f'VM Server: {self._vm_server_port}, VNC: {self._vnc_port}, '
            f'Chromium: {self._chromium_port}, VLC: {self._vlc_port}, QEMU PID: {self.qemu_pid}'
        )
    
    def check_if_alive(self) -> None:
        """Check if the OSWorld VM server is alive."""
        try:
            response = httpx.get(
                f'http://localhost:{self._vm_server_port}/screenshot',
                timeout=5.0
            )
            if response.status_code != 200:
                raise AgentRuntimeDisconnectedError('OSWorld VM server is not responding')
        except Exception as e:
            raise AgentRuntimeDisconnectedError(
                f'OSWorld VM server is not reachable: {str(e)}'
            )
    
    def close(self, rm_all_containers: bool | None = None):
        """Close the OSWorld runtime and stop QEMU VM."""
        # Close QEMU log file handles
        try:
            if self._qemu_stdout is not None:
                self._qemu_stdout.close()
                self._qemu_stdout = None
        except Exception as e:
            logger.warning(f'Failed to close QEMU stdout: {e}')
        try:
            if self._qemu_stderr is not None:
                self._qemu_stderr.close()
                self._qemu_stderr = None
        except Exception as e:
            logger.warning(f'Failed to close QEMU stderr: {e}')
        
        # Stop QEMU process if we started it
        if self.qemu_pid is not None:
            try:
                self.log('info', f'Stopping QEMU VM with PID: {self.qemu_pid}')
                os.kill(self.qemu_pid, signal.SIGTERM)
                time.sleep(2)
                # Force kill if still alive
                try:
                    os.kill(self.qemu_pid, 0)
                    os.kill(self.qemu_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.qemu_pid = None
            except Exception as e:
                self.log('warning', f'Failed to stop QEMU VM: {e}')
        
        # Call parent close
        super().close(rm_all_containers)
    
    @property
    def osworld_vm_url(self) -> str:
        """Get the OSWorld VM server URL."""
        return f'http://localhost:{self._vm_server_port}'
    
    @property
    def vnc_url(self) -> str:
        """Get the VNC URL for the VM.
        
        QEMU's built-in VNC server listens on the allocated VNC port.
        Connect with: vncviewer localhost:{display} (where display = port - 5900)
        """
        return f'vnc://localhost:{self._vnc_port}'
    
    @property
    def chromium_devtools_url(self) -> str:
        """Get the Chrome DevTools Protocol URL.
        
        Access the Chromium browser's DevTools interface inside the VM.
        Example usage: Connect Puppeteer or Chrome DevTools to this endpoint.
        """
        return f'http://localhost:{self._chromium_port}'
    
    @property
    def vlc_url(self) -> str:
        """Get the VLC web interface URL.
        
        Access the VLC media player's web interface inside the VM.
        """
        return f'http://localhost:{self._vlc_port}'
    
    def get_vm_screenshot(self) -> bytes | None:
        """Get screenshot from the VM.
        
        Returns:
            PNG image bytes or None if failed
        """
        try:
            response = httpx.get(
                f'{self.osworld_vm_url}/screenshot',
                timeout=30.0
            )
            if response.status_code == 200:
                return response.content
            return None
        except Exception as e:
            self.log('error', f'Failed to get VM screenshot: {e}')
            return None

    def get_vm_accessibility_tree(self) -> str | None:
        """Get accessibility tree from the VM.
        
        Returns:
            Accessibility tree string or None if failed
        """
        try:
            response = httpx.get(
                f'{self.osworld_vm_url}/accessibility',
                timeout=30.0
            )
            if response.status_code == 200:
                at = response.json().get('AT', '')
                return at
            return None
        except Exception as e:
            self.log('error', f'Failed to get VM screenshot: {e}')
            return None
    
    def _execute_pyautogui_command(self, pyautogui_command: str) -> dict:
        """Execute a PyAutoGUI command string in the VM.
        
        Args:
            pyautogui_command: Raw PyAutoGUI command(s) to execute
            
        Returns:
            Response dictionary from OSWorld server
        """
        try:
            # Wrap the command with necessary imports
            command = f"import pyautogui; import time; pyautogui.FAILSAFE = False; {pyautogui_command}"
            command_list = ["python3", "-c", command]
            payload = {"command": command_list, "shell": False}
            
            response = httpx.post(
                f'{self.osworld_vm_url}/execute',
                json=payload,
                timeout=30.0
            )
            return response.json()
        except Exception as e:
            self.log('error', f'Failed to execute PyAutoGUI command: {e}')
            return {'status': 'error', 'message': str(e)}
    
    def execute_vm_action(self, action_data: dict) -> dict:
        """Execute an action in the OSWorld VM.
        
        Args:
            action_data: Action data dictionary with 'action_type' and 'parameters'
            
        Returns:
            Response from OSWorld server
        """
        try:
            action_type = action_data.get('action_type')
            parameters = action_data.get('parameters', {})
            
            # Convert action to PyAutoGUI command
            pyautogui_command = self._action_to_pyautogui_command(action_type, parameters)
            
            if pyautogui_command is None:
                return {'status': 'error', 'message': f'Unknown action type: {action_type}'}
            
            # Execute using the common method
            return self._execute_pyautogui_command(pyautogui_command)
        except Exception as e:
            self.log('error', f'Failed to execute VM action: {e}')
            return {'status': 'error', 'message': str(e)}
    
    def _action_to_pyautogui_command(self, action_type: str, parameters: dict) -> str | None:
        """Convert an action dictionary to a PyAutoGUI command string.
        
        Args:
            action_type: Type of action (e.g., 'CLICK', 'TYPING', 'PRESS')
            parameters: Action parameters
            
        Returns:
            PyAutoGUI command string or None if unknown action type
        """
        import random
        
        # For MOVE_TO actions with duration
        move_mode = random.choice([
            "pyautogui.easeInQuad", "pyautogui.easeOutQuad", "pyautogui.easeInOutQuad",
            "pyautogui.easeInBounce", "pyautogui.easeInElastic"
        ])
        
        if action_type == "CLICK":
            x = parameters.get('x')
            y = parameters.get('y')
            button = parameters.get('button', 'left')
            num_clicks = parameters.get('clicks', 1)
            interval = parameters.get('interval', 0.0)
            duration = parameters.get('duration', 0.0)
            
            if x is not None and y is not None:
                return f"pyautogui.click(x={x}, y={y}, button='{button}', clicks={num_clicks}, interval={interval}, duration={duration})"
            else:
                return "pyautogui.click()"
        
        elif action_type == "DOUBLE_CLICK":
            x = parameters.get('x')
            y = parameters.get('y')
            button = parameters.get('button', 'left')
            interval = parameters.get('interval', 0.0)
            duration = parameters.get('duration', 0.0)
            if x is not None and y is not None:
                return f"pyautogui.doubleClick(x={x}, y={y}, button='{button}', interval={interval}, duration={duration})"
            else:
                return "pyautogui.doubleClick()"
        
        elif action_type == "TRIPLE_CLICK":
            x = parameters.get('x')
            y = parameters.get('y')
            button = parameters.get('button', 'left')
            interval = parameters.get('interval', 0.0)
            duration = parameters.get('duration', 0.0)
            if x is not None and y is not None:
                return f"pyautogui.tripleClick(x={x}, y={y}, button='{button}', interval={interval}, duration={duration})"
            else:
                return "pyautogui.tripleClick()"
        
        elif action_type == "RIGHT_CLICK":
            x = parameters.get('x')
            y = parameters.get('y')
            interval = parameters.get('interval', 0.0)
            duration = parameters.get('duration', 0.0)
            if x is not None and y is not None:
                return f"pyautogui.rightClick(x={x}, y={y}, interval={interval}, duration={duration})"
            else:
                return "pyautogui.rightClick()"
        
        elif action_type == "MIDDLE_CLICK":
            x = parameters.get('x')
            y = parameters.get('y')
            interval = parameters.get('interval', 0.0)
            duration = parameters.get('duration', 0.0)
            if x is not None and y is not None:
                return f"pyautogui.middleClick(x={x}, y={y}, interval={interval}, duration={duration})"
            else:
                return "pyautogui.click(button='middle')"
        
        elif action_type == "MOVE_TO":
            x = parameters.get('x')
            y = parameters.get('y')
            duration = parameters.get('duration', 0.0)
            if x is not None and y is not None:
                return f"pyautogui.moveTo({x}, {y}, {duration}, {move_mode})"
            else:
                return "pyautogui.moveTo()"
        
        elif action_type == "DRAG_TO":
            x = parameters.get('x')
            y = parameters.get('y')
            duration = parameters.get('duration', 0.0)
            button = parameters.get('button', 'left')
            mouseDownUp = parameters.get('mouseDownUp', True)
            if x is not None and y is not None:
                return f"pyautogui.dragTo({x}, {y}, button='{button}', duration={duration}, mouseDownUp={mouseDownUp})"
            return None
        
        elif action_type == "SCROLL":
            x = parameters.get('x', None)
            y = parameters.get('y', None)
            amount = parameters.get('amount', 1)
            return f"pyautogui.scroll({amount}, x={x}, y={y})"

        elif action_type == "HSCROLL":
            x = parameters.get('x', None)
            y = parameters.get('y', None)
            amount = parameters.get('amount', 1)
            return f"pyautogui.hscroll({amount}, x={x}, y={y})"
        
        elif action_type == "TYPING":
            text = parameters.get('text', '')
            interval = parameters.get('interval', 0.0)
            # Use repr() to properly escape the text (same as OSWorld)
            return f"pyautogui.typewrite({repr(text)}, interval={interval})"
        
        elif action_type == "PRESS":
            key = parameters.get('key', '')
            if isinstance(key, list):
                # Multiple keys - treat as hotkey
                keys_str = "', '".join(key)
                return f"pyautogui.hotkey('{keys_str}')"
            else:
                return f"pyautogui.press('{key}')"
        
        elif action_type == "HOTKEY":
            keys = parameters.get('keys', [])
            if isinstance(keys, list) and keys:
                keys_str = "', '".join(keys)
                return f"pyautogui.hotkey('{keys_str}')"
            return None
        
        elif action_type == "KEY_DOWN":
            key = parameters.get('key', '')
            return f"pyautogui.keyDown('{key}')"
        
        elif action_type == "KEY_UP":
            key = parameters.get('key', '')
            return f"pyautogui.keyUp('{key}')"
        
        elif action_type == "MOUSE_DOWN":
            button = parameters.get('button', 'left')
            return f"pyautogui.mouseDown(button='{button}')"
        
        elif action_type == "MOUSE_UP":
            button = parameters.get('button', 'left')
            return f"pyautogui.mouseUp(button='{button}')"

        elif action_type == "WAIT":
            seconds = parameters.get('seconds', 1)
            return f"time.sleep({seconds})"
        
        else:
            return None
    
    def run_action(self, action):
        """Run an action in the OSWorld VM.
        
        This method is called by the runtime system to execute actions.
        For OSWorldInteractiveAction, it converts the actions to PyAutoGUI commands.
        """
        from openhands.events.action.os import OSWorldInteractiveAction
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        
        if isinstance(action, OSWorldInteractiveAction):
            return self.osworld_interactive(action)
        else:
            # Fallback to parent implementation for other actions
            return super().run_action(action)
    
    def osworld_interactive(self, action) -> 'Observation':
        """Handle OSWorld interactive actions.
        
        Dispatches to appropriate handler based on action.method.
        Supports all PythonController methods from OSWorld.
        
        Args:
            action: OSWorldInteractiveAction with method and params
            
        Returns:
            Appropriate Observation based on the method
        """
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        
        try:
            method = action.method
            params = action.params or {}
            
            # Dispatch to appropriate handler
            if method == 'execute_action':
                return self._handle_execute_action(params)
            elif method == 'execute_agentic_action':
                return self._handle_execute_agentic_action(params, action.tool_call_metadata)
            elif method == 'get_screenshot':
                return self._handle_get_screenshot()
            elif method == 'get_accessibility_tree':
                return self._handle_get_accessibility_tree()
            elif method == 'get_terminal_output':
                return self._handle_get_terminal_output()
            elif method == 'get_file':
                return self._handle_get_file(params)
            elif method == 'execute_python_command':
                return self._handle_execute_python_command(params)
            elif method == 'run_python_script':
                return self._handle_run_python_script(params)
            elif method == 'run_bash_script':
                return self._handle_run_bash_script(params)
            elif method == 'start_recording':
                return self._handle_start_recording()
            elif method == 'end_recording':
                return self._handle_end_recording(params)
            elif method == 'get_vm_platform':
                return self._handle_get_vm_platform()
            elif method == 'get_vm_screen_size':
                return self._handle_get_vm_screen_size()
            elif method == 'get_vm_window_size':
                return self._handle_get_vm_window_size(params)
            elif method == 'get_vm_wallpaper':
                return self._handle_get_vm_wallpaper()
            elif method == 'get_vm_desktop_path':
                return self._handle_get_vm_desktop_path()
            elif method == 'get_vm_directory_tree':
                return self._handle_get_vm_directory_tree(params)
            else:
                return ErrorObservation(f'Unknown OSWorld method: {method}')
                
        except Exception as e:
            self.log('error', f'Failed to execute OSWorld interactive action: {e}')
            return ErrorObservation(f'Failed to execute OSWorld action: {str(e)}')
    
    # Handler methods for each PythonController method
    
    def _handle_execute_action(self, params: dict) -> 'Observation':
        """Handle execute_action - PyAutoGUI actions like CLICK, TYPING, etc."""
        from openhands.events.observation import CmdOutputObservation
        
        action_data = params.get('action', params)
        result = self.execute_vm_action(action_data)
        
        if result.get('status') == 'success':
            return CmdOutputObservation(
                content=result.get('output', 'Action executed successfully'),
                command=str(action_data),
                exit_code=0,
            )
        else:
            error_msg = result.get('error', result.get('message', 'Unknown error'))
            return CmdOutputObservation(
                content=f"Error: {error_msg}",
                command=str(action_data),
                exit_code=1,
            )

    def _handle_execute_agentic_action(self, params: dict, tool_call_metadata: ToolCallMetadata | None) -> 'Observation':
        """Handle execute_action - PyAutoGUI actions like CLICK, TYPING, etc."""
        from openhands.events.observation.osworld import OSWorldOutputObservation  
        from openhands.events.observation import ErrorObservation   
        import base64

        # Always save screenshot and accessibility tree. Will leave message formatting to the agent.
        include_screenshot = True #self.config.agents['agent'].enable_vision
        include_a11y_tree = True #self.config.agents['agent'].enable_a11y_tree
        
        action_data = params.get('action', params)

        # Convert normalized coordinates to pixel coordinates
        if not hasattr(self, 'screen_size'):
            self._handle_get_vm_screen_size()
        width, height = self.screen_size
        if 'parameters' in action_data:
            parameters = action_data['parameters']
            if 'x' in parameters:
                parameters['x'] = int(parameters['x'] * width)
            if 'y' in parameters:
                parameters['y'] = int(parameters['y'] * height)
            logger.info(f"Converted normalized coordinates to pixel coordinates: {action_data}. Screen size: {width}x{height}.")

        result = self.execute_vm_action(action_data)
        
        if result.get('status') == 'success':
            if include_screenshot:
                screenshot_bytes = self.get_vm_screenshot()
                screenshot_bytes = base64.b64encode(screenshot_bytes).decode('utf-8')
            else:
                screenshot_bytes = None

            if include_a11y_tree:
                accessibility_tree = self.get_vm_accessibility_tree()
                #accessibility_tree = linearize_accessibility_tree(accessibility_tree)
            else:
                accessibility_tree = None

            return OSWorldOutputObservation(
                command=str(action_data),
                screenshot=screenshot_bytes,
                accessibility_tree=accessibility_tree,
                tool_call_id=tool_call_metadata.tool_call_id,
                name=tool_call_metadata.function_name,
            )
        else:
            error_msg = result.get('error', result.get('message', 'Unknown error'))
            logger.error(f"Error in agentic action: action_data={action_data}, error={error_msg}")
            return ErrorObservation(
                content=f"Error: {error_msg}",
                command=str(action_data),
                error_id=tool_call_metadata.tool_call_id,
                name=tool_call_metadata.function_name,
            )
    
    def _handle_get_screenshot(self) -> 'Observation':
        """Handle get_screenshot - returns screenshot as base64."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        import base64
        
        screenshot_bytes = self.get_vm_screenshot()
        if screenshot_bytes:
            # Return as base64 encoded string
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            return CmdOutputObservation(
                content=f"Screenshot captured ({len(screenshot_bytes)} bytes)",
                command='get_screenshot',
                exit_code=0,
            )
        else:
            return ErrorObservation('Failed to capture screenshot')
    
    def _handle_get_accessibility_tree(self) -> 'Observation':
        """Handle get_accessibility_tree."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        
        at = self.get_vm_accessibility_tree()
        if at:
            return CmdOutputObservation(
                content=at,
                command='get_accessibility_tree',
                exit_code=0,
            )
        return ErrorObservation('Failed to get accessibility tree')
    
    def _handle_get_terminal_output(self) -> 'Observation':
        """Handle get_terminal_output."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        
        try:
            response = httpx.get(
                f'{self.osworld_vm_url}/terminal',
                timeout=10.0
            )
            if response.status_code == 200:
                output = response.json().get('output', '')
                return CmdOutputObservation(
                    content=output,
                    command='get_terminal_output',
                    exit_code=0,
                )
            return ErrorObservation(f'Failed to get terminal output: {response.status_code}')
        except Exception as e:
            return ErrorObservation(f'Failed to get terminal output: {e}')
    
    def _handle_get_file(self, params: dict) -> 'Observation':
        """Handle get_file - downloads file from VM.
        
        Returns the full file content as base64-encoded string in observation.content.
        Format: "base64:<base64_data>"
        """
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        import base64
        
        file_path = params.get('file_path', '')
        if not file_path:
            return ErrorObservation('file_path parameter required')
        
        try:
            response = httpx.post(
                f'{self.osworld_vm_url}/file',
                data={'file_path': file_path},
                timeout=30.0
            )
            if response.status_code == 200:
                file_content = response.content
                # Return full base64 with prefix for easy parsing
                content_b64 = base64.b64encode(file_content).decode('utf-8')
                return CmdOutputObservation(
                    content=f"base64:{content_b64}",
                    command=f'get_file {file_path}',
                    exit_code=0,
                )
            return ErrorObservation(f'Failed to get file: {response.status_code}')
        except Exception as e:
            return ErrorObservation(f'Failed to get file: {e}')
    
    def _handle_execute_python_command(self, params: dict) -> 'Observation':
        """Handle execute_python_command - raw Python command execution."""
        from openhands.events.observation import CmdOutputObservation
        
        command = params.get('command', '')
        if not command:
            return CmdOutputObservation(
                content='Error: command parameter required',
                command='execute_python_command',
                exit_code=1,
            )
        
        result = self._execute_pyautogui_command(command)
        
        if result.get('status') == 'success':
            return CmdOutputObservation(
                content=result.get('output', ''),
                command=command,
                exit_code=result.get('returncode', 0),
            )
        else:
            error_msg = result.get('error', result.get('message', 'Unknown error'))
            return CmdOutputObservation(
                content=f"Error: {error_msg}",
                command=command,
                exit_code=result.get('returncode', 1),
            )
    
    def _handle_run_python_script(self, params: dict) -> 'Observation':
        """Handle run_python_script."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        
        script = params.get('script', '')
        if not script:
            return ErrorObservation('script parameter required')
        
        try:
            payload = {'code': script}
            response = httpx.post(
                f'{self.osworld_vm_url}/run_python',
                json=payload,
                timeout=90.0
            )
            if response.status_code == 200:
                result = response.json()
                output = result.get('output', '')
                error = result.get('error', '')
                content = f"Output:\n{output}"
                if error:
                    content += f"\nError:\n{error}"
                return CmdOutputObservation(
                    content=content,
                    command='run_python_script',
                    exit_code=result.get('returncode', 0),
                )
            # Try to get error details from response
            try:
                error_detail = response.json()
                error_msg = error_detail.get('output', error_detail.get('message', 'Unknown error'))
            except:
                error_msg = response.text or 'Unknown error'
            return ErrorObservation(f'Failed to run Python script (HTTP {response.status_code}): {error_msg}')
        except Exception as e:
            return ErrorObservation(f'Failed to run Python script: {e}')
    
    def _handle_run_bash_script(self, params: dict) -> 'Observation':
        """Handle run_bash_script.
        
        Note: The /run_bash_script endpoint has a bug (missing _append_event function).
        As a workaround, we use /execute with base64 encoding to safely transfer scripts.
        """
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        import base64
        import uuid
        
        script = params.get('script', '')
        if not script:
            return ErrorObservation('script parameter required')
        
        timeout = params.get('timeout', 30)
        working_dir = params.get('working_dir')
        
        try:
            # Workaround: Use /execute endpoint instead of /run_bash_script
            # Encode script as base64 to avoid escaping issues
            script_name = f'/tmp/bash_script_{uuid.uuid4().hex}.sh'
            
            # Add shebang if not present
            if '#!/bin/bash' not in script:
                script = '#!/bin/bash\n\n' + script
            
            # Base64 encode the script for safe transfer
            script_b64 = base64.b64encode(script.encode('utf-8')).decode('ascii')
            
            # Build command to decode, write, and execute the script
            commands = [
                f'echo "{script_b64}" | base64 -d > {script_name}',
                f'chmod +x {script_name}',
            ]
            
            # If working_dir is specified, cd to it before executing
            if working_dir:
                commands.append(f'cd {working_dir}')
            
            # Execute and capture exit code
            commands.extend([
                f'{script_name}',
                f'SCRIPT_EXIT_CODE=$?',
                f'rm -f {script_name}',
                f'exit $SCRIPT_EXIT_CODE'
            ])
            
            bash_command = ' && '.join(commands)
            
            # Use /execute endpoint with shell=True
            payload = {
                'command': bash_command,
                'shell': True
            }
            
            response = httpx.post(
                f'{self.osworld_vm_url}/execute',
                json=payload,
                timeout=timeout + 10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                output = result.get('output', '')
                error = result.get('error', '')
                
                # Combine output and error
                content = output
                if error:
                    content += f'\n{error}' if content else error
                
                return CmdOutputObservation(
                    content=content,
                    command='run_bash_script',
                    exit_code=result.get('returncode', 0),
                )
            
            # Try to get error details from response
            try:
                error_detail = response.json()
                error_msg = error_detail.get('output', error_detail.get('message', 'Unknown error'))
            except:
                error_msg = response.text or 'Unknown error'
            return ErrorObservation(f'Failed to run bash script (HTTP {response.status_code}): {error_msg}')
        
        except Exception as e:
            return ErrorObservation(f'Failed to run bash script: {e}')
    
    def _handle_start_recording(self) -> 'Observation':
        """Handle start_recording."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        
        try:
            response = httpx.post(
                f'{self.osworld_vm_url}/start_recording',
                timeout=10.0
            )
            if response.status_code == 200:
                return CmdOutputObservation(
                    content='Recording started',
                    command='start_recording',
                    exit_code=0,
                )
            return ErrorObservation(f'Failed to start recording: {response.status_code}')
        except Exception as e:
            return ErrorObservation(f'Failed to start recording: {e}')
    
    def _handle_end_recording(self, params: dict) -> 'Observation':
        """Handle end_recording.
        
        Note: The /end_recording endpoint returns the actual recording video file (binary).
        The 'dest' parameter is IGNORED by the OSWorld server - the recording is always
        saved to /tmp/recording.mp4 inside the VM. We return the video as base64-encoded data.
        """
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        import base64
        
        dest = params.get('dest', '/tmp/recording.mp4')  # Ignored by server, kept for documentation
        
        try:
            response = httpx.post(
                f'{self.osworld_vm_url}/end_recording',
                timeout=60.0
            )
            if response.status_code == 200:
                video_content = response.content
                # Return as base64 for consistency with get_file
                content_b64 = base64.b64encode(video_content).decode('utf-8')
                return CmdOutputObservation(
                    content=f'base64:{content_b64}',
                    command=f'end_recording',
                    exit_code=0,
                )
            return ErrorObservation(f'Failed to end recording: {response.status_code}')
        except Exception as e:
            return ErrorObservation(f'Failed to end recording: {e}')
    
    def _handle_get_vm_platform(self) -> 'Observation':
        """Handle get_vm_platform."""
        from openhands.events.observation import CmdOutputObservation
        
        command = "import platform; print(platform.system())"
        result = self._execute_pyautogui_command(command)
        
        if result.get('status') == 'success':
            platform = result.get('output', '').strip()
            return CmdOutputObservation(
                content=platform,
                command='get_vm_platform',
                exit_code=0,
            )
        else:
            return CmdOutputObservation(
                content='Unknown',
                command='get_vm_platform',
                exit_code=1,
            )
    
    def _handle_get_vm_screen_size(self) -> 'Observation':
        """Handle get_vm_screen_size."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        
        try:
            if hasattr(self, 'screen_size'):
                width, height = self.screen_size
            else:
                response = httpx.post(
                    f'{self.osworld_vm_url}/screen_size',
                    timeout=10.0
                )
                if response.status_code == 200:
                    size = response.json()
                    width, height = size.get('width', 1920), size.get('height', 1080)
                    self.screen_size = (width, height)
            content = f"Width: {width}, Height: {height}"
            return CmdOutputObservation(
                content=content,
                command='get_vm_screen_size',
                exit_code=0,
            )
            return ErrorObservation(f'Failed to get screen size: {response.status_code}')
        except Exception as e:
            return ErrorObservation(f'Failed to get screen size: {e}')
    
    def _handle_get_vm_window_size(self, params: dict) -> 'Observation':
        """Handle get_vm_window_size."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        
        app_class_name = params.get('app_class_name', '')
        if not app_class_name:
            return ErrorObservation('app_class_name parameter required')
        
        try:
            response = httpx.post(
                f'{self.osworld_vm_url}/window_size',
                data={'app_class_name': app_class_name},
                timeout=10.0
            )
            if response.status_code == 200:
                size = response.json()
                content = f"Width: {size.get('width', 'unknown')}, Height: {size.get('height', 'unknown')}"
                return CmdOutputObservation(
                    content=content,
                    command=f'get_vm_window_size {app_class_name}',
                    exit_code=0,
                )
            return ErrorObservation(f'Failed to get window size: {response.status_code}')
        except Exception as e:
            return ErrorObservation(f'Failed to get window size: {e}')
    
    def _handle_get_vm_wallpaper(self) -> 'Observation':
        """Handle get_vm_wallpaper.
        
        Note: The /wallpaper endpoint returns the actual wallpaper image file (binary),
        not the path. We return it as base64-encoded data.
        """
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        import base64
        
        try:
            response = httpx.post(
                f'{self.osworld_vm_url}/wallpaper',
                timeout=30.0
            )
            if response.status_code == 200:
                wallpaper_bytes = response.content
                # Return as base64 for consistency with get_file
                content_b64 = base64.b64encode(wallpaper_bytes).decode('utf-8')
                return CmdOutputObservation(
                    content=f'base64:{content_b64}',
                    command='get_vm_wallpaper',
                    exit_code=0,
                )
            return ErrorObservation(f'Failed to get wallpaper: {response.status_code}')
        except Exception as e:
            return ErrorObservation(f'Failed to get wallpaper: {e}')
    
    def _handle_get_vm_desktop_path(self) -> 'Observation':
        """Handle get_vm_desktop_path."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        
        try:
            response = httpx.post(
                f'{self.osworld_vm_url}/desktop_path',
                timeout=10.0
            )
            if response.status_code == 200:
                desktop_path = response.json().get('desktop_path', '')
                return CmdOutputObservation(
                    content=desktop_path,
                    command='get_vm_desktop_path',
                    exit_code=0,
                )
            return ErrorObservation(f'Failed to get desktop path: {response.status_code}')
        except Exception as e:
            return ErrorObservation(f'Failed to get desktop path: {e}')
    
    def _handle_get_vm_directory_tree(self, params: dict) -> 'Observation':
        """Handle get_vm_directory_tree."""
        from openhands.events.observation import CmdOutputObservation, ErrorObservation
        import json
        
        path = params.get('path', '')
        if not path:
            return ErrorObservation('path parameter required')
        
        try:
            payload = {'path': path}
            response = httpx.post(
                f'{self.osworld_vm_url}/list_directory',
                json=payload,
                timeout=30.0
            )
            if response.status_code == 200:
                directory_tree = response.json().get('directory_tree', {})
                content = json.dumps(directory_tree, indent=2)
                return CmdOutputObservation(
                    content=content,
                    command=f'get_vm_directory_tree {path}',
                    exit_code=0,
                )
            return ErrorObservation(f'Failed to get directory tree: {response.status_code}')
        except Exception as e:
            return ErrorObservation(f'Failed to get directory tree: {e}')

    def get_microagents_from_selected_repo(
        self, selected_repository: str | None
    ):
        return []