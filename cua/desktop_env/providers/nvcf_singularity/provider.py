import logging
import os
import random
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import requests

from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.nvcf_singularity.NVCFSingularityProvider")
logger.setLevel(logging.INFO)

RETRY_INTERVAL = 10
WAIT_TIME = 3
DEFAULT_SIF_PATH = "/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/nvcf-osworld-eval/osworld-linux.sif"
# Port ranges for parallel instance support (matches singularity provider)
API_PORT_RANGE = (15000, 19999)
VNC_PORT_RANGE = (18000, 22999)
CHROME_PORT_RANGE = (19000, 22999)
VLC_PORT_RANGE = (20000, 22999)


class PortAllocationError(Exception):
    pass


class NVCFSingularityProvider(Provider):
    """
    Singularity-based provider that runs a prebuilt NVCF image directly.
    Unlike the default singularity provider, it does not use qcow2 VM images.
    Equivalent to NVCFDummyProvider but uses Singularity instead of Docker.

    Supports running multiple instances in parallel via dynamic port allocation.
    Each instance gets unique ports for API, VNC, Chrome DevTools, and VLC,
    passed to the container via environment variables.
    """

    _port_allocation_lock = threading.Lock()

    def __init__(self, region: str = None):
        super().__init__(region)
        self.process: subprocess.Popen = None
        self.process_pid: int = None
        self._stdout_fh = None
        self._stderr_fh = None

        self.server_port = None
        self.chromium_port = None
        self.vnc_port = None
        self.vlc_port = None

        self._check_singularity_availability()

    @staticmethod
    def _check_singularity_availability():
        try:
            subprocess.run(
                ['singularity', '--version'],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                'Singularity/Apptainer is not available. '
                'Please install it to use NVCFSingularityProvider.'
            ) from e

    @staticmethod
    def _check_port_available(port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _find_available_port(self, min_port: int, max_port: int, max_attempts: int = 50) -> int:
        rng = random.SystemRandom()
        ports = list(range(min_port, max_port + 1))
        rng.shuffle(ports)

        for port in ports[:max_attempts]:
            if self._check_port_available(port):
                return port
        raise PortAllocationError(f"No available ports found in range {min_port}-{max_port}")

    def _allocate_ports(self) -> tuple:
        """Allocate unique ports for API, VNC, Chrome, and VLC."""
        with NVCFSingularityProvider._port_allocation_lock:
            api_port = self._find_available_port(*API_PORT_RANGE)
            vnc_port = self._find_available_port(*VNC_PORT_RANGE)
            chrome_port = self._find_available_port(*CHROME_PORT_RANGE)
            vlc_port = self._find_available_port(*VLC_PORT_RANGE)
            return api_port, vnc_port, chrome_port, vlc_port

    def _wait_for_vm_ready(self, timeout: int = 300):
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"http://localhost:{self.server_port}/screenshot",
                    timeout=(10, 10),
                )
                if response.status_code == 200:
                    return True
            except Exception:
                pass

            if self.process and self.process.poll() is not None:
                self._read_and_raise_error()

            logger.info("Checking if nvcf_singularity container is ready...")
            time.sleep(RETRY_INTERVAL)

        raise TimeoutError("nvcf_singularity failed to become ready within timeout period")

    def _read_and_raise_error(self):
        self._close_log_handles()
        log_dir = Path("/tmp/osworld_nvcf_singularity_logs")
        err_files = sorted(log_dir.glob("singularity_*.err"), reverse=True)
        error_output = ""
        if err_files:
            error_output = err_files[0].read_text()
        raise RuntimeError(
            f"Singularity container exited unexpectedly. "
            f"Return code: {self.process.returncode}\nError: {error_output}"
        )

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"):
        del path_to_vm, headless, os_type

        sif_path = os.environ.get("NVCF_SINGULARITY_SIF_PATH", DEFAULT_SIF_PATH)
        if not os.path.exists(sif_path):
            raise FileNotFoundError(
                f"Singularity image not found: {sif_path}. "
                f"Please build or download the .sif image first."
            )

        try:
            self.server_port, self.vnc_port, self.chromium_port, self.vlc_port = self._allocate_ports()

            logger.info(
                "Allocated ports - API: %d, VNC: %d, Chrome: %d, VLC: %d",
                self.server_port, self.vnc_port, self.chromium_port, self.vlc_port,
            )

            cmd = [
                'singularity', 'run',
                '--contain',
                '--cleanenv',
                '--pid',
                '--writable-tmpfs',
                '--no-mount', 'home,cwd,tmp',
                '--home', '/home/user',
                '--env', f'API_PORT={self.server_port}',
                '--env', f'VNC_PORT={self.vnc_port}',
                '--env', f'CHROME_PORT={self.chromium_port}',
                '--env', f'VLC_PORT={self.vlc_port}',
                sif_path,
            ]

            log_dir = Path("/tmp/osworld_nvcf_singularity_logs")
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time())
            stdout_path = log_dir / f"singularity_{timestamp}.out"
            stderr_path = log_dir / f"singularity_{timestamp}.err"

            self._stdout_fh = open(stdout_path, 'w')
            self._stderr_fh = open(stderr_path, 'w')

            self.process = subprocess.Popen(
                cmd,
                stdout=self._stdout_fh,
                stderr=self._stderr_fh,
                text=True,
                start_new_session=True,
            )
            self.process_pid = self.process.pid

            time.sleep(2)
            if self.process.poll() is not None:
                self._close_log_handles()
                with open(stderr_path, 'r') as f:
                    error_output = f.read()
                raise RuntimeError(
                    f"Singularity container failed to start. "
                    f"Return code: {self.process.returncode}\nError: {error_output}"
                )

            logger.info("Singularity process started with PID: %d", self.process_pid)
            logger.info("Logs: stdout=%s, stderr=%s", stdout_path, stderr_path)

            logger.info(
                "Started nvcf_singularity with image '%s' "
                "(api=%d, vnc=%d, chrome=%d, vlc=%d)",
                sif_path,
                self.server_port, self.vnc_port, self.chromium_port, self.vlc_port,
            )
            self._wait_for_vm_ready()
        except Exception:
            self.stop_emulator("")
            raise

    def _close_log_handles(self):
        for fh in (self._stdout_fh, self._stderr_fh):
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass
        self._stdout_fh = None
        self._stderr_fh = None

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("Container not started - ports not allocated")
        return f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for nvcf_singularity provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        del path_to_vm, region, args, kwargs

        try:
            self._close_log_handles()

            # With --containall (which includes --pid), the container runs in
            # its own PID namespace.  Killing the Singularity CLI process
            # tears down the namespace and the kernel reaps every container
            # process automatically — no manual tree-walk needed.
            if self.process_pid is not None:
                logger.info("Stopping Singularity process (PID: %d)", self.process_pid)
                try:
                    os.kill(self.process_pid, signal.SIGTERM)
                    time.sleep(2)
                    try:
                        os.kill(self.process_pid, 0)
                        os.kill(self.process_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logger.warning("Failed to kill Singularity process: %s", e)

            time.sleep(WAIT_TIME)
        except Exception as e:
            logger.error("Error stopping nvcf_singularity container: %s", e)
        finally:
            self.process = None
            self.process_pid = None
            self.server_port = None
            self.chromium_port = None
            self.vnc_port = None
            self.vlc_port = None
