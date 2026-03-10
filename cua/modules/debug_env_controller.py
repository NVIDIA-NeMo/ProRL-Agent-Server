import logging
import os
import sys
import re
import uuid
import time
import threading
import requests
from typing import Dict, List, Optional, Tuple

# Ensure OSWorld is importable
_osworld_path = "/lustre/fsw/portfolios/nvr/users/bcui/OSWorld"
if _osworld_path not in sys.path:
    sys.path.insert(0, _osworld_path)

from desktop_env.desktop_env import DesktopEnv
from openhands.core.logger import openhands_logger

# Create a child logger
logger = openhands_logger.getChild('env_controller')
logger.setLevel(logging.DEBUG)

# Semaphore to limit concurrent downloads (shared with setup.py via env var)
_DOWNLOAD_SEMAPHORE = threading.Semaphore(int(os.environ.get('OSWORLD_MAX_CONCURRENT_DOWNLOADS', '3')))


class EnvController:
    """
    Static wrapper class that interfaces with OSWorld's DesktopEnv.
    Replaces the previous OpenHands runtime-based approach with OSWorld's
    native DesktopEnv + NVCFProvider for NVCF deployments.
    """

    @staticmethod
    def pre_download_setup_files(osworld_setup: Dict, cache_dir: str = "/tmp/osworld_cache") -> bool:
        """
        Pre-download all setup files to local cache BEFORE deploying NVCF.
        This avoids wasting NVCF resources if downloads fail (e.g., HF 429 errors).

        Returns True if all downloads succeeded, False otherwise.
        """
        config_list = osworld_setup.get("config", [])
        if not config_list:
            return True

        os.makedirs(cache_dir, exist_ok=True)

        # Build headers with HF token if available
        dl_headers = {}
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if hf_token:
            dl_headers['Authorization'] = f'Bearer {hf_token}'

        for cfg in config_list:
            if cfg.get("type") != "download":
                continue

            files = cfg.get("parameters", {}).get("files", [])
            for f in files:
                url = f.get("url", "")
                path = f.get("path", "")
                if not url or not path:
                    continue

                cache_path = os.path.join(cache_dir, "{:}_{:}".format(
                    uuid.uuid5(uuid.NAMESPACE_URL, url),
                    os.path.basename(path)))

                if os.path.exists(cache_path):
                    logger.info(f"[pre_download] Cache hit: {cache_path}")
                    continue

                logger.info(f"[pre_download] Downloading {url} to cache...")
                max_retries = 8
                downloaded = False
                last_error = None

                with _DOWNLOAD_SEMAPHORE:
                    for i in range(max_retries):
                        try:
                            backoff = min(2 ** i + 1, 60)
                            if i > 0:
                                logger.info(f"[pre_download] Waiting {backoff}s before retry {i+1}/{max_retries}")
                                time.sleep(backoff)

                            response = requests.get(url, stream=True, timeout=300, headers=dl_headers)
                            response.raise_for_status()

                            downloaded_size = 0
                            with open(cache_path, 'wb') as fh:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        fh.write(chunk)
                                        downloaded_size += len(chunk)

                            logger.info(f"[pre_download] Downloaded {downloaded_size / (1024*1024):.2f} MB to {cache_path}")
                            downloaded = True
                            break

                        except requests.RequestException as e:
                            last_error = e
                            logger.warning(f"[pre_download] Failed {url}: {e} ({max_retries - i - 1} retries left)")
                            if os.path.exists(cache_path):
                                os.remove(cache_path)

                if not downloaded:
                    logger.error(f"[pre_download] All retries exhausted for {url}. Last error: {last_error}")
                    return False

        return True

    @staticmethod
    async def initialize_runtime(
        job_id: str,
        vm_image_path: str,
        os_type: str,
        osworld_setup: Dict,
        runtime_type: str = "singularity",
        nvcf_function_id: Optional[str] = None,
        nvcf_version_id: Optional[str] = None,
        nvcf_api_key: Optional[str] = None,
        nvcf_org: Optional[str] = None,
    ):
        """
        Initialize runtime using OSWorld's DesktopEnv.

        For NVCF runtime: creates DesktopEnv(provider_name='nvcf') which
        auto-deploys an NVCF function and starts a local proxy.

        For singularity runtime: creates DesktopEnv(provider_name='singularity')
        which uses the local KVM-based approach.
        """
        logger.debug(f"[initialize_runtime] Creating {runtime_type} DesktopEnv for {job_id}")

        if runtime_type == "nvcf":
            # Set env vars that OSWorld's NVCFProvider reads
            if nvcf_api_key:
                os.environ.setdefault("NGC_API_KEY", nvcf_api_key)
            if nvcf_org:
                os.environ.setdefault("NGC_ORG", nvcf_org)
            if nvcf_function_id:
                os.environ["NVCF_FUNCTION_ID"] = nvcf_function_id
            if nvcf_version_id:
                os.environ["NVCF_VERSION_ID"] = nvcf_version_id

            provider_name = "nvcf"
        else:
            provider_name = "singularity"

        env = DesktopEnv(
            provider_name=provider_name,
            path_to_vm=vm_image_path if runtime_type != "nvcf" else "",
            action_space="pyautogui",
            headless=True,
            os_type="Ubuntu" if os_type == "linux" else os_type,
            require_a11y_tree=False,
        )

        logger.debug(f"[initialize_runtime] DesktopEnv created, resetting with OSWorld setup...")

        # DesktopEnv.reset() handles: start emulator, NVCF deploy, proxy, snapshot revert, setup
        env.reset(task_config=osworld_setup)

        logger.debug(f"[initialize_runtime] DesktopEnv reset complete for {job_id}")

        return env

    @staticmethod
    def execute_pyautogui_command(env, pyautogui_command: str):
        """Execute a pyautogui command on the remote VM via OSWorld's PythonController."""
        try:
            env.controller.execute_python_command(pyautogui_command)
            logger.debug("[execute_pyautogui_command] Action complete")
        except Exception as e:
            logger.debug(f"[execute_pyautogui_command] Error in Action: {e}")

    @staticmethod
    def get_screen_size(env) -> Tuple[int, int]:
        """Get the screen size of the remote VM."""
        try:
            size = env.controller.get_vm_screen_size()
            if isinstance(size, tuple) and len(size) == 2:
                return size
            # Fallback: parse from string if needed
            if isinstance(size, str):
                match = re.search(r"(\d+)\D+(\d+)", size)
                if match:
                    return int(match.group(1)), int(match.group(2))
        except Exception as e:
            logger.warning(f"[get_screen_size] Failed: {e}, using defaults")

        return env.screen_width, env.screen_height

    @staticmethod
    def get_screenshot(env) -> bytes:
        """
        Returns the current screenshot from the DesktopEnv as bytes.
        """
        screenshot = env.controller.get_screenshot()
        if not screenshot:
            logger.debug("Failed to get screenshot from DesktopEnv.")
            raise RuntimeError("Failed to get screenshot from DesktopEnv.")
        return screenshot
