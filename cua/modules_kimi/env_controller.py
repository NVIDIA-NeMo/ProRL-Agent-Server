import logging
import os
import sys
import re
import uuid
import time
import threading
import requests
from typing import Dict, Optional, Tuple

from openhands.core.logger import openhands_logger

logger = openhands_logger.getChild('kimi_env_controller')
logger.setLevel(logging.DEBUG)

# Semaphore to limit concurrent downloads
_DOWNLOAD_SEMAPHORE = threading.Semaphore(int(os.environ.get('OSWORLD_MAX_CONCURRENT_DOWNLOADS', '3')))


def _is_desktop_env(env_or_runtime) -> bool:
    """Check if the object is a DesktopEnv instance (duck-type check to avoid hard import)."""
    return hasattr(env_or_runtime, 'controller') and hasattr(env_or_runtime, 'reset')


class EnvController:
    """
    Static wrapper class that interfaces with either:
    - OSWorldSingularityRuntime (runtime_type='singularity')
    - OSWorld DesktopEnv (runtime_type='nvcf' or 'nvcf_singularity')
    """

    @staticmethod
    def pre_download_setup_files(osworld_setup: Dict, cache_dir: str = "/tmp/osworld_cache") -> bool:
        """
        Pre-download all setup files to local cache BEFORE deploying NVCF.
        Returns True if all downloads succeeded, False otherwise.
        """
        config_list = osworld_setup.get("config", [])
        if not config_list:
            return True

        os.makedirs(cache_dir, exist_ok=True)

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
        Initialize runtime.

        runtime_type='singularity': uses OSWorldSingularityRuntime (local KVM).
        runtime_type='nvcf_singularity': uses OSWorld DesktopEnv with NVCFSingularityProvider (local .sif).
        runtime_type='nvcf': uses OSWorld DesktopEnv with NVCFProvider.
        """
        logger.debug(f"[initialize_runtime] Creating {runtime_type} runtime for {job_id}")

        if runtime_type == "nvcf":
            from desktop_env.desktop_env import DesktopEnv

            if nvcf_api_key:
                os.environ.setdefault("NGC_API_KEY", nvcf_api_key)
            if nvcf_org:
                os.environ.setdefault("NGC_ORG", nvcf_org)
            if nvcf_function_id:
                os.environ["NVCF_FUNCTION_ID"] = nvcf_function_id
            # Set function name prefix for NVCF deployments
            os.environ.setdefault("OSWORLD_SETUP_CACHE_DIR", "/tmp/osworld_cache")
            fn_prefix = os.environ.get("NVCF_FUNCTION_NAME_PREFIX", "data-collection")
            os.environ.setdefault("NVCF_FUNCTION_NAME_PREFIX", fn_prefix)

            if nvcf_version_id:
                os.environ["NVCF_VERSION_ID"] = nvcf_version_id

            env = None
            try:
                env = DesktopEnv(
                    provider_name="nvcf",
                    path_to_vm="",
                    action_space="pyautogui",
                    headless=True,
                    os_type="Ubuntu" if os_type == "linux" else os_type,
                    require_a11y_tree=False,
                )

                logger.debug(f"[initialize_runtime] DesktopEnv created, resetting with OSWorld setup...")
                env.reset(task_config=osworld_setup)
                logger.debug(f"[initialize_runtime] DesktopEnv reset complete for {job_id}")
                return env
            except Exception:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                raise

        elif runtime_type == "nvcf_singularity":
            from desktop_env.desktop_env import DesktopEnv

            env = None
            try:
                env = DesktopEnv(
                    provider_name="nvcf_singularity",
                    path_to_vm="",
                    action_space="pyautogui",
                    headless=True,
                    os_type="Ubuntu" if os_type == "linux" else os_type,
                    require_a11y_tree=False,
                )

                logger.debug(f"[initialize_runtime] DesktopEnv (nvcf_singularity) created, resetting with OSWorld setup...")
                env.reset(task_config=osworld_setup)
                logger.debug(f"[initialize_runtime] DesktopEnv reset complete for {job_id}")
                return env
            except Exception:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                raise

        else:
            # Singularity backend
            from examples.setup import SetupController
            from openhands.core.config import OpenHandsConfig
            from openhands.events import EventStream
            from openhands.runtime.impl.singularity.osworld_singularity_runtime import OSWorldSingularityRuntime
            from openhands.storage import get_file_store

            config = OpenHandsConfig()
            config.runtime = "osworld"
            config.sandbox.base_container_image = "ubuntu:24.04"
            config.sandbox.run_as_fakeroot = False
            config.sandbox.runtime_container_image = None

            file_store = get_file_store('local', f'/tmp/synthetic_data_gen_{job_id}')
            event_stream = EventStream(sid=job_id, file_store=file_store)

            logger.debug(f"[initialize_runtime]   VM image: {vm_image_path}")

            runtime = OSWorldSingularityRuntime(
                config=config,
                event_stream=event_stream,
                sid=job_id,
                os_type=os_type,
                vm_image_path=vm_image_path,
                attach_to_existing=False,
            )

            await runtime.connect()
            logger.debug(f"[initialize_runtime] Runtime initialized and connected for {job_id}")

            if osworld_setup and os_type == "linux" and osworld_setup.get('config'):
                logger.info(f"[initialize_runtime] [{job_id}] Setting up OSWorld with {len(osworld_setup['config'])} config step(s)")
                setup_controller = SetupController(
                    vm_ip="127.0.0.1",
                    server_port=runtime._vm_server_port,
                    chromium_port=runtime._chromium_port,
                    cache_dir="/tmp/osworld_example",
                    client_password="password",
                    runtime=runtime
                )
                try:
                    await setup_controller.setup(osworld_setup['config'])
                    logger.info(f"[initialize_runtime] [{job_id}] OSWorld setup completed successfully")
                except Exception as e:
                    logger.error(f"[initialize_runtime] [{job_id}] OSWorld setup FAILED: {e}")
                    raise
            else:
                logger.debug(f"[initialize_runtime] No OSWorld setup provided")

            return runtime

    @staticmethod
    def execute_pyautogui_command(env_or_runtime, pyautogui_command: str):
        """Execute a pyautogui command. Works with both DesktopEnv and OSWorldSingularityRuntime."""
        if _is_desktop_env(env_or_runtime):
            try:
                env_or_runtime.controller.execute_python_command(pyautogui_command)
                logger.debug("[execute_pyautogui_command] Action complete")
            except Exception as e:
                logger.debug(f"[execute_pyautogui_command] Error in Action: {e}")
        else:
            from openhands.events.action.os import OSWorldInteractiveAction
            from openhands.events.observation import ErrorObservation
            action = OSWorldInteractiveAction(
                method="execute_python_command",
                params={"command": pyautogui_command},
            )
            result = env_or_runtime.run_action(action)
            if not isinstance(result, ErrorObservation):
                logger.debug("[execute_pyautogui_command] Action complete")
            else:
                logger.debug(f"[execute_pyautogui_command] Error in Action: {result}")

    @staticmethod
    def get_screen_size(env_or_runtime) -> Tuple[int, int]:
        """Get screen size. Works with both DesktopEnv and OSWorldSingularityRuntime."""
        if _is_desktop_env(env_or_runtime):
            try:
                size = env_or_runtime.controller.get_vm_screen_size()
                if isinstance(size, tuple) and len(size) == 2:
                    return size
                if isinstance(size, str):
                    match = re.search(r"(\d+)\D+(\d+)", size)
                    if match:
                        return int(match.group(1)), int(match.group(2))
            except Exception as e:
                logger.warning(f"[get_screen_size] Failed: {e}, using defaults")
            return env_or_runtime.screen_width, env_or_runtime.screen_height
        else:
            from openhands.events.action.os import OSWorldInteractiveAction
            observation = env_or_runtime.run_action(OSWorldInteractiveAction(
                method="get_vm_screen_size",
                params={},
                thought=""
            ))
            assert hasattr(observation, "content"), "get_screen_size failed."
            match = re.search(r"Width: (\d+), Height: (\d+)", observation.content)
            return int(match.group(1)), int(match.group(2))

    @staticmethod
    def get_screenshot(env_or_runtime) -> bytes:
        """Get screenshot. Works with both DesktopEnv and OSWorldSingularityRuntime."""
        if _is_desktop_env(env_or_runtime):
            screenshot = env_or_runtime.controller.get_screenshot()
            if not screenshot:
                raise RuntimeError("Failed to get screenshot from DesktopEnv.")
            return screenshot
        else:
            screenshot = env_or_runtime.get_vm_screenshot()
            if not screenshot:
                raise RuntimeError("Failed to get screenshot from runtime.")
            return screenshot
