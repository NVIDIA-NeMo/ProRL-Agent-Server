from typing import Dict
import requests

from openhands.core.logger import openhands_logger as logger


def _make_post_request(env, endpoint: str, **kwargs):
    """Make a POST request using env.client if available, otherwise direct."""
    if hasattr(env, 'client') and env.client:
        return env.client.post(endpoint, **kwargs)
    else:
        vm_ip = env.vm_ip
        port = env.server_port
        url = f"http://{vm_ip}:{port}{endpoint}"
        return requests.post(url, **kwargs)


def get_vm_command_line(env, config: Dict[str, str]):
    command = config["command"]
    shell = config.get("shell", False)

    response = _make_post_request(env, "/execute", json={"command": command, "shell": shell})

    if response.status_code == 200:
        try:
            result = response.json()
            logger.debug(f"VM command response: {result}")
            return result.get("output")
        except Exception as e:
            logger.error(f"Failed to parse VM command response: {e}")
            return None
    else:
        logger.error("Failed to get vm command line. Status code: %d, Response: %s", 
                     response.status_code, response.text[:200] if response.text else "empty")
        return None

def get_vm_command_error(env, config: Dict[str, str]):
    command = config["command"]
    shell = config.get("shell", False)

    response = _make_post_request(env, "/execute", json={"command": command, "shell": shell})

    if response.status_code == 200:
        try:
            result = response.json()
            logger.debug(f"VM command error response: {result}")
            return result.get("error")
        except Exception as e:
            logger.error(f"Failed to parse VM command error response: {e}")
            return None
    else:
        logger.error("Failed to get vm command line error. Status code: %d, Response: %s", 
                     response.status_code, response.text[:200] if response.text else "empty")
        return None


def get_vm_terminal_output(env, config: Dict[str, str]):
    return env.controller.get_terminal_output()
