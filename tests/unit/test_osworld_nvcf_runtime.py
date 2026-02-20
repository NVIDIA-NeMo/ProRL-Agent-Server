"""Unit tests for OSWorld NVCF Runtime (openhands.runtime.impl.nvcf)."""

import os
import unittest
from unittest.mock import MagicMock, Mock, patch

from openhands.core.config import OpenHandsConfig
from openhands.events import EventStream
from openhands.events.action.os import OSWorldInteractiveAction
from openhands.runtime.impl.nvcf import NVCFRuntime, OSWorldNVCFRuntime


class TestNVCFRuntime(unittest.TestCase):
    """Test base NVCFRuntime (ActionExecutionClient, deploy/close)."""

    def setUp(self):
        self.config = OpenHandsConfig()
        self.config.runtime = "osworld_nvcf"
        self.event_stream = Mock(spec=EventStream)
        self.event_stream.file_store = Mock()
        self.event_stream.file_store.write = Mock()
        self.event_stream.file_store.read = Mock(side_effect=FileNotFoundError)
        self.event_stream.file_store.delete = Mock()

    def test_nvcf_runtime_extends_action_execution_client(self):
        """NVCFRuntime must extend ActionExecutionClient, not SingularityRuntime."""
        from openhands.runtime.impl.action_execution.action_execution_client import (
            ActionExecutionClient,
        )
        self.assertEqual(NVCFRuntime.__bases__[0], ActionExecutionClient)
        self.assertTrue(issubclass(NVCFRuntime, ActionExecutionClient))

    def test_nvcf_runtime_requires_api_key(self):
        """Without NGC_API_KEY / nvcf_api_key, init raises."""
        with patch.dict(os.environ, {}, clear=False):
            for k in ("NGC_API_KEY", "NVCF_FUNCTION_ID", "NGC_ORG"):
                if k in os.environ:
                    del os.environ[k]
            with self.assertRaises(ValueError) as ctx:
                NVCFRuntime(
                    config=self.config,
                    event_stream=self.event_stream,
                    sid="test",
                )
            self.assertIn("api key", str(ctx.exception).lower())

    def test_nvcf_runtime_init_with_env(self):
        """With NGC_API_KEY and NVCF_FUNCTION_ID, init succeeds."""
        with patch.dict(
            os.environ,
            {"NGC_API_KEY": "test-key", "NVCF_FUNCTION_ID": "test-fid"},
            clear=False,
        ):
            r = NVCFRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid="test",
            )
            self.assertEqual(r._nvcf_function_id, "test-fid")
            self.assertEqual(r._nvcf_api_key, "test-key")


class TestOSWorldNVCFRuntime(unittest.TestCase):
    """Test OSWorld NVCF runtime (same API as OSWorld Singularity, NVCF backend)."""

    def setUp(self):
        self.config = OpenHandsConfig()
        self.config.runtime = "osworld_nvcf"
        self.event_stream = Mock(spec=EventStream)
        self.event_stream.file_store = Mock()
        self.event_stream.file_store.write = Mock()
        self.event_stream.file_store.read = Mock(side_effect=FileNotFoundError)
        self.event_stream.file_store.delete = Mock()

    def test_osworld_nvcf_extends_nvcf_runtime(self):
        """OSWorldNVCFRuntime extends NVCFRuntime."""
        self.assertTrue(issubclass(OSWorldNVCFRuntime, NVCFRuntime))

    def test_osworld_nvcf_init_os_type_and_screen_size(self):
        """OS type and default screen_size are set."""
        with patch.dict(
            os.environ,
            {"NGC_API_KEY": "k", "NVCF_FUNCTION_ID": "f"},
            clear=False,
        ):
            r = OSWorldNVCFRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid="test",
                os_type="linux",
            )
            self.assertEqual(r.os_type, "linux")
            self.assertEqual(r.screen_size, (1920, 1080))

    def test_osworld_vm_url(self):
        """osworld_vm_url is NVCF API base."""
        with patch.dict(
            os.environ,
            {"NGC_API_KEY": "k", "NVCF_FUNCTION_ID": "f"},
            clear=False,
        ):
            r = OSWorldNVCFRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid="test",
            )
            self.assertEqual(r.osworld_vm_url, "https://grpc.nvcf.nvidia.com/api")

    def test_action_to_pyautogui_click(self):
        """_action_to_pyautogui_command produces click command."""
        with patch.dict(
            os.environ,
            {"NGC_API_KEY": "k", "NVCF_FUNCTION_ID": "f"},
            clear=False,
        ):
            r = OSWorldNVCFRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid="test",
            )
            cmd = r._action_to_pyautogui_command(
                "CLICK",
                {"x": 100, "y": 200, "button": "left"},
            )
            self.assertIn("pyautogui.click", cmd)
            self.assertIn("100", cmd)
            self.assertIn("200", cmd)

    def test_action_to_pyautogui_typing(self):
        """_action_to_pyautogui_command produces typewrite for TYPING."""
        with patch.dict(
            os.environ,
            {"NGC_API_KEY": "k", "NVCF_FUNCTION_ID": "f"},
            clear=False,
        ):
            r = OSWorldNVCFRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid="test",
            )
            cmd = r._action_to_pyautogui_command(
                "TYPING",
                {"text": "hello"},
            )
            self.assertIn("pyautogui.typewrite", cmd)
            self.assertIn("hello", cmd)

    def test_run_action_get_screenshot_mocked_client(self):
        """run_action(get_screenshot) uses _nvcf_client and returns observation."""
        with patch.dict(
            os.environ,
            {"NGC_API_KEY": "k", "NVCF_FUNCTION_ID": "f"},
            clear=False,
        ):
            r = OSWorldNVCFRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid="test",
            )
            mock_client = MagicMock()
            mock_client.get.return_value = MagicMock(status_code=200, content=b"png")
            r._nvcf_client = mock_client
            r._runtime_initialized = True

            action = OSWorldInteractiveAction(
                method="get_screenshot",
                params={},
                thought="test",
            )
            obs = r.run_action(action)
            self.assertIsNotNone(obs)
            self.assertEqual(obs.content, "Screenshot captured")
            mock_client.get.assert_called()

    def test_run_action_execute_action_mocked_client(self):
        """run_action(execute_action) calls execute_vm_action and returns observation."""
        with patch.dict(
            os.environ,
            {"NGC_API_KEY": "k", "NVCF_FUNCTION_ID": "f"},
            clear=False,
        ):
            r = OSWorldNVCFRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid="test",
            )
            mock_client = MagicMock()
            mock_client.post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"status": "success", "output": "ok"},
            )
            r._nvcf_client = mock_client
            r._runtime_initialized = True

            action = OSWorldInteractiveAction(
                method="execute_action",
                params={
                    "action": {
                        "action_type": "CLICK",
                        "parameters": {"x": 10, "y": 10},
                    }
                },
                thought="test",
            )
            obs = r.run_action(action)
            self.assertIsNotNone(obs)
            self.assertEqual(obs.exit_code, 0)
            mock_client.post.assert_called()

    def test_run_action_unknown_method_returns_error_observation(self):
        """Unknown method returns ErrorObservation."""
        with patch.dict(
            os.environ,
            {"NGC_API_KEY": "k", "NVCF_FUNCTION_ID": "f"},
            clear=False,
        ):
            r = OSWorldNVCFRuntime(
                config=self.config,
                event_stream=self.event_stream,
                sid="test",
            )
            r._nvcf_client = MagicMock()
            r._runtime_initialized = True

            action = OSWorldInteractiveAction(
                method="no_such_method",
                params={},
                thought="test",
            )
            obs = r.run_action(action)
            from openhands.events.observation import ErrorObservation
            self.assertIsInstance(obs, ErrorObservation)
            self.assertIn("Unknown", obs.content)


if __name__ == "__main__":
    unittest.main()
