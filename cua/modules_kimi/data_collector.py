"""
DataCollector for the Kimi-K2.5 pipeline.

Flat action loop: goal generation -> direct action execution (no subgoal decomposition).
Self-contained — does not import from modules/.
"""
import copy
import json
import logging
import os
import random
import socket
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


from modules_kimi.kimi_actor import KimiActor
from modules_kimi.env_controller import EnvController
from modules_kimi.util import (
    save_image,
    bytes_to_base64,
    load_persona_dataset,
    load_osworld_setup_list,
    load_example_instructions,
)
from openhands.core.logger import openhands_logger

logger = openhands_logger.getChild('kimi_data_collector')
logger.setLevel(logging.INFO)


class DataCollector:
    """
    Orchestrates VM init + Kimi trajectory collection.
    """

    def __init__(self, args: Namespace):
        self.actor = KimiActor(args)

        self.vm_image_path = args.vm_image_path
        self.os_type = 'linux' if 'Ubuntu' in self.vm_image_path else 'windows'

        # Runtime type: 'singularity' (local KVM) or 'nvcf' (NVCF via OSWorld DesktopEnv)
        self.runtime_type = getattr(args, 'runtime', 'singularity')

        self.max_steps_per_trajectory = args.max_steps_per_trajectory

        # Output directory: cua/trajectories/kimi/<hostname>--<timestamp>
        # self.output_root = (
        #     Path("./trajectories/kimi")
        #     / f"{socket.gethostname()}--{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # )
        self.output_root = (
            Path(args.trajectory_save_dir)
            / f"{socket.gethostname()}--{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Load datasets
        self.persona_dfs, self.persona_df_weights = load_persona_dataset(
            args.persona_dataset_path, logger
        )
        self.osworld_setup_list = load_osworld_setup_list(args.osworld_setup_path, logger)
        self.example_instructions = load_example_instructions(
            args.example_instructions_path, logger
        )

        # Default screen dimensions
        if self.os_type == 'windows':
            self.default_screen_width, self.default_screen_height = 1280, 800
        else:
            self.default_screen_width, self.default_screen_height = 1920, 1080

    @staticmethod
    def save_trajectory(trajectory: Dict, trajectory_save_dir: Path):
        """Save trajectory data to json (strip base64 screenshots from saved copy)."""
        trajectory_to_save = copy.deepcopy(trajectory)
        for step in trajectory_to_save["steps"]:
            for action in step["actions"]:
                action.pop("screenshot_base64", None)

        with open(trajectory_save_dir / "trajectory.json", "w") as f:
            json.dump(trajectory_to_save, f, indent=4)

        logger.debug(f"Saved trajectory to {trajectory_save_dir / 'trajectory.json'}")

    async def init_runtime_for_job(self, trajectory_idx: int, nvcf_function_id: str = None, nvcf_version_id: str = None) -> Tuple:
        """
        Stage 1: Initialize the VM and OSWorld setup.
        Returns: (runtime, trajectory, trajectory_save_dir, trajectory_id, osworld_setup)
        """
        job_id = f"job_{trajectory_idx:04d}"
        trajectory_id = f"{trajectory_idx:04d}"

        trajectory_save_dir = self.output_root / trajectory_id
        os.makedirs(trajectory_save_dir, exist_ok=True)

        # Sample osworld setup (filter VLC configs)
        osworld_setup_ready, osworld_setup = False, None
        while not osworld_setup_ready:
            osworld_setup = random.choice(self.osworld_setup_list)
            osworld_setup_ready = True
            # if osworld_setup and any(
            #     "VLC_VERBOSE=-1" in config.get("parameters", {}).get("command", "")
            #     for config in osworld_setup.get("config", [])
            # ):
            #     continue
            # else:
            #     osworld_setup_ready = True

        # Pre-download setup files before NVCF deploy to avoid wasting GPU resources
        if self.runtime_type == 'nvcf':
            logger.info(f'[job {trajectory_idx:04d}] Pre-downloading setup files before NVCF deploy...')
            download_ok = EnvController.pre_download_setup_files(osworld_setup)
            if not download_ok:
                raise RuntimeError(
                    f'[job {trajectory_idx:04d}] Setup file pre-download failed. '
                    f'Skipping NVCF deploy to avoid wasting resources.'
                )
            logger.info(f'[job {trajectory_idx:04d}] Pre-download complete.')

        # Initialize runtime
        env_or_runtime = await EnvController.initialize_runtime(
            job_id, self.vm_image_path, self.os_type, osworld_setup,
            runtime_type=self.runtime_type,
            nvcf_function_id=nvcf_function_id,
            nvcf_version_id=nvcf_version_id,
        )

        # Get screen size
        width, height = EnvController.get_screen_size(env_or_runtime)

        trajectory = {
            "trajectory_id": trajectory_id,
            "metadata": {
                "vm_image": self.vm_image_path,
                "screen_size": f"{width}x{height}",
                "osworld_setup": osworld_setup,
                "pipeline": "kimi",
            },
            "goal": None,
            "steps": [],
        }

        return env_or_runtime, trajectory, trajectory_save_dir, trajectory_id, osworld_setup

    async def collect_trajectory(
        self, runtime, trajectory: Dict, trajectory_save_dir: Path, osworld_setup: Dict
    ):
        """
        Stage 2: Flat action loop — goal generation then direct action execution.
        """
        # Wait for UI initialization
        time.sleep(3.0)

        # Initial screenshot
        screenshot_bytes = EnvController.get_screenshot(runtime)
        image_filename = trajectory_save_dir / "0-0.png"
        save_image(screenshot_bytes, image_filename, logger)

        # Update actor screen size from runtime
        width, height = trajectory["metadata"]["screen_size"].split("x")
        self.actor.screen_size = (int(width), int(height))

        # --- 1. Generate Goal --- #
        osworld_config = osworld_setup["config"] if osworld_setup else []
        example_goals = random.sample(self.example_instructions, 1)

        goal, requirements = self.actor.generate_goal(
            screenshot_bytes, osworld_config, example_goals
        )

        trajectory["goal"] = goal
        logger.info(f"Generated Goal: {goal}")

        if not trajectory["goal"]:
            logger.warning("Failed to generate goal.")
            return trajectory

        # --- 2. Flat Action Loop --- #
        # Single step entry (no subgoal decomposition)
        step = {
            "subgoal": goal,
            "subgoal_intent": "Same as high-level goal",
            "actions": [],
        }

        # History lists (kept locally for thread safety)
        # history_cots stores dicts with "thought" and "action" keys, matching reference's cots format
        history_screenshots: list[bytes] = []
        history_actions: list[str] = []
        history_cots: list[Dict] = []

        while len(step["actions"]) < self.max_steps_per_trajectory:
            action_result = self.actor.generate_action(
                instruction=goal,
                screenshot_bytes=screenshot_bytes,
                history_screenshots=history_screenshots,
                history_actions=history_actions,
                history_cots=history_cots,
            )

            if action_result is None:
                logger.warning("generate_action returned None, stopping trajectory.")
                break

            action_type = action_result["action_type"]
            pyautogui_command = action_result["pyautogui_command"]
            action_generation = action_result["action_generation"]
            raw_response = action_result.get("raw_response", "")
            raw_reasoning = action_result.get("raw_reasoning", "")

            # Handle special action types
            if action_type == "done":
                logger.info("Kimi signaled task completion (done).")
                step["actions"].append({
                    "screenshot": str(image_filename.absolute()),
                    "pyautogui_command": "",
                    "action_type": "done",
                    "action_generation": action_generation,
                    "raw_response": raw_response,
                    "raw_reasoning": raw_reasoning,
                })
                break

            if action_type == "fail":
                logger.info("Kimi signaled task failure (fail).")
                step["actions"].append({
                    "screenshot": str(image_filename.absolute()),
                    "pyautogui_command": "",
                    "action_type": "fail",
                    "action_generation": action_generation,
                    "raw_response": raw_response,
                    "raw_reasoning": raw_reasoning,
                })
                break

            if action_type == "wait":
                logger.debug("Kimi requested wait (10s).")
                time.sleep(10)
            else:
                # Execute pyautogui command on VM
                EnvController.execute_pyautogui_command(runtime, pyautogui_command)

            # Wait & observe
            time.sleep(3.0)

            # Update history BEFORE taking new screenshot
            # action_generation is the sections dict from parse_response_to_cot_and_action
            history_screenshots.append(screenshot_bytes)
            history_actions.append(action_generation.get("action", ""))
            history_cots.append({
                "thought": action_generation.get("thought", ""),
                "action": action_generation.get("action", ""),
            })

            # Capture new state
            screenshot_bytes = EnvController.get_screenshot(runtime)

            action_idx = len(step["actions"])
            image_filename = trajectory_save_dir / f"0-{action_idx + 1}.png"
            save_image(screenshot_bytes, image_filename, logger)

            step["actions"].append({
                "screenshot": str(image_filename.absolute()),
                "screenshot_base64": bytes_to_base64(screenshot_bytes),
                "pyautogui_command": pyautogui_command,
                "action_type": action_type,
                "action_generation": action_generation,
                "raw_reasoning": raw_reasoning,
                "raw_response": raw_response,
            })

            # Incremental save
            trajectory["steps"] = [step]
            self.save_trajectory(trajectory, trajectory_save_dir)

        # Final save
        trajectory["steps"] = [step]
        self.save_trajectory(trajectory, trajectory_save_dir)
        return trajectory

    async def single_trajectory_job(self, trajectory_idx: int):
        """
        Single-trajectory generation for debugging.
        Chains init + collect sequentially.
        """
        runtime, trajectory_data, save_dir, t_id, setup = await self.init_runtime_for_job(
            trajectory_idx
        )

        try:
            await self.collect_trajectory(runtime, trajectory_data, save_dir, setup)
        finally:
            runtime.close()
