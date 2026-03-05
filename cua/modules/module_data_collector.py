import copy
import json
import logging
import os
import random
import socket
import threading
import time
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from modules.actors.debug_uitars_actor import UITarsActor
from modules.debug_planner import Planner
from modules.debug_env_controller import EnvController
from modules.util import *
from openhands.core.logger import openhands_logger


# Create a child logger
logger = openhands_logger.getChild('data_controller')
logger.setLevel(logging.INFO)  # todo for debugging, set this to logging.DEBUG


class DataCollector:
    """
    Class managing actual workflow to collect trajectory data.
    Orchestrates usage of EnvController and OpenAIWrapper.
    """
    def __init__(self, args: Namespace):
        # load helper classes
        self.planner = Planner(args)
        if args.actor_model_name == "ByteDance-Seed/UI-TARS-1.5-7B":
            self.actor = UITarsActor(args)
        else:
            raise NotImplementedError(f"actor_model_name `{args.actor_model_name}` is unknown.`")

        self.vm_image_path = args.vm_image_path
        self.os_type = 'linux' if 'Ubuntu' in self.vm_image_path else 'windows'

        # Runtime type: "singularity" (local KVM) or "nvcf" (NVCF via OSWorld DesktopEnv)
        self.runtime_type = getattr(args, 'runtime', 'singularity')

        # NVCF credentials (passed via env vars to OSWorld's NVCFProvider)
        self.nvcf_api_key = getattr(args, 'nvcf_api_key', None)
        self.nvcf_org = getattr(args, 'nvcf_org', None)

        self.max_steps_per_trajectory = args.max_steps_per_trajectory
        self.max_steps_per_goal = args.max_steps_per_goal

        # We need these directories relative to where the script is run
        self.output_root = (Path("./trajectories") /
                            f"{socket.gethostname()}--{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_root.mkdir(parents=True, exist_ok=True)

        # load persona dataset if args.persona_dataset_path is set
        self.persona_dfs, self.persona_df_weights = load_persona_dataset(args.persona_dataset_path, logger)

        # load osworld setup list
        self.osworld_setup_list = load_osworld_setup_list(args.osworld_setup_path, logger)

        # load example instructions
        self.example_instructions = load_example_instructions(args.example_instructions_path, logger)

        # Default screen dimensions (will be updated per runtime)
        if self.os_type == 'windows':
            self.default_screen_width, self.default_screen_height = 1280, 800
        else:
            self.default_screen_width, self.default_screen_height = 1920, 1080

    def sample_persona(self) -> Optional[Dict[str, Any]]:
        """
        Sample a random persona from the dataset.

        Returns:
            Dictionary containing persona information, or None if dataset not loaded
        """
        if not self.persona_dfs:
            return None

        # Randomly select a dataframe (weighted by number of records)
        selected_df = random.choices(self.persona_dfs, weights=self.persona_df_weights, k=1)[0]
        age = 1
        persona_info = None

        while age < 18:
            # Sample random persona from the selected dataframe
            persona_record = selected_df.sample(n=1).iloc[0].to_dict()

            # Extract key fields
            persona_info = {
                'professional': persona_record.get('professional_persona', ''),
                'hobbies': persona_record.get('hobbies_and_interests', ''),
                'occupation': persona_record.get('occupation', ''),
                'age': persona_record.get('age', ''),
                'education': persona_record.get('education_level', ''),
                'city': persona_record.get('city', ''),
                'state': persona_record.get('state', ''),
                'skills': persona_record.get('skills_and_expertise', ''),
                'interests_list': eval(persona_record.get('hobbies_and_interests_list', '[]')),
                'career_goals': persona_record.get('career_goals_and_ambitions', ''),
            }
            age = persona_record.get('age', 20)

        return persona_info

    @staticmethod
    def save_trajectory(trajectory: Dict, trajectory_save_dir: Path):
        """
        Save trajectory data to json.
        """
        trajectory_to_save = copy.deepcopy(trajectory)
        for step in trajectory_to_save["steps"]:
            for action in step['actions']:
                if "screenshot_base64" in action:
                    action.pop("screenshot_base64")

        with open(trajectory_save_dir / "trajectory.json", "w") as f:
            json.dump(trajectory_to_save, f, indent=4)

        logger.debug(f"✓ [save_trajectory] Saved to {str(trajectory_save_dir / 'trajectory.json')}")

    async def init_runtime_for_job(self, trajectory_idx: int,
                                   nvcf_function_id: str = None,
                                   nvcf_version_id: str = None) -> Tuple:
        """
        Stage 1: Initialize the VM and OSWorld setup.
        Returns: (env, trajectory, trajectory_save_dir, trajectory_id, osworld_setup)

        Uses OSWorld's DesktopEnv which handles NVCF deploy, local proxy,
        and environment setup internally.
        """
        # Create unique IDs
        job_id = f"job_{trajectory_idx:04d}"
        trajectory_id = f"{trajectory_idx:04d}"

        trajectory_save_dir = self.output_root / trajectory_id
        os.makedirs(trajectory_save_dir, exist_ok=True)

        # Sample osworld setup
        osworld_setup_ready, osworld_setup = False, None
        while not osworld_setup_ready:
            osworld_setup = random.choice(self.osworld_setup_list)
            # Filter unstable VLC configs
            if osworld_setup and any("VLC_VERBOSE=-1" in config.get('parameters', {}).get("command", "")
                                     for config in osworld_setup.get('config', [])):
                continue
            else:
                osworld_setup_ready = True

        logger.info(f"[job {trajectory_idx:04d}] Sampled OSWorld config: id={osworld_setup.get('id', 'unknown')}, "
                     f"snapshot={osworld_setup.get('snapshot', 'unknown')}, "
                     f"apps={osworld_setup.get('related_apps', [])}, "
                     f"instruction={osworld_setup.get('instruction', '')[:80]}")

        # Pre-download setup files to local cache BEFORE deploying NVCF.
        # This avoids wasting expensive NVCF GPU resources if downloads fail.
        if self.runtime_type == "nvcf":
            logger.info(f"[job {trajectory_idx:04d}] Pre-downloading setup files before NVCF deploy...")
            download_ok = EnvController.pre_download_setup_files(osworld_setup)
            if not download_ok:
                raise RuntimeError(
                    f"[job {trajectory_idx:04d}] Setup file pre-download failed. "
                    f"Skipping NVCF deploy to avoid wasting resources."
                )
            logger.info(f"[job {trajectory_idx:04d}] Pre-download complete, proceeding with NVCF deploy.")

        # Initialize DesktopEnv (handles NVCF deploy + proxy + setup internally)
        env = await EnvController.initialize_runtime(
            job_id, self.vm_image_path, self.os_type, osworld_setup,
            runtime_type=self.runtime_type,
            nvcf_function_id=nvcf_function_id,
            nvcf_version_id=nvcf_version_id,
            nvcf_api_key=self.nvcf_api_key,
            nvcf_org=self.nvcf_org,
        )

        # Get screen size
        width, height = EnvController.get_screen_size(env)

        # Prepare Metadata
        trajectory = {
            'trajectory_id': trajectory_id,
            'metadata': {
                'vm_image': self.vm_image_path,
                'screen_size': f"{width}x{height}",
                'osworld_setup': osworld_setup
            },
            'goal': None,
            'steps': [],
        }

        return env, trajectory, trajectory_save_dir, trajectory_id, osworld_setup

    async def collect_trajectory(self, env, trajectory: Dict, trajectory_save_dir: Path, osworld_setup: Dict):
        """
        Stage 2: Run the Agent Loop (Goal Generation -> Action Execution).
        `env` is an OSWorld DesktopEnv instance.
        """
        # Wait for UI initialization
        time.sleep(3.0)

        # Initial Screenshot
        screenshot_bytes = EnvController.get_screenshot(env)
        image_filename = trajectory_save_dir / f"0-0.png"
        save_image(screenshot_bytes, image_filename, logger)

        # --- 1. Generate High Level Goal --- #
        prev_requirements = []
        example_goals = random.sample(self.example_instructions, 1)
        goal, requirements = self.planner.generate_goal_with_long_horizon(
            screenshot_bytes, osworld_setup["config"], example_goals, prev_requirements,
        )

        trajectory['goal'] = goal
        logger.debug(f"Generated Goal: {goal}")

        if not trajectory['goal']:
            logger.warning("Failed to generate goal.")
            return trajectory

        # --- 2. Action Loop --- #
        while sum(len(s['actions']) for s in trajectory['steps']) < self.max_steps_per_trajectory:
            # Prepare context for Planner
            prev_subgoal_intents = [g['subgoal_intent'] for g in trajectory['steps']]
            prev_subgoals = [g['subgoal'] for g in trajectory['steps']]
            prev_actor_infos = [
                g["actions"][-1]["action_generation"]["thought"]
                if g["actions"] else "None"
                for g in trajectory['steps']
            ]

            # Generate Subgoal
            subgoal_intent, subgoal = self.planner.generate_subgoal(
                screenshot_bytes, trajectory['goal'],
                prev_subgoal_intents, prev_subgoals, prev_actor_infos
            )

            logger.debug(f"Subgoal: {subgoal} (Intent: {subgoal_intent})")

            if subgoal.lower().strip() in ["done", "impossible"]:
                break

            step_for_this_subgoal = {
                "subgoal": subgoal,
                "subgoal_intent": subgoal_intent,
                "actions": []
            }

            subgoal_idx = len(trajectory['steps'])

            # Actor Loop for this Subgoal
            while len(step_for_this_subgoal['actions']) < self.max_steps_per_goal:
                history_images = [s['screenshot_base64'] for s in step_for_this_subgoal['actions']]
                history_responses = [s['action_generation']['generation'] for s in step_for_this_subgoal['actions']]

                # Generate Action
                action_result = self.actor.generate_action(
                    subgoal, screenshot_bytes, history_images, history_responses
                )

                if action_result is None:
                    break

                pyautogui_command = action_result["pyautogui_command"]
                action_generation = action_result["action_generation"]

                # Execute
                EnvController.execute_pyautogui_command(env, pyautogui_command)

                # Wait & Observe
                time.sleep(3.0)

                # Capture new state
                screenshot_bytes = EnvController.get_screenshot(env)

                # Save step info
                action_idx = len(step_for_this_subgoal['actions'])
                image_filename = trajectory_save_dir / f"{subgoal_idx}-{action_idx + 1}.png"
                save_image(screenshot_bytes, image_filename, logger)

                step_for_this_subgoal['actions'].append({
                    "screenshot": str(image_filename.absolute()),
                    "screenshot_base64": bytes_to_base64(screenshot_bytes),
                    "pyautogui_command": pyautogui_command,
                    "action_generation": action_generation,
                })

                # Check for finished
                if any(a["action_type"] == "finished" for a in action_generation["parsed_actions"]):
                    break

            trajectory['steps'].append(step_for_this_subgoal)
            self.save_trajectory(trajectory, trajectory_save_dir)

        # Final Save
        self.save_trajectory(trajectory, trajectory_save_dir)
        return trajectory

    async def single_trajectory_job(self, trajectory_idx: int):
        """
        Original single-trajectory generation for debugging.
        Simply chains the two stages sequentially in the main thread.
        """
        # 1. Init
        env, trajectory_data, save_dir, t_id, setup = await self.init_runtime_for_job(trajectory_idx)

        try:
            # 2. Collect
            await self.collect_trajectory(env, trajectory_data, save_dir, setup)
        finally:
            # Cleanup
            env.close()
