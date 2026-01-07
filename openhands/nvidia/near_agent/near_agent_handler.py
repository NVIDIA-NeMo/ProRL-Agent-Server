# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""NeAR (NeMo Agent Runtime) Agent Handler for deep research tasks."""

from typing import Any, Optional

from evaluation.utils.shared import EvalMetadata
from openhands.core.config import OpenHandsConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.nvidia.registry import AgentHandler, JobDetails
from openhands.nvidia.reward import Reward
from openhands.nvidia.utils import (
    eval_exception,
    final_result as utils_final_result,
    initialize_exception,
    run_exception,
)
from openhands.runtime.base import Runtime

from openhands.nvidia.near_agent.utils import (
    evaluate_agent as near_evaluate_agent,
    initialize_agents as near_initialize_agents,
    run_agent as near_run_agent,
)


class NeARHandler(AgentHandler):
    """Handler for NeAR (NeMo Agent Runtime) integration for deep research tasks.

    NeAR provides enhanced tooling for research-oriented tasks with:
    - Browser automation (Playwright)
    - Stateful IPython execution
    - Web search via Tavily MCP
    - Bash command execution
    - File editing with str_replace
    - Specialized workspace filesystem
    """

    @property
    def name(self) -> str:
        """The name identifier for this agent handler.

        This name is used for routing instances to the appropriate handler.
        Instances with data_source='near' will be routed to this handler.
        """
        return 'near'

    async def init(
        self,
        job_details: JobDetails,
        sid: str | None = None,
    ) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:
        """Initialize the NeAR Agent with enhanced tooling and workspace.

        Args:
            job_details: Job details containing instance, LLM config, and agent config
            sid: Session ID for tracking

        Returns:
            Tuple of (Runtime, EvalMetadata, OpenHandsConfig)
        """
        instance = job_details.instance
        llm_config = job_details.llm_config

        return await near_initialize_agents(
            instance=instance,
            llm_config=llm_config,
            sid=sid,
            agent_config=job_details.agent_config,
        )

    async def run(
        self,
        job_details: JobDetails,
        sid: str | None = None,
    ) -> dict[str, object]:
        """Run the NeAR Agent with full research capabilities.

        Args:
            job_details: Job details with runtime, metadata, and config
            sid: Session ID for tracking

        Returns:
            Dict with outputs, success status, error info, and messages
        """
        return await near_run_agent(
            job_details=job_details,
            sid=sid,
        )

    async def eval(
        self,
        job_details: JobDetails,
        sid: str | None = None,
        allow_skip: bool = True,
        reward: Optional[Reward] = None,
    ) -> dict[str, Any]:
        """Evaluate the NeAR Agent results.

        Supports multiple evaluation modes:
        1. RLER reward (dr-tulu methodology) - set instance['use_rler_reward']=True
        2. Reward-based (using custom reward function)
        3. Output-based (checking for required outputs)
        4. Manual review (returning outputs for human evaluation)

        Args:
            job_details: Job details with run results
            sid: Session ID for tracking
            allow_skip: Whether to skip evaluation if no results
            reward: Optional reward function for automatic evaluation

        Returns:
            Dict with evaluation results (resolved status, outputs, etc.)
        """
        # Check if RLER reward should be used
        use_rler = job_details.instance.get('use_rler_reward', False)

        return await near_evaluate_agent(
            run_results=job_details.run_results,
            instance=job_details.instance,
            sid=sid,
            allow_skip=allow_skip,
            reward=reward,
            use_rler_reward=use_rler,
        )

    def init_exception(
        self, job_details: JobDetails, exception: Exception
    ) -> dict[str, Any]:
        """Handle exceptions during initialization.

        Args:
            job_details: Job details
            exception: Exception that was raised

        Returns:
            Dict with error information
        """
        return initialize_exception(job_details, exception)

    def run_exception(
        self, job_details: JobDetails, exception: Exception
    ) -> dict[str, Any]:
        """Handle exceptions during run.

        Args:
            job_details: Job details
            exception: Exception that was raised

        Returns:
            Dict with error information
        """
        return run_exception(job_details, exception)

    def eval_exception(
        self, job_details: JobDetails, exception: Exception
    ) -> dict[str, Any]:
        """Handle exceptions during evaluation.

        Args:
            job_details: Job details
            exception: Exception that was raised

        Returns:
            Dict with error information
        """
        return eval_exception(job_details, exception)

    def final_result(self, job_details: JobDetails) -> dict[str, Any]:
        """Process final results.

        Args:
            job_details: Job details with all results

        Returns:
            Dict with final processed results
        """
        return utils_final_result(job_details)
