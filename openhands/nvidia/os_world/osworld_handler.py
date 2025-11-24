from typing import Any, Optional
from evaluation.utils.shared import EvalMetadata  # type: ignore
from openhands.core.config import OpenHandsConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.nvidia.registry import AgentHandler, JobDetails
from openhands.runtime.base import Runtime
from openhands.nvidia.os_world.osworld_utils import (
    initialize_exception,
    run_exception,
    eval_exception,
)
from openhands.nvidia.utils import final_result as utils_final_result

# Import the existing functions from utils
from openhands.nvidia.os_world.osworld_utils import (  # type: ignore
    initialize_agents,
    run_agent,
    evaluate_agent,
)

from openhands.nvidia.reward import Reward

class OSWorldHandler(AgentHandler):
    """Handler for SWE Agent integration, reusing functions from utils.py."""

    @property
    def name(self) -> str:
        """The name identifier for this agent handler."""
        return "osworld"

    async def init(
        self,
        job_details: JobDetails,
        sid: str | None = None,
    ) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:
        """Initialize the SWE Agent with instance and config using utils functions."""
        instance = job_details.instance
        llm_config = job_details.llm_config

        return await initialize_agents(
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
        return await run_agent(
            job_details=job_details,
            sid=sid,
        )

    async def eval(
            self, job_details: JobDetails,
            sid: str | None = None,
            allow_skip: bool = True,
            reward: Optional[Reward] = None,
        ) -> dict[str, Any]:

        if reward is None:
            raise ValueError('Reward is required for evaluation of math problems.')

        return await evaluate_agent(
            run_results=job_details.run_results,
            instance=job_details.instance,
        )

    def init_exception(self, job_details: JobDetails, exception: Exception) -> dict[str, Any]:
        """Handle exceptions during initialization using utils functions."""
        return initialize_exception(job_details, exception)

    def run_exception(self, job_details: JobDetails, exception: Exception) -> dict[str, Any]:
        """Handle exceptions during run using utils functions."""
        return run_exception(job_details, exception)

    def eval_exception(self, job_details: JobDetails, exception: Exception) -> dict[str, Any]:
        """Handle exceptions during evaluation using utils functions."""
        return eval_exception(job_details, exception)

    def final_result(self, job_details: JobDetails) -> dict[str, Any]:
        """Process final results using utils functions."""
        return utils_final_result(job_details)
