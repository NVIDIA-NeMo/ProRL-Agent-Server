"""Skill loading and mounting utilities.

This module provides functions for loading skills from configurations
and mounting them into workspace sandboxes.
"""

from typing import TYPE_CHECKING

from openhands.sdk.logger import get_logger
from openhands.sdk.skill.config import SkillConfig
from openhands.sdk.skill.exceptions import SkillValidationError
from openhands.sdk.skill.skill import Skill


if TYPE_CHECKING:
    from openhands.sdk.workspace.base import BaseWorkspace

logger = get_logger(__name__)


def load_skills(configs: list[SkillConfig]) -> list[Skill]:
    """Load and validate skills from configs.

    Called at Agent initialization. Fails fast if any skill is invalid.

    Args:
        configs: List of SkillConfig objects specifying skill sources.

    Returns:
        List of validated Skill objects.

    Raises:
        SkillValidationError: If any skill fails validation.

    Example:
        >>> configs = [
        ...     SkillConfig(source=Path("./skills/pdf")),
        ...     SkillConfig(source=Path("./skills/docx")),
        ... ]
        >>> skills = load_skills(configs)
    """
    skills = []
    for config in configs:
        try:
            skill = Skill.load(config.source)
            # Apply custom mount path if specified
            if config.mount_path:
                skill.sandbox_path = config.mount_path
            skills.append(skill)
            logger.debug(f"Loaded skill: {skill.name} from {config.source}")
        except SkillValidationError:
            raise
        except Exception as e:
            raise SkillValidationError(
                f"Failed to load skill from {config.source}: {e}"
            ) from e

    logger.info(f"Loaded {len(skills)} skills: {[s.name for s in skills]}")
    return skills


def mount_skills(
    skills: list[Skill],
    workspace: "BaseWorkspace",
) -> None:
    """Mount skills into the workspace sandbox.

    All skills must be loaded from SKILL.md directories and uploaded via the
    workspace's upload_skill() method.

    Args:
        skills: List of validated Skill objects.
        workspace: Workspace to mount skills into.
    """
    for skill in skills:
        if skill.source_dir is None:
            raise SkillValidationError(
                f"Skill '{skill.name}' is missing source_dir. "
                "Programmatic skills are not supported; use SkillConfig "
                "to load skills from SKILL.md."
            )

        mount_path = skill.sandbox_path or f"/workspace/skills/{skill.name}"
        try:
            skill.sandbox_path = workspace.upload_skill(skill.source_dir, mount_path)
        except Exception as e:
            raise SkillValidationError(
                f"Failed to upload skill '{skill.name}' from {skill.source_dir}: {e}"
            ) from e

        logger.info(f"Mounted skill '{skill.name}' to {skill.sandbox_path}")
