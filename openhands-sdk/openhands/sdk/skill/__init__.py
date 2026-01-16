"""Agent Skills module.

Skills are Anthropic-standard agent capabilities that provide specialized
knowledge and procedures. They are loaded from SKILL.md files.

Example (from file):
    >>> from openhands.sdk.skill import SkillConfig, load_skills
    >>> configs = [SkillConfig(source=Path("./skills/pdf"))]
    >>> skills = load_skills(configs)

Example (with Agent):
    >>> from openhands.sdk import Agent
    >>> from openhands.sdk.skill import SkillConfig
    >>> agent = Agent(
    ...     tools=[...],
    ...     skills=[
    ...         SkillConfig(source=Path("./skills/pdf")),
    ...         SkillConfig(source=Path("./skills/docx")),
    ...     ],
    ... )
"""

from openhands.sdk.skill.config import SkillConfig
from openhands.sdk.skill.exceptions import SkillError, SkillValidationError
from openhands.sdk.skill.loader import load_skills, mount_skills
from openhands.sdk.skill.skill import Skill


__all__ = [
    "Skill",
    "SkillConfig",
    "SkillError",
    "SkillValidationError",
    "load_skills",
    "mount_skills",
]
