"""Anthropic-standard Agent Skill model.

Skills provide specialized knowledge and procedures that agents can use.
This implementation follows the AgentSkills specification (https://agentskills.io).
"""

from pathlib import Path
from typing import Any

import frontmatter
from pydantic import BaseModel, Field, field_validator

from openhands.sdk.skill.exceptions import SkillValidationError
from openhands.sdk.skill.utils import find_skill_md, validate_skill_name


class Skill(BaseModel):
    """Anthropic-standard Agent Skill (agentskills.io).

    Skills are loaded from SKILL.md files. The LLM decides which skills to use
    based on the description field.

    Example:
        >>> skill = Skill.load(Path("./skills/pdf"))
    """

    # Required fields
    name: str = Field(max_length=64)
    """Skill name (max 64 chars, lowercase alphanumeric + hyphens)."""

    description: str = Field(default="", max_length=1024)
    """Brief description of what the skill does and when to use it."""

    content: str
    """The skill instructions (SKILL.md body after frontmatter)."""

    # Source information
    source_dir: Path | None = Field(default=None)
    """Absolute path to skill directory on host."""

    sandbox_path: str | None = Field(default=None, exclude=True)
    """Path in sandbox after mounting (set by loader)."""

    # Optional AgentSkills fields
    license: str | None = None
    """License name or reference to LICENSE.txt."""

    compatibility: str | None = Field(default=None, max_length=500)
    """Environment requirements (e.g., 'Requires Python 3.10+')."""

    metadata: dict[str, str] | None = None
    """Arbitrary key-value metadata."""

    allowed_tools: list[str] | None = None
    """List of pre-approved tools for this skill."""

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        """Validate name follows AgentSkills naming conventions."""
        # Basic validation; full validation happens in load() when loading from file
        if not v:
            raise ValueError("Name cannot be empty")
        return v

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def parse_allowed_tools(cls, v: str | list | None) -> list[str] | None:
        """Parse allowed-tools from space-delimited string or list."""
        if v is None:
            return None
        if isinstance(v, str):
            return v.split()
        return [str(t) for t in v]

    @field_validator("metadata", mode="before")
    @classmethod
    def convert_metadata_values(cls, v: dict | None) -> dict[str, str] | None:
        """Convert metadata values to strings."""
        if v is None:
            return None
        if isinstance(v, dict):
            return {str(k): str(val) for k, val in v.items()}
        raise ValueError("metadata must be a dictionary")

    @classmethod
    def load(cls, skill_dir: Path) -> "Skill":
        """Load skill from directory containing SKILL.md.

        Args:
            skill_dir: Path to the skill directory.

        Returns:
            Skill object with parsed metadata and content.

        Raises:
            SkillValidationError: If SKILL.md is missing or invalid.
        """
        skill_dir = Path(skill_dir).resolve()

        # Find SKILL.md (case-insensitive)
        skill_md = find_skill_md(skill_dir)
        if skill_md is None:
            raise SkillValidationError(f"SKILL.md not found in {skill_dir}")

        # Parse frontmatter and content
        with open(skill_md) as f:
            post = frontmatter.load(f)

        fm: dict[str, Any] = post.metadata or {}
        directory_name = skill_dir.name

        # Get name from frontmatter or use directory name
        name = str(fm.get("name", directory_name))

        # Validate name matches directory
        name_errors = validate_skill_name(name, directory_name)
        if name_errors:
            raise SkillValidationError(
                f"Invalid skill name '{name}': {'; '.join(name_errors)}"
            )

        # Handle allowed-tools field (hyphenated in YAML)
        allowed_tools_value = fm.get("allowed-tools", fm.get("allowed_tools"))

        return cls(
            name=name,
            description=fm.get("description", ""),
            content=post.content,
            source_dir=skill_dir,
            license=fm.get("license"),
            compatibility=fm.get("compatibility"),
            metadata=fm.get("metadata"),
            allowed_tools=allowed_tools_value,
        )

    def get_sandbox_skill_path(self) -> str:
        """Get the path to SKILL.md in sandbox.

        Returns:
            Full path to SKILL.md in the sandbox filesystem.
        """
        if self.sandbox_path:
            return f"{self.sandbox_path}/SKILL.md"
        return f"/workspace/skills/{self.name}/SKILL.md"

    def get_sandbox_resource_path(self, resource: str) -> str:
        """Get path to a resource file in sandbox.

        Args:
            resource: Relative path to the resource (e.g., 'scripts/extract.py').

        Returns:
            Full path to the resource in the sandbox filesystem.
        """
        base = self.sandbox_path or f"/workspace/skills/{self.name}"
        return f"{base}/{resource}"

    def get_source_resource_path(self, resource: str) -> Path | None:
        """Get path to a resource file on the host.

        Args:
            resource: Relative path to the resource.

        Returns:
            Path to the resource on host, or None if no source_dir.
        """
        if self.source_dir is None:
            return None
        return self.source_dir / resource
