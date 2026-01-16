"""Skill configuration for mounting skills into workspaces."""

from pathlib import Path

from pydantic import BaseModel, field_validator


class SkillConfig(BaseModel):
    """Configuration for mounting a skill into the sandbox.

    Skills are configured on the host side and mounted into the agent's
    sandbox filesystem. This enables validation before execution and
    portability across different workspace types.

    Example:
        >>> config = SkillConfig(source=Path("./skill_factory/pdf"))
        >>> # Uses default mount path: /workspace/skills/pdf/

        >>> config = SkillConfig(
        ...     source=Path("./custom-skills/analyzer"),
        ...     mount_path="/tools/analyzer",
        ... )
        >>> # Uses custom mount path
    """

    source: Path
    """Local path to the skill directory (must contain SKILL.md)."""

    mount_path: str | None = None
    """
    Path in sandbox where skill will be mounted.
    Defaults to /workspace/skills/<skill-name>/ if not specified.
    """

    @field_validator("source", mode="before")
    @classmethod
    def convert_to_path(cls, v: str | Path) -> Path:
        """Convert string to Path if needed."""
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("source", mode="after")
    @classmethod
    def validate_source(cls, v: Path) -> Path:
        """Ensure source directory exists and contains SKILL.md."""
        # Resolve to absolute path
        v = v.resolve()

        if not v.is_dir():
            raise ValueError(f"Skill source must be a directory: {v}")

        skill_md = v / "SKILL.md"
        if not skill_md.exists():
            # Try case-insensitive match
            found = False
            for item in v.iterdir():
                if item.is_file() and item.name.lower() == "skill.md":
                    found = True
                    break
            if not found:
                raise ValueError(f"Skill directory must contain SKILL.md: {v}")

        return v

    def get_mount_path(self, skill_name: str) -> str:
        """Get the mount path, using default if not specified.

        Args:
            skill_name: Name of the skill (used for default path).

        Returns:
            Mount path in the sandbox filesystem.
        """
        if self.mount_path:
            return self.mount_path
        return f"/workspace/skills/{skill_name}"
