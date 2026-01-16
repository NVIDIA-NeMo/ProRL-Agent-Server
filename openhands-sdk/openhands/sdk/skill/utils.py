"""Utility functions for skill validation."""

import re
from pathlib import Path


# Regex pattern for valid AgentSkills names
# - 1-64 characters
# - Lowercase alphanumeric + hyphens only (a-z, 0-9, -)
# - Must not start or end with hyphen
# - Must not contain consecutive hyphens (--)
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def validate_skill_name(name: str, directory_name: str | None = None) -> list[str]:
    """Validate skill name according to AgentSkills spec.

    Per https://agentskills.io/specification:
    - Max 64 characters
    - Lowercase letters, numbers, and hyphens only
    - Must not start/end with hyphen or contain consecutive hyphens
    - Must match parent directory name

    Args:
        name: The skill name to validate.
        directory_name: Optional directory name to check for match.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    if not name:
        errors.append("Name cannot be empty")
        return errors

    if len(name) > 64:
        errors.append(f"Name exceeds 64 characters: {len(name)}")

    if not SKILL_NAME_PATTERN.match(name):
        errors.append(
            "Name must be lowercase alphanumeric with single hyphens "
            "(e.g., 'my-skill', 'pdf-tools')"
        )

    if directory_name and name != directory_name:
        errors.append(f"Name '{name}' does not match directory '{directory_name}'")

    return errors


def find_skill_md(skill_dir: Path) -> Path | None:
    """Find SKILL.md file in a directory (case-insensitive).

    Args:
        skill_dir: Path to the skill directory to search.

    Returns:
        Path to SKILL.md if found, None otherwise.
    """
    if not skill_dir.is_dir():
        return None
    for item in skill_dir.iterdir():
        if item.is_file() and item.name.lower() == "skill.md":
            return item
    return None
