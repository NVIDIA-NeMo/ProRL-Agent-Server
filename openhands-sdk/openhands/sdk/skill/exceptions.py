"""Exceptions for skill loading and validation."""


class SkillError(Exception):
    """Base exception for all skill errors."""

    pass


class SkillValidationError(SkillError):
    """Raised when there's a validation error in skill metadata or structure."""

    def __init__(self, message: str = "Skill validation failed") -> None:
        super().__init__(message)
