"""Custom exception taxonomy for the Production-Grade Agentic Orchestration System."""

from typing import Any


class OrchestrationBaseError(Exception):
    """Base exception class for all orchestration errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize error object to dictionary for structured logging or state storage."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class TransientError(OrchestrationBaseError):
    """Temporary network, rate limit, timeout, or transient API failure."""

    pass


class ValidationError(OrchestrationBaseError):
    """Schema validation failure or boundary object mismatch."""

    pass


class ToolError(OrchestrationBaseError):
    """Tool execution or external dependency integration failure."""

    pass


class HumanRequiredError(OrchestrationBaseError):
    """Signal when human review or intervention is mandatory."""

    pass


class ConfigurationError(OrchestrationBaseError):
    """Missing, invalid, or misconfigured environment settings."""

    pass


class UnexpectedError(OrchestrationBaseError):
    """Unclassified runtime exception wrapped with contextual metadata."""

    pass
