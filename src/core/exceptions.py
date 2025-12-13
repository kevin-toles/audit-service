"""Custom exceptions for audit-service."""

from __future__ import annotations


class AuditServiceError(Exception):
    """Base exception for audit-service errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(AuditServiceError):
    """Raised when validation fails."""

    pass


class RuleNotFoundError(AuditServiceError):
    """Raised when a requested rule is not found."""

    pass


class ConfigurationError(AuditServiceError):
    """Raised when configuration is invalid."""

    pass


class ExternalServiceError(AuditServiceError):
    """Raised when an external service call fails."""

    def __init__(
        self,
        message: str,
        service: str,
        status_code: int | None = None,
        details: dict | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            service: Name of the external service that failed.
            status_code: HTTP status code if applicable.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message, details)
        self.service = service
        self.status_code = status_code
