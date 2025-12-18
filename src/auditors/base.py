"""Base auditor interface.

All auditors must inherit from BaseAuditor and implement the audit method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal


# =============================================================================
# AC-5.4.4: Audit Status Enum
# =============================================================================


class AuditStatus(str, Enum):
    """AC-5.4.4: Status enum for audit results.

    Values:
        VERIFIED: Code matches reference with high confidence
        SUSPICIOUS: Code similarity below threshold, potential issue
        FALSE_POSITIVE: Flagged but confirmed as acceptable
    """

    VERIFIED = "verified"
    SUSPICIOUS = "suspicious"
    FALSE_POSITIVE = "false_positive"


# Type alias for status literal values
AuditStatusLiteral = Literal["verified", "suspicious", "false_positive"]


class AuditResult:
    """Result of an audit operation."""

    def __init__(
        self,
        passed: bool,
        findings: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        status: AuditStatus | None = None,
    ) -> None:
        """Initialize audit result.

        Args:
            passed: Whether the audit passed.
            findings: List of findings (violations, warnings, etc.).
            metadata: Optional metadata about the audit run.
            status: AC-5.4.4: Status enum (verified/suspicious/false_positive).
        """
        self.passed = passed
        self.findings = findings
        self.metadata = metadata or {}
        # AC-5.4.4: Derive status from passed if not explicitly set
        if status is not None:
            self.status = status
        else:
            self.status = AuditStatus.VERIFIED if passed else AuditStatus.SUSPICIOUS


class BaseAuditor(ABC):
    """Abstract base class for all auditors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the auditor name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this auditor checks."""
        ...

    @abstractmethod
    async def audit(self, code: str, context: dict[str, Any]) -> AuditResult:
        """Perform the audit.

        Args:
            code: Source code to audit.
            context: Additional context (references, config, etc.).

        Returns:
            AuditResult with findings.
        """
        ...
