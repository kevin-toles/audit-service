"""Response models for audit-service API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Severity levels for findings."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Finding(BaseModel):
    """A single finding from an audit."""

    rule_id: str = Field(..., description="Unique rule identifier")
    severity: Severity = Field(..., description="Severity level")
    message: str = Field(..., description="Human-readable description")
    line: int | None = Field(None, description="Line number if applicable")
    column: int | None = Field(None, description="Column number if applicable")
    suggestion: str | None = Field(None, description="Suggested fix")
    reference: str | None = Field(None, description="Reference material citation")


class AuditResponse(BaseModel):
    """Response model for full audit endpoint."""

    passed: bool = Field(..., description="Whether the audit passed")
    findings: list[Finding] = Field(default_factory=list, description="List of findings")
    summary: dict[str, int] = Field(
        default_factory=dict,
        description="Summary counts by severity",
    )
    auditors_run: list[str] = Field(
        default_factory=list,
        description="List of auditors that were executed",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Audit completion timestamp",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class CheckpointResponse(BaseModel):
    """Response model for checkpoint validation."""

    passed: bool = Field(..., description="Whether the checkpoint passed")
    checkpoint_id: int = Field(..., description="Checkpoint that was validated")
    findings: list[Finding] = Field(default_factory=list, description="List of findings")
    can_proceed: bool = Field(
        ...,
        description="Whether the pipeline can proceed to next stage",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Validation completion timestamp",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
