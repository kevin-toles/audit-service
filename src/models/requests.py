"""Request models for audit-service API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AuditRequest(BaseModel):
    """Request model for full audit endpoint."""

    code: str = Field(..., description="Source code to audit")
    language: str = Field(default="python", description="Programming language")
    references: list[str] = Field(
        default_factory=list,
        description="List of reference material IDs to check against",
    )
    auditors: list[str] = Field(
        default_factory=list,
        description="Specific auditors to run (empty = all)",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for the audit",
    )


class CheckpointRequest(BaseModel):
    """Request model for checkpoint validation."""

    code: str = Field(..., description="Source code to validate")
    checkpoint_id: int = Field(..., ge=1, le=3, description="Checkpoint number (1-3)")
    previous_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Results from previous checkpoints",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for validation",
    )
