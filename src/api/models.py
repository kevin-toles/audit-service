"""
EEP-5.5: API Models for Cross-Reference Audit

Pydantic models for audit request/response.

WBS Mapping:
- 5.5.2: Request/Response Pydantic models

Anti-Patterns Avoided:
- S1192: Constants for repeated values
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Constants
# =============================================================================

_DEFAULT_THRESHOLD = 0.5


# =============================================================================
# Reference Model
# =============================================================================


class ReferenceChapter(BaseModel):
    """A reference chapter containing code examples."""

    chapter_id: str = Field(description="Unique identifier for the chapter")
    title: str = Field(default="", description="Chapter title")
    content: str = Field(description="Markdown content with code blocks")


# =============================================================================
# Request Model
# =============================================================================


class CrossReferenceAuditRequest(BaseModel):
    """Request model for cross-reference audit.

    WBS 5.5.2: Request Pydantic model.
    """

    code: str = Field(description="Generated code to audit")
    references: list[ReferenceChapter] = Field(
        default_factory=list,
        description="Reference chapters to compare against",
    )
    threshold: float = Field(
        default=_DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Similarity threshold (0.0-1.0)",
    )


# =============================================================================
# Response Model
# =============================================================================


class CrossReferenceAuditResponse(BaseModel):
    """Response model for cross-reference audit.

    WBS 5.5.2: Response Pydantic model.
    """

    passed: bool = Field(description="Whether the audit passed")
    findings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of findings with similarity scores",
    )
    best_similarity: float = Field(
        ge=0.0,
        le=1.0,
        description="Best similarity score found",
    )
    threshold: float = Field(
        default=_DEFAULT_THRESHOLD,
        description="Threshold used for audit",
    )
