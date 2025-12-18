"""Audit endpoints.

POST /v1/audit - Full code audit against references
POST /v1/audit/cross-reference - Cross-reference audit using CodeBERT

WBS EEP-5.5: Audit endpoints for code validation.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.models import CrossReferenceAuditRequest, CrossReferenceAuditResponse
from src.auditors.cross_reference_auditor import CrossReferenceAuditor

router = APIRouter(prefix="/v1/audit", tags=["audit"])


@router.post("/cross-reference", response_model=CrossReferenceAuditResponse)
async def audit_cross_reference(
    request: CrossReferenceAuditRequest,
) -> CrossReferenceAuditResponse:
    """Audit code against reference chapters using CodeBERT similarity.

    WBS EEP-5.5.1: POST /v1/audit/cross-reference endpoint.

    Args:
        request: CrossReferenceAuditRequest with code, references, threshold

    Returns:
        CrossReferenceAuditResponse with passed, findings, best_similarity
    """
    auditor = CrossReferenceAuditor()

    # Convert request to auditor context format
    context = {
        "references": [
            {
                "chapter_id": ref.chapter_id,
                "title": ref.title,
                "content": ref.content,
            }
            for ref in request.references
        ],
        "threshold": request.threshold,
    }

    result = await auditor.audit(request.code, context)

    return CrossReferenceAuditResponse(
        passed=result.passed,
        status=result.status.value,  # AC-5.4.4: Include status enum
        findings=result.findings,
        best_similarity=result.metadata.get("best_similarity", 0.0),
        threshold=request.threshold,
        theory_impl_count=result.metadata.get("theory_impl_count", 0),  # AC-5.4.3
    )


# TODO: Implement full audit in Phase 6
# @router.post("")
# async def audit(request: AuditRequest) -> AuditResponse:
#     """Full audit of code against reference materials."""
#     pass
