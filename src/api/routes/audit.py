"""Audit endpoints.

POST /v1/audit - Full code audit against references
POST /v1/audit/cross-reference - Cross-reference audit using CodeBERT

WBS EEP-5.5: Audit endpoints for code validation.

FIX (2026-01-10): Added dependency injection for CodeBERTClient.
Previously used FakeCodeBERTClient by default, producing random similarity scores.
Now uses environment variable CODEBERT_CLIENT_MODE to select client:
- "real" → CodeBERTClient (calls Code-Orchestrator:8083)
- "fake" → FakeCodeBERTClient (hash-based, for testing)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from fastapi import APIRouter, Depends

logger = logging.getLogger(__name__)

from src.api.models import CrossReferenceAuditRequest, CrossReferenceAuditResponse
from src.auditors.cross_reference_auditor import CrossReferenceAuditor
from src.clients.codebert_client import (
    CodeBERTClient,
    CodeBERTClientProtocol,
    FakeCodeBERTClient,
)

router = APIRouter(prefix="/v1/audit", tags=["audit"])


# =============================================================================
# Dependency Injection for CodeBERT Client
# =============================================================================

_CODE_ORCHESTRATOR_URL = os.getenv(
    "CODE_ORCHESTRATOR_URL", "http://localhost:8083"
)


@lru_cache(maxsize=1)
def _get_client_mode() -> str:
    """Get client mode from environment (cached)."""
    return os.getenv("CODEBERT_CLIENT_MODE", "fake")


def get_codebert_client() -> CodeBERTClientProtocol:
    """Dependency injection for CodeBERT client.
    
    Uses CODEBERT_CLIENT_MODE environment variable:
    - "real": Use CodeBERTClient (HTTP calls to Code-Orchestrator)
    - "fake": Use FakeCodeBERTClient (hash-based, for testing)
    
    Returns:
        CodeBERTClientProtocol implementation
    """
    mode = _get_client_mode()
    if mode == "real":
        client = CodeBERTClient(base_url=_CODE_ORCHESTRATOR_URL)
        logger.info(f"Using REAL CodeBERTClient -> {_CODE_ORCHESTRATOR_URL}")
        return client
    logger.info("Using FakeCodeBERTClient (hash-based)")
    return FakeCodeBERTClient()


@router.post("/cross-reference", response_model=CrossReferenceAuditResponse)
async def audit_cross_reference(
    request: CrossReferenceAuditRequest,
    codebert_client: CodeBERTClientProtocol = Depends(get_codebert_client),
) -> CrossReferenceAuditResponse:
    """Audit code against reference chapters using CodeBERT similarity.

    WBS EEP-5.5.1: POST /v1/audit/cross-reference endpoint.

    Args:
        request: CrossReferenceAuditRequest with code, references, threshold
        codebert_client: Injected CodeBERT client (real or fake)

    Returns:
        CrossReferenceAuditResponse with passed, findings, best_similarity
    """
    auditor = CrossReferenceAuditor(codebert_client=codebert_client)

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
