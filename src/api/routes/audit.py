"""Audit endpoints.

POST /v1/audit - Full code audit against references
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/v1/audit", tags=["audit"])


# TODO: Implement in Phase 6
# @router.post("")
# async def audit(request: AuditRequest) -> AuditResponse:
#     """Full audit of code against reference materials."""
#     pass
