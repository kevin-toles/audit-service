"""Health check endpoints.

Implements Kubernetes-style health probes:
- /health: Liveness probe (is the service running?)
- /health/ready: Readiness probe (is the service ready to accept traffic?)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    service: str
    version: str
    timestamp: str


class ReadinessResponse(BaseModel):
    """Readiness check response model."""

    status: str
    service: str
    checks: dict[str, Any]
    timestamp: str


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Check if the service is running",
)
async def health() -> HealthResponse:
    """Liveness probe endpoint.

    Returns 200 if the service is running.
    Used by Kubernetes liveness probe.
    """
    return HealthResponse(
        status="healthy",
        service="audit-service",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Check if the service is ready to accept traffic",
)
async def readiness() -> ReadinessResponse:
    """Readiness probe endpoint.

    Returns 200 if the service is ready to handle requests.
    Used by Kubernetes readiness probe.

    Checks:
    - Rules loaded
    - Configuration valid
    """
    # TODO: Add actual readiness checks in Phase 6
    checks: dict[str, Any] = {
        "rules_loaded": True,
        "config_valid": True,
    }

    return ReadinessResponse(
        status="ready",
        service="audit-service",
        checks=checks,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
