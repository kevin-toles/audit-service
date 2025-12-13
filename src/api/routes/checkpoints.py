"""Checkpoint endpoints.

POST /v1/audit/checkpoint/{id} - Stage-specific validation
- Checkpoint 1: Post-draft (validates CodeT5+ output)
- Checkpoint 2: Post-LLM (validates LLM implementation)
- Checkpoint 3: Post-external (final validation)
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/v1/audit/checkpoint", tags=["checkpoints"])


# TODO: Implement in Phase 6
# @router.post("/{checkpoint_id}")
# async def checkpoint(checkpoint_id: int, request: CheckpointRequest) -> CheckpointResponse:
#     """Execute checkpoint validation."""
#     pass
