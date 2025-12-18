"""
EEP-5.5 Tests: Audit Endpoint

TDD RED Phase: Write failing tests first.

Tests for REST API endpoint exposing cross-reference audit functionality.

WBS Mapping:
- 5.5.1: POST /v1/audit/cross-reference endpoint
- 5.5.2: Request/Response Pydantic models
- 5.5.3: Integration with CrossReferenceAuditor

Anti-Patterns Avoided:
- #12: TestClient with FastAPI (no real server)
- S1192: Constants for repeated strings
- S3776: Small, focused test methods
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient


# =============================================================================
# Constants (Anti-Pattern S1192 Prevention)
# =============================================================================

_ENDPOINT_URL = "/v1/audit/cross-reference"

_SAMPLE_AUDIT_REQUEST: dict[str, Any] = {
    "code": """def chunk_text(text: str, chunk_size: int = 100) -> list[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]""",
    "references": [
        {
            "chapter_id": "ch1",
            "title": "Text Chunking",
            "content": '''# Text Chunking

```python
def chunk_text(text: str, chunk_size: int = 100) -> list[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
```
''',
        }
    ],
    "threshold": 0.5,
}

_SAMPLE_EMPTY_CODE_REQUEST: dict[str, Any] = {
    "code": "",
    "references": [],
    "threshold": 0.5,
}


# =============================================================================
# EEP-5.5.1: Endpoint Existence Tests
# =============================================================================


class TestCrossReferenceEndpoint:
    """Tests for /v1/audit/cross-reference endpoint."""

    def test_endpoint_exists(self, client: TestClient) -> None:
        """POST /v1/audit/cross-reference endpoint exists.

        AC-5.5.1: POST /v1/audit/cross-reference endpoint.
        """
        response = client.post(_ENDPOINT_URL, json=_SAMPLE_AUDIT_REQUEST)
        # Should not be 404
        assert response.status_code != 404

    def test_endpoint_returns_json(self, client: TestClient) -> None:
        """Endpoint returns JSON response."""
        response = client.post(_ENDPOINT_URL, json=_SAMPLE_AUDIT_REQUEST)
        assert response.headers.get("content-type") == "application/json"

    def test_endpoint_returns_200_on_success(self, client: TestClient) -> None:
        """Endpoint returns 200 on successful audit."""
        response = client.post(_ENDPOINT_URL, json=_SAMPLE_AUDIT_REQUEST)
        assert response.status_code == 200


# =============================================================================
# EEP-5.5.2: Request/Response Models Tests
# =============================================================================


class TestAuditRequestModel:
    """Tests for audit request model."""

    def test_request_model_exists(self) -> None:
        """CrossReferenceAuditRequest model exists.

        AC-5.5.2: Request/Response Pydantic models.
        """
        from src.api.models import CrossReferenceAuditRequest

        assert CrossReferenceAuditRequest is not None

    def test_request_requires_code(self) -> None:
        """Request requires code field."""
        from pydantic import ValidationError
        from src.api.models import CrossReferenceAuditRequest

        with pytest.raises(ValidationError):
            CrossReferenceAuditRequest(references=[])  # type: ignore

    def test_request_has_references_field(self) -> None:
        """Request has references field."""
        from src.api.models import CrossReferenceAuditRequest

        request = CrossReferenceAuditRequest(
            code="def test(): pass",
            references=[],
        )
        assert hasattr(request, "references")

    def test_request_has_optional_threshold(self) -> None:
        """Request has optional threshold with default."""
        from src.api.models import CrossReferenceAuditRequest

        request = CrossReferenceAuditRequest(
            code="def test(): pass",
            references=[],
        )
        assert hasattr(request, "threshold")
        assert request.threshold > 0  # Has default value


class TestAuditResponseModel:
    """Tests for audit response model."""

    def test_response_model_exists(self) -> None:
        """CrossReferenceAuditResponse model exists."""
        from src.api.models import CrossReferenceAuditResponse

        assert CrossReferenceAuditResponse is not None

    def test_response_has_passed_field(self) -> None:
        """Response has passed field."""
        from src.api.models import CrossReferenceAuditResponse

        response = CrossReferenceAuditResponse(
            passed=True,
            findings=[],
            best_similarity=0.9,
        )
        assert response.passed is True

    def test_response_has_findings_field(self) -> None:
        """Response has findings list."""
        from src.api.models import CrossReferenceAuditResponse

        response = CrossReferenceAuditResponse(
            passed=True,
            findings=[{"chapter_id": "ch1", "similarity": 0.9}],
            best_similarity=0.9,
        )
        assert len(response.findings) == 1

    def test_response_has_best_similarity(self) -> None:
        """Response has best_similarity field."""
        from src.api.models import CrossReferenceAuditResponse

        response = CrossReferenceAuditResponse(
            passed=True,
            findings=[],
            best_similarity=0.85,
        )
        assert response.best_similarity == 0.85


# =============================================================================
# EEP-5.5.3: Integration with Auditor Tests
# =============================================================================


class TestEndpointIntegration:
    """Tests for endpoint integration with CrossReferenceAuditor."""

    def test_response_contains_passed(self, client: TestClient) -> None:
        """Response contains passed field.

        AC-5.5.3: Integration with CrossReferenceAuditor.
        """
        response = client.post(_ENDPOINT_URL, json=_SAMPLE_AUDIT_REQUEST)
        data = response.json()
        assert "passed" in data

    def test_response_contains_findings(self, client: TestClient) -> None:
        """Response contains findings list."""
        response = client.post(_ENDPOINT_URL, json=_SAMPLE_AUDIT_REQUEST)
        data = response.json()
        assert "findings" in data
        assert isinstance(data["findings"], list)

    def test_response_contains_best_similarity(self, client: TestClient) -> None:
        """Response contains best_similarity score."""
        response = client.post(_ENDPOINT_URL, json=_SAMPLE_AUDIT_REQUEST)
        data = response.json()
        assert "best_similarity" in data

    def test_identical_code_gives_high_similarity(self, client: TestClient) -> None:
        """Identical code in reference gives high similarity score."""
        response = client.post(_ENDPOINT_URL, json=_SAMPLE_AUDIT_REQUEST)
        data = response.json()
        
        # Identical code should pass
        assert data["passed"] is True
        assert data["best_similarity"] >= 0.9

    def test_empty_code_fails_audit(self, client: TestClient) -> None:
        """Empty code fails the audit."""
        response = client.post(_ENDPOINT_URL, json=_SAMPLE_EMPTY_CODE_REQUEST)
        data = response.json()
        assert data["passed"] is False


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestEndpointErrorHandling:
    """Tests for endpoint error handling."""

    def test_invalid_json_returns_422(self, client: TestClient) -> None:
        """Invalid JSON returns 422 validation error."""
        response = client.post(
            _ENDPOINT_URL,
            content="not valid json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_code_returns_422(self, client: TestClient) -> None:
        """Missing code field returns 422."""
        response = client.post(_ENDPOINT_URL, json={"references": []})
        assert response.status_code == 422

    def test_invalid_threshold_returns_422(self, client: TestClient) -> None:
        """Invalid threshold value returns 422."""
        request = {
            "code": "def test(): pass",
            "references": [],
            "threshold": 2.0,  # Invalid - must be 0-1
        }
        response = client.post(_ENDPOINT_URL, json=request)
        assert response.status_code == 422


# =============================================================================
# AC-5.4.4: Status Field in Response Tests
# =============================================================================


class TestEndpointStatusField:
    """Tests for status field in audit response."""

    def test_response_has_status_field(self, client: TestClient) -> None:
        """AC-5.4.4: Response contains status field."""
        response = client.post(_ENDPOINT_URL, json=_SAMPLE_AUDIT_REQUEST)
        data = response.json()
        assert "status" in data

    def test_passing_audit_has_verified_status(self, client: TestClient) -> None:
        """AC-5.4.4: Passing audit returns verified status."""
        response = client.post(_ENDPOINT_URL, json=_SAMPLE_AUDIT_REQUEST)
        data = response.json()
        
        assert data["passed"] is True
        assert data["status"] == "verified"

    def test_failing_audit_has_suspicious_status(self, client: TestClient) -> None:
        """AC-5.4.4: Failing audit returns suspicious status."""
        request = {
            "code": "completely_different_code_xyz_123",
            "references": [
                {
                    "chapter_id": "ch1",
                    "title": "Test",
                    "content": "```python\ndef something_else(): pass\n```",
                }
            ],
            "threshold": 0.99,
        }
        response = client.post(_ENDPOINT_URL, json=request)
        data = response.json()
        
        assert data["passed"] is False
        assert data["status"] == "suspicious"

    def test_response_model_has_status_type(self) -> None:
        """AC-5.4.4: Response model supports status type."""
        from src.api.models import CrossReferenceAuditResponse

        response = CrossReferenceAuditResponse(
            passed=True,
            status="verified",
            findings=[],
            best_similarity=0.9,
        )
        assert response.status == "verified"

    def test_status_must_be_valid_value(self) -> None:
        """AC-5.4.4: Status must be verified/suspicious/false_positive."""
        from pydantic import ValidationError
        from src.api.models import CrossReferenceAuditResponse

        with pytest.raises(ValidationError):
            CrossReferenceAuditResponse(
                passed=True,
                status="invalid_status",  # type: ignore
                findings=[],
                best_similarity=0.9,
            )


# =============================================================================
# AC-5.4.3: Theory→Implementation in Response Tests
# =============================================================================


class TestEndpointTheoryImplementation:
    """Tests for theory→implementation count in response."""

    def test_response_has_theory_impl_count(self, client: TestClient) -> None:
        """AC-5.4.3: Response contains theory_impl_count field."""
        response = client.post(_ENDPOINT_URL, json=_SAMPLE_AUDIT_REQUEST)
        data = response.json()
        assert "theory_impl_count" in data

    def test_theory_impl_count_is_integer(self, client: TestClient) -> None:
        """AC-5.4.3: theory_impl_count is an integer."""
        response = client.post(_ENDPOINT_URL, json=_SAMPLE_AUDIT_REQUEST)
        data = response.json()
        assert isinstance(data["theory_impl_count"], int)

    def test_response_model_has_theory_impl_count(self) -> None:
        """AC-5.4.3: Response model has theory_impl_count field."""
        from src.api.models import CrossReferenceAuditResponse

        response = CrossReferenceAuditResponse(
            passed=True,
            findings=[],
            best_similarity=0.9,
            theory_impl_count=2,
        )
        assert response.theory_impl_count == 2

    def test_theory_impl_count_defaults_to_zero(self) -> None:
        """AC-5.4.3: theory_impl_count defaults to 0."""
        from src.api.models import CrossReferenceAuditResponse

        response = CrossReferenceAuditResponse(
            passed=True,
            findings=[],
            best_similarity=0.9,
        )
        assert response.theory_impl_count == 0

