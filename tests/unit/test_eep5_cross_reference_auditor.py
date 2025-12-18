"""
EEP-5.4 Tests: Cross-Reference Auditor

TDD RED Phase: Write failing tests first.

Tests for auditing code cross-references using CodeBERT similarity.

WBS Mapping:
- 5.4.1: CrossReferenceAuditor inherits from BaseAuditor
- 5.4.2: Audit code against reference chapters
- 5.4.3: Generate audit findings with confidence scores
- 5.4.4: Integration with EEP-3 multi-signal scoring

Anti-Patterns Avoided:
- #12: Fake embedder for testing
- S1192: Constants for repeated strings
- S3776: Small, focused test methods

Depends On:
- EEP-5.1 (CodeExtractor)
- EEP-5.2 (CodeBERTEmbedder)
- EEP-5.3 (CodeSimilarityScorer)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    pass


# =============================================================================
# Constants (Anti-Pattern S1192 Prevention)
# =============================================================================

_SAMPLE_GENERATED_CODE = '''def chunk_text(text: str, chunk_size: int = 100) -> list[str]:
    """Split text into chunks."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]'''

_SAMPLE_REFERENCE_CODE = '''def split_document(doc: str, size: int = 100) -> list[str]:
    """Split document into smaller parts."""
    chunks = []
    for i in range(0, len(doc), size):
        chunks.append(doc[i:i+size])
    return chunks'''

_SAMPLE_UNRELATED_CODE = '''async def fetch_user(user_id: int) -> dict:
    """Fetch user by ID."""
    return {"id": user_id, "name": "test"}'''

_SAMPLE_CHAPTER_WITH_CODE = '''# Chapter: Text Chunking

This chapter covers text chunking strategies.

```python
def split_document(doc: str, size: int = 100) -> list[str]:
    """Split document into smaller parts."""
    chunks = []
    for i in range(0, len(doc), size):
        chunks.append(doc[i:i+size])
    return chunks
```

## Overlap Strategy

For better context, use overlapping chunks.
'''

_SAMPLE_AUDIT_CONTEXT: dict[str, Any] = {
    "references": [
        {
            "chapter_id": "ch1",
            "title": "Text Chunking",
            "content": _SAMPLE_CHAPTER_WITH_CODE,
        }
    ],
    "threshold": 0.5,
}


# =============================================================================
# EEP-5.4.1: CrossReferenceAuditor Class Tests
# =============================================================================


class TestCrossReferenceAuditorClass:
    """Tests for CrossReferenceAuditor class existence and interface."""

    def test_cross_reference_auditor_class_exists(self) -> None:
        """CrossReferenceAuditor class exists and is importable."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        assert CrossReferenceAuditor is not None

    def test_auditor_inherits_from_base(self) -> None:
        """CrossReferenceAuditor inherits from BaseAuditor.

        AC-5.4.1: Inherits from BaseAuditor.
        """
        from src.auditors.base import BaseAuditor
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        assert issubclass(CrossReferenceAuditor, BaseAuditor)

    def test_auditor_has_name_property(self) -> None:
        """Auditor has name property."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        assert auditor.name == "cross_reference_auditor"

    def test_auditor_has_description_property(self) -> None:
        """Auditor has description property."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        assert isinstance(auditor.description, str)
        assert len(auditor.description) > 0


# =============================================================================
# EEP-5.4.2: Audit Code Against References Tests
# =============================================================================


class TestAuditCodeAgainstReferences:
    """Tests for auditing code against reference chapters."""

    @pytest.mark.asyncio
    async def test_audit_method_exists(self) -> None:
        """audit() method exists and is async.

        AC-5.4.2: Audit code against reference chapters.
        """
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        assert hasattr(auditor, "audit")
        assert callable(auditor.audit)

    @pytest.mark.asyncio
    async def test_audit_returns_audit_result(self) -> None:
        """audit() returns AuditResult."""
        from src.auditors.base import AuditResult
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, _SAMPLE_AUDIT_CONTEXT)

        assert isinstance(result, AuditResult)

    @pytest.mark.asyncio
    async def test_audit_result_has_passed_flag(self) -> None:
        """AuditResult has passed flag."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, _SAMPLE_AUDIT_CONTEXT)

        assert hasattr(result, "passed")
        assert isinstance(result.passed, bool)

    @pytest.mark.asyncio
    async def test_audit_with_matching_reference(self) -> None:
        """Audit passes when generated code matches reference.
        
        Note: Uses the SAME code in reference and audit since FakeCodeBERTEmbedder
        uses hash-based embeddings (identical code = identical embedding = 1.0 similarity).
        """
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        # Use same code as reference for reliable test with fake embedder
        same_code_context = {
            "references": [
                {
                    "chapter_id": "ch1",
                    "title": "Text Chunking",
                    "content": f"```python\n{_SAMPLE_GENERATED_CODE}\n```",
                }
            ],
            "threshold": 0.5,
        }

        auditor = CrossReferenceAuditor()
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, same_code_context)

        # Should pass - identical code gives similarity = 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_audit_with_no_matching_reference(self) -> None:
        """Audit fails when generated code doesn't match any reference."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        context = {
            "references": [
                {
                    "chapter_id": "ch1",
                    "title": "User Management",
                    "content": "No code examples here.",
                }
            ],
            "threshold": 0.5,
        }
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, context)

        # Should fail - no matching reference code
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_audit_with_empty_references(self) -> None:
        """Audit handles empty references gracefully."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        context: dict[str, Any] = {"references": [], "threshold": 0.5}
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, context)

        # Should fail - no references to match
        assert result.passed is False


# =============================================================================
# EEP-5.4.3: Audit Findings with Confidence Scores Tests
# =============================================================================


class TestAuditFindings:
    """Tests for audit findings with confidence scores."""

    @pytest.mark.asyncio
    async def test_audit_result_has_findings(self) -> None:
        """AuditResult contains findings list.

        AC-5.4.3: Generate audit findings with confidence scores.
        """
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, _SAMPLE_AUDIT_CONTEXT)

        assert hasattr(result, "findings")
        assert isinstance(result.findings, list)

    @pytest.mark.asyncio
    async def test_finding_has_confidence_score(self) -> None:
        """Each finding has a confidence score."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, _SAMPLE_AUDIT_CONTEXT)

        if result.findings:
            finding = result.findings[0]
            assert "confidence" in finding or "score" in finding

    @pytest.mark.asyncio
    async def test_finding_has_matched_reference(self) -> None:
        """Passing finding includes matched reference info."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, _SAMPLE_AUDIT_CONTEXT)

        if result.passed and result.findings:
            finding = result.findings[0]
            assert "matched_chapter" in finding or "reference" in finding

    @pytest.mark.asyncio
    async def test_finding_has_similarity_score(self) -> None:
        """Finding includes the similarity score."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, _SAMPLE_AUDIT_CONTEXT)

        if result.findings:
            finding = result.findings[0]
            assert "similarity" in finding or "score" in finding

    @pytest.mark.asyncio
    async def test_multiple_findings_sorted_by_score(self) -> None:
        """Multiple findings are sorted by confidence score descending."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        context = {
            "references": [
                {"chapter_id": "ch1", "title": "Text Chunking", "content": _SAMPLE_CHAPTER_WITH_CODE},
                {"chapter_id": "ch2", "title": "User Fetch", "content": f"```python\n{_SAMPLE_UNRELATED_CODE}\n```"},
            ],
            "threshold": 0.2,  # Low threshold to get multiple findings
        }
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, context)

        if len(result.findings) > 1:
            scores = [f.get("similarity", f.get("score", 0)) for f in result.findings]
            assert scores == sorted(scores, reverse=True)


# =============================================================================
# EEP-5.4.4: Multi-Signal Scoring Integration Tests
# =============================================================================


class TestMultiSignalIntegration:
    """Tests for integration with EEP-3 multi-signal scoring."""

    @pytest.mark.asyncio
    async def test_audit_result_has_metadata(self) -> None:
        """AuditResult includes metadata for multi-signal scoring.

        AC-5.4.4: Integration with EEP-3 multi-signal scoring.
        """
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, _SAMPLE_AUDIT_CONTEXT)

        assert hasattr(result, "metadata")
        assert isinstance(result.metadata, dict)

    @pytest.mark.asyncio
    async def test_metadata_has_signal_scores(self) -> None:
        """Metadata includes signal scores for EEP-3 fusion."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, _SAMPLE_AUDIT_CONTEXT)

        # Should have code_similarity signal
        assert "code_similarity_score" in result.metadata or "best_similarity" in result.metadata

    @pytest.mark.asyncio
    async def test_metadata_has_auditor_name(self) -> None:
        """Metadata includes auditor name for tracing."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, _SAMPLE_AUDIT_CONTEXT)

        assert result.metadata.get("auditor") == "cross_reference_auditor"

    @pytest.mark.asyncio
    async def test_metadata_has_threshold_used(self) -> None:
        """Metadata includes the threshold used for audit."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, _SAMPLE_AUDIT_CONTEXT)

        assert "threshold" in result.metadata


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestCrossReferenceAuditorEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_audit_with_empty_code(self) -> None:
        """Audit handles empty code string."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        result = await auditor.audit("", _SAMPLE_AUDIT_CONTEXT)

        # Should fail - nothing to audit
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_audit_with_whitespace_code(self) -> None:
        """Audit handles whitespace-only code."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        result = await auditor.audit("   \n  ", _SAMPLE_AUDIT_CONTEXT)

        assert result.passed is False

    @pytest.mark.asyncio
    async def test_audit_without_context(self) -> None:
        """Audit handles missing context gracefully."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, {})

        # Should fail - no references in context
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_threshold_from_context(self) -> None:
        """Threshold from context overrides default."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        
        # Very high threshold - should fail
        high_threshold_context = {
            "references": [
                {"chapter_id": "ch1", "title": "Text Chunking", "content": _SAMPLE_CHAPTER_WITH_CODE}
            ],
            "threshold": 0.99,
        }
        result = await auditor.audit(_SAMPLE_GENERATED_CODE, high_threshold_context)

        # High threshold should cause failure
        assert result.passed is False

