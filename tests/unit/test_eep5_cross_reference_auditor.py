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


# =============================================================================
# AC-5.4.3: Theory→Implementation Detection Tests
# =============================================================================


class TestTheoryImplementationDetection:
    """Tests for detecting theory→implementation relationships."""

    def test_detect_theory_implementation_method_exists(self) -> None:
        """detect_theory_implementation() method exists."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        assert hasattr(auditor, "detect_theory_implementation")
        assert callable(auditor.detect_theory_implementation)

    def test_detects_theory_to_impl_with_algorithm_context(self) -> None:
        """AC-5.4.3: Detects theory→implementation with algorithm keyword."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        content = '''## The Algorithm

The algorithm for text chunking works as follows:

```python
def chunk_text(text: str, size: int = 100) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size)]
```
'''
        relationships = auditor.detect_theory_implementation(content)

        assert len(relationships) >= 1
        assert relationships[0]["relationship_type"] == "theory_to_implementation"

    def test_detects_theory_to_impl_with_pattern_context(self) -> None:
        """AC-5.4.3: Detects theory→implementation with pattern keyword."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        content = '''## The Factory Pattern

The factory pattern provides object creation without exposing logic:

```python
class ShapeFactory:
    def create(self, shape_type: str):
        if shape_type == "circle":
            return Circle()
        return Rectangle()
```
'''
        relationships = auditor.detect_theory_implementation(content)

        assert len(relationships) >= 1

    def test_detects_explicit_implementation_language(self) -> None:
        """AC-5.4.3: Detects explicit 'implementing the above' language."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        content = '''Some theoretical concept here.

Implementing the above concept:

```python
def my_implementation():
    pass
```
'''
        relationships = auditor.detect_theory_implementation(content)

        assert len(relationships) >= 1

    def test_no_detection_for_standalone_code(self) -> None:
        """AC-5.4.3: No detection for code without theory context."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        content = '''```python
x = 1 + 2
```
'''
        relationships = auditor.detect_theory_implementation(content)

        # Simple code without theory context should not be detected
        assert len(relationships) == 0

    @pytest.mark.asyncio
    async def test_audit_includes_theory_impl_count(self) -> None:
        """AC-5.4.3: Audit result includes theory_impl_count in metadata."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        
        # Reference with theory→impl pattern
        context = {
            "references": [
                {
                    "chapter_id": "ch1",
                    "title": "Algorithm Chapter",
                    "content": '''## The Algorithm

The algorithm for processing:

```python
def process(data):
    return data.strip()
```
''',
                }
            ],
            "threshold": 0.3,
        }
        
        result = await auditor.audit("def process(data): return data.strip()", context)

        assert "theory_impl_count" in result.metadata
        assert isinstance(result.metadata["theory_impl_count"], int)

    @pytest.mark.asyncio
    async def test_findings_include_theory_impl_flag(self) -> None:
        """AC-5.4.3: Individual findings include is_theory_impl flag."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        
        context = {
            "references": [
                {
                    "chapter_id": "ch1",
                    "title": "Test",
                    "content": '''The concept here.

```python
def test(): pass
```
''',
                }
            ],
            "threshold": 0.3,
        }
        
        result = await auditor.audit("def test(): pass", context)

        assert len(result.findings) >= 1
        assert "is_theory_impl" in result.findings[0]


# =============================================================================
# AC-5.4.4: Status Enum Tests
# =============================================================================


class TestAuditStatusEnum:
    """Tests for AuditStatus enum in audit results."""

    def test_audit_status_enum_exists(self) -> None:
        """AC-5.4.4: AuditStatus enum exists."""
        from src.auditors.base import AuditStatus

        assert AuditStatus is not None
        assert hasattr(AuditStatus, "VERIFIED")
        assert hasattr(AuditStatus, "SUSPICIOUS")
        assert hasattr(AuditStatus, "FALSE_POSITIVE")

    def test_audit_result_has_status_field(self) -> None:
        """AC-5.4.4: AuditResult has status field."""
        from src.auditors.base import AuditResult, AuditStatus

        result = AuditResult(passed=True, findings=[])
        
        assert hasattr(result, "status")
        assert isinstance(result.status, AuditStatus)

    def test_passing_audit_has_verified_status(self) -> None:
        """AC-5.4.4: Passing audit has VERIFIED status."""
        from src.auditors.base import AuditResult, AuditStatus

        result = AuditResult(passed=True, findings=[])

        assert result.status == AuditStatus.VERIFIED

    def test_failing_audit_has_suspicious_status(self) -> None:
        """AC-5.4.4: Failing audit has SUSPICIOUS status."""
        from src.auditors.base import AuditResult, AuditStatus

        result = AuditResult(passed=False, findings=[])

        assert result.status == AuditStatus.SUSPICIOUS

    def test_explicit_status_overrides_default(self) -> None:
        """AC-5.4.4: Explicit status parameter overrides default."""
        from src.auditors.base import AuditResult, AuditStatus

        result = AuditResult(
            passed=False, 
            findings=[], 
            status=AuditStatus.FALSE_POSITIVE
        )

        assert result.status == AuditStatus.FALSE_POSITIVE

    @pytest.mark.asyncio
    async def test_auditor_returns_verified_status_on_pass(self) -> None:
        """AC-5.4.4: CrossReferenceAuditor returns VERIFIED on pass."""
        from src.auditors.base import AuditStatus
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        
        # Use identical code for reliable pass with fake embedder
        code = "def test(): pass"
        context = {
            "references": [
                {"chapter_id": "ch1", "title": "Test", "content": f"```python\n{code}\n```"}
            ],
            "threshold": 0.5,
        }
        
        result = await auditor.audit(code, context)

        assert result.passed is True
        assert result.status == AuditStatus.VERIFIED

    @pytest.mark.asyncio
    async def test_auditor_returns_suspicious_status_on_fail(self) -> None:
        """AC-5.4.4: CrossReferenceAuditor returns SUSPICIOUS on fail."""
        from src.auditors.base import AuditStatus
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor

        auditor = CrossReferenceAuditor()
        
        # Use very different code for reliable fail
        context = {
            "references": [
                {"chapter_id": "ch1", "title": "Test", "content": "```python\ndef completely_different(): pass\n```"}
            ],
            "threshold": 0.9,  # High threshold
        }
        
        result = await auditor.audit("async def another_thing(): await something()", context)

        assert result.passed is False
        assert result.status == AuditStatus.SUSPICIOUS

    def test_status_enum_values(self) -> None:
        """AC-5.4.4: Status enum has correct string values."""
        from src.auditors.base import AuditStatus

        assert AuditStatus.VERIFIED.value == "verified"
        assert AuditStatus.SUSPICIOUS.value == "suspicious"
        assert AuditStatus.FALSE_POSITIVE.value == "false_positive"

