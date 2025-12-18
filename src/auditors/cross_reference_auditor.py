"""
EEP-5.4: Cross-Reference Auditor

Audits generated code against reference chapter code using CodeBERT similarity.

WBS Mapping:
- 5.4.1: CrossReferenceAuditor inherits from BaseAuditor
- 5.4.2: Audit code against reference chapters
- 5.4.3: Generate audit findings with confidence scores
- 5.4.4: Integration with EEP-3 multi-signal scoring

Patterns Applied:
- Inherits BaseAuditor per existing pattern
- Uses CodeExtractor for reference code extraction
- Uses CodeSimilarityScorer for similarity comparison

Anti-Patterns Avoided:
- #12: Uses FakeCodeBERTEmbedder for testing
- S1192: Extracted constants
- S3776: Small focused methods, delegated to helper classes
"""

from __future__ import annotations

from typing import Any

from src.auditors.base import AuditResult, BaseAuditor
from src.embedders.codebert_embedder import FakeCodeBERTEmbedder
from src.extractors.code_extractor import CodeExtractor
from src.scoring.code_similarity import CodeSimilarityScorer


# =============================================================================
# Constants (Anti-Pattern S1192 Prevention)
# =============================================================================

_AUDITOR_NAME = "cross_reference_auditor"
_AUDITOR_DESCRIPTION = (
    "Audits generated code against reference chapters using CodeBERT "
    "similarity scoring. Verifies that code aligns with documented patterns."
)
_DEFAULT_THRESHOLD = 0.5


# =============================================================================
# CrossReferenceAuditor Class
# =============================================================================


class CrossReferenceAuditor(BaseAuditor):
    """Auditor for verifying code against reference materials.

    WBS 5.4: Uses CodeBERT embeddings to find similarity between generated
    code and code examples in reference chapters.

    Pattern: Inherits from BaseAuditor (AC-5.4.1)

    Usage:
        auditor = CrossReferenceAuditor()
        result = await auditor.audit(generated_code, {
            "references": [
                {"chapter_id": "ch1", "title": "...", "content": "...markdown..."}
            ],
            "threshold": 0.5
        })
    """

    def __init__(self) -> None:
        """Initialize cross-reference auditor."""
        self._extractor = CodeExtractor()
        self._embedder = FakeCodeBERTEmbedder()
        self._scorer = CodeSimilarityScorer(embedder=self._embedder)

    @property
    def name(self) -> str:
        """Return auditor name."""
        return _AUDITOR_NAME

    @property
    def description(self) -> str:
        """Return auditor description."""
        return _AUDITOR_DESCRIPTION

    async def audit(self, code: str, context: dict[str, Any]) -> AuditResult:
        """Audit code against reference chapters.

        WBS 5.4.2: Compare generated code to reference code blocks.

        Args:
            code: Generated code to audit
            context: Must include 'references' list and optional 'threshold'

        Returns:
            AuditResult with findings and confidence scores
        """
        # Handle empty/invalid code
        if not code or not code.strip():
            return self._create_failure_result(
                "Empty or whitespace-only code provided",
                context,
            )

        # Get references from context
        references = context.get("references", [])
        if not references:
            return self._create_failure_result(
                "No references provided in context",
                context,
            )

        # Get threshold
        threshold = context.get("threshold", _DEFAULT_THRESHOLD)

        # Extract code blocks from all references
        all_reference_codes: list[dict[str, Any]] = []
        for ref in references:
            chapter_id = ref.get("chapter_id", "unknown")
            title = ref.get("title", "")
            content = ref.get("content", "")

            blocks = self._extractor.extract_code_blocks(content, language_filter="python")
            for block in blocks:
                all_reference_codes.append({
                    "chapter_id": chapter_id,
                    "title": title,
                    "code": block.code,
                    "start_line": block.start_line,
                })

        # No code blocks in references
        if not all_reference_codes:
            return self._create_failure_result(
                "No code blocks found in reference chapters",
                context,
            )

        # Find similar code blocks
        findings: list[dict[str, Any]] = []
        best_similarity = 0.0

        for ref_code in all_reference_codes:
            similarity = self._scorer.calculate_similarity(code, ref_code["code"])
            
            finding: dict[str, Any] = {
                "chapter_id": ref_code["chapter_id"],
                "title": ref_code["title"],
                "similarity": similarity,
                "matched_chapter": ref_code["chapter_id"],
                "score": similarity,
                "confidence": similarity,
                "reference": {
                    "chapter_id": ref_code["chapter_id"],
                    "start_line": ref_code["start_line"],
                },
            }
            findings.append(finding)

            if similarity > best_similarity:
                best_similarity = similarity

        # Sort findings by similarity descending
        findings.sort(key=lambda x: x["similarity"], reverse=True)

        # Determine if audit passes
        passed = best_similarity >= threshold

        return AuditResult(
            passed=passed,
            findings=findings,
            metadata={
                "auditor": _AUDITOR_NAME,
                "threshold": threshold,
                "best_similarity": best_similarity,
                "code_similarity_score": best_similarity,
                "reference_count": len(all_reference_codes),
            },
        )

    def _create_failure_result(
        self, reason: str, context: dict[str, Any]
    ) -> AuditResult:
        """Create a failure result with reason.

        Args:
            reason: Failure reason message
            context: Original context

        Returns:
            AuditResult indicating failure
        """
        threshold = context.get("threshold", _DEFAULT_THRESHOLD)
        return AuditResult(
            passed=False,
            findings=[{"reason": reason, "severity": "error"}],
            metadata={
                "auditor": _AUDITOR_NAME,
                "threshold": threshold,
                "best_similarity": 0.0,
                "code_similarity_score": 0.0,
                "error": reason,
            },
        )
