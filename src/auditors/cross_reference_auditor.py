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

import re
from typing import TYPE_CHECKING, Any

from src.auditors.base import AuditResult, AuditStatus, BaseAuditor
from src.extractors.code_extractor import CodeExtractor
from src.scoring.code_similarity import CodeSimilarityScorer

if TYPE_CHECKING:
    from src.clients.codebert_client import CodeBERTClientProtocol


# =============================================================================
# Constants (Anti-Pattern S1192 Prevention)
# =============================================================================

_AUDITOR_NAME = "cross_reference_auditor"
_AUDITOR_DESCRIPTION = (
    "Audits generated code against reference chapters using CodeBERT "
    "similarity scoring. Verifies that code aligns with documented patterns."
)
_DEFAULT_THRESHOLD = 0.5

# AC-5.4.3: Theory→Implementation detection patterns
# Keywords that indicate theoretical/conceptual content
_THEORY_KEYWORDS = [
    "algorithm",
    "concept",
    "theory",
    "principle",
    "pattern",
    "approach",
    "strategy",
    "method",
    "technique",
    "best practice",
    "design",
    "architecture",
]

# Keywords that indicate implementation/practical content
_IMPLEMENTATION_KEYWORDS = [
    "def ",
    "class ",
    "function",
    "import ",
    "from ",
    "return ",
    "async def",
    "await ",
    "self.",
    "raise ",
    "try:",
    "except:",
]

# Regex pattern for detecting theory→implementation relationships
_THEORY_IMPL_PATTERN = re.compile(
    r"(?:implement(?:s|ing|ation)?|based on|following|using|apply(?:ing)?)\s+"
    r"(?:the\s+)?(?:above|this|that|previous|described|mentioned)",
    re.IGNORECASE,
)


# =============================================================================
# CrossReferenceAuditor Class
# =============================================================================


class CrossReferenceAuditor(BaseAuditor):
    """Auditor for verifying code against reference materials.

    WBS 5.4: Uses CodeBERT embeddings to find similarity between generated
    code and code examples in reference chapters.

    Pattern: Inherits from BaseAuditor (AC-5.4.1)

    AC-5.2.1: Accepts CodeBERTClientProtocol for dependency injection.
    Uses FakeCodeBERTClient by default for testability.

    Usage:
        auditor = CrossReferenceAuditor()
        result = await auditor.audit(generated_code, {
            "references": [
                {"chapter_id": "ch1", "title": "...", "content": "...markdown..."}
            ],
            "threshold": 0.5
        })
    """

    def __init__(
        self,
        codebert_client: CodeBERTClientProtocol | None = None,
    ) -> None:
        """Initialize cross-reference auditor.

        Args:
            codebert_client: CodeBERT client for embeddings (uses FakeCodeBERTClient if None)
        """
        self._extractor = CodeExtractor()

        # AC-5.2.1: Use injected client or default to FakeCodeBERTClient
        if codebert_client is None:
            from src.clients.codebert_client import FakeCodeBERTClient
            codebert_client = FakeCodeBERTClient()

        self._codebert_client = codebert_client
        self._scorer = CodeSimilarityScorer()

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
        theory_impl_relationships: list[dict[str, Any]] = []

        for ref in references:
            chapter_id = ref.get("chapter_id", "unknown")
            title = ref.get("title", "")
            content = ref.get("content", "")

            blocks = self._extractor.extract_code_blocks(content, language_filter="python")
            for block in blocks:
                # AC-5.4.3: Detect theory→implementation relationships
                is_theory_impl = self._detect_theory_impl_relationship(
                    block.context_before, block.code
                )

                all_reference_codes.append({
                    "chapter_id": chapter_id,
                    "title": title,
                    "code": block.code,
                    "start_line": block.start_line,
                    "is_theory_impl": is_theory_impl,
                })

                if is_theory_impl:
                    theory_impl_relationships.append({
                        "chapter_id": chapter_id,
                        "context": block.context_before[:100] + "..." if len(block.context_before) > 100 else block.context_before,
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
                "is_theory_impl": ref_code.get("is_theory_impl", False),
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

        # AC-5.4.4: Determine status based on similarity
        passed = best_similarity >= threshold
        status = AuditStatus.VERIFIED if passed else AuditStatus.SUSPICIOUS

        return AuditResult(
            passed=passed,
            findings=findings,
            metadata={
                "auditor": _AUDITOR_NAME,
                "threshold": threshold,
                "best_similarity": best_similarity,
                "code_similarity_score": best_similarity,
                "reference_count": len(all_reference_codes),
                "theory_impl_count": len(theory_impl_relationships),
                "theory_impl_references": theory_impl_relationships,
            },
            status=status,
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
            status=AuditStatus.SUSPICIOUS,
        )

    def _detect_theory_impl_relationship(
        self, context: str, code: str
    ) -> bool:
        """AC-5.4.3: Detect if code implements theoretical concepts.

        Analyzes context before code block and code itself to determine
        if this represents a theory→implementation relationship.

        Args:
            context: Text before the code block
            code: The code content

        Returns:
            True if this appears to be theory→implementation
        """
        # Check context for theory keywords
        context_lower = context.lower()
        has_theory_context = any(
            keyword in context_lower for keyword in _THEORY_KEYWORDS
        )

        # Check code for implementation patterns
        has_impl_code = any(
            keyword in code for keyword in _IMPLEMENTATION_KEYWORDS
        )

        # Check for explicit relationship indicators
        has_explicit_relationship = bool(_THEORY_IMPL_PATTERN.search(context))

        # Theory→impl if context has theory keywords and code has impl patterns
        # OR if there's explicit relationship language
        return (has_theory_context and has_impl_code) or has_explicit_relationship

    def detect_theory_implementation(
        self, content: str
    ) -> list[dict[str, Any]]:
        """AC-5.4.3: Public method to detect theory→implementation relationships.

        Scans content for code blocks that implement theoretical concepts.

        Args:
            content: Markdown content to analyze

        Returns:
            List of detected theory→implementation relationships
        """
        relationships: list[dict[str, Any]] = []
        blocks = self._extractor.extract_code_blocks(content)

        for block in blocks:
            if self._detect_theory_impl_relationship(block.context_before, block.code):
                relationships.append({
                    "start_line": block.start_line,
                    "end_line": block.end_line,
                    "language": block.language,
                    "context_summary": block.context_before[:200] if block.context_before else "",
                    "code_preview": block.code[:100] + "..." if len(block.code) > 100 else block.code,
                    "relationship_type": "theory_to_implementation",
                })

        return relationships
