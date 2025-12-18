"""
EEP-5.3: Code Similarity Scorer

Scores code similarity between code blocks using CodeBERT embeddings.

WBS Mapping:
- 5.3.1: Cosine similarity calculation between code embeddings
- 5.3.2: Pairwise similarity scoring for code block pairs
- 5.3.3: Similarity threshold configuration

Patterns Applied:
- Dependency injection for embedder (testability)
- Pydantic models for structured results
- Numpy for efficient vector operations

Anti-Patterns Avoided:
- #12: Uses injected embedder (FakeCodeBERTEmbedder for tests)
- S1192: Extracted constants
- S3776: Small focused methods
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.embedders.codebert_embedder import CodeBERTEmbedderProtocol


# =============================================================================
# Constants (Anti-Pattern S1192 Prevention)
# =============================================================================

_DEFAULT_THRESHOLD = 0.7  # Default similarity threshold


# =============================================================================
# Pydantic Models
# =============================================================================


class SimilarityResult(BaseModel):
    """Result of a similarity comparison.

    WBS 5.3: Structured output for similarity scoring.
    """

    code_a: str = Field(description="First code snippet")
    code_b: str = Field(description="Second code snippet")
    score: float = Field(ge=0.0, le=1.0, description="Similarity score (0-1)")
    is_similar: bool = Field(description="Whether codes are similar per threshold")


# =============================================================================
# CodeSimilarityScorer Class
# =============================================================================


class CodeSimilarityScorer:
    """Scores code similarity using CodeBERT embeddings.

    WBS 5.3: Provides cosine similarity scoring with configurable threshold.

    Usage:
        # With custom embedder (for testing)
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder
        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder, threshold=0.8)

        # Calculate similarity
        score = scorer.calculate_similarity(code_a, code_b)

        # Check if similar
        is_sim = scorer.is_similar(code_a, code_b)

        # Pairwise matrix
        matrix = scorer.calculate_pairwise_similarity([code1, code2, code3])
    """

    def __init__(
        self,
        embedder: "CodeBERTEmbedderProtocol | None" = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        """Initialize similarity scorer.

        Args:
            embedder: CodeBERT embedder instance (uses default if None)
            threshold: Similarity threshold for is_similar() (0.0-1.0)
        """
        if embedder is None:
            # Create default embedder
            from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

            embedder = FakeCodeBERTEmbedder()

        self._embedder = embedder
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """Return the similarity threshold."""
        return self._threshold

    def calculate_similarity(self, code_a: str, code_b: str) -> float:
        """Calculate cosine similarity between two code snippets.

        WBS 5.3.1: Cosine similarity using CodeBERT embeddings.

        Args:
            code_a: First code snippet
            code_b: Second code snippet

        Returns:
            Cosine similarity score in [0.0, 1.0]
        """
        emb_a = self._embedder.get_embedding(code_a)
        emb_b = self._embedder.get_embedding(code_b)

        return self.similarity_from_embeddings(emb_a, emb_b)

    def similarity_from_embeddings(
        self,
        emb_a: npt.NDArray[np.floating[Any]],
        emb_b: npt.NDArray[np.floating[Any]],
    ) -> float:
        """Calculate cosine similarity from embedding vectors.

        Args:
            emb_a: First embedding vector
            emb_b: Second embedding vector

        Returns:
            Cosine similarity score in [0.0, 1.0]
        """
        # Flatten to 1D
        emb_a = emb_a.flatten()
        emb_b = emb_b.flatten()

        # Calculate norms
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)

        # Handle zero vectors
        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Cosine similarity
        dot_product = np.dot(emb_a, emb_b)
        similarity = dot_product / (norm_a * norm_b)

        # Clamp to [0, 1] (cosine can be negative for opposite vectors)
        return float(max(0.0, min(1.0, similarity)))

    def calculate_pairwise_similarity(
        self, codes: list[str]
    ) -> npt.NDArray[np.floating[Any]]:
        """Calculate pairwise similarity matrix for multiple codes.

        WBS 5.3.2: Pairwise similarity scoring.

        Args:
            codes: List of code snippets

        Returns:
            NxN similarity matrix where matrix[i,j] = similarity(codes[i], codes[j])
        """
        n = len(codes)
        if n == 0:
            return np.array([]).reshape(0, 0)

        # Get all embeddings
        embeddings = self._embedder.get_embeddings_batch(codes)

        # Build similarity matrix
        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i, n):
                sim = self.similarity_from_embeddings(embeddings[i], embeddings[j])
                matrix[i, j] = sim
                matrix[j, i] = sim  # Symmetric

        return matrix

    def is_similar(self, code_a: str, code_b: str) -> bool:
        """Check if two code snippets are similar per threshold.

        WBS 5.3.3: Threshold-based similarity check.

        Args:
            code_a: First code snippet
            code_b: Second code snippet

        Returns:
            True if similarity >= threshold
        """
        score = self.calculate_similarity(code_a, code_b)
        return score >= self._threshold

    def find_similar_codes(
        self, query_code: str, codes: list[str]
    ) -> list[tuple[int, float]]:
        """Find codes similar to query above threshold.

        WBS 5.3.3: Find similar code snippets.

        Args:
            query_code: Code to compare against
            codes: List of candidate codes

        Returns:
            List of (index, score) tuples for codes above threshold, sorted by score
        """
        if not codes:
            return []

        query_emb = self._embedder.get_embedding(query_code)
        code_embeddings = self._embedder.get_embeddings_batch(codes)

        similar: list[tuple[int, float]] = []
        for i, emb in enumerate(code_embeddings):
            score = self.similarity_from_embeddings(query_emb, emb)
            if score >= self._threshold:
                similar.append((i, score))

        # Sort by score descending
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar

    def compare_codes(self, code_a: str, code_b: str) -> SimilarityResult:
        """Compare two codes and return structured result.

        Args:
            code_a: First code snippet
            code_b: Second code snippet

        Returns:
            SimilarityResult with score and is_similar flag
        """
        score = self.calculate_similarity(code_a, code_b)
        return SimilarityResult(
            code_a=code_a,
            code_b=code_b,
            score=score,
            is_similar=score >= self._threshold,
        )
