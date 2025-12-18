"""
EEP-5.3 Tests: Code Similarity Scorer

TDD RED Phase: Write failing tests first.

Tests for scoring code similarity between code blocks using CodeBERT embeddings.

WBS Mapping:
- 5.3.1: Cosine similarity calculation between code embeddings
- 5.3.2: Pairwise similarity scoring for code block pairs
- 5.3.3: Similarity threshold configuration

Anti-Patterns Avoided:
- #12: Uses FakeCodeBERTEmbedder (no real model)
- S1192: Constants for repeated strings
- S3776: Small, focused test methods

Depends On: EEP-3 multi-signal scores (this scorer can be one signal)
"""

from __future__ import annotations

import numpy as np
import pytest


# =============================================================================
# Constants (Anti-Pattern S1192 Prevention)
# =============================================================================

_SAMPLE_CODE_1 = """def chunk_text(text: str, chunk_size: int = 100) -> list[str]:
    \"\"\"Split text into chunks.\"\"\"
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]"""

_SAMPLE_CODE_2 = """def split_text(text: str, size: int = 100) -> list[str]:
    \"\"\"Split text into smaller parts.\"\"\"
    result = []
    for i in range(0, len(text), size):
        result.append(text[i:i+size])
    return result"""

_SAMPLE_CODE_3 = """function calculateSum(numbers) {
    return numbers.reduce((a, b) => a + b, 0);
}"""

_HIGH_SIMILARITY_THRESHOLD = 0.8
_LOW_SIMILARITY_THRESHOLD = 0.3


# =============================================================================
# EEP-5.3.1: Cosine Similarity Calculation Tests
# =============================================================================


class TestCodeSimilarityScorerClass:
    """Tests for CodeSimilarityScorer class existence and interface."""

    def test_code_similarity_scorer_class_exists(self) -> None:
        """CodeSimilarityScorer class exists and is importable."""
        from src.scoring.code_similarity import CodeSimilarityScorer

        assert CodeSimilarityScorer is not None

    def test_scorer_initializes_with_embedder(self) -> None:
        """Scorer accepts an embedder in constructor."""
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder)
        assert scorer is not None

    def test_scorer_has_default_embedder(self) -> None:
        """Scorer creates default embedder if none provided."""
        from src.scoring.code_similarity import CodeSimilarityScorer

        scorer = CodeSimilarityScorer()
        assert scorer is not None


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_calculate_similarity_method_exists(self) -> None:
        """calculate_similarity() method exists.

        AC-5.3.1: Calculate cosine similarity between code embeddings.
        """
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder)

        assert hasattr(scorer, "calculate_similarity")
        assert callable(scorer.calculate_similarity)

    def test_similarity_returns_float(self) -> None:
        """calculate_similarity() returns a float score."""
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder)

        score = scorer.calculate_similarity(_SAMPLE_CODE_1, _SAMPLE_CODE_2)
        assert isinstance(score, float)

    def test_similarity_in_valid_range(self) -> None:
        """Similarity score is in [0.0, 1.0] range."""
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder)

        score = scorer.calculate_similarity(_SAMPLE_CODE_1, _SAMPLE_CODE_2)
        assert 0.0 <= score <= 1.0

    def test_identical_code_has_high_similarity(self) -> None:
        """Identical code has similarity of 1.0."""
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder)

        score = scorer.calculate_similarity(_SAMPLE_CODE_1, _SAMPLE_CODE_1)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_calculate_similarity_from_embeddings(self) -> None:
        """Can calculate similarity directly from embedding vectors."""
        from src.scoring.code_similarity import CodeSimilarityScorer

        scorer = CodeSimilarityScorer()

        # Create two similar vectors
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])

        score = scorer.similarity_from_embeddings(emb1, emb2)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_vectors_zero_similarity(self) -> None:
        """Orthogonal vectors have zero similarity."""
        from src.scoring.code_similarity import CodeSimilarityScorer

        scorer = CodeSimilarityScorer()

        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])

        score = scorer.similarity_from_embeddings(emb1, emb2)
        assert score == pytest.approx(0.0, abs=0.01)


# =============================================================================
# EEP-5.3.2: Pairwise Similarity Tests
# =============================================================================


class TestPairwiseSimilarity:
    """Tests for pairwise similarity scoring."""

    def test_pairwise_similarity_method_exists(self) -> None:
        """calculate_pairwise_similarity() method exists.

        AC-5.3.2: Pairwise similarity scoring for code block pairs.
        """
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder)

        assert hasattr(scorer, "calculate_pairwise_similarity")
        assert callable(scorer.calculate_pairwise_similarity)

    def test_pairwise_returns_matrix(self) -> None:
        """Pairwise similarity returns a similarity matrix."""
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder)

        codes = [_SAMPLE_CODE_1, _SAMPLE_CODE_2, _SAMPLE_CODE_3]
        matrix = scorer.calculate_pairwise_similarity(codes)

        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)

    def test_pairwise_diagonal_is_one(self) -> None:
        """Diagonal of pairwise matrix is 1.0 (self-similarity)."""
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder)

        codes = [_SAMPLE_CODE_1, _SAMPLE_CODE_2, _SAMPLE_CODE_3]
        matrix = scorer.calculate_pairwise_similarity(codes)

        for i in range(len(codes)):
            assert matrix[i, i] == pytest.approx(1.0, abs=0.01)

    def test_pairwise_is_symmetric(self) -> None:
        """Pairwise similarity matrix is symmetric."""
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder)

        codes = [_SAMPLE_CODE_1, _SAMPLE_CODE_2, _SAMPLE_CODE_3]
        matrix = scorer.calculate_pairwise_similarity(codes)

        assert np.allclose(matrix, matrix.T)

    def test_pairwise_empty_list(self) -> None:
        """Pairwise similarity handles empty list."""
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder)

        matrix = scorer.calculate_pairwise_similarity([])
        assert matrix.shape == (0, 0)


# =============================================================================
# EEP-5.3.3: Similarity Threshold Tests
# =============================================================================


class TestSimilarityThreshold:
    """Tests for similarity threshold configuration."""

    def test_scorer_has_default_threshold(self) -> None:
        """Scorer has a default similarity threshold.

        AC-5.3.3: Similarity threshold configuration.
        """
        from src.scoring.code_similarity import CodeSimilarityScorer

        scorer = CodeSimilarityScorer()
        assert hasattr(scorer, "threshold")
        assert isinstance(scorer.threshold, float)

    def test_custom_threshold_at_initialization(self) -> None:
        """Threshold can be set at initialization."""
        from src.scoring.code_similarity import CodeSimilarityScorer

        scorer = CodeSimilarityScorer(threshold=0.75)
        assert scorer.threshold == 0.75

    def test_is_similar_method_exists(self) -> None:
        """is_similar() method uses threshold for comparison."""
        from src.scoring.code_similarity import CodeSimilarityScorer

        scorer = CodeSimilarityScorer()
        assert hasattr(scorer, "is_similar")
        assert callable(scorer.is_similar)

    def test_is_similar_respects_threshold(self) -> None:
        """is_similar() returns True if above threshold."""
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder, threshold=0.5)

        # Identical code should be similar (score = 1.0)
        assert scorer.is_similar(_SAMPLE_CODE_1, _SAMPLE_CODE_1) is True

    def test_find_similar_codes_method_exists(self) -> None:
        """find_similar_codes() finds codes above threshold."""
        from src.scoring.code_similarity import CodeSimilarityScorer

        scorer = CodeSimilarityScorer()
        assert hasattr(scorer, "find_similar_codes")
        assert callable(scorer.find_similar_codes)

    def test_find_similar_returns_list(self) -> None:
        """find_similar_codes() returns list of similar code indices."""
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder, threshold=0.5)

        codes = [_SAMPLE_CODE_1, _SAMPLE_CODE_2, _SAMPLE_CODE_3]
        # Find codes similar to _SAMPLE_CODE_1
        similar = scorer.find_similar_codes(_SAMPLE_CODE_1, codes)

        assert isinstance(similar, list)


# =============================================================================
# SimilarityResult Model Tests
# =============================================================================


class TestSimilarityResultModel:
    """Tests for SimilarityResult Pydantic model."""

    def test_similarity_result_model_exists(self) -> None:
        """SimilarityResult model exists."""
        from src.scoring.code_similarity import SimilarityResult

        assert SimilarityResult is not None

    def test_similarity_result_has_required_fields(self) -> None:
        """SimilarityResult has score and is_similar fields."""
        from src.scoring.code_similarity import SimilarityResult

        result = SimilarityResult(
            code_a="code1",
            code_b="code2",
            score=0.85,
            is_similar=True,
        )

        assert result.score == 0.85
        assert result.is_similar is True

    def test_compare_codes_returns_result(self) -> None:
        """compare_codes() returns SimilarityResult."""
        from src.scoring.code_similarity import CodeSimilarityScorer, SimilarityResult
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder)

        result = scorer.compare_codes(_SAMPLE_CODE_1, _SAMPLE_CODE_2)
        assert isinstance(result, SimilarityResult)


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestSimilarityScorerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_code_similarity(self) -> None:
        """Empty code strings are handled gracefully."""
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder)

        score = scorer.calculate_similarity("", "")
        assert isinstance(score, float)

    def test_whitespace_only_code(self) -> None:
        """Whitespace-only code is handled."""
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder)

        score = scorer.calculate_similarity("   \n  ", "   ")
        assert isinstance(score, float)

    def test_single_code_pairwise(self) -> None:
        """Pairwise with single code returns 1x1 matrix."""
        from src.scoring.code_similarity import CodeSimilarityScorer
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        scorer = CodeSimilarityScorer(embedder=embedder)

        matrix = scorer.calculate_pairwise_similarity([_SAMPLE_CODE_1])
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == pytest.approx(1.0, abs=0.01)

