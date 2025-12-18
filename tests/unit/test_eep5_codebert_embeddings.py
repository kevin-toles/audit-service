"""
EEP-5.2 Tests: CodeBERT Embeddings for Code

TDD RED Phase: Write failing tests first.

Tests for generating CodeBERT embeddings for extracted code blocks.

WBS Mapping:
- 5.2.1: CodeBERT embedding generation (768-dimensional vectors)
- 5.2.2: Batch embedding for multiple code blocks
- 5.2.3: Embedding caching/optimization

Anti-Patterns Avoided:
- #12: FakeCodeBERTEmbedder for testing (no real model loading)
- S1192: Constants for repeated strings
- S3776: Small, focused test methods

Design Decision:
- Uses FakeCodeBERTEmbedder for unit tests
- Real implementation calls Code-Orchestrator-Service via HTTP client
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

if TYPE_CHECKING:
    pass

# =============================================================================
# Constants (Anti-Pattern S1192 Prevention)
# =============================================================================

_SAMPLE_CODE_BLOCK = """def chunk_text(text: str, chunk_size: int = 100) -> list[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]"""

_SAMPLE_CODE_BLOCK_2 = """function chunkText(text, chunkSize = 100) {
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
        chunks.push(text.slice(i, i + chunkSize));
    }
    return chunks;
}"""

_EMBEDDING_DIM = 768  # CodeBERT base model dimension


# =============================================================================
# EEP-5.2.1: CodeBERT Embedding Generation Tests
# =============================================================================


class TestCodeBERTEmbedderClass:
    """Tests for CodeBERTEmbedder class existence and interface."""

    def test_codebert_embedder_class_exists(self) -> None:
        """CodeBERTEmbedder class exists and is importable."""
        from src.embedders.codebert_embedder import CodeBERTEmbedder

        assert CodeBERTEmbedder is not None

    def test_codebert_embedder_protocol_exists(self) -> None:
        """CodeBERTEmbedderProtocol defines the interface."""
        from src.embedders.codebert_embedder import CodeBERTEmbedderProtocol

        assert CodeBERTEmbedderProtocol is not None

    def test_fake_codebert_embedder_exists(self) -> None:
        """FakeCodeBERTEmbedder exists for testing."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        assert FakeCodeBERTEmbedder is not None

    def test_fake_embedder_implements_protocol(self) -> None:
        """FakeCodeBERTEmbedder implements CodeBERTEmbedderProtocol."""
        from src.embedders.codebert_embedder import (
            CodeBERTEmbedderProtocol,
            FakeCodeBERTEmbedder,
        )

        fake = FakeCodeBERTEmbedder()
        assert isinstance(fake, CodeBERTEmbedderProtocol)


class TestEmbeddingGeneration:
    """Tests for single embedding generation."""

    def test_get_embedding_method_exists(self) -> None:
        """get_embedding() method exists.

        AC-5.2.1: Generate 768-dimensional embeddings.
        """
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        assert hasattr(embedder, "get_embedding")
        assert callable(embedder.get_embedding)

    def test_get_embedding_returns_numpy_array(self) -> None:
        """get_embedding() returns a numpy array."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        embedding = embedder.get_embedding(_SAMPLE_CODE_BLOCK)

        assert isinstance(embedding, np.ndarray)

    def test_get_embedding_has_correct_dimension(self) -> None:
        """get_embedding() returns 768-dimensional vector.

        AC-5.2.1: 768-dimensional embeddings (CodeBERT base).
        """
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        embedding = embedder.get_embedding(_SAMPLE_CODE_BLOCK)

        assert embedding.shape == (_EMBEDDING_DIM,)

    def test_get_embedding_is_normalized(self) -> None:
        """Embedding vector is L2-normalized (unit length)."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        embedding = embedder.get_embedding(_SAMPLE_CODE_BLOCK)

        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=0.01)

    def test_get_embedding_deterministic_for_same_input(self) -> None:
        """Same code produces same embedding (deterministic)."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        emb1 = embedder.get_embedding(_SAMPLE_CODE_BLOCK)
        emb2 = embedder.get_embedding(_SAMPLE_CODE_BLOCK)

        assert np.allclose(emb1, emb2)

    def test_get_embedding_different_for_different_input(self) -> None:
        """Different code produces different embeddings."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        emb1 = embedder.get_embedding(_SAMPLE_CODE_BLOCK)
        emb2 = embedder.get_embedding(_SAMPLE_CODE_BLOCK_2)

        # Should not be identical
        assert not np.allclose(emb1, emb2)


# =============================================================================
# EEP-5.2.2: Batch Embedding Tests
# =============================================================================


class TestBatchEmbedding:
    """Tests for batch embedding generation."""

    def test_get_embeddings_batch_method_exists(self) -> None:
        """get_embeddings_batch() method exists.

        AC-5.2.2: Batch embedding for multiple code blocks.
        """
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        assert hasattr(embedder, "get_embeddings_batch")
        assert callable(embedder.get_embeddings_batch)

    def test_batch_embedding_returns_list(self) -> None:
        """get_embeddings_batch() returns list of arrays."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        codes = [_SAMPLE_CODE_BLOCK, _SAMPLE_CODE_BLOCK_2]
        embeddings = embedder.get_embeddings_batch(codes)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 2

    def test_batch_embedding_correct_dimensions(self) -> None:
        """Each batch embedding has correct dimension."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        codes = [_SAMPLE_CODE_BLOCK, _SAMPLE_CODE_BLOCK_2]
        embeddings = embedder.get_embeddings_batch(codes)

        for emb in embeddings:
            assert emb.shape == (_EMBEDDING_DIM,)

    def test_batch_embedding_empty_list(self) -> None:
        """get_embeddings_batch() handles empty list."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        embeddings = embedder.get_embeddings_batch([])

        assert embeddings == []

    def test_batch_embedding_matches_single(self) -> None:
        """Batch embedding results match individual calls."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        codes = [_SAMPLE_CODE_BLOCK, _SAMPLE_CODE_BLOCK_2]

        batch_emb = embedder.get_embeddings_batch(codes)
        single_emb1 = embedder.get_embedding(codes[0])
        single_emb2 = embedder.get_embedding(codes[1])

        assert np.allclose(batch_emb[0], single_emb1)
        assert np.allclose(batch_emb[1], single_emb2)


# =============================================================================
# EEP-5.2.3: Embedding Caching/Optimization Tests
# =============================================================================


class TestEmbeddingCaching:
    """Tests for embedding caching and optimization."""

    def test_embedder_has_cache_option(self) -> None:
        """Embedder can be initialized with caching.

        AC-5.2.3: Embedding caching/optimization.
        """
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder(enable_cache=True)
        assert embedder._cache_enabled is True

    def test_cache_stores_embeddings(self) -> None:
        """Cache stores computed embeddings."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder(enable_cache=True)
        _ = embedder.get_embedding(_SAMPLE_CODE_BLOCK)

        # Should have cache entry
        assert embedder.cache_size > 0

    def test_cache_returns_cached_embedding(self) -> None:
        """Cached embedding is returned on second call."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder(enable_cache=True)

        emb1 = embedder.get_embedding(_SAMPLE_CODE_BLOCK)
        emb2 = embedder.get_embedding(_SAMPLE_CODE_BLOCK)

        assert np.allclose(emb1, emb2)

    def test_cache_disabled_by_default(self) -> None:
        """Cache is disabled by default."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        assert embedder._cache_enabled is False

    def test_clear_cache_method_exists(self) -> None:
        """clear_cache() method exists."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder(enable_cache=True)
        assert hasattr(embedder, "clear_cache")
        assert callable(embedder.clear_cache)

    def test_clear_cache_empties_cache(self) -> None:
        """clear_cache() removes all cached embeddings."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder(enable_cache=True)
        _ = embedder.get_embedding(_SAMPLE_CODE_BLOCK)
        assert embedder.cache_size > 0

        embedder.clear_cache()
        assert embedder.cache_size == 0


# =============================================================================
# CodeBlock Embedding Integration Tests
# =============================================================================


class TestCodeBlockEmbedding:
    """Tests for embedding CodeBlock objects."""

    def test_embed_code_block_method_exists(self) -> None:
        """embed_code_block() method for CodeBlock objects."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        assert hasattr(embedder, "embed_code_block")
        assert callable(embedder.embed_code_block)

    def test_embed_code_block_returns_embedding(self) -> None:
        """embed_code_block() returns embedding for CodeBlock."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder
        from src.extractors.code_extractor import CodeBlock

        block = CodeBlock(
            code=_SAMPLE_CODE_BLOCK,
            language="python",
            start_line=1,
            end_line=2,
            index=0,
            context_before="",
            context_after="",
        )

        embedder = FakeCodeBERTEmbedder()
        embedding = embedder.embed_code_block(block)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (_EMBEDDING_DIM,)

    def test_embed_code_blocks_batch(self) -> None:
        """embed_code_blocks() handles list of CodeBlock objects."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder
        from src.extractors.code_extractor import CodeBlock

        blocks = [
            CodeBlock(
                code=_SAMPLE_CODE_BLOCK,
                language="python",
                start_line=1,
                end_line=2,
                index=0,
                context_before="",
                context_after="",
            ),
            CodeBlock(
                code=_SAMPLE_CODE_BLOCK_2,
                language="javascript",
                start_line=10,
                end_line=15,
                index=1,
                context_before="",
                context_after="",
            ),
        ]

        embedder = FakeCodeBERTEmbedder()
        embeddings = embedder.embed_code_blocks(blocks)

        assert len(embeddings) == 2
        for emb in embeddings:
            assert emb.shape == (_EMBEDDING_DIM,)


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEmbeddingEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_code_returns_zero_vector(self) -> None:
        """Empty code string returns zero vector."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        embedding = embedder.get_embedding("")

        assert embedding.shape == (_EMBEDDING_DIM,)
        # Zero vector or near-zero
        assert np.linalg.norm(embedding) < 0.1

    def test_whitespace_only_code(self) -> None:
        """Whitespace-only code handled gracefully."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        embedder = FakeCodeBERTEmbedder()
        embedding = embedder.get_embedding("   \n\n   ")

        assert embedding.shape == (_EMBEDDING_DIM,)

    def test_very_long_code_truncated(self) -> None:
        """Very long code is truncated (max 512 tokens)."""
        from src.embedders.codebert_embedder import FakeCodeBERTEmbedder

        # Create very long code (way more than 512 tokens)
        long_code = "x = 1\n" * 1000

        embedder = FakeCodeBERTEmbedder()
        embedding = embedder.get_embedding(long_code)

        # Should still return valid embedding
        assert embedding.shape == (_EMBEDDING_DIM,)

