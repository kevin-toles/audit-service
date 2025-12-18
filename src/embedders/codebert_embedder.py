"""
EEP-5.2: CodeBERT Embeddings for Code

Generates CodeBERT embeddings for extracted code blocks.

WBS Mapping:
- 5.2.1: CodeBERT embedding generation (768-dimensional vectors)
- 5.2.2: Batch embedding for multiple code blocks
- 5.2.3: Embedding caching/optimization

Patterns Applied:
- Protocol typing for dependency injection
- FakeCodeBERTEmbedder for unit testing (Anti-Pattern #12)
- Caching for performance optimization

Anti-Patterns Avoided:
- #12: FakeCodeBERTEmbedder avoids real model in unit tests
- S1192: Extracted constants for repeated values
- S3776: Small focused methods

Design Decision:
- FakeCodeBERTEmbedder uses deterministic hash-based embeddings
- Real implementation would call Code-Orchestrator-Service
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from src.extractors.code_extractor import CodeBlock


# =============================================================================
# Constants (Anti-Pattern S1192 Prevention)
# =============================================================================

_EMBEDDING_DIM = 768  # CodeBERT base model dimension
_MAX_TOKENS = 512  # Max tokens for CodeBERT


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class CodeBERTEmbedderProtocol(Protocol):
    """Protocol for CodeBERT embedding generation.

    Defines the interface for both real and fake embedders.
    """

    def get_embedding(self, code: str) -> npt.NDArray[np.floating[Any]]:
        """Generate embedding for code string.

        Args:
            code: Source code to embed

        Returns:
            768-dimensional normalized embedding
        """
        ...

    def get_embeddings_batch(
        self, codes: list[str]
    ) -> list[npt.NDArray[np.floating[Any]]]:
        """Generate embeddings for multiple code strings.

        Args:
            codes: List of source code strings

        Returns:
            List of 768-dimensional normalized embeddings
        """
        ...


# =============================================================================
# Fake Implementation for Testing
# =============================================================================


class FakeCodeBERTEmbedder:
    """Fake CodeBERT embedder for unit testing.

    Uses deterministic hash-based embeddings instead of real model.
    This avoids loading HuggingFace models during unit tests.

    Pattern: FakeClient per CODING_PATTERNS_ANALYSIS.md (Anti-Pattern #12)
    """

    def __init__(self, enable_cache: bool = False) -> None:
        """Initialize fake embedder.

        Args:
            enable_cache: Whether to cache computed embeddings
        """
        self._cache_enabled = enable_cache
        self._cache: dict[str, npt.NDArray[np.floating[Any]]] = {}

    @property
    def cache_size(self) -> int:
        """Return number of cached embeddings."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()

    def get_embedding(self, code: str) -> npt.NDArray[np.floating[Any]]:
        """Generate deterministic embedding from code hash.

        Args:
            code: Source code to embed

        Returns:
            768-dimensional normalized embedding
        """
        # Handle empty/whitespace input
        code_stripped = code.strip()
        if not code_stripped:
            return np.zeros(_EMBEDDING_DIM, dtype=np.float32)

        # Check cache
        if self._cache_enabled and code in self._cache:
            return self._cache[code]

        # Generate deterministic embedding from hash
        embedding = self._hash_to_embedding(code_stripped)

        # Cache if enabled
        if self._cache_enabled:
            self._cache[code] = embedding

        return embedding

    def get_embeddings_batch(
        self, codes: list[str]
    ) -> list[npt.NDArray[np.floating[Any]]]:
        """Generate embeddings for multiple code strings.

        Args:
            codes: List of source code strings

        Returns:
            List of 768-dimensional normalized embeddings
        """
        if not codes:
            return []

        return [self.get_embedding(code) for code in codes]

    def embed_code_block(
        self, block: "CodeBlock"
    ) -> npt.NDArray[np.floating[Any]]:
        """Generate embedding for a CodeBlock object.

        Args:
            block: CodeBlock to embed

        Returns:
            768-dimensional normalized embedding
        """
        return self.get_embedding(block.code)

    def embed_code_blocks(
        self, blocks: list["CodeBlock"]
    ) -> list[npt.NDArray[np.floating[Any]]]:
        """Generate embeddings for multiple CodeBlock objects.

        Args:
            blocks: List of CodeBlocks to embed

        Returns:
            List of 768-dimensional normalized embeddings
        """
        codes = [block.code for block in blocks]
        return self.get_embeddings_batch(codes)

    def _hash_to_embedding(self, text: str) -> npt.NDArray[np.floating[Any]]:
        """Convert text hash to deterministic embedding vector.

        Uses SHA-256 hash expanded to fill 768 dimensions.
        Produces consistent embeddings for same input.

        Args:
            text: Text to hash

        Returns:
            768-dimensional normalized embedding
        """
        # Create hash
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()

        # Expand hash to fill 768 dimensions
        # SHA-256 gives 32 bytes = 256 bits
        # Need 768 floats, so repeat and expand
        rng = np.random.default_rng(seed=int.from_bytes(hash_bytes[:8], "big"))
        embedding = rng.standard_normal(_EMBEDDING_DIM).astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


# =============================================================================
# Real Implementation (placeholder for production)
# =============================================================================


class CodeBERTEmbedder:
    """Production CodeBERT embedder.

    In production, this would call Code-Orchestrator-Service
    or load the CodeBERT model directly.

    For now, delegates to FakeCodeBERTEmbedder.
    """

    def __init__(
        self,
        service_url: str | None = None,
        enable_cache: bool = False,
    ) -> None:
        """Initialize CodeBERT embedder.

        Args:
            service_url: URL for Code-Orchestrator-Service (optional)
            enable_cache: Whether to cache embeddings
        """
        self._service_url = service_url
        self._cache_enabled = enable_cache
        # For now, use fake implementation
        self._fake = FakeCodeBERTEmbedder(enable_cache=enable_cache)

    @property
    def cache_size(self) -> int:
        """Return number of cached embeddings."""
        return self._fake.cache_size

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self._fake.clear_cache()

    def get_embedding(self, code: str) -> npt.NDArray[np.floating[Any]]:
        """Generate embedding for code string.

        Args:
            code: Source code to embed

        Returns:
            768-dimensional normalized embedding
        """
        # TODO: In production, call Code-Orchestrator-Service
        return self._fake.get_embedding(code)

    def get_embeddings_batch(
        self, codes: list[str]
    ) -> list[npt.NDArray[np.floating[Any]]]:
        """Generate embeddings for multiple code strings.

        Args:
            codes: List of source code strings

        Returns:
            List of 768-dimensional normalized embeddings
        """
        return self._fake.get_embeddings_batch(codes)

    def embed_code_block(
        self, block: "CodeBlock"
    ) -> npt.NDArray[np.floating[Any]]:
        """Generate embedding for a CodeBlock object.

        Args:
            block: CodeBlock to embed

        Returns:
            768-dimensional normalized embedding
        """
        return self._fake.embed_code_block(block)

    def embed_code_blocks(
        self, blocks: list["CodeBlock"]
    ) -> list[npt.NDArray[np.floating[Any]]]:
        """Generate embeddings for multiple CodeBlock objects.

        Args:
            blocks: List of CodeBlocks to embed

        Returns:
            List of 768-dimensional normalized embeddings
        """
        return self._fake.embed_code_blocks(blocks)
