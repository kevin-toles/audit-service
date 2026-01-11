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
# Real Implementation - Calls Code-Orchestrator-Service
# =============================================================================


class CodeBERTEmbedder:
    """Production CodeBERT embedder calling Code-Orchestrator-Service.

    FIX (2026-01-10): Now actually calls Code-Orchestrator HTTP API
    instead of delegating to FakeCodeBERTEmbedder.
    
    Uses synchronous requests (httpx) to match the sync Protocol interface.
    """

    def __init__(
        self,
        service_url: str | None = None,
        enable_cache: bool = False,
    ) -> None:
        """Initialize CodeBERT embedder.

        Args:
            service_url: URL for Code-Orchestrator-Service (default: localhost:8083)
            enable_cache: Whether to cache embeddings locally
        """
        self._service_url = service_url or "http://localhost:8083"
        self._cache_enabled = enable_cache
        self._cache: dict[str, npt.NDArray[np.floating[Any]]] = {}
        
        # Import httpx for sync HTTP calls
        try:
            import httpx
            self._httpx = httpx
        except ImportError:
            self._httpx = None

    @property
    def cache_size(self) -> int:
        """Return number of cached embeddings."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()

    def get_embedding(self, code: str) -> npt.NDArray[np.floating[Any]]:
        """Generate embedding via Code-Orchestrator-Service.

        Args:
            code: Source code to embed

        Returns:
            768-dimensional normalized embedding
        """
        # Handle empty input
        code_stripped = code.strip()
        if not code_stripped:
            return np.zeros(_EMBEDDING_DIM, dtype=np.float32)
        
        # Check cache
        if self._cache_enabled and code in self._cache:
            return self._cache[code]
        
        # Call Code-Orchestrator API
        if self._httpx is None:
            raise RuntimeError("httpx not installed - required for real CodeBERT embeddings")
        
        try:
            response = self._httpx.post(
                f"{self._service_url}/api/v1/codebert/embed",
                json={"code": code},
                timeout=30.0,
            )
            response.raise_for_status()
            
            data = response.json()
            embedding = np.array(data["embedding"], dtype=np.float32)
            
            # Cache if enabled
            if self._cache_enabled:
                self._cache[code] = embedding
            
            return embedding
            
        except Exception as e:
            # Log error and return zero embedding on failure
            import logging
            logging.getLogger(__name__).warning(
                f"CodeBERT embedding failed, returning zeros: {e}"
            )
            return np.zeros(_EMBEDDING_DIM, dtype=np.float32)

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
        
        # Try batch API first
        if self._httpx is not None:
            try:
                response = self._httpx.post(
                    f"{self._service_url}/api/v1/codebert/embed/batch",
                    json={"codes": codes},
                    timeout=60.0,
                )
                response.raise_for_status()
                
                data = response.json()
                return [
                    np.array(emb, dtype=np.float32)
                    for emb in data["embeddings"]
                ]
            except Exception:
                pass  # Fall back to individual calls
        
        # Fallback: call get_embedding for each
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
        return self.get_embeddings_batch([b.code for b in blocks])
