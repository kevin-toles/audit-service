"""
EEP-5.2: CodeBERT Client for audit-service

HTTP client calling Code-Orchestrator-Service /api/v1/codebert/embed.

WBS Mapping:
- AC-5.2.1: Use existing CodeBERTRanker from codebert_ranker.py (via HTTP)
- AC-5.2.2: Generate 768-dim embeddings for each code block
- AC-5.2.3: Cache embeddings to avoid recomputation (Anti-Pattern #12)

Patterns Applied:
- Protocol typing for dependency injection (CODING_PATTERNS_ANALYSIS.md L126)
- FakeCodeBERTClient for unit testing (CODING_PATTERNS_ANALYSIS.md L155)
- Connection pooling with httpx.AsyncClient (Anti-Pattern #12)

Anti-Patterns Avoided:
- #12: Reuses httpx.AsyncClient, caches embeddings
- S1192: Extracted constants for repeated strings
- S3776: Small focused methods
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


# =============================================================================
# Constants (Anti-Pattern S1192 Prevention)
# =============================================================================

_EMBEDDING_DIM = 768
_DEFAULT_BASE_URL = "http://localhost:8082"
_ENDPOINT_EMBED = "/api/v1/codebert/embed"
_ENDPOINT_EMBED_BATCH = "/api/v1/codebert/embed/batch"
_ENDPOINT_SIMILARITY = "/api/v1/codebert/similarity"
_DEFAULT_TIMEOUT = 30.0


# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class CodeBERTClientProtocol(Protocol):
    """Protocol for CodeBERT client implementations.

    AC-5.2.1: Defines interface for CodeBERT embedding operations.
    Enables duck typing for FakeCodeBERTClient in tests.

    Pattern: Protocol typing per CODING_PATTERNS_ANALYSIS.md L126
    """

    async def get_embedding(self, code: str) -> npt.NDArray[np.floating[Any]]:
        """Generate embedding for code string.

        Args:
            code: Source code to embed

        Returns:
            768-dimensional normalized embedding
        """
        ...

    async def get_embeddings_batch(
        self, codes: list[str]
    ) -> list[npt.NDArray[np.floating[Any]]]:
        """Generate embeddings for multiple code strings.

        Args:
            codes: List of source code strings

        Returns:
            List of 768-dimensional normalized embeddings
        """
        ...

    async def calculate_similarity(self, code_a: str, code_b: str) -> float:
        """Calculate cosine similarity between two code snippets.

        Args:
            code_a: First code snippet
            code_b: Second code snippet

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        ...


# =============================================================================
# Fake Implementation for Testing
# =============================================================================


class FakeCodeBERTClient:
    """Fake CodeBERT client for unit testing.

    Uses deterministic hash-based embeddings instead of HTTP calls.
    Pattern: FakeClient per CODING_PATTERNS_ANALYSIS.md L155

    AC-5.2.3: Supports embedding caching.
    """

    def __init__(self, enable_cache: bool = False) -> None:
        """Initialize fake client.

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

    async def get_embedding(self, code: str) -> npt.NDArray[np.floating[Any]]:
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

    async def get_embeddings_batch(
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

        return [await self.get_embedding(code) for code in codes]

    async def calculate_similarity(self, code_a: str, code_b: str) -> float:
        """Calculate cosine similarity between two code snippets.

        Args:
            code_a: First code snippet
            code_b: Second code snippet

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        emb_a = await self.get_embedding(code_a)
        emb_b = await self.get_embedding(code_b)

        # Flatten
        emb_a = emb_a.flatten()
        emb_b = emb_b.flatten()

        # Calculate norms
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Cosine similarity
        dot_product = np.dot(emb_a, emb_b)
        similarity = dot_product / (norm_a * norm_b)

        return float(max(0.0, min(1.0, similarity)))

    def _hash_to_embedding(self, text: str) -> npt.NDArray[np.floating[Any]]:
        """Convert text hash to deterministic embedding vector.

        Uses SHA-256 hash expanded to fill 768 dimensions.

        Args:
            text: Text to hash

        Returns:
            768-dimensional normalized embedding
        """
        # Create hash
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()

        # Use hash to seed numpy random for deterministic values
        seed = int.from_bytes(hash_bytes[:4], "big")
        rng = np.random.default_rng(seed)

        # Generate 768 values
        embedding = rng.standard_normal(_EMBEDDING_DIM).astype(np.float32)

        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


# =============================================================================
# HTTP Client Implementation
# =============================================================================


class CodeBERTClient:
    """HTTP client for Code-Orchestrator-Service CodeBERT API.

    AC-5.2.1: Calls existing CodeBERTRanker via HTTP endpoint.
    AC-5.2.3: Reuses httpx.AsyncClient for connection pooling.

    Pattern: Anti-Pattern #12 prevention - connection reuse
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        enable_cache: bool = False,
    ) -> None:
        """Initialize HTTP client.

        Args:
            base_url: Code-Orchestrator-Service base URL
            timeout: Request timeout in seconds
            enable_cache: Whether to cache embeddings locally
        """
        self._base_url = base_url
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._cache_enabled = enable_cache
        self._cache: dict[str, npt.NDArray[np.floating[Any]]] = {}

    @property
    def cache_size(self) -> int:
        """Return number of cached embeddings."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx client.

        Anti-Pattern #12: Reuse client for connection pooling.
        """
        if self._client is None:
            if httpx is None:
                raise RuntimeError("httpx not installed")
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_embedding(self, code: str) -> npt.NDArray[np.floating[Any]]:
        """Get embedding from Code-Orchestrator-Service.

        AC-5.2.1: Calls /api/v1/codebert/embed endpoint.
        AC-5.2.2: Returns 768-dimensional embedding.

        Args:
            code: Source code to embed

        Returns:
            768-dimensional embedding vector
        """
        # Check cache
        if self._cache_enabled and code in self._cache:
            return self._cache[code]

        client = await self._get_client()
        response = await client.post(
            _ENDPOINT_EMBED,
            json={"code": code},
        )
        response.raise_for_status()

        data = response.json()
        embedding = np.array(data["embedding"], dtype=np.float32)

        # Cache if enabled
        if self._cache_enabled:
            self._cache[code] = embedding

        return embedding

    async def get_embeddings_batch(
        self, codes: list[str]
    ) -> list[npt.NDArray[np.floating[Any]]]:
        """Get embeddings for multiple codes from Code-Orchestrator-Service.

        AC-5.2.2: Returns 768-dimensional embedding for each code.

        Args:
            codes: List of source code strings

        Returns:
            List of embedding vectors
        """
        if not codes:
            return []

        client = await self._get_client()
        response = await client.post(
            _ENDPOINT_EMBED_BATCH,
            json={"codes": codes},
        )
        response.raise_for_status()

        data = response.json()
        embeddings = [
            np.array(emb, dtype=np.float32)
            for emb in data["embeddings"]
        ]

        return embeddings

    async def calculate_similarity(self, code_a: str, code_b: str) -> float:
        """Calculate similarity via Code-Orchestrator-Service.

        Args:
            code_a: First code snippet
            code_b: Second code snippet

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        client = await self._get_client()
        response = await client.post(
            _ENDPOINT_SIMILARITY,
            json={"code_a": code_a, "code_b": code_b},
        )
        response.raise_for_status()

        data = response.json()
        return float(data["similarity"])
