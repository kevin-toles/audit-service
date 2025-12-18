"""
EEP-5.2: CodeBERT HTTP Client Tests

TDD RED Phase: Tests for audit-service client calling Code-Orchestrator-Service.

WBS Mapping:
- AC-5.2.1: Use existing CodeBERTRanker from codebert_ranker.py
- AC-5.2.2: Generate 768-dim embeddings for each code block
- AC-5.2.3: Cache embeddings to avoid recomputation (Anti-Pattern #12)

Anti-Patterns Avoided:
- #12: FakeCodeBERTClient for unit testing (no real HTTP calls)
- S1192: Extracted constants for repeated strings

Design Pattern:
- Protocol + FakeClient per CODING_PATTERNS_ANALYSIS.md
- HTTP client calls Code-Orchestrator-Service /api/v1/codebert/embed
"""

from __future__ import annotations

import pytest
import numpy as np
import numpy.typing as npt
from typing import Any, Protocol, runtime_checkable

# =============================================================================
# Constants (Anti-Pattern S1192 Prevention)
# =============================================================================

_EMBEDDING_DIM = 768
_TEST_CODE_PYTHON = "def hello(): print('world')"
_TEST_CODE_JAVASCRIPT = "function hello() { console.log('world'); }"
_DEFAULT_BASE_URL = "http://localhost:8082"


# =============================================================================
# Protocol Definition Tests
# =============================================================================


class TestCodeBERTClientProtocolExists:
    """Tests for CodeBERTClientProtocol existence (AC-5.2.1)."""

    def test_protocol_exists(self) -> None:
        """AC-5.2.1: CodeBERTClientProtocol is defined."""
        from src.clients.codebert_client import CodeBERTClientProtocol
        
        assert CodeBERTClientProtocol is not None

    def test_protocol_has_get_embedding_method(self) -> None:
        """AC-5.2.1: Protocol defines get_embedding method."""
        from src.clients.codebert_client import CodeBERTClientProtocol
        
        # Check method exists in protocol annotations
        assert hasattr(CodeBERTClientProtocol, "get_embedding")

    def test_protocol_has_get_embeddings_batch_method(self) -> None:
        """AC-5.2.1: Protocol defines get_embeddings_batch method."""
        from src.clients.codebert_client import CodeBERTClientProtocol
        
        assert hasattr(CodeBERTClientProtocol, "get_embeddings_batch")

    def test_protocol_has_calculate_similarity_method(self) -> None:
        """AC-5.2.1: Protocol defines calculate_similarity method."""
        from src.clients.codebert_client import CodeBERTClientProtocol
        
        assert hasattr(CodeBERTClientProtocol, "calculate_similarity")


class TestFakeCodeBERTClientExists:
    """Tests for FakeCodeBERTClient existence (Anti-Pattern #12)."""

    def test_fake_client_exists(self) -> None:
        """FakeCodeBERTClient is defined for testing."""
        from src.clients.codebert_client import FakeCodeBERTClient
        
        assert FakeCodeBERTClient is not None

    def test_fake_client_implements_protocol(self) -> None:
        """FakeCodeBERTClient implements CodeBERTClientProtocol."""
        from src.clients.codebert_client import (
            CodeBERTClientProtocol,
            FakeCodeBERTClient,
        )
        
        client = FakeCodeBERTClient()
        assert isinstance(client, CodeBERTClientProtocol)


class TestFakeCodeBERTClientGetEmbedding:
    """Tests for FakeCodeBERTClient.get_embedding (AC-5.2.2)."""

    @pytest.fixture
    def client(self) -> Any:
        """Create fake client for testing."""
        from src.clients.codebert_client import FakeCodeBERTClient
        
        return FakeCodeBERTClient()

    @pytest.mark.asyncio
    async def test_get_embedding_returns_768_dim_vector(self, client: Any) -> None:
        """AC-5.2.2: Returns 768-dimensional embedding."""
        embedding = await client.get_embedding(_TEST_CODE_PYTHON)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (_EMBEDDING_DIM,) or embedding.shape == (1, _EMBEDDING_DIM)

    @pytest.mark.asyncio
    async def test_get_embedding_empty_code_returns_zeros(self, client: Any) -> None:
        """AC-5.2.2: Empty code returns zero vector."""
        embedding = await client.get_embedding("")
        
        assert embedding.shape[0] == _EMBEDDING_DIM or embedding.shape[-1] == _EMBEDDING_DIM
        # All zeros or near-zeros
        assert np.allclose(embedding, 0, atol=0.01)

    @pytest.mark.asyncio
    async def test_get_embedding_deterministic(self, client: Any) -> None:
        """AC-5.2.3: Same code returns same embedding (deterministic)."""
        emb1 = await client.get_embedding(_TEST_CODE_PYTHON)
        emb2 = await client.get_embedding(_TEST_CODE_PYTHON)
        
        np.testing.assert_array_equal(emb1, emb2)


class TestFakeCodeBERTClientBatchEmbedding:
    """Tests for FakeCodeBERTClient.get_embeddings_batch (AC-5.2.2)."""

    @pytest.fixture
    def client(self) -> Any:
        """Create fake client for testing."""
        from src.clients.codebert_client import FakeCodeBERTClient
        
        return FakeCodeBERTClient()

    @pytest.mark.asyncio
    async def test_batch_returns_list_of_embeddings(self, client: Any) -> None:
        """AC-5.2.2: Batch returns list of embeddings."""
        codes = [_TEST_CODE_PYTHON, _TEST_CODE_JAVASCRIPT]
        embeddings = await client.get_embeddings_batch(codes)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2

    @pytest.mark.asyncio
    async def test_batch_embeddings_have_correct_dim(self, client: Any) -> None:
        """AC-5.2.2: Each batch embedding is 768-dim."""
        codes = [_TEST_CODE_PYTHON, _TEST_CODE_JAVASCRIPT]
        embeddings = await client.get_embeddings_batch(codes)
        
        for emb in embeddings:
            flat = emb.flatten()
            assert len(flat) == _EMBEDDING_DIM

    @pytest.mark.asyncio
    async def test_batch_empty_list_returns_empty(self, client: Any) -> None:
        """Batch with empty list returns empty list."""
        embeddings = await client.get_embeddings_batch([])
        
        assert embeddings == []


class TestFakeCodeBERTClientSimilarity:
    """Tests for FakeCodeBERTClient.calculate_similarity."""

    @pytest.fixture
    def client(self) -> Any:
        """Create fake client for testing."""
        from src.clients.codebert_client import FakeCodeBERTClient
        
        return FakeCodeBERTClient()

    @pytest.mark.asyncio
    async def test_similarity_identical_returns_high(self, client: Any) -> None:
        """Identical code returns similarity >= 0.9."""
        similarity = await client.calculate_similarity(
            _TEST_CODE_PYTHON, _TEST_CODE_PYTHON
        )
        
        assert similarity >= 0.9

    @pytest.mark.asyncio
    async def test_similarity_returns_float(self, client: Any) -> None:
        """Similarity returns float between 0 and 1."""
        similarity = await client.calculate_similarity(
            _TEST_CODE_PYTHON, _TEST_CODE_JAVASCRIPT
        )
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0


class TestCodeBERTClientHTTP:
    """Tests for CodeBERTClient HTTP implementation."""

    def test_http_client_exists(self) -> None:
        """CodeBERTClient HTTP implementation exists."""
        from src.clients.codebert_client import CodeBERTClient
        
        assert CodeBERTClient is not None

    def test_http_client_has_base_url(self) -> None:
        """CodeBERTClient accepts base_url parameter."""
        from src.clients.codebert_client import CodeBERTClient
        
        client = CodeBERTClient(base_url=_DEFAULT_BASE_URL)
        assert client._base_url == _DEFAULT_BASE_URL

    def test_http_client_implements_protocol(self) -> None:
        """CodeBERTClient implements CodeBERTClientProtocol."""
        from src.clients.codebert_client import (
            CodeBERTClientProtocol,
            CodeBERTClient,
        )
        
        client = CodeBERTClient(base_url=_DEFAULT_BASE_URL)
        assert isinstance(client, CodeBERTClientProtocol)


class TestCodeBERTClientCaching:
    """Tests for embedding caching (AC-5.2.3)."""

    @pytest.fixture
    def client(self) -> Any:
        """Create fake client with caching enabled."""
        from src.clients.codebert_client import FakeCodeBERTClient
        
        return FakeCodeBERTClient(enable_cache=True)

    @pytest.mark.asyncio
    async def test_cache_enabled_stores_embedding(self, client: Any) -> None:
        """AC-5.2.3: Cache stores embedding after first call."""
        await client.get_embedding(_TEST_CODE_PYTHON)
        
        assert client.cache_size > 0

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self, client: Any) -> None:
        """AC-5.2.3: Second call returns cached embedding."""
        emb1 = await client.get_embedding(_TEST_CODE_PYTHON)
        emb2 = await client.get_embedding(_TEST_CODE_PYTHON)
        
        np.testing.assert_array_equal(emb1, emb2)
        assert client.cache_size == 1  # Only one entry

    @pytest.mark.asyncio
    async def test_cache_clear_removes_entries(self, client: Any) -> None:
        """Cache clear removes all entries."""
        await client.get_embedding(_TEST_CODE_PYTHON)
        assert client.cache_size > 0
        
        client.clear_cache()
        assert client.cache_size == 0


class TestCrossReferenceAuditorUsesClient:
    """Tests for CrossReferenceAuditor using CodeBERTClient (AC-5.2.1)."""

    def test_auditor_accepts_client_injection(self) -> None:
        """AC-5.2.1: Auditor accepts CodeBERTClientProtocol injection."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor
        from src.clients.codebert_client import FakeCodeBERTClient
        
        client = FakeCodeBERTClient()
        auditor = CrossReferenceAuditor(codebert_client=client)
        
        assert auditor._codebert_client is client

    def test_auditor_default_uses_fake_client(self) -> None:
        """Auditor uses FakeCodeBERTClient by default (for testing)."""
        from src.auditors.cross_reference_auditor import CrossReferenceAuditor
        from src.clients.codebert_client import FakeCodeBERTClient
        
        auditor = CrossReferenceAuditor()
        
        # Default should be FakeCodeBERTClient (testable without network)
        assert isinstance(auditor._codebert_client, FakeCodeBERTClient)
