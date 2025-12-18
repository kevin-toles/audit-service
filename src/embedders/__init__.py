"""Embedders package for code and text embedding generation."""

from src.embedders.codebert_embedder import (
    CodeBERTEmbedder,
    CodeBERTEmbedderProtocol,
    FakeCodeBERTEmbedder,
)

__all__ = [
    "CodeBERTEmbedder",
    "CodeBERTEmbedderProtocol",
    "FakeCodeBERTEmbedder",
]
