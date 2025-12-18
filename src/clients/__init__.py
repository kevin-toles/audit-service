"""Clients package for external service integrations."""

from src.clients.codebert_client import (
    CodeBERTClient,
    CodeBERTClientProtocol,
    FakeCodeBERTClient,
)

__all__ = [
    "CodeBERTClient",
    "CodeBERTClientProtocol",
    "FakeCodeBERTClient",
]
