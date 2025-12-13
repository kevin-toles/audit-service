"""Tests for health check endpoints.

TDD Task 1.2.9 (RED): Write test_health_endpoint
These tests should FAIL initially until the /health endpoint is implemented.
"""

from __future__ import annotations

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.main import app


class TestHealthEndpoint:
    """Test cases for /health endpoint."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_health_returns_200(self, client: TestClient) -> None:
        """Test that /health returns HTTP 200 OK."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK

    def test_health_returns_healthy_status(self, client: TestClient) -> None:
        """Test that /health returns status='healthy'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_returns_service_name(self, client: TestClient) -> None:
        """Test that /health returns correct service name."""
        response = client.get("/health")
        data = response.json()
        assert data["service"] == "audit-service"

    def test_health_returns_version(self, client: TestClient) -> None:
        """Test that /health returns version string."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_returns_timestamp(self, client: TestClient) -> None:
        """Test that /health returns ISO timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data
        # Should be ISO format
        assert "T" in data["timestamp"]


class TestReadinessEndpoint:
    """Test cases for /health/ready endpoint."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_readiness_returns_200(self, client: TestClient) -> None:
        """Test that /health/ready returns HTTP 200 OK."""
        response = client.get("/health/ready")
        assert response.status_code == status.HTTP_200_OK

    def test_readiness_returns_ready_status(self, client: TestClient) -> None:
        """Test that /health/ready returns status='ready'."""
        response = client.get("/health/ready")
        data = response.json()
        assert data["status"] == "ready"

    def test_readiness_returns_checks(self, client: TestClient) -> None:
        """Test that /health/ready returns checks dictionary."""
        response = client.get("/health/ready")
        data = response.json()
        assert "checks" in data
        assert isinstance(data["checks"], dict)
