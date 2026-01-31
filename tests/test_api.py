"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from app.main import app
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_liveness(self, client):
        """Test liveness endpoint."""
        response = client.get("/live")
        assert response.status_code == 200
        assert response.json()["alive"] is True

    def test_readiness_without_model(self, client):
        """Test readiness endpoint when model is not loaded."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        # Without model, ready should be False
        assert "ready" in data

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime_seconds" in data
        assert "version" in data


class TestChatEndpoints:
    """Test chat completion endpoints."""

    def test_chat_without_model(self, client):
        """Test chat endpoint when model is not loaded."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            },
        )
        # Should return 503 when model is not loaded
        assert response.status_code == 503

    def test_models_endpoint(self, client):
        """Test models listing endpoint."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)


class TestValidation:
    """Test request validation."""

    def test_invalid_temperature(self, client):
        """Test invalid temperature value."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 3.0,  # Invalid: > 2.0
            },
        )
        assert response.status_code == 422

    def test_invalid_max_tokens(self, client):
        """Test invalid max_tokens value."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": -1,  # Invalid: < 1
            },
        )
        assert response.status_code == 422

    def test_empty_messages(self, client):
        """Test empty messages list."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [],
            },
        )
        assert response.status_code == 422


class TestRateLimiting:
    """Test rate limiting middleware."""

    def test_rate_limit_headers(self, client):
        """Test that rate limit headers are present."""
        response = client.get("/v1/models")
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
