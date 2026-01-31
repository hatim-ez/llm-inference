"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from app.models import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPart,
    HealthStatus,
    Role,
)


class TestChatMessage:
    """Test ChatMessage model."""

    def test_simple_message(self):
        """Test simple text message."""
        msg = ChatMessage(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_multimodal_message(self):
        """Test multimodal message with text and image."""
        content = [
            ContentPart(type="text", text="What's in this image?"),
            ContentPart(
                type="image_url",
                image_url={"url": "https://example.com/image.jpg"},
            ),
        ]
        msg = ChatMessage(role=Role.USER, content=content)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""

    def test_minimal_request(self):
        """Test minimal valid request."""
        req = ChatCompletionRequest(
            messages=[ChatMessage(role=Role.USER, content="Hello")]
        )
        assert len(req.messages) == 1
        assert req.temperature == 0.7  # default
        assert req.max_tokens == 256  # default
        assert req.stream is False  # default

    def test_full_request(self):
        """Test request with all parameters."""
        req = ChatCompletionRequest(
            messages=[
                ChatMessage(role=Role.SYSTEM, content="You are helpful"),
                ChatMessage(role=Role.USER, content="Hello"),
            ],
            temperature=0.5,
            top_p=0.9,
            max_tokens=100,
            stream=True,
            stop=["END"],
            presence_penalty=0.1,
            frequency_penalty=0.1,
            n=2,
        )
        assert req.temperature == 0.5
        assert req.stream is True
        assert req.n == 2

    def test_invalid_temperature(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role=Role.USER, content="Hello")],
                temperature=2.5,  # > 2.0
            )

    def test_invalid_top_p(self):
        """Test that invalid top_p raises error."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role=Role.USER, content="Hello")],
                top_p=1.5,  # > 1.0
            )

    def test_empty_messages(self):
        """Test that empty messages raises error."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(messages=[])


class TestHealthStatus:
    """Test HealthStatus model."""

    def test_healthy_status(self):
        """Test healthy status."""
        status = HealthStatus(
            status="healthy",
            model_loaded=True,
            gpu_available=True,
            gpu_memory_used_mb=16000,
            gpu_memory_total_mb=24000,
            gpu_utilization_percent=75.0,
            uptime_seconds=3600,
            version="0.1.0",
        )
        assert status.status == "healthy"
        assert status.model_loaded is True

    def test_degraded_status(self):
        """Test degraded status without model."""
        status = HealthStatus(
            status="degraded",
            model_loaded=False,
            gpu_available=True,
            uptime_seconds=100,
            version="0.1.0",
        )
        assert status.status == "degraded"
        assert status.model_loaded is False
