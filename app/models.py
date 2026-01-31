"""Pydantic models for API request/response schemas."""

from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message role enum."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ContentPart(BaseModel):
    """Content part for multimodal messages."""

    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[dict[str, str]] = None


class ChatMessage(BaseModel):
    """Chat message model (OpenAI-compatible)."""

    role: Role
    content: Union[str, list[ContentPart]]
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions (OpenAI-compatible)."""

    model: Optional[str] = Field(
        default=None,
        description="Model to use (ignored, uses configured model)",
    )
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="List of messages in the conversation",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability",
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=4096,
        description="Maximum tokens to generate",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response",
    )
    stop: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Stop sequences",
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty",
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty",
    )
    n: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of completions to generate",
    )
    user: Optional[str] = Field(
        default=None,
        description="User identifier for tracking",
    )


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Response model for chat completions (OpenAI-compatible)."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class ChatCompletionChunkDelta(BaseModel):
    """Delta for streaming response."""

    role: Optional[Role] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """Single streaming chunk choice."""

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk response (OpenAI-compatible)."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class HealthStatus(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy", "degraded"]
    model_loaded: bool
    gpu_available: bool
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    uptime_seconds: float
    version: str


class ErrorResponse(BaseModel):
    """Error response model."""

    error: dict[str, Any] = Field(
        ...,
        description="Error details",
    )


class ModelInfo(BaseModel):
    """Model information response."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "local"


class ModelsResponse(BaseModel):
    """Response for listing models."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]
