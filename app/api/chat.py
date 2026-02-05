"""Chat completion API endpoints (OpenAI-compatible)."""

import time
import uuid
from typing import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.core.vllm_engine import VLLMEngine
from app.models import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelInfo,
    ModelsResponse,
    Role,
    Usage,
)

router = APIRouter()
logger = structlog.get_logger(__name__)


def get_engine() -> VLLMEngine:
    """Get vLLM engine dependency."""
    from app.main import get_engine as _get_engine

    engine = _get_engine()
    if engine is None or not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please try again later.",
        )
    return engine


def messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Convert chat messages to a single prompt string.

    Args:
        messages: List of chat messages.

    Returns:
        Formatted prompt string.
    """
    prompt_parts = []

    for message in messages:
        role = message.role.value
        content = message.content

        if isinstance(content, str):
            prompt_parts.append(f"{role}: {content}")
        elif isinstance(content, list):
            # Handle multimodal content
            text_parts = []
            for part in content:
                if part.type == "text" and part.text:
                    text_parts.append(part.text)
                elif part.type == "image_url":
                    text_parts.append("[Image]")
            prompt_parts.append(f"{role}: {' '.join(text_parts)}")

    # Add assistant prompt prefix
    prompt_parts.append("assistant:")

    return "\n".join(prompt_parts)


@router.get("/models")
async def list_models(
    settings: Settings = Depends(get_settings),
) -> ModelsResponse:
    """List available models (OpenAI-compatible)."""
    return ModelsResponse(
        data=[
            ModelInfo(
                id=settings.model_name,
                created=int(time.time()),
            )
        ]
    )


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    engine: VLLMEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
) -> ChatCompletionResponse | StreamingResponse:
    """Create chat completion (OpenAI-compatible).

    Supports both text and multimodal (image) inputs.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    logger.info(
        "chat_completion_request",
        request_id=request_id,
        num_messages=len(request.messages),
        max_tokens=request.max_tokens,
        stream=request.stream,
    )

    # Convert messages to prompt
    prompt = messages_to_prompt(request.messages)

    # Handle stop sequences
    stop = None
    if request.stop:
        stop = [request.stop] if isinstance(request.stop, str) else request.stop

    if request.stream:
        return StreamingResponse(
            _stream_response(
                request_id=request_id,
                created=created,
                model=settings.model_name,
                prompt=prompt,
                engine=engine,
                request=request,
                stop=stop,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response
    try:
        result = await engine.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            n=request.n,
        )
    except Exception as e:
        logger.error("generation_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    # Build response
    choices = []
    for r in result["results"]:
        choices.append(
            ChatCompletionChoice(
                index=r["index"],
                message=ChatMessage(
                    role=Role.ASSISTANT,
                    content=r["text"].strip(),
                ),
                finish_reason=r["finish_reason"],
            )
        )

    response = ChatCompletionResponse(
        id=request_id,
        created=created,
        model=settings.model_name,
        choices=choices,
        usage=Usage(
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["total_tokens"],
        ),
    )

    logger.info(
        "chat_completion_complete",
        request_id=request_id,
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
    )

    return response


async def _stream_response(
    request_id: str,
    created: int,
    model: str,
    prompt: str,
    engine: VLLMEngine,
    request: ChatCompletionRequest,
    stop: list[str] | None,
) -> AsyncIterator[str]:
    """Generate streaming response chunks.

    Args:
        request_id: Request ID.
        created: Creation timestamp.
        model: Model name.
        prompt: Formatted prompt.
        engine: vLLM engine.
        request: Original request.
        stop: Stop sequences.

    Yields:
        SSE-formatted response chunks.
    """
    import json

    # Send initial chunk with role
    initial_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role=Role.ASSISTANT),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Stream content chunks
    try:
        async for chunk in engine.generate_stream(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
        ):
            content_chunk = ChatCompletionChunk(
                id=request_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=chunk["content"]),
                        finish_reason=chunk["finish_reason"],
                    )
                ],
            )
            yield f"data: {content_chunk.model_dump_json()}\n\n"

    except Exception as e:
        logger.error("streaming_failed", request_id=request_id, error=str(e))
        # Send error as final chunk
        error_data = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"

    # Send final [DONE] marker
    yield "data: [DONE]\n\n"

    logger.info("streaming_complete", request_id=request_id)
