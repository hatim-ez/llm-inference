"""FastAPI application entry point."""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app import __version__
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.config import get_settings
from app.core.vllm_engine import VLLMEngine
from app.middleware.logging import setup_logging
from app.middleware.rate_limit import RateLimitMiddleware
from app.utils.metrics import setup_metrics

# Initialize settings
settings = get_settings()

# Setup structured logging
setup_logging(settings.log_level, settings.log_format)
logger = structlog.get_logger(__name__)

# Global engine instance
engine: VLLMEngine | None = None
start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global engine, start_time

    start_time = time.time()
    logger.info(
        "starting_application",
        version=__version__,
        model_path=settings.model_path,
    )

    # Initialize vLLM engine
    try:
        engine = VLLMEngine(
            model_path=settings.model_path,
            tensor_parallel_size=settings.tensor_parallel_size,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            max_model_length=settings.max_model_length,
            max_num_seqs=settings.max_num_seqs,
            enforce_eager=settings.enforce_eager,
            swap_space=settings.swap_space,
        )
        await engine.initialize()
        logger.info("vllm_engine_initialized")
    except Exception as e:
        logger.error("failed_to_initialize_engine", error=str(e))
        # Allow startup without model for development
        engine = None

    yield

    # Cleanup
    logger.info("shutting_down_application")
    if engine:
        await engine.shutdown()


# Create FastAPI app
app = FastAPI(
    title="LLM Inference API",
    description="Production-grade multimodal LLM inference system",
    version=__version__,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=settings.rate_limit,
    burst_size=settings.rate_limit_burst,
)

# Setup Prometheus metrics
if settings.enable_metrics:
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
    setup_metrics()

# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(chat_router, prefix="/v1", tags=["Chat"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
            }
        },
    )


def get_engine() -> VLLMEngine | None:
    """Get the global vLLM engine instance."""
    return engine


def get_uptime() -> float:
    """Get application uptime in seconds."""
    return time.time() - start_time


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
