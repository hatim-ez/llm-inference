"""Logging configuration and request logging middleware."""

import logging
import sys
import time
import uuid
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


def setup_logging(level: str = "INFO", log_format: str = "json") -> None:
    """Configure structured logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_format: Output format (json or text).
    """
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure processors based on format
    if log_format.lower() == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger.info(
        "logging_configured",
        level=level,
        format=log_format,
    )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""

    # Paths to exclude from logging
    EXCLUDE_PATHS = {"/health", "/ready", "/live", "/metrics"}

    def __init__(self, app, log_request_body: bool = False):
        """Initialize logging middleware.

        Args:
            app: FastAPI application.
            log_request_body: Whether to log request body.
        """
        super().__init__(app)
        self.log_request_body = log_request_body

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Log request and response.

        Args:
            request: FastAPI request.
            call_next: Next middleware/handler.

        Returns:
            Response object.
        """
        # Skip logging for excluded paths
        if request.url.path in self.EXCLUDE_PATHS:
            return await call_next(request)

        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Extract client info
        client_ip = None
        if request.client:
            client_ip = request.client.host

        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        # Log request
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": client_ip,
            "user_agent": request.headers.get("User-Agent", "")[:100],
        }

        # Add query params if present
        if request.query_params:
            log_data["query_params"] = dict(request.query_params)

        logger.info("request_started", **log_data)

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                "request_completed",
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "request_failed",
                request_id=request_id,
                error=str(e),
                duration_ms=round(duration * 1000, 2),
                exc_info=True,
            )
            raise


def get_request_logger(request: Request) -> structlog.BoundLogger:
    """Get a logger bound to request context.

    Args:
        request: FastAPI request.

    Returns:
        Bound logger with request context.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    return logger.bind(request_id=request_id)
