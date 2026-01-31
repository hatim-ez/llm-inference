"""Rate limiting middleware using sliding window algorithm."""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = structlog.get_logger(__name__)


@dataclass
class RateLimitState:
    """Rate limit state for a client."""

    requests: list[float]  # Timestamps of requests
    blocked_until: Optional[float] = None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiting middleware."""

    def __init__(
        self,
        app,
        requests_per_minute: int = 100,
        burst_size: int = 20,
        block_duration: float = 60.0,
    ):
        """Initialize rate limiter.

        Args:
            app: FastAPI application.
            requests_per_minute: Maximum requests per minute.
            burst_size: Maximum burst requests allowed.
            block_duration: Duration to block after exceeding limit (seconds).
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.block_duration = block_duration
        self.window_size = 60.0  # 1 minute window
        self._states: dict[str, RateLimitState] = defaultdict(
            lambda: RateLimitState(requests=[])
        )

    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier.

        Args:
            request: FastAPI request.

        Returns:
            Client identifier string.
        """
        # Use API key if available, otherwise use IP
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key[:16]}"

        # Get IP, considering X-Forwarded-For for proxies
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        elif request.client:
            client_ip = request.client.host
        else:
            client_ip = "unknown"

        return f"ip:{client_ip}"

    def _cleanup_old_requests(
        self,
        state: RateLimitState,
        current_time: float,
    ) -> None:
        """Remove requests outside the sliding window.

        Args:
            state: Client rate limit state.
            current_time: Current timestamp.
        """
        cutoff = current_time - self.window_size
        state.requests = [ts for ts in state.requests if ts > cutoff]

    def _check_rate_limit(
        self,
        client_id: str,
        current_time: float,
    ) -> tuple[bool, Optional[float], int]:
        """Check if request is within rate limits.

        Args:
            client_id: Client identifier.
            current_time: Current timestamp.

        Returns:
            Tuple of (allowed, retry_after, remaining).
        """
        state = self._states[client_id]

        # Check if client is blocked
        if state.blocked_until and current_time < state.blocked_until:
            retry_after = state.blocked_until - current_time
            return False, retry_after, 0

        # Clear block if expired
        if state.blocked_until and current_time >= state.blocked_until:
            state.blocked_until = None
            state.requests = []

        # Cleanup old requests
        self._cleanup_old_requests(state, current_time)

        # Check burst limit (short-term)
        recent_requests = [
            ts for ts in state.requests
            if ts > current_time - 1.0  # Last 1 second
        ]
        if len(recent_requests) >= self.burst_size:
            # Block the client
            state.blocked_until = current_time + self.block_duration
            logger.warning(
                "rate_limit_exceeded",
                client_id=client_id,
                reason="burst",
                blocked_for=self.block_duration,
            )
            return False, self.block_duration, 0

        # Check rate limit (per minute)
        if len(state.requests) >= self.requests_per_minute:
            # Calculate when oldest request will fall out of window
            oldest = min(state.requests)
            retry_after = oldest + self.window_size - current_time
            return False, max(0.1, retry_after), 0

        # Request allowed
        remaining = self.requests_per_minute - len(state.requests) - 1
        return True, None, remaining

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request through rate limiter.

        Args:
            request: FastAPI request.
            call_next: Next middleware/handler.

        Returns:
            Response object.
        """
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/ready", "/live", "/metrics"):
            return await call_next(request)

        client_id = self._get_client_id(request)
        current_time = time.time()

        allowed, retry_after, remaining = self._check_rate_limit(
            client_id, current_time
        )

        if not allowed:
            logger.info(
                "rate_limited",
                client_id=client_id,
                path=request.url.path,
                retry_after=retry_after,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit exceeded. Please retry later.",
                        "type": "rate_limit_error",
                        "retry_after": retry_after,
                    }
                },
                headers={
                    "Retry-After": str(int(retry_after) if retry_after else 60),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + (retry_after or 60))),
                },
            )

        # Record the request
        self._states[client_id].requests.append(current_time)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(
            int(current_time + self.window_size)
        )

        return response

    def get_stats(self) -> dict:
        """Get rate limiter statistics.

        Returns:
            Dictionary with stats.
        """
        current_time = time.time()
        active_clients = 0
        blocked_clients = 0

        for client_id, state in self._states.items():
            self._cleanup_old_requests(state, current_time)
            if state.requests:
                active_clients += 1
            if state.blocked_until and current_time < state.blocked_until:
                blocked_clients += 1

        return {
            "active_clients": active_clients,
            "blocked_clients": blocked_clients,
            "requests_per_minute_limit": self.requests_per_minute,
            "burst_limit": self.burst_size,
        }
