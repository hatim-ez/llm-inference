"""Authentication middleware for API key validation."""

import hashlib
import hmac
import secrets
from typing import Optional

import structlog
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.config import get_settings

logger = structlog.get_logger(__name__)

# API key header configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str, secret_key: str) -> bool:
    """Verify API key using constant-time comparison.

    Args:
        api_key: The provided API key.
        secret_key: The expected secret key.

    Returns:
        True if keys match, False otherwise.
    """
    if not api_key or not secret_key:
        return False

    # Use constant-time comparison to prevent timing attacks
    return secrets.compare_digest(api_key, secret_key)


async def get_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(API_KEY_HEADER),
) -> Optional[str]:
    """Extract and validate API key from request.

    Args:
        request: FastAPI request object.
        api_key_header: API key from header.

    Returns:
        Validated API key or None if not required.

    Raises:
        HTTPException: If API key is required but invalid/missing.
    """
    settings = get_settings()

    if not settings.require_api_key:
        return None

    if not api_key_header:
        # Also check Authorization header for Bearer token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key_header = auth_header[7:]

    if not api_key_header:
        logger.warning(
            "missing_api_key",
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Missing API key. Include it in X-API-Key header or Authorization: Bearer <key>",
                    "type": "authentication_error",
                }
            },
        )

    if not settings.api_key_secret:
        logger.error("api_key_secret_not_configured")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "API key authentication not configured",
                    "type": "configuration_error",
                }
            },
        )

    if not verify_api_key(api_key_header, settings.api_key_secret):
        logger.warning(
            "invalid_api_key",
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key",
                    "type": "authentication_error",
                }
            },
        )

    return api_key_header


def generate_api_key(prefix: str = "llm") -> str:
    """Generate a new random API key.

    Args:
        prefix: Prefix for the API key.

    Returns:
        Generated API key string.
    """
    # Generate 32 random bytes and encode as hex
    random_bytes = secrets.token_bytes(32)
    key_hash = hashlib.sha256(random_bytes).hexdigest()[:48]
    return f"{prefix}_{key_hash}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage.

    Args:
        api_key: The API key to hash.

    Returns:
        SHA-256 hash of the API key.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()
