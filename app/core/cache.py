"""Caching utilities for LLM inference."""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    value: Any
    created_at: float
    expires_at: Optional[float]
    access_count: int = 0
    last_accessed: float = 0


class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of entries.
            default_ttl: Default time-to-live in seconds.
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            # Check expiration
            if entry.expires_at is not None and time.time() > entry.expires_at:
                del self._cache[key]
                self._misses += 1
                logger.debug("cache_expired", key=key)
                return None

            # Update access metadata and move to end (most recently used)
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._cache.move_to_end(key)
            self._hits += 1

            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds (overrides default).
        """
        async with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug("cache_evicted", key=oldest_key)

            now = time.time()
            effective_ttl = ttl if ttl is not None else self.default_ttl
            expires_at = now + effective_ttl if effective_ttl else None

            self._cache[key] = CacheEntry(
                value=value,
                created_at=now,
                expires_at=expires_at,
                last_accessed=now,
            )

    async def delete(self, key: str) -> bool:
        """Delete entry from cache.

        Args:
            key: Cache key.

        Returns:
            True if key was deleted, False if not found.
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("cache_cleared", entries_cleared=count)
            return count

    async def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            now = time.time()
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if entry.expires_at is not None and now > entry.expires_at
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.debug("cache_cleanup", entries_removed=len(expired_keys))

            return len(expired_keys)

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


class PromptCache:
    """Cache for LLM prompt responses."""

    def __init__(self, max_size: int = 500, ttl: float = 3600):
        """Initialize prompt cache.

        Args:
            max_size: Maximum number of cached prompts.
            ttl: Time-to-live in seconds.
        """
        self._cache = LRUCache(max_size=max_size, default_ttl=ttl)

    @staticmethod
    def _make_key(
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> str:
        """Create cache key from prompt parameters.

        Args:
            prompt: Input prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            top_p: Nucleus sampling probability.

        Returns:
            Cache key string.
        """
        # Create deterministic key from parameters
        key_data = json.dumps(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            },
            sort_keys=True,
        )
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    async def get(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> Optional[dict]:
        """Get cached response for prompt.

        Note: Only cache deterministic responses (temperature=0).

        Args:
            prompt: Input prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            top_p: Nucleus sampling probability.

        Returns:
            Cached response or None.
        """
        # Don't cache non-deterministic responses
        if temperature > 0:
            return None

        key = self._make_key(prompt, temperature, max_tokens, top_p)
        return await self._cache.get(key)

    async def set(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        response: dict,
    ) -> None:
        """Cache response for prompt.

        Args:
            prompt: Input prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            top_p: Nucleus sampling probability.
            response: Response to cache.
        """
        # Don't cache non-deterministic responses
        if temperature > 0:
            return

        key = self._make_key(prompt, temperature, max_tokens, top_p)
        await self._cache.set(key, response)

    def stats(self) -> dict:
        """Get cache statistics."""
        return self._cache.stats()


# Global prompt cache instance
_prompt_cache: Optional[PromptCache] = None


def get_prompt_cache() -> PromptCache:
    """Get global prompt cache instance."""
    global _prompt_cache
    if _prompt_cache is None:
        _prompt_cache = PromptCache()
    return _prompt_cache
