"""Tests for caching utilities."""

import asyncio

import pytest

from app.core.cache import LRUCache, PromptCache


class TestLRUCache:
    """Test LRU cache implementation."""

    @pytest.fixture
    def cache(self):
        """Create cache instance."""
        return LRUCache(max_size=5, default_ttl=10.0)

    @pytest.mark.asyncio
    async def test_basic_set_get(self, cache):
        """Test basic set and get operations."""
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_missing_key(self, cache):
        """Test getting a missing key."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill cache
        for i in range(5):
            await cache.set(f"key{i}", f"value{i}")

        # Add one more to trigger eviction
        await cache.set("key5", "value5")

        # First key should be evicted
        assert await cache.get("key0") is None
        # Last key should exist
        assert await cache.get("key5") == "value5"

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = LRUCache(max_size=5, default_ttl=0.1)

        await cache.set("key", "value")
        assert await cache.get("key") == "value"

        # Wait for expiration
        await asyncio.sleep(0.15)
        assert await cache.get("key") is None

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Test delete operation."""
        await cache.set("key", "value")
        assert await cache.delete("key") is True
        assert await cache.get("key") is None
        assert await cache.delete("key") is False

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clear operation."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        count = await cache.clear()
        assert count == 2
        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_stats(self, cache):
        """Test cache statistics."""
        await cache.set("key", "value")
        await cache.get("key")  # hit
        await cache.get("missing")  # miss

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestPromptCache:
    """Test prompt-specific cache."""

    @pytest.fixture
    def prompt_cache(self):
        """Create prompt cache instance."""
        return PromptCache(max_size=10, ttl=60.0)

    @pytest.mark.asyncio
    async def test_deterministic_caching(self, prompt_cache):
        """Test that deterministic responses are cached."""
        response = {"text": "Hello!"}

        # Set with temperature=0 (deterministic)
        await prompt_cache.set(
            prompt="Hello",
            temperature=0,
            max_tokens=100,
            top_p=1.0,
            response=response,
        )

        # Get should return cached value
        result = await prompt_cache.get(
            prompt="Hello",
            temperature=0,
            max_tokens=100,
            top_p=1.0,
        )
        assert result == response

    @pytest.mark.asyncio
    async def test_non_deterministic_not_cached(self, prompt_cache):
        """Test that non-deterministic responses are not cached."""
        response = {"text": "Hello!"}

        # Set with temperature>0 (non-deterministic)
        await prompt_cache.set(
            prompt="Hello",
            temperature=0.7,
            max_tokens=100,
            top_p=1.0,
            response=response,
        )

        # Get should return None (not cached)
        result = await prompt_cache.get(
            prompt="Hello",
            temperature=0.7,
            max_tokens=100,
            top_p=1.0,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_different_params_different_cache(self, prompt_cache):
        """Test that different parameters create different cache entries."""
        response1 = {"text": "Response 1"}
        response2 = {"text": "Response 2"}

        await prompt_cache.set(
            prompt="Hello",
            temperature=0,
            max_tokens=100,
            top_p=1.0,
            response=response1,
        )

        await prompt_cache.set(
            prompt="Hello",
            temperature=0,
            max_tokens=200,  # Different max_tokens
            top_p=1.0,
            response=response2,
        )

        # Get first
        result1 = await prompt_cache.get(
            prompt="Hello",
            temperature=0,
            max_tokens=100,
            top_p=1.0,
        )
        assert result1 == response1

        # Get second
        result2 = await prompt_cache.get(
            prompt="Hello",
            temperature=0,
            max_tokens=200,
            top_p=1.0,
        )
        assert result2 == response2
