"""Unit tests for SwapCache."""

import base64
import time
from unittest.mock import patch

import pytest

from app.services.swap_cache import SwapCache


@pytest.fixture
def swap_cache():
    """Create a SwapCache instance with short TTL for testing."""
    return SwapCache(ttl=2, max_size=10)


@pytest.fixture
def sample_image_b64():
    """Create sample base64 image for testing."""
    return base64.b64encode(b"test_image_data").decode("utf-8")


@pytest.fixture
def sample_result_bytes():
    """Create sample result bytes for testing."""
    return b"swapped_result_image_data"


class TestSwapCache:
    """Tests for SwapCache."""

    def test_init_with_defaults(self):
        """Test cache initialization with default values."""
        cache = SwapCache()
        assert cache.ttl == 3600
        assert cache.max_size == 100

    def test_init_with_custom_values(self):
        """Test cache initialization with custom values."""
        cache = SwapCache(ttl=600, max_size=50)
        assert cache.ttl == 600
        assert cache.max_size == 50

    def test_generate_key(self, swap_cache, sample_image_b64):
        """Test cache key generation."""
        key1 = swap_cache.generate_key(sample_image_b64, sample_image_b64)
        key2 = swap_cache.generate_key(sample_image_b64, sample_image_b64)

        # Same inputs should produce same key
        assert key1 == key2

        # Different inputs should produce different keys
        different_b64 = base64.b64encode(b"different_image").decode("utf-8")
        key3 = swap_cache.generate_key(different_b64, sample_image_b64)
        assert key1 != key3

    def test_generate_key_deterministic(self, swap_cache, sample_image_b64):
        """Test that key generation is deterministic."""
        keys = [
            swap_cache.generate_key(sample_image_b64, sample_image_b64)
            for _ in range(10)
        ]
        assert len(set(keys)) == 1  # All keys should be identical

    def test_set_and_get(self, swap_cache, sample_result_bytes):
        """Test basic set and get operations."""
        key = "test_key"

        # Initially cache miss
        assert swap_cache.get(key) is None

        # Set value
        swap_cache.set(key, sample_result_bytes)

        # Now should hit
        result = swap_cache.get(key)
        assert result == sample_result_bytes

    def test_ttl_expiry(self, swap_cache, sample_result_bytes):
        """Test that cached values expire after TTL."""
        key = "expiring_key"

        swap_cache.set(key, sample_result_bytes)
        assert swap_cache.get(key) == sample_result_bytes

        # Wait for TTL to expire (cache has 2 second TTL)
        time.sleep(2.5)

        # Should be expired now
        assert swap_cache.get(key) is None

    def test_max_size_eviction(self):
        """Test that cache evicts old entries when max size is reached."""
        cache = SwapCache(ttl=3600, max_size=3)

        # Fill cache
        cache.set("key1", b"value1")
        cache.set("key2", b"value2")
        cache.set("key3", b"value3")

        # All should be present
        assert cache.get("key1") is not None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

        # Add one more (should evict oldest)
        cache.set("key4", b"value4")

        # key4 should be present
        assert cache.get("key4") is not None

    def test_contains(self, swap_cache, sample_result_bytes):
        """Test contains check."""
        key = "check_key"

        assert not swap_cache.contains(key)

        swap_cache.set(key, sample_result_bytes)

        assert swap_cache.contains(key)

    def test_delete(self, swap_cache, sample_result_bytes):
        """Test delete operation."""
        key = "delete_key"

        swap_cache.set(key, sample_result_bytes)
        assert swap_cache.get(key) is not None

        swap_cache.delete(key)
        assert swap_cache.get(key) is None

    def test_clear(self, swap_cache, sample_result_bytes):
        """Test clear operation."""
        swap_cache.set("key1", sample_result_bytes)
        swap_cache.set("key2", sample_result_bytes)

        assert swap_cache.get("key1") is not None
        assert swap_cache.get("key2") is not None

        swap_cache.clear()

        assert swap_cache.get("key1") is None
        assert swap_cache.get("key2") is None

    def test_size(self, swap_cache, sample_result_bytes):
        """Test size reporting."""
        assert swap_cache.size() == 0

        swap_cache.set("key1", sample_result_bytes)
        assert swap_cache.size() == 1

        swap_cache.set("key2", sample_result_bytes)
        assert swap_cache.size() == 2

        swap_cache.delete("key1")
        assert swap_cache.size() == 1


class TestSwapCacheIntegration:
    """Integration tests for SwapCache with typical usage patterns."""

    def test_full_workflow(self, swap_cache, sample_image_b64, sample_result_bytes):
        """Test complete caching workflow."""
        ideal_image_b64 = base64.b64encode(b"ideal_face").decode("utf-8")

        # Generate key
        key = swap_cache.generate_key(sample_image_b64, ideal_image_b64)

        # Check cache miss
        assert swap_cache.get(key) is None

        # Store result
        swap_cache.set(key, sample_result_bytes)

        # Verify cache hit
        cached = swap_cache.get(key)
        assert cached == sample_result_bytes

        # Different images should miss
        different_key = swap_cache.generate_key(ideal_image_b64, sample_image_b64)
        assert swap_cache.get(different_key) is None
