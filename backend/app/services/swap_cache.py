"""In-memory cache for face swap results."""

import hashlib
import logging
from typing import Optional

from cachetools import TTLCache

logger = logging.getLogger(__name__)


class SwapCache:
    """TTL-based cache for face swap results.

    Uses cachetools.TTLCache for in-memory caching with automatic expiration.
    Cache keys are generated from hash of input images to avoid redundant API calls.
    """

    def __init__(self, ttl: int = 3600, max_size: int = 100):
        """Initialize the swap cache.

        Args:
            ttl: Time-to-live in seconds (default: 1 hour).
            max_size: Maximum number of items in cache (default: 100).
        """
        self.ttl = ttl
        self.max_size = max_size
        self._cache: TTLCache = TTLCache(maxsize=max_size, ttl=ttl)

    def generate_key(self, current_image_b64: str, ideal_image_b64: str) -> str:
        """Generate a cache key from input images.

        Args:
            current_image_b64: Base64 encoded current face image.
            ideal_image_b64: Base64 encoded ideal face image.

        Returns:
            SHA256 hash string as cache key.
        """
        # Combine both images for unique key
        combined = f"{current_image_b64}:{ideal_image_b64}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[bytes]:
        """Get cached result by key.

        Args:
            key: Cache key.

        Returns:
            Cached image bytes, or None if not found/expired.
        """
        try:
            return self._cache.get(key)
        except KeyError:
            return None

    def set(self, key: str, value: bytes) -> None:
        """Store result in cache.

        Args:
            key: Cache key.
            value: Image bytes to cache.
        """
        self._cache[key] = value
        logger.debug(f"Cached swap result with key: {key[:16]}...")

    def contains(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key.

        Returns:
            True if key exists and not expired, False otherwise.
        """
        return key in self._cache

    def delete(self, key: str) -> None:
        """Delete entry from cache.

        Args:
            key: Cache key to delete.
        """
        try:
            del self._cache[key]
        except KeyError:
            pass

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Swap cache cleared")

    def size(self) -> int:
        """Get current number of cached items.

        Returns:
            Number of items in cache.
        """
        return len(self._cache)


# Singleton instance
_swap_cache: Optional[SwapCache] = None


def get_swap_cache() -> SwapCache:
    """Get the singleton SwapCache instance.

    Returns:
        SwapCache instance.
    """
    global _swap_cache

    if _swap_cache is None:
        from app.config import get_settings

        settings = get_settings()
        _swap_cache = SwapCache(
            ttl=settings.swap_cache_ttl,
            max_size=settings.swap_cache_max_size,
        )

    return _swap_cache
