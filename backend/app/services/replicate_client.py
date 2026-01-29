"""Replicate API client for face swap operations."""

import asyncio
import base64
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class ReplicateError(Exception):
    """Custom exception for Replicate API errors."""

    pass


class ReplicateClient:
    """Client for Replicate API face swap operations."""

    # Replicate model for face swap
    MODEL_ID = "lucataco/faceswap:9a4298548422074c3f57258c5d544497314ae4112df80d116f0d2109e843d20d"

    def __init__(self, api_token: str):
        """Initialize the Replicate client.

        Args:
            api_token: Replicate API token.

        Raises:
            ValueError: If API token is empty.
        """
        if not api_token:
            raise ValueError("Replicate API token is required")
        self.api_token = api_token

    def _bytes_to_data_uri(self, image_bytes: bytes) -> str:
        """Convert image bytes to a data URI.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            Data URI string (e.g., "data:image/png;base64,...")
        """
        # Detect image type from magic bytes
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            mime_type = "image/png"
        elif image_bytes[:2] == b"\xff\xd8":
            mime_type = "image/jpeg"
        elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            mime_type = "image/webp"
        else:
            # Default to PNG
            mime_type = "image/png"

        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{b64_data}"

    async def run_faceswap(
        self,
        source_image: bytes,
        target_image: bytes,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> bytes:
        """Run face swap using Replicate API.

        Args:
            source_image: Source face image bytes (face to swap onto target).
            target_image: Target face image bytes (image to receive the face).
            max_retries: Maximum number of retry attempts.
            retry_delay: Delay between retries in seconds.

        Returns:
            Swapped face image bytes.

        Raises:
            ReplicateError: If face swap fails after all retries.
        """
        import replicate

        # Convert images to data URIs
        source_uri = self._bytes_to_data_uri(source_image)
        target_uri = self._bytes_to_data_uri(target_image)

        # Set API token for replicate library
        import os
        os.environ["REPLICATE_API_TOKEN"] = self.api_token

        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                logger.info(f"Running face swap (attempt {attempt + 1}/{max_retries})")

                # Run the model
                output = replicate.run(
                    self.MODEL_ID,
                    input={
                        "swap_image": source_uri,
                        "target_image": target_uri,
                    },
                )

                # Output can be a URL string, FileOutput object, or list
                result_url = output
                if isinstance(output, list) and len(output) > 0:
                    result_url = output[0]

                # Handle FileOutput object (newer replicate library versions)
                if hasattr(result_url, 'url'):
                    result_url = result_url.url
                elif not isinstance(result_url, str):
                    result_url = str(result_url)

                logger.info(f"Face swap completed, downloading result from: {result_url}")

                # Download the result image
                async with httpx.AsyncClient() as client:
                    response = await client.get(result_url)

                    if response.status_code != 200:
                        raise ReplicateError(
                            f"Failed to download result image: HTTP {response.status_code}"
                        )

                    return response.content

            except ReplicateError:
                raise
            except Exception as e:
                last_error = e
                error_str = str(e)
                logger.warning(f"Face swap attempt {attempt + 1} failed: {type(e).__name__}")
                logger.warning(f"  Details:\n{error_str}")

                # Check for rate limit (429) errors - need longer wait
                if "429" in error_str or "throttled" in error_str.lower():
                    wait_time = 20.0  # Wait 20 seconds on rate limit
                    logger.info(f"Rate limited, waiting {wait_time}s before retry")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)
                    else:
                        # Final attempt also rate limited
                        raise ReplicateError(
                            "サーバーが混雑しています。しばらく待ってからお試しください。"
                        )
                elif attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff

        raise ReplicateError(f"Face swap failed after {max_retries} attempts: {last_error}")


# Singleton instance
_replicate_client: Optional[ReplicateClient] = None


def get_replicate_client() -> ReplicateClient:
    """Get the singleton ReplicateClient instance.

    Returns:
        ReplicateClient instance.

    Raises:
        ValueError: If Replicate API token is not configured.
    """
    global _replicate_client

    if _replicate_client is None:
        from app.config import get_settings

        settings = get_settings()
        _replicate_client = ReplicateClient(api_token=settings.replicate_api_token)

    return _replicate_client
