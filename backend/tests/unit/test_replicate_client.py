"""Unit tests for ReplicateClient."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.replicate_client import ReplicateClient, ReplicateError


@pytest.fixture
def replicate_client():
    """Create a ReplicateClient instance."""
    return ReplicateClient(api_token="test_token")


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing."""
    # Simple 1x1 red pixel PNG
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )
    return png_data


class TestReplicateClient:
    """Tests for ReplicateClient."""

    def test_init_with_token(self):
        """Test client initialization with API token."""
        client = ReplicateClient(api_token="test_token")
        assert client.api_token == "test_token"

    def test_init_without_token_raises_error(self):
        """Test that initialization without token raises error."""
        with pytest.raises(ValueError, match="Replicate API token is required"):
            ReplicateClient(api_token="")

    @pytest.mark.asyncio
    async def test_run_faceswap_success(self, replicate_client, sample_image_bytes):
        """Test successful face swap operation."""
        mock_output = "https://replicate.delivery/test/output.png"

        with patch("replicate.run") as mock_run:
            mock_run.return_value = mock_output

            with patch("httpx.AsyncClient") as mock_http_client:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_response.content = sample_image_bytes
                mock_http_client_instance = AsyncMock()
                mock_http_client_instance.get = AsyncMock(return_value=mock_response)
                mock_http_client_instance.__aenter__ = AsyncMock(return_value=mock_http_client_instance)
                mock_http_client_instance.__aexit__ = AsyncMock(return_value=None)
                mock_http_client.return_value = mock_http_client_instance

                result = await replicate_client.run_faceswap(
                    source_image=sample_image_bytes,
                    target_image=sample_image_bytes,
                )

                assert result is not None
                assert len(result) > 0

    @pytest.mark.asyncio
    async def test_run_faceswap_replicate_error(self, replicate_client, sample_image_bytes):
        """Test face swap when Replicate API fails."""
        with patch("replicate.run") as mock_run:
            mock_run.side_effect = Exception("Replicate API error")

            with pytest.raises(ReplicateError, match="Face swap failed"):
                await replicate_client.run_faceswap(
                    source_image=sample_image_bytes,
                    target_image=sample_image_bytes,
                )

    @pytest.mark.asyncio
    async def test_run_faceswap_download_error(self, replicate_client, sample_image_bytes):
        """Test face swap when result download fails."""
        mock_output = "https://replicate.delivery/test/output.png"

        with patch("replicate.run") as mock_run:
            mock_run.return_value = mock_output

            with patch("httpx.AsyncClient") as mock_http_client:
                mock_response = AsyncMock()
                mock_response.status_code = 500
                mock_http_client_instance = AsyncMock()
                mock_http_client_instance.get = AsyncMock(return_value=mock_response)
                mock_http_client_instance.__aenter__ = AsyncMock(return_value=mock_http_client_instance)
                mock_http_client_instance.__aexit__ = AsyncMock(return_value=None)
                mock_http_client.return_value = mock_http_client_instance

                with pytest.raises(ReplicateError, match="Failed to download"):
                    await replicate_client.run_faceswap(
                        source_image=sample_image_bytes,
                        target_image=sample_image_bytes,
                    )

    @pytest.mark.asyncio
    async def test_run_faceswap_retry_on_failure(self, replicate_client, sample_image_bytes):
        """Test that face swap retries on transient failures."""
        mock_output = "https://replicate.delivery/test/output.png"
        call_count = 0

        def mock_run_with_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Transient error")
            return mock_output

        with patch("replicate.run", side_effect=mock_run_with_failure):
            with patch("httpx.AsyncClient") as mock_http_client:
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_response.content = sample_image_bytes
                mock_http_client_instance = AsyncMock()
                mock_http_client_instance.get = AsyncMock(return_value=mock_response)
                mock_http_client_instance.__aenter__ = AsyncMock(return_value=mock_http_client_instance)
                mock_http_client_instance.__aexit__ = AsyncMock(return_value=None)
                mock_http_client.return_value = mock_http_client_instance

                result = await replicate_client.run_faceswap(
                    source_image=sample_image_bytes,
                    target_image=sample_image_bytes,
                    max_retries=3,
                )

                assert result is not None
                assert call_count == 2  # First call failed, second succeeded

    @pytest.mark.asyncio
    async def test_run_faceswap_max_retries_exceeded(self, replicate_client, sample_image_bytes):
        """Test that face swap fails after max retries exceeded."""
        with patch("replicate.run") as mock_run:
            mock_run.side_effect = Exception("Persistent error")

            with pytest.raises(ReplicateError, match="Face swap failed"):
                await replicate_client.run_faceswap(
                    source_image=sample_image_bytes,
                    target_image=sample_image_bytes,
                    max_retries=2,
                )


class TestImageEncoding:
    """Tests for image encoding/decoding utilities."""

    def test_bytes_to_data_uri(self, replicate_client, sample_image_bytes):
        """Test converting bytes to data URI."""
        data_uri = replicate_client._bytes_to_data_uri(sample_image_bytes)
        assert data_uri.startswith("data:image/png;base64,")

    def test_bytes_to_data_uri_jpeg(self, replicate_client):
        """Test data URI detection for JPEG."""
        # JPEG magic bytes
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        data_uri = replicate_client._bytes_to_data_uri(jpeg_bytes)
        assert data_uri.startswith("data:image/jpeg;base64,")
