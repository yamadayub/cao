"""Integration tests for swap API endpoints."""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def api_base_url():
    """Base URL for API endpoints."""
    return "/api/v1"


@pytest.fixture
def sample_image_b64():
    """Create a minimal valid PNG as base64."""
    # 1x1 red pixel PNG
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )
    return base64.b64encode(png_bytes).decode("utf-8")


class TestSwapGenerateEndpoint:
    """Test cases for POST /api/v1/swap/generate endpoint."""

    def test_generate_without_current_image_returns_validation_error(
        self, client, api_base_url, sample_image_b64
    ):
        """Generate without current_image should return VALIDATION_ERROR."""
        response = client.post(
            f"{api_base_url}/swap/generate",
            json={"ideal_image": sample_image_b64},
        )

        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data

    def test_generate_without_ideal_image_returns_validation_error(
        self, client, api_base_url, sample_image_b64
    ):
        """Generate without ideal_image should return VALIDATION_ERROR."""
        response = client.post(
            f"{api_base_url}/swap/generate",
            json={"current_image": sample_image_b64},
        )

        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data

    def test_generate_with_invalid_base64_returns_error(
        self, client, api_base_url
    ):
        """Generate with invalid base64 should return error."""
        response = client.post(
            f"{api_base_url}/swap/generate",
            json={
                "current_image": "not-valid-base64!!!",
                "ideal_image": "also-not-valid!!!",
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False

    @patch("app.routers.swap.get_replicate_client")
    @patch("app.routers.swap.get_swap_cache")
    def test_generate_returns_cached_result(
        self, mock_get_cache, mock_get_client, client, api_base_url, sample_image_b64
    ):
        """Generate should return cached result if available."""
        # Setup mock cache with hit
        mock_cache = MagicMock()
        mock_cache.generate_key.return_value = "test_cache_key"
        mock_cache.get.return_value = b"cached_result_bytes"
        mock_get_cache.return_value = mock_cache

        response = client.post(
            f"{api_base_url}/swap/generate",
            json={
                "current_image": sample_image_b64,
                "ideal_image": sample_image_b64,
            },
        )

        # Should return success with job info (swapped_image is retrieved via GET endpoint)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "completed"
        assert data["data"]["job_id"] is not None

    @patch("app.routers.swap.get_replicate_client")
    @patch("app.routers.swap.get_swap_cache")
    def test_generate_calls_replicate_on_cache_miss(
        self, mock_get_cache, mock_get_client, client, api_base_url, sample_image_b64
    ):
        """Generate should call Replicate API on cache miss."""
        # Setup mock cache with miss
        mock_cache = MagicMock()
        mock_cache.generate_key.return_value = "test_cache_key"
        mock_cache.get.return_value = None  # Cache miss
        mock_get_cache.return_value = mock_cache

        # Setup mock client to fail (since we can't run real API)
        mock_client = MagicMock()
        mock_client.run_faceswap = AsyncMock(side_effect=Exception("API not configured"))
        mock_get_client.return_value = mock_client

        response = client.post(
            f"{api_base_url}/swap/generate",
            json={
                "current_image": sample_image_b64,
                "ideal_image": sample_image_b64,
            },
        )

        # Should fail since Replicate API is mocked to fail
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False


class TestSwapPartsEndpoint:
    """Test cases for POST /api/v1/swap/parts endpoint."""

    def test_parts_without_current_image_returns_validation_error(
        self, client, api_base_url, sample_image_b64
    ):
        """Parts without current_image should return VALIDATION_ERROR."""
        response = client.post(
            f"{api_base_url}/swap/parts",
            json={
                "swapped_image": sample_image_b64,
                "parts": {"nose": 1.0},
            },
        )

        assert response.status_code == 422

    def test_parts_without_swapped_image_returns_validation_error(
        self, client, api_base_url, sample_image_b64
    ):
        """Parts without swapped_image should return VALIDATION_ERROR."""
        response = client.post(
            f"{api_base_url}/swap/parts",
            json={
                "current_image": sample_image_b64,
                "parts": {"nose": 1.0},
            },
        )

        assert response.status_code == 422

    def test_parts_without_parts_dict_returns_validation_error(
        self, client, api_base_url, sample_image_b64
    ):
        """Parts without parts dict should return VALIDATION_ERROR."""
        response = client.post(
            f"{api_base_url}/swap/parts",
            json={
                "current_image": sample_image_b64,
                "swapped_image": sample_image_b64,
            },
        )

        assert response.status_code == 422

    def test_parts_with_empty_parts_returns_original(
        self, client, api_base_url, sample_image_b64
    ):
        """Parts with empty parts dict should return original image."""
        response = client.post(
            f"{api_base_url}/swap/parts",
            json={
                "current_image": sample_image_b64,
                "swapped_image": sample_image_b64,
                "parts": {},
            },
        )

        # Should succeed but return original image
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["result_image"] is not None

    def test_parts_with_invalid_base64_returns_error(
        self, client, api_base_url, sample_image_b64
    ):
        """Parts with invalid base64 should return error."""
        response = client.post(
            f"{api_base_url}/swap/parts",
            json={
                "current_image": "not-valid-base64!!!",
                "swapped_image": sample_image_b64,
                "parts": {"nose": 1.0},
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False


class TestSwapPreviewAllEndpoint:
    """Test cases for POST /api/v1/swap/preview-all endpoint."""

    def test_preview_all_without_required_fields_returns_error(
        self, client, api_base_url
    ):
        """Preview-all without required fields should return error."""
        response = client.post(
            f"{api_base_url}/swap/preview-all",
            json={},
        )

        assert response.status_code == 422

    def test_preview_all_with_valid_request(
        self, client, api_base_url, sample_image_b64
    ):
        """Preview-all with valid request should process."""
        response = client.post(
            f"{api_base_url}/swap/preview-all",
            json={
                "current_image": sample_image_b64,
                "swapped_image": sample_image_b64,
                "parts": {"nose": 1.0, "lips": 0.5},
            },
        )

        # Should succeed (returns original since images don't have faces)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["result_image"] is not None


class TestSwapErrorHandling:
    """Test error handling across swap endpoints."""

    def test_error_response_format(self, client, api_base_url):
        """Error responses should follow standard format."""
        response = client.post(f"{api_base_url}/swap/generate", json={})

        data = response.json()
        # FastAPI validation error format
        assert "detail" in data or ("success" in data and data["success"] is False)

    def test_generate_with_malformed_json_returns_error(self, client, api_base_url):
        """Generate with malformed JSON should return error."""
        response = client.post(
            f"{api_base_url}/swap/generate",
            content="not json at all",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422
