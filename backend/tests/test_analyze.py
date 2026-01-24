"""Tests for analyze endpoint."""

import io

import pytest


class TestAnalyzeEndpoint:
    """Test cases for POST /api/v1/analyze endpoint."""

    def test_analyze_without_image_returns_validation_error(self, client, api_base_url):
        """Analyze without image should return VALIDATION_ERROR."""
        response = client.post(f"{api_base_url}/analyze")

        assert response.status_code == 400

        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert data["error"]["message"] == "Image file is required"

    def test_analyze_with_empty_file_returns_validation_error(self, client, api_base_url):
        """Analyze with empty file should return VALIDATION_ERROR."""
        response = client.post(
            f"{api_base_url}/analyze",
            files={"image": ("test.jpg", b"", "image/jpeg")},
        )

        assert response.status_code == 400

        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_analyze_with_invalid_format_returns_invalid_image_format(self, client, api_base_url):
        """Analyze with invalid image format should return INVALID_IMAGE_FORMAT."""
        # Create a GIF header (invalid format)
        gif_data = b"GIF89a" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/analyze",
            files={"image": ("test.gif", gif_data, "image/gif")},
        )

        assert response.status_code == 400

        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "INVALID_IMAGE_FORMAT"
        assert "JPEG and PNG" in data["error"]["message"]

    def test_analyze_error_response_format(self, client, api_base_url):
        """Error response should follow standard format."""
        response = client.post(f"{api_base_url}/analyze")

        data = response.json()

        # Check error response structure
        assert "success" in data
        assert data["success"] is False
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
