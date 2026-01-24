"""Tests for morph endpoints."""

import json

import pytest


class TestMorphEndpoint:
    """Test cases for POST /api/v1/morph endpoint."""

    def test_morph_without_current_image_returns_validation_error(self, client, api_base_url):
        """Morph without current_image should return VALIDATION_ERROR."""
        # Create a minimal valid PNG for ideal_image only
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/morph",
            files={"ideal_image": ("ideal.png", png_header, "image/png")},
            data={"progress": "0.5"},
        )

        assert response.status_code == 400

        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_morph_without_ideal_image_returns_validation_error(self, client, api_base_url):
        """Morph without ideal_image should return VALIDATION_ERROR."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/morph",
            files={"current_image": ("current.png", png_header, "image/png")},
            data={"progress": "0.5"},
        )

        assert response.status_code == 400

        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_morph_with_invalid_progress_negative(self, client, api_base_url):
        """Morph with negative progress should return VALIDATION_ERROR."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/morph",
            files={
                "current_image": ("current.png", png_header, "image/png"),
                "ideal_image": ("ideal.png", png_header, "image/png"),
            },
            data={"progress": "-0.5"},
        )

        assert response.status_code == 400

        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert "0.0 and 1.0" in data["error"]["message"]

    def test_morph_with_invalid_progress_over_one(self, client, api_base_url):
        """Morph with progress > 1.0 should return VALIDATION_ERROR."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/morph",
            files={
                "current_image": ("current.png", png_header, "image/png"),
                "ideal_image": ("ideal.png", png_header, "image/png"),
            },
            data={"progress": "1.5"},
        )

        assert response.status_code == 400

        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"


class TestMorphStagesEndpoint:
    """Test cases for POST /api/v1/morph/stages endpoint."""

    def test_morph_stages_without_images_returns_validation_error(self, client, api_base_url):
        """Morph stages without images should return VALIDATION_ERROR."""
        response = client.post(f"{api_base_url}/morph/stages")

        assert response.status_code == 400

        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_morph_stages_with_invalid_stages_json(self, client, api_base_url):
        """Morph stages with invalid JSON should return VALIDATION_ERROR."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/morph/stages",
            files={
                "current_image": ("current.png", png_header, "image/png"),
                "ideal_image": ("ideal.png", png_header, "image/png"),
            },
            data={"stages": "invalid-json"},
        )

        assert response.status_code == 400

        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_morph_stages_with_out_of_range_values(self, client, api_base_url):
        """Morph stages with out of range values should return VALIDATION_ERROR."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/morph/stages",
            files={
                "current_image": ("current.png", png_header, "image/png"),
                "ideal_image": ("ideal.png", png_header, "image/png"),
            },
            data={"stages": json.dumps([0, 0.5, 1.5])},  # 1.5 is out of range
        )

        assert response.status_code == 400

        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_morph_stages_error_response_format(self, client, api_base_url):
        """Error response should follow standard format."""
        response = client.post(f"{api_base_url}/morph/stages")

        data = response.json()

        assert "success" in data
        assert data["success"] is False
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
