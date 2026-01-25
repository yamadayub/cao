"""Integration tests for blend API endpoints."""

import json

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


class TestBlendPartsEndpoint:
    """Test cases for POST /api/v1/blend/parts endpoint."""

    def test_blend_without_current_image_returns_validation_error(
        self, client, api_base_url
    ):
        """Blend without current_image should return VALIDATION_ERROR."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/blend/parts",
            files={"ideal_image": ("ideal.png", png_header, "image/png")},
            data={"parts": json.dumps({"nose": True})},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_blend_without_ideal_image_returns_validation_error(
        self, client, api_base_url
    ):
        """Blend without ideal_image should return VALIDATION_ERROR."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/blend/parts",
            files={"current_image": ("current.png", png_header, "image/png")},
            data={"parts": json.dumps({"nose": True})},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_blend_without_parts_returns_validation_error(self, client, api_base_url):
        """Blend without parts should return VALIDATION_ERROR."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/blend/parts",
            files={
                "current_image": ("current.png", png_header, "image/png"),
                "ideal_image": ("ideal.png", png_header, "image/png"),
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_blend_with_invalid_parts_json_returns_validation_error(
        self, client, api_base_url
    ):
        """Blend with invalid JSON should return VALIDATION_ERROR."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/blend/parts",
            files={
                "current_image": ("current.png", png_header, "image/png"),
                "ideal_image": ("ideal.png", png_header, "image/png"),
            },
            data={"parts": "not-valid-json"},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_blend_with_no_parts_selected_returns_validation_error(
        self, client, api_base_url
    ):
        """Blend with all parts set to false should return VALIDATION_ERROR."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/blend/parts",
            files={
                "current_image": ("current.png", png_header, "image/png"),
                "ideal_image": ("ideal.png", png_header, "image/png"),
            },
            data={
                "parts": json.dumps(
                    {
                        "left_eye": False,
                        "right_eye": False,
                        "left_eyebrow": False,
                        "right_eyebrow": False,
                        "nose": False,
                        "lips": False,
                    }
                )
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert "At least one part" in data["error"]["message"]

    def test_blend_with_empty_parts_object_returns_validation_error(
        self, client, api_base_url
    ):
        """Blend with empty parts object should return VALIDATION_ERROR."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/blend/parts",
            files={
                "current_image": ("current.png", png_header, "image/png"),
                "ideal_image": ("ideal.png", png_header, "image/png"),
            },
            data={"parts": json.dumps({})},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_blend_with_unknown_part_returns_validation_error(
        self, client, api_base_url
    ):
        """Blend with unknown part should return VALIDATION_ERROR."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/blend/parts",
            files={
                "current_image": ("current.png", png_header, "image/png"),
                "ideal_image": ("ideal.png", png_header, "image/png"),
            },
            data={"parts": json.dumps({"unknown_part": True, "nose": True})},
        )

        # Unknown parts should be ignored, but at least one valid part (nose) is selected
        # So this should proceed to face detection, which will fail on invalid image
        assert response.status_code in [400, 422]

    def test_error_response_format(self, client, api_base_url):
        """Error response should follow standard format."""
        response = client.post(f"{api_base_url}/blend/parts")

        data = response.json()
        assert "success" in data
        assert data["success"] is False
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]

    def test_blend_with_invalid_image_format_returns_error(self, client, api_base_url):
        """Blend with invalid image format should return error."""
        invalid_image = b"not an image at all"

        response = client.post(
            f"{api_base_url}/blend/parts",
            files={
                "current_image": ("current.txt", invalid_image, "text/plain"),
                "ideal_image": ("ideal.txt", invalid_image, "text/plain"),
            },
            data={"parts": json.dumps({"nose": True})},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "INVALID_IMAGE_FORMAT"


class TestBlendPartsValidSelection:
    """Test cases for valid parts selection scenarios."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def api_base_url(self):
        """Base URL for API endpoints."""
        return "/api/v1"

    def test_valid_single_part_selection(self, client, api_base_url):
        """Valid single part selection should be accepted (may fail at face detection)."""
        # Create a minimal PNG that won't be detected as a face
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/blend/parts",
            files={
                "current_image": ("current.png", png_header, "image/png"),
                "ideal_image": ("ideal.png", png_header, "image/png"),
            },
            data={"parts": json.dumps({"nose": True})},
        )

        # Will fail at face detection, but should pass validation
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] in ["FACE_NOT_DETECTED", "INVALID_IMAGE_FORMAT"]

    def test_valid_multiple_parts_selection(self, client, api_base_url):
        """Valid multiple parts selection should be accepted."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/blend/parts",
            files={
                "current_image": ("current.png", png_header, "image/png"),
                "ideal_image": ("ideal.png", png_header, "image/png"),
            },
            data={
                "parts": json.dumps(
                    {"left_eye": True, "right_eye": True, "nose": True, "lips": True}
                )
            },
        )

        # Will fail at face detection, but should pass validation
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] in ["FACE_NOT_DETECTED", "INVALID_IMAGE_FORMAT"]

    def test_valid_all_parts_selection(self, client, api_base_url):
        """Valid all parts selection should be accepted."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = client.post(
            f"{api_base_url}/blend/parts",
            files={
                "current_image": ("current.png", png_header, "image/png"),
                "ideal_image": ("ideal.png", png_header, "image/png"),
            },
            data={
                "parts": json.dumps(
                    {
                        "left_eye": True,
                        "right_eye": True,
                        "left_eyebrow": True,
                        "right_eyebrow": True,
                        "nose": True,
                        "lips": True,
                    }
                )
            },
        )

        # Will fail at face detection, but should pass validation
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] in ["FACE_NOT_DETECTED", "INVALID_IMAGE_FORMAT"]
