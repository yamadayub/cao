"""Unit tests for SwapCompositor."""

import base64
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.swap_compositor import SwapCompositor


@pytest.fixture
def swap_compositor():
    """Create a SwapCompositor instance."""
    return SwapCompositor()


@pytest.fixture
def sample_rgb_image():
    """Create a sample 100x100 RGB image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_bgr_image():
    """Create a sample 100x100 BGR image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


class TestSwapCompositor:
    """Tests for SwapCompositor."""

    def test_init(self):
        """Test compositor initialization."""
        compositor = SwapCompositor()
        assert compositor is not None

    def test_normalize_parts_dict_with_aliases(self, swap_compositor):
        """Test that part aliases are expanded."""
        parts = {"eyes": 0.8, "nose": 1.0}
        normalized = swap_compositor._normalize_parts_dict(parts)

        # "eyes" should expand to both left_eye and right_eye
        assert "left_eye" in normalized
        assert "right_eye" in normalized
        assert normalized["left_eye"] == 0.8
        assert normalized["right_eye"] == 0.8
        assert normalized["nose"] == 1.0

    def test_normalize_parts_dict_with_eyebrows_alias(self, swap_compositor):
        """Test that eyebrows alias is expanded."""
        parts = {"eyebrows": 0.5}
        normalized = swap_compositor._normalize_parts_dict(parts)

        assert "left_eyebrow" in normalized
        assert "right_eyebrow" in normalized
        assert normalized["left_eyebrow"] == 0.5
        assert normalized["right_eyebrow"] == 0.5

    def test_normalize_parts_dict_direct_parts(self, swap_compositor):
        """Test direct part names pass through."""
        parts = {"left_eye": 0.5, "nose": 1.0, "lips": 0.3}
        normalized = swap_compositor._normalize_parts_dict(parts)

        assert normalized["left_eye"] == 0.5
        assert normalized["nose"] == 1.0
        assert normalized["lips"] == 0.3

    def test_normalize_parts_dict_clamps_values(self, swap_compositor):
        """Test that intensity values are clamped to 0-1."""
        parts = {"nose": 1.5, "lips": -0.5}
        normalized = swap_compositor._normalize_parts_dict(parts)

        assert normalized["nose"] == 1.0
        assert normalized["lips"] == 0.0

    def test_blend_with_intensity_full(self, swap_compositor, sample_bgr_image):
        """Test blending with full intensity returns swapped image."""
        original = sample_bgr_image.copy()
        swapped = np.ones_like(sample_bgr_image) * 255  # White image

        result = swap_compositor._blend_with_intensity(
            original, swapped, intensity=1.0
        )

        np.testing.assert_array_equal(result, swapped)

    def test_blend_with_intensity_zero(self, swap_compositor, sample_bgr_image):
        """Test blending with zero intensity returns original."""
        original = sample_bgr_image.copy()
        swapped = np.ones_like(sample_bgr_image) * 255  # White image

        result = swap_compositor._blend_with_intensity(
            original, swapped, intensity=0.0
        )

        np.testing.assert_array_equal(result, original)

    def test_blend_with_intensity_half(self, swap_compositor):
        """Test blending with 50% intensity."""
        original = np.zeros((10, 10, 3), dtype=np.uint8)  # Black
        swapped = np.ones((10, 10, 3), dtype=np.uint8) * 254  # Almost white

        result = swap_compositor._blend_with_intensity(
            original, swapped, intensity=0.5
        )

        # Result should be approximately 127 (middle value)
        assert 120 <= result.mean() <= 134

    @patch("app.services.swap_compositor.get_face_detection_service")
    @patch("app.services.swap_compositor.PartBlender")
    def test_compose_parts_success(
        self, mock_part_blender_cls, mock_get_face_detection, swap_compositor, sample_bgr_image
    ):
        """Test successful parts composition."""
        # Mock face detection
        mock_detection_service = MagicMock()
        mock_get_face_detection.return_value = mock_detection_service

        # Mock PartBlender
        mock_blender = MagicMock()
        mock_blender.blend.return_value = sample_bgr_image.copy()
        mock_part_blender_cls.return_value = mock_blender

        parts = {"nose": 1.0, "lips": 0.5}

        result = swap_compositor.compose_parts(
            original=sample_bgr_image,
            swapped=sample_bgr_image,
            parts=parts,
        )

        assert result is not None
        assert result.shape == sample_bgr_image.shape

    def test_compose_parts_empty_parts(self, swap_compositor, sample_bgr_image):
        """Test composition with empty parts returns original."""
        result = swap_compositor.compose_parts(
            original=sample_bgr_image,
            swapped=sample_bgr_image,
            parts={},
        )

        np.testing.assert_array_equal(result, sample_bgr_image)

    def test_compose_parts_all_zero_intensity(self, swap_compositor, sample_bgr_image):
        """Test composition with all zero intensities returns original."""
        result = swap_compositor.compose_parts(
            original=sample_bgr_image,
            swapped=sample_bgr_image,
            parts={"nose": 0.0, "lips": 0.0},
        )

        np.testing.assert_array_equal(result, sample_bgr_image)


class TestSwapCompositorIntegration:
    """Integration tests for SwapCompositor."""

    def test_valid_part_names(self, swap_compositor):
        """Test that all valid part names are recognized."""
        valid_parts = [
            "left_eye",
            "right_eye",
            "left_eyebrow",
            "right_eyebrow",
            "nose",
            "lips",
            "eyes",  # alias
            "eyebrows",  # alias
        ]

        for part in valid_parts:
            normalized = swap_compositor._normalize_parts_dict({part: 1.0})
            assert len(normalized) > 0, f"Part '{part}' should be recognized"
