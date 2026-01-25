"""Unit tests for PartBlender service."""

import json
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.services.part_blender import (
    FACIAL_PART_LANDMARKS,
    PartsSelection,
)


class TestPartsSelection:
    """Test cases for PartsSelection model."""

    def test_valid_parts_selection(self):
        """Valid parts selection should be created successfully."""
        parts = PartsSelection(
            left_eye=True,
            right_eye=True,
            left_eyebrow=False,
            right_eyebrow=False,
            nose=True,
            lips=False,
        )
        assert parts.left_eye is True
        assert parts.right_eye is True
        assert parts.nose is True
        assert parts.lips is False

    def test_default_values(self):
        """Default values should be False."""
        parts = PartsSelection()
        assert parts.left_eye is False
        assert parts.right_eye is False
        assert parts.left_eyebrow is False
        assert parts.right_eyebrow is False
        assert parts.nose is False
        assert parts.lips is False

    def test_get_selected_parts_empty(self):
        """get_selected_parts should return empty list when no parts selected."""
        parts = PartsSelection()
        assert parts.get_selected_parts() == []

    def test_get_selected_parts_some(self):
        """get_selected_parts should return list of selected part names."""
        parts = PartsSelection(left_eye=True, nose=True, lips=True)
        selected = parts.get_selected_parts()
        assert "left_eye" in selected
        assert "nose" in selected
        assert "lips" in selected
        assert len(selected) == 3

    def test_has_any_selection_true(self):
        """has_any_selection should return True when at least one part is selected."""
        parts = PartsSelection(nose=True)
        assert parts.has_any_selection() is True

    def test_has_any_selection_false(self):
        """has_any_selection should return False when no parts selected."""
        parts = PartsSelection()
        assert parts.has_any_selection() is False

    def test_from_dict(self):
        """Parts selection should be created from dictionary."""
        data = {"left_eye": True, "nose": True, "lips": False}
        parts = PartsSelection(**data)
        assert parts.left_eye is True
        assert parts.nose is True
        assert parts.lips is False


class TestFacialPartLandmarks:
    """Test cases for FACIAL_PART_LANDMARKS constant."""

    def test_all_parts_defined(self):
        """All facial parts should have landmark definitions."""
        expected_parts = [
            "left_eye",
            "right_eye",
            "left_eyebrow",
            "right_eyebrow",
            "nose",
            "lips",
        ]
        for part in expected_parts:
            assert part in FACIAL_PART_LANDMARKS
            assert len(FACIAL_PART_LANDMARKS[part]) > 0

    def test_landmarks_are_integers(self):
        """Landmark indices should be integers."""
        for part, indices in FACIAL_PART_LANDMARKS.items():
            for idx in indices:
                assert isinstance(idx, int)
                assert 0 <= idx <= 477  # MediaPipe Face Mesh has 478 landmarks


class TestPartBlenderMethods:
    """Test cases for PartBlender internal methods using mocks."""

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:] = (128, 128, 128)
        return img

    @patch("app.services.part_blender.get_face_detection_service")
    def test_extract_part_mask_returns_mask(self, mock_get_service, test_image):
        """extract_part_mask should return a binary mask."""
        from app.services.part_blender import PartBlender

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        blender = PartBlender()
        landmarks = [(100 + i, 100 + i % 50) for i in range(478)]
        part_indices = FACIAL_PART_LANDMARKS["nose"]

        mask = blender._extract_part_mask(landmarks, part_indices, test_image.shape[:2])

        assert mask is not None
        assert mask.shape == test_image.shape[:2]
        assert mask.dtype == np.uint8

    @patch("app.services.part_blender.get_face_detection_service")
    def test_extract_part_region_shape(self, mock_get_service, test_image):
        """extract_part_region should return correct shapes."""
        from app.services.part_blender import PartBlender

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        blender = PartBlender()
        landmarks = [(100 + i % 100, 100 + i % 100) for i in range(478)]
        part_indices = FACIAL_PART_LANDMARKS["left_eye"]

        region, mask, center = blender._extract_part_region(
            test_image, landmarks, part_indices
        )

        assert region is not None
        assert mask is not None
        assert center is not None
        assert len(center) == 2

    @patch("app.services.part_blender.get_face_detection_service")
    def test_calculate_part_transform(self, mock_get_service):
        """calculate_part_transform should compute affine transform."""
        from app.services.part_blender import PartBlender

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        blender = PartBlender()
        src_landmarks = [(50 + i, 50 + i % 30) for i in range(478)]
        dst_landmarks = [(55 + i, 55 + i % 30) for i in range(478)]
        part_indices = FACIAL_PART_LANDMARKS["nose"]

        transform = blender._calculate_part_transform(
            src_landmarks, dst_landmarks, part_indices
        )

        assert transform is not None
        assert transform.shape == (2, 3)

    @patch("app.services.part_blender.get_face_detection_service")
    def test_color_transfer(self, mock_get_service, test_image):
        """color_transfer should adjust colors of source to match target."""
        from app.services.part_blender import PartBlender

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        blender = PartBlender()
        src = np.zeros((100, 100, 3), dtype=np.uint8)
        src[:] = (200, 100, 50)
        tgt = np.zeros((100, 100, 3), dtype=np.uint8)
        tgt[:] = (50, 100, 200)
        mask = np.ones((100, 100), dtype=np.uint8) * 255

        result = blender._color_transfer(src, tgt, mask)

        assert result is not None
        assert result.shape == src.shape
        assert result.dtype == np.uint8

    @patch("app.services.part_blender.get_face_detection_service")
    def test_seamless_blend(self, mock_get_service, test_image):
        """seamless_blend should blend part into base image."""
        from app.services.part_blender import PartBlender

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        blender = PartBlender()
        base = test_image.copy()
        part = np.ones((200, 200, 3), dtype=np.uint8) * 200
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255
        center = (100, 100)

        result = blender._seamless_blend(base, part, mask, center)

        assert result is not None
        assert result.shape == base.shape
        assert result.dtype == np.uint8


class TestPartBlenderIntegration:
    """Integration tests for PartBlender.blend method."""

    @patch("app.services.part_blender.get_face_detection_service")
    def test_blend_no_parts_selected_raises_error(self, mock_get_service):
        """blend should raise error when no parts are selected."""
        from app.services.part_blender import PartBlender

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        blender = PartBlender()
        parts = PartsSelection()
        img1 = np.zeros((200, 200, 3), dtype=np.uint8)
        img2 = np.zeros((200, 200, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="At least one part must be selected"):
            blender.blend(img1, img2, parts)


class TestGetPartBlenderService:
    """Test cases for get_part_blender_service function."""

    @patch("app.services.part_blender.get_face_detection_service")
    def test_returns_instance(self, mock_get_service):
        """get_part_blender_service should return a PartBlender instance."""
        # Reset singleton for test
        import app.services.part_blender as module
        module._part_blender_service = None

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        from app.services.part_blender import PartBlender, get_part_blender_service
        service = get_part_blender_service()
        assert isinstance(service, PartBlender)

    @patch("app.services.part_blender.get_face_detection_service")
    def test_returns_same_instance(self, mock_get_service):
        """get_part_blender_service should return singleton instance."""
        # Reset singleton for test
        import app.services.part_blender as module
        module._part_blender_service = None

        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        from app.services.part_blender import get_part_blender_service
        service1 = get_part_blender_service()
        service2 = get_part_blender_service()
        assert service1 is service2
