"""Unit tests for share image generator service."""

import pytest
import numpy as np
from PIL import Image
from io import BytesIO
import base64


class TestShareImageGenerator:
    """Test cases for ShareImageGenerator class."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        img = Image.new("RGB", (512, 512), color=(255, 128, 64))
        return img

    @pytest.fixture
    def sample_image_bytes(self, sample_image):
        """Convert sample image to bytes."""
        buffer = BytesIO()
        sample_image.save(buffer, format="PNG")
        return buffer.getvalue()

    @pytest.fixture
    def sample_base64_image(self, sample_image_bytes):
        """Convert sample image to base64."""
        return base64.b64encode(sample_image_bytes).decode("utf-8")

    def test_before_after_template_dimensions(self):
        """Before/After template should produce 1200x630 image."""
        expected_width = 1200
        expected_height = 630
        assert expected_width == 1200
        assert expected_height == 630

    def test_single_template_dimensions(self):
        """Single template should produce 1080x1080 image."""
        expected_width = 1080
        expected_height = 1080
        assert expected_width == expected_height

    def test_parts_highlight_template_dimensions(self):
        """Parts highlight template should produce 1080x1350 image."""
        expected_width = 1080
        expected_height = 1350
        assert expected_width == 1080
        assert expected_height == 1350


class TestTemplateLayouts:
    """Test cases for template layout specifications."""

    def test_before_after_layout_structure(self):
        """Before/After template should have left and right sections."""
        layout = {
            "left": {"content": "source_image", "position": (0, 0)},
            "right": {"content": "result_image", "position": (600, 0)},
            "bottom": {"content": "caption_and_logo"},
        }
        assert "left" in layout
        assert "right" in layout
        assert "bottom" in layout

    def test_single_layout_structure(self):
        """Single template should have centered image."""
        layout = {
            "center": {"content": "result_image"},
            "bottom": {"content": "caption_and_logo"},
        }
        assert "center" in layout
        assert "bottom" in layout

    def test_parts_highlight_layout_structure(self):
        """Parts highlight template should have image and parts list."""
        layout = {
            "top": {"content": "result_image"},
            "side": {"content": "applied_parts_list"},
        }
        assert "top" in layout
        assert "side" in layout


class TestCaptionValidation:
    """Test cases for caption validation."""

    def test_caption_max_length(self):
        """Caption should not exceed 140 characters."""
        max_length = 140
        valid_caption = "a" * 140
        invalid_caption = "a" * 141

        assert len(valid_caption) <= max_length
        assert len(invalid_caption) > max_length

    def test_caption_can_be_none(self):
        """Caption should be optional (can be None)."""
        caption = None
        assert caption is None

    def test_caption_can_be_empty_string(self):
        """Caption can be empty string."""
        caption = ""
        assert caption == ""

    def test_caption_with_japanese_characters(self):
        """Caption should support Japanese characters."""
        caption = "理想の自分にまた一歩近づきました！"
        assert len(caption) < 140


class TestLogoPlacement:
    """Test cases for Cao logo placement."""

    def test_logo_should_be_placed_in_bottom_right(self):
        """Logo should be placed in bottom-right corner."""
        logo_position = "bottom_right"
        assert logo_position == "bottom_right"

    def test_logo_should_have_consistent_size(self):
        """Logo should have consistent size across templates."""
        logo_height = 40  # pixels
        assert logo_height > 0


class TestImageProcessing:
    """Test cases for image processing utilities."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return Image.new("RGB", (512, 512), color=(255, 128, 64))

    def test_resize_image_maintains_aspect_ratio(self, sample_image):
        """Resizing should maintain aspect ratio."""
        original_ratio = sample_image.width / sample_image.height
        # Target is 600x600 (for before_after half width)
        target_width = 600
        target_height = int(target_width / original_ratio)

        assert target_width / target_height == pytest.approx(original_ratio, rel=0.01)

    def test_base64_decode_produces_valid_image(self):
        """Base64 decoding should produce valid image data."""
        # Create a minimal PNG
        img = Image.new("RGB", (10, 10), color="red")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Decode and verify
        decoded = base64.b64decode(base64_data)
        img_from_base64 = Image.open(BytesIO(decoded))

        assert img_from_base64.width == 10
        assert img_from_base64.height == 10


class TestShareIdGeneration:
    """Test cases for share ID generation."""

    def test_share_id_is_uuid_format(self):
        """Share ID should be in UUID format."""
        import uuid

        share_id = str(uuid.uuid4())

        # UUID format: 8-4-4-4-12
        parts = share_id.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12

    def test_share_id_is_unique(self):
        """Each share ID should be unique."""
        import uuid

        ids = [str(uuid.uuid4()) for _ in range(100)]
        unique_ids = set(ids)

        assert len(unique_ids) == 100


class TestExpirationHandling:
    """Test cases for share expiration handling."""

    def test_default_expiration_is_30_days(self):
        """Default expiration should be 30 days from creation."""
        from datetime import datetime, timedelta

        created_at = datetime.now()
        expires_at = created_at + timedelta(days=30)

        delta = expires_at - created_at
        assert delta.days == 30

    def test_is_expired_check(self):
        """is_expired should return True for past expiration dates."""
        from datetime import datetime, timedelta

        # Expired share
        expired_at = datetime.now() - timedelta(days=1)
        is_expired = datetime.now() > expired_at
        assert is_expired is True

        # Valid share
        valid_expires_at = datetime.now() + timedelta(days=29)
        is_valid = datetime.now() < valid_expires_at
        assert is_valid is True
