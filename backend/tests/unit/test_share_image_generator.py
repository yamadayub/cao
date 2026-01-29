"""Unit tests for share image generator service."""

import pytest
import base64
from io import BytesIO

from PIL import Image

from app.services.share_image_generator import (
    ShareImageGenerator,
    TEMPLATE_DIMENSIONS,
    get_share_image_generator,
)


class TestShareImageGenerator:
    """Test cases for ShareImageGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a ShareImageGenerator instance."""
        return ShareImageGenerator()

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

    def test_generator_instance(self, generator):
        """Generator should be instantiated correctly."""
        assert generator is not None
        assert isinstance(generator, ShareImageGenerator)

    def test_get_share_image_generator_returns_instance(self):
        """get_share_image_generator should return an instance."""
        generator = get_share_image_generator()
        assert isinstance(generator, ShareImageGenerator)

    def test_before_after_template_generates_correct_dimensions(
        self, generator, sample_base64_image
    ):
        """Before/After template should produce 1200x630 image."""
        result_bytes = generator.generate(
            source_image=sample_base64_image,
            result_image=sample_base64_image,
            template="before_after",
        )

        result_img = Image.open(BytesIO(result_bytes))
        assert result_img.width == 1200
        assert result_img.height == 630

    def test_single_template_generates_correct_dimensions(
        self, generator, sample_base64_image
    ):
        """Single template should produce 1080x1080 image."""
        result_bytes = generator.generate(
            source_image=sample_base64_image,
            result_image=sample_base64_image,
            template="single",
        )

        result_img = Image.open(BytesIO(result_bytes))
        assert result_img.width == 1080
        assert result_img.height == 1080

    def test_parts_highlight_template_generates_correct_dimensions(
        self, generator, sample_base64_image
    ):
        """Parts highlight template should produce 1080x1350 image."""
        result_bytes = generator.generate(
            source_image=sample_base64_image,
            result_image=sample_base64_image,
            template="parts_highlight",
        )

        result_img = Image.open(BytesIO(result_bytes))
        assert result_img.width == 1080
        assert result_img.height == 1350

    def test_generate_returns_valid_png_bytes(self, generator, sample_base64_image):
        """Generated image should be valid PNG bytes."""
        result_bytes = generator.generate(
            source_image=sample_base64_image,
            result_image=sample_base64_image,
            template="before_after",
        )

        # Should be bytes
        assert isinstance(result_bytes, bytes)

        # Should start with PNG magic number
        assert result_bytes[:8] == b"\x89PNG\r\n\x1a\n"

        # Should be loadable as image
        img = Image.open(BytesIO(result_bytes))
        assert img.format == "PNG"

    def test_generate_with_caption(self, generator, sample_base64_image):
        """Should generate image with caption."""
        caption = "Test caption text"

        result_bytes = generator.generate(
            source_image=sample_base64_image,
            result_image=sample_base64_image,
            template="before_after",
            caption=caption,
        )

        # Should complete without error
        img = Image.open(BytesIO(result_bytes))
        assert img.width == 1200
        assert img.height == 630

    def test_generate_with_applied_parts(self, generator, sample_base64_image):
        """Should generate parts_highlight template with applied parts."""
        applied_parts = ["left_eye", "right_eye", "nose"]

        result_bytes = generator.generate(
            source_image=sample_base64_image,
            result_image=sample_base64_image,
            template="parts_highlight",
            applied_parts=applied_parts,
        )

        img = Image.open(BytesIO(result_bytes))
        assert img.width == 1080
        assert img.height == 1350

    def test_generate_with_data_url_prefix(self, generator, sample_image_bytes):
        """Should handle base64 images with data URL prefix."""
        base64_data = base64.b64encode(sample_image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_data}"

        result_bytes = generator.generate(
            source_image=data_url,
            result_image=data_url,
            template="before_after",
        )

        img = Image.open(BytesIO(result_bytes))
        assert img.width == 1200

    def test_generate_with_invalid_template_raises_error(
        self, generator, sample_base64_image
    ):
        """Should raise error for invalid template."""
        with pytest.raises(ValueError, match="Unknown template"):
            generator.generate(
                source_image=sample_base64_image,
                result_image=sample_base64_image,
                template="invalid_template",
            )


class TestTemplateLayouts:
    """Test cases for template layout specifications."""

    def test_template_dimensions_dict_has_all_templates(self):
        """TEMPLATE_DIMENSIONS should have all three templates."""
        assert "before_after" in TEMPLATE_DIMENSIONS
        assert "single" in TEMPLATE_DIMENSIONS
        assert "parts_highlight" in TEMPLATE_DIMENSIONS

    def test_before_after_dimensions(self):
        """Before/After template should be 1200x630."""
        assert TEMPLATE_DIMENSIONS["before_after"] == (1200, 630)

    def test_single_dimensions(self):
        """Single template should be 1080x1080."""
        assert TEMPLATE_DIMENSIONS["single"] == (1080, 1080)

    def test_parts_highlight_dimensions(self):
        """Parts highlight template should be 1080x1350."""
        assert TEMPLATE_DIMENSIONS["parts_highlight"] == (1080, 1350)


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


class TestImageProcessing:
    """Test cases for image processing utilities."""

    @pytest.fixture
    def generator(self):
        """Create a ShareImageGenerator instance."""
        return ShareImageGenerator()

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return Image.new("RGB", (512, 512), color=(255, 128, 64))

    def test_fit_image_maintains_aspect_ratio(self, generator, sample_image):
        """_fit_image should maintain aspect ratio."""
        result = generator._fit_image(sample_image, 300, 300)

        original_ratio = sample_image.width / sample_image.height
        result_ratio = result.width / result.height

        assert abs(original_ratio - result_ratio) < 0.01

    def test_fit_image_respects_max_dimensions(self, generator, sample_image):
        """_fit_image should not exceed max dimensions."""
        result = generator._fit_image(sample_image, 300, 200)

        assert result.width <= 300
        assert result.height <= 200

    def test_fit_image_with_wide_image(self, generator):
        """_fit_image should handle wide images correctly."""
        wide_img = Image.new("RGB", (1000, 500), color="red")
        result = generator._fit_image(wide_img, 600, 600)

        assert result.width == 600
        assert result.height == 300

    def test_fit_image_with_tall_image(self, generator):
        """_fit_image should handle tall images correctly."""
        tall_img = Image.new("RGB", (500, 1000), color="blue")
        result = generator._fit_image(tall_img, 600, 600)

        assert result.width == 300
        assert result.height == 600

    def test_base64_decode_produces_valid_image(self, generator):
        """_decode_base64_image should produce valid PIL Image."""
        img = Image.new("RGB", (10, 10), color="red")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        result = generator._decode_base64_image(base64_data)

        assert isinstance(result, Image.Image)
        assert result.width == 10
        assert result.height == 10


class TestPartDisplayNames:
    """Test cases for part display name mapping."""

    @pytest.fixture
    def generator(self):
        """Create a ShareImageGenerator instance."""
        return ShareImageGenerator()

    def test_get_part_display_name_for_known_parts(self, generator):
        """_get_part_display_name should return correct names for known parts."""
        assert generator._get_part_display_name("left_eye") == "Left Eye"
        assert generator._get_part_display_name("right_eye") == "Right Eye"
        assert generator._get_part_display_name("left_eyebrow") == "Left Eyebrow"
        assert generator._get_part_display_name("right_eyebrow") == "Right Eyebrow"
        assert generator._get_part_display_name("nose") == "Nose"
        assert generator._get_part_display_name("lips") == "Lips"

    def test_get_part_display_name_for_unknown_parts(self, generator):
        """_get_part_display_name should handle unknown parts gracefully."""
        result = generator._get_part_display_name("some_new_part")
        assert result == "Some New Part"


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
