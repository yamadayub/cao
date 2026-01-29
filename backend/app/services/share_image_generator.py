"""Share image generator service for SNS sharing."""

import base64
from io import BytesIO
from typing import Literal, Optional

from PIL import Image, ImageDraw, ImageFont

# Template dimensions
TEMPLATE_DIMENSIONS = {
    "before_after": (1200, 630),
    "single": (1080, 1080),
    "parts_highlight": (1080, 1350),
}

# Colors
BACKGROUND_COLOR = (255, 255, 255)  # White
TEXT_COLOR = (64, 64, 64)  # Dark gray
ACCENT_COLOR = (59, 130, 246)  # Blue-500
LOGO_TEXT = "Cao"


class ShareImageGenerator:
    """Generate share images from simulation results."""

    def __init__(self):
        """Initialize the generator."""
        # Load fonts (use default if custom not available)
        try:
            self.title_font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 32)
            self.caption_font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 24)
            self.logo_font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 28)
        except (OSError, IOError):
            self.title_font = ImageFont.load_default()
            self.caption_font = ImageFont.load_default()
            self.logo_font = ImageFont.load_default()

    def generate(
        self,
        source_image: str,
        result_image: str,
        template: Literal["before_after", "single", "parts_highlight"],
        caption: Optional[str] = None,
        applied_parts: Optional[list[str]] = None,
    ) -> bytes:
        """Generate a share image.

        Args:
            source_image: Base64 encoded source image
            result_image: Base64 encoded result image
            template: Template type
            caption: Optional caption text
            applied_parts: Optional list of applied parts (for parts_highlight)

        Returns:
            PNG image bytes
        """
        # Validate template
        if template not in TEMPLATE_DIMENSIONS:
            raise ValueError(f"Unknown template: {template}")

        # Decode images
        source_img = self._decode_base64_image(source_image)
        result_img = self._decode_base64_image(result_image)

        # Get dimensions for template
        width, height = TEMPLATE_DIMENSIONS[template]

        # Generate based on template
        if template == "before_after":
            share_img = self._generate_before_after(source_img, result_img, width, height, caption)
        elif template == "single":
            share_img = self._generate_single(result_img, width, height, caption)
        elif template == "parts_highlight":
            share_img = self._generate_parts_highlight(
                result_img, width, height, caption, applied_parts
            )
        else:
            raise ValueError(f"Unknown template: {template}")

        # Convert to bytes
        buffer = BytesIO()
        share_img.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    def _decode_base64_image(self, base64_str: str) -> Image.Image:
        """Decode a base64 string to PIL Image."""
        # Handle data URL prefix
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]

        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data)).convert("RGB")

    def _generate_before_after(
        self,
        source_img: Image.Image,
        result_img: Image.Image,
        width: int,
        height: int,
        caption: Optional[str],
    ) -> Image.Image:
        """Generate before/after template (1200x630).

        Layout:
        - Left half (600px): source image
        - Right half (600px): result image
        - Bottom: caption and logo
        """
        canvas = Image.new("RGB", (width, height), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(canvas)

        # Calculate image area (leave space for caption/logo at bottom)
        image_height = height - 80 if caption else height - 50
        half_width = width // 2

        # Resize and place source image (left side)
        source_resized = self._fit_image(source_img, half_width, image_height)
        source_x = (half_width - source_resized.width) // 2
        source_y = (image_height - source_resized.height) // 2
        canvas.paste(source_resized, (source_x, source_y))

        # Resize and place result image (right side)
        result_resized = self._fit_image(result_img, half_width, image_height)
        result_x = half_width + (half_width - result_resized.width) // 2
        result_y = (image_height - result_resized.height) // 2
        canvas.paste(result_resized, (result_x, result_y))

        # Draw divider line
        draw.line([(half_width, 0), (half_width, image_height)], fill=(220, 220, 220), width=2)

        # Draw labels
        self._draw_text(draw, "Before", half_width // 2, image_height + 10, TEXT_COLOR, center=True)
        self._draw_text(draw, "After", half_width + half_width // 2, image_height + 10, TEXT_COLOR, center=True)

        # Draw caption if provided
        if caption:
            self._draw_text(draw, caption, width // 2, height - 50, TEXT_COLOR, center=True)

        # Draw logo
        self._draw_logo(draw, width - 60, height - 35)

        return canvas

    def _generate_single(
        self,
        result_img: Image.Image,
        width: int,
        height: int,
        caption: Optional[str],
    ) -> Image.Image:
        """Generate single image template (1080x1080).

        Layout:
        - Centered result image
        - Bottom: caption and logo
        """
        canvas = Image.new("RGB", (width, height), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(canvas)

        # Calculate image area
        padding = 40
        image_area = width - (padding * 2)
        image_height = height - 100 if caption else height - 60

        # Resize and center result image
        result_resized = self._fit_image(result_img, image_area, image_height - padding)
        x = (width - result_resized.width) // 2
        y = (image_height - result_resized.height) // 2
        canvas.paste(result_resized, (x, y))

        # Draw caption if provided
        if caption:
            self._draw_text(draw, caption, width // 2, height - 60, TEXT_COLOR, center=True)

        # Draw logo
        self._draw_logo(draw, width - 60, height - 35)

        return canvas

    def _generate_parts_highlight(
        self,
        result_img: Image.Image,
        width: int,
        height: int,
        caption: Optional[str],
        applied_parts: Optional[list[str]],
    ) -> Image.Image:
        """Generate parts highlight template (1080x1350).

        Layout:
        - Top: result image
        - Middle: applied parts list
        - Bottom: caption and logo
        """
        canvas = Image.new("RGB", (width, height), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(canvas)

        # Calculate image area (top portion)
        padding = 40
        image_height = height - 300 if applied_parts else height - 100

        # Resize and place result image
        result_resized = self._fit_image(result_img, width - (padding * 2), image_height - padding)
        x = (width - result_resized.width) // 2
        y = padding // 2
        canvas.paste(result_resized, (x, y))

        # Draw parts list if provided
        if applied_parts:
            parts_y = image_height + 20
            self._draw_text(draw, "Applied Parts", width // 2, parts_y, ACCENT_COLOR, center=True)

            parts_y += 40
            for part in applied_parts:
                part_display = self._get_part_display_name(part)
                self._draw_text(draw, f"  {part_display}", width // 2, parts_y, TEXT_COLOR, center=True)
                parts_y += 30

        # Draw caption if provided
        if caption:
            self._draw_text(draw, caption, width // 2, height - 60, TEXT_COLOR, center=True)

        # Draw logo
        self._draw_logo(draw, width - 60, height - 35)

        return canvas

    def _fit_image(
        self, img: Image.Image, max_width: int, max_height: int
    ) -> Image.Image:
        """Resize image to fit within max dimensions while maintaining aspect ratio."""
        # Calculate scale to fit
        width_ratio = max_width / img.width
        height_ratio = max_height / img.height
        scale = min(width_ratio, height_ratio)

        new_width = int(img.width * scale)
        new_height = int(img.height * scale)

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _draw_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        color: tuple,
        center: bool = False,
    ) -> None:
        """Draw text on the canvas."""
        font = self.caption_font

        if center:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            x = x - text_width // 2

        draw.text((x, y), text, fill=color, font=font)

    def _draw_logo(self, draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
        """Draw Cao logo."""
        draw.text((x, y), LOGO_TEXT, fill=ACCENT_COLOR, font=self.logo_font)

    def _get_part_display_name(self, part: str) -> str:
        """Get display name for a part."""
        part_names = {
            "left_eye": "Left Eye",
            "right_eye": "Right Eye",
            "left_eyebrow": "Left Eyebrow",
            "right_eyebrow": "Right Eyebrow",
            "nose": "Nose",
            "lips": "Lips",
            "eyes": "Eyes",
            "eyebrows": "Eyebrows",
        }
        return part_names.get(part, part.replace("_", " ").title())


def get_share_image_generator() -> ShareImageGenerator:
    """Get ShareImageGenerator instance."""
    return ShareImageGenerator()
