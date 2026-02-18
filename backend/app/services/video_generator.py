"""Morphing video generator service.

Generates a slider-style Before/After morphing video (9:16, 1080x1920)
suitable for TikTok, Instagram Reels, and YouTube Shorts.
"""

import logging
import os
import tempfile
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Video dimensions (9:16 vertical)
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30

# Layout
FACE_IMAGE_SIZE = 1080  # Square face image area
FACE_IMAGE_Y = 200  # Top offset for face image area

# Colors (BGR for OpenCV, RGB for PIL)
BG_COLOR_RGB = (255, 255, 255)
SLIDER_LINE_COLOR_BGR = (255, 255, 255)  # White slider line
SLIDER_HANDLE_COLOR_BGR = (255, 255, 255)  # White handle
SLIDER_SHADOW_COLOR_BGR = (180, 180, 180)  # Shadow
TEXT_COLOR_RGB = (64, 64, 64)
ACCENT_COLOR_RGB = (59, 130, 246)  # Blue-500
LABEL_BG_COLOR_RGB = (0, 0, 0)

# Timing (seconds)
HOLD_BEFORE = 0.5  # Hold on Before
SLIDE_FORWARD = 2.0  # Slide left to right
HOLD_AFTER = 0.5  # Hold on After
SLIDE_BACK = 0.5  # Slide right to left (fast)
HOLD_END = 0.5  # Final hold


def _ease_in_out(t: float) -> float:
    """Smooth ease-in-out interpolation."""
    return t * t * (3.0 - 2.0 * t)


class MorphVideoGenerator:
    """Generate slider-style morphing videos from Before/After images."""

    def __init__(self):
        """Initialize with fonts."""
        try:
            self.label_font = ImageFont.truetype(
                "/System/Library/Fonts/Hiragino Sans GB.ttc", 36
            )
            self.brand_font = ImageFont.truetype(
                "/System/Library/Fonts/Hiragino Sans GB.ttc", 32
            )
            self.small_font = ImageFont.truetype(
                "/System/Library/Fonts/Hiragino Sans GB.ttc", 24
            )
        except (OSError, IOError):
            self.label_font = ImageFont.load_default()
            self.brand_font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()

    def generate(self, source_image: bytes, result_image: bytes) -> bytes:
        """Generate a morphing video from Before/After images.

        Args:
            source_image: Before image bytes (JPEG/PNG)
            result_image: After image bytes (JPEG/PNG)

        Returns:
            MP4 video bytes
        """
        # Decode images
        before_img = self._decode_image(source_image)
        after_img = self._decode_image(result_image)

        # Resize both to face image area
        before_face = self._fit_and_crop(before_img, FACE_IMAGE_SIZE, FACE_IMAGE_SIZE)
        after_face = self._fit_and_crop(after_img, FACE_IMAGE_SIZE, FACE_IMAGE_SIZE)

        # Create base frame (background + branding)
        base_frame = self._create_base_frame()

        # Generate all frames
        frames = self._generate_frames(base_frame, before_face, after_face)

        # Encode to MP4
        return self._encode_to_mp4(frames)

    def _decode_image(self, image_data: bytes) -> np.ndarray:
        """Decode image bytes to OpenCV BGR array."""
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img

    def _fit_and_crop(
        self, img: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        """Resize and center-crop image to target dimensions."""
        h, w = img.shape[:2]
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Center crop
        x_start = (new_w - target_w) // 2
        y_start = (new_h - target_h) // 2
        return resized[y_start : y_start + target_h, x_start : x_start + target_w]

    def _create_base_frame(self) -> np.ndarray:
        """Create the base frame with background and branding elements.

        Uses PIL for text rendering, then converts to OpenCV format.
        """
        # Create white background with PIL
        pil_img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), BG_COLOR_RGB)
        draw = ImageDraw.Draw(pil_img)

        # Brand text area (below face image)
        brand_y = FACE_IMAGE_Y + FACE_IMAGE_SIZE + 80

        # "Before -> After" label
        label_text = "Before  \u2192  After"
        bbox = draw.textbbox((0, 0), label_text, font=self.label_font)
        text_w = bbox[2] - bbox[0]
        draw.text(
            ((VIDEO_WIDTH - text_w) // 2, brand_y),
            label_text,
            fill=TEXT_COLOR_RGB,
            font=self.label_font,
        )

        # Brand name
        brand_name = "Cao"
        bbox = draw.textbbox((0, 0), brand_name, font=self.brand_font)
        brand_w = bbox[2] - bbox[0]
        draw.text(
            ((VIDEO_WIDTH - brand_w) // 2, brand_y + 60),
            brand_name,
            fill=ACCENT_COLOR_RGB,
            font=self.brand_font,
        )

        # URL
        url_text = "cao-ai.com"
        bbox = draw.textbbox((0, 0), url_text, font=self.small_font)
        url_w = bbox[2] - bbox[0]
        draw.text(
            ((VIDEO_WIDTH - url_w) // 2, brand_y + 100),
            url_text,
            fill=(150, 150, 150),
            font=self.small_font,
        )

        # Convert PIL to OpenCV BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _create_slider_frame(
        self,
        base: np.ndarray,
        before_face: np.ndarray,
        after_face: np.ndarray,
        position: float,
    ) -> np.ndarray:
        """Create a single frame with the slider at given position.

        Args:
            base: Base frame (background + branding)
            before_face: Before face image (1080x1080 BGR)
            after_face: After face image (1080x1080 BGR)
            position: Slider position 0.0 (all Before) to 1.0 (all After)
        """
        frame = base.copy()
        y = FACE_IMAGE_Y

        # Calculate split point in pixels
        split_x = int(FACE_IMAGE_SIZE * position)
        split_x = max(0, min(FACE_IMAGE_SIZE, split_x))

        # Compose: left side = Before, right side = After
        if split_x > 0:
            frame[y : y + FACE_IMAGE_SIZE, 0:split_x] = before_face[:, 0:split_x]
        if split_x < FACE_IMAGE_SIZE:
            frame[y : y + FACE_IMAGE_SIZE, split_x:FACE_IMAGE_SIZE] = after_face[
                :, split_x:FACE_IMAGE_SIZE
            ]

        # Draw slider line (vertical white line with shadow)
        if 0 < split_x < FACE_IMAGE_SIZE:
            line_x = split_x
            # Shadow
            cv2.line(
                frame,
                (line_x + 2, y),
                (line_x + 2, y + FACE_IMAGE_SIZE),
                SLIDER_SHADOW_COLOR_BGR,
                3,
            )
            # Main line
            cv2.line(
                frame,
                (line_x, y),
                (line_x, y + FACE_IMAGE_SIZE),
                SLIDER_LINE_COLOR_BGR,
                3,
            )

            # Slider handle (circle in the middle)
            handle_y = y + FACE_IMAGE_SIZE // 2
            cv2.circle(frame, (line_x, handle_y), 20, SLIDER_SHADOW_COLOR_BGR, -1)
            cv2.circle(frame, (line_x, handle_y), 18, SLIDER_HANDLE_COLOR_BGR, -1)

            # Arrows on handle
            # Left arrow
            pts_left = np.array(
                [
                    [line_x - 8, handle_y],
                    [line_x - 2, handle_y - 6],
                    [line_x - 2, handle_y + 6],
                ],
                np.int32,
            )
            cv2.fillPoly(frame, [pts_left], SLIDER_SHADOW_COLOR_BGR)
            # Right arrow
            pts_right = np.array(
                [
                    [line_x + 8, handle_y],
                    [line_x + 2, handle_y - 6],
                    [line_x + 2, handle_y + 6],
                ],
                np.int32,
            )
            cv2.fillPoly(frame, [pts_right], SLIDER_SHADOW_COLOR_BGR)

        # Draw Before/After labels using PIL for proper text rendering
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_frame)

        # "Before" label (left side) - show when slider hasn't passed
        if split_x > 150:
            label = "Before"
            bbox = draw.textbbox((0, 0), label, font=self.small_font)
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1]
            lx = max(10, split_x // 2 - lw // 2)
            ly = y + 20
            # Semi-transparent background
            draw.rectangle(
                [lx - 8, ly - 4, lx + lw + 8, ly + lh + 4],
                fill=(0, 0, 0),
            )
            draw.text((lx, ly), label, fill=(255, 255, 255), font=self.small_font)

        # "After" label (right side) - show when slider hasn't covered it
        if split_x < FACE_IMAGE_SIZE - 150:
            label = "After"
            bbox = draw.textbbox((0, 0), label, font=self.small_font)
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1]
            lx = min(
                FACE_IMAGE_SIZE - lw - 10,
                split_x + (FACE_IMAGE_SIZE - split_x) // 2 - lw // 2,
            )
            ly = y + 20
            draw.rectangle(
                [lx - 8, ly - 4, lx + lw + 8, ly + lh + 4],
                fill=(0, 0, 0),
            )
            draw.text((lx, ly), label, fill=(255, 255, 255), font=self.small_font)

        return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

    def _generate_frames(
        self,
        base: np.ndarray,
        before_face: np.ndarray,
        after_face: np.ndarray,
    ) -> List[np.ndarray]:
        """Generate all video frames.

        Timeline:
        - 0.0s-0.5s: Hold Before (position=0)
        - 0.5s-2.5s: Slide left->right (position 0->1)
        - 2.5s-3.0s: Hold After (position=1)
        - 3.0s-3.5s: Slide right->left (position 1->0)
        - 3.5s-4.0s: Hold Before (position=0)
        """
        frames = []

        # Phase 1: Hold Before
        hold_before_frames = int(HOLD_BEFORE * FPS)
        frame_before = self._create_slider_frame(base, before_face, after_face, 0.0)
        for _ in range(hold_before_frames):
            frames.append(frame_before)

        # Phase 2: Slide forward (Before -> After)
        slide_forward_frames = int(SLIDE_FORWARD * FPS)
        for i in range(slide_forward_frames):
            t = i / max(slide_forward_frames - 1, 1)
            position = _ease_in_out(t)
            frame = self._create_slider_frame(base, before_face, after_face, position)
            frames.append(frame)

        # Phase 3: Hold After
        hold_after_frames = int(HOLD_AFTER * FPS)
        frame_after = self._create_slider_frame(base, before_face, after_face, 1.0)
        for _ in range(hold_after_frames):
            frames.append(frame_after)

        # Phase 4: Slide back (After -> Before)
        slide_back_frames = int(SLIDE_BACK * FPS)
        for i in range(slide_back_frames):
            t = i / max(slide_back_frames - 1, 1)
            position = 1.0 - _ease_in_out(t)
            frame = self._create_slider_frame(base, before_face, after_face, position)
            frames.append(frame)

        # Phase 5: Hold end (Before)
        hold_end_frames = int(HOLD_END * FPS)
        for _ in range(hold_end_frames):
            frames.append(frame_before)

        return frames

    def _encode_to_mp4(self, frames: List[np.ndarray]) -> bytes:
        """Encode frames to MP4 bytes using OpenCV VideoWriter.

        Uses mp4v (MPEG-4 Part 2) codec which is built into OpenCV
        and requires no external ffmpeg dependency.

        Args:
            frames: List of BGR OpenCV frames

        Returns:
            MP4 file bytes
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            tmp_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT)
        )

        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter with mp4v codec")

        for frame in frames:
            writer.write(frame)

        writer.release()

        with open(tmp_path, "rb") as f:
            data = f.read()
        os.unlink(tmp_path)

        return data


def get_video_generator() -> MorphVideoGenerator:
    """Get MorphVideoGenerator instance."""
    return MorphVideoGenerator()
