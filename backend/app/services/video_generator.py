"""Morphing video generator service.

Generates a slider-style Before/After morphing video (9:16, 1080x1920)
suitable for TikTok, Instagram Reels, and YouTube Shorts.
"""

import logging
import os
import tempfile
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Video dimensions (9:16 vertical)
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30

# Layout
FACE_IMAGE_SIZE = 1080  # Square face image area
FACE_IMAGE_Y = 200  # Top offset for face image area

# Colors (BGR for OpenCV)
BG_COLOR_BGR = (255, 255, 255)
SLIDER_LINE_COLOR_BGR = (255, 255, 255)  # White slider line
SLIDER_HANDLE_COLOR_BGR = (255, 255, 255)  # White handle
SLIDER_SHADOW_COLOR_BGR = (180, 180, 180)  # Shadow
TEXT_COLOR_BGR = (64, 64, 64)
ACCENT_COLOR_BGR = (246, 130, 59)  # Blue-500 in BGR
LABEL_BG_COLOR_BGR = (0, 0, 0)

# Timing (seconds)
HOLD_BEFORE = 0.5  # Hold on Before
SLIDE_FORWARD = 2.0  # Slide left to right
HOLD_AFTER = 0.5  # Hold on After
SLIDE_BACK = 0.5  # Slide right to left (fast)
HOLD_END = 0.5  # Final hold

# Codec fallback chain: (fourcc, extension, content_type)
# avc1 (H.264 / MP4) is the most widely browser-supported format.
# VP80 (WebM/VP8) is browser-native but needs libvpx (not always available).
# mp4v (MPEG-4 Part 2) plays in most modern browsers.
# MJPG is a last-resort universal fallback.
CODEC_CHAIN: List[Tuple[str, str, str]] = [
    ("avc1", ".mp4", "video/mp4"),
    ("VP80", ".webm", "video/webm"),
    ("mp4v", ".mp4", "video/mp4"),
    ("MJPG", ".avi", "video/x-msvideo"),
]


class VideoResult:
    """Result of video generation with format metadata."""

    def __init__(self, data: bytes, content_type: str, extension: str):
        self.data = data
        self.content_type = content_type
        self.extension = extension


def _ease_in_out(t: float) -> float:
    """Smooth ease-in-out interpolation."""
    return t * t * (3.0 - 2.0 * t)


def _put_centered_text(
    frame: np.ndarray,
    text: str,
    center_x: int,
    y: int,
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    """Draw text centered horizontally at given position."""
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = center_x - tw // 2
    cv2.putText(frame, text, (x, y + th), font, font_scale, color, thickness, cv2.LINE_AA)


class MorphVideoGenerator:
    """Generate slider-style morphing videos from Before/After images."""

    def __init__(self):
        """Initialize generator (no external font dependencies)."""
        pass

    def generate(self, source_image: bytes, result_image: bytes) -> VideoResult:
        """Generate a morphing video from Before/After images.

        Args:
            source_image: Before image bytes (JPEG/PNG)
            result_image: After image bytes (JPEG/PNG)

        Returns:
            VideoResult with video bytes and format metadata
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

        # Encode to video
        return self._encode_video(frames)

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

        Uses cv2.putText() with Hershey fonts (no external font dependencies).
        """
        frame = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), 255, dtype=np.uint8)

        # Brand text area (below face image)
        brand_y = FACE_IMAGE_Y + FACE_IMAGE_SIZE + 80
        center_x = VIDEO_WIDTH // 2

        # "Before -> After" label
        _put_centered_text(
            frame, "Before  ->  After", center_x, brand_y,
            font_scale=1.2, color=TEXT_COLOR_BGR, thickness=2,
        )

        # Brand name "Cao"
        _put_centered_text(
            frame, "Cao", center_x, brand_y + 60,
            font_scale=1.4, color=ACCENT_COLOR_BGR, thickness=3,
        )

        # URL "cao-ai.com"
        _put_centered_text(
            frame, "cao-ai.com", center_x, brand_y + 110,
            font_scale=0.8, color=(150, 150, 150), thickness=2,
        )

        return frame

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

        # Draw Before/After labels directly with OpenCV
        # "Before" label (left side) - show when slider hasn't passed
        if split_x > 150:
            label = "Before"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (lw, lh), _ = cv2.getTextSize(label, font, font_scale, thickness)
            lx = max(10, split_x // 2 - lw // 2)
            ly = y + 20
            # Background rectangle
            cv2.rectangle(
                frame,
                (lx - 8, ly - 4),
                (lx + lw + 8, ly + lh + 8),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame, label, (lx, ly + lh),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )

        # "After" label (right side) - show when slider hasn't covered it
        if split_x < FACE_IMAGE_SIZE - 150:
            label = "After"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (lw, lh), _ = cv2.getTextSize(label, font, font_scale, thickness)
            lx = min(
                FACE_IMAGE_SIZE - lw - 10,
                split_x + (FACE_IMAGE_SIZE - split_x) // 2 - lw // 2,
            )
            ly = y + 20
            # Background rectangle
            cv2.rectangle(
                frame,
                (lx - 8, ly - 4),
                (lx + lw + 8, ly + lh + 8),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame, label, (lx, ly + lh),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )

        return frame

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

    def _encode_video(self, frames: List[np.ndarray]) -> VideoResult:
        """Encode frames to a browser-playable video.

        Tries codecs in order: VP8 (WebM, browser-native) → mp4v → MJPG.

        Args:
            frames: List of BGR OpenCV frames

        Returns:
            VideoResult with video bytes and format metadata
        """
        for codec, ext, content_type in CODEC_CHAIN:
            try:
                path, data = self._try_encode(frames, codec, ext)
                if data and len(data) > 1024:
                    logger.info(
                        f"Video encoded with {codec} codec ({ext}): "
                        f"{len(data)} bytes"
                    )
                    if path:
                        try:
                            os.unlink(path)
                        except OSError:
                            pass
                    return VideoResult(
                        data=data,
                        content_type=content_type,
                        extension=ext,
                    )
                logger.warning(
                    f"Codec {codec} produced too-small output "
                    f"({len(data) if data else 0} bytes), trying next"
                )
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            except Exception as e:
                logger.warning(f"Codec {codec} failed: {e}")

        raise RuntimeError(
            "All video codecs failed: "
            + ", ".join(c for c, _, _ in CODEC_CHAIN)
        )

    def _try_encode(
        self, frames: List[np.ndarray], codec: str, ext: str
    ) -> Tuple[str | None, bytes | None]:
        """Attempt to encode frames with a specific codec.

        Returns (temp_file_path, video_bytes) or (None, None) on failure.
        The caller is responsible for cleaning up the temp file.
        """
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                tmp_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT)
            )

            if not writer.isOpened():
                logger.warning(f"VideoWriter failed to open with codec {codec}")
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                return None, None

            for frame in frames:
                writer.write(frame)

            writer.release()

            with open(tmp_path, "rb") as f:
                data = f.read()

            return tmp_path, data
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def get_video_generator() -> MorphVideoGenerator:
    """Get MorphVideoGenerator instance."""
    return MorphVideoGenerator()
