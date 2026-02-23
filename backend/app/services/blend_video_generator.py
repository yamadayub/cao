"""Blend reveal video generator — TikTok-optimized patterns.

Two patterns built around snap cuts and seamless loop bridges:

Pattern A (default, ~4.0s, loop-optimized):
  Before hold (1.5s) → SNAP CUT (1 frame) → After hold (2.0s)
  → Loop bridge crossfade to Before (0.5s)

Pattern B (~6.0s, morph showcase):
  Before hold (1.0s) → SNAP CUT → After hold (1.5s)
  → Hard cut back to Before (0.5s) → Slow morph (2.5s)
  → Loop bridge (0.5s)

Both patterns include:
- "Before"/"After" labels — bottom-left, white text on semi-transparent tag
- Cao watermark — small logo (64px) bottom-right, 35% opacity
- No full-screen logo, no end card

All images are cover-fitted to fill the entire frame (no borders).
Uses 720x1280 @ 30fps to fit within Heroku's 512MB / 30s limits.
"""

import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.services.video_generator import (
    VideoResult,
    get_ffmpeg_path,
)

# Logo image path (used as watermark)
_LOGO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "images", "cao-logo.jpg"
)

logger = logging.getLogger(__name__)

# ── Video settings ────────────────────────────
BLEND_WIDTH = 720
BLEND_HEIGHT = 1280
BLEND_FPS = 30

# Only codecs that actually work on Heroku's opencv-python-headless.
BLEND_CODEC_CHAIN: List[Tuple[str, str, str]] = [
    ("mp4v", ".mp4", "video/mp4"),
    ("MJPG", ".avi", "video/x-msvideo"),
]

# ── Pattern A timeline (seconds) ──────────────
PA_BEFORE_HOLD = 1.5   # Show before face
PA_AFTER_HOLD = 2.0    # Show after face
PA_LOOP_BRIDGE = 0.5   # Crossfade back to before for seamless loop
PA_TOTAL = PA_BEFORE_HOLD + PA_AFTER_HOLD + PA_LOOP_BRIDGE  # 4.0s

# ── Pattern B timeline (seconds) ──────────────
PB_BEFORE_HOLD = 1.0   # Show before face
PB_AFTER_HOLD = 1.5    # Show after face
PB_HARD_CUT = 0.5      # Hard cut back to before
PB_SLOW_MORPH = 2.5    # Slow morph before → after
PB_LOOP_BRIDGE = 0.5   # Crossfade back to before for seamless loop
PB_TOTAL = (PB_BEFORE_HOLD + PB_AFTER_HOLD + PB_HARD_CUT
            + PB_SLOW_MORPH + PB_LOOP_BRIDGE)  # 6.0s

# ── Label / watermark config ─────────────────
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 1.0
LABEL_THICKNESS = 2
LABEL_BG_ALPHA = 0.6       # Semi-transparent black background
LABEL_MARGIN_X = 24        # Left margin
LABEL_MARGIN_Y = 24        # Bottom margin from frame bottom
LABEL_PADDING_X = 12       # Padding inside tag
LABEL_PADDING_Y = 8

WATERMARK_SIZE = 64         # 64x64 pixels
WATERMARK_ALPHA = 0.35      # 35% opacity
WATERMARK_MARGIN = 24       # Margin from edges

# "After" label fade-in: 0.2s = 6 frames at 30fps
AFTER_LABEL_FADE_FRAMES = 6


class BlendVideoGenerator:
    """Generate TikTok-optimized blend-reveal videos."""

    # ── public API ──────────────────────────

    def generate(
        self,
        current_image: bytes,
        ideal_image: Optional[bytes],
        result_image: bytes,
        pattern: str = "A",
    ) -> VideoResult:
        """Generate blend-reveal video.

        Args:
            current_image: Before face image bytes.
            ideal_image: Ignored (backward compatibility).
            result_image: After face image bytes.
            pattern: "A" (4s loop) or "B" (6s morph showcase).

        Returns:
            VideoResult with video bytes, duration, and metadata.
        """
        before = self._fit(self._decode(current_image))
        after = self._fit(self._decode(result_image))
        return self._generate_and_encode(before, after, pattern)

    # ── image helpers ───────────────────────

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img

    @staticmethod
    def _fit(img: np.ndarray) -> np.ndarray:
        """Cover-fit image to full 720x1280 frame (no borders)."""
        h, w = img.shape[:2]
        scale = max(BLEND_WIDTH / w, BLEND_HEIGHT / h)
        nw, nh = max(int(w * scale), BLEND_WIDTH), max(int(h * scale), BLEND_HEIGHT)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        x0 = (nw - BLEND_WIDTH) // 2
        y0 = (nh - BLEND_HEIGHT) // 2
        cropped = resized[y0 : y0 + BLEND_HEIGHT, x0 : x0 + BLEND_WIDTH]
        if cropped.shape[:2] != (BLEND_HEIGHT, BLEND_WIDTH):
            cropped = cv2.resize(cropped, (BLEND_WIDTH, BLEND_HEIGHT))
        return np.ascontiguousarray(cropped)

    # ── labels ──────────────────────────────

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        text: str,
        bg_alpha: float = LABEL_BG_ALPHA,
    ) -> np.ndarray:
        """Draw text label at bottom-left with semi-transparent black background.

        Returns a new frame (does not modify in-place).
        """
        frame = frame.copy()
        (tw, th), baseline = cv2.getTextSize(
            text, LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS
        )

        # Position: bottom-left
        x = LABEL_MARGIN_X
        y = BLEND_HEIGHT - LABEL_MARGIN_Y - LABEL_PADDING_Y - th

        # Background rectangle region
        rx1 = x - LABEL_PADDING_X
        ry1 = y - LABEL_PADDING_Y
        rx2 = x + tw + LABEL_PADDING_X
        ry2 = y + th + baseline + LABEL_PADDING_Y

        # Clamp to frame bounds
        rx1 = max(0, rx1)
        ry1 = max(0, ry1)
        rx2 = min(BLEND_WIDTH, rx2)
        ry2 = min(BLEND_HEIGHT, ry2)

        # Alpha-blend black rectangle
        roi = frame[ry1:ry2, rx1:rx2].astype(np.float32)
        black = np.zeros_like(roi, dtype=np.float32)
        blended = roi * (1.0 - bg_alpha) + black * bg_alpha
        frame[ry1:ry2, rx1:rx2] = blended.clip(0, 255).astype(np.uint8)

        # Draw white text
        cv2.putText(
            frame, text, (x, y + th),
            LABEL_FONT, LABEL_FONT_SCALE, (255, 255, 255),
            LABEL_THICKNESS, cv2.LINE_AA,
        )
        return frame

    # ── watermark ───────────────────────────

    def _load_watermark(self) -> np.ndarray:
        """Load and cache the Cao logo as a 64x64 watermark."""
        if not hasattr(self, "_watermark_cache"):
            logo = cv2.imread(_LOGO_PATH, cv2.IMREAD_COLOR)
            if logo is None:
                logger.warning(f"Logo not found at {_LOGO_PATH}, using blank")
                self._watermark_cache = np.zeros(
                    (WATERMARK_SIZE, WATERMARK_SIZE, 3), dtype=np.uint8
                )
            else:
                self._watermark_cache = cv2.resize(
                    logo, (WATERMARK_SIZE, WATERMARK_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
        return self._watermark_cache

    def _apply_watermark(self, frame: np.ndarray) -> np.ndarray:
        """Alpha-blend small logo at bottom-right with 35% opacity."""
        wm = self._load_watermark()
        h, w = wm.shape[:2]

        # Position: bottom-right
        x = BLEND_WIDTH - w - WATERMARK_MARGIN
        y = BLEND_HEIGHT - h - WATERMARK_MARGIN

        roi = frame[y : y + h, x : x + w].astype(np.float32)
        wm_f = wm.astype(np.float32)
        blended = roi * (1.0 - WATERMARK_ALPHA) + wm_f * WATERMARK_ALPHA
        frame[y : y + h, x : x + w] = blended.clip(0, 255).astype(np.uint8)
        return frame

    # ── transitions ─────────────────────────

    @staticmethod
    def _cross_dissolve(
        img_from: np.ndarray,
        img_to: np.ndarray,
        progress: float,
    ) -> np.ndarray:
        """Linear cross-dissolve between two frames."""
        alpha = max(0.0, min(1.0, progress))
        result = (
            img_to.astype(np.float32) * alpha
            + img_from.astype(np.float32) * (1.0 - alpha)
        )
        return result.clip(0, 255).astype(np.uint8)

    # ── pattern rendering ───────────────────

    def _render_frame_pattern_a(
        self,
        frame_index: int,
        total_frames: int,
        before: np.ndarray,
        after: np.ndarray,
    ) -> np.ndarray:
        """Render a single frame for Pattern A.

        Timeline (frame-based at 30fps):
        - Frames 0-44 (1.5s): Before hold + "Before" label
        - Frame 45: SNAP CUT to After
        - Frames 45-104 (2.0s): After hold + "After" label (fades in)
        - Frames 105-119 (0.5s): Loop bridge crossfade After→Before
        """
        before_frames = int(PA_BEFORE_HOLD * BLEND_FPS)   # 45
        after_frames = int(PA_AFTER_HOLD * BLEND_FPS)      # 60
        # bridge_frames = total_frames - before_frames - after_frames  # 15

        if frame_index < before_frames:
            # Before hold
            frame = before.copy()
            frame = self._draw_label(frame, "Before")
        elif frame_index < before_frames + after_frames:
            # After hold
            frame = after.copy()
            # "After" label fades in over first AFTER_LABEL_FADE_FRAMES
            frames_into_after = frame_index - before_frames
            if frames_into_after < AFTER_LABEL_FADE_FRAMES:
                fade_alpha = LABEL_BG_ALPHA * (frames_into_after / AFTER_LABEL_FADE_FRAMES)
            else:
                fade_alpha = LABEL_BG_ALPHA
            frame = self._draw_label(frame, "After", bg_alpha=fade_alpha)
        else:
            # Loop bridge: crossfade After → Before
            bridge_start = before_frames + after_frames
            bridge_frames = total_frames - bridge_start
            progress = (frame_index - bridge_start) / max(bridge_frames - 1, 1)
            frame = self._cross_dissolve(after, before, progress)
            # Fade out "After" label during bridge
            remaining_alpha = LABEL_BG_ALPHA * (1.0 - progress)
            if remaining_alpha > 0.05:
                frame = self._draw_label(frame, "After", bg_alpha=remaining_alpha)

        return frame

    def _render_frame_pattern_b(
        self,
        frame_index: int,
        total_frames: int,
        before: np.ndarray,
        after: np.ndarray,
    ) -> np.ndarray:
        """Render a single frame for Pattern B.

        Timeline (frame-based at 30fps):
        - Frames 0-29 (1.0s): Before hold + "Before" label
        - Frame 30: SNAP CUT to After
        - Frames 30-74 (1.5s): After hold + "After" label (fades in)
        - Frames 75-89 (0.5s): Hard cut back to Before + "Before" label
        - Frames 90-164 (2.5s): Slow morph Before→After
        - Frames 165-179 (0.5s): Loop bridge crossfade After→Before
        """
        f_before = int(PB_BEFORE_HOLD * BLEND_FPS)     # 30
        f_after = int(PB_AFTER_HOLD * BLEND_FPS)        # 45
        f_hard = int(PB_HARD_CUT * BLEND_FPS)           # 15
        f_morph = int(PB_SLOW_MORPH * BLEND_FPS)        # 75
        # f_bridge = total_frames - f_before - f_after - f_hard - f_morph  # 15

        e1 = f_before
        e2 = e1 + f_after
        e3 = e2 + f_hard
        e4 = e3 + f_morph

        if frame_index < e1:
            # Before hold
            frame = before.copy()
            frame = self._draw_label(frame, "Before")
        elif frame_index < e2:
            # After hold
            frame = after.copy()
            frames_into_after = frame_index - e1
            if frames_into_after < AFTER_LABEL_FADE_FRAMES:
                fade_alpha = LABEL_BG_ALPHA * (frames_into_after / AFTER_LABEL_FADE_FRAMES)
            else:
                fade_alpha = LABEL_BG_ALPHA
            frame = self._draw_label(frame, "After", bg_alpha=fade_alpha)
        elif frame_index < e3:
            # Hard cut back to Before
            frame = before.copy()
            frame = self._draw_label(frame, "Before")
        elif frame_index < e4:
            # Slow morph Before → After
            progress = (frame_index - e3) / max(f_morph - 1, 1)
            frame = self._cross_dissolve(before, after, progress)
            # Morph label: transition from "Before" to "After"
            if progress < 0.5:
                frame = self._draw_label(frame, "Before", bg_alpha=LABEL_BG_ALPHA * (1.0 - progress * 2))
            else:
                alpha = LABEL_BG_ALPHA * ((progress - 0.5) * 2)
                if alpha > 0.05:
                    frame = self._draw_label(frame, "After", bg_alpha=alpha)
        else:
            # Loop bridge: crossfade After → Before
            bridge_start = e4
            bridge_frames = total_frames - bridge_start
            progress = (frame_index - bridge_start) / max(bridge_frames - 1, 1)
            frame = self._cross_dissolve(after, before, progress)
            remaining_alpha = LABEL_BG_ALPHA * (1.0 - progress)
            if remaining_alpha > 0.05:
                frame = self._draw_label(frame, "After", bg_alpha=remaining_alpha)

        return frame

    # ── frame dispatch ──────────────────────

    def _render_frame(
        self,
        frame_index: int,
        total_frames: int,
        before: np.ndarray,
        after: np.ndarray,
        pattern: str,
    ) -> np.ndarray:
        """Render a single frame, dispatching to the appropriate pattern."""
        if pattern == "B":
            frame = self._render_frame_pattern_b(
                frame_index, total_frames, before, after
            )
        else:
            frame = self._render_frame_pattern_a(
                frame_index, total_frames, before, after
            )

        # Apply watermark on every frame
        frame = self._apply_watermark(frame)

        # Safety: guarantee exact frame size and type for encoder
        if frame.shape != (BLEND_HEIGHT, BLEND_WIDTH, 3):
            frame = cv2.resize(frame, (BLEND_WIDTH, BLEND_HEIGHT))
        if frame.dtype != np.uint8:
            frame = frame.clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    # ── beat sync metadata ──────────────────

    @staticmethod
    def _get_beat_sync_points(pattern: str) -> List[float]:
        """Return timestamps (seconds) where snap cuts occur."""
        if pattern == "B":
            return [
                PB_BEFORE_HOLD,                          # Snap cut to After
                PB_BEFORE_HOLD + PB_AFTER_HOLD,          # Hard cut back to Before
            ]
        else:
            return [PA_BEFORE_HOLD]  # Single snap cut

    # ── encoding (streaming) ─────────────────

    def _generate_and_encode(
        self,
        before: np.ndarray,
        after: np.ndarray,
        pattern: str = "A",
    ) -> VideoResult:
        """Generate frames and encode to browser-playable H.264 video.

        Strategy:
        1. Try ffmpeg pipe (raw BGR → H.264 mp4) — browser-native playback.
        2. Fallback: OpenCV mp4v then ffmpeg re-encode to H.264.
        3. Last resort: OpenCV mp4v as-is (may not play in all browsers).
        """
        duration = PA_TOTAL if pattern != "B" else PB_TOTAL
        total_frames = int(duration * BLEND_FPS)
        ffmpeg_bin = get_ffmpeg_path()

        beat_sync = self._get_beat_sync_points(pattern)
        metadata: Dict = {
            "pattern": pattern,
            "loop_friendly": True,
            "beat_sync_points": beat_sync,
        }

        # ── Strategy 1: Pipe raw frames directly to ffmpeg ────
        if ffmpeg_bin:
            try:
                result_video = self._encode_with_ffmpeg_pipe(
                    before, after, total_frames, pattern, ffmpeg_bin
                )
                if result_video:
                    result_video.duration = duration
                    result_video.metadata = metadata
                    return result_video
            except Exception as e:
                logger.warning(f"ffmpeg pipe encoding failed: {e}")

        # ── Strategy 2 & 3: OpenCV write + optional ffmpeg re-encode ──
        for codec, ext, content_type in BLEND_CODEC_CHAIN:
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=ext, delete=False
                ) as tmp:
                    tmp_path = tmp.name

                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    tmp_path, fourcc, BLEND_FPS, (BLEND_WIDTH, BLEND_HEIGHT)
                )
                if not writer.isOpened():
                    os.unlink(tmp_path)
                    continue

                for i in range(total_frames):
                    frame = self._render_frame(
                        i, total_frames, before, after, pattern
                    )
                    writer.write(frame)
                writer.release()

                # Try to re-encode with ffmpeg for browser compatibility
                if ffmpeg_bin:
                    h264_data = self._ffmpeg_reencode(tmp_path, ffmpeg_bin)
                    os.unlink(tmp_path)
                    if h264_data and len(h264_data) > 1024:
                        logger.info(
                            f"Blend video re-encoded to H.264: "
                            f"{len(h264_data)} bytes, {duration:.1f}s"
                        )
                        return VideoResult(
                            h264_data, "video/mp4", ".mp4",
                            duration=duration, metadata=metadata,
                        )

                # Fallback: use mp4v as-is
                with open(tmp_path, "rb") as fh:
                    data = fh.read()
                os.unlink(tmp_path)

                if len(data) > 1024:
                    logger.info(
                        f"Blend video encoded ({codec}{ext}): "
                        f"{len(data)} bytes, {duration:.1f}s"
                    )
                    return VideoResult(
                        data, content_type, ext,
                        duration=duration, metadata=metadata,
                    )
            except Exception as e:
                logger.warning(f"Blend video codec {codec} failed: {e}")

        raise RuntimeError("All video codecs failed")

    def _encode_with_ffmpeg_pipe(
        self,
        before: np.ndarray,
        after: np.ndarray,
        total_frames: int,
        pattern: str,
        ffmpeg_bin: str = "ffmpeg",
    ) -> "VideoResult | None":
        """Pipe raw BGR frames to ffmpeg for direct H.264 encoding."""
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as tmp:
            out_path = tmp.name

        try:
            cmd = [
                ffmpeg_bin, "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{BLEND_WIDTH}x{BLEND_HEIGHT}",
                "-r", str(BLEND_FPS),
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                out_path,
            ]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            for i in range(total_frames):
                frame = self._render_frame(
                    i, total_frames, before, after, pattern
                )
                try:
                    proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    break
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

            proc.wait(timeout=60)
            stderr = proc.stderr.read()
            if proc.returncode != 0:
                logger.warning(
                    f"ffmpeg pipe failed (rc={proc.returncode}): "
                    f"{stderr.decode('utf-8', errors='replace')[:200]}"
                )
                os.unlink(out_path)
                return None

            with open(out_path, "rb") as fh:
                data = fh.read()
            os.unlink(out_path)

            if len(data) > 1024:
                duration = total_frames / BLEND_FPS
                logger.info(
                    f"Blend video encoded via ffmpeg pipe (H.264): "
                    f"{len(data)} bytes, {duration:.1f}s"
                )
                return VideoResult(data, "video/mp4", ".mp4")
            return None
        except Exception:
            try:
                os.unlink(out_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _ffmpeg_reencode(
        input_path: str, ffmpeg_bin: str = "ffmpeg"
    ) -> "bytes | None":
        """Re-encode an existing video file to H.264 with ffmpeg."""
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as tmp:
            out_path = tmp.name

        try:
            cmd = [
                ffmpeg_bin, "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                out_path,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.warning(
                    f"ffmpeg re-encode failed: "
                    f"{result.stderr.decode('utf-8', errors='replace')[:200]}"
                )
                os.unlink(out_path)
                return None

            with open(out_path, "rb") as fh:
                data = fh.read()
            os.unlink(out_path)
            return data
        except Exception:
            try:
                os.unlink(out_path)
            except OSError:
                pass
            raise


def get_blend_video_generator() -> BlendVideoGenerator:
    """Get BlendVideoGenerator instance."""
    return BlendVideoGenerator()
