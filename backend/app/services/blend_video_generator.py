"""Blend reveal video generator.

Creates a cinematic vertical video (9:16, 720x1280, 24fps, ~3.0s):
1. Before face – 1.0s
   → Wipe transition (0.4s)
2. After face – 1.0s
   → Cross-dissolve (0.3s)
3. Cao logo – 0.3s

No captions burned in – users add their own text via TikTok/CapCut.
All images are cover-fitted to fill the entire frame (no borders).
Uses 720x1280 @ 24fps to fit within Heroku's 512MB / 30s limits.
"""

import logging
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.services.video_generator import (
    VideoResult,
    _ease_in_out,
    get_ffmpeg_path,
)

# Logo image path
_LOGO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "images", "cao-logo.jpg"
)

logger = logging.getLogger(__name__)

# ── Blend-specific video settings ────────────
# Smaller than morph video (1080x1920) to stay within Heroku limits.
BLEND_WIDTH = 720
BLEND_HEIGHT = 1280
BLEND_FPS = 24

# Only codecs that actually work on Heroku's opencv-python-headless.
# avc1/VP80 fail on Heroku (no h264 hw encoder, no libvpx).
BLEND_CODEC_CHAIN: List[Tuple[str, str, str]] = [
    ("mp4v", ".mp4", "video/mp4"),
    ("MJPG", ".avi", "video/x-msvideo"),
]

# ── Timeline (seconds) ──────────────────────
PHASE_BEFORE = 1.0        # Show before face
PHASE_WIPE = 0.4          # Wipe transition
PHASE_AFTER = 1.0         # Show after face
PHASE_TRANS = 0.3         # Cross-dissolve transition
PHASE_BRAND = 0.3         # Logo

TOTAL_DURATION = (
    PHASE_BEFORE
    + PHASE_WIPE        # before → after (wipe)
    + PHASE_AFTER
    + PHASE_TRANS       # after → brand (dissolve)
    + PHASE_BRAND
)


class BlendVideoGenerator:
    """Generate cinematic blend-reveal videos."""

    # ── public API ──────────────────────────

    def generate(
        self,
        current_image: bytes,
        ideal_image: Optional[bytes],
        result_image: bytes,
    ) -> VideoResult:
        """Generate blend-reveal video (streaming, memory-safe).

        ideal_image is accepted for backward compatibility but ignored.
        """
        before = self._fit(self._decode(current_image))
        after = self._fit(self._decode(result_image))
        return self._generate_and_encode(before, after)

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
        # Guarantee exact output size
        if cropped.shape[:2] != (BLEND_HEIGHT, BLEND_WIDTH):
            cropped = cv2.resize(cropped, (BLEND_WIDTH, BLEND_HEIGHT))
        return np.ascontiguousarray(cropped)

    # ── effects ─────────────────────────────

    @staticmethod
    def _ken_burns(
        img: np.ndarray, progress: float, zoom: float = 0.04
    ) -> np.ndarray:
        """Subtle slow zoom for cinematic feel."""
        h, w = img.shape[:2]
        z = 1.0 + zoom * progress
        nw, nh = int(w * z), int(h * z)
        zoomed = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x0 = (nw - w) // 2
        y0 = (nh - h) // 2
        return zoomed[y0 : y0 + h, x0 : x0 + w]

    # ── transitions ─────────────────────────

    @staticmethod
    def _cross_dissolve(
        img_from: np.ndarray,
        img_to: np.ndarray,
        progress: float,
    ) -> np.ndarray:
        """Cross-dissolve (alpha blend) transition between two frames."""
        alpha = float(_ease_in_out(progress))
        # Use numpy blend to avoid cv2 shape-mismatch errors
        result = (
            img_to.astype(np.float32) * alpha
            + img_from.astype(np.float32) * (1.0 - alpha)
        )
        return result.clip(0, 255).astype(np.uint8)

    @staticmethod
    def _wipe_transition(
        img_from: np.ndarray,
        img_to: np.ndarray,
        progress: float,
    ) -> np.ndarray:
        """Left-to-right wipe with bright divider line and glow.

        A thick white vertical line sweeps across the frame,
        making the before/after boundary unmistakable.
        All operations vectorized with numpy (no per-pixel loops).
        """
        h, w = img_from.shape[:2]
        t = _ease_in_out(progress)
        boundary = max(0, min(w, int(w * t)))

        # Composite: after on left, before on right
        frame = img_from.copy()
        if boundary > 0:
            frame[:, :boundary] = img_to[:, :min(boundary, w)]

        # Bright glow zone (40px each side) + solid white line (6px)
        LINE_W = 6
        GLOW_W = 40

        if 0 < boundary < w:
            frame_f = frame.astype(np.float32)

            # Vectorized glow: build intensity array for the glow region
            glow_left = max(0, boundary - GLOW_W)
            glow_right = min(w, boundary + GLOW_W)
            xs = np.arange(glow_left, glow_right, dtype=np.float32)
            dist = np.abs(xs - boundary)
            intensity = (1.0 - dist / GLOW_W)
            intensity = intensity * intensity * 0.6  # quadratic falloff
            # Reshape to (1, num_cols, 1) for broadcasting with (H, num_cols, 3)
            glow_mask = intensity.reshape(1, -1, 1)
            frame_f[:, glow_left:glow_right] += 255.0 * glow_mask

            # Solid white divider line
            line_left = max(0, boundary - LINE_W // 2)
            line_right = min(w, boundary + LINE_W // 2)
            if line_right > line_left:
                frame_f[:, line_left:line_right] = 255.0

            return np.ascontiguousarray(
                frame_f.clip(0, 255).astype(np.uint8)
            )

        return np.ascontiguousarray(frame)

    def _load_logo(self) -> np.ndarray:
        """Load and cache the Cao logo image."""
        if not hasattr(self, "_logo_cache"):
            logo = cv2.imread(_LOGO_PATH, cv2.IMREAD_COLOR)
            if logo is None:
                logger.warning(f"Logo not found at {_LOGO_PATH}, using blank")
                self._logo_cache = np.zeros(
                    (BLEND_HEIGHT, BLEND_WIDTH, 3), dtype=np.uint8
                )
            else:
                self._logo_cache = self._fit(logo)
        return self._logo_cache

    def _render_brand_frame(self, progress: float) -> np.ndarray:
        """Render the brand/logo frame."""
        return self._load_logo().copy()

    def _render_frame(
        self,
        t: float,
        before: np.ndarray,
        after: np.ndarray,
    ) -> np.ndarray:
        """Render a single full-bleed frame at time *t*.

        Timeline:
        1. Before face hold (1.0s)
        2. Wipe: before → after (0.4s)
        3. After face hold (1.0s)
        4. Cross-dissolve: after → brand (0.3s)
        5. Brand/logo hold (0.3s)
        """
        # Build cumulative timeline boundaries
        e1 = PHASE_BEFORE                         # end of before hold
        e2 = e1 + PHASE_WIPE                      # end of wipe
        e3 = e2 + PHASE_AFTER                     # end of after hold
        e4 = e3 + PHASE_TRANS                     # end of after→brand dissolve

        if t < e1:
            p = t / PHASE_BEFORE
            frame = self._ken_burns(before, p)
        elif t < e2:
            p = (t - e1) / PHASE_WIPE
            frame = self._wipe_transition(before, after, p)
        elif t < e3:
            p = (t - e2) / PHASE_AFTER
            frame = self._ken_burns(after, p, zoom=0.03)
        elif t < e4:
            p = (t - e3) / PHASE_TRANS
            frame = self._cross_dissolve(
                after, self._render_brand_frame(1.0), p
            )
        else:
            frame = self._render_brand_frame(1.0)

        # Safety: guarantee exact frame size and type for encoder
        if frame.shape != (BLEND_HEIGHT, BLEND_WIDTH, 3):
            frame = cv2.resize(frame, (BLEND_WIDTH, BLEND_HEIGHT))
        if frame.dtype != np.uint8:
            frame = frame.clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    # ── encoding (streaming) ─────────────────

    def _generate_and_encode(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> VideoResult:
        """Generate frames and encode to browser-playable H.264 video.

        Strategy:
        1. Try ffmpeg pipe (raw BGR → H.264 mp4) — browser-native playback.
        2. Fallback: OpenCV mp4v then ffmpeg re-encode to H.264.
        3. Last resort: OpenCV mp4v as-is (may not play in all browsers).
        """
        total_frames = int(TOTAL_DURATION * BLEND_FPS)
        ffmpeg_bin = get_ffmpeg_path()

        # ── Strategy 1: Pipe raw frames directly to ffmpeg ────
        if ffmpeg_bin:
            try:
                result_video = self._encode_with_ffmpeg_pipe(
                    before, after, total_frames, ffmpeg_bin
                )
                if result_video:
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
                    t = i / BLEND_FPS
                    frame = self._render_frame(t, before, after)
                    writer.write(frame)
                writer.release()

                # Try to re-encode with ffmpeg for browser compatibility
                if ffmpeg_bin:
                    h264_data = self._ffmpeg_reencode(tmp_path, ffmpeg_bin)
                    os.unlink(tmp_path)
                    if h264_data and len(h264_data) > 1024:
                        logger.info(
                            f"Blend video re-encoded to H.264: "
                            f"{len(h264_data)} bytes, {TOTAL_DURATION:.1f}s"
                        )
                        return VideoResult(h264_data, "video/mp4", ".mp4")

                # Fallback: use mp4v as-is
                with open(tmp_path, "rb") as fh:
                    data = fh.read()
                os.unlink(tmp_path)

                if len(data) > 1024:
                    logger.info(
                        f"Blend video encoded ({codec}{ext}): "
                        f"{len(data)} bytes, {TOTAL_DURATION:.1f}s"
                    )
                    return VideoResult(data, content_type, ext)
            except Exception as e:
                logger.warning(f"Blend video codec {codec} failed: {e}")

        raise RuntimeError("All video codecs failed")

    def _encode_with_ffmpeg_pipe(
        self,
        before: np.ndarray,
        after: np.ndarray,
        total_frames: int,
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
                t = i / BLEND_FPS
                frame = self._render_frame(t, before, after)
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
                logger.info(
                    f"Blend video encoded via ffmpeg pipe (H.264): "
                    f"{len(data)} bytes, {TOTAL_DURATION:.1f}s"
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
