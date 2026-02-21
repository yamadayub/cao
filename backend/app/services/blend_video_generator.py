"""Blend reveal video generator.

Creates a cinematic vertical video (9:16, 720x1280, 24fps, ~6s):
1. Current face – full bleed, slow Ken Burns
2. Brief flash transition
3. Ideal face – full bleed, slow Ken Burns
4. Horizontal gradient wipe → simulation result
5. Result hold
6. Minimal brand overlay

All images are cover-fitted to fill the entire frame (no borders).
Uses 720x1280 @ 24fps to fit within Heroku's 512MB / 30s limits.
"""

import logging
import math
import os
import shutil
import subprocess
import tempfile
from typing import List, Tuple

import cv2
import numpy as np

from app.services.video_generator import (
    VideoResult,
    _ease_in_out,
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
PHASE_CURRENT = 1.0
PHASE_FLASH = 0.3
PHASE_IDEAL = 0.8
PHASE_WIPE = 1.2
PHASE_RESULT = 1.7
PHASE_BRAND = 1.0

TOTAL_DURATION = (
    PHASE_CURRENT
    + PHASE_FLASH
    + PHASE_IDEAL
    + PHASE_WIPE
    + PHASE_RESULT
    + PHASE_BRAND
)


class BlendVideoGenerator:
    """Generate cinematic blend-reveal videos."""

    # ── public API ──────────────────────────

    def generate(
        self,
        current_image: bytes,
        ideal_image: bytes,
        result_image: bytes,
    ) -> VideoResult:
        """Generate blend-reveal video (streaming, memory-safe)."""
        current = self._fit(self._decode(current_image))
        ideal = self._fit(self._decode(ideal_image))
        result = self._fit(self._decode(result_image))
        return self._generate_and_encode(current, ideal, result)

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
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        x0 = (nw - BLEND_WIDTH) // 2
        y0 = (nh - BLEND_HEIGHT) // 2
        return resized[y0 : y0 + BLEND_HEIGHT, x0 : x0 + BLEND_WIDTH]

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

    @staticmethod
    def _gradient_wipe(
        img_from: np.ndarray,
        img_to: np.ndarray,
        progress: float,
        feather: int = 200,
    ) -> np.ndarray:
        """Horizontal left-to-right wipe with soft feathered edge and warm glow."""
        h, w = img_from.shape[:2]
        cols = np.arange(w, dtype=np.float32)

        sweep_pos = -feather + progress * (w + 2 * feather)
        mask_1d = np.clip((sweep_pos - cols) / feather, 0.0, 1.0)

        # Broadcast to full frame
        mask_3d = mask_1d[np.newaxis, :, np.newaxis]

        blended = (
            img_from.astype(np.float32) * (1 - mask_3d)
            + img_to.astype(np.float32) * mask_3d
        )

        # Warm glow at the leading edge
        edge_sigma = feather * 0.35
        glow_1d = np.exp(-((cols - sweep_pos) ** 2) / (2 * edge_sigma**2))
        glow_3d = glow_1d[np.newaxis, :, np.newaxis]
        glow_strength = 50.0

        # BGR: add warm tint (R > G >> B)
        blended[:, :, 2:3] = np.clip(
            blended[:, :, 2:3] + glow_3d * glow_strength, 0, 255
        )
        blended[:, :, 1:2] = np.clip(
            blended[:, :, 1:2] + glow_3d * glow_strength * 0.35, 0, 255
        )

        return blended.astype(np.uint8)

    # ── frame rendering ─────────────────────

    def _render_frame(
        self,
        t: float,
        current: np.ndarray,
        ideal: np.ndarray,
        result: np.ndarray,
    ) -> np.ndarray:
        """Render a single full-bleed frame at time *t*."""
        t1 = PHASE_CURRENT
        t2 = t1 + PHASE_FLASH
        t3 = t2 + PHASE_IDEAL
        t4 = t3 + PHASE_WIPE
        t5 = t4 + PHASE_RESULT

        if t < t1:
            # ── Current face with Ken Burns ────────
            p = t / t1
            return self._ken_burns(current, p)

        elif t < t2:
            # ── Flash transition current → ideal ───
            p = (t - t1) / PHASE_FLASH
            brightness = math.sin(p * math.pi)
            img = current if p < 0.5 else ideal
            frame = img.copy()
            white = np.full_like(frame, 255)
            cv2.addWeighted(
                frame, 1 - brightness * 0.85, white, brightness * 0.85, 0, frame
            )
            return frame

        elif t < t3:
            # ── Ideal face with Ken Burns ──────────
            p = (t - t2) / PHASE_IDEAL
            return self._ken_burns(ideal, p)

        elif t < t4:
            # ── Gradient wipe ideal → result ───────
            p = (t - t3) / PHASE_WIPE
            eased = _ease_in_out(p)
            return self._gradient_wipe(ideal, result, eased)

        elif t < t5:
            # ── Result hold with Ken Burns ─────────
            p = (t - t4) / PHASE_RESULT
            return self._ken_burns(result, p, zoom=0.02)

        else:
            # ── Brand overlay ──────────────────────
            p = (t - t5) / PHASE_BRAND
            frame = result.copy()

            # Bottom gradient darken for text readability
            grad_h = BLEND_HEIGHT // 3
            fade = min(p * 3.0, 1.0)
            if fade > 0.01:
                alpha_col = np.linspace(0, 0.55 * fade, grad_h, dtype=np.float32)
                alpha_3d = alpha_col[:, np.newaxis, np.newaxis]
                y_start = BLEND_HEIGHT - grad_h
                region = frame[y_start:, :, :].astype(np.float32)
                frame[y_start:, :, :] = (region * (1 - alpha_3d)).astype(np.uint8)

            # Text fade-in
            text_alpha = min(p * 2.5, 1.0)
            if text_alpha > 0.05:
                self._overlay_text(
                    frame,
                    "Cao",
                    BLEND_WIDTH // 2,
                    BLEND_HEIGHT - 110,
                    scale=1.4,
                    color=(255, 255, 255),
                    thickness=2,
                    alpha=text_alpha,
                )
                self._overlay_text(
                    frame,
                    "cao.style-elements.jp",
                    BLEND_WIDTH // 2,
                    BLEND_HEIGHT - 55,
                    scale=0.55,
                    color=(200, 200, 200),
                    thickness=1,
                    alpha=text_alpha,
                )
            return frame

    @staticmethod
    def _overlay_text(
        frame: np.ndarray,
        text: str,
        cx: int,
        y: int,
        scale: float,
        color: tuple,
        thickness: int,
        alpha: float = 1.0,
    ) -> None:
        """Draw centered text with optional alpha blending."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, _th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = cx - tw // 2
        if alpha < 0.99:
            overlay = frame.copy()
            cv2.putText(
                overlay, text, (x, y), font, scale, color, thickness, cv2.LINE_AA
            )
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            cv2.putText(
                frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA
            )

    # ── encoding (streaming) ─────────────────

    @staticmethod
    def _has_ffmpeg() -> bool:
        """Check if ffmpeg is available on the system."""
        return shutil.which("ffmpeg") is not None

    def _generate_and_encode(
        self,
        current: np.ndarray,
        ideal: np.ndarray,
        result: np.ndarray,
    ) -> VideoResult:
        """Generate frames and encode to browser-playable H.264 video.

        Strategy:
        1. Try ffmpeg pipe (raw BGR → H.264 mp4) — browser-native playback.
        2. Fallback: OpenCV mp4v then ffmpeg re-encode to H.264.
        3. Last resort: OpenCV mp4v as-is (may not play in all browsers).
        """
        total_frames = int(TOTAL_DURATION * BLEND_FPS)
        has_ffmpeg = self._has_ffmpeg()

        # ── Strategy 1: Pipe raw frames directly to ffmpeg ────
        if has_ffmpeg:
            try:
                result_video = self._encode_with_ffmpeg_pipe(
                    current, ideal, result, total_frames
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
                    frame = self._render_frame(t, current, ideal, result)
                    writer.write(frame)
                writer.release()

                # Try to re-encode with ffmpeg for browser compatibility
                if has_ffmpeg:
                    h264_data = self._ffmpeg_reencode(tmp_path)
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
        current: np.ndarray,
        ideal: np.ndarray,
        result: np.ndarray,
        total_frames: int,
    ) -> "VideoResult | None":
        """Pipe raw BGR frames to ffmpeg for direct H.264 encoding."""
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as tmp:
            out_path = tmp.name

        try:
            cmd = [
                "ffmpeg", "-y",
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
                frame = self._render_frame(t, current, ideal, result)
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
    def _ffmpeg_reencode(input_path: str) -> "bytes | None":
        """Re-encode an existing video file to H.264 with ffmpeg."""
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as tmp:
            out_path = tmp.name

        try:
            cmd = [
                "ffmpeg", "-y",
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
