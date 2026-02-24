"""Blend reveal video generator — Gap-maximized photo slideshow for TikTok.

Timeline (~5.0s, loop-optimized):
  Before hold (2.0s) → Transition (0.3s) → After hold (2.2s)
  → Loop bridge crossfade back to Before (0.5s)

Features:
- Gap maximization via PIL ImageEnhance (Before darker/desaturated, After brighter/saturated)
- Quality gate: measures face diff, warns if too small
- Transition styles: blur (default), crossfade, snap
- Cao watermark (60px, 30% opacity, bottom-right)
- Seamless loop bridge (80% blend back to Before at final frame)
- High quality encoding (CRF 18, ~800-1200kbps)
- NO text labels, NO morph, NO flash — clean frames as TikTok raw material

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

# Try PIL for image enhancement
try:
    from PIL import Image, ImageEnhance
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────
_LOGO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "images", "cao-logo.jpg"
)

# ── Configuration (all parameters in one place) ─
CONFIG = {
    # Timing
    "before_hold_sec": 2.0,
    "transition_sec": 0.3,
    "after_hold_sec": 2.2,
    "loop_bridge_sec": 0.5,

    # Transition settings
    "transition_style": "blur",  # "blur", "crossfade", "snap"
    "blur_kernel_max": 51,       # Max Gaussian blur kernel size at peak

    # Enhancement (gap maximization)
    "enhance_before": {
        "brightness": 0.90,
        "color": 0.85,
        "contrast": 0.95,
    },
    "enhance_after": {
        "brightness": 1.08,
        "color": 1.10,
        "contrast": 1.10,
    },

    # Quality gate thresholds
    "quality_gate": {
        "sufficient": 15,   # face_diff >= 15 → OK
        "low": 5,           # face_diff >= 5  → low warning
        # face_diff < 5     → very_low warning
    },

    # Loop bridge settings
    "loop_bridge_blend_max": 0.8,  # Final frame: 80% Before, 20% After

    # Display settings
    "show_watermark": True,
    "watermark_opacity": 0.30,

    # Output settings
    "output_resolution": (720, 1280),
    "fps": 30,
    "crf": 18,
}

# ── Derived constants ─────────────────────────
BLEND_WIDTH, BLEND_HEIGHT = CONFIG["output_resolution"]
BLEND_FPS = CONFIG["fps"]
TOTAL_DURATION = (
    CONFIG["before_hold_sec"]
    + CONFIG["transition_sec"]
    + CONFIG["after_hold_sec"]
    + CONFIG["loop_bridge_sec"]
)  # 5.0s

# Only codecs that work on Heroku's opencv-python-headless.
BLEND_CODEC_CHAIN: List[Tuple[str, str, str]] = [
    ("mp4v", ".mp4", "video/mp4"),
    ("MJPG", ".avi", "video/x-msvideo"),
]

# ── Watermark ─────────────────────────────────
WATERMARK_SIZE = 60
WATERMARK_MARGIN = 24


class BlendVideoGenerator:
    """Generate gap-maximized photo slideshow videos for TikTok."""

    def __init__(self):
        self._watermark_cache = None

    # ── public API ──────────────────────────

    def generate(
        self,
        current_image: bytes,
        ideal_image: Optional[bytes],
        result_image: bytes,
        transition_style: str = "blur",
    ) -> VideoResult:
        """Generate gap-maximized blend-reveal video.

        Args:
            current_image: Before face image bytes.
            ideal_image: Ignored (backward compatibility).
            result_image: After face image bytes.
            transition_style: "blur" (default), "crossfade", or "snap".

        Returns:
            VideoResult with video bytes, duration, and metadata.
        """
        before_raw = self._fit(self._decode(current_image))
        after_raw = self._fit(self._decode(result_image))

        # Gap maximization: enhance before/after
        before = self._enhance_before(before_raw)
        after = self._enhance_after(after_raw)

        # Quality gate
        quality = self._quality_gate(before_raw, after_raw)

        # Load watermark
        if CONFIG["show_watermark"]:
            self._load_watermark()

        return self._generate_and_encode(before, after, transition_style, quality)

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

    # ── gap maximization (PIL ImageEnhance) ──

    @staticmethod
    def _enhance_pil(img_bgr: np.ndarray, params: dict) -> np.ndarray:
        """Apply brightness/color/contrast enhancement via PIL."""
        if not _HAS_PIL:
            return img_bgr

        # BGR → RGB → PIL
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Apply enhancements
        if "brightness" in params:
            pil_img = ImageEnhance.Brightness(pil_img).enhance(params["brightness"])
        if "color" in params:
            pil_img = ImageEnhance.Color(pil_img).enhance(params["color"])
        if "contrast" in params:
            pil_img = ImageEnhance.Contrast(pil_img).enhance(params["contrast"])

        # PIL → RGB → BGR
        enhanced = np.array(pil_img)
        return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    @classmethod
    def _enhance_before(cls, img: np.ndarray) -> np.ndarray:
        """Make Before image slightly darker and desaturated."""
        return cls._enhance_pil(img, CONFIG["enhance_before"])

    @classmethod
    def _enhance_after(cls, img: np.ndarray) -> np.ndarray:
        """Make After image slightly brighter and more saturated."""
        return cls._enhance_pil(img, CONFIG["enhance_after"])

    # ── quality gate ─────────────────────────

    @staticmethod
    def _quality_gate(before: np.ndarray, after: np.ndarray) -> dict:
        """Measure face diff and return quality verdict.

        Returns:
            dict with keys: face_diff, verdict, warning (optional)
        """
        # Mean absolute pixel difference
        diff = np.mean(np.abs(
            before.astype(np.float32) - after.astype(np.float32)
        ))

        thresholds = CONFIG["quality_gate"]
        if diff >= thresholds["sufficient"]:
            return {"face_diff": round(float(diff), 1), "verdict": "sufficient"}
        elif diff >= thresholds["low"]:
            return {
                "face_diff": round(float(diff), 1),
                "verdict": "low",
                "warning": "Before/After差分が小さいため、動画の効果が弱い可能性があります",
            }
        else:
            return {
                "face_diff": round(float(diff), 1),
                "verdict": "very_low",
                "warning": "Before/After差分が非常に小さいです。異なる画像を使用してください",
            }

    # ── cross dissolve ──────────────────────

    @staticmethod
    def _cross_dissolve(
        img_from: np.ndarray,
        img_to: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Linear cross-dissolve between two frames."""
        alpha = max(0.0, min(1.0, alpha))
        result = (
            img_to.astype(np.float32) * alpha
            + img_from.astype(np.float32) * (1.0 - alpha)
        )
        return result.clip(0, 255).astype(np.uint8)

    # ── blur transition ─────────────────────

    @staticmethod
    def _blur_transition(
        img_from: np.ndarray,
        img_to: np.ndarray,
        progress: float,
    ) -> np.ndarray:
        """Crossfade with Gaussian blur peaking at midpoint.

        progress: 0.0 (fully from) → 1.0 (fully to)
        Blur peaks at progress=0.5, zero at edges.
        """
        # Blur intensity: peaks at midpoint (triangle envelope)
        blur_amount = 1.0 - abs(2.0 * progress - 1.0)  # 0→1→0
        kernel_max = CONFIG["blur_kernel_max"]
        kernel_size = int(blur_amount * kernel_max)
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(1, kernel_size)

        # Crossfade
        alpha = max(0.0, min(1.0, progress))
        blended = (
            img_to.astype(np.float32) * alpha
            + img_from.astype(np.float32) * (1.0 - alpha)
        )
        blended = blended.clip(0, 255).astype(np.uint8)

        # Apply blur if significant
        if kernel_size >= 3:
            blended = cv2.GaussianBlur(blended, (kernel_size, kernel_size), 0)

        return blended

    # ── watermark ───────────────────────────

    def _load_watermark(self) -> np.ndarray:
        """Load and cache the Cao logo as a 60x60 watermark."""
        if self._watermark_cache is None:
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
        """Alpha-blend small logo at bottom-right."""
        wm = self._load_watermark()
        h, w = wm.shape[:2]
        wm_alpha = CONFIG["watermark_opacity"]

        x = BLEND_WIDTH - w - WATERMARK_MARGIN
        y = BLEND_HEIGHT - h - WATERMARK_MARGIN

        roi = frame[y : y + h, x : x + w].astype(np.float32)
        wm_f = wm.astype(np.float32)
        blended = roi * (1.0 - wm_alpha) + wm_f * wm_alpha
        frame[y : y + h, x : x + w] = blended.clip(0, 255).astype(np.uint8)
        return frame

    # ── frame rendering ─────────────────────

    def _render_frame(
        self,
        frame_index: int,
        total_frames: int,
        before: np.ndarray,
        after: np.ndarray,
        transition_style: str,
    ) -> np.ndarray:
        """Render a single frame of the photo slideshow timeline.

        Timeline (frame-based at 30fps, total 150 frames = 5.0s):
        - Frames 0-59 (2.0s): Before hold
        - Frames 60-68 (0.3s): Transition (blur/crossfade/snap)
        - Frames 69-134 (2.2s): After hold
        - Frames 135-149 (0.5s): Loop bridge crossfade After→Before (80%)
        """
        cfg = CONFIG
        fps = cfg["fps"]

        # Phase boundaries (in frames)
        f_before = int(cfg["before_hold_sec"] * fps)       # 60
        f_transition = int(cfg["transition_sec"] * fps)     # 9
        f_after = int(cfg["after_hold_sec"] * fps)          # 66
        f_bridge = total_frames - f_before - f_transition - f_after  # 15

        e1 = f_before                        # 60: transition starts
        e2 = e1 + f_transition              # 69: after hold starts
        e3 = e2 + f_after                   # 135: loop bridge starts

        # ── Base frame ──
        if frame_index < e1:
            # Before hold
            frame = before.copy()
        elif frame_index < e2:
            # Transition: Before → After
            t = (frame_index - e1) / max(f_transition - 1, 1)
            if transition_style == "snap":
                # Instant cut at midpoint
                frame = before.copy() if t < 0.5 else after.copy()
            elif transition_style == "crossfade":
                frame = self._cross_dissolve(before, after, t)
            else:
                # Default: blur transition
                frame = self._blur_transition(before, after, t)
        elif frame_index < e3:
            # After hold
            frame = after.copy()
        else:
            # Loop bridge: After → Before (up to 80%)
            t = (frame_index - e3) / max(f_bridge - 1, 1)
            blend = t * cfg["loop_bridge_blend_max"]
            frame = self._cross_dissolve(after, before, blend)

        # ── Watermark ──
        if cfg["show_watermark"]:
            frame = self._apply_watermark(frame)

        # Safety: guarantee exact frame size and type
        if frame.shape != (BLEND_HEIGHT, BLEND_WIDTH, 3):
            frame = cv2.resize(frame, (BLEND_WIDTH, BLEND_HEIGHT))
        if frame.dtype != np.uint8:
            frame = frame.clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    # ── encoding ────────────────────────────

    def _generate_and_encode(
        self,
        before: np.ndarray,
        after: np.ndarray,
        transition_style: str,
        quality: dict,
    ) -> VideoResult:
        """Generate frames and encode to browser-playable H.264 video.

        Strategy:
        1. Try ffmpeg pipe (raw BGR → H.264 mp4) — browser-native playback.
        2. Fallback: OpenCV mp4v then ffmpeg re-encode to H.264.
        3. Last resort: OpenCV mp4v as-is.
        """
        cfg = CONFIG
        duration = TOTAL_DURATION
        total_frames = int(duration * BLEND_FPS)
        crf = str(cfg["crf"])
        ffmpeg_bin = get_ffmpeg_path()

        # Metadata
        transition_start = cfg["before_hold_sec"]
        after_reveal = transition_start + cfg["transition_sec"]
        metadata: Dict = {
            "loop_friendly": True,
            "transition_start": transition_start,
            "after_reveal": after_reveal,
            "transition_style": transition_style,
            "quality_check": quality,
            "resolution": [BLEND_WIDTH, BLEND_HEIGHT],
            "fps": BLEND_FPS,
        }

        # ── Strategy 1: Pipe raw frames directly to ffmpeg ────
        if ffmpeg_bin:
            try:
                result_video = self._encode_with_ffmpeg_pipe(
                    before, after, total_frames, transition_style, crf, ffmpeg_bin
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
                        i, total_frames, before, after, transition_style
                    )
                    writer.write(frame)
                writer.release()

                # Try to re-encode with ffmpeg for browser compatibility
                if ffmpeg_bin:
                    h264_data = self._ffmpeg_reencode(tmp_path, crf, ffmpeg_bin)
                    os.unlink(tmp_path)
                    if h264_data and len(h264_data) > 1024:
                        logger.info(
                            f"Blend video re-encoded to H.264 (crf={crf}): "
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
        transition_style: str,
        crf: str,
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
                "-crf", crf,
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
                    i, total_frames, before, after, transition_style
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
                    f"Blend video encoded via ffmpeg pipe (H.264, crf={crf}): "
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
        input_path: str,
        crf: str = "18",
        ffmpeg_bin: str = "ffmpeg",
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
                "-crf", crf,
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
