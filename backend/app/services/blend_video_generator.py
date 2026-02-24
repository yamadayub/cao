"""Blend reveal video generator v8 — 4-phase timeline with detailed flash & bounce.

Timeline (~5.5s, no loop bridge — ends on After):
  Phase A: Before zoom-in (2.0s)
  Phase B: White flash with hold (0.5s)
  Phase C: After spring-bounce (0.5s)
  Phase D: After zoom-out (2.5s)

Features:
- Gap maximization via PIL ImageEnhance (Before darker/desaturated, After brighter/saturated)
- Quality gate: measures face diff, warns if too small
- Image motion: Before slow zoom-in, After spring-bounce + slow zoom-out
- Asymmetrical white flash with "hold" at peak (~0.1s pure white)
- Spring-bounce easing (damped sine wave: 1.06 → 0.99 → 1.01 → 1.00)
- Zoom-out border fix: pre-scale image to avoid black borders
- Cao watermark (60px, 30% opacity, bottom-right)
- NO loop bridge — video ends on After frame (TikTok jump cut on loop)
- High quality encoding (CRF 18, ~800-1200kbps)
- NO text labels, NO morph — clean frames as TikTok raw material

All images are cover-fitted to fill the entire frame (no borders).
Uses 720x1280 @ 30fps to fit within Heroku's 512MB / 30s limits.
"""

import logging
import math
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
    # Timing — 4-phase timeline
    "before_hold_sec": 2.0,       # Phase A: Before zoom-in
    "flash_sec": 0.5,             # Phase B: White flash with hold
    "bounce_sec": 0.5,            # Phase C: After spring-bounce
    "after_hold_sec": 2.5,        # Phase D: After zoom-out
    # Total: 5.5s

    # Transition settings
    "transition_style": "flash",       # "flash", "blur", "snap"
    "blur_kernel_max": 51,             # Max Gaussian blur kernel for blur transition

    # Motion settings — all zoom scales >= 1.0 to avoid black borders
    "motion_style": "zoom",                # "zoom" only for now
    "before_zoom_end_scale": 1.08,         # Before: zoom in to this scale (8%)
    "after_bounce_start_scale": 1.12,      # After: bounce pop-in zoom scale (12%)
    "after_bounce_settle_scale": 1.05,     # After: bounce zoom settles here
    "after_zoom_out_end_scale": 1.00,      # After: zoom out ends at natural (1.0)

    # Vertical bounce (Phase C) — dramatic up/down spring
    "bounce_amplitude_px": 80,             # Peak vertical displacement (pixels)

    # Enhancement (gap maximization) — dialed back to natural levels
    "enhance_enabled": True,
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
    "quality_gate_enabled": True,
    "quality_gate": {
        "sufficient": 15,   # face_diff >= 15 → OK
        "low": 5,           # face_diff >= 5  → low warning
        # face_diff < 5     → very_low warning
    },

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
    + CONFIG["flash_sec"]
    + CONFIG["bounce_sec"]
    + CONFIG["after_hold_sec"]
)  # 5.5s

# Only codecs that work on Heroku's opencv-python-headless.
BLEND_CODEC_CHAIN: List[Tuple[str, str, str]] = [
    ("mp4v", ".mp4", "video/mp4"),
    ("MJPG", ".avi", "video/x-msvideo"),
]

# ── Watermark ─────────────────────────────────
WATERMARK_SIZE = 60
WATERMARK_MARGIN = 24


class BlendVideoGenerator:
    """Generate gap-maximized TikTok slideshow videos with motion."""

    def __init__(self):
        self._watermark_cache = None

    # ── public API ──────────────────────────

    def generate(
        self,
        current_image: bytes,
        ideal_image: Optional[bytes],
        result_image: bytes,
        transition_style: str = "flash",
        motion_style: str = "zoom",
    ) -> VideoResult:
        """Generate gap-maximized blend-reveal video.

        Args:
            current_image: Before face image bytes.
            ideal_image: Ignored (backward compatibility).
            result_image: After face image bytes.
            transition_style: "flash" (default), "blur", or "snap".
            motion_style: "zoom" (default). Future: "slide", "ken_burns".

        Returns:
            VideoResult with video bytes, duration, and metadata.
        """
        before_raw = self._fit(self._decode(current_image))
        after_raw = self._fit(self._decode(result_image))

        # Gap maximization: enhance before/after
        if CONFIG["enhance_enabled"]:
            before = self._enhance_before(before_raw)
            after = self._enhance_after(after_raw)
        else:
            before = before_raw
            after = after_raw

        # Quality gate
        if CONFIG["quality_gate_enabled"]:
            quality = self._quality_gate(before_raw, after_raw)
        else:
            quality = {"face_diff": 0, "verdict": "skipped"}

        # Load watermark
        if CONFIG["show_watermark"]:
            self._load_watermark()

        return self._generate_and_encode(
            before, after, transition_style, motion_style, quality
        )

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

        if "brightness" in params:
            pil_img = ImageEnhance.Brightness(pil_img).enhance(params["brightness"])
        if "color" in params:
            pil_img = ImageEnhance.Color(pil_img).enhance(params["color"])
        if "contrast" in params:
            pil_img = ImageEnhance.Contrast(pil_img).enhance(params["contrast"])

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
        """Measure face diff and return quality verdict."""
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
                "warning": "変化が小さいため、SNSでのインパクトが弱い可能性があります",
            }
        else:
            return {
                "face_diff": round(float(diff), 1),
                "verdict": "very_low",
                "warning": "別の理想の顔を選び直すことをおすすめします",
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

    # ── zoom / crop helper ──────────────────

    @staticmethod
    def _apply_zoom(img: np.ndarray, scale: float) -> np.ndarray:
        """Zoom into the center of the image by the given scale factor.

        scale > 1.0: zoom in (crop center region, then upscale)
        scale < 1.0: zoom out — first upscale image by 1/scale, then
                     center-crop back to original size. This ensures no
                     black borders appear at edges.
        scale == 1.0: no change
        """
        if abs(scale - 1.0) < 0.001:
            return img

        h, w = img.shape[:2]

        if scale < 1.0:
            # Zoom-out: upscale image first so we have extra pixels to crop
            up_scale = 1.0 / scale  # e.g. 1/0.96 ≈ 1.042
            up_w = int(w * up_scale)
            up_h = int(h * up_scale)
            upscaled = cv2.resize(img, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
            # Center-crop back to original size
            x0 = (up_w - w) // 2
            y0 = (up_h - h) // 2
            return np.ascontiguousarray(upscaled[y0 : y0 + h, x0 : x0 + w])
        else:
            # Zoom-in: crop center region, then resize back
            new_h = int(h / scale)
            new_w = int(w / scale)
            new_h = max(1, min(h, new_h))
            new_w = max(1, min(w, new_w))
            y0 = (h - new_h) // 2
            x0 = (w - new_w) // 2
            cropped = img[y0 : y0 + new_h, x0 : x0 + new_w]
            return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def _apply_translate_y(img: np.ndarray, offset_y: float) -> np.ndarray:
        """Shift image vertically. Positive = down, negative = up.

        Uses BORDER_REPLICATE to fill exposed edges (no black borders).
        """
        if abs(offset_y) < 0.5:
            return img
        h, w = img.shape[:2]
        M = np.float32([[1, 0, 0], [0, 1, offset_y]])
        return cv2.warpAffine(
            img, M, (w, h),
            borderMode=cv2.BORDER_REPLICATE,
        )

    # ── easing functions ────────────────────

    @staticmethod
    def _ease_in(t: float) -> float:
        """Ease-in (quadratic): slow start → accelerate."""
        return t * t

    @staticmethod
    def _ease_out(t: float) -> float:
        """Ease-out (quadratic): fast start → decelerate."""
        return 1.0 - (1.0 - t) * (1.0 - t)

    @staticmethod
    def _bounce_spring(t: float) -> float:
        """Damped spring returning a factor that starts at 1.0 and settles at 0.0.

        Used for both zoom-settle and vertical bounce.
        freq=2.5π ensures cos(freq*1.0)≈0 → clean settle at t=1.0.
        """
        frequency = 2.5 * math.pi  # ≈ 7.854
        decay = 2.5
        return math.cos(frequency * t) * math.exp(-decay * t)

    # ── transition helpers ──────────────────

    @staticmethod
    def _flash_transition(
        img_from: np.ndarray,
        img_to: np.ndarray,
        progress: float,
    ) -> np.ndarray:
        """Asymmetrical white flash with hold at peak.

        3 sub-phases within progress [0.0, 1.0]:
          0.0–0.4: Before fades to white (overlay 0% → 100%)
          0.4–0.6: Pure white hold (溜め/anticipation)
          0.6–1.0: After fades in from white (overlay 100% → 0%)
        """
        if progress <= 0.4:
            # Before → white
            flash_t = progress / 0.4  # 0→1
            frame_f = img_from.astype(np.float32)
            frame_f += (255.0 - frame_f) * flash_t
            return frame_f.clip(0, 255).astype(np.uint8)
        elif progress <= 0.6:
            # White hold — pure white frames
            return np.full_like(img_from, 255)
        else:
            # White → After reveal
            reveal_t = (progress - 0.6) / 0.4  # 0→1
            frame_f = img_to.astype(np.float32)
            white_amount = 1.0 - reveal_t
            frame_f += (255.0 - frame_f) * white_amount
            return frame_f.clip(0, 255).astype(np.uint8)

    @staticmethod
    def _blur_transition(
        img_from: np.ndarray,
        img_to: np.ndarray,
        progress: float,
    ) -> np.ndarray:
        """Crossfade with Gaussian blur peaking at midpoint."""
        blur_amount = 1.0 - abs(2.0 * progress - 1.0)
        kernel_max = CONFIG["blur_kernel_max"]
        kernel_size = int(blur_amount * kernel_max)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(1, kernel_size)

        alpha = max(0.0, min(1.0, progress))
        blended = (
            img_to.astype(np.float32) * alpha
            + img_from.astype(np.float32) * (1.0 - alpha)
        )
        blended = blended.clip(0, 255).astype(np.uint8)

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
        motion_style: str,
    ) -> np.ndarray:
        """Render a single frame with 4-phase timeline.

        Timeline (at 30fps, total 165 frames = 5.5s):
          Phase A: Frames 0-59   (2.0s) Before with slow zoom-in
          Phase B: Frames 60-74  (0.5s) White flash with hold
          Phase C: Frames 75-89  (0.5s) After spring-bounce
          Phase D: Frames 90-164 (2.5s) After slow zoom-out
        """
        cfg = CONFIG
        fps = cfg["fps"]

        # Phase boundaries (in frames)
        f_before = int(cfg["before_hold_sec"] * fps)   # 60
        f_flash = int(cfg["flash_sec"] * fps)           # 15
        f_bounce = int(cfg["bounce_sec"] * fps)         # 15
        # f_after = remaining frames

        e_a = f_before                              # 60: flash starts
        e_b = e_a + f_flash                         # 75: bounce starts
        e_c = e_b + f_bounce                        # 90: zoom-out starts

        # ── Phase A: Before zoom-in ──
        if frame_index < e_a:
            t = frame_index / max(f_before - 1, 1)
            eased_t = self._ease_in(t)  # slow start → accelerate

            if motion_style == "zoom":
                scale = 1.0 + (cfg["before_zoom_end_scale"] - 1.0) * eased_t
                frame = self._apply_zoom(before, scale)
            else:
                frame = before.copy()

        # ── Phase B: White flash ──
        elif frame_index < e_b:
            t = (frame_index - e_a) / max(f_flash - 1, 1)

            # Apply zoom to source frames for continuity
            before_zoomed = self._apply_zoom(before, cfg["before_zoom_end_scale"]) \
                if motion_style == "zoom" else before
            after_bounced = self._apply_zoom(after, cfg["after_bounce_start_scale"]) \
                if motion_style == "zoom" else after

            if transition_style == "snap":
                frame = before_zoomed.copy() if t < 0.5 else after_bounced.copy()
            elif transition_style == "blur":
                frame = self._blur_transition(before_zoomed, after_bounced, t)
            else:
                # Default: flash transition with asymmetrical hold
                frame = self._flash_transition(before_zoomed, after_bounced, t)

        # ── Phase C: After spring-bounce (vertical + zoom settle) ──
        elif frame_index < e_c:
            bt = (frame_index - e_b) / max(f_bounce - 1, 1)

            if motion_style == "zoom":
                spring = self._bounce_spring(bt)

                # Zoom: settle from bounce_start → bounce_settle
                zoom_amp = cfg["after_bounce_start_scale"] - cfg["after_bounce_settle_scale"]
                scale = cfg["after_bounce_settle_scale"] + zoom_amp * spring
                frame = self._apply_zoom(after, max(scale, 1.001))

                # Vertical bounce: dramatic up/down spring
                offset_y = -cfg["bounce_amplitude_px"] * spring
                frame = self._apply_translate_y(frame, offset_y)
            else:
                frame = after.copy()

        # ── Phase D: After zoom-out (perceived pull-back: 1.05 → 1.0) ──
        else:
            f_after = total_frames - e_c
            zt = (frame_index - e_c) / max(f_after - 1, 1)
            eased_zt = self._ease_out(zt)

            if motion_style == "zoom":
                settle = cfg["after_bounce_settle_scale"]
                end = cfg["after_zoom_out_end_scale"]
                scale = settle + (end - settle) * eased_zt
                frame = self._apply_zoom(after, max(scale, 1.001))
            else:
                frame = after.copy()

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
        motion_style: str,
        quality: dict,
    ) -> VideoResult:
        """Generate frames and encode to browser-playable H.264 video."""
        cfg = CONFIG
        duration = TOTAL_DURATION
        total_frames = int(duration * BLEND_FPS)
        crf = str(cfg["crf"])
        ffmpeg_bin = get_ffmpeg_path()

        # Metadata for beat sync — 4-phase timeline markers
        flash_start = cfg["before_hold_sec"]
        flash_peak = flash_start + cfg["flash_sec"] * 0.4  # white hold begins
        after_reveal = flash_start + cfg["flash_sec"]
        bounce_end = after_reveal + cfg["bounce_sec"]
        metadata: Dict = {
            "loop_friendly": False,
            "flash_start": flash_start,
            "flash_peak": flash_peak,
            "after_reveal": after_reveal,
            "bounce_end": bounce_end,
            "transition_style": transition_style,
            "motion_style": motion_style,
            "quality_check": quality,
            "resolution": [BLEND_WIDTH, BLEND_HEIGHT],
            "fps": BLEND_FPS,
        }

        # ── Strategy 1: Pipe raw frames directly to ffmpeg ────
        if ffmpeg_bin:
            try:
                result_video = self._encode_with_ffmpeg_pipe(
                    before, after, total_frames,
                    transition_style, motion_style, crf, ffmpeg_bin,
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
                        i, total_frames, before, after,
                        transition_style, motion_style,
                    )
                    writer.write(frame)
                writer.release()

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
        motion_style: str,
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
                    i, total_frames, before, after,
                    transition_style, motion_style,
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
