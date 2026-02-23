"""Blend reveal video generator — Morph-centered, TikTok-optimized.

Timeline (~4.5s, loop-optimized):
  Before hold (0.5s) → Slow morph with ease-in-out (2.0s)
  → Flash accent at morph completion → After hold (1.5s)
  → Loop bridge crossfade back to Before (0.5s)

Features:
- Morph (cross-dissolve with cubic ease-in-out) as the main visual hook
- White flash at morph completion for beat sync
- "Before"/"After" labels with fade transitions (PIL + Noto Sans JP Bold)
- Cao watermark (64px, 35% opacity, bottom-right)
- Seamless loop bridge (80% blend back to Before at final frame)
- High quality encoding (CRF 18, ~800-1200kbps)

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

# Try PIL for label rendering
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────
_LOGO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "images", "cao-logo.jpg"
)
_FONT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "fonts", "NotoSansJP-subset.ttf"
)

# ── Configuration (all parameters in one place) ─
CONFIG = {
    # Timing
    "before_hold_sec": 0.5,
    "morph_sec": 2.0,
    "after_hold_sec": 1.5,
    "loop_bridge_sec": 0.5,

    # Morph settings
    "morph_easing": "ease_in_out",       # "linear" or "ease_in_out"
    "flash_enabled": True,
    "flash_opacity": 0.3,
    "flash_duration_sec": 0.1,

    # Loop bridge settings
    "loop_bridge_blend_max": 0.8,        # Final frame: 80% Before, 20% After

    # Display settings
    "show_labels": True,
    "show_watermark": True,
    "watermark_opacity": 0.35,

    # Output settings
    "output_resolution": (720, 1280),
    "fps": 30,
    "crf": 18,                           # High quality (18=high, 23=default)
}

# ── Derived constants ─────────────────────────
BLEND_WIDTH, BLEND_HEIGHT = CONFIG["output_resolution"]
BLEND_FPS = CONFIG["fps"]
TOTAL_DURATION = (
    CONFIG["before_hold_sec"]
    + CONFIG["morph_sec"]
    + CONFIG["after_hold_sec"]
    + CONFIG["loop_bridge_sec"]
)  # 4.5s

# Only codecs that work on Heroku's opencv-python-headless.
BLEND_CODEC_CHAIN: List[Tuple[str, str, str]] = [
    ("mp4v", ".mp4", "video/mp4"),
    ("MJPG", ".avi", "video/x-msvideo"),
]

# ── Label position ────────────────────────────
LABEL_X = 40                 # Left margin from frame edge
LABEL_Y_FROM_BOTTOM = 120   # From bottom of frame
LABEL_FONT_SIZE = 36
LABEL_PAD_X = 16
LABEL_PAD_Y = 10
LABEL_BG_ALPHA = 128         # 0-255 for PIL (≈0.5 opacity)

# ── Watermark ─────────────────────────────────
WATERMARK_SIZE = 64
WATERMARK_MARGIN = 24


class BlendVideoGenerator:
    """Generate morph-centered blend-reveal videos for TikTok."""

    def __init__(self):
        self._before_label = None   # Pre-rendered BGRA numpy array
        self._after_label = None
        self._watermark_cache = None
        self._font = None

    # ── public API ──────────────────────────

    def generate(
        self,
        current_image: bytes,
        ideal_image: Optional[bytes],
        result_image: bytes,
        pattern: str = "A",
    ) -> VideoResult:
        """Generate morph-centered blend-reveal video.

        Args:
            current_image: Before face image bytes.
            ideal_image: Ignored (backward compatibility).
            result_image: After face image bytes.
            pattern: Ignored (single morph pattern). Kept for backward compat.

        Returns:
            VideoResult with video bytes, duration, and metadata.
        """
        before = self._fit(self._decode(current_image))
        after = self._fit(self._decode(result_image))
        self._prepare_overlays()
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
        if cropped.shape[:2] != (BLEND_HEIGHT, BLEND_WIDTH):
            cropped = cv2.resize(cropped, (BLEND_WIDTH, BLEND_HEIGHT))
        return np.ascontiguousarray(cropped)

    # ── easing ──────────────────────────────

    @staticmethod
    def _ease_in_out_cubic(t: float) -> float:
        """Cubic ease-in-out: slow start → fast middle → slow end."""
        if t < 0.5:
            return 4.0 * t * t * t
        else:
            return 1.0 - pow(-2.0 * t + 2.0, 3) / 2.0

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

    # ── overlay preparation ─────────────────

    def _prepare_overlays(self):
        """Pre-render labels and load watermark (called once per generate)."""
        if CONFIG["show_labels"]:
            self._load_font()
            self._before_label = self._prerender_label("Before")
            self._after_label = self._prerender_label("After")
        if CONFIG["show_watermark"]:
            self._load_watermark()

    def _load_font(self):
        """Load Noto Sans JP Bold font for label rendering."""
        if not _HAS_PIL:
            self._font = None
            return

        # Try Noto Sans JP subset (includes Latin glyphs for Before/After)
        if os.path.exists(_FONT_PATH):
            try:
                self._font = ImageFont.truetype(_FONT_PATH, LABEL_FONT_SIZE)
                return
            except Exception:
                pass

        # Try system DejaVu Sans Bold as fallback
        for fallback in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        ]:
            if os.path.exists(fallback):
                try:
                    self._font = ImageFont.truetype(fallback, LABEL_FONT_SIZE)
                    return
                except Exception:
                    pass

        self._font = None  # Will use cv2 fallback

    def _prerender_label(self, text: str) -> np.ndarray:
        """Pre-render a label as a BGRA numpy array.

        Uses PIL for anti-aliased text when available, cv2 fallback otherwise.
        """
        if _HAS_PIL and self._font is not None:
            return self._prerender_label_pil(text)
        return self._prerender_label_cv2(text)

    def _prerender_label_pil(self, text: str) -> np.ndarray:
        """Pre-render label using PIL for anti-aliased text."""
        font = self._font

        # Measure text
        temp_img = Image.new("RGBA", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        w = tw + LABEL_PAD_X * 2
        h = th + LABEL_PAD_Y * 2

        # Semi-transparent black background with white text
        img = Image.new("RGBA", (w, h), (0, 0, 0, LABEL_BG_ALPHA))
        draw = ImageDraw.Draw(img)
        draw.text(
            (LABEL_PAD_X - bbox[0], LABEL_PAD_Y - bbox[1]),
            text,
            fill=(255, 255, 255, 255),
            font=font,
        )

        # Convert RGBA (PIL) → BGRA (OpenCV)
        rgba = np.array(img)
        bgra = np.empty_like(rgba)
        bgra[:, :, 0] = rgba[:, :, 2]  # B
        bgra[:, :, 1] = rgba[:, :, 1]  # G
        bgra[:, :, 2] = rgba[:, :, 0]  # R
        bgra[:, :, 3] = rgba[:, :, 3]  # A
        return bgra

    def _prerender_label_cv2(self, text: str) -> np.ndarray:
        """Fallback: pre-render label using cv2 Hershey font."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

        w = tw + LABEL_PAD_X * 2
        h = th + baseline + LABEL_PAD_Y * 2

        # Draw white text on a temp image to get the text mask
        text_img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            text_img, text, (LABEL_PAD_X, LABEL_PAD_Y + th),
            font, scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

        # Build BGRA: black bg with semi-transparent alpha + opaque text
        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        bgra[:, :, 3] = LABEL_BG_ALPHA  # Background alpha
        bgra[:, :, :3] = text_img
        text_mask = text_img[:, :, 0] > 0
        bgra[text_mask, 3] = 255  # Full alpha for text pixels

        return bgra

    # ── label compositing ───────────────────

    @staticmethod
    def _composite_label(
        frame: np.ndarray,
        label_bgra: np.ndarray,
        x: int,
        y: int,
        opacity: float,
    ) -> np.ndarray:
        """Alpha-composite a pre-rendered BGRA label onto a BGR frame."""
        if label_bgra is None or opacity <= 0.01:
            return frame

        lh, lw = label_bgra.shape[:2]

        # Clamp to frame bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(x + lw, BLEND_WIDTH)
        y2 = min(y + lh, BLEND_HEIGHT)

        if x2 <= x1 or y2 <= y1:
            return frame

        # Crop label region
        lx1 = x1 - x
        ly1 = y1 - y
        lx2 = lx1 + (x2 - x1)
        ly2 = ly1 + (y2 - y1)

        label_roi = label_bgra[ly1:ly2, lx1:lx2]
        frame_roi = frame[y1:y2, x1:x2].astype(np.float32)

        # Alpha channel scaled by opacity
        alpha = label_roi[:, :, 3:4].astype(np.float32) / 255.0 * opacity
        label_bgr = label_roi[:, :, :3].astype(np.float32)

        blended = frame_roi * (1.0 - alpha) + label_bgr * alpha
        frame[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)

        return frame

    # ── watermark ───────────────────────────

    def _load_watermark(self) -> np.ndarray:
        """Load and cache the Cao logo as a 64x64 watermark."""
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
    ) -> np.ndarray:
        """Render a single frame of the morph-centered timeline.

        Timeline (frame-based at 30fps, total 135 frames = 4.5s):
        - Frames 0-14 (0.5s): Before hold
        - Frames 15-74 (2.0s): Slow morph Before→After (ease-in-out cubic)
        - Frames 75-119 (1.5s): After hold
        - Frames 120-134 (0.5s): Loop bridge crossfade After→Before (80%)
        """
        cfg = CONFIG
        fps = cfg["fps"]

        # Phase boundaries (in frames)
        f_before = int(cfg["before_hold_sec"] * fps)   # 15
        f_morph = int(cfg["morph_sec"] * fps)           # 60
        f_after = int(cfg["after_hold_sec"] * fps)      # 45
        f_bridge = total_frames - f_before - f_morph - f_after  # 15

        e1 = f_before                    # 15: morph starts
        e2 = e1 + f_morph               # 75: after hold starts
        e3 = e2 + f_after               # 120: loop bridge starts

        # ── Base frame ──
        if frame_index < e1:
            # Before hold
            frame = before.copy()
        elif frame_index < e2:
            # Morph: Before → After with ease-in-out
            t = (frame_index - e1) / max(f_morph - 1, 1)
            if cfg["morph_easing"] == "ease_in_out":
                eased = self._ease_in_out_cubic(t)
            else:
                eased = t
            frame = self._cross_dissolve(before, after, eased)
        elif frame_index < e3:
            # After hold
            frame = after.copy()
        else:
            # Loop bridge: After → Before (up to 80%)
            t = (frame_index - e3) / max(f_bridge - 1, 1)
            blend = t * cfg["loop_bridge_blend_max"]
            frame = self._cross_dissolve(after, before, blend)

        # ── Flash effect at morph completion ──
        if cfg["flash_enabled"]:
            frame = self._apply_flash(frame, frame_index, e2, fps)

        # ── Labels ──
        if cfg["show_labels"]:
            frame = self._apply_labels(frame, frame_index, e1, e2, e3, total_frames, fps)

        # ── Watermark ──
        if cfg["show_watermark"]:
            frame = self._apply_watermark(frame)

        # Safety: guarantee exact frame size and type
        if frame.shape != (BLEND_HEIGHT, BLEND_WIDTH, 3):
            frame = cv2.resize(frame, (BLEND_WIDTH, BLEND_HEIGHT))
        if frame.dtype != np.uint8:
            frame = frame.clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    # ── flash effect ────────────────────────

    @staticmethod
    def _apply_flash(
        frame: np.ndarray,
        frame_index: int,
        morph_end_frame: int,
        fps: int,
    ) -> np.ndarray:
        """Apply white flash at morph completion.

        Flash peaks 1-2 frames before morph end (morph_end_frame),
        with 0.1s triangle envelope (fade in → peak → fade out).
        """
        cfg = CONFIG
        flash_frames = max(1, round(cfg["flash_duration_sec"] * fps))  # 3
        # Peak 2 frames before after-hold starts
        flash_center = morph_end_frame - 2
        flash_half = flash_frames / 2.0

        dist = abs(frame_index - flash_center)
        if dist > flash_half:
            return frame

        # Triangle envelope: peak at center, zero at edges
        flash_opacity = cfg["flash_opacity"] * (1.0 - dist / flash_half)
        if flash_opacity < 0.01:
            return frame

        # Blend toward white: frame + (white - frame) * opacity
        frame_f = frame.astype(np.float32)
        frame_f += (255.0 - frame_f) * flash_opacity
        return frame_f.clip(0, 255).astype(np.uint8)

    # ── label application ───────────────────

    def _apply_labels(
        self,
        frame: np.ndarray,
        fi: int,
        e1: int,
        e2: int,
        e3: int,
        total_frames: int,
        fps: int,
    ) -> np.ndarray:
        """Apply Before/After labels with fade transitions.

        Before label: visible 0.0s-0.8s, fade out 0.8s-1.1s (0.3s)
        After label: fade in at 2.5s (0.2s), visible until 4.0s, fade out 4.0s-4.3s (0.3s)
        """
        # Label Y: bottom-left, anchored from bottom
        if self._before_label is not None:
            label_h = self._before_label.shape[0]
        elif self._after_label is not None:
            label_h = self._after_label.shape[0]
        else:
            return frame
        label_y = BLEND_HEIGHT - LABEL_Y_FROM_BOTTOM - label_h

        # ── Before label ──
        before_visible_end = int(0.8 * fps)    # frame 24
        before_fade_end = int(1.1 * fps)       # frame 33

        if fi < before_visible_end:
            before_opacity = 1.0
        elif fi < before_fade_end:
            before_opacity = 1.0 - (fi - before_visible_end) / (before_fade_end - before_visible_end)
        else:
            before_opacity = 0.0

        if before_opacity > 0.01:
            frame = self._composite_label(
                frame, self._before_label, LABEL_X, label_y, before_opacity
            )

        # ── After label ──
        after_fade_in_start = e2                             # frame 75 (2.5s)
        after_fade_in_end = e2 + int(0.2 * fps)             # frame 81
        after_visible_end = e3                                # frame 120 (4.0s)
        after_fade_out_end = min(e3 + int(0.3 * fps), total_frames)  # frame 129

        if fi < after_fade_in_start:
            after_opacity = 0.0
        elif fi < after_fade_in_end:
            after_opacity = (fi - after_fade_in_start) / max(after_fade_in_end - after_fade_in_start, 1)
        elif fi < after_visible_end:
            after_opacity = 1.0
        elif fi < after_fade_out_end:
            after_opacity = 1.0 - (fi - after_visible_end) / max(after_fade_out_end - after_visible_end, 1)
        else:
            after_opacity = 0.0

        if after_opacity > 0.01:
            frame = self._composite_label(
                frame, self._after_label, LABEL_X, label_y, after_opacity
            )

        return frame

    # ── beat sync metadata ──────────────────

    @staticmethod
    def _get_beat_sync_points() -> List[Dict]:
        """Return beat sync points for BGM alignment."""
        cfg = CONFIG
        morph_complete_time = cfg["before_hold_sec"] + cfg["morph_sec"]
        return [
            {
                "time_sec": morph_complete_time,
                "type": "morph_complete",
                "description": "Morph complete (flash peak). Align BGM beat drop here.",
            }
        ]

    # ── encoding ────────────────────────────

    def _generate_and_encode(
        self,
        before: np.ndarray,
        after: np.ndarray,
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

        beat_sync = self._get_beat_sync_points()
        metadata: Dict = {
            "loop_friendly": True,
            "beat_sync_points": beat_sync,
            "resolution": [BLEND_WIDTH, BLEND_HEIGHT],
            "fps": BLEND_FPS,
        }

        # ── Strategy 1: Pipe raw frames directly to ffmpeg ────
        if ffmpeg_bin:
            try:
                result_video = self._encode_with_ffmpeg_pipe(
                    before, after, total_frames, crf, ffmpeg_bin
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
                    frame = self._render_frame(i, total_frames, before, after)
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
                frame = self._render_frame(i, total_frames, before, after)
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
