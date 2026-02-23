"""Blend reveal video generator.

Creates a cinematic vertical video (9:16, 720x1280, 24fps, ~7.1s):
1. Ideal face ("こんな顔になれたら？") – 1.0s
   → Cross-dissolve (0.3s)
2. Current face ("ビフォー") – 1.0s
   → Flash reveal (0.5s)
3. Result face ("アフター") – 3.0s
   → Cross-dissolve (0.3s)
4. Cao logo + tagline – 1.0s

Captions are placed in the upper area of the frame to avoid TikTok UI overlap.
All images are cover-fitted to fill the entire frame (no borders).
Uses 720x1280 @ 24fps to fit within Heroku's 512MB / 30s limits.
"""

import logging
import os
import subprocess
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.services.video_generator import (
    VideoResult,
    _ease_in_out,
    get_ffmpeg_path,
)

# Font path for Japanese captions (try Bold first, then old subset as fallback)
_FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "fonts")
_FONT_PATH = os.path.join(_FONT_DIR, "NotoSansJP-Bold.ttf")
_FONT_PATH_FALLBACK = os.path.join(_FONT_DIR, "NotoSansJP-subset.ttf")

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
PHASE_IDEAL = 1.0         # Show ideal face
PHASE_TRANS = 0.3         # Cross-dissolve transition
PHASE_REVEAL = 0.5        # Flash reveal (longer for drama)
PHASE_CURRENT = 1.0       # Show current face
PHASE_RESULT = 3.0        # Show result face
PHASE_BRAND = 1.0         # Logo + tagline

TOTAL_DURATION = (
    PHASE_IDEAL
    + PHASE_TRANS       # ideal → current (dissolve)
    + PHASE_CURRENT
    + PHASE_REVEAL      # current → result (flash reveal)
    + PHASE_RESULT
    + PHASE_TRANS       # result → brand (dissolve)
    + PHASE_BRAND
)

# ── Captions ─────────────────────────────────
CAPTION_IDEAL = "こんな顔になれたら？"
CAPTION_CURRENT = "ビフォー"
CAPTION_RESULT = "アフター"
CAPTION_BRAND = "Caoでなりたい顔をシミュレーション"


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

    # ── frame rendering ─────────────────────

    def _add_caption(
        self, frame: np.ndarray, text: str, alpha: float = 1.0
    ) -> np.ndarray:
        """Add a Japanese caption with semi-transparent background at top.

        Placed at the top of the frame to avoid TikTok caption/UI overlap.
        """
        frame = frame.copy()

        # Semi-transparent dark gradient at top for text readability
        grad_h = 280
        alpha_col = np.linspace(0.75, 0, grad_h, dtype=np.float32)
        alpha_3d = alpha_col[:, np.newaxis, np.newaxis]
        region = frame[:grad_h, :, :].astype(np.float32)
        frame[:grad_h, :, :] = (region * (1 - alpha_3d * alpha)).astype(
            np.uint8
        )

        # Caption text using PIL for Japanese support (large, bold, with outline)
        if alpha > 0.05:
            self._draw_pil_text(
                frame,
                text,
                BLEND_WIDTH // 2,
                160,
                font_size=64,
                color=(255, 255, 255),
                alpha=alpha,
                outline=True,
            )
        return frame

    def _draw_pil_text(
        self,
        frame: np.ndarray,
        text: str,
        cx: int,
        cy: int,
        font_size: int = 40,
        color: tuple = (255, 255, 255),
        alpha: float = 1.0,
        outline: bool = False,
        max_width: int = 0,
    ) -> None:
        """Draw centered text on OpenCV frame using PIL (supports Japanese).

        When outline=True, draws a dark stroke around the text for visibility.
        When max_width>0, wraps text to fit within that pixel width.
        """
        font = None
        for path in (_FONT_PATH, _FONT_PATH_FALLBACK):
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except (OSError, IOError):
                continue
        if font is None:
            self._overlay_text(
                frame, text, cx, cy,
                scale=1.0, color=color, thickness=2, alpha=alpha,
            )
            return

        # Create transparent overlay with PIL
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Wrap text if max_width is set and text overflows
        lines = [text]
        if max_width > 0:
            bbox = draw.textbbox((0, 0), text, font=font)
            if bbox[2] - bbox[0] > max_width:
                lines = self._wrap_text(draw, text, font, max_width)

        a = int(255 * alpha)
        stroke_w = max(3, font_size // 14) if outline else 0

        # Measure total block height
        line_heights = []
        line_widths = []
        for line in lines:
            bb = draw.textbbox((0, 0), line, font=font)
            line_widths.append(bb[2] - bb[0])
            line_heights.append(bb[3] - bb[1])
        line_spacing = int(font_size * 0.3)
        total_h = sum(line_heights) + line_spacing * (len(lines) - 1)

        # Draw each line centered
        cur_y = cy - total_h // 2
        for i, line in enumerate(lines):
            lw = line_widths[i]
            x = cx - lw // 2
            if outline:
                draw.text(
                    (x, cur_y), line, font=font,
                    fill=(*color, a),
                    stroke_width=stroke_w,
                    stroke_fill=(0, 0, 0, a),
                )
            else:
                draw.text((x, cur_y), line, font=font, fill=(*color, a))
            cur_y += line_heights[i] + line_spacing

        # Composite onto frame
        pil_img = pil_img.convert("RGBA")
        pil_img = Image.alpha_composite(pil_img, overlay)
        result = cv2.cvtColor(
            np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR
        )
        frame[:] = result

    @staticmethod
    def _wrap_text(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
    ) -> list:
        """Wrap text into multiple lines to fit within max_width pixels."""
        lines = []
        current = ""
        for ch in text:
            test = current + ch
            bb = draw.textbbox((0, 0), test, font=font)
            if bb[2] - bb[0] > max_width and current:
                lines.append(current)
                current = ch
            else:
                current = test
        if current:
            lines.append(current)
        return lines

    @staticmethod
    def _cross_dissolve(
        img_from: np.ndarray,
        img_to: np.ndarray,
        progress: float,
    ) -> np.ndarray:
        """Cross-dissolve (alpha blend) transition between two frames."""
        alpha = _ease_in_out(progress)
        return cv2.addWeighted(
            img_to, alpha, img_from, 1.0 - alpha, 0
        )

    @staticmethod
    def _flash_reveal(
        img_from: np.ndarray,
        img_to: np.ndarray,
        progress: float,
    ) -> np.ndarray:
        """Dramatic flash-burst reveal transition.

        Timeline within the transition:
        0.0–0.35: img_from brightens rapidly toward white
        0.35–0.50: peak white flash
        0.50–1.0: white fades out revealing img_to with zoom punch
        """
        h, w = img_from.shape[:2]

        if progress < 0.35:
            # Brighten img_from toward white
            t = progress / 0.35  # 0→1
            t2 = t * t  # accelerate
            frame = img_from.astype(np.float32)
            white = np.full_like(frame, 255.0)
            frame = frame * (1.0 - t2) + white * t2
            return frame.clip(0, 255).astype(np.uint8)

        elif progress < 0.50:
            # Peak white flash
            return np.full((h, w, 3), 255, dtype=np.uint8)

        else:
            # Fade from white revealing img_to with zoom punch
            t = (progress - 0.50) / 0.50  # 0→1
            t_ease = t * t * (3.0 - 2.0 * t)  # smoothstep

            # Zoom punch: start slightly zoomed in, settle to normal
            zoom = 1.0 + 0.08 * (1.0 - t_ease)
            nw, nh = int(w * zoom), int(h * zoom)
            zoomed = cv2.resize(img_to, (nw, nh), interpolation=cv2.INTER_LINEAR)
            x0 = (nw - w) // 2
            y0 = (nh - h) // 2
            revealed = zoomed[y0 : y0 + h, x0 : x0 + w]

            # Blend from white to revealed
            white_alpha = 1.0 - t_ease
            frame = revealed.astype(np.float32)
            white = np.full_like(frame, 255.0)
            frame = frame * (1.0 - white_alpha) + white * white_alpha
            return frame.clip(0, 255).astype(np.uint8)

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
        """Render the brand/logo frame with tagline."""
        frame = self._load_logo().copy()

        # Darken bottom area for tagline readability
        fade = min(progress * 3.0, 1.0)
        if fade > 0.01:
            grad_h = BLEND_HEIGHT // 3
            y_start = BLEND_HEIGHT - grad_h
            alpha_col = np.linspace(0, 0.6 * fade, grad_h, dtype=np.float32)
            alpha_3d = alpha_col[:, np.newaxis, np.newaxis]
            region = frame[y_start:, :, :].astype(np.float32)
            frame[y_start:, :, :] = (region * (1 - alpha_3d)).astype(np.uint8)

        # Tagline below logo center (large, with outline, wrapped)
        text_alpha = min(progress * 3.0, 1.0)
        if text_alpha > 0.05:
            self._draw_pil_text(
                frame,
                CAPTION_BRAND,
                BLEND_WIDTH // 2,
                BLEND_HEIGHT * 3 // 4,
                font_size=48,
                color=(255, 255, 255),
                alpha=text_alpha,
                outline=True,
                max_width=BLEND_WIDTH - 80,
            )
        return frame

    def _render_frame(
        self,
        t: float,
        current: np.ndarray,
        ideal: np.ndarray,
        result: np.ndarray,
    ) -> np.ndarray:
        """Render a single full-bleed frame at time *t*.

        Timeline (with transitions):
        1. Ideal face hold (1.0s)
        2. Cross-dissolve: ideal → current (0.3s)
        3. Current face hold (1.0s)
        4. Flash reveal: current → result (0.5s)
        5. Result face hold (3.0s)
        6. Cross-dissolve: result → brand (0.3s)
        7. Brand/logo hold (1.0s)
        """
        # Build cumulative timeline boundaries
        e1 = PHASE_IDEAL                          # end of ideal hold
        e2 = e1 + PHASE_TRANS                     # end of ideal→current dissolve
        e3 = e2 + PHASE_CURRENT                   # end of current hold
        e4 = e3 + PHASE_REVEAL                    # end of current→result flash
        e5 = e4 + PHASE_RESULT                    # end of result hold
        e6 = e5 + PHASE_TRANS                     # end of result→brand dissolve

        if t < e1:
            # ── Ideal face hold ──────────────────────
            p = t / PHASE_IDEAL
            frame = self._ken_burns(ideal, p)
            return self._add_caption(frame, CAPTION_IDEAL)

        elif t < e2:
            # ── Cross-dissolve: ideal → current ──────
            p = (t - e1) / PHASE_TRANS
            img_from = self._add_caption(ideal.copy(), CAPTION_IDEAL)
            img_to = self._add_caption(current.copy(), CAPTION_CURRENT)
            return self._cross_dissolve(img_from, img_to, p)

        elif t < e3:
            # ── Current face hold ────────────────────
            p = (t - e2) / PHASE_CURRENT
            frame = self._ken_burns(current, p)
            return self._add_caption(frame, CAPTION_CURRENT)

        elif t < e4:
            # ── Flash reveal: current → result ───────
            p = (t - e3) / PHASE_REVEAL
            img_from = current.copy()
            img_to = self._add_caption(result.copy(), CAPTION_RESULT)
            return self._flash_reveal(img_from, img_to, p)

        elif t < e5:
            # ── Result face hold ─────────────────────
            p = (t - e4) / PHASE_RESULT
            frame = self._ken_burns(result, p, zoom=0.03)
            return self._add_caption(frame, CAPTION_RESULT)

        elif t < e6:
            # ── Cross-dissolve: result → brand ───────
            p = (t - e5) / PHASE_TRANS
            img_from = self._add_caption(result.copy(), CAPTION_RESULT)
            img_to = self._render_brand_frame(1.0)
            return self._cross_dissolve(img_from, img_to, p)

        else:
            # ── Brand/logo hold ──────────────────────
            return self._render_brand_frame(1.0)

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
        ffmpeg_bin = get_ffmpeg_path()

        # ── Strategy 1: Pipe raw frames directly to ffmpeg ────
        if ffmpeg_bin:
            try:
                result_video = self._encode_with_ffmpeg_pipe(
                    current, ideal, result, total_frames, ffmpeg_bin
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
        current: np.ndarray,
        ideal: np.ndarray,
        result: np.ndarray,
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
