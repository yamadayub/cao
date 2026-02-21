"""Artistic blend reveal video generator.

Creates a shareable video showing:
1. Current face (1s) with Ken Burns effect
2. Ideal face (1s) with Ken Burns effect
3. Artistic blend transition (circular reveal + golden glow + sparkles)
4. Simulation result revealed
5. Branding

Output: 9:16 vertical (1080x1920), 30fps, ~6s
"""

import logging
import math
from typing import List, Tuple

import cv2
import numpy as np

from app.services.video_generator import (
    CODEC_CHAIN,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    FPS,
    FACE_IMAGE_SIZE,
    FACE_IMAGE_Y,
    VideoResult,
    _ease_in_out,
    _put_centered_text,
)

logger = logging.getLogger(__name__)

# ── Timeline (seconds) ──────────────────────
PHASE_HOLD_CURRENT = 1.0
PHASE_FLASH_1 = 0.15
PHASE_HOLD_IDEAL = 1.0
PHASE_FLASH_2 = 0.2
PHASE_BLEND = 1.2
PHASE_GLOW_SETTLE = 0.3
PHASE_HOLD_RESULT = 1.5
PHASE_BRANDING = 0.65

TOTAL_DURATION = (
    PHASE_HOLD_CURRENT
    + PHASE_FLASH_1
    + PHASE_HOLD_IDEAL
    + PHASE_FLASH_2
    + PHASE_BLEND
    + PHASE_GLOW_SETTLE
    + PHASE_HOLD_RESULT
    + PHASE_BRANDING
)

# ── Colors (BGR) ────────────────────────────
ACCENT_BGR = (59, 130, 246)  # warm orange-ish in BGR
GOLD_BGR = (0, 200, 255)
TEXT_GRAY = (100, 100, 100)
LIGHT_GRAY = (180, 180, 180)


class BlendVideoGenerator:
    """Generate artistic blend-reveal videos from 3 images."""

    def __init__(self) -> None:
        # Pre-generate sparkle animation data
        rng = np.random.RandomState(42)
        self._sparkles: List[dict] = []
        for _ in range(80):
            self._sparkles.append(
                {
                    "x": rng.randint(0, FACE_IMAGE_SIZE),
                    "y": rng.randint(0, FACE_IMAGE_SIZE),
                    "birth": rng.uniform(0.0, 0.7),
                    "life": rng.uniform(0.15, 0.45),
                    "brightness": rng.uniform(0.4, 1.0),
                    "radius": rng.randint(2, 6),
                }
            )

    # ── public API ──────────────────────────

    def generate(
        self,
        current_image: bytes,
        ideal_image: bytes,
        result_image: bytes,
    ) -> VideoResult:
        """Generate blend-reveal video.

        Args:
            current_image: Before face (JPEG/PNG bytes)
            ideal_image:   Ideal face  (JPEG/PNG bytes)
            result_image:  Simulation result (JPEG/PNG bytes)

        Returns:
            VideoResult with encoded video bytes + format metadata
        """
        current = self._fit(self._decode(current_image))
        ideal = self._fit(self._decode(ideal_image))
        result = self._fit(self._decode(result_image))

        frames = self._generate_all_frames(current, ideal, result)
        return self._encode(frames)

    # ── image helpers ───────────────────────

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img

    @staticmethod
    def _fit(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        s = FACE_IMAGE_SIZE
        scale = max(s / w, s / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        x0 = (nw - s) // 2
        y0 = (nh - s) // 2
        return resized[y0 : y0 + s, x0 : x0 + s]

    # ── effects ─────────────────────────────

    @staticmethod
    def _ken_burns(
        img: np.ndarray, progress: float, zoom: float = 0.05
    ) -> np.ndarray:
        """Slow zoom-in for dynamic feel on still images."""
        h, w = img.shape[:2]
        z = 1.0 + zoom * progress
        nw, nh = int(w * z), int(h * z)
        zoomed = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        x0 = (nw - w) // 2
        y0 = (nh - h) // 2
        return zoomed[y0 : y0 + h, x0 : x0 + w]

    @staticmethod
    def _circular_mask(size: int, radius: float) -> np.ndarray:
        """Soft-edged circular mask (float32, 0-1)."""
        y, x = np.ogrid[:size, :size]
        center = size / 2.0
        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        feather = 35.0
        return np.clip((radius - dist) / feather, 0.0, 1.0).astype(np.float32)

    def _render_sparkles(
        self, size: int, blend_progress: float
    ) -> np.ndarray:
        """Animated sparkle overlay (float32 BGR, additive)."""
        overlay = np.zeros((size, size, 3), dtype=np.float32)
        for sp in self._sparkles:
            age = blend_progress - sp["birth"]
            if age < 0 or age > sp["life"]:
                continue
            # fade in → hold → fade out
            fade_in = 0.05
            fade_out = 0.10
            if age < fade_in:
                b = (age / fade_in) * sp["brightness"]
            elif age > sp["life"] - fade_out:
                b = ((sp["life"] - age) / fade_out) * sp["brightness"]
            else:
                b = sp["brightness"]
            color = (b * 220, b * 235, b * 255)  # warm white
            cv2.circle(overlay, (sp["x"], sp["y"]), sp["radius"], color, -1)
        return cv2.GaussianBlur(overlay, (7, 7), 0)

    @staticmethod
    def _edge_glow(mask: np.ndarray) -> np.ndarray:
        """Golden glow on the expanding reveal edge (float32 BGR)."""
        blurred = cv2.GaussianBlur(
            (mask * 255).astype(np.uint8), (51, 51), 0
        ).astype(np.float32) / 255.0
        edge = np.clip(blurred - mask, 0.0, 1.0) * 2.0
        # warm gold: B=180, G=220, R=255
        return np.stack(
            [edge * 180, edge * 220, edge * 255], axis=-1
        )

    @staticmethod
    def _light_rays(size: int, intensity: float, num_rays: int = 16) -> np.ndarray:
        """Radial starburst emanating from center (float32 BGR)."""
        canvas = np.zeros((size, size), dtype=np.float32)
        cx = cy = size // 2
        length = int(size * 0.75)
        for i in range(num_rays):
            angle = (2 * math.pi * i) / num_rays
            x2 = int(cx + length * math.cos(angle))
            y2 = int(cy + length * math.sin(angle))
            cv2.line(canvas, (cx, cy), (x2, y2), float(intensity), 6)
        canvas = cv2.GaussianBlur(canvas, (41, 41), 0)
        return np.stack([canvas * 200, canvas * 225, canvas * 255], axis=-1)

    # ── frame rendering ─────────────────────

    def _render_frame(
        self,
        t: float,
        current: np.ndarray,
        ideal: np.ndarray,
        result: np.ndarray,
    ) -> np.ndarray:
        """Render a single frame at time *t* seconds."""
        frame = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), 255, dtype=np.uint8)
        y = FACE_IMAGE_Y
        cx = VIDEO_WIDTH // 2

        # Phase boundaries (cumulative)
        t1 = PHASE_HOLD_CURRENT
        t2 = t1 + PHASE_FLASH_1
        t3 = t2 + PHASE_HOLD_IDEAL
        t4 = t3 + PHASE_FLASH_2
        t5 = t4 + PHASE_BLEND
        t6 = t5 + PHASE_GLOW_SETTLE
        t7 = t6 + PHASE_HOLD_RESULT

        if t < t1:
            # ── Phase 1: Hold current ────────────────
            p = t / t1
            face = self._ken_burns(current, p)
            frame[y : y + FACE_IMAGE_SIZE, :] = face
            _put_centered_text(
                frame, "Before", cx, y + FACE_IMAGE_SIZE + 50,
                font_scale=1.3, color=TEXT_GRAY, thickness=2,
            )

        elif t < t2:
            # ── Phase 2: Flash → ideal ───────────────
            p = (t - t1) / PHASE_FLASH_1
            alpha = math.sin(p * math.pi)
            face = current if p < 0.5 else ideal
            frame[y : y + FACE_IMAGE_SIZE, :] = face
            white = np.full_like(frame, 255)
            cv2.addWeighted(frame, 1 - alpha, white, alpha, 0, frame)

        elif t < t3:
            # ── Phase 3: Hold ideal ──────────────────
            p = (t - t2) / PHASE_HOLD_IDEAL
            face = self._ken_burns(ideal, p)
            frame[y : y + FACE_IMAGE_SIZE, :] = face
            _put_centered_text(
                frame, "Ideal", cx, y + FACE_IMAGE_SIZE + 50,
                font_scale=1.3, color=TEXT_GRAY, thickness=2,
            )

        elif t < t4:
            # ── Phase 4: Golden flash builds ─────────
            p = (t - t3) / PHASE_FLASH_2
            alpha = p * 0.85
            frame[y : y + FACE_IMAGE_SIZE, :] = ideal
            warm = np.full_like(frame, 255)
            warm[:, :, 0] = 200  # less blue → warm
            cv2.addWeighted(frame, 1 - alpha, warm, alpha, 0, frame)

        elif t < t5:
            # ── Phase 5: Artistic blend (circular reveal) ─
            p = (t - t4) / PHASE_BLEND
            eased = _ease_in_out(p)
            max_r = FACE_IMAGE_SIZE * 0.82
            mask = self._circular_mask(FACE_IMAGE_SIZE, eased * max_r)
            m3 = np.stack([mask] * 3, axis=-1)

            face = (
                ideal.astype(np.float32) * (1 - m3)
                + result.astype(np.float32) * m3
            ).astype(np.uint8)
            frame[y : y + FACE_IMAGE_SIZE, :] = face

            # sparkle particles
            sparkle = self._render_sparkles(FACE_IMAGE_SIZE, p)
            region = frame[y : y + FACE_IMAGE_SIZE, :].astype(np.float32)
            frame[y : y + FACE_IMAGE_SIZE, :] = np.clip(
                region + sparkle, 0, 255
            ).astype(np.uint8)

            # golden edge glow
            glow = self._edge_glow(mask)
            region = frame[y : y + FACE_IMAGE_SIZE, :].astype(np.float32)
            frame[y : y + FACE_IMAGE_SIZE, :] = np.clip(
                region + glow, 0, 255
            ).astype(np.uint8)

            # light rays (strongest in the middle of transition)
            ray_intensity = math.sin(p * math.pi) * 0.35
            if ray_intensity > 0.02:
                rays = self._light_rays(FACE_IMAGE_SIZE, ray_intensity)
                region = frame[y : y + FACE_IMAGE_SIZE, :].astype(np.float32)
                frame[y : y + FACE_IMAGE_SIZE, :] = np.clip(
                    region + rays, 0, 255
                ).astype(np.uint8)

            # fading warm overlay
            flash_a = max(0.0, 0.55 - p * 0.7)
            if flash_a > 0.01:
                warm = np.full_like(frame, 255)
                warm[:, :, 0] = 210
                cv2.addWeighted(frame, 1 - flash_a, warm, flash_a, 0, frame)

        elif t < t6:
            # ── Phase 6: Glow settle ─────────────────
            p = (t - t5) / PHASE_GLOW_SETTLE
            frame[y : y + FACE_IMAGE_SIZE, :] = result
            alpha = (1 - p) * 0.12
            warm = np.full_like(frame, 255)
            cv2.addWeighted(frame, 1 - alpha, warm, alpha, 0, frame)

        elif t < t7:
            # ── Phase 7: Hold result ─────────────────
            p = (t - t6) / PHASE_HOLD_RESULT
            face = self._ken_burns(result, p, zoom=0.02)
            frame[y : y + FACE_IMAGE_SIZE, :] = face
            _put_centered_text(
                frame, "Simulation Result", cx, y + FACE_IMAGE_SIZE + 50,
                font_scale=1.3, color=ACCENT_BGR, thickness=2,
            )

        else:
            # ── Phase 8: Branding ────────────────────
            frame[y : y + FACE_IMAGE_SIZE, :] = result
            _put_centered_text(
                frame, "Cao", cx, y + FACE_IMAGE_SIZE + 70,
                font_scale=2.0, color=ACCENT_BGR, thickness=3,
            )
            _put_centered_text(
                frame, "cao.style-elements.jp", cx, y + FACE_IMAGE_SIZE + 130,
                font_scale=0.8, color=LIGHT_GRAY, thickness=2,
            )

        return frame

    # ── full pipeline ───────────────────────

    def _generate_all_frames(
        self,
        current: np.ndarray,
        ideal: np.ndarray,
        result: np.ndarray,
    ) -> List[np.ndarray]:
        total_frames = int(TOTAL_DURATION * FPS)
        frames: List[np.ndarray] = []
        for i in range(total_frames):
            t = i / FPS
            frames.append(self._render_frame(t, current, ideal, result))
        return frames

    def _encode(self, frames: List[np.ndarray]) -> VideoResult:
        """Encode frames using the shared codec fallback chain."""
        import os
        import tempfile

        for codec, ext, content_type in CODEC_CHAIN:
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=ext, delete=False
                ) as tmp:
                    tmp_path = tmp.name

                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    tmp_path, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT)
                )
                if not writer.isOpened():
                    os.unlink(tmp_path)
                    continue

                for f in frames:
                    writer.write(f)
                writer.release()

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


def get_blend_video_generator() -> BlendVideoGenerator:
    """Get BlendVideoGenerator instance."""
    return BlendVideoGenerator()
