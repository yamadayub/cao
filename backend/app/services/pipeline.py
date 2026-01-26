"""Generation pipeline for face simulation.

This module orchestrates the face generation pipeline, combining:
- Face detection and validation
- Morphing (full face)
- Parts blending (selective features)
- (Future) Diffusion inpainting for quality enhancement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from app.services.face_detection import (
    FaceDetectionService,
    ImageValidationError,
    get_face_detection_service,
)
from app.services.morphing import MorphingService, get_morphing_service
from app.services.part_blender import PartBlender, PartsSelection, get_part_blender_service
from app.utils.image import bytes_to_cv2, cv2_to_base64

logger = logging.getLogger(__name__)


# Valid parts for parts mode
VALID_PARTS = {"left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "nose", "lips"}

# Simplified parts mapping (user-friendly names to internal names)
PARTS_ALIAS = {
    "eyes": ["left_eye", "right_eye"],
    "eyebrows": ["left_eyebrow", "right_eyebrow"],
    "eye": ["left_eye", "right_eye"],
    "eyebrow": ["left_eyebrow", "right_eyebrow"],
    "mouth": ["lips"],
}


@dataclass
class PipelineResult:
    """Result from pipeline execution."""

    success: bool
    image: Optional[np.ndarray] = None
    image_base64: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PipelineError(Exception):
    """Pipeline execution error."""

    pass


class GenerationPipeline:
    """
    Face generation pipeline.

    Handles preprocessing, generation (morph/parts), and postprocessing.
    Designed for future extension with Diffusion-based refinement.
    """

    def __init__(
        self,
        face_service: Optional[FaceDetectionService] = None,
        morphing_service: Optional[MorphingService] = None,
        part_blender: Optional[PartBlender] = None,
    ):
        """Initialize pipeline with services."""
        self.face_service = face_service or get_face_detection_service()
        self.morphing_service = morphing_service or get_morphing_service()
        self.part_blender = part_blender or get_part_blender_service()

    def generate(
        self,
        base_image_data: str,
        target_image_data: str,
        mode: str,
        parts: Optional[List[str]] = None,
        strength: float = 0.5,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> PipelineResult:
        """
        Execute the generation pipeline.

        Args:
            base_image_data: Base64 encoded base image
            target_image_data: Base64 encoded target image
            mode: 'morph' or 'parts'
            parts: List of parts to blend (for mode='parts')
            strength: Blend strength (0=base, 1=target)
            seed: Random seed (for future use with diffusion)
            progress_callback: Callback for progress updates (progress, message)

        Returns:
            PipelineResult with generated image or error
        """
        if progress_callback:
            progress_callback(0, "Starting pipeline")

        try:
            # 1. Preprocess and validate images
            if progress_callback:
                progress_callback(10, "Validating images")

            base_img, target_img = self._preprocess_images(
                base_image_data, target_image_data
            )

            # 2. Validate faces
            if progress_callback:
                progress_callback(20, "Detecting faces")

            self._validate_faces(base_img, target_img)

            # 3. Execute generation based on mode
            if mode == "morph":
                if progress_callback:
                    progress_callback(30, "Morphing faces")
                result_img = self._execute_morph(base_img, target_img, strength)
            elif mode == "parts":
                if progress_callback:
                    progress_callback(30, "Blending parts")
                result_img = self._execute_parts(base_img, target_img, parts or [])
            else:
                raise PipelineError(f"Unknown mode: {mode}")

            # 4. Postprocess (quality enhancement, future: diffusion)
            if progress_callback:
                progress_callback(80, "Postprocessing")

            result_img = self._postprocess(result_img)

            # 5. Convert to base64
            if progress_callback:
                progress_callback(90, "Encoding result")

            result_base64 = cv2_to_base64(result_img)

            if progress_callback:
                progress_callback(100, "Complete")

            return PipelineResult(
                success=True,
                image=result_img,
                image_base64=result_base64,
                metadata={
                    "mode": mode,
                    "strength": strength,
                    "parts": parts,
                },
            )

        except ImageValidationError as e:
            logger.error(f"Face detection failed: {e}")
            return PipelineResult(success=False, error=str(e))
        except PipelineError as e:
            logger.error(f"Pipeline error: {e}")
            return PipelineResult(success=False, error=str(e))
        except Exception as e:
            logger.exception(f"Unexpected pipeline error: {e}")
            return PipelineResult(success=False, error=f"Processing failed: {str(e)}")

    def _preprocess_images(
        self, base_data: str, target_data: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess input images.

        - Decode from base64
        - Normalize to RGB
        - Apply max size limit
        - Handle EXIF rotation
        """
        MAX_DIMENSION = 2048

        # Decode base64 to CV2 images
        base_img = self._decode_image(base_data, "base")
        target_img = self._decode_image(target_data, "target")

        # Resize if too large
        base_img = self._limit_size(base_img, MAX_DIMENSION)
        target_img = self._limit_size(target_img, MAX_DIMENSION)

        return base_img, target_img

    def _decode_image(self, data: str, label: str) -> np.ndarray:
        """Decode base64 image data."""
        # Remove data URL prefix if present
        if data.startswith("data:"):
            data = data.split(",", 1)[1]

        import base64

        try:
            image_bytes = base64.b64decode(data)
        except Exception as e:
            raise PipelineError(f"Invalid base64 encoding for {label} image: {e}")

        img = bytes_to_cv2(image_bytes)
        if img is None:
            raise PipelineError(f"Failed to decode {label} image")

        return img

    def _limit_size(self, img: np.ndarray, max_dim: int) -> np.ndarray:
        """Limit image to maximum dimension while preserving aspect ratio."""
        h, w = img.shape[:2]
        if max(h, w) <= max_dim:
            return img

        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _validate_faces(self, base_img: np.ndarray, target_img: np.ndarray) -> None:
        """
        Validate that both images contain detectable faces.

        Raises ImageValidationError if face detection fails.
        """
        # This will raise ImageValidationError if no face is detected
        self.face_service.get_landmark_points(base_img, "base")
        self.face_service.get_landmark_points(target_img, "target")

    def _execute_morph(
        self,
        base_img: np.ndarray,
        target_img: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """
        Execute full face morphing.

        Uses landmark-based warping to morph inner face features
        while preserving the base face outline.
        """
        result = self.morphing_service.morph(
            base_img,
            target_img,
            progress=strength,
            img1_label="base",
            img2_label="target",
        )
        return result

    def _execute_parts(
        self,
        base_img: np.ndarray,
        target_img: np.ndarray,
        parts: List[str],
    ) -> np.ndarray:
        """
        Execute parts-based blending.

        Blends selected facial features from target to base.
        """
        # Normalize part names (handle aliases)
        normalized_parts = self._normalize_parts(parts)

        if not normalized_parts:
            raise PipelineError("No valid parts specified for parts mode")

        # Create PartsSelection
        selection = PartsSelection(
            left_eye="left_eye" in normalized_parts,
            right_eye="right_eye" in normalized_parts,
            left_eyebrow="left_eyebrow" in normalized_parts,
            right_eyebrow="right_eyebrow" in normalized_parts,
            nose="nose" in normalized_parts,
            lips="lips" in normalized_parts,
        )

        result = self.part_blender.blend(
            base_img,
            target_img,
            selection,
            current_label="base",
            ideal_label="target",
        )
        return result

    def _normalize_parts(self, parts: List[str]) -> List[str]:
        """Normalize part names, expanding aliases."""
        normalized = set()
        for part in parts:
            part_lower = part.lower().strip()

            # Check aliases first
            if part_lower in PARTS_ALIAS:
                normalized.update(PARTS_ALIAS[part_lower])
            elif part_lower in VALID_PARTS:
                normalized.add(part_lower)
            else:
                logger.warning(f"Unknown part: {part}, ignoring")

        return list(normalized)

    def _postprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Postprocess the generated image.

        Current implementation: basic quality adjustments.
        Future: Diffusion-based refinement for natural appearance.
        """
        # Apply slight sharpening to enhance details
        kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0],
        ])

        # Apply only to high-detail areas
        sharpened = cv2.filter2D(img, -1, kernel)

        # Blend with original (subtle sharpening)
        result = cv2.addWeighted(img, 0.7, sharpened, 0.3, 0)

        return result


# Global instance
_pipeline: Optional[GenerationPipeline] = None


def get_generation_pipeline() -> GenerationPipeline:
    """Get or create generation pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = GenerationPipeline()
    return _pipeline
