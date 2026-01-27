"""Swap compositor service for selective parts composition.

Orchestrates the composition of individual facial parts from a swapped face
onto the original face with configurable intensity levels.
"""

import logging
from typing import Dict, Optional

import cv2
import numpy as np

from app.services.face_parsing import get_face_parsing_service
from app.services.part_blender import PartBlender, PartsSelection

logger = logging.getLogger(__name__)

# Part aliases for convenience
PART_ALIASES: Dict[str, list[str]] = {
    "eyes": ["left_eye", "right_eye"],
    "eyebrows": ["left_eyebrow", "right_eyebrow"],
}

# Valid individual part names
VALID_PARTS = {"left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "nose", "lips"}


class SwapCompositor:
    """Orchestrates selective parts composition from swapped face to original.

    Uses PartBlender for the actual blending operation and provides
    intensity control for each part.
    """

    def __init__(
        self,
        use_bisenet: bool = True,
        use_seamless_clone: bool = True,
    ):
        """Initialize the swap compositor.

        Args:
            use_bisenet: Whether to use BiSeNet for mask generation.
            use_seamless_clone: Whether to use seamless clone for blending.
        """
        self.use_bisenet = use_bisenet
        self.use_seamless_clone = use_seamless_clone

    def _normalize_parts_dict(self, parts: Dict[str, float]) -> Dict[str, float]:
        """Normalize parts dictionary by expanding aliases and clamping values.

        Args:
            parts: Dictionary of part names to intensity values.

        Returns:
            Normalized dictionary with expanded aliases and clamped values.
        """
        normalized: Dict[str, float] = {}

        for part, intensity in parts.items():
            # Clamp intensity to 0-1
            clamped_intensity = max(0.0, min(1.0, intensity))

            # Expand aliases
            if part in PART_ALIASES:
                for expanded_part in PART_ALIASES[part]:
                    normalized[expanded_part] = clamped_intensity
            elif part in VALID_PARTS:
                normalized[part] = clamped_intensity
            else:
                logger.warning(f"Unknown part name: {part}, ignoring")

        return normalized

    def _blend_with_intensity(
        self,
        original: np.ndarray,
        blended: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Blend result with original based on intensity.

        Args:
            original: Original image (BGR).
            blended: Blended image (BGR).
            intensity: Blend intensity (0.0 = original, 1.0 = blended).

        Returns:
            Intensity-adjusted blended image.
        """
        if intensity >= 1.0:
            return blended.copy()
        if intensity <= 0.0:
            return original.copy()

        # Linear interpolation
        result = cv2.addWeighted(
            original, 1.0 - intensity, blended, intensity, 0
        )
        return result

    def compose_parts(
        self,
        original: np.ndarray,
        swapped: np.ndarray,
        parts: Dict[str, float],
    ) -> np.ndarray:
        """Compose selected parts from swapped face onto original.

        Args:
            original: Original face image (BGR format).
            swapped: Swapped face image (BGR format).
            parts: Dictionary of part names to intensity values.
                   Supports aliases like "eyes" -> ["left_eye", "right_eye"].

        Returns:
            Composed image with selected parts from swapped face.
        """
        # Normalize parts dictionary
        normalized_parts = self._normalize_parts_dict(parts)

        # If no parts selected or all zero intensity, return original
        if not normalized_parts or all(v == 0.0 for v in normalized_parts.values()):
            return original.copy()

        # Group parts by intensity for efficient processing
        intensity_groups: Dict[float, list[str]] = {}
        for part, intensity in normalized_parts.items():
            if intensity > 0:
                if intensity not in intensity_groups:
                    intensity_groups[intensity] = []
                intensity_groups[intensity].append(part)

        result = original.copy()

        # Process each intensity group
        for intensity, group_parts in intensity_groups.items():
            # Create PartsSelection for this group
            selection = PartsSelection(
                left_eye="left_eye" in group_parts,
                right_eye="right_eye" in group_parts,
                left_eyebrow="left_eyebrow" in group_parts,
                right_eyebrow="right_eyebrow" in group_parts,
                nose="nose" in group_parts,
                lips="lips" in group_parts,
            )

            # Skip if no parts selected
            if not selection.get_selected_parts():
                continue

            try:
                # Create blender and blend parts
                blender = PartBlender(
                    use_bisenet=self.use_bisenet,
                    use_seamless_clone=self.use_seamless_clone,
                )

                # Blend swapped parts onto current result
                blended = blender.blend(
                    current_img=result,
                    ideal_img=swapped,
                    parts=selection,
                    current_label="original",
                    ideal_label="swapped",
                )

                # Apply intensity
                result = self._blend_with_intensity(result, blended, intensity)

            except Exception as e:
                logger.error(f"Failed to blend parts {group_parts}: {e}")
                # Continue with other parts on failure
                continue

        return result

    def compose_all_parts(
        self,
        original: np.ndarray,
        swapped: np.ndarray,
        parts: Dict[str, float],
    ) -> np.ndarray:
        """Compose all parts at once with individual intensities.

        This is a convenience method that processes all parts in a single pass.

        Args:
            original: Original face image (BGR format).
            swapped: Swapped face image (BGR format).
            parts: Dictionary of part names to intensity values.

        Returns:
            Composed image with all selected parts applied.
        """
        return self.compose_parts(original, swapped, parts)

    def preserve_hair(
        self,
        original: np.ndarray,
        swapped: np.ndarray,
        blur_size: int = 21,
        dilate_pixels: int = 3,
    ) -> np.ndarray:
        """Preserve original hair on swapped face result.

        Uses BiSeNet to detect hair region in original image and composites
        it onto the swapped result to maintain hair quality.

        Args:
            original: Original face image (BGR format) - source of hair.
            swapped: Swapped face image (BGR format) - target to apply hair.
            blur_size: Gaussian blur size for soft edge blending.
            dilate_pixels: Pixels to dilate hair mask for coverage.

        Returns:
            Swapped image with original hair preserved.
        """
        # Ensure images have same dimensions
        if original.shape != swapped.shape:
            swapped = cv2.resize(swapped, (original.shape[1], original.shape[0]))

        # Get face parsing service
        face_parsing = get_face_parsing_service()

        if not face_parsing.is_available():
            logger.warning("BiSeNet not available, skipping hair preservation")
            return swapped.copy()

        try:
            # Get hair mask from original image
            hair_mask = face_parsing.get_part_mask(
                original,
                "hair",
                landmarks=None,
                dilate_pixels=dilate_pixels,
                blur_size=blur_size,
            )

            # If no hair detected, return swapped as-is
            if hair_mask.sum() == 0:
                logger.info("No hair detected in original image")
                return swapped.copy()

            # Normalize mask to float for blending
            mask_float = hair_mask.astype(np.float32) / 255.0
            mask_3ch = np.stack([mask_float] * 3, axis=-1)

            # Blend: original hair where mask is white, swapped elsewhere
            result = (original * mask_3ch + swapped * (1.0 - mask_3ch)).astype(np.uint8)

            logger.info("Hair preserved from original image")
            return result

        except Exception as e:
            logger.error(f"Failed to preserve hair: {e}")
            return swapped.copy()


# Singleton instance
_swap_compositor: Optional[SwapCompositor] = None


def get_swap_compositor() -> SwapCompositor:
    """Get the singleton SwapCompositor instance.

    Returns:
        SwapCompositor instance.
    """
    global _swap_compositor

    if _swap_compositor is None:
        _swap_compositor = SwapCompositor()

    return _swap_compositor
