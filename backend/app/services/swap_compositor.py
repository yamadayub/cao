"""Swap compositor service for selective parts composition.

Orchestrates the composition of individual facial parts from a swapped face
onto the original face with configurable intensity levels.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.services.face_detection import get_face_detection_service
from app.services.face_parsing import get_face_parsing_service
from app.services.part_blender import PartBlender, PartsSelection

logger = logging.getLogger(__name__)

# Part aliases for convenience
# "eyes" now includes both eyes AND eyebrows as a single unit
PART_ALIASES: Dict[str, list[str]] = {
    "eyes": ["left_eye", "right_eye", "left_eyebrow", "right_eyebrow"],
}

# Valid individual part names (internal use)
VALID_PARTS = {"left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "nose", "lips"}

# Valid simplified part names (API interface)
VALID_SIMPLIFIED_PARTS = {"eyes", "nose", "lips"}

# MediaPipe Face Mesh face oval landmark indices
# These form the outer contour of the face (forehead to chin)
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


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

    def _create_face_mask_from_landmarks(
        self,
        image: np.ndarray,
        landmarks: List[Tuple[float, float]],
        dilate_pixels: int = 10,
        blur_size: int = 31,
    ) -> np.ndarray:
        """Create a face mask from MediaPipe Face Mesh landmarks.

        Uses the face oval contour landmarks to create a filled polygon mask
        at the full image resolution.

        Args:
            image: Input image (BGR format) for dimensions.
            landmarks: MediaPipe landmark points as (x, y) pixel coordinates.
            dilate_pixels: Pixels to dilate face mask.
            blur_size: Gaussian blur size for soft edges.

        Returns:
            Face mask (H, W) with values 0-255.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Get face oval contour points
        contour_points = []
        for idx in FACE_OVAL_INDICES:
            if idx < len(landmarks):
                x, y = landmarks[idx]
                contour_points.append([int(x), int(y)])

        if len(contour_points) < 10:
            return mask

        contour_points = np.array(contour_points, dtype=np.int32)

        # Fill the face oval polygon
        cv2.fillPoly(mask, [contour_points], 255)

        # Close small gaps
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        # Dilate to cover face-hair boundary
        if dilate_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1),
            )
            mask = cv2.dilate(mask, kernel)

        # Blur for soft transition
        if blur_size > 0:
            ks = blur_size if blur_size % 2 == 1 else blur_size + 1
            mask = cv2.GaussianBlur(mask, (ks, ks), 0)

        return mask

    def _composite_with_face_mask(
        self,
        original: np.ndarray,
        swapped: np.ndarray,
        face_mask: np.ndarray,
    ) -> np.ndarray:
        """Composite swapped face onto original using a face mask.

        Face mask area: use swapped pixels.
        Non-face area: use original pixels (hair, background at full res).

        Args:
            original: Original image (BGR).
            swapped: Swapped image (BGR).
            face_mask: Face mask (H, W) with values 0-255.

        Returns:
            Composited image.
        """
        h, w = original.shape[:2]
        mask_float = face_mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_float] * 3, axis=-1)

        result = (
            swapped * mask_3ch + original * (1.0 - mask_3ch)
        ).astype(np.uint8)

        logger.info(
            f"Hair preserved: face mask covers "
            f"{(face_mask > 127).sum() / (h * w) * 100:.1f}%% of image"
        )
        return result

    def preserve_hair(
        self,
        original: np.ndarray,
        swapped: np.ndarray,
        blur_size: int = 31,
        dilate_pixels: int = 10,
    ) -> np.ndarray:
        """Preserve original hair/background by overlaying only the swapped face.

        Detects the FACE region and only uses swapped pixels there.
        Everything else (hair, background, ears, neck) comes directly
        from the original at full resolution.

        Uses BiSeNet if available, otherwise falls back to MediaPipe
        Face Mesh landmarks for face mask creation.

        Args:
            original: Original face image (BGR format) - kept for non-face.
            swapped: Swapped face image (BGR format) - used for face region.
            blur_size: Gaussian blur size for soft edge blending.
            dilate_pixels: Pixels to dilate face mask for seamless coverage.

        Returns:
            Original image with swapped face composited in.
        """
        # Ensure images have same dimensions
        if original.shape != swapped.shape:
            logger.info(
                f"Resizing swapped {swapped.shape[:2]} → {original.shape[:2]}"
            )
            swapped = cv2.resize(
                swapped,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )

        h, w = original.shape[:2]

        # Try BiSeNet first (more accurate semantic segmentation)
        face_parsing = get_face_parsing_service()
        if face_parsing.is_available():
            try:
                seg_map = face_parsing._parse_native(original)

                face_labels = {1, 2, 3, 4, 5, 10, 11, 12, 13}
                face_mask_native = np.zeros(seg_map.shape, dtype=np.uint8)
                for label in face_labels:
                    face_mask_native[seg_map == label] = 255

                if face_mask_native.sum() > 0:
                    close_kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (15, 15)
                    )
                    face_mask_native = cv2.morphologyEx(
                        face_mask_native, cv2.MORPH_CLOSE, close_kernel
                    )

                    face_mask = cv2.resize(
                        face_mask_native, (w, h), interpolation=cv2.INTER_LINEAR
                    )

                    if dilate_pixels > 0:
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1),
                        )
                        face_mask = cv2.dilate(face_mask, kernel)

                    if blur_size > 0:
                        ks = blur_size if blur_size % 2 == 1 else blur_size + 1
                        face_mask = cv2.GaussianBlur(face_mask, (ks, ks), 0)

                    logger.info("Using BiSeNet face mask for hair preservation")
                    return self._composite_with_face_mask(original, swapped, face_mask)

            except Exception as e:
                logger.warning(f"BiSeNet face mask failed: {e}, trying MediaPipe")

        # Fallback: MediaPipe Face Mesh landmark-based face mask
        logger.info("Using MediaPipe Face Mesh for hair preservation")
        try:
            face_detection = get_face_detection_service()
            landmarks, lm_w, lm_h = face_detection.get_landmark_points(
                swapped, image_label="swapped"
            )

            face_mask = self._create_face_mask_from_landmarks(
                original, landmarks,
                dilate_pixels=dilate_pixels,
                blur_size=blur_size,
            )

            if face_mask.sum() == 0:
                logger.warning("MediaPipe face mask is empty")
                return swapped.copy()

            return self._composite_with_face_mask(original, swapped, face_mask)

        except Exception as e:
            logger.error(f"MediaPipe face mask also failed: {e}")
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
