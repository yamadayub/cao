"""Part-by-part face blending service."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel

from app.services.face_detection import get_face_detection_service
from app.utils.image import ImageValidationError


# MediaPipe Face Mesh landmark indices for each facial part
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
FACIAL_PART_LANDMARKS: Dict[str, List[int]] = {
    # Left eye (user's perspective - right side of image)
    "left_eye": [
        # Eye contour
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        # Upper eyelid
        246, 161, 160, 159, 158, 157, 173,
        # Lower eyelid
        247, 30, 29, 27, 28, 56, 190,
        # Inner corner
        130, 25, 110, 24, 23, 22, 26, 112, 243,
        # Outer corner
        226, 31, 228, 229, 230, 231, 232, 233,
    ],
    # Right eye (user's perspective - left side of image)
    "right_eye": [
        # Eye contour
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        # Upper eyelid
        466, 388, 387, 386, 385, 384, 398,
        # Lower eyelid
        467, 260, 259, 257, 258, 286, 414,
        # Inner corner
        359, 255, 339, 254, 253, 252, 256, 341, 463,
        # Outer corner
        446, 261, 448, 449, 450, 451, 452, 453,
    ],
    # Left eyebrow
    "left_eyebrow": [
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
        # Additional points for fuller coverage
        156, 35, 124, 143, 111, 117, 118, 119, 120, 121,
    ],
    # Right eyebrow
    "right_eyebrow": [
        300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
        # Additional points for fuller coverage
        383, 265, 353, 372, 340, 346, 347, 348, 349, 350,
    ],
    # Nose
    "nose": [
        # Nose bridge
        6, 168, 197, 195, 5,
        # Nose tip
        4, 1, 19, 94, 2,
        # Left nostril
        98, 97, 99, 100, 240, 235, 129, 64, 48, 115,
        # Right nostril
        327, 326, 328, 329, 460, 455, 358, 294, 278, 344,
        # Nose sides
        131, 134, 102, 49, 238, 20, 242,
        360, 363, 331, 279, 458, 250, 462,
        # Additional nose contour
        75, 76, 77, 78, 79, 80, 81, 82,
        305, 306, 307, 308, 309, 310, 311, 312,
    ],
    # Lips/Mouth
    "lips": [
        # Outer upper lip
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        # Outer lower lip
        375, 321, 405, 314, 17, 84, 181, 91, 146,
        # Inner upper lip
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
        # Inner lower lip
        324, 318, 402, 317, 14, 87, 178, 88, 95,
        # Additional lip contour
        62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292,
    ],
}


class PartsSelection(BaseModel):
    """Model for selected parts configuration."""

    left_eye: bool = False
    right_eye: bool = False
    left_eyebrow: bool = False
    right_eyebrow: bool = False
    nose: bool = False
    lips: bool = False

    def get_selected_parts(self) -> List[str]:
        """Get list of selected part names."""
        selected = []
        if self.left_eye:
            selected.append("left_eye")
        if self.right_eye:
            selected.append("right_eye")
        if self.left_eyebrow:
            selected.append("left_eyebrow")
        if self.right_eyebrow:
            selected.append("right_eyebrow")
        if self.nose:
            selected.append("nose")
        if self.lips:
            selected.append("lips")
        return selected

    def has_any_selection(self) -> bool:
        """Check if any part is selected."""
        return any(
            [
                self.left_eye,
                self.right_eye,
                self.left_eyebrow,
                self.right_eyebrow,
                self.nose,
                self.lips,
            ]
        )


class PartBlender:
    """Service for blending specific facial parts from one face to another."""

    def __init__(self):
        """Initialize part blender service."""
        self.face_service = get_face_detection_service()

    def blend(
        self,
        current_img: np.ndarray,
        ideal_img: np.ndarray,
        parts: PartsSelection,
        current_label: str = "current",
        ideal_label: str = "ideal",
    ) -> np.ndarray:
        """
        Blend selected facial parts from ideal image onto current image.

        Args:
            current_img: Current face image (BGR format) - base image
            ideal_img: Ideal face image (BGR format) - source of parts
            parts: Selection of which parts to blend
            current_label: Label for current image in error messages
            ideal_label: Label for ideal image in error messages

        Returns:
            Blended image (BGR format)

        Raises:
            ValueError: If no parts are selected
            ImageValidationError: If face detection fails
        """
        if not parts.has_any_selection():
            raise ValueError("At least one part must be selected")

        # Get landmarks for both images
        current_landmarks, c_w, c_h = self.face_service.get_landmark_points(
            current_img, current_label
        )
        ideal_landmarks, i_w, i_h = self.face_service.get_landmark_points(
            ideal_img, ideal_label
        )

        # Resize images to common size (use current image size as reference)
        out_w, out_h = c_w, c_h

        if i_w != out_w or i_h != out_h:
            ideal_img = cv2.resize(ideal_img, (out_w, out_h))
            ideal_landmarks = [
                (x * out_w / i_w, y * out_h / i_h) for x, y in ideal_landmarks
            ]

        # Start with current image as base
        result = current_img.copy()

        # Process each selected part
        selected_parts = parts.get_selected_parts()
        for part_name in selected_parts:
            part_indices = FACIAL_PART_LANDMARKS[part_name]

            try:
                result = self._blend_part(
                    result,
                    ideal_img,
                    current_landmarks,
                    ideal_landmarks,
                    part_indices,
                    (out_w, out_h),
                )
            except Exception:
                # Skip parts that fail to blend
                continue

        return result

    def _blend_part(
        self,
        base_img: np.ndarray,
        src_img: np.ndarray,
        base_landmarks: List[Tuple[float, float]],
        src_landmarks: List[Tuple[float, float]],
        part_indices: List[int],
        img_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Blend a single facial part from source to base image.

        Args:
            base_img: Base image to blend onto
            src_img: Source image to extract part from
            base_landmarks: Landmarks of base face
            src_landmarks: Landmarks of source face
            part_indices: Landmark indices defining the part
            img_size: (width, height) of images

        Returns:
            Image with blended part
        """
        w, h = img_size

        # Extract part region from source
        src_region, src_mask, src_center = self._extract_part_region(
            src_img, src_landmarks, part_indices
        )

        if src_region is None or src_mask is None:
            return base_img

        # Calculate transformation from source to base
        transform = self._calculate_part_transform(
            src_landmarks, base_landmarks, part_indices
        )

        # Warp source part to base position
        warped_region = cv2.warpAffine(
            src_region,
            transform,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        warped_mask = cv2.warpAffine(
            src_mask,
            transform,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Find center for seamless cloning
        base_center = self._get_part_center(base_landmarks, part_indices, (w, h))

        # Apply color transfer to match skin tones
        warped_region = self._color_transfer(warped_region, base_img, warped_mask)

        # Seamless blend
        result = self._seamless_blend(base_img, warped_region, warped_mask, base_center)

        return result

    def _extract_part_mask(
        self,
        landmarks: List[Tuple[float, float]],
        part_indices: List[int],
        img_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Create a mask for a facial part.

        Args:
            landmarks: Face landmarks
            part_indices: Indices of landmarks defining the part
            img_shape: (height, width) of image

        Returns:
            Binary mask (uint8)
        """
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Get points for the part
        points = []
        for idx in part_indices:
            if idx < len(landmarks):
                x, y = landmarks[idx]
                points.append([int(x), int(y)])

        if len(points) < 3:
            return mask

        points = np.array(points, dtype=np.int32)

        # Create convex hull for the part
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

        # Dilate mask slightly for better coverage
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Smooth edges
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        return mask

    def _extract_part_region(
        self,
        img: np.ndarray,
        landmarks: List[Tuple[float, float]],
        part_indices: List[int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """
        Extract a facial part region with mask.

        Args:
            img: Source image
            landmarks: Face landmarks
            part_indices: Indices defining the part

        Returns:
            Tuple of (region image, mask, center point)
        """
        h, w = img.shape[:2]
        mask = self._extract_part_mask(landmarks, part_indices, (h, w))

        if mask.sum() == 0:
            return None, None, None

        # Find center of mask
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return None, None, None

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        return img.copy(), mask, (cx, cy)

    def _calculate_part_transform(
        self,
        src_landmarks: List[Tuple[float, float]],
        dst_landmarks: List[Tuple[float, float]],
        part_indices: List[int],
    ) -> np.ndarray:
        """
        Calculate affine transformation to map source part to destination.

        Args:
            src_landmarks: Source face landmarks
            dst_landmarks: Destination face landmarks
            part_indices: Indices of landmarks for the part

        Returns:
            2x3 affine transformation matrix
        """
        # Get corresponding points
        src_points = []
        dst_points = []

        for idx in part_indices:
            if idx < len(src_landmarks) and idx < len(dst_landmarks):
                src_points.append(src_landmarks[idx])
                dst_points.append(dst_landmarks[idx])

        if len(src_points) < 3:
            # Return identity transform if not enough points
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # Use estimateAffinePartial2D for robust estimation
        transform, _ = cv2.estimateAffinePartial2D(
            src_points, dst_points, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )

        if transform is None:
            # Fallback to identity
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        return transform

    def _get_part_center(
        self,
        landmarks: List[Tuple[float, float]],
        part_indices: List[int],
        img_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Get the center point of a facial part.

        Args:
            landmarks: Face landmarks
            part_indices: Indices of landmarks for the part
            img_size: (width, height) of image

        Returns:
            Center point (x, y)
        """
        w, h = img_size
        points = []

        for idx in part_indices:
            if idx < len(landmarks):
                x, y = landmarks[idx]
                points.append([x, y])

        if not points:
            return (w // 2, h // 2)

        points = np.array(points)
        cx = int(np.mean(points[:, 0]))
        cy = int(np.mean(points[:, 1]))

        # Clamp to image bounds
        cx = max(1, min(w - 2, cx))
        cy = max(1, min(h - 2, cy))

        return (cx, cy)

    def _color_transfer(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Transfer colors from source to match target using histogram matching.

        Args:
            src: Source image (part to be blended)
            tgt: Target image (base image)
            mask: Mask of the part region

        Returns:
            Color-adjusted source image
        """
        if mask.sum() == 0:
            return src

        result = src.copy()

        # Convert to LAB color space for better color transfer
        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Create expanded mask for statistics
        mask_bool = mask > 127

        # Calculate statistics for each channel
        for i in range(3):
            src_channel = src_lab[:, :, i]
            tgt_channel = tgt_lab[:, :, i]

            # Get masked values
            src_masked = src_channel[mask_bool]
            tgt_masked = tgt_channel[mask_bool]

            if len(src_masked) == 0 or len(tgt_masked) == 0:
                continue

            # Calculate mean and std
            src_mean = np.mean(src_masked)
            src_std = np.std(src_masked) + 1e-6
            tgt_mean = np.mean(tgt_masked)
            tgt_std = np.std(tgt_masked) + 1e-6

            # Apply transfer
            src_lab[:, :, i] = (src_channel - src_mean) * (tgt_std / src_std) + tgt_mean

        # Clip values
        src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)

        # Convert back to BGR
        result = cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)

        return result

    def _seamless_blend(
        self,
        base: np.ndarray,
        part: np.ndarray,
        mask: np.ndarray,
        center: Tuple[int, int],
    ) -> np.ndarray:
        """
        Seamlessly blend part into base image using Poisson blending.

        Args:
            base: Base image
            part: Part image to blend
            mask: Mask of the part
            center: Center point for blending

        Returns:
            Blended image
        """
        h, w = base.shape[:2]

        # Ensure center is within valid bounds
        cx, cy = center
        cx = max(1, min(w - 2, cx))
        cy = max(1, min(h - 2, cy))

        # Ensure mask has non-zero region
        if mask.sum() == 0:
            return base

        # Use seamless cloning
        try:
            result = cv2.seamlessClone(
                part,
                base,
                mask,
                (cx, cy),
                cv2.NORMAL_CLONE,
            )
        except cv2.error:
            # Fallback to simple alpha blending if seamless clone fails
            mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
            result = (part * mask_3ch + base * (1 - mask_3ch)).astype(np.uint8)

        return result


# Global instance
_part_blender_service: Optional[PartBlender] = None


def get_part_blender_service() -> PartBlender:
    """Get or create part blender service instance."""
    global _part_blender_service
    if _part_blender_service is None:
        _part_blender_service = PartBlender()
    return _part_blender_service
