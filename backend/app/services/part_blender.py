"""Part-by-part face blending service."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel
from scipy.interpolate import RBFInterpolator

from app.services.face_detection import get_face_detection_service

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
    # Nose - more precise definition focusing on core nose structure
    "nose": [
        # Nose bridge (top to bottom center line)
        6, 168, 197, 195, 5, 4,
        # Nose tip center
        1, 2, 98, 327,
        # Nose tip and bottom
        19, 94, 141, 370,
        # Left nostril outline
        239, 238, 20, 79, 218, 237,
        # Right nostril outline
        459, 458, 250, 309, 438, 457,
        # Nose wing left
        129, 49, 131, 134, 51, 45,
        # Nose wing right
        358, 279, 360, 363, 281, 275,
        # Alar base (nostril base)
        48, 115, 220, 64,
        278, 344, 440, 294,
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

# Extended landmark indices for warping context (surrounding area)
PART_CONTEXT_LANDMARKS: Dict[str, List[int]] = {
    "left_eye": [
        # Eyebrow nearby
        70, 63, 105, 66, 107,
        # Cheek nearby
        116, 123, 147, 213, 192,
        # Nose bridge
        6, 168,
    ],
    "right_eye": [
        # Eyebrow nearby
        300, 293, 334, 296, 336,
        # Cheek nearby
        345, 352, 376, 433, 416,
        # Nose bridge
        6, 168,
    ],
    "left_eyebrow": [
        # Forehead nearby - upper area only
        109, 10, 67, 69, 104, 68, 71, 139,
        # Temple area
        21, 54, 103, 108,
    ],
    "right_eyebrow": [
        # Forehead nearby - upper area only
        109, 10, 338, 297, 332, 299, 301, 368,
        # Temple area
        251, 284, 332, 337,
    ],
    "nose": [
        # Between eyes
        6, 168, 197,
        # Cheeks
        116, 123, 345, 352,
        # Upper lip
        164, 167, 393, 391,
    ],
    "lips": [
        # Chin
        152, 175, 199, 200, 18, 400, 428, 369,
        # Nose bottom
        2, 164, 0,
        # Cheeks - extended for better blending
        187, 207, 411, 427, 205, 425, 32, 262,
        # Lower face contour
        149, 150, 136, 172, 378, 379, 365, 397,
    ],
}

# Exclusion masks: landmarks that should be EXCLUDED from the mask
PART_EXCLUSION_LANDMARKS: Dict[str, List[int]] = {
    "left_eyebrow": FACIAL_PART_LANDMARKS["left_eye"],  # Exclude left eye from eyebrow
    "right_eyebrow": FACIAL_PART_LANDMARKS["right_eye"],  # Exclude right eye from eyebrow
}

# Key alignment landmarks for precise positioning of each part
# These are the most stable reference points for calculating centroid and angle
PART_ALIGNMENT_LANDMARKS: Dict[str, List[int]] = {
    "nose": [
        # Nose bridge top (between eyes)
        6,
        # Nose tip
        4, 1,
        # Nose bottom center
        2,
        # Left and right alar base (nostril sides) - for width/angle
        129, 358,
        # Left and right nose wing tips
        48, 278,
    ],
    "lips": [
        # Upper lip center
        0,
        # Lower lip center
        17,
        # Left corner
        61,
        # Right corner
        291,
        # Cupid's bow points
        37, 267,
    ],
    "left_eye": [
        # Inner corner
        133,
        # Outer corner
        33,
        # Upper lid center
        159,
        # Lower lid center
        145,
    ],
    "right_eye": [
        # Inner corner
        362,
        # Outer corner
        263,
        # Upper lid center
        386,
        # Lower lid center
        374,
    ],
    "left_eyebrow": [
        # Inner point
        107,
        # Outer point
        70,
        # Center top
        105,
    ],
    "right_eyebrow": [
        # Inner point
        336,
        # Outer point
        300,
        # Center top
        334,
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
            context_indices = PART_CONTEXT_LANDMARKS.get(part_name, [])

            try:
                result = self._blend_part(
                    result,
                    ideal_img,
                    current_landmarks,
                    ideal_landmarks,
                    part_indices,
                    context_indices,
                    (out_w, out_h),
                    part_name,
                )
            except Exception as e:
                # Log error but continue with other parts
                print(f"Warning: Failed to blend {part_name}: {e}")
                continue

        return result

    def _blend_part(
        self,
        base_img: np.ndarray,
        src_img: np.ndarray,
        base_landmarks: List[Tuple[float, float]],
        src_landmarks: List[Tuple[float, float]],
        part_indices: List[int],
        context_indices: List[int],
        img_size: Tuple[int, int],
        part_name: str = "",
    ) -> np.ndarray:
        """
        Blend a single facial part from source to base image.

        Uses centroid/angle alignment + RBF-based warping for natural shape adaptation.
        """
        w, h = img_size

        # Use alignment landmarks if available (more stable reference points)
        alignment_indices = PART_ALIGNMENT_LANDMARKS.get(part_name, part_indices)

        # Calculate centroid and angle for alignment using key landmarks
        src_centroid, src_angle, src_scale = self._calculate_part_geometry(
            src_landmarks, alignment_indices, part_name
        )
        dst_centroid, dst_angle, dst_scale = self._calculate_part_geometry(
            base_landmarks, alignment_indices, part_name
        )

        if src_centroid is None or dst_centroid is None:
            return base_img

        # Combine part and context indices for warping
        all_indices = list(set(part_indices + context_indices))

        # Get source and destination points for warping
        src_points = []
        dst_points = []
        for idx in all_indices:
            if idx < len(src_landmarks) and idx < len(base_landmarks):
                src_points.append(src_landmarks[idx])
                dst_points.append(base_landmarks[idx])

        if len(src_points) < 4:
            return base_img

        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # First, apply centroid-based alignment transform
        aligned_src = self._align_by_centroid_and_angle(
            src_img,
            src_centroid, src_angle, src_scale,
            dst_centroid, dst_angle, dst_scale,
            (w, h)
        )

        # Adjust source points to match the alignment transform
        aligned_src_points = self._transform_points(
            src_points,
            src_centroid, src_angle, src_scale,
            dst_centroid, dst_angle, dst_scale
        )

        # Warp the aligned source image for fine-grained adjustment
        warped_src = self._rbf_warp(aligned_src, aligned_src_points, dst_points, (w, h))

        # Create soft mask for the part (with exclusions if applicable)
        exclusion_indices = PART_EXCLUSION_LANDMARKS.get(part_name, [])
        mask = self._create_soft_mask_with_exclusion(
            base_landmarks, part_indices, exclusion_indices, (h, w)
        )

        if mask.sum() == 0:
            return base_img

        # Apply extra feathering for lips to blend better with surrounding skin
        if part_name == "lips":
            mask = self._feather_mask(mask, iterations=3, blur_size=31, erode_size=7)

        # Apply color correction to match skin tones
        warped_src = self._advanced_color_transfer(warped_src, base_img, mask)

        # Multi-band blend for natural transition
        # Use more levels for lips for smoother blending
        num_levels = 5 if part_name == "lips" else 4
        result = self._multiband_blend(base_img, warped_src, mask, num_levels=num_levels)

        return result

    def _calculate_part_geometry(
        self,
        landmarks: List[Tuple[float, float]],
        part_indices: List[int],
        part_name: str = "",
    ) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Calculate centroid, principal angle, and scale of a facial part.

        Returns:
            centroid: (x, y) center point
            angle: rotation angle in radians
            scale: approximate size (mean distance from centroid)
        """
        points = []
        for idx in part_indices:
            if idx < len(landmarks):
                points.append(landmarks[idx])

        if len(points) < 3:
            return None, 0.0, 1.0

        points = np.array(points, dtype=np.float32)

        # Special handling for nose - use specific reference points
        if part_name == "nose":
            return self._calculate_nose_geometry(landmarks)

        # Calculate centroid
        centroid = np.mean(points, axis=0)

        # Calculate principal angle using PCA
        centered = points - centroid
        cov = np.cov(centered.T)
        if cov.shape == (2, 2):
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Principal axis is the eigenvector with largest eigenvalue
            principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
            angle = np.arctan2(principal_axis[1], principal_axis[0])
        else:
            angle = 0.0

        # Calculate scale as mean distance from centroid
        distances = np.linalg.norm(centered, axis=1)
        scale = np.mean(distances) if len(distances) > 0 else 1.0
        scale = max(scale, 1.0)  # Prevent division by zero

        return centroid, angle, scale

    def _calculate_nose_geometry(
        self,
        landmarks: List[Tuple[float, float]],
    ) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Calculate nose geometry using precise anatomical reference points.

        Uses:
        - Bridge top (6) and tip (4) for vertical axis/angle
        - Nostril sides (129, 358) for width/scale
        - Nose tip (1) for centroid
        """
        # Key nose landmarks
        NOSE_BRIDGE_TOP = 6      # Top of nose bridge (between eyes)
        NOSE_TIP = 4             # Nose tip (prónasale)
        NOSE_TIP_CENTER = 1     # Nose tip center
        NOSE_BOTTOM = 2          # Bottom of nose (subnasale)
        LEFT_ALAR = 129          # Left nostril wing
        RIGHT_ALAR = 358         # Right nostril wing

        try:
            bridge_top = np.array(landmarks[NOSE_BRIDGE_TOP], dtype=np.float32)
            nose_tip = np.array(landmarks[NOSE_TIP], dtype=np.float32)
            nose_tip_center = np.array(landmarks[NOSE_TIP_CENTER], dtype=np.float32)
            nose_bottom = np.array(landmarks[NOSE_BOTTOM], dtype=np.float32)
            left_alar = np.array(landmarks[LEFT_ALAR], dtype=np.float32)
            right_alar = np.array(landmarks[RIGHT_ALAR], dtype=np.float32)
        except (IndexError, TypeError):
            return None, 0.0, 1.0

        # Centroid: weighted center focusing on tip area
        # Use nose tip center as the main reference, slightly adjusted toward bridge
        centroid = nose_tip_center * 0.6 + nose_tip * 0.2 + nose_bottom * 0.2

        # Angle: from bridge top to nose tip (should be roughly vertical)
        nose_axis = nose_tip - bridge_top
        # Calculate angle from vertical (not horizontal)
        # A perfectly vertical nose has angle = pi/2 from horizontal
        angle = np.arctan2(nose_axis[1], nose_axis[0])

        # Scale: combination of length and width
        # Nose length: bridge to tip
        length = np.linalg.norm(nose_tip - bridge_top)
        # Nose width: left to right alar
        width = np.linalg.norm(right_alar - left_alar)
        # Use geometric mean for balanced scale
        scale = np.sqrt(length * width)
        scale = max(scale, 1.0)

        return centroid, angle, scale

    def _align_by_centroid_and_angle(
        self,
        img: np.ndarray,
        src_centroid: np.ndarray,
        src_angle: float,
        src_scale: float,
        dst_centroid: np.ndarray,
        dst_angle: float,
        dst_scale: float,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Align source image by matching centroid, angle, and scale.
        """
        w, h = output_size

        # Calculate rotation angle difference
        angle_diff = dst_angle - src_angle

        # Calculate scale ratio (with limits to prevent extreme scaling)
        scale_ratio = np.clip(dst_scale / src_scale, 0.7, 1.4)

        # Build transformation matrix
        # 1. Translate to origin
        # 2. Scale
        # 3. Rotate
        # 4. Translate to destination
        cos_a = np.cos(angle_diff)
        sin_a = np.sin(angle_diff)

        # Combined transform matrix
        transform = np.array([
            [scale_ratio * cos_a, -scale_ratio * sin_a,
             dst_centroid[0] - scale_ratio * (cos_a * src_centroid[0] - sin_a * src_centroid[1])],
            [scale_ratio * sin_a, scale_ratio * cos_a,
             dst_centroid[1] - scale_ratio * (sin_a * src_centroid[0] + cos_a * src_centroid[1])],
        ], dtype=np.float32)

        # Apply transformation
        aligned = cv2.warpAffine(
            img, transform, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return aligned

    def _transform_points(
        self,
        points: np.ndarray,
        src_centroid: np.ndarray,
        src_angle: float,
        src_scale: float,
        dst_centroid: np.ndarray,
        dst_angle: float,
        dst_scale: float,
    ) -> np.ndarray:
        """
        Transform points using the same centroid/angle/scale alignment.
        """
        angle_diff = dst_angle - src_angle
        scale_ratio = np.clip(dst_scale / src_scale, 0.7, 1.4)

        cos_a = np.cos(angle_diff)
        sin_a = np.sin(angle_diff)

        # Transform each point
        transformed = []
        for pt in points:
            # Translate to origin
            x = pt[0] - src_centroid[0]
            y = pt[1] - src_centroid[1]

            # Scale and rotate
            new_x = scale_ratio * (cos_a * x - sin_a * y)
            new_y = scale_ratio * (sin_a * x + cos_a * y)

            # Translate to destination
            new_x += dst_centroid[0]
            new_y += dst_centroid[1]

            transformed.append([new_x, new_y])

        return np.array(transformed, dtype=np.float32)

    def _rbf_warp(
        self,
        img: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Warp image using RBF (Radial Basis Function) interpolation.

        This provides smooth, natural warping that preserves local structure.
        """
        w, h = output_size
        out_h, out_w = img.shape[:2]

        # Add corner points for stability
        corners_src = np.array([
            [0, 0], [out_w-1, 0], [0, out_h-1], [out_w-1, out_h-1],
            [out_w//2, 0], [out_w//2, out_h-1], [0, out_h//2], [out_w-1, out_h//2],
        ], dtype=np.float32)
        corners_dst = np.array([
            [0, 0], [w-1, 0], [0, h-1], [w-1, h-1],
            [w//2, 0], [w//2, h-1], [0, h//2], [w-1, h//2],
        ], dtype=np.float32)

        all_src = np.vstack([src_points, corners_src])
        all_dst = np.vstack([dst_points, corners_dst])

        # Create RBF interpolator for x and y coordinates
        try:
            rbf_x = RBFInterpolator(all_dst, all_src[:, 0], kernel='thin_plate_spline', smoothing=1.0)
            rbf_y = RBFInterpolator(all_dst, all_src[:, 1], kernel='thin_plate_spline', smoothing=1.0)
        except Exception:
            # Fallback to affine transform if RBF fails
            return self._affine_warp(img, src_points, dst_points, output_size)

        # Create output grid
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        # Compute source coordinates
        src_x = rbf_x(grid_points).reshape(h, w).astype(np.float32)
        src_y = rbf_y(grid_points).reshape(h, w).astype(np.float32)

        # Remap the image
        warped = cv2.remap(img, src_x, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        return warped

    def _affine_warp(
        self,
        img: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """Fallback affine warping when RBF fails."""
        w, h = output_size

        # Use affine transform
        transform, _ = cv2.estimateAffinePartial2D(
            src_points, dst_points, method=cv2.RANSAC
        )

        if transform is None:
            return cv2.resize(img, (w, h))

        return cv2.warpAffine(
            img, transform, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    def _create_soft_mask(
        self,
        landmarks: List[Tuple[float, float]],
        part_indices: List[int],
        img_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Create a soft-edged mask for natural blending.
        """
        return self._create_soft_mask_with_exclusion(landmarks, part_indices, [], img_shape)

    def _create_soft_mask_with_exclusion(
        self,
        landmarks: List[Tuple[float, float]],
        part_indices: List[int],
        exclusion_indices: List[int],
        img_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Create a soft-edged mask for natural blending, with exclusion regions.

        Args:
            landmarks: Face landmarks
            part_indices: Indices for the part to include
            exclusion_indices: Indices for areas to exclude (e.g., eyes from eyebrow)
            img_shape: (height, width)
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

        # Create convex hull
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

        # Create exclusion mask if needed
        if exclusion_indices:
            exclusion_mask = np.zeros((h, w), dtype=np.uint8)
            exclusion_points = []
            for idx in exclusion_indices:
                if idx < len(landmarks):
                    x, y = landmarks[idx]
                    exclusion_points.append([int(x), int(y)])

            if len(exclusion_points) >= 3:
                exclusion_points = np.array(exclusion_points, dtype=np.int32)
                exclusion_hull = cv2.convexHull(exclusion_points)
                cv2.fillConvexPoly(exclusion_mask, exclusion_hull, 255)

                # Expand exclusion area slightly with blur for soft edge
                kernel = np.ones((11, 11), np.uint8)
                exclusion_mask = cv2.dilate(exclusion_mask, kernel, iterations=2)
                exclusion_mask = cv2.GaussianBlur(exclusion_mask, (21, 21), 0)

                # Subtract exclusion from main mask
                mask = cv2.subtract(mask, exclusion_mask)

        # Expand mask slightly
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Apply strong Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (31, 31), 0)

        # Apply additional feathering at edges
        mask = self._feather_mask(mask, iterations=2, blur_size=21, erode_size=5)

        return mask

    def _feather_mask(
        self,
        mask: np.ndarray,
        iterations: int = 1,
        blur_size: int = 21,
        erode_size: int = 5,
    ) -> np.ndarray:
        """Apply feathering to mask edges for smoother blending."""
        result = mask.astype(np.float32)

        for _ in range(iterations):
            # Erode and blur
            eroded = cv2.erode(mask, np.ones((erode_size, erode_size), np.uint8), iterations=1)
            blurred = cv2.GaussianBlur(result, (blur_size, blur_size), 0)

            # Combine: keep center, use blurred for edges
            center_mask = eroded.astype(np.float32) / 255.0
            result = center_mask * result + (1 - center_mask) * blurred

        return result.astype(np.uint8)

    def _advanced_color_transfer(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Advanced color transfer using local statistics.
        """
        if mask.sum() == 0:
            return src

        result = src.copy()

        # Convert to LAB color space
        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Create mask for statistics (use a dilated version for context)
        kernel = np.ones((15, 15), np.uint8)
        stat_mask = cv2.dilate(mask, kernel, iterations=2)
        mask_bool = stat_mask > 127

        if not np.any(mask_bool):
            return src

        # Transfer each channel
        for i in range(3):
            src_channel = src_lab[:, :, i]
            tgt_channel = tgt_lab[:, :, i]

            src_masked = src_channel[mask_bool]
            tgt_masked = tgt_channel[mask_bool]

            if len(src_masked) == 0 or len(tgt_masked) == 0:
                continue

            # Robust statistics (use median and percentile range)
            src_median = np.median(src_masked)
            src_p25, src_p75 = np.percentile(src_masked, [25, 75])
            src_iqr = max(src_p75 - src_p25, 1.0)

            tgt_median = np.median(tgt_masked)
            tgt_p25, tgt_p75 = np.percentile(tgt_masked, [25, 75])
            tgt_iqr = max(tgt_p75 - tgt_p25, 1.0)

            # Apply transfer with blending factor
            scale = tgt_iqr / src_iqr
            # Limit scale to prevent extreme changes
            scale = np.clip(scale, 0.5, 2.0)

            transferred = (src_channel - src_median) * scale + tgt_median

            # Blend with original based on mask intensity
            blend_mask = mask.astype(np.float32) / 255.0
            src_lab[:, :, i] = src_channel * (1 - blend_mask) + transferred * blend_mask

        # Clip and convert back
        src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)

        return result

    def _multiband_blend(
        self,
        base: np.ndarray,
        src: np.ndarray,
        mask: np.ndarray,
        num_levels: int = 4,
    ) -> np.ndarray:
        """
        Multi-band blending (Laplacian pyramid blending) for seamless transitions.
        """
        # Ensure images are same size
        if base.shape != src.shape:
            src = cv2.resize(src, (base.shape[1], base.shape[0]))

        # Normalize mask to float
        mask_float = mask.astype(np.float32) / 255.0
        mask_3ch = np.dstack([mask_float, mask_float, mask_float])

        # Build Gaussian pyramid for mask
        mask_pyramid = [mask_3ch]
        current_mask = mask_3ch
        for _ in range(num_levels):
            current_mask = cv2.pyrDown(current_mask)
            mask_pyramid.append(current_mask)

        # Build Laplacian pyramid for base
        base_lap = []
        current = base.astype(np.float32)
        for _ in range(num_levels):
            down = cv2.pyrDown(current)
            up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
            lap = current - up
            base_lap.append(lap)
            current = down
        base_lap.append(current)

        # Build Laplacian pyramid for source
        src_lap = []
        current = src.astype(np.float32)
        for _ in range(num_levels):
            down = cv2.pyrDown(current)
            up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
            lap = current - up
            src_lap.append(lap)
            current = down
        src_lap.append(current)

        # Blend pyramids
        blended_lap = []
        for i in range(num_levels + 1):
            m = mask_pyramid[min(i, len(mask_pyramid) - 1)]
            # Resize mask if needed
            if m.shape[:2] != base_lap[i].shape[:2]:
                m = cv2.resize(m, (base_lap[i].shape[1], base_lap[i].shape[0]))
            blended = src_lap[i] * m + base_lap[i] * (1 - m)
            blended_lap.append(blended)

        # Reconstruct
        result = blended_lap[-1]
        for i in range(num_levels - 1, -1, -1):
            result = cv2.pyrUp(result, dstsize=(blended_lap[i].shape[1], blended_lap[i].shape[0]))
            result = result + blended_lap[i]

        return np.clip(result, 0, 255).astype(np.uint8)

    # Keep legacy methods for backward compatibility
    def _extract_part_mask(
        self,
        landmarks: List[Tuple[float, float]],
        part_indices: List[int],
        img_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Create a mask for a facial part (legacy method)."""
        return self._create_soft_mask(landmarks, part_indices, img_shape)

    def _extract_part_region(
        self,
        img: np.ndarray,
        landmarks: List[Tuple[float, float]],
        part_indices: List[int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Extract a facial part region with mask (legacy method)."""
        h, w = img.shape[:2]
        mask = self._create_soft_mask(landmarks, part_indices, (h, w))

        if mask.sum() == 0:
            return None, None, None

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
        """Calculate affine transformation (legacy method)."""
        src_points = []
        dst_points = []

        for idx in part_indices:
            if idx < len(src_landmarks) and idx < len(dst_landmarks):
                src_points.append(src_landmarks[idx])
                dst_points.append(dst_landmarks[idx])

        if len(src_points) < 3:
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        transform, _ = cv2.estimateAffinePartial2D(
            src_points, dst_points, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )

        if transform is None:
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        return transform

    def _get_part_center(
        self,
        landmarks: List[Tuple[float, float]],
        part_indices: List[int],
        img_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Get the center point of a facial part (legacy method)."""
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

        cx = max(1, min(w - 2, cx))
        cy = max(1, min(h - 2, cy))

        return (cx, cy)

    def _color_transfer(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Color transfer (legacy method)."""
        return self._advanced_color_transfer(src, tgt, mask)

    def _seamless_blend(
        self,
        base: np.ndarray,
        part: np.ndarray,
        mask: np.ndarray,
        center: Tuple[int, int],
    ) -> np.ndarray:
        """Seamless blend (legacy method)."""
        return self._multiband_blend(base, part, mask)


# Global instance
_part_blender_service: Optional[PartBlender] = None


def get_part_blender_service() -> PartBlender:
    """Get or create part blender service instance."""
    global _part_blender_service
    if _part_blender_service is None:
        _part_blender_service = PartBlender()
    return _part_blender_service
