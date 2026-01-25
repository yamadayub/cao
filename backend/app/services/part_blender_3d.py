"""3D-based part blending service using canonical face model fitting."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay

from app.services.face_detection import get_face_detection_service
from app.services.part_blender import (
    FACIAL_PART_LANDMARKS,
    PART_CONTEXT_LANDMARKS,
    PartsSelection,
)

logger = logging.getLogger(__name__)

# Depth estimation model singleton
_depth_model = None
_depth_processor = None

# ============================================================================
# MediaPipe Canonical Face Model - Key Landmarks (3D coordinates in cm)
# Based on the official MediaPipe canonical_face_model.obj
# Selected key landmarks that define the face shape for Procrustes fitting
# ============================================================================

# Key landmark indices for pose estimation (stable points)
POSE_LANDMARKS = {
    "nose_tip": 4,
    "nose_bridge": 6,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_outer": 263,
    "right_eye_inner": 362,
    "left_mouth": 61,
    "right_mouth": 291,
    "chin": 152,
    "forehead": 10,
    "left_cheek": 234,
    "right_cheek": 454,
}

# Nose-specific landmark indices for centerline and alignment
NOSE_ALIGNMENT_LANDMARKS = {
    "nose_tip": 4,           # Tip of nose (pronnasale)
    "nose_bridge_top": 6,    # Top of nose bridge (nasion area)
    "nose_bridge_mid": 168,  # Middle of nose bridge
    "nose_bottom": 2,        # Bottom of nose (subnasale)
    "left_alar": 129,        # Left nostril wing (alar)
    "right_alar": 358,       # Right nostril wing (alar)
    "left_nostril": 48,      # Left nostril base
    "right_nostril": 278,    # Right nostril base
}

# Eye landmarks for computing eye center
EYE_CENTER_LANDMARKS = {
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
}

# Canonical 3D coordinates for key landmarks (from MediaPipe canonical_face_model.obj)
# Units are in centimeters, Y-down, Z toward camera
CANONICAL_LANDMARKS_3D = {
    4: np.array([0.0, -0.463, 7.587]),       # nose tip
    6: np.array([0.0, 0.366, 7.243]),        # nose bridge
    33: np.array([-3.20, 1.99, 3.80]),       # left eye outer
    133: np.array([-1.30, 1.42, 4.83]),      # left eye inner
    263: np.array([3.20, 1.99, 3.80]),       # right eye outer
    362: np.array([1.30, 1.42, 4.83]),       # right eye inner
    61: np.array([-1.83, -4.10, 4.25]),      # left mouth
    291: np.array([1.83, -4.10, 4.25]),      # right mouth
    152: np.array([0.0, -6.15, 5.07]),       # chin
    10: np.array([0.0, 4.89, 5.39]),         # forehead
    234: np.array([-5.80, 2.35, 2.20]),      # left cheek
    454: np.array([5.80, 2.35, 2.20]),       # right cheek
    # Additional nose landmarks for better fitting
    1: np.array([0.0, -1.13, 7.48]),         # nose bridge top
    2: np.array([0.0, -2.09, 6.06]),         # nose dorsum
    94: np.array([0.0, -1.72, 6.60]),        # nose center
    # Eye corners for roll estimation
    130: np.array([-1.41, 0.97, 4.56]),      # left eye corner
    359: np.array([1.41, 0.97, 4.56]),       # right eye corner
}

# All indices used for fitting
FITTING_INDICES = list(CANONICAL_LANDMARKS_3D.keys())


def get_depth_model():
    """Get or create depth estimation model (lazy loading)."""
    global _depth_model, _depth_processor

    if _depth_model is None:
        try:
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            # Use Depth Anything V2 small model for balance of speed and quality
            model_name = "depth-anything/Depth-Anything-V2-Small-hf"

            logger.info(f"Loading depth model: {model_name}")
            _depth_processor = AutoImageProcessor.from_pretrained(model_name)
            _depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)

            # Use CPU for inference (Heroku doesn't have GPU)
            _depth_model.eval()
            logger.info("Depth model loaded successfully")
        except ImportError as e:
            logger.warning(f"Depth model dependencies not available: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            return None, None

    return _depth_model, _depth_processor


class CanonicalFaceModel:
    """
    Canonical 3D face model based on MediaPipe's face geometry.

    Uses Procrustes analysis to fit observed 2D landmarks to the
    canonical 3D model and recover pose (rotation, translation, scale).
    """

    def __init__(self):
        """Initialize with canonical 3D landmarks."""
        self.canonical_3d = np.array([
            CANONICAL_LANDMARKS_3D[idx] for idx in FITTING_INDICES
        ], dtype=np.float32)
        self.fitting_indices = FITTING_INDICES

        # Center canonical model at origin
        self.canonical_center = np.mean(self.canonical_3d, axis=0)
        self.canonical_centered = self.canonical_3d - self.canonical_center

        # Compute canonical model scale
        self.canonical_scale = np.std(self.canonical_centered)

    def fit_to_landmarks(
        self,
        landmarks_2d: np.ndarray,
        img_size: Tuple[int, int],
        depth_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Fit canonical model to observed 2D landmarks using Procrustes analysis.

        Args:
            landmarks_2d: (N, 2) array of all 2D landmarks
            img_size: (width, height) of image
            depth_map: Optional depth map for Z estimation

        Returns:
            rotation: (3, 3) rotation matrix
            translation: (3,) translation vector
            scale: float scale factor
            landmarks_3d: (N, 3) reconstructed 3D landmarks
        """
        w, h = img_size

        # Extract fitting landmarks
        observed_2d = []
        canonical_pts = []
        valid_indices = []

        for i, idx in enumerate(self.fitting_indices):
            if idx < len(landmarks_2d):
                observed_2d.append(landmarks_2d[idx])
                canonical_pts.append(self.canonical_centered[i])
                valid_indices.append(idx)

        if len(observed_2d) < 6:
            # Fallback to identity transform
            return np.eye(3), np.zeros(3), 1.0, self._simple_3d_from_2d(landmarks_2d, img_size)

        observed_2d = np.array(observed_2d, dtype=np.float32)
        canonical_pts = np.array(canonical_pts, dtype=np.float32)

        # Normalize observed 2D to [-1, 1] range
        observed_normalized = observed_2d.copy()
        observed_normalized[:, 0] = (observed_2d[:, 0] / w) * 2 - 1
        observed_normalized[:, 1] = (observed_2d[:, 1] / h) * 2 - 1

        # Estimate Z coordinates from depth map or canonical model
        if depth_map is not None:
            observed_z = self._estimate_z_from_depth(
                observed_2d, depth_map, canonical_pts[:, 2]
            )
        else:
            # Use canonical Z scaled appropriately
            observed_z = canonical_pts[:, 2] / self.canonical_scale

        # Create 3D observed points
        observed_3d = np.column_stack([
            observed_normalized[:, 0],
            observed_normalized[:, 1],
            observed_z
        ])

        # Center observed points
        observed_center = np.mean(observed_3d, axis=0)
        observed_centered = observed_3d - observed_center

        # Normalize canonical points to match observed scale
        canonical_normalized = canonical_pts / self.canonical_scale

        # Procrustes: find optimal rotation from canonical to observed
        rotation, scale = self._orthogonal_procrustes(
            canonical_normalized, observed_centered
        )

        # Compute translation
        translation = observed_center - scale * (rotation @ (np.zeros(3)))

        # Reconstruct all 3D landmarks
        landmarks_3d = self._reconstruct_all_3d(
            landmarks_2d, img_size, rotation, translation, scale, depth_map
        )

        return rotation, translation, scale, landmarks_3d

    def _orthogonal_procrustes(
        self,
        A: np.ndarray,
        B: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Solve orthogonal Procrustes: find R that minimizes ||B - RA||^2.

        Args:
            A: (N, 3) source points (canonical)
            B: (N, 3) target points (observed)

        Returns:
            R: (3, 3) rotation matrix
            s: scale factor
        """
        # Compute cross-covariance matrix
        H = A.T @ B

        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)

        # Optimal rotation
        R = Vt.T @ U.T

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute optimal scale
        scale = np.sum(S) / np.sum(A ** 2)

        return R, scale

    def _estimate_z_from_depth(
        self,
        points_2d: np.ndarray,
        depth_map: np.ndarray,
        canonical_z: np.ndarray,
    ) -> np.ndarray:
        """Estimate Z coordinates using depth map and canonical shape prior."""
        h, w = depth_map.shape
        z_values = []

        for i, (x, y) in enumerate(points_2d):
            ix = int(np.clip(x, 0, w - 1))
            iy = int(np.clip(y, 0, h - 1))

            # Sample depth in local region
            kernel = 3
            x_start, x_end = max(0, ix - kernel), min(w, ix + kernel + 1)
            y_start, y_end = max(0, iy - kernel), min(h, iy + kernel + 1)
            depth_val = np.median(depth_map[y_start:y_end, x_start:x_end])

            # Blend depth estimate with canonical prior
            # (depth is noisy, canonical provides structure)
            canonical_weight = 0.3
            z = (1 - canonical_weight) * depth_val + \
                canonical_weight * (canonical_z[i] / self.canonical_scale + 0.5)

            z_values.append(z)

        return np.array(z_values, dtype=np.float32)

    def _simple_3d_from_2d(
        self,
        landmarks_2d: np.ndarray,
        img_size: Tuple[int, int],
    ) -> np.ndarray:
        """Fallback: simple 3D reconstruction without fitting."""
        w, h = img_size
        n = len(landmarks_2d)
        landmarks_3d = np.zeros((n, 3), dtype=np.float32)

        for i, (x, y) in enumerate(landmarks_2d):
            landmarks_3d[i] = [
                (x / w) * 2 - 1,
                (y / h) * 2 - 1,
                0.5  # Default depth
            ]

        return landmarks_3d

    def _reconstruct_all_3d(
        self,
        landmarks_2d: np.ndarray,
        img_size: Tuple[int, int],
        rotation: np.ndarray,
        translation: np.ndarray,
        scale: float,
        depth_map: Optional[np.ndarray],
    ) -> np.ndarray:
        """Reconstruct 3D coordinates for all landmarks."""
        w, h = img_size
        n = len(landmarks_2d)
        landmarks_3d = np.zeros((n, 3), dtype=np.float32)

        for i, (x, y) in enumerate(landmarks_2d):
            # Normalize 2D
            x_norm = (x / w) * 2 - 1
            y_norm = (y / h) * 2 - 1

            # Estimate Z from depth or model
            if depth_map is not None:
                ix = int(np.clip(x, 0, w - 1))
                iy = int(np.clip(y, 0, h - 1))
                z = depth_map[iy, ix]
            else:
                z = 0.5

            landmarks_3d[i] = [x_norm, y_norm, z]

        return landmarks_3d

    def get_euler_angles(self, rotation: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract Euler angles (yaw, pitch, roll) from rotation matrix.

        Args:
            rotation: (3, 3) rotation matrix

        Returns:
            yaw: rotation around Y axis (left-right)
            pitch: rotation around X axis (up-down)
            roll: rotation around Z axis (tilt)
        """
        # Handle gimbal lock
        sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)

        if sy > 1e-6:
            pitch = np.arctan2(-rotation[2, 0], sy)
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        else:
            pitch = np.arctan2(-rotation[2, 0], sy)
            yaw = np.arctan2(-rotation[1, 2], rotation[1, 1])
            roll = 0

        return float(yaw), float(pitch), float(roll)


class FaceMesh3D:
    """3D face mesh fitted to canonical model."""

    def __init__(
        self,
        landmarks_2d: np.ndarray,
        depth_map: np.ndarray,
        texture: np.ndarray,
        img_size: Tuple[int, int],
        canonical_model: CanonicalFaceModel,
    ):
        """
        Initialize 3D face mesh with canonical model fitting.

        Args:
            landmarks_2d: (N, 2) array of 2D landmark coordinates
            depth_map: (H, W) depth map
            texture: (H, W, 3) BGR texture image
            img_size: (width, height)
            canonical_model: Shared canonical face model for fitting
        """
        self.landmarks_2d = landmarks_2d
        self.depth_map = depth_map
        self.texture = texture
        self.img_size = img_size
        self.canonical_model = canonical_model

        # Fit canonical model to get pose and 3D landmarks
        self.rotation, self.translation, self.scale, self.vertices_3d = \
            canonical_model.fit_to_landmarks(landmarks_2d, img_size, depth_map)

        # Get Euler angles for easy comparison
        self.yaw, self.pitch, self.roll = canonical_model.get_euler_angles(self.rotation)

        # Create triangulation
        self.triangles = self._create_triangulation()

    def _create_triangulation(self) -> np.ndarray:
        """Create Delaunay triangulation of landmarks."""
        try:
            tri = Delaunay(self.landmarks_2d)
            return tri.simplices
        except Exception:
            return np.array([], dtype=np.int32)

    def get_part_vertices(self, part_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Get 3D vertices and 2D coordinates for a facial part."""
        indices = [i for i in part_indices if i < len(self.vertices_3d)]
        vertices_3d = self.vertices_3d[indices]
        vertices_2d = self.landmarks_2d[indices]
        return vertices_3d, vertices_2d

    def get_yaw_pitch_roll(self) -> Tuple[float, float, float]:
        """Get face rotation angles."""
        return self.yaw, self.pitch, self.roll

    def transform_point_to_other(
        self,
        point_3d: np.ndarray,
        other: "FaceMesh3D",
    ) -> np.ndarray:
        """
        Transform a 3D point from this face's coordinate system to another's.

        This is the key operation for parts blending: we transform the source
        part's 3D position to match the target face's pose.
        """
        # Remove this face's transform
        point_centered = point_3d - self.translation
        point_canonical = (self.rotation.T @ point_centered.T).T / self.scale

        # Apply other face's transform
        point_other = other.scale * (other.rotation @ point_canonical.T).T + other.translation

        return point_other


class PartBlender3D:
    """3D-based facial part blending service using canonical model fitting."""

    def __init__(self):
        """Initialize 3D part blender service."""
        self.face_service = get_face_detection_service()
        self._depth_available = None
        self._has_real_depth = False  # Track if we have real depth data
        self.canonical_model = CanonicalFaceModel()

    def is_depth_available(self) -> bool:
        """
        Check if 3D blending is available.

        Always returns True since canonical model fitting works without depth.
        Depth estimation enhances results but is optional.
        """
        return True

    def has_depth_model(self) -> bool:
        """Check if depth estimation model is available (optional enhancement)."""
        if self._depth_available is None:
            model, processor = get_depth_model()
            self._depth_available = model is not None
        return self._depth_available

    def blend(
        self,
        current_img: np.ndarray,
        ideal_img: np.ndarray,
        parts: PartsSelection,
        current_label: str = "current",
        ideal_label: str = "ideal",
    ) -> np.ndarray:
        """
        Blend selected facial parts using canonical model fitting.
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

        # Resize images to common size
        out_w, out_h = c_w, c_h

        if i_w != out_w or i_h != out_h:
            ideal_img = cv2.resize(ideal_img, (out_w, out_h))
            ideal_landmarks = [
                (x * out_w / i_w, y * out_h / i_h) for x, y in ideal_landmarks
            ]

        # Estimate depth for both images (optional - works without depth too)
        current_depth = self._estimate_depth(current_img)
        ideal_depth = self._estimate_depth(ideal_img)

        # Track if we have real depth data for 3D transformation
        self._has_real_depth = current_depth is not None and ideal_depth is not None

        # If depth is not available, use dummy depth maps (canonical model still works)
        if current_depth is None:
            logger.info("Depth estimation not available, using 2D landmark alignment only")
            current_depth = np.ones((out_h, out_w), dtype=np.float32) * 0.5
        if ideal_depth is None:
            ideal_depth = np.ones((out_h, out_w), dtype=np.float32) * 0.5

        # Create 3D face meshes with canonical model fitting
        current_mesh = FaceMesh3D(
            np.array(current_landmarks, dtype=np.float32),
            current_depth,
            current_img,
            (out_w, out_h),
            self.canonical_model,
        )
        ideal_mesh = FaceMesh3D(
            np.array(ideal_landmarks, dtype=np.float32),
            ideal_depth,
            ideal_img,
            (out_w, out_h),
            self.canonical_model,
        )

        # Log pose information for debugging
        src_yaw, src_pitch, src_roll = ideal_mesh.get_yaw_pitch_roll()
        dst_yaw, dst_pitch, dst_roll = current_mesh.get_yaw_pitch_roll()
        logger.info(f"Source face pose (from Procrustes): yaw={np.degrees(src_yaw):.1f}°, "
                    f"pitch={np.degrees(src_pitch):.1f}°, roll={np.degrees(src_roll):.1f}°")
        logger.info(f"Target face pose (from Procrustes): yaw={np.degrees(dst_yaw):.1f}°, "
                    f"pitch={np.degrees(dst_pitch):.1f}°, roll={np.degrees(dst_roll):.1f}°")

        # Start with current image as base
        result = current_img.copy()

        # Process each selected part
        selected_parts = parts.get_selected_parts()
        for part_name in selected_parts:
            part_indices = FACIAL_PART_LANDMARKS[part_name]
            context_indices = PART_CONTEXT_LANDMARKS.get(part_name, [])

            try:
                result = self._blend_part_3d(
                    result,
                    current_mesh,
                    ideal_mesh,
                    part_indices,
                    context_indices,
                    (out_w, out_h),
                    part_name,
                )
            except Exception as e:
                logger.warning(f"Failed to blend {part_name} in 3D: {e}")
                import traceback
                logger.warning(traceback.format_exc())
                continue

        return result

    def _estimate_depth(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth map from image using Depth Anything."""
        model, processor = get_depth_model()
        if model is None or processor is None:
            return None

        try:
            import torch

            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Prepare input
            inputs = processor(images=rgb_img, return_tensors="pt")

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to original size
            h, w = img.shape[:2]
            depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze().numpy()

            # Normalize to [0, 1]
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth = np.zeros_like(depth)

            return depth.astype(np.float32)

        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None

    def _blend_part_3d(
        self,
        base_img: np.ndarray,
        base_mesh: FaceMesh3D,
        src_mesh: FaceMesh3D,
        part_indices: List[int],
        context_indices: List[int],
        img_size: Tuple[int, int],
        part_name: str = "",
    ) -> np.ndarray:
        """
        Blend a single facial part using canonical model-based transformation.

        When real depth data is available: Use 3D rotation matrices for pose alignment.
        When depth is not available: Use direct 2D landmark correspondence with
        enhanced alignment (scale, translation) without 3D rotation.
        """
        w, h = img_size

        # Get 3D and 2D vertices for the part
        src_3d, src_2d = src_mesh.get_part_vertices(part_indices)
        dst_3d, dst_2d = base_mesh.get_part_vertices(part_indices)

        if len(src_3d) < 4 or len(dst_3d) < 4:
            return base_img

        # Use nose-specific alignment for better results
        if part_name == "nose":
            aligned_2d = self._compute_nose_aligned_points(
                src_2d, dst_2d,
                src_mesh.landmarks_2d, base_mesh.landmarks_2d,
                part_name
            )
        elif self._has_real_depth:
            # Full 3D transformation with rotation for non-nose parts
            aligned_2d = self._compute_3d_aligned_points(
                src_3d, dst_3d, src_mesh, base_mesh, img_size, part_name
            )
        else:
            # Enhanced 2D alignment without 3D rotation (avoids broken results)
            aligned_2d = self._compute_similarity_aligned_points(
                src_2d, dst_2d, part_name
            )

        # Build correspondence for warping
        src_points = list(src_2d)
        dst_points = list(aligned_2d)

        # Add context points (use target landmarks directly for context)
        for idx in context_indices:
            if idx < len(src_mesh.landmarks_2d) and idx < len(base_mesh.landmarks_2d):
                src_points.append(src_mesh.landmarks_2d[idx])
                dst_points.append(base_mesh.landmarks_2d[idx])

        if len(src_points) < 4:
            return base_img

        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # RBF warping for smooth deformation
        warped_src = self._rbf_warp(src_mesh.texture, src_points, dst_points, img_size)

        # Create mask using target positions
        mask = self._create_soft_mask(base_mesh.landmarks_2d, part_indices, (h, w))

        if mask.sum() == 0:
            return base_img

        # Apply color transfer
        warped_src = self._depth_aware_color_transfer(
            warped_src, base_img, mask,
            src_mesh.depth_map, base_mesh.depth_map
        )

        # Multi-band blend
        num_levels = 5 if part_name == "nose" else 4
        result = self._multiband_blend(base_img, warped_src, mask, num_levels)

        return result

    def _compute_3d_aligned_points(
        self,
        src_3d: np.ndarray,
        dst_3d: np.ndarray,
        src_mesh: FaceMesh3D,
        base_mesh: FaceMesh3D,
        img_size: Tuple[int, int],
        part_name: str,
    ) -> np.ndarray:
        """
        Compute aligned 2D points using full 3D transformation (when depth is available).

        This applies rotation matrices from Procrustes analysis for pose alignment.
        """
        # Compute the relative rotation between faces
        R_relative = base_mesh.rotation @ src_mesh.rotation.T

        # Extract relative Euler angles for logging
        rel_yaw, rel_pitch, rel_roll = self.canonical_model.get_euler_angles(R_relative)
        logger.info(f"3D alignment for {part_name}: relative yaw={np.degrees(rel_yaw):.1f}°, "
                    f"pitch={np.degrees(rel_pitch):.1f}°, roll={np.degrees(rel_roll):.1f}°")

        # Transform source 3D points to match target pose
        src_center = np.mean(src_3d, axis=0)
        dst_center = np.mean(dst_3d, axis=0)
        src_centered = src_3d - src_center

        # Apply relative rotation
        src_rotated = (R_relative @ src_centered.T).T

        # Compute scale adjustment
        src_spread = np.std(src_centered)
        dst_spread = np.std(dst_3d - dst_center)
        scale_ratio = dst_spread / max(src_spread, 1e-6)
        scale_ratio = np.clip(scale_ratio, 0.7, 1.4)

        # Apply scale and translate to destination
        src_transformed = src_rotated * scale_ratio + dst_center

        # Project transformed 3D points to 2D
        return self._project_3d_to_2d(src_transformed, img_size)

    def _compute_similarity_aligned_points(
        self,
        src_2d: np.ndarray,
        dst_2d: np.ndarray,
        part_name: str,
    ) -> np.ndarray:
        """
        Compute aligned 2D points using similarity transform (rotation, scale, translation).

        This approach directly maps source part landmarks to destination part
        landmark positions using optimal rotation and scale. This naturally handles
        face orientation differences since the destination landmarks already reflect
        the target face's pose.

        Args:
            src_2d: Source part 2D landmarks
            dst_2d: Destination part 2D landmarks (target positions)
            part_name: Name of the part for logging

        Returns:
            Aligned 2D points that map source shape to destination position/orientation
        """
        try:
            # Compute centroids
            src_center = np.mean(src_2d, axis=0)
            dst_center = np.mean(dst_2d, axis=0)

            # Center the points
            src_centered = src_2d - src_center
            dst_centered = dst_2d - dst_center

            # Compute optimal rotation using SVD (2D Procrustes)
            # This finds the rotation that best aligns src points to dst points
            H = src_centered.T @ dst_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            # Ensure proper rotation (det = 1, not -1 which would be reflection)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            # Compute scale ratio
            src_scale = np.sqrt(np.sum(src_centered ** 2))
            dst_scale = np.sqrt(np.sum(dst_centered ** 2))
            scale = dst_scale / max(src_scale, 1e-6)

            # Part-specific scale clamping
            if part_name == "nose":
                scale = np.clip(scale, 0.7, 1.4)
            else:
                scale = np.clip(scale, 0.8, 1.25)

            # Apply transformation: rotate, scale, then translate to destination center
            aligned = (src_centered @ R.T) * scale + dst_center

            # Extract rotation angle for logging
            angle_rad = np.arctan2(R[1, 0], R[0, 0])
            logger.info(f"{part_name} alignment (similarity transform): "
                        f"rotation={np.degrees(angle_rad):.1f}°, scale={scale:.3f}")

            return aligned.astype(np.float32)

        except Exception as e:
            logger.warning(f"Similarity alignment failed for {part_name}: {e}, "
                          f"using direct destination landmarks")
            # Fallback: use destination landmarks directly
            return dst_2d.astype(np.float32)

    def _compute_nose_aligned_points(
        self,
        src_2d: np.ndarray,
        dst_2d: np.ndarray,
        src_landmarks: np.ndarray,
        dst_landmarks: np.ndarray,
        part_name: str,
    ) -> np.ndarray:
        """
        Compute aligned 2D points for nose using eye-center and centerline alignment.

        This method:
        1. Positions the nose center at the midpoint between both eyes
        2. Aligns the nose centerline (from bridge to tip) with the target face centerline
        3. Scales the nose appropriately based on inter-eye distance

        Args:
            src_2d: Source nose 2D landmarks
            dst_2d: Destination nose 2D landmarks
            src_landmarks: All source face landmarks
            dst_landmarks: All destination face landmarks
            part_name: Name of the part for logging

        Returns:
            Aligned 2D points for the nose
        """
        try:
            # Get eye center landmarks
            left_eye_inner_idx = EYE_CENTER_LANDMARKS["left_eye_inner"]
            left_eye_outer_idx = EYE_CENTER_LANDMARKS["left_eye_outer"]
            right_eye_inner_idx = EYE_CENTER_LANDMARKS["right_eye_inner"]
            right_eye_outer_idx = EYE_CENTER_LANDMARKS["right_eye_outer"]

            # Compute eye centers for source
            src_left_eye = (
                np.array(src_landmarks[left_eye_inner_idx]) +
                np.array(src_landmarks[left_eye_outer_idx])
            ) / 2
            src_right_eye = (
                np.array(src_landmarks[right_eye_inner_idx]) +
                np.array(src_landmarks[right_eye_outer_idx])
            ) / 2
            src_inter_eye_dist = np.linalg.norm(src_right_eye - src_left_eye)

            # Compute eye centers for destination
            dst_left_eye = (
                np.array(dst_landmarks[left_eye_inner_idx]) +
                np.array(dst_landmarks[left_eye_outer_idx])
            ) / 2
            dst_right_eye = (
                np.array(dst_landmarks[right_eye_inner_idx]) +
                np.array(dst_landmarks[right_eye_outer_idx])
            ) / 2
            dst_eyes_center = (dst_left_eye + dst_right_eye) / 2
            dst_inter_eye_dist = np.linalg.norm(dst_right_eye - dst_left_eye)

            # Get nose centerline landmarks
            nose_tip_idx = NOSE_ALIGNMENT_LANDMARKS["nose_tip"]
            nose_bridge_idx = NOSE_ALIGNMENT_LANDMARKS["nose_bridge_top"]
            left_alar_idx = NOSE_ALIGNMENT_LANDMARKS["left_alar"]
            right_alar_idx = NOSE_ALIGNMENT_LANDMARKS["right_alar"]

            # Compute source nose centerline
            src_nose_tip = np.array(src_landmarks[nose_tip_idx])
            src_nose_bridge = np.array(src_landmarks[nose_bridge_idx])
            src_left_alar = np.array(src_landmarks[left_alar_idx])
            src_right_alar = np.array(src_landmarks[right_alar_idx])

            # Source nose center (midpoint of alars)
            src_nose_center = (src_left_alar + src_right_alar) / 2

            # Source nose centerline vector (from bridge to tip)
            src_centerline = src_nose_tip - src_nose_bridge
            src_centerline_angle = np.arctan2(src_centerline[1], src_centerline[0])

            # Compute destination nose centerline
            dst_nose_tip = np.array(dst_landmarks[nose_tip_idx])
            dst_nose_bridge = np.array(dst_landmarks[nose_bridge_idx])
            dst_left_alar = np.array(dst_landmarks[left_alar_idx])
            dst_right_alar = np.array(dst_landmarks[right_alar_idx])

            # Destination nose center (midpoint of alars)
            dst_nose_center = (dst_left_alar + dst_right_alar) / 2

            # Destination nose centerline vector
            dst_centerline = dst_nose_tip - dst_nose_bridge
            dst_centerline_angle = np.arctan2(dst_centerline[1], dst_centerline[0])

            # Compute rotation angle to align centerlines
            rotation_angle = dst_centerline_angle - src_centerline_angle

            # Compute scale based on inter-eye distance
            scale = dst_inter_eye_dist / max(src_inter_eye_dist, 1e-6)
            scale = np.clip(scale, 0.7, 1.4)

            # Create rotation matrix
            cos_a = np.cos(rotation_angle)
            sin_a = np.sin(rotation_angle)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

            # Target position: position nose center under the eye center
            # Use destination eye center's X coordinate and destination nose center's Y relationship
            dst_vertical_offset = dst_nose_center[1] - dst_eyes_center[1]

            # Target position: X aligned with destination eye center, Y based on destination proportions
            target_center = np.array([
                dst_eyes_center[0],  # X: centered under eyes
                dst_eyes_center[1] + dst_vertical_offset  # Y: proper vertical position
            ])

            # Apply transformation:
            # 1. Center source nose points around source nose center
            src_centered = src_2d - src_nose_center

            # 2. Apply rotation to align centerlines
            src_rotated = (src_centered @ R.T)

            # 3. Apply scale
            src_scaled = src_rotated * scale

            # 4. Translate to target position
            aligned = src_scaled + target_center

            logger.info(f"Nose alignment: rotation={np.degrees(rotation_angle):.1f}°, "
                        f"scale={scale:.3f}, "
                        f"src_eye_dist={src_inter_eye_dist:.1f}, "
                        f"dst_eye_dist={dst_inter_eye_dist:.1f}, "
                        f"target_center=({target_center[0]:.1f}, {target_center[1]:.1f})")

            return aligned.astype(np.float32)

        except (IndexError, KeyError, TypeError) as e:
            logger.warning(f"Nose-specific alignment failed: {e}, "
                          f"falling back to similarity transform")
            return self._compute_similarity_aligned_points(src_2d, dst_2d, part_name)

    def _project_3d_to_2d(
        self,
        points_3d: np.ndarray,
        img_size: Tuple[int, int],
    ) -> np.ndarray:
        """Project normalized 3D points to 2D image coordinates."""
        w, h = img_size

        points_2d = np.zeros((len(points_3d), 2), dtype=np.float32)

        for i, (x, y, _z) in enumerate(points_3d):
            # Simple orthographic projection from normalized coordinates
            # (the perspective effect is already captured in the landmarks)
            px = (x + 1) / 2 * w
            py = (y + 1) / 2 * h
            points_2d[i] = [px, py]

        return points_2d

    def _perspective_warp(
        self,
        img: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """Apply perspective transformation."""
        w, h = output_size

        if len(src_points) >= 4 and len(dst_points) >= 4:
            M = cv2.getPerspectiveTransform(
                src_points[:4].astype(np.float32),
                dst_points[:4].astype(np.float32)
            )
            warped = cv2.warpPerspective(
                img, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
        else:
            warped = img

        return warped

    def _rbf_warp(
        self,
        img: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """RBF-based image warping."""
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

        try:
            rbf_x = RBFInterpolator(
                all_dst, all_src[:, 0],
                kernel='thin_plate_spline', smoothing=1.0
            )
            rbf_y = RBFInterpolator(
                all_dst, all_src[:, 1],
                kernel='thin_plate_spline', smoothing=1.0
            )
        except Exception:
            return cv2.resize(img, (w, h))

        grid_y, grid_x = np.mgrid[0:h, 0:w]
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        src_x = rbf_x(grid_points).reshape(h, w).astype(np.float32)
        src_y = rbf_y(grid_points).reshape(h, w).astype(np.float32)

        warped = cv2.remap(
            img, src_x, src_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        return warped

    def _create_soft_mask(
        self,
        landmarks: np.ndarray,
        part_indices: List[int],
        img_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Create soft-edged mask for blending."""
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        points = []
        for idx in part_indices:
            if idx < len(landmarks):
                x, y = landmarks[idx]
                points.append([int(x), int(y)])

        if len(points) < 3:
            return mask

        points = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

        # Expand and blur for soft edges
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)

        return mask

    def _depth_aware_color_transfer(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        mask: np.ndarray,
        src_depth: np.ndarray,
        tgt_depth: np.ndarray,
    ) -> np.ndarray:
        """Transfer color with depth-aware adjustment."""
        if mask.sum() == 0:
            return src

        result = src.copy()
        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB).astype(np.float32)

        kernel = np.ones((15, 15), np.uint8)
        stat_mask = cv2.dilate(mask, kernel, iterations=2)
        mask_bool = stat_mask > 127

        if not np.any(mask_bool):
            return src

        # Compute depth difference for lighting adjustment
        depth_diff = np.mean(tgt_depth[mask_bool]) - np.mean(src_depth[mask_bool])
        lighting_factor = 1.0 + depth_diff * 0.3

        for i in range(3):
            src_channel = src_lab[:, :, i]
            tgt_channel = tgt_lab[:, :, i]

            src_masked = src_channel[mask_bool]
            tgt_masked = tgt_channel[mask_bool]

            if len(src_masked) == 0 or len(tgt_masked) == 0:
                continue

            src_median = np.median(src_masked)
            src_p25, src_p75 = np.percentile(src_masked, [25, 75])
            src_iqr = max(src_p75 - src_p25, 1.0)

            tgt_median = np.median(tgt_masked)
            tgt_p25, tgt_p75 = np.percentile(tgt_masked, [25, 75])
            tgt_iqr = max(tgt_p75 - tgt_p25, 1.0)

            scale = np.clip(tgt_iqr / src_iqr, 0.5, 2.0)

            if i == 0:  # L channel
                scale *= lighting_factor

            transferred = (src_channel - src_median) * scale + tgt_median

            blend_mask = mask.astype(np.float32) / 255.0
            src_lab[:, :, i] = src_channel * (1 - blend_mask) + transferred * blend_mask

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
        """Multi-band (Laplacian pyramid) blending."""
        if base.shape != src.shape:
            src = cv2.resize(src, (base.shape[1], base.shape[0]))

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
            if m.shape[:2] != base_lap[i].shape[:2]:
                m = cv2.resize(m, (base_lap[i].shape[1], base_lap[i].shape[0]))
            blended = src_lap[i] * m + base_lap[i] * (1 - m)
            blended_lap.append(blended)

        # Reconstruct
        result = blended_lap[-1]
        for i in range(num_levels - 1, -1, -1):
            result = cv2.pyrUp(
                result, dstsize=(blended_lap[i].shape[1], blended_lap[i].shape[0])
            )
            result = result + blended_lap[i]

        return np.clip(result, 0, 255).astype(np.uint8)


# Global instance
_part_blender_3d_service: Optional[PartBlender3D] = None


def get_part_blender_3d_service() -> PartBlender3D:
    """Get or create 3D part blender service instance."""
    global _part_blender_3d_service
    if _part_blender_3d_service is None:
        _part_blender_3d_service = PartBlender3D()
    return _part_blender_3d_service
