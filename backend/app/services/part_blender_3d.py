"""3D-based part blending service using depth estimation."""

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


class FaceMesh3D:
    """3D face mesh constructed from 2D landmarks and depth map."""

    def __init__(
        self,
        landmarks_2d: np.ndarray,
        depth_map: np.ndarray,
        texture: np.ndarray,
        img_size: Tuple[int, int],
    ):
        """
        Initialize 3D face mesh.

        Args:
            landmarks_2d: (N, 2) array of 2D landmark coordinates
            depth_map: (H, W) depth map
            texture: (H, W, 3) BGR texture image
            img_size: (width, height)
        """
        self.landmarks_2d = landmarks_2d
        self.depth_map = depth_map
        self.texture = texture
        self.img_size = img_size

        # Construct 3D vertices from 2D landmarks + depth
        self.vertices_3d = self._construct_3d_vertices()

        # Create triangulation for mesh
        self.triangles = self._create_triangulation()

        # Estimate face pose from 3D vertices
        self.face_normal, self.face_center = self._estimate_face_pose()

    def _construct_3d_vertices(self) -> np.ndarray:
        """
        Construct 3D vertices from 2D landmarks and depth map.

        Uses pinhole camera model with focal length estimation.
        """
        w, h = self.img_size
        n_landmarks = len(self.landmarks_2d)
        vertices = np.zeros((n_landmarks, 3), dtype=np.float32)

        # Estimate focal length (typical for face images)
        focal_length = max(w, h)
        cx, cy = w / 2, h / 2

        for i, (x, y) in enumerate(self.landmarks_2d):
            # Clamp coordinates to image bounds
            ix = int(np.clip(x, 0, w - 1))
            iy = int(np.clip(y, 0, h - 1))

            # Get depth value at landmark position
            # Use area sampling for more stable depth
            kernel_size = 3
            x_start = max(0, ix - kernel_size)
            x_end = min(w, ix + kernel_size + 1)
            y_start = max(0, iy - kernel_size)
            y_end = min(h, iy + kernel_size + 1)
            z = np.median(self.depth_map[y_start:y_end, x_start:x_end])

            # Convert depth to actual Z (depth map is inverted - closer = higher value)
            # Scale to reasonable range (face depth ~40-60cm)
            z_scaled = (1 - z) * 0.5 + 0.3  # Range [0.3, 0.8] meters

            # Back-project to 3D using pinhole camera model
            X = (x - cx) * z_scaled / focal_length
            Y = (y - cy) * z_scaled / focal_length
            Z = z_scaled

            vertices[i] = [X, Y, Z]

        return vertices

    def _create_triangulation(self) -> np.ndarray:
        """Create Delaunay triangulation of landmarks."""
        try:
            tri = Delaunay(self.landmarks_2d)
            return tri.simplices
        except Exception:
            return np.array([], dtype=np.int32)

    def _estimate_face_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate face normal and center from 3D vertices.

        Uses key facial landmarks to compute face plane.
        """
        # Key landmarks for pose estimation
        # Left eye outer, right eye outer, nose tip, chin
        key_indices = [33, 263, 4, 152]

        key_points = []
        for idx in key_indices:
            if idx < len(self.vertices_3d):
                key_points.append(self.vertices_3d[idx])

        if len(key_points) < 3:
            return np.array([0, 0, 1], dtype=np.float32), np.zeros(3, dtype=np.float32)

        key_points = np.array(key_points, dtype=np.float32)

        # Compute face center
        center = np.mean(key_points, axis=0)

        # Compute face normal using cross product of two face vectors
        # Vector from left eye to right eye
        if len(key_points) >= 2:
            v1 = key_points[1] - key_points[0]  # Right eye - Left eye
        else:
            v1 = np.array([1, 0, 0])

        # Vector from eyes to nose
        if len(key_points) >= 3:
            eye_center = (key_points[0] + key_points[1]) / 2
            v2 = key_points[2] - eye_center  # Nose - Eye center
        else:
            v2 = np.array([0, 1, 0])

        # Normal is cross product (points outward from face)
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        else:
            normal = np.array([0, 0, 1], dtype=np.float32)

        # Ensure normal points toward camera (positive Z)
        if normal[2] < 0:
            normal = -normal

        return normal, center

    def get_part_vertices(self, part_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Get 3D vertices and 2D coordinates for a facial part."""
        indices = [i for i in part_indices if i < len(self.vertices_3d)]
        vertices_3d = self.vertices_3d[indices]
        vertices_2d = self.landmarks_2d[indices]
        return vertices_3d, vertices_2d

    def get_yaw_pitch_roll(self) -> Tuple[float, float, float]:
        """
        Get face rotation angles from face normal.

        Returns:
            yaw: Left-right rotation (radians)
            pitch: Up-down rotation (radians)
            roll: Computed from eye line (radians)
        """
        nx, ny, nz = self.face_normal

        # Yaw (left-right): angle between normal projection on XZ plane and Z axis
        yaw = np.arctan2(nx, nz)

        # Pitch (up-down): angle between normal and XZ plane
        pitch = np.arcsin(np.clip(-ny, -1, 1))

        # Roll: compute from eye landmarks
        left_eye_idx = 33
        right_eye_idx = 263
        if left_eye_idx < len(self.landmarks_2d) and right_eye_idx < len(self.landmarks_2d):
            eye_vec = self.landmarks_2d[right_eye_idx] - self.landmarks_2d[left_eye_idx]
            roll = np.arctan2(eye_vec[1], eye_vec[0])
        else:
            roll = 0.0

        return float(yaw), float(pitch), float(roll)


class PartBlender3D:
    """3D-based facial part blending service."""

    def __init__(self):
        """Initialize 3D part blender service."""
        self.face_service = get_face_detection_service()
        self._depth_available = None

    def is_depth_available(self) -> bool:
        """Check if depth estimation is available."""
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
        Blend selected facial parts using 3D reconstruction.
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

        # Estimate depth for both images
        current_depth = self._estimate_depth(current_img)
        ideal_depth = self._estimate_depth(ideal_img)

        if current_depth is None or ideal_depth is None:
            logger.warning("Depth estimation failed, falling back to 2D method")
            from app.services.part_blender import get_part_blender_service
            return get_part_blender_service().blend(
                current_img, ideal_img, parts, current_label, ideal_label
            )

        # Create 3D face meshes
        current_mesh = FaceMesh3D(
            np.array(current_landmarks, dtype=np.float32),
            current_depth,
            current_img,
            (out_w, out_h)
        )
        ideal_mesh = FaceMesh3D(
            np.array(ideal_landmarks, dtype=np.float32),
            ideal_depth,
            ideal_img,
            (out_w, out_h)
        )

        # Log pose information for debugging
        src_yaw, src_pitch, src_roll = ideal_mesh.get_yaw_pitch_roll()
        dst_yaw, dst_pitch, dst_roll = current_mesh.get_yaw_pitch_roll()
        logger.info(f"Source face pose: yaw={np.degrees(src_yaw):.1f}°, "
                   f"pitch={np.degrees(src_pitch):.1f}°, roll={np.degrees(src_roll):.1f}°")
        logger.info(f"Target face pose: yaw={np.degrees(dst_yaw):.1f}°, "
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
                continue

        return result

    def _estimate_depth(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth map from image using Depth Anything.
        """
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
        Blend a single facial part using 3D-aware transformation.

        Key improvements:
        1. Compute view-normalized transformation
        2. Apply depth-based parallax correction
        3. Use 3D rotation to handle different face orientations
        """
        w, h = img_size

        # Get face poses
        src_yaw, src_pitch, src_roll = src_mesh.get_yaw_pitch_roll()
        dst_yaw, dst_pitch, dst_roll = base_mesh.get_yaw_pitch_roll()

        # Compute pose difference
        yaw_diff = dst_yaw - src_yaw
        pitch_diff = dst_pitch - src_pitch
        roll_diff = dst_roll - src_roll

        logger.info(f"Pose diff for {part_name}: yaw={np.degrees(yaw_diff):.1f}°, "
                   f"pitch={np.degrees(pitch_diff):.1f}°, roll={np.degrees(roll_diff):.1f}°")

        # Get 3D and 2D vertices for the part
        src_3d, src_2d = src_mesh.get_part_vertices(part_indices)
        dst_3d, dst_2d = base_mesh.get_part_vertices(part_indices)

        if len(src_3d) < 4 or len(dst_3d) < 4:
            return base_img

        # Compute part centroids in 3D
        src_center_3d = np.mean(src_3d, axis=0)
        dst_center_3d = np.mean(dst_3d, axis=0)

        # Build 3D rotation matrix to align source pose to target pose
        R = self._build_rotation_matrix(yaw_diff, pitch_diff, roll_diff)

        # Transform source 3D points to match target pose
        src_3d_centered = src_3d - src_center_3d
        src_3d_rotated = (R @ src_3d_centered.T).T

        # Compute scale from 3D point spread
        src_spread = np.std(src_3d_centered)
        dst_spread = np.std(dst_3d - dst_center_3d)
        scale_3d = dst_spread / max(src_spread, 1e-6)
        scale_3d = np.clip(scale_3d, 0.7, 1.4)

        # Apply scale and translate to target position
        src_3d_transformed = src_3d_rotated * scale_3d + dst_center_3d

        # Project transformed 3D points back to 2D
        aligned_src_2d = self._project_3d_to_2d(
            src_3d_transformed,
            base_mesh.img_size,
        )

        # Compute 2D transformation for the entire source image
        src_points = []
        dst_points = []

        for i, _idx in enumerate(part_indices):
            if i < len(aligned_src_2d):
                src_points.append(src_2d[i])
                dst_points.append(aligned_src_2d[i])

        # Add context points (use original positions with offset)
        for idx in context_indices:
            if idx < len(src_mesh.landmarks_2d) and idx < len(base_mesh.landmarks_2d):
                src_points.append(src_mesh.landmarks_2d[idx])
                dst_points.append(base_mesh.landmarks_2d[idx])

        if len(src_points) < 4:
            return base_img

        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # Apply perspective-aware warping for significant pose differences
        if abs(yaw_diff) > 0.05 or abs(pitch_diff) > 0.05:  # > ~3 degrees
            warped_src = self._perspective_warp(
                src_mesh.texture,
                src_points[:4] if len(src_points) >= 4 else src_points,
                dst_points[:4] if len(dst_points) >= 4 else dst_points,
                img_size,
            )
        else:
            warped_src = src_mesh.texture

        # Fine-tune with RBF warping
        warped_src = self._rbf_warp(warped_src, src_points, dst_points, img_size)

        # Create mask using target positions
        mask = self._create_soft_mask(base_mesh.landmarks_2d, part_indices, (h, w))

        if mask.sum() == 0:
            return base_img

        # Apply depth-aware color correction
        warped_src = self._depth_aware_color_transfer(
            warped_src, base_img, mask,
            src_mesh.depth_map, base_mesh.depth_map
        )

        # Multi-band blend with extra levels for better transition
        num_levels = 5 if part_name == "nose" else 4
        result = self._multiband_blend(base_img, warped_src, mask, num_levels)

        return result

    def _build_rotation_matrix(
        self,
        yaw: float,
        pitch: float,
        roll: float,
    ) -> np.ndarray:
        """Build 3D rotation matrix from Euler angles."""
        # Rotation around Y axis (yaw)
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ], dtype=np.float32)

        # Rotation around X axis (pitch)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ], dtype=np.float32)

        # Rotation around Z axis (roll)
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Combined rotation: R = Rz @ Ry @ Rx
        return Rz @ Ry @ Rx

    def _project_3d_to_2d(
        self,
        points_3d: np.ndarray,
        img_size: Tuple[int, int],
    ) -> np.ndarray:
        """Project 3D points to 2D using pinhole camera model."""
        w, h = img_size
        focal_length = max(w, h)
        cx, cy = w / 2, h / 2

        points_2d = np.zeros((len(points_3d), 2), dtype=np.float32)

        for i, (X, Y, Z) in enumerate(points_3d):
            # Avoid division by zero
            Z = max(Z, 0.1)

            # Project using pinhole model
            x = (X * focal_length / Z) + cx
            y = (Y * focal_length / Z) + cy

            points_2d[i] = [x, y]

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
            # Compute perspective transform
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
            rbf_x = RBFInterpolator(all_dst, all_src[:, 0], kernel='thin_plate_spline', smoothing=1.0)
            rbf_y = RBFInterpolator(all_dst, all_src[:, 1], kernel='thin_plate_spline', smoothing=1.0)
        except Exception:
            return cv2.resize(img, (w, h))

        grid_y, grid_x = np.mgrid[0:h, 0:w]
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        src_x = rbf_x(grid_points).reshape(h, w).astype(np.float32)
        src_y = rbf_y(grid_points).reshape(h, w).astype(np.float32)

        warped = cv2.remap(img, src_x, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

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
        """
        Transfer color with depth-aware adjustment.

        Accounts for lighting differences due to surface orientation.
        """
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
        lighting_factor = 1.0 + depth_diff * 0.3  # Subtle adjustment

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

            # Apply lighting factor to luminance channel
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
            result = cv2.pyrUp(result, dstsize=(blended_lap[i].shape[1], blended_lap[i].shape[0]))
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
