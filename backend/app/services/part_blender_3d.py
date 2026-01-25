"""3D-based part blending service using depth estimation."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

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

    def _construct_3d_vertices(self) -> np.ndarray:
        """Construct 3D vertices from 2D landmarks and depth map."""
        w, h = self.img_size
        n_landmarks = len(self.landmarks_2d)
        vertices = np.zeros((n_landmarks, 3), dtype=np.float32)

        for i, (x, y) in enumerate(self.landmarks_2d):
            # Clamp coordinates to image bounds
            ix = int(np.clip(x, 0, w - 1))
            iy = int(np.clip(y, 0, h - 1))

            # Get depth value at landmark position
            z = self.depth_map[iy, ix]

            # Normalize coordinates to [-1, 1] range
            nx = (x / w) * 2 - 1
            ny = (y / h) * 2 - 1
            nz = z  # Depth is already normalized

            vertices[i] = [nx, ny, nz]

        return vertices

    def _create_triangulation(self) -> np.ndarray:
        """Create Delaunay triangulation of landmarks."""
        try:
            tri = Delaunay(self.landmarks_2d)
            return tri.simplices
        except Exception:
            # Fallback: create simple grid triangulation
            return np.array([], dtype=np.int32)

    def get_part_vertices(self, part_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Get 3D vertices and 2D coordinates for a facial part."""
        indices = [i for i in part_indices if i < len(self.vertices_3d)]
        vertices_3d = self.vertices_3d[indices]
        vertices_2d = self.landmarks_2d[indices]
        return vertices_3d, vertices_2d


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

        Args:
            current_img: Current face image (BGR format) - base image
            ideal_img: Ideal face image (BGR format) - source of parts
            parts: Selection of which parts to blend
            current_label: Label for current image in error messages
            ideal_label: Label for ideal image in error messages

        Returns:
            Blended image (BGR format)
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

        Args:
            img: BGR image

        Returns:
            Normalized depth map (H, W) with values in [0, 1]
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
        Blend a single facial part using 3D alignment.

        The process:
        1. Get 3D vertices for the part in both meshes
        2. Compute 3D transformation (rotation, translation, scale) to align
        3. Transform source texture according to 3D alignment
        4. Project back to 2D and blend
        """
        w, h = img_size

        # Get 3D and 2D vertices for the part
        src_3d, src_2d = src_mesh.get_part_vertices(part_indices)
        dst_3d, dst_2d = base_mesh.get_part_vertices(part_indices)

        if len(src_3d) < 4 or len(dst_3d) < 4:
            return base_img

        # Compute transformation to align source to destination in 3D
        # Using Procrustes analysis for optimal rotation and scale
        transform_3d = self._compute_3d_alignment(src_3d, dst_3d)

        # Transform source vertices to destination space
        aligned_src_3d = self._apply_3d_transform(src_3d, transform_3d)

        # Project aligned 3D vertices back to 2D
        aligned_src_2d = self._project_to_2d(aligned_src_3d, img_size)

        # Include context for warping
        all_indices = list(set(part_indices + context_indices))

        # Get all source points for RBF warping
        src_points = []
        dst_points = []
        for idx in all_indices:
            if idx < len(src_mesh.landmarks_2d) and idx < len(base_mesh.landmarks_2d):
                # For part vertices, use aligned positions
                if idx in part_indices:
                    local_idx = part_indices.index(idx)
                    if local_idx < len(aligned_src_2d):
                        src_points.append(aligned_src_2d[local_idx])
                    else:
                        src_points.append(src_mesh.landmarks_2d[idx])
                else:
                    src_points.append(src_mesh.landmarks_2d[idx])
                dst_points.append(base_mesh.landmarks_2d[idx])

        if len(src_points) < 4:
            return base_img

        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # Pre-warp source image based on 3D alignment
        warped_src = self._warp_with_3d_alignment(
            src_mesh.texture,
            src_mesh.landmarks_2d,
            aligned_src_2d,
            part_indices,
            img_size,
        )

        # Fine-tune with RBF warping
        warped_src = self._rbf_warp(warped_src, src_points, dst_points, img_size)

        # Create soft mask
        mask = self._create_soft_mask(base_mesh.landmarks_2d, part_indices, (h, w))

        if mask.sum() == 0:
            return base_img

        # Apply color correction
        warped_src = self._color_transfer(warped_src, base_img, mask)

        # Multi-band blend
        result = self._multiband_blend(base_img, warped_src, mask)

        return result

    def _compute_3d_alignment(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> Dict:
        """
        Compute 3D alignment transformation using Procrustes analysis.

        Returns:
            Dictionary with rotation, scale, and translation
        """
        # Center points
        src_centroid = np.mean(src_points, axis=0)
        dst_centroid = np.mean(dst_points, axis=0)

        src_centered = src_points - src_centroid
        dst_centered = dst_points - dst_centroid

        # Compute scale
        src_scale = np.sqrt(np.sum(src_centered ** 2) / len(src_centered))
        dst_scale = np.sqrt(np.sum(dst_centered ** 2) / len(dst_centered))

        scale = dst_scale / max(src_scale, 1e-6)
        scale = np.clip(scale, 0.5, 2.0)  # Limit scale range

        # Normalize
        src_norm = src_centered / max(src_scale, 1e-6)
        dst_norm = dst_centered / max(dst_scale, 1e-6)

        # Compute rotation using SVD (Kabsch algorithm)
        H = src_norm.T @ dst_norm
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        return {
            'rotation': R,
            'scale': scale,
            'src_centroid': src_centroid,
            'dst_centroid': dst_centroid,
        }

    def _apply_3d_transform(
        self,
        points: np.ndarray,
        transform: Dict,
    ) -> np.ndarray:
        """Apply 3D transformation to points."""
        R = transform['rotation']
        scale = transform['scale']
        src_centroid = transform['src_centroid']
        dst_centroid = transform['dst_centroid']

        # Center, scale, rotate, translate
        centered = points - src_centroid
        scaled = centered * scale
        rotated = scaled @ R.T
        translated = rotated + dst_centroid

        return translated

    def _project_to_2d(
        self,
        points_3d: np.ndarray,
        img_size: Tuple[int, int],
    ) -> np.ndarray:
        """Project 3D points back to 2D image coordinates."""
        w, h = img_size

        # Convert from normalized [-1, 1] to pixel coordinates
        points_2d = np.zeros((len(points_3d), 2), dtype=np.float32)
        points_2d[:, 0] = (points_3d[:, 0] + 1) * w / 2
        points_2d[:, 1] = (points_3d[:, 1] + 1) * h / 2

        return points_2d

    def _warp_with_3d_alignment(
        self,
        img: np.ndarray,
        src_landmarks: np.ndarray,
        aligned_landmarks: np.ndarray,
        part_indices: List[int],
        img_size: Tuple[int, int],
    ) -> np.ndarray:
        """Warp image based on 3D-aligned landmarks."""
        w, h = img_size

        # Compute affine transformation from original to aligned landmarks
        src_pts = []
        dst_pts = []

        for idx in part_indices:
            if idx < len(src_landmarks) and idx < len(aligned_landmarks):
                src_pts.append(src_landmarks[idx])

        for i, idx in enumerate(part_indices):
            if idx < len(aligned_landmarks):
                dst_pts.append(aligned_landmarks[i])

        if len(src_pts) < 3 or len(dst_pts) < 3:
            return img

        src_pts = np.array(src_pts[:3], dtype=np.float32)
        dst_pts = np.array(dst_pts[:3], dtype=np.float32)

        # Compute affine transform
        M = cv2.getAffineTransform(src_pts, dst_pts)

        # Apply transformation
        warped = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

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

    def _color_transfer(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Transfer color from target to source in masked region."""
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
