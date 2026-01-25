"""Face morphing service using OpenCV with landmark-based warping."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.services.face_detection import get_face_detection_service

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh landmark indices for face regions
# These define the face oval/boundary
FACE_OVAL_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

# Forehead region (above eyebrows, below hairline)
FOREHEAD_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356,  # Top of face oval
    109, 67, 103, 54, 21, 162, 127, 234,    # Continue around
    # Eyebrow top points
    107, 66, 105, 63, 70,  # Left eyebrow
    336, 296, 334, 293, 300,  # Right eyebrow
]

# Hair/background region landmarks (outside face but inside image)
HAIR_REGION_LANDMARKS = [
    # Top of head (estimated positions above face oval)
    10, 151, 9, 8, 168, 6, 197, 195, 5,
]


def get_triangulation_indices(points: List[Tuple[float, float]], w: int, h: int) -> List[Tuple[int, int, int]]:
    """
    Get Delaunay triangulation indices for a set of points.

    Args:
        points: List of (x, y) landmark points
        w: Image width
        h: Image height

    Returns:
        List of triangles as (idx1, idx2, idx3) tuples
    """
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)

    point_to_idx = {}
    for idx, (x, y) in enumerate(points):
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        try:
            subdiv.insert((x, y))
            point_to_idx[(int(x), int(y))] = idx
        except cv2.error:
            pass

    triangles = subdiv.getTriangleList()
    triangle_indices = []

    for t in triangles:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if all(0 <= pt[0] < w and 0 <= pt[1] < h for pt in [pt1, pt2, pt3]):
            idx1 = point_to_idx.get(pt1)
            idx2 = point_to_idx.get(pt2)
            idx3 = point_to_idx.get(pt3)

            if idx1 is not None and idx2 is not None and idx3 is not None:
                triangle_indices.append((idx1, idx2, idx3))

    return triangle_indices


def warp_triangle(
    src: np.ndarray,
    img_out: np.ndarray,
    t_src: np.ndarray,
    t_dst: np.ndarray,
) -> None:
    """
    Warp a single triangle from source to destination.

    Unlike morph_triangle, this does NOT blend two images - it only warps
    one source image to the destination position.

    Args:
        src: Source image
        img_out: Output image (modified in place)
        t_src: Triangle vertices in source image
        t_dst: Triangle vertices in destination (output) image
    """
    # Find bounding rectangles
    r_src = cv2.boundingRect(np.array([t_src], dtype=np.float32))
    r_dst = cv2.boundingRect(np.array([t_dst], dtype=np.float32))

    # Offset triangles
    t_src_offset = [(t_src[i][0] - r_src[0], t_src[i][1] - r_src[1]) for i in range(3)]
    t_dst_offset = [(t_dst[i][0] - r_dst[0], t_dst[i][1] - r_dst[1]) for i in range(3)]

    # Get source patch
    patch = src[r_src[1]:r_src[1]+r_src[3], r_src[0]:r_src[0]+r_src[2]]

    if patch.size == 0:
        return

    # Warp patch
    size = (r_dst[2], r_dst[3])
    warp_mat = cv2.getAffineTransform(
        np.float32(t_src_offset),
        np.float32(t_dst_offset)
    )
    warped = cv2.warpAffine(
        patch,
        warp_mat,
        size,
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Create mask
    mask = np.zeros((r_dst[3], r_dst[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_dst_offset), (1.0, 1.0, 1.0), 16, 0)

    # Copy to output
    y1, y2 = r_dst[1], r_dst[1] + r_dst[3]
    x1, x2 = r_dst[0], r_dst[0] + r_dst[2]

    if y2 <= img_out.shape[0] and x2 <= img_out.shape[1] and y2 > y1 and x2 > x1:
        img_out[y1:y2, x1:x2] = img_out[y1:y2, x1:x2] * (1 - mask) + warped * mask


class MorphingService:
    """Service for morphing faces using landmark-based warping."""

    def __init__(self):
        """Initialize morphing service."""
        self.face_service = get_face_detection_service()

    def morph(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        progress: float,
        img1_label: str = "current",
        img2_label: str = "ideal",
    ) -> np.ndarray:
        """
        Morph between two face images.

        Key approach:
        1. Compute intermediate landmark positions (linear interpolation)
        2. Warp the CURRENT image to intermediate positions (for hair/background)
        3. Warp the IDEAL image to intermediate positions (for facial features)
        4. Blend based on face region mask
        5. Apply color correction for seamless blending

        Args:
            img1: Current image (BGR format) - hair will come from this
            img2: Ideal image (BGR format) - face features will be warped toward this
            progress: Morphing progress (0.0 = img1, 1.0 = img2)
            img1_label: Label for img1 in error messages
            img2_label: Label for img2 in error messages

        Returns:
            Morphed image (BGR format)
        """
        # Get landmarks for both images
        points1, w1, h1 = self.face_service.get_landmark_points(img1, img1_label)
        points2, w2, h2 = self.face_service.get_landmark_points(img2, img2_label)

        # Resize to common size
        out_w = max(w1, w2)
        out_h = max(h1, h2)

        if w1 != out_w or h1 != out_h:
            img1 = cv2.resize(img1, (out_w, out_h))
            points1 = [(x * out_w / w1, y * out_h / h1) for x, y in points1]

        if w2 != out_w or h2 != out_h:
            img2 = cv2.resize(img2, (out_w, out_h))
            points2 = [(x * out_w / w2, y * out_h / h2) for x, y in points2]

        # Add corner and edge points
        corner_points = [
            (0, 0), (out_w - 1, 0), (out_w - 1, out_h - 1), (0, out_h - 1),
            (out_w // 2, 0), (out_w // 2, out_h - 1),
            (0, out_h // 2), (out_w - 1, out_h // 2),
            # Additional edge points for better hair coverage
            (out_w // 4, 0), (3 * out_w // 4, 0),
            (0, out_h // 4), (out_w - 1, out_h // 4),
        ]
        points1_full = list(points1) + corner_points
        points2_full = list(points2) + corner_points

        # Compute intermediate positions
        points_mid = [
            ((1 - progress) * p1[0] + progress * p2[0],
             (1 - progress) * p1[1] + progress * p2[1])
            for p1, p2 in zip(points1_full, points2_full)
        ]

        # Warp current image (img1) to intermediate positions
        # This gives us the hair and background
        img1_warped = self._warp_image_triangles(
            img1, points1_full, points_mid, (out_w, out_h)
        )

        # Warp ideal image (img2) to intermediate positions
        # This gives us the facial features at the target shape
        img2_warped = self._warp_image_triangles(
            img2, points2_full, points_mid, (out_w, out_h)
        )

        # Create the morphed face mask at intermediate position
        face_mask_mid = self._create_face_mask(points_mid[:len(points1)], (out_h, out_w))

        # Apply color transfer from img1 to img2_warped within face region
        img2_color_matched = self._color_transfer_face(
            img2_warped, img1_warped, face_mask_mid
        )

        # Blend: use img1_warped for hair/background, img2_color_matched for face
        # The blend factor is controlled by progress
        result = self._blend_with_mask(
            img1_warped, img2_color_matched, face_mask_mid, progress
        )

        return result

    def morph_stages(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        stages: List[float],
        img1_label: str = "current",
        img2_label: str = "ideal",
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Generate multiple morphing stages.

        Args:
            img1: First image (BGR format)
            img2: Second image (BGR format)
            stages: List of progress values (0.0 - 1.0)
            img1_label: Label for img1 in error messages
            img2_label: Label for img2 in error messages

        Returns:
            List of (progress, morphed_image) tuples
        """
        # Get landmarks once
        points1, w1, h1 = self.face_service.get_landmark_points(img1, img1_label)
        points2, w2, h2 = self.face_service.get_landmark_points(img2, img2_label)

        # Common size
        out_w = max(w1, w2)
        out_h = max(h1, h2)

        if w1 != out_w or h1 != out_h:
            img1 = cv2.resize(img1, (out_w, out_h))
            points1 = [(x * out_w / w1, y * out_h / h1) for x, y in points1]

        if w2 != out_w or h2 != out_h:
            img2 = cv2.resize(img2, (out_w, out_h))
            points2 = [(x * out_w / w2, y * out_h / h2) for x, y in points2]

        # Add corner and edge points
        corner_points = [
            (0, 0), (out_w - 1, 0), (out_w - 1, out_h - 1), (0, out_h - 1),
            (out_w // 2, 0), (out_w // 2, out_h - 1),
            (0, out_h // 2), (out_w - 1, out_h // 2),
            (out_w // 4, 0), (3 * out_w // 4, 0),
            (0, out_h // 4), (out_w - 1, out_h // 4),
        ]
        points1_full = list(points1) + corner_points
        points2_full = list(points2) + corner_points

        results = []
        for progress in stages:
            logger.info(f"Generating morph stage: {progress:.0%}")

            # Compute intermediate positions
            points_mid = [
                ((1 - progress) * p1[0] + progress * p2[0],
                 (1 - progress) * p1[1] + progress * p2[1])
                for p1, p2 in zip(points1_full, points2_full)
            ]

            # Warp both images
            img1_warped = self._warp_image_triangles(
                img1, points1_full, points_mid, (out_w, out_h)
            )
            img2_warped = self._warp_image_triangles(
                img2, points2_full, points_mid, (out_w, out_h)
            )

            # Face mask at intermediate position
            face_mask_mid = self._create_face_mask(points_mid[:len(points1)], (out_h, out_w))

            # Color transfer and blend
            img2_color_matched = self._color_transfer_face(
                img2_warped, img1_warped, face_mask_mid
            )

            result = self._blend_with_mask(
                img1_warped, img2_color_matched, face_mask_mid, progress
            )

            results.append((progress, result))

        return results

    def _create_face_mask(
        self,
        landmarks: List[Tuple[float, float]],
        img_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Create a soft face mask from landmarks.

        Args:
            landmarks: Face landmarks
            img_shape: (height, width)

        Returns:
            Soft mask (0-255, uint8)
        """
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Get face oval points
        oval_points = []
        for idx in FACE_OVAL_LANDMARKS:
            if idx < len(landmarks):
                x, y = landmarks[idx]
                oval_points.append([int(x), int(y)])

        if len(oval_points) < 3:
            return mask

        # Fill face oval
        oval_points = np.array(oval_points, dtype=np.int32)
        hull = cv2.convexHull(oval_points)
        cv2.fillConvexPoly(mask, hull, 255)

        # Expand slightly and blur for soft edges
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)

        return mask

    def _warp_image_triangles(
        self,
        img: np.ndarray,
        src_points: List[Tuple[float, float]],
        dst_points: List[Tuple[float, float]],
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Warp image using Delaunay triangulation.

        Memory-efficient method that warps triangles individually.

        Args:
            img: Source image
            src_points: Source landmark positions
            dst_points: Destination landmark positions
            output_size: (width, height)

        Returns:
            Warped image
        """
        w, h = output_size

        # Resize image if needed
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))

        # Get triangulation from destination points
        triangles = get_triangulation_indices(dst_points, w, h)

        # Create output image
        result = np.zeros((h, w, 3), dtype=np.float32)

        for tri in triangles:
            i1, i2, i3 = tri

            # Get triangle vertices
            t_src = np.array([
                src_points[i1],
                src_points[i2],
                src_points[i3],
            ], dtype=np.float32)

            t_dst = np.array([
                dst_points[i1],
                dst_points[i2],
                dst_points[i3],
            ], dtype=np.float32)

            # Warp this triangle
            warp_triangle(img, result, t_src, t_dst)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _color_transfer_face(
        self,
        src: np.ndarray,
        ref: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Transfer color from reference to source within masked region.

        Args:
            src: Source image (to be color-corrected)
            ref: Reference image (color reference)
            mask: Face mask

        Returns:
            Color-corrected source image
        """
        if mask.sum() == 0:
            return src

        result = src.copy()

        # Convert to LAB
        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)

        mask_bool = mask > 127

        if not np.any(mask_bool):
            return src

        # Transfer each channel
        for i in range(3):
            src_channel = src_lab[:, :, i]
            ref_channel = ref_lab[:, :, i]

            src_masked = src_channel[mask_bool]
            ref_masked = ref_channel[mask_bool]

            if len(src_masked) == 0 or len(ref_masked) == 0:
                continue

            # Robust statistics
            src_median = np.median(src_masked)
            src_p25, src_p75 = np.percentile(src_masked, [25, 75])
            src_iqr = max(src_p75 - src_p25, 1.0)

            ref_median = np.median(ref_masked)
            ref_p25, ref_p75 = np.percentile(ref_masked, [25, 75])
            ref_iqr = max(ref_p75 - ref_p25, 1.0)

            # Scale factor
            scale = np.clip(ref_iqr / src_iqr, 0.5, 2.0)

            # Transfer
            transferred = (src_channel - src_median) * scale + ref_median

            # Apply with mask blending
            blend_mask = mask.astype(np.float32) / 255.0
            src_lab[:, :, i] = src_channel * (1 - blend_mask) + transferred * blend_mask

        src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)

        return result

    def _blend_with_mask(
        self,
        img_bg: np.ndarray,
        img_face: np.ndarray,
        face_mask: np.ndarray,
        progress: float,
    ) -> np.ndarray:
        """
        Blend background and face images using mask.

        For hair/background: always use img_bg (current image)
        For face region: blend img_bg and img_face based on progress

        Args:
            img_bg: Background/hair image (current image warped)
            img_face: Face image (ideal image warped and color-matched)
            face_mask: Face region mask
            progress: Morphing progress (0=img_bg, 1=img_face)

        Returns:
            Blended result
        """
        # Normalize mask
        mask_float = face_mask.astype(np.float32) / 255.0
        mask_3ch = np.dstack([mask_float, mask_float, mask_float])

        # Convert to float
        bg_float = img_bg.astype(np.float32)
        face_float = img_face.astype(np.float32)

        # Compute face blend based on progress
        # At progress=0: 100% current (img_bg)
        # At progress=1: 100% ideal (img_face)
        face_blend = (1 - progress) * bg_float + progress * face_float

        # Final blend: face region from face_blend, background from img_bg
        result = bg_float * (1 - mask_3ch) + face_blend * mask_3ch

        # Multi-band blending for smooth transitions
        result = self._multiband_blend(img_bg, result.astype(np.uint8), face_mask)

        return result

    def _multiband_blend(
        self,
        base: np.ndarray,
        overlay: np.ndarray,
        mask: np.ndarray,
        num_levels: int = 4,
    ) -> np.ndarray:
        """
        Multi-band (Laplacian pyramid) blending for seamless transitions.
        """
        if base.shape != overlay.shape:
            overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]))

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

        # Build Laplacian pyramid for overlay
        overlay_lap = []
        current = overlay.astype(np.float32)
        for _ in range(num_levels):
            down = cv2.pyrDown(current)
            up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
            lap = current - up
            overlay_lap.append(lap)
            current = down
        overlay_lap.append(current)

        # Blend pyramids
        blended_lap = []
        for i in range(num_levels + 1):
            m = mask_pyramid[min(i, len(mask_pyramid) - 1)]
            if m.shape[:2] != base_lap[i].shape[:2]:
                m = cv2.resize(m, (base_lap[i].shape[1], base_lap[i].shape[0]))
            blended = overlay_lap[i] * m + base_lap[i] * (1 - m)
            blended_lap.append(blended)

        # Reconstruct
        result = blended_lap[-1]
        for i in range(num_levels - 1, -1, -1):
            result = cv2.pyrUp(result, dstsize=(blended_lap[i].shape[1], blended_lap[i].shape[0]))
            result = result + blended_lap[i]

        return np.clip(result, 0, 255).astype(np.uint8)


# Global instance
_morphing_service: Optional[MorphingService] = None


def get_morphing_service() -> MorphingService:
    """Get or create morphing service instance."""
    global _morphing_service
    if _morphing_service is None:
        _morphing_service = MorphingService()
    return _morphing_service
