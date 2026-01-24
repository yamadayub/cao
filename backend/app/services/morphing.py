"""Face morphing service using OpenCV."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.services.face_detection import get_face_detection_service


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
    # Create subdivision
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)

    # Add points to subdivision (clamped to image bounds)
    point_to_idx = {}
    for idx, (x, y) in enumerate(points):
        # Clamp points to image boundaries
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        try:
            subdiv.insert((x, y))
            point_to_idx[(int(x), int(y))] = idx
        except cv2.error:
            # Skip points that cause issues
            pass

    # Get triangles
    triangles = subdiv.getTriangleList()
    triangle_indices = []

    for t in triangles:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        # Check if triangle is inside image and we have indices for all points
        if all(0 <= pt[0] < w and 0 <= pt[1] < h for pt in [pt1, pt2, pt3]):
            # Find corresponding indices
            idx1 = point_to_idx.get(pt1)
            idx2 = point_to_idx.get(pt2)
            idx3 = point_to_idx.get(pt3)

            if idx1 is not None and idx2 is not None and idx3 is not None:
                triangle_indices.append((idx1, idx2, idx3))

    return triangle_indices


def apply_affine_transform(
    src: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray,
    size: Tuple[int, int],
) -> np.ndarray:
    """
    Apply affine transformation to warp a triangular region.

    Args:
        src: Source image
        src_tri: Source triangle vertices
        dst_tri: Destination triangle vertices
        size: Output size (width, height)

    Returns:
        Warped image patch
    """
    # Find bounding rectangle for destination triangle
    r_dst = cv2.boundingRect(np.array([dst_tri], dtype=np.float32))
    x, y, w, h = r_dst

    # Offset destination triangle
    dst_tri_offset = dst_tri - np.array([x, y], dtype=np.float32)

    # Find affine transform
    warp_mat = cv2.getAffineTransform(
        np.float32(src_tri),
        np.float32(dst_tri_offset)
    )

    # Apply affine transform
    dst = cv2.warpAffine(
        src,
        warp_mat,
        (w, h),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return dst


def morph_triangle(
    img1: np.ndarray,
    img2: np.ndarray,
    img_out: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    t_out: np.ndarray,
    alpha: float,
) -> None:
    """
    Morph a single triangle from two images.

    Args:
        img1: First source image
        img2: Second source image
        img_out: Output image (modified in place)
        t1: Triangle vertices in img1
        t2: Triangle vertices in img2
        t_out: Triangle vertices in output image
        alpha: Blend factor (0.0 = img1, 1.0 = img2)
    """
    # Find bounding rectangles
    r1 = cv2.boundingRect(np.array([t1], dtype=np.float32))
    r2 = cv2.boundingRect(np.array([t2], dtype=np.float32))
    r_out = cv2.boundingRect(np.array([t_out], dtype=np.float32))

    # Offset triangles
    t1_offset = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2_offset = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]
    t_out_offset = [(t_out[i][0] - r_out[0], t_out[i][1] - r_out[1]) for i in range(3)]

    # Get patches
    patch1 = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    patch2 = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

    if patch1.size == 0 or patch2.size == 0:
        return

    # Warp patches
    size = (r_out[2], r_out[3])

    warp1 = apply_affine_transform(patch1, np.float32(t1_offset), np.float32(t_out_offset), size)
    warp2 = apply_affine_transform(patch2, np.float32(t2_offset), np.float32(t_out_offset), size)

    # Blend warped patches
    img_patch = (1 - alpha) * warp1 + alpha * warp2

    # Create mask
    mask = np.zeros((r_out[3], r_out[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_out_offset), (1.0, 1.0, 1.0), 16, 0)

    # Copy to output
    y1, y2 = r_out[1], r_out[1] + r_out[3]
    x1, x2 = r_out[0], r_out[0] + r_out[2]

    if y2 <= img_out.shape[0] and x2 <= img_out.shape[1] and y2 > y1 and x2 > x1:
        img_out[y1:y2, x1:x2] = img_out[y1:y2, x1:x2] * (1 - mask) + img_patch * mask


class MorphingService:
    """Service for morphing faces between two images."""

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

        Args:
            img1: First image (BGR format)
            img2: Second image (BGR format)
            progress: Morphing progress (0.0 = img1, 1.0 = img2)
            img1_label: Label for img1 in error messages
            img2_label: Label for img2 in error messages

        Returns:
            Morphed image (BGR format)

        Raises:
            ImageValidationError: If face detection fails
        """
        # Get landmarks for both images
        points1, w1, h1 = self.face_service.get_landmark_points(img1, img1_label)
        points2, w2, h2 = self.face_service.get_landmark_points(img2, img2_label)

        # Resize images to common size
        out_w = max(w1, w2)
        out_h = max(h1, h2)

        if w1 != out_w or h1 != out_h:
            img1 = cv2.resize(img1, (out_w, out_h))
            points1 = [(x * out_w / w1, y * out_h / h1) for x, y in points1]

        if w2 != out_w or h2 != out_h:
            img2 = cv2.resize(img2, (out_w, out_h))
            points2 = [(x * out_w / w2, y * out_h / h2) for x, y in points2]

        # Add corner points for better coverage
        corner_points = [
            (0, 0),
            (out_w - 1, 0),
            (out_w - 1, out_h - 1),
            (0, out_h - 1),
            (out_w // 2, 0),
            (out_w // 2, out_h - 1),
            (0, out_h // 2),
            (out_w - 1, out_h // 2),
        ]
        points1 = points1 + corner_points
        points2 = points2 + corner_points

        # Calculate intermediate points
        points_out = [
            ((1 - progress) * p1[0] + progress * p2[0],
             (1 - progress) * p1[1] + progress * p2[1])
            for p1, p2 in zip(points1, points2)
        ]

        # Get triangulation
        triangles = get_triangulation_indices(points_out, out_w, out_h)

        # Create output image
        img_out = np.zeros((out_h, out_w, 3), dtype=np.float32)
        img1_float = img1.astype(np.float32)
        img2_float = img2.astype(np.float32)

        # Morph each triangle
        for t_idx in triangles:
            try:
                t1 = np.array([points1[i] for i in t_idx], dtype=np.float32)
                t2 = np.array([points2[i] for i in t_idx], dtype=np.float32)
                t_out = np.array([points_out[i] for i in t_idx], dtype=np.float32)

                morph_triangle(img1_float, img2_float, img_out, t1, t2, t_out, progress)
            except (IndexError, cv2.error):
                # Skip problematic triangles
                continue

        return np.uint8(np.clip(img_out, 0, 255))

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
        # Pre-compute landmarks once
        points1, w1, h1 = self.face_service.get_landmark_points(img1, img1_label)
        points2, w2, h2 = self.face_service.get_landmark_points(img2, img2_label)

        # Common size
        out_w = max(w1, w2)
        out_h = max(h1, h2)

        # Resize if needed
        if w1 != out_w or h1 != out_h:
            img1 = cv2.resize(img1, (out_w, out_h))
            points1 = [(x * out_w / w1, y * out_h / h1) for x, y in points1]

        if w2 != out_w or h2 != out_h:
            img2 = cv2.resize(img2, (out_w, out_h))
            points2 = [(x * out_w / w2, y * out_h / h2) for x, y in points2]

        # Add corner points
        corner_points = [
            (0, 0),
            (out_w - 1, 0),
            (out_w - 1, out_h - 1),
            (0, out_h - 1),
            (out_w // 2, 0),
            (out_w // 2, out_h - 1),
            (0, out_h // 2),
            (out_w - 1, out_h // 2),
        ]
        points1 = points1 + corner_points
        points2 = points2 + corner_points

        img1_float = img1.astype(np.float32)
        img2_float = img2.astype(np.float32)

        results = []
        for progress in stages:
            # Calculate intermediate points
            points_out = [
                ((1 - progress) * p1[0] + progress * p2[0],
                 (1 - progress) * p1[1] + progress * p2[1])
                for p1, p2 in zip(points1, points2)
            ]

            # Get triangulation
            triangles = get_triangulation_indices(points_out, out_w, out_h)

            # Create output
            img_out = np.zeros((out_h, out_w, 3), dtype=np.float32)

            # Morph triangles
            for t_idx in triangles:
                try:
                    t1 = np.array([points1[i] for i in t_idx], dtype=np.float32)
                    t2 = np.array([points2[i] for i in t_idx], dtype=np.float32)
                    t_out = np.array([points_out[i] for i in t_idx], dtype=np.float32)

                    morph_triangle(img1_float, img2_float, img_out, t1, t2, t_out, progress)
                except (IndexError, cv2.error):
                    continue

            results.append((progress, np.uint8(np.clip(img_out, 0, 255))))

        return results


# Global instance
_morphing_service: Optional[MorphingService] = None


def get_morphing_service() -> MorphingService:
    """Get or create morphing service instance."""
    global _morphing_service
    if _morphing_service is None:
        _morphing_service = MorphingService()
    return _morphing_service
