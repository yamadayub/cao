"""Face detection service using MediaPipe."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from app.models.schemas import FaceLandmark, FaceRegion, ImageInfo
from app.utils.image import ImageValidationError


@dataclass
class FaceDetectionResult:
    """Result of face detection."""

    face_detected: bool
    face_count: int
    face_region: Optional[FaceRegion]
    landmarks: Optional[List[FaceLandmark]]
    image_info: ImageInfo


class FaceDetectionService:
    """Service for detecting faces using MediaPipe."""

    def __init__(self):
        """Initialize MediaPipe Face Mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,  # Detect up to 5 faces to check for multiple
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(
        self,
        img: np.ndarray,
        image_format: str = "jpeg",
        require_single_face: bool = False,
    ) -> FaceDetectionResult:
        """
        Detect faces in an image.

        Args:
            img: OpenCV image in BGR format
            image_format: Original image format ('jpeg' or 'png')
            require_single_face: If True, raises error for 0 or multiple faces

        Returns:
            FaceDetectionResult with detection results

        Raises:
            ImageValidationError: If require_single_face is True and face count != 1
        """
        h, w = img.shape[:2]
        image_info = ImageInfo(width=w, height=h, format=image_format)

        # Convert BGR to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image
        results = self.face_mesh.process(rgb_img)

        if not results.multi_face_landmarks:
            if require_single_face:
                raise ImageValidationError(
                    code="FACE_NOT_DETECTED",
                    message="No face detected in the uploaded image",
                )
            return FaceDetectionResult(
                face_detected=False,
                face_count=0,
                face_region=None,
                landmarks=None,
                image_info=image_info,
            )

        face_count = len(results.multi_face_landmarks)

        if require_single_face and face_count > 1:
            raise ImageValidationError(
                code="MULTIPLE_FACES",
                message="Multiple faces detected. Please upload an image with a single face",
                details={"face_count": face_count},
            )

        # Get the first face's landmarks
        face_landmarks = results.multi_face_landmarks[0]

        # Convert landmarks to our format
        landmarks = []
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        for idx, landmark in enumerate(face_landmarks.landmark):
            # Clamp values to 0-1 range
            x = max(0.0, min(1.0, landmark.x))
            y = max(0.0, min(1.0, landmark.y))

            landmarks.append(
                FaceLandmark(
                    index=idx,
                    x=x,
                    y=y,
                    z=landmark.z,
                )
            )

            # Track bounding box
            min_x = min(min_x, landmark.x)
            min_y = min(min_y, landmark.y)
            max_x = max(max_x, landmark.x)
            max_y = max(max_y, landmark.y)

        # Calculate face region in pixel coordinates
        face_region = FaceRegion(
            x=max(0, int(min_x * w)),
            y=max(0, int(min_y * h)),
            width=min(w, int((max_x - min_x) * w)),
            height=min(h, int((max_y - min_y) * h)),
        )

        return FaceDetectionResult(
            face_detected=True,
            face_count=face_count,
            face_region=face_region,
            landmarks=landmarks,
            image_info=image_info,
        )

    def get_landmark_points(
        self, img: np.ndarray, image_label: str = "image"
    ) -> Tuple[List[Tuple[float, float]], int, int]:
        """
        Get landmark points as (x, y) tuples in pixel coordinates.

        Args:
            img: OpenCV image in BGR format
            image_label: Label for error messages ('current' or 'ideal')

        Returns:
            Tuple of (landmark points, image width, image height)

        Raises:
            ImageValidationError: If no single face is detected
        """
        h, w = img.shape[:2]
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)

        if not results.multi_face_landmarks:
            raise ImageValidationError(
                code="FACE_NOT_DETECTED",
                message=f"No face detected in {image_label} image",
            )

        if len(results.multi_face_landmarks) > 1:
            raise ImageValidationError(
                code="MULTIPLE_FACES",
                message=f"Multiple faces detected in {image_label} image",
            )

        face_landmarks = results.multi_face_landmarks[0]
        points = [
            (landmark.x * w, landmark.y * h)
            for landmark in face_landmarks.landmark
        ]

        return points, w, h

    def close(self):
        """Release resources."""
        self.face_mesh.close()


# Global instance for reuse
_face_detection_service: Optional[FaceDetectionService] = None


def get_face_detection_service() -> FaceDetectionService:
    """Get or create face detection service instance."""
    global _face_detection_service
    if _face_detection_service is None:
        _face_detection_service = FaceDetectionService()
    return _face_detection_service
