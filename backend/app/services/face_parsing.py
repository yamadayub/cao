"""
BiSeNet-based face parsing service for semantic segmentation.

Provides accurate pixel-level segmentation of facial parts:
- Skin, eyebrows, eyes, nose, lips, hair, etc.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# BiSeNet face parsing class indices
# Based on CelebAMask-HQ dataset labels
FACE_PARSING_LABELS = {
    0: "background",
    1: "skin",
    2: "left_eyebrow",
    3: "right_eyebrow",
    4: "left_eye",
    5: "right_eye",
    6: "eyeglasses",  # Not commonly present
    7: "left_ear",
    8: "right_ear",
    9: "earrings",
    10: "nose",
    11: "mouth",  # Inner mouth
    12: "upper_lip",
    13: "lower_lip",
    14: "neck",
    15: "necklace",
    16: "cloth",
    17: "hair",
    18: "hat",
}

# Mapping from our part names to BiSeNet label indices
PART_TO_BISENET_LABELS: Dict[str, List[int]] = {
    "left_eye": [4],  # left_eye
    "right_eye": [5],  # right_eye
    "left_eyebrow": [2],  # left_eyebrow
    "right_eyebrow": [3],  # right_eyebrow
    "nose": [10],  # nose
    "lips": [11, 12, 13],  # mouth, upper_lip, lower_lip
}


class FaceParsingService:
    """Service for semantic face segmentation using BiSeNet."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize face parsing service.

        Args:
            model_path: Path to BiSeNet ONNX model. If None, uses default path.
        """
        self._model = None
        self._model_path = model_path
        self._input_size = (512, 512)  # BiSeNet input size

    def _load_model(self):
        """Lazy load the BiSeNet model."""
        if self._model is not None:
            return

        # Try to load ONNX model first (faster, no torch dependency)
        model_path = self._model_path
        if model_path is None:
            # Default model path
            model_dir = Path(__file__).parent.parent.parent / "models"
            model_path = model_dir / "face_parsing.onnx"

        if model_path and Path(model_path).exists():
            try:
                import onnxruntime as ort

                self._model = ort.InferenceSession(
                    str(model_path),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self._model_type = "onnx"
                print(f"Loaded BiSeNet ONNX model from {model_path}")
                return
            except ImportError:
                print("onnxruntime not available, trying torch...")
            except Exception as e:
                print(f"Failed to load ONNX model: {e}")

        # Fallback: try to load PyTorch model
        try:
            import torch
            from .bisenet_model import BiSeNet

            # Check for PyTorch model
            pth_path = Path(model_path).with_suffix(".pth") if model_path else None
            if pth_path is None:
                model_dir = Path(__file__).parent.parent.parent / "models"
                pth_path = model_dir / "face_parsing.pth"

            if pth_path.exists():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._model = BiSeNet(n_classes=19)
                self._model.load_state_dict(torch.load(str(pth_path), map_location=device))
                self._model.to(device)
                self._model.eval()
                self._model_type = "torch"
                self._device = device
                print(f"Loaded BiSeNet PyTorch model from {pth_path}")
                return
        except ImportError:
            pass
        except Exception as e:
            print(f"Failed to load PyTorch model: {e}")

        # No model available - will use fallback landmark-based masks
        self._model = None
        self._model_type = None
        print("BiSeNet model not available, using landmark-based fallback")

    def is_available(self) -> bool:
        """Check if the face parsing model is available."""
        self._load_model()
        return self._model is not None

    def parse(self, image: np.ndarray) -> np.ndarray:
        """
        Parse face image and return segmentation map.

        Args:
            image: Input image in BGR format (H, W, 3)

        Returns:
            Segmentation map with class indices (H, W)
        """
        self._load_model()

        if self._model is None:
            # Return empty segmentation if model not available
            return np.zeros(image.shape[:2], dtype=np.uint8)

        # Preprocess image
        h, w = image.shape[:2]
        input_image = cv2.resize(image, self._input_size)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_image = (input_image - mean) / std

        # Add batch dimension and transpose to NCHW
        input_tensor = np.transpose(input_image, (2, 0, 1))[np.newaxis, ...]

        if self._model_type == "onnx":
            # ONNX inference
            input_name = self._model.get_inputs()[0].name
            outputs = self._model.run(None, {input_name: input_tensor.astype(np.float32)})
            segmentation = outputs[0][0]  # Remove batch dim
        else:
            # PyTorch inference
            import torch

            with torch.no_grad():
                input_tensor = torch.from_numpy(input_tensor).to(self._device)
                outputs = self._model(input_tensor)
                segmentation = outputs[0].cpu().numpy()

        # Get class predictions
        seg_map = np.argmax(segmentation, axis=0).astype(np.uint8)

        # Resize back to original size
        seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)

        return seg_map

    def get_part_mask(
        self,
        image: np.ndarray,
        part_name: str,
        landmarks: Optional[List[Tuple[float, float]]] = None,
        dilate_pixels: int = 0,
        blur_size: int = 0,
    ) -> np.ndarray:
        """
        Get binary mask for a specific facial part.

        Args:
            image: Input image in BGR format
            part_name: Name of the part ('left_eye', 'right_eye', 'nose', 'lips', etc.)
            landmarks: Optional landmarks for fallback if model not available
            dilate_pixels: Number of pixels to dilate the mask
            blur_size: Gaussian blur kernel size for soft edges (0 = no blur)

        Returns:
            Binary mask (H, W) with values 0-255
        """
        if part_name not in PART_TO_BISENET_LABELS:
            raise ValueError(f"Unknown part: {part_name}")

        self._load_model()

        if self._model is None and landmarks is not None:
            # Fallback to landmark-based mask
            return self._create_landmark_mask(image, part_name, landmarks, dilate_pixels, blur_size)

        # Parse the full face
        seg_map = self.parse(image)

        # Get labels for this part
        label_indices = PART_TO_BISENET_LABELS[part_name]

        # Create mask by combining all labels for this part
        mask = np.zeros(seg_map.shape, dtype=np.uint8)
        for label_idx in label_indices:
            mask[seg_map == label_idx] = 255

        # Dilate if requested
        if dilate_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel)

        # Blur for soft edges
        if blur_size > 0:
            blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        return mask

    def get_combined_mask(
        self,
        image: np.ndarray,
        part_names: List[str],
        landmarks: Optional[List[Tuple[float, float]]] = None,
        dilate_pixels: int = 0,
        blur_size: int = 0,
    ) -> np.ndarray:
        """
        Get combined binary mask for multiple facial parts.

        Args:
            image: Input image in BGR format
            part_names: List of part names
            landmarks: Optional landmarks for fallback
            dilate_pixels: Number of pixels to dilate the mask
            blur_size: Gaussian blur kernel size for soft edges

        Returns:
            Binary mask (H, W) with values 0-255
        """
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for part_name in part_names:
            part_mask = self.get_part_mask(
                image, part_name, landmarks, dilate_pixels=0, blur_size=0
            )
            combined_mask = cv2.bitwise_or(combined_mask, part_mask)

        # Apply dilation and blur to combined mask
        if dilate_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1)
            )
            combined_mask = cv2.dilate(combined_mask, kernel)

        if blur_size > 0:
            blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
            combined_mask = cv2.GaussianBlur(combined_mask, (blur_size, blur_size), 0)

        return combined_mask

    def _create_landmark_mask(
        self,
        image: np.ndarray,
        part_name: str,
        landmarks: List[Tuple[float, float]],
        dilate_pixels: int = 0,
        blur_size: int = 0,
    ) -> np.ndarray:
        """
        Fallback method: create mask from landmarks when BiSeNet not available.

        Args:
            image: Input image
            part_name: Part name
            landmarks: MediaPipe face mesh landmarks
            dilate_pixels: Dilation amount
            blur_size: Blur kernel size

        Returns:
            Binary mask
        """
        from .part_blender import FACIAL_PART_LANDMARKS

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        part_indices = FACIAL_PART_LANDMARKS.get(part_name, [])
        if not part_indices:
            return mask

        # Get points for the part
        points = []
        for idx in part_indices:
            if idx < len(landmarks):
                x, y = landmarks[idx]
                points.append([int(x), int(y)])

        if len(points) < 3:
            return mask

        points = np.array(points, dtype=np.int32)

        # Create convex hull and fill
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

        # Dilate
        if dilate_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_pixels * 2 + 1, dilate_pixels * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel)

        # Blur
        if blur_size > 0:
            blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        return mask


# Global instance
_face_parsing_service: Optional[FaceParsingService] = None


def get_face_parsing_service() -> FaceParsingService:
    """Get or create face parsing service instance."""
    global _face_parsing_service
    if _face_parsing_service is None:
        _face_parsing_service = FaceParsingService()
    return _face_parsing_service
