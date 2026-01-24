"""Image processing utilities."""

import base64
import io
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Register HEIF/HEIC support with Pillow (for iPhone images)
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # pillow-heif not installed, HEIC support disabled

from app.config import get_settings

settings = get_settings()

# Magic bytes for supported formats
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
WEBP_MAGIC = b"RIFF"
WEBP_MAGIC2 = b"WEBP"
# HEIC/HEIF uses ISO Base Media File Format with ftyp box
HEIC_FTYP = b"ftyp"
HEIC_BRANDS = (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1")


class ImageValidationError(Exception):
    """Exception raised for image validation errors."""

    def __init__(self, code: str, message: str, details: Optional[dict] = None):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


def validate_magic_bytes(data: bytes) -> Literal["jpeg", "png", "webp", "heic"]:
    """
    Validate image format by checking magic bytes.

    Args:
        data: Raw image data bytes

    Returns:
        Image format ('jpeg', 'png', 'webp', or 'heic')

    Raises:
        ImageValidationError: If format is not supported
    """
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    elif data.startswith(PNG_MAGIC):
        return "png"
    elif data.startswith(WEBP_MAGIC) and len(data) > 11 and data[8:12] == WEBP_MAGIC2:
        return "webp"
    elif len(data) > 12 and data[4:8] == HEIC_FTYP and data[8:12] in HEIC_BRANDS:
        return "heic"
    else:
        raise ImageValidationError(
            code="INVALID_IMAGE_FORMAT",
            message="Only JPEG, PNG, WebP, and HEIC formats are supported",
        )


def validate_image_size(data: bytes) -> None:
    """
    Validate image file size.

    Args:
        data: Raw image data bytes

    Raises:
        ImageValidationError: If image exceeds size limit
    """
    if len(data) > settings.max_image_size_bytes:
        raise ImageValidationError(
            code="IMAGE_TOO_LARGE",
            message="Image size must be under 10MB",
            details={"max_size_mb": settings.max_image_size_mb, "actual_size_mb": len(data) / (1024 * 1024)},
        )


def bytes_to_cv2(data: bytes) -> np.ndarray:
    """
    Convert bytes to OpenCV image (BGR).

    Args:
        data: Raw image data bytes

    Returns:
        OpenCV image array in BGR format
    """
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageValidationError(
            code="INVALID_IMAGE_FORMAT",
            message="Failed to decode image",
        )
    return img


def cv2_to_bytes(img: np.ndarray, format: str = "png") -> bytes:
    """
    Convert OpenCV image to bytes.

    Args:
        img: OpenCV image array in BGR format
        format: Output format ('png' or 'jpeg')

    Returns:
        Image as bytes
    """
    if format == "png":
        _, buffer = cv2.imencode(".png", img)
    else:
        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


def cv2_to_base64(img: np.ndarray, format: str = "png") -> str:
    """
    Convert OpenCV image to base64 string.

    Args:
        img: OpenCV image array in BGR format
        format: Output format ('png' or 'jpeg')

    Returns:
        Base64 encoded image string
    """
    image_bytes = cv2_to_bytes(img, format)
    return base64.b64encode(image_bytes).decode("utf-8")


def bytes_to_base64(data: bytes) -> str:
    """
    Convert bytes to base64 string.

    Args:
        data: Raw bytes

    Returns:
        Base64 encoded string
    """
    return base64.b64encode(data).decode("utf-8")


def get_image_dimensions(data: bytes) -> Tuple[int, int]:
    """
    Get image dimensions from bytes.

    Args:
        data: Raw image data bytes

    Returns:
        Tuple of (width, height)
    """
    img = Image.open(io.BytesIO(data))
    return img.size


def resize_image_if_needed(img: np.ndarray, max_dimension: Optional[int] = None) -> np.ndarray:
    """
    Resize image if it exceeds maximum dimension.

    Args:
        img: OpenCV image array
        max_dimension: Maximum dimension (width or height). Uses config default if None.

    Returns:
        Resized image or original if within limits
    """
    if max_dimension is None:
        max_dimension = settings.max_image_dimension

    h, w = img.shape[:2]
    if max(h, w) <= max_dimension:
        return img

    scale = max_dimension / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def validate_image(data: bytes) -> Tuple[Literal["jpeg", "png"], np.ndarray]:
    """
    Validate and load image.

    Args:
        data: Raw image data bytes

    Returns:
        Tuple of (format, opencv_image)

    Raises:
        ImageValidationError: If validation fails
    """
    # Check size
    validate_image_size(data)

    # Try to detect format by magic bytes first
    try:
        format = validate_magic_bytes(data)
    except ImageValidationError:
        # If magic bytes don't match, try to open with PIL
        # This handles cases like HEIC, BMP, or other formats
        try:
            pil_img = Image.open(io.BytesIO(data))
            pil_format = pil_img.format
            if pil_format not in ("JPEG", "PNG", "WEBP", "GIF", "BMP", "HEIC", "HEIF"):
                raise ImageValidationError(
                    code="INVALID_IMAGE_FORMAT",
                    message="Only JPEG, PNG, WebP, and HEIC formats are supported",
                )
            # Convert to RGB if needed
            if pil_img.mode in ("RGBA", "LA", "P"):
                pil_img = pil_img.convert("RGBA")
            elif pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            # Convert to PNG for consistency
            output = io.BytesIO()
            pil_img.save(output, format="PNG")
            data = output.getvalue()
            format = "png"
        except Exception as e:
            raise ImageValidationError(
                code="INVALID_IMAGE_FORMAT",
                message="Only JPEG, PNG, WebP, and HEIC formats are supported",
                details={"error": str(e)},
            )

    # If WebP or HEIC, convert to PNG for OpenCV compatibility
    if format in ("webp", "heic"):
        try:
            pil_img = Image.open(io.BytesIO(data))
            if pil_img.mode in ("RGBA", "LA", "P"):
                pil_img = pil_img.convert("RGBA")
            elif pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            output = io.BytesIO()
            pil_img.save(output, format="PNG")
            data = output.getvalue()
            format = "png"
        except Exception as e:
            raise ImageValidationError(
                code="INVALID_IMAGE_FORMAT",
                message=f"Failed to process {format.upper()} image",
                details={"error": str(e)},
            )

    # Load image
    img = bytes_to_cv2(data)

    # Resize if needed
    img = resize_image_if_needed(img)

    return format, img
