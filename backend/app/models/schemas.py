"""Pydantic models for API request/response schemas."""

from datetime import datetime
from typing import Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# ============================================
# Common Response Schemas
# ============================================


class ErrorDetail(BaseModel):
    """Error detail model."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Standard error response."""

    success: Literal[False] = False
    error: ErrorDetail


class ResponseMeta(BaseModel):
    """Response metadata."""

    request_id: str = Field(..., description="Unique request identifier")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class SuccessResponse(BaseModel, Generic[T]):
    """Standard success response."""

    success: Literal[True] = True
    data: T
    meta: Optional[ResponseMeta] = None


# ============================================
# Health Check Schemas
# ============================================


class HealthData(BaseModel):
    """Health check data."""

    status: Literal["ok", "degraded"] = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Current timestamp")


class HealthResponse(SuccessResponse[HealthData]):
    """Health check response."""

    pass


# ============================================
# Face Analysis Schemas
# ============================================


class FaceLandmark(BaseModel):
    """Face landmark point."""

    index: int = Field(..., description="Landmark index (0-477)")
    x: float = Field(..., ge=0, le=1, description="Normalized X coordinate (0.0 - 1.0)")
    y: float = Field(..., ge=0, le=1, description="Normalized Y coordinate (0.0 - 1.0)")
    z: float = Field(..., description="Z coordinate (depth)")


class FaceRegion(BaseModel):
    """Face bounding box region."""

    x: int = Field(..., ge=0, description="X coordinate of top-left corner")
    y: int = Field(..., ge=0, description="Y coordinate of top-left corner")
    width: int = Field(..., gt=0, description="Width of face region")
    height: int = Field(..., gt=0, description="Height of face region")


class ImageInfo(BaseModel):
    """Image metadata."""

    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")
    format: Literal["jpeg", "png"] = Field(..., description="Image format")


class AnalyzeData(BaseModel):
    """Face analysis result data."""

    face_detected: bool = Field(..., description="Whether a face was detected")
    face_count: int = Field(..., ge=0, description="Number of faces detected")
    face_region: Optional[FaceRegion] = Field(None, description="Face bounding box")
    landmarks: Optional[List[FaceLandmark]] = Field(None, description="Face landmarks")
    image_info: ImageInfo = Field(..., description="Image information")


class AnalyzeResponse(SuccessResponse[AnalyzeData]):
    """Face analysis response."""

    pass


# ============================================
# Morphing Schemas
# ============================================


class ImageDimensions(BaseModel):
    """Image dimensions."""

    width: int = Field(..., gt=0, description="Image width")
    height: int = Field(..., gt=0, description="Image height")


class MorphData(BaseModel):
    """Single morphing result data."""

    image: str = Field(..., description="Base64 encoded result image")
    format: Literal["png"] = Field(default="png", description="Output image format")
    progress: float = Field(..., ge=0, le=1, description="Morphing progress (0.0 - 1.0)")
    dimensions: ImageDimensions = Field(..., description="Output image dimensions")


class MorphResponse(SuccessResponse[MorphData]):
    """Single morphing response."""

    pass


class StageImage(BaseModel):
    """Single stage image in staged morphing."""

    progress: float = Field(..., ge=0, le=1, description="Stage progress value")
    image: str = Field(..., description="Base64 encoded image")


class StagedMorphData(BaseModel):
    """Staged morphing result data."""

    images: List[StageImage] = Field(..., description="List of stage images")
    format: Literal["png"] = Field(default="png", description="Output image format")
    dimensions: ImageDimensions = Field(..., description="Output image dimensions")


class StagedMorphResponse(SuccessResponse[StagedMorphData]):
    """Staged morphing response."""

    pass


# ============================================
# Error Code Constants
# ============================================


class ErrorCodes:
    """Error code constants."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    FACE_NOT_DETECTED = "FACE_NOT_DETECTED"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    UNAUTHORIZED = "UNAUTHORIZED"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"
