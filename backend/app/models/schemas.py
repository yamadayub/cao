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
# Parts Blend Schemas
# ============================================


class PartsBlendData(BaseModel):
    """Parts blend result data."""

    image: str = Field(..., description="Base64 encoded result image")
    format: Literal["png"] = Field(default="png", description="Output image format")
    applied_parts: List[str] = Field(..., description="List of parts that were applied")
    dimensions: ImageDimensions = Field(..., description="Output image dimensions")


class PartsBlendResponse(SuccessResponse[PartsBlendData]):
    """Parts blend response."""

    pass


# ============================================
# Error Code Constants
# ============================================


# ============================================
# Simulation Schemas
# ============================================


class ResultImageItem(BaseModel):
    """Result image with progress."""

    progress: float = Field(..., ge=0, le=1, description="Progress value (0.0 - 1.0)")
    image: str = Field(..., description="Base64 encoded image or URL")


class SimulationSettings(BaseModel):
    """Simulation settings."""

    selected_progress: Optional[float] = Field(None, ge=0, le=1, description="User selected progress")
    notes: Optional[str] = Field(None, description="User notes")


class CreateSimulationRequest(BaseModel):
    """Request to create a simulation."""

    current_image: str = Field(..., description="Base64 encoded current face image")
    ideal_image: str = Field(..., description="Base64 encoded ideal face image")
    result_images: List[ResultImageItem] = Field(..., description="List of result images")
    settings: Optional[SimulationSettings] = Field(None, description="Optional settings")


class SimulationData(BaseModel):
    """Full simulation data."""

    id: str = Field(..., description="Simulation UUID")
    user_id: str = Field(..., description="User ID")
    current_image_url: str = Field(..., description="URL to current face image")
    ideal_image_url: str = Field(..., description="URL to ideal face image")
    result_images: List[ResultImageItem] = Field(..., description="Result images")
    settings: dict = Field(default_factory=dict, description="Simulation settings")
    share_token: Optional[str] = Field(None, description="Share token if public")
    is_public: bool = Field(False, description="Whether simulation is publicly shared")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class SimulationResponse(SuccessResponse[SimulationData]):
    """Simulation response."""

    pass


class SimulationSummary(BaseModel):
    """Simulation summary for list view."""

    id: str = Field(..., description="Simulation UUID")
    thumbnail_url: str = Field(..., description="Thumbnail URL (50% progress)")
    created_at: datetime = Field(..., description="Creation timestamp")
    is_public: bool = Field(False, description="Whether simulation is publicly shared")


class PaginationInfo(BaseModel):
    """Pagination information."""

    total: int = Field(..., ge=0, description="Total count")
    limit: int = Field(..., ge=1, description="Page size")
    offset: int = Field(..., ge=0, description="Offset")
    has_more: bool = Field(..., description="Whether there are more results")


class SimulationListData(BaseModel):
    """Simulation list data."""

    simulations: List[SimulationSummary] = Field(..., description="List of simulations")
    pagination: PaginationInfo = Field(..., description="Pagination info")


class SimulationListResponse(SuccessResponse[SimulationListData]):
    """Simulation list response."""

    pass


class DeletedData(BaseModel):
    """Delete response data."""

    deleted: bool = Field(True, description="Whether deletion succeeded")
    id: str = Field(..., description="Deleted resource ID")


class DeleteResponse(SuccessResponse[DeletedData]):
    """Delete response."""

    pass


class ShareData(BaseModel):
    """Share URL data."""

    share_token: str = Field(..., description="Share token")
    share_url: str = Field(..., description="Full share URL")
    expires_at: Optional[datetime] = Field(None, description="Expiration (null for no expiry)")


class ShareResponse(SuccessResponse[ShareData]):
    """Share response."""

    pass


class SharedSimulationData(BaseModel):
    """Shared simulation data (public view)."""

    result_images: List[ResultImageItem] = Field(..., description="Result images")
    created_at: datetime = Field(..., description="Creation timestamp")


class SharedSimulationResponse(SuccessResponse[SharedSimulationData]):
    """Shared simulation response."""

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
    JOB_FAILED = "JOB_FAILED"


# ============================================
# Generation Job Schemas
# ============================================


class GenerationMode(BaseModel):
    """Generation mode enum-like constants."""

    MORPH: str = "morph"
    PARTS: str = "parts"


class CreateGenerationJobRequest(BaseModel):
    """Request to create a generation job."""

    base_image: str = Field(..., description="Base64 encoded base face image")
    target_image: str = Field(..., description="Base64 encoded target/ideal face image")
    mode: Literal["morph", "parts"] = Field(..., description="Generation mode: 'morph' or 'parts'")
    parts: Optional[List[str]] = Field(
        None,
        description="For mode='parts': list of parts to blend ['left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow', 'nose', 'lips']",
    )
    strength: Optional[float] = Field(
        0.5, ge=0, le=1, description="Blend strength (0=base, 1=target)"
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class GenerationJobStatus(BaseModel):
    """Generation job status data."""

    job_id: str = Field(..., description="Unique job identifier")
    status: Literal["queued", "running", "succeeded", "failed"] = Field(
        ..., description="Current job status"
    )
    progress: int = Field(0, ge=0, le=100, description="Processing progress percentage")
    result_image_url: Optional[str] = Field(
        None, description="Result image URL (when succeeded)"
    )
    error: Optional[str] = Field(None, description="Error message (when failed)")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")


class CreateGenerationJobResponse(SuccessResponse[GenerationJobStatus]):
    """Response for job creation."""

    pass


class GenerationJobStatusResponse(SuccessResponse[GenerationJobStatus]):
    """Response for job status query."""

    pass


class GenerationResultData(BaseModel):
    """Generation result data."""

    job_id: str = Field(..., description="Job identifier")
    image: str = Field(..., description="Base64 encoded result image")
    format: Literal["png"] = Field(default="png", description="Image format")
    mode: Literal["morph", "parts"] = Field(..., description="Generation mode used")
    strength: float = Field(..., description="Strength value used")


class GenerationResultResponse(SuccessResponse[GenerationResultData]):
    """Response for generation result."""

    pass
