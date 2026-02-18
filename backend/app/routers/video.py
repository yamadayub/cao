"""Video generation API router."""

import base64
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings
from app.models.schemas import ErrorCodes, ErrorDetail, ErrorResponse, SuccessResponse
from app.services.auth import get_current_user
from app.services.supabase_client import get_supabase_client
from app.services.video_generator import get_video_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video", tags=["video"])

# Rate limiter for video generation (heavier processing)
limiter = Limiter(key_func=get_remote_address)


# ============================================
# Request/Response Schemas
# ============================================


class VideoGenerateRequest(BaseModel):
    """Request to generate a morphing video."""

    source_image: str = Field(..., description="Base64 encoded Before image")
    result_image: str = Field(..., description="Base64 encoded After image")


class VideoGenerateData(BaseModel):
    """Video generation result data."""

    video_url: str = Field(..., description="URL or data URI of the generated video")
    duration: float = Field(..., description="Video duration in seconds")
    format: str = Field(default="mp4", description="Video format")


class VideoGenerateResponse(SuccessResponse[VideoGenerateData]):
    """Response for video generation."""

    pass


# ============================================
# Endpoints
# ============================================


async def _upload_video_to_supabase(
    video_bytes: bytes,
    video_id: str,
) -> Optional[str]:
    """Upload video to Supabase storage.

    Returns the public URL if successful, None otherwise.
    """
    client = get_supabase_client()
    if client is None:
        return None

    try:
        settings = get_settings()
        bucket_name = "videos"
        file_path = f"{video_id}/morph.mp4"

        client.storage.from_(bucket_name).upload(
            file_path,
            video_bytes,
            file_options={"content-type": "video/mp4"},
        )

        storage_base = settings.supabase_url.replace(
            ".supabase.co",
            ".supabase.co/storage/v1/object/public",
        )
        return f"{storage_base}/{bucket_name}/{file_path}"
    except Exception as e:
        logger.warning(f"Failed to upload video to Supabase: {e}")
        return None


def _decode_base64_image(base64_str: str) -> bytes:
    """Decode base64 string to image bytes."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    return base64.b64decode(base64_str)


@router.post("/generate", response_model=VideoGenerateResponse)
async def generate_morph_video(
    request: Request,
    body: VideoGenerateRequest,
    user: dict = Depends(get_current_user),
):
    """Generate a morphing video from Before/After images.

    Requires authentication. Creates a slider-style morphing video
    (9:16, 1080x1920) suitable for TikTok/Reels/Shorts.

    Rate limited to 5 requests per minute.
    """
    # Decode images
    try:
        source_bytes = _decode_base64_image(body.source_image)
        result_bytes = _decode_base64_image(body.result_image)
    except Exception:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.VALIDATION_ERROR,
                    message="Invalid base64 image data",
                )
            ).model_dump(),
        )

    # Generate video
    try:
        generator = get_video_generator()
        video_bytes = generator.generate(source_bytes, result_bytes)
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.PROCESSING_ERROR,
                    message=f"Failed to generate video: {str(e)}",
                )
            ).model_dump(),
        )

    # Try to upload to Supabase storage
    video_id = str(uuid.uuid4())
    video_url = await _upload_video_to_supabase(video_bytes, video_id)

    # Fallback to base64 data URL
    if not video_url:
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        video_url = f"data:video/mp4;base64,{video_b64}"

    # Calculate duration
    duration = HOLD_BEFORE + SLIDE_FORWARD + HOLD_AFTER + SLIDE_BACK + HOLD_END

    return VideoGenerateResponse(
        data=VideoGenerateData(
            video_url=video_url,
            duration=duration,
            format="mp4",
        )
    )


# Import timing constants for duration calculation
from app.services.video_generator import (
    HOLD_AFTER,
    HOLD_BEFORE,
    HOLD_END,
    SLIDE_BACK,
    SLIDE_FORWARD,
)
