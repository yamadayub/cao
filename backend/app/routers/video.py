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
from app.services.blend_video_generator import (
    get_blend_video_generator,
)
from app.services.supabase_client import get_supabase_client
from app.services.video_generator import (
    HOLD_AFTER,
    HOLD_BEFORE,
    HOLD_END,
    SLIDE_BACK,
    SLIDE_FORWARD,
    get_video_generator,
)

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


class BlendVideoGenerateRequest(BaseModel):
    """Request to generate a before/after blend video."""

    current_image: str = Field(..., description="Base64 encoded before face")
    ideal_image: Optional[str] = Field(None, description="Deprecated, ignored")
    result_image: str = Field(..., description="Base64 encoded after face")
    video_pattern: str = Field(default="A", description="Video pattern: A (4s loop) or B (6s morph)")


class VideoGenerateData(BaseModel):
    """Video generation result data."""

    video_url: str = Field(..., description="URL or data URI of the generated video")
    duration: float = Field(..., description="Video duration in seconds")
    format: str = Field(default="mp4", description="Video format")
    loop_friendly: Optional[bool] = Field(default=None, description="Whether the video loops seamlessly")
    beat_sync_points: Optional[list] = Field(default=None, description="Timestamps of snap cuts for beat sync")


class VideoGenerateResponse(SuccessResponse[VideoGenerateData]):
    """Response for video generation."""

    pass


# ============================================
# Endpoints
# ============================================


async def _upload_video_to_supabase(
    video_bytes: bytes,
    video_id: str,
    extension: str = ".webm",
    content_type: str = "video/webm",
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
        file_path = f"{video_id}/morph{extension}"

        client.storage.from_(bucket_name).upload(
            file_path,
            video_bytes,
            file_options={"content-type": content_type},
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

    # Validate decoded data
    if not source_bytes or not result_bytes:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.VALIDATION_ERROR,
                    message="Empty image data after decoding",
                )
            ).model_dump(),
        )

    # Generate video
    try:
        generator = get_video_generator()
        result = generator.generate(source_bytes, result_bytes)
        logger.info(
            f"Video generated: {len(result.data)} bytes, "
            f"format={result.content_type}"
        )
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
    video_url = await _upload_video_to_supabase(
        result.data, video_id, result.extension, result.content_type
    )

    # Fallback to base64 data URL
    if not video_url:
        video_b64 = base64.b64encode(result.data).decode("utf-8")
        video_url = f"data:{result.content_type};base64,{video_b64}"

    # Calculate duration
    duration = HOLD_BEFORE + SLIDE_FORWARD + HOLD_AFTER + SLIDE_BACK + HOLD_END
    video_format = result.extension.lstrip(".")

    return VideoGenerateResponse(
        data=VideoGenerateData(
            video_url=video_url,
            duration=duration,
            format=video_format,
        )
    )


@router.post("/blend", response_model=VideoGenerateResponse)
async def generate_blend_video(
    request: Request,
    body: BlendVideoGenerateRequest,
    user: dict = Depends(get_current_user),
):
    """Generate a before/after blend video with wipe transition.

    Shows before face → wipe → after face → logo.
    Suitable for sharing on SNS (9:16, 720x1280, ~3s).

    Requires authentication (Bearer JWT or X-API-Key).
    """
    # Decode images
    try:
        current_bytes = _decode_base64_image(body.current_image)
        ideal_bytes = (
            _decode_base64_image(body.ideal_image) if body.ideal_image else None
        )
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

    if not current_bytes or not result_bytes:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.VALIDATION_ERROR,
                    message="Empty image data after decoding",
                )
            ).model_dump(),
        )

    # Validate pattern
    pattern = body.video_pattern.upper()
    if pattern not in ("A", "B"):
        pattern = "A"

    # Generate video
    try:
        generator = get_blend_video_generator()
        result = generator.generate(
            current_bytes, ideal_bytes, result_bytes, pattern=pattern
        )
        logger.info(
            f"Blend video generated: {len(result.data)} bytes, "
            f"format={result.content_type}, duration={result.duration:.1f}s, "
            f"pattern={pattern}"
        )
    except Exception as e:
        logger.error(f"Blend video generation failed: {e}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.PROCESSING_ERROR,
                    message=f"Failed to generate blend video: {str(e)}",
                )
            ).model_dump(),
        )

    # Upload to Supabase
    video_id = str(uuid.uuid4())
    video_url = await _upload_video_to_supabase(
        result.data, video_id, result.extension, result.content_type
    )

    # Fallback to base64 data URL
    if not video_url:
        video_b64 = base64.b64encode(result.data).decode("utf-8")
        video_url = f"data:{result.content_type};base64,{video_b64}"

    video_format = result.extension.lstrip(".")

    return VideoGenerateResponse(
        data=VideoGenerateData(
            video_url=video_url,
            duration=result.duration,
            format=video_format,
            loop_friendly=result.metadata.get("loop_friendly"),
            beat_sync_points=result.metadata.get("beat_sync_points"),
        )
    )
