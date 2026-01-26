"""Face Swap API router.

Provides endpoints for:
- Generating face swaps using Replicate API
- Applying selective parts composition
- Previewing all parts at once
"""

import base64
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.models.schemas import (
    ErrorCodes,
    ErrorDetail,
    ErrorResponse,
    SwapGenerateData,
    SwapGenerateRequest,
    SwapGenerateResponse,
    SwapPartsData,
    SwapPartsRequest,
    SwapPartsResponse,
    SwapPreviewAllData,
    SwapPreviewAllRequest,
    SwapPreviewAllResponse,
    SwapResultData,
    SwapResultResponse,
)
from app.services.replicate_client import ReplicateError, get_replicate_client
from app.services.swap_cache import get_swap_cache
from app.services.swap_compositor import get_swap_compositor
from app.utils.image import bytes_to_cv2, cv2_to_base64

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/swap", tags=["swap"])


def _decode_base64_image(b64_string: str) -> bytes:
    """Decode base64 image string to bytes.

    Args:
        b64_string: Base64 encoded image (with or without data URL prefix).

    Returns:
        Image bytes.

    Raises:
        ValueError: If decoding fails.
    """
    # Remove data URL prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    try:
        return base64.b64decode(b64_string)
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {e}") from e


def _encode_bytes_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        Base64 encoded string.
    """
    return base64.b64encode(image_bytes).decode("utf-8")


@router.post("/generate", response_model=SwapGenerateResponse)
async def generate_swap(
    request: Request,
    data: SwapGenerateRequest,
):
    """
    Generate a face swap using Replicate API.

    Swaps the face from the ideal image onto the current image.
    Results are cached to avoid redundant API calls.

    - **current_image**: Base64 encoded current face image (receives the face)
    - **ideal_image**: Base64 encoded ideal face image (face to swap)

    Returns immediately with completed result if cached, otherwise
    calls Replicate API synchronously (may take 10-30 seconds).
    """
    try:
        # Decode images
        try:
            current_bytes = _decode_base64_image(data.current_image)
            ideal_bytes = _decode_base64_image(data.ideal_image)
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.VALIDATION_ERROR,
                        message=str(e),
                    )
                ).model_dump(),
            )

        # Check cache
        cache = get_swap_cache()
        cache_key = cache.generate_key(data.current_image, data.ideal_image)

        cached_result = cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for swap request: {cache_key[:16]}...")
            return SwapGenerateResponse(
                data=SwapGenerateData(
                    job_id=cache_key[:16],
                    status="completed",
                )
            )

        # Call Replicate API
        try:
            client = get_replicate_client()
            result_bytes = await client.run_faceswap(
                source_image=ideal_bytes,  # Face to swap FROM
                target_image=current_bytes,  # Face to swap ONTO
            )

            # Cache result
            cache.set(cache_key, result_bytes)

            logger.info(f"Swap completed and cached: {cache_key[:16]}...")

            return SwapGenerateResponse(
                data=SwapGenerateData(
                    job_id=cache_key[:16],
                    status="completed",
                )
            )

        except ReplicateError as e:
            logger.error(f"Replicate API error: {e}")
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.PROCESSING_ERROR,
                        message=f"Face swap failed: {e}",
                    )
                ).model_dump(),
            )

    except Exception as e:
        logger.exception(f"Unexpected error in generate_swap: {e}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.INTERNAL_ERROR,
                    message=f"Internal error: {e}",
                )
            ).model_dump(),
        )


@router.get("/generate/{job_id}", response_model=SwapResultResponse)
async def get_swap_result(
    request: Request,
    job_id: str,
):
    """
    Get the result of a face swap operation.

    For synchronous swap operations, the job_id is the first 16 characters
    of the cache key. Use this endpoint to retrieve the swapped image.

    Returns the swapped image as base64 encoded data when complete.
    """
    try:
        cache = get_swap_cache()

        # Try to find cached result by checking all keys
        # In production, this should use a proper job ID mapping
        for key in list(cache._cache.keys()):
            if key.startswith(job_id) or job_id in key[:16]:
                result_bytes = cache.get(key)
                if result_bytes:
                    return SwapResultResponse(
                        data=SwapResultData(
                            status="completed",
                            swapped_image=_encode_bytes_to_base64(result_bytes),
                        )
                    )

        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.NOT_FOUND,
                    message=f"Swap result not found for job {job_id}",
                )
            ).model_dump(),
        )

    except Exception as e:
        logger.exception(f"Error getting swap result: {e}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.INTERNAL_ERROR,
                    message=f"Internal error: {e}",
                )
            ).model_dump(),
        )


@router.post("/parts", response_model=SwapPartsResponse)
async def apply_parts(
    request: Request,
    data: SwapPartsRequest,
):
    """
    Apply selective parts composition from swapped face onto original.

    Takes the original face and a swapped face, then selectively applies
    individual facial parts with configurable intensity.

    - **current_image**: Base64 encoded original face image
    - **swapped_image**: Base64 encoded swapped face image
    - **parts**: Dictionary of parts to apply with intensity (0.0-1.0)
      - Supports: "left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "nose", "lips"
      - Aliases: "eyes" (both eyes), "eyebrows" (both eyebrows)

    Example parts: {"eyes": 0.8, "nose": 1.0, "lips": 0.5}
    """
    try:
        # Decode images
        try:
            current_bytes = _decode_base64_image(data.current_image)
            swapped_bytes = _decode_base64_image(data.swapped_image)
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.VALIDATION_ERROR,
                        message=str(e),
                    )
                ).model_dump(),
            )

        # Convert to OpenCV format
        try:
            current_img = bytes_to_cv2(current_bytes)
            swapped_img = bytes_to_cv2(swapped_bytes)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.INVALID_IMAGE_FORMAT,
                        message=f"Failed to decode image: {e}",
                    )
                ).model_dump(),
            )

        # If no parts or all zero intensity, return original
        if not data.parts or all(v == 0 for v in data.parts.values()):
            result_b64 = cv2_to_base64(current_img)
            return SwapPartsResponse(
                data=SwapPartsData(result_image=result_b64)
            )

        # Apply parts composition
        compositor = get_swap_compositor()
        result_img = compositor.compose_parts(
            original=current_img,
            swapped=swapped_img,
            parts=data.parts,
        )

        # Encode result
        result_b64 = cv2_to_base64(result_img)

        return SwapPartsResponse(
            data=SwapPartsData(result_image=result_b64)
        )

    except Exception as e:
        logger.exception(f"Error applying parts: {e}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.INTERNAL_ERROR,
                    message=f"Parts composition failed: {e}",
                )
            ).model_dump(),
        )


@router.post("/preview-all", response_model=SwapPreviewAllResponse)
async def preview_all_parts(
    request: Request,
    data: SwapPreviewAllRequest,
):
    """
    Preview all parts composed at once.

    Convenience endpoint that applies all specified parts in a single pass.
    Same functionality as /parts but optimized for previewing multiple parts.

    - **current_image**: Base64 encoded original face image
    - **swapped_image**: Base64 encoded swapped face image
    - **parts**: Dictionary of parts to apply with intensity

    Example: {"left_eye": 0.8, "right_eye": 0.8, "nose": 1.0, "lips": 0.5}
    """
    try:
        # Decode images
        try:
            current_bytes = _decode_base64_image(data.current_image)
            swapped_bytes = _decode_base64_image(data.swapped_image)
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.VALIDATION_ERROR,
                        message=str(e),
                    )
                ).model_dump(),
            )

        # Convert to OpenCV format
        try:
            current_img = bytes_to_cv2(current_bytes)
            swapped_img = bytes_to_cv2(swapped_bytes)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.INVALID_IMAGE_FORMAT,
                        message=f"Failed to decode image: {e}",
                    )
                ).model_dump(),
            )

        # If no parts, return original
        if not data.parts or all(v == 0 for v in data.parts.values()):
            result_b64 = cv2_to_base64(current_img)
            return SwapPreviewAllResponse(
                data=SwapPreviewAllData(result_image=result_b64)
            )

        # Apply all parts composition
        compositor = get_swap_compositor()
        result_img = compositor.compose_all_parts(
            original=current_img,
            swapped=swapped_img,
            parts=data.parts,
        )

        # Encode result
        result_b64 = cv2_to_base64(result_img)

        return SwapPreviewAllResponse(
            data=SwapPreviewAllData(result_image=result_b64)
        )

    except Exception as e:
        logger.exception(f"Error in preview-all: {e}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.INTERNAL_ERROR,
                    message=f"Preview failed: {e}",
                )
            ).model_dump(),
        )
