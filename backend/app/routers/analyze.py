"""Face analysis endpoint."""

from __future__ import annotations

import time
from typing import Optional, Union

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.models.schemas import (
    AnalyzeData,
    AnalyzeResponse,
    ErrorCodes,
    ErrorDetail,
    ErrorResponse,
    ResponseMeta,
)
from app.services.face_detection import get_face_detection_service
from app.utils.image import ImageValidationError, validate_image

router = APIRouter(tags=["analyze"])


@router.post("/analyze", response_model=None)
async def analyze_face(
    request: Request,
    image: Optional[UploadFile] = File(None),
):
    """
    Analyze a face in an uploaded image.

    Detects face and extracts landmarks using MediaPipe Face Mesh.

    Args:
        request: FastAPI request object
        image: Uploaded image file (JPEG or PNG, max 10MB)

    Returns:
        Face analysis results including landmarks and face region
    """
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")

    # Validate image file is provided
    if image is None or image.filename == "":
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.VALIDATION_ERROR,
                    message="Image file is required",
                )
            ).model_dump(),
        )

    try:
        # Read image data
        image_data = await image.read()

        if len(image_data) == 0:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.VALIDATION_ERROR,
                        message="Image file is required",
                    )
                ).model_dump(),
            )

        # Validate and load image
        image_format, cv2_image = validate_image(image_data)

        # Detect faces
        face_service = get_face_detection_service()
        result = face_service.detect(
            cv2_image,
            image_format=image_format,
            require_single_face=True,
        )

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        return AnalyzeResponse(
            data=AnalyzeData(
                face_detected=result.face_detected,
                face_count=result.face_count,
                face_region=result.face_region,
                landmarks=result.landmarks,
                image_info=result.image_info,
            ),
            meta=ResponseMeta(
                request_id=request_id,
                processing_time_ms=processing_time_ms,
            ),
        )

    except ImageValidationError as e:
        # Map error code to HTTP status
        status_code_map = {
            ErrorCodes.IMAGE_TOO_LARGE: 413,
            ErrorCodes.INVALID_IMAGE_FORMAT: 400,
            ErrorCodes.FACE_NOT_DETECTED: 400,
            ErrorCodes.MULTIPLE_FACES: 400,
        }
        status_code = status_code_map.get(e.code, 400)

        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=e.code,
                    message=e.message,
                    details=e.details,
                )
            ).model_dump(),
        )

    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Analyze error: {e}")

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.PROCESSING_ERROR,
                    message="An unexpected error occurred during image analysis",
                )
            ).model_dump(),
        )
