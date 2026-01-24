"""Face morphing endpoints."""

from __future__ import annotations

import json
import time
from typing import List, Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.models.schemas import (
    ErrorCodes,
    ErrorDetail,
    ErrorResponse,
    ImageDimensions,
    MorphData,
    MorphResponse,
    ResponseMeta,
    StagedMorphData,
    StagedMorphResponse,
    StageImage,
)
from app.services.morphing import get_morphing_service
from app.utils.image import ImageValidationError, cv2_to_base64, validate_image

router = APIRouter(tags=["morph"])

DEFAULT_STAGES = [0.0, 0.25, 0.5, 0.75, 1.0]


@router.post("/morph", response_model=None)
async def morph_faces(
    request: Request,
    current_image: Optional[UploadFile] = File(None),
    ideal_image: Optional[UploadFile] = File(None),
    progress: float = Form(0.5),
):
    """
    Morph between two face images.

    Args:
        request: FastAPI request object
        current_image: Current face image (JPEG/PNG, max 10MB)
        ideal_image: Ideal/target face image (JPEG/PNG, max 10MB)
        progress: Morphing progress (0.0 = current, 1.0 = ideal)

    Returns:
        Morphed image as base64-encoded PNG
    """
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")

    # Validate progress parameter
    if progress < 0.0 or progress > 1.0:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.VALIDATION_ERROR,
                    message="Progress must be between 0.0 and 1.0",
                )
            ).model_dump(),
        )

    # Validate current_image is provided
    if current_image is None or current_image.filename == "":
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.VALIDATION_ERROR,
                    message="current_image is required",
                )
            ).model_dump(),
        )

    # Validate ideal_image is provided
    if ideal_image is None or ideal_image.filename == "":
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.VALIDATION_ERROR,
                    message="ideal_image is required",
                )
            ).model_dump(),
        )

    try:
        # Read and validate images
        current_data = await current_image.read()
        ideal_data = await ideal_image.read()

        if len(current_data) == 0:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.VALIDATION_ERROR,
                        message="current_image is required",
                    )
                ).model_dump(),
            )

        if len(ideal_data) == 0:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.VALIDATION_ERROR,
                        message="ideal_image is required",
                    )
                ).model_dump(),
            )

        # Validate and load images
        _, cv2_current = validate_image(current_data)
        _, cv2_ideal = validate_image(ideal_data)

        # Perform morphing
        morph_service = get_morphing_service()
        result = morph_service.morph(
            cv2_current,
            cv2_ideal,
            progress,
            img1_label="current",
            img2_label="ideal",
        )

        # Get dimensions
        h, w = result.shape[:2]

        # Convert to base64
        image_base64 = cv2_to_base64(result, "png")

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        return MorphResponse(
            data=MorphData(
                image=image_base64,
                format="png",
                progress=progress,
                dimensions=ImageDimensions(width=w, height=h),
            ),
            meta=ResponseMeta(
                request_id=request_id,
                processing_time_ms=processing_time_ms,
            ),
        )

    except ImageValidationError as e:
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
        print(f"Morph error: {e}")

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.PROCESSING_ERROR,
                    message="Failed to generate morphed image",
                )
            ).model_dump(),
        )


@router.post("/morph/stages", response_model=None)
async def morph_stages(
    request: Request,
    current_image: Optional[UploadFile] = File(None),
    ideal_image: Optional[UploadFile] = File(None),
    stages: Optional[str] = Form(None),
):
    """
    Generate multiple morphing stages between two face images.

    Args:
        request: FastAPI request object
        current_image: Current face image (JPEG/PNG, max 10MB)
        ideal_image: Ideal/target face image (JPEG/PNG, max 10MB)
        stages: JSON array of progress values (e.g., "[0, 0.25, 0.5, 0.75, 1.0]")

    Returns:
        Multiple morphed images at different progress levels
    """
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "unknown")

    # Parse stages
    stage_values: List[float] = DEFAULT_STAGES
    if stages is not None and stages.strip():
        try:
            parsed = json.loads(stages)
            if not isinstance(parsed, list):
                return JSONResponse(
                    status_code=400,
                    content=ErrorResponse(
                        error=ErrorDetail(
                            code=ErrorCodes.VALIDATION_ERROR,
                            message="stages must be a JSON array of numbers",
                        )
                    ).model_dump(),
                )

            stage_values = [float(v) for v in parsed]

            # Validate all values are in range
            for v in stage_values:
                if v < 0.0 or v > 1.0:
                    return JSONResponse(
                        status_code=400,
                        content=ErrorResponse(
                            error=ErrorDetail(
                                code=ErrorCodes.VALIDATION_ERROR,
                                message="All stage values must be between 0.0 and 1.0",
                                details={"invalid_value": v},
                            )
                        ).model_dump(),
                    )

        except (json.JSONDecodeError, ValueError, TypeError):
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.VALIDATION_ERROR,
                        message="stages must be a valid JSON array of numbers",
                    )
                ).model_dump(),
            )

    # Validate images
    if current_image is None or current_image.filename == "":
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.VALIDATION_ERROR,
                    message="current_image is required",
                )
            ).model_dump(),
        )

    if ideal_image is None or ideal_image.filename == "":
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.VALIDATION_ERROR,
                    message="ideal_image is required",
                )
            ).model_dump(),
        )

    try:
        # Read and validate images
        current_data = await current_image.read()
        ideal_data = await ideal_image.read()

        if len(current_data) == 0 or len(ideal_data) == 0:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.VALIDATION_ERROR,
                        message="Image files are required",
                    )
                ).model_dump(),
            )

        _, cv2_current = validate_image(current_data)
        _, cv2_ideal = validate_image(ideal_data)

        # Perform staged morphing
        morph_service = get_morphing_service()
        results = morph_service.morph_stages(
            cv2_current,
            cv2_ideal,
            stage_values,
            img1_label="current",
            img2_label="ideal",
        )

        # Convert results
        stage_images = []
        dimensions = None

        for progress, img in results:
            h, w = img.shape[:2]
            if dimensions is None:
                dimensions = ImageDimensions(width=w, height=h)

            image_base64 = cv2_to_base64(img, "png")
            stage_images.append(StageImage(progress=progress, image=image_base64))

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        return StagedMorphResponse(
            data=StagedMorphData(
                images=stage_images,
                format="png",
                dimensions=dimensions or ImageDimensions(width=0, height=0),
            ),
            meta=ResponseMeta(
                request_id=request_id,
                processing_time_ms=processing_time_ms,
            ),
        )

    except ImageValidationError as e:
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
        print(f"Morph stages error: {e}")

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.PROCESSING_ERROR,
                    message="Failed to generate morphed images",
                )
            ).model_dump(),
        )
