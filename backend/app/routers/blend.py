"""Blend API endpoints."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.models.schemas import (
    ErrorDetail,
    ErrorResponse,
    ImageDimensions,
    PartsBlendData,
    PartsBlendResponse,
)
from app.services.part_blender import PartsSelection, get_part_blender_service
from app.services.part_blender_3d import get_part_blender_3d_service
from app.utils.image import ImageValidationError, cv2_to_base64, validate_image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/blend", tags=["blend"])


@router.post(
    "/parts",
    response_model=PartsBlendResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation or processing error"},
    },
)
async def blend_parts(
    request: Request,
    current_image: Optional[UploadFile] = File(None),
    ideal_image: Optional[UploadFile] = File(None),
    parts: Optional[str] = Form(None),
    method: Optional[str] = Form("auto"),
) -> JSONResponse:
    """
    Blend selected facial parts from ideal image onto current image.

    Args:
        current_image: Current face image (base image)
        ideal_image: Ideal face image (source of parts)
        parts: JSON string specifying which parts to blend
        method: Blending method - "2d", "3d", or "auto" (default: "auto")
                "auto" uses 3D if available, falls back to 2D

    Returns:
        Blended image with selected parts applied
    """
    # Validate required files
    if current_image is None:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="current_image is required",
                )
            ).model_dump(),
        )

    if ideal_image is None:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="ideal_image is required",
                )
            ).model_dump(),
        )

    if parts is None:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="parts is required",
                )
            ).model_dump(),
        )

    # Parse parts JSON
    try:
        parts_dict = json.loads(parts)
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="Invalid parts JSON format",
                )
            ).model_dump(),
        )

    # Validate parts selection
    try:
        parts_selection = PartsSelection(**parts_dict)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message=f"Invalid parts selection format: {str(e)}",
                )
            ).model_dump(),
        )

    # Check if at least one part is selected
    if not parts_selection.has_any_selection():
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="At least one part must be selected",
                )
            ).model_dump(),
        )

    # Read and validate images
    try:
        current_data = await current_image.read()
        _, current_img = validate_image(current_data)
    except ImageValidationError as e:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=e.code,
                    message=e.message,
                    details=e.details,
                )
            ).model_dump(),
        )

    try:
        ideal_data = await ideal_image.read()
        _, ideal_img = validate_image(ideal_data)
    except ImageValidationError as e:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=e.code,
                    message=e.message,
                    details=e.details,
                )
            ).model_dump(),
        )

    # Perform blending - choose method
    try:
        use_3d = False

        if method == "3d":
            use_3d = True
        elif method == "auto":
            # Auto mode: try 3D if available
            blender_3d = get_part_blender_3d_service()
            use_3d = blender_3d.is_depth_available()

        if use_3d:
            logger.info("Using 3D blending method")
            blender = get_part_blender_3d_service()
        else:
            logger.info("Using 2D blending method")
            blender = get_part_blender_service()

        result_img = blender.blend(
            current_img,
            ideal_img,
            parts_selection,
            current_label="current",
            ideal_label="ideal",
        )
    except ImageValidationError as e:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=e.code,
                    message=e.message,
                    details=e.details,
                )
            ).model_dump(),
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message=str(e),
                )
            ).model_dump(),
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="PROCESSING_ERROR",
                    message="Failed to generate blended image",
                    details={"error": str(e)},
                )
            ).model_dump(),
        )

    # Encode result
    h, w = result_img.shape[:2]
    image_base64 = cv2_to_base64(result_img, "png")

    return JSONResponse(
        status_code=200,
        content=PartsBlendResponse(
            data=PartsBlendData(
                image=image_base64,
                format="png",
                applied_parts=parts_selection.get_selected_parts(),
                dimensions=ImageDimensions(width=w, height=h),
            )
        ).model_dump(),
    )
