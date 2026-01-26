"""Generation Jobs API router - async face generation."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models.schemas import (
    CreateGenerationJobRequest,
    CreateGenerationJobResponse,
    ErrorCodes,
    ErrorDetail,
    ErrorResponse,
    GenerationJobStatus,
    GenerationJobStatusResponse,
    GenerationResultData,
    GenerationResultResponse,
)
from app.services.job_queue import JobNotFoundError, get_job_queue_service

router = APIRouter(prefix="/simulations/generate", tags=["generation"])


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime string, handling timezone suffix."""
    if not value:
        return None
    try:
        # Handle 'Z' suffix and '+00:00'
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _job_to_status(job: dict) -> GenerationJobStatus:
    """Convert job record to status response."""
    # For result URL, use the stored base64 directly
    result_url = None
    if job.get("result_image_path") and job.get("status") == "succeeded":
        # Return as data URL for now (or storage URL in future)
        result_path = job["result_image_path"]
        if result_path.startswith("data:") or len(result_path) > 200:
            # It's base64 data, return as-is or as data URL
            result_url = result_path if result_path.startswith("data:") else f"data:image/png;base64,{result_path}"
        else:
            result_url = result_path  # Storage URL

    return GenerationJobStatus(
        job_id=job["id"],
        status=job["status"],
        progress=job.get("progress", 0),
        result_image_url=result_url,
        error=job.get("error"),
        created_at=_parse_datetime(job.get("created_at")) or datetime.utcnow(),
        started_at=_parse_datetime(job.get("started_at")),
        completed_at=_parse_datetime(job.get("completed_at")),
    )


def _verify_optional_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Optionally verify JWT token and return user_id."""
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization.replace("Bearer ", "")

    try:
        import jwt

        decoded = jwt.decode(token, options={"verify_signature": False})
        return decoded.get("sub")
    except Exception:
        return None


@router.post("", response_model=CreateGenerationJobResponse)
async def create_generation_job(
    request: Request,
    data: CreateGenerationJobRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Create a new face generation job.

    The job will be queued and processed asynchronously by the worker.
    Poll the status endpoint to check progress and get results.

    - **base_image**: Base64 encoded base face image (the person)
    - **target_image**: Base64 encoded target/ideal face image (features to apply)
    - **mode**: 'morph' for full face blending, 'parts' for selective features
    - **parts**: For mode='parts', list of parts to blend (eyes, nose, lips, etc.)
    - **strength**: How much to blend toward target (0-1, default 0.5)
    """
    try:
        # Validate parts for parts mode
        if data.mode == "parts":
            if not data.parts or len(data.parts) == 0:
                return JSONResponse(
                    status_code=400,
                    content=ErrorResponse(
                        error=ErrorDetail(
                            code=ErrorCodes.VALIDATION_ERROR,
                            message="Parts list is required for mode='parts'",
                        )
                    ).model_dump(),
                )

        # Get optional user_id
        user_id = _verify_optional_token(authorization)

        # Create job
        job_service = get_job_queue_service()
        job = job_service.create_job(
            mode=data.mode,
            base_image_path=data.base_image,
            target_image_path=data.target_image,
            parts=data.parts,
            strength=data.strength or 0.5,
            seed=data.seed,
            user_id=user_id,
        )

        return CreateGenerationJobResponse(data=_job_to_status(job))

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.INTERNAL_ERROR,
                    message=f"Failed to create job: {str(e)}",
                )
            ).model_dump(),
        )


@router.get("/{job_id}", response_model=GenerationJobStatusResponse)
async def get_job_status(
    request: Request,
    job_id: str,
):
    """
    Get the status of a generation job.

    Returns current status, progress percentage, and result URL when complete.

    Status values:
    - **queued**: Waiting to be processed
    - **running**: Currently being processed
    - **succeeded**: Complete, result available
    - **failed**: Processing failed, see error field
    """
    try:
        job_service = get_job_queue_service()
        job = job_service.get_job(job_id)

        return GenerationJobStatusResponse(data=_job_to_status(job))

    except JobNotFoundError:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.NOT_FOUND,
                    message=f"Job {job_id} not found",
                )
            ).model_dump(),
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.INTERNAL_ERROR,
                    message=f"Failed to get job status: {str(e)}",
                )
            ).model_dump(),
        )


@router.get("/{job_id}/result", response_model=GenerationResultResponse)
async def get_job_result(
    request: Request,
    job_id: str,
):
    """
    Get the result of a completed generation job.

    Returns the generated image as base64 encoded data.
    Only available when job status is 'succeeded'.
    """
    try:
        job_service = get_job_queue_service()
        job = job_service.get_job(job_id)

        if job["status"] != "succeeded":
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.JOB_FAILED,
                        message=f"Job is not complete. Status: {job['status']}",
                        details={"status": job["status"], "error": job.get("error")},
                    )
                ).model_dump(),
            )

        result_path = job.get("result_image_path")
        if not result_path:
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.INTERNAL_ERROR,
                        message="Result image not found",
                    )
                ).model_dump(),
            )

        # Extract base64 data (remove data URL prefix if present)
        image_data = result_path
        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]

        return GenerationResultResponse(
            data=GenerationResultData(
                job_id=job_id,
                image=image_data,
                format="png",
                mode=job["mode"],
                strength=job.get("strength", 0.5),
            )
        )

    except JobNotFoundError:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.NOT_FOUND,
                    message=f"Job {job_id} not found",
                )
            ).model_dump(),
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.INTERNAL_ERROR,
                    message=f"Failed to get result: {str(e)}",
                )
            ).model_dump(),
        )
