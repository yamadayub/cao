"""Health check endpoint."""

from datetime import datetime

from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import HealthData, HealthResponse

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the current status of the API including version and timestamp.
    """
    return HealthResponse(
        data=HealthData(
            status="ok",
            version=settings.api_version,
            timestamp=datetime.utcnow(),
        )
    )
