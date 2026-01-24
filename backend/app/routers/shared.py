"""Shared simulations API router - public access."""

from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.models.schemas import (
    ErrorCodes,
    ErrorDetail,
    ErrorResponse,
    ResultImageItem,
    SharedSimulationData,
    SharedSimulationResponse,
)
from app.services.supabase_client import get_supabase

router = APIRouter(prefix="/shared", tags=["shared"])


@router.get("/{token}", response_model=SharedSimulationResponse)
async def get_shared_simulation(
    request: Request,
    token: str,
):
    """Get a shared simulation by token (public access)."""
    try:
        supabase = get_supabase()

        # Find simulation by share token
        result = (
            supabase.table("simulations")
            .select("result_images, created_at, is_public")
            .eq("share_token", token)
            .eq("is_public", True)
            .execute()
        )

        if not result.data or len(result.data) == 0:
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.NOT_FOUND,
                        message="Shared simulation not found",
                    )
                ).model_dump(),
            )

        sim = result.data[0]
        return SharedSimulationResponse(
            data=SharedSimulationData(
                result_images=[
                    ResultImageItem(progress=img["progress"], image=img["path"])
                    for img in sim["result_images"]
                ],
                created_at=datetime.fromisoformat(sim["created_at"].replace("Z", "+00:00")),
            )
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.INTERNAL_ERROR,
                    message=f"Failed to get shared simulation: {str(e)}",
                )
            ).model_dump(),
        )
