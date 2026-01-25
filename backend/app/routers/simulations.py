"""Simulations API router - requires authentication."""

import secrets
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models.schemas import (
    CreateSimulationRequest,
    DeletedData,
    DeleteResponse,
    ErrorCodes,
    ErrorDetail,
    ErrorResponse,
    PaginationInfo,
    ResultImageItem,
    ShareData,
    ShareResponse,
    SimulationData,
    SimulationListData,
    SimulationListResponse,
    SimulationResponse,
    SimulationSummary,
)
from app.services.supabase_client import get_supabase

router = APIRouter(prefix="/simulations", tags=["simulations"])
settings = get_settings()


def verify_token(authorization: Optional[str] = Header(None)) -> str:
    """Verify JWT token and return user_id."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.UNAUTHORIZED,
                    message="Authentication required",
                )
            ).model_dump(),
        )

    token = authorization.replace("Bearer ", "")

    try:
        import jwt

        # Decode JWT to get user_id (Clerk JWT)
        # In production, verify with Clerk's public key
        # For now, we decode without verification for development
        decoded = jwt.decode(token, options={"verify_signature": False})
        user_id = decoded.get("sub")
        if not user_id:
            raise ValueError("No user_id in token")
        return user_id
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.UNAUTHORIZED,
                    message=f"Invalid token: {str(e)}",
                )
            ).model_dump(),
        ) from e


@router.post("", response_model=SimulationResponse)
async def create_simulation(
    request: Request,
    data: CreateSimulationRequest,
    user_id: str = Depends(verify_token),
):
    """Create a new simulation."""
    try:
        supabase = get_supabase()
        simulation_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # Store images as base64 for MVP (in production, upload to storage)
        result_images_json = [
            {"progress": img.progress, "path": img.image} for img in data.result_images
        ]

        # Insert into database
        insert_data = {
            "id": simulation_id,
            "user_id": user_id,
            "current_image_path": data.current_image,
            "ideal_image_path": data.ideal_image,
            "result_images": result_images_json,
            "settings": data.settings.model_dump() if data.settings else {},
            "is_public": False,
            "created_at": now,
            "updated_at": now,
        }

        result = supabase.table("simulations").insert(insert_data).execute()

        if not result.data:
            raise Exception("Failed to insert simulation")

        sim = result.data[0]

        return SimulationResponse(
            data=SimulationData(
                id=sim["id"],
                user_id=sim["user_id"],
                current_image_url=sim["current_image_path"],
                ideal_image_url=sim["ideal_image_path"],
                result_images=[
                    ResultImageItem(progress=img["progress"], image=img["path"])
                    for img in sim["result_images"]
                ],
                settings=sim["settings"] or {},
                share_token=sim.get("share_token"),
                is_public=sim["is_public"],
                created_at=datetime.fromisoformat(sim["created_at"].replace("Z", "+00:00")),
                updated_at=datetime.fromisoformat(sim["updated_at"].replace("Z", "+00:00")),
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.INTERNAL_ERROR,
                    message=f"Failed to create simulation: {str(e)}",
                )
            ).model_dump(),
        )


@router.get("", response_model=SimulationListResponse)
async def list_simulations(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(verify_token),
):
    """List user's simulations."""
    try:
        supabase = get_supabase()

        # Get total count
        count_result = (
            supabase.table("simulations")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .execute()
        )
        total = count_result.count or 0

        # Get simulations with pagination
        result = (
            supabase.table("simulations")
            .select("id, result_images, is_public, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )

        simulations = []
        for sim in result.data or []:
            # Get thumbnail from 50% progress result
            result_images = sim.get("result_images", [])
            thumbnail_url = ""
            for img in result_images:
                if img.get("progress") == 0.5:
                    thumbnail_url = img.get("path", "")
                    break
            if not thumbnail_url and result_images:
                thumbnail_url = result_images[len(result_images) // 2].get("path", "")

            simulations.append(
                SimulationSummary(
                    id=sim["id"],
                    thumbnail_url=thumbnail_url,
                    created_at=datetime.fromisoformat(sim["created_at"].replace("Z", "+00:00")),
                    is_public=sim["is_public"],
                )
            )

        return SimulationListResponse(
            data=SimulationListData(
                simulations=simulations,
                pagination=PaginationInfo(
                    total=total,
                    limit=limit,
                    offset=offset,
                    has_more=offset + limit < total,
                ),
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.INTERNAL_ERROR,
                    message=f"Failed to list simulations: {str(e)}",
                )
            ).model_dump(),
        )


@router.get("/{simulation_id}", response_model=SimulationResponse)
async def get_simulation(
    request: Request,
    simulation_id: str,
    user_id: str = Depends(verify_token),
):
    """Get a specific simulation."""
    try:
        supabase = get_supabase()

        result = (
            supabase.table("simulations")
            .select("*")
            .eq("id", simulation_id)
            .eq("user_id", user_id)
            .execute()
        )

        if not result.data or len(result.data) == 0:
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.NOT_FOUND,
                        message="Simulation not found",
                    )
                ).model_dump(),
            )

        sim = result.data[0]
        return SimulationResponse(
            data=SimulationData(
                id=sim["id"],
                user_id=sim["user_id"],
                current_image_url=sim["current_image_path"],
                ideal_image_url=sim["ideal_image_path"],
                result_images=[
                    ResultImageItem(progress=img["progress"], image=img["path"])
                    for img in sim["result_images"]
                ],
                settings=sim["settings"] or {},
                share_token=sim.get("share_token"),
                is_public=sim["is_public"],
                created_at=datetime.fromisoformat(sim["created_at"].replace("Z", "+00:00")),
                updated_at=datetime.fromisoformat(sim["updated_at"].replace("Z", "+00:00")),
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.INTERNAL_ERROR,
                    message=f"Failed to get simulation: {str(e)}",
                )
            ).model_dump(),
        )


@router.delete("/{simulation_id}", response_model=DeleteResponse)
async def delete_simulation(
    request: Request,
    simulation_id: str,
    user_id: str = Depends(verify_token),
):
    """Delete a simulation."""
    try:
        supabase = get_supabase()

        # Check if simulation exists and belongs to user
        check_result = (
            supabase.table("simulations")
            .select("id")
            .eq("id", simulation_id)
            .eq("user_id", user_id)
            .execute()
        )

        if not check_result.data or len(check_result.data) == 0:
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.NOT_FOUND,
                        message="Simulation not found",
                    )
                ).model_dump(),
            )

        # Delete
        supabase.table("simulations").delete().eq("id", simulation_id).execute()

        return DeleteResponse(
            data=DeletedData(
                deleted=True,
                id=simulation_id,
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.INTERNAL_ERROR,
                    message=f"Failed to delete simulation: {str(e)}",
                )
            ).model_dump(),
        )


@router.post("/{simulation_id}/share", response_model=ShareResponse)
async def share_simulation(
    request: Request,
    simulation_id: str,
    user_id: str = Depends(verify_token),
):
    """Generate a share URL for a simulation."""
    try:
        supabase = get_supabase()

        # Check if simulation exists and belongs to user
        check_result = (
            supabase.table("simulations")
            .select("id, share_token")
            .eq("id", simulation_id)
            .eq("user_id", user_id)
            .execute()
        )

        if not check_result.data or len(check_result.data) == 0:
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code=ErrorCodes.NOT_FOUND,
                        message="Simulation not found",
                    )
                ).model_dump(),
            )

        # Use existing token or generate new one
        share_token = check_result.data[0].get("share_token")
        if not share_token:
            share_token = secrets.token_urlsafe(16)

            # Update simulation with share token and make public
            supabase.table("simulations").update(
                {
                    "share_token": share_token,
                    "is_public": True,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            ).eq("id", simulation_id).execute()

        # Build share URL (use frontend URL from CORS origins)
        frontend_url = settings.cors_origins_list[0] if settings.cors_origins_list else "https://cao.app"
        share_url = f"{frontend_url}/s/{share_token}"

        return ShareResponse(
            data=ShareData(
                share_token=share_token,
                share_url=share_url,
                expires_at=None,
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.INTERNAL_ERROR,
                    message=f"Failed to share simulation: {str(e)}",
                )
            ).model_dump(),
        )
