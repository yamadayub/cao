"""SNS Share API router."""

import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.models.schemas import (
    CreateSnsShareRequest,
    CreateSnsShareResponse,
    ErrorCodes,
    ErrorDetail,
    ErrorResponse,
    GetSnsShareData,
    GetSnsShareResponse,
    SnsShareData,
)
from app.services.auth import get_current_user

router = APIRouter(prefix="/share", tags=["share"])

# In-memory storage for MVP (replace with database in production)
_shares_store: dict = {}

# Share expiration period
SHARE_EXPIRATION_DAYS = 30


@router.post("/create", response_model=CreateSnsShareResponse)
async def create_share(
    request: Request,
    body: CreateSnsShareRequest,
    user: dict = Depends(get_current_user),
):
    """Create a new SNS share image.

    Requires authentication. Creates a shareable image with the selected template
    and returns URLs for sharing on social media.
    """
    # Validate caption length
    if body.caption and len(body.caption) > 140:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.VALIDATION_ERROR,
                    message="Caption must be 140 characters or less",
                )
            ).model_dump(),
        )

    # Generate share ID
    share_id = str(uuid.uuid4())

    # Calculate expiration
    created_at = datetime.utcnow()
    expires_at = created_at + timedelta(days=SHARE_EXPIRATION_DAYS)

    # TODO: Generate actual share image using ShareImageGenerator service
    # For MVP, we'll use placeholder URLs
    base_url = "https://cao.app"
    storage_base = "https://storage.cao.app"

    share_url = f"{base_url}/share/{share_id}"
    share_image_url = f"{storage_base}/shares/{share_id}/share.png"
    og_image_url = f"{storage_base}/shares/{share_id}/og.png"

    # Store share data (in-memory for MVP)
    _shares_store[share_id] = {
        "share_id": share_id,
        "user_id": user["id"],
        "simulation_id": body.simulation_id,
        "source_image": body.source_image,
        "result_image": body.result_image,
        "template": body.template,
        "caption": body.caption,
        "applied_parts": body.applied_parts,
        "share_image_url": share_image_url,
        "og_image_url": og_image_url,
        "created_at": created_at,
        "expires_at": expires_at,
    }

    return CreateSnsShareResponse(
        data=SnsShareData(
            share_id=share_id,
            share_url=share_url,
            share_image_url=share_image_url,
            og_image_url=og_image_url,
            expires_at=expires_at,
        )
    )


@router.get("/{share_id}", response_model=GetSnsShareResponse)
async def get_share(
    request: Request,
    share_id: str,
):
    """Get share data by ID (public access).

    Returns share details including the image URL and metadata.
    Does not require authentication.
    """
    # Check if share exists
    share = _shares_store.get(share_id)

    if not share:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.NOT_FOUND,
                    message="Share not found",
                )
            ).model_dump(),
        )

    # Check if expired
    now = datetime.utcnow()
    is_expired = now > share["expires_at"]

    if is_expired:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.NOT_FOUND,
                    message="Share has expired",
                    details={"expired_at": share["expires_at"].isoformat()},
                )
            ).model_dump(),
        )

    return GetSnsShareResponse(
        data=GetSnsShareData(
            share_id=share["share_id"],
            share_image_url=share["share_image_url"],
            caption=share["caption"],
            template=share["template"],
            created_at=share["created_at"],
            expires_at=share["expires_at"],
            is_expired=is_expired,
        )
    )
