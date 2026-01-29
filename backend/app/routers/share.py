"""SNS Share API router."""

import base64
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
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
from app.services.share_image_generator import get_share_image_generator
from app.services.supabase_client import get_supabase_client

router = APIRouter(prefix="/share", tags=["share"])

# Share expiration period
SHARE_EXPIRATION_DAYS = 30

# In-memory fallback storage (when Supabase is not configured)
_shares_store: dict = {}


def _get_base_urls() -> tuple[str, str]:
    """Get base URLs for share and storage."""
    settings = get_settings()
    if settings.is_production:
        base_url = "https://cao.app"
        storage_base = settings.supabase_url.replace(".supabase.co", ".supabase.co/storage/v1/object/public") if settings.supabase_url else "https://storage.cao.app"
    else:
        base_url = "http://localhost:3000"
        storage_base = settings.supabase_url.replace(".supabase.co", ".supabase.co/storage/v1/object/public") if settings.supabase_url else "http://localhost:3000/storage"
    return base_url, storage_base


async def _upload_to_supabase(
    image_bytes: bytes,
    share_id: str,
    filename: str,
) -> Optional[str]:
    """Upload image to Supabase storage.

    Returns the public URL if successful, None otherwise.
    """
    client = get_supabase_client()
    if client is None:
        return None

    try:
        settings = get_settings()
        bucket_name = "shares"
        file_path = f"{share_id}/{filename}"

        # Upload to Supabase storage
        client.storage.from_(bucket_name).upload(
            file_path,
            image_bytes,
            file_options={"content-type": "image/png"}
        )

        # Get public URL
        storage_base = settings.supabase_url.replace(
            ".supabase.co",
            ".supabase.co/storage/v1/object/public"
        )
        return f"{storage_base}/{bucket_name}/{file_path}"
    except Exception as e:
        print(f"Failed to upload to Supabase: {e}")
        return None


async def _save_share_to_database(
    share_id: str,
    user_id: str,
    share_data: dict,
) -> bool:
    """Save share metadata to Supabase database.

    Returns True if successful, False otherwise.
    """
    client = get_supabase_client()
    if client is None:
        return False

    try:
        client.table("sns_shares").insert({
            "id": share_id,
            "user_id": user_id,
            "simulation_id": share_data.get("simulation_id"),
            "template": share_data["template"],
            "caption": share_data.get("caption"),
            "applied_parts": share_data.get("applied_parts"),
            "share_image_url": share_data["share_image_url"],
            "og_image_url": share_data["og_image_url"],
            "expires_at": share_data["expires_at"].isoformat(),
        }).execute()
        return True
    except Exception as e:
        print(f"Failed to save to database: {e}")
        return False


async def _get_share_from_database(share_id: str) -> Optional[dict]:
    """Get share from Supabase database.

    Returns share data if found, None otherwise.
    """
    client = get_supabase_client()
    if client is None:
        return None

    try:
        result = client.table("sns_shares").select("*").eq("id", share_id).single().execute()
        if result.data:
            # Convert ISO string back to datetime
            data = result.data
            data["expires_at"] = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
            data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            return data
        return None
    except Exception as e:
        print(f"Failed to get from database: {e}")
        return None


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

    # Get base URLs
    base_url, storage_base = _get_base_urls()

    # Generate share image
    try:
        generator = get_share_image_generator()
        share_image_bytes = generator.generate(
            source_image=body.source_image,
            result_image=body.result_image,
            template=body.template,
            caption=body.caption,
            applied_parts=body.applied_parts,
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.PROCESSING_ERROR,
                    message=f"Failed to generate share image: {str(e)}",
                )
            ).model_dump(),
        )

    # Try to upload to Supabase storage
    share_image_url = await _upload_to_supabase(share_image_bytes, share_id, "share.png")
    og_image_url = await _upload_to_supabase(share_image_bytes, share_id, "og.png")

    # Fallback to base64 data URL if Supabase is not available
    if not share_image_url:
        share_image_b64 = base64.b64encode(share_image_bytes).decode("utf-8")
        share_image_url = f"data:image/png;base64,{share_image_b64}"
    if not og_image_url:
        og_image_b64 = base64.b64encode(share_image_bytes).decode("utf-8")
        og_image_url = f"data:image/png;base64,{og_image_b64}"

    share_url = f"{base_url}/share/{share_id}"

    # Prepare share data
    share_data = {
        "share_id": share_id,
        "user_id": user["id"],
        "simulation_id": body.simulation_id,
        "source_image": body.source_image[:100] + "..." if len(body.source_image) > 100 else body.source_image,  # Don't store full image
        "result_image": body.result_image[:100] + "..." if len(body.result_image) > 100 else body.result_image,
        "template": body.template,
        "caption": body.caption,
        "applied_parts": body.applied_parts,
        "share_image_url": share_image_url,
        "og_image_url": og_image_url,
        "created_at": created_at,
        "expires_at": expires_at,
    }

    # Try to save to database
    saved = await _save_share_to_database(share_id, user["id"], share_data)

    # Fallback to in-memory storage if database not available
    if not saved:
        _shares_store[share_id] = share_data

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
    # Try to get from database first
    share = await _get_share_from_database(share_id)

    # Fallback to in-memory storage
    if not share:
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
    expires_at = share["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))

    now = datetime.utcnow()
    # Make both timezone-naive for comparison
    if expires_at.tzinfo is not None:
        expires_at = expires_at.replace(tzinfo=None)
    is_expired = now > expires_at

    if is_expired:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.NOT_FOUND,
                    message="Share has expired",
                    details={"expired_at": expires_at.isoformat()},
                )
            ).model_dump(),
        )

    # Get created_at
    created_at = share.get("created_at", datetime.utcnow())
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

    return GetSnsShareResponse(
        data=GetSnsShareData(
            share_id=share.get("share_id") or share.get("id"),
            share_image_url=share["share_image_url"],
            caption=share.get("caption"),
            template=share["template"],
            created_at=created_at,
            expires_at=expires_at,
            is_expired=is_expired,
        )
    )
