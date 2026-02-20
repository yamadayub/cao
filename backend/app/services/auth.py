"""Authentication service for API endpoints."""

from typing import Optional

from fastapi import Header, HTTPException

from app.config import get_settings
from app.models.schemas import ErrorCodes, ErrorDetail, ErrorResponse

# Internal API key user ID prefix
_INTERNAL_USER_PREFIX = "internal_api"


def get_current_user(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
) -> dict:
    """Verify JWT token or API key and return user info.

    Supports two authentication methods:
    1. Bearer JWT token (Clerk) via Authorization header
    2. Internal API key via X-API-Key header (for scripts/automation)

    Args:
        authorization: Bearer token from Authorization header
        x_api_key: API key from X-API-Key header

    Returns:
        dict with user info including 'id' and 'email'

    Raises:
        HTTPException: If no valid credentials provided
    """
    # Try API key authentication first
    if x_api_key:
        settings = get_settings()
        if settings.internal_api_key and x_api_key == settings.internal_api_key:
            return {
                "id": _INTERNAL_USER_PREFIX,
                "email": None,
            }
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCodes.UNAUTHORIZED,
                    message="Invalid API key",
                )
            ).model_dump(),
        )

    # Fall back to Bearer JWT token
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

        # Decode JWT to get user info (Clerk JWT)
        # In production, verify with Clerk's public key
        # For now, we decode without verification for development
        decoded = jwt.decode(token, options={"verify_signature": False})
        user_id = decoded.get("sub")
        if not user_id:
            raise ValueError("No user_id in token")

        return {
            "id": user_id,
            "email": decoded.get("email"),
        }
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
