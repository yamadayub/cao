"""Authentication service for API endpoints."""

from typing import Optional

from fastapi import Header, HTTPException
from fastapi.responses import JSONResponse

from app.models.schemas import ErrorCodes, ErrorDetail, ErrorResponse


def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """Verify JWT token and return user info.

    Args:
        authorization: Bearer token from Authorization header

    Returns:
        dict with user info including 'id' and 'email'

    Raises:
        HTTPException: If token is missing or invalid
    """
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
