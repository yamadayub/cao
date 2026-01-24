"""FastAPI Application Entry Point."""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings
from app.models.schemas import ErrorDetail, ErrorResponse
from app.routers import analyze, health, morph

settings = get_settings()

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    print(f"Starting Cao Backend API v{settings.api_version}")
    print(f"Environment: {settings.app_env}")
    yield
    # Shutdown
    print("Shutting down Cao Backend API")


app = FastAPI(
    title="Cao API",
    description="AI Face Analysis and Morphing API",
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# State for rate limiter
app.state.limiter = limiter

# Add rate limit exceeded handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Custom rate limit exceeded handler with our response format
@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Handle rate limit exceeded with custom response format."""
    return JSONResponse(
        status_code=429,
        content=ErrorResponse(
            error=ErrorDetail(
                code="RATE_LIMITED",
                message="Too many requests. Please try again later.",
                details={"retry_after": str(exc.detail)},
            )
        ).model_dump(),
    )


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def add_request_metadata(request: Request, call_next) -> Response:
    """Add request ID and processing time to requests."""
    # Get or generate request ID
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    request.state.start_time = time.time()

    response = await call_next(request)

    # Add headers
    response.headers["X-Request-ID"] = request_id
    processing_time = (time.time() - request.state.start_time) * 1000
    response.headers["X-Processing-Time-Ms"] = f"{processing_time:.2f}"

    return response


# Include routers
app.include_router(health.router)
app.include_router(analyze.router, prefix="/api/v1")
app.include_router(morph.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint - redirect to docs or return basic info."""
    return {
        "name": "Cao API",
        "version": settings.api_version,
        "docs": "/docs" if settings.debug else None,
    }
