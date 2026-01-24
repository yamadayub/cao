"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_env: str = "development"
    debug: bool = True
    api_version: str = "1.0.0"

    # CORS
    cors_origins: str = "http://localhost:3000"

    # Supabase (optional for MVP)
    supabase_url: str = ""
    supabase_service_key: str = ""

    # Rate Limiting
    rate_limit_per_minute_unauthenticated: int = 10
    rate_limit_per_minute_authenticated: int = 30

    # Image Processing
    max_image_size_mb: int = 10
    max_image_dimension: int = 2048

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def max_image_size_bytes(self) -> int:
        """Max image size in bytes."""
        return self.max_image_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
