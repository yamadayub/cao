"""Supabase client service for database operations."""

from functools import lru_cache
from typing import Optional

from supabase import Client, create_client

from app.config import get_settings


@lru_cache
def get_supabase_client() -> Optional[Client]:
    """Get cached Supabase client instance."""
    settings = get_settings()

    if not settings.supabase_url or not settings.supabase_service_key:
        return None

    return create_client(settings.supabase_url, settings.supabase_service_key)


def get_supabase() -> Client:
    """Get Supabase client, raise error if not configured."""
    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY.")
    return client
