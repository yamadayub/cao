"""Clerk Webhook endpoint for syncing user data to Supabase."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from svix.webhooks import Webhook, WebhookVerificationError

from app.config import get_settings
from app.services.encryption import encrypt_pii
from app.services.supabase_client import get_supabase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def _extract_line_account(external_accounts: list) -> dict:
    """Extract LINE OAuth account info from Clerk external_accounts."""
    for account in external_accounts:
        if account.get("provider") == "oauth_line":
            return {
                "line_user_id": account.get("provider_user_id", ""),
                "line_display_name": (
                    account.get("username")
                    or account.get("first_name")
                    or ""
                ),
            }
    return {}


def _build_external_accounts_summary(external_accounts: list) -> list:
    """Build a summary of external accounts (provider type + id only)."""
    return [
        {"provider": a.get("provider", ""), "id": a.get("provider_user_id", "")}
        for a in external_accounts
    ]


def _upsert_profile(data: dict) -> None:
    """Upsert a profile row from Clerk user data."""
    supabase = get_supabase()
    user_id = data["id"]

    # Email
    email_addresses = data.get("email_addresses") or []
    primary_email = email_addresses[0]["email_address"] if email_addresses else None

    # Phone
    phone_numbers = data.get("phone_numbers") or []
    primary_phone = phone_numbers[0]["phone_number"] if phone_numbers else None

    # External accounts (LINE etc.)
    external_accounts = data.get("external_accounts") or []
    line_info = _extract_line_account(external_accounts)

    # last_sign_in_at (Clerk sends as Unix ms timestamp)
    last_sign_in_at = None
    raw_sign_in = data.get("last_sign_in_at")
    if raw_sign_in:
        last_sign_in_at = datetime.fromtimestamp(
            raw_sign_in / 1000, tz=timezone.utc
        ).isoformat()

    profile = {
        "id": user_id,
        "email_encrypted": encrypt_pii(primary_email) if primary_email else None,
        "phone_encrypted": encrypt_pii(primary_phone) if primary_phone else None,
        "first_name": data.get("first_name") or None,
        "last_name": data.get("last_name") or None,
        "clerk_external_accounts": _build_external_accounts_summary(external_accounts),
        "last_sign_in_at": last_sign_in_at,
        "clerk_synced_at": datetime.now(timezone.utc).isoformat(),
    }

    # LINE-specific fields
    if line_info:
        line_uid = line_info.get("line_user_id")
        profile["line_user_id_encrypted"] = encrypt_pii(line_uid) if line_uid else None
        profile["line_display_name"] = line_info.get("line_display_name") or None

    supabase.table("profiles").upsert(profile, on_conflict="id").execute()
    logger.info("Upserted profile for user %s", user_id)


def _delete_profile(data: dict) -> None:
    """Delete a profile row."""
    supabase = get_supabase()
    user_id = data.get("id")
    if not user_id:
        logger.warning("user.deleted event missing user id")
        return
    supabase.table("profiles").delete().eq("id", user_id).execute()
    logger.info("Deleted profile for user %s", user_id)


@router.post("/clerk")
async def clerk_webhook(request: Request):
    """Receive Clerk webhook events and sync user data to Supabase."""
    settings = get_settings()

    if not settings.clerk_webhook_secret:
        logger.error("CLERK_WEBHOOK_SECRET is not configured")
        return JSONResponse(
            status_code=500,
            content={"error": "Webhook secret not configured"},
        )

    # Read raw body for signature verification
    body = await request.body()
    headers = dict(request.headers)

    # Verify svix signature
    try:
        wh = Webhook(settings.clerk_webhook_secret)
        payload = wh.verify(body, headers)
    except WebhookVerificationError:
        logger.warning("Webhook signature verification failed")
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid webhook signature"},
        )

    event_type = payload.get("type")
    data = payload.get("data", {})

    logger.info("Received Clerk webhook: %s", event_type)

    try:
        if event_type in ("user.created", "user.updated"):
            _upsert_profile(data)
        elif event_type == "user.deleted":
            _delete_profile(data)
        else:
            logger.info("Ignoring unhandled event type: %s", event_type)
    except Exception:
        logger.exception("Error processing webhook event %s", event_type)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )

    return JSONResponse(status_code=200, content={"status": "ok"})
