"""
WhatsApp webhook endpoints (dormant — activate via WHATSAPP_ENABLED=true).
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import scripts.config as cfg
from scripts.utils import rate_limit_check, verify_admin_token
from scripts.rag_core import get_site_language_cached

router = APIRouter()


@router.get("/api/whatsapp/webhook")
async def whatsapp_verify(request: Request):
    """Meta webhook verification (challenge-response handshake)."""
    if not cfg.WHATSAPP_ENABLED or cfg.whatsapp_handler is None:
        return JSONResponse({"error": "WhatsApp handler is disabled"}, status_code=503)

    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    result = cfg.whatsapp_handler.verify_webhook(mode, token, challenge)
    if result is not None:
        return JSONResponse(content=int(result))
    return JSONResponse({"error": "Verification failed"}, status_code=403)


@router.post("/api/whatsapp/webhook")
async def whatsapp_incoming(request: Request):
    """Receive inbound WhatsApp messages from BSP."""
    if not cfg.WHATSAPP_ENABLED or cfg.whatsapp_handler is None:
        return JSONResponse({"error": "WhatsApp handler is disabled"}, status_code=503)

    blocked = rate_limit_check(request, "whatsapp", 100, 60)
    if blocked:
        return blocked

    body = await request.body()

    if cfg.WHATSAPP_APP_SECRET:
        from scripts.whatsapp_handler import WhatsAppHandler
        signature = request.headers.get("x-hub-signature-256", "")
        if not WhatsAppHandler.verify_signature(body, signature, cfg.WHATSAPP_APP_SECRET):
            cfg.log.warning("WhatsApp webhook signature verification failed")
            return JSONResponse({"error": "Invalid signature"}, status_code=403)

    payload = json.loads(body)

    asyncio.get_event_loop().create_task(
        asyncio.to_thread(
            cfg.whatsapp_handler.handle_webhook,
            payload,
            cfg._session_history,
            cfg._session_lock,
            get_site_language_cached,
        )
    )

    return JSONResponse({"status": "ok"})


@router.post("/api/whatsapp/register")
async def whatsapp_register(request: Request):
    """Admin endpoint: register a WhatsApp business number for a site."""
    if not cfg.WHATSAPP_ENABLED or cfg.whatsapp_handler is None:
        return JSONResponse({"error": "WhatsApp handler is disabled"}, status_code=503)

    admin_token = request.headers.get("x-admin-token", "")
    if not verify_admin_token(admin_token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    data = await request.json()
    site_id = data.get("site_id")
    phone_number = data.get("phone_number")
    display_name = data.get("display_name", "")
    api_token = data.get("api_token")
    provider = data.get("provider", "360dialog")

    if not all([site_id, phone_number, api_token]):
        return JSONResponse(
            {"error": "Required fields: site_id, phone_number, api_token"},
            status_code=400,
        )

    try:
        account = cfg.whatsapp_handler.register_account(
            site_id, phone_number, display_name, api_token, provider,
        )
        return JSONResponse({"ok": True, "account": account})
    except Exception as e:
        cfg.log.error(f"WhatsApp register error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
