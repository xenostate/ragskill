"""
Chat, health, and widget endpoints.
"""

from __future__ import annotations

import asyncio
import time
from urllib.parse import urlparse

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import scripts.config as cfg
from scripts.utils import rate_limit_check
from scripts.rag_core import do_rag_sync, get_site_language_cached

router = APIRouter()


class ChatRequest(BaseModel):
    site_id: int
    query: str = Field(..., max_length=2000)
    session_id: str | None = None
    top_k: int = Field(default=cfg.RAG_TOP_K, le=20)
    origin_domain: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: str


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": cfg.embed_model is not None,
        "llm_enabled": cfg.openai_client is not None,
        "uptime_seconds": round(time.time() - cfg.start_time, 1),
    }


@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    blocked = rate_limit_check(request, "chat", 20, 60)
    if blocked:
        return blocked

    # Domain verification: use the browser-set Origin header (not spoofable by JS)
    origin_header = request.headers.get("origin", "")
    origin_domain = ""
    if origin_header:
        try:
            origin_domain = urlparse(origin_header).hostname or ""
        except Exception:
            pass
    if not origin_domain:
        origin_domain = req.origin_domain or ""

    if origin_domain:
        client = cfg.sb_public or cfg.sb
        site_row = client.table("sites").select("domain, settings").eq("id", req.site_id).execute()
        if site_row.data:
            site_domain = site_row.data[0].get("domain", "")
            site_settings = site_row.data[0].get("settings") or {}
            is_landing = site_settings.get("landing", False)
            is_trial = "trial" in site_domain
            if not is_landing and not is_trial and origin_domain not in ("wrs.kz", "localhost"):
                if origin_domain != site_domain and not origin_domain.endswith(f".{site_domain}"):
                    cfg.log.warning(f"Domain mismatch: site {req.site_id} domain={site_domain} origin={origin_domain}")
                    return JSONResponse({"error": "Widget not authorized for this domain"}, status_code=403)

    t0 = time.time()
    language = get_site_language_cached(req.site_id)

    result = await asyncio.to_thread(
        do_rag_sync, req.site_id, req.query, req.top_k, req.session_id, language
    )

    cfg.log.info(
        f"chat site={req.site_id} q=\"{req.query[:50]}\" "
        f"confidence={result['confidence']} chunks={len(result['sources'])} "
        f"time={time.time()-t0:.2f}s"
    )

    return ChatResponse(**result)


@router.get("/widget.js")
async def serve_widget():
    js_path = cfg.WIDGET_DIR / "widget.js"
    if not js_path.exists():
        return JSONResponse({"error": "widget.js not found"}, status_code=404)
    return FileResponse(js_path, media_type="application/javascript")


# ── Telegram webhook (DISABLED) ─────────────────────────────────────────

@router.post("/api/telegram")
async def telegram_webhook(request: Request):
    return JSONResponse({"error": "Telegram handler is disabled"}, status_code=503)


@router.post("/api/telegram/set-webhook")
async def set_telegram_webhook(request: Request):
    return JSONResponse({"error": "Telegram handler is disabled"}, status_code=503)
