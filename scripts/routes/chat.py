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
from scripts.assistant_features import get_public_assistant_config, submit_assistant_form
from scripts.utils import rate_limit_check, get_client_ip, parse_user_agent
from scripts.rag_core import do_rag_sync, get_site_language_cached

router = APIRouter()


class ChatRequest(BaseModel):
    site_id: int
    query: str = Field(..., max_length=2000)
    session_id: str | None = None
    top_k: int = Field(default=cfg.RAG_TOP_K, le=20)
    origin_domain: str | None = None  # kept for backwards compat, not trusted


class TrackRequest(BaseModel):
    site_id: int
    session_id: str = Field(..., max_length=128)
    referer: str | None = Field(default=None, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: str


class WidgetConfigResponse(BaseModel):
    assistant: dict


class FormSubmitRequest(BaseModel):
    site_id: int
    form_id: str = Field(..., max_length=128)
    values: dict = Field(default_factory=dict)
    session_id: str | None = Field(default=None, max_length=128)
    page_url: str | None = Field(default=None, max_length=2000)


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": cfg.embed_model is not None,
        "llm_enabled": cfg.openai_client is not None,
        "uptime_seconds": round(time.time() - cfg.start_time, 1),
    }


def _extract_origin_domain(request: Request) -> str:
    origin_header = request.headers.get("origin", "")
    if origin_header:
        try:
            return urlparse(origin_header).hostname or ""
        except Exception:
            return ""

    referer = request.headers.get("referer", "")
    if referer:
        try:
            return urlparse(referer).hostname or ""
        except Exception:
            return ""
    return ""


def _authorize_site_request(site_id: int, request: Request):
    """Load a site row and enforce origin checks for embedded widget traffic."""
    client = cfg.sb_public or cfg.sb
    site_row = client.table("sites").select("id, domain, settings").eq("id", site_id).execute()
    if not site_row.data:
        return None, JSONResponse({"error": "Site not found"}, status_code=404)

    site = site_row.data[0]
    site_domain = site.get("domain", "")
    site_settings = site.get("settings") or {}
    is_landing = site_settings.get("landing", False)
    is_trial = "trial" in site_domain
    is_internal = site_settings.get("internal_assistant", False)

    if not is_landing and not is_trial and not is_internal:
        origin_domain = _extract_origin_domain(request)
        if not origin_domain:
            cfg.log.warning(f"No Origin/Referer header for site {site_id} — rejecting")
            return None, JSONResponse({"error": "Origin header required"}, status_code=403)
        if origin_domain not in ("wrs.kz", "localhost"):
            if origin_domain != site_domain and not origin_domain.endswith(f".{site_domain}"):
                cfg.log.warning(f"Domain mismatch: site {site_id} domain={site_domain} origin={origin_domain}")
                return None, JSONResponse({"error": "Widget not authorized for this domain"}, status_code=403)

    return site, None


@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    blocked = rate_limit_check(request, "chat", 20, 60)
    if blocked:
        return blocked

    site, error = _authorize_site_request(req.site_id, request)
    if error:
        return error

    t0 = time.time()
    language = get_site_language_cached(req.site_id)

    result = await asyncio.to_thread(
        do_rag_sync, req.site_id, req.query, req.top_k, req.session_id, language
    )

    elapsed_ms = int((time.time() - t0) * 1000)
    cfg.log.info(
        f"chat site={req.site_id} q=\"{req.query[:50]}\" "
        f"confidence={result['confidence']} chunks={len(result['sources'])} "
        f"time={elapsed_ms}ms"
    )

    # Fire-and-forget analytics log (never blocks the response)
    asyncio.create_task(_log_query(
        req.site_id, req.query, result["confidence"], elapsed_ms, len(result["sources"])
    ))

    return ChatResponse(**result)


@router.get("/api/widget/config/{site_id}", response_model=WidgetConfigResponse)
async def widget_config(site_id: int, request: Request):
    blocked = rate_limit_check(request, "widget_config", 60, 60)
    if blocked:
        return blocked

    site, error = _authorize_site_request(site_id, request)
    if error:
        return error

    return WidgetConfigResponse(
        assistant=get_public_assistant_config(site.get("settings") or {})
    )


@router.post("/api/widget/forms/submit")
async def submit_widget_form(req: FormSubmitRequest, request: Request):
    blocked = rate_limit_check(request, f"assistant_form:{req.form_id}", 10, 300)
    if blocked:
        return blocked

    site, error = _authorize_site_request(req.site_id, request)
    if error:
        return error

    result = await asyncio.to_thread(
        submit_assistant_form,
        req.site_id,
        site.get("domain", ""),
        site.get("settings") or {},
        req.session_id,
        req.form_id,
        req.values or {},
        req.page_url,
        request.headers.get("user-agent", ""),
    )
    if result.get("error"):
        return JSONResponse(
            {"error": result["error"], "errors": result.get("errors", {})},
            status_code=result.get("status_code", 400),
        )
    return result


async def _log_query(site_id: int, query: str, confidence: str, response_ms: int, chunk_count: int) -> None:
    try:
        await asyncio.to_thread(
            lambda: cfg.sb.table("chat_logs").insert({
                "site_id": site_id,
                "query": query[:500],
                "confidence": confidence,
                "response_time_ms": response_ms,
                "chunk_count": chunk_count,
            }).execute()
        )
    except Exception as e:
        cfg.log.debug(f"chat_logs insert failed: {e}")


# ── Visitor tracking ────────────────────────────────────────────────────────

@router.post("/api/track")
async def track_visitor(req: TrackRequest, request: Request):
    """Lightweight page-view beacon called by widget.js on every load."""
    # Rate-limit per session to prevent abuse (10 pings/min is plenty)
    blocked = rate_limit_check(request, f"track:{req.session_id[:32]}", 10, 60)
    if blocked:
        return {}  # silent — never show errors to end-users

    ua_str = request.headers.get("user-agent", "")
    ip = get_client_ip(request)
    ua = parse_user_agent(ua_str)

    asyncio.create_task(_log_visitor(
        req.site_id, req.session_id, ip, ua_str[:500],
        ua["device"], ua["browser"], ua["os"],
        (req.referer or "")[:1000],
    ))
    return {}


async def _log_visitor(site_id: int, session_id: str, ip: str, user_agent: str,
                       device: str, browser: str, os_name: str, referer: str) -> None:
    try:
        await asyncio.to_thread(
            lambda: cfg.sb.table("visitor_logs").insert({
                "site_id": site_id,
                "session_id": session_id,
                "ip": ip,
                "user_agent": user_agent,
                "device_type": device,
                "browser": browser,
                "os": os_name,
                "referer": referer,
            }).execute()
        )
    except Exception as e:
        cfg.log.debug(f"visitor_logs insert failed: {e}")


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
