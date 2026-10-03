"""
Chat, health, and widget endpoints.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import scripts.config as cfg
from scripts.assistant_features import (
    get_assistant_config,
    get_public_assistant_config,
    match_intent_actions,
    resolve_text_value,
    submit_assistant_form,
)
from scripts.utils import rate_limit_check, get_client_ip, parse_user_agent, verify_admin_token
from scripts.rag_core import (
    do_rag_stream_sync,
    do_rag_sync,
    get_site_language_cached,
    resolve_response_language,
    resolve_trial_response_language,
)

router = APIRouter()
_health_cache: tuple[float, dict, int] | None = None
_HEALTH_CACHE_SECONDS = 20


class ChatRequest(BaseModel):
    site_id: int
    query: str = Field(..., max_length=2000)
    session_id: str | None = None
    top_k: int = Field(default=cfg.RAG_TOP_K, le=20)
    origin_domain: str | None = None  # kept for backwards compat, not trusted
    response_language: str | None = Field(default=None, max_length=12)


class TrackRequest(BaseModel):
    site_id: int
    session_id: str = Field(..., max_length=128)
    referer: str | None = Field(default=None, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: str
    message_id: str
    actions: list[dict] = Field(default_factory=list)


class WidgetConfigResponse(BaseModel):
    assistant: dict


class FormSubmitRequest(BaseModel):
    site_id: int
    form_id: str = Field(..., max_length=128)
    values: dict = Field(default_factory=dict)
    session_id: str | None = Field(default=None, max_length=128)
    page_url: str | None = Field(default=None, max_length=2000)
    response_language: str | None = Field(default=None, max_length=12)


class FeedbackRequest(BaseModel):
    site_id: int
    session_id: str | None = Field(default=None, max_length=128)
    rating: str = Field(..., pattern="^(up|down)$")
    message_id: str | None = Field(default=None, max_length=128)
    page_url: str | None = Field(default=None, max_length=2000)


@router.get("/health/live")
async def liveness():
    return {
        "status": "ok",
        "environment": cfg.APP_ENV,
        "version": cfg.APP_VERSION,
        "uptime_seconds": round(max(0, time.time() - cfg.start_time), 1),
    }


@router.get("/health")
async def health():
    global _health_cache
    now = time.monotonic()
    if _health_cache and now - _health_cache[0] < _HEALTH_CACHE_SECONDS:
        return JSONResponse(_health_cache[1], status_code=_health_cache[2])

    async def run_check(name: str, callback) -> tuple[str, dict]:
        started = time.perf_counter()
        try:
            await asyncio.wait_for(
                asyncio.to_thread(callback),
                timeout=cfg.HEALTHCHECK_TIMEOUT,
            )
            return name, {
                "status": "ok",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        except asyncio.TimeoutError:
            return name, {
                "status": "error",
                "error": "timeout",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        except Exception as exc:
            cfg.log.warning(
                "health dependency check failed",
                extra={
                    "event": "health.check_failed",
                    "dependency": name,
                    "error_code": exc.__class__.__name__,
                },
            )
            return name, {
                "status": "error",
                "error": exc.__class__.__name__,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }

    def check_database() -> None:
        if cfg.sb is None:
            raise RuntimeError("database client is not initialized")
        cfg.sb.table("sites").select("id").limit(1).execute()

    def check_openai() -> None:
        if cfg.openai_client is None:
            raise RuntimeError("OpenAI client is not initialized")
        cfg.openai_client.models.retrieve(cfg.RAG_MODEL)

    checks = dict(await asyncio.gather(run_check("database", check_database)))
    if cfg.OPENAI_API_KEY:
        checks.update(await asyncio.gather(run_check("openai", check_openai)))
    else:
        checks["openai"] = {"status": "skipped", "reason": "not_configured"}

    checks["embedding_model"] = {
        "status": "ok" if cfg.embed_model is not None else "error",
    }
    ready = all(item["status"] in {"ok", "skipped"} for item in checks.values())
    payload = {
        "status": "ok" if ready else "degraded",
        "environment": cfg.APP_ENV,
        "version": cfg.APP_VERSION,
        "uptime_seconds": round(max(0, time.time() - cfg.start_time), 1),
        "checks": checks,
    }
    status_code = 200 if ready else 503
    _health_cache = (time.monotonic(), payload, status_code)
    return JSONResponse(payload, status_code=status_code)


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


def _is_admin_preview_request(request: Request) -> bool:
    """Allow the admin page to render a real widget preview for any site."""
    if request.headers.get("x-widget-preview", "").lower() != "true":
        return False
    admin_token = request.headers.get("x-admin-token", "")
    return verify_admin_token(admin_token)


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
    is_admin_preview = _is_admin_preview_request(request)

    if not is_landing and not is_trial and not is_internal and not is_admin_preview:
        origin_domain = _extract_origin_domain(request)
        if not origin_domain:
            cfg.log.warning(f"No Origin/Referer header for site {site_id} — rejecting")
            return None, JSONResponse({"error": "Origin header required"}, status_code=403)
        if origin_domain not in ("wrs.kz", "localhost"):
            if origin_domain != site_domain and not origin_domain.endswith(f".{site_domain}"):
                cfg.log.warning(f"Domain mismatch: site {site_id} domain={site_domain} origin={origin_domain}")
                return None, JSONResponse({"error": "Widget not authorized for this domain"}, status_code=403)

    return site, None


def _chat_language(site: dict, req: ChatRequest) -> str | None:
    site_language = get_site_language_cached(req.site_id)
    if (site.get("settings") or {}).get("trial"):
        return resolve_trial_response_language(req.query, req.response_language, site_language)
    return resolve_response_language(req.query, req.response_language, site_language)


@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    blocked = rate_limit_check(request, "chat", 20, 60)
    if blocked:
        return blocked

    site, error = _authorize_site_request(req.site_id, request)
    if error:
        return error

    t0 = time.time()
    interaction_id = f"msg_{uuid.uuid4().hex}"
    language = _chat_language(site, req)
    assistant_config = get_assistant_config(site.get("settings") or {})
    intent_result = match_intent_actions(assistant_config, req.query)

    try:
        result = await asyncio.to_thread(
            do_rag_sync,
            req.site_id,
            req.query,
            req.top_k,
            req.session_id,
            language,
            assistant_config.get("behavior"),
        )
    except Exception as exc:
        elapsed_ms = int((time.time() - t0) * 1000)
        cfg.log.exception(f"chat failed site={req.site_id} interaction={interaction_id}")
        asyncio.create_task(_log_query(
            site_id=req.site_id,
            query=req.query,
            confidence="low",
            response_ms=elapsed_ms,
            chunk_count=0,
            interaction_id=interaction_id,
            session_id=req.session_id,
            status="error",
            error_code=exc.__class__.__name__,
            error_message=str(exc)[:500],
        ))
        return JSONResponse({
            "error": "The assistant could not answer this question",
            "message_id": interaction_id,
        }, status_code=500)
    if intent_result["actions"] and result.get("confidence") == "low":
        result["answer"] = resolve_text_value(
            intent_result.get("response_message"),
            language,
        ) or resolve_text_value({
            "ru": "Я могу помочь с этим. Выберите подходящее действие ниже.",
            "en": "I can help with that. Choose one of the options below.",
            "ko": "도와드릴 수 있어요. 아래에서 원하는 작업을 선택해 주세요.",
        }, language)

    elapsed_ms = int((time.time() - t0) * 1000)
    cfg.log.info(
        "chat completed",
        extra={
            "event": "chat.completed",
            "site_id": req.site_id,
            "query_length": len(req.query),
            "confidence": result["confidence"],
            "chunk_count": len(result["sources"]),
            "intent_action_count": len(intent_result["actions"]),
            "duration_ms": elapsed_ms,
        },
    )

    # Fire-and-forget analytics log (never blocks the response)
    asyncio.create_task(_log_query(
        site_id=req.site_id,
        query=req.query,
        confidence=result["confidence"],
        response_ms=elapsed_ms,
        chunk_count=len(result["sources"]),
        interaction_id=interaction_id,
        session_id=req.session_id,
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
    ))

    return ChatResponse(
        **result,
        message_id=interaction_id,
        actions=intent_result["actions"],
    )


@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """Stream answer deltas over SSE while preserving the regular JSON API."""
    blocked = rate_limit_check(request, "chat", 20, 60)
    if blocked:
        return blocked

    site, error = _authorize_site_request(req.site_id, request)
    if error:
        return error

    started = time.time()
    interaction_id = f"msg_{uuid.uuid4().hex}"
    language = _chat_language(site, req)
    assistant_config = get_assistant_config(site.get("settings") or {})
    intent_result = match_intent_actions(assistant_config, req.query)
    intent_fallback = None
    if intent_result["actions"]:
        intent_fallback = resolve_text_value(
            intent_result.get("response_message"),
            language,
        ) or resolve_text_value({
            "ru": "Я могу помочь с этим. Выберите подходящее действие ниже.",
            "en": "I can help with that. Choose one of the options below.",
            "ko": "도와드릴 수 있어요. 아래에서 원하는 작업을 선택해 주세요.",
        }, language)

    async def events():
        event_queue: asyncio.Queue[dict] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce() -> None:
            try:
                for event in do_rag_stream_sync(
                    req.site_id,
                    req.query,
                    req.top_k,
                    req.session_id,
                    language,
                    assistant_config.get("behavior"),
                    intent_fallback,
                ):
                    loop.call_soon_threadsafe(event_queue.put_nowait, event)
            except Exception as exc:
                cfg.log.exception(
                    f"streaming chat failed site={req.site_id} interaction={interaction_id}"
                )
                loop.call_soon_threadsafe(event_queue.put_nowait, {
                    "type": "error",
                    "error": "The assistant could not answer this question",
                    "error_code": exc.__class__.__name__,
                    "error_message": str(exc)[:500],
                })
            finally:
                loop.call_soon_threadsafe(event_queue.put_nowait, {"type": "_end"})

        producer = asyncio.create_task(asyncio.to_thread(produce))
        yield f"data: {json.dumps({'type': 'start', 'message_id': interaction_id})}\n\n"

        while True:
            event = await event_queue.get()
            event_type = event.get("type")
            if event_type == "_end":
                break

            elapsed_ms = int((time.time() - started) * 1000)
            if event_type == "done":
                event["message_id"] = interaction_id
                event["actions"] = intent_result["actions"]
                asyncio.create_task(_log_query(
                    site_id=req.site_id,
                    query=req.query,
                    confidence=event["confidence"],
                    response_ms=elapsed_ms,
                    chunk_count=len(event["sources"]),
                    interaction_id=interaction_id,
                    session_id=req.session_id,
                    answer=event.get("answer", ""),
                    sources=event.get("sources", []),
                ))
                cfg.log.info(
                    "streaming chat completed",
                    extra={
                        "event": "chat.completed",
                        "site_id": req.site_id,
                        "query_length": len(req.query),
                        "confidence": event["confidence"],
                        "chunk_count": len(event["sources"]),
                        "intent_action_count": len(intent_result["actions"]),
                        "duration_ms": elapsed_ms,
                    },
                )
            elif event_type == "error":
                asyncio.create_task(_log_query(
                    site_id=req.site_id,
                    query=req.query,
                    confidence="low",
                    response_ms=elapsed_ms,
                    chunk_count=0,
                    interaction_id=interaction_id,
                    session_id=req.session_id,
                    status="error",
                    error_code=event.get("error_code"),
                    error_message=event.get("error_message"),
                ))

            public_event = {
                key: value for key, value in event.items()
                if key not in {"error_code", "error_message"}
            }
            yield f"data: {json.dumps(public_event, ensure_ascii=False)}\n\n"

        await producer

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


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
        req.response_language,
    )
    if result.get("error"):
        return JSONResponse(
            {"error": result["error"], "errors": result.get("errors", {})},
            status_code=result.get("status_code", 400),
        )
    return result


@router.post("/api/widget/feedback")
async def submit_widget_feedback(req: FeedbackRequest, request: Request):
    blocked = rate_limit_check(request, "widget_feedback", 30, 300)
    if blocked:
        return blocked

    site, error = _authorize_site_request(req.site_id, request)
    if error:
        return error

    try:
        await asyncio.to_thread(
            lambda: cfg.sb.table("assistant_feedback").insert({
                "site_id": req.site_id,
                "session_id": req.session_id,
                "rating": req.rating,
                "message_id": req.message_id,
                "page_url": req.page_url,
            }).execute()
        )
    except Exception as exc:
        # Feedback must never interrupt the visitor's conversation. This also
        # keeps deployments compatible until the optional table is migrated.
        cfg.log.debug(f"assistant_feedback insert failed: {exc}")
    return {"success": True}


async def _log_query(
    site_id: int,
    query: str,
    confidence: str,
    response_ms: int,
    chunk_count: int,
    *,
    interaction_id: str,
    session_id: str | None = None,
    answer: str = "",
    sources: list[dict] | None = None,
    status: str = "ok",
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    try:
        await asyncio.to_thread(
            lambda: cfg.sb.table("chat_logs").insert({
                "site_id": site_id,
                "query": query[:500],
                "answer": answer[:10000],
                "confidence": confidence,
                "response_time_ms": response_ms,
                "chunk_count": chunk_count,
                "interaction_id": interaction_id,
                "session_id": (session_id or "")[:128] or None,
                "sources": sources or [],
                "status": status,
                "error_code": error_code,
                "error_message": error_message,
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
    return FileResponse(
        js_path,
        media_type="application/javascript",
        headers={
            "Cache-Control": "public, max-age=300, stale-while-revalidate=86400",
        },
    )


# ── Telegram webhook (DISABLED) ─────────────────────────────────────────

@router.post("/api/telegram")
async def telegram_webhook(request: Request):
    return JSONResponse({"error": "Telegram handler is disabled"}, status_code=503)


@router.post("/api/telegram/set-webhook")
async def set_telegram_webhook(request: Request):
    return JSONResponse({"error": "Telegram handler is disabled"}, status_code=503)
