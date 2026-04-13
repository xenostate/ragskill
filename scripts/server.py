#!/usr/bin/env python3
"""
RAG API Server — FastAPI entry point.

Delegates to route modules in scripts/routes/.
Start: uvicorn scripts.server:app --host 0.0.0.0 --port 8090
"""

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from supabase import create_client

import scripts.config as cfg
from scripts.utils import rate_limiter

# Route modules
from scripts.routes.chat import router as chat_router
from scripts.routes.trial import router as trial_router
from scripts.routes.admin import router as admin_router, setup_landing_site
from scripts.routes.auth import router as auth_router
from scripts.routes.internal import router as internal_router
from scripts.routes.whatsapp import router as whatsapp_router


# ── Background cleanup task ─────────────────────────────────────────────────

async def cleanup_expired_trials():
    """Periodic task: delete expired trial sites + clean rate limiter + tokens."""
    while True:
        await asyncio.sleep(3600)
        rate_limiter.cleanup()

        # Clean expired trial sites
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            resp = cfg.sb.table("sites") \
                .delete() \
                .eq("is_trial", True) \
                .lt("expires_at", now_iso) \
                .execute()
            if resp.data:
                expired_ids = {row["id"] for row in resp.data}
                for sid in expired_ids:
                    cfg.trial_progress.pop(sid, None)
                cfg.log.info(f"Cleaned up {len(resp.data)} expired trial site(s)")
        except Exception as e:
            cfg.log.error(f"Trial cleanup error: {e}")

        # Clean stale session histories
        try:
            now = time.time()
            with cfg._session_lock:
                stale = [sid for sid, t in cfg._session_last_access.items() if now - t > cfg.SESSION_TTL]
                for sid in stale:
                    cfg._session_history.pop(sid, None)
                    cfg._session_last_access.pop(sid, None)
            if stale:
                cfg.log.info(f"Cleaned up {len(stale)} stale session(s)")
        except Exception as e:
            cfg.log.error(f"Session cleanup error: {e}")

        # Clean expired admin + internal tokens
        try:
            now = time.time()
            expired_admin = [t for t, exp in cfg.admin_tokens.items() if now > exp]
            for t in expired_admin:
                cfg.admin_tokens.pop(t, None)
            expired_internal = [
                t for t, s in cfg._internal_tokens.items()
                if now - s.get("created", 0) > cfg.INTERNAL_TOKEN_TTL
            ]
            for t in expired_internal:
                cfg._internal_tokens.pop(t, None)
            cleaned = len(expired_admin) + len(expired_internal)
            if cleaned:
                cfg.save_tokens()
                cfg.log.info(f"Cleaned up {len(expired_admin)} admin + {len(expired_internal)} internal expired token(s)")
        except Exception as e:
            cfg.log.error(f"Token cleanup error: {e}")

        # Clean stale trial_progress entries (done or errored for > 1 hour)
        try:
            now = time.time()
            stale_trials = [
                sid for sid, p in cfg.trial_progress.items()
                if p.get("done") and p.get("_finished_at", 0) < now - 3600
            ]
            for sid in stale_trials:
                cfg.trial_progress.pop(sid, None)
            if stale_trials:
                cfg.log.info(f"Cleaned up {len(stale_trials)} stale trial progress entries")
        except Exception as e:
            cfg.log.error(f"Trial progress cleanup error: {e}")


# ── App lifespan ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg.start_time = time.time()

    # Set bounded thread pool so asyncio.to_thread doesn't saturate with embedding work
    asyncio.get_event_loop().set_default_executor(cfg.thread_pool)
    cfg.log.info(f"Thread pool: {cfg._THREAD_POOL_SIZE} workers")

    cfg.log.info(f"Loading embedding model: {cfg.EMBED_MODEL}")
    cfg.embed_model = SentenceTransformer(cfg.EMBED_MODEL)
    cfg.log.info("Embedding model loaded")

    cfg.sb = create_client(cfg.SUPABASE_URL, cfg.SUPABASE_KEY)
    cfg.log.info("Supabase client initialized (service key)")

    # Public client uses anon key (subject to RLS) for unauthenticated endpoints.
    # Falls back to service key if SUPABASE_ANON_KEY is not set.
    if cfg.SUPABASE_ANON_KEY:
        cfg.sb_public = create_client(cfg.SUPABASE_URL, cfg.SUPABASE_ANON_KEY)
        cfg.log.info("Supabase public client initialized (anon key)")
    else:
        cfg.sb_public = cfg.sb
        cfg.log.warning("SUPABASE_ANON_KEY not set — public endpoints use service key (no RLS)")

    if cfg.OPENAI_API_KEY:
        cfg.openai_client = OpenAI(api_key=cfg.OPENAI_API_KEY)
        cfg.log.info("OpenAI client initialized")
    else:
        cfg.log.warning("No OPENAI_API_KEY — LLM answers disabled, retrieval-only mode")

    # WhatsApp handler (dormant unless WHATSAPP_ENABLED=true)
    if cfg.WHATSAPP_ENABLED:
        from scripts.whatsapp_handler import WhatsAppHandler
        from scripts.rag_core import do_rag_sync
        def _wa_rag_fn(site_id, query, top_k=cfg.RAG_TOP_K, session_id=None, language=None):
            return do_rag_sync(site_id, query, top_k, session_id, language)
        cfg.whatsapp_handler = WhatsAppHandler(cfg.sb, _wa_rag_fn, cfg.WHATSAPP_VERIFY_TOKEN)
        cfg.log.info("WhatsApp handler initialized")
        if not cfg.WHATSAPP_APP_SECRET:
            cfg.log.warning("WHATSAPP_APP_SECRET not set — webhook signature verification is DISABLED.")
    else:
        cfg.log.info("WhatsApp handler disabled (set WHATSAPP_ENABLED=true to activate)")

    # Restore tokens from previous run
    cfg.load_tokens()

    # Start background cleanup task
    cleanup_task = asyncio.create_task(cleanup_expired_trials())
    cfg.log.info("Cleanup task started (runs hourly)")

    # Setup landing chat site
    try:
        setup_landing_site()
    except Exception as e:
        cfg.log.error(f"Landing site setup failed: {e}")

    yield

    # Shutdown
    cfg.save_tokens()
    cleanup_task.cancel()
    cfg.thread_pool.shutdown(wait=False)
    cfg.log.info("Shutting down")


# ── Body size limiter ──────────────────────────────────────────────────────

class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Content-Length exceeds MAX_BODY_SIZE."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > cfg.MAX_BODY_SIZE:
            return JSONResponse(
                {"error": f"Request body too large (max {cfg.MAX_BODY_SIZE // (1024*1024)}MB)"},
                status_code=413,
            )
        return await call_next(request)


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="web-rag API", lifespan=lifespan)

# Middleware (order: last added = outermost = runs first)
app.add_middleware(BodySizeLimitMiddleware)

# CORS: open for widget.js (embedded on customer sites).
# Auth uses custom headers, not cookies, so allow_credentials is not needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Token", "X-Internal-Token"],
)

# Register route modules
app.include_router(chat_router)
app.include_router(trial_router)
app.include_router(admin_router)
app.include_router(auth_router)
app.include_router(internal_router)
app.include_router(whatsapp_router)
