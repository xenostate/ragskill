#!/usr/bin/env python3
"""
RAG API Server — FastAPI service with warm embedding model.

Endpoints:
    POST /api/chat          — widget/client chat
    POST /api/telegram      — Telegram webhook
    GET  /widget.js         — serve the chat widget
    GET  /trial             — self-service trial page
    POST /api/trial/start   — start trial indexing
    GET  /api/trial/progress/{site_id} — SSE progress stream
    GET  /health            — status check

Start:
    uvicorn scripts.server:app --host 0.0.0.0 --port 8090
"""

import asyncio
import hashlib
import io
import json
import os
import re
import sys
import time
import uuid
import logging
import threading
from collections import deque, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from pypdf import PdfReader

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from supabase import create_client
import bcrypt

# from scripts.telegram_handler import TelegramHandler  # DISABLED
# WhatsApp handler — imported conditionally in lifespan if enabled

# ── Config ──────────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
RAG_MODEL = os.environ.get("RAG_MODEL", "gpt-4o-mini")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("OPENAI_KEY", ""))
# TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")  # DISABLED

# WhatsApp Business (dormant — set WHATSAPP_ENABLED=true to activate)
WHATSAPP_ENABLED = os.environ.get("WHATSAPP_ENABLED", "").lower() in ("true", "1", "yes")
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
WHATSAPP_APP_SECRET = os.environ.get("WHATSAPP_APP_SECRET", "")

WIDGET_DIR = Path(__file__).resolve().parent.parent / "widget"
SCRIPTS_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag-server")

# ── Globals (warm on startup) ──────────────────────────────────────────────

embed_model: SentenceTransformer | None = None
sb = None
openai_client: OpenAI | None = None
whatsapp_handler = None  # initialized in lifespan if WHATSAPP_ENABLED
start_time = 0.0

# Trial indexing progress: { site_id: { step, total, message, done, error } }
trial_progress: dict[int, dict] = {}

MAX_PDF_SIZE = 10 * 1024 * 1024  # 10 MB

# Landing page chat site_id (set on startup or via /api/landing/setup)
landing_site_id: int | None = None

LANDING_DOMAIN = "landing.wrs.kz"

# ── Session history (conversation context) ────────────────────────────────

_session_history: dict[str, list[dict]] = {}   # session_id -> [{role, content}, ...]
_session_last_access: dict[str, float] = {}     # session_id -> timestamp
_session_lock = threading.Lock()
SESSION_HISTORY_LIMIT = 5   # max exchanges (10 messages) per session
SESSION_TTL = 3600          # evict sessions idle for 1 hour

# ── Site language cache ───────────────────────────────────────────────────

_site_lang_cache: dict[int, tuple[str | None, float]] = {}  # site_id -> (lang, timestamp)
LANG_CACHE_TTL = 3600  # 1 hour


# ── Rate Limiter ──────────────────────────────────────────────────────────

class RateLimiter:
    """In-memory sliding-window rate limiter per IP address."""

    def __init__(self):
        self._lock = threading.Lock()
        # { "bucket_name:ip": deque([timestamp, ...]) }
        self._hits: dict[str, deque] = defaultdict(deque)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Return True if request is within limit, False if rate-exceeded."""
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            q = self._hits[key]
            # Evict old entries
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= max_requests:
                return False
            q.append(now)
            return True

    def cleanup(self):
        """Remove stale keys (call periodically)."""
        now = time.time()
        with self._lock:
            stale = [k for k, q in self._hits.items() if not q or q[-1] < now - 3600]
            for k in stale:
                del self._hits[k]


rate_limiter = RateLimiter()


def get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For from Nginx."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit_check(request: Request, bucket: str, max_requests: int, window: int):
    """Check rate limit; raise 429 if exceeded."""
    ip = get_client_ip(request)
    key = f"{bucket}:{ip}"
    if not rate_limiter.is_allowed(key, max_requests, window):
        log.warning(f"Rate limit hit: {bucket} from {ip}")
        return JSONResponse(
            {"error": "Too many requests. Please try again later."},
            status_code=429,
        )
    return None

LANDING_CONTENT = [
    {
        "title": "О WebRAG Systems",
        "text": """WebRAG Systems — это AI-ассистент для сайтов. Наш продукт автоматически отвечает на вопросы ваших клиентов, основываясь на содержимом вашего сайта.

WebRAG Systems is an AI assistant for websites. Our product automatically answers your customers' questions based on your website content.

WebRAG Systems — сайттарға арналған AI-көмекші. Біздің өнім сайтыңыздың мазмұнына негізделіп тұтынушыларыңыздың сұрақтарына автоматты түрде жауап береді.""",
    },
    {
        "title": "Как это работает / How it works",
        "text": """Процесс работы WebRAG очень простой:
1. Вы регистрируетесь и указываете URL вашего сайта.
2. Наш AI сканирует и индексирует весь контент сайта, включая PDF-файлы.
3. Вы тестируете ассистента в чате — задаёте вопросы и проверяете качество ответов.
4. Копируете одну строчку кода и вставляете в HTML вашего сайта.
После этого виджет-ассистент появится на вашем сайте и будет отвечать посетителям 24/7.

The WebRAG process is very simple: register, paste your website URL, our AI crawls and indexes all content including PDFs, test the assistant in chat, then copy one line of code into your site HTML. The widget assistant will appear on your site and answer visitors 24/7.""",
    },
    {
        "title": "Возможности и особенности / Features",
        "text": """Ключевые возможности WebRAG:
- Установка за 5 минут — одна строчка кода на сайт
- Мультиязычность — русский, английский, казахский и другие языки
- Поддержка PDF-файлов — загружайте документы для индексации
- Работает 24/7 — ассистент всегда онлайн
- Точные ответы — на основе реального контента вашего сайта, без выдумок
- Автоматическое сканирование — AI сам находит и индексирует страницы сайта

Key features: 5-minute setup with one line of code, multilingual support (Russian, English, Kazakh), PDF support, 24/7 availability, accurate answers based on real site content, automatic crawling.""",
    },
    {
        "title": "Для кого подходит / Who it's for",
        "text": """WebRAG идеально подходит для:
- Интернет-магазинов — ответы на вопросы о товарах, наличии, доставке
- Сервисных компаний — информация об услугах, ценах, графике работы
- Образовательных платформ — ответы по учебным материалам
- Корпоративных сайтов — помощь клиентам и сотрудникам
- Любого бизнеса в Казахстане, который хочет автоматизировать поддержку клиентов

Особенно актуально для бизнеса в Казахстане: поддержка казахского, русского и английского языков из коробки.""",
    },
    
    {
        "title": "Контакты и пробная версия / Contact and trial",
        "text": """Вы можете попробовать WebRAG бесплатно прямо на нашем сайте wrs.kz — нажмите кнопку 'Начать бесплатно' внизу страницы.

Для связи: телефон +7 700 534 5949.
Сайт: wrs.kz

You can try WebRAG for free on our website wrs.kz. Contact us at +7 700 534 5949.""",
    },
]

# ── RAG prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant for a website. You answer visitors' questions using the provided source chunks from that website.

Rules:
1. Base your answers ONLY on the provided source chunks. Do not invent information that isn't in the sources.
2. For broad questions like "what is this website about?", "tell me about this site", or "what do you do?" — synthesize an overview from whatever source chunks are available. Summarize the main topics, services, or content you can see in the sources.
3. Only say "I don't have enough information to answer this" if the sources are truly empty or completely irrelevant to the question.
4. Keep answers concise and factual. Do not speculate beyond what the sources state.
5. Do NOT list sources or citations in your answer. Just provide the answer as plain text.
6. Answer in the same language as the user's question.
7. If the user's question is a follow-up referencing a previous message, use the conversation history to understand their intent, but still answer only from the provided source chunks.
8. Be conversational and helpful. If the sources partially cover the topic, answer what you can and note what you couldn't find.
"""

LANGUAGE_NAMES = {
    "en": "English", "ru": "Russian", "kk": "Kazakh", "es": "Spanish",
    "fr": "French", "de": "German", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "pt": "Portuguese", "ar": "Arabic", "hi": "Hindi",
    "it": "Italian", "tr": "Turkish", "nl": "Dutch", "pl": "Polish",
    "uk": "Ukrainian",
}


def get_system_prompt(language: str | None = None) -> str:
    """Return system prompt, optionally with strict language enforcement."""
    if language and language in LANGUAGE_NAMES:
        lang_name = LANGUAGE_NAMES[language]
        return SYSTEM_PROMPT + f"\nCRITICAL: You MUST reply strictly in {lang_name}. Every part of your response — the answer, the source list, everything — must be in {lang_name}. No exceptions.\n"
    return SYSTEM_PROMPT


# ── Lifespan (model loading) ──────────────────────────────────────────────

# rag_fn removed — Telegram handler is disabled


async def cleanup_expired_trials():
    """Periodic task: delete expired trial sites + clean rate limiter."""
    while True:
        await asyncio.sleep(3600)  # run every hour
        rate_limiter.cleanup()
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            resp = sb.table("sites") \
                .delete() \
                .eq("is_trial", True) \
                .lt("expires_at", now_iso) \
                .execute()
            if resp.data:
                expired_ids = {row["id"] for row in resp.data}
                for sid in expired_ids:
                    trial_progress.pop(sid, None)
                log.info(f"Cleaned up {len(resp.data)} expired trial site(s)")
        except Exception as e:
            log.error(f"Trial cleanup error: {e}")

        # Clean stale session histories
        try:
            now = time.time()
            with _session_lock:
                stale = [sid for sid, t in _session_last_access.items() if now - t > SESSION_TTL]
                for sid in stale:
                    _session_history.pop(sid, None)
                    _session_last_access.pop(sid, None)
            if stale:
                log.info(f"Cleaned up {len(stale)} stale session(s)")
        except Exception as e:
            log.error(f"Session cleanup error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, sb, openai_client, whatsapp_handler, start_time
    start_time = time.time()

    log.info(f"Loading embedding model: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL)
    log.info("Embedding model loaded")

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    log.info("Supabase client initialized")

    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        log.info("OpenAI client initialized")
    else:
        log.warning("No OPENAI_API_KEY — LLM answers disabled, retrieval-only mode")

    # WhatsApp handler (dormant unless WHATSAPP_ENABLED=true)
    if WHATSAPP_ENABLED:
        from scripts.whatsapp_handler import WhatsAppHandler
        def _wa_rag_fn(site_id, query, top_k=RAG_TOP_K, session_id=None, language=None):
            return _do_rag_sync(site_id, query, top_k, session_id, language)
        whatsapp_handler = WhatsAppHandler(sb, _wa_rag_fn, WHATSAPP_VERIFY_TOKEN)
        log.info("WhatsApp handler initialized")
        if not WHATSAPP_APP_SECRET:
            log.warning("WHATSAPP_APP_SECRET not set — webhook signature verification is DISABLED. "
                        "This is insecure for production use.")
    else:
        log.info("WhatsApp handler disabled (set WHATSAPP_ENABLED=true to activate)")

    # Start trial cleanup background task
    cleanup_task = asyncio.create_task(cleanup_expired_trials())
    log.info("Trial cleanup task started (runs hourly)")

    # Setup landing chat site
    try:
        setup_landing_site()
    except Exception as e:
        log.error(f"Landing site setup failed: {e}")

    yield

    cleanup_task.cancel()
    log.info("Shutting down")


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(title="web-rag API", lifespan=lifespan)

# CORS: open for widget.js (embedded on customer sites) + chat API
# Rate limiting protects against abuse instead of origin-based restriction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ── Models ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    site_id: int
    query: str = Field(..., max_length=2000)
    session_id: str | None = None
    top_k: int = Field(default=RAG_TOP_K, le=20)
    origin_domain: str | None = None  # sent by widget.js for domain verification


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: str


# ── Core RAG logic ─────────────────────────────────────────────────────────

def retrieve_chunks(site_id: int, query: str, top_k: int = 5) -> dict:
    """Embed query + call match_chunks. Uses warm model."""
    query_embedding = embed_model.encode(
        f"query: {query}", normalize_embeddings=True
    ).tolist()

    resp = sb.rpc("match_chunks", {
        "p_site_id": site_id,
        "p_query_embedding": query_embedding,
        "p_query_text": query,
        "p_match_count": top_k,
    }).execute()

    rows = resp.data or []

    results = []
    for row in rows:
        results.append({
            "chunk_text": row["text"],
            "url": row["url"],
            "title": row["title"],
            "score": round(row["score"], 4),
        })

    if not results:
        confidence = "low"
    else:
        top_score = results[0]["score"]
        # Check if top result is significantly better than others (isolated match)
        if len(results) > 1:
            avg_rest = sum(r["score"] for r in results[1:]) / len(results[1:])
            score_gap = top_score - avg_rest
        else:
            score_gap = 0

        if top_score >= 0.75 and score_gap < 0.4:
            confidence = "high"
        elif top_score >= 0.75 and score_gap >= 0.4:
            # High top score but nothing else matches — thin evidence
            confidence = "medium"
        elif top_score >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"

    return {"confidence": confidence, "results": results}


def build_context(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] (score: {r['score']}) {r['title']} — {r['url']}")
        lines.append(r["chunk_text"])
        lines.append("")
    return "\n".join(lines)


def get_site_language(site_id: int) -> str | None:
    """Look up the language setting for a site."""
    try:
        resp = sb.table("sites").select("language").eq("id", site_id).single().execute()
        return resp.data.get("language") if resp.data else None
    except Exception:
        return None


def generate_answer(query: str, context: str, confidence: str, language: str | None = None) -> str:
    """Call OpenAI to generate a cited answer."""
    if openai_client is None:
        return "LLM not configured. Set OPENAI_API_KEY in .env to enable answers."

    user_msg = f"Source chunks:\n{context}\n\nQuestion: {query}"
    prompt = get_system_prompt(language)

    resp = openai_client.chat.completions.create(
        model=RAG_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=2000,
    )

    return resp.choices[0].message.content


def get_site_language_cached(site_id: int) -> str | None:
    """Cached version of get_site_language (TTL = 5 min)."""
    now = time.time()
    cached = _site_lang_cache.get(site_id)
    if cached and now - cached[1] < LANG_CACHE_TTL:
        return cached[0]
    lang = get_site_language(site_id)
    _site_lang_cache[site_id] = (lang, now)
    return lang


def _do_rag_sync(site_id: int, query: str, top_k: int,
                 session_id: str | None = None,
                 language: str | None = None) -> dict:
    """Full RAG pipeline (synchronous) — meant to run in asyncio.to_thread.

    Handles retrieval, conversation context, and LLM generation.
    """
    # Detect broad/generic queries that need wider context
    _BROAD_PATTERNS = re.compile(
        r'\b(what is this|tell me about|about the (website|site|company|page)|'
        r'what do you do|what (can|does) (this|the) (site|website|company)|'
        r'who are you|what are you|overview|summary|introduce)\b',
        re.IGNORECASE,
    )
    is_broad = bool(_BROAD_PATTERNS.search(query))

    # Expand short follow-up queries using conversation history
    retrieval_query = query
    if session_id:
        with _session_lock:
            history = _session_history.get(session_id, [])
        if history and len(query.split()) < 6:
            last_user = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"),
                None,
            )
            if last_user:
                retrieval_query = f"{last_user} {query}"

    # 1. Retrieve — fetch more chunks for broad questions to give LLM wider context
    effective_top_k = min(top_k * 2, 15) if is_broad else top_k
    retrieval = retrieve_chunks(site_id, retrieval_query, effective_top_k)
    context = build_context(retrieval["results"])

    # 2. Build messages with conversation history
    system = get_system_prompt(language)
    messages = [{"role": "system", "content": system}]

    if session_id:
        with _session_lock:
            history = _session_history.get(session_id, [])
        for msg in history[-(SESSION_HISTORY_LIMIT * 2):]:
            messages.append(msg)

    messages.append({
        "role": "user",
        "content": f"Source chunks:\n{context}\n\nQuestion: {query}",
    })

    # 3. Generate
    if openai_client is None:
        answer = "LLM not configured. Set OPENAI_API_KEY in .env to enable answers."
    else:
        resp = openai_client.chat.completions.create(
            model=RAG_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=2000,
        )
        answer = resp.choices[0].message.content

    # 4. Store in session history
    if session_id:
        with _session_lock:
            if session_id not in _session_history:
                _session_history[session_id] = []
            _session_history[session_id].append({"role": "user", "content": query})
            _session_history[session_id].append({"role": "assistant", "content": answer})
            # Trim to limit
            if len(_session_history[session_id]) > SESSION_HISTORY_LIMIT * 2:
                _session_history[session_id] = _session_history[session_id][-(SESSION_HISTORY_LIMIT * 2):]
            _session_last_access[session_id] = time.time()

    sources = [
        {"title": r["title"], "url": r["url"], "score": r["score"]}
        for r in retrieval["results"]
    ]
    return {"answer": answer, "sources": sources, "confidence": retrieval["confidence"]}


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": embed_model is not None,
        "llm_enabled": openai_client is not None,
        "uptime_seconds": round(time.time() - start_time, 1),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    # Rate limit: 20 messages per minute per IP
    blocked = rate_limit_check(request, "chat", 20, 60)
    if blocked:
        return blocked

    # Domain verification: check that widget is used on the correct site
    if req.origin_domain:
        site_row = sb.table("sites").select("domain, settings").eq("id", req.site_id).execute()
        if site_row.data:
            site_domain = site_row.data[0].get("domain", "")
            site_settings = site_row.data[0].get("settings") or {}
            is_landing = site_settings.get("landing", False)
            is_trial = "trial" in site_domain
            # Skip check for landing site, trial sites, and wrs.kz itself
            if not is_landing and not is_trial and req.origin_domain not in ("wrs.kz", "localhost"):
                # Check if the origin matches the registered domain
                if req.origin_domain != site_domain and not req.origin_domain.endswith(f".{site_domain}"):
                    log.warning(f"Domain mismatch: site {req.site_id} domain={site_domain} origin={req.origin_domain}")
                    return JSONResponse({"error": "Widget not authorized for this domain"}, status_code=403)

    t0 = time.time()
    language = get_site_language_cached(req.site_id)

    # Run the full RAG pipeline in a thread to avoid blocking the event loop
    result = await asyncio.to_thread(
        _do_rag_sync, req.site_id, req.query, req.top_k, req.session_id, language
    )

    log.info(
        f"chat site={req.site_id} q=\"{req.query[:50]}\" "
        f"confidence={result['confidence']} chunks={len(result['sources'])} "
        f"time={time.time()-t0:.2f}s"
    )

    return ChatResponse(**result)


@app.get("/widget.js")
async def serve_widget():
    js_path = WIDGET_DIR / "widget.js"
    if not js_path.exists():
        return JSONResponse({"error": "widget.js not found"}, status_code=404)
    return FileResponse(js_path, media_type="application/javascript")


# ── Telegram webhook (DISABLED) ───────────────────────────────────────────

@app.post("/api/telegram")
async def telegram_webhook(request: Request):
    return JSONResponse({"error": "Telegram handler is disabled"}, status_code=503)


@app.post("/api/telegram/set-webhook")
async def set_telegram_webhook(request: Request):
    return JSONResponse({"error": "Telegram handler is disabled"}, status_code=503)


# ── WhatsApp webhook ─────────────────────────────────────────────────────

@app.get("/api/whatsapp/webhook")
async def whatsapp_verify(request: Request):
    """Meta webhook verification (challenge-response handshake)."""
    if not WHATSAPP_ENABLED or whatsapp_handler is None:
        return JSONResponse({"error": "WhatsApp handler is disabled"}, status_code=503)

    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    result = whatsapp_handler.verify_webhook(mode, token, challenge)
    if result is not None:
        # Meta expects the challenge echoed back as a plain integer
        return JSONResponse(content=int(result))
    return JSONResponse({"error": "Verification failed"}, status_code=403)


@app.post("/api/whatsapp/webhook")
async def whatsapp_incoming(request: Request):
    """Receive inbound WhatsApp messages from BSP."""
    if not WHATSAPP_ENABLED or whatsapp_handler is None:
        return JSONResponse({"error": "WhatsApp handler is disabled"}, status_code=503)

    # Rate limit the webhook endpoint (BSP server IPs)
    blocked = rate_limit_check(request, "whatsapp", 100, 60)
    if blocked:
        return blocked

    # Read raw body for signature verification
    body = await request.body()

    # Verify HMAC signature if app secret is configured
    if WHATSAPP_APP_SECRET:
        from scripts.whatsapp_handler import WhatsAppHandler
        signature = request.headers.get("x-hub-signature-256", "")
        if not WhatsAppHandler.verify_signature(body, signature, WHATSAPP_APP_SECRET):
            log.warning("WhatsApp webhook signature verification failed")
            return JSONResponse({"error": "Invalid signature"}, status_code=403)

    payload = json.loads(body)

    # Process in background thread — Meta requires 200 response within 5 seconds
    asyncio.get_event_loop().create_task(
        asyncio.to_thread(
            whatsapp_handler.handle_webhook,
            payload,
            _session_history,
            _session_lock,
            get_site_language_cached,
        )
    )

    return JSONResponse({"status": "ok"})


@app.post("/api/whatsapp/register")
async def whatsapp_register(request: Request):
    """Admin endpoint: register a WhatsApp business number for a site."""
    if not WHATSAPP_ENABLED or whatsapp_handler is None:
        return JSONResponse({"error": "WhatsApp handler is disabled"}, status_code=503)

    # Verify admin token
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
        account = whatsapp_handler.register_account(
            site_id, phone_number, display_name, api_token, provider,
        )
        return JSONResponse({"ok": True, "account": account})
    except Exception as e:
        log.error(f"WhatsApp register error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Trial / Demo ──────────────────────────────────────────────────────────

# Import reusable functions from indexer
from scripts.indexer import (
    clean_html, chunk_text, extract_headings, extract_links,
    content_hash, StaticRenderer, PlaywrightRenderer,
)


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/trial")


@app.get("/trial")
async def serve_trial_page():
    html_path = WIDGET_DIR / "trial.html"
    if not html_path.exists():
        return JSONResponse({"error": "trial.html not found"}, status_code=404)
    return FileResponse(html_path, media_type="text/html")


def run_trial_indexing(site_id: int, url: str, max_pages: int, pdf_data: list[dict], use_playwright: bool = False):
    """Synchronous trial indexing — runs in asyncio.to_thread."""
    progress = trial_progress[site_id]

    try:
        all_docs = []  # list of (url_or_filename, title, text, html_or_none)

        # ── Phase 1: Crawl the URL ──
        progress["message"] = f"Crawling {url}..."
        if use_playwright:
            log.info(f"Trial {site_id}: using Playwright renderer")
            renderer = PlaywrightRenderer()
        else:
            renderer = StaticRenderer()

        try:
            queue = deque([url])
            visited = set()
            pages_crawled = 0
            allowed_domain = urlparse(url).netloc

            while queue and pages_crawled < max_pages:
                page_url = queue.popleft()
                if page_url in visited:
                    continue
                visited.add(page_url)

                result = renderer.fetch(page_url)
                if result is None:
                    continue
                html, status = result
                if status != 200:
                    continue

                title, text = clean_html(html)
                if len(text) < 50:
                    # Still extract links
                    for link in extract_links(html, page_url, allowed_domain):
                        if link not in visited:
                            queue.append(link)
                    continue

                all_docs.append((page_url, title or page_url, text, html))
                pages_crawled += 1
                progress["message"] = f"Crawled {pages_crawled} page(s)..."

                if pages_crawled < max_pages:
                    for link in extract_links(html, page_url, allowed_domain):
                        if link not in visited:
                            queue.append(link)

                time.sleep(0.3)
        finally:
            renderer.close()

        # ── Phase 2: Extract text from PDFs ──
        for pdf_item in pdf_data:
            progress["message"] = f"Processing PDF: {pdf_item['filename']}..."
            try:
                reader = PdfReader(io.BytesIO(pdf_item["content"]))
                pdf_text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pdf_text += page_text + "\n\n"

                if pdf_text.strip():
                    safe_filename = re.sub(r'[^\w\s\-.]', '_', pdf_item['filename'])
                    all_docs.append((
                        f"pdf://{safe_filename}",
                        pdf_item["filename"],
                        pdf_text.strip(),
                        None,
                    ))
                else:
                    log.warning(f"PDF {pdf_item['filename']} yielded no text (possibly scanned/image PDF)")
            except Exception as e:
                log.warning(f"Trial PDF extraction failed for {pdf_item['filename']}: {e}")

        if not all_docs:
            progress["error"] = "No content could be extracted from the URL or PDFs."
            progress["done"] = True
            return

        # ── Phase 3: Chunk, embed, store ──
        total_chunks = 0
        progress["total"] = len(all_docs)

        for doc_idx, (doc_url, doc_title, doc_text, doc_html) in enumerate(all_docs):
            progress["step"] = doc_idx + 1
            progress["message"] = f"Indexing {doc_idx+1}/{len(all_docs)}: {doc_title[:50]}..."

            c_hash = content_hash(doc_text)

            # Insert document
            ins = sb.table("documents").insert({
                "site_id": site_id,
                "url": doc_url,
                "title": doc_title,
                "content_hash": c_hash,
            }).execute()
            doc_id = ins.data[0]["id"]

            # Chunk
            headings = extract_headings(doc_html) if doc_html else []
            chunks = chunk_text(doc_text)

            if not chunks:
                continue

            # Embed (using warm model from server.py global)
            texts_to_embed = [f"passage: {c}" for c in chunks]
            embeddings = embed_model.encode(
                texts_to_embed, show_progress_bar=False, normalize_embeddings=True
            )

            # Insert chunks
            rows = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                rows.append({
                    "document_id": doc_id,
                    "chunk_index": i,
                    "text": chunk,
                    "headings": headings,
                    "embedding": emb.tolist(),
                })

            sb.table("chunks").insert(rows).execute()
            total_chunks += len(rows)

        progress["message"] = f"Done! Indexed {len(all_docs)} document(s), {total_chunks} chunks."
        progress["done"] = True
        log.info(f"Trial site {site_id}: indexed {len(all_docs)} docs, {total_chunks} chunks")

    except Exception as e:
        log.error(f"Trial indexing error for site {site_id}: {e}")
        progress["error"] = str(e)
        progress["done"] = True


@app.post("/api/trial/start")
async def trial_start(
    request: Request,
    url: str = Form(...),
    max_pages: int = Form(default=5),
    language: str = Form(default="en"),
    use_playwright: str = Form(default="0"),
    pdfs: list[UploadFile] = File(default=[]),
):
    # Rate limit: 3 trials per hour per IP
    blocked = rate_limit_check(request, "trial_start", 3, 3600)
    if blocked:
        return blocked

    # Validate URL
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)
    if not parsed.netloc:
        return JSONResponse({"error": "Invalid URL"}, status_code=400)

    # Cap max pages for trial
    max_pages = max(1, min(max_pages, 50))

    domain = f"trial-{uuid.uuid4().hex[:8]}.demo"
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()

    # Create trial site in Supabase
    site_resp = sb.table("sites").insert({
        "domain": domain,
        "language": language,
        "is_trial": True,
        "expires_at": expires_at,
        "settings": {"source_url": url, "trial": True},
    }).execute()
    site_id = site_resp.data[0]["id"]

    # Read PDF bytes into memory before background task
    pdf_data = []
    for pdf in pdfs:
        content = await pdf.read()
        if len(content) > MAX_PDF_SIZE:
            return JSONResponse(
                {"error": f"PDF '{pdf.filename}' exceeds 10MB limit"},
                status_code=400,
            )
        pdf_data.append({"filename": pdf.filename, "content": content})

    # Initialize progress
    trial_progress[site_id] = {
        "step": 0, "total": 0, "message": "Starting...",
        "done": False, "error": None,
    }

    # Launch indexing in background thread
    pw = use_playwright == "1"
    asyncio.get_event_loop().create_task(
        asyncio.to_thread(run_trial_indexing, site_id, url, max_pages, pdf_data, pw)
    )

    log.info(f"Trial started: site_id={site_id} url={url} pdfs={len(pdf_data)} max_pages={max_pages} playwright={pw}")
    return {"site_id": site_id, "message": "Indexing started", "expires_at": expires_at}


@app.get("/api/trial/progress/{site_id}")
async def trial_progress_stream(site_id: int, request: Request):
    # Rate limit: 10 SSE connections per minute per IP
    blocked = rate_limit_check(request, "sse", 10, 60)
    if blocked:
        return blocked
    async def event_generator():
        while True:
            progress = trial_progress.get(site_id)
            if progress is None:
                yield f"data: {json.dumps({'error': 'Unknown site_id'})}\n\n"
                break

            yield f"data: {json.dumps(progress)}\n\n"

            if progress.get("done") or progress.get("error"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.delete("/api/trial/stop/{site_id}")
async def trial_stop(site_id: int):
    """Immediately delete a trial site and all its data."""
    try:
        # Verify it's a trial site
        site = sb.table("sites").select("is_trial").eq("id", site_id).execute()
        if not site.data or not site.data[0].get("is_trial"):
            return JSONResponse({"error": "Not a trial site"}, status_code=400)

        # Delete site (cascade deletes documents + chunks)
        sb.table("sites").delete().eq("id", site_id).execute()
        trial_progress.pop(site_id, None)

        log.info(f"Trial site {site_id} stopped and deleted by user")
        return {"success": True, "message": "Trial data deleted"}
    except Exception as e:
        log.error(f"Trial stop error for site {site_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/trial/cleanup")
async def trigger_trial_cleanup():
    """Manual cleanup of expired trial sites."""
    now_iso = datetime.now(timezone.utc).isoformat()
    resp = sb.table("sites") \
        .delete() \
        .eq("is_trial", True) \
        .lt("expires_at", now_iso) \
        .execute()
    deleted = len(resp.data) if resp.data else 0
    return {"deleted": deleted}


# ── Landing page chat ─────────────────────────────────────────────────────

def setup_landing_site():
    """Create (or find existing) landing chat site and index product content."""
    global landing_site_id

    # Check if landing site already exists
    existing = sb.table("sites").select("id").eq("domain", LANDING_DOMAIN).execute()
    if existing.data:
        landing_site_id = existing.data[0]["id"]
        # Ensure language is set to auto-detect (None)
        sb.table("sites").update({"language": None}).eq("id", landing_site_id).execute()
        log.info(f"Landing site already exists: site_id={landing_site_id}")
        return landing_site_id

    # Create new site
    site_resp = sb.table("sites").insert({
        "domain": LANDING_DOMAIN,
        "language": None,
        "is_trial": False,
        "settings": {"landing": True},
    }).execute()
    landing_site_id = site_resp.data[0]["id"]
    log.info(f"Created landing site: site_id={landing_site_id}")

    # Index content
    for item in LANDING_CONTENT:
        c_hash = content_hash(item["text"])
        ins = sb.table("documents").insert({
            "site_id": landing_site_id,
            "url": f"landing://{LANDING_DOMAIN}",
            "title": item["title"],
            "content_hash": c_hash,
        }).execute()
        doc_id = ins.data[0]["id"]

        chunks = chunk_text(item["text"])
        if not chunks:
            continue

        texts_to_embed = [f"passage: {c}" for c in chunks]
        embeddings = embed_model.encode(
            texts_to_embed, show_progress_bar=False, normalize_embeddings=True
        )

        rows = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            rows.append({
                "document_id": doc_id,
                "chunk_index": i,
                "text": chunk,
                "headings": [],
                "embedding": emb.tolist(),
            })

        sb.table("chunks").insert(rows).execute()

    log.info(f"Landing site {landing_site_id} indexed with {len(LANDING_CONTENT)} documents")
    return landing_site_id


@app.get("/api/landing/config")
async def landing_config():
    """Return landing chat site_id."""
    return {"site_id": landing_site_id}


@app.post("/api/landing/setup")
async def landing_setup():
    """Create landing site and index product content (idempotent)."""
    sid = await asyncio.to_thread(setup_landing_site)
    return {"site_id": sid, "message": "Landing site ready"}


# ── Admin Dashboard ──────────────────────────────────────────────────────

ADMIN_CODE = os.environ.get("ADMIN_CODE", os.environ.get("ACTIVATION_CODE", ""))
INTERNAL_SETUP_CODE = os.environ.get("INTERNAL_SETUP_CODE", os.environ.get("ACTIVATION_CODE", ""))
# Simple token store: { token: expiry_timestamp }
admin_tokens: dict[str, float] = {}
ADMIN_TOKEN_TTL = 3600 * 8  # 8 hours


def verify_admin(request: Request) -> bool:
    """Check X-Admin-Token header."""
    token = request.headers.get("x-admin-token", "")
    expiry = admin_tokens.get(token)
    if not expiry or time.time() > expiry:
        admin_tokens.pop(token, None)
        return False
    return True


def verify_admin_token(token: str) -> bool:
    """Check a raw admin token string."""
    expiry = admin_tokens.get(token)
    if not expiry or time.time() > expiry:
        admin_tokens.pop(token, None)
        return False
    return True


@app.get("/admin")
async def serve_admin_page():
    html_path = WIDGET_DIR / "admin.html"
    if not html_path.exists():
        return JSONResponse({"error": "admin.html not found"}, status_code=404)
    return FileResponse(html_path, media_type="text/html")


@app.post("/api/admin/auth")
async def admin_auth(request: Request):
    # Rate limit: 5 attempts per hour per IP
    blocked = rate_limit_check(request, "admin_auth", 5, 3600)
    if blocked:
        return blocked

    if not ADMIN_CODE:
        return JSONResponse({"error": "Admin access not configured"}, status_code=503)

    body = await request.json()
    code = body.get("code", "").strip()

    if code != ADMIN_CODE:
        log.warning(f"Failed admin login from {get_client_ip(request)}")
        return JSONResponse({"error": "Invalid code"}, status_code=403)

    token = f"adm_{uuid.uuid4().hex}"
    admin_tokens[token] = time.time() + ADMIN_TOKEN_TTL
    log.info(f"Admin login from {get_client_ip(request)}")
    return {"success": True, "token": token}


@app.get("/api/admin/documents")
async def admin_list_documents(request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if not landing_site_id:
        return {"documents": []}

    docs = sb.table("documents").select("id, title, url").eq("site_id", landing_site_id).execute()

    if not docs.data:
        return {"documents": []}

    doc_ids = [doc["id"] for doc in docs.data]
    # Batch fetch all chunks for these documents
    all_chunks = sb.table("chunks").select("document_id, text").in_("document_id", doc_ids).execute()

    # Group chunks by document_id
    chunks_by_doc = {}
    for c in (all_chunks.data or []):
        chunks_by_doc.setdefault(c["document_id"], []).append(c["text"])

    result = []
    for doc in docs.data:
        chunk_texts = chunks_by_doc.get(doc["id"], [])
        preview = chunk_texts[0][:300] if chunk_texts else ""
        result.append({
            "id": doc["id"],
            "title": doc["title"],
            "chunk_count": len(chunk_texts),
            "preview": preview,
        })

    return {"documents": result}


@app.post("/api/admin/documents")
async def admin_add_document(request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if not landing_site_id:
        return JSONResponse({"error": "Landing site not initialized"}, status_code=500)

    body = await request.json()
    title = body.get("title", "").strip()
    text = body.get("text", "").strip()

    if not title or not text:
        return JSONResponse({"error": "Title and text are required"}, status_code=400)

    # Index the document (run in thread since embedding is CPU-bound)
    def do_index():
        c_hash = content_hash(text)
        doc_uid = uuid.uuid4().hex[:8]
        ins = sb.table("documents").insert({
            "site_id": landing_site_id,
            "url": f"admin://{LANDING_DOMAIN}/{doc_uid}",
            "title": title,
            "content_hash": c_hash,
        }).execute()
        doc_id = ins.data[0]["id"]

        chunks = chunk_text(text)
        if not chunks:
            return 0

        texts_to_embed = [f"passage: {c}" for c in chunks]
        embeddings = embed_model.encode(
            texts_to_embed, show_progress_bar=False, normalize_embeddings=True
        )

        rows = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            rows.append({
                "document_id": doc_id,
                "chunk_index": i,
                "text": chunk,
                "headings": [],
                "embedding": emb.tolist(),
            })

        sb.table("chunks").insert(rows).execute()
        return len(rows)

    chunk_count = await asyncio.to_thread(do_index)
    log.info(f"Admin added document '{title}' to landing site: {chunk_count} chunks")
    return {"success": True, "chunks": chunk_count}


@app.delete("/api/admin/documents/{doc_id}")
async def admin_delete_document(doc_id: int, request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Verify document belongs to landing site
    doc = sb.table("documents").select("site_id").eq("id", doc_id).execute()
    if not doc.data or doc.data[0]["site_id"] != landing_site_id:
        return JSONResponse({"error": "Document not found"}, status_code=404)

    # Delete chunks first, then document
    sb.table("chunks").delete().eq("document_id", doc_id).execute()
    sb.table("documents").delete().eq("id", doc_id).execute()

    log.info(f"Admin deleted document {doc_id} from landing site")
    return {"success": True}


@app.get("/api/admin/sites")
async def admin_list_sites(request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    sites = sb.table("sites").select("id, domain, language, is_trial, settings").execute()

    result = []
    for s in (sites.data or []):
        settings = s.get("settings") or {}
        doc_count = sb.table("documents").select("id", count="exact").eq("site_id", s["id"]).execute()
        result.append({
            "id": s["id"],
            "domain": s["domain"],
            "language": s.get("language"),
            "is_trial": s.get("is_trial", False),
            "is_landing": settings.get("landing", False),
            "doc_count": doc_count.count if doc_count.count is not None else 0,
        })

    return {"sites": result}


@app.delete("/api/admin/sites/{site_id}")
async def admin_delete_site(site_id: int, request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Don't allow deleting the landing site from here
    if site_id == landing_site_id:
        return JSONResponse({"error": "Cannot delete the landing site"}, status_code=400)

    # Verify site exists
    site = sb.table("sites").select("id, domain").eq("id", site_id).execute()
    if not site.data:
        return JSONResponse({"error": "Site not found"}, status_code=404)

    domain = site.data[0]["domain"]

    # Delete chunks → documents → site
    docs = sb.table("documents").select("id").eq("site_id", site_id).execute()
    for doc in (docs.data or []):
        sb.table("chunks").delete().eq("document_id", doc["id"]).execute()
    sb.table("documents").delete().eq("site_id", site_id).execute()
    sb.table("sites").delete().eq("id", site_id).execute()

    # Clean up trial progress if any
    trial_progress.pop(site_id, None)

    log.info(f"Admin deleted site {site_id} ({domain})")
    return {"success": True}


@app.get("/api/admin/sites/{site_id}/documents")
async def admin_site_documents(site_id: int, request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Get site info
    site = sb.table("sites").select("id, domain, language, settings").eq("id", site_id).execute()
    if not site.data:
        return JSONResponse({"error": "Site not found"}, status_code=404)

    # Get all documents for this site
    docs = sb.table("documents").select("id, title, url, content_hash, last_crawled").eq("site_id", site_id).execute()

    if not docs.data:
        return {"site": site.data[0], "documents": []}

    doc_ids = [doc["id"] for doc in docs.data]
    all_chunks = sb.table("chunks").select("id, document_id, chunk_index, text").in_("document_id", doc_ids).execute()

    chunks_by_doc = {}
    for c in (all_chunks.data or []):
        chunks_by_doc.setdefault(c["document_id"], []).append({
            "id": c["id"],
            "chunk_index": c["chunk_index"],
            "preview": c["text"][:200],
        })

    result = []
    for doc in docs.data:
        doc_chunks = chunks_by_doc.get(doc["id"], [])
        doc_chunks.sort(key=lambda x: x["chunk_index"])
        result.append({
            "id": doc["id"],
            "title": doc["title"],
            "url": doc["url"],
            "last_crawled": doc.get("last_crawled"),
            "chunk_count": len(doc_chunks),
            "chunks": doc_chunks,
        })

    return {"site": site.data[0], "documents": result}


@app.delete("/api/admin/sites/{site_id}/chunks/{chunk_id}")
async def admin_delete_chunk(site_id: int, chunk_id: int, request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Verify chunk belongs to this site
    chunk = sb.table("chunks").select("id, document_id").eq("id", chunk_id).execute()
    if not chunk.data:
        return JSONResponse({"error": "Chunk not found"}, status_code=404)

    doc_id = chunk.data[0]["document_id"]
    doc = sb.table("documents").select("site_id").eq("id", doc_id).execute()
    if not doc.data or doc.data[0]["site_id"] != site_id:
        return JSONResponse({"error": "Chunk does not belong to this site"}, status_code=403)

    sb.table("chunks").delete().eq("id", chunk_id).execute()
    log.info(f"Admin deleted chunk {chunk_id} from site {site_id}")
    return {"success": True}


@app.post("/api/admin/sites/{site_id}/upload-pdf")
async def admin_upload_pdf(site_id: int, request: Request, pdf: UploadFile = File(...)):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Verify site exists
    site = sb.table("sites").select("id, domain").eq("id", site_id).execute()
    if not site.data:
        return JSONResponse({"error": "Site not found"}, status_code=404)

    content = await pdf.read()
    if len(content) > MAX_PDF_SIZE:
        return JSONResponse({"error": f"PDF exceeds {MAX_PDF_SIZE // (1024*1024)}MB limit"}, status_code=400)

    def do_pdf_index():
        from pypdf import PdfReader as _PdfReader
        reader = _PdfReader(io.BytesIO(content))
        pdf_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text + "\n\n"

        if not pdf_text.strip():
            return {"error": "No text could be extracted from this PDF (possibly scanned/image PDF)"}

        safe_filename = re.sub(r'[^\w\s\-.]', '_', pdf.filename or "upload.pdf")
        c_hash = content_hash(pdf_text.strip())

        ins = sb.table("documents").insert({
            "site_id": site_id,
            "url": f"pdf://{safe_filename}",
            "title": pdf.filename or "Uploaded PDF",
            "content_hash": c_hash,
        }).execute()
        doc_id = ins.data[0]["id"]

        chunks = chunk_text(pdf_text.strip())
        if not chunks:
            return {"success": True, "doc_id": doc_id, "chunks": 0}

        texts_to_embed = [f"passage: {c}" for c in chunks]
        embeddings = embed_model.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)

        rows = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            rows.append({
                "document_id": doc_id,
                "chunk_index": i,
                "text": chunk,
                "headings": [],
                "embedding": emb.tolist(),
            })

        sb.table("chunks").insert(rows).execute()
        return {"success": True, "doc_id": doc_id, "chunks": len(rows), "filename": pdf.filename}

    result = await asyncio.to_thread(do_pdf_index)
    if "error" in result:
        return JSONResponse({"error": result["error"]}, status_code=400)

    log.info(f"Admin uploaded PDF '{pdf.filename}' to site {site_id}: {result['chunks']} chunks")
    return result


@app.post("/api/admin/sites/{site_id}/recrawl")
async def admin_recrawl_site(site_id: int, request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    max_pages = min(int(body.get("max_pages", 100)), 300)
    use_playwright = body.get("use_playwright", False)

    # Get site and source URL
    site = sb.table("sites").select("id, domain, settings").eq("id", site_id).execute()
    if not site.data:
        return JSONResponse({"error": "Site not found"}, status_code=404)

    settings = site.data[0].get("settings") or {}
    source_url = body.get("url") or settings.get("source_url") or f"https://{site.data[0]['domain']}"

    # Update source_url in settings if provided
    if body.get("url"):
        settings["source_url"] = body["url"]
        sb.table("sites").update({"settings": settings}).eq("id", site_id).execute()

    # Initialize progress
    trial_progress[site_id] = {
        "step": 0, "total": 0, "message": "Starting re-crawl...",
        "done": False, "error": None,
    }

    def do_recrawl():
        # Delete existing documents and chunks
        docs = sb.table("documents").select("id").eq("site_id", site_id).execute()
        for doc in (docs.data or []):
            sb.table("chunks").delete().eq("document_id", doc["id"]).execute()
        sb.table("documents").delete().eq("site_id", site_id).execute()
        # Re-index
        run_trial_indexing(site_id, source_url, max_pages, [], use_playwright)

    asyncio.get_event_loop().create_task(asyncio.to_thread(do_recrawl))

    log.info(f"Admin triggered re-crawl for site {site_id}: url={source_url} max_pages={max_pages}")
    return {"success": True, "site_id": site_id, "source_url": source_url, "message": "Re-crawl started"}


# ── Registration & Activation ─────────────────────────────────────────────

ACTIVATION_CODE = os.environ.get("ACTIVATION_CODE", "")


@app.post("/api/register")
async def register(request: Request):
    # Rate limit: 5 registrations per hour per IP
    blocked = rate_limit_check(request, "register", 5, 3600)
    if blocked:
        return blocked

    body = await request.json()
    name = body.get("name", "").strip()
    email = body.get("email", "").strip().lower()
    company = body.get("company", "").strip()

    if not name or not email:
        return JSONResponse({"error": "Name and email are required"}, status_code=400)

    # Check if already registered
    existing = sb.table("registrations").select("token").eq("email", email).execute()
    if existing.data:
        token = existing.data[0]["token"]
        log.info(f"Returning existing registration for {email}")
        return {"token": token, "message": "Welcome back"}

    # New registration
    token = f"wr_{uuid.uuid4().hex}"
    sb.table("registrations").insert({
        "name": name,
        "email": email,
        "company": company or None,
        "token": token,
    }).execute()

    log.info(f"New registration: {email} ({name})")
    return {"token": token, "message": "Registered successfully"}


@app.post("/api/activate")
async def activate(request: Request):
    # Rate limit: 10 activation attempts per hour per IP (brute-force protection)
    blocked = rate_limit_check(request, "activate", 10, 3600)
    if blocked:
        return blocked

    body = await request.json()
    code = body.get("code", "").strip()
    site_id = body.get("site_id")
    token = body.get("token", "").strip()
    max_pages = min(int(body.get("max_pages", 100)), 300)

    if not code or not site_id or not token:
        return JSONResponse({"error": "Code, site_id, and token are required"}, status_code=400)

    # Verify token
    reg = sb.table("registrations").select("email").eq("token", token).execute()
    if not reg.data:
        return JSONResponse({"error": "Invalid session"}, status_code=401)

    # Verify code
    if not ACTIVATION_CODE:
        return JSONResponse({"error": "Activation not configured"}, status_code=503)

    if code != ACTIVATION_CODE:
        return JSONResponse({"error": "Invalid activation code"}, status_code=403)

    # Verify site exists and is a trial
    site = sb.table("sites").select("*").eq("id", site_id).execute()
    if not site.data:
        return JSONResponse({"error": "Site not found"}, status_code=404)

    site_data = site.data[0]
    source_url = site_data.get("settings", {}).get("source_url", "")

    # Convert trial to permanent: remove trial flag and expiry
    sb.table("sites").update({
        "is_trial": False,
        "expires_at": None,
    }).eq("id", site_id).execute()

    # Re-index with more pages in background
    if source_url and max_pages > 0:
        trial_progress[site_id] = {
            "step": 0, "total": 0, "message": "Re-indexing with full page count...",
            "done": False, "error": None,
        }
        # Clear existing docs/chunks before re-indexing
        def reindex():
            docs = sb.table("documents").select("id").eq("site_id", site_id).execute()
            for doc in (docs.data or []):
                sb.table("chunks").delete().eq("document_id", doc["id"]).execute()
            sb.table("documents").delete().eq("site_id", site_id).execute()
            run_trial_indexing(site_id, source_url, max_pages, [])

        asyncio.get_event_loop().create_task(asyncio.to_thread(reindex))

    log.info(f"Site {site_id} activated by {reg.data[0]['email']} — re-indexing {max_pages} pages")

    return {
        "success": True,
        "site_id": site_id,
        "widget_code": f'<script src="{os.environ.get("PUBLIC_URL", "https://wrs.kz")}/widget.js" data-site-id="{site_id}"></script>',
        "source_url": source_url,
    }


@app.post("/api/quick-activate")
async def quick_activate(
    request: Request,
    url: str = Form(...),
    code: str = Form(...),
    max_pages: int = Form(default=100),
    language: str = Form(default=""),
    use_playwright: str = Form(default="0"),
    pdfs: list[UploadFile] = File(default=[]),
):
    """Activate directly from landing page — no trial/registration needed."""
    # Rate limit: 3 per hour per IP
    blocked = rate_limit_check(request, "quick_activate", 3, 3600)
    if blocked:
        return blocked

    code = code.strip()
    url = url.strip()
    max_pages = max(1, min(max_pages, 300))

    if not code or not url:
        return JSONResponse({"error": "Code and URL are required"}, status_code=400)

    if not ACTIVATION_CODE:
        return JSONResponse({"error": "Activation not configured"}, status_code=503)

    if code != ACTIVATION_CODE:
        return JSONResponse({"error": "Invalid activation code"}, status_code=403)

    # Validate URL
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)
    if not parsed.netloc:
        return JSONResponse({"error": "Invalid URL"}, status_code=400)

    domain = parsed.netloc

    # Read PDF bytes
    pdf_data = []
    for pdf in pdfs:
        content = await pdf.read()
        if len(content) > MAX_PDF_SIZE:
            return JSONResponse(
                {"error": f"PDF '{pdf.filename}' exceeds 10MB limit"},
                status_code=400,
            )
        pdf_data.append({"filename": pdf.filename, "content": content})

    # Create permanent site
    site_resp = sb.table("sites").insert({
        "domain": domain,
        "language": language or None,
        "is_trial": False,
        "settings": {"source_url": url},
    }).execute()
    site_id = site_resp.data[0]["id"]

    # Start indexing in background
    pw = use_playwright == "1"
    trial_progress[site_id] = {
        "step": 0, "total": 0, "message": "Starting full indexing...",
        "done": False, "error": None,
    }
    asyncio.get_event_loop().create_task(
        asyncio.to_thread(run_trial_indexing, site_id, url, max_pages, pdf_data, pw)
    )

    log.info(f"Quick-activate: site_id={site_id} domain={domain} max_pages={max_pages} pdfs={len(pdf_data)} playwright={pw}")

    return {
        "success": True,
        "site_id": site_id,
        "widget_code": f'<script src="{os.environ.get("PUBLIC_URL", "https://wrs.kz")}/widget.js" data-site-id="{site_id}"></script>',
    }


# ── Internal Assistants ──────────────────────────────────────────────────

@app.get("/assistant/{slug}")
async def serve_assistant_page(slug: str):
    """Serve the internal assistant page."""
    return FileResponse(WIDGET_DIR / "assistant.html", media_type="text/html")


@app.post("/api/internal/setup")
async def setup_internal_assistant(request: Request):
    """Create a new internal assistant with a purchase code."""
    blocked = rate_limit_check(request, "internal_setup", 3, 3600)
    if blocked:
        return blocked

    body = await request.json()
    code = body.get("code", "").strip()
    name = body.get("name", "").strip()
    slug = body.get("slug", "").strip().lower()
    admin_password = body.get("admin_password", "").strip()
    user_password = body.get("user_password", "").strip()
    language = body.get("language", "").strip()
    created_by = body.get("email", "").strip()

    if not code or not name or not slug or not admin_password or not user_password:
        return JSONResponse({"error": "All fields are required: code, name, slug, admin_password, user_password"}, status_code=400)

    # Validate setup code
    if not INTERNAL_SETUP_CODE:
        return JSONResponse({"error": "Internal assistant setup not configured"}, status_code=503)
    if code != INTERNAL_SETUP_CODE:
        return JSONResponse({"error": "Invalid setup code"}, status_code=403)

    # Validate slug format
    if not re.match(r'^[a-z0-9][a-z0-9\-]{1,48}[a-z0-9]$', slug):
        return JSONResponse({"error": "Slug must be 3-50 chars, lowercase letters, numbers, and hyphens only"}, status_code=400)

    if admin_password == user_password:
        return JSONResponse({"error": "Admin and user passwords must be different"}, status_code=400)

    if len(admin_password) < 4 or len(user_password) < 4:
        return JSONResponse({"error": "Passwords must be at least 4 characters"}, status_code=400)

    # Check slug uniqueness
    existing = sb.table("internal_assistants").select("id").eq("slug", slug).execute()
    if existing.data:
        return JSONResponse({"error": "This slug is already taken"}, status_code=409)

    # Create a site for this assistant
    domain = f"internal-{slug}"
    try:
        site_resp = sb.table("sites").insert({
            "domain": domain,
            "language": language or "en",
            "is_trial": False,
            "settings": {"internal_assistant": True, "slug": slug},
        }).execute()
    except Exception as e:
        if "duplicate" in str(e).lower():
            return JSONResponse({"error": "An assistant with a similar name already exists"}, status_code=409)
        raise
    site_id = site_resp.data[0]["id"]

    # Hash passwords
    admin_hash = bcrypt.hashpw(admin_password.encode(), bcrypt.gensalt()).decode()
    user_hash = bcrypt.hashpw(user_password.encode(), bcrypt.gensalt()).decode()

    # Create assistant record
    assistant = sb.table("internal_assistants").insert({
        "slug": slug,
        "name": name,
        "site_id": site_id,
        "admin_password": admin_hash,
        "user_password": user_hash,
        "created_by": created_by or None,
        "settings": {"language": language or "en"},
    }).execute()

    public_url = os.environ.get("PUBLIC_URL", "https://wrs.kz")
    log.info(f"Internal assistant created: {name} (slug={slug}, site_id={site_id})")

    return {
        "success": True,
        "slug": slug,
        "site_id": site_id,
        "url": f"{public_url}/assistant/{slug}",
    }


@app.post("/api/internal/{slug}/auth")
async def internal_assistant_auth(slug: str, request: Request):
    """Authenticate to an internal assistant. Returns role (admin/user) and a token."""
    blocked = rate_limit_check(request, f"internal_auth:{slug}", 10, 300)
    if blocked:
        return blocked

    body = await request.json()
    password = body.get("password", "").strip()

    if not password:
        return JSONResponse({"error": "Password is required"}, status_code=400)

    assistant = sb.table("internal_assistants").select("*").eq("slug", slug).execute()
    if not assistant.data:
        return JSONResponse({"error": "Assistant not found"}, status_code=404)

    a = assistant.data[0]

    # Check admin password first, then user password
    if bcrypt.checkpw(password.encode(), a["admin_password"].encode()):
        role = "admin"
    elif bcrypt.checkpw(password.encode(), a["user_password"].encode()):
        role = "user"
    else:
        return JSONResponse({"error": "Invalid password"}, status_code=401)

    # Generate a simple session token
    token = hashlib.sha256(f"{slug}:{role}:{uuid.uuid4().hex}".encode()).hexdigest()[:48]

    # Store in memory (similar to admin tokens)
    _internal_tokens[token] = {
        "slug": slug,
        "role": role,
        "site_id": a["site_id"],
        "assistant_id": a["id"],
        "created": time.time(),
    }

    return {
        "success": True,
        "role": role,
        "token": token,
        "name": a["name"],
        "site_id": a["site_id"],
        "settings": a.get("settings", {}),
    }


# In-memory token store for internal assistant sessions (TTL: 8 hours)
_internal_tokens: dict[str, dict] = {}
INTERNAL_TOKEN_TTL = 8 * 3600


def verify_internal(request: Request, slug: str, require_admin: bool = False) -> dict | None:
    """Verify internal assistant token. Returns session dict or None."""
    token = request.headers.get("x-internal-token", "")
    if not token or token not in _internal_tokens:
        return None
    session = _internal_tokens[token]
    # Check TTL
    if time.time() - session["created"] > INTERNAL_TOKEN_TTL:
        _internal_tokens.pop(token, None)
        return None
    # Check slug match
    if session["slug"] != slug:
        return None
    # Check admin requirement
    if require_admin and session["role"] != "admin":
        return None
    return session


@app.get("/api/internal/{slug}/documents")
async def internal_get_documents(slug: str, request: Request):
    """List documents and chunks for an internal assistant (admin only)."""
    session = verify_internal(request, slug, require_admin=True)
    if not session:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    site_id = session["site_id"]
    docs = sb.table("documents").select("id, title, url, content_hash, last_crawled").eq("site_id", site_id).execute()

    if not docs.data:
        return {"documents": []}

    doc_ids = [doc["id"] for doc in docs.data]
    all_chunks = sb.table("chunks").select("id, document_id, chunk_index, text").in_("document_id", doc_ids).execute()

    chunks_by_doc = {}
    for c in (all_chunks.data or []):
        chunks_by_doc.setdefault(c["document_id"], []).append({
            "id": c["id"],
            "chunk_index": c["chunk_index"],
            "preview": c["text"][:200],
        })

    result = []
    for doc in docs.data:
        doc_chunks = chunks_by_doc.get(doc["id"], [])
        doc_chunks.sort(key=lambda x: x["chunk_index"])
        result.append({
            "id": doc["id"],
            "title": doc["title"],
            "url": doc["url"],
            "last_crawled": doc.get("last_crawled"),
            "chunk_count": len(doc_chunks),
            "chunks": doc_chunks,
        })

    return {"documents": result}


@app.post("/api/internal/{slug}/add-text")
async def internal_add_text(slug: str, request: Request):
    """Add plain text content to an internal assistant (admin only)."""
    session = verify_internal(request, slug, require_admin=True)
    if not session:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    title = body.get("title", "").strip()
    text = body.get("text", "").strip()

    if not title or not text:
        return JSONResponse({"error": "Title and text are required"}, status_code=400)

    site_id = session["site_id"]

    def do_index():
        c_hash = content_hash(text)
        ins = sb.table("documents").insert({
            "site_id": site_id,
            "url": f"text://{title[:50]}",
            "title": title,
            "content_hash": c_hash,
        }).execute()
        doc_id = ins.data[0]["id"]

        chunks = chunk_text(text)
        if not chunks:
            return {"success": True, "doc_id": doc_id, "chunks": 0}

        texts_to_embed = [f"passage: {c}" for c in chunks]
        embeddings = embed_model.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)

        rows = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            rows.append({
                "document_id": doc_id,
                "chunk_index": i,
                "text": chunk,
                "headings": [],
                "embedding": emb.tolist(),
            })

        sb.table("chunks").insert(rows).execute()
        return {"success": True, "doc_id": doc_id, "chunks": len(rows)}

    result = await asyncio.to_thread(do_index)
    log.info(f"Internal assistant {slug}: added text '{title}' — {result.get('chunks', 0)} chunks")
    return result


@app.post("/api/internal/{slug}/upload-pdf")
async def internal_upload_pdf(slug: str, request: Request, pdf: UploadFile = File(...)):
    """Upload a PDF to an internal assistant (admin only)."""
    token = request.headers.get("x-internal-token", "")
    if not token or token not in _internal_tokens:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    session = _internal_tokens[token]
    if session["slug"] != slug or session["role"] != "admin":
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if time.time() - session["created"] > INTERNAL_TOKEN_TTL:
        _internal_tokens.pop(token, None)
        return JSONResponse({"error": "Session expired"}, status_code=401)

    site_id = session["site_id"]

    content = await pdf.read()
    if len(content) > MAX_PDF_SIZE:
        return JSONResponse({"error": f"PDF exceeds {MAX_PDF_SIZE // (1024*1024)}MB limit"}, status_code=400)

    def do_pdf_index():
        from pypdf import PdfReader as _PdfReader
        reader = _PdfReader(io.BytesIO(content))
        pdf_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text + "\n\n"

        if not pdf_text.strip():
            return {"error": "No text could be extracted from this PDF"}

        safe_filename = re.sub(r'[^\w\s\-.]', '_', pdf.filename or "upload.pdf")
        c_hash = content_hash(pdf_text.strip())

        ins = sb.table("documents").insert({
            "site_id": site_id,
            "url": f"pdf://{safe_filename}",
            "title": pdf.filename or "Uploaded PDF",
            "content_hash": c_hash,
        }).execute()
        doc_id = ins.data[0]["id"]

        chunks = chunk_text(pdf_text.strip())
        if not chunks:
            return {"success": True, "doc_id": doc_id, "chunks": 0}

        texts_to_embed = [f"passage: {c}" for c in chunks]
        embeddings = embed_model.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)

        rows = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            rows.append({
                "document_id": doc_id,
                "chunk_index": i,
                "text": chunk,
                "headings": [],
                "embedding": emb.tolist(),
            })

        sb.table("chunks").insert(rows).execute()
        return {"success": True, "doc_id": doc_id, "chunks": len(rows), "filename": pdf.filename}

    result = await asyncio.to_thread(do_pdf_index)
    if "error" in result:
        return JSONResponse({"error": result["error"]}, status_code=400)

    log.info(f"Internal assistant {slug}: uploaded PDF '{pdf.filename}' — {result['chunks']} chunks")
    return result


@app.post("/api/internal/{slug}/crawl")
async def internal_crawl(slug: str, request: Request):
    """Crawl a website and add its content to an internal assistant (admin only)."""
    session = verify_internal(request, slug, require_admin=True)
    if not session:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    url = body.get("url", "").strip()
    max_pages = min(int(body.get("max_pages", 50)), 200)

    if not url:
        return JSONResponse({"error": "URL is required"}, status_code=400)

    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"

    site_id = session["site_id"]

    trial_progress[site_id] = {
        "step": 0, "total": 0, "message": "Starting crawl...",
        "done": False, "error": None,
    }

    def do_crawl():
        run_trial_indexing(site_id, url, max_pages, [], False)

    asyncio.get_event_loop().create_task(asyncio.to_thread(do_crawl))

    log.info(f"Internal assistant {slug}: started crawl url={url} max_pages={max_pages}")
    return {"success": True, "site_id": site_id, "message": "Crawl started"}


@app.delete("/api/internal/{slug}/chunks/{chunk_id}")
async def internal_delete_chunk(slug: str, chunk_id: int, request: Request):
    """Delete a specific chunk from an internal assistant (admin only)."""
    session = verify_internal(request, slug, require_admin=True)
    if not session:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    site_id = session["site_id"]

    chunk = sb.table("chunks").select("id, document_id").eq("id", chunk_id).execute()
    if not chunk.data:
        return JSONResponse({"error": "Chunk not found"}, status_code=404)

    doc = sb.table("documents").select("site_id").eq("id", chunk.data[0]["document_id"]).execute()
    if not doc.data or doc.data[0]["site_id"] != site_id:
        return JSONResponse({"error": "Chunk does not belong to this assistant"}, status_code=403)

    sb.table("chunks").delete().eq("id", chunk_id).execute()
    log.info(f"Internal assistant {slug}: deleted chunk {chunk_id}")
    return {"success": True}


@app.delete("/api/internal/{slug}/documents/{doc_id}")
async def internal_delete_document(slug: str, doc_id: int, request: Request):
    """Delete a document and all its chunks from an internal assistant (admin only)."""
    session = verify_internal(request, slug, require_admin=True)
    if not session:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    site_id = session["site_id"]

    doc = sb.table("documents").select("site_id, title").eq("id", doc_id).execute()
    if not doc.data or doc.data[0]["site_id"] != site_id:
        return JSONResponse({"error": "Document not found"}, status_code=404)

    sb.table("chunks").delete().eq("document_id", doc_id).execute()
    sb.table("documents").delete().eq("id", doc_id).execute()

    log.info(f"Internal assistant {slug}: deleted document {doc_id}")
    return {"success": True}


# ── Admin: Internal Assistants management ────────────────────────────────

@app.get("/api/admin/internal-assistants")
async def admin_list_internal_assistants(request: Request):
    """List all internal assistants (admin dashboard)."""
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    assistants = sb.table("internal_assistants").select("id, slug, name, site_id, created_by, settings, created_at").execute()

    result = []
    for a in (assistants.data or []):
        # Get doc/chunk count for each
        docs = sb.table("documents").select("id", count="exact").eq("site_id", a["site_id"]).execute()
        chunks_count = 0
        if docs.data:
            doc_ids = [d["id"] for d in docs.data]
            if doc_ids:
                chunks_resp = sb.table("chunks").select("id", count="exact").in_("document_id", doc_ids).execute()
                chunks_count = len(chunks_resp.data) if chunks_resp.data else 0

        result.append({
            "id": a["id"],
            "slug": a["slug"],
            "name": a["name"],
            "site_id": a["site_id"],
            "created_by": a.get("created_by"),
            "settings": a.get("settings", {}),
            "created_at": a["created_at"],
            "doc_count": len(docs.data) if docs.data else 0,
            "chunk_count": chunks_count,
        })

    return {"assistants": result}


@app.delete("/api/admin/internal-assistants/{assistant_id}")
async def admin_delete_internal_assistant(assistant_id: int, request: Request):
    """Delete an internal assistant and its site (admin dashboard)."""
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    assistant = sb.table("internal_assistants").select("id, slug, site_id").eq("id", assistant_id).execute()
    if not assistant.data:
        return JSONResponse({"error": "Assistant not found"}, status_code=404)

    a = assistant.data[0]

    # Delete chunks, documents, site (cascade should handle it, but be explicit)
    docs = sb.table("documents").select("id").eq("site_id", a["site_id"]).execute()
    for doc in (docs.data or []):
        sb.table("chunks").delete().eq("document_id", doc["id"]).execute()
    sb.table("documents").delete().eq("site_id", a["site_id"]).execute()
    sb.table("internal_assistants").delete().eq("id", assistant_id).execute()
    sb.table("sites").delete().eq("id", a["site_id"]).execute()

    # Clean up any tokens for this slug
    to_remove = [t for t, s in _internal_tokens.items() if s["slug"] == a["slug"]]
    for t in to_remove:
        _internal_tokens.pop(t, None)

    log.info(f"Admin deleted internal assistant {a['slug']} (id={assistant_id})")
    return {"success": True}
