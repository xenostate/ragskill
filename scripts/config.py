"""
Shared configuration, globals, and constants for the RAG server.

All mutable server state lives here so modules can import it without circular deps.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Config ──────────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
RAG_MODEL = os.environ.get("RAG_MODEL", "gpt-4o-mini")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("OPENAI_KEY", ""))

# WhatsApp Business (dormant — set WHATSAPP_ENABLED=true to activate)
WHATSAPP_ENABLED = os.environ.get("WHATSAPP_ENABLED", "").lower() in ("true", "1", "yes")
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
WHATSAPP_APP_SECRET = os.environ.get("WHATSAPP_APP_SECRET", "")

# Trusted reverse proxies whose X-Forwarded-For header we respect.
_raw_proxies = os.environ.get("TRUSTED_PROXIES", "127.0.0.1,::1")
TRUSTED_PROXY_NETS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
for _p in _raw_proxies.split(","):
    _p = _p.strip()
    if _p:
        try:
            TRUSTED_PROXY_NETS.append(ipaddress.ip_network(_p, strict=False))
        except ValueError:
            pass

WIDGET_DIR = Path(__file__).resolve().parent.parent / "widget"
SCRIPTS_DIR = Path(__file__).resolve().parent

MAX_PDF_SIZE = 10 * 1024 * 1024   # 10 MB
MAX_BODY_SIZE = 15 * 1024 * 1024  # 15 MB (covers PDF + multipart overhead)

ADMIN_CODE = os.environ.get("ADMIN_CODE", os.environ.get("ACTIVATION_CODE", ""))
ACTIVATION_CODE = os.environ.get("ACTIVATION_CODE", "")
INTERNAL_SETUP_CODE = os.environ.get("INTERNAL_SETUP_CODE", os.environ.get("ACTIVATION_CODE", ""))

LANDING_DOMAIN = "landing.wrs.kz"

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag-server")

# ── Mutable globals (warm on startup) ───────────────────────────────────────

embed_model = None         # SentenceTransformer, set in lifespan
sb = None                  # Supabase client (service key — full access), set in lifespan
sb_public = None           # Supabase client (anon key — RLS-restricted), set in lifespan
openai_client = None       # OpenAI client, set in lifespan
whatsapp_handler = None    # WhatsAppHandler, set in lifespan if enabled
start_time = 0.0

# Trial indexing progress: { site_id: { step, total, message, done, error } }
trial_progress: dict[int, dict] = {}

# Landing page chat site_id (set on startup or via /api/landing/setup)
landing_site_id: int | None = None

# Bounded thread pool for CPU-heavy work (embedding, PDF parsing).
_THREAD_POOL_SIZE = int(os.environ.get("THREAD_POOL_SIZE", "4"))
thread_pool = ThreadPoolExecutor(max_workers=_THREAD_POOL_SIZE, thread_name_prefix="rag-worker")

# ── Auth token stores ───────────────────────────────────────────────────────

admin_tokens: dict[str, float] = {}          # token -> expiry_timestamp
ADMIN_TOKEN_TTL = 3600 * 8                   # 8 hours

_internal_tokens: dict[str, dict] = {}       # token -> session_dict
INTERNAL_TOKEN_TTL = 8 * 3600               # 8 hours

# ── Session history (conversation context) ──────────────────────────────────

_session_history: dict[str, list[dict]] = {}   # session_id -> [{role, content}, ...]
_session_last_access: dict[str, float] = {}     # session_id -> timestamp
_session_lock = threading.Lock()
SESSION_HISTORY_LIMIT = 5   # max exchanges (10 messages) per session
SESSION_TTL = 3600          # evict sessions idle for 1 hour

# ── Site language cache ─────────────────────────────────────────────────────

_site_lang_cache: dict[int, tuple] = {}  # site_id -> (lang, timestamp)
LANG_CACHE_TTL = 3600  # 1 hour

# ── Token persistence ───────────────────────────────────────────────────────

_TOKEN_FILE = Path(__file__).resolve().parent.parent / ".tokens.json"


def save_tokens():
    """Persist admin + internal tokens to disk so they survive restarts."""
    try:
        data = {
            "admin": {t: exp for t, exp in admin_tokens.items()},
            "internal": {t: s for t, s in _internal_tokens.items()},
        }
        _TOKEN_FILE.write_text(json.dumps(data), encoding="utf-8")
    except Exception as e:
        log.error(f"Failed to save tokens: {e}")


def load_tokens():
    """Restore tokens from disk, discarding any that have expired."""
    if not _TOKEN_FILE.exists():
        return
    try:
        data = json.loads(_TOKEN_FILE.read_text(encoding="utf-8"))
        now = time.time()
        for token, expiry in data.get("admin", {}).items():
            if expiry > now:
                admin_tokens[token] = expiry
        for token, session in data.get("internal", {}).items():
            if now - session.get("created", 0) < INTERNAL_TOKEN_TTL:
                _internal_tokens[token] = session
        log.info(f"Restored {len(admin_tokens)} admin + {len(_internal_tokens)} internal token(s) from disk")
    except Exception as e:
        log.warning(f"Could not load tokens from disk: {e}")
