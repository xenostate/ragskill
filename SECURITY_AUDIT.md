# Security Audit — RagSkill

**Date:** 2026-04-11
**Scope:** Full codebase review (scripts/, widget/, references/)

---

## Vulnerabilities Fixed

### 1. SSRF (Server-Side Request Forgery) — SEVERITY: HIGH

**Status:** FIXED

**Location:** `scripts/server.py` (URL validation in `/api/trial/start`, `/api/quick-activate`, `/api/internal/{slug}/crawl`, `/api/admin/sites/{site_id}/recrawl`)

**Problem:** User-supplied URLs were validated only for having a scheme and netloc, but never checked against private/internal IP ranges. An attacker could submit URLs like:
- `http://169.254.169.254/latest/meta-data/` (AWS/GCP instance metadata)
- `http://127.0.0.1:8090/admin` (localhost services)
- `http://10.0.0.1/` (internal network)
- `http://[::1]/` (IPv6 loopback)

The crawler would fetch the content and index it, leaking internal data through the RAG chatbot.

**Fix:** Added `is_url_safe()` validator that resolves hostnames to IPs and blocks RFC-1918 private ranges, loopback, link-local, multicast, and cloud metadata addresses. Applied to all four URL ingestion endpoints.

---

### 2. CORS Misconfiguration — SEVERITY: HIGH

**Status:** FIXED

**Location:** `scripts/server.py` (CORS middleware)

**Problem:** CORS was configured with `allow_origins=["*"]` and `allow_credentials=True`. While browsers technically block `Access-Control-Allow-Credentials` with a `*` origin, the combination signals intent to accept credentials from any origin, and some misconfigured proxies/CDNs may not enforce the browser restriction. More critically, it allowed any malicious site to make cross-origin requests to the admin API endpoints.

**Fix:** Removed `allow_credentials=True` (the API uses custom headers for auth, not cookies, so credentials mode is unnecessary). Added `X-Admin-Token` and `X-Internal-Token` to `allow_headers` so browser preflight requests for admin/internal endpoints actually work. Kept `allow_origins=["*"]` because the widget must work on any customer's domain.

---

### 3. X-Forwarded-For Spoofing — SEVERITY: MEDIUM

**Status:** FIXED

**Location:** `scripts/server.py` (`get_client_ip()`)

**Problem:** `get_client_ip()` blindly trusted the first value in the `X-Forwarded-For` header. Without a trusted proxy allowlist, any direct client can set this header to an arbitrary IP, completely bypassing all rate limiting (chat, admin auth, trial start, etc.).

**Fix:** Added `TRUSTED_PROXIES` configuration (env var `TRUSTED_PROXIES`, comma-separated CIDRs/IPs). `X-Forwarded-For` is only trusted when the direct connecting IP (`request.client.host`) is in the trusted proxy list. Uses rightmost-untrusted-IP strategy for correct proxy chain resolution. Defaults to trusting `127.0.0.1` and `::1` (localhost) for common reverse proxy setups.

---

### 4. Domain Verification is Client-Controlled — SEVERITY: MEDIUM

**Status:** FIXED

**Location:** `scripts/server.py` (`/api/chat` endpoint)

**Problem:** The `origin_domain` field was sent by widget JavaScript, which is trivially spoofable. Anyone could POST directly to `/api/chat` with any `origin_domain` value.

**Fix:** Now reads the browser-set `Origin` HTTP header first (not controllable by JavaScript on cross-origin requests). Falls back to the client-sent `origin_domain` field only for backward compatibility with non-browser clients.

---

### 5. No MIME Validation on PDF Uploads — SEVERITY: MEDIUM

**Status:** FIXED

**Location:** `scripts/server.py` (all 3 PDF upload paths: trial, admin, internal)

**Problem:** Only file size was checked. No magic-bytes validation ensured the uploaded file was actually a PDF. Crafted non-PDF files could reach `PdfReader` and potentially trigger parser vulnerabilities.

**Fix:** Added `is_valid_pdf()` that checks for `%PDF-` magic bytes before passing to PdfReader. Applied to all three PDF ingestion paths: trial upload, admin upload, and internal assistant upload.

---

### 6. Weak Password Policy — SEVERITY: LOW-MEDIUM

**Status:** ACCEPTED (not fixing)

**Location:** `scripts/server.py` (internal assistant setup)

**Problem:** Internal assistant passwords only require 4 characters minimum. Combined with rate limiting of 10 attempts per 5 minutes, a determined attacker could enumerate short passwords.

**Note:** Accepted for now as the project is not publicly known. Should be revisited before wider launch.

---

### 7. In-Memory State Lost on Restart — SEVERITY: LOW

**Status:** FIXED

**Location:** `scripts/server.py` (token stores)

**Problem:** Admin tokens and internal assistant tokens were stored in Python dicts. Server restarts wiped all active sessions, forcing re-login.

**Fix:** Added file-backed persistence (`.tokens.json`). Tokens are saved to disk on creation and on shutdown, loaded on startup with expiry filtering. Session history and rate limiter state remain ephemeral (acceptable to lose on restart). File added to `.gitignore`.

---

### 8. Admin Token Dict Grows Unbounded — SEVERITY: LOW

**Status:** FIXED

**Location:** `scripts/server.py` (hourly cleanup task)

**Problem:** Expired admin tokens were only removed when someone tried to use them. Stale entries from unused tokens accumulated indefinitely. Same issue with `_internal_tokens`.

**Fix:** Added periodic cleanup of both `admin_tokens` and `_internal_tokens` in the hourly background task. Expired entries are purged and the token file is updated.

---

### 9. Trial Progress Dict Never Cleaned — SEVERITY: LOW

**Status:** FIXED

**Location:** `scripts/server.py` (`trial_progress` dict)

**Problem:** If a trial indexing task finished (successfully or with error), its `trial_progress[site_id]` entry stayed in memory forever.

**Fix:** Added `_finished_at` timestamp when indexing completes. The hourly cleanup task now removes entries that have been done for more than 1 hour.

---

### 10. Blocking Embedding Model in Async Server — SEVERITY: LOW

**Status:** FIXED

**Location:** `scripts/server.py` (`asyncio.to_thread` calls)

**Problem:** `embed_model.encode()` is CPU-heavy and ran in the default `ThreadPoolExecutor` (size `min(32, cpu_count+4)`). Under load, embedding threads could saturate the pool and block other async operations like health checks.

**Fix:** Created a bounded `ThreadPoolExecutor` with configurable size (`THREAD_POOL_SIZE` env var, default 4) and set it as the event loop's default executor. This limits concurrent embedding calls and keeps async handlers responsive. Executor is shut down cleanly on server stop.

---

### 11. Playwright Renderer Shares Single Page — SEVERITY: LOW

**Status:** FIXED

**Location:** `scripts/indexer.py` (`PlaywrightRenderer`)

**Problem:** The `PlaywrightRenderer` created one browser context and one page, then reused it across all navigations. If used concurrently from multiple threads (multiple trial crawls), this would cause race conditions. Also, `ignore_https_errors=True` was set, making crawling vulnerable to MITM.

**Fix:** Each `fetch()` call now creates a fresh browser context and page, which is closed after use. This isolates state between navigations and is safe for concurrent use. Removed `ignore_https_errors=True` so TLS is properly validated.

---

## Architecture & Design Improvements

### 12. Monolithic server.py — SEVERITY: DESIGN

**Status:** FIXED

**Location:** `scripts/server.py` (was ~2100 lines)

**Problem:** All endpoints, config, utils, and RAG logic were in a single file. This made the codebase hard to navigate, review, and test.

**Fix:** Split into focused modules using FastAPI `APIRouter`:
- `scripts/config.py` — shared config, constants, mutable globals, token persistence
- `scripts/utils.py` — rate limiter, SSRF protection, IP resolution, PDF validation, auth helpers
- `scripts/rag_core.py` — retrieval, context building, answer generation, RAG pipeline
- `scripts/routes/chat.py` — chat, health, widget.js endpoints
- `scripts/routes/trial.py` — trial flow, background indexing
- `scripts/routes/admin.py` — admin dashboard, site management, landing page
- `scripts/routes/auth.py` — registration and activation
- `scripts/routes/internal.py` — internal assistant CRUD
- `scripts/routes/whatsapp.py` — WhatsApp webhook endpoints
- `scripts/server.py` — slim ~200 line entry point (lifespan, middleware, router registration)

Also added `from __future__ import annotations` to all modules for Python 3.9 compatibility.

---

### 13. No Automated Tests — SEVERITY: DESIGN

**Status:** FIXED

**Location:** `tests/` (new)

**Problem:** No test suite existed. Regressions in critical paths (auth, rate limiting, SSRF, chunking) could go unnoticed.

**Fix:** Added 66 unit tests in `tests/test_utils.py` and `tests/test_indexer.py` covering:
- **RateLimiter** — under/over limit, window expiry, key isolation, thread safety, cleanup
- **SSRF protection** — private IPs, loopback, metadata, link-local, scheme validation, unresolvable hosts
- **PDF validation** — magic bytes check
- **IP resolution** — direct IP, trusted/untrusted proxy, proxy chains
- **Auth helpers** — admin token (valid/expired/missing), internal token (admin/user roles, slug matching, expiry)
- **HTML cleaning** — title extraction, script/nav/footer stripping, reference removal, table conversion
- **Text chunking** — empty text, paragraph boundaries, large text splitting, overlap, multi-paragraph
- **Sentence splitting** — basic split, abbreviation handling, question/exclamation marks
- **Link extraction** — same-domain, media filtering, fragment removal, relative URL resolution
- **Content hashing** — determinism, uniqueness, format

Tests use `conftest.py` to mock heavy dependencies (sentence-transformers, supabase, openai) so they run fast without the full server environment.

Run: `python3 -m pytest tests/ -v`

---

### 14. Service Key Used for All Operations — SEVERITY: MEDIUM

**Status:** FIXED

**Location:** `scripts/config.py`, `scripts/server.py`, `scripts/rag_core.py`, `scripts/routes/chat.py`

**Problem:** The Supabase service key (bypasses Row Level Security) was used for every database operation, including public-facing endpoints like `/api/chat`. If RLS policies were configured, they would be ignored.

**Fix:** Added `SUPABASE_ANON_KEY` config and a second Supabase client (`cfg.sb_public`). Public-facing read-only operations (chunk retrieval, site language lookup, domain verification) now use the anon-key client when configured. Falls back to service key if `SUPABASE_ANON_KEY` is not set.

**Setup required:** Set `SUPABASE_ANON_KEY` in `.env` and configure RLS policies on the `sites` and `chunks` tables to allow `SELECT` for the anon role.

---

### 15. No Request Body Size Limits — SEVERITY: LOW-MEDIUM

**Status:** FIXED

**Location:** `scripts/server.py` (new `BodySizeLimitMiddleware`)

**Problem:** No limit on incoming request body size. A malicious client could send a multi-GB payload, causing the server to buffer it into memory and potentially OOM.

**Fix:** Added `BodySizeLimitMiddleware` that checks `Content-Length` against `MAX_BODY_SIZE` (15 MB, configurable in `config.py`) and returns HTTP 413 for oversized requests. This runs before any route handler or file parsing. The 15 MB default allows the 10 MB PDF limit plus multipart encoding overhead.

---

## Remaining Recommendations

### Security
- **Security headers** — Add middleware for `X-Frame-Options`, `X-Content-Type-Options`, `Content-Security-Policy` on HTML responses
- **CSRF tokens** — Consider adding for admin/internal endpoints (currently mitigated by custom header auth)

### Product Features
- **Streaming chat responses** — SSE from OpenAI to client for lower perceived latency
- **Analytics dashboard** — per-site query volume, confidence distribution, top questions, unanswered queries
- **Semantic caching** — cache embeddings and retrieval results for repeated/similar queries
- **Chunk quality scoring** — auto-flag low-quality chunks (nav fragments, cookie banners)
- **Per-document re-indexing** — update a single page without recrawling the entire site
- **Webhook callbacks** — notify customers when indexing completes
- **Multi-model embeddings** — support OpenAI embeddings as alternative to local model

### Code Quality
- **Dependency health check** — extend `/health` to verify Supabase and OpenAI connectivity
