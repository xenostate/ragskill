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
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from pypdf import PdfReader

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from supabase import create_client

from scripts.telegram_handler import TelegramHandler

# ── Config ──────────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
RAG_MODEL = os.environ.get("RAG_MODEL", "gpt-4o-mini")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("OPENAI_KEY", ""))
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

WIDGET_DIR = Path(__file__).resolve().parent.parent / "widget"
SCRIPTS_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag-server")

# ── Globals (warm on startup) ──────────────────────────────────────────────

embed_model: SentenceTransformer | None = None
sb = None
openai_client: OpenAI | None = None
tg_handler: TelegramHandler | None = None
start_time = 0.0

# Trial indexing progress: { site_id: { step, total, message, done, error } }
trial_progress: dict[int, dict] = {}

MAX_PDF_SIZE = 10 * 1024 * 1024  # 10 MB

# ── RAG prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise assistant that answers questions using ONLY the provided source chunks.

Rules:
1. Answer ONLY from the sources below. If the sources don't contain the answer, say "I don't have enough information from the indexed sources to answer this."
2. Cite every claim with [N] where N is the source number.
3. If multiple sources support a claim, cite all of them: [1][3].
4. At the end, list all used sources as:
   Sources:
   [1] Title — URL
   [2] Title — URL
5. Keep answers concise and factual. Do not speculate beyond what the sources state.
6. If confidence is "low", preface with: "⚠️ Low confidence — the indexed sources may not cover this topic well."
7. Answer in the same language as the user's question.
"""

LANGUAGE_NAMES = {
    "en": "English", "ru": "Russian", "es": "Spanish", "fr": "French",
    "de": "German", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "pt": "Portuguese", "ar": "Arabic", "hi": "Hindi", "it": "Italian",
    "tr": "Turkish", "nl": "Dutch", "pl": "Polish", "uk": "Ukrainian",
}


def get_system_prompt(language: str | None = None) -> str:
    """Return system prompt, optionally with strict language enforcement."""
    if language and language in LANGUAGE_NAMES:
        lang_name = LANGUAGE_NAMES[language]
        return SYSTEM_PROMPT + f"\nCRITICAL: You MUST reply strictly in {lang_name}. Every part of your response — the answer, the source list, everything — must be in {lang_name}. No exceptions.\n"
    return SYSTEM_PROMPT


# ── Lifespan (model loading) ──────────────────────────────────────────────

def rag_fn(site_id: int, query: str) -> dict:
    """RAG function passed to TelegramHandler — uses warm model."""
    retrieval = retrieve_chunks(site_id, query)
    context = build_context(retrieval["results"])
    language = get_site_language(site_id)
    answer = generate_answer(query, context, retrieval["confidence"], language)
    sources = [
        {"title": r["title"], "url": r["url"], "score": r["score"]}
        for r in retrieval["results"]
    ]
    return {"answer": answer, "sources": sources, "confidence": retrieval["confidence"]}


async def cleanup_expired_trials():
    """Periodic task: delete expired trial sites (cascade deletes docs + chunks)."""
    while True:
        await asyncio.sleep(3600)  # run every hour
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, sb, openai_client, tg_handler, start_time
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

    if TELEGRAM_BOT_TOKEN:
        tg_handler = TelegramHandler(TELEGRAM_BOT_TOKEN, sb, rag_fn)
        log.info("Telegram handler initialized")
    else:
        log.warning("No TELEGRAM_BOT_TOKEN — Telegram bot disabled")

    # Start trial cleanup background task
    cleanup_task = asyncio.create_task(cleanup_expired_trials())
    log.info("Trial cleanup task started (runs hourly)")

    yield

    cleanup_task.cancel()
    log.info("Shutting down")


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(title="web-rag API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    site_id: int
    query: str
    session_id: str | None = None
    top_k: int = RAG_TOP_K


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
    elif results[0]["score"] >= 0.75:
        confidence = "high"
    elif results[0]["score"] >= 0.5:
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
async def chat(req: ChatRequest):
    t0 = time.time()

    # 1. Retrieve
    retrieval = retrieve_chunks(req.site_id, req.query, req.top_k)
    context = build_context(retrieval["results"])

    # 2. Generate answer (with language enforcement)
    language = get_site_language(req.site_id)
    answer = generate_answer(req.query, context, retrieval["confidence"], language)

    # 3. Build source list (slim version for response)
    sources = [
        {"title": r["title"], "url": r["url"], "score": r["score"]}
        for r in retrieval["results"]
    ]

    log.info(
        f"chat site={req.site_id} q=\"{req.query[:50]}\" "
        f"confidence={retrieval['confidence']} chunks={len(retrieval['results'])} "
        f"time={time.time()-t0:.2f}s"
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence=retrieval["confidence"],
    )


@app.get("/widget.js")
async def serve_widget():
    js_path = WIDGET_DIR / "widget.js"
    if not js_path.exists():
        return JSONResponse({"error": "widget.js not found"}, status_code=404)
    return FileResponse(js_path, media_type="application/javascript")


# ── Telegram webhook ───────────────────────────────────────────────────────

@app.post("/api/telegram")
async def telegram_webhook(request: Request):
    if tg_handler is None:
        return JSONResponse({"error": "Telegram not configured"}, status_code=503)

    body = await request.json()
    log.info(f"Telegram update: {json.dumps(body)[:200]}")

    # Process in background-ish (synchronous but fast return isn't critical here)
    try:
        tg_handler.handle_update(body)
    except Exception as e:
        log.error(f"Telegram handler error: {e}")

    return {"ok": True}


# ── Telegram webhook registration helper ──────────────────────────────────

@app.post("/api/telegram/set-webhook")
async def set_telegram_webhook(request: Request):
    """Call this once to register the webhook URL with Telegram."""
    if not TELEGRAM_BOT_TOKEN:
        return JSONResponse({"error": "No TELEGRAM_BOT_TOKEN"}, status_code=503)

    body = await request.json()
    webhook_url = body.get("url")
    if not webhook_url:
        return JSONResponse({"error": "Provide {\"url\": \"https://your-domain/api/telegram\"}"}, status_code=400)

    import requests as http_req
    resp = http_req.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook",
        json={"url": webhook_url},
        timeout=10,
    )
    return resp.json()


# ── Trial / Demo ──────────────────────────────────────────────────────────

# Import reusable functions from indexer
from scripts.indexer import (
    clean_html, chunk_text, extract_headings, extract_links,
    content_hash, StaticRenderer,
)


@app.get("/trial")
async def serve_trial_page():
    html_path = WIDGET_DIR / "trial.html"
    if not html_path.exists():
        return JSONResponse({"error": "trial.html not found"}, status_code=404)
    return FileResponse(html_path, media_type="text/html")


def run_trial_indexing(site_id: int, url: str, max_pages: int, pdf_data: list[dict]):
    """Synchronous trial indexing — runs in asyncio.to_thread."""
    progress = trial_progress[site_id]

    try:
        all_docs = []  # list of (url_or_filename, title, text, html_or_none)

        # ── Phase 1: Crawl the URL ──
        progress["message"] = f"Crawling {url}..."
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
                    all_docs.append((
                        f"pdf://{pdf_item['filename']}",
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
    url: str = Form(...),
    max_pages: int = Form(default=5),
    language: str = Form(default="en"),
    pdfs: list[UploadFile] = File(default=[]),
):
    # Validate URL
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)
    if not parsed.netloc:
        return JSONResponse({"error": "Invalid URL"}, status_code=400)

    # Cap max pages at 50
    max_pages = min(max_pages, 50)

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
    asyncio.get_event_loop().create_task(
        asyncio.to_thread(run_trial_indexing, site_id, url, max_pages, pdf_data)
    )

    log.info(f"Trial started: site_id={site_id} url={url} pdfs={len(pdf_data)} max_pages={max_pages}")
    return {"site_id": site_id, "message": "Indexing started", "expires_at": expires_at}


@app.get("/api/trial/progress/{site_id}")
async def trial_progress_stream(site_id: int):
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
