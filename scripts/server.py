#!/usr/bin/env python3
"""
RAG API Server — FastAPI service with warm embedding model.

Endpoints:
    POST /api/chat          — widget/client chat
    POST /api/telegram      — Telegram webhook
    GET  /widget.js         — serve the chat widget
    GET  /health            — status check

Start:
    uvicorn scripts.server:app --host 0.0.0.0 --port 8090
"""

import json
import os
import sys
import time
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

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


# ── Lifespan (model loading) ──────────────────────────────────────────────

def rag_fn(site_id: int, query: str) -> dict:
    """RAG function passed to TelegramHandler — uses warm model."""
    retrieval = retrieve_chunks(site_id, query)
    context = build_context(retrieval["results"])
    answer = generate_answer(query, context, retrieval["confidence"])
    sources = [
        {"title": r["title"], "url": r["url"], "score": r["score"]}
        for r in retrieval["results"]
    ]
    return {"answer": answer, "sources": sources, "confidence": retrieval["confidence"]}


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

    yield

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


def generate_answer(query: str, context: str, confidence: str) -> str:
    """Call OpenAI to generate a cited answer."""
    if openai_client is None:
        return "LLM not configured. Set OPENAI_API_KEY in .env to enable answers."

    user_msg = f"Source chunks:\n{context}\n\nQuestion: {query}"

    resp = openai_client.chat.completions.create(
        model=RAG_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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

    # 2. Generate answer
    answer = generate_answer(req.query, context, retrieval["confidence"])

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
