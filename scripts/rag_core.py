"""
Core RAG pipeline: retrieval, context building, answer generation.
"""

from __future__ import annotations

import re
import time

import scripts.config as cfg


# ── RAG prompt ──────────────────────────────────────────────────────────────

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
        return SYSTEM_PROMPT + (
            f"\nCRITICAL: You MUST reply strictly in {lang_name}. "
            f"Every part of your response — the answer, the source list, everything — "
            f"must be in {lang_name}. No exceptions.\n"
        )
    return SYSTEM_PROMPT


# ── Retrieval ───────────────────────────────────────────────────────────────

def retrieve_chunks(site_id: int, query: str, top_k: int = 5) -> dict:
    """Embed query + call match_chunks. Uses warm model."""
    query_embedding = cfg.embed_model.encode(
        f"query: {query}", normalize_embeddings=True
    ).tolist()

    # Use public (anon-key) client for retrieval — respects RLS when configured
    client = cfg.sb_public or cfg.sb
    resp = client.rpc("match_chunks", {
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
        if len(results) > 1:
            avg_rest = sum(r["score"] for r in results[1:]) / len(results[1:])
            score_gap = top_score - avg_rest
        else:
            score_gap = 0

        if top_score >= 0.75 and score_gap < 0.4:
            confidence = "high"
        elif top_score >= 0.75 and score_gap >= 0.4:
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


# ── Language helpers ────────────────────────────────────────────────────────

def get_site_language(site_id: int) -> str | None:
    try:
        client = cfg.sb_public or cfg.sb
        resp = client.table("sites").select("language").eq("id", site_id).single().execute()
        return resp.data.get("language") if resp.data else None
    except Exception:
        return None


def get_site_language_cached(site_id: int) -> str | None:
    now = time.time()
    cached = cfg._site_lang_cache.get(site_id)
    if cached and now - cached[1] < cfg.LANG_CACHE_TTL:
        return cached[0]
    lang = get_site_language(site_id)
    cfg._site_lang_cache[site_id] = (lang, now)
    return lang


# ── Answer generation ───────────────────────────────────────────────────────

def generate_answer(query: str, context: str, confidence: str, language: str | None = None) -> str:
    if cfg.openai_client is None:
        return "LLM not configured. Set OPENAI_API_KEY in .env to enable answers."

    user_msg = f"Source chunks:\n{context}\n\nQuestion: {query}"
    prompt = get_system_prompt(language)

    resp = cfg.openai_client.chat.completions.create(
        model=cfg.RAG_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=2000,
    )

    return resp.choices[0].message.content


# ── Full RAG pipeline ───────────────────────────────────────────────────────

_BROAD_PATTERNS = re.compile(
    r'\b(what is this|tell me about|about the (website|site|company|page)|'
    r'what do you do|what (can|does) (this|the) (site|website|company)|'
    r'who are you|what are you|overview|summary|introduce)\b',
    re.IGNORECASE,
)


def do_rag_sync(site_id: int, query: str, top_k: int,
                session_id: str | None = None,
                language: str | None = None) -> dict:
    """Full RAG pipeline (synchronous) — meant to run in asyncio.to_thread."""
    is_broad = bool(_BROAD_PATTERNS.search(query))

    # Expand short follow-up queries using conversation history
    retrieval_query = query
    if session_id:
        with cfg._session_lock:
            history = cfg._session_history.get(session_id, [])
        if history and len(query.split()) < 6:
            last_user = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"),
                None,
            )
            if last_user:
                retrieval_query = f"{last_user} {query}"

    # 1. Retrieve — fetch more chunks for broad questions
    effective_top_k = min(top_k * 2, 15) if is_broad else top_k
    retrieval = retrieve_chunks(site_id, retrieval_query, effective_top_k)
    context = build_context(retrieval["results"])

    # 2. Build messages with conversation history
    system = get_system_prompt(language)
    messages = [{"role": "system", "content": system}]

    if session_id:
        with cfg._session_lock:
            history = cfg._session_history.get(session_id, [])
        if history:
            messages.extend(history[-cfg.SESSION_HISTORY_LIMIT * 2:])

    user_msg = f"Source chunks:\n{context}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_msg})

    # 3. Generate answer
    if cfg.openai_client is None:
        answer = "LLM not configured. Set OPENAI_API_KEY in .env to enable answers."
    else:
        resp = cfg.openai_client.chat.completions.create(
            model=cfg.RAG_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=2000,
        )
        answer = resp.choices[0].message.content

    # 4. Update session history
    if session_id:
        with cfg._session_lock:
            if session_id not in cfg._session_history:
                cfg._session_history[session_id] = []
            cfg._session_history[session_id].append({"role": "user", "content": query})
            cfg._session_history[session_id].append({"role": "assistant", "content": answer})
            if len(cfg._session_history[session_id]) > cfg.SESSION_HISTORY_LIMIT * 2:
                cfg._session_history[session_id] = cfg._session_history[session_id][-(cfg.SESSION_HISTORY_LIMIT * 2):]
            cfg._session_last_access[session_id] = time.time()

    sources = [
        {"title": r["title"], "url": r["url"], "score": r["score"]}
        for r in retrieval["results"]
    ]
    return {"answer": answer, "sources": sources, "confidence": retrieval["confidence"]}
