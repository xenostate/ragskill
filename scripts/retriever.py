#!/usr/bin/env python3
"""
web-rag retriever: embed a query, call Supabase match_chunks, return ranked results.

Usage as CLI:
    python3 retriever.py --site-id 1 --query "how to configure nginx" --top-k 5

Usage as module:
    from retriever import retrieve
    results = retrieve(site_id=1, query="how to configure nginx", top_k=5)
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from sentence_transformers import SentenceTransformer
from supabase import create_client

# ── Config ──────────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")

# Confidence thresholds
THRESHOLD_HIGH = float(os.environ.get("THRESHOLD_HIGH", "0.75"))
THRESHOLD_MEDIUM = float(os.environ.get("THRESHOLD_MEDIUM", "0.5"))

# ── Globals ─────────────────────────────────────────────────────────────────

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


# ── Retrieval ──────────────────────────────────────────────────────────────

def retrieve(site_id: int, query: str, top_k: int = 5, model=None) -> dict:
    """
    Given (site_id, query), return top chunks with sources and confidence.

    Args:
        model: Optional pre-loaded SentenceTransformer. If None, loads on demand.
               Pass a warm model from server.py to avoid reload overhead.

    Returns:
        {
            "confidence": "high" | "medium" | "low",
            "results": [
                {"chunk_text": str, "url": str, "title": str, "score": float}
            ]
        }
    """
    if model is None:
        model = get_model()

    # Embed query (e5 models use "query: " prefix)
    query_embedding = model.encode(
        f"query: {query}", normalize_embeddings=True
    ).tolist()

    # Call Supabase RPC: match_chunks(site_id, query_embedding, query_text, match_count)
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

    # Determine confidence
    if not results:
        confidence = "low"
    elif results[0]["score"] >= THRESHOLD_HIGH:
        confidence = "high"
    elif results[0]["score"] >= THRESHOLD_MEDIUM:
        confidence = "medium"
    else:
        confidence = "low"

    return {"confidence": confidence, "results": results}


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="web-rag retriever")
    parser.add_argument("--site-id", type=int, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    try:
        result = retrieve(args.site_id, args.query, args.top_k)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except KeyError as e:
        print(f"Error: missing env var {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
