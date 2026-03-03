#!/usr/bin/env python3
"""
web-rag RAG answerer: retrieve chunks then generate a cited answer.

Usage:
    python3 rag.py --site-id 1 --query "how to set up SSL certificates"

Outputs JSON:
    {
        "answer": "...",
        "sources": [...],
        "confidence": "high|medium|low"
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from retriever import retrieve

# ── Config ──────────────────────────────────────────────────────────────────

# The LLM call is done by the ZeptoClaw agent itself — this script prepares
# the prompt context. If OPENAI_API_KEY is set, it can also call the LLM
# directly for standalone usage.

TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

# ── Prompt builder ─────────────────────────────────────────────────────────

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
"""


def build_context(retrieval_result: dict) -> str:
    """Format retrieved chunks into numbered context block."""
    lines = []
    for i, r in enumerate(retrieval_result["results"], 1):
        lines.append(f"[{i}] (score: {r['score']}) {r['title']} — {r['url']}")
        lines.append(r["chunk_text"])
        lines.append("")
    return "\n".join(lines)


def build_rag_payload(site_id: int, query: str, top_k: int = TOP_K) -> dict:
    """
    Retrieve chunks and build a complete RAG payload.

    Returns a dict the agent can feed directly to its LLM:
        {
            "system": str,
            "context": str,
            "query": str,
            "sources": [...],
            "confidence": str
        }
    """
    retrieval = retrieve(site_id, query, top_k)
    context = build_context(retrieval)

    return {
        "system": SYSTEM_PROMPT,
        "context": context,
        "query": query,
        "sources": retrieval["results"],
        "confidence": retrieval["confidence"],
    }


# ── Standalone mode (optional, if OPENAI_API_KEY set) ──────────────────────

def answer_standalone(payload: dict) -> dict:
    """Call OpenAI-compatible API to generate the answer."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package needed for standalone mode", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    user_msg = f"""Source chunks:
{payload['context']}

Question: {payload['query']}"""

    resp = client.chat.completions.create(
        model=os.environ.get("RAG_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": payload["system"]},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=2000,
    )

    return {
        "answer": resp.choices[0].message.content,
        "sources": payload["sources"],
        "confidence": payload["confidence"],
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="web-rag RAG answerer")
    parser.add_argument("--site-id", type=int, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--context-only", action="store_true",
        help="Only output the RAG payload (system+context), don't call LLM"
    )
    args = parser.parse_args()

    try:
        payload = build_rag_payload(args.site_id, args.query, args.top_k)

        if args.context_only:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            result = answer_standalone(payload)
            print(json.dumps(result, indent=2, ensure_ascii=False))
    except KeyError as e:
        print(f"Error: missing env var {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
