"""Helpers for adding customer-supplied knowledge to a site."""

from __future__ import annotations

import uuid

import scripts.config as cfg
from scripts.indexer import chunk_text, content_hash


def index_answer_document(site_id: int, question: str, answer: str, title: str | None = None) -> dict:
    """Index a concise Q&A document and return its identifiers."""
    clean_question = str(question or "").strip()
    clean_answer = str(answer or "").strip()
    if not clean_question or not clean_answer:
        raise ValueError("Question and answer are required")
    if len(clean_question) > 2000:
        raise ValueError("Question is too long")
    if len(clean_answer) > 20000:
        raise ValueError("Answer is too long")

    document_title = str(title or f"Answer: {clean_question[:80]}").strip()[:200]
    text = f"Question: {clean_question}\n\nAnswer: {clean_answer}"
    doc_uid = uuid.uuid4().hex
    inserted = cfg.sb.table("documents").insert({
        "site_id": site_id,
        "url": f"answer://site-{site_id}/{doc_uid}",
        "title": document_title,
        "content_hash": content_hash(text),
    }).execute()
    doc_id = inserted.data[0]["id"]

    chunks = chunk_text(text)
    if chunks:
        embeddings = cfg.embed_model.encode(
            [f"passage: {chunk}" for chunk in chunks],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        cfg.sb.table("chunks").insert([
            {
                "document_id": doc_id,
                "chunk_index": index,
                "text": chunk,
                "headings": ["Customer-provided answer"],
                "embedding": embedding.tolist(),
            }
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]).execute()

    return {"document_id": doc_id, "title": document_title, "chunks": len(chunks)}
