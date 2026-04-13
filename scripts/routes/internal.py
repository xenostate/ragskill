"""
Internal assistant endpoints: setup, auth, documents, PDF upload, crawl.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import re
import time
import uuid
from urllib.parse import urlparse

import bcrypt
from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pypdf import PdfReader

import scripts.config as cfg
from scripts.utils import rate_limit_check, verify_internal, is_url_safe, is_valid_pdf
from scripts.indexer import chunk_text, content_hash
from scripts.routes.trial import run_trial_indexing

router = APIRouter()


@router.get("/assistant/{slug}")
async def serve_assistant_page(slug: str):
    return FileResponse(cfg.WIDGET_DIR / "assistant.html", media_type="text/html")


@router.post("/api/internal/setup")
async def setup_internal_assistant(request: Request):
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

    if not cfg.INTERNAL_SETUP_CODE:
        return JSONResponse({"error": "Internal assistant setup not configured"}, status_code=503)
    if code != cfg.INTERNAL_SETUP_CODE:
        return JSONResponse({"error": "Invalid setup code"}, status_code=403)

    if not re.match(r'^[a-z0-9][a-z0-9\-]{1,48}[a-z0-9]$', slug):
        return JSONResponse({"error": "Slug must be 3-50 chars, lowercase letters, numbers, and hyphens only"}, status_code=400)

    if admin_password == user_password:
        return JSONResponse({"error": "Admin and user passwords must be different"}, status_code=400)

    if len(admin_password) < 4 or len(user_password) < 4:
        return JSONResponse({"error": "Passwords must be at least 4 characters"}, status_code=400)

    existing = cfg.sb.table("internal_assistants").select("id").eq("slug", slug).execute()
    if existing.data:
        return JSONResponse({"error": "This slug is already taken"}, status_code=409)

    domain = f"internal-{slug}"
    try:
        site_resp = cfg.sb.table("sites").insert({
            "domain": domain, "language": language or "en", "is_trial": False,
            "settings": {"internal_assistant": True, "slug": slug},
        }).execute()
    except Exception as e:
        if "duplicate" in str(e).lower():
            return JSONResponse({"error": "An assistant with a similar name already exists"}, status_code=409)
        raise
    site_id = site_resp.data[0]["id"]

    admin_hash = bcrypt.hashpw(admin_password.encode(), bcrypt.gensalt()).decode()
    user_hash = bcrypt.hashpw(user_password.encode(), bcrypt.gensalt()).decode()

    cfg.sb.table("internal_assistants").insert({
        "slug": slug, "name": name, "site_id": site_id,
        "admin_password": admin_hash, "user_password": user_hash,
        "created_by": created_by or None, "settings": {"language": language or "en"},
    }).execute()

    public_url = os.environ.get("PUBLIC_URL", "https://wrs.kz")
    cfg.log.info(f"Internal assistant created: {name} (slug={slug}, site_id={site_id})")
    return {"success": True, "slug": slug, "site_id": site_id, "url": f"{public_url}/assistant/{slug}"}


@router.post("/api/internal/{slug}/auth")
async def internal_assistant_auth(slug: str, request: Request):
    blocked = rate_limit_check(request, f"internal_auth:{slug}", 10, 300)
    if blocked:
        return blocked

    body = await request.json()
    password = body.get("password", "").strip()
    if not password:
        return JSONResponse({"error": "Password is required"}, status_code=400)

    assistant = cfg.sb.table("internal_assistants").select("*").eq("slug", slug).execute()
    if not assistant.data:
        return JSONResponse({"error": "Assistant not found"}, status_code=404)

    a = assistant.data[0]

    if bcrypt.checkpw(password.encode(), a["admin_password"].encode()):
        role = "admin"
    elif bcrypt.checkpw(password.encode(), a["user_password"].encode()):
        role = "user"
    else:
        return JSONResponse({"error": "Invalid password"}, status_code=401)

    token = hashlib.sha256(f"{slug}:{role}:{uuid.uuid4().hex}".encode()).hexdigest()[:48]

    cfg._internal_tokens[token] = {
        "slug": slug, "role": role, "site_id": a["site_id"],
        "assistant_id": a["id"], "created": time.time(),
    }
    cfg.save_tokens()

    return {
        "success": True, "role": role, "token": token,
        "name": a["name"], "site_id": a["site_id"], "settings": a.get("settings", {}),
    }


@router.get("/api/internal/{slug}/documents")
async def internal_get_documents(slug: str, request: Request):
    session = verify_internal(request, slug, require_admin=True)
    if not session:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    site_id = session["site_id"]
    docs = cfg.sb.table("documents").select("id, title, url, content_hash, last_crawled").eq("site_id", site_id).execute()
    if not docs.data:
        return {"documents": []}

    doc_ids = [doc["id"] for doc in docs.data]
    all_chunks = cfg.sb.table("chunks").select("id, document_id, chunk_index, text").in_("document_id", doc_ids).execute()

    chunks_by_doc = {}
    for c in (all_chunks.data or []):
        chunks_by_doc.setdefault(c["document_id"], []).append({"id": c["id"], "chunk_index": c["chunk_index"], "preview": c["text"][:200]})

    result = []
    for doc in docs.data:
        doc_chunks = chunks_by_doc.get(doc["id"], [])
        doc_chunks.sort(key=lambda x: x["chunk_index"])
        result.append({
            "id": doc["id"], "title": doc["title"], "url": doc["url"],
            "last_crawled": doc.get("last_crawled"), "chunk_count": len(doc_chunks), "chunks": doc_chunks,
        })
    return {"documents": result}


@router.post("/api/internal/{slug}/add-text")
async def internal_add_text(slug: str, request: Request):
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
        ins = cfg.sb.table("documents").insert({
            "site_id": site_id, "url": f"text://{title[:50]}", "title": title, "content_hash": c_hash,
        }).execute()
        doc_id = ins.data[0]["id"]
        chunks = chunk_text(text)
        if not chunks:
            return {"success": True, "doc_id": doc_id, "chunks": 0}
        texts_to_embed = [f"passage: {c}" for c in chunks]
        embeddings = cfg.embed_model.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)
        rows = [{"document_id": doc_id, "chunk_index": i, "text": chunk, "headings": [], "embedding": emb.tolist()} for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
        cfg.sb.table("chunks").insert(rows).execute()
        return {"success": True, "doc_id": doc_id, "chunks": len(rows)}

    result = await asyncio.to_thread(do_index)
    cfg.log.info(f"Internal assistant {slug}: added text '{title}' — {result.get('chunks', 0)} chunks")
    return result


@router.post("/api/internal/{slug}/upload-pdf")
async def internal_upload_pdf(slug: str, request: Request, pdf: UploadFile = File(...)):
    session = verify_internal(request, slug, require_admin=True)
    if not session:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    site_id = session["site_id"]

    content = await pdf.read()
    if len(content) > cfg.MAX_PDF_SIZE:
        return JSONResponse({"error": f"PDF exceeds {cfg.MAX_PDF_SIZE // (1024*1024)}MB limit"}, status_code=400)
    if not is_valid_pdf(content):
        return JSONResponse({"error": "Invalid file: not a PDF"}, status_code=400)

    def do_pdf_index():
        reader = PdfReader(io.BytesIO(content))
        pdf_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text + "\n\n"
        if not pdf_text.strip():
            return {"error": "No text could be extracted from this PDF"}
        safe_filename = re.sub(r'[^\w\s\-.]', '_', pdf.filename or "upload.pdf")
        c_hash = content_hash(pdf_text.strip())
        ins = cfg.sb.table("documents").insert({"site_id": site_id, "url": f"pdf://{safe_filename}", "title": pdf.filename or "Uploaded PDF", "content_hash": c_hash}).execute()
        doc_id = ins.data[0]["id"]
        chunks = chunk_text(pdf_text.strip())
        if not chunks:
            return {"success": True, "doc_id": doc_id, "chunks": 0}
        texts_to_embed = [f"passage: {c}" for c in chunks]
        embeddings = cfg.embed_model.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)
        rows = [{"document_id": doc_id, "chunk_index": i, "text": chunk, "headings": [], "embedding": emb.tolist()} for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
        cfg.sb.table("chunks").insert(rows).execute()
        return {"success": True, "doc_id": doc_id, "chunks": len(rows), "filename": pdf.filename}

    result = await asyncio.to_thread(do_pdf_index)
    if "error" in result:
        return JSONResponse({"error": result["error"]}, status_code=400)
    cfg.log.info(f"Internal assistant {slug}: uploaded PDF '{pdf.filename}' — {result['chunks']} chunks")
    return result


@router.post("/api/internal/{slug}/crawl")
async def internal_crawl(slug: str, request: Request):
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

    safe, reason = is_url_safe(url)
    if not safe:
        cfg.log.warning(f"SSRF blocked in internal_crawl: {url} — {reason}")
        return JSONResponse({"error": reason}, status_code=400)

    site_id = session["site_id"]
    cfg.trial_progress[site_id] = {"step": 0, "total": 0, "message": "Starting crawl...", "done": False, "error": None}

    def do_crawl():
        run_trial_indexing(site_id, url, max_pages, [], False)

    asyncio.get_event_loop().create_task(asyncio.to_thread(do_crawl))

    cfg.log.info(f"Internal assistant {slug}: started crawl url={url} max_pages={max_pages}")
    return {"success": True, "site_id": site_id, "message": "Crawl started"}


@router.delete("/api/internal/{slug}/chunks/{chunk_id}")
async def internal_delete_chunk(slug: str, chunk_id: int, request: Request):
    session = verify_internal(request, slug, require_admin=True)
    if not session:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    site_id = session["site_id"]
    chunk = cfg.sb.table("chunks").select("id, document_id").eq("id", chunk_id).execute()
    if not chunk.data:
        return JSONResponse({"error": "Chunk not found"}, status_code=404)
    doc = cfg.sb.table("documents").select("site_id").eq("id", chunk.data[0]["document_id"]).execute()
    if not doc.data or doc.data[0]["site_id"] != site_id:
        return JSONResponse({"error": "Chunk does not belong to this assistant"}, status_code=403)
    cfg.sb.table("chunks").delete().eq("id", chunk_id).execute()
    cfg.log.info(f"Internal assistant {slug}: deleted chunk {chunk_id}")
    return {"success": True}


@router.delete("/api/internal/{slug}/documents/{doc_id}")
async def internal_delete_document(slug: str, doc_id: int, request: Request):
    session = verify_internal(request, slug, require_admin=True)
    if not session:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    site_id = session["site_id"]
    doc = cfg.sb.table("documents").select("site_id, title").eq("id", doc_id).execute()
    if not doc.data or doc.data[0]["site_id"] != site_id:
        return JSONResponse({"error": "Document not found"}, status_code=404)
    cfg.sb.table("chunks").delete().eq("document_id", doc_id).execute()
    cfg.sb.table("documents").delete().eq("id", doc_id).execute()
    cfg.log.info(f"Internal assistant {slug}: deleted document {doc_id}")
    return {"success": True}
