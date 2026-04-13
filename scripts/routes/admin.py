"""
Admin dashboard endpoints: auth, documents, sites, landing page, PDF upload, recrawl.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import re
import uuid
import time

from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pypdf import PdfReader

import scripts.config as cfg
from scripts.utils import rate_limit_check, get_client_ip, verify_admin, is_url_safe, is_valid_pdf
from scripts.indexer import chunk_text, content_hash
from scripts.routes.trial import run_trial_indexing

router = APIRouter()


# ── Landing page content ────────────────────────────────────────────────────

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


def setup_landing_site():
    """Create (or find existing) landing chat site and index product content."""
    existing = cfg.sb.table("sites").select("id").eq("domain", cfg.LANDING_DOMAIN).execute()
    if existing.data:
        cfg.landing_site_id = existing.data[0]["id"]
        cfg.sb.table("sites").update({"language": None}).eq("id", cfg.landing_site_id).execute()
        cfg.log.info(f"Landing site already exists: site_id={cfg.landing_site_id}")
        return cfg.landing_site_id

    site_resp = cfg.sb.table("sites").insert({
        "domain": cfg.LANDING_DOMAIN,
        "language": None,
        "is_trial": False,
        "settings": {"landing": True},
    }).execute()
    cfg.landing_site_id = site_resp.data[0]["id"]
    cfg.log.info(f"Created landing site: site_id={cfg.landing_site_id}")

    for item in LANDING_CONTENT:
        c_hash = content_hash(item["text"])
        ins = cfg.sb.table("documents").insert({
            "site_id": cfg.landing_site_id,
            "url": f"landing://{cfg.LANDING_DOMAIN}",
            "title": item["title"],
            "content_hash": c_hash,
        }).execute()
        doc_id = ins.data[0]["id"]

        chunks = chunk_text(item["text"])
        if not chunks:
            continue

        texts_to_embed = [f"passage: {c}" for c in chunks]
        embeddings = cfg.embed_model.encode(
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

        cfg.sb.table("chunks").insert(rows).execute()

    cfg.log.info(f"Landing site {cfg.landing_site_id} indexed with {len(LANDING_CONTENT)} documents")
    return cfg.landing_site_id


# ── Landing endpoints ───────────────────────────────────────────────────────

@router.get("/api/landing/config")
async def landing_config():
    return {"site_id": cfg.landing_site_id}


@router.post("/api/landing/setup")
async def landing_setup():
    sid = await asyncio.to_thread(setup_landing_site)
    return {"site_id": sid, "message": "Landing site ready"}


# ── Admin auth ──────────────────────────────────────────────────────────────

@router.get("/admin")
async def serve_admin_page():
    html_path = cfg.WIDGET_DIR / "admin.html"
    if not html_path.exists():
        return JSONResponse({"error": "admin.html not found"}, status_code=404)
    return FileResponse(html_path, media_type="text/html")


@router.post("/api/admin/auth")
async def admin_auth(request: Request):
    blocked = rate_limit_check(request, "admin_auth", 5, 3600)
    if blocked:
        return blocked

    if not cfg.ADMIN_CODE:
        return JSONResponse({"error": "Admin access not configured"}, status_code=503)

    body = await request.json()
    code = body.get("code", "").strip()

    if code != cfg.ADMIN_CODE:
        cfg.log.warning(f"Failed admin login from {get_client_ip(request)}")
        return JSONResponse({"error": "Invalid code"}, status_code=403)

    token = f"adm_{uuid.uuid4().hex}"
    cfg.admin_tokens[token] = time.time() + cfg.ADMIN_TOKEN_TTL
    cfg.save_tokens()
    cfg.log.info(f"Admin login from {get_client_ip(request)}")
    return {"success": True, "token": token}


# ── Landing documents ───────────────────────────────────────────────────────

@router.get("/api/admin/documents")
async def admin_list_documents(request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not cfg.landing_site_id:
        return {"documents": []}

    docs = cfg.sb.table("documents").select("id, title, url").eq("site_id", cfg.landing_site_id).execute()
    if not docs.data:
        return {"documents": []}

    doc_ids = [doc["id"] for doc in docs.data]
    all_chunks = cfg.sb.table("chunks").select("document_id, text").in_("document_id", doc_ids).execute()

    chunks_by_doc = {}
    for c in (all_chunks.data or []):
        chunks_by_doc.setdefault(c["document_id"], []).append(c["text"])

    result = []
    for doc in docs.data:
        chunk_texts = chunks_by_doc.get(doc["id"], [])
        preview = chunk_texts[0][:300] if chunk_texts else ""
        result.append({"id": doc["id"], "title": doc["title"], "chunk_count": len(chunk_texts), "preview": preview})

    return {"documents": result}


@router.post("/api/admin/documents")
async def admin_add_document(request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not cfg.landing_site_id:
        return JSONResponse({"error": "Landing site not initialized"}, status_code=500)

    body = await request.json()
    title = body.get("title", "").strip()
    text = body.get("text", "").strip()

    if not title or not text:
        return JSONResponse({"error": "Title and text are required"}, status_code=400)

    def do_index():
        c_hash = content_hash(text)
        doc_uid = uuid.uuid4().hex[:8]
        ins = cfg.sb.table("documents").insert({
            "site_id": cfg.landing_site_id,
            "url": f"admin://{cfg.LANDING_DOMAIN}/{doc_uid}",
            "title": title,
            "content_hash": c_hash,
        }).execute()
        doc_id = ins.data[0]["id"]

        chunks = chunk_text(text)
        if not chunks:
            return 0

        texts_to_embed = [f"passage: {c}" for c in chunks]
        embeddings = cfg.embed_model.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)

        rows = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            rows.append({"document_id": doc_id, "chunk_index": i, "text": chunk, "headings": [], "embedding": emb.tolist()})

        cfg.sb.table("chunks").insert(rows).execute()
        return len(rows)

    chunk_count = await asyncio.to_thread(do_index)
    cfg.log.info(f"Admin added document '{title}' to landing site: {chunk_count} chunks")
    return {"success": True, "chunks": chunk_count}


@router.delete("/api/admin/documents/{doc_id}")
async def admin_delete_document(doc_id: int, request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    doc = cfg.sb.table("documents").select("site_id").eq("id", doc_id).execute()
    if not doc.data or doc.data[0]["site_id"] != cfg.landing_site_id:
        return JSONResponse({"error": "Document not found"}, status_code=404)
    cfg.sb.table("chunks").delete().eq("document_id", doc_id).execute()
    cfg.sb.table("documents").delete().eq("id", doc_id).execute()
    cfg.log.info(f"Admin deleted document {doc_id} from landing site")
    return {"success": True}


# ── Site management ─────────────────────────────────────────────────────────

@router.get("/api/admin/sites")
async def admin_list_sites(request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    sites = cfg.sb.table("sites").select("id, domain, language, is_trial, settings").execute()
    result = []
    for s in (sites.data or []):
        settings = s.get("settings") or {}
        doc_count = cfg.sb.table("documents").select("id", count="exact").eq("site_id", s["id"]).execute()
        result.append({
            "id": s["id"], "domain": s["domain"], "language": s.get("language"),
            "is_trial": s.get("is_trial", False), "is_landing": settings.get("landing", False),
            "doc_count": doc_count.count if doc_count.count is not None else 0,
        })
    return {"sites": result}


@router.delete("/api/admin/sites/{site_id}")
async def admin_delete_site(site_id: int, request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if site_id == cfg.landing_site_id:
        return JSONResponse({"error": "Cannot delete the landing site"}, status_code=400)
    site = cfg.sb.table("sites").select("id, domain").eq("id", site_id).execute()
    if not site.data:
        return JSONResponse({"error": "Site not found"}, status_code=404)
    domain = site.data[0]["domain"]
    docs = cfg.sb.table("documents").select("id").eq("site_id", site_id).execute()
    for doc in (docs.data or []):
        cfg.sb.table("chunks").delete().eq("document_id", doc["id"]).execute()
    cfg.sb.table("documents").delete().eq("site_id", site_id).execute()
    cfg.sb.table("sites").delete().eq("id", site_id).execute()
    cfg.trial_progress.pop(site_id, None)
    cfg.log.info(f"Admin deleted site {site_id} ({domain})")
    return {"success": True}


@router.get("/api/admin/sites/{site_id}/documents")
async def admin_site_documents(site_id: int, request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    site = cfg.sb.table("sites").select("id, domain, language, settings").eq("id", site_id).execute()
    if not site.data:
        return JSONResponse({"error": "Site not found"}, status_code=404)
    docs = cfg.sb.table("documents").select("id, title, url, content_hash, last_crawled").eq("site_id", site_id).execute()
    if not docs.data:
        return {"site": site.data[0], "documents": []}
    doc_ids = [doc["id"] for doc in docs.data]
    all_chunks = cfg.sb.table("chunks").select("id, document_id, chunk_index, text").in_("document_id", doc_ids).execute()
    chunks_by_doc = {}
    for c in (all_chunks.data or []):
        chunks_by_doc.setdefault(c["document_id"], []).append({"id": c["id"], "chunk_index": c["chunk_index"], "preview": c["text"][:200]})
    result = []
    for doc in docs.data:
        doc_chunks = chunks_by_doc.get(doc["id"], [])
        doc_chunks.sort(key=lambda x: x["chunk_index"])
        result.append({"id": doc["id"], "title": doc["title"], "url": doc["url"], "last_crawled": doc.get("last_crawled"), "chunk_count": len(doc_chunks), "chunks": doc_chunks})
    return {"site": site.data[0], "documents": result}


@router.delete("/api/admin/sites/{site_id}/chunks/{chunk_id}")
async def admin_delete_chunk(site_id: int, chunk_id: int, request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    chunk = cfg.sb.table("chunks").select("id, document_id").eq("id", chunk_id).execute()
    if not chunk.data:
        return JSONResponse({"error": "Chunk not found"}, status_code=404)
    doc_id = chunk.data[0]["document_id"]
    doc = cfg.sb.table("documents").select("site_id").eq("id", doc_id).execute()
    if not doc.data or doc.data[0]["site_id"] != site_id:
        return JSONResponse({"error": "Chunk does not belong to this site"}, status_code=403)
    cfg.sb.table("chunks").delete().eq("id", chunk_id).execute()
    cfg.log.info(f"Admin deleted chunk {chunk_id} from site {site_id}")
    return {"success": True}


@router.post("/api/admin/sites/{site_id}/upload-pdf")
async def admin_upload_pdf(site_id: int, request: Request, pdf: UploadFile = File(...)):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    site = cfg.sb.table("sites").select("id, domain").eq("id", site_id).execute()
    if not site.data:
        return JSONResponse({"error": "Site not found"}, status_code=404)

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
            return {"error": "No text could be extracted from this PDF (possibly scanned/image PDF)"}
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
    cfg.log.info(f"Admin uploaded PDF '{pdf.filename}' to site {site_id}: {result['chunks']} chunks")
    return result


@router.post("/api/admin/sites/{site_id}/recrawl")
async def admin_recrawl_site(site_id: int, request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    max_pages = min(int(body.get("max_pages", 100)), 300)
    use_playwright = body.get("use_playwright", False)

    site = cfg.sb.table("sites").select("id, domain, settings").eq("id", site_id).execute()
    if not site.data:
        return JSONResponse({"error": "Site not found"}, status_code=404)

    settings = site.data[0].get("settings") or {}
    source_url = body.get("url") or settings.get("source_url") or f"https://{site.data[0]['domain']}"

    safe, reason = is_url_safe(source_url)
    if not safe:
        cfg.log.warning(f"SSRF blocked in admin_recrawl: {source_url} — {reason}")
        return JSONResponse({"error": reason}, status_code=400)

    if body.get("url"):
        settings["source_url"] = body["url"]
        cfg.sb.table("sites").update({"settings": settings}).eq("id", site_id).execute()

    cfg.trial_progress[site_id] = {"step": 0, "total": 0, "message": "Starting re-crawl...", "done": False, "error": None}

    def do_recrawl():
        docs = cfg.sb.table("documents").select("id").eq("site_id", site_id).execute()
        for doc in (docs.data or []):
            cfg.sb.table("chunks").delete().eq("document_id", doc["id"]).execute()
        cfg.sb.table("documents").delete().eq("site_id", site_id).execute()
        run_trial_indexing(site_id, source_url, max_pages, [], use_playwright)

    asyncio.get_event_loop().create_task(asyncio.to_thread(do_recrawl))

    cfg.log.info(f"Admin triggered re-crawl for site {site_id}: url={source_url} max_pages={max_pages}")
    return {"success": True, "site_id": site_id, "source_url": source_url, "message": "Re-crawl started"}


# ── Admin: Internal Assistants management ────────────────────────────────────

@router.get("/api/admin/internal-assistants")
async def admin_list_internal_assistants(request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    assistants = cfg.sb.table("internal_assistants").select("id, slug, name, site_id, created_by, settings, created_at").execute()
    result = []
    for a in (assistants.data or []):
        docs = cfg.sb.table("documents").select("id", count="exact").eq("site_id", a["site_id"]).execute()
        chunks_count = 0
        if docs.data:
            doc_ids = [d["id"] for d in docs.data]
            if doc_ids:
                chunks_resp = cfg.sb.table("chunks").select("id", count="exact").in_("document_id", doc_ids).execute()
                chunks_count = len(chunks_resp.data) if chunks_resp.data else 0
        result.append({
            "id": a["id"], "slug": a["slug"], "name": a["name"], "site_id": a["site_id"],
            "created_by": a.get("created_by"), "settings": a.get("settings", {}),
            "created_at": a["created_at"], "doc_count": len(docs.data) if docs.data else 0,
            "chunk_count": chunks_count,
        })
    return {"assistants": result}


@router.delete("/api/admin/internal-assistants/{assistant_id}")
async def admin_delete_internal_assistant(assistant_id: int, request: Request):
    if not verify_admin(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    assistant = cfg.sb.table("internal_assistants").select("id, slug, site_id").eq("id", assistant_id).execute()
    if not assistant.data:
        return JSONResponse({"error": "Assistant not found"}, status_code=404)
    a = assistant.data[0]
    docs = cfg.sb.table("documents").select("id").eq("site_id", a["site_id"]).execute()
    for doc in (docs.data or []):
        cfg.sb.table("chunks").delete().eq("document_id", doc["id"]).execute()
    cfg.sb.table("documents").delete().eq("site_id", a["site_id"]).execute()
    cfg.sb.table("internal_assistants").delete().eq("id", assistant_id).execute()
    cfg.sb.table("sites").delete().eq("id", a["site_id"]).execute()
    to_remove = [t for t, s in cfg._internal_tokens.items() if s["slug"] == a["slug"]]
    for t in to_remove:
        cfg._internal_tokens.pop(t, None)
    cfg.log.info(f"Admin deleted internal assistant {a['slug']} (id={assistant_id})")
    return {"success": True}
