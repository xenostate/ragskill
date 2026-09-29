"""
Customer account endpoints and ownership-scoped dashboard APIs.
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from pypdf import PdfReader

import scripts.config as cfg
from scripts.analytics import (
    load_site_analytics,
    monthly_report_csv,
    utc_month_period,
    utc_period_for_days,
)
from scripts.assistant_features import normalize_assistant_config
from scripts.config_assistant import ConfigAssistantError, draft_assistant_config
from scripts.indexer import chunk_text, content_hash
from scripts.knowledge import index_answer_document
from scripts.routes.trial import schedule_indexing_job
from scripts.utils import (
    rate_limit_check,
    verify_user,
    is_url_safe,
    is_valid_pdf,
    get_client_ip,
)

router = APIRouter()


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _check_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except Exception:
        return False


def _session_expiry() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=cfg.APP_SESSION_TTL_HOURS)).isoformat()


def _create_user_session(user_id: int, request: Request) -> str:
    token = f"usr_{secrets.token_urlsafe(32)}"
    cfg.sb.table("app_sessions").insert({
        "token": token,
        "user_id": user_id,
        "user_agent": request.headers.get("user-agent", "")[:500],
        "ip": get_client_ip(request),
        "expires_at": _session_expiry(),
    }).execute()
    return token


def bootstrap_app_user_ownership() -> None:
    """Ensure a bootstrap admin app user exists and owns legacy data."""
    try:
        admin_user = None

        if cfg.APP_ADMIN_EMAIL:
            resp = (
                cfg.sb.table("app_users")
                .select("id, email, role")
                .eq("email", cfg.APP_ADMIN_EMAIL)
                .limit(1)
                .execute()
            )
            if resp.data:
                admin_user = resp.data[0]
            elif cfg.APP_ADMIN_PASSWORD:
                created = cfg.sb.table("app_users").insert({
                    "email": cfg.APP_ADMIN_EMAIL,
                    "password_hash": _hash_password(cfg.APP_ADMIN_PASSWORD),
                    "name": cfg.APP_ADMIN_NAME,
                    "role": "admin",
                    "status": "active",
                }).execute()
                admin_user = created.data[0]
                cfg.log.info(f"Bootstrap app admin created for {cfg.APP_ADMIN_EMAIL}")

        if admin_user is None:
            resp = (
                cfg.sb.table("app_users")
                .select("id, email, role")
                .eq("role", "admin")
                .limit(1)
                .execute()
            )
            if resp.data:
                admin_user = resp.data[0]

        if admin_user is None:
            cfg.log.warning("No app admin user available for ownership backfill.")
            return

        legacy_sites = cfg.sb.table("sites").select("id").is_("owner_user_id", "null").execute()
        site_ids = [row["id"] for row in (legacy_sites.data or [])]
        if site_ids:
            cfg.sb.table("sites").update({"owner_user_id": admin_user["id"]}).in_("id", site_ids).execute()
            cfg.log.info(f"Backfilled {len(site_ids)} site(s) to app admin owner {admin_user['id']}")

        legacy_assistants = (
            cfg.sb.table("internal_assistants")
            .select("id")
            .is_("owner_user_id", "null")
            .execute()
        )
        assistant_ids = [row["id"] for row in (legacy_assistants.data or [])]
        if assistant_ids:
            cfg.sb.table("internal_assistants").update({"owner_user_id": admin_user["id"]}).in_("id", assistant_ids).execute()
            cfg.log.info(f"Backfilled {len(assistant_ids)} internal assistant(s) to app admin owner {admin_user['id']}")
    except Exception as e:
        cfg.log.warning(f"App user bootstrap skipped: {e}")


def _user_can_access_site(user: dict, site_id: int) -> dict | None:
    query = cfg.sb.table("sites").select("id, domain, language, settings, is_trial, owner_user_id").eq("id", site_id)
    if user.get("role") != "admin":
        query = query.eq("owner_user_id", user["user_id"])
    resp = query.limit(1).execute()
    return resp.data[0] if resp.data else None


def _site_documents(site_id: int) -> list[dict]:
    docs = cfg.sb.table("documents").select("id, title, url, content_hash, last_crawled").eq("site_id", site_id).execute()
    if not docs.data:
        return []
    doc_ids = [doc["id"] for doc in docs.data]
    all_chunks = cfg.sb.table("chunks").select("id, document_id, chunk_index, text").in_("document_id", doc_ids).execute()
    chunks_by_doc = {}
    for c in (all_chunks.data or []):
        chunks_by_doc.setdefault(c["document_id"], []).append({
            "id": c["id"],
            "chunk_index": c["chunk_index"],
            "preview": c["text"][:200],
        })
    result = []
    for doc in docs.data:
        doc_chunks = chunks_by_doc.get(doc["id"], [])
        doc_chunks.sort(key=lambda x: x["chunk_index"])
        result.append({
            "id": doc["id"],
            "title": doc["title"],
            "url": doc["url"],
            "last_crawled": doc.get("last_crawled"),
            "chunk_count": len(doc_chunks),
            "chunks": doc_chunks,
        })
    return result


@router.get("/app")
async def serve_user_page():
    html_path = cfg.WIDGET_DIR / "user.html"
    if not html_path.exists():
        return JSONResponse({"error": "user.html not found"}, status_code=404)
    return FileResponse(html_path, media_type="text/html")


@router.post("/api/user/signup")
async def user_signup(request: Request):
    blocked = rate_limit_check(request, "user_signup", 10, 3600)
    if blocked:
        return blocked

    body = await request.json()
    name = body.get("name", "").strip()
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    if not name or not email or not password:
        return JSONResponse({"error": "Name, email, and password are required"}, status_code=400)
    if "@" not in email:
        return JSONResponse({"error": "Please enter a valid email"}, status_code=400)
    if len(password) < 6:
        return JSONResponse({"error": "Password must be at least 6 characters"}, status_code=400)

    existing = cfg.sb.table("app_users").select("id").eq("email", email).limit(1).execute()
    if existing.data:
        return JSONResponse({"error": "An account with this email already exists"}, status_code=409)

    created = cfg.sb.table("app_users").insert({
        "email": email,
        "password_hash": _hash_password(password),
        "name": name,
        "role": "client",
        "status": "active",
    }).execute()
    user = created.data[0]
    token = _create_user_session(user["id"], request)
    cfg.log.info(f"App user signup: {email}")
    return {
        "success": True,
        "token": token,
        "user": {
            "id": user["id"],
            "email": email,
            "name": name,
            "role": "client",
            "status": "active",
        },
    }


@router.post("/api/user/login")
async def user_login(request: Request):
    blocked = rate_limit_check(request, "user_login", 15, 3600)
    if blocked:
        return blocked

    body = await request.json()
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")
    if not email or not password:
        return JSONResponse({"error": "Email and password are required"}, status_code=400)

    resp = (
        cfg.sb.table("app_users")
        .select("id, email, name, role, status, password_hash")
        .eq("email", email)
        .limit(1)
        .execute()
    )
    if not resp.data:
        return JSONResponse({"error": "Invalid email or password"}, status_code=401)

    user = resp.data[0]
    if not _check_password(password, user.get("password_hash", "")):
        return JSONResponse({"error": "Invalid email or password"}, status_code=401)
    if user.get("status") != "active":
        return JSONResponse({"error": "This account is not active"}, status_code=403)

    cfg.sb.table("app_users").update({
        "last_login_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", user["id"]).execute()
    token = _create_user_session(user["id"], request)
    return {
        "success": True,
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
            "status": user["status"],
        },
    }


@router.post("/api/user/logout")
async def user_logout(request: Request):
    user = verify_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    cfg.sb.table("app_sessions").delete().eq("token", user["token"]).execute()
    return {"success": True}


@router.get("/api/user/me")
async def user_me(request: Request):
    user = verify_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    sites_count_query = cfg.sb.table("sites").select("id", count="exact")
    assistants_count_query = cfg.sb.table("internal_assistants").select("id", count="exact")
    if user.get("role") != "admin":
        sites_count_query = sites_count_query.eq("owner_user_id", user["user_id"])
        assistants_count_query = assistants_count_query.eq("owner_user_id", user["user_id"])

    sites_count = sites_count_query.execute().count or 0
    assistants_count = assistants_count_query.execute().count or 0

    return {
        "user": {
            "id": user["user_id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
            "status": user["status"],
        },
        "stats": {
            "sites": sites_count,
            "assistants": assistants_count,
        },
    }


@router.get("/api/user/sites")
async def user_list_sites(request: Request):
    user = verify_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    query = cfg.sb.table("sites").select("id, domain, language, is_trial, settings, owner_user_id")
    if user.get("role") != "admin":
        query = query.eq("owner_user_id", user["user_id"])
    sites = query.execute()

    result = []
    for s in (sites.data or []):
        settings = s.get("settings") or {}
        doc_count = cfg.sb.table("documents").select("id", count="exact").eq("site_id", s["id"]).execute()
        query_count = cfg.sb.table("chat_logs").select("id", count="exact").eq("site_id", s["id"]).execute()
        lead_count = cfg.sb.table("assistant_form_submissions").select("id", count="exact").eq("site_id", s["id"]).execute()
        result.append({
            "id": s["id"],
            "domain": s["domain"],
            "language": s.get("language"),
            "is_trial": s.get("is_trial", False),
            "is_landing": settings.get("landing", False),
            "doc_count": doc_count.count if doc_count.count is not None else 0,
            "query_count": query_count.count if query_count.count is not None else 0,
            "lead_count": lead_count.count if lead_count.count is not None else 0,
        })
    return {"sites": result}


@router.get("/api/user/sites/{site_id}/documents")
async def user_site_documents(site_id: int, request: Request):
    user = verify_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    site = _user_can_access_site(user, site_id)
    if not site:
        return JSONResponse({"error": "Site not found"}, status_code=404)
    return {"site": site, "documents": _site_documents(site_id)}


@router.get("/api/user/sites/{site_id}/analytics")
async def user_site_analytics(site_id: int, request: Request):
    """Customer-facing pilot report for a rolling period (14 days by default)."""
    user = verify_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    site = _user_can_access_site(user, site_id)
    if not site:
        return JSONResponse({"error": "Site not found"}, status_code=404)
    try:
        days = int(request.query_params.get("days", "14"))
    except ValueError:
        return JSONResponse({"error": "days must be a number"}, status_code=400)
    start, end = utc_period_for_days(days)
    try:
        report, _ = await asyncio.to_thread(load_site_analytics, cfg.sb, site_id, start, end)
    except Exception:
        cfg.log.exception(f"Could not load analytics for site {site_id}")
        return JSONResponse({"error": "Analytics unavailable. Apply the latest schema migration."}, status_code=500)
    report.pop("_feedback_rows", None)
    report["site"] = {"id": site_id, "domain": site.get("domain")}
    return report


@router.get("/api/user/sites/{site_id}/analytics/export")
async def user_site_analytics_export(site_id: int, request: Request):
    """Download a monthly CSV with metrics and question/answer detail."""
    user = verify_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    site = _user_can_access_site(user, site_id)
    if not site:
        return JSONResponse({"error": "Site not found"}, status_code=404)
    try:
        start, end, month = utc_month_period(request.query_params.get("month", ""))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    try:
        report, rows = await asyncio.to_thread(load_site_analytics, cfg.sb, site_id, start, end)
        csv_content = monthly_report_csv(site, month, report, rows)
    except Exception:
        cfg.log.exception(f"Could not export analytics for site {site_id}")
        return JSONResponse({"error": "Report unavailable. Apply the latest schema migration."}, status_code=500)
    safe_domain = re.sub(r"[^a-zA-Z0-9.-]", "_", site.get("domain") or f"site-{site_id}")
    return Response(
        csv_content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="wrs-{safe_domain}-{month}.csv"'},
    )


@router.post("/api/user/sites/{site_id}/answers")
async def user_add_site_answer(site_id: int, request: Request):
    """Turn a failed/low-confidence question into indexed customer knowledge."""
    user = verify_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    site = _user_can_access_site(user, site_id)
    if not site:
        return JSONResponse({"error": "Site not found"}, status_code=404)
    body = await request.json()
    interaction_id = str(body.get("interaction_id") or "").strip()[:128]
    chat_log_id = body.get("chat_log_id")
    try:
        result = await asyncio.to_thread(
            index_answer_document,
            site_id,
            body.get("question", ""),
            body.get("answer", ""),
            body.get("title"),
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        cfg.log.exception(f"Could not add answer for site {site_id}")
        return JSONResponse({"error": "Could not index the answer"}, status_code=500)
    if interaction_id:
        cfg.sb.table("chat_logs").update({
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "resolved_document_id": result["document_id"],
        }).eq("site_id", site_id).eq("interaction_id", interaction_id).execute()
    elif isinstance(chat_log_id, int):
        cfg.sb.table("chat_logs").update({
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "resolved_document_id": result["document_id"],
        }).eq("site_id", site_id).eq("id", chat_log_id).execute()
    return {"success": True, **result}


@router.post("/api/user/sites/{site_id}/assistant-config")
async def user_update_assistant_config(site_id: int, request: Request):
    user = verify_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    site = _user_can_access_site(user, site_id)
    if not site:
        return JSONResponse({"error": "Site not found"}, status_code=404)

    body = await request.json()
    assistant = body.get("assistant")
    if not isinstance(assistant, dict):
        return JSONResponse({"error": "assistant config must be a JSON object"}, status_code=400)

    settings = site.get("settings") or {}
    settings["assistant"] = normalize_assistant_config(assistant)
    cfg.sb.table("sites").update({"settings": settings}).eq("id", site_id).execute()
    return {"success": True, "assistant": settings["assistant"]}


@router.post("/api/user/sites/{site_id}/assistant-config/draft")
async def user_draft_assistant_config(site_id: int, request: Request):
    """Create an LLM-edited configuration draft without publishing it."""
    user = verify_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    site = _user_can_access_site(user, site_id)
    if not site:
        return JSONResponse({"error": "Site not found"}, status_code=404)
    blocked = rate_limit_check(request, f"assistant_config_draft:{site_id}", 12, 600)
    if blocked:
        return blocked
    body = await request.json()
    instruction = str(body.get("instruction") or "").strip()
    current_config = body.get("current_config")
    response_language = str(body.get("response_language") or "").strip().lower()
    if current_config is not None and not isinstance(current_config, dict):
        return JSONResponse({"error": "current_config must be a JSON object"}, status_code=400)
    if cfg.openai_client is None:
        return JSONResponse({"error": "The configuration assistant is not configured"}, status_code=503)
    try:
        result = await asyncio.to_thread(
            draft_assistant_config,
            current_config if current_config is not None else (site.get("settings") or {}).get("assistant"),
            instruction,
            response_language,
        )
    except ConfigAssistantError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"success": True, **result}


@router.delete("/api/user/sites/{site_id}")
async def user_delete_site(site_id: int, request: Request):
    user = verify_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    site = _user_can_access_site(user, site_id)
    if not site:
        return JSONResponse({"error": "Site not found"}, status_code=404)

    docs = cfg.sb.table("documents").select("id").eq("site_id", site_id).execute()
    for doc in (docs.data or []):
        cfg.sb.table("chunks").delete().eq("document_id", doc["id"]).execute()
    cfg.sb.table("documents").delete().eq("site_id", site_id).execute()
    cfg.sb.table("sites").delete().eq("id", site_id).execute()
    cfg.trial_progress.pop(site_id, None)
    return {"success": True}


@router.delete("/api/user/sites/{site_id}/chunks/{chunk_id}")
async def user_delete_chunk(site_id: int, chunk_id: int, request: Request):
    user = verify_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    site = _user_can_access_site(user, site_id)
    if not site:
        return JSONResponse({"error": "Site not found"}, status_code=404)

    chunk = cfg.sb.table("chunks").select("id, document_id").eq("id", chunk_id).execute()
    if not chunk.data:
        return JSONResponse({"error": "Chunk not found"}, status_code=404)
    doc_id = chunk.data[0]["document_id"]
    doc = cfg.sb.table("documents").select("site_id").eq("id", doc_id).execute()
    if not doc.data or doc.data[0]["site_id"] != site_id:
        return JSONResponse({"error": "Chunk does not belong to this site"}, status_code=403)
    cfg.sb.table("chunks").delete().eq("id", chunk_id).execute()
    return {"success": True}


@router.post("/api/user/sites/{site_id}/upload-pdf")
async def user_upload_pdf(
    site_id: int,
    request: Request,
    pdf: UploadFile = File(...),
    interaction_id: str = Form(""),
    chat_log_id: str = Form(""),
):
    user = verify_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    site = _user_can_access_site(user, site_id)
    if not site:
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
        safe_filename = re.sub(r"[^\w\s\-.]", "_", pdf.filename or "upload.pdf")
        c_hash = content_hash(pdf_text.strip())
        ins = cfg.sb.table("documents").insert({
            "site_id": site_id,
            "url": f"pdf://{safe_filename}",
            "title": pdf.filename or "Uploaded PDF",
            "content_hash": c_hash,
        }).execute()
        doc_id = ins.data[0]["id"]
        chunks = chunk_text(pdf_text.strip())
        if not chunks:
            return {"success": True, "doc_id": doc_id, "chunks": 0}
        texts_to_embed = [f"passage: {c}" for c in chunks]
        embeddings = cfg.embed_model.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)
        rows = [{
            "document_id": doc_id,
            "chunk_index": i,
            "text": chunk,
            "headings": [],
            "embedding": emb.tolist(),
        } for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True))]
        cfg.sb.table("chunks").insert(rows).execute()
        return {"success": True, "doc_id": doc_id, "chunks": len(rows), "filename": pdf.filename}

    result = await asyncio.to_thread(do_pdf_index)
    if "error" in result:
        return JSONResponse({"error": result["error"]}, status_code=400)
    interaction_id = interaction_id.strip()[:128]
    if interaction_id:
        cfg.sb.table("chat_logs").update({
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "resolved_document_id": result["doc_id"],
        }).eq("site_id", site_id).eq("interaction_id", interaction_id).execute()
    elif chat_log_id.isdigit():
        cfg.sb.table("chat_logs").update({
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "resolved_document_id": result["doc_id"],
        }).eq("site_id", site_id).eq("id", int(chat_log_id)).execute()
    return result


@router.post("/api/user/sites/{site_id}/recrawl")
async def user_recrawl_site(site_id: int, request: Request):
    user = verify_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    site = _user_can_access_site(user, site_id)
    if not site:
        return JSONResponse({"error": "Site not found"}, status_code=404)

    body = await request.json()
    max_pages = min(int(body.get("max_pages", 100)), 300)
    source_url = (site.get("settings") or {}).get("source_url", "")
    if not source_url:
        return JSONResponse({"error": "No source_url stored for this site"}, status_code=400)

    safe, reason = is_url_safe(source_url)
    if not safe:
        return JSONResponse({"error": reason}, status_code=400)

    settings = site.get("settings") or {}
    use_playwright = bool(settings.get("use_playwright", False))
    job_id = schedule_indexing_job(
        site_id,
        source_url,
        max_pages,
        [],
        use_playwright,
        kind="recrawl",
        replace_existing=True,
        message="Customer re-crawl queued",
    )
    return {"success": True, "job_id": job_id, "message": "Re-crawl started"}


@router.get("/api/user/internal-assistants")
async def user_list_internal_assistants(request: Request):
    user = verify_user(request)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    query = cfg.sb.table("internal_assistants").select("id, slug, name, site_id, created_by, settings, created_at, owner_user_id")
    if user.get("role") != "admin":
        query = query.eq("owner_user_id", user["user_id"])
    assistants = query.execute()

    result = []
    for a in (assistants.data or []):
        docs = cfg.sb.table("documents").select("id", count="exact").eq("site_id", a["site_id"]).execute()
        doc_count = docs.count if docs.count is not None else 0
        result.append({
            "id": a["id"],
            "slug": a["slug"],
            "name": a["name"],
            "site_id": a["site_id"],
            "doc_count": doc_count,
            "created_by": a.get("created_by"),
            "settings": a.get("settings", {}),
            "created_at": a.get("created_at"),
            "url": f"{os.environ.get('PUBLIC_URL', 'https://wrs.kz')}/assistant/{a['slug']}",
        })
    return {"assistants": result}
