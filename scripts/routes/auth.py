"""
Registration and activation endpoints.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse

import scripts.config as cfg
from scripts.utils import rate_limit_check, is_url_safe, verify_user
from scripts.routes.trial import run_trial_indexing

router = APIRouter()


def _widget_snippet(site_id: int) -> str:
    public_url = os.environ.get("PUBLIC_URL", "https://wrs.kz").rstrip("/")
    return f'<script src="{public_url}/widget.js" data-site-id="{site_id}" data-api="{public_url}"></script>'


@router.post("/api/register")
async def register(request: Request):
    blocked = rate_limit_check(request, "register", 5, 3600)
    if blocked:
        return blocked

    body = await request.json()
    name = body.get("name", "").strip()
    email = body.get("email", "").strip().lower()
    company = body.get("company", "").strip()

    if not name or not email:
        return JSONResponse({"error": "Name and email are required"}, status_code=400)

    existing = cfg.sb.table("registrations").select("token").eq("email", email).execute()
    if existing.data:
        token = existing.data[0]["token"]
        cfg.log.info(f"Returning existing registration for {email}")
        return {"token": token, "message": "Welcome back"}

    token = f"wr_{uuid.uuid4().hex}"
    cfg.sb.table("registrations").insert({
        "name": name, "email": email, "company": company or None, "token": token,
    }).execute()

    cfg.log.info(f"New registration: {email} ({name})")
    return {"token": token, "message": "Registered successfully"}


@router.post("/api/activate")
async def activate(request: Request):
    blocked = rate_limit_check(request, "activate", 10, 3600)
    if blocked:
        return blocked

    body = await request.json()
    code = body.get("code", "").strip()
    site_id = body.get("site_id")
    token = body.get("token", "").strip()
    max_pages = min(int(body.get("max_pages", 100)), 300)

    current_user = verify_user(request, require_active=True)
    if not current_user:
        return JSONResponse(
            {"error": "Please sign in or create an account in the customer portal before activating an assistant."},
            status_code=401,
        )
    if not code or not site_id or (not token and not current_user):
        return JSONResponse({"error": "Code, site_id, and token are required"}, status_code=400)

    reg_email = current_user["email"] if current_user else None
    reg = cfg.sb.table("registrations").select("email").eq("token", token).execute() if token else None
    if token and (not reg or not reg.data):
        return JSONResponse({"error": "Invalid session"}, status_code=401)
    if reg and reg.data:
        reg_email = reg.data[0]["email"]

    if not cfg.ACTIVATION_CODE:
        return JSONResponse({"error": "Activation not configured"}, status_code=503)

    if code != cfg.ACTIVATION_CODE:
        return JSONResponse({"error": "Invalid activation code"}, status_code=403)

    site = cfg.sb.table("sites").select("*").eq("id", site_id).execute()
    if not site.data:
        return JSONResponse({"error": "Site not found"}, status_code=404)

    site_data = site.data[0]

    # Only trial sites can be activated — prevents re-activation.
    if not site_data.get("is_trial"):
        return JSONResponse({"error": "Site is already activated"}, status_code=400)

    # Verify the caller owns this trial site.  The registration token
    # is stored in settings.owner_token when trial_start creates the
    # site.  If it was set, the activating token must match.
    site_settings = site_data.get("settings") or {}
    owner_token = site_settings.get("owner_token")
    owner_user_id = site_data.get("owner_user_id")
    if current_user and owner_user_id and owner_user_id != current_user["user_id"] and current_user.get("role") != "admin":
        cfg.log.warning(f"Activate denied: user mismatch for site {site_id}")
        return JSONResponse({"error": "You are not the owner of this trial"}, status_code=403)
    if owner_token and not token and not current_user:
        return JSONResponse({"error": "You are not the owner of this trial"}, status_code=403)
    if owner_token and token and owner_token != token:
        cfg.log.warning(f"Activate denied: token mismatch for site {site_id}")
        return JSONResponse({"error": "You are not the owner of this trial"}, status_code=403)

    source_url = site_settings.get("source_url", "")

    update_payload = {"is_trial": False, "expires_at": None}
    if current_user:
        update_payload["owner_user_id"] = current_user["user_id"]
    cfg.sb.table("sites").update(update_payload).eq("id", site_id).execute()

    if source_url and max_pages > 0:
        cfg.trial_progress[site_id] = {
            "step": 0, "total": 0, "message": "Re-indexing with full page count...",
            "done": False, "error": None,
        }

        def reindex():
            docs = cfg.sb.table("documents").select("id").eq("site_id", site_id).execute()
            for doc in (docs.data or []):
                cfg.sb.table("chunks").delete().eq("document_id", doc["id"]).execute()
            cfg.sb.table("documents").delete().eq("site_id", site_id).execute()
            run_trial_indexing(site_id, source_url, max_pages, [])

        asyncio.get_event_loop().create_task(asyncio.to_thread(reindex))

    cfg.log.info(f"Site {site_id} activated by {reg_email or 'unknown'} — re-indexing {max_pages} pages")

    return {
        "success": True,
        "site_id": site_id,
        "widget_code": _widget_snippet(site_id),
        "source_url": source_url,
    }


@router.post("/api/quick-activate")
async def quick_activate(
    request: Request,
    url: str = Form(...),
    code: str = Form(...),
    max_pages: int = Form(default=100),
    language: str = Form(default=""),
    use_playwright: str = Form(default="0"),
    pdfs: list[UploadFile] = File(default=[]),
):
    """Activate directly from landing page — no trial/registration needed."""
    blocked = rate_limit_check(request, "quick_activate", 3, 3600)
    if blocked:
        return blocked

    current_user = verify_user(request, require_active=True)
    if not current_user:
        return JSONResponse(
            {"error": "Please sign in or create an account in the customer portal before activating an assistant."},
            status_code=401,
        )

    code = code.strip()
    url = url.strip()
    max_pages = max(1, min(max_pages, 300))

    if not code or not url:
        return JSONResponse({"error": "Code and URL are required"}, status_code=400)

    if not cfg.ACTIVATION_CODE:
        return JSONResponse({"error": "Activation not configured"}, status_code=503)

    if code != cfg.ACTIVATION_CODE:
        return JSONResponse({"error": "Invalid activation code"}, status_code=403)

    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)
    if not parsed.netloc:
        return JSONResponse({"error": "Invalid URL"}, status_code=400)

    safe, reason = is_url_safe(url)
    if not safe:
        cfg.log.warning(f"SSRF blocked in quick_activate: {url} — {reason}")
        return JSONResponse({"error": reason}, status_code=400)

    domain = parsed.netloc

    pdf_data = []
    for pdf in pdfs:
        content = await pdf.read()
        if len(content) > cfg.MAX_PDF_SIZE:
            return JSONResponse({"error": f"PDF '{pdf.filename}' exceeds 10MB limit"}, status_code=400)
        pdf_data.append({"filename": pdf.filename, "content": content})

    # Check if site exists
    existing = cfg.sb.table("sites").select("id, settings").eq("domain", domain).execute()
    if existing.data:
        site_id = existing.data[0]["id"]
        existing_settings = existing.data[0].get("settings") or {}
        existing_site = cfg.sb.table("sites").select("owner_user_id").eq("id", site_id).limit(1).execute()
        existing_owner = existing_site.data[0].get("owner_user_id") if existing_site.data else None
        if current_user and existing_owner and existing_owner != current_user["user_id"] and current_user.get("role") != "admin":
            return JSONResponse({"error": "This domain is already owned by another account"}, status_code=403)
        existing_settings["source_url"] = url
        existing_settings["use_playwright"] = use_playwright == "1"
        update_payload = {
            "is_trial": False,
            "expires_at": None,
            "language": language or "en",
            "settings": existing_settings,
        }
        if current_user:
            update_payload["owner_user_id"] = current_user["user_id"]
        cfg.sb.table("sites").update(update_payload).eq("id", site_id).execute()
        # Clear existing documents/chunks so run_trial_indexing can
        # insert fresh ones without hitting unique(site_id, url).
        old_docs = cfg.sb.table("documents").select("id").eq("site_id", site_id).execute()
        for doc in (old_docs.data or []):
            cfg.sb.table("chunks").delete().eq("document_id", doc["id"]).execute()
        cfg.sb.table("documents").delete().eq("site_id", site_id).execute()
    else:
        site_resp = cfg.sb.table("sites").insert({
            "domain": domain, "language": language or "en", "is_trial": False,
            "owner_user_id": current_user["user_id"] if current_user else None,
            "settings": {"source_url": url, "use_playwright": use_playwright == "1"},
        }).execute()
        site_id = site_resp.data[0]["id"]

    cfg.trial_progress[site_id] = {"step": 0, "total": 0, "message": "Starting...", "done": False, "error": None}

    pw = use_playwright == "1"
    asyncio.get_event_loop().create_task(
        asyncio.to_thread(run_trial_indexing, site_id, url, max_pages, pdf_data, pw)
    )

    public_url = os.environ.get("PUBLIC_URL", "https://wrs.kz")
    cfg.log.info(f"Quick-activate: site_id={site_id} domain={domain} url={url} pdfs={len(pdf_data)}")
    return {
        "success": True,
        "site_id": site_id,
        "widget_code": _widget_snippet(site_id),
    }
