"""
Trial / Demo endpoints and the background indexing function.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from pypdf import PdfReader

import scripts.config as cfg
from scripts.utils import rate_limit_check, is_url_safe, is_valid_pdf, verify_user
from scripts.indexer import (
    clean_html, chunk_text, extract_headings, extract_links,
    content_hash, StaticRenderer, PlaywrightRenderer,
)
from scripts.jobs import (
    complete_indexing_job,
    create_indexing_job,
    fail_indexing_job,
    latest_indexing_job,
    progress_payload,
    start_indexing_job,
    update_indexing_job,
)

router = APIRouter()


# ── Background indexing ─────────────────────────────────────────────────────

def run_trial_indexing(site_id: int, url: str, max_pages: int,
                       pdf_data: list[dict], use_playwright: bool = False,
                       job_id: str | None = None):
    """Synchronous trial indexing — runs in asyncio.to_thread."""
    if job_id is None:
        job_id = create_indexing_job(
            site_id,
            "crawl",
            url=url,
            max_pages=max_pages,
            use_playwright=use_playwright,
            pdf_count=len(pdf_data),
        )
    start_indexing_job(job_id, site_id)

    try:
        all_docs = []

        # ── Phase 1: Crawl the URL ──
        update_indexing_job(job_id, site_id, message="Crawling website", status="running")
        renderer = PlaywrightRenderer() if use_playwright else StaticRenderer()

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
                    for link in extract_links(html, page_url, allowed_domain):
                        if link not in visited:
                            queue.append(link)
                    continue

                all_docs.append((page_url, title or page_url, text, html))
                pages_crawled += 1
                update_indexing_job(
                    job_id,
                    site_id,
                    message=f"Crawled {pages_crawled} page(s)",
                    step=pages_crawled,
                    status="running",
                )

                if pages_crawled < max_pages:
                    for link in extract_links(html, page_url, allowed_domain):
                        if link not in visited:
                            queue.append(link)

                time.sleep(0.3)
        finally:
            renderer.close()

        # ── Phase 2: Extract text from PDFs ──
        for pdf_item in pdf_data:
            update_indexing_job(
                job_id,
                site_id,
                message=f"Processing PDF: {pdf_item['filename'][:120]}",
                status="running",
            )
            try:
                if not is_valid_pdf(pdf_item["content"]):
                    cfg.log.warning(f"Skipping invalid PDF (bad magic bytes): {pdf_item['filename']}")
                    continue
                reader = PdfReader(io.BytesIO(pdf_item["content"]))
                pdf_text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pdf_text += page_text + "\n\n"

                if pdf_text.strip():
                    safe_filename = re.sub(r'[^\w\s\-.]', '_', pdf_item['filename'])
                    all_docs.append((
                        f"pdf://{safe_filename}",
                        pdf_item["filename"],
                        pdf_text.strip(),
                        None,
                    ))
                else:
                    cfg.log.warning(f"PDF {pdf_item['filename']} yielded no text")
            except Exception as e:
                cfg.log.warning(f"Trial PDF extraction failed for {pdf_item['filename']}: {e}")

        if not all_docs:
            fail_indexing_job(
                job_id,
                site_id,
                RuntimeError("No content could be extracted from the URL or PDFs."),
            )
            return

        # ── Phase 3: Chunk, embed, store ──
        total_chunks = 0
        update_indexing_job(job_id, site_id, total=len(all_docs), step=0, status="running")

        for doc_idx, (doc_url, doc_title, doc_text, doc_html) in enumerate(all_docs):
            update_indexing_job(
                job_id,
                site_id,
                step=doc_idx + 1,
                total=len(all_docs),
                message=f"Indexing {doc_idx + 1}/{len(all_docs)}: {doc_title[:80]}",
                status="running",
            )

            c_hash = content_hash(doc_text)

            ins = cfg.sb.table("documents").insert({
                "site_id": site_id,
                "url": doc_url,
                "title": doc_title,
                "content_hash": c_hash,
            }).execute()
            doc_id = ins.data[0]["id"]

            headings = extract_headings(doc_html) if doc_html else []
            chunks = chunk_text(doc_text)

            if not chunks:
                continue

            texts_to_embed = [f"passage: {c}" for c in chunks]
            embeddings = cfg.embed_model.encode(
                texts_to_embed, show_progress_bar=False, normalize_embeddings=True
            )

            rows = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
                rows.append({
                    "document_id": doc_id,
                    "chunk_index": i,
                    "text": chunk,
                    "headings": headings,
                    "embedding": emb.tolist(),
                })

            cfg.sb.table("chunks").insert(rows).execute()
            total_chunks += len(rows)

        result_message = f"Done! Indexed {len(all_docs)} document(s), {total_chunks} chunks."
        complete_indexing_job(
            job_id,
            site_id,
            result_message,
            step=len(all_docs),
            total=len(all_docs),
        )
        cfg.log.info(f"Trial site {site_id}: indexed {len(all_docs)} docs, {total_chunks} chunks")

    except Exception as e:
        fail_indexing_job(job_id, site_id, e)


def schedule_indexing_job(
    site_id: int,
    url: str,
    max_pages: int,
    pdf_data: list[dict],
    use_playwright: bool = False,
    *,
    kind: str = "crawl",
    replace_existing: bool = False,
    message: str = "Queued",
) -> str:
    """Persist job state, then start the existing in-process worker."""
    job_id = create_indexing_job(
        site_id,
        kind,
        url=url,
        max_pages=max_pages,
        use_playwright=use_playwright,
        pdf_count=len(pdf_data),
        message=message,
    )

    def run() -> None:
        try:
            if replace_existing:
                docs = cfg.sb.table("documents").select("id").eq("site_id", site_id).execute()
                for doc in (docs.data or []):
                    cfg.sb.table("chunks").delete().eq("document_id", doc["id"]).execute()
                cfg.sb.table("documents").delete().eq("site_id", site_id).execute()
            run_trial_indexing(site_id, url, max_pages, pdf_data, use_playwright, job_id)
        except Exception as exc:
            fail_indexing_job(job_id, site_id, exc)

    asyncio.get_event_loop().create_task(asyncio.to_thread(run))
    return job_id


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/")
async def root_redirect():
    return RedirectResponse(url="/trial")


@router.get("/trial")
async def serve_trial_page():
    html_path = cfg.WIDGET_DIR / "trial.html"
    if not html_path.exists():
        return JSONResponse({"error": "trial.html not found"}, status_code=404)
    return FileResponse(html_path, media_type="text/html")


@router.get("/designs")
@router.get("/designs/{concept_id}")
async def serve_design_concepts(concept_id: int | None = None):
    """Serve the isolated landing-page concepts without changing the live trial page."""
    if concept_id is not None and concept_id not in {1, 2, 3, 4}:
        return RedirectResponse(url="/designs")
    html_path = cfg.WIDGET_DIR / "designs.html"
    if not html_path.exists():
        return JSONResponse({"error": "designs.html not found"}, status_code=404)
    return FileResponse(html_path, media_type="text/html")


@router.get("/editorial.css")
async def serve_editorial_styles():
    """Serve the shared visual system used by all first-party frontend pages."""
    css_path = cfg.WIDGET_DIR / "editorial.css"
    if not css_path.exists():
        return JSONResponse({"error": "editorial.css not found"}, status_code=404)
    return FileResponse(css_path, media_type="text/css")


@router.post("/api/trial/start")
async def trial_start(
    request: Request,
    url: str = Form(...),
    max_pages: int = Form(default=5),
    language: str = Form(default="en"),
    use_playwright: str = Form(default="0"),
    token: str = Form(default=""),
    pdfs: list[UploadFile] = File(default=[]),
):
    blocked = rate_limit_check(request, "trial_start", 3, 3600)
    if blocked:
        return blocked
    current_user = verify_user(request, require_active=True)

    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)
    if not parsed.netloc:
        return JSONResponse({"error": "Invalid URL"}, status_code=400)

    safe, reason = is_url_safe(url)
    if not safe:
        cfg.log.warning(f"SSRF blocked in trial_start: {url} — {reason}")
        return JSONResponse({"error": reason}, status_code=400)

    max_pages = max(1, min(max_pages, 50))

    domain = f"trial-{uuid.uuid4().hex[:8]}.demo"
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()

    site_settings = {"source_url": url, "trial": True}
    if token.strip():
        site_settings["owner_token"] = token.strip()

    site_resp = cfg.sb.table("sites").insert({
        "domain": domain,
        "language": language,
        "is_trial": True,
        "expires_at": expires_at,
        "owner_user_id": current_user["user_id"] if current_user else None,
        "settings": site_settings,
    }).execute()
    site_id = site_resp.data[0]["id"]

    pdf_data = []
    for pdf in pdfs:
        content = await pdf.read()
        if len(content) > cfg.MAX_PDF_SIZE:
            return JSONResponse(
                {"error": f"PDF '{pdf.filename}' exceeds 10MB limit"},
                status_code=400,
            )
        pdf_data.append({"filename": pdf.filename, "content": content})

    pw = use_playwright == "1"
    job_id = schedule_indexing_job(
        site_id,
        url,
        max_pages,
        pdf_data,
        pw,
        kind="trial",
        message="Trial indexing queued",
    )

    cfg.log.info(f"Trial started: site_id={site_id} url={url} pdfs={len(pdf_data)} max_pages={max_pages} playwright={pw}")
    return {"site_id": site_id, "job_id": job_id, "message": "Indexing started", "expires_at": expires_at}


@router.get("/api/trial/progress/{site_id}")
async def trial_progress_stream(site_id: int, request: Request):
    blocked = rate_limit_check(request, "sse", 10, 60)
    if blocked:
        return blocked

    async def event_generator():
        while True:
            progress = cfg.trial_progress.get(site_id)
            if progress is None:
                try:
                    durable_job = await asyncio.to_thread(latest_indexing_job, site_id)
                    if durable_job:
                        progress = progress_payload(durable_job)
                except Exception:
                    cfg.log.exception(
                        "could not read durable indexing progress",
                        extra={"event": "indexing_job.progress_read_failed", "site_id": site_id},
                    )
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


@router.delete("/api/trial/stop/{site_id}")
async def trial_stop(site_id: int):
    try:
        site = cfg.sb.table("sites").select("is_trial").eq("id", site_id).execute()
        if not site.data or not site.data[0].get("is_trial"):
            return JSONResponse({"error": "Not a trial site"}, status_code=400)
        cfg.sb.table("sites").delete().eq("id", site_id).execute()
        cfg.trial_progress.pop(site_id, None)
        cfg.log.info(f"Trial site {site_id} stopped and deleted by user")
        return {"success": True, "message": "Trial data deleted"}
    except Exception as e:
        cfg.log.error(f"Trial stop error for site {site_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/trial/cleanup")
async def trigger_trial_cleanup():
    now_iso = datetime.now(timezone.utc).isoformat()
    resp = cfg.sb.table("sites") \
        .delete() \
        .eq("is_trial", True) \
        .lt("expires_at", now_iso) \
        .execute()
    deleted = len(resp.data) if resp.data else 0
    return {"deleted": deleted}
