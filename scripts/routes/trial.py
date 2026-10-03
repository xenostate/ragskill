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
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlsplit, urlunsplit

from bs4 import BeautifulSoup
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response, StreamingResponse
from pypdf import PdfReader

import scripts.config as cfg
from scripts.utils import rate_limit_check, is_url_safe, is_valid_pdf, verify_user
from scripts.indexer import (
    clean_html, chunk_text, extract_headings, extract_links,
    content_hash, AdaptiveRenderer, PlaywrightRenderer, normalize_url,
    should_index_page,
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
from scripts.site_language import SUPPORTED_LANGUAGES, detect_page_language

router = APIRouter()

QUICK_DEMO_TTL = timedelta(hours=1)
REGISTERED_TRIAL_TTL = timedelta(hours=3)


def _fallback_website_name(url: str) -> str:
    host = (urlsplit(url).hostname or "website").removeprefix("www.")
    return host[:80]


def _website_name(html: str, title: str, url: str) -> str:
    """Get a short, plain-text name for the demo greeting."""
    soup = BeautifulSoup(html, "lxml")
    name = ""
    for meta in soup.find_all("meta"):
        key = str(meta.get("property") or meta.get("name") or "").lower()
        if key == "og:site_name":
            name = str(meta.get("content") or "")
            break
    if not name:
        parts = [part.strip() for part in re.split(r"\s+[|—–-]\s+", title or "") if part.strip()]
        if parts:
            generic = {"home", "homepage", "главная", "welcome"}
            name = parts[-1] if parts[0].casefold() in generic and len(parts) > 1 else parts[0]
            if name.casefold() in generic:
                name = ""
    name = re.sub(r"[\x00-\x1f\x7f]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:80] or _fallback_website_name(url)


def _locale_query(url: str) -> str:
    """Carry an explicitly selected language to same-site pages in this demo."""
    for key, value in parse_qsl(urlsplit(url).query, keep_blank_values=False):
        if key.lower() not in {"lang", "language", "locale", "hl"}:
            continue
        code = value.lower().replace("_", "-").split("-", 1)[0]
        if code in SUPPORTED_LANGUAGES:
            return urlencode({key: value})
    return ""


def _is_spa_route(url: str) -> bool:
    return urlsplit(url).fragment.lstrip("!").startswith("/")


def _locale_path_code(segment: str) -> str | None:
    if not re.fullmatch(r"[a-z]{2}(?:[-_][a-z]{2})?", segment, flags=re.IGNORECASE):
        return None
    code = segment.lower().replace("_", "-").split("-", 1)[0]
    return code if code in SUPPORTED_LANGUAGES else None


def _trial_page_links(html: str, page_url: str, allowed_domain: str,
                      locale_query: str, locale_path: str, spa_locale: str | None) -> list[str]:
    links = extract_links(html, page_url, allowed_domain)
    if locale_path:
        links = [
            link for link in links
            if urlsplit(link).path == locale_path.rstrip("/") or urlsplit(link).path.startswith(locale_path)
        ]
    if locale_query:
        selected_code = parse_qsl(locale_query)[0][1].lower().replace("_", "-").split("-", 1)[0]
        links = [
            link for link in links
            if not (
                (segment := urlsplit(link).path.strip("/").split("/", 1)[0])
                and (code := _locale_path_code(segment))
                and code != selected_code
            )
        ]
        links = [urlunsplit((*urlsplit(link)[:3], locale_query, "")) for link in links]
    if spa_locale is not None:
        soup = BeautifulSoup(html, "lxml")
        allowed_host = urlsplit(page_url).hostname
        route_links = []
        for anchor in soup.find_all("a", href=True):
            full = urljoin(page_url, str(anchor["href"]))
            parsed = urlsplit(full)
            if parsed.scheme in {"http", "https"} and parsed.hostname == allowed_host and _is_spa_route(full):
                route_path = parsed.fragment.lstrip("!")
                route_segment = route_path.strip("/").split("/", 1)[0]
                route_locale = _locale_path_code(route_segment)
                if spa_locale and route_locale != spa_locale:
                    continue
                route_links.append(full)
        # extract_links() strips hash routes. Do not also crawl their bare root,
        # which may show a different language from the submitted SPA route.
        route_bases = {normalize_url(link) for link in route_links}
        links = [link for link in links if normalize_url(link) not in route_bases]
        links.extend(route_links)
    links = list(dict.fromkeys(links))
    return links


def purge_trial_runtime_state(site_ids: set[int]) -> None:
    """Drop in-process state belonging to deleted temporary sites."""
    if not site_ids:
        return
    for site_id in site_ids:
        cfg.trial_progress.pop(site_id, None)
        cfg._site_lang_cache.pop(site_id, None)
    prefixes = tuple(f"{site_id}:" for site_id in site_ids)
    with cfg._session_lock:
        keys = {
            key for key in (*cfg._session_history, *cfg._session_last_access)
            if key.startswith(prefixes)
        }
        for key in keys:
            cfg._session_history.pop(key, None)
            cfg._session_last_access.pop(key, None)


# ── Background indexing ─────────────────────────────────────────────────────

def run_trial_indexing(site_id: int, url: str, max_pages: int,
                       pdf_data: list[dict], use_playwright: bool = False,
                       job_id: str | None = None, *, trial_metadata: bool = False,
                       auto_language: bool = False):
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
        renderer = PlaywrightRenderer() if use_playwright else AdaptiveRenderer()

        try:
            # Preserve the selected locale query or SPA hash on the first fetch.
            # normalize_url() intentionally removes both, which would index a
            # different language version from the one the visitor submitted.
            queue = deque([url if trial_metadata else normalize_url(url)])
            visited = set()
            seen_content_hashes = set()
            pages_crawled = 0
            allowed_domain = urlparse(url).netloc
            locale_query = _locale_query(url) if trial_metadata else ""
            segments = [segment for segment in urlsplit(url).path.split("/") if segment]
            locale_segment = segments[0] if segments else ""
            locale_path = f"/{locale_segment}/" if trial_metadata and _locale_path_code(locale_segment) else ""
            spa_locale = None
            if trial_metadata and _is_spa_route(url):
                route_segment = urlsplit(url).fragment.lstrip("!").strip("/").split("/", 1)[0]
                spa_locale = _locale_path_code(route_segment) or ""
            metadata_captured = False

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

                if not should_index_page(html):
                    cfg.log.info(
                        "Skipping noindex or authentication page",
                        extra={"event": "indexing.page_skipped", "site_id": site_id, "url": page_url},
                    )
                    continue

                title, text = clean_html(html)
                if trial_metadata and not metadata_captured:
                    metadata_captured = True
                    progress = cfg.trial_progress.get(site_id)
                    if progress is not None:
                        progress["website_name"] = _website_name(html, title, url)
                    if auto_language:
                        detected_language = detect_page_language(html, url, text)
                        if detected_language:
                            cfg.sb.table("sites").update({"language": detected_language}).eq("id", site_id).execute()
                            cfg._site_lang_cache.pop(site_id, None)
                        if progress is not None:
                            progress["language"] = detected_language or "en"

                if len(text) < 50:
                    for link in _trial_page_links(html, page_url, allowed_domain, locale_query, locale_path, spa_locale):
                        if link not in visited:
                            queue.append(link)
                    continue

                page_content_hash = content_hash(text)
                if page_content_hash in seen_content_hashes:
                    cfg.log.info(
                        "Skipping duplicate page content",
                        extra={"event": "indexing.page_duplicate", "site_id": site_id, "url": page_url},
                    )
                    for link in _trial_page_links(html, page_url, allowed_domain, locale_query, locale_path, spa_locale):
                        if link not in visited:
                            queue.append(link)
                    continue
                seen_content_hashes.add(page_content_hash)

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
                    for link in _trial_page_links(html, page_url, allowed_domain, locale_query, locale_path, spa_locale):
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
    auto_language: bool = False,
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
    if kind == "trial":
        cfg.trial_progress[site_id]["website_name"] = _fallback_website_name(url)

    def run() -> None:
        try:
            if replace_existing:
                docs = cfg.sb.table("documents").select("id").eq("site_id", site_id).execute()
                for doc in (docs.data or []):
                    cfg.sb.table("chunks").delete().eq("document_id", doc["id"]).execute()
                cfg.sb.table("documents").delete().eq("site_id", site_id).execute()
            run_trial_indexing(
                site_id, url, max_pages, pdf_data, use_playwright, job_id,
                trial_metadata=kind == "trial", auto_language=auto_language,
            )
        except Exception as exc:
            fail_indexing_job(job_id, site_id, exc)

    asyncio.get_event_loop().create_task(asyncio.to_thread(run))
    return job_id


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/")
async def root_redirect():
    return RedirectResponse(url="/trial", status_code=308)


@router.get("/trial")
async def serve_trial_page():
    html_path = cfg.WIDGET_DIR / "trial.html"
    if not html_path.exists():
        return JSONResponse({"error": "trial.html not found"}, status_code=404)
    return FileResponse(html_path, media_type="text/html", headers={"Cache-Control": "no-cache"})


@router.get("/robots.txt", response_class=PlainTextResponse)
async def serve_robots():
    return "\n".join([
        "User-agent: *",
        "Allow: /",
        "Disallow: /app",
        "Disallow: /admin",
        "Disallow: /designs",
        "Disallow: /internal/",
        "Disallow: /api/",
        f"Sitemap: {cfg.PUBLIC_URL}/sitemap.xml",
        "",
    ])


@router.get("/sitemap.xml")
async def serve_sitemap():
    last_modified = datetime.now(timezone.utc).date().isoformat()
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f'<url><loc>{cfg.PUBLIC_URL}/trial</loc><lastmod>{last_modified}</lastmod>'
        '<changefreq>weekly</changefreq><priority>1.0</priority></url>'
        '</urlset>'
    )
    return Response(content=xml, media_type="application/xml", headers={"Cache-Control": "public, max-age=3600"})


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
    return FileResponse(
        css_path,
        media_type="text/css",
        headers={"Cache-Control": "public, max-age=300, stale-while-revalidate=86400"},
    )


@router.post("/api/trial/start")
async def trial_start(
    request: Request,
    url: str = Form(...),
    max_pages: int = Form(default=10),
    language: str = Form(default="auto"),
    use_playwright: str = Form(default="auto"),
    token: str = Form(default=""),
    pdfs: list[UploadFile] = File(default=[]),
):
    blocked = rate_limit_check(request, "trial_start", 3, 3600)
    if blocked:
        return blocked
    current_user = verify_user(request, require_active=True)

    url = url.strip()
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
    auto_language = language.strip().lower() in {"", "auto"}
    initial_language = "en" if auto_language else language.strip().lower()
    quick_demo = not token.strip()

    domain = f"trial-{uuid.uuid4().hex[:8]}.demo"
    expires_at = (datetime.now(timezone.utc) + (QUICK_DEMO_TTL if quick_demo else REGISTERED_TRIAL_TTL)).isoformat()

    site_settings = {"source_url": url, "trial": True, "quick_demo": quick_demo}
    if token.strip():
        site_settings["owner_token"] = token.strip()

    pdf_data = []
    for pdf in pdfs:
        content = await pdf.read()
        if len(content) > cfg.MAX_PDF_SIZE:
            return JSONResponse(
                {"error": f"PDF '{pdf.filename}' exceeds 10MB limit"},
                status_code=400,
            )
        pdf_data.append({"filename": pdf.filename, "content": content})

    site_resp = cfg.sb.table("sites").insert({
        "domain": domain,
        "language": initial_language,
        "is_trial": True,
        "expires_at": expires_at,
        "owner_user_id": current_user["user_id"] if current_user else None,
        "settings": site_settings,
    }).execute()
    site_id = site_resp.data[0]["id"]

    pw = use_playwright == "1" or _is_spa_route(url)
    job_id = schedule_indexing_job(
        site_id,
        url,
        max_pages,
        pdf_data,
        pw,
        kind="trial",
        message="Trial indexing queued",
        auto_language=auto_language,
    )
    if not auto_language:
        cfg.trial_progress[site_id]["language"] = initial_language

    cfg.log.info(f"Trial started: site_id={site_id} url={url} pdfs={len(pdf_data)} max_pages={max_pages} playwright={pw}")
    return {
        "site_id": site_id,
        "job_id": job_id,
        "message": "Indexing started",
        "expires_at": expires_at,
        "website_name": _fallback_website_name(url),
    }


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
        purge_trial_runtime_state({site_id})
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
    deleted_ids = {row["id"] for row in (resp.data or [])}
    purge_trial_runtime_state(deleted_ids)
    deleted = len(deleted_ids)
    return {"deleted": deleted}
