#!/usr/bin/env python3
"""
web-rag indexer: crawl a site, clean HTML, chunk text, embed, store in Supabase.

Supports both static sites (requests) and JS-rendered SPAs (Playwright).

Usage:
    python3 indexer.py --site-id 1 --max-pages 200
    python3 indexer.py --site-id 1 --max-pages 50 --start-url https://example.com/docs
    python3 indexer.py --site-id 1 --max-pages 50 --renderer playwright
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import requests
from bs4 import BeautifulSoup, Comment
from sentence_transformers import SentenceTransformer
from supabase import create_client

# ── Config ──────────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
REQUEST_DELAY = float(os.environ.get("REQUEST_DELAY", "0.5"))
USER_AGENT = "ZeptoClaw-WebRAG/1.0"

# ── Supabase client ────────────────────────────────────────────────────────

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Embedding model (lazy-loaded) ──────────────────────────────────────────

_model = None


def get_model():
    global _model
    if _model is None:
        print(f"[embed] Loading model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


# ── Renderers ──────────────────────────────────────────────────────────────

class StaticRenderer:
    """Fetch pages with requests (fast, no JS)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def fetch(self, url: str) -> tuple[str, int] | None:
        """Returns (html, status_code) or None on error."""
        try:
            resp = self.session.get(url, timeout=15)
            if "text/html" not in resp.headers.get("content-type", ""):
                return None
            return resp.text, resp.status_code
        except requests.RequestException as e:
            print(f"  skip  {url} ({e})")
            return None

    def close(self):
        pass


class PlaywrightRenderer:
    """Fetch pages with headless Chromium (handles JS-rendered SPAs)."""

    def __init__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._context = self._browser.new_context(
            user_agent=USER_AGENT,
            ignore_https_errors=True,
        )
        self._page = self._context.new_page()

    def fetch(self, url: str) -> tuple[str, int] | None:
        """Navigate, wait for network idle, return rendered HTML."""
        try:
            resp = self._page.goto(url, wait_until="networkidle", timeout=30000)
            if resp is None:
                return None
            # Wait a bit more for lazy-loaded content
            self._page.wait_for_timeout(1000)
            html = self._page.content()
            return html, resp.status
        except Exception as e:
            print(f"  skip  {url} ({e})")
            return None

    def close(self):
        try:
            self._browser.close()
            self._pw.stop()
        except Exception:
            pass


def get_renderer(name: str):
    if name == "playwright":
        print("[render] Using Playwright (headless Chromium)")
        return PlaywrightRenderer()
    else:
        print("[render] Using static requests")
        return StaticRenderer()


# ── HTML cleaning ──────────────────────────────────────────────────────────

STRIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg", "iframe"}


def clean_html(html: str) -> tuple[str, str]:
    """Extract title and clean body text from raw HTML."""
    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Remove unwanted tags
    for tag in soup.find_all(STRIP_TAGS):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Extract text, collapse whitespace
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    return title, text


def extract_headings(html: str) -> list[str]:
    """Pull h1-h3 headings from HTML for chunk metadata."""
    soup = BeautifulSoup(html, "lxml")
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        t = tag.get_text(strip=True)
        if t:
            headings.append(t)
    return headings


# ── Chunking ───────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap

    return chunks


# ── Link extraction ────────────────────────────────────────────────────────

def extract_links(html: str, base_url: str, allowed_domain: str) -> list[str]:
    """Extract same-domain links from HTML."""
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        # Same domain, no fragments, no common non-page extensions
        if parsed.netloc == allowed_domain and not parsed.fragment:
            skip_ext = (".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp4", ".mp3")
            if not parsed.path.lower().endswith(skip_ext):
                clean = parsed._replace(fragment="", query="").geturl()
                links.append(clean)
    return links


# ── Crawl + index ──────────────────────────────────────────────────────────

def fetch_site_config(site_id: int) -> dict:
    """Load site row from Supabase."""
    resp = sb.table("sites").select("*").eq("id", site_id).single().execute()
    return resp.data


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def index_site(site_id: int, max_pages: int, start_url: str | None = None, renderer_name: str = "static"):
    site = fetch_site_config(site_id)
    domain = site["domain"]
    base = start_url or f"https://{domain}"
    allowed_domain = urlparse(base).netloc

    print(f"[crawl] Site #{site_id}: {domain}")
    print(f"[crawl] Start URL: {base}")
    print(f"[crawl] Max pages: {max_pages}")
    print()

    renderer = get_renderer(renderer_name)

    queue = deque([base])
    visited = set()

    stats = {
        "pages_crawled": 0,
        "pages_skipped": 0,
        "chunks_inserted": 0,
        "embeddings_generated": 0,
    }

    model = get_model()

    try:
        while queue and stats["pages_crawled"] < max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            # Fetch page
            result = renderer.fetch(url)
            if result is None:
                stats["pages_skipped"] += 1
                continue

            html, status = result
            if status != 200:
                stats["pages_skipped"] += 1
                print(f"  skip  {url} (HTTP {status})")
                continue

            title, text = clean_html(html)

            if len(text) < 50:
                stats["pages_skipped"] += 1
                print(f"  skip  {url} (too short)")
                # Still extract links from rendered HTML
                for link in extract_links(html, url, allowed_domain):
                    if link not in visited:
                        queue.append(link)
                continue

            # Check if content changed (dedup via hash)
            c_hash = content_hash(text)
            existing = (
                sb.table("documents")
                .select("id, content_hash")
                .eq("site_id", site_id)
                .eq("url", url)
                .execute()
            )

            doc_id = None
            if existing.data:
                doc = existing.data[0]
                if doc["content_hash"] == c_hash:
                    stats["pages_skipped"] += 1
                    print(f"  skip  {url} (unchanged)")
                    for link in extract_links(html, url, allowed_domain):
                        if link not in visited:
                            queue.append(link)
                    continue
                # Content changed — update doc, delete old chunks
                doc_id = doc["id"]
                sb.table("chunks").delete().eq("document_id", doc_id).execute()
                sb.table("documents").update({
                    "title": title or url,
                    "content_hash": c_hash,
                    "last_crawled": "now()",
                }).eq("id", doc_id).execute()
            else:
                # New document
                ins = sb.table("documents").insert({
                    "site_id": site_id,
                    "url": url,
                    "title": title or url,
                    "content_hash": c_hash,
                }).execute()
                doc_id = ins.data[0]["id"]

            stats["pages_crawled"] += 1

            # Chunk
            headings = extract_headings(html)
            chunks = chunk_text(text)

            if not chunks:
                print(f"  crawl {url} → 0 chunks")
                for link in extract_links(html, url, allowed_domain):
                    if link not in visited:
                        queue.append(link)
                continue

            # Embed all chunks in batch
            texts_to_embed = [f"passage: {c}" for c in chunks]
            embeddings = model.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)
            stats["embeddings_generated"] += len(embeddings)

            # Insert chunks
            rows = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                rows.append({
                    "document_id": doc_id,
                    "chunk_index": i,
                    "text": chunk,
                    "headings": headings,
                    "embedding": emb.tolist(),
                })

            sb.table("chunks").insert(rows).execute()
            stats["chunks_inserted"] += len(rows)

            print(f"  crawl {url} → {len(chunks)} chunks")

            # Enqueue links
            for link in extract_links(html, url, allowed_domain):
                if link not in visited:
                    queue.append(link)

            time.sleep(REQUEST_DELAY)
    finally:
        renderer.close()

    # Final report
    print()
    print("─" * 50)
    print(f"  Pages crawled:       {stats['pages_crawled']}")
    print(f"  Pages skipped:       {stats['pages_skipped']}")
    print(f"  Chunks inserted:     {stats['chunks_inserted']}")
    print(f"  Embeddings generated:{stats['embeddings_generated']}")
    print("─" * 50)

    return stats


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="web-rag indexer")
    parser.add_argument("--site-id", type=int, required=True, help="Site ID from sites table")
    parser.add_argument("--max-pages", type=int, default=100, help="Max pages to crawl")
    parser.add_argument("--start-url", type=str, default=None, help="Override start URL")
    parser.add_argument(
        "--renderer", type=str, default="static", choices=["static", "playwright"],
        help="Renderer: 'static' (requests) or 'playwright' (headless browser for JS sites)"
    )
    args = parser.parse_args()

    try:
        stats = index_site(args.site_id, args.max_pages, args.start_url, args.renderer)
        print(json.dumps(stats))
    except KeyError as e:
        print(f"Error: missing env var {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
