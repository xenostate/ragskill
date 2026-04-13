#!/usr/bin/env python3
"""
web-rag indexer: crawl a site, clean HTML, chunk text, embed, store in Supabase.

Supports both static sites (requests) and JS-rendered SPAs (Playwright).

Usage:
    python3 indexer.py --site-id 1 --max-pages 200
    python3 indexer.py --site-id 1 --max-pages 50 --start-url https://example.com/docs
    python3 indexer.py --site-id 1 --max-pages 50 --renderer playwright
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone
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
    """Fetch pages with headless Chromium (handles JS-rendered SPAs).

    Each fetch() creates a fresh page to avoid state leaking between navigations
    and to be safe if multiple crawls run concurrently in separate threads.
    """

    def __init__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)

    def fetch(self, url: str) -> tuple[str, int] | None:
        """Navigate in a fresh context/page, wait for network idle, return rendered HTML."""
        context = None
        try:
            context = self._browser.new_context(user_agent=USER_AGENT)
            page = context.new_page()
            resp = page.goto(url, wait_until="networkidle", timeout=30000)
            if resp is None:
                return None
            # Wait a bit more for lazy-loaded content
            page.wait_for_timeout(1000)
            html = page.content()
            return html, resp.status
        except Exception as e:
            print(f"  skip  {url} ({e})")
            return None
        finally:
            if context:
                try:
                    context.close()
                except Exception:
                    pass

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

# CSS classes / IDs that indicate junk sections (references, navboxes, sidebars, etc.)
JUNK_CLASSES = re.compile(
    r"reflist|references|ref-list|mw-references|citation|navbox|sidebar|"
    r"infobox|mw-editsection|catlinks|printfooter|mw-jump-link|"
    r"noprint|metadata|hatnote|ambox|dmbox|tmbox|fmbox|ombox|"
    r"external[_-]links|see-also|authority-control|portal-bar",
    re.IGNORECASE,
)


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

    # Remove citation markers [1], [2], etc.
    for sup in soup.find_all("sup", class_="reference"):
        sup.decompose()

    # Remove junk sections: references, navboxes, sidebars, infoboxes, etc.
    for tag in list(soup.find_all(True)):
        if not tag.parent:
            continue
        classes = " ".join(tag.get("class") or [])
        tag_id = tag.get("id") or ""
        if JUNK_CLASSES.search(classes) or JUNK_CLASSES.search(tag_id):
            tag.decompose()

    # Convert tables to readable text (preserve header→value context)
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        headers = []
        text_lines = []
        for row in rows:
            cells = row.find_all(["th", "td"])
            cell_texts = [c.get_text(strip=True) for c in cells]
            # Detect header row
            if row.find("th") and not row.find("td"):
                headers = cell_texts
            elif headers and len(cell_texts) == len(headers):
                pairs = [f"{h}: {v}" for h, v in zip(headers, cell_texts) if v]
                if pairs:
                    text_lines.append(" | ".join(pairs))
            else:
                line = " | ".join(c for c in cell_texts if c)
                if line:
                    text_lines.append(line)
        table.replace_with(BeautifulSoup("\n".join(text_lines) + "\n", "lxml"))

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


# ── Sentence splitting ─────────────────────────────────────────────────────

# Common abbreviations that end with a period but aren't sentence endings
_ABBREVIATIONS = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc',
                  'inc', 'ltd', 'co', 'corp', 'st', 'ave', 'fig', 'no', 'vol',
                  'dept', 'univ', 'approx', 'e.g', 'i.e', 'u.s', 'u.k'}

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting abbreviations and decimal numbers."""
    # Use a regex that finds potential sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', text)

    sentences = []
    buffer = ""
    for part in parts:
        if buffer:
            buffer += " " + part
        else:
            buffer = part

        # Check if buffer ends with an abbreviation or decimal number
        last_word = buffer.rstrip('.!?').rsplit(None, 1)[-1].lower() if buffer.rstrip('.!?') else ""
        ends_with_abbrev = last_word in _ABBREVIATIONS
        ends_with_number = bool(re.search(r'\d\.\d*$', buffer.rstrip()))

        if not ends_with_abbrev and not ends_with_number:
            sentences.append(buffer)
            buffer = ""

    if buffer:
        if sentences:
            sentences[-1] += " " + buffer
        else:
            sentences.append(buffer)

    return [s for s in sentences if s.strip()]


# ── Chunking ───────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks respecting paragraph and sentence boundaries."""
    if not text.strip():
        return []

    # Split into paragraphs (double-newline separated)
    paragraphs = re.split(r'\n\s*\n', text)

    # Break large paragraphs into sentence-level segments
    segments = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_words = para.split()
        if len(para_words) <= chunk_size:
            segments.append(para)
        else:
            # Paragraph exceeds chunk_size — split by sentences
            sentences = _split_sentences(para)
            current = []
            current_len = 0
            for sent in sentences:
                sent_words = sent.split()
                if current and current_len + len(sent_words) > chunk_size:
                    segments.append(" ".join(current))
                    current = []
                    current_len = 0
                current.extend(sent_words)
                current_len += len(sent_words)
            if current:
                segments.append(" ".join(current))

    if not segments:
        return []

    # Greedily merge segments into chunks up to chunk_size words,
    # with overlap by carrying last segment(s) from previous chunk
    chunks = []
    current_words = []
    current_segs = []

    for seg in segments:
        seg_words = seg.split()

        if current_words and len(current_words) + len(seg_words) > chunk_size:
            # Emit current chunk
            chunks.append(" ".join(current_words))

            # Overlap: carry last segment(s) fitting within overlap word count
            overlap_words = []
            overlap_segs = []
            for prev_seg in reversed(current_segs):
                pw = prev_seg.split()
                if len(overlap_words) + len(pw) <= overlap:
                    overlap_words = pw + overlap_words
                    overlap_segs.insert(0, prev_seg)
                else:
                    break
            current_words = overlap_words
            current_segs = overlap_segs

        current_words.extend(seg_words)
        current_segs.append(seg)

    if current_words:
        chunks.append(" ".join(current_words))

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
                # Normalize: remove trailing slash for consistency
                clean_path = parsed.path.rstrip('/')
                if not clean_path:
                    clean_path = '/'
                clean = parsed._replace(fragment="", query="", path=clean_path).geturl()
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
                    "last_crawled": datetime.now(timezone.utc).isoformat(),
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
