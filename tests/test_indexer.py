"""
Tests for scripts/indexer.py — chunking, HTML cleaning, link extraction, content hashing.
"""

from requests import Response

from scripts.indexer import (
    StaticRenderer,
    clean_html,
    chunk_text,
    extract_headings,
    extract_links,
    content_hash,
    normalize_url,
    needs_js_rendering,
    should_index_page,
    _split_sentences,
)


def test_static_renderer_decodes_utf8_without_http_charset(monkeypatch):
    response = Response()
    response.status_code = 200
    response.headers["content-type"] = "text/html"
    response._content = '<html><head><title>Roman Ten — Applied AI</title></head></html>'.encode("utf-8")
    renderer = StaticRenderer()
    monkeypatch.setattr(renderer.session, "get", lambda *_args, **_kwargs: response)

    html, status = renderer.fetch("https://example.com")

    assert status == 200
    assert "Roman Ten — Applied AI" in html


# ── clean_html ─────────────────────────────────────────────────────────────

class TestCleanHtml:
    def test_extracts_title(self):
        html = "<html><head><title>My Page</title></head><body><p>Hello world</p></body></html>"
        title, text = clean_html(html)
        assert title == "My Page"
        assert "Hello world" in text

    def test_strips_script_style(self):
        html = """
        <html><body>
            <script>alert('xss')</script>
            <style>.foo{color:red}</style>
            <p>Real content here</p>
        </body></html>
        """
        title, text = clean_html(html)
        assert "alert" not in text
        assert "color" not in text
        assert "Real content here" in text

    def test_strips_nav_footer(self):
        html = """
        <html><body>
            <nav><a href="/">Home</a></nav>
            <main><p>Main content</p></main>
            <footer>Copyright 2024</footer>
        </body></html>
        """
        title, text = clean_html(html)
        assert "Main content" in text
        # nav and footer should be stripped
        assert "Copyright" not in text

    def test_removes_reference_sections(self):
        html = """
        <html><body>
            <p>Article content.</p>
            <div class="reflist"><p>[1] Some reference</p></div>
        </body></html>
        """
        title, text = clean_html(html)
        assert "Article content" in text
        assert "Some reference" not in text

    def test_empty_html(self):
        title, text = clean_html("<html><body></body></html>")
        assert title == ""

    def test_no_title(self):
        html = "<html><body><p>Content only</p></body></html>"
        title, text = clean_html(html)
        assert title == ""
        assert "Content only" in text

    def test_table_conversion(self):
        html = """
        <html><body>
            <table>
                <tr><th>Name</th><th>Age</th></tr>
                <tr><td>Alice</td><td>30</td></tr>
            </table>
        </body></html>
        """
        title, text = clean_html(html)
        assert "Alice" in text
        assert "30" in text


# ── chunk_text ─────────────────────────────────────────────────────────────

class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        text = "This is a short paragraph."
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_respects_paragraph_boundaries(self):
        text = "First paragraph.\n\nSecond paragraph."
        chunks = chunk_text(text, chunk_size=100, overlap=0)
        # Both paragraphs should fit in one chunk
        assert len(chunks) == 1
        assert "First" in chunks[0]
        assert "Second" in chunks[0]

    def test_splits_large_text(self):
        # Create text with multiple paragraphs that exceed chunk_size
        paragraphs = [" ".join(["word"] * 30) for _ in range(8)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=50, overlap=5)
        assert len(chunks) > 1

    def test_overlap_present(self):
        # Use small paragraphs that merge into chunks, forcing overlap on split
        # Each paragraph is ~10 words; chunk_size=25 means 2 paragraphs fit per chunk
        paras = [f"para{i} " + " ".join([f"w{i}"] * 8) for i in range(6)]
        text = "\n\n".join(paras)
        chunks = chunk_text(text, chunk_size=25, overlap=10)
        assert len(chunks) >= 2
        # With overlap, the last paragraph from chunk N should appear at start of chunk N+1
        # Check that chunk 1 begins with content that also appeared in chunk 0
        c0_words = set(chunks[0].split())
        c1_first_words = chunks[1].split()[:12]
        overlap_found = any(w in c0_words for w in c1_first_words)
        assert overlap_found, "No overlap found between chunks"

    def test_multiple_paragraphs(self):
        paragraphs = [f"Paragraph {i} content." for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=20, overlap=3)
        assert len(chunks) >= 2
        # All content should be represented
        all_text = " ".join(chunks)
        for i in range(10):
            assert f"Paragraph {i}" in all_text


# ── _split_sentences ───────────────────────────────────────────────────────

class TestSplitSentences:
    def test_basic_split(self):
        text = "First sentence. Second sentence."
        sentences = _split_sentences(text)
        assert len(sentences) == 2

    def test_abbreviation_preserved(self):
        text = "Dr. Smith went home. He was tired."
        sentences = _split_sentences(text)
        # "Dr." should NOT cause a split
        assert any("Dr." in s and "Smith" in s for s in sentences)

    def test_single_sentence(self):
        text = "Just one sentence."
        sentences = _split_sentences(text)
        assert len(sentences) == 1

    def test_question_and_exclamation(self):
        text = "Is this a test? Yes it is! Great."
        sentences = _split_sentences(text)
        assert len(sentences) == 3


# ── extract_headings ───────────────────────────────────────────────────────

class TestExtractHeadings:
    def test_basic_headings(self):
        html = "<html><body><h1>Title</h1><h2>Subtitle</h2><h3>Section</h3><p>text</p></body></html>"
        headings = extract_headings(html)
        assert headings == ["Title", "Subtitle", "Section"]

    def test_no_headings(self):
        html = "<html><body><p>Just text</p></body></html>"
        assert extract_headings(html) == []

    def test_ignores_h4_and_below(self):
        html = "<html><body><h1>H1</h1><h4>H4</h4><h5>H5</h5></body></html>"
        headings = extract_headings(html)
        assert headings == ["H1"]


# ── extract_links ──────────────────────────────────────────────────────────

class TestExtractLinks:
    def test_same_domain_links(self):
        html = '<html><body><a href="/about">About</a><a href="https://other.com">Other</a></body></html>'
        links = extract_links(html, "https://example.com/", "example.com")
        assert any("/about" in link for link in links)
        assert not any("other.com" in link for link in links)

    def test_skips_media_files(self):
        html = '<html><body><a href="/image.png">Pic</a><a href="/page">Page</a></body></html>'
        links = extract_links(html, "https://example.com/", "example.com")
        assert not any(".png" in link for link in links)
        assert any("/page" in link for link in links)

    def test_no_fragments(self):
        html = '<html><body><a href="/page#section">Section</a></body></html>'
        links = extract_links(html, "https://example.com/", "example.com")
        for link in links:
            assert "#" not in link

    def test_relative_urls_resolved(self):
        html = '<html><body><a href="subpage">Sub</a></body></html>'
        links = extract_links(html, "https://example.com/docs/", "example.com")
        assert any("subpage" in link for link in links)

    def test_deduplicates_root_and_removes_query(self):
        html = '''
        <a href="https://example.com">Root</a>
        <a href="https://example.com/?campaign=test">Root campaign</a>
        <a href="/#top">Root fragment</a>
        '''
        links = extract_links(html, "https://example.com", "example.com")
        assert links == ["https://example.com/"]


class TestUrlAndRenderDetection:
    def test_normalize_url_canonicalizes_root(self):
        assert normalize_url("HTTPS://Example.COM:443?campaign=x#top") == "https://example.com/"
        assert normalize_url("https://example.com/docs/") == "https://example.com/docs"

    def test_detects_empty_client_translation_bindings(self):
        html = '''
        <html><body>
          <h1 data-i18n="title"></h1>
          <p data-i18n="summary"></p>
          <button data-i18n="cta"></button>
          <script src="/app.js"></script>
        </body></html>
        '''
        assert needs_js_rendering(html, "Navigation") is True

    def test_does_not_flag_content_rich_page(self):
        text = " ".join(["useful"] * 250)
        html = f'<html><body><main>{text}</main><span data-i18n="label"></span></body></html>'
        assert needs_js_rendering(html, text) is False

    def test_skips_noindex_and_password_pages(self):
        assert should_index_page('<meta name="robots" content="noindex, follow"><p>Private</p>') is False
        assert should_index_page('<meta name="robots" content="noindex nofollow"><p>Private</p>') is False
        assert should_index_page('<form><input type="password"></form>') is False
        assert should_index_page('<main><p>Public documentation</p></main>') is True

    def test_does_not_skip_content_rich_page_with_login_modal(self):
        copy = " ".join(["documentation"] * 220)
        html = f'<title>Product docs</title><main>{copy}</main><form><input type="password"></form>'
        assert should_index_page(html) is True


# ── content_hash ───────────────────────────────────────────────────────────

class TestContentHash:
    def test_deterministic(self):
        h1 = content_hash("hello world")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_different_inputs(self):
        h1 = content_hash("hello")
        h2 = content_hash("world")
        assert h1 != h2

    def test_returns_hex_string(self):
        h = content_hash("test")
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)
