"""Infer the language of the page a trial visitor chose to crawl."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

from bs4 import BeautifulSoup


SUPPORTED_LANGUAGES = frozenset({
    "en", "ru", "kk", "es", "fr", "de", "zh", "ja", "ko", "pt", "ar",
    "hi", "it", "tr", "nl", "pl", "uk",
})
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_KAZAKH_RE = re.compile(r"[ӘәҒғҚқҢңӨөҰұҮүҺһІі]")
_UKRAINIAN_RE = re.compile(r"[ЄєЇїҐґ]")


def _supported_code(value: str | None) -> str | None:
    """Normalize a locale such as ru-RU or ru_RU to a supported base code."""
    code = str(value or "").strip().lower().replace("_", "-").split(",", 1)[0]
    code = code.split(";", 1)[0].split("-", 1)[0]
    return code if code in SUPPORTED_LANGUAGES else None


def _path_locale(path: str) -> str | None:
    segments = [segment for segment in path.lstrip("!").split("/") if segment]
    if not segments:
        return None
    first = segments[0]
    if re.fullmatch(r"[a-z]{2}(?:[-_][a-z]{2})?", first, flags=re.IGNORECASE):
        return _supported_code(first)
    return None


def detect_page_language(html: str, url: str, text: str | None = None) -> str | None:
    """Use the selected URL, page metadata, then visible copy as language signals.

    The URL wins when it explicitly selects a localized version (for example,
    ``/ru/``, ``/#/ru``, or ``?lang=ru``). Generic Cyrillic copy is only treated as Russian
    when it makes up a substantial part of the page.
    """
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    for key in ("lang", "language", "locale", "hl"):
        for value in query.get(key, []):
            code = _supported_code(value)
            if code:
                return code

    for path in (parsed.path, parsed.fragment):
        code = _path_locale(path)
        if code:
            return code

    subdomain = (parsed.hostname or "").split(".", 1)[0]
    if "." in (parsed.hostname or ""):
        code = _supported_code(subdomain)
        if code:
            return code

    soup = BeautifulSoup(html, "lxml")
    if soup.html:
        code = _supported_code(soup.html.get("lang") or soup.html.get("xml:lang"))
        if code:
            return code

    for meta in soup.find_all("meta"):
        key = str(meta.get("http-equiv") or meta.get("name") or meta.get("property") or "").lower()
        if key in {"content-language", "language", "og:locale"}:
            code = _supported_code(meta.get("content"))
            if code:
                return code

    if text is None:
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
    cyrillic_count = len(_CYRILLIC_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))
    if cyrillic_count >= 30 and cyrillic_count >= latin_count * 0.3:
        if len(_KAZAKH_RE.findall(text)) >= 2:
            return "kk"
        if len(_UKRAINIAN_RE.findall(text)) >= 2:
            return "uk"
        return "ru"
    return None
