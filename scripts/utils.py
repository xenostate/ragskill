"""
Utility functions: rate limiting, SSRF protection, IP resolution, PDF validation.
"""

from __future__ import annotations

import ipaddress
import socket
import time
import threading
from collections import deque, defaultdict
from urllib.parse import urlparse

from fastapi import Request
from fastapi.responses import JSONResponse

from scripts.config import TRUSTED_PROXY_NETS, log


# ── Rate Limiter ──────────────────────────────────────────────────────────

class RateLimiter:
    """In-memory sliding-window rate limiter per IP address."""

    def __init__(self):
        self._lock = threading.Lock()
        self._hits: dict[str, deque] = defaultdict(deque)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            q = self._hits[key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= max_requests:
                return False
            q.append(now)
            return True

    def cleanup(self):
        now = time.time()
        with self._lock:
            stale = [k for k, q in self._hits.items() if not q or q[-1] < now - 3600]
            for k in stale:
                del self._hits[k]


rate_limiter = RateLimiter()


# ── IP Resolution ────────────────────────────────────────────────────────

def _is_trusted_proxy(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
        return any(addr in net for net in TRUSTED_PROXY_NETS)
    except ValueError:
        return False


def get_client_ip(request: Request) -> str:
    """Extract client IP, trusting X-Forwarded-For only from known proxies."""
    direct_ip = request.client.host if request.client else "unknown"
    if direct_ip != "unknown" and _is_trusted_proxy(direct_ip):
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            parts = [p.strip() for p in forwarded.split(",")]
            for ip in reversed(parts):
                if not _is_trusted_proxy(ip):
                    return ip
    return direct_ip


def rate_limit_check(request: Request, bucket: str, max_requests: int, window: int):
    """Return a 429 JSONResponse if exceeded, else None."""
    ip = get_client_ip(request)
    key = f"{bucket}:{ip}"
    if not rate_limiter.is_allowed(key, max_requests, window):
        log.warning(f"Rate limit hit: {bucket} from {ip}")
        return JSONResponse(
            {"error": "Too many requests. Please try again later."},
            status_code=429,
        )
    return None


# ── SSRF Protection ──────────────────────────────────────────────────────

def is_url_safe(url: str) -> tuple[bool, str]:
    """Check that a URL does not point to private/internal network addresses."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid URL: no hostname"
        if parsed.scheme not in ("http", "https"):
            return False, f"Invalid URL scheme: {parsed.scheme}"
        try:
            addr_infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror:
            return False, f"Cannot resolve hostname: {hostname}"
        for addr_info in addr_infos:
            ip = ipaddress.ip_address(addr_info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                return False, "URL points to a private/internal network address"
            if ip == ipaddress.ip_address("169.254.169.254"):
                return False, "URL points to cloud metadata service"
        return True, ""
    except Exception as e:
        return False, f"URL validation error: {e}"


# ── PDF Validation ───────────────────────────────────────────────────────

def is_valid_pdf(content: bytes) -> bool:
    """Check PDF magic bytes (%PDF-)."""
    return content[:5] == b"%PDF-"


# ── Auth helpers ─────────────────────────────────────────────────────────

def verify_admin(request: Request) -> bool:
    """Check X-Admin-Token header against in-memory store."""
    from scripts.config import admin_tokens
    token = request.headers.get("x-admin-token", "")
    expiry = admin_tokens.get(token)
    if not expiry or time.time() > expiry:
        admin_tokens.pop(token, None)
        return False
    return True


def verify_admin_token(token: str) -> bool:
    """Check a raw admin token string."""
    from scripts.config import admin_tokens
    expiry = admin_tokens.get(token)
    if not expiry or time.time() > expiry:
        admin_tokens.pop(token, None)
        return False
    return True


def verify_internal(request: Request, slug: str, require_admin: bool = False) -> dict | None:
    """Verify internal assistant token. Returns session dict or None."""
    from scripts.config import _internal_tokens, INTERNAL_TOKEN_TTL
    token = request.headers.get("x-internal-token", "")
    if not token or token not in _internal_tokens:
        return None
    session = _internal_tokens[token]
    if time.time() - session["created"] > INTERNAL_TOKEN_TTL:
        _internal_tokens.pop(token, None)
        return None
    if session["slug"] != slug:
        return None
    if require_admin and session["role"] != "admin":
        return None
    return session
