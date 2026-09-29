import asyncio

from starlette.requests import Request
from starlette.responses import HTMLResponse

import scripts.config as cfg
from scripts.routes.trial import root_redirect, serve_robots, serve_sitemap
from scripts.routes.user import _session_response
from scripts.server import SecurityHeadersMiddleware


def test_robots_and_sitemap_publish_only_the_public_landing(monkeypatch):
    monkeypatch.setattr(cfg, "PUBLIC_URL", "https://wrs.example")

    robots = asyncio.run(serve_robots())
    sitemap = asyncio.run(serve_sitemap())
    sitemap_text = sitemap.body.decode()

    assert "Disallow: /app" in robots
    assert "Disallow: /internal/" in robots
    assert "Disallow: /api/" in robots
    assert "Sitemap: https://wrs.example/sitemap.xml" in robots
    assert "https://wrs.example/trial" in sitemap_text
    assert "/app" not in sitemap_text
    assert asyncio.run(root_redirect()).status_code == 308


async def _security_response(*, forwarded_proto: str | None = None):
    headers = []
    if forwarded_proto:
        headers.append((b"x-forwarded-proto", forwarded_proto.encode()))
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": headers,
        "scheme": "http",
        "server": ("localhost", 80),
        "client": ("127.0.0.1", 1234),
    })
    middleware = SecurityHeadersMiddleware(lambda _scope, _receive, _send: None)

    async def call_next(_request):
        return HTMLResponse("<h1>Safe</h1>")

    return await middleware.dispatch(request, call_next)


def test_security_headers_are_added_to_html_responses():
    response = asyncio.run(_security_response(forwarded_proto="https"))

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "SAMEORIGIN"
    assert response.headers["strict-transport-security"].startswith("max-age=31536000")
    assert "object-src 'none'" in response.headers["content-security-policy"]
    assert "upgrade-insecure-requests" in response.headers["content-security-policy"]


def test_http_development_pages_do_not_upgrade_local_assets():
    response = asyncio.run(_security_response())

    assert "upgrade-insecure-requests" not in response.headers["content-security-policy"]


def test_browser_session_cookie_is_http_only_and_secure(monkeypatch):
    monkeypatch.setattr(cfg, "USER_SESSION_COOKIE", "test_session")
    monkeypatch.setattr(cfg, "USER_SESSION_COOKIE_SECURE", True)

    response = _session_response({"success": True}, "secret-token")
    cookie = response.headers["set-cookie"]

    assert cookie.startswith("test_session=secret-token;")
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Secure" in cookie
