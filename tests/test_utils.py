"""
Tests for scripts/utils.py — rate limiter, SSRF protection, IP resolution, PDF validation, auth.
"""

import time
import threading
from unittest.mock import MagicMock, patch

import pytest

# We need to mock config before importing utils, since utils imports from config
import scripts.config as cfg

from scripts.utils import (
    RateLimiter,
    is_url_safe,
    is_valid_pdf,
    get_client_ip,
    rate_limit_check,
    verify_admin,
    verify_admin_token,
    verify_internal,
)


# ── RateLimiter ────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_allows_under_limit(self):
        rl = RateLimiter()
        for _ in range(5):
            assert rl.is_allowed("key1", 5, 60) is True

    def test_blocks_over_limit(self):
        rl = RateLimiter()
        for _ in range(3):
            rl.is_allowed("key1", 3, 60)
        assert rl.is_allowed("key1", 3, 60) is False

    def test_different_keys_independent(self):
        rl = RateLimiter()
        for _ in range(3):
            rl.is_allowed("a", 3, 60)
        assert rl.is_allowed("a", 3, 60) is False
        assert rl.is_allowed("b", 3, 60) is True

    def test_window_expiry(self):
        rl = RateLimiter()
        # Fill the bucket
        for _ in range(3):
            rl.is_allowed("key1", 3, 1)
        assert rl.is_allowed("key1", 3, 1) is False
        # Wait for window to expire
        time.sleep(1.1)
        assert rl.is_allowed("key1", 3, 1) is True

    def test_cleanup_removes_stale(self):
        rl = RateLimiter()
        # Manually insert an old entry (cleanup checks q[-1] < now - 3600)
        from collections import deque
        rl._hits["old"] = deque([time.time() - 7200])  # 2 hours ago
        rl.cleanup()
        assert "old" not in rl._hits

    def test_thread_safety(self):
        rl = RateLimiter()
        results = []

        def hammer():
            for _ in range(100):
                results.append(rl.is_allowed("shared", 1000, 60))

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 1000
        assert all(r is True for r in results)


# ── SSRF Protection ────────────────────────────────────────────────────────

class TestIsUrlSafe:
    def test_private_ipv4_blocked(self):
        safe, reason = is_url_safe("http://192.168.1.1/")
        assert safe is False
        assert "private" in reason.lower()

    def test_loopback_blocked(self):
        safe, reason = is_url_safe("http://127.0.0.1/")
        assert safe is False

    def test_metadata_blocked(self):
        safe, reason = is_url_safe("http://169.254.169.254/latest/meta-data/")
        assert safe is False
        assert "metadata" in reason.lower() or "private" in reason.lower()

    def test_valid_public_url(self):
        safe, reason = is_url_safe("https://example.com")
        assert safe is True
        assert reason == ""

    def test_no_hostname(self):
        safe, reason = is_url_safe("not-a-url")
        assert safe is False

    def test_ftp_scheme_blocked(self):
        safe, reason = is_url_safe("ftp://example.com/file")
        assert safe is False
        assert "scheme" in reason.lower()

    def test_unresolvable_host(self):
        safe, reason = is_url_safe("http://thisdomaindoesnotexist12345.invalid/")
        assert safe is False
        assert "resolve" in reason.lower()

    def test_10_x_blocked(self):
        safe, _ = is_url_safe("http://10.0.0.1/")
        assert safe is False

    def test_link_local_blocked(self):
        safe, _ = is_url_safe("http://169.254.1.1/")
        assert safe is False


# ── PDF Validation ─────────────────────────────────────────────────────────

class TestIsValidPdf:
    def test_valid_pdf_magic(self):
        assert is_valid_pdf(b"%PDF-1.4 ...") is True

    def test_invalid_magic(self):
        assert is_valid_pdf(b"<html>not a pdf</html>") is False

    def test_empty(self):
        assert is_valid_pdf(b"") is False

    def test_short_valid(self):
        assert is_valid_pdf(b"%PDF-") is True

    def test_almost_valid(self):
        assert is_valid_pdf(b"%PD") is False


# ── IP Resolution ──────────────────────────────────────────────────────────

def _make_request(client_host="1.2.3.4", forwarded_for=None):
    """Create a mock FastAPI Request."""
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = client_host
    headers = {}
    if forwarded_for:
        headers["x-forwarded-for"] = forwarded_for
    req.headers = headers
    return req


class TestGetClientIp:
    def test_direct_ip_no_proxy(self):
        req = _make_request("8.8.8.8")
        assert get_client_ip(req) == "8.8.8.8"

    def test_untrusted_proxy_ignores_forwarded(self):
        req = _make_request("8.8.8.8", "10.0.0.1, 1.1.1.1")
        # 8.8.8.8 is not a trusted proxy, so XFF is ignored
        assert get_client_ip(req) == "8.8.8.8"

    def test_trusted_proxy_uses_forwarded(self):
        req = _make_request("127.0.0.1", "203.0.113.5")
        # 127.0.0.1 is in default TRUSTED_PROXIES
        assert get_client_ip(req) == "203.0.113.5"

    def test_trusted_proxy_chain(self):
        req = _make_request("127.0.0.1", "203.0.113.5, 127.0.0.1")
        # rightmost-untrusted should be 203.0.113.5
        assert get_client_ip(req) == "203.0.113.5"

    def test_no_client(self):
        req = MagicMock()
        req.client = None
        req.headers = {}
        assert get_client_ip(req) == "unknown"


# ── rate_limit_check ───────────────────────────────────────────────────────

class TestRateLimitCheck:
    def test_returns_none_when_allowed(self):
        req = _make_request("1.2.3.4")
        result = rate_limit_check(req, "test_bucket_ok", 100, 60)
        assert result is None

    def test_returns_429_when_exceeded(self):
        req = _make_request("1.2.3.4")
        for _ in range(3):
            rate_limit_check(req, "test_bucket_exceed", 3, 60)
        result = rate_limit_check(req, "test_bucket_exceed", 3, 60)
        assert result is not None
        assert result.status_code == 429


# ── Auth helpers ───────────────────────────────────────────────────────────

class TestVerifyAdmin:
    def test_valid_token(self):
        cfg.admin_tokens["test_token_1"] = time.time() + 3600
        req = MagicMock()
        req.headers = {"x-admin-token": "test_token_1"}
        assert verify_admin(req) is True
        cfg.admin_tokens.pop("test_token_1", None)

    def test_expired_token(self):
        cfg.admin_tokens["expired_1"] = time.time() - 10
        req = MagicMock()
        req.headers = {"x-admin-token": "expired_1"}
        assert verify_admin(req) is False
        assert "expired_1" not in cfg.admin_tokens

    def test_missing_token(self):
        req = MagicMock()
        req.headers = {"x-admin-token": ""}
        assert verify_admin(req) is False

    def test_nonexistent_token(self):
        req = MagicMock()
        req.headers = {"x-admin-token": "doesnt_exist"}
        assert verify_admin(req) is False


class TestVerifyAdminToken:
    def test_valid(self):
        cfg.admin_tokens["raw_test_1"] = time.time() + 3600
        assert verify_admin_token("raw_test_1") is True
        cfg.admin_tokens.pop("raw_test_1", None)

    def test_expired(self):
        cfg.admin_tokens["raw_expired"] = time.time() - 10
        assert verify_admin_token("raw_expired") is False


class TestVerifyInternal:
    def setup_method(self):
        self.token = "int_test_token"
        cfg._internal_tokens[self.token] = {
            "slug": "my-assistant",
            "role": "admin",
            "site_id": 999,
            "assistant_id": 1,
            "created": time.time(),
        }

    def teardown_method(self):
        cfg._internal_tokens.pop(self.token, None)

    def test_valid_admin(self):
        req = MagicMock()
        req.headers = {"x-internal-token": self.token}
        session = verify_internal(req, "my-assistant", require_admin=True)
        assert session is not None
        assert session["role"] == "admin"

    def test_wrong_slug(self):
        req = MagicMock()
        req.headers = {"x-internal-token": self.token}
        assert verify_internal(req, "other-slug") is None

    def test_user_blocked_from_admin(self):
        cfg._internal_tokens[self.token]["role"] = "user"
        req = MagicMock()
        req.headers = {"x-internal-token": self.token}
        assert verify_internal(req, "my-assistant", require_admin=True) is None

    def test_user_allowed_without_admin_requirement(self):
        cfg._internal_tokens[self.token]["role"] = "user"
        req = MagicMock()
        req.headers = {"x-internal-token": self.token}
        session = verify_internal(req, "my-assistant", require_admin=False)
        assert session is not None

    def test_expired_internal_token(self):
        cfg._internal_tokens[self.token]["created"] = time.time() - cfg.INTERNAL_TOKEN_TTL - 10
        req = MagicMock()
        req.headers = {"x-internal-token": self.token}
        assert verify_internal(req, "my-assistant") is None

    def test_missing_token(self):
        req = MagicMock()
        req.headers = {"x-internal-token": ""}
        assert verify_internal(req, "my-assistant") is None
