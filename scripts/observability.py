"""Structured logging, request correlation, and error-report sanitizing."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any


request_id_var: ContextVar[str] = ContextVar("request_id", default="")

_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "email",
    "ip",
    "openai_api_key",
    "password",
    "phone",
    "secret",
    "service_key",
    "session_id",
    "token",
}
_KEY_VALUE_RE = re.compile(
    r"(?i)\b(password|passwd|secret|token|api[_-]?key|authorization|cookie|email|phone|ip)"
    r"(\s*[=:]\s*|\s+)([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_IPV4_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_IPV6_RE = re.compile(r"(?i)(?<![\w:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![\w:])")
_URL_CREDENTIAL_RE = re.compile(r"(?i)(https?://)[^/@\s]+@")

_STANDARD_LOG_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__)
_STANDARD_LOG_RECORD_FIELDS.update({"message", "asctime"})


def _secret_values() -> list[str]:
    values = []
    for key, value in os.environ.items():
        normalized = key.lower()
        if any(part in normalized for part in ("key", "token", "password", "secret", "dsn")):
            if len(value) >= 8:
                values.append(value)
    return sorted(values, key=len, reverse=True)


def redact_text(value: str) -> str:
    """Remove common credentials and personal identifiers from log text."""
    text = value
    for secret in _secret_values():
        text = text.replace(secret, "[REDACTED]")
    text = _KEY_VALUE_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", text)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _OPENAI_KEY_RE.sub("[REDACTED]", text)
    text = _JWT_RE.sub("[REDACTED]", text)
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _IPV4_RE.sub("[REDACTED_IP]", text)
    text = _IPV6_RE.sub("[REDACTED_IP]", text)
    text = _URL_CREDENTIAL_RE.sub(r"\1[REDACTED]@", text)
    return text


def sanitize(value: Any, key: str = "") -> Any:
    """Recursively sanitize structured fields before logs or Sentry leave the app."""
    normalized_key = key.lower().replace("-", "_")
    if normalized_key in _SENSITIVE_KEYS or any(
        part in normalized_key for part in ("password", "secret", "token", "authorization", "cookie")
    ):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(item, key) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize(item, key) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact_text(str(value))


class SensitiveDataFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_text(str(record.msg))
        if record.args:
            record.args = sanitize(record.args)
        for key, value in list(record.__dict__.items()):
            if key not in _STANDARD_LOG_RECORD_FIELDS:
                record.__dict__[key] = sanitize(value, key)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id

        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_FIELDS and not key.startswith("_"):
                payload[key] = sanitize(value, key)

        if record.exc_info:
            payload["exception"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> logging.Logger:
    """Configure one stdout handler for machine-readable production logs."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = os.environ.get("LOG_FORMAT", "json").lower()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(SensitiveDataFilter())
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers = [handler]
        uvicorn_logger.propagate = False
        uvicorn_logger.setLevel(level)
    return logging.getLogger("rag-server")


def sentry_before_send(event: dict, _hint: dict) -> dict:
    """Final privacy boundary for error events sent to Sentry."""
    return sanitize(event)
