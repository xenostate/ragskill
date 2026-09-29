import json
import logging

from scripts.observability import JsonFormatter, SensitiveDataFilter, redact_text, sanitize


def test_redact_text_removes_common_secrets_and_pii():
    value = (
        "email=person@example.com ip=203.0.113.8 peer=::1 "
        "authorization=Bearer abc.def.ghi token=super-secret-token "
        "phone=+77001234567 sk-example123456789"
    )

    result = redact_text(value)

    assert "person@example.com" not in result
    assert "203.0.113.8" not in result
    assert "::1" not in result
    assert "super-secret-token" not in result
    assert "+77001234567" not in result
    assert "sk-example123456789" not in result


def test_sanitize_redacts_sensitive_structured_fields():
    result = sanitize({
        "site_id": 7,
        "password": "dont-log-me",
        "nested": {"api_key": "also-secret", "status": "ok"},
    })

    assert result == {
        "site_id": 7,
        "password": "[REDACTED]",
        "nested": {"api_key": "[REDACTED]", "status": "ok"},
    }


def test_json_formatter_emits_structured_sanitized_log():
    record = logging.LogRecord(
        "test",
        logging.INFO,
        __file__,
        1,
        "login email=user@example.com",
        (),
        None,
    )
    record.event = "auth.login"
    record.token = "secret-token"
    SensitiveDataFilter().filter(record)

    payload = json.loads(JsonFormatter().format(record))

    assert payload["event"] == "auth.login"
    assert payload["token"] == "[REDACTED]"
    assert "user@example.com" not in payload["message"]
