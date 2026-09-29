import asyncio
from unittest.mock import MagicMock

import scripts.config as cfg
from scripts.routes.chat import _log_query


def test_chat_log_stores_answer_identity_timing_sources_and_session(monkeypatch):
    execute = MagicMock()
    execute.execute.return_value = MagicMock(data=[{"id": 1}])
    table = MagicMock()
    table.insert.return_value = execute
    sb = MagicMock()
    sb.table.return_value = table
    monkeypatch.setattr(cfg, "sb", sb)

    asyncio.run(_log_query(
        site_id=4,
        query="What are your hours?",
        confidence="high",
        response_ms=321,
        chunk_count=2,
        interaction_id="msg_abc",
        session_id="session-1",
        answer="We are open from 9 to 5.",
        sources=[{"title": "Hours", "url": "https://example.com/hours", "score": 0.91}],
    ))

    payload = table.insert.call_args.args[0]
    assert payload["site_id"] == 4
    assert payload["interaction_id"] == "msg_abc"
    assert payload["session_id"] == "session-1"
    assert payload["answer"] == "We are open from 9 to 5."
    assert payload["response_time_ms"] == 321
    assert payload["status"] == "ok"
    assert payload["sources"][0]["score"] == 0.91


def test_chat_log_stores_failures(monkeypatch):
    execute = MagicMock()
    execute.execute.return_value = MagicMock(data=[{"id": 2}])
    table = MagicMock()
    table.insert.return_value = execute
    sb = MagicMock()
    sb.table.return_value = table
    monkeypatch.setattr(cfg, "sb", sb)

    asyncio.run(_log_query(
        site_id=4,
        query="Will this fail?",
        confidence="low",
        response_ms=900,
        chunk_count=0,
        interaction_id="msg_error",
        status="error",
        error_code="TimeoutError",
        error_message="request timed out",
    ))

    payload = table.insert.call_args.args[0]
    assert payload["status"] == "error"
    assert payload["error_code"] == "TimeoutError"
    assert payload["error_message"] == "request timed out"
