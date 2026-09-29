import asyncio
from unittest.mock import MagicMock

from starlette.requests import Request

import scripts.config as cfg
import scripts.routes.chat as chat_routes
from scripts.routes.chat import ChatRequest, _log_query


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


def test_streaming_chat_emits_incremental_and_final_events(monkeypatch):
    monkeypatch.setattr(chat_routes, "rate_limit_check", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        chat_routes,
        "_authorize_site_request",
        lambda *_args, **_kwargs: ({"settings": {}}, None),
    )
    monkeypatch.setattr(chat_routes, "get_site_language_cached", lambda _site_id: "en")
    monkeypatch.setattr(chat_routes, "get_assistant_config", lambda _settings: {"behavior": {}})
    monkeypatch.setattr(
        chat_routes,
        "match_intent_actions",
        lambda *_args, **_kwargs: {"actions": [], "response_message": None},
    )

    def fake_stream(*_args, **_kwargs):
        yield {"type": "delta", "text": "Hello"}
        yield {
            "type": "done",
            "answer": "Hello",
            "sources": [],
            "confidence": "high",
        }

    async def ignore_log(*_args, **_kwargs):
        return None

    monkeypatch.setattr(chat_routes, "do_rag_stream_sync", fake_stream)
    monkeypatch.setattr(chat_routes, "_log_query", ignore_log)

    async def exercise():
        request = Request({
            "type": "http",
            "method": "POST",
            "path": "/api/chat/stream",
            "query_string": b"",
            "headers": [],
            "scheme": "https",
            "server": ("testserver", 443),
            "client": ("127.0.0.1", 1234),
        })
        response = await chat_routes.chat_stream(
            ChatRequest(site_id=1, query="Hi", session_id="session-1"),
            request,
        )
        chunks = [chunk async for chunk in response.body_iterator]
        await asyncio.sleep(0)
        return response, "".join(chunks)

    response, body = asyncio.run(exercise())

    assert response.media_type == "text/event-stream"
    assert response.headers["x-accel-buffering"] == "no"
    assert '"type": "delta", "text": "Hello"' in body
    assert '"type": "done", "answer": "Hello"' in body
    assert '"message_id": "msg_' in body
