from types import SimpleNamespace
from unittest.mock import MagicMock

import scripts.config as cfg
import scripts.rag_core as rag_core


def test_streaming_rag_yields_deltas_and_the_regular_result(monkeypatch):
    monkeypatch.setattr(rag_core, "retrieve_chunks", lambda *_args, **_kwargs: {
        "confidence": "high",
        "results": [{
            "title": "About",
            "url": "https://example.com/about",
            "score": 0.92,
            "chunk_text": "Example source text.",
        }],
    })
    create = MagicMock(return_value=[
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="world"))]),
    ])
    monkeypatch.setattr(
        cfg,
        "openai_client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )

    events = list(rag_core.do_rag_stream_sync(1, "What is this?", 5))

    assert [event["text"] for event in events[:-1]] == ["Hello ", "world"]
    assert events[-1] == {
        "type": "done",
        "answer": "Hello world",
        "sources": [{
            "title": "About",
            "url": "https://example.com/about",
            "score": 0.92,
        }],
        "confidence": "high",
    }
    assert create.call_args.kwargs["stream"] is True


def test_streaming_rag_uses_intent_fallback_without_calling_llm(monkeypatch):
    monkeypatch.setattr(rag_core, "retrieve_chunks", lambda *_args, **_kwargs: {
        "confidence": "low",
        "results": [],
    })
    create = MagicMock()
    monkeypatch.setattr(
        cfg,
        "openai_client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )

    events = list(rag_core.do_rag_stream_sync(
        1,
        "Contact me",
        5,
        low_confidence_answer="Choose an action.",
    ))

    assert events[0] == {"type": "delta", "text": "Choose an action."}
    assert events[-1]["answer"] == "Choose an action."
    create.assert_not_called()
