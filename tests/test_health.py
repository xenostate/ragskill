import asyncio
import json
from unittest.mock import MagicMock

import scripts.config as cfg
import scripts.routes.chat as chat_routes
from scripts.routes.chat import health, liveness


def test_liveness_does_not_call_dependencies():
    payload = asyncio.run(liveness())

    assert payload["status"] == "ok"
    assert "checks" not in payload


def test_health_checks_database_openai_and_embedding_model(monkeypatch):
    monkeypatch.setattr(chat_routes, "_health_cache", None)
    database = MagicMock()
    database.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    openai_client = MagicMock()
    openai_client.models.retrieve.return_value = MagicMock(id=cfg.RAG_MODEL)
    monkeypatch.setattr(cfg, "sb", database)
    monkeypatch.setattr(cfg, "openai_client", openai_client)
    monkeypatch.setattr(cfg, "embed_model", MagicMock())

    response = asyncio.run(health())
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["checks"]["database"]["status"] == "ok"
    assert payload["checks"]["openai"]["status"] == "ok"
    openai_client.models.retrieve.assert_called_once_with(cfg.RAG_MODEL)


def test_health_returns_503_when_database_is_down(monkeypatch):
    monkeypatch.setattr(chat_routes, "_health_cache", None)
    database = MagicMock()
    database.table.return_value.select.return_value.limit.return_value.execute.side_effect = RuntimeError("down")
    monkeypatch.setattr(cfg, "sb", database)
    monkeypatch.setattr(cfg, "openai_client", MagicMock())
    monkeypatch.setattr(cfg, "embed_model", MagicMock())

    response = asyncio.run(health())
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["status"] == "degraded"
    assert payload["checks"]["database"]["error"] == "RuntimeError"
