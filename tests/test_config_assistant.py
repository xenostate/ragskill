import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import scripts.config as cfg
from scripts.config_assistant import ConfigAssistantError, draft_assistant_config


def _client_with_content(payload: dict | str):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def test_draft_updates_requested_behavior_and_preserves_unrelated_config(monkeypatch):
    client = _client_with_content({
        "message": "I made the assistant warmer and more detailed. Review and save it.",
        "config": {
            "behavior": {
                "tone": "warm",
                "answer_length": "detailed",
                "instructions": "Explain unfamiliar terms.",
            }
        },
    })
    monkeypatch.setattr(cfg, "openai_client", client)
    monkeypatch.setattr(cfg, "ASSISTANT_CONFIG_MODEL", "test-config-model")
    current = {
        "display": {"title": "Acme Assistant"},
        "forms": [{
            "id": "lead",
            "title": "Contact",
            "fields": [],
            "destinations": {"email": ["sales@example.com"]},
        }],
    }

    result = draft_assistant_config(current, "Make it warmer and more detailed")

    assert result["config"]["behavior"]["tone"] == "warm"
    assert result["config"]["behavior"]["answer_length"] == "detailed"
    assert result["config"]["display"]["title"] == "Acme Assistant"
    assert result["config"]["forms"][0]["destinations"]["email"] == ["sales@example.com"]
    assert result["model"] == "test-config-model"
    call = client.chat.completions.create.call_args.kwargs
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"][0]["role"] == "system"
    assert "does not add factual knowledge" in call["messages"][0]["content"]


def test_draft_rejects_empty_or_invalid_model_output(monkeypatch):
    monkeypatch.setattr(cfg, "openai_client", _client_with_content("not json"))
    with pytest.raises(ConfigAssistantError, match="invalid JSON"):
        draft_assistant_config({}, "Make it friendly")

    with pytest.raises(ConfigAssistantError, match="Describe"):
        draft_assistant_config({}, "   ")


def test_draft_requires_configured_llm(monkeypatch):
    monkeypatch.setattr(cfg, "openai_client", None)
    with pytest.raises(ConfigAssistantError, match="not configured"):
        draft_assistant_config({}, "Make it concise")


def test_draft_passes_requested_ui_language_to_model(monkeypatch):
    client = _client_with_content({"config": {}})
    monkeypatch.setattr(cfg, "openai_client", client)
    monkeypatch.setattr(cfg, "ASSISTANT_CONFIG_MODEL", "test-config-model")

    result = draft_assistant_config({}, "Сделай стиль дружелюбнее", "ru")

    call = client.chat.completions.create.call_args.kwargs
    payload = json.loads(call["messages"][1]["content"])
    assert payload["response_language"] == "Russian"
    assert result["message"].startswith("Черновик обновлён")
