"""
Tests for assistant configuration normalization and form handling.
"""

from unittest.mock import MagicMock, patch

import scripts.config as cfg

from scripts.assistant_features import (
    assistant_config_template,
    get_public_assistant_config,
    match_intent_actions,
    normalize_assistant_config,
    resolve_text_value,
    submit_assistant_form,
    validate_form_submission,
)


class TestAssistantConfig:
    def test_normalize_config_keeps_supported_shapes(self):
        raw = {
            "display": {"title": "School Bot", "input_placeholder": "Ask here"},
            "greeting": {"enabled": True, "message": "Hello", "delay_ms": 500},
            "starters": [
                {"label": "Pricing", "action": "send_message", "message": "Tell me about pricing"},
                {"label": "Contact", "action": "open_form", "form_id": "lead-form"},
            ],
            "forms": [
                {
                    "id": "lead-form",
                    "title": "Lead form",
                    "fields": [{"name": "email", "label": "Email", "type": "email", "required": True}],
                    "destinations": {"email": ["sales@example.com"]},
                }
            ],
        }

        config = normalize_assistant_config(raw)
        assert config["version"] == 1
        assert config["display"]["title"] == "School Bot"
        assert config["greeting"]["enabled"] is True
        assert config["starters"][0]["action"] == "send_message"
        assert config["starters"][1]["form_id"] == "lead_form"
        assert config["forms"][0]["destinations"]["email"] == ["sales@example.com"]

    def test_public_config_hides_destinations(self):
        settings = {
            "assistant": {
                "forms": [
                    {
                        "id": "contact",
                        "title": "Contact",
                        "fields": [],
                        "destinations": {"email": ["private@example.com"]},
                    }
                ]
            }
        }
        public = get_public_assistant_config(settings)
        assert "destinations" not in public["forms"][0]

    def test_template_is_reusable_copy(self):
        template = assistant_config_template()
        template["greeting"]["enabled"] = True
        template2 = assistant_config_template()
        assert template2["greeting"]["enabled"] is False

    def test_multilingual_text_values_are_preserved(self):
        config = normalize_assistant_config({
            "display": {
                "title": {"ru": "Связаться с нами", "en": "Contact us"},
            },
            "forms": [
                {
                    "id": "lead",
                    "title": {"ru": "Форма", "en": "Form"},
                    "fields": [
                        {
                            "name": "level",
                            "label": {"ru": "Уровень", "en": "Level"},
                            "type": "select",
                            "options": [
                                {"ru": "Начальный", "en": "Beginner"},
                                {"value": "advanced", "label": {"ru": "Продвинутый", "en": "Advanced"}},
                            ],
                        }
                    ],
                }
            ],
        })

        assert config["display"]["title"]["ru"] == "Связаться с нами"
        assert resolve_text_value(config["display"]["title"], "en") == "Contact us"
        assert config["forms"][0]["fields"][0]["options"][0]["label"]["en"] == "Beginner"
        assert config["forms"][0]["fields"][0]["options"][1]["value"] == "advanced"

    def test_intent_rules_are_normalized_and_match_actions(self):
        config = normalize_assistant_config({
            "intent_rules": [
                {
                    "id": "lead_capture",
                    "match": {
                        "keywords": ["оставить заявку", "связаться"]
                    },
                    "response_message": {"ru": "Оставьте заявку ниже", "en": "Leave a request below"},
                    "actions": [
                        {
                            "type": "open_form",
                            "label": {"ru": "Оставить заявку", "en": "Leave a request"},
                            "form_id": "contact_form",
                        },
                        {
                            "type": "send_message",
                            "label": {"ru": "Узнать стоимость", "en": "Ask about pricing"},
                            "message": {"ru": "Расскажите о стоимости", "en": "Tell me about pricing"},
                        },
                    ],
                }
            ]
        })

        assert config["intent_rules"][0]["match"]["keywords"] == ["оставить заявку", "связаться"]
        assert config["intent_rules"][0]["actions"][0]["form_id"] == "contact_form"

        matched = match_intent_actions(config, "Хочу оставить заявку на обучение")
        assert matched["matched_rule_ids"] == ["lead_capture"]
        assert len(matched["actions"]) == 2
        assert matched["actions"][0]["type"] == "open_form"
        assert matched["response_message"]["ru"] == "Оставьте заявку ниже"


class TestAssistantForms:
    def test_validate_form_submission_required_and_email(self):
        config = normalize_assistant_config({
            "forms": [
                {
                    "id": "lead",
                    "title": "Lead",
                    "fields": [
                        {"name": "name", "label": "Name", "type": "text", "required": True},
                        {"name": "email", "label": "Email", "type": "email", "required": True},
                    ],
                }
            ]
        })
        form, cleaned, errors = validate_form_submission(config, "lead", {"name": "", "email": "bad"})
        assert form is not None
        assert cleaned == {}
        assert "name" in errors
        assert "email" in errors

    def test_validate_form_submission_accepts_normalized_select_values(self):
        config = normalize_assistant_config({
            "forms": [
                {
                    "id": "lead",
                    "title": {"ru": "Заявка", "en": "Lead"},
                    "fields": [
                        {
                            "name": "level",
                            "label": {"ru": "Уровень", "en": "Level"},
                            "type": "select",
                            "required": True,
                            "options": [
                                {"value": "beginner", "label": {"ru": "Начальный", "en": "Beginner"}},
                            ],
                        },
                    ],
                }
            ]
        })
        form, cleaned, errors = validate_form_submission(config, "lead", {"level": "beginner"})
        assert form is not None
        assert cleaned["level"] == "beginner"
        assert errors == {}

    @patch("scripts.assistant_features._send_email_notifications", return_value={"configured": 1, "sent": 1})
    @patch("scripts.assistant_features._send_telegram_notifications", return_value={"configured": 0, "sent": 0})
    @patch("scripts.assistant_features._send_whatsapp_notifications", return_value={"configured": 0, "sent": 0})
    def test_submit_assistant_form_stores_and_notifies(self, _wa, _tg, _email):
        insert_execute = MagicMock()
        insert_execute.execute.return_value = MagicMock(data=[{"id": 77}])
        update_execute = MagicMock()
        update_execute.eq.return_value.execute.return_value = MagicMock(data=[{"id": 77}])

        table_mock = MagicMock()
        table_mock.insert.return_value = insert_execute
        table_mock.update.return_value = update_execute
        cfg.sb = MagicMock()
        cfg.sb.table.return_value = table_mock

        settings = {
            "assistant": {
                "forms": [
                    {
                        "id": "contact_form",
                        "title": "Contact",
                        "success_message": "Sent",
                        "fields": [
                            {"name": "name", "label": "Name", "type": "text", "required": True},
                        ],
                        "destinations": {"email": ["sales@example.com"]},
                    }
                ]
            }
        }

        result = submit_assistant_form(
            5,
            "example.com",
            settings,
            "session-1",
            "contact_form",
            {"name": "Jane"},
            "https://example.com/contact",
            "Mozilla/5.0",
        )

        assert result["success"] is True
        assert result["submission_id"] == 77
        cfg.sb.table.assert_called_with("assistant_form_submissions")

    def test_submit_assistant_form_returns_validation_error(self):
        settings = {
            "assistant": {
                "forms": [
                    {
                        "id": "contact_form",
                        "title": "Contact",
                        "fields": [
                            {"name": "phone", "label": "Phone", "type": "tel", "required": True},
                        ],
                    }
                ]
            }
        }
        result = submit_assistant_form(
            5,
            "example.com",
            settings,
            "session-1",
            "contact_form",
            {"phone": ""},
            None,
            None,
        )
        assert result["status_code"] == 400
        assert "phone" in result["errors"]

    @patch("scripts.assistant_features._send_email_notifications", return_value={"configured": 0, "sent": 0})
    @patch("scripts.assistant_features._send_telegram_notifications", return_value={"configured": 0, "sent": 0})
    @patch("scripts.assistant_features._send_whatsapp_notifications", return_value={"configured": 0, "sent": 0})
    def test_submit_assistant_form_resolves_multilingual_success_message(self, _wa, _tg, _email):
        insert_execute = MagicMock()
        insert_execute.execute.return_value = MagicMock(data=[{"id": 88}])
        update_execute = MagicMock()
        update_execute.eq.return_value.execute.return_value = MagicMock(data=[{"id": 88}])

        table_mock = MagicMock()
        table_mock.insert.return_value = insert_execute
        table_mock.update.return_value = update_execute
        cfg.sb = MagicMock()
        cfg.sb.table.return_value = table_mock

        settings = {
            "assistant": {
                "forms": [
                    {
                        "id": "contact_form",
                        "title": {"ru": "Контакты", "en": "Contact"},
                        "success_message": {"ru": "Отправлено", "en": "Sent"},
                        "fields": [
                            {"name": "name", "label": {"ru": "Имя", "en": "Name"}, "type": "text", "required": True},
                        ],
                    }
                ]
            }
        }
        result = submit_assistant_form(
            5,
            "example.com",
            settings,
            "session-1",
            "contact_form",
            {"name": "Jane"},
            None,
            None,
            "en",
        )
        assert result["message"] == "Sent"
