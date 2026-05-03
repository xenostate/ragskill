"""
Reusable assistant configuration + workflow helpers.

This layer sits above RAG so each tenant can configure greetings,
predefined actions, and lead-capture forms without code changes.
"""

from __future__ import annotations

import copy
import json
import re
import smtplib
from email.message import EmailMessage

import requests as http_requests

import scripts.config as cfg

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SAFE_ID_RE = re.compile(r"[^a-z0-9_]+")
DEFAULT_TEXT_FALLBACK_LANG = "ru"

ASSISTANT_CONFIG_TEMPLATE = {
    "display": {
        "title": "",
        "input_placeholder": "",
    },
    "language_switch": {
        "enabled": False,
        "default": "",
        "options": [
            {"code": "ru", "label": "RU"},
            {"code": "en", "label": "EN"},
        ],
    },
    "greeting": {
        "enabled": False,
        "message": "",
        "show_once": True,
        "delay_ms": 0,
    },
    "starters": [
        {
            "id": "example_info",
            "label": "Ask about the program",
            "action": "send_message",
            "message": "Can you tell me about your main program and how it works?",
        },
        {
            "id": "example_form",
            "label": "Leave my details",
            "action": "open_form",
            "form_id": "contact_form",
        },
    ],
    "forms": [
        {
            "id": "contact_form",
            "title": "Contact request",
            "description": "Leave your details and our team can follow up.",
            "submit_label": "Send",
            "success_message": "Thanks. Your request has been sent.",
            "fields": [
                {
                    "name": "name",
                    "label": "Name",
                    "type": "text",
                    "required": True,
                    "placeholder": "Your name",
                },
                {
                    "name": "phone",
                    "label": "Phone",
                    "type": "tel",
                    "required": True,
                    "placeholder": "+82 10 1234 5678",
                },
                {
                    "name": "notes",
                    "label": "Notes",
                    "type": "textarea",
                    "required": False,
                    "placeholder": "What would you like help with?",
                },
            ],
            "destinations": {
                "email": [],
                "telegram_chat_ids": [],
                "whatsapp_numbers": [],
            },
        }
    ],
}


def assistant_config_template() -> dict:
    """Return a deep copy of the starter template."""
    return copy.deepcopy(ASSISTANT_CONFIG_TEMPLATE)


def _clean_str(value, max_len: int = 500) -> str:
    if value is None:
        return ""
    return str(value).strip()[:max_len]


def _safe_id(value: str, fallback: str) -> str:
    candidate = SAFE_ID_RE.sub("_", _clean_str(value, 80).lower()).strip("_")
    return candidate or fallback


def _coerce_options(options) -> list[str]:
    if not isinstance(options, list):
        return []
    cleaned = []
    for item in options:
        text = _clean_str(item, 120)
        if text:
            cleaned.append(text)
    return cleaned[:20]


def _clean_text_value(value, max_len: int = 500):
    """Allow either a plain string or a {lang: text} mapping."""
    if isinstance(value, dict):
        cleaned = {}
        for lang, text in value.items():
            code = _safe_id(lang, "")
            label = _clean_str(text, max_len)
            if code and label:
                cleaned[code] = label
        return cleaned or ""
    return _clean_str(value, max_len)


def resolve_text_value(value, lang: str | None = None, fallback: str = DEFAULT_TEXT_FALLBACK_LANG) -> str:
    """Resolve a string or localized mapping into one display string."""
    if isinstance(value, dict):
        preferred = _safe_id(lang or "", "")
        fallback_code = _safe_id(fallback, "")
        if preferred and value.get(preferred):
            return value[preferred]
        if fallback_code and value.get(fallback_code):
            return value[fallback_code]
        if preferred and "_" in preferred:
            base = preferred.split("_", 1)[0]
            if value.get(base):
                return value[base]
        return next((str(text).strip() for text in value.values() if str(text).strip()), "")
    return _clean_str(value, 2000)


def _normalize_field_options(options) -> list[dict]:
    if not isinstance(options, list):
        return []
    normalized = []
    for idx, item in enumerate(options):
        if isinstance(item, dict) and ("value" in item or "label" in item):
            value = _clean_str(item.get("value"), 120)
            label = _clean_text_value(item.get("label"), 120)
        else:
            label = _clean_text_value(item, 120)
            value = resolve_text_value(label)[:120]
        if not value:
            value = f"option_{idx + 1}"
        if not resolve_text_value(label):
            label = value
        normalized.append({
            "value": value,
            "label": label,
        })
    return normalized[:20]


def _normalize_language_options(options) -> list[dict]:
    if not isinstance(options, list):
        return []
    normalized = []
    seen = set()
    for item in options:
        if isinstance(item, dict):
            code = _safe_id(item.get("code"), "")
            label = _clean_str(item.get("label"), 12) or code.upper()
        else:
            code = _safe_id(item, "")
            label = code.upper()
        if not code or code in seen:
            continue
        seen.add(code)
        normalized.append({"code": code, "label": label})
    return normalized[:8]


def normalize_assistant_config(raw: dict | None) -> dict:
    """Normalize tenant assistant config into a safe, predictable shape."""
    raw = raw or {}
    display = raw.get("display") if isinstance(raw.get("display"), dict) else {}
    language_switch = raw.get("language_switch") if isinstance(raw.get("language_switch"), dict) else {}
    greeting = raw.get("greeting") if isinstance(raw.get("greeting"), dict) else {}
    language_options = _normalize_language_options(language_switch.get("options"))

    config = {
        "version": 1,
        "display": {
            "title": _clean_text_value(display.get("title"), 80),
            "input_placeholder": _clean_text_value(display.get("input_placeholder"), 120),
        },
        "language_switch": {
            "enabled": bool(language_switch.get("enabled", False)),
            "default": _safe_id(language_switch.get("default"), ""),
            "options": language_options,
        },
        "greeting": {
            "enabled": bool(greeting.get("enabled", False)),
            "message": _clean_text_value(greeting.get("message"), 1200),
            "show_once": bool(greeting.get("show_once", True)),
            "delay_ms": max(0, min(int(greeting.get("delay_ms", 0) or 0), 10000)),
        },
        "starters": [],
        "forms": [],
    }

    if config["language_switch"]["default"]:
        valid_codes = {item["code"] for item in config["language_switch"]["options"]}
        if config["language_switch"]["default"] not in valid_codes:
            config["language_switch"]["default"] = ""

    starters = raw.get("starters") if isinstance(raw.get("starters"), list) else []
    for idx, item in enumerate(starters):
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        if action not in ("send_message", "open_form"):
            action = "send_message"
        label = _clean_text_value(item.get("label"), 80)
        if not resolve_text_value(label):
            continue
        config["starters"].append({
            "id": _safe_id(item.get("id") or resolve_text_value(label), f"starter_{idx + 1}"),
            "label": label,
            "action": action,
            "message": _clean_text_value(item.get("message"), 1200),
            "form_id": _safe_id(item.get("form_id"), "") if item.get("form_id") else "",
        })

    forms = raw.get("forms") if isinstance(raw.get("forms"), list) else []
    for idx, item in enumerate(forms):
        if not isinstance(item, dict):
            continue
        form_id = _safe_id(item.get("id"), f"form_{idx + 1}")
        title = _clean_text_value(item.get("title"), 120)
        if not resolve_text_value(title):
            title = form_id.replace("_", " ").title()
        description = _clean_text_value(item.get("description"), 500)
        submit_label = _clean_text_value(item.get("submit_label"), 40)
        if not resolve_text_value(submit_label):
            submit_label = "Submit"
        success_message = _clean_text_value(item.get("success_message"), 240)
        if not resolve_text_value(success_message):
            success_message = "Thanks. Your request has been received."
        fields = []
        for field_idx, field in enumerate(item.get("fields") or []):
            if not isinstance(field, dict):
                continue
            field_name = _safe_id(field.get("name"), f"field_{field_idx + 1}")
            field_type = _clean_str(field.get("type"), 20).lower() or "text"
            if field_type not in ("text", "textarea", "email", "tel", "number", "select"):
                field_type = "text"
            label = _clean_text_value(field.get("label"), 80)
            if not resolve_text_value(label):
                label = field_name.replace("_", " ").title()
            fields.append({
                "name": field_name,
                "label": label,
                "type": field_type,
                "required": bool(field.get("required", False)),
                "placeholder": _clean_text_value(field.get("placeholder"), 120),
                "options": _normalize_field_options(field.get("options")),
            })
        destinations = item.get("destinations") if isinstance(item.get("destinations"), dict) else {}
        config["forms"].append({
            "id": form_id,
            "title": title,
            "description": description,
            "submit_label": submit_label,
            "success_message": success_message,
            "fields": fields,
            "destinations": {
                "email": _coerce_options(destinations.get("email")),
                "telegram_chat_ids": _coerce_options(destinations.get("telegram_chat_ids")),
                "whatsapp_numbers": _coerce_options(destinations.get("whatsapp_numbers")),
            },
        })

    return config


def get_assistant_config(site_settings: dict | None) -> dict:
    settings = site_settings or {}
    return normalize_assistant_config(settings.get("assistant"))


def get_public_assistant_config(site_settings: dict | None) -> dict:
    config = get_assistant_config(site_settings)
    public_forms = []
    for form in config["forms"]:
        public_forms.append({
            "id": form["id"],
            "title": form["title"],
            "description": form["description"],
            "submit_label": form["submit_label"],
            "success_message": form["success_message"],
            "fields": form["fields"],
        })
    return {
        "version": config["version"],
        "display": config["display"],
        "language_switch": config["language_switch"],
        "greeting": config["greeting"],
        "starters": config["starters"],
        "forms": public_forms,
    }


def validate_form_submission(assistant_config: dict, form_id: str, values: dict | None) -> tuple[dict | None, dict, dict]:
    """Return (form, cleaned_values, field_errors)."""
    values = values or {}
    form = next((f for f in assistant_config.get("forms", []) if f["id"] == form_id), None)
    if form is None:
        return None, {}, {"form": "Unknown form."}

    cleaned: dict[str, str] = {}
    errors: dict[str, str] = {}
    for field in form.get("fields", []):
        raw_value = values.get(field["name"], "")
        if isinstance(raw_value, (dict, list)):
            raw_value = json.dumps(raw_value, ensure_ascii=False)
        value = _clean_str(raw_value, 2000)
        if field.get("required") and not value:
            errors[field["name"]] = "This field is required."
            continue
        if value and field["type"] == "email" and not EMAIL_RE.match(value):
            errors[field["name"]] = "Please enter a valid email."
            continue
        if value and field["type"] == "select":
            options = field.get("options") or []
            option_values = {opt.get("value") for opt in options if isinstance(opt, dict)}
            if option_values and value not in option_values:
                errors[field["name"]] = "Please choose one of the available options."
                continue
        cleaned[field["name"]] = value
    return form, cleaned, errors


def _build_notification_text(site_id: int, site_domain: str, form: dict, payload: dict,
                             page_url: str | None, language: str | None = None) -> str:
    lines = [
        "New assistant form submission",
        "",
        f"Site: {site_domain} (#{site_id})",
        f"Form: {resolve_text_value(form['title'], language)} ({form['id']})",
    ]
    if page_url:
        lines.append(f"Page: {page_url}")
    lines.append("")
    lines.append("Fields:")
    for field in form.get("fields", []):
        name = field["name"]
        label = resolve_text_value(field["label"], language)
        value = payload.get(name, "")
        lines.append(f"- {label}: {value or '(empty)'}")
    return "\n".join(lines)


def _send_email_notifications(recipients: list[str], subject: str, body: str) -> dict:
    if not recipients:
        return {"configured": 0, "sent": 0}
    if not cfg.SMTP_HOST or not cfg.SMTP_FROM:
        return {
            "configured": len(recipients),
            "sent": 0,
            "error": "SMTP is not configured on the server.",
        }

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.SMTP_FROM
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    try:
        with smtplib.SMTP(cfg.SMTP_HOST, cfg.SMTP_PORT, timeout=10) as smtp:
            if cfg.SMTP_USE_TLS:
                smtp.starttls()
            if cfg.SMTP_USERNAME:
                smtp.login(cfg.SMTP_USERNAME, cfg.SMTP_PASSWORD)
            smtp.send_message(msg)
        return {"configured": len(recipients), "sent": len(recipients)}
    except Exception as e:
        cfg.log.error(f"Email notification failed: {e}")
        return {"configured": len(recipients), "sent": 0, "error": str(e)}


def _send_telegram_notifications(chat_ids: list[str], text: str) -> dict:
    if not chat_ids:
        return {"configured": 0, "sent": 0}
    if not cfg.TELEGRAM_BOT_TOKEN:
        return {
            "configured": len(chat_ids),
            "sent": 0,
            "error": "TELEGRAM_BOT_TOKEN is not configured on the server.",
        }

    sent = 0
    last_error = None
    api = f"https://api.telegram.org/bot{cfg.TELEGRAM_BOT_TOKEN}/sendMessage"
    for chat_id in chat_ids:
        try:
            resp = http_requests.post(
                api,
                json={"chat_id": chat_id, "text": text},
                timeout=10,
            )
            if resp.status_code == 200:
                sent += 1
            else:
                last_error = f"status {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_error = str(e)
    result = {"configured": len(chat_ids), "sent": sent}
    if last_error:
        result["error"] = last_error
    return result


def _send_whatsapp_notifications(site_id: int, numbers: list[str], text: str) -> dict:
    if not numbers:
        return {"configured": 0, "sent": 0}
    if cfg.whatsapp_handler is None:
        return {
            "configured": len(numbers),
            "sent": 0,
            "error": "WhatsApp is not enabled on the server.",
        }

    try:
        resp = (
            cfg.sb.table("whatsapp_accounts")
            .select("api_token, provider")
            .eq("site_id", site_id)
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
    except Exception as e:
        cfg.log.error(f"WhatsApp account lookup failed: {e}")
        return {"configured": len(numbers), "sent": 0, "error": str(e)}

    if not resp.data:
        return {
            "configured": len(numbers),
            "sent": 0,
            "error": "No active WhatsApp account is registered for this site.",
        }

    account = resp.data[0]
    sent = 0
    last_error = None
    for number in numbers:
        try:
            cfg.whatsapp_handler._send_reply(  # noqa: SLF001 - internal helper is reused intentionally
                account["api_token"],
                number,
                text,
                account.get("provider", "360dialog"),
            )
            sent += 1
        except Exception as e:
            last_error = str(e)
    result = {"configured": len(numbers), "sent": sent}
    if last_error:
        result["error"] = last_error
    return result


def submit_assistant_form(site_id: int, site_domain: str, site_settings: dict | None,
                          session_id: str | None, form_id: str, values: dict | None,
                          page_url: str | None, user_agent: str | None,
                          response_language: str | None = None) -> dict:
    """Validate, store, and notify configured destinations for a form submission."""
    assistant_config = get_assistant_config(site_settings)
    form, cleaned, errors = validate_form_submission(assistant_config, form_id, values)
    if form is None:
        return {"error": "Unknown form.", "errors": errors, "status_code": 404}
    if errors:
        return {"error": "Please correct the highlighted fields.", "errors": errors, "status_code": 400}

    row = {
        "site_id": site_id,
        "session_id": _clean_str(session_id, 128),
        "form_id": form["id"],
        "form_title": resolve_text_value(form["title"], response_language),
        "payload": cleaned,
        "page_url": _clean_str(page_url, 2000),
        "user_agent": _clean_str(user_agent, 500),
        "delivery_status": {},
    }
    insert_resp = cfg.sb.table("assistant_form_submissions").insert(row).execute()
    submission_id = insert_resp.data[0]["id"] if insert_resp.data else None

    notification_text = _build_notification_text(site_id, site_domain, form, cleaned, page_url, response_language)
    subject = f"New assistant lead from {site_domain}"
    destinations = form.get("destinations") or {}
    delivery_status = {
        "email": _send_email_notifications(destinations.get("email") or [], subject, notification_text),
        "telegram": _send_telegram_notifications(destinations.get("telegram_chat_ids") or [], notification_text),
        "whatsapp": _send_whatsapp_notifications(site_id, destinations.get("whatsapp_numbers") or [], notification_text),
    }

    if submission_id is not None:
        try:
            cfg.sb.table("assistant_form_submissions").update({
                "delivery_status": delivery_status,
            }).eq("id", submission_id).execute()
        except Exception as e:
            cfg.log.error(f"Failed to update form submission delivery status: {e}")

    return {
        "success": True,
        "message": resolve_text_value(form["success_message"], response_language),
        "submission_id": submission_id,
        "delivery_status": delivery_status,
    }
