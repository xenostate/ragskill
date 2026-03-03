"""
Telegram bot handler for web-rag.

Multi-tenant routing:
  - /start site_<ID>  → binds this chat to a site
  - /site <ID>        → re-binds to a different site
  - /help             → shows commands
  - any message       → RAG answer from the bound site

Used by server.py — not run standalone.
"""

import logging
import os

import requests as http_requests
from supabase import Client as SupabaseClient

log = logging.getLogger("rag-server.telegram")

TELEGRAM_API = "https://api.telegram.org/bot{token}"


class TelegramHandler:
    def __init__(self, token: str, sb: SupabaseClient, rag_fn):
        """
        Args:
            token: Telegram bot token
            sb: Supabase client
            rag_fn: callable(site_id, query) -> {"answer": str, "sources": [...], "confidence": str}
        """
        self.token = token
        self.api = TELEGRAM_API.format(token=token)
        self.sb = sb
        self.rag_fn = rag_fn

    # ── Telegram API helpers ───────────────────────────────────────────

    def send_message(self, chat_id: int, text: str, parse_mode: str = "Markdown"):
        """Send a message via Telegram API."""
        # Truncate if too long (Telegram limit is 4096 chars)
        if len(text) > 4000:
            text = text[:4000] + "\n\n_(truncated)_"

        try:
            resp = http_requests.post(
                f"{self.api}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                # Fallback: send without parse_mode if Markdown fails
                http_requests.post(
                    f"{self.api}/sendMessage",
                    json={"chat_id": chat_id, "text": text},
                    timeout=10,
                )
        except Exception as e:
            log.error(f"Failed to send message to {chat_id}: {e}")

    def send_typing(self, chat_id: int):
        """Show typing indicator."""
        try:
            http_requests.post(
                f"{self.api}/sendChatAction",
                json={"chat_id": chat_id, "action": "typing"},
                timeout=5,
            )
        except Exception:
            pass

    # ── Binding logic ──────────────────────────────────────────────────

    def get_binding(self, chat_id: int) -> int | None:
        """Look up which site_id this chat is bound to."""
        resp = (
            self.sb.table("telegram_bindings")
            .select("site_id")
            .eq("chat_id", chat_id)
            .execute()
        )
        if resp.data:
            return resp.data[0]["site_id"]
        return None

    def set_binding(self, chat_id: int, site_id: int, username: str | None = None):
        """Bind a chat to a site (upsert)."""
        self.sb.table("telegram_bindings").upsert({
            "chat_id": chat_id,
            "site_id": site_id,
            "username": username,
        }).execute()

    def get_site_name(self, site_id: int) -> str | None:
        """Get domain name for a site_id."""
        resp = self.sb.table("sites").select("domain").eq("id", site_id).execute()
        if resp.data:
            return resp.data[0]["domain"]
        return None

    # ── Command handlers ───────────────────────────────────────────────

    def handle_start(self, chat_id: int, args: str, username: str | None):
        """Handle /start [site_<ID>]"""
        # Parse site_id from deep link: /start site_3
        site_id = None
        if args and args.startswith("site_"):
            try:
                site_id = int(args.replace("site_", ""))
            except ValueError:
                pass

        if site_id:
            domain = self.get_site_name(site_id)
            if domain:
                self.set_binding(chat_id, site_id, username)
                self.send_message(
                    chat_id,
                    f"👋 Welcome! I'm connected to *{domain}*.\n\n"
                    f"Ask me anything about this site and I'll answer based on its content.\n\n"
                    f"Commands:\n"
                    f"/site `<id>` — switch to a different site\n"
                    f"/help — show help",
                )
            else:
                self.send_message(chat_id, f"❌ Site #{site_id} not found.")
        else:
            self.send_message(
                chat_id,
                "👋 Hi! I'm a website assistant bot.\n\n"
                "To get started, ask the site owner for a link like:\n"
                "`t.me/xenostatebot?start=site_2`\n\n"
                "Or if you know the site ID, use: /site `<id>`",
            )

    def handle_site(self, chat_id: int, args: str, username: str | None):
        """Handle /site <ID>"""
        if not args:
            self.send_message(chat_id, "Usage: /site `<id>`\n\nExample: /site 2")
            return

        try:
            site_id = int(args.strip())
        except ValueError:
            self.send_message(chat_id, "❌ Invalid site ID. Use a number, e.g. /site 2")
            return

        domain = self.get_site_name(site_id)
        if domain:
            self.set_binding(chat_id, site_id, username)
            self.send_message(chat_id, f"✅ Switched to *{domain}* (site #{site_id})")
        else:
            self.send_message(chat_id, f"❌ Site #{site_id} not found.")

    def handle_help(self, chat_id: int):
        """Handle /help"""
        self.send_message(
            chat_id,
            "🕸️ *Web-RAG Assistant*\n\n"
            "I answer questions using content from indexed websites.\n\n"
            "*Commands:*\n"
            "/site `<id>` — switch to a different site\n"
            "/help — show this message\n\n"
            "Just type your question and I'll search the site for an answer!",
        )

    # ── Message handler ────────────────────────────────────────────────

    def handle_message(self, chat_id: int, text: str):
        """Handle a regular text message — RAG answer."""
        site_id = self.get_binding(chat_id)
        if site_id is None:
            self.send_message(
                chat_id,
                "⚠️ No site connected.\n\n"
                "Ask the site owner for a start link, or use /site `<id>` to connect.",
            )
            return

        self.send_typing(chat_id)

        try:
            result = self.rag_fn(site_id, text)
            answer = result["answer"]

            # Append confidence indicator
            conf = result["confidence"]
            if conf == "high":
                prefix = ""
            elif conf == "medium":
                prefix = ""
            else:
                prefix = "⚠️ _Low confidence — the sources may not fully cover this topic._\n\n"

            self.send_message(chat_id, prefix + answer)

        except Exception as e:
            log.error(f"RAG error for chat {chat_id}: {e}")
            self.send_message(chat_id, "❌ Sorry, something went wrong. Please try again.")

    # ── Main dispatcher ────────────────────────────────────────────────

    def handle_update(self, update: dict):
        """Process a Telegram Update object."""
        message = update.get("message")
        if not message:
            return

        chat_id = message["chat"]["id"]
        text = message.get("text", "").strip()
        username = message.get("from", {}).get("username")

        if not text:
            return

        # Route commands
        if text.startswith("/start"):
            args = text[7:].strip()  # everything after "/start "
            self.handle_start(chat_id, args, username)
        elif text.startswith("/site"):
            args = text[6:].strip()  # everything after "/site "
            self.handle_site(chat_id, args, username)
        elif text.startswith("/help"):
            self.handle_help(chat_id)
        else:
            self.handle_message(chat_id, text)
