"""
WhatsApp Business handler for web-rag.

Multi-tenant routing via BSP (360dialog by default):
  - Each business registers a WhatsApp number → mapped to a site_id
  - Customer messages are routed through the BSP webhook
  - Conversation history is persisted in Supabase (survives restarts)

Used by server.py — not run standalone.
"""

import hashlib
import hmac
import logging
import os
import re
import time

import requests as http_requests
from supabase import Client as SupabaseClient

log = logging.getLogger("rag-server.whatsapp")

# BSP API endpoints (provider-specific)
BSP_ENDPOINTS = {
    "360dialog": "https://waba.360dialog.io/v1/messages",
    "meta_cloud": "https://graph.facebook.com/{api_version}/{phone_number_id}/messages",
}

PHONE_HASH_SALT = os.environ.get("PHONE_HASH_SALT", "webrag-v1")
WHATSAPP_MAX_LENGTH = 4096
CONVERSATION_LIMIT = 5  # exchanges (10 messages) to load from DB


class WhatsAppHandler:
    def __init__(self, sb: SupabaseClient, rag_fn, verify_token: str):
        """
        Args:
            sb: Supabase client
            rag_fn: callable(site_id, query, top_k, session_id, language)
                    -> {"answer": str, "sources": [...], "confidence": str}
            verify_token: Token for Meta webhook verification handshake
        """
        self.sb = sb
        self.rag_fn = rag_fn
        self.verify_token = verify_token

    # ── Webhook verification ──────────────────────────────────────────

    def verify_webhook(self, mode: str | None, token: str | None,
                       challenge: str | None) -> str | None:
        """Meta webhook verification (GET challenge-response).

        Returns the challenge string on success, None on failure.
        """
        if mode == "subscribe" and token == self.verify_token and challenge:
            log.info("WhatsApp webhook verified successfully")
            return challenge
        log.warning(f"WhatsApp webhook verification failed: mode={mode}")
        return None

    @staticmethod
    def verify_signature(payload: bytes, signature_header: str,
                         app_secret: str) -> bool:
        """Verify HMAC-SHA256 signature from Meta/BSP.

        Args:
            payload: Raw request body bytes
            signature_header: Value of X-Hub-Signature-256 header
            app_secret: Meta App Secret or BSP app secret
        """
        if not signature_header:
            return False
        # Header format: "sha256=<hex_digest>"
        if not signature_header.startswith("sha256="):
            return False
        expected = signature_header[7:]
        computed = hmac.new(
            app_secret.encode(), payload, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(computed, expected)

    # ── Phone hashing ─────────────────────────────────────────────────

    @staticmethod
    def _hash_phone(phone: str) -> str:
        """Salted SHA-256 hash for customer phone privacy."""
        return hashlib.sha256(f"{PHONE_HASH_SALT}:{phone}".encode()).hexdigest()

    # ── Account resolution ────────────────────────────────────────────

    def _resolve_account(self, business_phone: str) -> dict | None:
        """Look up active WhatsApp account by business phone number.

        Returns full row dict or None.
        """
        try:
            # Normalize: strip all non-digit chars except leading +
            phone = re.sub(r'[^\d+]', '', business_phone.strip())
            if not phone.startswith('+'):
                phone = f'+{phone}'

            resp = (
                self.sb.table("whatsapp_accounts")
                .select("id, site_id, phone_number, api_token, provider, display_name")
                .eq("phone_number", phone)
                .eq("is_active", True)
                .execute()
            )
            if resp.data:
                return resp.data[0]
        except Exception as e:
            log.error(f"Account lookup failed for {business_phone}: {e}")
        return None

    # ── Conversation persistence ──────────────────────────────────────

    def _load_conversation(self, account_id: int,
                           phone_hash: str) -> list[dict]:
        """Load recent conversation history from DB.

        Returns list of {role, content} in chronological order.
        """
        try:
            resp = (
                self.sb.table("whatsapp_conversations")
                .select("role, content")
                .eq("account_id", account_id)
                .eq("customer_phone", phone_hash)
                .order("created_at", desc=True)
                .limit(CONVERSATION_LIMIT * 2)
                .execute()
            )
            if resp.data:
                # Reverse to chronological order (query returns newest first)
                return list(reversed(resp.data))
        except Exception as e:
            log.error(f"Failed to load conversation: account={account_id} phone_hash={phone_hash[:8]}... {e}")
        return []

    def _save_messages(self, account_id: int, phone_hash: str,
                       user_msg: str, assistant_msg: str):
        """Persist user + assistant messages to DB."""
        try:
            self.sb.table("whatsapp_conversations").insert([
                {
                    "account_id": account_id,
                    "customer_phone": phone_hash,
                    "role": "user",
                    "content": user_msg,
                },
                {
                    "account_id": account_id,
                    "customer_phone": phone_hash,
                    "role": "assistant",
                    "content": assistant_msg,
                },
            ]).execute()
        except Exception as e:
            log.error(f"Failed to save conversation: {e}")

    # ── BSP reply API ─────────────────────────────────────────────────

    def _send_reply(self, api_token: str, to_phone: str, text: str,
                    provider: str = "360dialog"):
        """Send a text message via the BSP API (with 1 retry)."""
        if len(text) > WHATSAPP_MAX_LENGTH - 50:
            text = text[: WHATSAPP_MAX_LENGTH - 50] + "\n\n_(truncated)_"

        payload = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "text",
            "text": {"body": text},
        }

        if provider == "360dialog":
            url = BSP_ENDPOINTS["360dialog"]
            headers = {"D360-API-KEY": api_token, "Content-Type": "application/json"}
        else:
            # Meta Cloud API fallback
            url = BSP_ENDPOINTS["meta_cloud"]
            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            }

        for attempt in range(2):
            try:
                resp = http_requests.post(url, json=payload, headers=headers, timeout=10)
                if resp.status_code in (200, 201):
                    return
                log.warning(f"BSP reply failed (attempt {attempt + 1}): "
                            f"status={resp.status_code} body={resp.text[:200]}")
            except Exception as e:
                log.error(f"BSP reply error (attempt {attempt + 1}): {e}")
            if attempt == 0:
                time.sleep(0.5)

    def _send_error_reply(self, api_token: str, to_phone: str,
                          provider: str = "360dialog"):
        """Send a generic error message to the customer."""
        self._send_reply(
            api_token, to_phone,
            "Sorry, I encountered an error processing your message. Please try again.",
            provider,
        )

    # ── Message handlers ──────────────────────────────────────────────

    def handle_message(self, account: dict, sender_phone: str, text: str,
                       session_history: dict, session_lock,
                       get_language_fn):
        """Process a text message through the RAG pipeline.

        Args:
            account: whatsapp_accounts row dict
            sender_phone: customer phone (raw, will be hashed)
            text: message text
            session_history: reference to server._session_history dict
            session_lock: reference to server._session_lock
            get_language_fn: callable(site_id) -> language string
        """
        phone_hash = self._hash_phone(sender_phone)
        account_id = account["id"]
        site_id = account["site_id"]
        api_token = account["api_token"]
        VALID_PROVIDERS = ("360dialog", "meta_cloud")
        provider = account.get("provider", "360dialog")
        if provider not in VALID_PROVIDERS:
            log.warning(f"Unknown provider '{provider}' for account {account_id}, defaulting to 360dialog")
            provider = "360dialog"

        try:
            # 1. Load persistent conversation history from DB
            db_history = self._load_conversation(account_id, phone_hash)

            # 2. Temporarily inject into in-memory session store
            #    so _do_rag_sync reuses existing context logic
            session_key = f"wa_{phone_hash}"
            with session_lock:
                session_history[session_key] = db_history

            # 3. Run RAG pipeline
            language = get_language_fn(site_id)
            result = self.rag_fn(
                site_id, text, 5, session_key, language,
            )
            answer = result["answer"]

            # 4. Clean up in-memory session (DB is source of truth)
            with session_lock:
                session_history.pop(session_key, None)

            # 5. Persist to DB
            self._save_messages(account_id, phone_hash, text, answer)

            # 6. Add confidence prefix for low confidence
            conf = result.get("confidence", "medium")
            if conf == "low":
                answer = ("⚠️ I'm not very confident about this answer — "
                          "the sources may not fully cover this topic.\n\n" + answer)

            # 7. Send reply
            self._send_reply(api_token, sender_phone, answer, provider)

            log.info(f"WhatsApp reply sent: account={account_id} "
                     f"confidence={conf} len={len(answer)}")

        except Exception as e:
            log.error(f"WhatsApp RAG error: account={account_id} phone_hash={phone_hash[:8]}... {e}")
            # Clean up on error too
            with session_lock:
                session_history.pop(f"wa_{phone_hash}", None)
            self._send_error_reply(api_token, sender_phone, provider)

    def handle_unsupported(self, account: dict, sender_phone: str,
                           msg_type: str):
        """Reply to non-text message types (image, video, audio, etc.)."""
        self._send_reply(
            account["api_token"], sender_phone,
            f"I can only process text messages at this time. "
            f"Please send your question as text.",
            account.get("provider", "360dialog"),
        )
        log.info(f"Unsupported message type '{msg_type}' from {self._hash_phone(sender_phone)[:8]}...")

    # ── Main dispatcher ───────────────────────────────────────────────

    def handle_webhook(self, payload: dict, session_history: dict,
                       session_lock, get_language_fn):
        """Process an incoming WhatsApp webhook payload.

        Meta webhook format:
        {
            "entry": [{
                "changes": [{
                    "value": {
                        "metadata": {"display_phone_number": "...", "phone_number_id": "..."},
                        "messages": [{"from": "...", "type": "text", "text": {"body": "..."}}],
                        "statuses": [...]  // delivery receipts — ignored
                    }
                }]
            }]
        }
        """
        try:
            entries = payload.get("entry", [])
            for entry in entries:
                for change in entry.get("changes", []):
                    value = change.get("value", {})

                    # Skip status updates (delivery receipts, read receipts)
                    if "statuses" in value and "messages" not in value:
                        continue

                    messages = value.get("messages", [])
                    if not messages:
                        continue

                    # Identify the business account
                    metadata = value.get("metadata", {})
                    business_phone = metadata.get("display_phone_number", "")
                    if not business_phone:
                        log.warning("Webhook missing display_phone_number in metadata")
                        continue

                    account = self._resolve_account(business_phone)
                    if not account:
                        log.warning(f"No active account for business phone {business_phone}")
                        continue

                    # Process each message
                    for msg in messages:
                        sender = msg.get("from", "")
                        msg_type = msg.get("type", "")

                        if not sender:
                            continue

                        if msg_type == "text":
                            body = msg.get("text", {}).get("body", "").strip()
                            if body:
                                self.handle_message(
                                    account, sender, body,
                                    session_history, session_lock,
                                    get_language_fn,
                                )
                        else:
                            self.handle_unsupported(account, sender, msg_type)

        except Exception as e:
            log.error(f"WhatsApp webhook dispatch error: {e}", exc_info=True)

    # ── Admin: register account ───────────────────────────────────────

    def register_account(self, site_id: int, phone_number: str,
                         display_name: str, api_token: str,
                         provider: str = "360dialog") -> dict:
        """Register or update a WhatsApp business account.

        Returns the upserted row.
        """
        # Normalize: strip all non-digit chars except leading +
        phone = re.sub(r'[^\d+]', '', phone_number.strip())
        if not phone.startswith('+'):
            phone = f'+{phone}'

        resp = self.sb.table("whatsapp_accounts").upsert(
            {
                "site_id": site_id,
                "phone_number": phone,
                "display_name": display_name,
                "api_token": api_token,
                "provider": provider,
                "is_active": True,
            },
            on_conflict="phone_number",
        ).execute()

        log.info(f"WhatsApp account registered: phone={phone} site={site_id}")
        return resp.data[0] if resp.data else {}
