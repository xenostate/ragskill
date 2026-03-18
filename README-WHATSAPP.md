# WhatsApp Business Integration — Activation Guide

The WhatsApp infrastructure is fully built but **dormant**. Follow these steps to activate it.

---

## 1. Choose a BSP (Business Solution Provider)

You need a BSP to connect to WhatsApp Business API. Recommended:

| BSP | Pricing | Setup time | Notes |
|-----|---------|------------|-------|
| **360dialog** (recommended) | ~$0.005/msg + Meta fees | 1-2 days | Cheapest, simple API |
| **Twilio** | ~$0.005-0.01/msg + Meta fees | Same day | Well-documented, higher cost |
| **Gupshup** | ~$0.004/msg + Meta fees | 1-2 days | Good for high volume |

Meta conversation fees: ~$0.02-0.08 per conversation depending on country.

### What you need from the BSP:
- A **verified phone number** for WhatsApp Business
- An **API token** (for sending messages)
- An **App Secret** (for webhook signature verification)

---

## 2. Run the Database Migration

In Supabase SQL Editor, run the contents of:

```
references/whatsapp_migration.sql
```

This creates two tables:
- `whatsapp_accounts` — maps business phone numbers to sites
- `whatsapp_conversations` — persistent chat history

---

## 3. Set Environment Variables

Add to your `.env` on the server:

```env
WHATSAPP_ENABLED=true
WHATSAPP_VERIFY_TOKEN=any-random-string-you-choose
WHATSAPP_APP_SECRET=your-app-secret-from-bsp
```

- `WHATSAPP_VERIFY_TOKEN` — you make this up; it's used during webhook registration to prove you own the endpoint
- `WHATSAPP_APP_SECRET` — from your BSP/Meta dashboard; used to verify incoming webhook signatures

---

## 4. Restart the Server

```bash
systemctl restart webrag
```

Check logs to confirm:
```bash
journalctl -u webrag -n 20
```

You should see: `WhatsApp handler initialized`

---

## 5. Register Webhook URL with BSP

Point your BSP's webhook configuration to:

```
https://your-domain/api/whatsapp/webhook
```

- **Verification method**: GET with `hub.mode`, `hub.verify_token`, `hub.challenge`
- **Message delivery**: POST with signed JSON payloads
- **Subscribe to**: `messages` events

For 360dialog: do this in the 360dialog Partner Hub dashboard.

---

## 6. Register a Business Number

Call the admin endpoint to link a WhatsApp number to a site:

```bash
curl -X POST https://your-domain/api/whatsapp/register \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: YOUR_ADMIN_TOKEN" \
  -d '{
    "site_id": 11,
    "phone_number": "+77001234567",
    "display_name": "My Business",
    "api_token": "your-bsp-api-token",
    "provider": "360dialog"
  }'
```

You can register multiple numbers for different sites (multi-tenant).

---

## 7. Test It

1. Send a WhatsApp message to the registered business number
2. Check server logs: `journalctl -u webrag -f`
3. You should see the message being processed and a reply sent back

---

## Architecture Overview

```
Customer WhatsApp → BSP (360dialog) → POST /api/whatsapp/webhook
                                          ↓
                                    Verify signature
                                    Parse message
                                    Resolve business phone → site_id
                                    Load conversation history (Supabase)
                                    Run RAG pipeline (_do_rag_sync)
                                    Save to whatsapp_conversations
                                    Send reply via BSP API
                                          ↓
Customer WhatsApp ← BSP (360dialog) ← Reply
```

### Key files:
- `scripts/whatsapp_handler.py` — handler class (parsing, BSP API, conversation persistence)
- `scripts/server.py` — endpoints (`/api/whatsapp/webhook`, `/api/whatsapp/register`)
- `references/whatsapp_migration.sql` — database tables

### Multi-tenant:
Each business gets their own row in `whatsapp_accounts` with their own phone number, API token, and linked `site_id`. All messages route through the same webhook — the handler resolves which business account to use based on the phone number in the webhook metadata.

### Conversation memory:
Chat history is stored in `whatsapp_conversations` (Supabase). The last 5 exchanges are loaded before each RAG call, giving the LLM context for follow-up questions. Old conversations (30+ days) can be pruned with:

```sql
DELETE FROM whatsapp_conversations WHERE created_at < now() - interval '30 days';
```

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| Endpoints return 503 | `WHATSAPP_ENABLED=true` in .env? Server restarted? |
| Webhook verification fails | `WHATSAPP_VERIFY_TOKEN` matches what you configured in BSP dashboard? |
| Messages received but no reply | Check logs for BSP API errors. Is `api_token` correct? Is account `is_active = true`? |
| Wrong site answers | Check `whatsapp_accounts.site_id` matches the correct site |
| Signature verification fails | `WHATSAPP_APP_SECRET` matches BSP app secret? |
