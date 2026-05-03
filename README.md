# Web-RAG

Multi-tenant website assistant that crawls sites, embeds content, and answers questions using RAG (Retrieval-Augmented Generation) with citations.

Works as an **embeddable chat widget** on any website, a **private knowledge base** for businesses (Internal Assistants), and as a **Telegram bot** — all powered by the same backend.

The widget now supports a tenant-configurable assistant layer on top of RAG:

- Custom greeting message
- Custom widget title and input placeholder
- Quick action buttons
- In-chat structured forms
- Form submission storage in Supabase
- Optional Telegram / email / WhatsApp notifications per form

## Architecture

```
                  ┌─────────────┐
                  │  Supabase   │
                  │  pgvector   │
                  └──────┬──────┘
                         │
  ┌──────────┐    ┌──────┴──────┐    ┌──────────────┐
  │ Indexer   │───▶│  FastAPI    │◀───│ Chat Widget  │
  │ (crawl)  │    │  Server     │    │ (Shadow DOM) │
  └──────────┘    └──────┬──────┘    └──────────────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
       ┌──────┴──┐ ┌────┴─────┐ ┌──┴──────┐
       │Internal │ │ Telegram │ │WhatsApp │
       │Assistant│ │ Webhook  │ │ Webhook │
       └─────────┘ └──────────┘ └─────────┘
```

**Pipeline:** Crawl → Clean HTML → Chunk (500 words, overlapping) → Embed (`intfloat/multilingual-e5-base`, 768-dim) → Store in Supabase → Hybrid retrieval (70% cosine + 30% tsvector) → LLM answer with citations

## Prerequisites

- **Python 3.11+**
- **Supabase** account (free tier works) — [supabase.com](https://supabase.com)
- **OpenAI API key** — for LLM answer generation (uses `gpt-4o-mini`, ~$0.001/query)
- **Telegram Bot Token** (optional) — used for Telegram notifications and legacy bot support

## Quick Start

### 1. Clone and set up Python environment

```bash
git clone https://github.com/xenostate/ragskill.git
cd ragskill

# Create virtual environment
python3 -m venv .venv

# Install dependencies
.venv/bin/pip install -r scripts/requirements.txt

# For JS-rendered sites (SPAs), also install Playwright:
.venv/bin/pip install playwright
.venv/bin/playwright install chromium
```

### 2. Create Supabase project

1. Go to [supabase.com](https://supabase.com) → **New Project**
2. Pick a name, set a database password, choose a region
3. Once created, go to **Settings → API** and copy:
   - **Project URL** (e.g. `https://xxxxx.supabase.co`)
   - **Service role key** (the `service_role` one, NOT `anon`)
4. Go to **SQL Editor** → paste and run everything in `references/schema.sql`

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJ...your-service-role-key
EMBED_MODEL=intfloat/multilingual-e5-base
OPENAI_API_KEY=sk-proj-...your-openai-key
TELEGRAM_BOT_TOKEN=123456:ABC...your-bot-token    # optional
SMTP_HOST=smtp.example.com                        # optional
SMTP_PORT=587                                     # optional
SMTP_USERNAME=alerts@example.com                  # optional
SMTP_PASSWORD=your-password                       # optional
SMTP_FROM=alerts@example.com                      # optional
SMTP_USE_TLS=true                                 # optional
```

### 4. Add a site to index

In the Supabase **SQL Editor**:

```sql
INSERT INTO sites (domain, language) VALUES ('example.com', 'en');
-- Note the returned ID (e.g. 1)
```

### 5. Crawl and index the site

```bash
# For normal HTML sites:
.venv/bin/python3 scripts/indexer.py --site-id 1 --max-pages 50

# For JS-rendered SPAs (React, Svelte, Next.js, etc.):
.venv/bin/python3 scripts/indexer.py --site-id 1 --max-pages 50 --renderer playwright
```

### 6. Start the server

```bash
./start.sh
```

Server runs at `http://localhost:8090`. Test it:

```bash
curl -s http://localhost:8090/health | python3 -m json.tool

curl -s -X POST http://localhost:8090/api/chat \
  -H "Content-Type: application/json" \
  -d '{"site_id": 1, "query": "What does this company do?"}' | python3 -m json.tool
```

## Commands Cheatsheet

### Indexing (crawl a website)

```bash
.venv/bin/python3 scripts/indexer.py \
  --site-id <ID> \
  --max-pages <N> \
  --renderer <static|playwright> \
  --start-url <URL>              # optional: override crawl entry point
```

| Flag | Default | Description |
|------|---------|-------------|
| `--site-id` | required | Site ID from `sites` table |
| `--max-pages` | 100 | Max pages to BFS-crawl |
| `--renderer` | `static` | `static` = requests (fast), `playwright` = headless Chrome (for SPAs) |
| `--start-url` | `https://{domain}` | Override the starting URL |

### Retrieval (search chunks)

```bash
.venv/bin/python3 scripts/retriever.py \
  --site-id <ID> \
  --query "your question" \
  --top-k 5
```

Returns JSON with confidence level (`high` / `medium` / `low`) and ranked chunks.

### RAG Answer (retrieve + LLM)

```bash
# Full answer with citations:
.venv/bin/python3 scripts/rag.py --site-id <ID> --query "your question"

# Context only (no LLM call):
.venv/bin/python3 scripts/rag.py --site-id <ID> --query "your question" --context-only
```

### Server

```bash
./start.sh              # Start on default port 8090
./start.sh 9000         # Start on custom port
```

### Telegram Bot Setup

1. Create a bot via [@BotFather](https://t.me/BotFather)
2. Add the token to `.env` as `TELEGRAM_BOT_TOKEN`
3. Start the server, then register the webhook:

```bash
curl -X POST http://localhost:8090/api/telegram/set-webhook \
  -H "Content-Type: application/json" \
  -d '{"url": "https://YOUR-PUBLIC-URL/api/telegram"}'
```

4. Users open: `t.me/YOUR_BOT?start=site_<ID>` to bind to a site
5. Then they just type questions

**Bot commands:**
- `/start site_<ID>` — bind chat to a site
- `/site <ID>` — switch to a different site
- `/help` — show help

## Assistant Configuration

Each site can store an `assistant` JSON object inside `sites.settings`. This adds configurable widget behavior on top of RAG without changing code per client.

Supported features:

- `display` — widget title and input placeholder
- `greeting` — first assistant message shown when the widget opens
- `starters` — quick action buttons
- `forms` — structured in-chat forms
- `destinations` — where form submissions should be forwarded

The assistant config is managed from the Admin page for each site:

1. Open `/admin`
2. Open a site from **All Sites**
3. Edit **Assistant Configuration**
4. Save the JSON
5. Use the **Live Widget Preview** to test greeting, starters, forms, and normal chat

### Assistant Config Shape

```json
{
  "display": {
    "title": "Assistant title",
    "input_placeholder": "Type your question"
  },
  "greeting": {
    "enabled": true,
    "message": "Hello! How can I help?",
    "show_once": true,
    "delay_ms": 700
  },
  "starters": [
    {
      "id": "pricing",
      "label": "Pricing",
      "action": "send_message",
      "message": "Tell me about pricing"
    },
    {
      "id": "contact",
      "label": "Leave my details",
      "action": "open_form",
      "form_id": "contact_form"
    }
  ],
  "forms": [
    {
      "id": "contact_form",
      "title": "Contact request",
      "description": "Leave your details and our team will follow up.",
      "submit_label": "Send",
      "success_message": "Thanks! Your request has been sent.",
      "fields": [
        {
          "name": "name",
          "label": "Name",
          "type": "text",
          "required": true,
          "placeholder": "Your name"
        },
        {
          "name": "phone",
          "label": "Phone",
          "type": "tel",
          "required": true,
          "placeholder": "+82 10 1234 5678"
        }
      ],
      "destinations": {
        "email": ["sales@example.com"],
        "telegram_chat_ids": ["123456789"],
        "whatsapp_numbers": ["+821012345678"]
      }
    }
  ]
}
```

### Starter Actions

- `send_message` — sends a predefined message into the normal `/api/chat` RAG flow
- `open_form` — opens a configured form inside the widget

### Supported Form Field Types

- `text`
- `tel`
- `email`
- `number`
- `textarea`
- `select`

For `select`, provide an `options` array.

### Form Submission Flow

1. Visitor opens the widget
2. Greeting appears if enabled
3. Starter buttons appear if configured
4. User clicks a starter or types a normal message
5. `send_message` starters go to `/api/chat`
6. `open_form` starters render a structured form in the widget
7. Form submission is validated, stored in `assistant_form_submissions`, and optionally forwarded to configured destinations

### Telegram / Email / WhatsApp Notifications

Form notifications are configured per form, but the server credentials live in `.env`.

Current Telegram behavior:

- `TELEGRAM_BOT_TOKEN` is global for the server
- Multiple clients can use the same Telegram bot
- Each form can route to different `telegram_chat_ids`

Current WhatsApp behavior:

- WhatsApp notifications require a registered active WhatsApp account for that site
- Delivery uses the site-linked WhatsApp account plus the `whatsapp_numbers` configured on the form

If no destinations are configured, submissions are still stored in Supabase.

### Viewing Stored Form Submissions

Run this in Supabase SQL Editor:

```sql
select *
from assistant_form_submissions
order by created_at desc
limit 50;
```

### Internal Assistant (Private Knowledge Base)

Internal Assistants let businesses create password-protected knowledge bases. Employees or clients access them via a dedicated URL — no widget embedding needed.

**How it works:**

1. An admin creates the assistant by providing a setup code, name, URL slug, and two passwords (admin + user)
2. The assistant gets its own page at `https://YOUR-SERVER/assistant/{slug}`
3. **Admin role** — can add knowledge (text, PDFs, crawl URLs), manage documents and chunks
4. **User role** — can only chat with the assistant

**Setup (via API):**

```bash
curl -X POST https://YOUR-SERVER/api/internal/setup \
  -H "Content-Type: application/json" \
  -d '{
    "code": "YOUR_INTERNAL_SETUP_CODE",
    "name": "Company KB",
    "slug": "company-kb",
    "admin_password": "admin-pass-here",
    "user_password": "user-pass-here",
    "language": "en",
    "email": "admin@company.com"
  }'
```

Returns the assistant URL (e.g. `https://YOUR-SERVER/assistant/company-kb`).

**Adding knowledge (admin only):**

```bash
# Add plain text
curl -X POST https://YOUR-SERVER/api/internal/company-kb/add-text \
  -H "Content-Type: application/json" \
  -H "X-Internal-Token: YOUR_TOKEN" \
  -d '{"title": "Onboarding Guide", "text": "Full text content here..."}'

# Upload a PDF
curl -X POST https://YOUR-SERVER/api/internal/company-kb/upload-pdf \
  -H "X-Internal-Token: YOUR_TOKEN" \
  -F "pdf=@handbook.pdf"

# Crawl a website
curl -X POST https://YOUR-SERVER/api/internal/company-kb/crawl \
  -H "Content-Type: application/json" \
  -H "X-Internal-Token: YOUR_TOKEN" \
  -d '{"url": "https://docs.company.com", "max_pages": 50}'
```

**Environment variable:** Set `INTERNAL_SETUP_CODE` in `.env` to enable assistant creation.

### Embed Widget on Any Website

```html
<script
  src="https://YOUR-SERVER-URL/widget.js"
  data-site-id="1"
  data-title="Ask a question"
  data-color="#2563eb"
  data-position="right"
></script>
```

| Attribute | Default | Description |
|-----------|---------|-------------|
| `data-site-id` | `1` | Which indexed site to query |
| `data-api` | same origin | API server URL (set if widget is on a different domain) |
| `data-title` | "Ask a question" | Chat panel header text |
| `data-color` | `#2563eb` | Accent color (hex) |
| `data-position` | `right` | Bubble position: `right` or `left` |

If the site has an assistant config saved, the embed automatically supports:

- Greeting message
- Quick action buttons
- In-chat forms
- Form submission to your backend
- Telegram / email / WhatsApp notifications (if configured)

Important:

- The widget only works on the domain registered in `sites.domain` (or its subdomains)
- The widget fetches assistant config from the server on load
- Forms submit to `/api/widget/forms/submit`

## Project Structure

```
.
├── .env.example             # Template for environment variables
├── start.sh                 # One-command server launcher
├── scripts/
│   ├── requirements.txt     # Python dependencies
│   ├── server.py            # FastAPI entry point (lifespan, middleware)
│   ├── config.py            # Shared config, constants, mutable globals
│   ├── assistant_features.py # Assistant config, forms, notification routing
│   ├── utils.py             # Rate limiter, auth, SSRF protection, helpers
│   ├── rag_core.py          # Retrieval, context building, answer generation
│   ├── indexer.py           # Crawl → clean → chunk → embed → store
│   ├── retriever.py         # Semantic search (vector + keyword hybrid)
│   ├── rag.py               # CLI RAG answerer (retrieve + LLM)
│   ├── whatsapp_handler.py  # WhatsApp Business webhook handler
│   ├── telegram_handler.py  # Telegram bot multi-tenant routing
│   └── routes/
│       ├── chat.py          # /api/chat, /health, /widget.js
│       ├── trial.py         # /trial, /api/trial/* (free trial flow)
│       ├── admin.py         # /admin, /api/admin/* (site management)
│       ├── auth.py          # /api/admin/auth, /api/quick-activate
│       ├── internal.py      # /assistant/*, /api/internal/* (private KBs)
│       └── whatsapp.py      # /api/whatsapp/* (webhook endpoints)
├── tests/
│   ├── conftest.py          # Test config (mocks heavy deps)
│   ├── test_assistant_features.py # Assistant config + form handling tests
│   ├── test_utils.py        # 39 tests: rate limiter, auth, SSRF, IP
│   └── test_indexer.py      # 27 tests: HTML cleaning, chunking, links
├── widget/
│   ├── widget.js            # Embeddable chat widget (Shadow DOM)
│   ├── widget.html          # Local test page for the widget
│   ├── trial.html           # Free trial page
│   ├── admin.html           # Admin dashboard
│   └── assistant.html       # Internal assistant page
└── references/
    ├── schema.sql           # Supabase database schema (run once)
    └── korean_school_assistant.ru.json # Example tenant assistant config
```

## Multi-Tenant Architecture

One server instance serves multiple websites and assistants:

1. Each website/assistant is a row in the `sites` table with a unique `site_id`
2. The indexer crawls and stores chunks per `site_id`
3. The widget passes `data-site-id` with every request
4. Internal assistants get their own `site_id` on creation — knowledge is fully isolated
5. All retrieval and RAG is scoped to the bound `site_id`

Assistant behavior is also tenant-scoped:

6. Each site can have its own `settings.assistant`
7. Greetings, quick actions, forms, and destinations are isolated per `site_id`
8. One shared Telegram bot token can still notify different chats for different clients

**Adding a new site:**
```sql
INSERT INTO sites (domain, language) VALUES ('newsite.com', 'en');
-- Returns ID, e.g. 3
```
Then crawl it:
```bash
.venv/bin/python3 scripts/indexer.py --site-id 3 --max-pages 100
```

## Deployment (Production)

For always-on hosting without running on your local machine:

### VPS (Recommended — $4-6/mo)

1. Get a VPS (Hetzner, DigitalOcean, etc.)
2. Clone repo, create `.venv`, install deps
3. Set up as a systemd service:

```ini
# /etc/systemd/system/web-rag.service
[Unit]
Description=Web-RAG Server
After=network.target

[Service]
WorkingDirectory=/opt/web-rag
ExecStart=/opt/web-rag/.venv/bin/uvicorn scripts.server:app --host 0.0.0.0 --port 8090
Restart=always
EnvironmentFile=/opt/web-rag/.env

[Install]
WantedBy=multi-user.target
```

4. Reverse proxy with Caddy or Nginx for HTTPS
5. Register Telegram webhook once with the permanent domain

### Docker

```bash
docker build -t web-rag .
docker run -d --env-file .env -p 8090:8090 web-rag
```

### Running Tests

```bash
python3 -m pytest tests/ -v
```

Tests cover rate limiting, auth, assistant config normalization, form validation, SSRF protection, HTML cleaning, chunking, and link extraction.

### Quick Tunnel (Development)

For temporary public access from your machine:

```bash
# Install: brew install cloudflare/cloudflare/cloudflared
cloudflared tunnel --url http://localhost:8090
```

## Tuning

| Env Var | Default | Description |
|---------|---------|-------------|
| `EMBED_MODEL` | `intfloat/multilingual-e5-base` | Sentence-transformers model name |
| `RAG_MODEL` | `gpt-4o-mini` | OpenAI model for answer generation |
| `RAG_TOP_K` | `5` | Number of chunks to retrieve per query |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap words between chunks |
| `REQUEST_DELAY` | `0.5` | Seconds between HTTP requests when crawling |
| `THRESHOLD_HIGH` | `0.75` | Score threshold for "high" confidence |
| `THRESHOLD_MEDIUM` | `0.5` | Score threshold for "medium" confidence |
| `SUPABASE_ANON_KEY` | *(none)* | Supabase anon key for public endpoints (enables RLS) |
| `TELEGRAM_BOT_TOKEN` | *(none)* | Telegram bot used for notifications and legacy bot mode |
| `SMTP_HOST` | *(none)* | SMTP server for form notification emails |
| `SMTP_PORT` | `587` | SMTP port |
| `SMTP_USERNAME` | *(none)* | SMTP username |
| `SMTP_PASSWORD` | *(none)* | SMTP password |
| `SMTP_FROM` | *(none)* | Sender address for form notification emails |
| `SMTP_USE_TLS` | `true` | Enable STARTTLS for SMTP |
| `INTERNAL_SETUP_CODE` | `ACTIVATION_CODE` | Code required to create internal assistants |
| `TRUSTED_PROXIES` | `127.0.0.1,::1` | Trusted reverse proxy IPs/CIDRs for X-Forwarded-For |
| `THREAD_POOL_SIZE` | `4` | Max concurrent embedding/PDF threads |
| `WHATSAPP_ENABLED` | `false` | Enable WhatsApp Business webhook |

## Cost

- **Supabase**: Free tier (500MB database, 2GB storage)
- **Embedding**: Free — runs locally via `sentence-transformers` (~1GB RAM)
- **LLM answers**: OpenAI `gpt-4o-mini` — ~$0.001 per query
- **Hosting**: $0 (your machine) to ~$5/mo (VPS)

## License

MIT
