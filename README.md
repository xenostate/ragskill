# Web-RAG

Multi-tenant website assistant that crawls sites, embeds content, and answers questions using RAG (Retrieval-Augmented Generation) with citations.

Works as an **embeddable chat widget** on any website, a **private knowledge base** for businesses (Internal Assistants), and as a **Telegram bot** — all powered by the same backend.

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
- **Telegram Bot Token** (optional, currently disabled) — from [@BotFather](https://t.me/BotFather)

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

## Project Structure

```
.
├── .env.example             # Template for environment variables
├── start.sh                 # One-command server launcher
├── scripts/
│   ├── requirements.txt     # Python dependencies
│   ├── server.py            # FastAPI entry point (lifespan, middleware)
│   ├── config.py            # Shared config, constants, mutable globals
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
│   ├── test_utils.py        # 39 tests: rate limiter, auth, SSRF, IP
│   └── test_indexer.py      # 27 tests: HTML cleaning, chunking, links
├── widget/
│   ├── widget.js            # Embeddable chat widget (Shadow DOM)
│   ├── widget.html          # Local test page for the widget
│   ├── trial.html           # Free trial page
│   ├── admin.html           # Admin dashboard
│   └── assistant.html       # Internal assistant page
└── references/
    └── schema.sql           # Supabase database schema (run once)
```

## Multi-Tenant Architecture

One server instance serves multiple websites and assistants:

1. Each website/assistant is a row in the `sites` table with a unique `site_id`
2. The indexer crawls and stores chunks per `site_id`
3. The widget passes `data-site-id` with every request
4. Internal assistants get their own `site_id` on creation — knowledge is fully isolated
5. All retrieval and RAG is scoped to the bound `site_id`

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

66 tests covering rate limiting, auth, SSRF protection, HTML cleaning, chunking, and link extraction.

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
