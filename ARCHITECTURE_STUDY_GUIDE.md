# RAGSkill Architecture Study Guide

This note explains the project in a way that is easier to review from a phone.

## 1. What this project is

`ragskill` is a multi-tenant RAG assistant platform.

It can serve:

- an embeddable website chat widget
- a trial/demo onboarding flow
- an admin dashboard
- private internal assistants
- optional WhatsApp support

All of those product surfaces share the same backend and the same knowledge pipeline.

Core idea:

`content -> chunking -> embeddings -> Supabase/pgvector -> retrieval -> OpenAI answer`

## 2. The 3 layers

The easiest way to understand the system is as 3 layers.

### Experience layer

This is what users interact with.

- `widget/widget.js` = public website chat widget
- `widget/trial.html` = landing page, trial, activation flow
- `widget/admin.html` = admin dashboard
- `widget/assistant.html` = internal/private assistant UI
- WhatsApp webhook = messaging channel

### Application layer

This layer handles product logic and request routing.

- `scripts/routes/*.py`
- `scripts/assistant_features.py`
- auth, rate limits, form handling, analytics, onboarding

### Knowledge layer

This layer powers the RAG behavior.

- `scripts/indexer.py`
- `scripts/rag_core.py`
- Supabase tables
- pgvector retrieval
- OpenAI answer generation

## 3. Main backend entry point

File:

- `scripts/server.py`

What it does:

- creates the FastAPI app
- registers middleware
- loads the embedding model
- initializes Supabase clients
- initializes OpenAI client
- initializes WhatsApp handler if enabled
- restores tokens from disk
- starts cleanup background task
- registers all route modules

Important point:

`server.py` is the app bootstrap/composition layer, but much of the shared runtime state actually lives in `scripts/config.py`.

## 4. Global config and runtime state

File:

- `scripts/config.py`

This is one of the most important files architecturally.

It contains:

- environment variables
- model names and keys
- Supabase and OpenAI clients
- admin tokens
- internal assistant tokens
- in-memory session history
- language cache
- trial progress
- thread pool

Why it matters:

This file acts like the shared application state. Many modules import it directly, which makes the code convenient but tightly coupled.

## 5. Database model

Defined in:

- `references/schema.sql`

Important tables:

- `sites`
- `documents`
- `chunks`
- `chat_logs`
- `visitor_logs`
- `assistant_form_submissions`
- `registrations`
- `internal_assistants`

How they relate:

- `sites` = one tenant / one assistant knowledge space
- `documents` = crawled pages or uploaded content for a site
- `chunks` = embedded text chunks belonging to documents
- `chat_logs` = analytics for queries
- `visitor_logs` = page-view tracking from widget loads
- `assistant_form_submissions` = structured lead/form captures
- `registrations` = trial/activation identity records
- `internal_assistants` = password-protected private assistants tied to a site

Important design idea:

Everything is centered around `site_id`.

That means:

- a public website assistant is a site
- a landing/demo site is a site
- a temporary trial is a site
- an internal assistant is also a site

This is one of the cleanest ideas in the whole project.

## 6. Public widget request flow

Main files:

- `widget/widget.js`
- `scripts/routes/chat.py`
- `scripts/rag_core.py`

Flow:

1. A customer website includes `widget.js` with a `data-site-id`.
2. The widget creates a chat bubble inside a Shadow DOM.
3. The widget creates or reuses a browser session ID in `localStorage`.
4. The widget loads assistant UI settings from `/api/widget/config/{site_id}`.
5. The user sends a message.
6. The widget sends `POST /api/chat`.
7. Backend checks rate limits and validates the domain/origin.
8. Backend runs the RAG pipeline.
9. Backend returns the answer, source list, confidence, and optional actions.
10. Widget renders the response.

What `widget.js` also does:

- sends `/api/track` page-view events
- renders greetings
- supports language switching
- supports starter buttons
- supports in-chat forms
- supports admin preview mode

## 7. The RAG flow

Main file:

- `scripts/rag_core.py`

Flow:

1. Receive `site_id`, query, optional session ID, and language.
2. Check session history for short follow-up questions.
3. Optionally expand the retrieval query using the previous user message.
4. Embed the query using the e5 embedding model.
5. Call Supabase RPC `match_chunks(...)`.
6. Get top chunks with hybrid ranking.
7. Build a context block from those chunks.
8. Send system prompt + context + user question to OpenAI.
9. Return final answer and sources.
10. Save a short chat history in memory for the session.

Important retrieval detail:

Ranking is hybrid:

- `70%` vector similarity
- `30%` keyword/full-text score

That logic lives in SQL inside `references/schema.sql`.

## 8. Ingestion and indexing flow

Main files:

- `scripts/indexer.py`
- `scripts/routes/trial.py`

Flow:

1. Start with a URL.
2. Crawl pages using either:
   - `requests` for static sites
   - Playwright for JS-rendered sites
3. Clean HTML to remove junk content.
4. Extract readable text and headings.
5. Split text into overlapping chunks.
6. Embed chunks in batches.
7. Insert `documents` and `chunks` rows into Supabase.

Important helpers in `indexer.py`:

- `StaticRenderer`
- `PlaywrightRenderer`
- `clean_html()`
- `chunk_text()`
- `extract_links()`
- `index_site()`

The trial flow, admin recrawl flow, and internal assistant crawl flow all reuse the same core indexing logic.

## 9. Trial and activation flow

Main files:

- `widget/trial.html`
- `scripts/routes/trial.py`
- `scripts/routes/auth.py`

What happens:

1. User enters a site URL on the landing/trial page.
2. Backend creates a temporary `sites` row with `is_trial = true`.
3. Background indexing starts.
4. Frontend polls progress through `/api/trial/progress/{site_id}`.
5. User tests the assistant.
6. User activates it with a code.
7. Trial site becomes permanent.

Why this is clever:

The trial is not a separate system. It is just a temporary site using the same RAG backend.

## 10. Assistant configuration layer

Main file:

- `scripts/assistant_features.py`

This is the layer that turns the project from “plain RAG chatbot” into a customizable assistant product.

It supports:

- custom title
- custom placeholder
- greeting message
- quick starters
- language switch config
- in-chat forms
- intent-based actions
- notifications for form submissions

This config is stored inside:

- `sites.settings.assistant`

Main responsibilities of `assistant_features.py`:

- normalize raw assistant JSON into a safe structure
- expose a public/sanitized version to the widget
- match user intent keywords
- validate form submissions
- store form submissions
- send notifications by email, Telegram, or WhatsApp

## 11. Admin dashboard flow

Main files:

- `widget/admin.html`
- `scripts/routes/admin.py`

What admin can do:

- authenticate with admin code
- view all sites
- inspect documents and chunks
- edit assistant config JSON
- upload PDFs
- trigger recrawls
- delete sites/documents/chunks
- view analytics
- manage internal assistants

This is basically the operator control center for the whole product.

## 12. Internal/private assistant flow

Main files:

- `widget/assistant.html`
- `scripts/routes/internal.py`

What it is:

A private assistant for businesses or teams, protected by passwords.

Flow:

1. Open `/assistant/{slug}`.
2. Authenticate using password.
3. Backend checks `internal_assistants` table.
4. Backend returns a token, role, and `site_id`.
5. User chats through the same `/api/chat` endpoint as the public widget.
6. If role is admin, the page also allows:
   - adding text
   - uploading PDFs
   - crawling websites
   - deleting documents/chunks

Important idea:

This is not a different AI engine. It is the same system, just with a private access model and admin tools.

## 13. Analytics flow

Main file:

- `scripts/routes/admin.py`

What gets tracked:

- `visitor_logs` from widget page loads
- `chat_logs` from chat requests

What admin sees:

- total queries
- daily/weekly query counts
- response times
- confidence distribution
- top sites
- recent queries
- visitor counts
- device/browser/OS breakdown
- top pages

This gives product and usage visibility without storing full generated answers.

## 14. WhatsApp support

Main files:

- `scripts/routes/whatsapp.py`
- `scripts/whatsapp_handler.py`

What it does:

- verifies webhook requests
- resolves inbound number to the correct tenant/site
- loads conversation history
- runs the same RAG flow
- sends reply back through BSP API
- stores message history

Again, the same knowledge layer is reused.

## 15. Important architectural strengths

These are the cleanest ideas in the codebase.

- Everything is modeled around `site_id`.
- One backend supports multiple product surfaces.
- The ingestion flow is reused by trial, admin, and internal assistant flows.
- The assistant config layer makes the widget customizable without custom code per client.
- Hybrid retrieval is done close to the data in SQL.

## 16. Main architectural weaknesses

These are the areas most worth improving.

- Heavy reliance on global mutable state via `config.py`
- Route handlers and business logic are still fairly coupled
- In-memory session history and token stores limit scaling
- Trial progress and rate limiting are in memory rather than shared storage
- `server.py` is not the biggest problem by itself; it is more the global-state pattern around it

## 17. Best refactor direction

If the goal is cleaner architecture, the best first move is not just “refactor `server.py`”.

A better path is:

1. Move shared runtime concerns behind explicit services.
2. Reduce direct `cfg.*` access from feature modules.
3. Keep routes thinner and move business logic into service classes/modules.
4. Let `server.py` become mainly the composition root.

Good service candidates:

- `rag_service`
- `indexing_service`
- `assistant_service`
- `auth_service`
- `analytics_service`
- `runtime_state` or `app_context`

## 18. Best file reading order for learning

If you want to understand the project progressively, read files in this order:

1. `README.md`
2. `scripts/server.py`
3. `scripts/config.py`
4. `scripts/routes/chat.py`
5. `scripts/rag_core.py`
6. `references/schema.sql`
7. `scripts/indexer.py`
8. `scripts/assistant_features.py`
9. `scripts/routes/admin.py`
10. `scripts/routes/internal.py`
11. `widget/widget.js`
12. `widget/admin.html`
13. `widget/assistant.html`
14. `widget/trial.html`

That order moves from big-picture entry points to the knowledge layer to product surfaces.

## 19. Short summary

The project is best understood as:

- one FastAPI backend
- one shared knowledge pipeline
- many product surfaces
- one strong tenant model centered on `site_id`

The most important long-term architectural decision is whether the system continues using global in-memory runtime state, or moves toward explicit services and dependency-managed state.

If you understand that point, you understand the core architecture of this repo.
