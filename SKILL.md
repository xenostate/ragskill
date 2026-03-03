---
name: web-rag
description: Crawl websites, chunk and embed content into Supabase pgvector, then answer questions using RAG with citations.
metadata: {"zeptoclaw":{"emoji":"🕸️","requires":{"bins":[],"env":["SUPABASE_URL","SUPABASE_SERVICE_KEY"]},"install":[]}}
---

# web-rag

Crawl → chunk → embed → store → retrieve → answer with citations.

All scripts use a dedicated venv at `~/.zeptoclaw/skills/web-rag/.venv/`. Use the venv python directly — no activation needed.

## Environment

Required env vars:
- `SUPABASE_URL` — Supabase project URL
- `SUPABASE_SERVICE_KEY` — Supabase service role key
- `EMBED_MODEL` — (optional) sentence-transformers model, default `intfloat/multilingual-e5-base`

## Setup (one-time)

1. Create venv and install deps (already done if you see `.venv/`):
```bash
python3 -m venv ~/.zeptoclaw/skills/web-rag/.venv
~/.zeptoclaw/skills/web-rag/.venv/bin/pip install -r ~/.zeptoclaw/skills/web-rag/scripts/requirements.txt
```

2. Run the schema in Supabase SQL editor. Read the schema from `references/schema.sql`.

3. Insert a site row:
```sql
insert into sites (domain, language) values ('example.com', 'en');
```

## Workflow

### 1. Index a site

```bash
~/.zeptoclaw/skills/web-rag/.venv/bin/python3 ~/.zeptoclaw/skills/web-rag/scripts/indexer.py --site-id <ID> --max-pages <N>
```

Optional `--start-url` to override the crawl entry point. The indexer:
- BFS-crawls same-domain links
- Strips nav/footer/scripts, extracts clean text
- Splits into ~500-word overlapping chunks
- Embeds with sentence-transformers (e5 `passage:` prefix)
- Upserts to Supabase `documents` + `chunks` tables
- Skips unchanged pages via `content_hash`
- Prints progress: pages crawled/skipped, chunks inserted, embeddings generated

### 2. Retrieve chunks

```bash
~/.zeptoclaw/skills/web-rag/.venv/bin/python3 ~/.zeptoclaw/skills/web-rag/scripts/retriever.py --site-id <ID> --query "your question" --top-k 5
```

Returns JSON: `{confidence, results: [{chunk_text, url, title, score}]}`.

Confidence levels:
- **high**: top score ≥ 0.75
- **medium**: top score ≥ 0.5
- **low**: below 0.5 or empty

### 3. RAG answer (the primary behavior)

When the user asks a question about an indexed site:

**Option A — Agent-driven RAG** (preferred):
1. Run retriever.py to get chunks + confidence
2. Inject the chunks as context into your own prompt
3. Follow these rules:
   - Answer ONLY from retrieved sources
   - Cite every claim with `[N]`
   - If sources don't cover it, say so
   - If confidence is "low", warn the user
   - List sources at the end: `[N] Title — URL`

**Option B — Standalone**:
```bash
~/.zeptoclaw/skills/web-rag/.venv/bin/python3 ~/.zeptoclaw/skills/web-rag/scripts/rag.py --site-id <ID> --query "question"
```
Uses `--context-only` flag to get the payload without calling an LLM.

## Tuning

| Env var | Default | Purpose |
|---------|---------|---------|
| `CHUNK_SIZE` | 500 | Words per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap words between chunks |
| `REQUEST_DELAY` | 0.5 | Seconds between HTTP requests |
| `THRESHOLD_HIGH` | 0.75 | Score threshold for high confidence |
| `THRESHOLD_MEDIUM` | 0.5 | Score threshold for medium confidence |
| `RAG_TOP_K` | 5 | Chunks to retrieve per query |
