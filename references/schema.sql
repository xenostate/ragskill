-- web-rag Supabase schema
-- Run this once in the Supabase SQL editor to set up tables.

-- Enable pgvector
create extension if not exists vector;

-- Sites table
create table if not exists sites (
    id          bigint generated always as identity primary key,
    domain      text not null unique,
    language    text default 'en',
    settings    jsonb default '{}',
    created_at  timestamptz default now()
);

-- Documents table
create table if not exists documents (
    id              bigint generated always as identity primary key,
    site_id         bigint not null references sites(id) on delete cascade,
    url             text not null,
    title           text,
    content_hash    text,
    last_crawled    timestamptz default now(),
    created_at      timestamptz default now(),
    unique(site_id, url)
);

create index if not exists idx_documents_site on documents(site_id);
create index if not exists idx_documents_content_hash on documents(site_id, content_hash);

-- Chunks table with embedding vector and optional tsvector
create table if not exists chunks (
    id              bigint generated always as identity primary key,
    document_id     bigint not null references documents(id) on delete cascade,
    chunk_index     int not null,
    text            text not null,
    language        text,
    headings        text[],
    embedding       vector(768),  -- matches multilingual-e5-base output dim
    tsv             tsvector generated always as (to_tsvector('simple', text)) stored,
    created_at      timestamptz default now()
);

create index if not exists idx_chunks_doc on chunks(document_id);
create index if not exists idx_chunks_embedding on chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);
create index if not exists idx_chunks_tsv on chunks using gin(tsv);
create index if not exists idx_chunks_language on chunks(language) where language is not null;

-- Hybrid retrieval function: vector similarity + keyword matching
create or replace function match_chunks(
    p_site_id       bigint,
    p_query_embedding vector(768),
    p_query_text    text,
    p_match_count   int default 5
)
returns table (
    id          bigint,
    document_id bigint,
    chunk_index int,
    text        text,
    url         text,
    title       text,
    headings    text[],
    score       float
)
language plpgsql
as $$
begin
    return query
    select
        c.id,
        c.document_id,
        c.chunk_index,
        c.text,
        d.url,
        d.title,
        c.headings,
        -- Hybrid score: 0.7 * vector similarity + 0.3 * keyword rank
        -- ts_rank flag 32 normalizes to [0,1) range so both components are comparable
        (
            0.7 * (1 - (c.embedding <=> p_query_embedding)) +
            0.3 * coalesce(ts_rank(c.tsv, plainto_tsquery('simple', p_query_text), 32), 0)
        ) as score
    from chunks c
    join documents d on d.id = c.document_id
    where d.site_id = p_site_id
    order by score desc
    limit p_match_count;
end;
$$;

-- Telegram chat-to-site bindings
create table if not exists telegram_bindings (
    chat_id     bigint primary key,
    site_id     bigint not null references sites(id) on delete cascade,
    username    text,
    bound_at    timestamptz default now()
);

-- Trial site support (run this migration if tables already exist)
alter table sites add column if not exists is_trial boolean default false;
alter table sites add column if not exists expires_at timestamptz;

create index if not exists idx_sites_trial_expiry
    on sites(is_trial, expires_at)
    where is_trial = true;

-- User registrations for trial/purchase flow
create table if not exists registrations (
    id          bigint generated always as identity primary key,
    name        text not null,
    email       text not null unique,
    company     text,
    token       text not null unique,
    created_at  timestamptz default now()
);

-- Internal assistants (self-service knowledge bases for businesses)
create table if not exists internal_assistants (
    id              bigint generated always as identity primary key,
    slug            text not null unique,              -- URL-safe identifier, e.g. "acme-corp"
    name            text not null,                     -- Display name
    site_id         bigint not null references sites(id) on delete cascade,
    admin_password  text not null,                     -- bcrypt hash
    user_password   text not null,                     -- bcrypt hash
    created_by      text,                              -- email or identifier
    settings        jsonb default '{}',                -- custom settings (welcome message, language, etc.)
    created_at      timestamptz default now()
);

create index if not exists idx_internal_assistants_slug on internal_assistants(slug);
create index if not exists idx_internal_assistants_site on internal_assistants(site_id);

-- Chat analytics log (lightweight — query text only, no answers stored)
create table if not exists chat_logs (
    id              bigint generated always as identity primary key,
    site_id         bigint references sites(id) on delete cascade,
    query           text not null,
    confidence      text,           -- high / medium / low
    response_time_ms int,
    chunk_count     int,
    created_at      timestamptz default now()
);

create index if not exists idx_chat_logs_site    on chat_logs(site_id);
create index if not exists idx_chat_logs_created on chat_logs(created_at desc);

-- Widget visitor page-view log (sent as a beacon from widget.js on every page load)
create table if not exists visitor_logs (
    id          bigint generated always as identity primary key,
    site_id     bigint references sites(id) on delete cascade,
    session_id  text,           -- anonymous session from widget localStorage
    ip          text,           -- raw client IP (anonymise/hash for GDPR if needed)
    user_agent  text,
    device_type text,           -- desktop / mobile / tablet
    browser     text,           -- Chrome / Firefox / Safari / Edge / Other
    os          text,           -- Windows / macOS / iOS / Android / Linux / Other
    referer     text,           -- full page URL where the widget was loaded
    created_at  timestamptz default now()
);

create index if not exists idx_visitor_logs_site    on visitor_logs(site_id);
create index if not exists idx_visitor_logs_created on visitor_logs(created_at desc);
create index if not exists idx_visitor_logs_session on visitor_logs(session_id);
