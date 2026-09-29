-- Durable state for crawl/indexing work started by the API process.
create table if not exists indexing_jobs (
    id              uuid primary key,
    site_id         bigint not null references sites(id) on delete cascade,
    kind            text not null,
    status          text not null default 'queued'
                    check (status in ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    payload         jsonb not null default '{}'::jsonb,
    step            integer not null default 0 check (step >= 0),
    total           integer not null default 0 check (total >= 0),
    message         text,
    error           text,
    error_code      text,
    created_at      timestamptz not null default now(),
    started_at      timestamptz,
    finished_at     timestamptz,
    heartbeat_at    timestamptz,
    updated_at      timestamptz not null default now()
);

create index if not exists idx_indexing_jobs_site_created
    on indexing_jobs(site_id, created_at desc);
create index if not exists idx_indexing_jobs_active
    on indexing_jobs(status, updated_at)
    where status in ('queued', 'running');

-- Job state is internal. The service role bypasses RLS; the anon client gets no policy.
alter table indexing_jobs enable row level security;

comment on table indexing_jobs is
    'Durable operational state for crawl/indexing work; job execution remains in-process.';
