-- WhatsApp Business integration tables
-- Run this in Supabase SQL Editor to enable WhatsApp support

-- Business WhatsApp accounts (one per business phone number)
create table if not exists whatsapp_accounts (
    id              bigint generated always as identity primary key,
    site_id         bigint not null references sites(id) on delete cascade,
    phone_number    text not null unique,       -- business WhatsApp number (E.164 format, e.g. +77001234567)
    display_name    text,                       -- business display name shown to customers
    api_token       text,                       -- BSP API token for sending replies
    provider        text not null default '360dialog',  -- BSP provider name
    is_active       boolean not null default false,
    created_at      timestamptz not null default now()
);

create index if not exists idx_wa_accounts_site  on whatsapp_accounts(site_id);
create index if not exists idx_wa_accounts_phone on whatsapp_accounts(phone_number);

-- Persistent conversation history (survives server restarts)
create table if not exists whatsapp_conversations (
    id              bigint generated always as identity primary key,
    account_id      bigint not null references whatsapp_accounts(id) on delete cascade,
    customer_phone  text not null,              -- SHA-256 hash of customer phone (privacy)
    role            text not null check (role in ('user', 'assistant')),
    content         text not null,
    created_at      timestamptz not null default now()
);

-- Fast lookup: recent messages for a given customer on a given account
create index if not exists idx_wa_conv_lookup
    on whatsapp_conversations(account_id, customer_phone, created_at desc);

-- Cleanup function for old WhatsApp conversations (call via pg_cron or Supabase scheduled function)
create or replace function cleanup_old_conversations(days_old int default 30)
returns int
language plpgsql
as $$
declare
    deleted_count int;
begin
    delete from whatsapp_conversations
    where created_at < now() - make_interval(days => days_old);
    get diagnostics deleted_count = row_count;
    return deleted_count;
end;
$$;
