#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ ! -f .env.staging ]]; then
    echo "Missing .env.staging; copy .env.staging.example and fill its secrets." >&2
    exit 1
fi

docker compose -f compose.staging.yml build --pull
docker compose -f compose.staging.yml --profile tools run --rm migrate
docker compose -f compose.staging.yml up -d --remove-orphans
docker compose -f compose.staging.yml ps
