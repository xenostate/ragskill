#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo "Usage: DATABASE_URL=... $0 --confirm /absolute/path/to/wrs-public-TIMESTAMP.dump" >&2
}

if [[ "${1:-}" != "--confirm" || -z "${2:-}" || -n "${3:-}" ]]; then
    usage
    exit 2
fi
: "${DATABASE_URL:?DATABASE_URL must be set}"

backup_file="$2"
if [[ "$backup_file" != /* ]]; then
    echo "Use an absolute backup path." >&2
    exit 1
fi
if [[ ! -f "$backup_file" ]]; then
    echo "Backup does not exist: $backup_file" >&2
    exit 1
fi

command -v pg_restore >/dev/null || { echo "pg_restore is required" >&2; exit 1; }
pg_restore --list "$backup_file" >/dev/null

if [[ -f "$backup_file.sha256" ]]; then
    (cd "$(dirname "$backup_file")" && sha256sum --check "$(basename "$backup_file").sha256")
fi

echo "Restoring $backup_file. Existing objects represented in the backup may be replaced."
pg_restore \
    --dbname "$DATABASE_URL" \
    --clean \
    --if-exists \
    --no-owner \
    --no-acl \
    --single-transaction \
    "$backup_file"
echo "Restore complete. Run the application smoke tests before accepting traffic."
