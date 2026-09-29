#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

: "${DATABASE_URL:?DATABASE_URL must be set}"

backup_root="${BACKUP_DIR:-/var/backups/wrs}"
retention_days="${BACKUP_RETENTION_DAYS:-14}"
schema="${BACKUP_SCHEMA:-public}"

if [[ "$backup_root" == "/" || -z "$backup_root" ]]; then
    echo "Refusing unsafe BACKUP_DIR: $backup_root" >&2
    exit 1
fi
if ! [[ "$retention_days" =~ ^[0-9]+$ ]]; then
    echo "BACKUP_RETENTION_DAYS must be a non-negative integer" >&2
    exit 1
fi

command -v pg_dump >/dev/null || { echo "pg_dump is required" >&2; exit 1; }
command -v pg_restore >/dev/null || { echo "pg_restore is required" >&2; exit 1; }

mkdir -p "$backup_root"
backup_root="$(cd "$backup_root" && pwd -P)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
final_path="$backup_root/wrs-${schema}-${timestamp}.dump"
temporary_path="$final_path.partial"

cleanup() {
    rm -f -- "$temporary_path"
}
trap cleanup EXIT

pg_dump \
    --dbname "$DATABASE_URL" \
    --format custom \
    --compress 9 \
    --no-owner \
    --no-acl \
    --schema "$schema" \
    --file "$temporary_path"

pg_restore --list "$temporary_path" >/dev/null
mv -- "$temporary_path" "$final_path"
(cd "$backup_root" && sha256sum "$(basename "$final_path")") >"$final_path.sha256"

if [[ -n "${BACKUP_RCLONE_REMOTE:-}" ]]; then
    command -v rclone >/dev/null || { echo "BACKUP_RCLONE_REMOTE set but rclone is missing" >&2; exit 1; }
    rclone copy "$final_path" "$BACKUP_RCLONE_REMOTE"
    rclone copy "$final_path.sha256" "$BACKUP_RCLONE_REMOTE"
fi

find "$backup_root" -maxdepth 1 -type f \
    \( -name 'wrs-*.dump' -o -name 'wrs-*.dump.sha256' \) \
    -mtime "+$retention_days" -delete

echo "Backup complete: $final_path"
