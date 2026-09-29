#!/usr/bin/env python3
"""Apply ordered, checksummed Postgres migrations using DATABASE_URL."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = ROOT / "migrations"
load_dotenv(ROOT / ".env")


_INCLUDE_RE = re.compile(r"^\s*--\s*wrs:include\s+(.+?)\s*$", re.MULTILINE)


def migration_sql(path: Path) -> str:
    """Expand explicit repository-local includes used by the imported baseline."""
    sql = path.read_text(encoding="utf-8")

    def expand(match: re.Match) -> str:
        included = (ROOT / match.group(1)).resolve()
        if ROOT.resolve() not in included.parents:
            raise RuntimeError(f"Migration include escapes repository: {included}")
        return included.read_text(encoding="utf-8")

    return _INCLUDE_RE.sub(expand, sql)


def checksum(path: Path) -> str:
    return hashlib.sha256(migration_sql(path).encode("utf-8")).hexdigest()


def migration_files() -> list[Path]:
    files = sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9][0-9]_*.sql"))
    if not files:
        raise RuntimeError(f"No migrations found in {MIGRATIONS_DIR}")
    return files


def migrate(database_url: str, *, dry_run: bool = False) -> int:
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("psycopg is required; install scripts/requirements.txt") from exc

    applied_count = 0
    with psycopg.connect(database_url, autocommit=False) as connection:
        connection.execute("select pg_advisory_lock(hashtext('wrs_schema_migrations'))")
        try:
            connection.execute(
                """
                create table if not exists schema_migrations (
                    version text primary key,
                    checksum text not null,
                    applied_at timestamptz not null default now()
                )
                """
            )
            connection.commit()
            existing = {
                row[0]: row[1]
                for row in connection.execute(
                    "select version, checksum from schema_migrations order by version"
                ).fetchall()
            }

            for path in migration_files():
                version = path.stem
                digest = checksum(path)
                if version in existing:
                    if existing[version] != digest:
                        raise RuntimeError(
                            f"Checksum mismatch for applied migration {version}; "
                            "never edit an applied migration"
                        )
                    print(f"skip  {version}")
                    continue

                if dry_run:
                    print(f"would apply  {version}")
                    continue

                print(f"apply {version}")
                try:
                    connection.execute(migration_sql(path))
                    connection.execute(
                        "insert into schema_migrations (version, checksum) values (%s, %s)",
                        (version, digest),
                    )
                    connection.commit()
                    applied_count += 1
                except Exception:
                    connection.rollback()
                    raise
        finally:
            connection.execute("select pg_advisory_unlock(hashtext('wrs_schema_migrations'))")
            connection.commit()
    return applied_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.database_url:
        parser.error("DATABASE_URL or --database-url is required")

    try:
        count = migrate(args.database_url, dry_run=args.dry_run)
    except Exception as exc:
        print(f"migration failed: {exc}", file=sys.stderr)
        return 1
    print(f"done: {count} migration(s) applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
