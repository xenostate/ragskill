# Operations runbook

## Staging

1. Copy `.env.staging.example` to `.env.staging` and fill every required value.
2. Point the staging hostname at the VPS.
3. Run `ops/deploy-staging.sh`. It builds the pinned image, applies migrations, and starts the app behind Caddy TLS.
4. Set the GitHub Actions repository variable `STAGING_URL` to the staging base URL. `PRODUCTION_URL` defaults to `https://wrs.kz`.

Deployments must run `python -m scripts.migrate` before starting application code that depends on a new migration. Applied migration checksums are immutable.

## Database backup and restore

`ops/backup.sh` creates a compressed Postgres custom-format dump of the `public` schema, validates its catalog, writes a SHA-256 sidecar, removes expired local backups, and can copy the result off-site through rclone.

For a VPS, install `postgresql-client`, copy `ops/backup.env.example` to `/etc/wrs/backup.env` with mode `0600`, install the two files from `ops/systemd/`, then enable the timer. The service expects the checkout at `/opt/wrs/current` and a `wrs` system user:

```bash
sudo install -d -o wrs -g wrs -m 0700 /var/backups/wrs
sudo install -m 0644 ops/systemd/wrs-backup.service ops/systemd/wrs-backup.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wrs-backup.timer
systemctl list-timers wrs-backup.timer
```

Use a direct Postgres or session-pooler `DATABASE_URL`; transaction poolers are unsuitable for migrations and dumps. Regularly test restoration into a disposable database:

```bash
DATABASE_URL=postgresql://... ops/restore.sh --confirm /var/backups/wrs/wrs-public-TIMESTAMP.dump
python -m scripts.migrate
```

For a completely empty target, apply migrations once before the restore so Postgres extensions such as `vector` exist, then run migrations again afterward to verify the restored migration ledger. `restore.sh` is intentionally gated by `--confirm` because it uses `pg_restore --clean --if-exists`. Stop writers or isolate the target before restoring. A backup is not considered reliable until a restore drill has succeeded.
