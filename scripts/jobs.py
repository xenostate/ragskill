"""Durable indexing job state stored in Supabase/Postgres."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import scripts.config as cfg
from scripts.observability import redact_text


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_url(url: str) -> str:
    """Keep operationally useful URL parts without credentials or query strings."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return "[invalid-url]"


def create_indexing_job(
    site_id: int,
    kind: str,
    *,
    url: str,
    max_pages: int,
    use_playwright: bool,
    pdf_count: int = 0,
    message: str = "Queued",
) -> str:
    """Create the durable record before handing work to a background thread."""
    job_id = str(uuid.uuid4())
    now = _now_iso()
    row = {
        "id": job_id,
        "site_id": site_id,
        "kind": kind,
        "status": "queued",
        "payload": {
            "url": _safe_url(url),
            "max_pages": max_pages,
            "renderer": "playwright" if use_playwright else "static",
            "pdf_count": pdf_count,
        },
        "step": 0,
        "total": 0,
        "message": message,
        "created_at": now,
        "updated_at": now,
    }
    cfg.sb.table("indexing_jobs").insert(row).execute()
    cfg.trial_progress[site_id] = {
        "job_id": job_id,
        "step": 0,
        "total": 0,
        "message": message,
        "done": False,
        "error": None,
        "status": "queued",
    }
    cfg.log.info(
        "indexing job queued",
        extra={"event": "indexing_job.queued", "job_id": job_id, "site_id": site_id, "kind": kind},
    )
    return job_id


def update_indexing_job(job_id: str, site_id: int, **changes: Any) -> None:
    """Persist progress and mirror it in memory for low-latency SSE updates."""
    allowed = {
        "status",
        "step",
        "total",
        "message",
        "error",
        "error_code",
        "started_at",
        "finished_at",
        "heartbeat_at",
    }
    payload = {key: value for key, value in changes.items() if key in allowed}
    payload["updated_at"] = _now_iso()
    if payload.get("status") == "running" and "heartbeat_at" not in payload:
        payload["heartbeat_at"] = payload["updated_at"]

    cfg.sb.table("indexing_jobs").update(payload).eq("id", job_id).execute()

    progress = cfg.trial_progress.setdefault(site_id, {"job_id": job_id})
    for key in ("step", "total", "message", "error", "status"):
        if key in payload:
            progress[key] = payload[key]
    progress["job_id"] = job_id
    progress["done"] = payload.get("status", progress.get("status")) in TERMINAL_STATUSES
    if progress["done"]:
        progress["_finished_at"] = time.time()


def start_indexing_job(job_id: str, site_id: int) -> None:
    now = _now_iso()
    update_indexing_job(
        job_id,
        site_id,
        status="running",
        message="Starting indexing",
        started_at=now,
        heartbeat_at=now,
    )


def complete_indexing_job(job_id: str, site_id: int, message: str, *, step: int, total: int) -> None:
    update_indexing_job(
        job_id,
        site_id,
        status="succeeded",
        message=message,
        step=step,
        total=total,
        error=None,
        finished_at=_now_iso(),
    )
    cfg.log.info(
        "indexing job completed",
        extra={"event": "indexing_job.succeeded", "job_id": job_id, "site_id": site_id},
    )


def fail_indexing_job(job_id: str, site_id: int, exc: Exception) -> None:
    try:
        update_indexing_job(
            job_id,
            site_id,
            status="failed",
            message="Indexing failed",
            error=redact_text(str(exc))[:1000],
            error_code=exc.__class__.__name__,
            finished_at=_now_iso(),
        )
    except Exception as state_exc:
        cfg.log.error(
            "could not persist failed indexing job state",
            extra={
                "event": "indexing_job.state_write_failed",
                "job_id": job_id,
                "site_id": site_id,
                "error_code": state_exc.__class__.__name__,
            },
        )
    try:
        import sentry_sdk

        sentry_sdk.capture_exception(exc)
    except Exception:
        pass
    cfg.log.error(
        "indexing job failed",
        extra={
            "event": "indexing_job.failed",
            "job_id": job_id,
            "site_id": site_id,
            "error_code": exc.__class__.__name__,
        },
    )


def latest_indexing_job(site_id: int) -> dict | None:
    response = (
        cfg.sb.table("indexing_jobs")
        .select("id,site_id,kind,status,step,total,message,error,error_code,created_at,started_at,finished_at,updated_at")
        .eq("site_id", site_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


def progress_payload(job: dict) -> dict:
    status = job.get("status", "queued")
    return {
        "job_id": job.get("id"),
        "status": status,
        "step": job.get("step", 0),
        "total": job.get("total", 0),
        "message": job.get("message", ""),
        "done": status in TERMINAL_STATUSES,
        "error": job.get("error"),
        "error_code": job.get("error_code"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
    }


def mark_interrupted_jobs() -> int:
    """Make work abandoned by a prior process restart explicit and observable."""
    now = _now_iso()
    response = (
        cfg.sb.table("indexing_jobs")
        .update({
            "status": "failed",
            "message": "Interrupted by application restart; retry the indexing job",
            "error": "The worker stopped before this job completed.",
            "error_code": "WorkerRestart",
            "finished_at": now,
            "updated_at": now,
        })
        .in_("status", ["queued", "running"])
        .execute()
    )
    return len(response.data or [])
