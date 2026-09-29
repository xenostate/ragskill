"""Pilot analytics aggregation and monthly CSV reporting.

The aggregation functions in this module are deliberately independent of
FastAPI and Supabase so the business definitions can be unit tested without
external services.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone


_SPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCTUATION_RE = re.compile(r"[\s?!.,;:]+$")


def utc_period_for_days(days: int, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Return a bounded UTC period ending now."""
    end = now or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    safe_days = min(max(int(days), 1), 366)
    return end - timedelta(days=safe_days), end


def utc_month_period(month: str, now: datetime | None = None) -> tuple[datetime, datetime, str]:
    """Return [start, end) for ``YYYY-MM``; default to the current UTC month."""
    current = now or datetime.now(timezone.utc)
    selected = month or current.strftime("%Y-%m")
    if not re.fullmatch(r"\d{4}-\d{2}", selected):
        raise ValueError("month must use YYYY-MM format")
    try:
        start = datetime.strptime(selected, "%Y-%m").replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValueError("month must use YYYY-MM format") from exc
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end, start.strftime("%Y-%m")


def normalize_question(question: str | None) -> str:
    """Create a conservative key for grouping repeated questions."""
    normalized = _SPACE_RE.sub(" ", str(question or "").strip().casefold())
    return _TRAILING_PUNCTUATION_RE.sub("", normalized)


def _feedback_by_message(feedback: list[dict]) -> dict[str, str]:
    # The UI only permits one rating per rendered answer. If duplicated data
    # exists, the most recently returned row wins.
    result: dict[str, str] = {}
    for row in feedback:
        message_id = str(row.get("message_id") or "")
        rating = row.get("rating")
        if message_id and rating in {"up", "down"}:
            result[message_id] = rating
    return result


def aggregate_site_analytics(
    chat_logs: list[dict],
    feedback: list[dict] | None = None,
    form_submissions: list[dict] | None = None,
    visitor_logs: list[dict] | None = None,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict:
    """Build customer-facing usage and quality metrics for one site."""
    feedback = feedback or []
    form_submissions = form_submissions or []
    visitor_logs = visitor_logs or []
    ratings = _feedback_by_message(feedback)

    response_times: list[int] = []
    confidence = Counter({"high": 0, "medium": 0, "low": 0})
    question_groups: dict[str, dict] = {}
    needs_improvement: list[dict] = []
    recent_interactions: list[dict] = []
    sessions: set[str] = set()
    by_day: dict[str, dict[str, int]] = defaultdict(
        lambda: {"questions": 0, "low_confidence": 0, "errors": 0, "leads": 0}
    )
    answered = errors = resolved = 0

    sorted_logs = sorted(chat_logs, key=lambda row: str(row.get("created_at") or ""), reverse=True)
    for row in sorted_logs:
        status = row.get("status") or ("error" if row.get("error_code") else "ok")
        conf = str(row.get("confidence") or "low")
        if conf not in {"high", "medium", "low"}:
            conf = "low"
        confidence[conf] += 1
        is_error = status == "error" or bool(row.get("error_code"))
        has_answer = bool(str(row.get("answer") or "").strip()) and not is_error
        if has_answer:
            answered += 1
        if is_error:
            errors += 1
        if row.get("resolved_at"):
            resolved += 1

        response_ms = row.get("response_time_ms")
        if isinstance(response_ms, (int, float)) and response_ms >= 0:
            response_times.append(int(response_ms))

        session_id = str(row.get("session_id") or "")
        if session_id:
            sessions.add(session_id)

        message_id = str(row.get("interaction_id") or row.get("message_id") or "")
        rating = ratings.get(message_id)
        created_at = str(row.get("created_at") or "")
        day = created_at[:10]
        if day:
            by_day[day]["questions"] += 1
            if conf == "low":
                by_day[day]["low_confidence"] += 1
            if is_error:
                by_day[day]["errors"] += 1

        item = {
            "id": row.get("id"),
            "interaction_id": message_id,
            "query": row.get("query") or "",
            "answer": row.get("answer") or "",
            "confidence": conf,
            "status": "error" if is_error else "ok",
            "error_code": row.get("error_code"),
            "error_message": row.get("error_message"),
            "response_time_ms": int(response_ms or 0),
            "feedback": rating,
            "created_at": created_at,
            "resolved_at": row.get("resolved_at"),
            "resolved_document_id": row.get("resolved_document_id"),
        }
        recent_interactions.append(item)

        needs_work = is_error or conf == "low" or rating == "down" or not has_answer
        if needs_work and not row.get("resolved_at"):
            reasons = []
            if is_error or not has_answer:
                reasons.append("unanswered")
            if conf == "low":
                reasons.append("low_confidence")
            if rating == "down":
                reasons.append("negative_feedback")
            item = {**item, "reasons": reasons}
            needs_improvement.append(item)

        key = normalize_question(row.get("query"))
        if key:
            group = question_groups.setdefault(key, {
                "question": row.get("query") or "",
                "count": 0,
                "low_confidence": 0,
                "errors": 0,
                "negative_feedback": 0,
            })
            group["count"] += 1
            group["low_confidence"] += int(conf == "low")
            group["errors"] += int(is_error)
            group["negative_feedback"] += int(rating == "down")

    for submission in form_submissions:
        day = str(submission.get("created_at") or "")[:10]
        if day:
            by_day[day]["leads"] += 1

    top_questions = sorted(
        question_groups.values(),
        key=lambda item: (-item["count"], -item["low_confidence"], item["question"].casefold()),
    )[:10]
    response_times.sort()
    avg_response_ms = round(sum(response_times) / len(response_times)) if response_times else 0
    p95_index = max(0, int(len(response_times) * 0.95 + 0.999999) - 1)
    p95_response_ms = response_times[p95_index] if response_times else 0
    total = len(chat_logs)
    lead_count = len(form_submissions)
    converted_sessions = {
        str(row.get("session_id") or f"submission:{row.get('id')}")
        for row in form_submissions
    }
    session_count = len(sessions)
    feedback_up = sum(1 for row in feedback if row.get("rating") == "up")
    feedback_down = sum(1 for row in feedback if row.get("rating") == "down")
    feedback_total = feedback_up + feedback_down
    unique_visitors = len({row.get("session_id") for row in visitor_logs if row.get("session_id")})

    return {
        "period": {
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
        },
        "summary": {
            "questions": total,
            "answered": answered,
            "needs_improvement": len(needs_improvement),
            "low_confidence": confidence["low"],
            "errors": errors,
            "error_rate": round(errors * 100 / total, 1) if total else 0.0,
            "avg_response_ms": avg_response_ms,
            "p95_response_ms": p95_response_ms,
            "chat_sessions": session_count,
            "page_views": len(visitor_logs),
            "unique_visitors": unique_visitors,
            "leads": lead_count,
            "lead_sessions": len(converted_sessions),
            "lead_conversion_rate": round(len(converted_sessions) * 100 / session_count, 1) if session_count else 0.0,
            "feedback_up": feedback_up,
            "feedback_down": feedback_down,
            "helpful_rate": round(feedback_up * 100 / feedback_total, 1) if feedback_total else None,
            "resolved": resolved,
        },
        "confidence": dict(confidence),
        "top_questions": top_questions,
        "needs_improvement": needs_improvement[:50],
        "recent_interactions": recent_interactions[:50],
        "usage_by_day": [
            {"date": day, **counts}
            for day, counts in sorted(by_day.items())
        ],
    }


def load_site_analytics(client, site_id: int, start: datetime, end: datetime) -> tuple[dict, list[dict]]:
    """Load one site's period rows from Supabase and aggregate them."""
    start_iso = start.isoformat()
    end_iso = end.isoformat()
    chat_fields = (
        "id, site_id, interaction_id, session_id, query, answer, confidence, "
        "response_time_ms, chunk_count, sources, status, error_code, error_message, "
        "resolved_at, resolved_document_id, created_at"
    )
    chats = (
        client.table("chat_logs")
        .select(chat_fields)
        .eq("site_id", site_id)
        .gte("created_at", start_iso)
        .lt("created_at", end_iso)
        .order("created_at", desc=True)
        .limit(10000)
        .execute()
    ).data or []
    feedback = (
        client.table("assistant_feedback")
        .select("message_id, rating, created_at")
        .eq("site_id", site_id)
        .gte("created_at", start_iso)
        .lt("created_at", end_iso)
        .order("created_at", desc=False)
        .limit(10000)
        .execute()
    ).data or []
    forms = (
        client.table("assistant_form_submissions")
        .select("id, session_id, form_id, created_at")
        .eq("site_id", site_id)
        .gte("created_at", start_iso)
        .lt("created_at", end_iso)
        .limit(10000)
        .execute()
    ).data or []
    visitors = (
        client.table("visitor_logs")
        .select("session_id, created_at")
        .eq("site_id", site_id)
        .gte("created_at", start_iso)
        .lt("created_at", end_iso)
        .limit(20000)
        .execute()
    ).data or []
    report = aggregate_site_analytics(
        chats,
        feedback,
        forms,
        visitors,
        start=start,
        end=end,
    )
    report["_feedback_rows"] = feedback
    return report, chats


def monthly_report_csv(site: dict, month: str, report: dict, interactions: list[dict]) -> str:
    """Render a spreadsheet-friendly UTF-8 CSV with summary and interaction rows."""
    output = io.StringIO()
    writer = csv.writer(output)
    summary = report["summary"]
    writer.writerow(["WRS monthly report", site.get("domain", ""), month])
    writer.writerow([])
    writer.writerow(["Metric", "Value"])
    for label, key in (
        ("Questions", "questions"),
        ("Answered", "answered"),
        ("Needs improvement", "needs_improvement"),
        ("Low confidence", "low_confidence"),
        ("Errors", "errors"),
        ("Error rate (%)", "error_rate"),
        ("Average response time (ms)", "avg_response_ms"),
        ("P95 response time (ms)", "p95_response_ms"),
        ("Chat sessions", "chat_sessions"),
        ("Page views", "page_views"),
        ("Unique visitors", "unique_visitors"),
        ("Lead conversions", "leads"),
        ("Converted chat sessions", "lead_sessions"),
        ("Lead conversion rate (%)", "lead_conversion_rate"),
        ("Helpful feedback", "feedback_up"),
        ("Unhelpful feedback", "feedback_down"),
    ):
        writer.writerow([label, summary.get(key, "")])

    writer.writerow([])
    writer.writerow([
        "Date", "Question", "WRS answer", "Confidence", "Status", "Response time (ms)",
        "Feedback", "Error", "Resolved at",
    ])
    rating_map = _feedback_by_message(report.get("_feedback_rows", []))
    for row in sorted(interactions, key=lambda item: str(item.get("created_at") or "")):
        message_id = str(row.get("interaction_id") or row.get("message_id") or "")
        is_error = (row.get("status") == "error") or bool(row.get("error_code"))
        writer.writerow([
            row.get("created_at") or "",
            row.get("query") or "",
            row.get("answer") or "",
            row.get("confidence") or "low",
            "error" if is_error else "ok",
            row.get("response_time_ms") or 0,
            rating_map.get(message_id, ""),
            row.get("error_message") or row.get("error_code") or "",
            row.get("resolved_at") or "",
        ])
    return "\ufeff" + output.getvalue()
