from datetime import datetime, timezone

import pytest

from scripts.analytics import (
    aggregate_site_analytics,
    monthly_report_csv,
    normalize_question,
    utc_month_period,
)


def test_normalize_question_groups_case_spacing_and_trailing_punctuation():
    assert normalize_question("  What   are your HOURS? ") == "what are your hours"
    assert normalize_question("What are your hours!!!") == "what are your hours"


def test_aggregate_site_analytics_reports_usage_quality_feedback_and_leads():
    chats = [
        {
            "id": 1,
            "interaction_id": "msg_1",
            "session_id": "session-a",
            "query": "What are your hours?",
            "answer": "We open at nine.",
            "confidence": "high",
            "response_time_ms": 200,
            "status": "ok",
            "created_at": "2026-09-20T10:00:00+00:00",
        },
        {
            "id": 2,
            "interaction_id": "msg_2",
            "session_id": "session-a",
            "query": "what are your hours",
            "answer": "I do not have enough information.",
            "confidence": "low",
            "response_time_ms": 400,
            "status": "ok",
            "created_at": "2026-09-21T10:00:00+00:00",
        },
        {
            "id": 3,
            "interaction_id": "msg_3",
            "session_id": "session-b",
            "query": "Do you ship abroad?",
            "answer": "",
            "confidence": "low",
            "response_time_ms": 100,
            "status": "error",
            "error_code": "TimeoutError",
            "created_at": "2026-09-21T11:00:00+00:00",
        },
        {
            "id": 4,
            "interaction_id": "msg_4",
            "session_id": "session-c",
            "query": "Old failed question",
            "answer": "",
            "confidence": "low",
            "status": "error",
            "resolved_at": "2026-09-22T12:00:00+00:00",
            "created_at": "2026-09-21T12:00:00+00:00",
        },
    ]
    feedback = [
        {"message_id": "msg_1", "rating": "up"},
        {"message_id": "msg_2", "rating": "down"},
    ]
    forms = [
        {"session_id": "session-a", "created_at": "2026-09-21T13:00:00+00:00"},
    ]
    visitors = [
        {"session_id": "visitor-a", "created_at": "2026-09-20T09:00:00+00:00"},
        {"session_id": "visitor-a", "created_at": "2026-09-20T09:05:00+00:00"},
        {"session_id": "visitor-b", "created_at": "2026-09-21T09:00:00+00:00"},
    ]

    report = aggregate_site_analytics(chats, feedback, forms, visitors)

    assert report["summary"] == {
        "questions": 4,
        "answered": 2,
        "needs_improvement": 2,
        "low_confidence": 3,
        "errors": 2,
        "error_rate": 50.0,
        "avg_response_ms": 233,
        "p95_response_ms": 400,
        "chat_sessions": 3,
        "page_views": 3,
        "unique_visitors": 2,
        "leads": 1,
        "lead_sessions": 1,
        "lead_conversion_rate": 33.3,
        "feedback_up": 1,
        "feedback_down": 1,
        "helpful_rate": 50.0,
        "resolved": 1,
    }
    assert report["top_questions"][0]["question"] == "what are your hours"
    assert report["top_questions"][0]["count"] == 2
    assert {item["interaction_id"] for item in report["needs_improvement"]} == {"msg_2", "msg_3"}
    assert report["needs_improvement"][0]["feedback"] == "down" or report["needs_improvement"][1]["feedback"] == "down"


def test_month_period_and_csv_include_answer_detail():
    start, end, month = utc_month_period(
        "2026-09",
        now=datetime(2026, 9, 27, tzinfo=timezone.utc),
    )
    assert start.isoformat() == "2026-09-01T00:00:00+00:00"
    assert end.isoformat() == "2026-10-01T00:00:00+00:00"
    assert month == "2026-09"

    rows = [{
        "interaction_id": "msg_1",
        "query": "Pricing?",
        "answer": "Plans start at $10.",
        "confidence": "high",
        "status": "ok",
        "response_time_ms": 120,
        "created_at": "2026-09-03T12:00:00+00:00",
    }]
    report = aggregate_site_analytics(rows, [{"message_id": "msg_1", "rating": "up"}])
    report["_feedback_rows"] = [{"message_id": "msg_1", "rating": "up"}]
    content = monthly_report_csv({"domain": "example.com"}, month, report, rows)
    assert content.startswith("\ufeff")
    assert "WRS monthly report,example.com,2026-09" in content
    assert "Pricing?" in content
    assert "Plans start at $10." in content
    assert ",up," in content


def test_invalid_month_is_rejected():
    with pytest.raises(ValueError, match="YYYY-MM"):
        utc_month_period("September 2026")
