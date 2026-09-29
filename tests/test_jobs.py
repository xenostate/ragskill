from unittest.mock import MagicMock

import scripts.config as cfg
from scripts.jobs import create_indexing_job, progress_payload, update_indexing_job


def test_create_job_persists_safe_payload_and_memory_state(monkeypatch):
    table = MagicMock()
    table.insert.return_value.execute.return_value = MagicMock(data=[])
    client = MagicMock()
    client.table.return_value = table
    monkeypatch.setattr(cfg, "sb", client)
    cfg.trial_progress.clear()

    job_id = create_indexing_job(
        42,
        "trial",
        url="https://user:pass@example.com/path?token=secret",
        max_pages=5,
        use_playwright=False,
        pdf_count=1,
    )

    row = table.insert.call_args.args[0]
    assert row["id"] == job_id
    assert row["payload"]["url"] == "https://example.com/path"
    assert row["payload"]["pdf_count"] == 1
    assert cfg.trial_progress[42]["job_id"] == job_id
    assert cfg.trial_progress[42]["status"] == "queued"


def test_update_job_mirrors_terminal_progress(monkeypatch):
    execute = MagicMock(data=[])
    table = MagicMock()
    table.update.return_value.eq.return_value.execute.return_value = execute
    client = MagicMock()
    client.table.return_value = table
    monkeypatch.setattr(cfg, "sb", client)
    cfg.trial_progress[9] = {"job_id": "job-1", "status": "running"}

    update_indexing_job(
        "job-1",
        9,
        status="succeeded",
        step=3,
        total=3,
        message="Done",
    )

    assert cfg.trial_progress[9]["done"] is True
    assert cfg.trial_progress[9]["step"] == 3
    assert cfg.trial_progress[9]["message"] == "Done"


def test_progress_payload_has_legacy_sse_shape():
    payload = progress_payload({
        "id": "job-2",
        "status": "failed",
        "step": 1,
        "total": 2,
        "message": "Indexing failed",
        "error": "boom",
        "error_code": "RuntimeError",
    })

    assert payload["job_id"] == "job-2"
    assert payload["done"] is True
    assert payload["error_code"] == "RuntimeError"
