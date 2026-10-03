"""The public URL-to-chat demo keeps the selected page version and expires quickly."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from postgrest.exceptions import APIError

import scripts.config as cfg
import scripts.routes.trial as trial


class FakeQuery:
    def __init__(self, database, table):
        self.database = database
        self.table = table
        self.action = None
        self.payload = None

    def insert(self, payload):
        self.action, self.payload = "insert", payload
        return self

    def update(self, payload):
        self.action, self.payload = "update", payload
        return self

    def delete(self):
        self.action, self.payload = "delete", None
        return self

    def eq(self, *_args):
        return self

    def execute(self):
        if self.table == "indexing_jobs" and self.database.jobs_table_missing:
            raise APIError({"code": "PGRST205", "message": "Could not find indexing_jobs"})
        self.database.writes.append((self.table, self.action, self.payload))
        return SimpleNamespace(data=[{"id": 17 if self.table == "sites" else 31}])


class FakeDatabase:
    def __init__(self):
        self.writes = []
        self.jobs_table_missing = False

    def table(self, name):
        return FakeQuery(self, name)


def test_homepage_demo_defaults_to_ten_auto_rendered_pages_and_one_hour(monkeypatch):
    database = FakeDatabase()
    scheduled = []
    monkeypatch.setattr(cfg, "sb", database)
    monkeypatch.setattr(trial, "rate_limit_check", lambda *_args: None)
    monkeypatch.setattr(trial, "verify_user", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trial, "is_url_safe", lambda _url: (True, ""))
    monkeypatch.setattr(trial, "schedule_indexing_job", lambda *args, **kwargs: scheduled.append((args, kwargs)) or "job-1")

    app = FastAPI()
    app.include_router(trial.router)
    before = datetime.now(timezone.utc)
    response = TestClient(app).post("/api/trial/start", data={"url": "https://example.com/ru"})
    after = datetime.now(timezone.utc)

    assert response.status_code == 200
    data = response.json()
    expiry = datetime.fromisoformat(data["expires_at"])
    assert before + timedelta(hours=1) <= expiry <= after + timedelta(hours=1)
    assert data["website_name"] == "example.com"
    site_row = database.writes[0][2]
    assert site_row["settings"]["quick_demo"] is True
    assert site_row["language"] == "en"  # Updated from the first fetched page.
    args, options = scheduled[0]
    assert args[:3] == (17, "https://example.com/ru", 10)
    assert args[4] is False  # Adaptive rendering chooses Playwright only when needed.
    assert options["auto_language"] is True


def test_trial_start_returns_json_and_removes_site_when_queueing_fails(monkeypatch):
    database = FakeDatabase()
    monkeypatch.setattr(cfg, "sb", database)
    monkeypatch.setattr(trial, "rate_limit_check", lambda *_args: None)
    monkeypatch.setattr(trial, "verify_user", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trial, "is_url_safe", lambda _url: (True, ""))

    def fail_queue(*_args, **_kwargs):
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(trial, "schedule_indexing_job", fail_queue)
    app = FastAPI()
    app.include_router(trial.router)

    response = TestClient(app).post("/api/trial/start", data={"url": "https://example.com"})

    assert response.status_code == 503
    assert response.json()["error"] == "Could not start indexing. Please try again later."
    assert ("sites", "delete", None) in database.writes


def test_trial_start_succeeds_when_jobs_migration_is_missing(monkeypatch):
    database = FakeDatabase()
    database.jobs_table_missing = True
    monkeypatch.setattr(cfg, "sb", database)
    monkeypatch.setattr(cfg, "trial_progress", {})
    monkeypatch.setattr(trial, "rate_limit_check", lambda *_args: None)
    monkeypatch.setattr(trial, "verify_user", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trial, "is_url_safe", lambda _url: (True, ""))
    monkeypatch.setattr(trial, "run_trial_indexing", lambda *_args, **_kwargs: None)
    app = FastAPI()
    app.include_router(trial.router)

    response = TestClient(app).post("/api/trial/start", data={"url": "https://tenroman.com"})

    assert response.status_code == 200
    assert response.json()["site_id"] == 17
    assert cfg.trial_progress[17]["_volatile"] is True
    assert cfg.trial_progress[17]["status"] == "queued"


def test_trial_crawl_keeps_query_locale_and_publishes_page_language(monkeypatch):
    database = FakeDatabase()
    fetched = []
    pages = {
        "https://example.com/?lang=ru": (
            '<html lang="ru"><head><title>Главная | Пример</title>'
            '<meta property="og:site_name" content="Пример"></head><body>'
            '<p>Русская версия сайта рассказывает о продуктах и услугах нашей компании. '
            'Здесь можно узнать о командах, проектах, партнерах и контактах организации.</p>'
            '<a href="/about">О компании</a></body></html>'
        ),
        "https://example.com/about?lang=ru": (
            '<html lang="ru"><head><title>О компании | Пример</title></head><body>'
            '<p>Наша компания работает над новыми решениями и рассказывает о своих '
            'сотрудниках, услугах, достижениях, истории и возможностях сотрудничества.</p>'
            '</body></html>'
        ),
    }

    class FakeRenderer:
        def fetch(self, url):
            fetched.append(url)
            return pages[url], 200

        def close(self):
            pass

    completed = []
    monkeypatch.setattr(cfg, "sb", database)
    monkeypatch.setattr(cfg, "trial_progress", {17: {"status": "queued"}})
    monkeypatch.setattr(cfg, "_site_lang_cache", {17: ("en", 0)})
    monkeypatch.setattr(trial, "AdaptiveRenderer", FakeRenderer)
    monkeypatch.setattr(trial, "start_indexing_job", lambda *_args: None)
    monkeypatch.setattr(trial, "update_indexing_job", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trial, "complete_indexing_job", lambda *_args, **_kwargs: completed.append(True))
    monkeypatch.setattr(trial, "fail_indexing_job", lambda *_args: None)
    monkeypatch.setattr(trial, "chunk_text", lambda _text: [])
    monkeypatch.setattr(trial.time, "sleep", lambda _seconds: None)

    trial.run_trial_indexing(
        17, "https://example.com/?lang=ru", 2, [], job_id="job-1",
        trial_metadata=True, auto_language=True,
    )

    assert fetched == list(pages)
    assert cfg.trial_progress[17]["website_name"] == "Пример"
    assert cfg.trial_progress[17]["language"] == "ru"
    assert ("sites", "update", {"language": "ru"}) in database.writes
    assert 17 not in cfg._site_lang_cache
    assert completed
