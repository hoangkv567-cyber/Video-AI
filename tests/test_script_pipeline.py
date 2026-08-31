"""Vertical slice: selected discovery topic -> persisted source -> script Job."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from conftest import make_video_plan
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.ai.base import FakeScriptProvider
from app.db import Base, SessionLocal, engine
from app.main import app
from app.models import (
    AuditEvent,
    Campaign,
    CostEvent,
    Creative,
    Job,
    ScriptVersion,
    Source,
    User,
)
from app.states import CreativeState, JobStatus, Role
from app.workers.tasks import run_write_script


def _topic(title: str, suffix: str = "a") -> dict:
    accessed = datetime.now(UTC).isoformat()
    return {
        "title": title,
        "summary": f"Summary for {title}",
        "category": "ai",
        "score": 0.9,
        "sources": [
            {
                "url": f"https://openai.com/research/{suffix}",
                "title": "Official source",
                "publisher": "OpenAI",
                "is_official": True,
                "accessed_at": accessed,
            },
            {
                "url": f"https://reuters.com/technology/{suffix}",
                "title": "Independent report",
                "publisher": "Reuters",
                "is_official": False,
                "accessed_at": accessed,
            },
        ],
    }


@pytest.fixture(scope="module")
def api_schema() -> Iterator[str]:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        editor = User(
            email="script-editor@t.local",
            name="Script editor",
            role=Role.EDITOR.value,
            password_hash="x",
        )
        db.add(editor)
        db.commit()
        yield editor.id
    finally:
        db.close()


@pytest.fixture()
def client(api_schema: str) -> Iterator[TestClient]:
    with TestClient(app, headers={"X-User-Id": api_schema}) as test_client:
        yield test_client


@pytest.fixture()
def api_db(api_schema: str) -> Iterator[Session]:
    del api_schema
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _seed_discovery(db: Session, *, status: str = JobStatus.SUCCEEDED.value) -> Job:
    job = Job(
        kind="discover",
        queue="ai",
        status=status,
        payload={"brief": "Newest AI infrastructure", "category": "ai"},
        result={"topics": [_topic("Topic one", "one"), _topic("Topic two", "two")]},
    )
    db.add(job)
    db.commit()
    return job


def test_select_topic_creates_researched_creative_and_script_job(
    client: TestClient, api_db: Session, api_schema: str
) -> None:
    discovery = _seed_discovery(api_db)
    response = client.post(
        f"/api/v1/discoveries/{discovery.id}/select",
        json={"topic_index": 0, "campaign_name": "AI daily", "mode": "manual"},
        headers={
            "X-User-Id": api_schema,
            "Idempotency-Key": f"select-{uuid.uuid4()}",
        },
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["kind"] == "script"
    assert payload["selected_topic"] == {"index": 0, "title": "Topic one"}
    api_db.expire_all()
    creative = api_db.get(Creative, payload["creative_id"])
    script_job = api_db.get(Job, payload["job_id"])
    assert creative is not None and creative.state == CreativeState.RESEARCHED.value
    assert creative.topic_title == "Topic one"
    assert script_job is not None and script_job.kind == "script" and script_job.queue == "ai"
    sources = api_db.query(Source).filter(Source.creative_id == creative.id).all()
    assert len(sources) == 2
    assert {source.publisher for source in sources} == {"OpenAI", "Reuters"}
    assert all(source.citation for source in sources)
    audit = api_db.query(AuditEvent).filter_by(action="topic_selected", entity_id=creative.id).one()
    assert audit.data["script_job_id"] == script_job.id


def test_select_topic_replay_is_exact_and_creates_no_duplicates(
    client: TestClient, api_db: Session, api_schema: str
) -> None:
    discovery = _seed_discovery(api_db)
    key = f"select-{uuid.uuid4()}"
    body = {"topic_index": 1, "campaign_name": "Replay campaign", "mode": "manual"}
    headers = {"X-User-Id": api_schema, "Idempotency-Key": key}
    before = {
        "campaigns": api_db.query(Campaign).count(),
        "creatives": api_db.query(Creative).count(),
        "jobs": api_db.query(Job).count(),
    }

    first = client.post(f"/api/v1/discoveries/{discovery.id}/select", json=body, headers=headers)
    second = client.post(f"/api/v1/discoveries/{discovery.id}/select", json=body, headers=headers)

    assert first.status_code == second.status_code == 202
    assert second.json() == first.json()
    api_db.expire_all()
    assert api_db.query(Campaign).count() == before["campaigns"] + 1
    assert api_db.query(Creative).count() == before["creatives"] + 1
    # One script Job; the discovery was inserted before the baseline count.
    assert api_db.query(Job).count() == before["jobs"] + 1


def test_select_topic_same_key_different_index_conflicts(
    client: TestClient, api_db: Session, api_schema: str
) -> None:
    discovery = _seed_discovery(api_db)
    key = f"select-{uuid.uuid4()}"
    headers = {"X-User-Id": api_schema, "Idempotency-Key": key}
    first = client.post(
        f"/api/v1/discoveries/{discovery.id}/select",
        json={"topic_index": 0},
        headers=headers,
    )
    conflict = client.post(
        f"/api/v1/discoveries/{discovery.id}/select",
        json={"topic_index": 1},
        headers=headers,
    )
    assert first.status_code == 202
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_key_conflict"


def test_select_topic_rejects_tampered_client_snapshot_and_unready_job(
    client: TestClient, api_db: Session, api_schema: str
) -> None:
    discovery = _seed_discovery(api_db, status=JobStatus.RUNNING.value)
    headers = {
        "X-User-Id": api_schema,
        "Idempotency-Key": f"select-{uuid.uuid4()}",
    }
    unready = client.post(
        f"/api/v1/discoveries/{discovery.id}/select",
        json={"topic_index": 0},
        headers=headers,
    )
    assert unready.status_code == 409
    assert unready.json()["code"] == "discovery_not_ready"
    assert unready.json()["retryable"] is True

    discovery.status = JobStatus.SUCCEEDED.value
    api_db.commit()
    tampered = client.post(
        f"/api/v1/discoveries/{discovery.id}/select",
        json={"topic_index": 0, "title": "client supplied", "sources": []},
        headers={**headers, "Idempotency-Key": f"select-{uuid.uuid4()}"},
    )
    assert tampered.status_code == 422


def _seed_researched_creative(db: Session, *, mode: str = "manual", cap: float = 6.0) -> Creative:
    campaign = Campaign(name="Worker campaign", brief="AI infrastructure", category="ai")
    db.add(campaign)
    db.flush()
    creative = Creative(
        campaign_id=campaign.id,
        state=CreativeState.RESEARCHED.value,
        topic_title="New AI model launch",
        angle="Why it matters",
        mode=mode,
        cost_cap_usd=cap,
    )
    db.add(creative)
    db.flush()
    now = datetime.now(UTC)
    db.add_all(
        [
            Source(
                creative_id=creative.id,
                url="https://openai.com/blog/new-model",
                title="Official",
                publisher="OpenAI",
                is_official=True,
                accessed_at=now,
                citation="Official",
            ),
            Source(
                creative_id=creative.id,
                url="https://techcrunch.com/2026/08/20/new-model",
                title="Coverage",
                publisher="TechCrunch",
                accessed_at=now,
                citation="Coverage",
            ),
        ]
    )
    db.commit()
    return creative


def _script_job(db: Session, creative: Creative) -> Job:
    job = Job(kind="script", queue="ai", creative_id=creative.id, payload={})
    db.add(job)
    db.commit()
    return job


def test_script_worker_persists_version_projection_and_manual_state(
    db_session: Session,
) -> None:
    creative = _seed_researched_creative(db_session)
    job = _script_job(db_session, creative)
    provider = FakeScriptProvider(make_video_plan())

    result = run_write_script(db_session, job.id, provider)

    assert "error" not in result
    assert result["creative_state"] == CreativeState.SCRIPT_READY.value
    assert result["auto_approved"] is False
    assert result["generate_job_id"] is None
    version = db_session.get(ScriptVersion, result["script_version_id"])
    assert version is not None and not version.is_approved and version.version == 1
    assert version.video_plan["topic"] == creative.topic_title
    assert [source["url"] for source in version.video_plan["sources"]] == [
        "https://openai.com/blog/new-model",
        "https://techcrunch.com/2026/08/20/new-model",
    ]
    projections = (
        db_session.query(CostEvent).filter_by(creative_id=creative.id, projected=True).all()
    )
    assert projections
    assert db_session.get(Job, job.id).status == JobStatus.SUCCEEDED.value


def test_script_worker_auto_approves_and_enqueues_generation(db_session: Session) -> None:
    creative = _seed_researched_creative(db_session, mode="auto")
    job = _script_job(db_session, creative)

    result = run_write_script(
        db_session,
        job.id,
        FakeScriptProvider(make_video_plan()),
        preflight_issues=[],
    )

    assert result["auto_approved"] is True
    assert result["creative_state"] == CreativeState.GENERATING.value
    version = db_session.get(ScriptVersion, result["script_version_id"])
    generate_job = db_session.get(Job, result["generate_job_id"])
    assert version is not None and version.is_approved and version.approved_by is None
    assert generate_job is not None and generate_job.kind == "generate"
    assert generate_job.creative_id == creative.id


def test_script_worker_auto_stops_for_unsourced_scene(db_session: Session) -> None:
    creative = _seed_researched_creative(db_session, mode="auto")
    job = _script_job(db_session, creative)
    plan = make_video_plan()
    plan["scenes"][3]["fact_ids"] = []

    result = run_write_script(db_session, job.id, FakeScriptProvider(plan))

    assert result["auto_mode_allowed"] is False
    assert result["auto_approved"] is False
    assert result["creative_state"] == CreativeState.SCRIPT_READY.value
    assert result["generate_job_id"] is None


def test_script_worker_redelivery_returns_stored_result_without_provider_call(
    db_session: Session,
) -> None:
    creative = _seed_researched_creative(db_session)
    job = _script_job(db_session, creative)
    first_provider = FakeScriptProvider(make_video_plan())
    first = run_write_script(db_session, job.id, first_provider)
    attempts = db_session.get(Job, job.id).attempts
    second_provider = FakeScriptProvider(make_video_plan())

    second = run_write_script(db_session, job.id, second_provider)

    assert second == first
    assert second_provider.generate_calls == 0
    assert db_session.get(Job, job.id).attempts == attempts
    assert db_session.query(ScriptVersion).filter_by(creative_id=creative.id).count() == 1


def test_script_worker_schema_and_cost_failures_leave_no_partial_rows(
    db_session: Session,
) -> None:
    invalid_creative = _seed_researched_creative(db_session)
    invalid_job = _script_job(db_session, invalid_creative)
    invalid = run_write_script(
        db_session,
        invalid_job.id,
        FakeScriptProvider({"schema_version": "1.0", "topic": "bad"}),
    )
    assert invalid["error"]["code"] == "validation_failed"
    assert invalid_creative.state == CreativeState.RESEARCHED.value
    assert db_session.query(ScriptVersion).filter_by(creative_id=invalid_creative.id).count() == 0

    capped_creative = _seed_researched_creative(db_session, cap=0.1)
    capped_job = _script_job(db_session, capped_creative)
    capped = run_write_script(db_session, capped_job.id, FakeScriptProvider(make_video_plan()))
    assert capped["error"]["code"] == "cost_cap_exceeded"
    assert capped_creative.state == CreativeState.RESEARCHED.value
    assert db_session.query(CostEvent).filter_by(creative_id=capped_creative.id).count() == 0
    assert db_session.query(ScriptVersion).filter_by(creative_id=capped_creative.id).count() == 0
