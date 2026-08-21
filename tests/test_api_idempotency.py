"""Idempotency-Key contract: replays return the SAME stored 202 and create no rows.

Double-click safety is the week-2 exit gate: two identical POSTs yield one Job;
a reused key with a different body is a 409; a second click with a fresh key
hits the state machine and 409s instead of double-generating.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from conftest import make_video_plan
from fastapi.testclient import TestClient

from app.db import Base, SessionLocal, engine
from app.main import app
from app.models import (
    Campaign,
    Creative,
    IdempotencyKey,
    Job,
    PublishTarget,
    Rendition,
    ScriptVersion,
    User,
)
from app.states import CreativeState, Role

pytestmark = pytest.mark.usefixtures("api_schema")


@pytest.fixture(scope="module")
def api_schema() -> Iterator[None]:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.add(
            User(
                email="pub-idem@t.local",
                name="Publisher",
                role=Role.PUBLISHER.value,
                password_hash="x",
            )
        )
        db.commit()
        yield
    finally:
        db.close()


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def db() -> Iterator:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _seed_approved_creative(db) -> str:
    campaign = Campaign(name="Idem", brief="b")
    db.add(campaign)
    db.flush()
    creative = Creative(
        campaign_id=campaign.id,
        topic_title="Idempotency test",
        state=CreativeState.SCRIPT_APPROVED.value,
        cost_cap_usd=6.0,
    )
    db.add(creative)
    db.flush()
    db.add(
        ScriptVersion(
            creative_id=creative.id,
            version=1,
            video_plan=make_video_plan(),
            is_approved=True,
            approved_at=datetime.now(UTC),
        )
    )
    db.commit()
    return creative.id


def _seed_publishable_creative(db) -> tuple[str, str]:
    creative_id = _seed_approved_creative(db)
    creative = db.get(Creative, creative_id)
    creative.state = CreativeState.FINAL_APPROVED.value
    rendition = Rendition(
        creative_id=creative_id, locale="vi", title="Title vi", is_approved=True
    )
    db.add(rendition)
    db.commit()
    return creative_id, rendition.id


def _job_count(db, kind: str) -> int:
    return db.query(Job).filter(Job.kind == kind).count()


def test_generate_double_post_same_key_returns_identical_response_and_no_new_job(
    client: TestClient, db
) -> None:
    creative_id = _seed_approved_creative(db)
    key = f"gen-{uuid.uuid4()}"
    jobs_before = _job_count(db, "generate")

    first = client.post(
        f"/api/v1/creatives/{creative_id}/generate",
        json={},
        headers={"Idempotency-Key": key},
    )
    assert first.status_code == 202
    second = client.post(
        f"/api/v1/creatives/{creative_id}/generate",
        json={},
        headers={"Idempotency-Key": key},
    )
    assert second.status_code == 202
    assert second.json() == first.json()  # identical stored 202, same job_id
    assert second.json()["job_id"] == first.json()["job_id"]

    db.expire_all()
    assert _job_count(db, "generate") == jobs_before + 1  # NO new Job row
    stored = (
        db.query(IdempotencyKey).filter(IdempotencyKey.key == key).all()
    )
    assert len(stored) == 1


def test_generate_same_key_different_body_conflicts(client: TestClient, db) -> None:
    creative_id = _seed_approved_creative(db)
    key = f"gen-{uuid.uuid4()}"
    first = client.post(
        f"/api/v1/creatives/{creative_id}/generate",
        json={"scene_ids": ["a"]},
        headers={"Idempotency-Key": key},
    )
    assert first.status_code == 202
    conflict = client.post(
        f"/api/v1/creatives/{creative_id}/generate",
        json={"scene_ids": ["b"]},
        headers={"Idempotency-Key": key},
    )
    assert conflict.status_code == 409
    payload = conflict.json()
    assert payload["code"] == "idempotency_key_conflict"
    assert set(payload) >= {"code", "message", "retryable", "details", "correlation_id"}


def test_generate_double_click_with_fresh_key_hits_state_machine(
    client: TestClient, db
) -> None:
    creative_id = _seed_approved_creative(db)
    jobs_before = _job_count(db, "generate")
    first = client.post(
        f"/api/v1/creatives/{creative_id}/generate",
        json={},
        headers={"Idempotency-Key": f"gen-{uuid.uuid4()}"},
    )
    assert first.status_code == 202
    second = client.post(
        f"/api/v1/creatives/{creative_id}/generate",
        json={},
        headers={"Idempotency-Key": f"gen-{uuid.uuid4()}"},
    )
    assert second.status_code == 409  # GENERATING has no GENERATING edge
    assert second.json()["code"] == "invalid_transition"
    db.expire_all()
    assert _job_count(db, "generate") == jobs_before + 1


def test_same_key_is_scoped_per_endpoint(client: TestClient, db) -> None:
    """One key on creative A's generate does not replay for creative B."""
    creative_a = _seed_approved_creative(db)
    creative_b = _seed_approved_creative(db)
    key = f"gen-{uuid.uuid4()}"
    first = client.post(
        f"/api/v1/creatives/{creative_a}/generate", json={}, headers={"Idempotency-Key": key}
    )
    second = client.post(
        f"/api/v1/creatives/{creative_b}/generate", json={}, headers={"Idempotency-Key": key}
    )
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["job_id"] != second.json()["job_id"]


def test_publications_double_post_same_key_creates_no_new_targets(
    client: TestClient, db
) -> None:
    creative_id, rendition_id = _seed_publishable_creative(db)
    key = f"pub-{uuid.uuid4()}"
    body = {
        "creative_id": creative_id,
        "targets": [{"rendition_id": rendition_id, "platform": "youtube"}],
    }
    first = client.post("/api/v1/publications", json=body, headers={"Idempotency-Key": key})
    assert first.status_code == 202, first.text
    targets_after_first = (
        db.query(PublishTarget).filter(PublishTarget.creative_id == creative_id).count()
    )
    second = client.post("/api/v1/publications", json=body, headers={"Idempotency-Key": key})
    assert second.status_code == 202
    assert second.json() == first.json()
    assert second.json()["target_ids"] == first.json()["target_ids"]

    db.expire_all()
    targets_after_second = (
        db.query(PublishTarget).filter(PublishTarget.creative_id == creative_id).count()
    )
    assert targets_after_second == targets_after_first == 1
    assert _job_count(db, "publish") >= 1
    publish_jobs = (
        db.query(Job)
        .filter(Job.kind == "publish", Job.creative_id == creative_id)
        .count()
    )
    assert publish_jobs == 1


def test_publications_requires_idempotency_key(client: TestClient, db) -> None:
    creative_id, rendition_id = _seed_publishable_creative(db)
    response = client.post(
        "/api/v1/publications",
        json={
            "creative_id": creative_id,
            "targets": [{"rendition_id": rendition_id, "platform": "youtube"}],
        },
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_failed"


def test_webhook_duplicate_event_stored_once(client: TestClient, db) -> None:
    creative_id, rendition_id = _seed_publishable_creative(db)
    target = PublishTarget(
        creative_id=creative_id,
        rendition_id=rendition_id,
        platform="facebook",
        remote_post_id="fb-idem-1",
    )
    db.add(target)
    db.commit()
    event = {"event_id": "fb-evt-1", "remote_post_id": "fb-idem-1", "status": "published"}
    first = client.post("/webhooks/facebook", json=event)
    second = client.post("/webhooks/facebook", json=event)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    rows = (
        db.query(IdempotencyKey)
        .filter(IdempotencyKey.endpoint == "webhook:facebook", IdempotencyKey.key == "fb-evt-1")
        .count()
    )
    assert rows == 1
