"""API contract tests: happy paths, role gates, error envelopes, OAuth, webhooks.

Runs fully offline over TestClient + SQLite (env pinned by tests/conftest.py
before any app import). Auth uses the dev/test X-User-Id header path.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from conftest import make_video_plan
from fastapi.testclient import TestClient

from app.api import oauth as oauth_module
from app.db import Base, SessionLocal, engine
from app.main import app
from app.models import (
    Campaign,
    ConnectedAccount,
    Creative,
    Job,
    PublishTarget,
    Rendition,
    ScriptVersion,
    User,
)
from app.publishing.crypto import decrypt_credentials
from app.states import Capability, CreativeState, PublishTargetStatus, Role

pytestmark = pytest.mark.usefixtures("api_schema")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_schema() -> Iterator[dict[str, str]]:
    """Fresh schema on the app engine plus one user per role."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        users = {
            "admin": User(email="admin@t.local", name="Admin", role=Role.ADMIN.value,
                          password_hash="x"),
            "editor": User(email="editor@t.local", name="Editor", role=Role.EDITOR.value,
                           password_hash="x"),
            "publisher": User(email="pub@t.local", name="Publisher",
                              role=Role.PUBLISHER.value, password_hash="x"),
        }
        db.add_all(users.values())
        db.commit()
        yield {name: user.id for name, user in users.items()}
    finally:
        db.close()


@pytest.fixture()
def users(api_schema: dict[str, str]) -> dict[str, str]:
    return api_schema


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


def _create_creative_via_api(client: TestClient, **overrides: object) -> dict:
    plan = make_video_plan()
    body = {
        "topic_title": "New AI model launch",
        "brief": "Emerging AI news",
        "sources": [
            {"url": s["url"], "title": s["title"], "is_official": s["is_official"]}
            for s in plan["sources"]
        ],
        "video_plan": plan,
    }
    body.update(overrides)
    response = client.post("/api/v1/creatives", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _approve_script(client: TestClient, script_version_id: str, users: dict[str, str]) -> dict:
    response = client.patch(
        f"/api/v1/scripts/{script_version_id}",
        json={"approve": True},
        headers={"X-User-Id": users["publisher"]},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _assert_envelope(payload: dict, code: str) -> None:
    assert payload["code"] == code
    assert set(payload) >= {"code", "message", "retryable", "details", "correlation_id"}


# ---------------------------------------------------------------------------
# /topics/discover and /jobs
# ---------------------------------------------------------------------------


def test_discover_returns_202_job(client: TestClient, db) -> None:
    response = client.post("/api/v1/topics/discover", json={"brief": "AI hardware news"})
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "QUEUED"
    job = db.get(Job, payload["job_id"])
    assert job is not None
    assert job.kind == "discover"
    assert job.queue == "ai"
    assert job.payload["brief"] == "AI hardware news"

    got = client.get(f"/api/v1/jobs/{payload['job_id']}")
    assert got.status_code == 200
    assert got.json()["status"] == "QUEUED"


def test_get_job_unknown_returns_404_envelope(client: TestClient) -> None:
    response = client.get(f"/api/v1/jobs/{uuid.uuid4()}")
    assert response.status_code == 404
    _assert_envelope(response.json(), "not_found")


# ---------------------------------------------------------------------------
# /creatives
# ---------------------------------------------------------------------------


def test_create_creative_with_sources_and_plan(client: TestClient, db) -> None:
    payload = _create_creative_via_api(client)
    assert payload["state"] == CreativeState.SCRIPT_READY.value
    assert payload["script_version_id"]
    assert all(not issue["blocking"] for issue in payload["issues"] or [])
    creative = db.get(Creative, payload["id"])
    assert creative is not None
    assert len(creative.sources) == 2
    assert creative.cost_cap_usd <= 6.0


def test_create_creative_invalid_plan_422(client: TestClient) -> None:
    bad_plan = make_video_plan()
    bad_plan["scenes"] = bad_plan["scenes"][:2]  # must be exactly 5
    response = client.post(
        "/api/v1/creatives",
        json={"topic_title": "Bad plan", "video_plan": bad_plan},
    )
    assert response.status_code == 422
    _assert_envelope(response.json(), "validation_failed")


# ---------------------------------------------------------------------------
# PATCH /scripts/{id}
# ---------------------------------------------------------------------------


def test_patch_script_creates_version_2_and_never_mutates_v1(
    client: TestClient, db, users: dict[str, str]
) -> None:
    created = _create_creative_via_api(client)
    v1_id = created["script_version_id"]
    v1_plan = db.get(ScriptVersion, v1_id).video_plan

    edited = make_video_plan()
    edited["locales"]["vi"]["title"] = "Tiêu đề đã chỉnh sửa"
    response = client.patch(
        f"/api/v1/scripts/{v1_id}",
        json={"video_plan": edited},
        headers={"X-User-Id": users["editor"]},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["version"] == 2
    assert payload["script_version_id"] != v1_id
    assert payload["creative_state"] == CreativeState.SCRIPT_READY.value

    db.expire_all()
    v1 = db.get(ScriptVersion, v1_id)
    v2 = db.get(ScriptVersion, payload["script_version_id"])
    assert v1.video_plan == v1_plan  # untouched
    assert v2.video_plan["locales"]["vi"]["title"] == "Tiêu đề đã chỉnh sửa"
    assert not v1.is_approved
    assert not v2.is_approved


def test_script_approve_requires_publisher_role(
    client: TestClient, users: dict[str, str]
) -> None:
    created = _create_creative_via_api(client)
    response = client.patch(
        f"/api/v1/scripts/{created['script_version_id']}",
        json={"approve": True},
        headers={"X-User-Id": users["editor"]},
    )
    assert response.status_code == 403
    _assert_envelope(response.json(), "forbidden")


def test_script_approve_blocked_while_blocking_issues_exist(
    client: TestClient, users: dict[str, str]
) -> None:
    plan = make_video_plan()
    plan["sources"] = plan["sources"][:1]  # sources_min is a blocking issue
    for fact in plan["facts"]:
        fact["source_ids"] = ["src1"]
    created = _create_creative_via_api(client, video_plan=plan)
    response = client.patch(
        f"/api/v1/scripts/{created['script_version_id']}",
        json={"approve": True},
        headers={"X-User-Id": users["publisher"]},
    )
    assert response.status_code == 422
    payload = response.json()
    _assert_envelope(payload, "policy_blocked")
    assert any(i["code"] == "sources_min" for i in payload["details"]["issues"])


def test_script_approve_happy_path(client: TestClient, db, users: dict[str, str]) -> None:
    created = _create_creative_via_api(client)
    payload = _approve_script(client, created["script_version_id"], users)
    assert payload["is_approved"] is True
    assert payload["creative_state"] == CreativeState.SCRIPT_APPROVED.value
    db.expire_all()
    version = db.get(ScriptVersion, created["script_version_id"])
    assert version.is_approved
    assert version.approved_by == users["publisher"]


# ---------------------------------------------------------------------------
# POST /creatives/{id}/generate
# ---------------------------------------------------------------------------


def test_generate_requires_idempotency_key(client: TestClient, users: dict[str, str]) -> None:
    created = _create_creative_via_api(client)
    _approve_script(client, created["script_version_id"], users)
    response = client.post(f"/api/v1/creatives/{created['id']}/generate", json={})
    assert response.status_code == 422
    _assert_envelope(response.json(), "validation_failed")


def test_generate_returns_202_and_moves_to_generating(
    client: TestClient, db, users: dict[str, str]
) -> None:
    created = _create_creative_via_api(client)
    _approve_script(client, created["script_version_id"], users)
    response = client.post(
        f"/api/v1/creatives/{created['id']}/generate",
        json={},
        headers={"Idempotency-Key": f"gen-{uuid.uuid4()}"},
    )
    assert response.status_code == 202
    payload = response.json()
    db.expire_all()
    creative = db.get(Creative, created["id"])
    assert creative.state == CreativeState.GENERATING.value
    job = db.get(Job, payload["job_id"])
    assert job.kind == "generate"
    assert job.queue == "ai"
    assert job.creative_id == created["id"]


def test_generate_invalid_transition_409(client: TestClient) -> None:
    response = client.post(
        "/api/v1/creatives",
        json={"topic_title": "Draft only"},  # stays in DRAFT
    )
    creative_id = response.json()["id"]
    result = client.post(
        f"/api/v1/creatives/{creative_id}/generate",
        json={},
        headers={"Idempotency-Key": f"gen-{uuid.uuid4()}"},
    )
    assert result.status_code == 409
    payload = result.json()
    _assert_envelope(payload, "invalid_transition")
    assert payload["details"] == {"current": "DRAFT", "target": "GENERATING"}


def test_generate_unknown_creative_404(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/creatives/{uuid.uuid4()}/generate",
        json={},
        headers={"Idempotency-Key": f"gen-{uuid.uuid4()}"},
    )
    assert response.status_code == 404
    _assert_envelope(response.json(), "not_found")


# ---------------------------------------------------------------------------
# POST /renditions/{id}/approve
# ---------------------------------------------------------------------------


def _seed_ready_creative(db, users: dict[str, str]) -> tuple[str, str, str]:
    """Creative in READY with an approved script and two renditions."""
    campaign = Campaign(name="Seeded", brief="b")
    db.add(campaign)
    db.flush()
    creative = Creative(
        campaign_id=campaign.id,
        topic_title="Seeded topic",
        state=CreativeState.READY.value,
        cost_cap_usd=6.0,
    )
    db.add(creative)
    db.flush()
    version = ScriptVersion(
        creative_id=creative.id,
        version=1,
        video_plan=make_video_plan(),
        is_approved=True,
        approved_by=users["publisher"],
        approved_at=datetime.now(UTC),
    )
    db.add(version)
    renditions = [
        Rendition(creative_id=creative.id, locale=locale, title=f"Title {locale}")
        for locale in ("vi", "en")
    ]
    db.add_all(renditions)
    db.commit()
    return creative.id, renditions[0].id, renditions[1].id


def test_rendition_approve_role_gate(client: TestClient, db, users: dict[str, str]) -> None:
    _, rendition_id, _ = _seed_ready_creative(db, users)
    response = client.post(
        f"/api/v1/renditions/{rendition_id}/approve",
        headers={"X-User-Id": users["editor"]},
    )
    assert response.status_code == 403
    _assert_envelope(response.json(), "forbidden")

    anonymous = client.post(f"/api/v1/renditions/{rendition_id}/approve")
    assert anonymous.status_code == 401
    _assert_envelope(anonymous.json(), "unauthorized")


def test_rendition_approve_blocked_on_failed_qc(
    client: TestClient, db, users: dict[str, str]
) -> None:
    _, rendition_id, _ = _seed_ready_creative(db, users)
    rendition = db.get(Rendition, rendition_id)
    rendition.qc_report = {"passed": False, "checks": []}
    db.commit()
    response = client.post(
        f"/api/v1/renditions/{rendition_id}/approve",
        headers={"X-User-Id": users["publisher"]},
    )
    assert response.status_code == 422
    _assert_envelope(response.json(), "policy_blocked")


def test_rendition_approve_both_locales_finalizes_creative(
    client: TestClient, db, users: dict[str, str]
) -> None:
    creative_id, vi_id, en_id = _seed_ready_creative(db, users)
    first = client.post(
        f"/api/v1/renditions/{vi_id}/approve", headers={"X-User-Id": users["publisher"]}
    )
    assert first.status_code == 200
    assert first.json()["creative_state"] == CreativeState.READY.value  # one of two approved

    second = client.post(
        f"/api/v1/renditions/{en_id}/approve", headers={"X-User-Id": users["admin"]}
    )
    assert second.status_code == 200
    assert second.json()["creative_state"] == CreativeState.FINAL_APPROVED.value
    db.expire_all()
    assert db.get(Creative, creative_id).state == CreativeState.FINAL_APPROVED.value


# ---------------------------------------------------------------------------
# POST /publications
# ---------------------------------------------------------------------------


def _finalize_creative(db, creative_id: str) -> list[str]:
    creative = db.get(Creative, creative_id)
    creative.state = CreativeState.FINAL_APPROVED.value
    rendition_ids = []
    for rendition in db.query(Rendition).filter(Rendition.creative_id == creative_id).all():
        rendition.is_approved = True
        rendition_ids.append(rendition.id)
    db.commit()
    return rendition_ids


def test_publications_creates_targets_and_schedules(
    client: TestClient, db, users: dict[str, str]
) -> None:
    creative_id, vi_id, en_id = _seed_ready_creative(db, users)
    _finalize_creative(db, creative_id)
    scheduled_at = (datetime.now(UTC) + timedelta(hours=3)).isoformat()
    response = client.post(
        "/api/v1/publications",
        json={
            "creative_id": creative_id,
            "targets": [
                {"rendition_id": vi_id, "platform": "youtube", "scheduled_at": scheduled_at},
                {"rendition_id": en_id, "platform": "facebook"},
            ],
        },
        headers={"Idempotency-Key": f"pub-{uuid.uuid4()}"},
    )
    assert response.status_code == 202, response.text
    payload = response.json()
    assert len(payload["target_ids"]) == 2
    db.expire_all()
    assert db.get(Creative, creative_id).state == CreativeState.SCHEDULED.value
    job = db.get(Job, payload["job_id"])
    assert job.kind == "publish"
    assert job.queue == "publish"
    targets = db.query(PublishTarget).filter(PublishTarget.creative_id == creative_id).all()
    assert {t.platform for t in targets} == {"youtube", "facebook"}
    assert all(t.status == PublishTargetStatus.PENDING.value for t in targets)


def test_publications_manual_mode_requires_approved_renditions(
    client: TestClient, db, users: dict[str, str]
) -> None:
    creative_id, vi_id, _ = _seed_ready_creative(db, users)
    creative = db.get(Creative, creative_id)
    creative.state = CreativeState.FINAL_APPROVED.value
    db.commit()  # renditions intentionally NOT approved
    response = client.post(
        "/api/v1/publications",
        json={"creative_id": creative_id, "targets": [{"rendition_id": vi_id, "platform": "youtube"}]},
        headers={"Idempotency-Key": f"pub-{uuid.uuid4()}"},
    )
    assert response.status_code == 422
    _assert_envelope(response.json(), "policy_blocked")


def test_publications_auto_mode_gates_on_account_capability(
    client: TestClient, db, users: dict[str, str]
) -> None:
    creative_id, vi_id, _ = _seed_ready_creative(db, users)
    _finalize_creative(db, creative_id)
    account = ConnectedAccount(platform="tiktok", capability=Capability.DRAFT.value)
    db.add(account)
    db.commit()

    blocked = client.post(
        "/api/v1/publications",
        json={
            "creative_id": creative_id,
            "mode": "auto",
            "targets": [
                {
                    "rendition_id": vi_id,
                    "platform": "tiktok",
                    "connected_account_id": account.id,
                }
            ],
        },
        headers={"Idempotency-Key": f"pub-{uuid.uuid4()}"},
    )
    assert blocked.status_code == 422
    payload = blocked.json()
    _assert_envelope(payload, "policy_blocked")
    assert payload["details"]["capability"] == Capability.DRAFT.value

    account.capability = Capability.DIRECT.value
    db.commit()
    allowed = client.post(
        "/api/v1/publications",
        json={
            "creative_id": creative_id,
            "mode": "auto",
            "targets": [
                {
                    "rendition_id": vi_id,
                    "platform": "tiktok",
                    "connected_account_id": account.id,
                }
            ],
        },
        headers={"Idempotency-Key": f"pub-{uuid.uuid4()}"},
    )
    assert allowed.status_code == 202, allowed.text


def test_publications_unknown_platform_422(
    client: TestClient, db, users: dict[str, str]
) -> None:
    creative_id, vi_id, _ = _seed_ready_creative(db, users)
    _finalize_creative(db, creative_id)
    response = client.post(
        "/api/v1/publications",
        json={"creative_id": creative_id, "targets": [{"rendition_id": vi_id, "platform": "myspace"}]},
        headers={"Idempotency-Key": f"pub-{uuid.uuid4()}"},
    )
    assert response.status_code == 422
    _assert_envelope(response.json(), "validation_failed")


# ---------------------------------------------------------------------------
# OAuth
# ---------------------------------------------------------------------------


def test_oauth_start_redirects_with_state_pkce_and_cookie(client: TestClient) -> None:
    response = client.get("/oauth/youtube/start", follow_redirects=False)
    assert response.status_code == 302
    location = response.headers["Location"]
    parsed = urlparse(location)
    assert parsed.hostname == "accounts.google.com"
    query = parse_qs(parsed.query)
    assert query["response_type"] == ["code"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["state"][0]
    assert query["code_challenge"][0]
    assert oauth_module.STATE_COOKIE in response.cookies


def test_oauth_start_unknown_platform_404(client: TestClient) -> None:
    response = client.get("/oauth/myspace/start", follow_redirects=False)
    assert response.status_code == 404
    _assert_envelope(response.json(), "not_found")


def test_oauth_callback_state_mismatch_rejected(client: TestClient) -> None:
    client.get("/oauth/youtube/start", follow_redirects=False)
    response = client.get(
        "/oauth/youtube/callback",
        params={"code": "abc", "state": "wrong-state"},
        follow_redirects=False,
    )
    assert response.status_code == 422
    _assert_envelope(response.json(), "oauth_state_mismatch")


def test_oauth_callback_stores_encrypted_credentials_and_probes(
    client: TestClient, db
) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.host == "oauth2.googleapis.com":
            return httpx.Response(
                200,
                json={
                    "access_token": "tok-secret-123",
                    "refresh_token": "ref-secret-456",
                    "expires_in": 3600,
                    "scope": "https://www.googleapis.com/auth/youtube.upload",
                },
            )
        return httpx.Response(200, json={"items": [{"id": "chan-1"}]})

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    app.dependency_overrides[oauth_module.get_http_client] = lambda: mock_client
    try:
        start = client.get("/oauth/youtube/start", follow_redirects=False)
        state = parse_qs(urlparse(start.headers["Location"]).query)["state"][0]
        callback = client.get(
            "/oauth/youtube/callback",
            params={"code": "auth-code", "state": state},
            follow_redirects=False,
        )
        assert callback.status_code == 303, callback.text
        assert callback.headers["Location"] == "/connections"
    finally:
        app.dependency_overrides.pop(oauth_module.get_http_client, None)
        mock_client.close()

    assert any("oauth2.googleapis.com" in url for url in calls)  # token exchange happened
    db.expire_all()
    account = (
        db.query(ConnectedAccount).filter(ConnectedAccount.platform == "youtube").one()
    )
    assert account.encrypted_credentials
    assert "tok-secret-123" not in account.encrypted_credentials  # never stored plaintext
    creds = decrypt_credentials(account.encrypted_credentials)
    assert creds["access_token"] == "tok-secret-123"
    assert account.token_expires_at is not None
    # scope includes youtube.upload -> the probe records SCHEDULE capability
    assert account.capability == Capability.SCHEDULE.value
    assert account.last_probe_at is not None


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------


def _seed_publish_target(db, users: dict[str, str], **kwargs: object) -> str:
    creative_id, vi_id, _ = _seed_ready_creative(db, users)
    target = PublishTarget(
        creative_id=creative_id,
        rendition_id=vi_id,
        platform="youtube",
        status=PublishTargetStatus.UPLOADING.value,
        remote_post_id="yt-123",
    )
    for key, value in kwargs.items():
        setattr(target, key, value)
    db.add(target)
    db.commit()
    return target.id


def test_webhook_updates_target_and_replay_changes_nothing(
    client: TestClient, db, users: dict[str, str]
) -> None:
    target_id = _seed_publish_target(db, users)
    event = {"event_id": f"evt-{uuid.uuid4()}", "remote_post_id": "yt-123", "status": "published"}

    first = client.post("/webhooks/youtube", json=event)
    assert first.status_code == 200
    assert first.json()["status"] == "processed"
    db.expire_all()
    target = db.get(PublishTarget, target_id)
    assert target.status == PublishTargetStatus.PUBLISHED.value

    # Simulate a later manual state change, then replay the SAME webhook:
    target.status = PublishTargetStatus.NEEDS_ACTION.value
    db.commit()
    replay = client.post("/webhooks/youtube", json=event)
    assert replay.status_code == 200
    assert replay.json() == first.json()  # identical stored response
    db.expire_all()
    assert (
        db.get(PublishTarget, target_id).status == PublishTargetStatus.NEEDS_ACTION.value
    )  # replay re-applied NOTHING


def test_webhook_unknown_provider_404(client: TestClient) -> None:
    response = client.post("/webhooks/myspace", json={"event_id": "e1"})
    assert response.status_code == 404


def test_webhook_signature_enforced_when_secret_configured(
    client: TestClient, db, users: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import get_settings

    _seed_publish_target(db, users, platform="facebook", remote_post_id="fb-9")
    monkeypatch.setattr(get_settings(), "facebook_app_secret", "whsec-test")
    body = {"event_id": f"evt-{uuid.uuid4()}", "remote_post_id": "fb-9", "status": "published"}
    raw = json.dumps(body).encode()

    missing = client.post(
        "/webhooks/facebook", content=raw, headers={"Content-Type": "application/json"}
    )
    assert missing.status_code == 401
    _assert_envelope(missing.json(), "webhook_signature_rejected")

    bad = client.post(
        "/webhooks/facebook",
        content=raw,
        headers={"Content-Type": "application/json", "X-Hub-Signature-256": "sha256=deadbeef"},
    )
    assert bad.status_code == 401

    signature = hmac.new(b"whsec-test", raw, hashlib.sha256).hexdigest()
    good = client.post(
        "/webhooks/facebook",
        content=raw,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": f"sha256={signature}",
        },
    )
    assert good.status_code == 200
    assert good.json()["status"] == "processed"
