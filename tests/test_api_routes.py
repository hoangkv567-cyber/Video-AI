"""API contract tests: happy paths, role gates, error envelopes, OAuth, webhooks.

Runs fully offline over TestClient + SQLite (env pinned by tests/conftest.py
before any app import). Auth uses the dev/test X-User-Id header path.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from conftest import make_video_plan
from fastapi.testclient import TestClient

from app import readiness as readiness_module
from app.api import oauth as oauth_module
from app.api import routes as routes_module
from app.db import Base, SessionLocal, engine
from app.main import app
from app.models import (
    Asset,
    AuditEvent,
    Campaign,
    ConnectedAccount,
    CostEvent,
    Creative,
    Job,
    PublishAttempt,
    PublishTarget,
    Rendition,
    Scene,
    ScriptVersion,
    Source,
    User,
)
from app.publishing.crypto import decrypt_credentials
from app.states import Capability, CreativeState, JobStatus, PublishTargetStatus, Role

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
            "admin": User(
                email="admin@t.local", name="Admin", role=Role.ADMIN.value, password_hash="x"
            ),
            "editor": User(
                email="editor@t.local", name="Editor", role=Role.EDITOR.value, password_hash="x"
            ),
            "publisher": User(
                email="pub@t.local", name="Publisher", role=Role.PUBLISHER.value, password_hash="x"
            ),
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
def client(api_schema: dict[str, str]) -> Iterator[TestClient]:
    with TestClient(app, headers={"X-User-Id": api_schema["admin"]}) as test_client:
        yield test_client


@pytest.fixture()
def db() -> Iterator:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(autouse=True)
def generation_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_module, "generation_preflight_issues", lambda: [])


def _create_creative_via_api(client: TestClient, **overrides: object) -> dict:
    plan = make_video_plan()
    body = {
        "topic_title": "New AI model launch",
        "brief": "Emerging AI news",
        "sources": [
            {
                "url": s["url"],
                "title": s["title"],
                "publisher": s["publisher"],
                "is_official": s["is_official"],
            }
            for s in plan["sources"]
        ],
        "video_plan": plan,
    }
    body.update(overrides)
    response = client.post(
        "/api/v1/creatives",
        json=body,
        headers={"Idempotency-Key": f"create-{uuid.uuid4()}"},
    )
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


def test_healthz_reports_versioned_runtime_config(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_config_version": "2026-08-31",
        "master_duration_seconds": pytest.approx(38.8),
    }


_READINESS_NAMES = ("postgresql", "redis", "minio", "ffmpeg")


def _fake_readiness_checks(
    calls: list[str], failures: frozenset[str] = frozenset()
) -> dict[str, Callable[[], None]]:
    def build_probe(name: str) -> Callable[[], None]:
        def probe() -> None:
            calls.append(name)
            if name in failures:
                raise RuntimeError(f"sentinel-secret-from-{name}")

        return probe

    return {name: build_probe(name) for name in _READINESS_NAMES}


def test_readyz_reports_all_dependencies_ready(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        readiness_module,
        "get_readiness_checks",
        lambda: _fake_readiness_checks(calls),
    )

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "status": "ready",
        "checks": {name: {"status": "ok"} for name in _READINESS_NAMES},
    }
    assert calls == list(_READINESS_NAMES)


@pytest.mark.parametrize("failed_name", _READINESS_NAMES)
def test_readyz_reports_failure_without_leaking_details(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    failed_name: str,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        readiness_module,
        "get_readiness_checks",
        lambda: _fake_readiness_checks(calls, frozenset({failed_name})),
    )

    response = client.get("/readyz")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["checks"][failed_name] == {"status": "failed"}
    assert all(
        payload["checks"][name] == {"status": "ok"}
        for name in _READINESS_NAMES
        if name != failed_name
    )
    assert calls == list(_READINESS_NAMES)
    assert "sentinel-secret" not in response.text


def test_healthz_never_runs_readiness_checks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_factory() -> dict[str, Callable[[], None]]:
        raise AssertionError("liveness called readiness probes")

    monkeypatch.setattr(readiness_module, "get_readiness_checks", unexpected_factory)

    response = client.get("/healthz")

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# /topics/discover and /jobs
# ---------------------------------------------------------------------------


def test_discover_returns_202_job(client: TestClient, db) -> None:
    response = client.post(
        "/api/v1/topics/discover",
        json={"brief": "AI hardware news"},
        headers={"Idempotency-Key": f"discover-{uuid.uuid4()}"},
    )
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


def test_asset_stream_supports_video_byte_ranges(
    client: TestClient, db, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _create_creative_via_api(client)
    data = b"0123456789"

    class MemoryStore:
        def get_url(self, key: str, expires_seconds: int = 3600) -> str:
            del key, expires_seconds
            return "file:///not-used"

        def get_bytes(self, key: str) -> bytes:
            assert key == "creatives/test/master/video.mp4"
            return data

    asset = Asset(
        creative_id=created["id"],
        kind="master",
        storage_key="creatives/test/master/video.mp4",
        sha256="0" * 64,
        size_bytes=len(data),
    )
    db.add(asset)
    db.commit()
    monkeypatch.setattr(routes_module, "get_asset_store", lambda: MemoryStore())

    full = client.get(f"/api/v1/assets/{asset.id}/stream")
    assert full.status_code == 200
    assert full.content == data
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-type"].startswith("video/mp4")

    partial = client.get(f"/api/v1/assets/{asset.id}/stream", headers={"Range": "bytes=2-5"})
    assert partial.status_code == 206
    assert partial.content == b"2345"
    assert partial.headers["content-range"] == "bytes 2-5/10"


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
        headers={"Idempotency-Key": f"create-{uuid.uuid4()}"},
    )
    assert response.status_code == 422
    _assert_envelope(response.json(), "validation_failed")


def test_create_creative_source_rejects_extra_fields(client: TestClient) -> None:
    response = client.post(
        "/api/v1/creatives",
        json={
            "topic_title": "Strict source input",
            "sources": [
                {
                    "url": "https://example.com/report",
                    "title": "Report",
                    "trusted_override": True,
                }
            ],
        },
        headers={"Idempotency-Key": f"create-{uuid.uuid4()}"},
    )

    assert response.status_code == 422
    assert any(
        error["type"] == "extra_forbidden"
        and error["loc"][-3:] == ["sources", 0, "trusted_override"]
        for error in response.json()["detail"]
    )


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "http://127.0.0.1/private",
        "http://localhost/private",
        "https://user:secret@example.com/report",
    ],
)
def test_create_creative_rejects_unsafe_source_urls(client: TestClient, url: str) -> None:
    response = client.post(
        "/api/v1/creatives",
        json={"topic_title": "Unsafe source", "sources": [{"url": url}]},
        headers={"Idempotency-Key": f"create-{uuid.uuid4()}"},
    )

    assert response.status_code == 422
    assert any(error["loc"][-1] == "url" for error in response.json()["detail"])


def test_create_creative_rejects_plan_source_url_mismatch(client: TestClient) -> None:
    plan = make_video_plan()
    sources = [
        {
            "url": source["url"],
            "title": source["title"],
            "publisher": source["publisher"],
            "is_official": source["is_official"],
        }
        for source in plan["sources"]
    ]
    expected_url = plan["sources"][1]["url"]
    unexpected_url = "https://example.org/unpersisted-report"
    plan["sources"][1]["url"] = unexpected_url

    response = client.post(
        "/api/v1/creatives",
        json={"topic_title": "Mismatched sources", "sources": sources, "video_plan": plan},
        headers={"Idempotency-Key": f"create-{uuid.uuid4()}"},
    )

    assert response.status_code == 422
    payload = response.json()
    _assert_envelope(payload, "source_integrity_mismatch")
    assert payload["details"]["missing_urls"] == [expected_url]
    assert payload["details"]["unexpected_urls"] == [unexpected_url]


def test_create_creative_canonicalizes_plan_source_metadata(client: TestClient, db) -> None:
    plan = make_video_plan()
    persisted_inputs = []
    for index, source_ref in enumerate(plan["sources"], start=1):
        persisted_inputs.append(
            {
                "url": source_ref["url"],
                "title": f"Canonical title {index}",
                "publisher": f"Canonical publisher {index}",
                "is_official": False,
            }
        )
        source_ref.update(
            {
                "title": "Spoofed title",
                "publisher": "Spoofed publisher",
                "is_official": True,
            }
        )

    response = client.post(
        "/api/v1/creatives",
        json={
            "topic_title": "Canonical source metadata",
            "sources": persisted_inputs,
            "video_plan": plan,
        },
        headers={"Idempotency-Key": f"create-{uuid.uuid4()}"},
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    version = db.get(ScriptVersion, payload["script_version_id"])
    persisted = {
        source.url: source
        for source in db.query(Source).filter(Source.creative_id == payload["id"]).all()
    }
    for source_ref in version.video_plan["sources"]:
        source = persisted[source_ref["url"]]
        assert source_ref["title"] == source.title
        assert source_ref["publisher"] == source.publisher
        assert source_ref["is_official"] is source.is_official is False
    assert any(issue["code"] == "sources_official" for issue in payload["issues"])


class RecordingDeleteStore:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        result: bool = True,
        fail_after: int | None = None,
    ) -> None:
        self.error = error
        self.result = result
        self.fail_after = fail_after
        self.deleted: list[str] = []

    def delete(self, key: str) -> bool:
        if self.error is not None and (
            self.fail_after is None or len(self.deleted) >= self.fail_after
        ):
            raise self.error
        self.deleted.append(key)
        return self.result


def _seed_deletable_creative_graph(client: TestClient, db) -> dict[str, str]:
    created = _create_creative_via_api(client, topic_title="Creative to delete")
    creative_id = created["id"]
    script_id = created["script_version_id"]
    scene = Scene(
        creative_id=creative_id,
        script_version_id=script_id,
        index=0,
        keyframe_prompt_en="keyframe",
        visual_prompt_en="visual",
        status="done",
    )
    db.add(scene)
    db.flush()
    storage_key = f"creatives/{creative_id}/keyframe/delete-me.png"
    asset = Asset(
        creative_id=creative_id,
        scene_id=scene.id,
        kind="keyframe",
        storage_key=storage_key,
        sha256="a" * 64,
        size_bytes=1234,
        pinned=True,
    )
    db.add(asset)
    db.flush()
    rendition = Rendition(
        creative_id=creative_id,
        locale="vi",
        master_asset_id=asset.id,
        title="Delete me",
    )
    db.add(rendition)
    db.flush()
    target = PublishTarget(
        creative_id=creative_id,
        rendition_id=rendition.id,
        platform="youtube",
        status=PublishTargetStatus.FAILED.value,
        last_error={"retryable": False},
    )
    db.add(target)
    db.flush()
    attempt = PublishAttempt(
        target_id=target.id,
        attempt_no=1,
        status="failed",
        finished_at=datetime.now(UTC),
    )
    db.add_all(
        [
            attempt,
            CostEvent(
                creative_id=creative_id,
                kind="other",
                model_id="test",
                units=1,
                amount_usd=0.25,
            ),
            Job(
                creative_id=creative_id,
                kind="render",
                queue="render",
                status=JobStatus.SUCCEEDED.value,
                attempts=1,
            ),
        ]
    )
    db.commit()
    return {
        "creative_id": creative_id,
        "campaign_id": created["campaign_id"],
        "scene_id": scene.id,
        "asset_id": asset.id,
        "rendition_id": rendition.id,
        "target_id": target.id,
        "attempt_id": attempt.id,
        "storage_key": storage_key,
    }


def test_admin_deletes_complete_creative_graph_and_owned_storage(
    client: TestClient,
    db,
    users: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = _seed_deletable_creative_graph(client, db)
    store = RecordingDeleteStore()
    monkeypatch.setattr(routes_module, "get_asset_store", lambda: store)

    response = client.delete(f"/api/v1/creatives/{seeded['creative_id']}")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["deleted"] is True
    assert payload["creative_id"] == seeded["creative_id"]
    assert payload["deleted_counts"]["assets"] == 1
    assert store.deleted == [seeded["storage_key"]]
    db.expire_all()
    assert db.get(Creative, seeded["creative_id"]) is None
    assert db.get(Campaign, seeded["campaign_id"]) is not None
    assert db.query(Source).filter_by(creative_id=seeded["creative_id"]).count() == 0
    assert db.query(ScriptVersion).filter_by(creative_id=seeded["creative_id"]).count() == 0
    assert db.query(Scene).filter_by(creative_id=seeded["creative_id"]).count() == 0
    assert db.query(Asset).filter_by(creative_id=seeded["creative_id"]).count() == 0
    assert db.query(Rendition).filter_by(creative_id=seeded["creative_id"]).count() == 0
    assert db.query(CostEvent).filter_by(creative_id=seeded["creative_id"]).count() == 0
    assert db.query(Job).filter_by(creative_id=seeded["creative_id"]).count() == 0
    assert db.get(PublishTarget, seeded["target_id"]) is None
    assert db.get(PublishAttempt, seeded["attempt_id"]) is None
    audit = db.query(AuditEvent).filter_by(
        action="creative_deleted", entity_id=seeded["creative_id"]
    ).one()
    assert audit.actor_id == users["admin"]
    assert audit.data["asset_bytes"] == 1234
    assert audit.data["actual_cost_usd"] == pytest.approx(0.25)


def test_creative_delete_is_admin_only(
    client: TestClient, db, users: dict[str, str]
) -> None:
    created = _create_creative_via_api(client, topic_title="Protected creative")

    for role in ("editor", "publisher"):
        response = client.delete(
            f"/api/v1/creatives/{created['id']}",
            headers={"X-User-Id": users[role]},
        )
        assert response.status_code == 403
        _assert_envelope(response.json(), "forbidden")
    anonymous = client.delete(
        f"/api/v1/creatives/{created['id']}",
        headers={"X-User-Id": ""},
    )
    assert anonymous.status_code == 401
    _assert_envelope(anonymous.json(), "unauthorized")
    db.expire_all()
    assert db.get(Creative, created["id"]) is not None


@pytest.mark.parametrize("job_status", [JobStatus.QUEUED.value, JobStatus.RUNNING.value])
def test_creative_delete_blocks_active_jobs(
    client: TestClient, db, job_status: str
) -> None:
    created = _create_creative_via_api(client, topic_title=f"Active {job_status}")
    db.add(
        Job(
            creative_id=created["id"],
            kind="generate",
            queue="ai",
            status=job_status,
        )
    )
    db.commit()

    response = client.delete(f"/api/v1/creatives/{created['id']}")

    assert response.status_code == 409
    _assert_envelope(response.json(), "creative_delete_blocked")
    assert response.json()["details"]["blockers"][0]["code"] == "creative_jobs_active"
    db.expire_all()
    assert db.get(Creative, created["id"]) is not None


@pytest.mark.parametrize(("attempts", "expected_status"), [(3, 409), (4, 200)])
def test_creative_delete_obeys_job_retry_limit(
    client: TestClient, db, attempts: int, expected_status: int
) -> None:
    created = _create_creative_via_api(client, topic_title=f"Retry boundary {attempts}")
    db.add(
        Job(
            creative_id=created["id"],
            kind="generate",
            queue="ai",
            status=JobStatus.FAILED.value,
            attempts=attempts,
            error={"retryable": True},
        )
    )
    db.commit()

    response = client.delete(f"/api/v1/creatives/{created['id']}")

    assert response.status_code == expected_status, response.text
    db.expire_all()
    if expected_status == 409:
        _assert_envelope(response.json(), "creative_delete_blocked")
        assert db.get(Creative, created["id"]) is not None
    else:
        assert response.json()["deleted"] is True
        assert db.get(Creative, created["id"]) is None


def test_creative_delete_blocks_remote_publication(client: TestClient, db) -> None:
    created = _create_creative_via_api(client, topic_title="Remote post")
    rendition = Rendition(creative_id=created["id"], locale="vi", title="Remote")
    db.add(rendition)
    db.flush()
    db.add(
        PublishTarget(
            creative_id=created["id"],
            rendition_id=rendition.id,
            platform="facebook",
            status=PublishTargetStatus.SCHEDULED_REMOTE.value,
            remote_post_id="remote-123",
        )
    )
    db.commit()

    response = client.delete(f"/api/v1/creatives/{created['id']}")

    assert response.status_code == 409
    _assert_envelope(response.json(), "creative_delete_blocked")
    assert any(
        blocker["code"] == "creative_publication_active_or_remote"
        for blocker in response.json()["details"]["blockers"]
    )
    db.expire_all()
    assert db.get(Creative, created["id"]) is not None


@pytest.mark.parametrize(("attempt_count", "expected_status"), [(3, 409), (4, 200)])
def test_creative_delete_obeys_publish_retry_limit(
    client: TestClient,
    db,
    monkeypatch: pytest.MonkeyPatch,
    attempt_count: int,
    expected_status: int,
) -> None:
    seeded = _seed_deletable_creative_graph(client, db)
    target = db.get(PublishTarget, seeded["target_id"])
    target.last_error = {"retryable": True}
    for attempt_no in range(2, attempt_count + 1):
        db.add(
            PublishAttempt(
                target_id=target.id,
                attempt_no=attempt_no,
                status="failed",
                finished_at=datetime.now(UTC),
            )
        )
    db.commit()
    store = RecordingDeleteStore()
    monkeypatch.setattr(routes_module, "get_asset_store", lambda: store)

    response = client.delete(f"/api/v1/creatives/{seeded['creative_id']}")

    assert response.status_code == expected_status, response.text
    db.expire_all()
    if expected_status == 409:
        _assert_envelope(response.json(), "creative_delete_blocked")
        assert db.get(Creative, seeded["creative_id"]) is not None
        assert store.deleted == []
    else:
        assert response.json()["deleted"] is True
        assert db.get(Creative, seeded["creative_id"]) is None
        assert store.deleted == [seeded["storage_key"]]


def test_creative_delete_blocks_persisted_remote_checkpoint(client: TestClient, db) -> None:
    seeded = _seed_deletable_creative_graph(client, db)
    attempt = db.get(PublishAttempt, seeded["attempt_id"])
    attempt.response = {"encrypted_checkpoint": "authenticated-ciphertext"}
    db.commit()

    response = client.delete(f"/api/v1/creatives/{seeded['creative_id']}")

    assert response.status_code == 409
    _assert_envelope(response.json(), "creative_delete_blocked")
    publication_blocker = next(
        blocker
        for blocker in response.json()["details"]["blockers"]
        if blocker["code"] == "creative_publication_active_or_remote"
    )
    assert publication_blocker["remote_evidence_attempt_ids"] == [seeded["attempt_id"]]
    db.expire_all()
    assert db.get(Creative, seeded["creative_id"]) is not None


@pytest.mark.parametrize("condition", ["claimed_target", "active_attempt"])
def test_creative_delete_blocks_in_flight_publish_state(
    client: TestClient, db, condition: str
) -> None:
    seeded = _seed_deletable_creative_graph(client, db)
    if condition == "claimed_target":
        target = db.get(PublishTarget, seeded["target_id"])
        target.claimed_at = datetime.now(UTC)
    else:
        attempt = db.get(PublishAttempt, seeded["attempt_id"])
        attempt.status = "uploading"
        attempt.finished_at = None
    db.commit()

    response = client.delete(f"/api/v1/creatives/{seeded['creative_id']}")

    assert response.status_code == 409
    _assert_envelope(response.json(), "creative_delete_blocked")
    db.expire_all()
    assert db.get(Creative, seeded["creative_id"]) is not None


def test_creative_delete_reports_already_missing_storage_object(
    client: TestClient,
    db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = _seed_deletable_creative_graph(client, db)
    store = RecordingDeleteStore(result=False)
    monkeypatch.setattr(routes_module, "get_asset_store", lambda: store)

    response = client.delete(f"/api/v1/creatives/{seeded['creative_id']}")

    assert response.status_code == 200, response.text
    assert response.json()["storage_objects_deleted"] == 0
    assert response.json()["storage_objects_missing"] == 1
    assert store.deleted == [seeded["storage_key"]]
    db.expire_all()
    assert db.get(Creative, seeded["creative_id"]) is None


def test_creative_delete_never_removes_shared_or_out_of_namespace_storage(
    client: TestClient,
    db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = _seed_deletable_creative_graph(client, db)
    other = _create_creative_via_api(client, topic_title="Storage owner")
    shared_key = f"creatives/{seeded['creative_id']}/shared.bin"
    db.add_all(
        [
            Asset(
                creative_id=seeded["creative_id"],
                kind="other",
                storage_key=shared_key,
                sha256="b" * 64,
            ),
            Asset(
                creative_id=other["id"],
                kind="other",
                storage_key=shared_key,
                sha256="b" * 64,
            ),
            Asset(
                creative_id=seeded["creative_id"],
                kind="other",
                storage_key="legacy/outside-creative-prefix.bin",
                sha256="c" * 64,
            ),
        ]
    )
    db.commit()
    store = RecordingDeleteStore()
    monkeypatch.setattr(routes_module, "get_asset_store", lambda: store)

    response = client.delete(f"/api/v1/creatives/{seeded['creative_id']}")

    assert response.status_code == 200, response.text
    assert response.json()["storage_objects_skipped"] == 2
    assert store.deleted == [seeded["storage_key"]]
    db.expire_all()
    assert db.get(Creative, seeded["creative_id"]) is None
    assert db.get(Creative, other["id"]) is not None
    assert db.query(Asset).filter_by(creative_id=other["id"], storage_key=shared_key).count() == 1


def test_creative_delete_rolls_back_database_when_storage_cleanup_fails(
    client: TestClient,
    db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = _seed_deletable_creative_graph(client, db)
    second_asset = Asset(
        creative_id=seeded["creative_id"],
        kind="other",
        storage_key=f"creatives/{seeded['creative_id']}/second-object.bin",
        sha256="d" * 64,
    )
    db.add(second_asset)
    db.commit()
    store = RecordingDeleteStore(
        error=OSError("sentinel storage error"),
        fail_after=1,
    )
    monkeypatch.setattr(
        routes_module,
        "get_asset_store",
        lambda: store,
    )

    response = client.delete(f"/api/v1/creatives/{seeded['creative_id']}")

    assert response.status_code == 502
    _assert_envelope(response.json(), "creative_delete_storage_failed")
    assert "sentinel storage error" not in response.text
    assert response.json()["details"]["deleted_objects"] == 1
    assert response.json()["details"]["remaining_objects"] == 1
    assert len(store.deleted) == 1
    db.expire_all()
    assert db.get(Creative, seeded["creative_id"]) is not None
    assert db.get(Asset, seeded["asset_id"]) is not None
    assert db.get(Asset, second_asset.id) is not None


def test_creative_delete_unknown_id_returns_404(client: TestClient) -> None:
    response = client.delete("/api/v1/creatives/does-not-exist")

    assert response.status_code == 404
    _assert_envelope(response.json(), "not_found")


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


def test_patch_script_rejects_plan_source_url_mismatch(
    client: TestClient, db, users: dict[str, str]
) -> None:
    created = _create_creative_via_api(client)
    edited = make_video_plan()
    edited["sources"][0]["url"] = "https://example.org/not-on-this-creative"

    response = client.patch(
        f"/api/v1/scripts/{created['script_version_id']}",
        json={"video_plan": edited},
        headers={"X-User-Id": users["editor"]},
    )

    assert response.status_code == 422
    _assert_envelope(response.json(), "source_integrity_mismatch")
    db.expire_all()
    assert db.query(ScriptVersion).filter(ScriptVersion.creative_id == created["id"]).count() == 1


def test_patch_script_canonicalizes_plan_source_metadata(
    client: TestClient, db, users: dict[str, str]
) -> None:
    plan = make_video_plan()
    persisted_inputs = [
        {
            "url": source_ref["url"],
            "title": f"DB title {index}",
            "publisher": f"DB publisher {index}",
            "is_official": False,
        }
        for index, source_ref in enumerate(plan["sources"], start=1)
    ]
    created = _create_creative_via_api(client, sources=persisted_inputs)
    edited = make_video_plan()
    for source_ref in edited["sources"]:
        source_ref.update(
            {
                "title": "Client override",
                "publisher": "Client publisher",
                "is_official": True,
            }
        )

    response = client.patch(
        f"/api/v1/scripts/{created['script_version_id']}",
        json={"video_plan": edited},
        headers={"X-User-Id": users["editor"]},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    version = db.get(ScriptVersion, payload["script_version_id"])
    persisted = {
        source.url: source
        for source in db.query(Source).filter(Source.creative_id == created["id"]).all()
    }
    for source_ref in version.video_plan["sources"]:
        source = persisted[source_ref["url"]]
        assert source_ref["title"] == source.title
        assert source_ref["publisher"] == source.publisher
        assert source_ref["is_official"] is source.is_official is False
    assert any(issue["code"] == "sources_official" for issue in payload["issues"])


def test_script_approve_requires_publisher_role(client: TestClient, users: dict[str, str]) -> None:
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
    source_ref = plan["sources"][0]
    created = _create_creative_via_api(
        client,
        video_plan=plan,
        sources=[
            {
                "url": source_ref["url"],
                "title": source_ref["title"],
                "publisher": source_ref["publisher"],
                "is_official": source_ref["is_official"],
            }
        ],
    )
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


def test_generate_stops_before_enqueue_when_provider_preflight_fails(
    client: TestClient,
    users: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_creative_via_api(client)
    _approve_script(client, created["script_version_id"], users)
    monkeypatch.setattr(
        routes_module,
        "generation_preflight_issues",
        lambda: [{"code": "groq_orpheus_terms_required", "message": "accept terms"}],
    )

    response = client.post(
        f"/api/v1/creatives/{created['id']}/generate",
        json={},
        headers={"Idempotency-Key": f"gen-{uuid.uuid4()}"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "generation_preflight_failed"


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
        headers={"Idempotency-Key": f"create-{uuid.uuid4()}"},
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
# POST /creatives/{id}/scenes/{id}/video-submission/reconcile
# ---------------------------------------------------------------------------


def _seed_ambiguous_video_submission(
    client: TestClient,
    db,
    users: dict[str, str],
    *,
    provider: str = "veo",
) -> dict[str, str]:
    created = _create_creative_via_api(client)
    _approve_script(client, created["script_version_id"], users)
    creative = db.get(Creative, created["id"])
    assert creative is not None
    scene = Scene(
        creative_id=creative.id,
        script_version_id=created["script_version_id"],
        index=0,
        duration_seconds=8.0,
        keyframe_prompt_en="keyframe",
        visual_prompt_en="video",
    )
    db.add(scene)
    db.flush()
    projection = CostEvent(
        creative_id=creative.id,
        kind=provider,
        model_id="paid-video-model",
        units=8.0,
        unit_price_usd=0.05,
        amount_usd=0.4,
        projected=True,
        note=f"{provider} scene {scene.id}",
    )
    db.add(projection)
    db.flush()
    operation = Job(
        kind="veo_operation",
        queue="ai",
        status=JobStatus.RUNNING.value,
        creative_id=creative.id,
        idempotency_key=f"veo:{scene.id}",
        payload={
            "scene_id": scene.id,
            "phase": "intent",
            "operation_name": "",
            "model_id": "paid-video-model",
            "duration_seconds": 8.0,
            "provider": provider,
            "creative_id": creative.id,
            "cost_event_id": projection.id,
        },
        started_at=datetime.now(UTC),
    )
    db.add(operation)
    creative.state = CreativeState.NEEDS_ACTION.value
    creative.last_error = {
        "code": "veo_submission_ambiguous",
        "message": "unknown outcome",
    }
    db.commit()
    return {
        "creative_id": creative.id,
        "scene_id": scene.id,
        "operation_job_id": operation.id,
        "projection_id": projection.id,
    }


def test_reconcile_video_submission_is_admin_only(
    client: TestClient,
    db,
    users: dict[str, str],
) -> None:
    seeded = _seed_ambiguous_video_submission(client, db, users)
    response = client.post(
        f"/api/v1/creatives/{seeded['creative_id']}/scenes/"
        f"{seeded['scene_id']}/video-submission/reconcile",
        json={
            "action": "confirm_not_submitted",
            "provider": "veo",
            "reason": "Verified in provider console",
        },
        headers={
            "X-User-Id": users["editor"],
            "Idempotency-Key": f"reconcile-{uuid.uuid4()}",
        },
    )
    assert response.status_code == 403
    _assert_envelope(response.json(), "forbidden")

    operation_read = client.get(
        f"/api/v1/jobs/{seeded['operation_job_id']}",
        headers={"X-User-Id": users["editor"]},
    )
    assert operation_read.status_code == 403
    _assert_envelope(operation_read.json(), "forbidden")
    assert client.get(f"/api/v1/jobs/{seeded['operation_job_id']}").status_code == 200


def test_reconcile_attaches_operation_and_enqueues_one_resume_job(
    client: TestClient,
    db,
    users: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = _seed_ambiguous_video_submission(client, db, users)
    monkeypatch.setattr(
        routes_module,
        "get_settings",
        lambda: SimpleNamespace(video_provider="veo"),
    )
    key = f"reconcile-{uuid.uuid4()}"
    url = (
        f"/api/v1/creatives/{seeded['creative_id']}/scenes/"
        f"{seeded['scene_id']}/video-submission/reconcile"
    )
    body = {
        "action": "attach_operation",
        "provider": "veo",
        "operation_name": "operations/provider-task-123",
        "reason": "Matched request timestamp and model in provider console",
    }

    first = client.post(url, json=body, headers={"Idempotency-Key": key})
    replay = client.post(url, json=body, headers={"Idempotency-Key": key})

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json() == first.json()
    db.expire_all()
    creative = db.get(Creative, seeded["creative_id"])
    scene = db.get(Scene, seeded["scene_id"])
    operation = db.get(Job, seeded["operation_job_id"])
    assert creative is not None and creative.state == CreativeState.GENERATING.value
    assert creative.last_error is None
    assert scene is not None and scene.status == "pending" and scene.last_error is None
    assert operation is not None and operation.status == JobStatus.RUNNING.value
    assert operation.payload["phase"] == "submitted"
    assert operation.payload["operation_name"] == "operations/provider-task-123"
    assert operation.payload["provider"] == "veo"
    assert db.get(CostEvent, seeded["projection_id"]) is not None
    resume_jobs = db.query(Job).filter_by(
        creative_id=seeded["creative_id"],
        kind="generate",
        idempotency_key=key,
    ).all()
    assert len(resume_jobs) == 1
    audit = db.query(AuditEvent).filter_by(
        action="video_submission_reconciled",
        entity_id=seeded["scene_id"],
    ).one()
    assert audit.actor_id == users["admin"]
    assert audit.data["action"] == "attach_operation"


def test_reconcile_confirm_not_submitted_releases_exact_projection(
    client: TestClient,
    db,
    users: dict[str, str],
) -> None:
    seeded = _seed_ambiguous_video_submission(client, db, users, provider="wan")
    key = f"reconcile-{uuid.uuid4()}"
    response = client.post(
        f"/api/v1/creatives/{seeded['creative_id']}/scenes/"
        f"{seeded['scene_id']}/video-submission/reconcile",
        json={
            "action": "confirm_not_submitted",
            "provider": "wan",
            "reason": "Provider support confirmed that no task was created",
        },
        headers={"Idempotency-Key": key},
    )

    assert response.status_code == 202
    db.expire_all()
    operation = db.get(Job, seeded["operation_job_id"])
    assert operation is not None and operation.status == JobStatus.CANCELLED.value
    assert operation.result == {
        "reason": "Provider support confirmed that no task was created",
        "reconciled": True,
    }
    assert db.get(CostEvent, seeded["projection_id"]) is None
    assert db.query(Job).filter_by(
        creative_id=seeded["creative_id"],
        kind="generate",
        idempotency_key=key,
    ).count() == 1


def test_reconcile_rolls_back_when_resume_preflight_fails(
    client: TestClient,
    db,
    users: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = _seed_ambiguous_video_submission(client, db, users)
    monkeypatch.setattr(
        routes_module,
        "get_settings",
        lambda: SimpleNamespace(video_provider="veo"),
    )
    monkeypatch.setattr(
        routes_module,
        "generation_preflight_issues",
        lambda: [{"code": "provider_unavailable", "message": "not configured"}],
    )
    response = client.post(
        f"/api/v1/creatives/{seeded['creative_id']}/scenes/"
        f"{seeded['scene_id']}/video-submission/reconcile",
        json={
            "action": "attach_operation",
            "provider": "veo",
            "operation_name": "operations/provider-task-123",
            "reason": "Matched in provider console",
        },
        headers={"Idempotency-Key": f"reconcile-{uuid.uuid4()}"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "generation_preflight_failed"
    db.expire_all()
    creative = db.get(Creative, seeded["creative_id"])
    operation = db.get(Job, seeded["operation_job_id"])
    assert creative is not None and creative.state == CreativeState.NEEDS_ACTION.value
    assert operation is not None and operation.payload["phase"] == "intent"
    assert operation.payload["operation_name"] == ""
    assert db.get(CostEvent, seeded["projection_id"]) is not None
    assert db.query(Job).filter_by(
        creative_id=seeded["creative_id"],
        kind="generate",
    ).count() == 0


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
        Rendition(
            creative_id=creative.id,
            locale=locale,
            title=f"Title {locale}",
            qc_report={"passed": True, "checks": []},
        )
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

    anonymous = client.post(
        f"/api/v1/renditions/{rendition_id}/approve",
        headers={"X-User-Id": ""},
    )
    assert anonymous.status_code == 401
    _assert_envelope(anonymous.json(), "unauthorized")


@pytest.mark.parametrize(
    "qc_report",
    [None, {}, {"passed": False}, {"passed": "true"}, {"passed": 1}],
)
def test_rendition_approve_requires_explicit_boolean_qc_pass(
    client: TestClient, db, users: dict[str, str], qc_report: object
) -> None:
    _, rendition_id, _ = _seed_ready_creative(db, users)
    rendition = db.get(Rendition, rendition_id)
    rendition.qc_report = qc_report
    db.commit()
    response = client.post(
        f"/api/v1/renditions/{rendition_id}/approve",
        headers={"X-User-Id": users["publisher"]},
    )
    assert response.status_code == 422
    _assert_envelope(response.json(), "policy_blocked")
    db.expire_all()
    assert db.get(Rendition, rendition_id).is_approved is False


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
        json={
            "creative_id": creative_id,
            "targets": [{"rendition_id": vi_id, "platform": "youtube"}],
        },
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


def test_publications_unknown_platform_422(client: TestClient, db, users: dict[str, str]) -> None:
    creative_id, vi_id, _ = _seed_ready_creative(db, users)
    _finalize_creative(db, creative_id)
    response = client.post(
        "/api/v1/publications",
        json={
            "creative_id": creative_id,
            "targets": [{"rendition_id": vi_id, "platform": "myspace"}],
        },
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


def test_oauth_start_requires_admin(client: TestClient, users: dict[str, str]) -> None:
    anonymous = client.get(
        "/oauth/youtube/start",
        headers={"X-User-Id": ""},
        follow_redirects=False,
    )
    assert anonymous.status_code == 401
    _assert_envelope(anonymous.json(), "unauthorized")

    editor = client.get(
        "/oauth/youtube/start",
        headers={"X-User-Id": users["editor"]},
        follow_redirects=False,
    )
    assert editor.status_code == 403
    _assert_envelope(editor.json(), "forbidden")


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


def test_oauth_callback_stores_encrypted_credentials_and_probes(client: TestClient, db) -> None:
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
    account = db.query(ConnectedAccount).filter(ConnectedAccount.platform == "youtube").one()
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


def test_manual_bundle_download_is_publisher_only_and_proxied_from_store(
    client: TestClient,
    db,
    users: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_id = _seed_publish_target(
        db,
        users,
        status=PublishTargetStatus.MANUAL_BUNDLE.value,
        remote_post_id=None,
    )
    target = db.get(PublishTarget, target_id)
    bundle_bytes = b"PK\x03\x04durable-bundle"
    storage_key = f"creatives/{target.creative_id}/bundle/manual_youtube_test.zip"
    asset = Asset(
        creative_id=target.creative_id,
        kind="bundle",
        platform="youtube",
        storage_key=storage_key,
        sha256=hashlib.sha256(bundle_bytes).hexdigest(),
        size_bytes=len(bundle_bytes),
        pinned=True,
    )
    target.bundle_path = storage_key
    db.add(asset)
    db.commit()

    class MemoryStore:
        def get_url(self, key: str, expires_seconds: int = 3600) -> str:
            raise AssertionError("authenticated bundle downloads must not redirect")

        def get_bytes(self, key: str) -> bytes:
            assert key == storage_key
            return bundle_bytes

    monkeypatch.setattr(routes_module, "get_asset_store", lambda: MemoryStore())
    url = f"/api/v1/publish-targets/{target_id}/bundle"

    anonymous = client.get(url, headers={"X-User-Id": ""})
    assert anonymous.status_code == 401
    editor = client.get(url, headers={"X-User-Id": users["editor"]})
    assert editor.status_code == 403

    response = client.get(url, headers={"X-User-Id": users["publisher"]})
    assert response.status_code == 200
    assert response.content == bundle_bytes
    assert "zip" in response.headers["content-type"]
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert 'filename="manual_youtube_test.zip"' in response.headers["content-disposition"]


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
