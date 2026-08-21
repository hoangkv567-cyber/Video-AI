"""YouTube contract tests over httpx.MockTransport (secret-free canned responses)."""

import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.sqlite3")

import json  # noqa: E402
import random  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from app.publishing.base import PublishContext, PublishNeedsAction  # noqa: E402
from app.publishing.ledger import InMemoryCostLedger  # noqa: E402
from app.publishing.retry import RetryPolicy  # noqa: E402
from app.publishing.youtube import (  # noqa: E402
    TOKEN_ENDPOINT,
    UPLOAD_ENDPOINT,
    VIDEOS_ENDPOINT,
    YouTubePublisher,
)

UPLOAD_SESSION_URL = "https://upload.example/session-1"
POLICY = RetryPolicy(base_delay_seconds=0.001, cap_delay_seconds=0.002, max_attempts=4)


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def make_ctx(tmp_path: Path, **overrides) -> PublishContext:
    video = tmp_path / "short.mp4"
    if not video.exists():
        video.write_bytes(b"\x00\x01" * 512)
    defaults = dict(
        creative_id="creative-1",
        rendition_id="rendition-vi",
        locale="vi",
        title="AI moi ra mat",
        description="Ban tin cong nghe 40 giay.",
        hashtags=["#AI", "#congnghe", "shorts"],
        file_path=str(video),
        privacy="private",
    )
    defaults.update(overrides)
    return PublishContext(**defaults)


def make_publisher(handler, *, credentials=None, ledger=None, sleep=None) -> YouTubePublisher:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return YouTubePublisher(
        credentials if credentials is not None else {"access_token": "test-token"},
        client=client,
        ledger=ledger,
        policy=POLICY,
        sleep=sleep if sleep is not None else SleepRecorder(),
        rng=random.Random(7),
    )


def happy_handler(recorded: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(UPLOAD_ENDPOINT) and request.method == "POST":
            recorded["init_body"] = json.loads(request.content.decode())
            recorded["init_auth"] = request.headers.get("Authorization")
            return httpx.Response(200, headers={"Location": UPLOAD_SESSION_URL})
        if url.startswith(UPLOAD_SESSION_URL) and request.method == "PUT":
            recorded.setdefault("puts", []).append(len(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "vid123",
                    "status": {"uploadStatus": "uploaded", "privacyStatus": "private"},
                },
            )
        if url.startswith(VIDEOS_ENDPOINT) and request.method == "GET":
            recorded["list_params"] = dict(request.url.params)
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": "vid123",
                            "status": {"uploadStatus": "processed", "privacyStatus": "private"},
                            "processingDetails": {"processingStatus": "succeeded"},
                        }
                    ]
                },
            )
        return httpx.Response(404, json={"error": "unexpected request"})

    return handler


class TestHappyFlow:
    def test_full_flow_private_default_disclosure_and_metadata(self, tmp_path: Path) -> None:
        recorded: dict = {}
        ledger = InMemoryCostLedger()
        pub = make_publisher(happy_handler(recorded), ledger=ledger)
        # Even an explicit "public" must not go out on an unaudited project.
        ctx = make_ctx(tmp_path, privacy="public")

        pub.validate(ctx, {"duration_seconds": 38.8})
        session = pub.prepare(ctx)
        assert session["upload_url"] == UPLOAD_SESSION_URL

        body = recorded["init_body"]
        assert body["status"]["privacyStatus"] == "private"
        assert body["status"]["containsSyntheticMedia"] is True
        assert body["status"]["selfDeclaredMadeForKids"] is False
        assert body["snippet"]["title"] == ctx.title
        assert body["snippet"]["tags"] == ["AI", "congnghe", "shorts"]
        assert recorded["init_auth"] == "Bearer test-token"

        session = pub.upload(ctx, session)
        result = pub.finalize(ctx, session)
        assert result.remote_post_id == "vid123"
        assert result.remote_status["upload_status"] == "uploaded"
        assert result.raw["id"] == "vid123"

        status = pub.poll_status("vid123")
        assert status["exists"] is True
        assert status["processing"]["processingStatus"] == "succeeded"
        assert recorded["list_params"]["part"] == "status,processingDetails"

        # Every external call is on the cost ledger (0 USD publish API calls).
        notes = [e["note"] for e in ledger.events]
        assert any("videos.insert.init" in n for n in notes)
        assert any("videos.insert.upload" in n for n in notes)
        assert any("videos.list" in n for n in notes)
        assert ledger.total_spent_usd() == 0.0

    def test_publish_at_scheduling_forces_private(self, tmp_path: Path) -> None:
        recorded: dict = {}
        pub = make_publisher(happy_handler(recorded))
        scheduled = datetime(2027, 1, 15, 9, 30, tzinfo=UTC)
        ctx = make_ctx(tmp_path, privacy="public", scheduled_at=scheduled)

        pub.prepare(ctx)
        status = recorded["init_body"]["status"]
        assert status["publishAt"] == "2027-01-15T09:30:00Z"
        assert status["privacyStatus"] == "private"


class TestErrorPaths:
    def test_expired_token_triggers_refresh_then_succeeds(self, tmp_path: Path) -> None:
        recorded: dict = {"init_calls": 0, "auths": []}

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.startswith(TOKEN_ENDPOINT):
                recorded["token_body"] = request.content.decode()
                return httpx.Response(
                    200, json={"access_token": "new-token", "expires_in": 3600}
                )
            if url.startswith(UPLOAD_ENDPOINT):
                recorded["init_calls"] += 1
                recorded["auths"].append(request.headers.get("Authorization"))
                if recorded["init_calls"] == 1:
                    return httpx.Response(401, json={"error": {"code": 401}})
                return httpx.Response(200, headers={"Location": UPLOAD_SESSION_URL})
            return httpx.Response(404)

        pub = make_publisher(
            handler,
            credentials={"access_token": "stale-token", "refresh_token": "refresh-1"},
        )
        session = pub.prepare(make_ctx(tmp_path))
        assert session["upload_url"] == UPLOAD_SESSION_URL
        assert pub.credentials["access_token"] == "new-token"
        assert recorded["auths"] == ["Bearer stale-token", "Bearer new-token"]
        assert "refresh_token=refresh-1" in recorded["token_body"]

    def test_429_then_success_uses_backoff(self, tmp_path: Path) -> None:
        recorded: dict = {"init_calls": 0}
        sleeps = SleepRecorder()

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).startswith(UPLOAD_ENDPOINT):
                recorded["init_calls"] += 1
                if recorded["init_calls"] == 1:
                    return httpx.Response(429, json={"error": "rate limit"})
                return httpx.Response(200, headers={"Location": UPLOAD_SESSION_URL})
            return httpx.Response(404)

        pub = make_publisher(handler, sleep=sleeps)
        session = pub.prepare(make_ctx(tmp_path))
        assert session["upload_url"] == UPLOAD_SESSION_URL
        assert recorded["init_calls"] == 2
        assert len(sleeps.calls) == 1
        assert 0.0 <= sleeps.calls[0] <= POLICY.cap_delay_seconds

    def test_timeout_probes_status_and_never_duplicates_upload(self, tmp_path: Path) -> None:
        recorded: dict = {"data_puts": 0, "probe_puts": 0}
        sleeps = SleepRecorder()

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.startswith(UPLOAD_ENDPOINT) and request.method == "POST":
                return httpx.Response(200, headers={"Location": UPLOAD_SESSION_URL})
            if url.startswith(UPLOAD_SESSION_URL) and request.method == "PUT":
                if request.headers.get("Content-Range", "").startswith("bytes */"):
                    recorded["probe_puts"] += 1
                    return httpx.Response(
                        201,
                        json={
                            "id": "vid123",
                            "status": {"uploadStatus": "uploaded", "privacyStatus": "private"},
                        },
                    )
                recorded["data_puts"] += 1
                raise httpx.ReadTimeout("simulated upload timeout", request=request)
            return httpx.Response(404)

        pub = make_publisher(handler, sleep=sleeps)
        ctx = make_ctx(tmp_path)
        session = pub.prepare(ctx)
        session = pub.upload(ctx, session)
        result = pub.finalize(ctx, session)

        assert result.remote_post_id == "vid123"
        assert recorded["data_puts"] == 1  # the byte upload was never re-sent
        assert recorded["probe_puts"] == 1  # exactly one status probe
        assert sleeps.calls == []  # recovered without blind backoff retries

    def test_policy_block_maps_to_needs_action_without_retry(self, tmp_path: Path) -> None:
        recorded: dict = {"init_calls": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            recorded["init_calls"] += 1
            return httpx.Response(403, json={"error": {"message": "uploads disabled"}})

        pub = make_publisher(handler)
        with pytest.raises(PublishNeedsAction):
            pub.prepare(make_ctx(tmp_path))
        assert recorded["init_calls"] == 1


class TestValidation:
    def test_good_target_passes(self, tmp_path: Path) -> None:
        pub = make_publisher(lambda request: httpx.Response(500))
        pub.validate(make_ctx(tmp_path), {"duration_seconds": 38.8})

    def test_missing_title_rejected(self, tmp_path: Path) -> None:
        pub = make_publisher(lambda request: httpx.Response(500))
        with pytest.raises(PublishNeedsAction) as excinfo:
            pub.validate(make_ctx(tmp_path, title="  "))
        assert "title" in str(excinfo.value.details["problems"])

    def test_naive_schedule_rejected(self, tmp_path: Path) -> None:
        pub = make_publisher(lambda request: httpx.Response(500))
        with pytest.raises(PublishNeedsAction):
            pub.validate(make_ctx(tmp_path, scheduled_at=datetime(2027, 1, 1, 8, 0)))

    def test_over_length_video_rejected(self, tmp_path: Path) -> None:
        pub = make_publisher(lambda request: httpx.Response(500))
        with pytest.raises(PublishNeedsAction):
            pub.validate(make_ctx(tmp_path), {"duration_seconds": 200.0})
