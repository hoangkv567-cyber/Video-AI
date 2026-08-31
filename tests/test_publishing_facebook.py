"""Facebook Page Reels contract tests over httpx.MockTransport."""

import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.sqlite3")

import random  # noqa: E402
from datetime import UTC, datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402
from urllib.parse import parse_qs  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from app.publishing.base import PublishContext, PublishNeedsAction  # noqa: E402
from app.publishing.facebook import GRAPH_BASE, FacebookPublisher  # noqa: E402
from app.publishing.retry import RetryPolicy  # noqa: E402

PAGE_ID = "1122334455"
UPLOAD_URL = "https://rupload.example/video-upload/v21.0/v900"
POLICY = RetryPolicy(base_delay_seconds=0.001, cap_delay_seconds=0.002, max_attempts=4)


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def make_ctx(tmp_path: Path, **overrides) -> PublishContext:
    video = tmp_path / "reel.mp4"
    if not video.exists():
        video.write_bytes(b"\x00\x01" * 256)
    defaults = dict(
        creative_id="creative-1",
        rendition_id="rendition-en",
        locale="en",
        title="New AI chip explained",
        description="A 40-second tech brief.",
        hashtags=["#AI", "#tech"],
        file_path=str(video),
    )
    defaults.update(overrides)
    return PublishContext(**defaults)


def make_publisher(handler, *, sleep=None) -> FacebookPublisher:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return FacebookPublisher(
        {"page_id": PAGE_ID, "page_access_token": "page-token"},
        client=client,
        policy=POLICY,
        sleep=sleep if sleep is not None else SleepRecorder(),
        rng=random.Random(11),
    )


def reels_handler(recorded: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(f"{GRAPH_BASE}/{PAGE_ID}/video_reels") and request.method == "POST":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            phase = form.get("upload_phase")
            if phase == "start":
                recorded["start_form"] = form
                return httpx.Response(200, json={"video_id": "v900", "upload_url": UPLOAD_URL})
            if phase == "finish":
                recorded["finish_form"] = form
                return httpx.Response(200, json={"success": True, "post_id": "post-777"})
            return httpx.Response(400, json={"error": "bad phase"})
        if url.startswith(UPLOAD_URL) and request.method == "POST":
            recorded["upload_headers"] = dict(request.headers)
            recorded["upload_bytes"] = len(request.content)
            return httpx.Response(200, json={"success": True})
        if url.startswith(f"{GRAPH_BASE}/v900") and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "status": {
                        "video_status": "ready",
                        "uploading_phase": {"status": "complete"},
                    }
                },
            )
        return httpx.Response(404, json={"error": "unexpected request"})

    return handler


class TestReelsFlow:
    def test_start_upload_finish_publishes_immediately(self, tmp_path: Path) -> None:
        recorded: dict = {}
        pub = make_publisher(reels_handler(recorded))
        ctx = make_ctx(tmp_path)

        pub.validate(ctx, {"duration_seconds": 38.8})
        session = pub.prepare(ctx)
        assert session == {"video_id": "v900", "upload_url": UPLOAD_URL}
        assert recorded["start_form"]["upload_phase"] == "start"

        session = pub.upload(ctx, session)
        assert recorded["upload_headers"]["authorization"] == "OAuth page-token"
        assert recorded["upload_headers"]["offset"] == "0"
        assert int(recorded["upload_headers"]["file_size"]) == recorded["upload_bytes"]

        result = pub.finalize(ctx, session)
        assert result.remote_post_id == "post-777"
        assert recorded["finish_form"]["video_state"] == "PUBLISHED"
        assert "scheduled_publish_time" not in recorded["finish_form"]
        assert ctx.title in recorded["finish_form"]["description"]
        assert "#AI" in recorded["finish_form"]["description"]

        status = pub.poll_status("v900")
        assert status["video_status"] == "ready"

    def test_scheduled_publish_inside_native_window(self, tmp_path: Path) -> None:
        recorded: dict = {}
        pub = make_publisher(reels_handler(recorded))
        scheduled = datetime.now(UTC) + timedelta(hours=6)
        ctx = make_ctx(tmp_path, scheduled_at=scheduled)

        pub.validate(ctx)
        session = pub.prepare(ctx)
        session = pub.upload(ctx, session)
        result = pub.finalize(ctx, session)

        assert recorded["finish_form"]["video_state"] == "SCHEDULED"
        assert recorded["finish_form"]["scheduled_publish_time"] == str(int(scheduled.timestamp()))
        assert result.remote_status["video_state"] == "SCHEDULED"

    def test_hosted_file_url_upload_sends_no_bytes(self, tmp_path: Path) -> None:
        recorded: dict = {}
        pub = make_publisher(reels_handler(recorded))
        ctx = make_ctx(
            tmp_path, file_path=None, file_url="https://cdn.example/renditions/reel.mp4"
        )

        session = pub.prepare(ctx)
        pub.upload(ctx, session)
        assert recorded["upload_headers"]["file_url"] == "https://cdn.example/renditions/reel.mp4"
        assert recorded["upload_bytes"] == 0


class TestSchedulingWindow:
    @pytest.mark.parametrize(
        "offset",
        [timedelta(minutes=2), timedelta(days=45), timedelta(minutes=-30)],
        ids=["too_soon", "too_far", "in_the_past"],
    )
    def test_outside_native_window_needs_action(self, tmp_path: Path, offset) -> None:
        pub = make_publisher(reels_handler({}))
        ctx = make_ctx(tmp_path, scheduled_at=datetime.now(UTC) + offset)
        with pytest.raises(PublishNeedsAction):
            pub.validate(ctx)

    def test_finalize_re_checks_the_window(self, tmp_path: Path) -> None:
        recorded: dict = {}
        pub = make_publisher(reels_handler(recorded))
        ctx = make_ctx(tmp_path, scheduled_at=datetime.now(UTC) + timedelta(minutes=1))
        with pytest.raises(PublishNeedsAction):
            pub.finalize(ctx, {"video_id": "v900", "upload_url": UPLOAD_URL})
        assert "finish_form" not in recorded  # rejected before any remote call


class TestResilience:
    def test_finish_408_fails_closed_without_duplicate_post(self, tmp_path: Path) -> None:
        recorded: dict = {"finish_calls": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.startswith(f"{GRAPH_BASE}/{PAGE_ID}/video_reels"):
                form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
                if form.get("upload_phase") == "start":
                    return httpx.Response(
                        200, json={"video_id": "v900", "upload_url": UPLOAD_URL}
                    )
                if form.get("upload_phase") == "finish":
                    recorded["finish_calls"] += 1
                    return httpx.Response(408, json={"error": "unknown outcome"})
            if url.startswith(UPLOAD_URL):
                return httpx.Response(200, json={"success": True})
            return httpx.Response(404)

        pub = make_publisher(handler)
        ctx = make_ctx(tmp_path)
        session = pub.upload(ctx, pub.prepare(ctx))

        with pytest.raises(PublishNeedsAction) as excinfo:
            pub.finalize(ctx, session)

        assert excinfo.value.details["reason"] == "remote_outcome_unknown"
        assert recorded["finish_calls"] == 1

    def test_500_on_start_retries_then_succeeds(self, tmp_path: Path) -> None:
        recorded: dict = {"start_calls": 0}
        sleeps = SleepRecorder()

        def handler(request: httpx.Request) -> httpx.Response:
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            if form.get("upload_phase") == "start":
                recorded["start_calls"] += 1
                if recorded["start_calls"] <= 2:
                    return httpx.Response(503, json={"error": "server busy"})
                return httpx.Response(200, json={"video_id": "v900", "upload_url": UPLOAD_URL})
            return httpx.Response(404)

        pub = make_publisher(handler, sleep=sleeps)
        session = pub.prepare(make_ctx(tmp_path))
        assert session["video_id"] == "v900"
        assert recorded["start_calls"] == 3
        assert len(sleeps.calls) == 2

    def test_upload_timeout_recovers_via_status_probe(self, tmp_path: Path) -> None:
        recorded: dict = {"data_posts": 0, "status_gets": 0}
        sleeps = SleepRecorder()

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            form = (
                {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
                if request.method == "POST" and url.startswith(GRAPH_BASE)
                else {}
            )
            if form.get("upload_phase") == "start":
                return httpx.Response(200, json={"video_id": "v900", "upload_url": UPLOAD_URL})
            if url.startswith(UPLOAD_URL):
                recorded["data_posts"] += 1
                raise httpx.ReadTimeout("simulated upload timeout", request=request)
            if url.startswith(f"{GRAPH_BASE}/v900") and request.method == "GET":
                recorded["status_gets"] += 1
                return httpx.Response(
                    200, json={"status": {"uploading_phase": {"status": "complete"}}}
                )
            return httpx.Response(404)

        pub = make_publisher(handler, sleep=sleeps)
        ctx = make_ctx(tmp_path)
        session = pub.prepare(ctx)
        session = pub.upload(ctx, session)
        assert session["upload_response"]["recovered_from_status_probe"] is True
        assert recorded["data_posts"] == 1  # bytes were never re-sent
        assert recorded["status_gets"] == 1  # exactly one remote status query
        assert sleeps.calls == []


class TestProbe:
    class Account:
        platform = "facebook"
        capability = "MANUAL"
        status = "active"
        last_probe_at = None
        last_probe_result = None

        def __init__(self, scopes, status="active"):
            self.scopes = scopes
            self.status = status

    def test_full_scopes_probe_schedule(self) -> None:
        pub = make_publisher(lambda request: httpx.Response(500))
        account = self.Account(
            ["pages_show_list", "pages_read_engagement", "pages_manage_posts"]
        )
        assert pub.probe_capability(account).value == "SCHEDULE"

    def test_missing_manage_posts_probe_manual(self) -> None:
        pub = make_publisher(lambda request: httpx.Response(500))
        assert pub.probe_capability(self.Account(["pages_show_list"])).value == "MANUAL"

    def test_revoked_account_probe_blocked(self) -> None:
        pub = make_publisher(lambda request: httpx.Response(500))
        account = self.Account(["pages_manage_posts"], status="revoked")
        assert pub.probe_capability(account).value == "BLOCKED"
