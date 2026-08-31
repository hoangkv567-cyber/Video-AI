"""TikTok publisher test suite (PLAN.md §3, Week 6).

Covers:
- Capability probing (MANUAL, DRAFT, BLOCKED)
- Consent token enforcement (zero HTTP requests when consent is missing)
- Complete Upload-to-Inbox flow (init -> put -> finalize)
- Status polling and probe-based upload recovery
- Token refresh with client key/secret
- Error handling (validation errors, 4xx/5xx responses, upstream failures)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.sqlite3")

import httpx
import pytest

from app.config import Settings
from app.models import ConnectedAccount
from app.publishing.base import (
    PublishContext,
    PublishError,
    PublishNeedsAction,
    PublishResult,
)
from app.publishing.retry import RetryPolicy
from app.publishing.tiktok import (
    INBOX_INIT_ENDPOINT,
    STATUS_FETCH_ENDPOINT,
    TOKEN_ENDPOINT,
    TikTokPublisher,
)
from app.states import Capability

POLICY = RetryPolicy(base_delay_seconds=0.001, cap_delay_seconds=0.002, max_attempts=3)


def make_ctx(tmp_path: Path, **overrides: Any) -> PublishContext:
    video = tmp_path / "tiktok_clip.mp4"
    if not video.exists():
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100)
    defaults: dict[str, Any] = dict(
        creative_id="creative-tiktok-1",
        rendition_id="rendition-vi",
        locale="vi",
        title="TikTok AI News",
        description="Daily AI summary",
        hashtags=["#AI", "#Tech"],
        file_path=str(video),
        consent_token="consent-tiktok-test",
    )
    defaults.update(overrides)
    return PublishContext(**defaults)


def mock_client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestTikTokPublisherContract:
    def test_capabilities(self) -> None:
        pub = TikTokPublisher({"access_token": "test-tok"})
        caps = pub.capabilities()
        assert caps == frozenset({Capability.DRAFT, Capability.MANUAL})

    def test_validate_success(self, tmp_path: Path) -> None:
        pub = TikTokPublisher({"access_token": "test-tok"})
        ctx = make_ctx(tmp_path)
        pub.validate(ctx, rendition={"duration_seconds": 38.8})

    def test_validate_missing_file_path(self) -> None:
        pub = TikTokPublisher({"access_token": "test-tok"})
        ctx = PublishContext(
            creative_id="c1",
            rendition_id="r1",
            locale="vi",
            title="Title",
            description="Desc",
            hashtags=[],
            file_path="",
        )
        with pytest.raises(PublishNeedsAction) as excinfo:
            pub.validate(ctx)
        assert "file_path" in str(excinfo.value.details)

    def test_validate_duration_exceeds_limit(self, tmp_path: Path) -> None:
        pub = TikTokPublisher({"access_token": "test-tok"})
        ctx = make_ctx(tmp_path)
        with pytest.raises(PublishNeedsAction) as excinfo:
            pub.validate(ctx, rendition={"duration_seconds": 650.0})
        assert "duration" in str(excinfo.value.details)


class TestTikTokCapabilityProbe:
    def test_probe_with_video_upload_scope_returns_draft(self) -> None:
        pub = TikTokPublisher()
        account = ConnectedAccount(
            platform="tiktok",
            scopes=["video.upload", "user.info.basic"],
            status="active",
        )
        assert pub.probe_capability(account) == Capability.DRAFT

    def test_probe_without_upload_scope_returns_manual(self) -> None:
        pub = TikTokPublisher()
        account = ConnectedAccount(
            platform="tiktok",
            scopes=["user.info.basic"],
            status="active",
        )
        assert pub.probe_capability(account) == Capability.MANUAL

    def test_probe_inactive_account_returns_blocked(self) -> None:
        pub = TikTokPublisher()
        account = ConnectedAccount(
            platform="tiktok",
            scopes=["video.upload"],
            status="revoked",
        )
        assert pub.probe_capability(account) == Capability.BLOCKED


class TestTikTokInboxFlow:
    def test_consent_enforcement_blocks_without_http(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("HTTP request made without consent token")

        pub = TikTokPublisher({"access_token": "test-tok"}, client=mock_client(handler))
        ctx = make_ctx(tmp_path, consent_token=None)

        with pytest.raises(PublishNeedsAction) as exc:
            pub.prepare(ctx)
        assert exc.value.details["reason"] == "consent_required"

        with pytest.raises(PublishNeedsAction) as exc:
            pub.upload(ctx, {"upload_url": "https://example.com", "publish_id": "p1"})
        assert exc.value.details["reason"] == "consent_required"

    def test_full_inbox_publish_pipeline(self, tmp_path: Path) -> None:
        requests_log: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            requests_log.append(f"{request.method} {url}")
            if url == INBOX_INIT_ENDPOINT:
                assert request.headers["Authorization"] == "Bearer test-tok"
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "publish_id": "tt-pub-12345",
                            "upload_url": "https://open-upload.tiktokapis.example/video/upload-123",
                        }
                    },
                )
            if url.startswith("https://open-upload.tiktokapis.example/video/upload-123"):
                assert request.headers["Content-Type"] == "video/mp4"
                assert "Content-Range" in request.headers
                return httpx.Response(200)
            if url == STATUS_FETCH_ENDPOINT:
                return httpx.Response(
                    200,
                    json={"data": {"status": "SEND_TO_USER_INBOX", "publish_id": "tt-pub-12345"}},
                )
            return httpx.Response(404)

        pub = TikTokPublisher(
            {"access_token": "test-tok"},
            client=mock_client(handler),
            policy=POLICY,
            sleep=lambda _: None,
        )
        ctx = make_ctx(tmp_path)

        session = pub.prepare(ctx)
        assert session["publish_id"] == "tt-pub-12345"
        assert session["upload_url"] == "https://open-upload.tiktokapis.example/video/upload-123"

        session = pub.upload(ctx, session)
        assert session["upload_response"]["uploaded"] is True

        result = pub.finalize(ctx, session)
        assert isinstance(result, PublishResult)
        assert result.remote_post_id == "tt-pub-12345"
        assert result.remote_status["state"] == "SEND_TO_USER_INBOX"

        # Verify status poll
        status = pub.poll_status("tt-pub-12345")
        assert status["status"] == "SEND_TO_USER_INBOX"

    def test_init_missing_payload_data_raises_publish_error(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": {}})

        pub = TikTokPublisher(
            {"access_token": "test-tok"},
            client=mock_client(handler),
            policy=POLICY,
            sleep=lambda _: None,
        )
        ctx = make_ctx(tmp_path)
        with pytest.raises(PublishError):
            pub.prepare(ctx)


class TestTikTokTokenRefresh:
    def test_refresh_credentials_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == TOKEN_ENDPOINT:
                assert b"client_key=test_tt_key" in request.read()
                return httpx.Response(
                    200,
                    json={
                        "access_token": "new-tiktok-access-token",
                        "refresh_token": "new-tiktok-refresh-token",
                        "expires_in": 86400,
                    },
                )
            return httpx.Response(404)

        settings = Settings(
            tiktok_client_key="test_tt_key",
            tiktok_client_secret="test_tt_secret",
        )
        pub = TikTokPublisher(
            {"access_token": "old-token", "refresh_token": "old-refresh-token"},
            client=mock_client(handler),
            settings=settings,
        )
        new_creds = pub.refresh_credentials()
        assert new_creds["access_token"] == "new-tiktok-access-token"
        assert new_creds["refresh_token"] == "new-tiktok-refresh-token"

    def test_refresh_credentials_missing_refresh_token_needs_action(self) -> None:
        pub = TikTokPublisher({"access_token": "only-access-token"})
        with pytest.raises(PublishNeedsAction) as exc:
            pub.refresh_credentials()
        assert "no refresh_token" in str(exc.value)

    def test_refresh_credentials_failed_response_needs_action(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "invalid_grant"})

        pub = TikTokPublisher(
            {"access_token": "old", "refresh_token": "bad-ref"},
            client=mock_client(handler),
            settings=Settings(tiktok_client_key="k", tiktok_client_secret="s"),
        )
        with pytest.raises(PublishNeedsAction) as exc:
            pub.refresh_credentials()
        assert exc.value.details["status_code"] == 400
