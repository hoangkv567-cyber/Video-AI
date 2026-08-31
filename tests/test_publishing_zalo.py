"""Zalo OA publisher test suite (PLAN.md §3, Week 6).

Covers:
- Capability probing (DIRECT, MANUAL, BLOCKED)
- OA video upload & article creation pipeline
- Body error code mapping (HTTP 200 with error != 0 -> PublishNeedsAction)
- Manual bundle degradation when account lacks DIRECT permission
- Token refresh with app ID/secret
- Validation and status verification
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
from app.publishing.zalo import (
    ARTICLE_CREATE_ENDPOINT,
    ARTICLE_VERIFY_ENDPOINT,
    TOKEN_ENDPOINT,
    UPLOAD_VIDEO_ENDPOINT,
    ZaloPublisher,
)
from app.states import Capability

POLICY = RetryPolicy(base_delay_seconds=0.001, cap_delay_seconds=0.002, max_attempts=3)


def make_ctx(tmp_path: Path, **overrides: Any) -> PublishContext:
    video = tmp_path / "zalo_clip.mp4"
    if not video.exists():
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100)
    defaults: dict[str, Any] = dict(
        creative_id="creative-zalo-1",
        rendition_id="rendition-vi",
        locale="vi",
        title="Bản tin AI hôm nay",
        description="Tổng hợp công nghệ mới nhất",
        hashtags=["#AI", "#Tech"],
        file_path=str(video),
    )
    defaults.update(overrides)
    return PublishContext(**defaults)


def mock_client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestZaloPublisherContract:
    def test_capabilities(self) -> None:
        pub = ZaloPublisher({"access_token": "oa-tok"})
        caps = pub.capabilities()
        assert caps == frozenset({Capability.DIRECT, Capability.SCHEDULE, Capability.MANUAL})

    def test_validate_success(self, tmp_path: Path) -> None:
        pub = ZaloPublisher({"access_token": "oa-tok"})
        ctx = make_ctx(tmp_path)
        pub.validate(ctx)

    def test_validate_missing_title(self, tmp_path: Path) -> None:
        pub = ZaloPublisher({"access_token": "oa-tok"})
        ctx = make_ctx(tmp_path, title="   ")
        with pytest.raises(PublishNeedsAction) as exc:
            pub.validate(ctx)
        assert "title" in str(exc.value.details)

    def test_validate_missing_file(self) -> None:
        pub = ZaloPublisher({"access_token": "oa-tok"})
        ctx = PublishContext(
            creative_id="c1",
            rendition_id="r1",
            locale="vi",
            title="Title",
            description="Desc",
            hashtags=[],
            file_path="",
            file_url=None,
        )
        with pytest.raises(PublishNeedsAction) as exc:
            pub.validate(ctx)
        assert "file_path or file_url" in str(exc.value.details)


class TestZaloCapabilityProbe:
    def test_probe_with_content_scope_returns_direct(self) -> None:
        pub = ZaloPublisher()
        account = ConnectedAccount(
            platform="zalo",
            scopes=["article.create", "oa.info"],
            status="active",
        )
        assert pub.probe_capability(account) == Capability.DIRECT

    def test_probe_with_oa_content_scope_returns_direct(self) -> None:
        pub = ZaloPublisher()
        account = ConnectedAccount(
            platform="zalo",
            scopes=["oa.content"],
            status="active",
        )
        assert pub.probe_capability(account) == Capability.DIRECT

    def test_probe_without_content_scope_returns_manual(self) -> None:
        pub = ZaloPublisher()
        account = ConnectedAccount(
            platform="zalo",
            scopes=["oa.info"],
            status="active",
        )
        assert pub.probe_capability(account) == Capability.MANUAL

    def test_probe_inactive_account_returns_blocked(self) -> None:
        pub = ZaloPublisher()
        account = ConnectedAccount(
            platform="zalo",
            scopes=["article.create"],
            status="expired",
        )
        assert pub.probe_capability(account) == Capability.BLOCKED


class TestZaloDirectPublishFlow:
    def test_full_direct_oa_publishing(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            assert request.headers.get("access_token") == "oa-valid-token"

            if url == UPLOAD_VIDEO_ENDPOINT:
                return httpx.Response(
                    200,
                    json={
                        "error": 0,
                        "message": "Success",
                        "data": {"token": "video-asset-token-999"},
                    },
                )
            if url == ARTICLE_CREATE_ENDPOINT:
                return httpx.Response(
                    200,
                    json={
                        "error": 0,
                        "message": "Success",
                        "data": {"token": "article-token-888"},
                    },
                )
            if url == ARTICLE_VERIFY_ENDPOINT:
                return httpx.Response(
                    200,
                    json={
                        "error": 0,
                        "message": "Success",
                        "data": {"id": "article-id-777", "status": "show"},
                    },
                )
            return httpx.Response(404)

        pub = ZaloPublisher(
            {"access_token": "oa-valid-token"},
            capability=Capability.DIRECT,
            client=mock_client(handler),
            policy=POLICY,
            sleep=lambda _: None,
        )
        ctx = make_ctx(tmp_path)

        session = pub.prepare(ctx)
        assert session["api"] == "zalo_oa"

        session = pub.upload(ctx, session)
        assert session["video_token"] == "video-asset-token-999"

        result = pub.finalize(ctx, session)
        assert isinstance(result, PublishResult)
        assert result.remote_post_id == "article-token-888"
        assert result.remote_status["video_token"] == "video-asset-token-999"

        status = pub.poll_status("article-token-888")
        assert status["id"] == "article-id-777"
        assert status["status"] == "show"

    def test_upload_missing_token_in_response_raises_error(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": 0, "data": {}})

        pub = ZaloPublisher(
            {"access_token": "oa-token"},
            capability=Capability.DIRECT,
            client=mock_client(handler),
            policy=POLICY,
            sleep=lambda _: None,
        )
        ctx = make_ctx(tmp_path)
        with pytest.raises(PublishError):
            pub.upload(ctx, pub.prepare(ctx))


class TestZaloManualFallback:
    def test_manual_capability_degrades_with_bundle(self, tmp_path: Path) -> None:
        pub = ZaloPublisher(
            {"access_token": "oa-token"},
            capability=Capability.MANUAL,
            bundle_dir=str(tmp_path / "zalo_bundles"),
        )
        ctx = make_ctx(tmp_path)
        with pytest.raises(PublishNeedsAction) as exc:
            pub.prepare(ctx)
        assert "manual bundle was created" in str(exc.value)
        assert "deep_link" in exc.value.details
        assert Path(exc.value.details["bundle"]["bundle_dir"]).is_dir()


class TestZaloTokenRefresh:
    def test_refresh_credentials_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == TOKEN_ENDPOINT:
                assert request.headers.get("secret_key") == "zalo_secret"
                return httpx.Response(
                    200,
                    json={
                        "access_token": "new-zalo-access-token",
                        "refresh_token": "new-zalo-refresh-token",
                        "expires_in": 90000,
                    },
                )
            return httpx.Response(404)

        settings = Settings(
            zalo_app_id="zalo_id_123",
            zalo_app_secret="zalo_secret",
        )
        pub = ZaloPublisher(
            {"access_token": "old-tok", "refresh_token": "old-ref"},
            client=mock_client(handler),
            settings=settings,
        )
        creds = pub.refresh_credentials()
        assert creds["access_token"] == "new-zalo-access-token"
        assert creds["refresh_token"] == "new-zalo-refresh-token"

    def test_refresh_missing_refresh_token_needs_action(self) -> None:
        pub = ZaloPublisher({"access_token": "only-access"})
        with pytest.raises(PublishNeedsAction) as exc:
            pub.refresh_credentials()
        assert "no refresh_token" in str(exc.value)

    def test_refresh_error_response_needs_action(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": -14014, "message": "Invalid token"})

        pub = ZaloPublisher(
            {"access_token": "old", "refresh_token": "bad-ref"},
            client=mock_client(handler),
            settings=Settings(zalo_app_id="id", zalo_app_secret="sec"),
        )
        with pytest.raises(PublishNeedsAction) as exc:
            pub.refresh_credentials()
        assert exc.value.details["status_code"] == 401
