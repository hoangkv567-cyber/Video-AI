"""Cross-platform publisher contract tests: registry, probing, consent, cost ledger."""

import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.sqlite3")

import random  # noqa: E402
from pathlib import Path  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db import Base  # noqa: E402
from app.errors import CostCapExceeded  # noqa: E402
from app.models import Campaign, ConnectedAccount, CostEvent, Creative  # noqa: E402
from app.publishing.base import (  # noqa: E402
    PublishContext,
    Publisher,
    PublishNeedsAction,
    PublishResult,
)
from app.publishing.ledger import (  # noqa: E402
    DbCostLedger,
    InMemoryCostLedger,
    ensure_within_cap,
)
from app.publishing.registry import (  # noqa: E402
    auto_mode_allowed,
    create_publisher,
    probe_and_record,
    supported_platforms,
)
from app.publishing.retry import RetryPolicy  # noqa: E402
from app.publishing.tiktok import TikTokPublisher  # noqa: E402
from app.publishing.zalo import ZaloPublisher  # noqa: E402
from app.states import Capability  # noqa: E402

CONTRACT_OPS = [
    "capabilities",
    "validate",
    "prepare",
    "upload",
    "finalize",
    "poll_status",
    "refresh_credentials",
    "probe_capability",
]

POLICY = RetryPolicy(base_delay_seconds=0.001, cap_delay_seconds=0.002, max_attempts=3)


def make_ctx(tmp_path: Path, **overrides) -> PublishContext:
    video = tmp_path / "clip.mp4"
    if not video.exists():
        video.write_bytes(b"\x00\x01" * 64)
    defaults = dict(
        creative_id="creative-1",
        rendition_id="rendition-vi",
        locale="vi",
        title="Tin AI hom nay",
        description="Mo ta ngan.",
        hashtags=["#AI"],
        file_path=str(video),
    )
    defaults.update(overrides)
    return PublishContext(**defaults)


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestRegistry:
    def test_all_platforms_registered(self) -> None:
        assert supported_platforms() == ["facebook", "tiktok", "youtube", "zalo"]

    @pytest.mark.parametrize("platform", ["youtube", "facebook", "tiktok", "zalo"])
    def test_factory_returns_full_contract(self, platform: str) -> None:
        pub = create_publisher(platform, {"access_token": "canned"})
        assert isinstance(pub, Publisher)
        assert pub.platform == platform
        for op in CONTRACT_OPS:
            assert callable(getattr(pub, op)), f"{platform} is missing {op}"
        caps = pub.capabilities()
        assert caps and all(isinstance(c, Capability) for c in caps)

    def test_unknown_platform_needs_action(self) -> None:
        with pytest.raises(PublishNeedsAction) as excinfo:
            create_publisher("instagram")
        assert "supported" in excinfo.value.details


class TestCapabilityProbe:
    def test_probe_records_on_connected_account(self) -> None:
        account = ConnectedAccount(
            platform="youtube",
            scopes=["https://www.googleapis.com/auth/youtube.upload"],
            status="active",
        )
        capability = probe_and_record(account, credentials={})
        assert capability is Capability.SCHEDULE
        assert account.capability == "SCHEDULE"
        assert account.last_probe_at is not None
        assert account.last_probe_result["capability"] == "SCHEDULE"
        assert auto_mode_allowed(account) is True

    def test_probe_without_scopes_is_manual_and_never_auto(self) -> None:
        account = ConnectedAccount(platform="youtube", scopes=[], status="active")
        assert probe_and_record(account, credentials={}) is Capability.MANUAL
        assert auto_mode_allowed(account) is False

    def test_tiktok_draft_capability_is_not_auto(self) -> None:
        account = ConnectedAccount(
            platform="tiktok", scopes=["video.upload"], status="active"
        )
        assert probe_and_record(account, credentials={}) is Capability.DRAFT
        assert auto_mode_allowed(account) is False

    def test_inactive_account_probes_blocked(self) -> None:
        account = ConnectedAccount(platform="facebook", scopes=[], status="revoked")
        assert probe_and_record(account, credentials={}) is Capability.BLOCKED
        assert auto_mode_allowed(account) is False


class TestTikTokConsent:
    def test_no_consent_token_needs_action_without_any_http(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no HTTP call may happen without consent")

        pub = TikTokPublisher({"access_token": "canned"}, client=mock_client(handler))
        with pytest.raises(PublishNeedsAction) as excinfo:
            pub.prepare(make_ctx(tmp_path))
        assert excinfo.value.details["reason"] == "consent_required"

    def test_consented_inbox_flow_reaches_the_users_inbox(self, tmp_path: Path) -> None:
        recorded: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "inbox/video/init" in url:
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "publish_id": "pub-1",
                            "upload_url": "https://upload.tiktokapis.example/pub-1",
                        }
                    },
                )
            if url.startswith("https://upload.tiktokapis.example/"):
                recorded["content_range"] = request.headers.get("Content-Range")
                return httpx.Response(201)
            if "status/fetch" in url:
                return httpx.Response(200, json={"data": {"status": "SEND_TO_USER_INBOX"}})
            return httpx.Response(404)

        pub = TikTokPublisher(
            {"access_token": "canned"},
            client=mock_client(handler),
            policy=POLICY,
            sleep=lambda s: None,
            rng=random.Random(3),
        )
        ctx = make_ctx(tmp_path, consent_token="consent-abc123")
        session = pub.prepare(ctx)
        session = pub.upload(ctx, session)
        result = pub.finalize(ctx, session)
        assert result.remote_post_id == "pub-1"
        assert result.remote_status["state"] == "SEND_TO_USER_INBOX"
        assert recorded["content_range"].startswith("bytes 0-")
        assert pub.poll_status("pub-1")["status"] == "SEND_TO_USER_INBOX"


class TestZaloDirect:
    def test_direct_capability_uses_the_oa_api(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "upload/video" in url:
                assert request.headers.get("access_token") == "oa-token"
                return httpx.Response(
                    200, json={"error": 0, "data": {"token": "video-token-1"}}
                )
            if "article/create" in url:
                return httpx.Response(
                    200, json={"error": 0, "data": {"token": "article-token-1"}}
                )
            if "article/verify" in url:
                return httpx.Response(
                    200, json={"error": 0, "data": {"id": "article-1", "status": "show"}}
                )
            return httpx.Response(404)

        pub = ZaloPublisher(
            {"access_token": "oa-token"},
            capability=Capability.DIRECT,
            client=mock_client(handler),
            policy=POLICY,
            sleep=lambda s: None,
        )
        ctx = make_ctx(tmp_path)
        session = pub.prepare(ctx)
        session = pub.upload(ctx, session)
        result = pub.finalize(ctx, session)
        assert isinstance(result, PublishResult)
        assert result.remote_post_id == "article-token-1"
        assert pub.poll_status("article-token-1")["id"] == "article-1"

    def test_zalo_body_error_code_maps_to_needs_action(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": -201, "message": "no permission"})

        pub = ZaloPublisher(
            {"access_token": "oa-token"},
            capability=Capability.DIRECT,
            client=mock_client(handler),
            policy=POLICY,
            sleep=lambda s: None,
        )
        ctx = make_ctx(tmp_path)
        with pytest.raises(PublishNeedsAction) as excinfo:
            pub.upload(ctx, pub.prepare(ctx))
        assert excinfo.value.details["error"] == -201


class TestCostLedger:
    def test_publisher_blocks_when_creative_is_over_the_hard_cap(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no upstream call may happen past the cost cap")

        ledger = InMemoryCostLedger(initial_spend_usd=6.5)  # over the 6 USD hard cap
        pub = create_publisher(
            "youtube", {"access_token": "canned"}, client=mock_client(handler), ledger=ledger
        )
        with pytest.raises(CostCapExceeded):
            pub.prepare(make_ctx(tmp_path))

    def test_lower_creative_cap_wins_over_hard_cap(self, tmp_path: Path) -> None:
        ledger = InMemoryCostLedger(initial_spend_usd=3.0)
        pub = create_publisher(
            "youtube",
            {"access_token": "canned"},
            client=mock_client(lambda request: httpx.Response(500)),
            ledger=ledger,
        )
        ctx = make_ctx(tmp_path, extra={"cost_cap_usd": 2.5})
        with pytest.raises(CostCapExceeded):
            pub.prepare(ctx)

    def test_db_ledger_records_cost_events_and_sums_actuals_only(self) -> None:
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            campaign = Campaign(name="pilot")
            session.add(campaign)
            session.flush()
            creative = Creative(campaign_id=campaign.id, cost_cap_usd=6.0)
            session.add(creative)
            session.flush()

            ledger = DbCostLedger(session, creative.id, job_id="job-1")
            ledger.record(kind="veo", model_id="veo-lite", units=40.0,
                          unit_price_usd=0.05, amount_usd=2.0, note="5 clips")
            ledger.record(kind="other", note="youtube:videos.insert.init")

            # Projected events must not count as actual spend.
            session.add(
                CostEvent(creative_id=creative.id, kind="veo", amount_usd=99.0, projected=True)
            )
            session.flush()

            assert ledger.total_spent_usd() == pytest.approx(2.0)
            rows = session.query(CostEvent).filter_by(creative_id=creative.id).all()
            assert len(rows) == 3
            assert {r.job_id for r in rows if not r.projected} == {"job-1"}

    def test_ensure_within_cap_counts_projected_spend(self) -> None:
        ledger = InMemoryCostLedger(initial_spend_usd=5.0)
        ensure_within_cap(ledger, creative_cap_usd=6.0, hard_cap_usd=6.0, projected_usd=0.5)
        with pytest.raises(CostCapExceeded):
            ensure_within_cap(
                ledger, creative_cap_usd=6.0, hard_cap_usd=6.0, projected_usd=1.5
            )
