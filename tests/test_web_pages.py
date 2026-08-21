"""Dashboard page tests: login flow, rendering, role gates, timezone display.

Runs fully offline against SQLite; env must be set before app modules import
because the engine is created at import time.
"""

import os

os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///./test.sqlite3"
os.environ["SECRET_KEY"] = "test-secret-key-not-a-real-secret"

import re  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import (  # noqa: E402
    Campaign,
    ConnectedAccount,
    CostEvent,
    Creative,
    PublishTarget,
    Rendition,
    Scene,
    ScriptVersion,
    Source,
    User,
)
from app.schemas.videoplan import (  # noqa: E402
    Fact,
    LocaleContent,
    ScenePlan,
    SourceRef,
    VideoPlan,
)
from app.states import Capability, CreativeState, Platform, PublishTargetStatus, Role  # noqa: E402
from app.web import auth  # noqa: E402
from app.web.filters import format_hcm_time, hcm_datetime_input  # noqa: E402

ADMIN_EMAIL = "admin@example.com"
PUBLISHER_EMAIL = "publisher@example.com"
PASSWORD = "correct horse battery staple"

# 10:30 UTC == 17:30 Asia/Ho_Chi_Minh (+07:00, no DST)
SCHEDULED_UTC = datetime(2026, 1, 1, 10, 30, tzinfo=UTC)
SCHEDULED_HCM_TEXT = "17:30 01/01/2026"


def _make_plan() -> dict:
    scenes = [
        ScenePlan(
            index=i,
            keyframe_prompt_en=f"keyframe prompt {i}",
            visual_prompt_en=f"visual prompt {i}",
            fact_ids=["f1"],
            is_hero=(i == 0),
        )
        for i in range(5)
    ]
    locales = {
        "vi": LocaleContent(
            narration=[f"Lời thoại cảnh {i + 1}" for i in range(5)],
            on_screen_text=[f"Chữ màn hình {i + 1}" for i in range(5)],
            title="Chip AI mới ra mắt",
            description="Mô tả tiếng Việt",
            hashtags=["#ai", "#congnghe"],
        ),
        "en": LocaleContent(
            narration=[f"Narration scene {i + 1}" for i in range(5)],
            on_screen_text=[f"Overlay {i + 1}" for i in range(5)],
            title="New AI chip launched",
            description="English description",
            hashtags=["#ai", "#tech"],
        ),
    }
    plan = VideoPlan(
        topic="Chip AI mới",
        angle="Tác động tới người dùng phổ thông",
        sources=[
            SourceRef(source_id="s1", url="https://example.com/official", is_official=True),
            SourceRef(source_id="s2", url="https://example.org/report"),
        ],
        facts=[Fact(fact_id="f1", claim="Chip nhanh gấp đôi", source_ids=["s1"])],
        scenes=scenes,
        locales=locales,
        expected_cost_usd=2.4,
    )
    return plan.model_dump(mode="json")


@pytest.fixture(scope="session")
def seeded() -> dict[str, str]:
    """Fresh schema + one user of each role, a campaign, and a full creative."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        admin = User(
            email=ADMIN_EMAIL,
            name="Quản trị viên",
            role=Role.ADMIN.value,
            password_hash=auth.hash_password(PASSWORD),
        )
        publisher = User(
            email=PUBLISHER_EMAIL,
            name="Người xuất bản",
            role=Role.PUBLISHER.value,
            password_hash=auth.hash_password(PASSWORD),
        )
        campaign = Campaign(name="Tin AI tuần 34", brief="Brief thử nghiệm", category="AI")
        db.add_all([admin, publisher, campaign])
        db.flush()

        creative = Creative(
            campaign_id=campaign.id,
            state=CreativeState.SCRIPT_READY.value,
            topic_title="Chip AI mới",
            angle="Tác động tới người dùng phổ thông",
        )
        db.add(creative)
        db.flush()

        db.add_all(
            [
                Source(
                    creative_id=creative.id,
                    url="https://example.com/official",
                    title="Thông cáo chính thức",
                    publisher="Example Corp",
                    is_official=True,
                ),
                Source(
                    creative_id=creative.id,
                    url="https://example.org/report",
                    title="Bài phân tích độc lập",
                    publisher="Example News",
                ),
            ]
        )

        script = ScriptVersion(
            creative_id=creative.id, version=1, video_plan=_make_plan(), created_by=admin.id
        )
        db.add(script)
        db.flush()
        for i in range(5):
            db.add(
                Scene(
                    creative_id=creative.id,
                    script_version_id=script.id,
                    index=i,
                    keyframe_prompt_en=f"keyframe prompt {i}",
                    visual_prompt_en=f"visual prompt {i}",
                    status="failed" if i == 2 else "done",
                    is_hero=(i == 0),
                    last_error={"code": "veo_timeout", "message": "hết thời gian"} if i == 2 else None,
                )
            )

        # Cost ledger: projected vs actual, well under the 6 USD hard cap.
        db.add_all(
            [
                CostEvent(
                    creative_id=creative.id,
                    kind="veo",
                    model_id="veo-3.1-lite-generate-preview",
                    units=40.0,
                    unit_price_usd=0.05,
                    amount_usd=2.0,
                    projected=True,
                ),
                CostEvent(
                    creative_id=creative.id,
                    kind="veo",
                    model_id="veo-3.1-lite-generate-preview",
                    units=16.0,
                    unit_price_usd=0.05,
                    amount_usd=0.8,
                    projected=False,
                ),
                CostEvent(
                    creative_id=creative.id,
                    kind="tts",
                    model_id="vi-VN-Neural2-A",
                    units=1200.0,
                    unit_price_usd=0.000016,
                    amount_usd=0.02,
                    projected=False,
                ),
            ]
        )

        rendition_vi = Rendition(
            creative_id=creative.id,
            locale="vi",
            title="Chip AI mới ra mắt",
            hashtags=["#ai"],
        )
        rendition_en = Rendition(
            creative_id=creative.id,
            locale="en",
            title="New AI chip launched",
            hashtags=["#ai"],
        )
        account = ConnectedAccount(
            platform=Platform.YOUTUBE.value,
            display_name="Kênh nội bộ",
            capability=Capability.DIRECT.value,
        )
        db.add_all([rendition_vi, rendition_en, account])
        db.flush()

        db.add_all(
            [
                PublishTarget(
                    creative_id=creative.id,
                    rendition_id=rendition_vi.id,
                    platform=Platform.YOUTUBE.value,
                    connected_account_id=account.id,
                    scheduled_at=SCHEDULED_UTC,
                    privacy="private",
                    status=PublishTargetStatus.SCHEDULED_REMOTE.value,
                    remote_post_id="yt-abc-123",
                ),
                PublishTarget(
                    creative_id=creative.id,
                    rendition_id=rendition_en.id,
                    platform=Platform.TIKTOK.value,
                    status=PublishTargetStatus.MANUAL_BUNDLE.value,
                    bundle_path="bundles/tiktok/abc.zip",
                    last_error={"code": "policy_blocked", "message": "cần thao tác thủ công"},
                ),
            ]
        )
        db.commit()
        return {"creative_id": creative.id, "campaign_id": campaign.id}
    finally:
        db.close()


@pytest.fixture()
def client(seeded: dict[str, str]) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _extract_csrf(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match, "csrf token not found in page"
    return match.group(1)


def _login(client: TestClient, email: str = ADMIN_EMAIL, password: str = PASSWORD):
    page = client.get("/login")
    assert page.status_code == 200
    return client.post(
        "/login",
        data={
            "email": email,
            "password": password,
            "csrf_token": _extract_csrf(page.text),
            "next": "/",
        },
        follow_redirects=False,
    )


# --- Password hashing / CSRF units ------------------------------------------


def test_password_hash_roundtrip() -> None:
    stored = auth.hash_password("mật khẩu bí mật")
    assert stored.startswith("pbkdf2_sha256$")
    assert auth.verify_password("mật khẩu bí mật", stored)
    assert not auth.verify_password("sai mật khẩu", stored)


def test_password_hash_uses_per_user_salt() -> None:
    assert auth.hash_password("same") != auth.hash_password("same")


def test_verify_password_rejects_malformed_stored_value() -> None:
    assert not auth.verify_password("x", "not-a-real-hash")
    assert not auth.verify_password("x", "")


def test_csrf_token_binding() -> None:
    token = auth.make_csrf_token("user-1")
    assert auth.verify_csrf_token(token, "user-1")
    assert not auth.verify_csrf_token(token, "user-2")
    assert not auth.verify_csrf_token("garbage", "user-1")


# --- Timezone filter ---------------------------------------------------------


def test_timezone_filter_converts_utc_to_hcm() -> None:
    assert format_hcm_time(SCHEDULED_UTC) == SCHEDULED_HCM_TEXT
    # Naive datetimes (SQLite round-trip) are treated as UTC.
    assert format_hcm_time(SCHEDULED_UTC.replace(tzinfo=None)) == SCHEDULED_HCM_TEXT
    assert format_hcm_time(None) == "—"
    assert hcm_datetime_input(SCHEDULED_UTC) == "2026-01-01T17:30"


# --- Login flow --------------------------------------------------------------


def test_login_page_renders(client: TestClient) -> None:
    resp = client.get("/login")
    assert resp.status_code == 200
    assert "Đăng nhập" in resp.text
    assert 'name="csrf_token"' in resp.text


def test_login_success_sets_cookie_and_redirects(client: TestClient) -> None:
    resp = _login(client)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/"
    assert auth.SESSION_COOKIE in resp.cookies

    home = client.get("/")
    assert home.status_code == 200
    assert "Tổng quan" in home.text


def test_login_wrong_password_rejected(client: TestClient) -> None:
    resp = _login(client, password="wrong-password")
    assert resp.status_code == 401
    assert auth.SESSION_COOKIE not in resp.cookies
    assert "Sai email hoặc mật khẩu" in resp.text


def test_login_without_csrf_rejected(client: TestClient) -> None:
    resp = client.post(
        "/login",
        data={"email": ADMIN_EMAIL, "password": PASSWORD, "csrf_token": "forged"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert auth.SESSION_COOKIE not in resp.cookies


def test_logout_clears_session(client: TestClient) -> None:
    _login(client)
    home = client.get("/")
    resp = client.post(
        "/logout", data={"csrf_token": _extract_csrf(home.text)}, follow_redirects=False
    )
    assert resp.status_code == 303
    after = client.get("/", follow_redirects=False)
    assert after.status_code == 303
    assert after.headers["location"].startswith("/login")


# --- Authorization -----------------------------------------------------------


def test_anonymous_user_redirected_to_login(client: TestClient) -> None:
    for path in ("/", "/briefs/new", "/connections", "/publishing"):
        resp = client.get(path, follow_redirects=False)
        assert resp.status_code == 303, path
        assert resp.headers["location"].startswith("/login"), path


def test_publisher_role_blocked_from_brief_form(client: TestClient) -> None:
    resp = _login(client, email=PUBLISHER_EMAIL)
    assert resp.status_code == 303
    blocked = client.get("/briefs/new")
    assert blocked.status_code == 403
    # Non-mutating pages remain accessible for publishers.
    assert client.get("/publishing").status_code == 200


# --- Page rendering ----------------------------------------------------------


def test_overview_lists_creatives_states_and_costs(client: TestClient) -> None:
    _login(client)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Tổng quan" in resp.text
    assert "Chip AI mới" in resp.text
    assert "Kịch bản sẵn sàng" in resp.text  # SCRIPT_READY label
    assert "Chi phí dự kiến" in resp.text
    assert "$2.00" in resp.text  # projected
    assert "$0.82" in resp.text  # actual
    assert "$6.00" in resp.text  # cap


def test_connections_page_shows_capability_badges(client: TestClient) -> None:
    _login(client)
    resp = client.get("/connections")
    assert resp.status_code == 200
    assert "Kết nối kênh" in resp.text
    assert "DIRECT" in resp.text
    assert "/oauth/youtube/start" in resp.text
    assert "/oauth/zalo/start" in resp.text


def test_brief_form_renders(client: TestClient) -> None:
    _login(client)
    resp = client.get("/briefs/new")
    assert resp.status_code == 200
    assert "Tạo brief mới" in resp.text
    assert "/api/v1/topics/discover" in resp.text


def test_creative_detail_renders_editor_and_planner(
    client: TestClient, seeded: dict[str, str]
) -> None:
    _login(client)
    resp = client.get(f"/creatives/{seeded['creative_id']}")
    assert resp.status_code == 200
    text = resp.text
    # Topic + sources
    assert "Chip AI mới" in text
    assert "Nguồn tham khảo" in text
    assert "Nguồn chính thức" in text
    # Script editor with per-scene VI/EN narration
    assert "Lời thoại (VI)" in text
    assert "Lời thoại (EN)" in text
    assert "Lời thoại cảnh 1" in text
    assert "Narration scene 1" in text
    # Failed scene gets a retry button, progress badges shown
    assert "Thử lại cảnh 3" in text
    assert "failed" in text
    # Costs projected vs actual
    assert "Chi phí dự kiến" in text
    assert "Chi phí thực tế" in text
    # Rendition previews + approve
    assert "Xem trước" in text
    assert "Duyệt bản vi" in text
    assert "Duyệt bản en" in text
    # Publish planner
    assert "Kế hoạch xuất bản" in text
    assert "Giờ đăng (Asia/Ho_Chi_Minh)" in text
    assert 'name="platforms" value="youtube"' in text
    assert "/api/v1/publications" in text


def test_creative_detail_404_for_unknown_id(client: TestClient) -> None:
    _login(client)
    resp = client.get("/creatives/does-not-exist")
    assert resp.status_code == 404


def test_publishing_board_shows_status_and_hcm_time(client: TestClient) -> None:
    _login(client)
    resp = client.get("/publishing")
    assert resp.status_code == 200
    text = resp.text
    assert "Trạng thái xuất bản" in text
    assert "yt-abc-123" in text
    assert "SCHEDULED_REMOTE" in text
    assert "MANUAL_BUNDLE" in text
    assert "Tải bundle" in text
    # UTC 10:30 rendered as 17:30 Asia/Ho_Chi_Minh
    assert SCHEDULED_HCM_TEXT in text


# --- Static assets -----------------------------------------------------------


def test_static_assets_served_locally(client: TestClient) -> None:
    css = client.get("/static/style.css")
    assert css.status_code == 200
    assert "--bg" in css.text
    js = client.get("/static/htmx.min.js")
    assert js.status_code == 200


def test_static_path_traversal_blocked(client: TestClient) -> None:
    resp = client.get("/static/%2e%2e/auth.py")
    assert resp.status_code == 404
