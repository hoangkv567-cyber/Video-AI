# ruff: noqa: E402 — test env vars must be set before app modules are imported.
"""Shared fixtures: sqlite session factory over Base.metadata and a generic
VideoPlan v1 dict factory. Kept generic so other suites can reuse them."""

import copy
import os
from collections.abc import Callable, Generator

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.sqlite3")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401 — registers all ORM tables on Base.metadata
from app.db import Base
from app.models import Campaign, Creative


@pytest.fixture()
def db_engine() -> Generator[Engine, None, None]:
    """Isolated in-memory SQLite engine with the full schema created."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine: Engine) -> Generator[Session, None, None]:
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def creative(db_session: Session) -> Creative:
    """A persisted Campaign + Creative with the default 6 USD cost cap."""
    campaign = Campaign(name="Test campaign", brief="Emerging AI news", category="ai")
    db_session.add(campaign)
    db_session.flush()
    row = Creative(
        campaign_id=campaign.id,
        topic_title="New AI model launch",
        angle="Why it matters",
        cost_cap_usd=6.0,
    )
    db_session.add(row)
    db_session.commit()
    return row


# ---------------------------------------------------------------------------
# VideoPlan v1 dict factory (generic — reused by other suites)
# ---------------------------------------------------------------------------

_VI_NARRATIONS = [
    "Một mô hình AI mới vừa được công bố với nhiều cải tiến lớn.",
    "Mô hình xử lý văn bản và hình ảnh nhanh hơn thế hệ trước.",
    "Chi phí vận hành giảm mạnh so với phiên bản cũ.",
    "Các nhà phát triển đã có thể đăng ký dùng thử ngay.",
    "Đây có thể là bước ngoặt cho ứng dụng AI tại Việt Nam.",
]

_EN_NARRATIONS = [
    "A brand new AI model just launched with major upgrades.",
    "It handles text and images faster than the last generation.",
    "Running costs drop sharply compared with the old version.",
    "Developers can already sign up for early access today.",
    "This could be a turning point for everyday AI apps.",
]


def make_video_plan(**overrides: object) -> dict:
    """Return a fresh, semantically valid VideoPlan v1 dict; top-level overrides applied."""
    plan: dict = {
        "schema_version": "1.0",
        "topic": "New AI model launch",
        "angle": "Why it matters for everyday users",
        "style_guide": {
            "palette": "electric blue and warm orange",
            "mood": "energetic, optimistic",
            "camera": "slow push-in, macro details",
            "consistency_notes": "same protagonist device across scenes",
        },
        "sources": [
            {
                "source_id": "src1",
                "url": "https://openai.com/blog/new-model",
                "title": "Introducing the new model",
                "publisher": "OpenAI",
                "is_official": True,
            },
            {
                "source_id": "src2",
                "url": "https://techcrunch.com/2026/08/20/new-model-launch",
                "title": "New model launches",
                "publisher": "TechCrunch",
                "is_official": False,
            },
        ],
        "facts": [
            {
                "fact_id": "f1",
                "claim": "The model was announced this week.",
                "source_ids": ["src1", "src2"],
            }
        ],
        "scenes": [
            {
                "index": i,
                "duration_seconds": 8.0,
                "keyframe_prompt_en": f"Scene {i}: sleek device on a desk, dramatic light",
                "visual_prompt_en": f"Scene {i}: slow cinematic push-in on a glowing device",
                "negative_prompt_en": "text, logos, watermarks",
                "continuity_note": "same device, same desk",
                "fact_ids": ["f1"],
                "is_hero": i == 0,
            }
            for i in range(5)
        ],
        "locales": {
            "vi": {
                "narration": list(_VI_NARRATIONS),
                "on_screen_text": [f"Điểm nhấn {i + 1}" for i in range(5)],
                "title": "Mô hình AI mới có gì đặc biệt?",
                "description": "Tóm tắt nhanh về mô hình AI mới nhất.",
                "hashtags": ["#AI", "#congnghe"],
            },
            "en": {
                "narration": list(_EN_NARRATIONS),
                "on_screen_text": [f"Highlight {i + 1}" for i in range(5)],
                "title": "What's special about the new AI model?",
                "description": "A quick rundown of the newest AI model.",
                "hashtags": ["#AI", "#tech"],
            },
        },
        "disclosure": {
            "synthetic_media": True,
            "made_for_kids": False,
            "content_risks": [],
        },
        "expected_cost_usd": 0.0,
    }
    plan.update(copy.deepcopy(overrides))
    return plan


@pytest.fixture()
def video_plan_factory() -> Callable[..., dict]:
    return make_video_plan


@pytest.fixture()
def video_plan_dict() -> dict:
    return make_video_plan()
