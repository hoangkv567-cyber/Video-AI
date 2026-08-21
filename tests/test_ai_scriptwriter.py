"""ScriptService: plan generation, shorten loop, cost projection, cap and auto gates."""

import copy
from collections.abc import Callable

import pytest
from sqlalchemy.orm import Session

from app.ai.base import FakeScriptProvider, SourceInfo
from app.ai.scriptwriter import ScriptService
from app.costs import CostLedger
from app.errors import CostCapExceeded
from app.models import CostEvent, Creative

SOURCES = [
    SourceInfo(url="https://openai.com/blog/new-model", title="Official", is_official=True),
    SourceInfo(url="https://techcrunch.com/2026/08/20/new-model", title="Coverage"),
]

LONG_VI = (
    "Đây là một câu thuyết minh quá dài vượt xa ngân sách bảy phẩy sáu giây cho một cảnh "
    "và chắc chắn cần được rút gọn lại đáng kể trước khi tổng hợp giọng nói tiếng Việt."
)


def make_service(
    db_session: Session, plan: dict, provider: FakeScriptProvider | None = None
) -> tuple[ScriptService, CostLedger, FakeScriptProvider]:
    ledger = CostLedger(db_session, default_cap_usd=6.0)
    provider = provider or FakeScriptProvider(plan)
    return ScriptService(provider, ledger), ledger, provider


class StubbornProvider(FakeScriptProvider):
    """Never actually shortens — exercises the bounded-retry exit."""

    def shorten_narrations(self, plan: dict, issues: list[dict]) -> dict:
        self.shorten_calls.append(issues)
        return copy.deepcopy(plan)


class TestHappyPath:
    def test_valid_plan_passes_and_projects_costs(
        self, db_session: Session, creative: Creative, video_plan_dict: dict
    ) -> None:
        service, ledger, provider = make_service(db_session, video_plan_dict)
        result = service.create_plan(creative, "New AI model", SOURCES, brief="ai news")

        assert result.blocking == []
        assert result.auto_mode_allowed
        assert result.shorten_attempts == 0
        assert provider.shorten_calls == []

        # 5 scenes x 8 s veo lite alone is 2.00 USD; keyframes/text/tts add cents.
        assert 2.0 < result.expected_cost_usd < 3.0
        assert result.plan.expected_cost_usd == pytest.approx(result.expected_cost_usd)
        assert result.plan_dict["expected_cost_usd"] == pytest.approx(result.expected_cost_usd)

        events = (
            db_session.query(CostEvent)
            .filter_by(creative_id=creative.id, projected=True)
            .all()
        )
        assert {e.kind for e in events} == {"veo", "gemini_image", "gemini_text", "tts"}
        assert ledger.total_projected(creative.id) == pytest.approx(
            result.expected_cost_usd, abs=1e-3
        )
        veo = next(e for e in events if e.kind == "veo")
        assert veo.units == pytest.approx(40.0)
        assert veo.unit_price_usd == pytest.approx(0.05)


class TestShortenLoop:
    def test_too_long_narration_is_shortened_and_revalidated(
        self, db_session: Session, creative: Creative, video_plan_factory: Callable[..., dict]
    ) -> None:
        plan = video_plan_factory()
        plan["locales"]["vi"]["narration"][2] = LONG_VI
        service, _, provider = make_service(db_session, plan)

        result = service.create_plan(creative, "New AI model", SOURCES)

        assert result.shorten_attempts == 1
        assert len(provider.shorten_calls) == 1
        issue = provider.shorten_calls[0][0]
        assert issue["code"] == "narration_too_long"
        assert issue["locale"] == "vi"
        assert issue["scene_index"] == 2
        assert not any(i.code == "narration_too_long" for i in result.issues)
        assert result.auto_mode_allowed

    def test_retries_are_bounded_and_issue_survives(
        self, db_session: Session, creative: Creative, video_plan_factory: Callable[..., dict]
    ) -> None:
        plan = video_plan_factory()
        plan["locales"]["vi"]["narration"][1] = LONG_VI
        provider = StubbornProvider(plan)
        service, _, _ = make_service(db_session, plan, provider)

        result = service.create_plan(creative, "New AI model", SOURCES)

        assert result.shorten_attempts == 2  # bounded at MAX_SHORTEN_RETRIES
        assert len(provider.shorten_calls) == 2
        assert any(i.code == "narration_too_long" for i in result.issues)
        assert not result.auto_mode_allowed  # blocking issue disables auto mode


class TestCostGate:
    def test_cap_exceeded_raises_and_records_nothing(
        self, db_session: Session, creative: Creative, video_plan_dict: dict
    ) -> None:
        creative.cost_cap_usd = 1.0  # below the ~2.26 USD projection
        service, ledger, _ = make_service(db_session, video_plan_dict)

        with pytest.raises(CostCapExceeded):
            service.create_plan(creative, "New AI model", SOURCES)
        assert ledger.total_projected(creative.id) == pytest.approx(0.0)
        assert db_session.query(CostEvent).filter_by(creative_id=creative.id).count() == 0


class TestSourceGates:
    def test_fact_with_unknown_source_blocks_auto_mode(
        self, db_session: Session, creative: Creative, video_plan_factory: Callable[..., dict]
    ) -> None:
        plan = video_plan_factory()
        plan["facts"][0]["source_ids"] = ["src-does-not-exist"]
        service, _, _ = make_service(db_session, plan)

        result = service.create_plan(creative, "New AI model", SOURCES)

        assert any(i.code == "fact_source_missing" for i in result.blocking)
        assert not result.auto_mode_allowed

    def test_scene_without_fact_ids_blocks_auto_mode_only(
        self, db_session: Session, creative: Creative, video_plan_factory: Callable[..., dict]
    ) -> None:
        plan = video_plan_factory()
        plan["scenes"][3]["fact_ids"] = []
        service, _, _ = make_service(db_session, plan)

        result = service.create_plan(creative, "New AI model", SOURCES)

        assert result.blocking == []  # not a hard schema failure...
        assert not result.auto_mode_allowed  # ...but auto mode requires sourced scenes
