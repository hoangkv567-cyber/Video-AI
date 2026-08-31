"""ScriptService: plan generation, shorten loop, cost projection, cap and auto gates."""

import copy
from collections.abc import Callable

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.ai.base import FakeScriptProvider, SourceInfo
from app.ai.scriptwriter import ScriptService
from app.config import Settings
from app.costs import CostLedger
from app.errors import CostCapExceeded
from app.models import CostEvent, Creative
from app.schemas.videoplan import VideoPlan, blocking_issues, validate_semantics

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
    settings = Settings(
        video_provider="keyframe_motion",
        tts_provider_en="edge",
        tts_provider_vi="edge",
        _env_file=None,
    )
    return ScriptService(provider, ledger, settings=settings), ledger, provider


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

        # Free-mode projection excludes Veo and prices Edge TTS at zero.
        assert result.expected_cost_usd == pytest.approx(0.422)
        assert result.plan.expected_cost_usd == pytest.approx(result.expected_cost_usd)
        assert result.plan_dict["expected_cost_usd"] == pytest.approx(result.expected_cost_usd)

        events = (
            db_session.query(CostEvent).filter_by(creative_id=creative.id, projected=True).all()
        )
        assert {e.kind for e in events} == {"gemini_image", "gemini_text", "tts"}
        assert all(e.amount_usd == 0 for e in events if e.kind == "tts")
        assert ledger.total_projected(creative.id) == pytest.approx(
            result.expected_cost_usd, abs=1e-3
        )
        assert not any(e.kind == "veo" for e in events)

    def test_paid_provider_projection_includes_veo_and_google_tts(
        self, db_session: Session, creative: Creative, video_plan_dict: dict
    ) -> None:
        ledger = CostLedger(db_session, default_cap_usd=6.0)
        settings = Settings(
            video_provider="veo",
            tts_provider_en="google",
            tts_provider_vi="google",
            _env_file=None,
        )
        service = ScriptService(FakeScriptProvider(video_plan_dict), ledger, settings=settings)

        result = service.create_plan(creative, "New AI model", SOURCES)

        assert 2.0 < result.expected_cost_usd < 3.0
        events = (
            db_session.query(CostEvent).filter_by(creative_id=creative.id, projected=True).all()
        )
        veo = next(event for event in events if event.kind == "veo")
        assert veo.units == pytest.approx(40.0)
        assert veo.unit_price_usd == pytest.approx(0.05)
        assert all(event.unit_price_usd > 0 for event in events if event.kind == "tts")


class TestSourceBoundary:
    def test_two_urls_on_one_domain_are_not_independent(
        self, video_plan_dict: dict
    ) -> None:
        video_plan_dict["sources"][1]["url"] = "https://openai.com/research/second"
        plan = VideoPlan.model_validate(video_plan_dict)

        assert {issue.code for issue in blocking_issues(validate_semantics(plan))} >= {
            "sources_min"
        }

    def test_executable_source_url_is_rejected(self, video_plan_dict: dict) -> None:
        video_plan_dict["sources"][0]["url"] = "javascript:alert(1)"

        with pytest.raises(ValidationError):
            VideoPlan.model_validate(video_plan_dict)


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
        creative.cost_cap_usd = 0.10  # below the free-mode image/text projection
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

    def test_declared_content_risk_requires_manual_review(
        self, db_session: Session, creative: Creative, video_plan_factory: Callable[..., dict]
    ) -> None:
        plan = video_plan_factory()
        plan["disclosure"]["content_risks"] = ["public_figure_likeness"]
        service, _, _ = make_service(db_session, plan)

        result = service.create_plan(creative, "New AI model", SOURCES)

        assert result.blocking == []
        assert not result.auto_mode_allowed


class TestFactRepair:
    """Bounded provider repair for plans whose scenes cite missing facts."""

    def test_repair_resolves_missing_facts(
        self,
        db_session: Session,
        creative: Creative,
        video_plan_factory: Callable[..., dict],
    ) -> None:
        broken = video_plan_factory()
        broken["facts"] = []  # scenes still cite f1 -> scene_fact_missing
        fixed = video_plan_factory()

        class RepairableProvider(FakeScriptProvider):
            def __init__(self) -> None:
                super().__init__(broken)
                self.repair_calls = 0

            def repair_plan(self, plan: dict, issues: list[dict]) -> dict:
                self.repair_calls += 1
                return copy.deepcopy(fixed)

        provider = RepairableProvider()
        service, _, _ = make_service(db_session, broken, provider=provider)

        result = service.create_plan(creative, "New AI model", SOURCES)

        assert provider.repair_calls == 1
        assert result.blocking == []
        assert result.auto_mode_allowed

    def test_provider_without_repair_keeps_blocking_issues(
        self,
        db_session: Session,
        creative: Creative,
        video_plan_factory: Callable[..., dict],
    ) -> None:
        broken = video_plan_factory()
        broken["facts"] = []
        service, _, _ = make_service(db_session, broken)

        result = service.create_plan(creative, "New AI model", SOURCES)

        assert any(i.code == "scene_fact_missing" for i in result.blocking)
        assert not result.auto_mode_allowed

    def test_failed_repair_keeps_original_plan(
        self,
        db_session: Session,
        creative: Creative,
        video_plan_factory: Callable[..., dict],
    ) -> None:
        broken = video_plan_factory()
        broken["facts"] = []

        class HopelessProvider(FakeScriptProvider):
            def repair_plan(self, plan: dict, issues: list[dict]) -> dict:
                still_broken = copy.deepcopy(plan)
                still_broken["facts"] = []
                return still_broken

        service, _, _ = make_service(db_session, broken, provider=HopelessProvider(broken))

        result = service.create_plan(creative, "New AI model", SOURCES)

        assert any(i.code == "scene_fact_missing" for i in result.blocking)


class TestFreeImageProjection:
    def test_procedural_image_provider_projects_zero_cost(
        self, db_session: Session, creative: Creative, video_plan_dict: dict
    ) -> None:
        ledger = CostLedger(db_session, default_cap_usd=6.0)
        settings = Settings(
            video_provider="keyframe_motion",
            image_provider="procedural",
            tts_provider_en="edge",
            tts_provider_vi="edge",
            _env_file=None,
        )
        service = ScriptService(
            FakeScriptProvider(video_plan_dict), ledger, settings=settings
        )

        service.create_plan(creative, "New AI model", SOURCES)

        image_events = [
            e
            for e in db_session.query(CostEvent).filter_by(
                creative_id=creative.id, projected=True
            )
            if e.kind in {"gemini_image", "free_image"}
        ]
        assert {e.kind for e in image_events} == {"free_image"}
        assert all(e.amount_usd == 0 for e in image_events)
