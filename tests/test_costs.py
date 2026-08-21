"""CostLedger: recording, totals, price lookups, cap enforcement and overrides."""

import pytest
from sqlalchemy.orm import Session

from app.ai.base import FakeImageProvider
from app.ai.keyframes import KeyframeService, prompt_hash
from app.config import get_model_config
from app.costs import (
    CostLedger,
    image_price_usd,
    text_call_price_usd,
    tts_price_per_char_usd,
    veo_price_per_second,
)
from app.errors import CostCapExceeded, ValidationFailed
from app.models import AuditEvent, CostEvent, Creative
from app.schemas.videoplan import VideoPlan


def make_ledger(db_session: Session) -> CostLedger:
    return CostLedger(db_session, default_cap_usd=6.0)


class TestPriceLookups:
    def test_veo_lite_price_from_model_config(self) -> None:
        cfg = get_model_config()
        assert veo_price_per_second(cfg.veo_model_lite, cfg) == pytest.approx(0.05)

    def test_veo_fast_price_from_model_config(self) -> None:
        cfg = get_model_config()
        assert veo_price_per_second(cfg.veo_model_fast, cfg) == pytest.approx(0.15)

    def test_unknown_veo_model_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown veo model"):
            veo_price_per_second("veo-9000-ultra", get_model_config())

    def test_other_prices(self) -> None:
        cfg = get_model_config()
        assert image_price_usd(cfg) == pytest.approx(0.039)
        assert text_call_price_usd(cfg) == pytest.approx(0.01)
        assert tts_price_per_char_usd(cfg) == pytest.approx(16.0 / 1_000_000)


class TestRecording:
    def test_projected_and_actual_totals(self, db_session: Session, creative: Creative) -> None:
        ledger = make_ledger(db_session)
        cfg = get_model_config()
        ledger.record_projected(
            creative.id, kind="veo", model_id=cfg.veo_model_lite, units=40.0, unit_price_usd=0.05
        )
        ledger.record_actual(
            creative.id, kind="veo", model_id=cfg.veo_model_lite, units=8.0, unit_price_usd=0.05
        )
        assert ledger.total_projected(creative.id) == pytest.approx(2.0)
        assert ledger.total_actual(creative.id) == pytest.approx(0.4)

        rows = db_session.query(CostEvent).filter_by(creative_id=creative.id).all()
        assert {row.projected for row in rows} == {True, False}
        assert all(row.amount_usd == pytest.approx(row.units * row.unit_price_usd) for row in rows)

    def test_explicit_amount_wins_over_units(self, db_session: Session, creative: Creative) -> None:
        ledger = make_ledger(db_session)
        event = ledger.record_actual(creative.id, kind="other", amount_usd=1.23)
        assert event.amount_usd == pytest.approx(1.23)

    def test_remove_projected_filters_by_kind_and_note(
        self, db_session: Session, creative: Creative
    ) -> None:
        ledger = make_ledger(db_session)
        ledger.record_projected(creative.id, kind="veo", units=8.0, unit_price_usd=0.05, note="s1")
        ledger.record_projected(creative.id, kind="veo", units=8.0, unit_price_usd=0.05, note="s2")
        ledger.record_projected(creative.id, kind="tts", units=100.0, unit_price_usd=0.00001)

        removed = ledger.remove_projected(creative.id, kind="veo", note="s1")
        assert removed == 1
        assert ledger.total_projected(creative.id) == pytest.approx(0.4 + 0.001)


class TestCapEnforcement:
    def test_under_cap_passes(self, db_session: Session, creative: Creative) -> None:
        ledger = make_ledger(db_session)
        check = ledger.check_cap(creative, 2.0)
        assert not check.overridden
        assert check.cap_usd == pytest.approx(6.0)

    def test_actual_plus_projected_plus_additional_exceeds(
        self, db_session: Session, creative: Creative
    ) -> None:
        ledger = make_ledger(db_session)
        ledger.record_actual(creative.id, kind="veo", amount_usd=3.0)
        ledger.record_projected(creative.id, kind="veo", amount_usd=2.5)
        with pytest.raises(CostCapExceeded) as excinfo:
            ledger.check_cap(creative, 1.0)
        assert excinfo.value.code == "cost_cap_exceeded"
        assert excinfo.value.details["cap_usd"] == pytest.approx(6.0)

    def test_creative_specific_cap_respected(self, db_session: Session, creative: Creative) -> None:
        creative.cost_cap_usd = 1.0
        ledger = make_ledger(db_session)
        with pytest.raises(CostCapExceeded):
            ledger.check_cap(creative, 1.5)

    def test_override_without_reason_rejected(
        self, db_session: Session, creative: Creative
    ) -> None:
        ledger = make_ledger(db_session)
        with pytest.raises(ValidationFailed):
            ledger.check_cap(creative, 10.0, override=True, override_reason="  ")
        assert db_session.query(AuditEvent).count() == 0

    def test_override_with_reason_records_audit_event(
        self, db_session: Session, creative: Creative
    ) -> None:
        ledger = make_ledger(db_session)
        check = ledger.check_cap(
            creative,
            10.0,
            override=True,
            override_reason="hero campaign, approved by lead",
            actor_id=None,
        )
        assert check.overridden

        audits = db_session.query(AuditEvent).filter_by(action="cost_cap_override").all()
        assert len(audits) == 1
        audit = audits[0]
        assert audit.entity_type == "creative"
        assert audit.entity_id == creative.id
        assert audit.data["reason"] == "hero campaign, approved by lead"
        assert audit.data["additional_usd"] == pytest.approx(10.0)


class TestKeyframeCosts:
    """KeyframeService goes through the ledger for every image call."""

    def test_style_board_records_actual_cost(
        self, db_session: Session, creative: Creative, video_plan_dict: dict
    ) -> None:
        ledger = make_ledger(db_session)
        service = KeyframeService(FakeImageProvider(), ledger)
        plan = VideoPlan.model_validate(video_plan_dict)

        board = service.generate_style_board(creative, plan)
        assert board.kind == "styleboard"
        assert board.image_bytes.startswith(FakeImageProvider.PNG_STUB)
        assert board.prompt_hash == prompt_hash(board.prompt, board.model_id)
        assert len(board.prompt_hash) == 64
        assert len(board.sha256) == 64
        assert board.as_asset_meta()["size_bytes"] == len(board.image_bytes)

        events = db_session.query(CostEvent).filter_by(creative_id=creative.id).all()
        assert len(events) == 1
        assert events[0].kind == "gemini_image"
        assert not events[0].projected
        assert events[0].amount_usd == pytest.approx(0.039)

    def test_generate_all_cap_blocks_when_exhausted(
        self, db_session: Session, creative: Creative, video_plan_dict: dict
    ) -> None:
        creative.cost_cap_usd = 0.10  # room for two images only
        ledger = make_ledger(db_session)
        service = KeyframeService(FakeImageProvider(), ledger)
        plan = VideoPlan.model_validate(video_plan_dict)

        with pytest.raises(CostCapExceeded):
            service.generate_all(creative, plan)
        # Two images landed before the third check tripped the cap.
        assert ledger.total_actual(creative.id) == pytest.approx(0.078)
