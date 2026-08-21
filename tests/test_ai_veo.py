"""VeoService: LRO submit/poll, resume safety after restart, escalation, actual cost."""

import pytest
from sqlalchemy.orm import Session

from app.ai.base import OP_FAILED, OP_RUNNING, OP_SUCCEEDED, FakeVideoProvider
from app.ai.veo import (
    InMemoryOperationStore,
    JobOperationStore,
    StoredOperation,
    VeoService,
)
from app.config import get_model_config
from app.costs import CostLedger
from app.errors import CostCapExceeded, NotFound
from app.models import CostEvent, Creative


def make_service(
    db_session: Session,
    store: InMemoryOperationStore | JobOperationStore | None = None,
    provider: FakeVideoProvider | None = None,
) -> tuple[VeoService, FakeVideoProvider, CostLedger, InMemoryOperationStore | JobOperationStore]:
    provider = provider or FakeVideoProvider()
    ledger = CostLedger(db_session, default_cap_usd=6.0)
    store = store if store is not None else InMemoryOperationStore()
    return VeoService(provider, ledger, store), provider, ledger, store


class TestSubmit:
    def test_submit_persists_operation_before_returning(
        self, db_session: Session, creative: Creative
    ) -> None:
        service, provider, ledger, store = make_service(db_session)
        submission = service.ensure_submitted(creative, "scene-1", "a glowing device")

        assert not submission.resumed
        stored = store.get("scene-1")
        assert stored is not None
        assert stored.operation_name == submission.operation_name
        assert stored.model_id == get_model_config().veo_model_lite
        assert len(provider.submit_calls) == 1
        # Projected cost recorded at submit: 8 s x 0.05 USD/s.
        assert ledger.total_projected(creative.id) == pytest.approx(0.4)

    def test_second_call_resumes_without_new_submit(
        self, db_session: Session, creative: Creative
    ) -> None:
        service, provider, _, _ = make_service(db_session)
        first = service.ensure_submitted(creative, "scene-1", "a glowing device")
        second = service.ensure_submitted(creative, "scene-1", "a glowing device")

        assert second.resumed
        assert second.operation_name == first.operation_name
        assert len(provider.submit_calls) == 1

    def test_cap_blocks_submit_and_stores_nothing(
        self, db_session: Session, creative: Creative
    ) -> None:
        creative.cost_cap_usd = 0.1
        service, provider, _, store = make_service(db_session)
        with pytest.raises(CostCapExceeded):
            service.ensure_submitted(creative, "scene-1", "a glowing device")
        assert provider.submit_calls == []
        assert store.get("scene-1") is None


class TestResumeSafety:
    def test_restart_with_stored_op_makes_zero_new_submit_calls(
        self, db_session: Session, creative: Creative
    ) -> None:
        """Worker restart: fresh provider + service over the surviving store."""
        store = InMemoryOperationStore()
        service, _, _, _ = make_service(db_session, store=store)
        submission = service.ensure_submitted(creative, "scene-1", "a glowing device")

        # --- simulated restart: brand-new provider and service instances -----
        restarted_provider = FakeVideoProvider()
        restarted_service = VeoService(
            restarted_provider, CostLedger(db_session, default_cap_usd=6.0), store
        )
        resumed = restarted_service.ensure_submitted(creative, "scene-1", "a glowing device")

        assert resumed.resumed
        assert resumed.operation_name == submission.operation_name
        assert restarted_provider.submit_calls == []  # ZERO new submits

        # The restarted worker re-polls the stored operation to completion.
        result = restarted_service.poll(creative, "scene-1")
        assert result.status == OP_SUCCEEDED
        assert result.video_bytes is not None
        assert restarted_provider.poll_calls == [submission.operation_name]

    def test_job_operation_store_survives_restart(
        self, db_session: Session, creative: Creative
    ) -> None:
        """DB-backed store: a second store instance over the same DB sees the op."""
        store = JobOperationStore(db_session)
        service, provider, _, _ = make_service(db_session, store=store)
        submission = service.ensure_submitted(creative, "scene-7", "hero shot", is_hero=True)
        assert len(provider.submit_calls) == 1

        fresh_store = JobOperationStore(db_session)
        stored = fresh_store.get("scene-7")
        assert stored is not None
        assert stored.operation_name == submission.operation_name
        assert stored.model_id == get_model_config().veo_model_fast

        restarted_provider = FakeVideoProvider()
        restarted = VeoService(
            restarted_provider, CostLedger(db_session, default_cap_usd=6.0), fresh_store
        )
        resumed = restarted.ensure_submitted(creative, "scene-7", "hero shot", is_hero=True)
        assert resumed.resumed
        assert restarted_provider.submit_calls == []

        result = restarted.poll(creative, "scene-7")
        assert result.status == OP_SUCCEEDED
        assert fresh_store.get("scene-7") is None  # cleared after download

    def test_poll_without_stored_operation_raises(
        self, db_session: Session, creative: Creative
    ) -> None:
        service, _, _, _ = make_service(db_session)
        with pytest.raises(NotFound):
            service.poll(creative, "scene-unknown")


class TestPollingAndCost:
    def test_success_records_actual_cost_per_generated_second(
        self, db_session: Session, creative: Creative
    ) -> None:
        service, _, ledger, store = make_service(db_session)
        service.ensure_submitted(creative, "scene-1", "a glowing device")
        result = service.poll(creative, "scene-1")

        assert result.status == OP_SUCCEEDED
        assert result.duration_seconds == pytest.approx(8.0)
        assert result.actual_cost_usd == pytest.approx(0.4)  # 8 s x 0.05 lite

        assert ledger.total_actual(creative.id) == pytest.approx(0.4)
        assert ledger.total_projected(creative.id) == pytest.approx(0.0)  # superseded
        actuals = (
            db_session.query(CostEvent)
            .filter_by(creative_id=creative.id, projected=False, kind="veo")
            .all()
        )
        assert len(actuals) == 1
        assert actuals[0].units == pytest.approx(8.0)
        assert store.get("scene-1") is None

    def test_running_operation_reports_running(
        self, db_session: Session, creative: Creative
    ) -> None:
        provider = FakeVideoProvider(pending_polls=2)
        service, _, _, _ = make_service(db_session, provider=provider)
        service.ensure_submitted(creative, "scene-1", "a glowing device")

        assert service.poll(creative, "scene-1").status == OP_RUNNING
        assert service.poll(creative, "scene-1").status == OP_RUNNING
        assert service.poll(creative, "scene-1").status == OP_SUCCEEDED


class TestEscalation:
    def test_model_resolution(self, db_session: Session) -> None:
        service, _, _, _ = make_service(db_session)
        cfg = get_model_config()
        assert service.resolve_model() == cfg.veo_model_lite
        assert service.resolve_model(is_hero=True) == cfg.veo_model_fast
        assert service.resolve_model(previously_failed=True) == cfg.veo_model_fast

    def test_failed_scene_resubmits_escalated_to_fast(
        self, db_session: Session, creative: Creative
    ) -> None:
        service, provider, ledger, store = make_service(db_session)
        submission = service.ensure_submitted(creative, "scene-2", "tricky shot")
        provider.fail_operations.add(submission.operation_name)

        failed = service.poll(creative, "scene-2")
        assert failed.status == OP_FAILED
        assert failed.error
        assert store.get("scene-2") is None  # cleared so escalation can resubmit

        retry = service.ensure_submitted(
            creative, "scene-2", "tricky shot", previously_failed=True
        )
        assert not retry.resumed
        assert retry.model_id == get_model_config().veo_model_fast
        assert provider.submit_calls[-1]["model_id"] == get_model_config().veo_model_fast

        provider.fail_operations.clear()
        result = service.poll(creative, "scene-2")
        assert result.status == OP_SUCCEEDED
        # Escalated actual cost: 8 s x 0.15 fast.
        assert result.actual_cost_usd == pytest.approx(1.2)


class TestStoredOperationDefaults:
    def test_dataclass_defaults(self) -> None:
        op = StoredOperation(operation_name="op-1", model_id="m")
        assert op.duration_seconds == pytest.approx(8.0)
