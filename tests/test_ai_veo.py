"""VeoService: LRO submit/poll, resume safety after restart, escalation, actual cost."""

from collections.abc import Callable

import pytest
from sqlalchemy.orm import Session

from app.ai.base import OP_FAILED, OP_RUNNING, OP_SUCCEEDED, FakeVideoProvider
from app.ai.veo import (
    CallbackOperationStore,
    InMemoryOperationStore,
    JobOperationStore,
    StoredOperation,
    VeoService,
)
from app.config import get_model_config
from app.costs import CostLedger
from app.errors import CostCapExceeded, NotFound, PolicyBlocked
from app.models import CostEvent, Creative, Job
from app.states import JobStatus


class SimulatedProcessKill(BaseException):
    """Not caught by normal ``except Exception`` worker cleanup."""


class SideEffectThenKilled(FakeVideoProvider):
    def __init__(self, projected_total: Callable[[], float]) -> None:
        super().__init__()
        self._projected_total = projected_total
        self.projected_seen_during_submit = 0.0

    def submit(
        self,
        *,
        prompt: str,
        model_id: str,
        duration_seconds: float,
        keyframe_bytes: bytes | None = None,
    ) -> str:
        # The parent fake records the remote side effect and creates an op, but
        # the worker dies before that operation name reaches local persistence.
        super().submit(
            prompt=prompt,
            model_id=model_id,
            duration_seconds=duration_seconds,
            keyframe_bytes=keyframe_bytes,
        )
        self.projected_seen_during_submit = float(self._projected_total())
        raise SimulatedProcessKill


class SideEffectThenError(FakeVideoProvider):
    def submit(
        self,
        *,
        prompt: str,
        model_id: str,
        duration_seconds: float,
        keyframe_bytes: bytes | None = None,
    ) -> str:
        super().submit(
            prompt=prompt,
            model_id=model_id,
            duration_seconds=duration_seconds,
            keyframe_bytes=keyframe_bytes,
        )
        raise RuntimeError("SDK lost the response after the remote side effect")


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
        assert store.get("scene-1") is not None
        restarted_service.checkpoint_success(creative, "scene-1", result)
        assert store.get("scene-1") is None

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
        assert stored.provider_kind == "veo"
        assert stored.creative_id == creative.id
        assert stored.cost_event_id

        restarted_provider = FakeVideoProvider()
        restarted = VeoService(
            restarted_provider, CostLedger(db_session, default_cap_usd=6.0), fresh_store
        )
        resumed = restarted.ensure_submitted(creative, "scene-7", "hero shot", is_hero=True)
        assert resumed.resumed
        assert restarted_provider.submit_calls == []

        result = restarted.poll(creative, "scene-7")
        assert result.status == OP_SUCCEEDED
        assert fresh_store.get("scene-7") is not None  # held until the caller stores bytes
        restarted.checkpoint_success(creative, "scene-7", result)
        assert fresh_store.get("scene-7") is None

        rows = (
            db_session.query(Job)
            .filter(Job.kind == JobOperationStore.KIND, Job.idempotency_key == "veo:scene-7")
            .all()
        )
        assert len(rows) == 1  # put completed the original intent row
        assert rows[0].payload["phase"] == "submitted"
        assert rows[0].creative_id == creative.id

    def test_provider_drift_fails_closed_before_poll_or_submit(
        self, db_session: Session, creative: Creative
    ) -> None:
        store = JobOperationStore(db_session)
        service, provider, _, _ = make_service(db_session, store=store)
        service.ensure_submitted(creative, "scene-provider-drift", "paid shot")
        assert len(provider.submit_calls) == 1

        class WrongProviderService(VeoService):
            cost_kind = "wan"

        restarted_provider = FakeVideoProvider()
        restarted = WrongProviderService(
            restarted_provider,
            CostLedger(db_session, default_cap_usd=6.0),
            JobOperationStore(db_session),
        )
        with pytest.raises(PolicyBlocked) as excinfo:
            restarted.ensure_submitted(creative, "scene-provider-drift", "paid shot")

        assert excinfo.value.code == "video_submission_provider_mismatch"
        assert excinfo.value.details["stored_provider"] == "veo"
        assert restarted_provider.submit_calls == []
        assert restarted_provider.poll_calls == []

    def test_poll_without_stored_operation_raises(
        self, db_session: Session, creative: Creative
    ) -> None:
        service, _, _, _ = make_service(db_session)
        with pytest.raises(NotFound):
            service.poll(creative, "scene-unknown")


class TestCrashBetweenSubmitAndOperationPersistence:
    def test_ordinary_submit_error_after_intent_is_also_ambiguous(
        self, db_session: Session, creative: Creative
    ) -> None:
        provider = SideEffectThenError()
        ledger = CostLedger(db_session, default_cap_usd=6.0)
        store = JobOperationStore(db_session)
        service = VeoService(provider, ledger, store)

        with pytest.raises(PolicyBlocked) as excinfo:
            service.ensure_submitted(creative, "scene-error", "paid shot")

        assert excinfo.value.code == "veo_submission_ambiguous"
        assert len(provider.submit_calls) == 1
        assert ledger.total_projected(creative.id) == pytest.approx(0.4)
        persisted = store.get("scene-error")
        assert persisted is not None and persisted.submission_ambiguous

    def test_db_intent_blocks_resubmit_after_remote_side_effect_then_process_kill(
        self, db_session: Session, creative: Creative
    ) -> None:
        ledger = CostLedger(db_session, default_cap_usd=6.0)
        provider = SideEffectThenKilled(lambda: ledger.total_projected(creative.id))
        store = JobOperationStore(db_session)
        service = VeoService(provider, ledger, store)

        with pytest.raises(SimulatedProcessKill):
            service.ensure_submitted(creative, "scene-killed", "paid cinematic shot")

        assert len(provider.submit_calls) == 1  # remote side effect happened
        assert provider.projected_seen_during_submit == pytest.approx(0.4)
        assert ledger.total_projected(creative.id) == pytest.approx(0.4)
        intent = JobOperationStore(db_session).get("scene-killed")
        assert intent is not None
        assert intent.submission_ambiguous
        assert intent.operation_name == ""

        restarted_provider = FakeVideoProvider()
        restarted = VeoService(
            restarted_provider,
            CostLedger(db_session, default_cap_usd=6.0),
            JobOperationStore(db_session),
        )
        with pytest.raises(PolicyBlocked) as excinfo:
            restarted.ensure_submitted(creative, "scene-killed", "paid cinematic shot")

        assert excinfo.value.code == "veo_submission_ambiguous"
        assert excinfo.value.retryable is False
        assert excinfo.value.details == {
            "scene_id": "scene-killed",
            "model_id": get_model_config().veo_model_lite,
            "operator_action_required": True,
        }
        assert restarted_provider.submit_calls == []  # fail closed: ZERO duplicate submits
        rows = (
            db_session.query(Job)
            .filter(
                Job.kind == JobOperationStore.KIND,
                Job.idempotency_key == "veo:scene-killed",
            )
            .all()
        )
        assert len(rows) == 1
        assert rows[0].payload["phase"] == "intent"

    def test_in_memory_intent_also_fails_closed_on_restart_simulation(
        self, db_session: Session, creative: Creative
    ) -> None:
        ledger = CostLedger(db_session, default_cap_usd=6.0)
        provider = SideEffectThenKilled(lambda: ledger.total_projected(creative.id))
        store = InMemoryOperationStore()
        service = VeoService(provider, ledger, store)

        with pytest.raises(SimulatedProcessKill):
            service.ensure_submitted(creative, "scene-memory", "paid shot")

        restarted_provider = FakeVideoProvider()
        restarted = VeoService(
            restarted_provider, CostLedger(db_session, default_cap_usd=6.0), store
        )
        with pytest.raises(PolicyBlocked, match="outcome is unknown") as excinfo:
            restarted.poll(creative, "scene-memory")
        assert excinfo.value.code == "veo_submission_ambiguous"
        assert restarted_provider.submit_calls == []
        assert restarted_provider.poll_calls == []


class TestCallbackStoreCapability:
    def test_legacy_callback_can_resume_but_cannot_start_a_paid_submission(
        self, db_session: Session, creative: Creative
    ) -> None:
        rows = {
            "existing": StoredOperation(
                operation_name="provider-op-existing", model_id="veo-model"
            )
        }
        store = CallbackOperationStore(rows.get, rows.__setitem__, rows.pop)
        provider = FakeVideoProvider()
        service = VeoService(provider, CostLedger(db_session), store)

        resumed = service.ensure_submitted(creative, "existing", "old prompt")
        assert resumed.resumed is True
        assert resumed.operation_name == "provider-op-existing"

        with pytest.raises(PolicyBlocked) as excinfo:
            service.ensure_submitted(creative, "new", "new paid prompt")
        assert excinfo.value.code == "veo_operation_store_unsafe"
        assert provider.submit_calls == []

    def test_callback_with_begin_capability_uses_intent_then_completes_it(
        self, db_session: Session, creative: Creative
    ) -> None:
        rows: dict[str, StoredOperation] = {}
        store = CallbackOperationStore(
            rows.get,
            rows.__setitem__,
            rows.pop,
            begin=rows.__setitem__,
        )
        provider = FakeVideoProvider()
        service = VeoService(provider, CostLedger(db_session), store)

        submission = service.ensure_submitted(creative, "callback", "paid prompt")

        assert submission.resumed is False
        assert rows["callback"].phase == "submitted"
        assert rows["callback"].operation_name == submission.operation_name
        assert len(provider.submit_calls) == 1


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

        # Polling alone is not an acknowledgement: the operation and reserved
        # cost survive until the caller has durably stored the downloaded bytes.
        assert ledger.total_actual(creative.id) == pytest.approx(0.0)
        assert ledger.total_projected(creative.id) == pytest.approx(0.4)
        assert store.get("scene-1") is not None

        service.checkpoint_success(creative, "scene-1", result)
        assert ledger.total_actual(creative.id) == pytest.approx(0.4)
        assert ledger.total_projected(creative.id) == pytest.approx(0.0)
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
        completed = service.poll(creative, "scene-1")
        assert completed.status == OP_SUCCEEDED
        service.checkpoint_success(creative, "scene-1", completed)


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

        retry = service.ensure_submitted(creative, "scene-2", "tricky shot", previously_failed=True)
        assert not retry.resumed
        assert retry.model_id == get_model_config().veo_model_fast
        assert provider.submit_calls[-1]["model_id"] == get_model_config().veo_model_fast

        provider.fail_operations.clear()
        result = service.poll(creative, "scene-2")
        assert result.status == OP_SUCCEEDED
        # Escalated actual cost: 8 s x 0.10 fast at 720p.
        assert result.actual_cost_usd == pytest.approx(0.8)
        service.checkpoint_success(creative, "scene-2", result)
        assert ledger.total_actual(creative.id) == pytest.approx(0.8)


class TestStoredOperationDefaults:
    def test_dataclass_defaults(self) -> None:
        op = StoredOperation(operation_name="op-1", model_id="m")
        assert op.duration_seconds == pytest.approx(8.0)
        assert op.phase == "submitted"
        assert op.submission_ambiguous is False

    def test_legacy_job_row_without_phase_remains_resumable(self, db_session: Session) -> None:
        db_session.add(
            Job(
                kind=JobOperationStore.KIND,
                queue="ai",
                status=JobStatus.RUNNING.value,
                idempotency_key="veo:legacy-scene",
                payload={
                    "scene_id": "legacy-scene",
                    "operation_name": "legacy-provider-op",
                    "model_id": "legacy-model",
                    "duration_seconds": 8.0,
                },
            )
        )
        db_session.commit()

        stored = JobOperationStore(db_session).get("legacy-scene")

        assert stored is not None
        assert stored.phase == "submitted"
        assert stored.submission_ambiguous is False
        assert stored.operation_name == "legacy-provider-op"
