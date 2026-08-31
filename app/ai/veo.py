"""VeoService: long-running-operation submit/poll with resume safety.

CRITICAL: a durable submission intent is persisted per scene BEFORE the paid
provider call. The returned operation name completes that same intent row. If
the process dies after the remote side effect but before the operation name is
known, a restart sees an ambiguous intent and fails closed instead of risking a
second charge. Completed submissions are re-polled without a new submit.

Failed or hero scenes escalate lite -> fast. Actual cost is recorded per
generated second; the matching projected row is removed when actuals land.
Result bytes must be downloaded promptly — Google keeps files only ~2 days —
so providers return the bytes inside the succeeded poll result.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sqlalchemy.orm import Session

from app.ai.base import OP_FAILED, OP_RUNNING, OP_SUCCEEDED, VideoOperation, VideoProvider
from app.config import ModelConfig, get_model_config
from app.costs import CostLedger, veo_price_per_second
from app.errors import NotFound, PolicyBlocked, ProviderQuotaExhausted
from app.models import Creative, Job, utcnow
from app.states import JobStatus


@dataclass(frozen=True)
class StoredOperation:
    operation_name: str
    model_id: str
    duration_seconds: float = 8.0
    phase: str = "submitted"
    provider_kind: str = ""
    creative_id: str = ""
    cost_event_id: str = ""

    @property
    def submission_ambiguous(self) -> bool:
        return self.phase != "submitted" or not self.operation_name

    @classmethod
    def intent(
        cls,
        *,
        model_id: str,
        duration_seconds: float,
        provider_kind: str = "",
        creative_id: str = "",
        cost_event_id: str = "",
    ) -> "StoredOperation":
        return cls(
            operation_name="",
            model_id=model_id,
            duration_seconds=duration_seconds,
            phase="intent",
            provider_kind=provider_kind,
            creative_id=creative_id,
            cost_event_id=cost_event_id,
        )


@runtime_checkable
class OperationStore(Protocol):
    """Persistence hook for in-flight Veo operations, keyed by scene id."""

    @property
    def supports_submission_intents(self) -> bool: ...

    def get(self, scene_id: str) -> StoredOperation | None: ...

    def begin(self, scene_id: str, intent: StoredOperation) -> None: ...

    def put(self, scene_id: str, operation: StoredOperation) -> None: ...

    def clear(self, scene_id: str) -> None: ...


class InMemoryOperationStore:
    """Dict-backed store for tests/dev mode."""

    def __init__(self) -> None:
        self._ops: dict[str, StoredOperation] = {}

    @property
    def supports_submission_intents(self) -> bool:
        return True

    def get(self, scene_id: str) -> StoredOperation | None:
        return self._ops.get(scene_id)

    def begin(self, scene_id: str, intent: StoredOperation) -> None:
        if scene_id in self._ops:
            raise RuntimeError("veo submission intent already exists")
        if not intent.submission_ambiguous:
            raise ValueError("begin requires an ambiguous submission intent")
        self._ops[scene_id] = intent

    def put(self, scene_id: str, operation: StoredOperation) -> None:
        intent = self._ops.get(scene_id)
        if intent is None or not intent.submission_ambiguous:
            raise RuntimeError("veo operation cannot complete without an active intent")
        if (
            intent.model_id != operation.model_id
            or intent.duration_seconds != operation.duration_seconds
            or (
                intent.provider_kind
                and intent.provider_kind != operation.provider_kind
            )
            or (intent.creative_id and intent.creative_id != operation.creative_id)
            or (intent.cost_event_id and intent.cost_event_id != operation.cost_event_id)
            or operation.submission_ambiguous
        ):
            raise ValueError("veo operation does not match its submission intent")
        self._ops[scene_id] = operation

    def clear(self, scene_id: str) -> None:
        self._ops.pop(scene_id, None)


class CallbackOperationStore:
    """Callback adapter; legacy three-callback construction remains valid.

    Paid submission is allowed only when ``begin`` is supplied and durably
    stores the intent before returning. Legacy adapters remain usable for
    reading/polling existing operations, but fail closed before a new submit.
    """

    def __init__(
        self,
        get: Callable[[str], StoredOperation | None],
        put: Callable[[str, StoredOperation], None],
        clear: Callable[[str], None],
        begin: Callable[[str, StoredOperation], None] | None = None,
    ) -> None:
        self._get = get
        self._put = put
        self._clear = clear
        self._begin = begin

    @property
    def supports_submission_intents(self) -> bool:
        return self._begin is not None

    def get(self, scene_id: str) -> StoredOperation | None:
        return self._get(scene_id)

    def begin(self, scene_id: str, intent: StoredOperation) -> None:
        if self._begin is None:
            raise RuntimeError("callback operation store has no durable begin callback")
        self._begin(scene_id, intent)

    def put(self, scene_id: str, operation: StoredOperation) -> None:
        self._put(scene_id, operation)

    def clear(self, scene_id: str) -> None:
        self._clear(scene_id)


class JobOperationStore:
    """DB-backed store using the jobs table so operations survive worker restarts.

    Rows use kind='veo_operation' and idempotency_key='veo:<scene_id>'. `begin`
    commits an intent before the remote call; `put` fills the operation name in
    that same row. ``commit=False`` is retained for read/test compatibility but
    is rejected by :class:`VeoService` for new paid submissions.
    """

    KIND = "veo_operation"

    def __init__(self, db: Session, commit: bool = True) -> None:
        self._db = db
        self._commit = commit

    @property
    def supports_submission_intents(self) -> bool:
        return self._commit

    @staticmethod
    def _key(scene_id: str) -> str:
        return f"veo:{scene_id}"

    def _rows(self, scene_id: str) -> list[Job]:
        return (
            self._db.query(Job)
            .filter(
                Job.kind == self.KIND,
                Job.idempotency_key == self._key(scene_id),
                Job.status == JobStatus.RUNNING.value,
            )
            .order_by(Job.created_at.desc())
            .all()
        )

    def get(self, scene_id: str) -> StoredOperation | None:
        rows = self._rows(scene_id)
        if not rows:
            return None
        payload = rows[0].payload or {}
        operation_name = str(payload.get("operation_name", ""))
        phase = str(payload.get("phase", ""))
        # Backward compatibility: rows written before submission intents had no
        # phase but did have a real operation name.
        if not phase:
            phase = "submitted" if operation_name else "intent"
        elif phase != "submitted" or not operation_name:
            phase = "intent"
        return StoredOperation(
            operation_name=operation_name,
            model_id=str(payload.get("model_id", "")),
            duration_seconds=float(payload.get("duration_seconds", 8.0)),
            phase=phase,
            provider_kind=str(payload.get("provider", "")),
            creative_id=str(payload.get("creative_id", "")),
            cost_event_id=str(payload.get("cost_event_id", "")),
        )

    def begin(self, scene_id: str, intent: StoredOperation) -> None:
        if not self._commit:
            raise RuntimeError("durable Veo intents require JobOperationStore(commit=True)")
        if self._rows(scene_id):
            raise RuntimeError("veo submission intent already exists")
        if not intent.submission_ambiguous:
            raise ValueError("begin requires an ambiguous submission intent")
        job = Job(
            kind=self.KIND,
            queue="ai",
            status=JobStatus.RUNNING.value,
            creative_id=intent.creative_id or None,
            idempotency_key=self._key(scene_id),
            payload={
                "scene_id": scene_id,
                "phase": "intent",
                "operation_name": "",
                "model_id": intent.model_id,
                "duration_seconds": intent.duration_seconds,
                "provider": intent.provider_kind,
                "creative_id": intent.creative_id,
                "cost_event_id": intent.cost_event_id,
            },
            started_at=utcnow(),
        )
        self._db.add(job)
        self._flush()

    def put(self, scene_id: str, operation: StoredOperation) -> None:
        rows = self._rows(scene_id)
        if not rows:
            raise RuntimeError("veo operation cannot complete without an active intent")
        row = rows[0]
        payload = row.payload or {}
        intent_provider = str(payload.get("provider", ""))
        intent_creative_id = str(payload.get("creative_id", ""))
        intent_cost_event_id = str(payload.get("cost_event_id", ""))
        if (
            str(payload.get("phase", "")) != "intent"
            or str(payload.get("model_id", "")) != operation.model_id
            or float(payload.get("duration_seconds", 8.0)) != operation.duration_seconds
            or (intent_provider and intent_provider != operation.provider_kind)
            or (intent_creative_id and intent_creative_id != operation.creative_id)
            or (intent_cost_event_id and intent_cost_event_id != operation.cost_event_id)
            or operation.submission_ambiguous
        ):
            raise ValueError("veo operation does not match its submission intent")
        row.payload = {
            **payload,
            "phase": "submitted",
            "operation_name": operation.operation_name,
            "provider": operation.provider_kind or intent_provider,
            "creative_id": operation.creative_id or intent_creative_id,
            "cost_event_id": operation.cost_event_id or intent_cost_event_id,
        }
        self._flush()

    def clear(self, scene_id: str) -> None:
        for row in self._rows(scene_id):
            row.status = JobStatus.SUCCEEDED.value
            row.finished_at = utcnow()
        self._flush()

    def cancel_ambiguous_intent(self, scene_id: str, *, reason: str) -> None:
        """Retire an intent after an operator proves no remote task exists.

        This is intentionally separate from :meth:`clear`: a reconciled
        no-submission outcome is not a successful provider operation.  The
        caller is responsible for removing the matching projected cost in the
        same database transaction.
        """
        rows = self._rows(scene_id)
        if not rows:
            raise RuntimeError("no active veo submission intent exists")
        row = rows[0]
        payload = row.payload or {}
        if str(payload.get("phase", "")) != "intent" or payload.get("operation_name"):
            raise ValueError("only an ambiguous veo submission intent can be cancelled")
        row.status = JobStatus.CANCELLED.value
        row.result = {"reason": reason, "reconciled": True}
        row.finished_at = utcnow()
        self._flush()

    def _flush(self) -> None:
        if self._commit:
            self._db.commit()
        else:
            self._db.flush()


@dataclass(frozen=True)
class VeoSubmission:
    scene_id: str
    operation_name: str
    model_id: str
    resumed: bool
    projected_cost_usd: float


@dataclass(frozen=True)
class VeoPollResult:
    scene_id: str
    operation_name: str
    status: str  # OP_RUNNING | OP_SUCCEEDED | OP_FAILED
    model_id: str = ""
    video_bytes: bytes | None = None
    duration_seconds: float = 0.0
    actual_cost_usd: float = 0.0
    error: str | None = None


class VeoService:
    #: Cost-ledger kind and per-second price are hooks so paid adapters with a
    #: different provider behind them (e.g. wan2.7-i2v) can reuse every safety
    #: property below by subclassing.
    cost_kind = "veo"

    def __init__(
        self,
        provider: VideoProvider,
        ledger: CostLedger,
        store: OperationStore,
        model_config: ModelConfig | None = None,
    ) -> None:
        self._provider = provider
        self._ledger = ledger
        self._store = store
        self._cfg = model_config or get_model_config()

    def resolve_model(self, *, is_hero: bool = False, previously_failed: bool = False) -> str:
        """Failed or hero scenes escalate lite -> fast."""
        if is_hero or previously_failed:
            return self._cfg.veo_model_fast
        return self._cfg.veo_model_lite

    def price_per_second(self, model_id: str) -> float:
        """USD per generated second for the configured model tier."""
        return veo_price_per_second(model_id, self._cfg)

    @staticmethod
    def _cost_note(scene_id: str) -> str:
        return f"veo scene {scene_id}"

    @staticmethod
    def _raise_ambiguous(
        scene_id: str,
        stored: StoredOperation,
        *,
        known_operation_name: str | None = None,
    ) -> None:
        details = {
            "scene_id": scene_id,
            "model_id": stored.model_id,
            "operator_action_required": True,
        }
        if known_operation_name:
            details["operation_name"] = known_operation_name
        raise PolicyBlocked(
            "Veo submission outcome is unknown; automatic resubmission is blocked "
            "pending operator reconciliation",
            code="veo_submission_ambiguous",
            details=details,
        )

    def _validate_stored_provider(
        self,
        scene_id: str,
        stored: StoredOperation,
    ) -> None:
        """Fail closed if configuration drift would poll through another adapter."""
        if stored.provider_kind and stored.provider_kind != self.cost_kind:
            raise PolicyBlocked(
                "stored video operation belongs to a different provider",
                code="video_submission_provider_mismatch",
                details={
                    "scene_id": scene_id,
                    "stored_provider": stored.provider_kind,
                    "configured_provider": self.cost_kind,
                    "operator_action_required": True,
                },
            )

    def _remove_projected_cost(
        self,
        creative: Creative,
        scene_id: str,
        stored: StoredOperation,
    ) -> None:
        if stored.cost_event_id:
            removed = self._ledger.remove_projected_event(
                stored.cost_event_id,
                creative_id=creative.id,
                kind=self.cost_kind,
            )
            if removed != 1:
                raise PolicyBlocked(
                    "linked video cost projection is missing or inconsistent",
                    code="video_submission_projection_mismatch",
                    details={
                        "scene_id": scene_id,
                        "operator_action_required": True,
                    },
                )
            return
        # Backward compatibility for operations created before projections
        # carried a durable event identifier.
        self._ledger.remove_projected(
            creative.id,
            kind=self.cost_kind,
            note=self._cost_note(scene_id),
        )

    def ensure_submitted(
        self,
        creative: Creative,
        scene_id: str,
        prompt: str,
        *,
        keyframe_bytes: bytes | None = None,
        is_hero: bool = False,
        previously_failed: bool = False,
        duration_seconds: float | None = None,
    ) -> VeoSubmission:
        """Submit once per scene: a stored operation short-circuits to a resume."""
        existing = self._store.get(scene_id)
        if existing is not None:
            self._validate_stored_provider(scene_id, existing)
            if existing.submission_ambiguous:
                self._raise_ambiguous(scene_id, existing)
            return VeoSubmission(
                scene_id=scene_id,
                operation_name=existing.operation_name,
                model_id=existing.model_id,
                resumed=True,
                projected_cost_usd=0.0,
            )

        duration = duration_seconds if duration_seconds is not None else self._cfg.scene_seconds
        model_id = self.resolve_model(is_hero=is_hero, previously_failed=previously_failed)
        price = self.price_per_second(model_id)
        projected = duration * price
        self._ledger.check_cap(creative, projected)

        if not self._store.supports_submission_intents:
            raise PolicyBlocked(
                "operation store cannot durably reserve a Veo submission",
                code="veo_operation_store_unsafe",
                details={"scene_id": scene_id, "operator_action_required": True},
            )

        # Record cost before the intent commit. JobOperationStore uses the same
        # Session and commits in begin(), making both the conservative spend
        # projection and the intent durable before the paid provider call.
        projected_event = self._ledger.record_projected(
            creative.id,
            kind=self.cost_kind,
            model_id=model_id,
            units=duration,
            unit_price_usd=price,
            note=self._cost_note(scene_id),
        )
        intent = StoredOperation.intent(
            model_id=model_id,
            duration_seconds=duration,
            provider_kind=self.cost_kind,
            creative_id=creative.id,
            cost_event_id=projected_event.id,
        )
        self._store.begin(scene_id, intent)
        try:
            operation_name = self._provider.submit(
                prompt=prompt,
                model_id=model_id,
                duration_seconds=duration,
                keyframe_bytes=keyframe_bytes,
            )
        except ProviderQuotaExhausted:
            # A definitive pre-charge rejection: the provider created no task
            # and billed nothing, so the intent row and its projection can be
            # released instead of blocking on operator reconciliation.
            self._remove_projected_cost(creative, scene_id, intent)
            self._store.clear(scene_id)
            raise
        except Exception:
            # Once the durable intent exists, an ordinary SDK/transport error
            # cannot prove that the paid remote side effect did not happen.
            self._raise_ambiguous(scene_id, intent)
        # Persist BEFORE returning: a restart re-polls this instead of resubmitting.
        try:
            self._store.put(
                scene_id,
                StoredOperation(
                    operation_name=operation_name,
                    model_id=model_id,
                    duration_seconds=duration,
                    provider_kind=self.cost_kind,
                    creative_id=creative.id,
                    cost_event_id=projected_event.id,
                ),
            )
        except Exception:
            self._raise_ambiguous(
                scene_id,
                intent,
                known_operation_name=operation_name,
            )
        return VeoSubmission(
            scene_id=scene_id,
            operation_name=operation_name,
            model_id=model_id,
            resumed=False,
            projected_cost_usd=projected,
        )

    def poll(self, creative: Creative, scene_id: str) -> VeoPollResult:
        """Poll without acknowledging success until the caller persists the video."""
        stored = self._store.get(scene_id)
        if stored is None:
            raise NotFound(f"no stored veo operation for scene {scene_id}")
        self._validate_stored_provider(scene_id, stored)
        if stored.submission_ambiguous:
            self._raise_ambiguous(scene_id, stored)

        operation: VideoOperation = self._provider.poll(stored.operation_name)
        if operation.status == OP_RUNNING:
            return VeoPollResult(
                scene_id=scene_id,
                operation_name=stored.operation_name,
                status=OP_RUNNING,
                model_id=stored.model_id,
            )
        if operation.status == OP_FAILED:
            # Clear so the next ensure_submitted escalates and resubmits.
            self._remove_projected_cost(creative, scene_id, stored)
            self._store.clear(scene_id)
            return VeoPollResult(
                scene_id=scene_id,
                operation_name=stored.operation_name,
                status=OP_FAILED,
                model_id=stored.model_id,
                error=operation.error or "veo generation failed",
            )

        seconds = operation.duration_seconds or stored.duration_seconds
        price = self.price_per_second(stored.model_id)
        return VeoPollResult(
            scene_id=scene_id,
            operation_name=stored.operation_name,
            status=OP_SUCCEEDED,
            model_id=stored.model_id,
            video_bytes=operation.video_bytes,
            duration_seconds=seconds,
            actual_cost_usd=round(seconds * price, 6),
        )

    def checkpoint_success(
        self,
        creative: Creative,
        scene_id: str,
        result: VeoPollResult,
    ) -> None:
        """Acknowledge a completed operation after its Asset row is ready.

        With JobOperationStore this commits the Asset/scene update, actual cost
        and operation completion in one database transaction.  Until this is
        called, a restart re-polls the same provider operation.
        """
        if result.status != OP_SUCCEEDED:
            raise ValueError("only a succeeded Veo result can be checkpointed")
        stored = self._store.get(scene_id)
        if stored is None:
            raise NotFound(f"no stored veo operation for scene {scene_id}")
        self._validate_stored_provider(scene_id, stored)
        if (
            stored.submission_ambiguous
            or stored.operation_name != result.operation_name
            or stored.model_id != result.model_id
        ):
            raise PolicyBlocked(
                "Veo completion does not match the active operation",
                code="veo_completion_mismatch",
                details={"scene_id": scene_id, "operator_action_required": True},
            )
        note = self._cost_note(scene_id)
        self._remove_projected_cost(creative, scene_id, stored)
        self._ledger.record_actual(
            creative.id,
            kind=self.cost_kind,
            model_id=stored.model_id,
            units=result.duration_seconds,
            unit_price_usd=self.price_per_second(stored.model_id),
            note=note,
        )
        self._store.clear(scene_id)
