"""VeoService: long-running-operation submit/poll with resume safety.

CRITICAL: the operation name is persisted per scene through an injectable
persistence hook (an `OperationStore` — repo class or callables via
`CallbackOperationStore`) as soon as the provider hands it back and BEFORE
`ensure_submitted` returns. A worker restart therefore finds the stored
operation and re-polls it instead of resubmitting (zero new submit calls).

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
from app.errors import NotFound
from app.models import Creative, Job, utcnow
from app.states import JobStatus


@dataclass(frozen=True)
class StoredOperation:
    operation_name: str
    model_id: str
    duration_seconds: float = 8.0


@runtime_checkable
class OperationStore(Protocol):
    """Persistence hook for in-flight Veo operations, keyed by scene id."""

    def get(self, scene_id: str) -> StoredOperation | None: ...

    def put(self, scene_id: str, operation: StoredOperation) -> None: ...

    def clear(self, scene_id: str) -> None: ...


class InMemoryOperationStore:
    """Dict-backed store for tests/dev mode."""

    def __init__(self) -> None:
        self._ops: dict[str, StoredOperation] = {}

    def get(self, scene_id: str) -> StoredOperation | None:
        return self._ops.get(scene_id)

    def put(self, scene_id: str, operation: StoredOperation) -> None:
        self._ops[scene_id] = operation

    def clear(self, scene_id: str) -> None:
        self._ops.pop(scene_id, None)


class CallbackOperationStore:
    """Adapter turning three callables into an OperationStore."""

    def __init__(
        self,
        get: Callable[[str], StoredOperation | None],
        put: Callable[[str, StoredOperation], None],
        clear: Callable[[str], None],
    ) -> None:
        self._get = get
        self._put = put
        self._clear = clear

    def get(self, scene_id: str) -> StoredOperation | None:
        return self._get(scene_id)

    def put(self, scene_id: str, operation: StoredOperation) -> None:
        self._put(scene_id, operation)

    def clear(self, scene_id: str) -> None:
        self._clear(scene_id)


class JobOperationStore:
    """DB-backed store using the jobs table so operations survive worker restarts.

    Rows use kind='veo_operation' and idempotency_key='veo:<scene_id>'. `put`
    commits by default so the operation name is durable before submit returns
    to the caller.
    """

    KIND = "veo_operation"

    def __init__(self, db: Session, commit: bool = True) -> None:
        self._db = db
        self._commit = commit

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
        return StoredOperation(
            operation_name=str(payload.get("operation_name", "")),
            model_id=str(payload.get("model_id", "")),
            duration_seconds=float(payload.get("duration_seconds", 8.0)),
        )

    def put(self, scene_id: str, operation: StoredOperation) -> None:
        job = Job(
            kind=self.KIND,
            queue="ai",
            status=JobStatus.RUNNING.value,
            idempotency_key=self._key(scene_id),
            payload={
                "scene_id": scene_id,
                "operation_name": operation.operation_name,
                "model_id": operation.model_id,
                "duration_seconds": operation.duration_seconds,
            },
            started_at=utcnow(),
        )
        self._db.add(job)
        self._flush()

    def clear(self, scene_id: str) -> None:
        for row in self._rows(scene_id):
            row.status = JobStatus.SUCCEEDED.value
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

    @staticmethod
    def _cost_note(scene_id: str) -> str:
        return f"veo scene {scene_id}"

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
            return VeoSubmission(
                scene_id=scene_id,
                operation_name=existing.operation_name,
                model_id=existing.model_id,
                resumed=True,
                projected_cost_usd=0.0,
            )

        duration = duration_seconds if duration_seconds is not None else self._cfg.scene_seconds
        model_id = self.resolve_model(is_hero=is_hero, previously_failed=previously_failed)
        price = veo_price_per_second(model_id, self._cfg)
        projected = duration * price
        self._ledger.check_cap(creative, projected)

        operation_name = self._provider.submit(
            prompt=prompt,
            model_id=model_id,
            duration_seconds=duration,
            keyframe_bytes=keyframe_bytes,
        )
        # Persist BEFORE returning: a restart re-polls this instead of resubmitting.
        self._store.put(
            scene_id,
            StoredOperation(
                operation_name=operation_name, model_id=model_id, duration_seconds=duration
            ),
        )
        self._ledger.record_projected(
            creative.id,
            kind="veo",
            model_id=model_id,
            units=duration,
            unit_price_usd=price,
            note=self._cost_note(scene_id),
        )
        return VeoSubmission(
            scene_id=scene_id,
            operation_name=operation_name,
            model_id=model_id,
            resumed=False,
            projected_cost_usd=projected,
        )

    def poll(self, creative: Creative, scene_id: str) -> VeoPollResult:
        """Poll the stored operation; on success record actual cost per generated second."""
        stored = self._store.get(scene_id)
        if stored is None:
            raise NotFound(f"no stored veo operation for scene {scene_id}")

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
            self._store.clear(scene_id)
            return VeoPollResult(
                scene_id=scene_id,
                operation_name=stored.operation_name,
                status=OP_FAILED,
                model_id=stored.model_id,
                error=operation.error or "veo generation failed",
            )

        seconds = operation.duration_seconds or stored.duration_seconds
        price = veo_price_per_second(stored.model_id, self._cfg)
        note = self._cost_note(scene_id)
        self._ledger.remove_projected(creative.id, kind="veo", note=note)
        self._ledger.record_actual(
            creative.id,
            kind="veo",
            model_id=stored.model_id,
            units=seconds,
            unit_price_usd=price,
            note=note,
        )
        self._store.clear(scene_id)
        return VeoPollResult(
            scene_id=scene_id,
            operation_name=stored.operation_name,
            status=OP_SUCCEEDED,
            model_id=stored.model_id,
            video_bytes=operation.video_bytes,
            duration_seconds=seconds,
            actual_cost_usd=round(seconds * price, 6),
        )
