"""Guarded, explicit deletion of an unpublished creative and its local data.

Creative rows do not have ORM or database delete cascades.  This module keeps
the dependency order in one place, blocks in-flight/remote side effects, and
purges owned objects before committing the database deletion.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.errors import Conflict, UpstreamError
from app.models import (
    Asset,
    AuditEvent,
    CostEvent,
    Creative,
    Job,
    PublishAttempt,
    PublishTarget,
    Rendition,
    Scene,
    ScriptVersion,
    Source,
)
from app.states import CreativeState, JobStatus, PublishTargetStatus
from app.storage import AssetStore

_MAX_JOB_ATTEMPTS = 4
_MAX_TARGET_ATTEMPTS = 4
_BLOCKED_CREATIVE_STATES = frozenset(
    {
        CreativeState.SCHEDULED.value,
        CreativeState.PUBLISHING.value,
        CreativeState.PUBLISHED.value,
        CreativeState.PARTIAL.value,
        CreativeState.NEEDS_ACTION.value,
    }
)
_SAFE_TARGET_STATUSES = frozenset(
    {
        PublishTargetStatus.FAILED.value,
        PublishTargetStatus.MANUAL_BUNDLE.value,
    }
)
_ACTIVE_PUBLISH_PHASES = frozenset(
    {"running", "validated", "preparing", "prepared", "uploading", "uploaded", "finalizing"}
)


def _retryable(error: dict[str, Any] | None) -> bool:
    return bool((error or {}).get("retryable"))


def creative_delete_blockers(
    db: Session,
    creative: Creative,
    *,
    lock_dependents: bool = False,
) -> list[dict[str, Any]]:
    """Return every reason a local hard-delete would be unsafe right now."""
    blockers: list[dict[str, Any]] = []
    if creative.state in _BLOCKED_CREATIVE_STATES:
        blockers.append(
            {
                "code": "creative_state_has_external_side_effects",
                "message": (
                    f"Creative đang ở trạng thái {creative.state}; hãy giữ lại để đối soát "
                    "lịch hoặc bài đăng trên nền tảng."
                ),
            }
        )

    jobs_stmt = select(Job).where(Job.creative_id == creative.id)
    if lock_dependents:
        jobs_stmt = jobs_stmt.with_for_update()
    jobs = db.execute(jobs_stmt).scalars().all()
    active_job_ids = [
        job.id
        for job in jobs
        if job.status in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}
        or (
            job.status == JobStatus.FAILED.value
            and _retryable(job.error)
            and job.attempts < _MAX_JOB_ATTEMPTS
        )
    ]
    if active_job_ids:
        blockers.append(
            {
                "code": "creative_jobs_active",
                "message": "Creative vẫn còn tác vụ đang chạy hoặc đang chờ tự thử lại.",
                "job_ids": active_job_ids,
            }
        )

    targets_stmt = select(PublishTarget).where(PublishTarget.creative_id == creative.id)
    if lock_dependents:
        targets_stmt = targets_stmt.with_for_update()
    targets = db.execute(targets_stmt).scalars().all()
    target_ids = [target.id for target in targets]
    attempts: list[PublishAttempt] = []
    if target_ids:
        attempts_stmt = select(PublishAttempt).where(PublishAttempt.target_id.in_(target_ids))
        if lock_dependents:
            attempts_stmt = attempts_stmt.with_for_update()
        attempts = list(db.execute(attempts_stmt).scalars())
    attempt_counts: dict[str, int] = {}
    for attempt in attempts:
        attempt_counts[attempt.target_id] = attempt_counts.get(attempt.target_id, 0) + 1

    unsafe_target_ids = [
        target.id
        for target in targets
        if target.remote_post_id
        or target.remote_status
        or target.claimed_at is not None
        or target.status not in _SAFE_TARGET_STATUSES
        or (
            target.status == PublishTargetStatus.FAILED.value
            and _retryable(target.last_error)
            and attempt_counts.get(target.id, 0) < _MAX_TARGET_ATTEMPTS
        )
    ]
    active_attempt_ids = [
        attempt.id
        for attempt in attempts
        if attempt.finished_at is None and attempt.status in _ACTIVE_PUBLISH_PHASES
    ]
    remote_evidence_attempt_ids = [
        attempt.id
        for attempt in attempts
        if isinstance(attempt.response, dict)
        and (
            bool(attempt.response.get("encrypted_checkpoint"))
            or bool(attempt.response.get("remote_post_id"))
        )
    ]
    if unsafe_target_ids or active_attempt_ids or remote_evidence_attempt_ids:
        blockers.append(
            {
                "code": "creative_publication_active_or_remote",
                "message": (
                    "Creative có lịch/bài remote hoặc phiên đăng chưa kết thúc; "
                    "xóa local không thể hủy nội dung trên nền tảng."
                ),
                "target_ids": unsafe_target_ids,
                "attempt_ids": active_attempt_ids,
                "remote_evidence_attempt_ids": remote_evidence_attempt_ids,
            }
        )
    return blockers


def require_creative_deletable(db: Session, creative: Creative) -> None:
    blockers = creative_delete_blockers(db, creative, lock_dependents=True)
    if blockers:
        raise Conflict(
            "Không thể xóa creative an toàn: " + blockers[0]["message"],
            code="creative_delete_blocked",
            details={"blockers": blockers},
        )


def _row_count(db: Session, model: type, condition: Any) -> int:
    return int(db.scalar(select(func.count()).select_from(model).where(condition)) or 0)


def _owned_storage_keys(db: Session, creative_id: str) -> tuple[list[str], int]:
    raw_keys = list(
        dict.fromkeys(
            db.execute(
                select(Asset.storage_key).where(
                    Asset.creative_id == creative_id,
                    Asset.storage_key != "",
                )
            ).scalars()
        )
    )
    if not raw_keys:
        return [], 0
    shared_keys = set(
        db.execute(
            select(Asset.storage_key).where(
                Asset.creative_id != creative_id,
                Asset.storage_key.in_(raw_keys),
            )
        ).scalars()
    )
    prefix = f"creatives/{creative_id}/"
    owned = [key for key in raw_keys if key.startswith(prefix) and key not in shared_keys]
    return owned, len(raw_keys) - len(owned)


def delete_creative_graph(
    db: Session,
    creative: Creative,
    store: AssetStore,
    *,
    actor_id: str,
) -> dict[str, Any]:
    """Purge local objects and dependent rows for an already-locked creative."""
    require_creative_deletable(db, creative)
    creative_id = creative.id
    title = creative.topic_title
    prior_state = creative.state

    target_ids = list(
        db.execute(
            select(PublishTarget.id).where(PublishTarget.creative_id == creative_id)
        ).scalars()
    )
    storage_keys, skipped_storage_keys = _owned_storage_keys(db, creative_id)
    asset_bytes = int(
        db.scalar(
            select(func.coalesce(func.sum(Asset.size_bytes), 0)).where(
                Asset.creative_id == creative_id
            )
        )
        or 0
    )
    counts = {
        "publish_attempts": (
            _row_count(db, PublishAttempt, PublishAttempt.target_id.in_(target_ids))
            if target_ids
            else 0
        ),
        "publish_targets": len(target_ids),
        "renditions": _row_count(db, Rendition, Rendition.creative_id == creative_id),
        "assets": _row_count(db, Asset, Asset.creative_id == creative_id),
        "scenes": _row_count(db, Scene, Scene.creative_id == creative_id),
        "script_versions": _row_count(
            db, ScriptVersion, ScriptVersion.creative_id == creative_id
        ),
        "sources": _row_count(db, Source, Source.creative_id == creative_id),
        "cost_events": _row_count(db, CostEvent, CostEvent.creative_id == creative_id),
        "jobs": _row_count(db, Job, Job.creative_id == creative_id),
    }
    projected_cost, actual_cost = 0.0, 0.0
    for projected, amount in db.execute(
        select(CostEvent.projected, func.coalesce(func.sum(CostEvent.amount_usd), 0.0))
        .where(CostEvent.creative_id == creative_id)
        .group_by(CostEvent.projected)
    ):
        if projected:
            projected_cost = float(amount)
        else:
            actual_cost = float(amount)

    audit_data: dict[str, Any] = {
        "topic_title": title,
        "prior_state": prior_state,
        "deleted_counts": counts,
        "asset_bytes": asset_bytes,
        "storage_objects_requested": len(storage_keys),
        "storage_objects_skipped": skipped_storage_keys,
        "projected_cost_usd": projected_cost,
        "actual_cost_usd": actual_cost,
    }
    audit = AuditEvent(
        actor_id=actor_id,
        actor_kind="user",
        action="creative_deleted",
        entity_type="creative",
        entity_id=creative_id,
        data=audit_data,
    )

    # Execute and flush every database mutation before touching object storage.
    # This catches ordinary FK/constraint failures while all files still exist.
    try:
        if target_ids:
            db.execute(delete(PublishAttempt).where(PublishAttempt.target_id.in_(target_ids)))
        db.execute(delete(PublishTarget).where(PublishTarget.creative_id == creative_id))
        db.execute(delete(Rendition).where(Rendition.creative_id == creative_id))
        db.execute(delete(Asset).where(Asset.creative_id == creative_id))
        db.execute(delete(Scene).where(Scene.creative_id == creative_id))
        db.execute(delete(ScriptVersion).where(ScriptVersion.creative_id == creative_id))
        db.execute(delete(Source).where(Source.creative_id == creative_id))
        db.execute(delete(CostEvent).where(CostEvent.creative_id == creative_id))
        db.execute(delete(Job).where(Job.creative_id == creative_id))
        db.execute(delete(Creative).where(Creative.id == creative_id))
        db.add(audit)
        db.flush()
    except Exception as exc:
        db.rollback()
        raise UpstreamError(
            "Không thể chuẩn bị xóa dữ liệu creative; chưa có tệp nào bị xóa.",
            code="creative_delete_database_failed",
            retryable=True,
            details={"storage_objects_deleted": 0},
        ) from exc

    deleted_objects = 0
    missing_objects = 0
    processed_objects = 0
    try:
        for storage_key in storage_keys:
            if store.delete(storage_key):
                deleted_objects += 1
            else:
                missing_objects += 1
            processed_objects += 1
    except Exception as exc:
        db.rollback()
        raise UpstreamError(
            "Xóa tệp của creative thất bại; có thể thử xóa lại an toàn.",
            code="creative_delete_storage_failed",
            retryable=True,
            details={
                "deleted_objects": deleted_objects,
                "missing_objects": missing_objects,
                "remaining_objects": len(storage_keys) - processed_objects,
            },
        ) from exc

    audit.data = {
        **audit_data,
        "storage_objects_deleted": deleted_objects,
        "storage_objects_missing": missing_objects,
    }
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise UpstreamError(
            "Xóa dữ liệu creative thất bại sau khi dọn tệp; hãy thử lại.",
            code="creative_delete_database_failed",
            retryable=True,
            details={"storage_objects_deleted": deleted_objects},
        ) from exc

    return {
        "creative_id": creative_id,
        "deleted": True,
        "deleted_counts": counts,
        "storage_objects_deleted": deleted_objects,
        "storage_objects_missing": missing_objects,
        "storage_objects_skipped": skipped_storage_keys,
    }
