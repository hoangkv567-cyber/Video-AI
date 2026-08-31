"""Publish scheduler loop (``python -m app.workers.scheduler``).

Every 60 seconds it claims due PublishTargets and enqueues one publish task per
claim. On PostgreSQL the claim uses ``SELECT ... FOR UPDATE SKIP LOCKED``; on
SQLite it falls back to an optimistic ``UPDATE ... WHERE claimed_at IS NULL``
whose rowcount decides the winner, so two racing sweeps can never both claim
the same target. Claims go stale after ``STALE_CLAIM_SECONDS`` and become
re-sweepable, which is what makes the loop restart-safe. All times are UTC.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Job, PublishAttempt, PublishTarget
from app.states import JobStatus, PublishTargetStatus
from app.workers.dispatch import DISPATCHABLE_JOB_KINDS, SendFn, enqueue_job

logger = logging.getLogger("videoai.scheduler")

SWEEP_INTERVAL_SECONDS = 60
STALE_CLAIM_SECONDS = 600
DEFAULT_CLAIM_LIMIT = 50
JOB_DISPATCH_STALE_SECONDS = 5 * 60
JOB_RETRY_DELAY_SECONDS = 60
JOB_RUNNING_STALE_SECONDS = 6 * 60 * 60
MAX_JOB_ATTEMPTS = 4
TARGET_RETRY_DELAY_SECONDS = 60
MAX_TARGET_ATTEMPTS = 4

DUE_STATUSES = (
    PublishTargetStatus.PENDING.value,
    PublishTargetStatus.VALIDATED.value,
)

TargetEnqueueFn = Callable[[str], None]


def _now() -> datetime:
    return datetime.now(UTC)


def _due_conditions(
    now: datetime,
    stale_cutoff: datetime,
    *,
    retry_delay_seconds: int = TARGET_RETRY_DELAY_SECONDS,
    max_attempts: int = MAX_TARGET_ATTEMPTS,
) -> list:
    retry_cutoff = now - timedelta(seconds=retry_delay_seconds)
    attempt_count = (
        select(func.count(PublishAttempt.id))
        .where(PublishAttempt.target_id == PublishTarget.id)
        .correlate(PublishTarget)
        .scalar_subquery()
    )
    last_attempt_finished = (
        select(func.max(PublishAttempt.finished_at))
        .where(PublishAttempt.target_id == PublishTarget.id)
        .correlate(PublishTarget)
        .scalar_subquery()
    )
    scheduled_due = and_(
        PublishTarget.status.in_(DUE_STATUSES),
        PublishTarget.scheduled_at.is_not(None),
        PublishTarget.scheduled_at <= now,
        or_(PublishTarget.claimed_at.is_(None), PublishTarget.claimed_at < stale_cutoff),
    )
    retryable_failed = and_(
        PublishTarget.status == PublishTargetStatus.FAILED.value,
        PublishTarget.last_error["retryable"].as_boolean().is_(True),
        attempt_count < max_attempts,
        or_(last_attempt_finished.is_(None), last_attempt_finished <= retry_cutoff),
        or_(PublishTarget.claimed_at.is_(None), PublishTarget.claimed_at <= retry_cutoff),
    )
    stale_uploading = and_(
        PublishTarget.status == PublishTargetStatus.UPLOADING.value,
        or_(
            PublishTarget.claimed_at < stale_cutoff,
            and_(
                PublishTarget.claimed_at.is_(None),
                PublishTarget.updated_at < stale_cutoff,
            ),
        ),
    )
    return [or_(scheduled_due, retryable_failed, stale_uploading)]


def _job_dispatch_conditions(
    now: datetime,
    *,
    dispatch_stale_after_seconds: int,
    retry_delay_seconds: int,
    running_stale_after_seconds: int,
    max_attempts: int,
) -> list:
    dispatch_cutoff = now - timedelta(seconds=dispatch_stale_after_seconds)
    retry_cutoff = now - timedelta(seconds=retry_delay_seconds)
    running_cutoff = now - timedelta(seconds=running_stale_after_seconds)
    queued = and_(
        Job.status == JobStatus.QUEUED.value,
        or_(Job.dispatched_at.is_(None), Job.dispatched_at <= dispatch_cutoff),
    )
    retryable_failed = and_(
        Job.status == JobStatus.FAILED.value,
        Job.error["retryable"].as_boolean().is_(True),
        or_(Job.finished_at.is_(None), Job.finished_at <= retry_cutoff),
    )
    stale_running = and_(
        Job.status == JobStatus.RUNNING.value,
        or_(Job.started_at.is_(None), Job.started_at <= running_cutoff),
    )
    return [
        Job.kind.in_(sorted(DISPATCHABLE_JOB_KINDS)),
        Job.attempts < max_attempts,
        or_(queued, retryable_failed, stale_running),
    ]


def _job_candidate_select(conditions: list, limit: int, *, skip_locked: bool):
    stmt = select(Job).where(*conditions).order_by(Job.created_at, Job.id).limit(limit)
    return stmt.with_for_update(skip_locked=True) if skip_locked else stmt


def claim_dispatchable_jobs(
    db: Session,
    *,
    now: datetime | None = None,
    dispatch_stale_after_seconds: int = JOB_DISPATCH_STALE_SECONDS,
    retry_delay_seconds: int = JOB_RETRY_DELAY_SECONDS,
    running_stale_after_seconds: int = JOB_RUNNING_STALE_SECONDS,
    max_attempts: int = MAX_JOB_ATTEMPTS,
    limit: int = DEFAULT_CLAIM_LIMIT,
) -> list[str]:
    """Atomically lease recoverable Job rows for another broker dispatch."""
    now = now or _now()
    conditions = _job_dispatch_conditions(
        now,
        dispatch_stale_after_seconds=dispatch_stale_after_seconds,
        retry_delay_seconds=retry_delay_seconds,
        running_stale_after_seconds=running_stale_after_seconds,
        max_attempts=max_attempts,
    )
    dialect = db.get_bind().dialect.name

    if dialect == "postgresql":
        rows = db.execute(
            _job_candidate_select(conditions, limit, skip_locked=True)
        ).scalars().all()
        for row in rows:
            row.status = JobStatus.QUEUED.value
            row.dispatched_at = now
            row.finished_at = None
        db.commit()
        return [row.id for row in rows]

    # SQLite (and other non-PostgreSQL test/dev databases) cannot SKIP LOCKED.
    # Every candidate is claimed with a guarded UPDATE that rechecks the full
    # eligibility predicate, so only one racing sweep can change the row.
    candidate_ids = list(
        db.execute(
            _job_candidate_select(conditions, limit, skip_locked=False).with_only_columns(Job.id)
        ).scalars()
    )
    claimed: list[str] = []
    for job_id in candidate_ids:
        result = cast(
            CursorResult[Any],
            db.execute(
                update(Job)
                .where(Job.id == job_id, *_job_dispatch_conditions(
                    now,
                    dispatch_stale_after_seconds=dispatch_stale_after_seconds,
                    retry_delay_seconds=retry_delay_seconds,
                    running_stale_after_seconds=running_stale_after_seconds,
                    max_attempts=max_attempts,
                ))
                .values(
                    status=JobStatus.QUEUED.value,
                    dispatched_at=now,
                    finished_at=None,
                )
            ),
        )
        if result.rowcount == 1:
            claimed.append(job_id)
    db.commit()
    return claimed


def sweep_job_dispatches_once(
    db: Session,
    *,
    send: SendFn | None = None,
    now: datetime | None = None,
    dispatch_stale_after_seconds: int = JOB_DISPATCH_STALE_SECONDS,
    retry_delay_seconds: int = JOB_RETRY_DELAY_SECONDS,
    running_stale_after_seconds: int = JOB_RUNNING_STALE_SECONDS,
    max_attempts: int = MAX_JOB_ATTEMPTS,
    limit: int = DEFAULT_CLAIM_LIMIT,
) -> list[str]:
    """Claim recoverable Jobs and dispatch each without losing the DB outbox row."""
    claimed = claim_dispatchable_jobs(
        db,
        now=now,
        dispatch_stale_after_seconds=dispatch_stale_after_seconds,
        retry_delay_seconds=retry_delay_seconds,
        running_stale_after_seconds=running_stale_after_seconds,
        max_attempts=max_attempts,
        limit=limit,
    )
    for job_id in claimed:
        job = db.get(Job, job_id)
        if job is None:  # pragma: no cover - claimed row cannot disappear with FKs intact
            continue
        try:
            enqueue_job(job, send=send)
        except Exception:
            # Keep the lease timestamp. It becomes eligible again after the
            # dispatch-stale interval instead of hot-looping on a dead broker.
            logger.exception("failed to redispatch job %s", job_id)
    if claimed:
        logger.info("claimed %d recoverable job(s): %s", len(claimed), ", ".join(claimed))
    return claimed


def claim_due_targets(
    db: Session,
    *,
    now: datetime | None = None,
    stale_after_seconds: int = STALE_CLAIM_SECONDS,
    retry_delay_seconds: int = TARGET_RETRY_DELAY_SECONDS,
    max_attempts: int = MAX_TARGET_ATTEMPTS,
    limit: int = DEFAULT_CLAIM_LIMIT,
) -> list[str]:
    """Atomically claim due targets; each id is returned by exactly one sweep."""
    now = now or _now()
    stale_cutoff = now - timedelta(seconds=stale_after_seconds)
    conditions = _due_conditions(
        now,
        stale_cutoff,
        retry_delay_seconds=retry_delay_seconds,
        max_attempts=max_attempts,
    )
    dialect = db.get_bind().dialect.name

    if dialect == "postgresql":
        stmt = (
            select(PublishTarget)
            .where(*conditions)
            .order_by(PublishTarget.scheduled_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        rows = db.execute(stmt).scalars().all()
        for row in rows:
            row.claimed_at = now
        db.commit()
        return [row.id for row in rows]

    # SQLite (and others): optimistic per-row claim — the UPDATE's WHERE clause
    # re-checks every due condition, so a lost race means rowcount == 0.
    candidate_ids = list(
        db.execute(
            select(PublishTarget.id)
            .where(*conditions)
            .order_by(PublishTarget.scheduled_at)
            .limit(limit)
        ).scalars()
    )
    claimed: list[str] = []
    for target_id in candidate_ids:
        result = cast(
            CursorResult[Any],
            db.execute(
                update(PublishTarget)
                .where(
                    PublishTarget.id == target_id,
                    *_due_conditions(
                        now,
                        stale_cutoff,
                        retry_delay_seconds=retry_delay_seconds,
                        max_attempts=max_attempts,
                    ),
                )
                .values(claimed_at=now)
            ),
        )
        if result.rowcount == 1:
            claimed.append(target_id)
    db.commit()
    return claimed


def _default_enqueue(target_id: str) -> None:
    from app.workers.celery_app import celery_app

    celery_app.send_task("app.workers.tasks.publish_target", args=[target_id], queue="publish")


def sweep_once(
    db: Session,
    *,
    enqueue: TargetEnqueueFn | None = None,
    now: datetime | None = None,
    stale_after_seconds: int = STALE_CLAIM_SECONDS,
    retry_delay_seconds: int = TARGET_RETRY_DELAY_SECONDS,
    max_attempts: int = MAX_TARGET_ATTEMPTS,
    limit: int = DEFAULT_CLAIM_LIMIT,
) -> list[str]:
    """One scheduler pass: claim due targets and enqueue a publish task per claim."""
    enqueue_fn = enqueue or _default_enqueue
    claimed = claim_due_targets(
        db,
        now=now,
        stale_after_seconds=stale_after_seconds,
        retry_delay_seconds=retry_delay_seconds,
        max_attempts=max_attempts,
        limit=limit,
    )
    for target_id in claimed:
        try:
            enqueue_fn(target_id)
        except Exception:
            # The claim stays in place and goes stale, so a later sweep retries.
            logger.exception("failed to enqueue publish target %s", target_id)
    if claimed:
        logger.info("claimed %d publish target(s): %s", len(claimed), ", ".join(claimed))
    return claimed


def run_forever(
    interval_seconds: float = SWEEP_INTERVAL_SECONDS,
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_sweeps: int | None = None,
) -> None:
    sweeps = 0
    while max_sweeps is None or sweeps < max_sweeps:
        with SessionLocal() as db:
            try:
                sweep_job_dispatches_once(db)
                sweep_once(db)
            except Exception:
                db.rollback()
                logger.exception("scheduler sweep failed; retrying next interval")
        sweeps += 1
        if max_sweeps is not None and sweeps >= max_sweeps:
            break
        sleep(interval_seconds)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    logger.info("scheduler started; sweeping every %ss (UTC)", SWEEP_INTERVAL_SECONDS)
    run_forever()


if __name__ == "__main__":
    main()
