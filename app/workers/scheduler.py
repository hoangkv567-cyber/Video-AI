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

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import PublishTarget
from app.states import PublishTargetStatus

logger = logging.getLogger("videoai.scheduler")

SWEEP_INTERVAL_SECONDS = 60
STALE_CLAIM_SECONDS = 600
DEFAULT_CLAIM_LIMIT = 50

DUE_STATUSES = (
    PublishTargetStatus.PENDING.value,
    PublishTargetStatus.VALIDATED.value,
)

EnqueueFn = Callable[[str], None]


def _now() -> datetime:
    return datetime.now(UTC)


def _due_conditions(now: datetime, stale_cutoff: datetime) -> list:
    return [
        PublishTarget.status.in_(DUE_STATUSES),
        PublishTarget.scheduled_at.is_not(None),
        PublishTarget.scheduled_at <= now,
        or_(PublishTarget.claimed_at.is_(None), PublishTarget.claimed_at < stale_cutoff),
    ]


def claim_due_targets(
    db: Session,
    *,
    now: datetime | None = None,
    stale_after_seconds: int = STALE_CLAIM_SECONDS,
    limit: int = DEFAULT_CLAIM_LIMIT,
) -> list[str]:
    """Atomically claim due targets; each id is returned by exactly one sweep."""
    now = now or _now()
    stale_cutoff = now - timedelta(seconds=stale_after_seconds)
    conditions = _due_conditions(now, stale_cutoff)
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
        result = db.execute(
            update(PublishTarget)
            .where(PublishTarget.id == target_id, *_due_conditions(now, stale_cutoff))
            .values(claimed_at=now)
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
    enqueue: EnqueueFn | None = None,
    now: datetime | None = None,
    stale_after_seconds: int = STALE_CLAIM_SECONDS,
    limit: int = DEFAULT_CLAIM_LIMIT,
) -> list[str]:
    """One scheduler pass: claim due targets and enqueue a publish task per claim."""
    enqueue_fn = enqueue or _default_enqueue
    claimed = claim_due_targets(
        db, now=now, stale_after_seconds=stale_after_seconds, limit=limit
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
