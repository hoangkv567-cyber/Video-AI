"""Scheduler sweep tests: atomic claims, no double-enqueue, stale-claim recovery.

Runs on SQLite, exercising the optimistic-update fallback path (PostgreSQL uses
FOR UPDATE SKIP LOCKED in production). All times are UTC."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.models import Creative, PublishAttempt, PublishTarget, Rendition
from app.states import PublishTargetStatus
from app.workers import scheduler
from app.workers.scheduler import STALE_CLAIM_SECONDS, claim_due_targets, sweep_once


def _rendition(db: Session, creative: Creative) -> Rendition:
    """One shared rendition per creative (unique constraint on creative+locale)."""
    row = db.query(Rendition).filter(Rendition.creative_id == creative.id).first()
    if row is None:
        row = Rendition(creative_id=creative.id, locale="vi", title="t")
        db.add(row)
        db.flush()
    return row


def _seed_target(
    db: Session,
    creative: Creative,
    *,
    scheduled_at: datetime | None,
    status: str = PublishTargetStatus.PENDING.value,
    claimed_at: datetime | None = None,
) -> PublishTarget:
    rendition = _rendition(db, creative)
    target = PublishTarget(
        creative_id=creative.id,
        rendition_id=rendition.id,
        platform="youtube",
        scheduled_at=scheduled_at,
        status=status,
        claimed_at=claimed_at,
    )
    db.add(target)
    db.commit()
    return target


def _now() -> datetime:
    return datetime.now(UTC)


def test_due_target_claimed_exactly_once_across_two_sweeps(
    db_session: Session, creative: Creative
) -> None:
    target = _seed_target(db_session, creative, scheduled_at=_now() - timedelta(minutes=5))
    first = claim_due_targets(db_session)
    second = claim_due_targets(db_session)
    assert first == [target.id]
    assert second == []  # already claimed; not re-claimable while fresh


def test_racing_claims_only_one_update_wins(db_session: Session, creative: Creative) -> None:
    """Both sweeps read the same candidate; the guarded UPDATE lets one win."""
    target = _seed_target(db_session, creative, scheduled_at=_now() - timedelta(minutes=1))
    now = _now()
    stale_cutoff = now - timedelta(seconds=STALE_CLAIM_SECONDS)
    conditions = scheduler._due_conditions(now, stale_cutoff)

    # Simulated race: two sweeps each run the optimistic claim for the same id.
    win = db_session.execute(
        update(PublishTarget)
        .where(PublishTarget.id == target.id, *conditions)
        .values(claimed_at=now)
    )
    lose = db_session.execute(
        update(PublishTarget)
        .where(PublishTarget.id == target.id, *conditions)
        .values(claimed_at=now)
    )
    db_session.commit()
    assert win.rowcount == 1
    assert lose.rowcount == 0


def test_future_and_unscheduled_targets_are_not_claimed(
    db_session: Session, creative: Creative
) -> None:
    _seed_target(db_session, creative, scheduled_at=_now() + timedelta(hours=2))
    _seed_target(db_session, creative, scheduled_at=None)  # immediate: publish_job's job
    assert claim_due_targets(db_session) == []


def test_non_pending_statuses_are_not_claimed(db_session: Session, creative: Creative) -> None:
    for status in (
        PublishTargetStatus.PUBLISHED.value,
        PublishTargetStatus.NEEDS_ACTION.value,
        PublishTargetStatus.UPLOADING.value,
    ):
        _seed_target(
            db_session, creative, scheduled_at=_now() - timedelta(minutes=10), status=status
        )
    failed = _seed_target(
        db_session,
        creative,
        scheduled_at=_now() - timedelta(minutes=10),
        status=PublishTargetStatus.FAILED.value,
    )
    failed.last_error = {"retryable": False}
    db_session.commit()
    assert claim_due_targets(db_session) == []


def test_retryable_failed_immediate_target_is_reclaimed_after_backoff(
    db_session: Session, creative: Creative
) -> None:
    now = _now()
    target = _seed_target(
        db_session,
        creative,
        scheduled_at=None,
        status=PublishTargetStatus.FAILED.value,
        claimed_at=now - timedelta(minutes=3),
    )
    target.last_error = {"code": "publish_retryable", "retryable": True}
    db_session.add(
        PublishAttempt(
            target_id=target.id,
            attempt_no=1,
            status="failed",
            finished_at=now - timedelta(minutes=2),
        )
    )
    db_session.commit()

    assert claim_due_targets(db_session, now=now) == [target.id]


def test_retryable_target_waits_and_stops_at_attempt_cap(
    db_session: Session, creative: Creative
) -> None:
    now = _now()
    recent = _seed_target(
        db_session,
        creative,
        scheduled_at=None,
        status=PublishTargetStatus.FAILED.value,
    )
    recent.last_error = {"retryable": True}
    db_session.add(
        PublishAttempt(
            target_id=recent.id,
            attempt_no=1,
            status="failed",
            finished_at=now,
        )
    )

    capped = _seed_target(
        db_session,
        creative,
        scheduled_at=now - timedelta(hours=1),
        status=PublishTargetStatus.FAILED.value,
        claimed_at=now - timedelta(hours=1),
    )
    capped.last_error = {"retryable": True}
    for attempt_no in range(1, scheduler.MAX_TARGET_ATTEMPTS + 1):
        db_session.add(
            PublishAttempt(
                target_id=capped.id,
                attempt_no=attempt_no,
                status="failed",
                finished_at=now - timedelta(minutes=5),
            )
        )
    db_session.commit()

    assert claim_due_targets(db_session, now=now) == []


def test_sweep_enqueues_each_claim_once(db_session: Session, creative: Creative) -> None:
    due_1 = _seed_target(db_session, creative, scheduled_at=_now() - timedelta(minutes=3))
    due_2 = _seed_target(db_session, creative, scheduled_at=_now() - timedelta(minutes=2))
    _seed_target(db_session, creative, scheduled_at=_now() + timedelta(hours=1))
    enqueued: list[str] = []

    claimed = sweep_once(db_session, enqueue=enqueued.append)
    assert set(claimed) == {due_1.id, due_2.id}
    assert sorted(enqueued) == sorted([due_1.id, due_2.id])

    # Second sweep (still claimed): nothing new is enqueued -> no double publish.
    again = sweep_once(db_session, enqueue=enqueued.append)
    assert again == []
    assert len(enqueued) == 2


def test_stale_claim_is_resweepable_after_restart(
    db_session: Session, creative: Creative
) -> None:
    """A claim from a worker that died goes stale and is picked up again."""
    stale = _now() - timedelta(seconds=STALE_CLAIM_SECONDS * 2)
    target = _seed_target(
        db_session,
        creative,
        scheduled_at=_now() - timedelta(hours=1),
        claimed_at=stale,
    )
    enqueued: list[str] = []
    claimed = sweep_once(db_session, enqueue=enqueued.append)
    assert claimed == [target.id]
    assert enqueued == [target.id]
    db_session.expire_all()
    refreshed = db_session.get(PublishTarget, target.id)
    assert refreshed.claimed_at is not None
    claimed_at = refreshed.claimed_at
    if claimed_at.tzinfo is None:
        claimed_at = claimed_at.replace(tzinfo=UTC)
    assert claimed_at > stale  # claim timestamp refreshed (UTC)


def test_stale_uploading_checkpoint_is_reclaimed_for_safe_reconciliation(
    db_session: Session, creative: Creative
) -> None:
    stale = _now() - timedelta(seconds=STALE_CLAIM_SECONDS * 2)
    target = _seed_target(
        db_session,
        creative,
        scheduled_at=None,
        status=PublishTargetStatus.UPLOADING.value,
        claimed_at=stale,
    )
    db_session.add(
        PublishAttempt(
            target_id=target.id,
            attempt_no=1,
            status="uploading",
            response={"encrypted_checkpoint": "opaque-ciphertext"},
            started_at=stale,
        )
    )
    db_session.commit()
    enqueued: list[str] = []

    claimed = sweep_once(db_session, enqueue=enqueued.append, now=_now())

    assert claimed == [target.id]
    assert enqueued == [target.id]
    assert db_session.query(PublishAttempt).filter_by(target_id=target.id).count() == 1


def test_fresh_claim_is_not_resweepable(db_session: Session, creative: Creative) -> None:
    _seed_target(
        db_session,
        creative,
        scheduled_at=_now() - timedelta(hours=1),
        claimed_at=_now() - timedelta(seconds=30),
    )
    assert sweep_once(db_session, enqueue=lambda _tid: None) == []


def test_enqueue_failure_keeps_the_claim_for_a_later_stale_resweep(
    db_session: Session, creative: Creative
) -> None:
    target = _seed_target(db_session, creative, scheduled_at=_now() - timedelta(minutes=1))

    def broken_enqueue(_target_id: str) -> None:
        raise ConnectionError("broker down")

    claimed = sweep_once(db_session, enqueue=broken_enqueue)
    assert claimed == [target.id]  # claim persisted despite the enqueue failure
    # Not immediately re-swept (claim is fresh) ...
    assert sweep_once(db_session, enqueue=lambda _tid: None) == []
    # ... but a sweep after the stale window recovers it.
    future_now = _now() + timedelta(seconds=STALE_CLAIM_SECONDS * 2)
    recovered: list[str] = []
    assert sweep_once(db_session, enqueue=recovered.append, now=future_now) == [target.id]
    assert recovered == [target.id]


def test_claim_limit_respected(db_session: Session, creative: Creative) -> None:
    for minutes in (30, 20, 10):
        _seed_target(db_session, creative, scheduled_at=_now() - timedelta(minutes=minutes))
    first = claim_due_targets(db_session, limit=2)
    assert len(first) == 2
    second = claim_due_targets(db_session, limit=2)
    assert len(second) == 1  # the remaining due target on the next pass
