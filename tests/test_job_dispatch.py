"""Durable Job dispatch and recovery-sweep tests (fully offline)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import update
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from app.models import Job
from app.states import JobStatus
from app.workers.dispatch import (
    TASK_BY_JOB_KIND,
    DispatchRequest,
    dispatch_job,
    dispatch_request,
)
from app.workers.scheduler import (
    JOB_DISPATCH_STALE_SECONDS,
    JOB_RETRY_DELAY_SECONDS,
    JOB_RUNNING_STALE_SECONDS,
    MAX_JOB_ATTEMPTS,
    _job_candidate_select,
    _job_dispatch_conditions,
    claim_dispatchable_jobs,
    sweep_job_dispatches_once,
)

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


def _seed_job(
    db: Session,
    *,
    kind: str = "discover",
    queue: str = "ai",
    status: str = JobStatus.QUEUED.value,
    attempts: int = 0,
    dispatched_at: datetime | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    error: dict | None = None,
) -> Job:
    job = Job(
        kind=kind,
        queue=queue,
        status=status,
        payload={},
        attempts=attempts,
        dispatched_at=dispatched_at,
        started_at=started_at,
        finished_at=finished_at,
        error=error,
    )
    db.add(job)
    db.commit()
    return job


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def test_every_dispatchable_job_kind_has_the_expected_task() -> None:
    expected = {
        "discover": "app.workers.tasks.discover_topics",
        "script": "app.workers.tasks.write_script",
        "generate": "app.workers.tasks.generate_creative",
        "render": "app.workers.tasks.render_rendition",
        "publish": "app.workers.tasks.publish_job",
    }

    assert expected == TASK_BY_JOB_KIND
    for kind, task_name in expected.items():
        job = Job(id=f"job-{kind}", kind=kind, queue="render" if kind == "render" else "ai")
        assert dispatch_request(job) == DispatchRequest(job.id, task_name, job.queue)


def test_api_dispatch_checkpoints_only_after_send_succeeds(db_session: Session) -> None:
    job = _seed_job(db_session)
    observed: list[DispatchRequest] = []

    def send(request: DispatchRequest) -> None:
        assert db_session.get(Job, job.id).dispatched_at is None
        observed.append(request)

    assert dispatch_job(db_session, job.id, send=send, now=NOW) is True
    assert observed == [DispatchRequest(job.id, TASK_BY_JOB_KIND["discover"], "ai")]
    db_session.expire_all()
    assert _as_utc(db_session.get(Job, job.id).dispatched_at) == NOW


def test_api_dispatch_failure_leaves_job_immediately_recoverable(db_session: Session) -> None:
    job = _seed_job(db_session)

    def broken_send(_request: DispatchRequest) -> None:
        raise ConnectionError("broker unavailable")

    assert dispatch_job(db_session, job.id, send=broken_send, now=NOW) is False
    db_session.expire_all()
    assert db_session.get(Job, job.id).dispatched_at is None
    assert claim_dispatchable_jobs(db_session, now=NOW) == [job.id]


def test_dispatch_does_not_send_an_already_checkpointed_job(db_session: Session) -> None:
    job = _seed_job(db_session, dispatched_at=NOW)
    sent: list[DispatchRequest] = []

    assert dispatch_job(db_session, job.id, send=sent.append, now=NOW) is False
    assert sent == []


def test_claims_new_stale_retryable_and_stale_running_jobs_only(
    db_session: Session,
) -> None:
    new = _seed_job(db_session)
    stale_dispatch = _seed_job(
        db_session,
        kind="render",
        queue="render",
        dispatched_at=NOW - timedelta(seconds=JOB_DISPATCH_STALE_SECONDS + 1),
    )
    _seed_job(
        db_session,
        dispatched_at=NOW - timedelta(seconds=JOB_DISPATCH_STALE_SECONDS - 1),
    )
    retryable = _seed_job(
        db_session,
        kind="script",
        status=JobStatus.FAILED.value,
        attempts=1,
        error={"retryable": True},
        finished_at=NOW - timedelta(seconds=JOB_RETRY_DELAY_SECONDS + 1),
    )
    _seed_job(
        db_session,
        status=JobStatus.FAILED.value,
        attempts=1,
        error={"retryable": True},
        finished_at=NOW - timedelta(seconds=JOB_RETRY_DELAY_SECONDS - 1),
    )
    _seed_job(
        db_session,
        status=JobStatus.FAILED.value,
        attempts=1,
        error={"retryable": False},
        finished_at=NOW - timedelta(hours=1),
    )
    stale_running = _seed_job(
        db_session,
        kind="generate",
        status=JobStatus.RUNNING.value,
        attempts=2,
        started_at=NOW - timedelta(seconds=JOB_RUNNING_STALE_SECONDS + 1),
    )
    _seed_job(
        db_session,
        status=JobStatus.RUNNING.value,
        attempts=2,
        started_at=NOW - timedelta(seconds=JOB_RUNNING_STALE_SECONDS - 1),
    )
    _seed_job(
        db_session,
        status=JobStatus.FAILED.value,
        attempts=MAX_JOB_ATTEMPTS,
        error={"retryable": True},
        finished_at=NOW - timedelta(hours=1),
    )
    _seed_job(db_session, kind="veo_operation")

    claimed = claim_dispatchable_jobs(db_session, now=NOW, limit=20)

    assert set(claimed) == {new.id, stale_dispatch.id, retryable.id, stale_running.id}
    for job_id in claimed:
        row = db_session.get(Job, job_id)
        assert row.status == JobStatus.QUEUED.value
        assert _as_utc(row.dispatched_at) == NOW
        assert row.finished_at is None
    assert claim_dispatchable_jobs(db_session, now=NOW, limit=20) == []


def test_sqlite_guarded_claim_allows_only_one_racing_update(db_session: Session) -> None:
    job = _seed_job(db_session)
    conditions = _job_dispatch_conditions(
        NOW,
        dispatch_stale_after_seconds=JOB_DISPATCH_STALE_SECONDS,
        retry_delay_seconds=JOB_RETRY_DELAY_SECONDS,
        running_stale_after_seconds=JOB_RUNNING_STALE_SECONDS,
        max_attempts=MAX_JOB_ATTEMPTS,
    )
    statement = (
        update(Job)
        .where(Job.id == job.id, *conditions)
        .values(dispatched_at=NOW, status=JobStatus.QUEUED.value)
    )

    winner = db_session.execute(statement)
    loser = db_session.execute(statement)
    db_session.commit()

    assert winner.rowcount == 1
    assert loser.rowcount == 0


def test_postgresql_candidate_query_uses_skip_locked() -> None:
    conditions = _job_dispatch_conditions(
        NOW,
        dispatch_stale_after_seconds=JOB_DISPATCH_STALE_SECONDS,
        retry_delay_seconds=JOB_RETRY_DELAY_SECONDS,
        running_stale_after_seconds=JOB_RUNNING_STALE_SECONDS,
        max_attempts=MAX_JOB_ATTEMPTS,
    )
    statement = _job_candidate_select(conditions, 10, skip_locked=True)

    sql = str(statement.compile(dialect=postgresql.dialect()))

    assert "FOR UPDATE SKIP LOCKED" in sql


def test_sweep_uses_central_mapping_and_recovers_failed_broker_send(
    db_session: Session,
) -> None:
    job = _seed_job(db_session, kind="publish", queue="publish")

    def broken_send(_request: DispatchRequest) -> None:
        raise ConnectionError("broker unavailable")

    first = sweep_job_dispatches_once(db_session, send=broken_send, now=NOW)
    assert first == [job.id]
    assert sweep_job_dispatches_once(db_session, send=lambda _request: None, now=NOW) == []

    sent: list[DispatchRequest] = []
    recovered_at = NOW + timedelta(seconds=JOB_DISPATCH_STALE_SECONDS + 1)
    recovered = sweep_job_dispatches_once(db_session, send=sent.append, now=recovered_at)

    assert recovered == [job.id]
    assert sent == [DispatchRequest(job.id, TASK_BY_JOB_KIND["publish"], "publish")]
