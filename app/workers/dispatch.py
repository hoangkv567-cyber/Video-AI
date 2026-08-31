"""Durable Job-to-Celery dispatch boundary.

The database Job row is the outbox record.  API handlers commit it first and
then call :func:`dispatch_job`; a successful broker send is followed by a
``dispatched_at`` checkpoint.  If either send or checkpoint fails, the
scheduler can safely redispatch the still-eligible row.

Delivery is intentionally at-least-once.  Worker execution still needs its
own lease/fencing guard because a process can die after the broker accepts a
message but before ``dispatched_at`` commits.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Job
from app.states import JobStatus

logger = logging.getLogger("videoai.dispatch")

TASK_BY_JOB_KIND: dict[str, str] = {
    "discover": "app.workers.tasks.discover_topics",
    "script": "app.workers.tasks.write_script",
    "generate": "app.workers.tasks.generate_creative",
    "render": "app.workers.tasks.render_rendition",
    "publish": "app.workers.tasks.publish_job",
}
DISPATCHABLE_JOB_KINDS = frozenset(TASK_BY_JOB_KIND)


@dataclass(frozen=True, slots=True)
class DispatchRequest:
    job_id: str
    task_name: str
    queue: str


SendFn = Callable[[DispatchRequest], None]


def dispatch_request(job: Job) -> DispatchRequest:
    """Map a persisted Job to its Celery task without importing Celery."""
    task_name = TASK_BY_JOB_KIND.get(job.kind)
    if task_name is None:
        raise ValueError(f"job kind {job.kind!r} is not dispatchable")
    return DispatchRequest(job_id=job.id, task_name=task_name, queue=job.queue)


def _default_send(request: DispatchRequest) -> None:
    from app.workers.celery_app import celery_app

    celery_app.send_task(request.task_name, args=[request.job_id], queue=request.queue)


def enqueue_job(job: Job, *, send: SendFn | None = None) -> None:
    """Send one Job to its mapped queue; broker errors intentionally propagate."""
    (send or _default_send)(dispatch_request(job))


def dispatch_job(
    db: Session,
    job_id: str,
    *,
    send: SendFn | None = None,
    now: datetime | None = None,
) -> bool:
    """Send a committed Job and checkpoint only after broker acceptance.

    The row lock prevents the PostgreSQL recovery sweep from claiming this Job
    while the API request is publishing it.  A broker/checkpoint failure is
    logged and returned as ``False`` so the already-committed API command stays
    available to the recovery sweep.
    """
    if send is None and get_settings().app_env == "test":
        return False

    try:
        job = db.execute(
            select(Job)
            .where(Job.id == job_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).scalar_one_or_none()
        if job is None:
            db.rollback()
            logger.error("cannot dispatch missing job %s", job_id)
            return False
        if job.kind not in DISPATCHABLE_JOB_KINDS:
            db.rollback()
            logger.error("cannot dispatch unsupported job %s kind=%s", job.id, job.kind)
            return False
        if job.status != JobStatus.QUEUED.value or job.dispatched_at is not None:
            db.rollback()
            return False

        enqueue_job(job, send=send)
        job.dispatched_at = now or datetime.now(UTC)
        db.commit()
        return True
    except Exception:
        db.rollback()
        logger.exception("failed to dispatch job %s; recovery sweep will retry", job_id)
        return False
