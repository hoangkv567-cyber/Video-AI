"""Celery application: redis broker/backend, three queues (ai/render/publish).

``task_acks_late=True`` plus DB-level idempotency in the tasks themselves means
a worker killed mid-task is safe: the redelivered task finds the persisted
operation names / Asset rows and never re-calls a paid provider.
"""

from __future__ import annotations

from celery import Celery
from kombu import Queue

from app.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "videoai",
    broker=_settings.redis_url,
    backend=_settings.redis_url,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_default_queue="ai",
    task_queues=(Queue("ai"), Queue("render"), Queue("publish")),
    task_routes={
        "app.workers.tasks.discover_topics": {"queue": "ai"},
        "app.workers.tasks.generate_creative": {"queue": "ai"},
        "app.workers.tasks.render_rendition": {"queue": "render"},
        "app.workers.tasks.publish_job": {"queue": "publish"},
        "app.workers.tasks.publish_target": {"queue": "publish"},
    },
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)
