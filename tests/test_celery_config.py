"""Celery routing configuration regression tests."""

from app.workers.celery_app import BROKER_VISIBILITY_TIMEOUT_SECONDS, celery_app
from app.workers.scheduler import JOB_RUNNING_STALE_SECONDS


def test_worker_queues_have_distinct_direct_routing_keys() -> None:
    queues = celery_app.conf.task_queues

    assert queues is not None
    bindings = {
        queue.name: (queue.exchange.name, queue.exchange.type, queue.routing_key)
        for queue in queues
    }
    assert bindings == {
        "ai": ("videoai", "direct", "ai"),
        "render": ("videoai", "direct", "render"),
        "publish": ("videoai", "direct", "publish"),
    }


def test_named_tasks_route_to_their_dedicated_queue() -> None:
    routes = celery_app.conf.task_routes

    assert routes["app.workers.tasks.discover_topics"] == {"queue": "ai"}
    assert routes["app.workers.tasks.write_script"] == {"queue": "ai"}
    assert routes["app.workers.tasks.generate_creative"] == {"queue": "ai"}
    assert routes["app.workers.tasks.render_rendition"] == {"queue": "render"}
    assert routes["app.workers.tasks.publish_job"] == {"queue": "publish"}
    assert routes["app.workers.tasks.publish_target"] == {"queue": "publish"}


def test_redis_visibility_timeout_exceeds_running_job_recovery_window() -> None:
    assert BROKER_VISIBILITY_TIMEOUT_SECONDS > JOB_RUNNING_STALE_SECONDS
    assert celery_app.conf.broker_transport_options == {
        "visibility_timeout": BROKER_VISIBILITY_TIMEOUT_SECONDS
    }
