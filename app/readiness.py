"""Bounded, read-only probes for infrastructure required by the application."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import urllib3
from minio import Minio
from redis import Redis
from sqlalchemy import text
from urllib3.util import Retry, Timeout

from app.config import get_settings
from app.db import engine
from app.media.runner import SubprocessRunner, check

READINESS_TIMEOUT_SECONDS = 2.0

ReadinessCheck = Callable[[], None]


def check_postgresql() -> None:
    """Verify that the configured SQL database accepts a trivial query."""
    with engine.connect() as connection:
        if connection.scalar(text("SELECT 1")) != 1:
            raise RuntimeError("database readiness query returned an unexpected value")


def check_redis() -> None:
    """Verify broker connectivity without publishing or consuming a message."""
    settings = get_settings()
    client = Redis.from_url(
        settings.redis_url,
        socket_connect_timeout=READINESS_TIMEOUT_SECONDS,
        socket_timeout=READINESS_TIMEOUT_SECONDS,
    )
    try:
        if not client.ping():
            raise RuntimeError("redis readiness ping returned a false response")
    finally:
        client.close()


def check_minio() -> None:
    """Verify MinIO connectivity and credentials without creating any objects."""
    settings = get_settings()
    http_client = urllib3.PoolManager(
        timeout=Timeout(
            connect=READINESS_TIMEOUT_SECONDS,
            read=READINESS_TIMEOUT_SECONDS,
        ),
        retries=Retry(total=0),
        cert_reqs="CERT_REQUIRED",
    )
    try:
        client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            http_client=http_client,
        )
        # A missing bucket is not a readiness failure: the storage backend
        # intentionally creates it lazily on the first write.
        client.bucket_exists(settings.minio_bucket)
    finally:
        http_client.clear()


def check_ffmpeg() -> None:
    """Verify both media binaries used by rendering and quality control."""
    runner = SubprocessRunner(timeout_seconds=READINESS_TIMEOUT_SECONDS)
    for binary in ("ffmpeg", "ffprobe"):
        check(runner.run((binary, "-version")))


def get_readiness_checks() -> Mapping[str, ReadinessCheck]:
    """Return the readiness probes; kept as a factory for offline tests."""
    return {
        "postgresql": check_postgresql,
        "redis": check_redis,
        "minio": check_minio,
        "ffmpeg": check_ffmpeg,
    }
