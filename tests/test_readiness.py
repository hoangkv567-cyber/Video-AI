"""Offline unit tests for the internal dependency readiness probes."""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Any

import pytest

from app import readiness
from app.media.runner import CommandResult


class _FakeConnection:
    def __init__(self) -> None:
        self.statement = ""

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def scalar(self, statement: Any) -> int:
        self.statement = str(statement)
        return 1


class _FakeEngine:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection
        self.connect_calls = 0

    def connect(self) -> _FakeConnection:
        self.connect_calls += 1
        return self.connection


def test_postgresql_probe_runs_read_only_scalar_query(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection()
    fake_engine = _FakeEngine(connection)
    monkeypatch.setattr(readiness, "engine", fake_engine)

    readiness.check_postgresql()

    assert fake_engine.connect_calls == 1
    assert connection.statement == "SELECT 1"


class _FakeRedisClient:
    def __init__(self) -> None:
        self.ping_calls = 0
        self.closed = False

    def ping(self) -> bool:
        self.ping_calls += 1
        return True

    def close(self) -> None:
        self.closed = True


def test_redis_probe_uses_bounded_timeouts_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeRedisClient()
    captured: dict[str, Any] = {}

    class FakeRedis:
        @classmethod
        def from_url(cls, url: str, **kwargs: Any) -> _FakeRedisClient:
            captured.update(url=url, **kwargs)
            return fake_client

    monkeypatch.setattr(readiness, "Redis", FakeRedis)

    readiness.check_redis()

    assert fake_client.ping_calls == 1
    assert fake_client.closed is True
    assert captured["socket_connect_timeout"] == readiness.READINESS_TIMEOUT_SECONDS
    assert captured["socket_timeout"] == readiness.READINESS_TIMEOUT_SECONDS


class _FakeHttpClient:
    def __init__(self) -> None:
        self.cleared = False

    def clear(self) -> None:
        self.cleared = True


def test_minio_probe_checks_configured_bucket_without_mutating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_http = _FakeHttpClient()
    captured: dict[str, Any] = {}

    class FakeMinioClient:
        def bucket_exists(self, bucket: str) -> bool:
            captured["bucket"] = bucket
            return False

    def fake_minio(endpoint: str, **kwargs: Any) -> FakeMinioClient:
        captured.update(endpoint=endpoint, **kwargs)
        return FakeMinioClient()

    monkeypatch.setattr(readiness.urllib3, "PoolManager", lambda **_kwargs: fake_http)
    monkeypatch.setattr(readiness, "Minio", fake_minio)

    readiness.check_minio()

    settings = readiness.get_settings()
    assert captured["endpoint"] == settings.minio_endpoint
    assert captured["bucket"] == settings.minio_bucket
    assert captured["http_client"] is fake_http
    assert fake_http.cleared is True


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv: Sequence[str]) -> CommandResult:
        command = tuple(argv)
        self.calls.append(command)
        return CommandResult(command, 0, "version", "")


def test_ffmpeg_probe_checks_both_required_binaries(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_runner = _FakeRunner()
    captured_timeout: list[float] = []

    def fake_runner_factory(*, timeout_seconds: float) -> _FakeRunner:
        captured_timeout.append(timeout_seconds)
        return fake_runner

    monkeypatch.setattr(readiness, "SubprocessRunner", fake_runner_factory)

    readiness.check_ffmpeg()

    assert captured_timeout == [readiness.READINESS_TIMEOUT_SECONDS]
    assert fake_runner.calls == [("ffmpeg", "-version"), ("ffprobe", "-version")]


def test_readiness_factory_exposes_all_required_checks() -> None:
    assert tuple(readiness.get_readiness_checks()) == (
        "postgresql",
        "redis",
        "minio",
        "ffmpeg",
    )
