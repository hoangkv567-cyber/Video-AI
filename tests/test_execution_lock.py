from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.workers.execution import ExecutionLockLost, advisory_lock_key, execution_lock


@pytest.fixture()
def postgres_engine() -> Generator[Engine, None, None]:
    url = os.environ.get("TEST_POSTGRES_URL", "")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not configured")
    engine = create_engine(url, pool_pre_ping=True)
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    try:
        yield engine
    finally:
        engine.dispose()


def test_advisory_lock_key_is_stable_scoped_signed_bigint() -> None:
    first = advisory_lock_key("job", "abc")

    assert first == advisory_lock_key("job", "abc")
    assert first != advisory_lock_key("publish_target", "abc")
    assert first != advisory_lock_key("job", "xyz")
    assert -(2**63) <= first < 2**63


def test_sqlite_execution_lock_is_always_acquired() -> None:
    engine = create_engine("sqlite://")

    with execution_lock(engine, "job", "abc") as acquired:
        assert acquired is True

    engine.dispose()


@pytest.mark.postgres
def test_postgres_execution_lock_serializes_same_key(
    postgres_engine: Engine,
) -> None:
    entity_id = f"contention-{uuid.uuid4()}"

    with execution_lock(postgres_engine, "job", entity_id) as first:
        assert first is True
        with execution_lock(postgres_engine, "job", entity_id) as contender:
            assert contender is False
        with execution_lock(postgres_engine, "job", f"{entity_id}-other") as other:
            assert other is True

    with execution_lock(postgres_engine, "job", entity_id) as reacquired:
        assert reacquired is True


@pytest.mark.postgres
def test_postgres_backend_termination_releases_lock_and_reports_loss(
    postgres_engine: Engine,
) -> None:
    holder_name = f"videoai-lock-holder-{uuid.uuid4()}"
    holder_engine = create_engine(
        os.environ["TEST_POSTGRES_URL"],
        connect_args={"application_name": holder_name},
        pool_pre_ping=False,
    )
    entity_id = f"failover-{uuid.uuid4()}"
    entered = threading.Event()
    release_holder = threading.Event()
    holder_errors: list[BaseException] = []

    def hold_lock() -> None:
        try:
            with execution_lock(holder_engine, "job", entity_id) as acquired:
                assert acquired is True
                entered.set()
                assert release_holder.wait(timeout=15)
        except BaseException as exc:
            holder_errors.append(exc)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    try:
        assert entered.wait(timeout=10)
        with postgres_engine.connect() as admin:
            holder_pid = admin.execute(
                text(
                    "SELECT pid FROM pg_stat_activity "
                    "WHERE application_name = :application_name"
                ),
                {"application_name": holder_name},
            ).scalar_one()
            terminated = admin.execute(
                text("SELECT pg_terminate_backend(:pid)"),
                {"pid": holder_pid},
            ).scalar_one()
            admin.commit()
        assert terminated is True

        # PostgreSQL releases session advisory locks when the backend dies,
        # even though the old Python worker is still inside the protected body.
        with execution_lock(postgres_engine, "job", entity_id) as recovered:
            assert recovered is True
    finally:
        release_holder.set()
        holder.join(timeout=10)
        holder_engine.dispose()

    assert holder.is_alive() is False
    assert len(holder_errors) == 1
    assert isinstance(holder_errors[0], ExecutionLockLost)
