"""Process-independent execution locks for at-least-once worker delivery.

Celery may redeliver a message while the first worker is still alive.  In
production PostgreSQL advisory locks provide one live executor per logical
entity.  SQLite is used only for local tests and therefore deliberately falls
back to an always-acquired lock.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger("videoai.execution")


class ExecutionLockLost(RuntimeError):
    """The PostgreSQL session lock disappeared before clean task exit."""


def advisory_lock_key(scope: str, entity_id: str) -> int:
    """Return a stable signed 64-bit key accepted by PostgreSQL."""
    payload = f"video-ai\x00{scope}\x00{entity_id}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)


@contextmanager
def execution_lock(engine: Engine, scope: str, entity_id: str) -> Iterator[bool]:
    """Try to hold an entity lock for the full duration of one worker call.

    A dedicated connection is intentional: worker sessions commit between
    expensive stages, while a PostgreSQL session-level advisory lock must stay
    attached to one connection until the task exits.
    """
    if engine.dialect.name != "postgresql":
        yield True
        return

    key = advisory_lock_key(scope, entity_id)
    with engine.connect() as connection:
        acquired = bool(
            connection.execute(
                text("SELECT pg_try_advisory_lock(:lock_key)"),
                {"lock_key": key},
            ).scalar_one()
        )
        connection.commit()
        body_failed = False
        try:
            yield acquired
        except BaseException:
            body_failed = True
            raise
        finally:
            if acquired:
                database_error: SQLAlchemyError | None = None
                released = False
                try:
                    released = bool(
                        connection.execute(
                            text("SELECT pg_advisory_unlock(:lock_key)"),
                            {"lock_key": key},
                        ).scalar_one()
                    )
                    connection.commit()
                except SQLAlchemyError as exc:
                    database_error = exc

                if not released:
                    message = (
                        f"PostgreSQL execution lock was lost for {scope}:{entity_id}; "
                        "the completed work must be treated as indeterminate"
                    )
                    if body_failed:
                        logger.error("%s (unlock error: %s)", message, database_error)
                    elif database_error is not None:
                        raise ExecutionLockLost(message) from database_error
                    else:
                        raise ExecutionLockLost(message)
