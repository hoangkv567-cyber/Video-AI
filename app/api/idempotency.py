"""Idempotency-Key handling for async command endpoints (and webhook dedupe).

Contract (PLAN.md §2): generate/publish commands REQUIRE an ``Idempotency-Key``
header. The first call stores its 202 response; any replay with the same
key+endpoint returns the EXACT stored response and creates no new rows.
Reusing a key with a different body is a 409 Conflict.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.errors import Conflict, ValidationFailed
from app.models import IdempotencyKey


@dataclass(frozen=True)
class IdempotencyReservation:
    """A key claimed before domain writes, or a committed replay response."""

    row: IdempotencyKey
    replay: bool


def require_idempotency_key(key: str | None) -> str:
    if not key or not key.strip():
        raise ValidationFailed(
            "Idempotency-Key header is required for this endpoint",
            details={"header": "Idempotency-Key"},
        )
    return key.strip()


def request_hash(body: Any) -> str:
    """Canonical sha256 over the parsed request body (dict/list/None)."""
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def find_stored(db: Session, *, key: str, endpoint: str, body_hash: str) -> IdempotencyKey | None:
    """Return the stored response row, or None. Body mismatch -> 409 Conflict."""
    row = (
        db.query(IdempotencyKey)
        .filter(IdempotencyKey.key == key, IdempotencyKey.endpoint == endpoint)
        .one_or_none()
    )
    if row is None:
        return None
    if row.request_hash != body_hash:
        raise Conflict(
            "Idempotency-Key was already used with a different request body",
            code="idempotency_key_conflict",
            details={"key": key, "endpoint": endpoint},
        )
    return row


def reserve_request(
    db: Session, *, key: str, endpoint: str, body_hash: str
) -> IdempotencyReservation:
    """Reserve ``key`` before creating any Job/Creative side effects.

    The reservation and domain writes remain in the same transaction. On
    PostgreSQL a concurrent insert waits for the winner; if it loses the unique
    race, rolling back is safe because this helper was the first write.
    """
    existing = find_stored(db, key=key, endpoint=endpoint, body_hash=body_hash)
    if existing is not None:
        return IdempotencyReservation(existing, replay=True)

    row = IdempotencyKey(
        key=key,
        endpoint=endpoint,
        request_hash=body_hash,
        response_status=0,
        response_body={},
    )
    db.add(row)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        winner = find_stored(db, key=key, endpoint=endpoint, body_hash=body_hash)
        if winner is None:  # pragma: no cover - unique winner must be visible
            raise
        return IdempotencyReservation(winner, replay=True)
    return IdempotencyReservation(row, replay=False)


def finalize_reservation(
    reservation: IdempotencyReservation,
    *,
    status_code: int,
    response_body: dict[str, Any],
) -> IdempotencyKey:
    if reservation.replay:
        return reservation.row
    reservation.row.response_status = status_code
    reservation.row.response_body = response_body
    return reservation.row


def store_response(
    db: Session,
    *,
    key: str,
    endpoint: str,
    body_hash: str,
    status_code: int,
    response_body: dict[str, Any],
) -> IdempotencyKey:
    """Persist the response for replays; a lost unique-race re-reads the winner."""
    row = IdempotencyKey(
        key=key,
        endpoint=endpoint,
        request_hash=body_hash,
        response_status=status_code,
        response_body=response_body,
    )
    db.add(row)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        existing = find_stored(db, key=key, endpoint=endpoint, body_hash=body_hash)
        if existing is None:  # pragma: no cover - the race winner must exist
            raise
        return existing
    return row
