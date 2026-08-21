"""Provider webhooks: signature check, event dedupe, idempotent status updates.

Dedupe reuses the IdempotencyKey table with ``endpoint="webhook:<provider>"``
and the provider's event id as the key: a repeated delivery returns the stored
response and re-applies NOTHING, so webhook retries can never duplicate state
changes. Signature verification (HMAC-SHA256 over the raw body) runs whenever
the provider's app secret is configured.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api import idempotency
from app.config import Settings, get_settings
from app.db import get_db
from app.errors import AppError, NotFound, ValidationFailed
from app.models import PublishTarget
from app.states import Platform, PublishTargetStatus

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

SIGNATURE_HEADERS = ("X-Hub-Signature-256", "X-Signature")

_STATUS_MAP: dict[str, str] = {
    "published": PublishTargetStatus.PUBLISHED.value,
    "live": PublishTargetStatus.PUBLISHED.value,
    "finished": PublishTargetStatus.PUBLISHED.value,
    "scheduled": PublishTargetStatus.SCHEDULED_REMOTE.value,
    "failed": PublishTargetStatus.FAILED.value,
    "rejected": PublishTargetStatus.FAILED.value,
    "needs_action": PublishTargetStatus.NEEDS_ACTION.value,
    "policy_blocked": PublishTargetStatus.NEEDS_ACTION.value,
}


class SignatureRejected(AppError):
    status_code = 401
    code = "webhook_signature_rejected"


def webhook_secret(provider: str, settings: Settings) -> str:
    attrs = {
        Platform.YOUTUBE.value: settings.youtube_client_secret,
        Platform.FACEBOOK.value: settings.facebook_app_secret,
        Platform.TIKTOK.value: settings.tiktok_client_secret,
        Platform.ZALO.value: settings.zalo_app_secret,
    }
    return attrs.get(provider, "")


def verify_signature(provider: str, raw_body: bytes, headers: Any, settings: Settings) -> None:
    """HMAC-SHA256 check when a secret is configured; skipped otherwise."""
    secret = webhook_secret(provider, settings)
    if not secret:
        return
    provided: str | None = None
    for header in SIGNATURE_HEADERS:
        value = headers.get(header)
        if value:
            provided = value
            break
    if not provided:
        raise SignatureRejected(f"missing webhook signature for {provider}")
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    candidate = provided.removeprefix("sha256=")
    if not hmac.compare_digest(candidate, expected):
        raise SignatureRejected(f"invalid webhook signature for {provider}")


def _event_id(payload: dict[str, Any], raw_body: bytes) -> str:
    for field in ("event_id", "id"):
        value = payload.get(field)
        if isinstance(value, str | int) and str(value):
            return str(value)
    return hashlib.sha256(raw_body).hexdigest()  # content-address unnamed events


def _find_target(db: Session, provider: str, payload: dict[str, Any]) -> PublishTarget | None:
    target_id = payload.get("target_id")
    if isinstance(target_id, str) and target_id:
        return db.get(PublishTarget, target_id)
    remote_post_id = payload.get("remote_post_id")
    if isinstance(remote_post_id, str | int) and str(remote_post_id):
        return (
            db.query(PublishTarget)
            .filter(
                PublishTarget.platform == provider,
                PublishTarget.remote_post_id == str(remote_post_id),
            )
            .order_by(PublishTarget.created_at.desc())
            .first()
        )
    return None


def _apply_event(db: Session, provider: str, payload: dict[str, Any]) -> dict[str, Any]:
    target = _find_target(db, provider, payload)
    if target is None:
        return {"status": "ignored", "reason": "no matching publish target"}

    remote_status = payload.get("status_detail")
    if not isinstance(remote_status, dict):
        remote_status = {k: v for k, v in payload.items() if k not in {"event_id", "id"}}
    target.remote_status = remote_status

    remote_post_id = payload.get("remote_post_id")
    if isinstance(remote_post_id, str | int) and str(remote_post_id):
        target.remote_post_id = str(remote_post_id)

    raw_status = str(payload.get("status", "")).lower()
    mapped = _STATUS_MAP.get(raw_status)
    if mapped is not None:
        target.status = mapped
        if mapped == PublishTargetStatus.FAILED.value:
            target.last_error = {
                "code": "remote_failure",
                "message": str(payload.get("message", "remote reported failure")),
                "retryable": False,
                "details": remote_status,
            }
    return {"status": "processed", "target_id": target.id, "target_status": target.status}


@router.post("/{provider}")
async def receive_webhook(
    provider: str,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> JSONResponse:
    if provider not in {p.value for p in Platform}:
        raise NotFound(f"unknown webhook provider {provider!r}")

    raw_body = await request.body()
    verify_signature(provider, raw_body, request.headers, get_settings())

    try:
        payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValidationFailed("webhook body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValidationFailed("webhook body must be a JSON object")

    endpoint = f"webhook:{provider}"
    event_id = _event_id(payload, raw_body)
    body_hash = idempotency.request_hash(payload)

    stored = idempotency.find_stored(db, key=event_id, endpoint=endpoint, body_hash=body_hash)
    if stored is not None:
        # Duplicate delivery: same response, zero re-applied state changes.
        return JSONResponse(status_code=stored.response_status, content=stored.response_body)

    result = _apply_event(db, provider, payload)
    idempotency.store_response(
        db,
        key=event_id,
        endpoint=endpoint,
        body_hash=body_hash,
        status_code=200,
        response_body=result,
    )
    db.commit()
    return JSONResponse(status_code=200, content=result)
