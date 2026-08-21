"""REST API /api/v1 — exactly the endpoints from PLAN.md §2.

Async commands (generate/publish) require an ``Idempotency-Key`` header and
answer ``202 {job_id}``; replays return the identical stored response and
create no new Job rows. Every state change goes through ``app.states.advance``
(invalid jumps -> 409 envelope). Script edits always create a NEW
ScriptVersion — approved versions are never mutated.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api import idempotency
from app.api.deps import get_api_user, require_roles
from app.config import get_settings
from app.db import get_db
from app.errors import Conflict, NotFound, PolicyBlocked, ValidationFailed
from app.models import (
    AuditEvent,
    Campaign,
    ConnectedAccount,
    Creative,
    Job,
    PublishTarget,
    Rendition,
    ScriptVersion,
    Source,
    User,
)
from app.publishing.registry import auto_mode_allowed as account_auto_mode_allowed
from app.schemas.videoplan import (
    VideoPlan,
    auto_mode_allowed,
    blocking_issues,
    validate_semantics,
)
from app.states import CreativeState, InvalidTransition, Platform, Role, advance

router = APIRouter(prefix="/api/v1", tags=["api"])

IdempotencyKeyHeader = Annotated[str | None, Header(alias="Idempotency-Key")]


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class DiscoverRequest(BaseModel):
    brief: str = Field(min_length=1)
    category: str = ""
    campaign_id: str | None = None


class SourceIn(BaseModel):
    url: str = Field(min_length=1)
    title: str = ""
    publisher: str = ""
    is_official: bool = False
    citation: str = ""


class CreativeCreateRequest(BaseModel):
    campaign_id: str | None = None
    campaign_name: str = ""
    brief: str = ""
    topic_title: str = Field(min_length=1)
    angle: str = ""
    mode: str = "manual"
    cost_cap_usd: float | None = None
    sources: list[SourceIn] = Field(default_factory=list)
    video_plan: dict[str, Any] | None = None


class GenerateRequest(BaseModel):
    scene_ids: list[str] = Field(default_factory=list)


class ScriptPatchRequest(BaseModel):
    video_plan: dict[str, Any] | None = None
    approve: bool = False


class PublicationTargetIn(BaseModel):
    rendition_id: str
    platform: str
    connected_account_id: str | None = None
    scheduled_at: datetime | None = None
    privacy: str = "private"


class PublicationRequest(BaseModel):
    creative_id: str
    targets: list[PublicationTargetIn] = Field(min_length=1)
    mode: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _advance_state(creative: Creative, target: CreativeState) -> None:
    """State transitions ONLY through app.states.advance; invalid -> 409 envelope."""
    try:
        creative.state = advance(CreativeState(creative.state), target).value
    except InvalidTransition as exc:
        raise Conflict(
            str(exc),
            code="invalid_transition",
            details={"current": exc.current.value, "target": exc.target.value},
        ) from exc


def _validate_plan(raw: dict[str, Any]) -> VideoPlan:
    try:
        return VideoPlan.model_validate(raw)
    except ValidationError as exc:
        raise ValidationFailed(
            "video_plan failed schema validation",
            details={"errors": exc.errors(include_url=False, include_input=False)},
        ) from exc


def _get_or_404(db: Session, model: type, entity_id: str, label: str) -> Any:
    row = db.get(model, entity_id)
    if row is None:
        raise NotFound(f"{label} {entity_id} not found")
    return row


def _latest_script_version(db: Session, creative_id: str) -> ScriptVersion | None:
    return (
        db.query(ScriptVersion)
        .filter(ScriptVersion.creative_id == creative_id)
        .order_by(ScriptVersion.version.desc())
        .first()
    )


def _latest_approved_script_version(db: Session, creative_id: str) -> ScriptVersion | None:
    return (
        db.query(ScriptVersion)
        .filter(ScriptVersion.creative_id == creative_id, ScriptVersion.is_approved.is_(True))
        .order_by(ScriptVersion.version.desc())
        .first()
    )


def _audit(
    db: Session,
    *,
    action: str,
    entity_type: str,
    entity_id: str,
    data: dict[str, Any],
    actor: User | None = None,
) -> None:
    db.add(
        AuditEvent(
            actor_id=actor.id if actor is not None else None,
            actor_kind="user" if actor is not None else "system",
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            data=data,
        )
    )


def _enqueue_job(job: Job) -> None:
    """Hand the Job to Celery; test env (and broker outages) never break the API.

    Jobs stay QUEUED in the database either way, so a worker sweep can pick up
    anything the broker missed.
    """
    if get_settings().app_env == "test":
        return
    task_names = {
        "discover": "app.workers.tasks.discover_topics",
        "generate": "app.workers.tasks.generate_creative",
        "render": "app.workers.tasks.render_rendition",
        "publish": "app.workers.tasks.publish_job",
    }
    task_name = task_names.get(job.kind)
    if task_name is None:
        return
    try:
        from app.workers.celery_app import celery_app

        celery_app.send_task(task_name, args=[job.id], queue=job.queue)
    except Exception:  # broker down must never fail the API request
        return


def _job_response(job: Job) -> dict[str, Any]:
    return {"job_id": job.id, "status": job.status, "kind": job.kind}


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/topics/discover", status_code=202)
def discover_topics(
    body: DiscoverRequest,
    db: Annotated[Session, Depends(get_db)],
    idempotency_key: IdempotencyKeyHeader = None,
) -> JSONResponse:
    endpoint = "POST /api/v1/topics/discover"
    body_hash = idempotency.request_hash(body.model_dump(mode="json"))
    if idempotency_key:
        stored = idempotency.find_stored(
            db, key=idempotency_key, endpoint=endpoint, body_hash=body_hash
        )
        if stored is not None:
            return JSONResponse(status_code=stored.response_status, content=stored.response_body)

    job = Job(
        kind="discover",
        queue="ai",
        payload={
            "brief": body.brief,
            "category": body.category,
            "campaign_id": body.campaign_id,
        },
        idempotency_key=idempotency_key,
    )
    db.add(job)
    db.flush()
    response = _job_response(job)
    if idempotency_key:
        idempotency.store_response(
            db,
            key=idempotency_key,
            endpoint=endpoint,
            body_hash=body_hash,
            status_code=202,
            response_body=response,
        )
    db.commit()
    _enqueue_job(job)
    return JSONResponse(status_code=202, content=response)


@router.post("/creatives", status_code=201)
def create_creative(
    body: CreativeCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User | None, Depends(get_api_user)],
) -> dict[str, Any]:
    if body.campaign_id is not None:
        campaign = _get_or_404(db, Campaign, body.campaign_id, "campaign")
    else:
        campaign = Campaign(
            name=body.campaign_name or body.topic_title,
            brief=body.brief,
            mode=body.mode,
            created_by=user.id if user else None,
        )
        db.add(campaign)
        db.flush()

    settings = get_settings()
    cap = body.cost_cap_usd if body.cost_cap_usd is not None else settings.cost_hard_cap_usd
    creative = Creative(
        campaign_id=campaign.id,
        topic_title=body.topic_title,
        angle=body.angle,
        mode=body.mode,
        cost_cap_usd=min(cap, settings.cost_hard_cap_usd),
    )
    db.add(creative)
    db.flush()

    if body.sources:
        for src in body.sources:
            db.add(
                Source(
                    creative_id=creative.id,
                    url=src.url,
                    title=src.title,
                    publisher=src.publisher,
                    is_official=src.is_official,
                    citation=src.citation or f"{src.title or src.url} ({src.url})",
                )
            )
        _advance_state(creative, CreativeState.RESEARCHED)

    issues_payload: list[dict[str, Any]] | None = None
    script_version_id: str | None = None
    if body.video_plan is not None:
        plan = _validate_plan(body.video_plan)
        issues = validate_semantics(plan, cost_cap_usd=creative.cost_cap_usd)
        if CreativeState(creative.state) == CreativeState.DRAFT:
            _advance_state(creative, CreativeState.RESEARCHED)
        _advance_state(creative, CreativeState.SCRIPT_READY)
        version = ScriptVersion(
            creative_id=creative.id,
            version=1,
            video_plan=plan.model_dump(mode="json"),
            created_by=user.id if user else None,
        )
        db.add(version)
        db.flush()
        issues_payload = [i.model_dump() for i in issues]
        script_version_id = version.id
        _audit(
            db,
            action="script_validated",
            entity_type="script_version",
            entity_id=version.id,
            data={"issues": issues_payload, "version": 1},
            actor=user,
        )

    _audit(
        db,
        action="creative_created",
        entity_type="creative",
        entity_id=creative.id,
        data={"topic_title": creative.topic_title, "mode": creative.mode},
        actor=user,
    )
    db.commit()
    return {
        "id": creative.id,
        "campaign_id": campaign.id,
        "state": creative.state,
        "script_version_id": script_version_id,
        "issues": issues_payload,
    }


@router.post("/creatives/{creative_id}/generate", status_code=202)
def generate_creative(
    creative_id: str,
    db: Annotated[Session, Depends(get_db)],
    body: GenerateRequest | None = None,
    idempotency_key: IdempotencyKeyHeader = None,
) -> JSONResponse:
    key = idempotency.require_idempotency_key(idempotency_key)
    endpoint = f"POST /api/v1/creatives/{creative_id}/generate"
    payload = body.model_dump(mode="json") if body is not None else {}
    body_hash = idempotency.request_hash(payload)

    stored = idempotency.find_stored(db, key=key, endpoint=endpoint, body_hash=body_hash)
    if stored is not None:
        return JSONResponse(status_code=stored.response_status, content=stored.response_body)

    creative = _get_or_404(db, Creative, creative_id, "creative")
    _advance_state(creative, CreativeState.GENERATING)

    job = Job(
        kind="generate",
        queue="ai",
        creative_id=creative.id,
        payload={"scene_ids": payload.get("scene_ids", [])},
        idempotency_key=key,
    )
    db.add(job)
    db.flush()
    response = _job_response(job)
    idempotency.store_response(
        db,
        key=key,
        endpoint=endpoint,
        body_hash=body_hash,
        status_code=202,
        response_body=response,
    )
    db.commit()
    _enqueue_job(job)
    return JSONResponse(status_code=202, content=response)


@router.patch("/scripts/{script_version_id}")
def patch_script(
    script_version_id: str,
    body: ScriptPatchRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User | None, Depends(get_api_user)],
) -> dict[str, Any]:
    base_version: ScriptVersion = _get_or_404(
        db, ScriptVersion, script_version_id, "script version"
    )
    creative: Creative = _get_or_404(db, Creative, base_version.creative_id, "creative")

    if body.video_plan is None and not body.approve:
        raise ValidationFailed("provide video_plan and/or approve=true")

    target_version = base_version
    issues_payload: list[dict[str, Any]] = []

    if body.video_plan is not None:
        plan = _validate_plan(body.video_plan)
        issues = validate_semantics(plan, cost_cap_usd=creative.cost_cap_usd)
        issues_payload = [i.model_dump() for i in issues]
        # Versioning: NEVER mutate an existing (possibly approved) version.
        _advance_state(creative, CreativeState.SCRIPT_READY)
        max_version = db.execute(
            select(func.coalesce(func.max(ScriptVersion.version), 0)).where(
                ScriptVersion.creative_id == creative.id
            )
        ).scalar_one()
        target_version = ScriptVersion(
            creative_id=creative.id,
            version=int(max_version) + 1,
            video_plan=plan.model_dump(mode="json"),
            created_by=user.id if user else None,
        )
        db.add(target_version)
        db.flush()
        _audit(
            db,
            action="script_validated",
            entity_type="script_version",
            entity_id=target_version.id,
            data={"issues": issues_payload, "version": target_version.version},
            actor=user,
        )
    else:
        plan = _validate_plan(target_version.video_plan)
        issues = validate_semantics(plan, cost_cap_usd=creative.cost_cap_usd)
        issues_payload = [i.model_dump() for i in issues]

    approved = False
    if body.approve:
        _require_approver(user)
        blocking = blocking_issues(issues)
        if blocking:
            raise PolicyBlocked(
                "script cannot be approved while blocking issues exist",
                details={"issues": [i.model_dump() for i in blocking]},
            )
        _advance_state(creative, CreativeState.SCRIPT_APPROVED)
        target_version.is_approved = True
        target_version.approved_by = user.id if user else None
        target_version.approved_at = datetime.now(UTC)
        approved = True
        _audit(
            db,
            action="script_approved",
            entity_type="script_version",
            entity_id=target_version.id,
            data={"version": target_version.version},
            actor=user,
        )

    db.commit()
    return {
        "script_version_id": target_version.id,
        "version": target_version.version,
        "creative_id": creative.id,
        "creative_state": creative.state,
        "is_approved": approved,
        "issues": issues_payload,
        "auto_mode_allowed": auto_mode_allowed(plan, issues),
    }


def _require_approver(user: User | None) -> User:
    from app.api.deps import Forbidden, Unauthorized

    if user is None:
        raise Unauthorized("authentication required to approve")
    if user.role not in {Role.PUBLISHER.value, Role.ADMIN.value}:
        raise Forbidden(
            "approval requires the publisher or admin role",
            details={"actual": user.role},
        )
    return user


@router.post("/renditions/{rendition_id}/approve")
def approve_rendition(
    rendition_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.PUBLISHER))],
) -> dict[str, Any]:
    rendition: Rendition = _get_or_404(db, Rendition, rendition_id, "rendition")
    creative: Creative = _get_or_404(db, Creative, rendition.creative_id, "creative")

    script = _latest_approved_script_version(db, creative.id) or _latest_script_version(
        db, creative.id
    )
    if script is not None:
        plan = _validate_plan(script.video_plan)
        blocking = blocking_issues(validate_semantics(plan, cost_cap_usd=creative.cost_cap_usd))
        if blocking:
            raise PolicyBlocked(
                "rendition cannot be approved while blocking script issues exist",
                details={"issues": [i.model_dump() for i in blocking]},
            )
    if rendition.qc_report is not None and rendition.qc_report.get("passed") is False:
        raise PolicyBlocked(
            "rendition failed QC; re-render before approval",
            details={"qc_report": rendition.qc_report},
        )

    rendition.is_approved = True
    rendition.approved_by = user.id
    rendition.approved_at = datetime.now(UTC)
    _audit(
        db,
        action="rendition_approved",
        entity_type="rendition",
        entity_id=rendition.id,
        data={"locale": rendition.locale},
        actor=user,
    )

    all_renditions = db.query(Rendition).filter(Rendition.creative_id == creative.id).all()
    if all_renditions and all(r.is_approved for r in all_renditions):
        _advance_state(creative, CreativeState.FINAL_APPROVED)

    db.commit()
    return {
        "rendition_id": rendition.id,
        "is_approved": True,
        "creative_id": creative.id,
        "creative_state": creative.state,
    }


@router.post("/publications", status_code=202)
def create_publication(
    body: PublicationRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User | None, Depends(get_api_user)],
    idempotency_key: IdempotencyKeyHeader = None,
) -> JSONResponse:
    key = idempotency.require_idempotency_key(idempotency_key)
    endpoint = "POST /api/v1/publications"
    body_hash = idempotency.request_hash(body.model_dump(mode="json"))
    stored = idempotency.find_stored(db, key=key, endpoint=endpoint, body_hash=body_hash)
    if stored is not None:
        return JSONResponse(status_code=stored.response_status, content=stored.response_body)

    creative: Creative = _get_or_404(db, Creative, body.creative_id, "creative")
    mode = (body.mode or creative.mode or "manual").lower()

    valid_platforms = {p.value for p in Platform}
    renditions: dict[str, Rendition] = {}
    for target in body.targets:
        if target.platform not in valid_platforms:
            raise ValidationFailed(
                f"unsupported platform {target.platform!r}",
                details={"supported": sorted(valid_platforms)},
            )
        rendition = _get_or_404(db, Rendition, target.rendition_id, "rendition")
        if rendition.creative_id != creative.id:
            raise ValidationFailed(
                f"rendition {rendition.id} does not belong to creative {creative.id}"
            )
        renditions[target.rendition_id] = rendition

    if mode == "auto":
        _enforce_auto_gate(db, creative, body.targets)
    else:
        unapproved = [r.id for r in renditions.values() if not r.is_approved]
        if unapproved:
            raise PolicyBlocked(
                "manual mode requires approved renditions before publishing",
                details={"unapproved_rendition_ids": unapproved},
            )

    _advance_state(creative, CreativeState.SCHEDULED)

    target_ids: list[str] = []
    for target in body.targets:
        row = PublishTarget(
            creative_id=creative.id,
            rendition_id=target.rendition_id,
            platform=target.platform,
            connected_account_id=target.connected_account_id,
            scheduled_at=_as_utc(target.scheduled_at),
            privacy=target.privacy,
        )
        db.add(row)
        db.flush()
        target_ids.append(row.id)

    job = Job(
        kind="publish",
        queue="publish",
        creative_id=creative.id,
        payload={"target_ids": target_ids, "mode": mode},
        idempotency_key=key,
    )
    db.add(job)
    db.flush()
    _audit(
        db,
        action="publication_requested",
        entity_type="creative",
        entity_id=creative.id,
        data={"target_ids": target_ids, "mode": mode},
        actor=user,
    )
    response = {**_job_response(job), "target_ids": target_ids}
    idempotency.store_response(
        db,
        key=key,
        endpoint=endpoint,
        body_hash=body_hash,
        status_code=202,
        response_body=response,
    )
    db.commit()
    _enqueue_job(job)
    return JSONResponse(status_code=202, content=response)


def _enforce_auto_gate(
    db: Session, creative: Creative, targets: list[PublicationTargetIn]
) -> None:
    """Auto mode gate: zero blocking issues, sourced facts, auto-capable accounts."""
    script = _latest_approved_script_version(db, creative.id)
    if script is None:
        raise PolicyBlocked("auto mode requires an approved script version")
    plan = _validate_plan(script.video_plan)
    issues = validate_semantics(plan, cost_cap_usd=creative.cost_cap_usd)
    if not auto_mode_allowed(plan, issues):
        raise PolicyBlocked(
            "auto mode is not allowed for this plan (blocking issues or unsourced scenes)",
            details={"issues": [i.model_dump() for i in blocking_issues(issues)]},
        )
    for target in targets:
        if target.connected_account_id is None:
            raise PolicyBlocked(
                f"auto mode requires a connected account for platform {target.platform}"
            )
        account = _get_or_404(
            db, ConnectedAccount, target.connected_account_id, "connected account"
        )
        if not account_auto_mode_allowed(account):
            raise PolicyBlocked(
                f"account capability {account.capability} on {target.platform} "
                "does not allow auto publishing",
                details={"platform": target.platform, "capability": account.capability},
            )
    _audit(
        db,
        action="auto_mode_gate_passed",
        entity_type="creative",
        entity_id=creative.id,
        data={"platforms": [t.platform for t in targets]},
    )


@router.get("/jobs/{job_id}")
def get_job(job_id: str, db: Annotated[Session, Depends(get_db)]) -> dict[str, Any]:
    job: Job = _get_or_404(db, Job, job_id, "job")
    return {
        "id": job.id,
        "kind": job.kind,
        "queue": job.queue,
        "status": job.status,
        "creative_id": job.creative_id,
        "payload": job.payload,
        "result": job.result,
        "error": job.error,
        "attempts": job.attempts,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }
