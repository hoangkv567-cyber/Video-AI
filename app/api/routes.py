"""REST API /api/v1 — exactly the endpoints from PLAN.md §2.

Async commands (generate/publish) require an ``Idempotency-Key`` header and
answer ``202 {job_id}``; replays return the identical stored response and
create no new Job rows. Every state change goes through ``app.states.advance``
(invalid jumps -> 409 envelope). Script edits always create a NEW
ScriptVersion — approved versions are never mutated.
"""

from __future__ import annotations

import mimetypes
from collections import Counter
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, Response
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai.factory import generation_preflight_issues
from app.ai.topic_snapshots import topic_snapshot_from_job_result, validate_http_source_url
from app.ai.topics import registrable_domain
from app.ai.veo import JobOperationStore, StoredOperation
from app.api import idempotency
from app.api.deps import Forbidden, require_roles
from app.config import get_settings
from app.costs import CostLedger
from app.db import get_db
from app.errors import Conflict, NotFound, PolicyBlocked, ValidationFailed
from app.maintenance.creative_deletion import delete_creative_graph
from app.media.qc import qc_report_passed
from app.models import (
    Asset,
    AuditEvent,
    Campaign,
    ConnectedAccount,
    CostEvent,
    Creative,
    Job,
    PublishTarget,
    Rendition,
    Scene,
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
from app.states import CreativeState, InvalidTransition, JobStatus, Platform, Role, advance
from app.storage import get_asset_store
from app.workers.dispatch import dispatch_job

router = APIRouter(prefix="/api/v1", tags=["api"])

IdempotencyKeyHeader = Annotated[str | None, Header(alias="Idempotency-Key")]


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class DiscoverRequest(BaseModel):
    brief: str = Field(min_length=1)
    category: str = ""
    campaign_id: str | None = None


class SelectTopicRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic_index: StrictInt = Field(ge=0)
    campaign_name: str | None = Field(default=None, min_length=1, max_length=255)
    mode: Literal["manual", "auto"] = "manual"


class SourceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    title: str = ""
    publisher: str = ""
    is_official: bool = False
    citation: str = ""

    _safe_http_url = field_validator("url")(validate_http_source_url)


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


class VideoSubmissionReconcileRequest(BaseModel):
    """An explicit admin decision for an indeterminate paid video submit."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["attach_operation", "confirm_not_submitted"]
    provider: Literal["veo", "wan"]
    operation_name: str | None = Field(default=None, max_length=500)
    reason: str = Field(min_length=3, max_length=500)

    @field_validator("operation_name", "reason", mode="before")
    @classmethod
    def strip_operator_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_action_fields(self) -> Self:
        if self.action == "attach_operation" and not self.operation_name:
            raise ValueError("operation_name is required when attaching an operation")
        if self.action == "confirm_not_submitted" and self.operation_name is not None:
            raise ValueError("operation_name is not allowed when confirming no submission")
        return self


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


def _canonicalize_plan_sources(
    db: Session,
    *,
    creative_id: str,
    plan: VideoPlan,
) -> VideoPlan:
    """Bind untrusted plan source references to the creative's persisted sources."""
    persisted_sources = (
        db.execute(select(Source).where(Source.creative_id == creative_id)).scalars().all()
    )
    persisted_by_url: dict[str, Source] = {}
    duplicate_persisted_urls: set[str] = set()
    for source in persisted_sources:
        if source.url in persisted_by_url:
            duplicate_persisted_urls.add(source.url)
        persisted_by_url[source.url] = source

    plan_urls = [source.url for source in plan.sources]
    plan_url_counts = Counter(plan_urls)
    duplicate_plan_urls = {url for url, count in plan_url_counts.items() if count > 1}
    persisted_urls = set(persisted_by_url)
    supplied_urls = set(plan_urls)
    missing_urls = persisted_urls - supplied_urls
    unexpected_urls = supplied_urls - persisted_urls
    if (
        missing_urls
        or unexpected_urls
        or duplicate_plan_urls
        or duplicate_persisted_urls
        or len(plan_urls) != len(persisted_sources)
    ):
        raise ValidationFailed(
            "video_plan sources must exactly match the creative's persisted sources",
            code="source_integrity_mismatch",
            details={
                "missing_urls": sorted(missing_urls),
                "unexpected_urls": sorted(unexpected_urls),
                "duplicate_plan_urls": sorted(duplicate_plan_urls),
                "duplicate_persisted_urls": sorted(duplicate_persisted_urls),
            },
        )

    canonical_sources = [
        source_ref.model_copy(
            update={
                "title": persisted_by_url[source_ref.url].title,
                "publisher": persisted_by_url[source_ref.url].publisher,
                "is_official": persisted_by_url[source_ref.url].is_official,
            }
        )
        for source_ref in plan.sources
    ]
    return plan.model_copy(update={"sources": canonical_sources})


def _validate_plan_for_creative(
    db: Session,
    *,
    creative_id: str,
    raw: dict[str, Any],
) -> VideoPlan:
    plan = _validate_plan(raw)
    return _canonicalize_plan_sources(db, creative_id=creative_id, plan=plan)


def _get_or_404(db: Session, model: type, entity_id: str, label: str) -> Any:
    row = db.get(model, entity_id)
    if row is None:
        raise NotFound(f"{label} {entity_id} not found")
    return row


def _get_creative_for_update(db: Session, creative_id: str) -> Creative:
    """Serialize mutually exclusive commands for one creative in PostgreSQL."""
    row = db.execute(
        select(Creative)
        .where(Creative.id == creative_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if row is None:
        raise NotFound(f"creative {creative_id} not found")
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


def _job_response(job: Job) -> dict[str, Any]:
    return {"job_id": job.id, "status": job.status, "kind": job.kind}


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    if not header:
        return None
    if not header.startswith("bytes=") or "," in header:
        raise ValidationFailed("invalid Range header")
    start_raw, separator, end_raw = header[6:].partition("-")
    if not separator:
        raise ValidationFailed("invalid Range header")
    try:
        if start_raw:
            start = int(start_raw)
            end = int(end_raw) if end_raw else size - 1
        else:
            suffix = int(end_raw)
            if suffix <= 0:
                raise ValueError
            start = max(0, size - suffix)
            end = size - 1
    except ValueError as exc:
        raise ValidationFailed("invalid Range header") from exc
    if start < 0 or start >= size or end < start:
        raise ValidationFailed("requested byte range is outside the asset")
    return start, min(end, size - 1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/topics/discover", status_code=202)
def discover_topics(
    body: DiscoverRequest,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(require_roles(Role.EDITOR))],
    idempotency_key: IdempotencyKeyHeader = None,
) -> JSONResponse:
    key = idempotency.require_idempotency_key(idempotency_key)
    endpoint = "POST /api/v1/topics/discover"
    body_hash = idempotency.request_hash(body.model_dump(mode="json"))
    reservation = idempotency.reserve_request(db, key=key, endpoint=endpoint, body_hash=body_hash)
    if reservation.replay:
        return JSONResponse(
            status_code=reservation.row.response_status,
            content=reservation.row.response_body,
        )

    job = Job(
        kind="discover",
        queue="ai",
        payload={
            "brief": body.brief,
            "category": body.category,
            "campaign_id": body.campaign_id,
        },
        idempotency_key=key,
    )
    db.add(job)
    db.flush()
    response = _job_response(job)
    idempotency.finalize_reservation(reservation, status_code=202, response_body=response)
    db.commit()
    dispatch_job(db, job.id)
    return JSONResponse(status_code=202, content=response)


@router.post("/discoveries/{discovery_job_id}/select", status_code=202)
def select_discovery_topic(
    discovery_job_id: str,
    body: SelectTopicRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.EDITOR))],
    idempotency_key: IdempotencyKeyHeader = None,
) -> JSONResponse:
    """Select one server-stored topic and enqueue structured script generation."""
    key = idempotency.require_idempotency_key(idempotency_key)
    discovery: Job = _get_or_404(db, Job, discovery_job_id, "discovery job")
    if discovery.kind != "discover":
        raise ValidationFailed(
            "job is not a topic discovery",
            details={"job_id": discovery.id, "kind": discovery.kind},
        )
    if discovery.status != JobStatus.SUCCEEDED.value:
        retryable = discovery.status in {
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
        } or bool((discovery.error or {}).get("retryable"))
        raise Conflict(
            "topic discovery is not ready for selection",
            code="discovery_not_ready",
            retryable=retryable,
            details={"job_id": discovery.id, "status": discovery.status},
        )

    snapshot = topic_snapshot_from_job_result(discovery.result, body.topic_index)
    candidate = snapshot.to_topic_candidate()
    discovery_payload = discovery.payload or {}
    campaign_id = discovery_payload.get("campaign_id")
    campaign: Campaign | None = None
    if campaign_id:
        if body.campaign_name is not None:
            raise ValidationFailed(
                "campaign_name cannot be supplied when discovery already has campaign_id"
            )
        campaign = _get_or_404(db, Campaign, str(campaign_id), "campaign")

    endpoint = f"POST /api/v1/discoveries/{discovery_job_id}/select"
    body_hash = idempotency.request_hash(body.model_dump(mode="json"))
    reservation = idempotency.reserve_request(db, key=key, endpoint=endpoint, body_hash=body_hash)
    if reservation.replay:
        return JSONResponse(
            status_code=reservation.row.response_status,
            content=reservation.row.response_body,
        )

    if campaign is None:
        campaign = Campaign(
            name=body.campaign_name or candidate.title,
            brief=str(discovery_payload.get("brief", "")),
            category=str(discovery_payload.get("category", "")),
            mode=body.mode,
            created_by=user.id,
        )
        db.add(campaign)
        db.flush()

    settings = get_settings()
    creative = Creative(
        campaign_id=campaign.id,
        state=CreativeState.DRAFT.value,
        topic_title=candidate.title,
        angle=candidate.summary,
        mode=body.mode,
        cost_cap_usd=settings.cost_hard_cap_usd,
    )
    db.add(creative)
    db.flush()

    for source in candidate.sources:
        publisher = source.publisher or registrable_domain(source.url)
        db.add(
            Source(
                creative_id=creative.id,
                url=source.url,
                title=source.title,
                publisher=publisher,
                is_official=source.is_official,
                accessed_at=source.accessed_at,
                citation=(f"{source.title or source.url} — {publisher} ({source.url})"),
            )
        )
    _advance_state(creative, CreativeState.RESEARCHED)

    script_job = Job(
        kind="script",
        queue="ai",
        creative_id=creative.id,
        payload={
            "discovery_job_id": discovery.id,
            "topic_index": body.topic_index,
        },
        idempotency_key=key,
    )
    db.add(script_job)
    db.flush()
    _audit(
        db,
        action="topic_selected",
        entity_type="creative",
        entity_id=creative.id,
        data={
            "discovery_job_id": discovery.id,
            "topic_index": body.topic_index,
            "script_job_id": script_job.id,
            "source_count": len(candidate.sources),
        },
        actor=user,
    )

    response = {
        **_job_response(script_job),
        "campaign_id": campaign.id,
        "creative_id": creative.id,
        "selected_topic": {"index": body.topic_index, "title": candidate.title},
    }
    idempotency.finalize_reservation(reservation, status_code=202, response_body=response)
    db.commit()
    dispatch_job(db, script_job.id)
    return JSONResponse(status_code=202, content=response)


@router.post("/creatives", status_code=201)
def create_creative(
    body: CreativeCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.EDITOR))],
    idempotency_key: IdempotencyKeyHeader = None,
) -> JSONResponse:
    key = idempotency.require_idempotency_key(idempotency_key)
    endpoint = "POST /api/v1/creatives"
    body_hash = idempotency.request_hash(body.model_dump(mode="json"))
    reservation = idempotency.reserve_request(db, key=key, endpoint=endpoint, body_hash=body_hash)
    if reservation.replay:
        return JSONResponse(
            status_code=reservation.row.response_status,
            content=reservation.row.response_body,
        )

    if body.campaign_id is not None:
        campaign = _get_or_404(db, Campaign, body.campaign_id, "campaign")
    else:
        campaign = Campaign(
            name=body.campaign_name or body.topic_title,
            brief=body.brief,
            mode=body.mode,
            created_by=user.id,
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
        db.flush()
        _advance_state(creative, CreativeState.RESEARCHED)

    issues_payload: list[dict[str, Any]] | None = None
    script_version_id: str | None = None
    if body.video_plan is not None:
        plan = _validate_plan_for_creative(
            db,
            creative_id=creative.id,
            raw=body.video_plan,
        )
        issues = validate_semantics(plan, cost_cap_usd=creative.cost_cap_usd)
        if CreativeState(creative.state) == CreativeState.DRAFT:
            _advance_state(creative, CreativeState.RESEARCHED)
        _advance_state(creative, CreativeState.SCRIPT_READY)
        version = ScriptVersion(
            creative_id=creative.id,
            version=1,
            video_plan=plan.model_dump(mode="json"),
            created_by=user.id,
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
    response = {
        "id": creative.id,
        "campaign_id": campaign.id,
        "state": creative.state,
        "script_version_id": script_version_id,
        "issues": issues_payload,
    }
    idempotency.finalize_reservation(reservation, status_code=201, response_body=response)
    db.commit()
    return JSONResponse(status_code=201, content=response)


@router.delete("/creatives/{creative_id}")
def delete_creative(
    creative_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.ADMIN))],
) -> JSONResponse:
    """Permanently remove one unpublished, idle creative and its local files."""
    creative = _get_creative_for_update(db, creative_id)
    result = delete_creative_graph(
        db,
        creative,
        get_asset_store(),
        actor_id=user.id,
    )
    return JSONResponse(status_code=200, content=result)


@router.post("/creatives/{creative_id}/generate", status_code=202)
def generate_creative(
    creative_id: str,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(require_roles(Role.EDITOR))],
    body: GenerateRequest | None = None,
    idempotency_key: IdempotencyKeyHeader = None,
) -> JSONResponse:
    preflight = generation_preflight_issues()
    if preflight:
        raise PolicyBlocked(
            "media generation providers are not ready",
            code="generation_preflight_failed",
            details={"issues": preflight},
        )
    key = idempotency.require_idempotency_key(idempotency_key)
    endpoint = f"POST /api/v1/creatives/{creative_id}/generate"
    payload = body.model_dump(mode="json") if body is not None else {}
    body_hash = idempotency.request_hash(payload)
    reservation = idempotency.reserve_request(db, key=key, endpoint=endpoint, body_hash=body_hash)
    if reservation.replay:
        return JSONResponse(
            status_code=reservation.row.response_status,
            content=reservation.row.response_body,
        )

    creative = _get_creative_for_update(db, creative_id)
    # GENERATING is a valid resume point: a transient upstream failure leaves
    # the creative there and a retry must re-enqueue (the worker task itself
    # re-validates and resumes from persisted assets). A double-click while a
    # generate job is still in flight stays a 409.
    if CreativeState(creative.state) == CreativeState.GENERATING:
        active = db.execute(
            select(Job)
            .where(
                Job.creative_id == creative.id,
                Job.kind == "generate",
                Job.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]),
            )
            .limit(1)
        ).scalars().first()
        if active is not None:
            raise Conflict(
                "generation already in progress",
                code="generation_in_progress",
                details={"active_job_id": active.id},
            )
    else:
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
    idempotency.finalize_reservation(reservation, status_code=202, response_body=response)
    db.commit()
    dispatch_job(db, job.id)
    return JSONResponse(status_code=202, content=response)


@router.post(
    "/creatives/{creative_id}/scenes/{scene_id}/video-submission/reconcile",
    status_code=202,
)
def reconcile_video_submission(
    creative_id: str,
    scene_id: str,
    body: VideoSubmissionReconcileRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.ADMIN))],
    idempotency_key: IdempotencyKeyHeader = None,
) -> JSONResponse:
    """Resolve a paid video submission whose remote outcome is unknown.

    The endpoint never guesses. An administrator either supplies the exact
    remote operation identifier to resume polling, or explicitly confirms
    that the provider created no task so a new submission may be made.
    """
    key = idempotency.require_idempotency_key(idempotency_key)
    endpoint = (
        f"POST /api/v1/creatives/{creative_id}/scenes/{scene_id}/"
        "video-submission/reconcile"
    )
    body_payload = body.model_dump(mode="json")
    body_hash = idempotency.request_hash(body_payload)
    reservation = idempotency.reserve_request(
        db,
        key=key,
        endpoint=endpoint,
        body_hash=body_hash,
    )
    if reservation.replay:
        return JSONResponse(
            status_code=reservation.row.response_status,
            content=reservation.row.response_body,
        )

    creative = _get_creative_for_update(db, creative_id)
    if CreativeState(creative.state) != CreativeState.NEEDS_ACTION:
        raise Conflict(
            "creative does not require video-submission reconciliation",
            code="video_submission_reconciliation_not_required",
            details={"state": creative.state},
        )

    scene = db.execute(
        select(Scene)
        .where(Scene.id == scene_id, Scene.creative_id == creative.id)
        .with_for_update()
    ).scalar_one_or_none()
    if scene is None:
        raise NotFound(f"scene {scene_id} not found for creative {creative.id}")
    script = _latest_approved_script_version(db, creative.id)
    if script is None or scene.script_version_id != script.id:
        raise Conflict(
            "scene does not belong to the current approved script",
            code="video_submission_scene_stale",
            details={"scene_id": scene.id},
        )

    active_generate = db.execute(
        select(Job)
        .where(
            Job.creative_id == creative.id,
            Job.kind == "generate",
            Job.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]),
        )
        .limit(1)
    ).scalars().first()
    if active_generate is not None:
        raise Conflict(
            "generation already in progress",
            code="generation_in_progress",
            details={"active_job_id": active_generate.id},
        )

    store = JobOperationStore(db, commit=False)
    stored = store.get(scene.id)
    if stored is None:
        raise NotFound(f"no active video submission intent for scene {scene.id}")
    if not stored.submission_ambiguous:
        raise Conflict(
            "video submission already has a remote operation identifier",
            code="video_submission_reconciliation_not_required",
            details={"scene_id": scene.id},
        )
    if stored.provider_kind and stored.provider_kind != body.provider:
        raise Conflict(
            "operator-selected provider does not match the durable intent",
            code="video_submission_provider_mismatch",
            details={
                "stored_provider": stored.provider_kind,
                "selected_provider": body.provider,
            },
        )
    if stored.creative_id and stored.creative_id != creative.id:
        raise Conflict(
            "video submission intent belongs to another creative",
            code="video_submission_intent_mismatch",
            details={"scene_id": scene.id},
        )

    cost_note = f"{body.provider} scene {scene.id}"
    projection_filters = [
        CostEvent.creative_id == creative.id,
        CostEvent.kind == body.provider,
        CostEvent.projected.is_(True),
    ]
    if stored.cost_event_id:
        projection_filters.append(CostEvent.id == stored.cost_event_id)
    else:
        projection_filters.append(CostEvent.note == cost_note)
    projected_rows = db.execute(
        select(CostEvent).where(*projection_filters).with_for_update()
    ).scalars().all()
    if len(projected_rows) != 1:
        raise Conflict(
            "submission intent does not have exactly one matching cost projection",
            code="video_submission_projection_mismatch",
            details={
                "scene_id": scene.id,
                "provider": body.provider,
                "matching_projections": len(projected_rows),
            },
        )

    if body.action == "attach_operation":
        configured_provider = get_settings().video_provider
        if configured_provider != body.provider:
            raise PolicyBlocked(
                "configured video provider cannot poll the supplied operation",
                code="video_submission_provider_mismatch",
                details={
                    "configured_provider": configured_provider,
                    "intent_provider": body.provider,
                },
            )
        assert body.operation_name is not None
        store.put(
            scene.id,
            StoredOperation(
                operation_name=body.operation_name,
                model_id=stored.model_id,
                duration_seconds=stored.duration_seconds,
                provider_kind=body.provider,
                creative_id=creative.id,
                cost_event_id=projected_rows[0].id,
            ),
        )
    else:
        store.cancel_ambiguous_intent(scene.id, reason=body.reason)
        ledger = CostLedger(db)
        removed = (
            ledger.remove_projected_event(
                stored.cost_event_id,
                creative_id=creative.id,
                kind=body.provider,
            )
            if stored.cost_event_id
            else ledger.remove_projected(
                creative.id,
                kind=body.provider,
                note=cost_note,
            )
        )
        if removed != 1:
            raise Conflict(
                "submission cost projection changed during reconciliation",
                code="video_submission_projection_mismatch",
                details={"scene_id": scene.id, "removed": removed},
            )

    preflight = generation_preflight_issues()
    if preflight:
        raise PolicyBlocked(
            "media generation providers are not ready",
            code="generation_preflight_failed",
            details={"issues": preflight},
        )

    scene.status = "pending"
    scene.last_error = None
    _advance_state(creative, CreativeState.GENERATING)
    creative.last_error = None
    job = Job(
        kind="generate",
        queue="ai",
        creative_id=creative.id,
        payload={"scene_ids": []},
        idempotency_key=key,
    )
    db.add(job)
    db.flush()
    _audit(
        db,
        action="video_submission_reconciled",
        entity_type="scene",
        entity_id=scene.id,
        data={
            "creative_id": creative.id,
            "action": body.action,
            "provider": body.provider,
            "operation_suffix": (
                body.operation_name[-12:] if body.operation_name is not None else None
            ),
            "reason": body.reason,
            "resume_job_id": job.id,
        },
        actor=user,
    )
    response = {
        **_job_response(job),
        "creative_id": creative.id,
        "scene_id": scene.id,
        "action": body.action,
        "provider": body.provider,
    }
    idempotency.finalize_reservation(reservation, status_code=202, response_body=response)
    db.commit()
    dispatch_job(db, job.id)
    return JSONResponse(status_code=202, content=response)


@router.patch("/scripts/{script_version_id}")
def patch_script(
    script_version_id: str,
    body: ScriptPatchRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.EDITOR, Role.PUBLISHER))],
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
        plan = _validate_plan_for_creative(
            db,
            creative_id=creative.id,
            raw=body.video_plan,
        )
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
    if not qc_report_passed(rendition.qc_report):
        raise PolicyBlocked(
            "rendition has not passed QC; re-render before approval",
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
    user: Annotated[User, Depends(require_roles(Role.PUBLISHER))],
    idempotency_key: IdempotencyKeyHeader = None,
) -> JSONResponse:
    key = idempotency.require_idempotency_key(idempotency_key)
    endpoint = "POST /api/v1/publications"
    body_hash = idempotency.request_hash(body.model_dump(mode="json"))
    reservation = idempotency.reserve_request(db, key=key, endpoint=endpoint, body_hash=body_hash)
    if reservation.replay:
        return JSONResponse(
            status_code=reservation.row.response_status,
            content=reservation.row.response_body,
        )

    creative = _get_creative_for_update(db, body.creative_id)
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

    if creative.state == CreativeState.READY.value:
        _advance_state(creative, CreativeState.FINAL_APPROVED)
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
    idempotency.finalize_reservation(reservation, status_code=202, response_body=response)
    db.commit()
    dispatch_job(db, job.id)
    return JSONResponse(status_code=202, content=response)


def _enforce_auto_gate(db: Session, creative: Creative, targets: list[PublicationTargetIn]) -> None:
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


def _asset_response(asset: Asset, request: Request, *, attachment: bool = False) -> Response:
    store = get_asset_store()
    data = store.get_bytes(asset.storage_key)
    size = len(data)
    selected = _byte_range(request.headers.get("Range"), size)
    media_type = mimetypes.guess_type(asset.storage_key)[0] or "application/octet-stream"
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, no-store" if attachment else "private, max-age=300",
        "Content-Disposition": (
            f'attachment; filename="{asset.storage_key.rsplit("/", 1)[-1]}"'
            if attachment
            else "inline"
        ),
        "X-Content-Type-Options": "nosniff",
    }
    if selected is None:
        headers["Content-Length"] = str(size)
        return Response(content=data, media_type=media_type, headers=headers)
    start, end = selected
    chunk = data[start : end + 1]
    headers.update(
        {
            "Content-Length": str(len(chunk)),
            "Content-Range": f"bytes {start}-{end}/{size}",
        }
    )
    return Response(content=chunk, status_code=206, media_type=media_type, headers=headers)


@router.get("/assets/{asset_id}/stream")
def stream_asset(
    asset_id: str,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(require_roles(Role.EDITOR, Role.PUBLISHER))],
) -> Response:
    """Serve local assets with Range support or redirect to a presigned object URL."""
    asset: Asset = _get_or_404(db, Asset, asset_id, "asset")
    return _asset_response(asset, request)


@router.get("/publish-targets/{target_id}/bundle")
def download_publish_bundle(
    target_id: str,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    _user: Annotated[User, Depends(require_roles(Role.PUBLISHER))],
) -> Response:
    target: PublishTarget = _get_or_404(db, PublishTarget, target_id, "publish target")
    if not target.bundle_path:
        raise NotFound(f"publish target {target.id} has no manual bundle")
    asset = (
        db.execute(
            select(Asset)
            .where(
                Asset.creative_id == target.creative_id,
                Asset.storage_key == target.bundle_path,
                Asset.kind == "bundle",
                Asset.platform == target.platform,
            )
            .order_by(Asset.created_at.desc())
        )
        .scalars()
        .first()
    )
    if asset is None:
        raise NotFound(f"manual bundle for publish target {target.id} not found")
    return _asset_response(asset, request, attachment=True)


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles(Role.EDITOR, Role.PUBLISHER))],
) -> dict[str, Any]:
    job: Job = _get_or_404(db, Job, job_id, "job")
    if job.kind == JobOperationStore.KIND and user.role != Role.ADMIN.value:
        raise Forbidden("video provider operation jobs are restricted to administrators")
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
