"""Idempotent worker tasks binding the pipeline (PLAN.md §2, weeks 3-6).

Every step checks for an existing Asset (by prompt hash / kind / scene) BEFORE
calling a paid provider, and Veo operations are persisted through the
OperationStore before submit returns — so a killed and restarted worker never
re-calls Veo, never regenerates keyframes and never re-synthesizes voices.
Job rows track queued/running/succeeded/failed with the unified error envelope.

The ``run_*`` functions take an explicit Session plus injectable providers so
tests exercise them directly without any broker; the Celery wrappers only add
session management and default (lazily imported) providers.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from subprocess import TimeoutExpired
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Any

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai.base import (
    OP_RUNNING,
    OP_SUCCEEDED,
    ImageProvider,
    ResearchProvider,
    ScriptProvider,
    SourceInfo,
    TTSProvider,
    VideoProvider,
)
from app.ai.factory import (
    PIPELINE_KEYFRAME_MOTION,
    PIPELINE_VEO,
    build_groq_script_provider,
    build_image_provider,
    build_script_provider,
    build_tts_provider,
    build_video_pipeline_kind,
    generation_preflight_issues,
)
from app.ai.keyframes import (
    KeyframeService,
    build_keyframe_prompt,
    build_style_board_prompt,
    prompt_hash,
)
from app.ai.scriptwriter import ScriptResult, ScriptService
from app.ai.topics import ScoredTopic, TopicDiscoveryService
from app.ai.tts import TTSService
from app.ai.veo import JobOperationStore, OperationStore, VeoPollResult, VeoService
from app.config import ModelConfig, get_model_config, get_settings
from app.costs import CostLedger
from app.db import SessionLocal, engine
from app.errors import (
    AppError,
    Conflict,
    CostCapExceeded,
    NotFound,
    PolicyBlocked,
    ProviderQuotaExhausted,
    UpstreamError,
    ValidationFailed,
    error_envelope,
)
from app.media import captions, motion
from app.media import ffmpeg as ff
from app.media.probe import ProbeResult, ffprobe_cmd, parse_ffprobe_json
from app.media.qc import (
    BlackInterval,
    FreezeInterval,
    blackdetect_cmd,
    evaluate_master,
    freezedetect_cmd,
    parse_blackdetect,
    parse_freezedetect,
    qc_report_passed,
)
from app.media.runner import FFmpegError, Runner, SubprocessRunner, check
from app.models import (
    Asset,
    AuditEvent,
    ConnectedAccount,
    Creative,
    Job,
    PublishAttempt,
    PublishTarget,
    Rendition,
    Scene,
    ScriptVersion,
    Source,
    utcnow,
)
from app.publishing.base import PublishContext, Publisher, PublishNeedsAction
from app.publishing.bundle import OA_MANAGER_DEEP_LINK, build_bundle
from app.publishing.crypto import (
    decrypt_credentials,
    decrypt_publish_checkpoint,
    encrypt_credentials,
    encrypt_publish_checkpoint,
)
from app.publishing.ledger import DbCostLedger
from app.publishing.registry import create_publisher
from app.schemas.videoplan import Disclosure, ScenePlan, VideoPlan
from app.states import (
    Capability,
    CreativeState,
    InvalidTransition,
    JobStatus,
    PublishTargetStatus,
    advance,
)
from app.storage import AssetStore, find_asset, get_asset_store, store_asset
from app.workers.celery_app import celery_app
from app.workers.dispatch import dispatch_job
from app.workers.execution import execution_lock

logger = logging.getLogger("videoai.tasks")

PublisherFactory = Callable[[str, dict[str, Any]], Publisher]

_IN_FLIGHT_TARGET_STATUSES = frozenset(
    {
        PublishTargetStatus.PENDING.value,
        PublishTargetStatus.VALIDATED.value,
        PublishTargetStatus.UPLOADING.value,
    }
)
_SUCCESS_TARGET_STATUSES = frozenset(
    {PublishTargetStatus.PUBLISHED.value, PublishTargetStatus.SCHEDULED_REMOTE.value}
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _exception_envelope(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, AppError):
        return error_envelope(exc)
    if isinstance(exc, FFmpegError):
        program = exc.argv[0] if exc.argv else "ffmpeg"
        return error_envelope(
            UpstreamError(
                "media processing command failed; the render can be retried safely",
                code="media_processing_failed",
                details={
                    "program": program,
                    "returncode": exc.returncode,
                },
            )
        )
    if isinstance(exc, TimeoutExpired):
        command = exc.cmd
        program = (
            str(command[0])
            if isinstance(command, list | tuple) and command
            else "media_command"
        )
        return error_envelope(
            UpstreamError(
                "media processing command timed out; the render can be retried safely",
                code="media_processing_timeout",
                details={"program": program},
            )
        )
    if isinstance(exc, PydanticValidationError):
        return error_envelope(
            ValidationFailed(
                "provider output failed schema validation",
                details={
                    "errors": exc.errors(include_url=False, include_input=False),
                },
            )
        )
    return {
        "code": "internal_error",
        # Provider/SDK exception strings can contain request payloads or credentials.
        # Keep the operator-visible Job envelope useful without persisting secrets.
        "message": "internal worker error",
        "retryable": False,
        "details": {
            "exception_type": type(exc).__name__,
            # Exception str for non-AppError types is already provider-scoped
            # (our own raise sites); full tracebacks stay in worker logs.
            "exception_detail": f"{exc}"[:300],
        },
        "correlation_id": str(uuid.uuid4()),
    }


def _get_job(db: Session, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise NotFound(f"job {job_id} not found")
    return job


def _job_start(db: Session, job: Job) -> dict[str, Any] | None:
    """Mark a job running, or return its stored result after redelivery.

    Celery acknowledges tasks late.  If a worker commits success and dies before
    the broker receives the ACK, the same task is delivered again.  A terminal
    guard is therefore required before any provider call.
    """
    if job.status == JobStatus.SUCCEEDED.value:
        return dict(job.result or {})
    if job.status == JobStatus.CANCELLED.value:
        return {
            "error": {
                "code": "job_cancelled",
                "message": "job was cancelled",
                "retryable": False,
                "details": {"job_id": job.id},
                "correlation_id": str(uuid.uuid4()),
            }
        }
    if job.status == JobStatus.FAILED.value and not bool((job.error or {}).get("retryable")):
        return {"error": dict(job.error or {})}
    job.status = JobStatus.RUNNING.value
    job.started_at = utcnow()
    job.finished_at = None
    job.error = None
    job.attempts = (job.attempts or 0) + 1
    db.commit()
    return None


def _job_succeed(db: Session, job: Job, result: dict[str, Any]) -> None:
    job.status = JobStatus.SUCCEEDED.value
    job.result = result
    job.finished_at = utcnow()
    db.commit()


def _job_fail(db: Session, job: Job, envelope: dict[str, Any]) -> None:
    job.status = JobStatus.FAILED.value
    job.error = envelope
    job.finished_at = utcnow()


def _advance_state(creative: Creative, target: CreativeState) -> None:
    try:
        creative.state = advance(CreativeState(creative.state), target).value
    except InvalidTransition as exc:
        raise Conflict(
            str(exc),
            code="invalid_transition",
            details={"current": exc.current.value, "target": exc.target.value},
        ) from exc


def _as_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _latest_approved_script(db: Session, creative_id: str) -> ScriptVersion:
    row = (
        db.query(ScriptVersion)
        .filter(ScriptVersion.creative_id == creative_id, ScriptVersion.is_approved.is_(True))
        .order_by(ScriptVersion.version.desc())
        .first()
    )
    if row is None:
        raise NotFound(f"no approved script version for creative {creative_id}")
    return row


def _scenes_for_script(db: Session, creative_id: str, script_version_id: str) -> list[Scene]:
    return (
        db.query(Scene)
        .filter(Scene.creative_id == creative_id, Scene.script_version_id == script_version_id)
        .order_by(Scene.index)
        .all()
    )


# ---------------------------------------------------------------------------
# discover_topics
# ---------------------------------------------------------------------------


def _serialize_topic(scored: ScoredTopic) -> dict[str, Any]:
    candidate = scored.candidate
    return {
        "title": candidate.title,
        "summary": candidate.summary,
        "category": candidate.category,
        "score": scored.score,
        "freshness": scored.freshness,
        "cross_verification": scored.cross_verification,
        "audience_fit": scored.audience_fit,
        "visual_potential": scored.visual_potential,
        "domains": list(scored.domains),
        "has_official": scored.has_official,
        "window_hours": scored.window_hours,
        "sources": [
            {
                "url": s.url,
                "title": s.title,
                "publisher": s.publisher,
                "is_official": s.is_official,
                "accessed_at": s.accessed_at.isoformat(),
            }
            for s in candidate.sources
        ],
    }


def run_discover_topics(
    db: Session, job_id: str, provider: ResearchProvider | None = None
) -> dict[str, Any]:
    job = _get_job(db, job_id)
    completed = _job_start(db, job)
    if completed is not None:
        return completed
    try:
        if provider is None:
            from app.ai.gemini import GeminiResearchProvider  # lazy: optional dependency

            provider = GeminiResearchProvider()
        service = TopicDiscoveryService(provider)
        payload = job.payload or {}
        scored = service.discover(str(payload.get("brief", "")), str(payload.get("category", "")))
        result = {"topics": [_serialize_topic(s) for s in scored]}
        _job_succeed(db, job, result)
        return result
    except Exception as exc:
        db.rollback()
        envelope = _exception_envelope(exc)
        _job_fail(db, job, envelope)
        db.commit()
        return {"error": envelope}


# ---------------------------------------------------------------------------
# write_script
# ---------------------------------------------------------------------------


def _script_sources(db: Session, creative_id: str) -> list[SourceInfo]:
    rows = (
        db.query(Source)
        .filter(Source.creative_id == creative_id)
        .order_by(Source.is_official.desc(), Source.accessed_at, Source.url, Source.id)
        .all()
    )
    if len(rows) < 2:
        raise ValidationFailed(
            "script generation requires at least two persisted sources",
            details={"creative_id": creative_id, "source_count": len(rows)},
        )
    return [
        SourceInfo(
            url=row.url,
            title=row.title,
            publisher=row.publisher,
            is_official=row.is_official,
            accessed_at=_as_aware(row.accessed_at) or datetime.now(UTC),
        )
        for row in rows
    ]


def _create_script_result(
    db: Session,
    creative: Creative,
    job: Job,
    provider: ScriptProvider,
) -> ScriptResult:
    campaign_brief = creative.campaign.brief if creative.campaign is not None else ""
    service = ScriptService(provider, CostLedger(db), get_model_config())
    return service.create_plan(
        creative,
        creative.topic_title,
        _script_sources(db, creative.id),
        brief=campaign_brief,
    )


def _run_default_script_provider(
    db: Session, creative: Creative, job: Job
) -> tuple[ScriptResult, str]:
    """Use Gemini first and Groq only for a retryable Gemini failure."""
    settings = get_settings()
    if not settings.gemini_api_key and not settings.groq_api_key:
        raise ValidationFailed(
            "no script provider is configured",
            details={"required_any": ["GEMINI_API_KEY", "GROQ_API_KEY"]},
        )

    if settings.gemini_api_key:
        try:
            return (
                _create_script_result(db, creative, job, build_script_provider(settings)),
                "gemini",
            )
        except UpstreamError as exc:
            if not exc.retryable or not settings.groq_api_key:
                raise
            # ScriptService does not record projections until the complete plan
            # passes validation, but rollback protects against future changes.
            db.rollback()
            refreshed = db.get(Creative, creative.id)
            if refreshed is None:  # pragma: no cover - FK protected
                raise NotFound("creative disappeared during provider fallback") from exc
            creative = refreshed

    return (
        _create_script_result(db, creative, job, build_groq_script_provider(settings)),
        "groq",
    )


def run_write_script(
    db: Session,
    job_id: str,
    provider: ScriptProvider | None = None,
    *,
    preflight_issues: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Generate and persist one immutable VideoPlan version for a selected topic."""
    job = _get_job(db, job_id)
    completed = _job_start(db, job)
    if completed is not None:
        return completed

    try:
        if job.kind != "script":
            raise ValidationFailed(f"job {job.id} has kind {job.kind!r}; expected 'script'")
        if job.creative_id is None:
            raise ValidationFailed(f"job {job.id} has no creative_id")
        creative = db.get(Creative, job.creative_id)
        if creative is None:
            raise NotFound(f"creative {job.creative_id} not found")
        if CreativeState(creative.state) != CreativeState.RESEARCHED:
            raise Conflict(
                f"creative {creative.id} is in {creative.state}; scripting requires RESEARCHED",
                code="invalid_transition",
                details={"current": creative.state, "target": CreativeState.SCRIPT_READY.value},
            )

        if provider is None:
            script_result, provider_name = _run_default_script_provider(db, creative, job)
        else:
            script_result = _create_script_result(db, creative, job, provider)
            provider_name = type(provider).__name__

        max_version = db.execute(
            select(func.coalesce(func.max(ScriptVersion.version), 0)).where(
                ScriptVersion.creative_id == creative.id
            )
        ).scalar_one()
        version = ScriptVersion(
            creative_id=creative.id,
            version=int(max_version) + 1,
            video_plan=script_result.plan_dict,
        )
        db.add(version)
        _advance_state(creative, CreativeState.SCRIPT_READY)
        db.flush()

        issues_payload = [issue.model_dump(mode="json") for issue in script_result.issues]
        blocking_payload = [issue.model_dump(mode="json") for issue in script_result.blocking]
        preflight = generation_preflight_issues() if preflight_issues is None else preflight_issues
        auto_approved = (
            creative.mode == "auto" and script_result.auto_mode_allowed and not preflight
        )
        generate_job: Job | None = None
        if auto_approved:
            _advance_state(creative, CreativeState.SCRIPT_APPROVED)
            version.is_approved = True
            version.approved_at = datetime.now(UTC)
            db.add(
                AuditEvent(
                    actor_kind="system",
                    action="script_auto_approved",
                    entity_type="script_version",
                    entity_id=version.id,
                    data={"job_id": job.id, "version": version.version},
                )
            )
            _advance_state(creative, CreativeState.GENERATING)
            generate_job = Job(
                kind="generate",
                queue="ai",
                creative_id=creative.id,
                payload={"scene_ids": [], "parent_job_id": job.id},
                idempotency_key=f"auto-script:{job.id}",
            )
            db.add(generate_job)
            db.flush()
            db.add(
                AuditEvent(
                    actor_kind="system",
                    action="generation_requested",
                    entity_type="creative",
                    entity_id=creative.id,
                    data={"script_job_id": job.id, "generate_job_id": generate_job.id},
                )
            )

        db.add(
            AuditEvent(
                actor_kind="system",
                action="script_generated",
                entity_type="script_version",
                entity_id=version.id,
                data={
                    "job_id": job.id,
                    "provider": provider_name,
                    "version": version.version,
                    "issues": issues_payload,
                    "auto_mode_allowed": script_result.auto_mode_allowed,
                },
            )
        )
        result = {
            "creative_id": creative.id,
            "creative_state": creative.state,
            "script_version_id": version.id,
            "version": version.version,
            "issues": issues_payload,
            "blocking_issues": blocking_payload,
            "auto_mode_allowed": script_result.auto_mode_allowed,
            "auto_approved": auto_approved,
            "generate_job_id": generate_job.id if generate_job is not None else None,
            "generation_preflight_issues": preflight,
            "expected_cost_usd": script_result.expected_cost_usd,
            "shorten_attempts": script_result.shorten_attempts,
            "provider": provider_name,
        }
        _job_succeed(db, job, result)
        if generate_job is not None:
            dispatch_job(db, generate_job.id)
        return result
    except Exception as exc:
        db.rollback()
        envelope = _exception_envelope(exc)
        job = _get_job(db, job_id)
        _job_fail(db, job, envelope)
        if job.creative_id:
            creative = db.get(Creative, job.creative_id)
            if creative is not None:
                creative.last_error = envelope
        db.commit()
        return {"error": envelope}


# ---------------------------------------------------------------------------
# generate_creative
# ---------------------------------------------------------------------------


@dataclass
class GenerationDeps:
    """Injectable collaborators for the generation pipeline."""

    image_provider: ImageProvider
    video_provider: VideoProvider
    tts_provider: TTSProvider
    asset_store: AssetStore
    operation_store: OperationStore | None = None  # default: DB-backed JobOperationStore
    model_config: ModelConfig = field(default_factory=get_model_config)
    poll_interval_seconds: float = 10.0
    max_polls: int = 90
    sleep: Callable[[float], None] = time.sleep
    # Free-tier wiring (PLAN.md "Chế độ miễn phí"): per-locale TTS providers
    # (Groq is EN-only, edge-tts covers VI) and the video pipeline kind.
    # "keyframe_motion" skips Veo entirely — motion clips are cut from the
    # keyframes at render time. Defaults keep the classic Veo path.
    tts_providers: dict[str, TTSProvider] | None = None
    # locale -> (billing model id, USD/character). Free providers use 0.0.
    tts_accounting: dict[str, tuple[str | None, float]] | None = None
    video_pipeline: str = PIPELINE_VEO
    # When the paid video provider's quota is definitively exhausted, scenes
    # fall back to this pipeline (render-stage Ken Burns) instead of failing
    # the whole creative. Empty string disables the fallback.
    video_fallback: str = PIPELINE_KEYFRAME_MOTION


def _default_generation_deps() -> GenerationDeps:
    """Production wiring; provider selection comes from settings via app.ai.factory."""
    from app.ai.gemini import GenAIVideoProvider

    settings = get_settings()
    preflight = generation_preflight_issues(settings)
    if preflight:
        raise PolicyBlocked(
            "media generation providers are not ready",
            code="generation_preflight_failed",
            details={"issues": preflight},
        )
    cfg = get_model_config()
    tts_en = build_tts_provider("en", settings, cfg)
    tts_vi = build_tts_provider("vi", settings, cfg)

    def accounting(locale: str, choice: str) -> tuple[str | None, float]:
        if choice == "google":
            voice = cfg.tts_voice_vi if locale == "vi" else cfg.tts_voice_en
            return voice, cfg.tts_usd_per_million_chars / 1_000_000.0
        if choice == "groq":
            return cfg.groq_tts_model_en, 0.0
        return None, 0.0  # edge-tts: SceneAudio records the returned voice

    if settings.video_provider == "wan":
        from app.ai.wan import WanVideoProvider  # noqa: PLC0415 — lazy

        video_provider: VideoProvider = WanVideoProvider(api_key=settings.wan_api_key)
    else:
        video_provider = GenAIVideoProvider(api_key=settings.gemini_api_key)

    return GenerationDeps(
        image_provider=build_image_provider(settings),
        video_provider=video_provider,
        tts_provider=tts_en,
        asset_store=get_asset_store(),
        model_config=cfg,
        tts_providers={"en": tts_en, "vi": tts_vi},
        tts_accounting={
            "en": accounting("en", settings.tts_provider_en),
            "vi": accounting("vi", settings.tts_provider_vi),
        },
        video_pipeline=build_video_pipeline_kind(settings),
        video_fallback=settings.video_fallback_provider,
    )


def _ensure_scene_rows(
    db: Session, creative: Creative, script: ScriptVersion, plan: VideoPlan
) -> list[Scene]:
    existing = {s.index: s for s in _scenes_for_script(db, creative.id, script.id)}
    for scene_plan in plan.scenes:
        if scene_plan.index in existing:
            continue
        row = Scene(
            creative_id=creative.id,
            script_version_id=script.id,
            index=scene_plan.index,
            duration_seconds=scene_plan.duration_seconds,
            keyframe_prompt_en=scene_plan.keyframe_prompt_en,
            visual_prompt_en=scene_plan.visual_prompt_en,
            negative_prompt_en=scene_plan.negative_prompt_en,
            continuity_note=scene_plan.continuity_note,
            fact_ids=list(scene_plan.fact_ids),
            is_hero=scene_plan.is_hero,
        )
        db.add(row)
        existing[scene_plan.index] = row
    db.commit()
    return [existing[i] for i in sorted(existing)]


def _ensure_keyframes(
    db: Session,
    creative: Creative,
    plan: VideoPlan,
    scenes: list[Scene],
    ledger: CostLedger,
    deps: GenerationDeps,
    counters: dict[str, int],
    force_scene_ids: frozenset[str] = frozenset(),
) -> dict[int, bytes]:
    """Style board + one keyframe per scene; Asset existence short-circuits providers."""
    cfg = deps.model_config
    service = KeyframeService(deps.image_provider, ledger, cfg)

    board_hash = prompt_hash(build_style_board_prompt(plan), cfg.gemini_image_model)
    if find_asset(db, creative.id, "styleboard", prompt_hash=board_hash) is None:
        image = service.generate_style_board(creative, plan)
        store_asset(
            db,
            deps.asset_store,
            creative_id=creative.id,
            kind="styleboard",
            data=image.image_bytes,
            filename=f"styleboard_{image.prompt_hash[:12]}.png",
            model_id=image.model_id,
            prompt_hash=image.prompt_hash,
            cost_usd=image.cost_usd,
        )
        counters["images_generated"] += 1
        db.commit()
    else:
        counters["images_reused"] += 1

    plan_by_index = {s.index: s for s in plan.scenes}
    keyframe_bytes: dict[int, bytes] = {}
    for scene in scenes:
        scene_plan = plan_by_index[scene.index]
        khash = prompt_hash(build_keyframe_prompt(plan, scene_plan), cfg.gemini_image_model)
        existing = find_asset(db, creative.id, "keyframe", scene_id=scene.id, prompt_hash=khash)
        if existing is not None and scene.id not in force_scene_ids:
            keyframe_bytes[scene.index] = deps.asset_store.get_bytes(existing.storage_key)
            counters["images_reused"] += 1
            continue
        image = service.generate_scene_keyframe(creative, plan, scene.index)
        runner = SubprocessRunner()
        image = dataclasses.replace(
            image, image_bytes=_image_to_portrait(runner, image.image_bytes)
        )
        luma = _image_mean_luma(runner, image.image_bytes)
        if luma is not None and luma < KEYFRAME_MIN_MEAN_LUMA:
            # A near-black keyframe animates into a near-black clip that the
            # master's black-frame QC will (correctly) reject. One provider
            # retry with a nudged prompt; if it is still dark, keep it and let
            # QC arbitrate — content choice stays with the generator.
            logger.warning(
                "keyframe scene %s too dark (mean luma %.1f < %.1f); retrying image",
                scene.index,
                luma,
                KEYFRAME_MIN_MEAN_LUMA,
            )
            image = service.generate_scene_keyframe(
                creative,
                plan,
                scene.index,
                brightness_hint="bright well-lit daylight scene, high-key lighting, clear visible detail",
            )
            image = dataclasses.replace(
                image, image_bytes=_image_to_portrait(runner, image.image_bytes)
            )
            retry_luma = _image_mean_luma(runner, image.image_bytes)
            if retry_luma is not None and retry_luma < KEYFRAME_MIN_MEAN_LUMA:
                logger.warning(
                    "keyframe scene %s still dark after retry (mean luma %.1f); keeping",
                    scene.index,
                    retry_luma,
                )
        store_asset(
            db,
            deps.asset_store,
            creative_id=creative.id,
            kind="keyframe",
            scene_id=scene.id,
            data=image.image_bytes,
            filename=f"keyframe_s{scene.index}_{image.prompt_hash[:12]}.png",
            model_id=image.model_id,
            prompt_hash=image.prompt_hash,
            cost_usd=image.cost_usd,
        )
        keyframe_bytes[scene.index] = image.image_bytes
        counters["images_generated"] += 1
        db.commit()
    return keyframe_bytes


def _await_operation(
    veo: VeoService, creative: Creative, scene_id: str, deps: GenerationDeps
) -> VeoPollResult:
    result = veo.poll(creative, scene_id)
    polls = 1
    while result.status == OP_RUNNING and polls < deps.max_polls:
        deps.sleep(deps.poll_interval_seconds)
        result = veo.poll(creative, scene_id)
        polls += 1
    if result.status == OP_RUNNING:
        # Operation stays persisted: a restart resumes polling, never resubmits.
        raise UpstreamError(
            f"veo operation for scene {scene_id} still running after {polls} polls",
            retryable=True,
        )
    return result


def _generate_clip(
    db: Session,
    veo: VeoService,
    creative: Creative,
    scene: Scene,
    scene_plan: ScenePlan,
    keyframe: bytes | None,
    deps: GenerationDeps,
) -> VeoPollResult:
    """Submit (or resume) the scene's Veo operation; one escalated retry on failure."""
    previously_failed = scene.status == "failed"
    veo.ensure_submitted(
        creative,
        scene.id,
        scene_plan.visual_prompt_en,
        keyframe_bytes=keyframe,
        is_hero=scene.is_hero,
        previously_failed=previously_failed,
        duration_seconds=scene.duration_seconds,
    )
    scene.status = "generating"
    db.commit()
    result = _await_operation(veo, creative, scene.id, deps)
    if result.status == OP_SUCCEEDED:
        return result

    scene.status = "failed"
    scene.last_error = {"code": "veo_failed", "message": result.error or "", "attempt": 1}
    db.commit()
    # Escalate lite -> fast exactly once (VeoService cleared the failed operation).
    veo.ensure_submitted(
        creative,
        scene.id,
        scene_plan.visual_prompt_en,
        keyframe_bytes=keyframe,
        is_hero=scene.is_hero,
        previously_failed=True,
        duration_seconds=scene.duration_seconds,
    )
    result = _await_operation(veo, creative, scene.id, deps)
    if result.status == OP_SUCCEEDED:
        return result
    scene.last_error = {"code": "veo_failed", "message": result.error or "", "attempt": 2}
    db.commit()
    raise UpstreamError(
        f"veo generation failed twice for scene {scene.index}: {result.error}",
        retryable=False,
        details={"scene_id": scene.id, "scene_index": scene.index},
    )


def _ensure_clips(
    db: Session,
    creative: Creative,
    plan: VideoPlan,
    scenes: list[Scene],
    keyframes: dict[int, bytes],
    ledger: CostLedger,
    operation_store: OperationStore,
    deps: GenerationDeps,
    counters: dict[str, int],
    force_scene_ids: frozenset[str] = frozenset(),
) -> None:
    if deps.video_pipeline == "wan":
        from app.ai.wan import WanService  # noqa: PLC0415 — lazy

        veo: VeoService = WanService(
            deps.video_provider, ledger, operation_store, deps.model_config
        )
    else:
        veo = VeoService(deps.video_provider, ledger, operation_store, deps.model_config)
    # The script's coarse video projection (whole-master estimate) hands its
    # reservation duty to the exact per-scene projections below; keeping both
    # would double-reserve and falsely trip the hard cap mid-generation.
    ledger.remove_projected(creative.id, kind=veo.cost_kind, note_prefix="projected:")
    plan_by_index = {s.index: s for s in plan.scenes}
    quota_exhausted = False
    fallback_scenes = 0
    for scene in scenes:
        if (
            scene.id not in force_scene_ids
            and find_asset(db, creative.id, "clip", scene_id=scene.id) is not None
        ):
            if scene.status != "done":
                scene.status = "done"
            counters["clips_reused"] += 1
            db.commit()
            continue
        if quota_exhausted:
            # The provider already proved the quota is gone; skip straight to
            # the render-stage motion clip instead of eating N rejections.
            scene.status = "done"
            fallback_scenes += 1
            db.commit()
            continue
        try:
            result = _generate_clip(
                db, veo, creative, scene, plan_by_index[scene.index],
                keyframes.get(scene.index), deps,
            )
        except ProviderQuotaExhausted as exc:
            if not deps.video_fallback:
                raise
            quota_exhausted = True
            fallback_scenes += 1
            logger.warning(
                "video provider quota exhausted on scene %s; falling back to %s "
                "for this and remaining scenes (%s)",
                scene.index,
                deps.video_fallback,
                exc.message,
            )
            db.add(
                AuditEvent(
                    actor_kind="system",
                    action="video_provider_quota_fallback",
                    entity_type="creative",
                    entity_id=creative.id,
                    data={
                        "scene_index": scene.index,
                        "fallback": deps.video_fallback,
                        "provider_error": exc.message,
                        "details": exc.details,
                    },
                )
            )
            scene.status = "done"
            db.commit()
            continue
        if not result.video_bytes:
            raise UpstreamError(
                f"veo operation for scene {scene.id} completed without downloadable bytes",
                retryable=True,
                details={"scene_id": scene.id},
            )
        store_asset(
            db,
            deps.asset_store,
            creative_id=creative.id,
            kind="clip",
            scene_id=scene.id,
            data=result.video_bytes,
            filename=(
                f"clip_s{scene.index}_"
                f"{prompt_hash(result.operation_name, result.model_id)[:12]}.mp4"
            ),
            model_id=result.model_id,
            prompt_hash=prompt_hash(plan_by_index[scene.index].visual_prompt_en, result.model_id),
            cost_usd=result.actual_cost_usd,
            ffprobe={"duration_seconds": result.duration_seconds},
        )
        scene.status = "done"
        scene.last_error = None
        counters["clips_generated"] += 1
        # JobOperationStore commits this together with the Asset, scene and
        # projected->actual cost handoff. Until now a crash safely re-polls.
        veo.checkpoint_success(creative, scene.id, result)
        db.commit()

    if fallback_scenes:
        counters["clips_fallback"] = fallback_scenes


def _ensure_voices(
    db: Session,
    creative: Creative,
    plan: VideoPlan,
    scenes: list[Scene],
    ledger: CostLedger,
    deps: GenerationDeps,
    counters: dict[str, int],
) -> None:
    for locale in ("vi", "en"):
        content = plan.locales.get(locale)
        if content is None:
            continue
        provider = (deps.tts_providers or {}).get(locale, deps.tts_provider)
        accounting = (deps.tts_accounting or {}).get(locale)
        if accounting is None:
            tts = TTSService(provider, ledger, deps.model_config)
        else:
            model_id, unit_price_usd = accounting
            tts = TTSService(
                provider,
                ledger,
                deps.model_config,
                model_id=model_id,
                unit_price_usd=unit_price_usd,
            )
        for scene in scenes:
            existing = find_asset(db, creative.id, "voice", scene_id=scene.id, locale=locale)
            if existing is not None:
                counters["voices_reused"] += 1
                continue
            audio = tts.synthesize_scene(creative, content.narration[scene.index], locale)
            ext = "wav" if audio.audio_mime_type == "audio/wav" else "mp3"
            store_asset(
                db,
                deps.asset_store,
                creative_id=creative.id,
                kind="voice",
                scene_id=scene.id,
                locale=locale,
                data=audio.audio_bytes,
                filename=f"voice_{locale}_s{scene.index}_{uuid.uuid4().hex[:8]}.{ext}",
                model_id=audio.voice,
                cost_usd=audio.cost_usd,
                ffprobe={
                    "duration_seconds": audio.duration_seconds,
                    "timepoints": [
                        {"mark": t.mark_name, "seconds": t.seconds} for t in audio.timepoints
                    ],
                    "shorten_needed": audio.shorten_needed,
                    "atempo_factor": audio.atempo_factor,
                },
            )
            counters["voices_generated"] += 1
            db.commit()


def _ensure_renditions(db: Session, creative: Creative, plan: VideoPlan) -> list[Rendition]:
    rows: list[Rendition] = []
    for locale, content in plan.locales.items():
        rendition = (
            db.query(Rendition)
            .filter(Rendition.creative_id == creative.id, Rendition.locale == locale)
            .one_or_none()
        )
        if rendition is None:
            rendition = Rendition(
                creative_id=creative.id,
                locale=locale,
                title=content.title,
                description=content.description,
                hashtags=list(content.hashtags),
            )
            db.add(rendition)
            db.flush()
        rows.append(rendition)
    db.commit()
    return rows


def _spawn_render_jobs(
    db: Session,
    creative: Creative,
    renditions: list[Rendition],
    *,
    force_new_attempt: bool = False,
) -> list[Job]:
    jobs: list[Job] = []
    for rendition in renditions:
        base_key = f"render:{creative.id}:{rendition.id}"
        existing_jobs = (
            db.query(Job)
            .filter(
                Job.kind == "render",
                Job.creative_id == creative.id,
                Job.idempotency_key.like(f"{base_key}%"),
            )
            .order_by(Job.created_at.desc(), Job.id.desc())
            .all()
        )
        latest = existing_jobs[0] if existing_jobs else None
        reuse_latest = not force_new_attempt and latest is not None and (
            latest.status in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}
            or (
                latest.status == JobStatus.FAILED.value
                and bool((latest.error or {}).get("retryable"))
            )
            or (
                latest.status == JobStatus.SUCCEEDED.value
                and qc_report_passed(rendition.qc_report)
            )
        )
        if reuse_latest:
            assert latest is not None
            job = latest
        else:
            attempt_no = len(existing_jobs) + 1
            idempotency_key = (
                base_key if attempt_no == 1 else f"{base_key}:attempt:{attempt_no}"
            )
            job = Job(
                kind="render",
                queue="render",
                creative_id=creative.id,
                payload={"rendition_id": rendition.id},
                idempotency_key=idempotency_key,
            )
            db.add(job)
            db.flush()
        jobs.append(job)
    return jobs


def _requested_scene_ids(job: Job, scenes: list[Scene]) -> frozenset[str]:
    """Validate an operator-requested targeted regeneration against this script."""
    raw = (job.payload or {}).get("scene_ids", [])
    if raw is None:
        raw = []
    if not isinstance(raw, list) or any(not isinstance(scene_id, str) for scene_id in raw):
        raise ValidationFailed(
            "generate job scene_ids must be a list of scene IDs",
            details={"job_id": job.id},
        )
    requested = frozenset(raw)
    known = {scene.id for scene in scenes}
    unknown = sorted(requested - known)
    if unknown:
        raise ValidationFailed(
            "generate job contains scenes outside the approved script",
            details={"job_id": job.id, "unknown_scene_ids": unknown},
        )
    return requested


def _invalidate_renditions_after_visual_change(renditions: list[Rendition]) -> None:
    """A new visual invalidates every locale that shares the visual master."""
    for rendition in renditions:
        rendition.master_asset_id = None
        rendition.thumbnail_asset_id = None
        rendition.srt_asset_id = None
        rendition.qc_report = None
        rendition.is_approved = False
        rendition.approved_by = None
        rendition.approved_at = None


def _retire_superseded_render_jobs(db: Session, creative_id: str) -> None:
    """Fence render jobs that were created for the previous visual assets."""
    rows = (
        db.query(Job)
        .filter(Job.creative_id == creative_id, Job.kind == "render")
        .all()
    )
    running = [row.id for row in rows if row.status == JobStatus.RUNNING.value]
    if running:
        raise Conflict(
            "targeted regeneration cannot start while a rendition is rendering",
            code="render_in_progress",
            retryable=True,
            details={"render_job_ids": running},
        )
    for row in rows:
        retryable_failed = row.status == JobStatus.FAILED.value and bool(
            (row.error or {}).get("retryable")
        )
        if row.status == JobStatus.QUEUED.value or retryable_failed:
            row.status = JobStatus.CANCELLED.value
            row.finished_at = utcnow()
    db.commit()


def run_generate_creative(
    db: Session, job_id: str, deps: GenerationDeps | None = None
) -> dict[str, Any]:
    job = _get_job(db, job_id)
    completed = _job_start(db, job)
    if completed is not None:
        return completed
    if job.creative_id is None:
        raise ValidationFailed(f"job {job_id} has no creative_id")
    creative = db.get(Creative, job.creative_id)
    if creative is None:
        raise NotFound(f"creative {job.creative_id} not found")

    try:
        if deps is None:
            deps = _default_generation_deps()
        operation_store = deps.operation_store or JobOperationStore(db)
        state = CreativeState(creative.state)
        if state == CreativeState.SCRIPT_APPROVED:
            _advance_state(creative, CreativeState.GENERATING)
            db.commit()
        elif state != CreativeState.GENERATING:
            raise Conflict(
                f"creative {creative.id} is in {state.value}; generation requires "
                "SCRIPT_APPROVED or GENERATING",
                code="invalid_transition",
            )

        script = _latest_approved_script(db, creative.id)
        plan = VideoPlan.model_validate(script.video_plan)
        scenes = _ensure_scene_rows(db, creative, script, plan)
        force_scene_ids = _requested_scene_ids(job, scenes)
        if force_scene_ids:
            _retire_superseded_render_jobs(db, creative.id)
        for scene in scenes:
            if scene.id in force_scene_ids:
                scene.status = "pending"
                scene.last_error = None
        if force_scene_ids:
            db.commit()
        ledger = CostLedger(db)
        counters = {
            "images_generated": 0,
            "images_reused": 0,
            "clips_generated": 0,
            "clips_reused": 0,
            "voices_generated": 0,
            "voices_reused": 0,
        }

        keyframes = _ensure_keyframes(
            db,
            creative,
            plan,
            scenes,
            ledger,
            deps,
            counters,
            force_scene_ids,
        )
        if deps.video_pipeline == PIPELINE_KEYFRAME_MOTION:
            # Free tier: Veo has no free tier — zero submits, zero veo cost.
            # Ken Burns motion clips are cut from these keyframes in the render
            # stage (app.media.motion via the injectable Runner, resume-safe on
            # the motion_clip asset); the scene's generate-stage work is done
            # once its keyframe asset exists.
            for scene in scenes:
                if scene.status != "done":
                    scene.status = "done"
            db.commit()
        else:
            _ensure_clips(
                db,
                creative,
                plan,
                scenes,
                keyframes,
                ledger,
                operation_store,
                deps,
                counters,
                force_scene_ids,
            )
        _ensure_voices(db, creative, plan, scenes, ledger, deps, counters)
        renditions = _ensure_renditions(db, creative, plan)
        if force_scene_ids:
            _invalidate_renditions_after_visual_change(renditions)

        _advance_state(creative, CreativeState.QC_REQUIRED)
        render_jobs = _spawn_render_jobs(
            db,
            creative,
            renditions,
            force_new_attempt=bool(force_scene_ids),
        )

        result = {
            **counters,
            "creative_id": creative.id,
            "state": creative.state,
            "rendition_ids": [r.id for r in renditions],
            "render_job_ids": [render_job.id for render_job in render_jobs],
            "regenerated_scene_ids": sorted(force_scene_ids),
        }
        # Commit the parent success, QC_REQUIRED state and child outbox rows as
        # one unit.  A crash after this point is repaired by the dispatch sweep.
        _job_succeed(db, job, result)
        for render_job in render_jobs:
            dispatch_job(db, render_job.id)
        return result
    except Exception as exc:
        db.rollback()
        logger.exception("generate job %s failed", job_id)
        envelope = _exception_envelope(exc)
        operator_reconciliation_required = isinstance(exc, PolicyBlocked) and exc.code in {
            "veo_submission_ambiguous",
            "video_submission_provider_mismatch",
            "video_submission_projection_mismatch",
        }
        if operator_reconciliation_required:
            _advance_state(creative, CreativeState.NEEDS_ACTION)
            creative.last_error = envelope
            db.add(
                AuditEvent(
                    actor_kind="system",
                    action="generation_needs_action",
                    entity_type="creative",
                    entity_id=creative.id,
                    data=envelope,
                )
            )
        elif envelope.get("retryable"):
            # Transient (e.g. Veo still running after the poll budget): leave the
            # creative in GENERATING so a re-run resumes from the persisted state.
            creative.last_error = envelope
        else:
            # No FAILED edge from the current state: keep the state as-is.
            with contextlib.suppress(Conflict):
                _advance_state(creative, CreativeState.FAILED)
            creative.last_error = envelope
            db.add(
                AuditEvent(
                    actor_kind="system",
                    action=(
                        "generation_aborted_cost_cap"
                        if isinstance(exc, CostCapExceeded)
                        else "generation_failed"
                    ),
                    entity_type="creative",
                    entity_id=creative.id,
                    data=envelope,
                )
            )
        _job_fail(db, job, envelope)
        db.commit()
        return {"error": envelope, "creative_id": creative.id, "state": creative.state}


# ---------------------------------------------------------------------------
# render_rendition
# ---------------------------------------------------------------------------


def _run_cmd(runner: Runner, argv: list[str]) -> Any:
    return check(runner.run(argv))


# Keyframes darker than this (0-255 luma, area average) animate into clips that
# trip the master's black-frame QC — reject and retry the image provider once.
KEYFRAME_MIN_MEAN_LUMA = 30.0


def _image_to_portrait(runner: Runner, image_bytes: bytes) -> bytes:
    """Force the 9:16 master frame on a keyframe of any source aspect.

    Free image backends return landscape/square photos; video providers keep
    the input aspect, so an un-normalized keyframe would later be center-cropped
    by the clip normalizer, discarding both sides of the composition.
    """
    with (
        NamedTemporaryFile(suffix=".in.img", delete=False) as fin,
        NamedTemporaryFile(suffix=".png", delete=False) as fout,
    ):
        fin.write(image_bytes)
        in_path, out_path = fin.name, fout.name
    try:
        result = runner.run(ff.portrait_crop_image_cmd(in_path, out_path))
        if result.returncode != 0:
            stderr = result.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            logger.warning(
                "keyframe portrait crop failed (%s); keeping original aspect", stderr[-200:]
            )
            return image_bytes
        cropped = Path(out_path).read_bytes()
        return cropped if cropped else image_bytes
    finally:
        Path(in_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


def _image_mean_luma(runner: Runner, image_bytes: bytes) -> float | None:
    """Area-average brightness via ffmpeg's 1x1 gray downsample (no Pillow)."""
    with NamedTemporaryFile(suffix=".img", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        result = runner.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                tmp_path,
                "-vf",
                "scale=1:1,format=gray",
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-",
            ]
        )
        if result.returncode != 0 or not result.stdout:
            return None
        # SubprocessRunner decodes stdout to str; both encodings carry one
        # raw gray byte for the 1x1 frame.
        first = result.stdout[0]
        return float(first if isinstance(first, int) else ord(first))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _measure_loudnorm(runner: Runner, path: str) -> ff.LoudnormMeasurement | None:
    result = runner.run(ff.loudnorm_measure_cmd(path))
    if result.returncode != 0:
        return None
    try:
        return ff.parse_loudnorm_json(result.stderr)
    except ValueError:
        return None


def _probe_file(runner: Runner, path: str) -> ProbeResult | None:
    result = runner.run(ffprobe_cmd(path))
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return parse_ffprobe_json(result.stdout)
    except (ValueError, KeyError, TypeError):
        return None


def _detect_black_frames(runner: Runner, path: str) -> list[BlackInterval] | None:
    result = runner.run(blackdetect_cmd(path))
    if result.returncode != 0:
        return None
    return parse_blackdetect(result.stderr)


def _detect_freeze_frames(runner: Runner, path: str) -> list[FreezeInterval] | None:
    result = runner.run(freezedetect_cmd(path))
    if result.returncode != 0:
        return None
    return parse_freezedetect(result.stderr)


def _ensure_motion_clip(
    db: Session,
    creative: Creative,
    scene: Scene,
    runner: Runner,
    store: AssetStore,
    work: Path,
) -> tuple[Asset, int]:
    """Keyframe-motion pipeline: reuse or render the scene's Ken Burns clip.

    Returns ``(asset, commands_run)``. Same resume-safety pattern as every
    generate step: ``find_asset`` first, so a restart never re-renders a
    motion clip that already exists.
    """
    keyframe = find_asset(db, creative.id, "keyframe", scene_id=scene.id)
    if keyframe is None:
        raise ValidationFailed(f"scene {scene.index} has no clip or keyframe asset; generate first")
    direction = motion.direction_for_scene(scene.index)
    motion_hash = prompt_hash(
        f"{keyframe.sha256}:{scene.duration_seconds}:{direction}",
        "ffmpeg-kenburns-v1",
    )
    existing = find_asset(
        db,
        creative.id,
        "motion_clip",
        scene_id=scene.id,
        prompt_hash=motion_hash,
    )
    if existing is not None:
        return existing, 0
    image_path = work / f"keyframe{scene.index}.png"
    image_path.write_bytes(store.get_bytes(keyframe.storage_key))
    clip_path = work / f"motion{scene.index}.mp4"
    _run_cmd(
        runner,
        motion.kenburns_clip_cmd(
            str(image_path),
            str(clip_path),
            duration_s=scene.duration_seconds,
            direction=direction,
        ),
    )
    asset = store_asset(
        db,
        store,
        creative_id=creative.id,
        kind="motion_clip",
        scene_id=scene.id,
        data=clip_path.read_bytes(),
        filename=f"motion_s{scene.index}_{uuid.uuid4().hex[:8]}.mp4",
        model_id=f"ffmpeg-kenburns:{direction}",
        prompt_hash=motion_hash,
        ffprobe={"duration_seconds": scene.duration_seconds, "direction": direction},
    )
    db.commit()
    return asset, 1


def run_render_rendition(
    db: Session,
    rendition_id: str,
    *,
    runner: Runner | None = None,
    asset_store: AssetStore | None = None,
    workdir: str | Path | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    rendition = db.get(Rendition, rendition_id)
    if rendition is None:
        raise NotFound(f"rendition {rendition_id} not found")
    creative = db.get(Creative, rendition.creative_id)
    if creative is None:
        raise NotFound(f"creative {rendition.creative_id} not found")

    if job_id is not None:
        job = _get_job(db, job_id)
    else:
        job = Job(
            kind="render",
            queue="render",
            creative_id=creative.id,
            payload={"rendition_id": rendition_id},
        )
        db.add(job)
        db.flush()
    completed = _job_start(db, job)
    if completed is not None:
        return completed

    runner = runner or SubprocessRunner()
    store = asset_store or get_asset_store()
    try:
        script = _latest_approved_script(db, creative.id)
        scenes = _scenes_for_script(db, creative.id, script.id)
        if not scenes:
            raise ValidationFailed(f"creative {creative.id} has no scenes to render")

        work = Path(workdir) if workdir else Path(mkdtemp(prefix=f"render_{rendition_id[:8]}_"))
        work.mkdir(parents=True, exist_ok=True)

        commands_run = 0
        clip_paths: list[str] = []
        voice_paths: list[str] = []
        durations: list[float] = []
        for scene in scenes:
            clip = find_asset(db, creative.id, "clip", scene_id=scene.id)
            if clip is None:
                # keyframe_motion pipeline: no Veo clip — cut (or reuse) the
                # Ken Burns motion clip rendered from the scene keyframe.
                clip, motion_commands = _ensure_motion_clip(
                    db, creative, scene, runner, store, work
                )
                commands_run += motion_commands
            voice = find_asset(db, creative.id, "voice", scene_id=scene.id, locale=rendition.locale)
            if voice is None:
                raise ValidationFailed(
                    f"scene {scene.index} has no {rendition.locale} voice asset; generate first"
                )
            clip_path = work / f"clip{scene.index}.mp4"
            clip_path.write_bytes(store.get_bytes(clip.storage_key))
            voice_path = work / f"voice_{rendition.locale}_{scene.index}.mp3"
            voice_path.write_bytes(store.get_bytes(voice.storage_key))
            clip_paths.append(str(clip_path))
            voice_paths.append(str(voice_path))
            durations.append(scene.duration_seconds)

        norm_paths: list[str] = []
        for idx, clip_path_str in enumerate(clip_paths):
            norm = work / f"norm{idx}.mp4"
            _run_cmd(runner, ff.normalize_clip_cmd(clip_path_str, str(norm)))
            commands_run += 1
            norm_paths.append(str(norm))

        visual = work / "master_visual.mp4"
        _run_cmd(runner, ff.concat_crossfade_cmd(norm_paths, str(visual), durations))
        commands_run += 1

        # Scene i starts at sum(previous durations) - i * fade (crossfade overlap).
        offsets = [
            round(sum(durations[:i]) - i * ff.CROSSFADE_SECONDS, 3) for i in range(len(durations))
        ]
        mixed = work / f"master_{rendition.locale}_mixed.mp4"
        _run_cmd(
            runner,
            ff.mix_voiceover_cmd(
                str(visual), list(zip(voice_paths, offsets, strict=True)), str(mixed)
            ),
        )
        commands_run += 1

        normalized_audio = work / f"master_{rendition.locale}_audio.mp4"
        measurement = _measure_loudnorm(runner, str(mixed))
        if measurement is not None:
            _run_cmd(runner, ff.loudnorm_apply_cmd(str(mixed), str(normalized_audio), measurement))
            commands_run += 1
        else:
            normalized_audio = mixed  # measurement unavailable (e.g. offline runner): skip pass 2

        # Generate SRT captions
        plan_dict = script.video_plan or {}
        loc_content = plan_dict.get("locales", {}).get(rendition.locale, {})
        narrations = loc_content.get("narration", [])
        master_duration = round(
            sum(durations) - (len(durations) - 1) * ff.CROSSFADE_SECONDS, 3
        )
        scene_marks = {f"s{i}": offsets[i] for i in range(len(offsets))}
        srt_events = []
        if narrations:
            try:
                srt_events = captions.events_from_timepoints(
                    narrations, scene_marks, total_duration=master_duration
                )
            except Exception as exc:  # noqa: BLE001 — captions degrade to none, render continues
                logger.warning(
                    "caption event generation failed for rendition %s locale=%s: %s",
                    rendition_id,
                    rendition.locale,
                    exc,
                )

        srt_text = captions.build_srt(srt_events) if srt_events else ""
        if srt_text:
            srt_asset = store_asset(
                db,
                store,
                creative_id=creative.id,
                kind="srt",
                locale=rendition.locale,
                data=srt_text.encode("utf-8"),
                filename=f"captions_{rendition.locale}_{uuid.uuid4().hex[:8]}.srt",
            )
            rendition.srt_asset_id = srt_asset.id

        # Burn subtitles into final master video
        final = work / f"master_{rendition.locale}.mp4"
        if srt_text:
            srt_file = work / f"captions_{rendition.locale}.srt"
            srt_file.write_text(srt_text, encoding="utf-8")
            _run_cmd(
                runner,
                ff.burn_captions_cmd(str(normalized_audio), str(srt_file), str(final)),
            )
            commands_run += 1
        else:
            final = normalized_audio

        # Extract thumbnail frame
        thumb_path = work / f"thumbnail_{rendition.locale}.jpg"
        try:
            _run_cmd(
                runner,
                ff.thumbnail_cmd(
                    str(final),
                    str(thumb_path),
                    at_seconds=1.0,
                    overlay_text=rendition.title or None,
                ),
            )
            commands_run += 1
        except Exception as exc:  # noqa: BLE001 — thumbnail is best-effort, render continues
            logger.warning(
                "thumbnail extraction failed for rendition %s locale=%s: %s",
                rendition_id,
                rendition.locale,
                exc,
            )
        if thumb_path.exists():
            thumb_asset = store_asset(
                db,
                store,
                creative_id=creative.id,
                kind="thumbnail",
                locale=rendition.locale,
                data=thumb_path.read_bytes(),
                filename=f"thumbnail_{rendition.locale}_{uuid.uuid4().hex[:8]}.jpg",
            )
            rendition.thumbnail_asset_id = thumb_asset.id

        data = final.read_bytes()
        probe_result = _probe_file(runner, str(final))
        final_measurement = _measure_loudnorm(runner, str(final))
        black_intervals = _detect_black_frames(runner, str(final))
        freeze_intervals = _detect_freeze_frames(runner, str(final))
        qc_report: dict[str, Any] | None = None
        if probe_result is not None:
            qc_report = evaluate_master(
                probe_result,
                final_measurement,
                black_intervals=black_intervals,
                freeze_intervals=freeze_intervals,
            ).to_dict()

        asset = store_asset(
            db,
            store,
            creative_id=creative.id,
            kind="master",
            locale=rendition.locale,
            data=data,
            filename=f"master_{rendition.locale}_{uuid.uuid4().hex[:8]}.mp4",
            ffprobe=probe_result.to_dict() if probe_result is not None else None,
        )
        rendition.master_asset_id = asset.id
        rendition.qc_report = qc_report
        db.flush()

        all_renditions = db.query(Rendition).filter(Rendition.creative_id == creative.id).all()
        # Fail closed: a missing/unparseable ffprobe produces no report,
        # and a malformed/failed report never promotes the creative.
        if (
            all(r.master_asset_id for r in all_renditions)
            and CreativeState(creative.state) == CreativeState.QC_REQUIRED
            and all(qc_report_passed(r.qc_report) for r in all_renditions)
        ):
            _advance_state(creative, CreativeState.READY)

        result = {
            "rendition_id": rendition.id,
            "master_asset_id": asset.id,
            "commands_run": commands_run,
            "qc_passed": None if qc_report is None else qc_report.get("passed"),
            "creative_state": creative.state,
        }
        _job_succeed(db, job, result)
        return result
    except Exception as exc:
        db.rollback()
        envelope = _exception_envelope(exc)
        _job_fail(db, job, envelope)
        db.commit()
        return {"error": envelope, "rendition_id": rendition_id}


# ---------------------------------------------------------------------------
# publish_target / publish_job
# ---------------------------------------------------------------------------


_PUBLISH_ACTIVE_PHASES = frozenset(
    {"running", "validated", "preparing", "prepared", "uploading", "uploaded", "finalizing"}
)
_PUBLISH_MUTATING_PHASES = frozenset(
    {"preparing", "prepared", "uploading", "uploaded", "finalizing"}
)
MAX_PUBLISH_RECOVERY_ATTEMPTS = 4


def _active_publish_attempt(db: Session, target_id: str) -> PublishAttempt | None:
    return (
        db.query(PublishAttempt)
        .filter(
            PublishAttempt.target_id == target_id,
            PublishAttempt.finished_at.is_(None),
            PublishAttempt.status.in_(_PUBLISH_ACTIVE_PHASES),
        )
        .order_by(PublishAttempt.attempt_no.desc(), PublishAttempt.started_at.desc())
        .first()
    )


def _publish_attempt_checkpoint(
    db: Session,
    target: PublishTarget,
    attempt: PublishAttempt,
    phase: str,
    session: dict[str, Any] | None = None,
) -> None:
    """Durably fence the next remote mutation and encrypt opaque session data."""
    if phase not in _PUBLISH_ACTIVE_PHASES:
        raise ValueError(f"unsupported publish phase {phase}")
    checkpoint: dict[str, Any] = {
        "target_id": target.id,
        "platform": target.platform,
        "phase": phase,
    }
    if session is not None:
        checkpoint["session"] = session
    attempt.status = phase
    # The JSON column contains only an authenticated ciphertext. Provider
    # upload URLs, tokens and opaque identifiers never reach plaintext DB JSON.
    raw_recovery_count = (attempt.response or {}).get("recovery_count", 0)
    recovery_count = raw_recovery_count if isinstance(raw_recovery_count, int) else 0
    attempt.response = {
        "encrypted_checkpoint": encrypt_publish_checkpoint(checkpoint),
        "recovery_count": max(recovery_count, 0),
    }
    if phase in _PUBLISH_MUTATING_PHASES:
        target.status = PublishTargetStatus.UPLOADING.value
    elif phase == "validated":
        target.status = PublishTargetStatus.VALIDATED.value
    target.claimed_at = utcnow()
    db.commit()


def _claim_publish_recovery(db: Session, attempt: PublishAttempt) -> None:
    """Count resumed deliveries and stop an unreconcilable hot loop."""
    response = dict(attempt.response or {})
    raw_count = response.get("recovery_count", 0)
    count = raw_count if isinstance(raw_count, int) and raw_count >= 0 else 0
    count += 1
    response["recovery_count"] = count
    attempt.response = response
    db.commit()
    if count > MAX_PUBLISH_RECOVERY_ATTEMPTS:
        raise PublishNeedsAction(
            "publish recovery attempt limit exceeded",
            details={"reason": "publish_recovery_exhausted"},
        )


def _load_publish_session(
    target: PublishTarget,
    attempt: PublishAttempt,
) -> dict[str, Any]:
    """Authenticate a persisted session and bind it to this target/phase."""
    token = (attempt.response or {}).get("encrypted_checkpoint")
    if not isinstance(token, str) or not token:
        raise PublishNeedsAction(
            "publish recovery checkpoint is missing",
            details={"reason": "publish_checkpoint_missing"},
        )
    try:
        checkpoint = decrypt_publish_checkpoint(token)
    except (TypeError, ValueError) as exc:
        raise PublishNeedsAction(
            "publish recovery checkpoint is invalid",
            details={"reason": "publish_checkpoint_invalid"},
        ) from exc
    if (
        checkpoint.get("target_id") != target.id
        or checkpoint.get("platform") != target.platform
        or checkpoint.get("phase") != attempt.status
    ):
        raise PublishNeedsAction(
            "publish recovery checkpoint does not match its target",
            details={"reason": "publish_checkpoint_mismatch"},
        )
    session = checkpoint.get("session")
    if not isinstance(session, dict):
        raise PublishNeedsAction(
            "publish recovery checkpoint has no session",
            details={"reason": "publish_checkpoint_session_missing"},
        )
    return dict(session)


def _ambiguous_publish_phase(phase: str) -> PublishNeedsAction:
    return PublishNeedsAction(
        f"publish worker stopped during {phase}; remote outcome needs reconciliation",
        details={
            "reason": "remote_outcome_unknown",
            "operation": phase,
            "operator_action_required": True,
        },
    )


def _publish_disclosure(db: Session, creative: Creative) -> Disclosure:
    try:
        script = _latest_approved_script(db, creative.id)
    except NotFound:
        return Disclosure()
    raw = (script.video_plan or {}).get("disclosure")
    if not isinstance(raw, dict):
        return Disclosure()
    try:
        return Disclosure.model_validate(raw)
    except ValueError:
        return Disclosure()


def _build_publish_context(
    db: Session,
    target: PublishTarget,
    rendition: Rendition,
    creative: Creative,
    asset_store: AssetStore | None,
    workdir: str | Path | None,
    *,
    now: datetime | None = None,
) -> PublishContext:
    file_path: str | None = None
    file_url: str | None = None
    thumbnail_path: str | None = None
    if rendition.master_asset_id and asset_store is not None:
        asset = db.get(Asset, rendition.master_asset_id)
        if asset is not None:
            file_url = asset_store.get_url(asset.storage_key)
            work = Path(workdir) if workdir else Path(mkdtemp(prefix="publish_"))
            work.mkdir(parents=True, exist_ok=True)
            local = work / f"{rendition.locale}_{asset.id[:8]}.mp4"
            local.write_bytes(asset_store.get_bytes(asset.storage_key))
            file_path = str(local)
            if rendition.thumbnail_asset_id:
                thumbnail = db.get(Asset, rendition.thumbnail_asset_id)
                if thumbnail is not None:
                    suffix = Path(thumbnail.storage_key).suffix or ".jpg"
                    local_thumbnail = work / f"thumbnail_{thumbnail.id[:8]}{suffix}"
                    local_thumbnail.write_bytes(asset_store.get_bytes(thumbnail.storage_key))
                    thumbnail_path = str(local_thumbnail)
    now_dt = _as_aware(now) or datetime.now(UTC)
    scheduled_at = _as_aware(target.scheduled_at)
    # Once the internal scheduler reaches the due time (or if within native window < 10 mins),
    # publish immediately so native APIs (Facebook/YouTube) accept the upload without rejection.
    if scheduled_at is not None and (
        scheduled_at <= now_dt
        or (
            target.platform == "facebook"
            and (scheduled_at - now_dt) < timedelta(minutes=10)
        )
    ):
        scheduled_at = None

    return PublishContext(
        creative_id=creative.id,
        rendition_id=rendition.id,
        locale=rendition.locale,
        title=rendition.title,
        description=rendition.description,
        hashtags=list(rendition.hashtags or []),
        file_path=file_path,
        file_url=file_url,
        thumbnail_path=thumbnail_path,
        privacy=target.privacy,
        scheduled_at=scheduled_at,
        disclosure=_publish_disclosure(db, creative),
        extra={"cost_cap_usd": creative.cost_cap_usd, "target_id": target.id},
    )


def _target_credentials(db: Session, target: PublishTarget) -> dict[str, Any]:
    if target.connected_account_id is None:
        return {}
    account = db.get(ConnectedAccount, target.connected_account_id)
    if account is None or not account.encrypted_credentials:
        return {}
    try:
        return decrypt_credentials(account.encrypted_credentials)
    except ValueError:
        return {}


def _target_account(db: Session, target: PublishTarget) -> ConnectedAccount | None:
    if target.connected_account_id is None:
        return None
    return db.get(ConnectedAccount, target.connected_account_id)


def _has_native_schedule(account: ConnectedAccount | None) -> bool:
    return bool(
        account is not None
        and account.status == "active"
        and account.capability == Capability.SCHEDULE.value
    )


def _manual_publish_reason(account: ConnectedAccount | None) -> str | None:
    """Why the default publisher must stay offline and create a bundle."""
    if account is None:
        return "no_connected_account"
    if account.status != "active":
        return "connected_account_inactive"
    if account.capability in {
        Capability.MANUAL.value,
        Capability.BLOCKED.value,
    }:
        return f"account_capability_{account.capability.lower()}"
    if not account.encrypted_credentials:
        return "connected_account_credentials_missing"
    return None


def _persist_changed_publisher_credentials(
    db: Session,
    account: ConnectedAccount | None,
    publisher: Publisher,
    persisted_credentials: dict[str, Any],
) -> dict[str, Any]:
    """Encrypt and commit credentials changed by an adapter refresh.

    Called from ``finally`` around every publisher operation so a successful
    refresh is durable even when the retried operation or a later phase fails.
    Credential values are never copied into an error, response, or log.
    """
    current = publisher.credentials
    if account is None or current == persisted_credentials:
        return persisted_credentials
    account.encrypted_credentials = encrypt_credentials(current)
    db.commit()
    return current


def _persist_manual_bundle(
    db: Session,
    target: PublishTarget,
    rendition: Rendition,
    creative: Creative,
    ctx: PublishContext,
    store: AssetStore,
    *,
    workdir: str | Path | None,
    deep_link: str | None = None,
) -> Asset:
    root = (
        Path(workdir) / "bundles"
        if workdir is not None
        else Path(mkdtemp(prefix=f"bundle_{target.id[:8]}_"))
    )
    manifest = build_bundle(
        ctx,
        target.platform,
        root,
        zip_output=True,
        deep_link=deep_link,
    )
    zip_path = Path(str(manifest["zip_path"]))
    asset = store_asset(
        db,
        store,
        creative_id=creative.id,
        kind="bundle",
        locale=rendition.locale,
        platform=target.platform,
        data=zip_path.read_bytes(),
        filename=f"manual_{target.platform}_{target.id}_{uuid.uuid4().hex[:8]}.zip",
        model_id="manual-publish-bundle-v1",
    )
    # Bundles are operator hand-off artifacts and must survive normal cache
    # cleanup even after the creative has reached a terminal state.
    asset.pinned = True
    target.bundle_path = asset.storage_key
    return asset


def _update_creative_publish_state(db: Session, creative: Creative) -> str:
    """Final creative state from per-target outcomes; only from PUBLISHING."""
    targets = db.query(PublishTarget).filter(PublishTarget.creative_id == creative.id).all()
    if not targets or CreativeState(creative.state) != CreativeState.PUBLISHING:
        return creative.state
    statuses = [t.status for t in targets]
    if any(s in _IN_FLIGHT_TARGET_STATUSES for s in statuses):
        return creative.state  # more targets still running
    ok = [s for s in statuses if s in _SUCCESS_TARGET_STATUSES]
    if len(ok) == len(statuses):
        _advance_state(creative, CreativeState.PUBLISHED)
    elif ok:
        _advance_state(creative, CreativeState.PARTIAL)
    elif any(
        s in {PublishTargetStatus.NEEDS_ACTION.value, PublishTargetStatus.MANUAL_BUNDLE.value}
        for s in statuses
    ):
        _advance_state(creative, CreativeState.NEEDS_ACTION)
    else:
        _advance_state(creative, CreativeState.FAILED)
    return creative.state


def _run_publish_target_unlocked(
    db: Session,
    target_id: str,
    *,
    publisher_factory: PublisherFactory | None = None,
    asset_store: AssetStore | None = None,
    workdir: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    target = db.get(PublishTarget, target_id)
    if target is None:
        raise NotFound(f"publish target {target_id} not found")
    if target.status in _SUCCESS_TARGET_STATUSES or target.status in {
        PublishTargetStatus.NEEDS_ACTION.value,
        PublishTargetStatus.MANUAL_BUNDLE.value,
    }:
        return {"target_id": target.id, "status": target.status, "skipped": True}
    if target.status == PublishTargetStatus.FAILED.value and not bool(
        (target.last_error or {}).get("retryable")
    ):
        return {"target_id": target.id, "status": target.status, "skipped": True}
    rendition = db.get(Rendition, target.rendition_id)
    creative = db.get(Creative, target.creative_id)
    if rendition is None or creative is None:
        raise NotFound(f"rendition/creative missing for target {target_id}")

    if CreativeState(creative.state) in {
        CreativeState.SCHEDULED,
        CreativeState.PARTIAL,
        CreativeState.NEEDS_ACTION,
    }:
        _advance_state(creative, CreativeState.PUBLISHING)

    attempt = _active_publish_attempt(db, target.id)
    resuming_attempt = attempt is not None
    if attempt is None:
        attempt_no = (
            db.query(func.count(PublishAttempt.id))
            .filter(PublishAttempt.target_id == target.id)
            .scalar()
            or 0
        ) + 1
        attempt = PublishAttempt(target_id=target.id, attempt_no=attempt_no, status="running")
        db.add(attempt)
        db.commit()
    else:
        attempt_no = attempt.attempt_no

    store = asset_store if asset_store is not None else get_asset_store()
    account = _target_account(db, target)
    try:
        ctx = _build_publish_context(
            db,
            target,
            rendition,
            creative,
            store,
            workdir,
            now=now,
        )
        credentials = _target_credentials(db, target)
        if publisher_factory is None:
            manual_reason = _manual_publish_reason(account)
            if manual_reason is None and not credentials:
                manual_reason = "connected_account_credentials_unavailable"
            if manual_reason is not None:
                details = {"reason": manual_reason}
                if target.platform == "zalo":
                    details["deep_link"] = OA_MANAGER_DEEP_LINK
                raise PublishNeedsAction(
                    f"{target.platform} requires manual publishing",
                    details=details,
                )
            publisher = create_publisher(
                target.platform,
                credentials,
                ledger=DbCostLedger(db, creative.id),
                settings=get_settings(),
            )
        else:
            publisher = publisher_factory(target.platform, credentials)
        persisted_credentials = dict(credentials)

        if resuming_attempt and attempt.status in _PUBLISH_MUTATING_PHASES:
            _claim_publish_recovery(db, attempt)

        phase = attempt.status
        try:
            publisher.validate(ctx)
        finally:
            persisted_credentials = _persist_changed_publisher_credentials(
                db, account, publisher, persisted_credentials
            )

        session: dict[str, Any]
        result = None
        if phase in {"running", "validated"}:
            # A legacy UPLOADING row without an authenticated phase cannot be
            # distinguished from a prepare call that succeeded before a crash.
            if target.status == PublishTargetStatus.UPLOADING.value:
                raise _ambiguous_publish_phase("legacy_prepare")
            _publish_attempt_checkpoint(db, target, attempt, "validated")
            _publish_attempt_checkpoint(db, target, attempt, "preparing")
            try:
                session = publisher.prepare(ctx)
            finally:
                persisted_credentials = _persist_changed_publisher_credentials(
                    db, account, publisher, persisted_credentials
                )
            _publish_attempt_checkpoint(db, target, attempt, "prepared", session)
            phase = "prepared"
        elif phase == "preparing":
            # The process may have died after the remote session was created but
            # before its response was checkpointed. Only an adapter with a
            # deterministic, non-mutating recovery path may continue.
            try:
                recovered_prepare = publisher.recover_prepare(ctx)
            finally:
                persisted_credentials = _persist_changed_publisher_credentials(
                    db, account, publisher, persisted_credentials
                )
            if recovered_prepare is None:
                raise _ambiguous_publish_phase("prepare")
            session = recovered_prepare
            _publish_attempt_checkpoint(db, target, attempt, "prepared", session)
            phase = "prepared"
        else:
            session = _load_publish_session(target, attempt)

        if phase == "prepared":
            _publish_attempt_checkpoint(db, target, attempt, "uploading", session)
            try:
                session = publisher.upload(ctx, session)
            finally:
                persisted_credentials = _persist_changed_publisher_credentials(
                    db, account, publisher, persisted_credentials
                )
            _publish_attempt_checkpoint(db, target, attempt, "uploaded", session)
            phase = "uploaded"
        elif phase == "uploading":
            try:
                recovered_upload = publisher.recover_upload(ctx, session)
            finally:
                persisted_credentials = _persist_changed_publisher_credentials(
                    db, account, publisher, persisted_credentials
                )
            if recovered_upload is None:
                raise _ambiguous_publish_phase("upload")
            session = recovered_upload
            _publish_attempt_checkpoint(db, target, attempt, "uploaded", session)
            phase = "uploaded"

        if phase == "uploaded":
            _publish_attempt_checkpoint(db, target, attempt, "finalizing", session)
            try:
                result = publisher.finalize(ctx, session)
            finally:
                _persist_changed_publisher_credentials(
                    db, account, publisher, persisted_credentials
                )
        elif phase == "finalizing":
            try:
                result = publisher.recover_finalize(ctx, session)
            finally:
                _persist_changed_publisher_credentials(
                    db, account, publisher, persisted_credentials
                )
            if result is None:
                raise _ambiguous_publish_phase("finalize")
        elif phase not in {"uploaded", "finalizing"}:
            raise PublishNeedsAction(
                "publish recovery phase is unsupported",
                details={"reason": "publish_checkpoint_phase_invalid"},
            )

        if result is None:  # pragma: no cover - guarded by the phase branches above
            raise PublishNeedsAction(
                "publish result could not be reconciled",
                details={"reason": "remote_outcome_unknown"},
            )

        target.remote_post_id = result.remote_post_id
        target.remote_status = result.remote_status
        if ctx.scheduled_at is not None:
            target.status = PublishTargetStatus.SCHEDULED_REMOTE.value
        else:
            target.status = PublishTargetStatus.PUBLISHED.value
        target.last_error = None
        target.claimed_at = None
        attempt.status = "succeeded"
        attempt.error = None
        attempt.response = {
            "remote_post_id": result.remote_post_id,
            "remote_status": result.remote_status,
        }
    except PublishNeedsAction as exc:
        safe_details = {
            key: value
            for key, value in exc.details.items()
            if key in {"reason", "status_code", "deep_link"}
            and isinstance(value, str | int | float | bool | type(None))
        }
        try:
            bundle = _persist_manual_bundle(
                db,
                target,
                rendition,
                creative,
                ctx,
                store,
                workdir=workdir,
                deep_link=(str(exc.details["deep_link"]) if exc.details.get("deep_link") else None),
            )
        except Exception as bundle_exc:
            safe_details["bundle_error"] = type(bundle_exc).__name__
            target.status = PublishTargetStatus.NEEDS_ACTION.value
        else:
            safe_details.update(
                {
                    "bundle_asset_id": bundle.id,
                    "bundle_download": f"/api/v1/publish-targets/{target.id}/bundle",
                }
            )
            target.status = PublishTargetStatus.MANUAL_BUNDLE.value
        envelope = error_envelope(PublishNeedsAction(exc.message, details=safe_details))
        target.last_error = envelope
        target.claimed_at = None
        attempt.status = "needs_action"
        attempt.error = envelope
    except AppError as exc:
        envelope = error_envelope(exc)
        target.last_error = envelope
        attempt.error = envelope
        if exc.retryable and attempt.status in _PUBLISH_MUTATING_PHASES:
            # Do not discard the session and start a fresh remote resource.
            # The scheduler will lease this same attempt after the stale window;
            # recovery then probes the provider before doing anything else.
            target.status = PublishTargetStatus.UPLOADING.value
            target.claimed_at = utcnow()
        else:
            target.status = PublishTargetStatus.FAILED.value
            target.claimed_at = None
            attempt.status = "failed"
    except Exception as exc:
        db.rollback()
        envelope = _exception_envelope(exc)
        target.status = PublishTargetStatus.FAILED.value
        target.claimed_at = None
        target.last_error = envelope
        attempt.status = "failed"
        attempt.error = envelope

    if attempt.status not in _PUBLISH_ACTIVE_PHASES:
        attempt.finished_at = utcnow()
    db.commit()
    _update_creative_publish_state(db, creative)
    db.commit()
    return {
        "target_id": target.id,
        "status": target.status,
        "remote_post_id": target.remote_post_id,
        "creative_state": creative.state,
        "attempt_no": attempt_no,
    }


def run_publish_target(
    db: Session,
    target_id: str,
    *,
    publisher_factory: PublisherFactory | None = None,
    asset_store: AssetStore | None = None,
    workdir: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Publish once while fencing duplicate parent/scheduler deliveries."""
    with execution_lock(engine, "publish_target", target_id) as acquired:
        if not acquired:
            return {
                "target_id": target_id,
                "skipped": True,
                "reason": "execution_locked",
            }
        return _run_publish_target_unlocked(
            db,
            target_id,
            publisher_factory=publisher_factory,
            asset_store=asset_store,
            workdir=workdir,
            now=now,
        )


def run_publish_job(
    db: Session,
    job_id: str,
    *,
    publisher_factory: PublisherFactory | None = None,
    asset_store: AssetStore | None = None,
    workdir: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fan the /publications job out per target with strict per-target isolation."""
    job = _get_job(db, job_id)
    completed = _job_start(db, job)
    if completed is not None:
        return completed
    now_dt = _as_aware(now) or datetime.now(UTC)
    results: dict[str, Any] = {}
    for target_id in (job.payload or {}).get("target_ids", []):
        target = db.get(PublishTarget, target_id)
        if target is None:
            results[target_id] = {"error": "target not found"}
            continue
        scheduled = _as_aware(target.scheduled_at)
        account = _target_account(db, target)
        if scheduled is not None and scheduled > now_dt and not _has_native_schedule(account):
            # No native scheduling capability: the internal scheduler sweep
            # will claim and enqueue the target when its due time arrives.
            results[target_id] = {
                "status": target.status,
                "deferred_until": scheduled.isoformat(),
            }
            continue
        target.claimed_at = datetime.now(UTC)  # keep the scheduler from double-claiming
        db.commit()
        try:
            results[target_id] = run_publish_target(
                db,
                target_id,
                publisher_factory=publisher_factory,
                asset_store=asset_store,
                workdir=workdir,
                now=now_dt,
            )
        except Exception as exc:  # one platform failing never rolls back the others
            db.rollback()
            results[target_id] = {"error": _exception_envelope(exc)}
    summary = {"targets": results}
    _job_succeed(db, job, summary)
    return summary


# ---------------------------------------------------------------------------
# Celery task wrappers (session + default providers only; logic stays above)
# ---------------------------------------------------------------------------


JobRunner = Callable[[Session, str], dict[str, Any]]


def _run_locked_job(job_id: str, runner: JobRunner) -> dict[str, Any]:
    with execution_lock(engine, "job", job_id) as acquired:
        if not acquired:
            return {
                "job_id": job_id,
                "skipped": True,
                "reason": "execution_locked",
            }
        with SessionLocal() as db:
            return runner(db, job_id)


def _run_render_job(db: Session, job_id: str) -> dict[str, Any]:
    job = _get_job(db, job_id)
    rendition_id = str((job.payload or {}).get("rendition_id", ""))
    with execution_lock(engine, "rendition_render", rendition_id) as acquired:
        if not acquired:
            return {
                "job_id": job_id,
                "rendition_id": rendition_id,
                "skipped": True,
                "reason": "execution_locked",
            }
        return run_render_rendition(db, rendition_id, job_id=job_id)


def _run_generate_job(db: Session, job_id: str) -> dict[str, Any]:
    job = _get_job(db, job_id)
    logical_id = job.creative_id or job_id
    with execution_lock(engine, "creative_generate", logical_id) as acquired:
        if not acquired:
            return {
                "job_id": job_id,
                "skipped": True,
                "reason": "execution_locked",
            }
        if job.creative_id:
            creative = db.get(Creative, job.creative_id)
            if creative is not None and CreativeState(creative.state) not in {
                CreativeState.SCRIPT_APPROVED,
                CreativeState.GENERATING,
            }:
                result = {
                    "creative_id": creative.id,
                    "state": creative.state,
                    "skipped": True,
                    "reason": "generation_already_advanced",
                }
                _job_succeed(db, job, result)
                return result
        return run_generate_creative(db, job_id)


@celery_app.task(name="app.workers.tasks.discover_topics")
def discover_topics(job_id: str) -> dict[str, Any]:
    return _run_locked_job(job_id, run_discover_topics)


@celery_app.task(name="app.workers.tasks.write_script")
def write_script(job_id: str) -> dict[str, Any]:
    return _run_locked_job(job_id, run_write_script)


@celery_app.task(name="app.workers.tasks.generate_creative")
def generate_creative(job_id: str) -> dict[str, Any]:
    return _run_locked_job(job_id, _run_generate_job)


@celery_app.task(name="app.workers.tasks.render_rendition")
def render_rendition(job_id: str) -> dict[str, Any]:
    return _run_locked_job(job_id, _run_render_job)


@celery_app.task(name="app.workers.tasks.publish_job")
def publish_job(job_id: str) -> dict[str, Any]:
    return _run_locked_job(job_id, run_publish_job)


@celery_app.task(name="app.workers.tasks.publish_target")
def publish_target(target_id: str) -> dict[str, Any]:
    with SessionLocal() as db:
        return run_publish_target(db, target_id)
