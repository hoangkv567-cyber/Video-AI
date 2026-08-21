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
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.ai.base import (
    OP_RUNNING,
    OP_SUCCEEDED,
    ImageProvider,
    ResearchProvider,
    TTSProvider,
    VideoProvider,
)
from app.ai.factory import (
    PIPELINE_KEYFRAME_MOTION,
    PIPELINE_VEO,
    build_tts_provider,
    build_video_pipeline_kind,
)
from app.ai.keyframes import (
    KeyframeService,
    build_keyframe_prompt,
    build_style_board_prompt,
    prompt_hash,
)
from app.ai.topics import ScoredTopic, TopicDiscoveryService
from app.ai.tts import TTSService
from app.ai.veo import JobOperationStore, OperationStore, VeoPollResult, VeoService
from app.config import ModelConfig, get_model_config, get_settings
from app.costs import CostLedger
from app.db import SessionLocal
from app.errors import (
    AppError,
    Conflict,
    CostCapExceeded,
    NotFound,
    UpstreamError,
    ValidationFailed,
    error_envelope,
)
from app.media import ffmpeg as ff
from app.media import motion
from app.media.probe import ProbeResult, ffprobe_cmd, parse_ffprobe_json
from app.media.qc import evaluate_master
from app.media.runner import Runner, SubprocessRunner, check
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
    utcnow,
)
from app.publishing.base import PublishContext, Publisher, PublishNeedsAction
from app.publishing.crypto import decrypt_credentials
from app.publishing.ledger import DbCostLedger
from app.publishing.registry import create_publisher
from app.schemas.videoplan import Disclosure, ScenePlan, VideoPlan
from app.states import (
    CreativeState,
    InvalidTransition,
    JobStatus,
    PublishTargetStatus,
    advance,
)
from app.storage import AssetStore, find_asset, get_asset_store, store_asset
from app.workers.celery_app import celery_app

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
    return {
        "code": "internal_error",
        "message": f"{type(exc).__name__}: {exc}",
        "retryable": False,
        "details": {},
        "correlation_id": str(uuid.uuid4()),
    }


def _get_job(db: Session, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise NotFound(f"job {job_id} not found")
    return job


def _job_start(db: Session, job: Job) -> None:
    job.status = JobStatus.RUNNING.value
    job.started_at = utcnow()
    job.attempts = (job.attempts or 0) + 1
    db.commit()


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
    _job_start(db, job)
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
    video_pipeline: str = PIPELINE_VEO


def _default_generation_deps() -> GenerationDeps:
    """Production wiring; provider selection comes from settings via app.ai.factory."""
    from app.ai.gemini import GeminiImageProvider, GenAIVideoProvider

    settings = get_settings()
    cfg = get_model_config()
    tts_en = build_tts_provider("en", settings, cfg)
    tts_vi = build_tts_provider("vi", settings, cfg)
    return GenerationDeps(
        image_provider=GeminiImageProvider(api_key=settings.gemini_api_key),
        video_provider=GenAIVideoProvider(api_key=settings.gemini_api_key),
        tts_provider=tts_en,
        asset_store=get_asset_store(),
        model_config=cfg,
        tts_providers={"en": tts_en, "vi": tts_vi},
        video_pipeline=build_video_pipeline_kind(settings),
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
        if existing is not None:
            keyframe_bytes[scene.index] = deps.asset_store.get_bytes(existing.storage_key)
            counters["images_reused"] += 1
            continue
        image = service.generate_scene_keyframe(creative, plan, scene.index)
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
) -> None:
    veo = VeoService(deps.video_provider, ledger, operation_store, deps.model_config)
    plan_by_index = {s.index: s for s in plan.scenes}
    for scene in scenes:
        if find_asset(db, creative.id, "clip", scene_id=scene.id) is not None:
            if scene.status != "done":
                scene.status = "done"
            counters["clips_reused"] += 1
            db.commit()
            continue
        result = _generate_clip(
            db, veo, creative, scene, plan_by_index[scene.index], keyframes.get(scene.index), deps
        )
        store_asset(
            db,
            deps.asset_store,
            creative_id=creative.id,
            kind="clip",
            scene_id=scene.id,
            data=result.video_bytes or b"",
            filename=f"clip_s{scene.index}_{uuid.uuid4().hex[:8]}.mp4",
            model_id=result.model_id,
            prompt_hash=prompt_hash(plan_by_index[scene.index].visual_prompt_en, result.model_id),
            cost_usd=result.actual_cost_usd,
            ffprobe={"duration_seconds": result.duration_seconds},
        )
        scene.status = "done"
        scene.last_error = None
        counters["clips_generated"] += 1
        db.commit()


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
        bind = getattr(provider, "bind_cost_ledger", None)
        if callable(bind):
            bind(ledger, creative.id)  # free-tier adapters log zero-cost usage rows
        tts = TTSService(provider, ledger, deps.model_config)
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


def _spawn_render_jobs(db: Session, creative: Creative, renditions: list[Rendition]) -> list[str]:
    job_ids: list[str] = []
    for rendition in renditions:
        job = Job(
            kind="render",
            queue="render",
            creative_id=creative.id,
            payload={"rendition_id": rendition.id},
        )
        db.add(job)
        db.flush()
        job_ids.append(job.id)
    db.commit()
    if get_settings().app_env != "test":
        for job_id in job_ids:
            # Broker outages never lose the queued Job row.
            with contextlib.suppress(Exception):
                celery_app.send_task(
                    "app.workers.tasks.render_rendition", args=[job_id], queue="render"
                )
    return job_ids


def run_generate_creative(
    db: Session, job_id: str, deps: GenerationDeps | None = None
) -> dict[str, Any]:
    job = _get_job(db, job_id)
    if job.creative_id is None:
        raise ValidationFailed(f"job {job_id} has no creative_id")
    creative = db.get(Creative, job.creative_id)
    if creative is None:
        raise NotFound(f"creative {job.creative_id} not found")
    _job_start(db, job)
    if deps is None:
        deps = _default_generation_deps()
    operation_store = deps.operation_store or JobOperationStore(db)

    try:
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
        ledger = CostLedger(db)
        counters = {
            "images_generated": 0,
            "images_reused": 0,
            "clips_generated": 0,
            "clips_reused": 0,
            "voices_generated": 0,
            "voices_reused": 0,
        }

        keyframes = _ensure_keyframes(db, creative, plan, scenes, ledger, deps, counters)
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
                db, creative, plan, scenes, keyframes, ledger, operation_store, deps, counters
            )
        _ensure_voices(db, creative, plan, scenes, ledger, deps, counters)
        renditions = _ensure_renditions(db, creative, plan)

        _advance_state(creative, CreativeState.QC_REQUIRED)
        db.commit()
        render_job_ids = _spawn_render_jobs(db, creative, renditions)

        result = {
            **counters,
            "creative_id": creative.id,
            "state": creative.state,
            "rendition_ids": [r.id for r in renditions],
            "render_job_ids": render_job_ids,
        }
        _job_succeed(db, job, result)
        return result
    except Exception as exc:
        db.rollback()
        envelope = _exception_envelope(exc)
        if envelope.get("retryable"):
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
    existing = find_asset(db, creative.id, "motion_clip", scene_id=scene.id)
    if existing is not None:
        return existing, 0
    keyframe = find_asset(db, creative.id, "keyframe", scene_id=scene.id)
    if keyframe is None:
        raise ValidationFailed(
            f"scene {scene.index} has no clip or keyframe asset; generate first"
        )
    image_path = work / f"keyframe{scene.index}.png"
    image_path.write_bytes(store.get_bytes(keyframe.storage_key))
    clip_path = work / f"motion{scene.index}.mp4"
    direction = motion.direction_for_scene(scene.index)
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
    _job_start(db, job)

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
            voice = find_asset(
                db, creative.id, "voice", scene_id=scene.id, locale=rendition.locale
            )
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
        for idx, clip_path in enumerate(clip_paths):
            norm = work / f"norm{idx}.mp4"
            _run_cmd(runner, ff.normalize_clip_cmd(clip_path, str(norm)))
            commands_run += 1
            norm_paths.append(str(norm))

        visual = work / "master_visual.mp4"
        _run_cmd(runner, ff.concat_crossfade_cmd(norm_paths, str(visual), durations))
        commands_run += 1

        # Scene i starts at sum(previous durations) - i * fade (crossfade overlap).
        offsets = [
            round(sum(durations[:i]) - i * ff.CROSSFADE_SECONDS, 3)
            for i in range(len(durations))
        ]
        mixed = work / f"master_{rendition.locale}_mixed.mp4"
        _run_cmd(
            runner,
            ff.mix_voiceover_cmd(
                str(visual), list(zip(voice_paths, offsets, strict=True)), str(mixed)
            ),
        )
        commands_run += 1

        final = work / f"master_{rendition.locale}.mp4"
        measurement = _measure_loudnorm(runner, str(mixed))
        if measurement is not None:
            _run_cmd(runner, ff.loudnorm_apply_cmd(str(mixed), str(final), measurement))
            commands_run += 1
        else:
            final = mixed  # measurement unavailable (e.g. offline runner): skip pass 2

        data = final.read_bytes()
        probe_result = _probe_file(runner, str(final))
        final_measurement = _measure_loudnorm(runner, str(final))
        qc_report: dict[str, Any] | None = None
        if probe_result is not None:
            qc_report = evaluate_master(probe_result, final_measurement).to_dict()

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

        all_renditions = (
            db.query(Rendition).filter(Rendition.creative_id == creative.id).all()
        )
        if (
            all(r.master_asset_id for r in all_renditions)
            and CreativeState(creative.state) == CreativeState.QC_REQUIRED
        ):
            qc_failed = any(
                r.qc_report is not None and r.qc_report.get("passed") is False
                for r in all_renditions
            )
            if not qc_failed:
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
) -> PublishContext:
    file_path: str | None = None
    file_url: str | None = None
    if rendition.master_asset_id and asset_store is not None:
        asset = db.get(Asset, rendition.master_asset_id)
        if asset is not None:
            file_url = asset_store.get_url(asset.storage_key)
            work = Path(workdir) if workdir else Path(mkdtemp(prefix="publish_"))
            work.mkdir(parents=True, exist_ok=True)
            local = work / f"{rendition.locale}_{asset.id[:8]}.mp4"
            local.write_bytes(asset_store.get_bytes(asset.storage_key))
            file_path = str(local)
    return PublishContext(
        creative_id=creative.id,
        rendition_id=rendition.id,
        locale=rendition.locale,
        title=rendition.title,
        description=rendition.description,
        hashtags=list(rendition.hashtags or []),
        file_path=file_path,
        file_url=file_url,
        privacy=target.privacy,
        scheduled_at=_as_aware(target.scheduled_at),
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
        s
        in {PublishTargetStatus.NEEDS_ACTION.value, PublishTargetStatus.MANUAL_BUNDLE.value}
        for s in statuses
    ):
        _advance_state(creative, CreativeState.NEEDS_ACTION)
    else:
        _advance_state(creative, CreativeState.FAILED)
    return creative.state


def run_publish_target(
    db: Session,
    target_id: str,
    *,
    publisher_factory: PublisherFactory | None = None,
    asset_store: AssetStore | None = None,
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    target = db.get(PublishTarget, target_id)
    if target is None:
        raise NotFound(f"publish target {target_id} not found")
    if target.status in _SUCCESS_TARGET_STATUSES:
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

    attempt_no = (
        db.query(func.count(PublishAttempt.id))
        .filter(PublishAttempt.target_id == target.id)
        .scalar()
        or 0
    ) + 1
    attempt = PublishAttempt(target_id=target.id, attempt_no=attempt_no, status="running")
    db.add(attempt)
    db.commit()

    store = asset_store if asset_store is not None else get_asset_store()
    try:
        credentials = _target_credentials(db, target)
        if publisher_factory is not None:
            publisher = publisher_factory(target.platform, credentials)
        else:
            publisher = create_publisher(
                target.platform,
                credentials,
                ledger=DbCostLedger(db, creative.id),
                settings=get_settings(),
            )
        ctx = _build_publish_context(db, target, rendition, creative, store, workdir)

        publisher.validate(ctx)
        target.status = PublishTargetStatus.VALIDATED.value
        db.commit()
        session = publisher.prepare(ctx)
        target.status = PublishTargetStatus.UPLOADING.value
        db.commit()
        session = publisher.upload(ctx, session)
        result = publisher.finalize(ctx, session)

        target.remote_post_id = result.remote_post_id
        target.remote_status = result.remote_status
        scheduled = _as_aware(target.scheduled_at)
        if scheduled is not None and scheduled > datetime.now(UTC):
            target.status = PublishTargetStatus.SCHEDULED_REMOTE.value
        else:
            target.status = PublishTargetStatus.PUBLISHED.value
        target.last_error = None
        attempt.status = "succeeded"
        attempt.response = {
            "remote_post_id": result.remote_post_id,
            "remote_status": result.remote_status,
        }
    except PublishNeedsAction as exc:
        envelope = error_envelope(exc)
        target.status = PublishTargetStatus.NEEDS_ACTION.value
        target.last_error = envelope
        attempt.status = "needs_action"
        attempt.error = envelope
    except AppError as exc:
        envelope = error_envelope(exc)
        target.status = PublishTargetStatus.FAILED.value
        target.last_error = envelope
        attempt.status = "failed"
        attempt.error = envelope
    except Exception as exc:
        db.rollback()
        envelope = _exception_envelope(exc)
        target.status = PublishTargetStatus.FAILED.value
        target.last_error = envelope
        attempt.status = "failed"
        attempt.error = envelope

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
    _job_start(db, job)
    now_dt = now or datetime.now(UTC)
    results: dict[str, Any] = {}
    for target_id in (job.payload or {}).get("target_ids", []):
        target = db.get(PublishTarget, target_id)
        if target is None:
            results[target_id] = {"error": "target not found"}
            continue
        scheduled = _as_aware(target.scheduled_at)
        if scheduled is not None and scheduled > now_dt:
            # Future target: the scheduler sweep will claim and enqueue it.
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


@celery_app.task(name="app.workers.tasks.discover_topics")
def discover_topics(job_id: str) -> dict[str, Any]:
    with SessionLocal() as db:
        return run_discover_topics(db, job_id)


@celery_app.task(name="app.workers.tasks.generate_creative")
def generate_creative(job_id: str) -> dict[str, Any]:
    with SessionLocal() as db:
        return run_generate_creative(db, job_id)


@celery_app.task(name="app.workers.tasks.render_rendition")
def render_rendition(job_id: str) -> dict[str, Any]:
    with SessionLocal() as db:
        job = _get_job(db, job_id)
        rendition_id = str((job.payload or {}).get("rendition_id", ""))
        return run_render_rendition(db, rendition_id, job_id=job_id)


@celery_app.task(name="app.workers.tasks.publish_job")
def publish_job(job_id: str) -> dict[str, Any]:
    with SessionLocal() as db:
        return run_publish_job(db, job_id)


@celery_app.task(name="app.workers.tasks.publish_target")
def publish_target(target_id: str) -> dict[str, Any]:
    with SessionLocal() as db:
        return run_publish_target(db, target_id)
