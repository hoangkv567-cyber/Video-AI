"""Worker task tests: generation resume safety, cost-cap abort, render, publish.

Task functions are exercised directly (no broker) with the deterministic fakes
from app.ai.base, a LocalDirBackend asset store and an injectable runner, so
everything runs offline. The key exit gates proven here:

- a restarted worker NEVER re-submits Veo (persisted operation names + Asset
  existence checks short-circuit every provider call);
- the 6 USD cap aborts generation with CostCapExceeded -> creative FAILED plus
  an audit row;
- publish failures are isolated per target and roll the creative up to
  PUBLISHED / PARTIAL / NEEDS_ACTION correctly.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import make_video_plan
from sqlalchemy.orm import Session

from app.ai.base import (
    OP_FAILED,
    FakeImageProvider,
    FakeTTSProvider,
    FakeVideoProvider,
    VideoOperation,
)
from app.media.runner import CommandResult
from app.models import (
    Asset,
    AuditEvent,
    CostEvent,
    Creative,
    Job,
    PublishAttempt,
    PublishTarget,
    Rendition,
    Scene,
    ScriptVersion,
)
from app.publishing.base import (
    PublishContext,
    Publisher,
    PublishError,
    PublishNeedsAction,
    PublishResult,
)
from app.states import Capability, CreativeState, JobStatus, PublishTargetStatus
from app.storage import LocalDirBackend
from app.workers.tasks import (
    GenerationDeps,
    run_generate_creative,
    run_publish_job,
    run_publish_target,
    run_render_rendition,
)

# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


def _seed_script(db: Session, creative: Creative, plan: dict | None = None) -> ScriptVersion:
    version = ScriptVersion(
        creative_id=creative.id,
        version=1,
        video_plan=plan or make_video_plan(),
        is_approved=True,
        approved_at=datetime.now(UTC),
    )
    db.add(version)
    db.commit()
    return version


def _seed_generate_job(db: Session, creative: Creative) -> Job:
    job = Job(kind="generate", queue="ai", creative_id=creative.id, payload={})
    db.add(job)
    db.commit()
    return job


def _deps(
    tmp_path: Path,
    *,
    image: FakeImageProvider | None = None,
    video: FakeVideoProvider | None = None,
    tts: FakeTTSProvider | None = None,
    **overrides: Any,
) -> GenerationDeps:
    return GenerationDeps(
        image_provider=image or FakeImageProvider(),
        video_provider=video or FakeVideoProvider(),
        tts_provider=tts or FakeTTSProvider(),
        asset_store=LocalDirBackend(tmp_path / "assets"),
        poll_interval_seconds=0.0,
        sleep=lambda _s: None,
        **overrides,
    )


def _asset_count(db: Session, creative_id: str, kind: str) -> int:
    return db.query(Asset).filter(Asset.creative_id == creative_id, Asset.kind == kind).count()


# ---------------------------------------------------------------------------
# generate_creative
# ---------------------------------------------------------------------------


def test_generate_happy_path(db_session: Session, creative: Creative, tmp_path: Path) -> None:
    creative.state = CreativeState.SCRIPT_APPROVED.value
    _seed_script(db_session, creative)
    job = _seed_generate_job(db_session, creative)
    deps = _deps(tmp_path)

    result = run_generate_creative(db_session, job.id, deps)

    assert "error" not in result
    assert result["images_generated"] == 6  # style board + 5 keyframes
    assert result["clips_generated"] == 5
    assert result["voices_generated"] == 10  # 5 scenes x vi/en
    assert creative.state == CreativeState.QC_REQUIRED.value
    assert job.status == JobStatus.SUCCEEDED.value

    assert _asset_count(db_session, creative.id, "styleboard") == 1
    assert _asset_count(db_session, creative.id, "keyframe") == 5
    assert _asset_count(db_session, creative.id, "clip") == 5
    assert _asset_count(db_session, creative.id, "voice") == 10

    scenes = db_session.query(Scene).filter(Scene.creative_id == creative.id).all()
    assert len(scenes) == 5
    assert all(s.status == "done" for s in scenes)

    renditions = db_session.query(Rendition).filter(Rendition.creative_id == creative.id).all()
    assert {r.locale for r in renditions} == {"vi", "en"}
    assert len(result["render_job_ids"]) == 2

    # Costs: 5 actual veo rows (projections superseded), 6 images, 10 tts rows.
    events = db_session.query(CostEvent).filter(CostEvent.creative_id == creative.id).all()
    veo_actual = [e for e in events if e.kind == "veo" and not e.projected]
    veo_projected = [e for e in events if e.kind == "veo" and e.projected]
    assert len(veo_actual) == 5
    assert veo_projected == []
    assert len([e for e in events if e.kind == "gemini_image"]) == 6
    assert len([e for e in events if e.kind == "tts"]) == 10
    # 4 lite scenes (8 s x 0.05) + 1 hero scene escalated to fast (8 s x 0.15).
    assert sum(e.amount_usd for e in veo_actual) == pytest.approx(4 * 0.4 + 1.2)


def test_generate_restart_reuses_assets_and_never_recalls_providers(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    creative.state = CreativeState.SCRIPT_APPROVED.value
    _seed_script(db_session, creative)
    first_job = _seed_generate_job(db_session, creative)
    run_generate_creative(db_session, first_job.id, _deps(tmp_path))
    assert creative.state == CreativeState.QC_REQUIRED.value

    # "Restart": scene retry re-enters GENERATING with completely fresh providers.
    creative.state = CreativeState.GENERATING.value
    db_session.commit()
    fresh_image = FakeImageProvider()
    fresh_video = FakeVideoProvider()
    fresh_tts = FakeTTSProvider()
    second_job = _seed_generate_job(db_session, creative)
    result = run_generate_creative(
        db_session,
        second_job.id,
        _deps(tmp_path, image=fresh_image, video=fresh_video, tts=fresh_tts),
    )

    assert "error" not in result
    assert fresh_video.submit_calls == []  # Veo NEVER re-called after restart
    assert fresh_video.poll_calls == []
    assert fresh_image.calls == []
    assert fresh_tts.calls == []
    assert result["clips_reused"] == 5
    assert result["images_reused"] == 6
    assert result["voices_reused"] == 10
    assert _asset_count(db_session, creative.id, "clip") == 5  # no duplicates
    assert creative.state == CreativeState.QC_REQUIRED.value


def test_generate_resumes_persisted_veo_operation_without_resubmitting(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    """Kill the worker mid-poll: the stored operation is re-polled, not re-submitted."""
    creative.state = CreativeState.SCRIPT_APPROVED.value
    _seed_script(db_session, creative)
    job = _seed_generate_job(db_session, creative)

    # Run 1: Veo never completes within the poll budget -> retryable failure.
    slow_video = FakeVideoProvider(pending_polls=100)
    result1 = run_generate_creative(
        db_session, job.id, _deps(tmp_path, video=slow_video, max_polls=2)
    )
    assert result1["error"]["retryable"] is True
    assert creative.state == CreativeState.GENERATING.value  # resumable, NOT FAILED
    assert len(slow_video.submit_calls) == 1  # only scene 0 was submitted
    assert job.status == JobStatus.FAILED.value
    submitted_prompt = slow_video.submit_calls[0]["prompt"]

    # Run 2 simulates a restarted worker: brand-new provider instance.
    fresh_video = FakeVideoProvider()
    job2 = _seed_generate_job(db_session, creative)
    result2 = run_generate_creative(db_session, job2.id, _deps(tmp_path, video=fresh_video))

    assert "error" not in result2
    submitted_after = [c["prompt"] for c in fresh_video.submit_calls]
    assert submitted_prompt not in submitted_after  # scene 0 resumed, never resubmitted
    assert len(fresh_video.submit_calls) == 4  # only scenes 1-4 still needed submits
    # The very first poll of run 2 hits the operation persisted by run 1.
    stored_op_name = slow_video.poll_calls[0]
    assert fresh_video.poll_calls[0] == stored_op_name
    assert _asset_count(db_session, creative.id, "clip") == 5
    assert creative.state == CreativeState.QC_REQUIRED.value


def test_generate_cost_cap_aborts_with_failed_creative_and_audit(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    creative.state = CreativeState.SCRIPT_APPROVED.value
    creative.cost_cap_usd = 0.05  # second image (0.039 + 0.039) blows the cap
    _seed_script(db_session, creative)
    job = _seed_generate_job(db_session, creative)
    video = FakeVideoProvider()

    result = run_generate_creative(db_session, job.id, _deps(tmp_path, video=video))

    assert result["error"]["code"] == "cost_cap_exceeded"
    assert creative.state == CreativeState.FAILED.value
    assert job.status == JobStatus.FAILED.value
    assert job.error["code"] == "cost_cap_exceeded"
    assert video.submit_calls == []  # aborted before any Veo spend
    audit = (
        db_session.query(AuditEvent)
        .filter(
            AuditEvent.entity_id == creative.id,
            AuditEvent.action == "generation_aborted_cost_cap",
        )
        .all()
    )
    assert len(audit) == 1
    assert audit[0].data["code"] == "cost_cap_exceeded"


def test_generate_escalates_failed_scene_to_fast_model(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    creative.state = CreativeState.SCRIPT_APPROVED.value
    plan = make_video_plan()
    for scene in plan["scenes"]:
        scene["is_hero"] = False  # scene 0 starts on lite so the escalation is visible
    _seed_script(db_session, creative, plan)
    job = _seed_generate_job(db_session, creative)

    class FirstPollFails(FakeVideoProvider):
        def __init__(self) -> None:
            super().__init__()
            self._failed_once = False

        def poll(self, operation_name: str) -> VideoOperation:
            if not self._failed_once:
                self._failed_once = True
                self.poll_calls.append(operation_name)
                return VideoOperation(operation_name, OP_FAILED, error="transient veo error")
            return super().poll(operation_name)

    video = FirstPollFails()
    result = run_generate_creative(db_session, job.id, _deps(tmp_path, video=video))

    assert "error" not in result
    models = [c["model_id"] for c in video.submit_calls]
    assert len(models) == 6  # 5 scenes + 1 escalated retry of scene 0
    assert "lite" in models[0]  # first attempt on the lite model
    assert "fast" in models[1]  # failed scene escalated lite -> fast
    assert all("lite" in m for m in models[2:])  # remaining scenes stay on lite
    assert creative.state == CreativeState.QC_REQUIRED.value
    # The retried scene's actual veo cost is billed at the fast rate (8 s x 0.15).
    veo_events = (
        db_session.query(CostEvent)
        .filter(
            CostEvent.creative_id == creative.id,
            CostEvent.kind == "veo",
            CostEvent.projected.is_(False),
        )
        .all()
    )
    assert len(veo_events) == 5
    assert max(e.amount_usd for e in veo_events) == pytest.approx(8 * 0.15)


# ---------------------------------------------------------------------------
# render_rendition
# ---------------------------------------------------------------------------


class FakeRunner:
    """Records every argv; fabricates ffmpeg output files so the task can read them."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, argv: Any) -> CommandResult:
        argv = list(argv)
        self.commands.append(argv)
        output = argv[-1]
        if argv[0] == "ffmpeg" and output.endswith(".mp4"):
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"FAKEMASTER:" + path.name.encode())
        return CommandResult(tuple(argv), 0, "", "")


def _generated_creative(db: Session, creative: Creative, tmp_path: Path) -> LocalDirBackend:
    creative.state = CreativeState.SCRIPT_APPROVED.value
    _seed_script(db, creative)
    job = _seed_generate_job(db, creative)
    deps = _deps(tmp_path)
    result = run_generate_creative(db, job.id, deps)
    assert "error" not in result
    return deps.asset_store


def test_render_rendition_builds_media_commands_and_stores_master(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    store = _generated_creative(db_session, creative, tmp_path)
    renditions = {
        r.locale: r
        for r in db_session.query(Rendition).filter(Rendition.creative_id == creative.id)
    }
    runner = FakeRunner()

    result = run_render_rendition(
        db_session,
        renditions["vi"].id,
        runner=runner,
        asset_store=store,
        workdir=tmp_path / "render_vi",
    )

    assert "error" not in result
    assert result["commands_run"] == 7  # 5 normalize + concat + voice mix
    joined = [" ".join(cmd) for cmd in runner.commands]
    normalize_cmds = [c for c in joined if "-an" in c.split() and "scale=" in c]
    assert len(normalize_cmds) == 5  # every scene normalized to 1080x1920/30fps, silent
    assert any("xfade=transition=fade:duration=0.3" in c for c in joined)  # crossfades
    assert any("adelay=7700" in c for c in joined)  # scene 1 voice offset at 7.7 s
    assert all(cmd[0] in {"ffmpeg", "ffprobe"} for cmd in runner.commands)

    db_session.expire_all()
    rendition = db_session.get(Rendition, renditions["vi"].id)
    assert rendition.master_asset_id == result["master_asset_id"]
    master = db_session.get(Asset, rendition.master_asset_id)
    assert master.kind == "master"
    assert master.locale == "vi"
    assert master.size_bytes > 0
    assert store.exists(master.storage_key)
    # Only one rendition rendered so far: creative must still be in QC.
    assert creative.state == CreativeState.QC_REQUIRED.value


def test_render_both_renditions_moves_creative_to_ready(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    store = _generated_creative(db_session, creative, tmp_path)
    renditions = db_session.query(Rendition).filter(Rendition.creative_id == creative.id).all()
    for i, rendition in enumerate(renditions):
        result = run_render_rendition(
            db_session,
            rendition.id,
            runner=FakeRunner(),
            asset_store=store,
            workdir=tmp_path / f"render_{i}",
        )
        assert "error" not in result
    assert creative.state == CreativeState.READY.value
    assert _asset_count(db_session, creative.id, "master") == 2


def test_render_without_clips_fails_cleanly(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    creative.state = CreativeState.QC_REQUIRED.value
    _seed_script(db_session, creative)
    rendition = Rendition(creative_id=creative.id, locale="vi", title="t")
    db_session.add(rendition)
    db_session.commit()
    result = run_render_rendition(
        db_session,
        rendition.id,
        runner=FakeRunner(),
        asset_store=LocalDirBackend(tmp_path / "assets"),
        workdir=tmp_path / "render",
    )
    assert result["error"]["code"] == "validation_failed"


# ---------------------------------------------------------------------------
# publish_target / publish_job
# ---------------------------------------------------------------------------


class FakePublisher(Publisher):
    """Configurable in-memory publisher covering the full contract."""

    def __init__(
        self,
        credentials: dict[str, Any] | None = None,
        *,
        behavior: str = "ok",
        platform_name: str = "youtube",
        **kwargs: Any,
    ) -> None:
        super().__init__(credentials, **kwargs)
        self.behavior = behavior
        self.platform = platform_name  # type: ignore[misc]
        self.calls: list[str] = []

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.DIRECT})

    def validate(
        self, target: PublishContext, rendition: Mapping[str, Any] | None = None
    ) -> None:
        self.calls.append("validate")
        if self.behavior == "needs_action":
            raise PublishNeedsAction("platform demands manual action")

    def prepare(self, ctx: PublishContext) -> dict[str, Any]:
        self.calls.append("prepare")
        if self.behavior == "fail_prepare":
            raise PublishError("prepare exploded")
        return {"session": "s1"}

    def upload(self, ctx: PublishContext, session: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("upload")
        return session

    def finalize(self, ctx: PublishContext, session: dict[str, Any]) -> PublishResult:
        self.calls.append("finalize")
        if self.behavior == "fail_finalize":
            raise PublishError("finalize exploded")
        return PublishResult(
            remote_post_id=f"post-{ctx.rendition_id[:8]}",
            remote_status={"status": "published"},
        )

    def poll_status(self, remote_post_id: str) -> dict[str, Any]:
        return {"status": "published"}

    def refresh_credentials(self) -> dict[str, Any]:
        return dict(self._credentials)

    def probe_capability(self, account: Any) -> Capability:
        return Capability.DIRECT


def _seed_targets(
    db: Session, creative: Creative, platforms: list[str]
) -> list[PublishTarget]:
    creative.state = CreativeState.SCHEDULED.value
    rendition = Rendition(
        creative_id=creative.id, locale="vi", title="Title vi", is_approved=True
    )
    db.add(rendition)
    db.flush()
    targets = [
        PublishTarget(
            creative_id=creative.id,
            rendition_id=rendition.id,
            platform=platform,
            scheduled_at=None,
        )
        for platform in platforms
    ]
    db.add_all(targets)
    db.commit()
    return targets


def _factory(behaviors: dict[str, str]):
    def make(platform: str, credentials: dict[str, Any]) -> FakePublisher:
        return FakePublisher(
            credentials, behavior=behaviors.get(platform, "ok"), platform_name=platform
        )

    return make


def test_publish_all_targets_succeed_creative_published(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    targets = _seed_targets(db_session, creative, ["youtube", "facebook"])
    factory = _factory({})
    store = LocalDirBackend(tmp_path / "assets")
    for target in targets:
        result = run_publish_target(
            db_session, target.id, publisher_factory=factory, asset_store=store
        )
        assert result["status"] == PublishTargetStatus.PUBLISHED.value
        assert result["remote_post_id"]
    assert creative.state == CreativeState.PUBLISHED.value


def test_publish_partial_failure_isolated_per_target(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    targets = _seed_targets(db_session, creative, ["youtube", "facebook"])
    factory = _factory({"facebook": "fail_finalize"})
    store = LocalDirBackend(tmp_path / "assets")

    ok = run_publish_target(
        db_session, targets[0].id, publisher_factory=factory, asset_store=store
    )
    assert ok["status"] == PublishTargetStatus.PUBLISHED.value
    assert creative.state == CreativeState.PUBLISHING.value  # facebook still pending

    failed = run_publish_target(
        db_session, targets[1].id, publisher_factory=factory, asset_store=store
    )
    assert failed["status"] == PublishTargetStatus.FAILED.value

    db_session.expire_all()
    yt = db_session.get(PublishTarget, targets[0].id)
    fb = db_session.get(PublishTarget, targets[1].id)
    assert yt.status == PublishTargetStatus.PUBLISHED.value  # success NOT rolled back
    assert yt.remote_post_id
    assert fb.status == PublishTargetStatus.FAILED.value
    assert fb.last_error["code"] == "publish_error"
    assert creative.state == CreativeState.PARTIAL.value

    attempts = {
        a.target_id: a
        for a in db_session.query(PublishAttempt).filter(
            PublishAttempt.target_id.in_([yt.id, fb.id])
        )
    }
    assert attempts[yt.id].status == "succeeded"
    assert attempts[fb.id].status == "failed"
    assert attempts[fb.id].error["code"] == "publish_error"


def test_publish_needs_action_rolls_up_to_needs_action(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    targets = _seed_targets(db_session, creative, ["tiktok"])
    result = run_publish_target(
        db_session,
        targets[0].id,
        publisher_factory=_factory({"tiktok": "needs_action"}),
        asset_store=LocalDirBackend(tmp_path / "assets"),
    )
    assert result["status"] == PublishTargetStatus.NEEDS_ACTION.value
    assert creative.state == CreativeState.NEEDS_ACTION.value
    target = db_session.get(PublishTarget, targets[0].id)
    assert target.last_error["code"] == "publish_needs_action"


def test_publish_target_already_published_is_skipped(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    targets = _seed_targets(db_session, creative, ["youtube"])
    targets[0].status = PublishTargetStatus.PUBLISHED.value
    db_session.commit()
    calls: list[str] = []

    def factory(platform: str, credentials: dict[str, Any]) -> FakePublisher:
        calls.append(platform)
        return FakePublisher(credentials, platform_name=platform)

    result = run_publish_target(
        db_session,
        targets[0].id,
        publisher_factory=factory,
        asset_store=LocalDirBackend(tmp_path / "assets"),
    )
    assert result["skipped"] is True
    assert calls == []  # no publisher even constructed


def test_publish_job_processes_due_and_defers_future_targets(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    targets = _seed_targets(db_session, creative, ["youtube", "facebook"])
    future = datetime.now(UTC) + timedelta(hours=6)
    targets[1].scheduled_at = future
    job = Job(
        kind="publish",
        queue="publish",
        creative_id=creative.id,
        payload={"target_ids": [t.id for t in targets]},
    )
    db_session.add(job)
    db_session.commit()

    summary = run_publish_job(
        db_session,
        job.id,
        publisher_factory=_factory({}),
        asset_store=LocalDirBackend(tmp_path / "assets"),
    )

    results = summary["targets"]
    assert results[targets[0].id]["status"] == PublishTargetStatus.PUBLISHED.value
    assert "deferred_until" in results[targets[1].id]
    db_session.expire_all()
    assert (
        db_session.get(PublishTarget, targets[1].id).status
        == PublishTargetStatus.PENDING.value
    )
    assert job.status == JobStatus.SUCCEEDED.value
    # One target still pending -> the creative must stay in PUBLISHING.
    assert creative.state == CreativeState.PUBLISHING.value
