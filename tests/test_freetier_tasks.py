"""Free-tier pipeline wiring through the worker tasks.

Proves that with ``settings.video_provider=keyframe_motion`` the generate flow
performs ZERO Veo submits/polls and records ZERO veo cost events, while the
render stage cuts per-scene Ken Burns motion clips from the keyframes through
the injectable Runner — with the same find_asset resume safety as every other
step (a second render re-renders nothing)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from conftest import make_video_plan
from sqlalchemy.orm import Session

from app.ai.base import FakeImageProvider, FakeTTSProvider, FakeVideoProvider
from app.ai.factory import PIPELINE_KEYFRAME_MOTION, build_video_pipeline_kind
from app.config import Settings
from app.media.runner import CommandResult
from app.models import Asset, CostEvent, Creative, Job, Rendition, Scene, ScriptVersion
from app.states import CreativeState, JobStatus
from app.storage import LocalDirBackend
from app.workers.tasks import GenerationDeps, run_generate_creative, run_render_rendition


class FakeRunner:
    """Records argv and fabricates ffmpeg output files (offline, no binaries)."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, argv: Any) -> CommandResult:
        argv = list(argv)
        self.commands.append(argv)
        output = argv[-1]
        if argv[0] == "ffmpeg" and output.endswith(".mp4"):
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"FAKEOUT:" + path.name.encode())
        return CommandResult(tuple(argv), 0, "", "")


def _seed_script_and_job(db: Session, creative: Creative) -> Job:
    creative.state = CreativeState.SCRIPT_APPROVED.value
    db.add(
        ScriptVersion(
            creative_id=creative.id,
            version=1,
            video_plan=make_video_plan(),
            is_approved=True,
            approved_at=datetime.now(UTC),
        )
    )
    job = Job(kind="generate", queue="ai", creative_id=creative.id, payload={})
    db.add(job)
    db.commit()
    return job


def _freetier_deps(tmp_path: Path, video: FakeVideoProvider) -> GenerationDeps:
    # The pipeline kind is resolved from Settings exactly like production
    # wiring does: video_provider=keyframe_motion -> "keyframe_motion".
    settings = Settings(video_provider="keyframe_motion", _env_file=None)
    pipeline = build_video_pipeline_kind(settings)
    assert pipeline == PIPELINE_KEYFRAME_MOTION
    return GenerationDeps(
        image_provider=FakeImageProvider(),
        video_provider=video,
        tts_provider=FakeTTSProvider(),
        asset_store=LocalDirBackend(tmp_path / "assets"),
        poll_interval_seconds=0.0,
        sleep=lambda _s: None,
        video_pipeline=pipeline,
    )


def _asset_count(db: Session, creative_id: str, kind: str) -> int:
    return db.query(Asset).filter(Asset.creative_id == creative_id, Asset.kind == kind).count()


def test_generate_keyframe_motion_performs_zero_veo_work(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    job = _seed_script_and_job(db_session, creative)
    video = FakeVideoProvider()
    deps = _freetier_deps(tmp_path, video)

    result = run_generate_creative(db_session, job.id, deps)

    assert "error" not in result
    # Zero Veo submits, zero polls, zero veo cost events (projected or actual).
    assert video.submit_calls == []
    assert video.poll_calls == []
    veo_events = (
        db_session.query(CostEvent)
        .filter(CostEvent.creative_id == creative.id, CostEvent.kind == "veo")
        .count()
    )
    assert veo_events == 0
    assert _asset_count(db_session, creative.id, "clip") == 0  # no Veo clip assets

    # Keyframes and voices proceed exactly as before.
    assert result["images_generated"] == 6  # style board + 5 keyframes
    assert result["voices_generated"] == 10
    assert result["clips_generated"] == 0
    assert _asset_count(db_session, creative.id, "styleboard") == 1
    assert _asset_count(db_session, creative.id, "keyframe") == 5
    assert _asset_count(db_session, creative.id, "voice") == 10

    scenes = db_session.query(Scene).filter(Scene.creative_id == creative.id).all()
    assert len(scenes) == 5
    assert all(s.status == "done" for s in scenes)
    assert creative.state == CreativeState.QC_REQUIRED.value
    assert job.status == JobStatus.SUCCEEDED.value
    assert len(result["render_job_ids"]) == 2


def test_render_cuts_motion_clips_from_keyframes_and_reuses_them(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    job = _seed_script_and_job(db_session, creative)
    deps = _freetier_deps(tmp_path, FakeVideoProvider())
    assert "error" not in run_generate_creative(db_session, job.id, deps)
    renditions = {
        r.locale: r
        for r in db_session.query(Rendition).filter(Rendition.creative_id == creative.id)
    }

    runner = FakeRunner()
    result = run_render_rendition(
        db_session,
        renditions["vi"].id,
        runner=runner,
        asset_store=deps.asset_store,
        workdir=tmp_path / "render_vi",
    )

    assert "error" not in result
    joined = [" ".join(cmd) for cmd in runner.commands]
    zoompan_cmds = [c for c in joined if "zoompan" in c]
    assert len(zoompan_cmds) == 5  # one Ken Burns render per scene
    assert all("d=240" in c and "s=1080x1920" in c for c in zoompan_cmds)
    # Motion clips enter the standard chain: 5 kenburns + 5 normalize + concat + mix.
    assert result["commands_run"] == 12
    # The 38.8 s crossfade structure is unchanged.
    assert any(
        "xfade=transition=fade:duration=0.3:offset=7.7" in c for c in joined
    )
    assert any("offset=30.8" in c for c in joined)

    # Motion clips are persisted as per-scene assets (resume safety anchor).
    assert _asset_count(db_session, creative.id, "motion_clip") == 5
    motion_assets = (
        db_session.query(Asset)
        .filter(Asset.creative_id == creative.id, Asset.kind == "motion_clip")
        .all()
    )
    assert all(a.scene_id is not None for a in motion_assets)
    assert {a.model_id.split(":")[0] for a in motion_assets} == {"ffmpeg-kenburns"}

    master = db_session.get(Asset, result["master_asset_id"])
    assert master is not None and master.kind == "master" and master.locale == "vi"

    # Second locale render: existing motion clips are reused, never re-rendered.
    runner_en = FakeRunner()
    result_en = run_render_rendition(
        db_session,
        renditions["en"].id,
        runner=runner_en,
        asset_store=deps.asset_store,
        workdir=tmp_path / "render_en",
    )
    assert "error" not in result_en
    assert not any("zoompan" in " ".join(cmd) for cmd in runner_en.commands)
    assert result_en["commands_run"] == 7  # 5 normalize + concat + mix only
    assert _asset_count(db_session, creative.id, "motion_clip") == 5  # no duplicates
    assert creative.state == CreativeState.READY.value


def test_generate_restart_keyframe_motion_reuses_everything(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    job = _seed_script_and_job(db_session, creative)
    run_generate_creative(db_session, job.id, _freetier_deps(tmp_path, FakeVideoProvider()))
    assert creative.state == CreativeState.QC_REQUIRED.value

    # "Restart": back to GENERATING with completely fresh fake providers.
    creative.state = CreativeState.GENERATING.value
    db_session.commit()
    fresh_video = FakeVideoProvider()
    deps = _freetier_deps(tmp_path, fresh_video)
    fresh_image = deps.image_provider
    fresh_tts = deps.tts_provider
    job2 = Job(kind="generate", queue="ai", creative_id=creative.id, payload={})
    db_session.add(job2)
    db_session.commit()

    result = run_generate_creative(db_session, job2.id, deps)

    assert "error" not in result
    assert fresh_video.submit_calls == [] and fresh_video.poll_calls == []
    assert fresh_image.calls == []  # keyframes reused via find_asset
    assert fresh_tts.calls == []  # voices reused via find_asset
    assert result["images_reused"] == 6
    assert result["voices_reused"] == 10
    assert creative.state == CreativeState.QC_REQUIRED.value
