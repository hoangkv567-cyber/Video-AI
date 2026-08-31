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
from app.storage import LocalDirBackend, store_asset
from app.workers.tasks import GenerationDeps, run_generate_creative, run_render_rendition

GOOD_FFPROBE_JSON = """{
  "streams": [
    {"codec_type": "video", "codec_name": "h264", "width": 1080,
     "height": 1920, "pix_fmt": "yuv420p", "avg_frame_rate": "30/1"},
    {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"}
  ],
  "format": {"duration": "38.8", "bit_rate": "4523000"}
}"""

GOOD_LOUDNORM_STDERR = """{
  "input_i": "-14.0", "input_tp": "-1.5", "input_lra": "2.0",
  "input_thresh": "-24.0", "output_i": "-14.0", "output_tp": "-1.5",
  "output_lra": "2.0", "output_thresh": "-24.0",
  "normalization_type": "linear", "target_offset": "0.0"
}"""


class FakeRunner:
    """Records argv and fabricates ffmpeg output files (offline, no binaries)."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, argv: Any) -> CommandResult:
        argv = list(argv)
        self.commands.append(argv)
        if argv[0] == "ffprobe":
            return CommandResult(tuple(argv), 0, GOOD_FFPROBE_JSON, "")
        if argv[0] == "ffmpeg" and "-af" in argv and argv[-1] == "-":
            audio_filter = argv[argv.index("-af") + 1]
            if "loudnorm=" in audio_filter:
                return CommandResult(tuple(argv), 0, "", GOOD_LOUDNORM_STDERR)
        output = argv[-1]
        if argv[0] == "ffmpeg" and (
            output.endswith(".mp4") or output.endswith(".jpg") or output.endswith(".png")
        ):
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
    # Motion clips enter the standard chain: 5 kenburns + 5 normalize + concat + mix + loudnorm + captions + thumbnail.
    assert result["commands_run"] == 15
    # The 38.8 s crossfade structure is unchanged.
    assert any("xfade=transition=fade:duration=0.3:offset=7.7" in c for c in joined)
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
    assert result_en["commands_run"] == 10  # 5 normalize + concat + mix + loudnorm + captions + thumbnail
    assert _asset_count(db_session, creative.id, "motion_clip") == 5  # no duplicates
    assert creative.state == CreativeState.READY.value


def test_motion_clip_cache_tracks_the_current_keyframe_checksum(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    job = _seed_script_and_job(db_session, creative)
    deps = _freetier_deps(tmp_path, FakeVideoProvider())
    run_generate_creative(db_session, job.id, deps)
    renditions = {
        row.locale: row
        for row in db_session.query(Rendition).filter_by(creative_id=creative.id)
    }
    run_render_rendition(
        db_session,
        renditions["vi"].id,
        runner=FakeRunner(),
        asset_store=deps.asset_store,
        workdir=tmp_path / "render_original",
    )
    scene = (
        db_session.query(Scene)
        .filter_by(creative_id=creative.id)
        .order_by(Scene.index)
        .first()
    )
    previous_motion = (
        db_session.query(Asset)
        .filter_by(scene_id=scene.id, kind="motion_clip")
        .one()
    )
    previous_hash = previous_motion.prompt_hash
    current_keyframe = (
        db_session.query(Asset)
        .filter_by(scene_id=scene.id, kind="keyframe")
        .order_by(Asset.created_at.desc())
        .first()
    )
    store_asset(
        db_session,
        deps.asset_store,
        creative_id=creative.id,
        kind="keyframe",
        scene_id=scene.id,
        data=b"replacement-keyframe-bytes",
        filename="replacement_keyframe.png",
        model_id=current_keyframe.model_id,
        prompt_hash=current_keyframe.prompt_hash,
    )
    db_session.commit()

    runner = FakeRunner()
    result = run_render_rendition(
        db_session,
        renditions["en"].id,
        runner=runner,
        asset_store=deps.asset_store,
        workdir=tmp_path / "render_after_keyframe_change",
    )

    assert "error" not in result
    assert len([cmd for cmd in runner.commands if "zoompan" in " ".join(cmd)]) == 1
    assert _asset_count(db_session, creative.id, "motion_clip") == 6
    latest_motion = (
        db_session.query(Asset)
        .filter_by(scene_id=scene.id, kind="motion_clip")
        .order_by(Asset.created_at.desc())
        .first()
    )
    assert latest_motion.prompt_hash
    assert latest_motion.prompt_hash != previous_hash


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
