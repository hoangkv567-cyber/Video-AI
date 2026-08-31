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

import io
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from subprocess import TimeoutExpired
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
from app.errors import ProviderQuotaExhausted
from app.media.runner import CommandResult
from app.models import (
    Asset,
    AuditEvent,
    ConnectedAccount,
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
from app.publishing.crypto import decrypt_credentials, encrypt_credentials
from app.states import Capability, CreativeState, JobStatus, PublishTargetStatus
from app.storage import LocalDirBackend, store_asset
from app.workers import tasks as tasks_module
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
    # 4 lite scenes (8 s x 0.05) + 1 hero scene escalated to fast (8 s x 0.10).
    assert sum(e.amount_usd for e in veo_actual) == pytest.approx(4 * 0.4 + 0.8)


def test_generate_restart_reuses_assets_and_never_recalls_providers(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    creative.state = CreativeState.SCRIPT_APPROVED.value
    _seed_script(db_session, creative)
    first_job = _seed_generate_job(db_session, creative)
    first_result = run_generate_creative(db_session, first_job.id, _deps(tmp_path))
    assert creative.state == CreativeState.QC_REQUIRED.value
    first_render_jobs = set(first_result["render_job_ids"])

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
    assert set(result["render_job_ids"]) == first_render_jobs
    assert db_session.query(Job).filter(Job.kind == "render").count() == 2
    assert creative.state == CreativeState.QC_REQUIRED.value


def test_generate_scene_ids_force_only_requested_visual_and_new_render_attempts(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    creative.state = CreativeState.SCRIPT_APPROVED.value
    _seed_script(db_session, creative)
    first_job = _seed_generate_job(db_session, creative)
    first_result = run_generate_creative(db_session, first_job.id, _deps(tmp_path))
    first_render_ids = set(first_result["render_job_ids"])
    scenes = (
        db_session.query(Scene)
        .filter(Scene.creative_id == creative.id)
        .order_by(Scene.index)
        .all()
    )
    requested = scenes[0]
    untouched = scenes[1]
    renditions = db_session.query(Rendition).filter_by(creative_id=creative.id).all()
    for rendition in renditions:
        rendition.qc_report = {"passed": True, "checks": []}
        rendition.is_approved = True
        rendition.approved_at = datetime.now(UTC)
    creative.state = CreativeState.GENERATING.value
    retry_job = Job(
        kind="generate",
        queue="ai",
        creative_id=creative.id,
        payload={"scene_ids": [requested.id]},
    )
    db_session.add(retry_job)
    db_session.commit()
    fresh_image = FakeImageProvider()
    fresh_video = FakeVideoProvider()
    fresh_tts = FakeTTSProvider()

    result = run_generate_creative(
        db_session,
        retry_job.id,
        _deps(
            tmp_path,
            image=fresh_image,
            video=fresh_video,
            tts=fresh_tts,
        ),
    )

    assert "error" not in result
    assert result["regenerated_scene_ids"] == [requested.id]
    assert result["images_generated"] == 1
    assert result["images_reused"] == 5  # styleboard + four untouched scenes
    assert result["clips_generated"] == 1
    assert result["clips_reused"] == 4
    assert result["voices_generated"] == 0
    assert result["voices_reused"] == 10
    assert len(fresh_image.calls) == 1
    assert len(fresh_video.submit_calls) == 1
    assert fresh_tts.calls == []
    for kind in ("keyframe", "clip"):
        assert db_session.query(Asset).filter_by(scene_id=requested.id, kind=kind).count() == 2
        assert db_session.query(Asset).filter_by(scene_id=untouched.id, kind=kind).count() == 1
    assert all(rendition.qc_report is None for rendition in renditions)
    assert all(rendition.is_approved is False for rendition in renditions)
    assert first_render_ids.isdisjoint(result["render_job_ids"])
    assert {
        db_session.get(Job, job_id).status for job_id in first_render_ids
    } == {JobStatus.CANCELLED.value}
    assert db_session.query(Job).filter(Job.kind == "render").count() == 4


def test_generate_rejects_scene_ids_outside_the_approved_script(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    creative.state = CreativeState.SCRIPT_APPROVED.value
    _seed_script(db_session, creative)
    job = Job(
        kind="generate",
        queue="ai",
        creative_id=creative.id,
        payload={"scene_ids": ["not-a-scene-in-this-script"]},
    )
    db_session.add(job)
    db_session.commit()
    image = FakeImageProvider()

    result = run_generate_creative(db_session, job.id, _deps(tmp_path, image=image))

    assert result["error"]["code"] == "validation_failed"
    assert result["error"]["details"]["unknown_scene_ids"] == [
        "not-a-scene-in-this-script"
    ]
    assert result["error"]["retryable"] is False
    assert image.calls == []


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


def test_generate_kill_after_poll_before_asset_repolls_without_resubmit(
    db_session: Session,
    creative: Creative,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedProcessKill(BaseException):
        pass

    creative.state = CreativeState.SCRIPT_APPROVED.value
    _seed_script(db_session, creative)
    job = _seed_generate_job(db_session, creative)
    first_provider = FakeVideoProvider()
    original_store_asset = tasks_module.store_asset
    kill_once = {"armed": True}

    def kill_before_clip_checkpoint(*args: Any, **kwargs: Any) -> Asset:
        if kwargs.get("kind") == "clip" and kill_once["armed"]:
            kill_once["armed"] = False
            raise SimulatedProcessKill
        return original_store_asset(*args, **kwargs)

    monkeypatch.setattr(tasks_module, "store_asset", kill_before_clip_checkpoint)
    with pytest.raises(SimulatedProcessKill):
        run_generate_creative(
            db_session,
            job.id,
            _deps(tmp_path, video=first_provider),
        )
    assert len(first_provider.submit_calls) == 1
    assert (
        db_session.query(Job)
        .filter(Job.kind == "veo_operation", Job.status == JobStatus.RUNNING.value)
        .count()
        == 1
    )

    # A fresh provider can re-poll the completed operation by its durable name.
    # It submits only the four scenes that genuinely never started.
    fresh_provider = FakeVideoProvider()
    result = run_generate_creative(
        db_session,
        job.id,
        _deps(tmp_path, video=fresh_provider),
    )

    assert "error" not in result
    assert len(fresh_provider.poll_calls) == 5
    assert len(fresh_provider.submit_calls) == 4
    actuals = (
        db_session.query(CostEvent)
        .filter_by(creative_id=creative.id, projected=False, kind="veo")
        .all()
    )
    assert len(actuals) == 5
    assert db_session.query(CostEvent).filter_by(
        creative_id=creative.id, projected=True, kind="veo"
    ).count() == 0


def test_generate_ambiguous_veo_intent_requires_operator_action(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    class SimulatedProcessKill(BaseException):
        pass

    class SideEffectThenKilled(FakeVideoProvider):
        def submit(
            self,
            prompt: str,
            *,
            model_id: str,
            duration_seconds: float,
            keyframe_bytes: bytes | None = None,
        ) -> VideoOperation:
            super().submit(
                prompt=prompt,
                model_id=model_id,
                duration_seconds=duration_seconds,
                keyframe_bytes=keyframe_bytes,
            )
            raise SimulatedProcessKill

    creative.state = CreativeState.SCRIPT_APPROVED.value
    _seed_script(db_session, creative)
    job = _seed_generate_job(db_session, creative)

    with pytest.raises(SimulatedProcessKill):
        run_generate_creative(
            db_session,
            job.id,
            _deps(tmp_path, video=SideEffectThenKilled()),
        )

    result = run_generate_creative(
        db_session,
        job.id,
        _deps(tmp_path, video=FakeVideoProvider()),
    )

    assert result["error"]["code"] == "veo_submission_ambiguous"
    assert result["error"]["details"]["operator_action_required"] is True
    assert creative.state == CreativeState.NEEDS_ACTION.value
    assert job.status == JobStatus.FAILED.value
    audit = (
        db_session.query(AuditEvent)
        .filter_by(entity_id=creative.id, action="generation_needs_action")
        .one()
    )
    assert audit.data["code"] == "veo_submission_ambiguous"


def test_generate_cost_cap_aborts_with_failed_creative_and_audit(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    creative.state = CreativeState.SCRIPT_APPROVED.value
    creative.cost_cap_usd = 0.10  # second image (0.067 + 0.067) blows the cap
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
    # The retried scene's actual veo cost is billed at the fast rate (8 s x 0.10).
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
    assert max(e.amount_usd for e in veo_events) == pytest.approx(8 * 0.10)


# ---------------------------------------------------------------------------
# render_rendition
# ---------------------------------------------------------------------------


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
    """Records every argv; fabricates ffmpeg output files so the task can read them."""

    def __init__(
        self,
        *,
        ffprobe_stdout: str = GOOD_FFPROBE_JSON,
        final_loudnorm_stderr: str | None = None,
        analysis_failures: frozenset[str] = frozenset(),
    ) -> None:
        self.commands: list[list[str]] = []
        self.ffprobe_stdout = ffprobe_stdout
        self.final_loudnorm_stderr = final_loudnorm_stderr
        self.analysis_failures = analysis_failures
        self.loudnorm_measurements = 0

    def run(self, argv: Any) -> CommandResult:
        argv = list(argv)
        self.commands.append(argv)
        if argv[0] == "ffprobe":
            return CommandResult(tuple(argv), 0, self.ffprobe_stdout, "")
        if argv[0] == "ffmpeg" and "-vf" in argv:
            video_filter = argv[argv.index("-vf") + 1]
            if video_filter.startswith("blackdetect="):
                return CommandResult(
                    tuple(argv),
                    1 if "blackdetect" in self.analysis_failures else 0,
                    "",
                    "",
                )
            if video_filter.startswith("freezedetect="):
                return CommandResult(
                    tuple(argv),
                    1 if "freezedetect" in self.analysis_failures else 0,
                    "",
                    "",
                )
        if argv[0] == "ffmpeg" and "-af" in argv and argv[-1] == "-":
            audio_filter = argv[argv.index("-af") + 1]
            if "loudnorm=" in audio_filter:
                self.loudnorm_measurements += 1
                stderr = GOOD_LOUDNORM_STDERR
                if self.loudnorm_measurements > 1 and self.final_loudnorm_stderr is not None:
                    stderr = self.final_loudnorm_stderr
                return CommandResult(tuple(argv), 0, "", stderr)
        output = argv[-1]
        if argv[0] == "ffmpeg" and (
            output.endswith(".mp4") or output.endswith(".jpg") or output.endswith(".png")
        ):
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
    assert result["commands_run"] == 10  # 5 normalize + concat + voice mix + loudnorm + captions + thumbnail
    joined = [" ".join(cmd) for cmd in runner.commands]
    normalize_cmds = [c for c in joined if "-an" in c.split() and "scale=" in c]
    assert len(normalize_cmds) == 5  # every scene normalized to 1080x1920/30fps, silent
    assert any("xfade=transition=fade:duration=0.3" in c for c in joined)  # crossfades
    assert any("adelay=7700" in c for c in joined)  # scene 1 voice offset at 7.7 s
    assert any("blackdetect=" in c for c in joined)
    assert any("freezedetect=" in c for c in joined)
    assert any("subtitles=" in c for c in joined)  # captions burned
    assert all(cmd[0] in {"ffmpeg", "ffprobe"} for cmd in runner.commands)

    db_session.expire_all()
    rendition = db_session.get(Rendition, renditions["vi"].id)
    assert rendition.master_asset_id == result["master_asset_id"]
    assert rendition.srt_asset_id is not None
    assert rendition.thumbnail_asset_id is not None
    master = db_session.get(Asset, rendition.master_asset_id)
    assert master.kind == "master"
    assert master.locale == "vi"
    assert master.size_bytes > 0
    assert store.exists(master.storage_key)
    assert rendition.qc_report is not None and rendition.qc_report["passed"] is True
    assert result["qc_passed"] is True
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


@pytest.mark.parametrize("ffprobe_stdout", ["", "not valid JSON"])
def test_render_missing_or_invalid_ffprobe_never_moves_creative_to_ready(
    db_session: Session,
    creative: Creative,
    tmp_path: Path,
    ffprobe_stdout: str,
) -> None:
    store = _generated_creative(db_session, creative, tmp_path)
    renditions = db_session.query(Rendition).filter(Rendition.creative_id == creative.id).all()

    first = run_render_rendition(
        db_session,
        renditions[0].id,
        runner=FakeRunner(),
        asset_store=store,
        workdir=tmp_path / "render_good",
    )
    second = run_render_rendition(
        db_session,
        renditions[1].id,
        runner=FakeRunner(ffprobe_stdout=ffprobe_stdout),
        asset_store=store,
        workdir=tmp_path / "render_bad_probe",
    )

    assert first["qc_passed"] is True
    assert second["qc_passed"] is None
    assert creative.state == CreativeState.QC_REQUIRED.value
    assert db_session.get(Rendition, renditions[1].id).qc_report is None


def test_render_failed_qc_keeps_creative_in_qc_required(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    store = _generated_creative(db_session, creative, tmp_path)
    renditions = db_session.query(Rendition).filter(Rendition.creative_id == creative.id).all()
    failed_qc_probe = GOOD_FFPROBE_JSON.replace('"38.8"', '"10.0"')

    run_render_rendition(
        db_session,
        renditions[0].id,
        runner=FakeRunner(),
        asset_store=store,
        workdir=tmp_path / "render_good",
    )
    result = run_render_rendition(
        db_session,
        renditions[1].id,
        runner=FakeRunner(ffprobe_stdout=failed_qc_probe),
        asset_store=store,
        workdir=tmp_path / "render_failed_qc",
    )

    assert result["qc_passed"] is False
    assert creative.state == CreativeState.QC_REQUIRED.value


def test_failed_qc_creates_a_new_business_render_attempt_and_can_recover(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    store = _generated_creative(db_session, creative, tmp_path)
    renditions = db_session.query(Rendition).filter(Rendition.creative_id == creative.id).all()
    render_jobs = {
        str((job.payload or {}).get("rendition_id")): job
        for job in db_session.query(Job).filter(Job.kind == "render").all()
    }
    first_job_ids = {rendition.id: render_jobs[rendition.id].id for rendition in renditions}
    failed_qc_probe = GOOD_FFPROBE_JSON.replace('"38.8"', '"10.0"')

    run_render_rendition(
        db_session,
        renditions[0].id,
        runner=FakeRunner(),
        asset_store=store,
        workdir=tmp_path / "business_render_good",
        job_id=first_job_ids[renditions[0].id],
    )
    failed = run_render_rendition(
        db_session,
        renditions[1].id,
        runner=FakeRunner(ffprobe_stdout=failed_qc_probe),
        asset_store=store,
        workdir=tmp_path / "business_render_bad",
        job_id=first_job_ids[renditions[1].id],
    )
    assert failed["qc_passed"] is False
    assert creative.state == CreativeState.QC_REQUIRED.value

    creative.state = CreativeState.GENERATING.value
    retry_job = _seed_generate_job(db_session, creative)
    regenerated = run_generate_creative(db_session, retry_job.id, _deps(tmp_path))
    retry_render_ids = set(regenerated["render_job_ids"])

    assert first_job_ids[renditions[0].id] in retry_render_ids
    assert first_job_ids[renditions[1].id] not in retry_render_ids
    retry_render_job = (
        db_session.query(Job)
        .filter(
            Job.kind == "render",
            Job.id.in_(retry_render_ids),
            Job.payload["rendition_id"].as_string() == renditions[1].id,
        )
        .one()
    )
    recovered = run_render_rendition(
        db_session,
        renditions[1].id,
        runner=FakeRunner(),
        asset_store=store,
        workdir=tmp_path / "business_render_recovered",
        job_id=retry_render_job.id,
    )

    assert recovered["qc_passed"] is True
    assert creative.state == CreativeState.READY.value


@pytest.mark.parametrize(
    ("missing_analysis", "failed_check"),
    [
        ("final_loudnorm", "loudness_measurement"),
        ("blackdetect", "blackdetect"),
        ("freezedetect", "freezedetect"),
    ],
)
def test_render_missing_required_analysis_fails_qc_closed(
    db_session: Session,
    creative: Creative,
    tmp_path: Path,
    missing_analysis: str,
    failed_check: str,
) -> None:
    store = _generated_creative(db_session, creative, tmp_path)
    rendition = db_session.query(Rendition).filter(Rendition.creative_id == creative.id).first()
    runner = FakeRunner(
        final_loudnorm_stderr="" if missing_analysis == "final_loudnorm" else None,
        analysis_failures=(
            frozenset({missing_analysis})
            if missing_analysis in {"blackdetect", "freezedetect"}
            else frozenset()
        ),
    )

    result = run_render_rendition(
        db_session,
        rendition.id,
        runner=runner,
        asset_store=store,
        workdir=tmp_path / f"render_missing_{missing_analysis}",
    )

    assert result["qc_passed"] is False
    report = db_session.get(Rendition, rendition.id).qc_report
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert failed == {failed_check}
    assert creative.state == CreativeState.QC_REQUIRED.value


def test_render_malformed_existing_qc_report_fails_closed(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    store = _generated_creative(db_session, creative, tmp_path)
    renditions = db_session.query(Rendition).filter(Rendition.creative_id == creative.id).all()
    run_render_rendition(
        db_session,
        renditions[0].id,
        runner=FakeRunner(),
        asset_store=store,
        workdir=tmp_path / "render_first",
    )
    renditions[0].qc_report = {"passed": "true", "checks": []}
    db_session.commit()

    result = run_render_rendition(
        db_session,
        renditions[1].id,
        runner=FakeRunner(),
        asset_store=store,
        workdir=tmp_path / "render_second",
    )

    assert result["qc_passed"] is True
    assert creative.state == CreativeState.QC_REQUIRED.value


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


class FailedMediaRunner:
    def run(self, argv: Any) -> CommandResult:
        command = tuple(argv)
        return CommandResult(command, 1, "", "simulated ffmpeg failure")


class TimedOutMediaRunner:
    def run(self, argv: Any) -> CommandResult:
        raise TimeoutExpired(cmd=list(argv), timeout=1.0)


@pytest.mark.parametrize(
    ("runner", "expected_code"),
    [
        (FailedMediaRunner(), "media_processing_failed"),
        (TimedOutMediaRunner(), "media_processing_timeout"),
    ],
)
def test_render_media_process_errors_are_retryable(
    db_session: Session,
    creative: Creative,
    tmp_path: Path,
    runner: Any,
    expected_code: str,
) -> None:
    store = _generated_creative(db_session, creative, tmp_path)
    rendition = db_session.query(Rendition).filter_by(creative_id=creative.id).first()

    result = run_render_rendition(
        db_session,
        rendition.id,
        runner=runner,
        asset_store=store,
        workdir=tmp_path / expected_code,
    )

    assert result["error"]["code"] == expected_code
    assert result["error"]["retryable"] is True
    render_job = db_session.query(Job).filter_by(id=result.get("job_id")).one_or_none()
    if render_job is None:
        render_job = db_session.query(Job).filter_by(kind="render").order_by(Job.created_at.desc()).first()
    assert render_job.status == JobStatus.FAILED.value
    assert render_job.error["retryable"] is True


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

    def validate(self, target: PublishContext, rendition: Mapping[str, Any] | None = None) -> None:
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


def _seed_targets(db: Session, creative: Creative, platforms: list[str]) -> list[PublishTarget]:
    creative.state = CreativeState.SCHEDULED.value
    rendition = Rendition(creative_id=creative.id, locale="vi", title="Title vi", is_approved=True)
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


def _connect_target(
    db: Session,
    target: PublishTarget,
    capability: Capability,
    *,
    credentials: dict[str, Any] | None = None,
) -> ConnectedAccount:
    account = ConnectedAccount(
        platform=target.platform,
        display_name=f"Test {target.platform}",
        capability=capability.value,
        encrypted_credentials=(encrypt_credentials(credentials) if credentials is not None else ""),
        status="active",
    )
    db.add(account)
    db.flush()
    target.connected_account_id = account.id
    db.commit()
    return account


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

    ok = run_publish_target(db_session, targets[0].id, publisher_factory=factory, asset_store=store)
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
    store = LocalDirBackend(tmp_path / "assets")
    rendition = db_session.get(Rendition, targets[0].rendition_id)
    video_bytes = b"fake-mp4-for-manual-bundle"
    master = store_asset(
        db_session,
        store,
        creative_id=creative.id,
        kind="master",
        locale="vi",
        data=video_bytes,
        filename="master_vi.mp4",
    )
    thumbnail_bytes = b"fake-thumbnail"
    thumbnail = store_asset(
        db_session,
        store,
        creative_id=creative.id,
        kind="thumbnail",
        locale="vi",
        data=thumbnail_bytes,
        filename="thumbnail_vi.jpg",
    )
    rendition.master_asset_id = master.id
    rendition.thumbnail_asset_id = thumbnail.id
    db_session.commit()
    result = run_publish_target(
        db_session,
        targets[0].id,
        publisher_factory=_factory({"tiktok": "needs_action"}),
        asset_store=store,
    )
    assert result["status"] == PublishTargetStatus.MANUAL_BUNDLE.value
    assert creative.state == CreativeState.NEEDS_ACTION.value
    target = db_session.get(PublishTarget, targets[0].id)
    assert target.last_error["code"] == "publish_needs_action"
    assert target.bundle_path
    bundle_asset = db_session.query(Asset).filter_by(storage_key=target.bundle_path).one()
    assert bundle_asset.kind == "bundle" and bundle_asset.platform == "tiktok"
    assert bundle_asset.pinned is True
    assert bundle_asset.storage_key.startswith(f"creatives/{creative.id}/bundle/")
    assert str(tmp_path) not in bundle_asset.storage_key
    with zipfile.ZipFile(io.BytesIO(store.get_bytes(bundle_asset.storage_key))) as archive:
        names = set(archive.namelist())
        assert archive.read("video.mp4") == video_bytes
        assert archive.read("thumbnail.jpg") == thumbnail_bytes
    assert "video.mp4" in names
    assert {"manifest.json", "title.txt", "disclosure.txt"} <= names


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
    now = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
    future = now + timedelta(hours=6)
    targets[1].scheduled_at = future
    _connect_target(
        db_session,
        targets[1],
        Capability.DIRECT,
        credentials={"access_token": "direct-test-token"},
    )
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
        now=now,
    )

    results = summary["targets"]
    assert results[targets[0].id]["status"] == PublishTargetStatus.PUBLISHED.value
    assert "deferred_until" in results[targets[1].id]
    db_session.expire_all()
    assert db_session.get(PublishTarget, targets[1].id).status == PublishTargetStatus.PENDING.value
    assert job.status == JobStatus.SUCCEEDED.value
    # One target still pending -> the creative must stay in PUBLISHING.
    assert creative.state == CreativeState.PUBLISHING.value


def test_publish_job_pushes_future_target_to_native_scheduler(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    target = _seed_targets(db_session, creative, ["youtube"])[0]
    now = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
    target.scheduled_at = now + timedelta(hours=6)
    _connect_target(
        db_session,
        target,
        Capability.SCHEDULE,
        credentials={"access_token": "schedule-test-token"},
    )
    job = Job(
        kind="publish",
        queue="publish",
        creative_id=creative.id,
        payload={"target_ids": [target.id]},
    )
    db_session.add(job)
    db_session.commit()
    scheduled_seen: list[datetime | None] = []

    class CaptureSchedulePublisher(FakePublisher):
        def validate(
            self,
            target_ctx: PublishContext,
            rendition: Mapping[str, Any] | None = None,
        ) -> None:
            scheduled_seen.append(target_ctx.scheduled_at)
            super().validate(target_ctx, rendition)

    summary = run_publish_job(
        db_session,
        job.id,
        publisher_factory=lambda platform, credentials: CaptureSchedulePublisher(
            credentials, platform_name=platform
        ),
        asset_store=LocalDirBackend(tmp_path / "assets"),
        now=now,
    )

    result = summary["targets"][target.id]
    assert "deferred_until" not in result
    assert result["status"] == PublishTargetStatus.SCHEDULED_REMOTE.value
    assert scheduled_seen == [target.scheduled_at]


def test_due_internal_schedule_publishes_without_elapsed_native_timestamp(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    target = _seed_targets(db_session, creative, ["facebook"])[0]
    now = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
    target.scheduled_at = now - timedelta(seconds=1)
    _connect_target(
        db_session,
        target,
        Capability.DIRECT,
        credentials={"access_token": "direct-test-token"},
    )
    job = Job(
        kind="publish",
        queue="publish",
        creative_id=creative.id,
        payload={"target_ids": [target.id]},
    )
    db_session.add(job)
    db_session.commit()
    scheduled_seen: list[datetime | None] = []

    class CaptureDuePublisher(FakePublisher):
        def validate(
            self,
            target_ctx: PublishContext,
            rendition: Mapping[str, Any] | None = None,
        ) -> None:
            scheduled_seen.append(target_ctx.scheduled_at)
            super().validate(target_ctx, rendition)

    summary = run_publish_job(
        db_session,
        job.id,
        publisher_factory=lambda platform, credentials: CaptureDuePublisher(
            credentials, platform_name=platform
        ),
        asset_store=LocalDirBackend(tmp_path / "assets"),
        now=now,
    )

    assert summary["targets"][target.id]["status"] == PublishTargetStatus.PUBLISHED.value
    assert scheduled_seen == [None]


@pytest.mark.parametrize("capability", [None, Capability.MANUAL, Capability.BLOCKED])
def test_default_factory_manual_targets_bundle_without_network(
    db_session: Session,
    creative: Creative,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capability: Capability | None,
) -> None:
    target = _seed_targets(db_session, creative, ["youtube"])[0]
    if capability is not None:
        _connect_target(
            db_session,
            target,
            capability,
            credentials={"access_token": "must-not-be-used"},
        )
    store = LocalDirBackend(tmp_path / "assets")
    rendition = db_session.get(Rendition, target.rendition_id)
    master = store_asset(
        db_session,
        store,
        creative_id=creative.id,
        kind="master",
        locale="vi",
        data=b"manual-master-video",
        filename="manual_master.mp4",
    )
    rendition.master_asset_id = master.id
    db_session.commit()

    def unexpected_factory(*args: Any, **kwargs: Any) -> Publisher:
        raise AssertionError("manual target must not construct a network publisher")

    monkeypatch.setattr(tasks_module, "create_publisher", unexpected_factory)

    result = run_publish_target(db_session, target.id, asset_store=store)

    assert result["status"] == PublishTargetStatus.MANUAL_BUNDLE.value
    assert db_session.get(PublishTarget, target.id).bundle_path


def test_refreshed_credentials_persist_even_when_retried_step_fails(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    target = _seed_targets(db_session, creative, ["youtube"])[0]
    account = _connect_target(
        db_session,
        target,
        Capability.DIRECT,
        credentials={
            "access_token": "old-secret-token",
            "refresh_token": "refresh-secret-token",
        },
    )

    class RefreshThenFailPublisher(FakePublisher):
        def upload(self, ctx: PublishContext, session: dict[str, Any]) -> dict[str, Any]:
            self.calls.append("upload")
            self.refresh_credentials()
            raise PublishError("upload still failed after refresh")

        def refresh_credentials(self) -> dict[str, Any]:
            self.calls.append("refresh")
            self._credentials["access_token"] = "new-secret-token"
            return dict(self._credentials)

    result = run_publish_target(
        db_session,
        target.id,
        publisher_factory=lambda platform, credentials: RefreshThenFailPublisher(
            credentials, platform_name=platform
        ),
        asset_store=LocalDirBackend(tmp_path / "assets"),
    )

    assert result["status"] == PublishTargetStatus.FAILED.value
    db_session.expire_all()
    stored_account = db_session.get(ConnectedAccount, account.id)
    assert "new-secret-token" not in stored_account.encrypted_credentials
    assert decrypt_credentials(stored_account.encrypted_credentials) == {
        "access_token": "new-secret-token",
        "refresh_token": "refresh-secret-token",
    }
    target_after = db_session.get(PublishTarget, target.id)
    exposed = repr({"result": result, "last_error": target_after.last_error})
    assert "new-secret-token" not in exposed
    assert "refresh-secret-token" not in exposed


class SimulatedWorkerDeath(BaseException):
    """Failure injection that bypasses the worker's normal Exception handler."""


def test_publish_prepare_crash_fails_closed_without_opening_another_session(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    target = _seed_targets(db_session, creative, ["youtube"])[0]
    remote = {"prepare_calls": 0}

    class DieDuringPrepare(FakePublisher):
        def prepare(self, ctx: PublishContext) -> dict[str, Any]:
            remote["prepare_calls"] += 1
            # Simulate the provider accepting START before the process dies and
            # before its response can be checkpointed locally.
            raise SimulatedWorkerDeath

    with pytest.raises(SimulatedWorkerDeath):
        run_publish_target(
            db_session,
            target.id,
            publisher_factory=lambda platform, credentials: DieDuringPrepare(
                credentials, platform_name=platform
            ),
            asset_store=LocalDirBackend(tmp_path / "assets"),
        )

    db_session.expire_all()
    interrupted = db_session.get(PublishTarget, target.id)
    attempt = db_session.query(PublishAttempt).filter_by(target_id=target.id).one()
    assert interrupted.status == PublishTargetStatus.UPLOADING.value
    assert attempt.status == "preparing"
    assert attempt.finished_at is None

    result = run_publish_target(
        db_session,
        target.id,
        publisher_factory=lambda platform, credentials: FakePublisher(
            credentials, platform_name=platform
        ),
        asset_store=LocalDirBackend(tmp_path / "assets"),
    )

    assert remote["prepare_calls"] == 1
    assert result["status"] == PublishTargetStatus.NEEDS_ACTION.value
    assert db_session.query(PublishAttempt).filter_by(target_id=target.id).count() == 1


def test_publish_upload_crash_recovers_encrypted_session_without_duplicate_upload(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    target = _seed_targets(db_session, creative, ["youtube"])[0]
    secret_url = "https://upload.example/private-session-secret"
    remote: dict[str, Any] = {
        "prepare_calls": 0,
        "upload_calls": 0,
        "recover_calls": 0,
        "finalize_calls": 0,
        "uploaded": False,
    }

    class DieDuringUpload(FakePublisher):
        def prepare(self, ctx: PublishContext) -> dict[str, Any]:
            remote["prepare_calls"] += 1
            return {"upload_url": secret_url, "remote_id": "remote-video-1"}

        def upload(self, ctx: PublishContext, session: dict[str, Any]) -> dict[str, Any]:
            remote["upload_calls"] += 1
            remote["uploaded"] = True
            raise SimulatedWorkerDeath

    with pytest.raises(SimulatedWorkerDeath):
        run_publish_target(
            db_session,
            target.id,
            publisher_factory=lambda platform, credentials: DieDuringUpload(
                credentials, platform_name=platform
            ),
            asset_store=LocalDirBackend(tmp_path / "assets"),
        )

    attempt = db_session.query(PublishAttempt).filter_by(target_id=target.id).one()
    assert attempt.status == "uploading"
    assert secret_url not in repr(attempt.response)

    class RecoverUploadedSession(FakePublisher):
        def recover_upload(
            self, ctx: PublishContext, session: dict[str, Any]
        ) -> dict[str, Any] | None:
            remote["recover_calls"] += 1
            assert remote["uploaded"] is True
            assert session["upload_url"] == secret_url
            return {**session, "upload_response": {"uploaded": True}}

        def finalize(self, ctx: PublishContext, session: dict[str, Any]) -> PublishResult:
            remote["finalize_calls"] += 1
            return PublishResult(
                remote_post_id=str(session["remote_id"]),
                remote_status={"status": "published"},
            )

    result = run_publish_target(
        db_session,
        target.id,
        publisher_factory=lambda platform, credentials: RecoverUploadedSession(
            credentials, platform_name=platform
        ),
        asset_store=LocalDirBackend(tmp_path / "assets"),
    )

    assert result["status"] == PublishTargetStatus.PUBLISHED.value
    assert remote == {
        "prepare_calls": 1,
        "upload_calls": 1,
        "recover_calls": 1,
        "finalize_calls": 1,
        "uploaded": True,
    }
    assert db_session.query(PublishAttempt).filter_by(target_id=target.id).count() == 1


def test_publish_finalize_crash_recovers_remote_post_without_duplicate_finalize(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    target = _seed_targets(db_session, creative, ["facebook"])[0]
    remote: dict[str, Any] = {"finalize_calls": 0, "recover_calls": 0, "post_id": None}

    class DieDuringFinalize(FakePublisher):
        def finalize(self, ctx: PublishContext, session: dict[str, Any]) -> PublishResult:
            remote["finalize_calls"] += 1
            remote["post_id"] = "remote-post-once"
            raise SimulatedWorkerDeath

    with pytest.raises(SimulatedWorkerDeath):
        run_publish_target(
            db_session,
            target.id,
            publisher_factory=lambda platform, credentials: DieDuringFinalize(
                credentials, platform_name=platform
            ),
            asset_store=LocalDirBackend(tmp_path / "assets"),
        )

    attempt = db_session.query(PublishAttempt).filter_by(target_id=target.id).one()
    assert attempt.status == "finalizing"

    class RecoverFinalizedPost(FakePublisher):
        def recover_finalize(
            self, ctx: PublishContext, session: dict[str, Any]
        ) -> PublishResult | None:
            remote["recover_calls"] += 1
            if remote["post_id"] is None:
                return None
            return PublishResult(
                remote_post_id=str(remote["post_id"]),
                remote_status={"status": "published", "recovered": True},
            )

    result = run_publish_target(
        db_session,
        target.id,
        publisher_factory=lambda platform, credentials: RecoverFinalizedPost(
            credentials, platform_name=platform
        ),
        asset_store=LocalDirBackend(tmp_path / "assets"),
    )

    assert result["status"] == PublishTargetStatus.PUBLISHED.value
    assert result["remote_post_id"] == "remote-post-once"
    assert remote["finalize_calls"] == 1
    assert remote["recover_calls"] == 1
    assert db_session.query(PublishAttempt).filter_by(target_id=target.id).count() == 1


def test_publish_recovery_cap_fails_closed_instead_of_polling_forever(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    target = _seed_targets(db_session, creative, ["youtube"])[0]

    class DieDuringUpload(FakePublisher):
        def prepare(self, ctx: PublishContext) -> dict[str, Any]:
            return {"upload_url": "https://upload.example/one-session"}

        def upload(self, ctx: PublishContext, session: dict[str, Any]) -> dict[str, Any]:
            raise SimulatedWorkerDeath

    with pytest.raises(SimulatedWorkerDeath):
        run_publish_target(
            db_session,
            target.id,
            publisher_factory=lambda platform, credentials: DieDuringUpload(
                credentials, platform_name=platform
            ),
            asset_store=LocalDirBackend(tmp_path / "assets"),
        )

    attempt = db_session.query(PublishAttempt).filter_by(target_id=target.id).one()
    attempt.response = {
        **dict(attempt.response or {}),
        "recovery_count": tasks_module.MAX_PUBLISH_RECOVERY_ATTEMPTS,
    }
    db_session.commit()

    class MustNotProbe(FakePublisher):
        def recover_upload(
            self, ctx: PublishContext, session: dict[str, Any]
        ) -> dict[str, Any] | None:
            raise AssertionError("recovery cap must stop before another remote probe")

    result = run_publish_target(
        db_session,
        target.id,
        publisher_factory=lambda platform, credentials: MustNotProbe(
            credentials, platform_name=platform
        ),
        asset_store=LocalDirBackend(tmp_path / "assets"),
    )

    assert result["status"] == PublishTargetStatus.NEEDS_ACTION.value
    assert db_session.get(PublishTarget, target.id).claimed_at is None
    assert attempt.error["details"]["reason"] == "publish_recovery_exhausted"


# ---------------------------------------------------------------------------
# video provider quota fallback
# ---------------------------------------------------------------------------


class QuotaExhaustedVideoProvider:
    """Every submit is definitively rejected before any billable effect."""

    def __init__(self) -> None:
        self.submit_calls = 0

    def submit(self, *, prompt: str, model_id: str, duration_seconds: float, keyframe_bytes=None) -> str:
        self.submit_calls += 1
        raise ProviderQuotaExhausted(
            "wan provider quota exhausted: AllocationQuota.FreeTierOnly",
            details={"provider_code": "AllocationQuota.FreeTierOnly", "status_code": 403},
        )

    def poll(self, operation_name: str):
        raise AssertionError("poll must not run when submit was rejected")


def test_generate_falls_back_to_keyframe_motion_on_quota_exhausted(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    creative.state = CreativeState.SCRIPT_APPROVED.value
    _seed_script(db_session, creative)
    job = _seed_generate_job(db_session, creative)
    video = QuotaExhaustedVideoProvider()
    deps = _deps(tmp_path, video=video)

    result = run_generate_creative(db_session, job.id, deps)

    assert "error" not in result
    assert result["clips_generated"] == 0
    assert result["clips_fallback"] == 5
    assert result["voices_generated"] == 10  # voices unaffected
    assert creative.state == CreativeState.QC_REQUIRED.value
    assert job.status == JobStatus.SUCCEEDED.value
    # Submit rejected exactly once; the remaining scenes skipped straight to
    # the render-stage motion clips.
    assert video.submit_calls == 1
    assert _asset_count(db_session, creative.id, "clip") == 0
    scenes = db_session.query(Scene).filter(Scene.creative_id == creative.id).all()
    assert all(s.status == "done" for s in scenes)
    fallback_audits = db_session.query(AuditEvent).filter_by(
        action="video_provider_quota_fallback", entity_id=creative.id
    ).all()
    assert len(fallback_audits) == 1
    assert fallback_audits[0].data["fallback"] == "keyframe_motion"
    assert fallback_audits[0].data["scene_index"] == 0


def test_generate_quota_exhausted_without_fallback_fails(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    creative.state = CreativeState.SCRIPT_APPROVED.value
    _seed_script(db_session, creative)
    job = _seed_generate_job(db_session, creative)
    deps = _deps(tmp_path, video=QuotaExhaustedVideoProvider(), video_fallback="")

    result = run_generate_creative(db_session, job.id, deps)

    assert "error" in result
    assert result["error"]["code"] == "provider_quota_exhausted"
    assert job.status == JobStatus.FAILED.value


def test_quota_rejection_does_not_brick_submission_intents(
    db_session: Session, creative: Creative, tmp_path: Path
) -> None:
    """After a quota rejection the scene can retry with a healthy provider."""
    from app.ai.veo import InMemoryOperationStore, VeoService
    from app.config import get_model_config
    from app.costs import CostLedger

    store = InMemoryOperationStore()
    ledger = CostLedger(db_session, default_cap_usd=6.0)
    quota_service = VeoService(
        QuotaExhaustedVideoProvider(), ledger, store, model_config=get_model_config()
    )
    with pytest.raises(ProviderQuotaExhausted):
        quota_service.ensure_submitted(creative, "scene-x", "prompt", keyframe_bytes=b"kf")

    # Intent row and projection were released, not left ambiguous.
    assert store.get("scene-x") is None
    projected = (
        db_session.query(CostEvent)
        .filter_by(creative_id=creative.id, kind="veo", projected=True)
        .all()
    )
    assert projected == []

    healthy_service = VeoService(
        FakeVideoProvider(), ledger, store, model_config=get_model_config()
    )
    submission = healthy_service.ensure_submitted(
        creative, "scene-x", "prompt", keyframe_bytes=b"kf"
    )
    assert submission.resumed is False


def test_image_to_portrait_crops_landscape_keyframe(tmp_path: Path) -> None:
    """A real landscape image becomes the 1080x1920 master frame."""
    import struct
    import subprocess

    landscape = tmp_path / "landscape.png"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=red:s=1600x900:d=1",
            "-frames:v", "1", str(landscape),
        ],
        check=True,
    )

    def png_size(path: Path) -> tuple[int, int]:
        head = path.read_bytes()[:24]
        assert head[:8] == b"\x89PNG\r\n\x1a\n"
        return struct.unpack(">II", head[16:24])

    assert png_size(landscape) == (1600, 900)

    from app.media.runner import SubprocessRunner

    cropped_bytes = tasks_module._image_to_portrait(
        SubprocessRunner(), landscape.read_bytes()
    )
    out = tmp_path / "cropped.png"
    out.write_bytes(cropped_bytes)
    assert png_size(out) == (1080, 1920)
