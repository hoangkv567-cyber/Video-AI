"""Tests for ffprobe parsing, the QC pass/fail matrix, black/freeze detection
parsers, and the shared-visual-master checksum helpers. All offline: the fake
runner returns canned output and no ffmpeg/ffprobe binary is ever invoked."""

import dataclasses
import json
from collections.abc import Sequence

import pytest

from app.media.ffmpeg import LoudnormMeasurement
from app.media.probe import ProbeResult, ffprobe_cmd, parse_ffprobe_json, probe
from app.media.qc import (
    BlackInterval,
    FreezeInterval,
    QCExpectations,
    blackdetect_cmd,
    evaluate_master,
    freezedetect_cmd,
    parse_blackdetect,
    parse_freezedetect,
    parse_hash_output,
    qc_report_passed,
    renditions_share_visual_master,
    visual_checksum_cmd,
)
from app.media.runner import CommandResult, FFmpegError

FFPROBE_JSON = json.dumps(
    {
        "streams": [
            {
                "index": 0,
                "codec_name": "h264",
                "codec_type": "video",
                "width": 1080,
                "height": 1920,
                "pix_fmt": "yuv420p",
                "avg_frame_rate": "30/1",
                "r_frame_rate": "30/1",
            },
            {
                "index": 1,
                "codec_name": "aac",
                "codec_type": "audio",
                "sample_rate": "48000",
                "channels": 2,
            },
        ],
        "format": {"duration": "38.800000", "bit_rate": "4523000"},
    }
)


class FakeRunner:
    """Injectable runner returning canned results; records every argv."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv: Sequence[str]) -> CommandResult:
        self.calls.append(tuple(argv))
        return CommandResult(tuple(argv), self.returncode, self.stdout, self.stderr)


def good_probe(**overrides: object) -> ProbeResult:
    base = ProbeResult(
        duration=38.8,
        width=1080,
        height=1920,
        fps=30.0,
        vcodec="h264",
        acodec="aac",
        sample_rate=48000,
        pix_fmt="yuv420p",
        bitrate=4_523_000,
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


def good_loudnorm(**overrides: object) -> LoudnormMeasurement:
    base = LoudnormMeasurement(
        input_i=-14.2,
        input_tp=-1.8,
        input_lra=4.0,
        input_thresh=-24.5,
        output_i=-14.0,
        output_tp=-1.5,
        output_lra=4.0,
        output_thresh=-24.3,
        normalization_type="linear",
        target_offset=0.1,
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        ({"passed": True}, True),
        (None, False),
        ({}, False),
        ({"passed": False}, False),
        ({"passed": 1}, False),
        ({"passed": "true"}, False),
        ([{"passed": True}], False),
    ],
)
def test_qc_report_passed_requires_explicit_boolean(report: object, expected: bool) -> None:
    assert qc_report_passed(report) is expected


def failed_names(report) -> set[str]:
    return {c.name for c in report.failures()}


def evaluate_complete(
    probe_result: ProbeResult,
    loudnorm: LoudnormMeasurement,
):
    return evaluate_master(
        probe_result,
        loudnorm,
        black_intervals=[],
        freeze_intervals=[],
    )


# ---------------------------------------------------------------------------
# ffprobe wrapper
# ---------------------------------------------------------------------------


class TestProbe:
    def test_ffprobe_cmd_flags(self):
        cmd = ffprobe_cmd("master.mp4")
        assert cmd[0] == "ffprobe"
        assert "-show_format" in cmd
        assert "-show_streams" in cmd
        assert cmd[cmd.index("-print_format") + 1] == "json"
        assert cmd[-1] == "master.mp4"

    def test_parse_ffprobe_json(self):
        result = parse_ffprobe_json(FFPROBE_JSON)
        assert result == good_probe()

    def test_parse_fractional_fps(self):
        data = json.loads(FFPROBE_JSON)
        data["streams"][0]["avg_frame_rate"] = "30000/1001"
        result = parse_ffprobe_json(json.dumps(data))
        assert result.fps == pytest.approx(29.97, abs=0.01)

    def test_parse_video_only_file(self):
        data = json.loads(FFPROBE_JSON)
        data["streams"] = [data["streams"][0]]
        result = parse_ffprobe_json(json.dumps(data))
        assert result.acodec is None
        assert result.sample_rate is None

    def test_parse_no_video_stream_raises(self):
        data = json.loads(FFPROBE_JSON)
        data["streams"] = [data["streams"][1]]
        with pytest.raises(ValueError):
            parse_ffprobe_json(json.dumps(data))

    def test_parse_invalid_json_raises(self):
        with pytest.raises(ValueError):
            parse_ffprobe_json("not json {")

    def test_probe_uses_injected_runner(self):
        runner = FakeRunner(stdout=FFPROBE_JSON)
        result = probe("master.mp4", runner)
        assert result.duration == 38.8
        assert runner.calls == [tuple(ffprobe_cmd("master.mp4"))]

    def test_probe_raises_on_failure(self):
        runner = FakeRunner(stderr="master.mp4: No such file or directory", returncode=1)
        with pytest.raises(FFmpegError):
            probe("master.mp4", runner)

    def test_probe_result_to_dict_is_json_safe(self):
        json.dumps(good_probe().to_dict())


# ---------------------------------------------------------------------------
# QC pass/fail matrix on synthetic ProbeResults
# ---------------------------------------------------------------------------


class TestQCMatrix:
    def test_good_rendition_passes(self):
        report = evaluate_complete(good_probe(), good_loudnorm())
        assert report.passed
        assert report.failures() == []

    @pytest.mark.parametrize(
        ("overrides", "expected_failed"),
        [
            ({"duration": 37.9}, {"duration"}),
            ({"duration": 39.4}, {"duration"}),
            ({"width": 1080, "height": 1080}, {"resolution", "aspect_ratio"}),
            ({"fps": 25.0}, {"fps"}),
            ({"vcodec": "hevc"}, {"video_codec"}),
            ({"pix_fmt": "yuv420p10le"}, {"pix_fmt"}),
            ({"acodec": None, "sample_rate": None}, {"audio_codec", "audio_sample_rate"}),
            ({"sample_rate": 44100}, {"audio_sample_rate"}),
        ],
    )
    def test_single_defect_flips_expected_check(self, overrides, expected_failed):
        report = evaluate_complete(good_probe(**overrides), good_loudnorm())
        assert failed_names(report) == expected_failed

    def test_duration_within_tolerance_passes(self):
        report = evaluate_complete(good_probe(duration=38.5), good_loudnorm())
        assert "duration" not in failed_names(report)

    def test_720x1280_fails_resolution_but_keeps_aspect(self):
        report = evaluate_complete(good_probe(width=720, height=1280), good_loudnorm())
        assert failed_names(report) == {"resolution"}

    def test_loudness_out_of_band_fails(self):
        report = evaluate_complete(good_probe(), good_loudnorm(input_i=-16.0))
        assert failed_names(report) == {"loudness_integrated"}

    def test_loudness_within_1_lu_passes(self):
        report = evaluate_complete(good_probe(), good_loudnorm(input_i=-14.9))
        assert "loudness_integrated" not in failed_names(report)

    def test_true_peak_above_ceiling_fails(self):
        report = evaluate_complete(good_probe(), good_loudnorm(input_tp=-1.2))
        assert failed_names(report) == {"true_peak"}

    def test_true_peak_at_ceiling_passes(self):
        report = evaluate_complete(good_probe(), good_loudnorm(input_tp=-1.5))
        assert "true_peak" not in failed_names(report)

    def test_no_final_loudnorm_fails_closed(self):
        report = evaluate_master(good_probe(), black_intervals=[], freeze_intervals=[])
        assert failed_names(report) == {"loudness_measurement"}

    def test_silent_master_expectations(self):
        # The visual master (pre voice-over) is intentionally silent.
        exp = QCExpectations(require_audio=False, require_loudness=False)
        report = evaluate_master(
            good_probe(acodec=None, sample_rate=None),
            expectations=exp,
            black_intervals=[],
            freeze_intervals=[],
        )
        assert report.passed

    def test_previsual_can_explicitly_disable_optional_analysis_gates(self):
        exp = QCExpectations(
            require_audio=False,
            require_loudness=False,
            require_blackdetect=False,
            require_freezedetect=False,
        )
        report = evaluate_master(
            good_probe(acodec=None, sample_rate=None),
            expectations=exp,
        )
        assert report.passed

    def test_black_and_freeze_intervals_fail_when_present(self):
        report = evaluate_master(
            good_probe(),
            good_loudnorm(),
            black_intervals=[BlackInterval(1.0, 2.0, 1.0)],
            freeze_intervals=[FreezeInterval(5.0, 7.0, 2.0)],
        )
        assert {"black_frames", "freeze_frames"} <= failed_names(report)

    def test_clean_detection_passes(self):
        report = evaluate_master(
            good_probe(), good_loudnorm(), black_intervals=[], freeze_intervals=[]
        )
        assert report.passed

    def test_detection_not_run_fails_closed(self):
        report = evaluate_master(good_probe(), good_loudnorm())
        assert failed_names(report) == {"blackdetect", "freezedetect"}

    def test_report_to_dict_json_serializable(self):
        report = evaluate_complete(good_probe(duration=10.0), good_loudnorm())
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["passed"] is False
        assert any(c["name"] == "duration" and not c["passed"] for c in payload["checks"])


# ---------------------------------------------------------------------------
# black / freeze detection builders and parsers
# ---------------------------------------------------------------------------

BLACKDETECT_STDERR = """\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'master.mp4':
[blackdetect @ 0x55d0f1c2] black_start:12.3456 black_end:13.0 black_duration:0.6544
frame= 600 fps=120 q=-0.0 size=N/A time=00:00:20.00 bitrate=N/A speed=  40x
[blackdetect @ 0x55d0f1c2] black_start:37.9 black_end:38.8 black_duration:0.9
"""

FREEZEDETECT_STDERR = """\
[freezedetect @ 0x5636] lavfi.freezedetect.freeze_start: 4.50446
[freezedetect @ 0x5636] lavfi.freezedetect.freeze_duration: 2.00284
[freezedetect @ 0x5636] lavfi.freezedetect.freeze_end: 6.5073
frame= 900 fps=150 q=-0.0 size=N/A time=00:00:30.00 bitrate=N/A speed=  50x
[freezedetect @ 0x5636] lavfi.freezedetect.freeze_start: 36.0
"""


class TestDetectors:
    def test_blackdetect_cmd(self):
        cmd = blackdetect_cmd("master.mp4")
        vf = cmd[cmd.index("-vf") + 1]
        assert vf == "blackdetect=d=0.4:pic_th=0.98:pix_th=0.1"
        assert cmd[-3:] == ["-f", "null", "-"]
        assert "-an" in cmd

    def test_freezedetect_cmd(self):
        cmd = freezedetect_cmd("master.mp4")
        vf = cmd[cmd.index("-vf") + 1]
        assert vf == "freezedetect=n=-60dB:d=2"
        assert cmd[-3:] == ["-f", "null", "-"]

    def test_parse_blackdetect(self):
        intervals = parse_blackdetect(BLACKDETECT_STDERR)
        assert intervals == [
            BlackInterval(start=12.3456, end=13.0, duration=0.6544),
            BlackInterval(start=37.9, end=38.8, duration=0.9),
        ]

    def test_parse_blackdetect_clean(self):
        assert parse_blackdetect("frame= 100 fps=30 nothing detected") == []

    def test_parse_freezedetect_with_open_interval(self):
        intervals = parse_freezedetect(FREEZEDETECT_STDERR)
        assert intervals == [
            FreezeInterval(start=4.50446, end=6.5073, duration=2.00284),
            FreezeInterval(start=36.0, end=None, duration=None),  # runs to EOF
        ]

    def test_parse_freezedetect_clean(self):
        assert parse_freezedetect("no freezes here") == []


# ---------------------------------------------------------------------------
# visual checksum (shared visual master proof)
# ---------------------------------------------------------------------------


class TestVisualChecksum:
    def test_cmd_hashes_video_stream_only(self):
        cmd = visual_checksum_cmd("vi.mp4")
        assert cmd[cmd.index("-map") + 1] == "0:v:0"
        assert cmd[cmd.index("-f") + 1] == "hash"
        assert cmd[cmd.index("-hash") + 1] == "sha256"
        assert cmd[-1] == "-"

    def test_parse_hash_output(self):
        digest = "AB" * 32
        assert parse_hash_output(f"SHA256={digest}\n") == digest.lower()

    def test_parse_hash_output_rejects_garbage(self):
        with pytest.raises(ValueError):
            parse_hash_output("MD5=deadbeef")

    def test_vi_en_share_master(self):
        digest = "ab" * 32
        assert renditions_share_visual_master(digest, digest.upper())

    def test_different_masters_do_not_match(self):
        assert not renditions_share_visual_master("ab" * 32, "cd" * 32)

    def test_empty_digest_never_matches(self):
        assert not renditions_share_visual_master("", "")
