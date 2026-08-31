"""Technical QC for finished renditions (PLAN.md week 4 exit gate).

Pure evaluation: a ``QCReport`` is derived from a ``ProbeResult`` plus optional
loudness measurement and black/freeze detection results. Command builders for
blackdetect/freezedetect and the visual checksum are pure argv functions; their
stderr/stdout parsers live here too, so tests run on captured fixture text.

The visual checksum hashes decoded video frames only (``-map 0:v:0 -f hash``),
so the VI and EN renditions — whose video stream is stream-copied from the same
master — provably share one visual master even though their audio differs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.media.ffmpeg import (
    AUDIO_SAMPLE_RATE,
    TARGET_FPS,
    TARGET_HEIGHT,
    TARGET_LUFS,
    TARGET_TRUE_PEAK_DBTP,
    TARGET_WIDTH,
    LoudnormMeasurement,
)
from app.media.probe import ProbeResult

DEFAULT_MASTER_DURATION = 38.8


@dataclass(frozen=True)
class QCExpectations:
    """Acceptance thresholds; defaults mirror PLAN.md §2 and §5."""

    duration_seconds: float = DEFAULT_MASTER_DURATION
    duration_tolerance: float = 0.5
    width: int = TARGET_WIDTH
    height: int = TARGET_HEIGHT
    fps: float = float(TARGET_FPS)
    fps_tolerance: float = 0.1
    vcodec: str = "h264"
    pix_fmt: str = "yuv420p"
    require_audio: bool = True
    require_loudness: bool = True
    require_blackdetect: bool = True
    require_freezedetect: bool = True
    acodec: str = "aac"
    sample_rate: int = AUDIO_SAMPLE_RATE
    loudness_lufs: float = TARGET_LUFS
    loudness_tolerance_lu: float = 1.6
    true_peak_max_dbtp: float = TARGET_TRUE_PEAK_DBTP


DEFAULT_EXPECTATIONS = QCExpectations()


@dataclass(frozen=True)
class QCCheck:
    name: str
    passed: bool
    expected: str
    actual: str


@dataclass
class QCReport:
    checks: list[QCCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failures(self) -> list[QCCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe shape for ``Rendition.qc_report``."""
        return {
            "passed": self.passed,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "expected": c.expected,
                    "actual": c.actual,
                }
                for c in self.checks
            ],
        }


def qc_report_passed(report: object) -> bool:
    """Return true only for an explicit boolean pass in a persisted report.

    Persisted JSON is a trust boundary: missing reports, malformed shapes and
    truthy substitutes such as ``1`` or ``"true"`` must all fail closed.
    """
    return isinstance(report, Mapping) and report.get("passed") is True


def evaluate_master(
    probe: ProbeResult,
    loudnorm: LoudnormMeasurement | None = None,
    *,
    expectations: QCExpectations = DEFAULT_EXPECTATIONS,
    black_intervals: Sequence[BlackInterval] | None = None,
    freeze_intervals: Sequence[FreezeInterval] | None = None,
) -> QCReport:
    """Evaluate a probed rendition against expectations.

    ``loudnorm`` must be a fresh measurement of the *final* file (pass the
    measure command over the output, parse stderr). ``black_intervals`` /
    ``freeze_intervals`` are results of the detect passes. For every analysis
    required by ``expectations``, ``None`` fails closed while an empty list is
    a successful detector run with no findings. Silent/pre-visual files can be
    assessed by explicitly disabling audio/loudness requirements.
    """
    exp = expectations
    checks: list[QCCheck] = []

    checks.append(
        QCCheck(
            name="duration",
            passed=abs(probe.duration - exp.duration_seconds) <= exp.duration_tolerance,
            expected=f"{exp.duration_seconds:g}s ± {exp.duration_tolerance:g}s",
            actual=f"{probe.duration:.2f}s",
        )
    )
    checks.append(
        QCCheck(
            name="resolution",
            passed=probe.width == exp.width and probe.height == exp.height,
            expected=f"{exp.width}x{exp.height}",
            actual=f"{probe.width}x{probe.height}",
        )
    )
    checks.append(
        QCCheck(
            name="aspect_ratio",
            passed=probe.height > 0 and probe.width * 16 == probe.height * 9,
            expected="9:16",
            actual=f"{probe.width}:{probe.height}",
        )
    )
    checks.append(
        QCCheck(
            name="fps",
            passed=abs(probe.fps - exp.fps) <= exp.fps_tolerance,
            expected=f"{exp.fps:g} ± {exp.fps_tolerance:g}",
            actual=f"{probe.fps:g}",
        )
    )
    checks.append(
        QCCheck(
            name="video_codec",
            passed=probe.vcodec.lower() == exp.vcodec,
            expected=exp.vcodec,
            actual=probe.vcodec,
        )
    )
    checks.append(
        QCCheck(
            name="pix_fmt",
            passed=probe.pix_fmt in (exp.pix_fmt, "yuvj420p") if exp.pix_fmt == "yuv420p" else probe.pix_fmt == exp.pix_fmt,
            expected=exp.pix_fmt,
            actual=probe.pix_fmt,
        )
    )

    if exp.require_audio:
        checks.append(
            QCCheck(
                name="audio_codec",
                passed=(probe.acodec or "").lower() == exp.acodec,
                expected=exp.acodec,
                actual=probe.acodec or "none",
            )
        )
        checks.append(
            QCCheck(
                name="audio_sample_rate",
                passed=probe.sample_rate == exp.sample_rate,
                expected=str(exp.sample_rate),
                actual=str(probe.sample_rate or "none"),
            )
        )

    if exp.require_loudness and loudnorm is None:
        checks.append(
            QCCheck(
                name="loudness_measurement",
                passed=False,
                expected="successful loudnorm analysis of final file",
                actual="missing or invalid",
            )
        )
    elif exp.require_loudness:
        assert loudnorm is not None
        checks.append(
            QCCheck(
                name="loudness_integrated",
                passed=abs(loudnorm.input_i - exp.loudness_lufs) <= exp.loudness_tolerance_lu,
                expected=f"{exp.loudness_lufs:g} LUFS ± {exp.loudness_tolerance_lu:g} LU",
                actual=f"{loudnorm.input_i:g} LUFS",
            )
        )
        checks.append(
            QCCheck(
                name="true_peak",
                passed=loudnorm.input_tp <= exp.true_peak_max_dbtp + 1e-6,
                expected=f"<= {exp.true_peak_max_dbtp:g} dBTP",
                actual=f"{loudnorm.input_tp:g} dBTP",
            )
        )

    if exp.require_blackdetect and black_intervals is None:
        checks.append(
            QCCheck(
                name="blackdetect",
                passed=False,
                expected="successful blackdetect analysis",
                actual="missing or failed",
            )
        )
    elif exp.require_blackdetect:
        assert black_intervals is not None
        checks.append(
            QCCheck(
                name="black_frames",
                passed=len(black_intervals) == 0,
                expected="no black intervals",
                actual=f"{len(black_intervals)} interval(s)",
            )
        )
    if exp.require_freezedetect and freeze_intervals is None:
        checks.append(
            QCCheck(
                name="freezedetect",
                passed=False,
                expected="successful freezedetect analysis",
                actual="missing or failed",
            )
        )
    elif exp.require_freezedetect:
        assert freeze_intervals is not None
        checks.append(
            QCCheck(
                name="freeze_frames",
                passed=len(freeze_intervals) == 0,
                expected="no freeze intervals",
                actual=f"{len(freeze_intervals)} interval(s)",
            )
        )

    return QCReport(checks=checks)


# ---------------------------------------------------------------------------
# Black / freeze frame detection (builders + stderr parsers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlackInterval:
    start: float
    end: float
    duration: float


@dataclass(frozen=True)
class FreezeInterval:
    start: float
    end: float | None
    duration: float | None


def blackdetect_cmd(
    input_path: str,
    *,
    min_duration: float = 0.4,
    picture_threshold: float = 0.98,
    pixel_threshold: float = 0.10,
) -> list[str]:
    """argv for a blackdetect analysis pass (results land on stderr)."""
    vf = f"blackdetect=d={min_duration:g}:pic_th={picture_threshold:g}:pix_th={pixel_threshold:g}"
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        input_path,
        "-vf",
        vf,
        "-an",
        "-f",
        "null",
        "-",
    ]


def freezedetect_cmd(
    input_path: str,
    *,
    noise_db: float = -60.0,
    min_duration: float = 2.0,
) -> list[str]:
    """argv for a freezedetect analysis pass (results land on stderr)."""
    vf = f"freezedetect=n={noise_db:g}dB:d={min_duration:g}"
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        input_path,
        "-vf",
        vf,
        "-an",
        "-f",
        "null",
        "-",
    ]


_BLACKDETECT_RE = re.compile(
    r"black_start:\s*(?P<start>[\d.]+)\s+black_end:\s*(?P<end>[\d.]+)"
    r"\s+black_duration:\s*(?P<duration>[\d.]+)"
)


def parse_blackdetect(stderr_text: str) -> list[BlackInterval]:
    """Parse ``[blackdetect @ ...] black_start:... black_end:...`` stderr lines."""
    return [
        BlackInterval(
            start=float(m.group("start")),
            end=float(m.group("end")),
            duration=float(m.group("duration")),
        )
        for m in _BLACKDETECT_RE.finditer(stderr_text)
    ]


_FREEZE_EVENT_RE = re.compile(
    r"lavfi\.freezedetect\.freeze_(?P<kind>start|duration|end):\s*(?P<value>[\d.]+)"
)


def parse_freezedetect(stderr_text: str) -> list[FreezeInterval]:
    """Parse freezedetect stderr events into intervals.

    Events arrive as freeze_start / freeze_duration / freeze_end triplets; a
    freeze running to end-of-stream may lack duration/end, which stay ``None``.
    """
    intervals: list[FreezeInterval] = []
    current_start: float | None = None
    current_duration: float | None = None
    for m in _FREEZE_EVENT_RE.finditer(stderr_text):
        kind = m.group("kind")
        value = float(m.group("value"))
        if kind == "start":
            if current_start is not None:
                intervals.append(
                    FreezeInterval(start=current_start, end=None, duration=current_duration)
                )
            current_start = value
            current_duration = None
        elif kind == "duration":
            current_duration = value
        elif kind == "end" and current_start is not None:
            intervals.append(
                FreezeInterval(start=current_start, end=value, duration=current_duration)
            )
            current_start = None
            current_duration = None
    if current_start is not None:
        intervals.append(FreezeInterval(start=current_start, end=None, duration=current_duration))
    return intervals


# ---------------------------------------------------------------------------
# Visual checksum: prove VI/EN renditions share the visual master
# ---------------------------------------------------------------------------


def visual_checksum_cmd(input_path: str) -> list[str]:
    """argv hashing the *decoded video stream only* (audio excluded).

    Because per-locale renditions stream-copy the master's video, decoding
    yields identical frames and therefore identical SHA-256 digests, proving
    the shared visual master without comparing containers byte-for-byte.
    """
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-f",
        "hash",
        "-hash",
        "sha256",
        "-",
    ]


_HASH_RE = re.compile(r"SHA256=(?P<digest>[0-9a-fA-F]{64})")


def parse_hash_output(text: str) -> str:
    """Extract the lowercase hex digest from ``-f hash`` output (``SHA256=...``)."""
    m = _HASH_RE.search(text)
    if not m:
        raise ValueError("no SHA256 digest found in hash output")
    return m.group("digest").lower()


def renditions_share_visual_master(digest_a: str, digest_b: str) -> bool:
    """Case-insensitive digest comparison for the shared-master QC gate."""
    return bool(digest_a) and digest_a.lower() == digest_b.lower()
