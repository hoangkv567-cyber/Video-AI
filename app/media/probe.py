"""ffprobe wrapper: pure command builder + JSON parser + injectable runner.

``ffprobe_cmd`` and ``parse_ffprobe_json`` are pure; ``probe`` glues them to a
``Runner`` so tests use a fake and production uses ``SubprocessRunner``. The
parsed ``ProbeResult`` feeds both ``Asset.ffprobe`` provenance and QC.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from app.media.runner import Runner, check


@dataclass(frozen=True)
class ProbeResult:
    """Normalized subset of ffprobe output that QC and provenance care about."""

    duration: float
    width: int
    height: int
    fps: float
    vcodec: str
    acodec: str | None
    sample_rate: int | None
    pix_fmt: str
    bitrate: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ffprobe_cmd(input_path: str) -> list[str]:
    """argv for a machine-readable probe: JSON with format + streams sections."""
    return [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        input_path,
    ]


def _parse_rate(value: str | None) -> float:
    """Parse ffprobe frame-rate fractions like '30/1' or '30000/1001'."""
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            denominator = float(den)
            return float(num) / denominator if denominator else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_ffprobe_json(text: str) -> ProbeResult:
    """Parse ``ffprobe -print_format json`` output into a ProbeResult.

    Raises ``ValueError`` when the JSON is malformed or has no video stream.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid ffprobe JSON: {exc}") from exc

    streams: list[dict[str, Any]] = data.get("streams") or []
    fmt: dict[str, Any] = data.get("format") or {}

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("ffprobe output has no video stream")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = _to_float(fmt.get("duration"), default=0.0)
    if duration <= 0:
        duration = _to_float(video.get("duration"), default=0.0)

    fps = _parse_rate(video.get("avg_frame_rate"))
    if fps <= 0:
        fps = _parse_rate(video.get("r_frame_rate"))

    return ProbeResult(
        duration=duration,
        width=_to_int(video.get("width")) or 0,
        height=_to_int(video.get("height")) or 0,
        fps=fps,
        vcodec=str(video.get("codec_name") or ""),
        acodec=str(audio["codec_name"]) if audio and audio.get("codec_name") else None,
        sample_rate=_to_int(audio.get("sample_rate")) if audio else None,
        pix_fmt=str(video.get("pix_fmt") or ""),
        bitrate=_to_int(fmt.get("bit_rate")),
    )


def probe(input_path: str, runner: Runner) -> ProbeResult:
    """Run ffprobe via the injected runner and parse the result.

    Raises ``FFmpegError`` on non-zero exit, ``ValueError`` on unparsable output.
    """
    result = check(runner.run(ffprobe_cmd(input_path)))
    return parse_ffprobe_json(result.stdout)
