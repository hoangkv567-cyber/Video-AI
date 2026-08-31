"""Pure ffmpeg command builders for the render pipeline (PLAN.md week 4).

Every public ``*_cmd`` function returns a complete argv as ``list[str]`` and
performs no I/O; execution goes through ``app.media.runner``. Because argv is
passed straight to ``subprocess`` (no shell), only ffmpeg's own filtergraph
escaping matters here — see ``escape_filter_path`` / ``escape_drawtext_text``.

Targets from the plan: 1080x1920 @ 30 fps, H.264 High, yuv420p, AAC 48 kHz,
``+faststart``, loudness -14 LUFS with true peak <= -1.5 dBTP, and a ~38.8 s
master from five 8 s scenes joined by four 0.3 s crossfades. Veo's native
audio is stripped at normalize time (the visual master is silent by design).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.media.captions import FONT_NAME, SAFE_BOTTOM_MARGIN

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_FPS = 30
TARGET_LUFS = -14.0
TARGET_TRUE_PEAK_DBTP = -1.5
TARGET_LRA = 11.0
AUDIO_SAMPLE_RATE = 48000
AUDIO_BITRATE = "192k"
CROSSFADE_SECONDS = 0.3
SCENE_SECONDS = 8.0
VIDEO_CRF = 18
VIDEO_PRESET = "medium"

_H264_ARGS = ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p"]
_ENCODE_QUALITY_ARGS = ["-preset", VIDEO_PRESET, "-crf", str(VIDEO_CRF)]
_FASTSTART_ARGS = ["-movflags", "+faststart"]


def _fmt(value: float) -> str:
    """Compact float formatting for filter args: 0.3 -> '0.3', -14.0 -> '-14'."""
    return f"{value:g}"


# ---------------------------------------------------------------------------
# Filtergraph escaping (mandatory unit-test list)
# ---------------------------------------------------------------------------


def escape_filter_path(path: str) -> str:
    """Escape a filesystem path for use as an ffmpeg filter option value.

    Windows paths are the classic failure: the filtergraph parser treats ``:``
    as an option separator and ``\\`` as an escape character. We convert
    backslashes to forward slashes (accepted by ffmpeg on Windows), escape the
    drive colon, escape embedded single quotes, and wrap the whole value in
    single quotes — the canonical ``subtitles=filename='C\\:/path/subs.srt'``
    recipe.
    """
    escaped = path.replace("\\", "/")
    escaped = escaped.replace("'", "'\\''")  # close quote, escaped quote, reopen
    escaped = escaped.replace(":", "\\:")
    return f"'{escaped}'"


_DRAWTEXT_SPECIALS = frozenset("\\':,;[]=%")


def escape_drawtext_text(text: str) -> str:
    """Escape arbitrary user text for the drawtext ``text`` option.

    Follows the documented ffmpeg pattern: backslash-escape filter specials and
    wrap in single quotes. Newlines are normalized to ``\\n`` characters and
    preserved (argv is passed without a shell), which drawtext renders as line
    breaks. Callers should also set ``expansion=none`` so ``%`` sequences are
    never expanded.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    escaped = "".join(f"\\{ch}" if ch in _DRAWTEXT_SPECIALS else ch for ch in normalized)
    return f"'{escaped}'"


def quote_filter_value(value: str) -> str:
    """Quote a generic filter option value (e.g. force_style with commas)."""
    return "'" + value.replace("\\", "\\\\").replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------
# Normalization and concat
# ---------------------------------------------------------------------------


def normalize_clip_cmd(input_path: str, output_path: str) -> list[str]:
    """Normalize one Veo scene clip to the master spec, stripping source audio.

    Forces 1080x1920 (scale-to-cover then center crop), 30 fps, square pixels,
    H.264 High / yuv420p, ``+faststart``. ``-an`` drops Veo's native audio per
    the plan: the visual master carries no speech, text, or lip-sync.
    """
    vf = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_WIDTH}:{TARGET_HEIGHT},"
        f"fps={TARGET_FPS},setsar=1"
    )
    return [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        vf,
        "-an",
        *_H264_ARGS,
        *_ENCODE_QUALITY_ARGS,
        *_FASTSTART_ARGS,
        output_path,
    ]


def crossfade_offsets(durations: Sequence[float], fade: float = CROSSFADE_SECONDS) -> list[float]:
    """xfade offsets for chaining clips with crossfades of ``fade`` seconds.

    For five 8 s clips and a 0.3 s fade: [7.7, 15.4, 23.1, 30.8], for a
    38.8 s master. Offsets are cumulative: each is the previous offset plus the
    next clip's duration minus the fade. Rounded to milliseconds to keep float
    noise out of filtergraphs.
    """
    if len(durations) < 2:
        raise ValueError("need at least 2 clips to crossfade")
    if fade <= 0:
        raise ValueError("fade must be positive")
    for i, duration in enumerate(durations):
        if duration <= fade:
            raise ValueError(f"clip {i} duration {duration}s must exceed fade {fade}s")
    offsets: list[float] = []
    cursor = 0.0
    for duration in durations[:-1]:
        cursor = cursor + duration - fade
        offsets.append(round(cursor, 3))
    return offsets


def crossfade_total_duration(
    durations: Sequence[float], fade: float = CROSSFADE_SECONDS
) -> float:
    """Total master duration after (n-1) crossfades: sum(d) - (n-1)*fade."""
    return round(sum(durations) - (len(durations) - 1) * fade, 3)


def concat_crossfade_cmd(
    inputs: Sequence[str],
    output_path: str,
    durations: Sequence[float] | None = None,
    fade: float = CROSSFADE_SECONDS,
) -> list[str]:
    """Join normalized scene clips with an xfade chain into the visual master.

    Offsets are computed from ``durations`` (default 8 s each), never hardcoded.
    The master is video-only (``-an``); voice-over is mixed per locale later.
    """
    if len(inputs) < 2:
        raise ValueError("need at least 2 inputs to concat with crossfades")
    if durations is None:
        durations = [SCENE_SECONDS] * len(inputs)
    if len(durations) != len(inputs):
        raise ValueError(f"got {len(inputs)} inputs but {len(durations)} durations")
    offsets = crossfade_offsets(durations, fade)

    steps: list[str] = []
    prev_label = "[0:v]"
    for i in range(1, len(inputs)):
        out_label = "[vout]" if i == len(inputs) - 1 else f"[vx{i}]"
        steps.append(
            f"{prev_label}[{i}:v]"
            f"xfade=transition=fade:duration={_fmt(fade)}:offset={_fmt(offsets[i - 1])}"
            f"{out_label}"
        )
        prev_label = out_label
    filter_complex = ";".join(steps)

    cmd = ["ffmpeg", "-y"]
    for path in inputs:
        cmd += ["-i", path]
    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-an",
        "-r",
        str(TARGET_FPS),
        *_H264_ARGS,
        *_ENCODE_QUALITY_ARGS,
        *_FASTSTART_ARGS,
        output_path,
    ]
    return cmd


# ---------------------------------------------------------------------------
# Voice-over mix
# ---------------------------------------------------------------------------


def mix_voiceover_cmd(
    master_path: str,
    voice_tracks: Sequence[tuple[str, float]],
    output_path: str,
) -> list[str]:
    """Mix per-scene voice tracks onto the silent master at given offsets.

    ``voice_tracks`` is ``[(path, offset_seconds), ...]``. Each track is
    delayed with ``adelay`` (millisecond precision) and the set is combined
    with ``amix`` without renormalization (the loudnorm pass follows). Video
    is stream-copied so both locales provably share the visual master; audio
    is AAC 48 kHz.
    """
    if not voice_tracks:
        raise ValueError("at least one voice track is required")

    cmd = ["ffmpeg", "-y", "-i", master_path]
    for path, _offset in voice_tracks:
        cmd += ["-i", path]

    steps: list[str] = []
    labels: list[str] = []
    for idx, (_path, offset) in enumerate(voice_tracks, start=1):
        if offset < 0:
            raise ValueError(f"voice track {idx - 1} has negative offset {offset}")
        delay_ms = round(offset * 1000)
        steps.append(f"[{idx}:a]adelay={delay_ms}:all=1[a{idx}]")
        labels.append(f"[a{idx}]")
    if len(labels) == 1:
        mix_label = labels[0]
        aout = mix_label  # single track: the delayed track is the output
        filter_complex = ";".join(steps)
    else:
        steps.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0[aout]")
        aout = "[aout]"
        filter_complex = ";".join(steps)

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v",
        "-map",
        aout,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-b:a",
        AUDIO_BITRATE,
        *_FASTSTART_ARGS,
        output_path,
    ]
    return cmd


# ---------------------------------------------------------------------------
# Two-pass loudness normalization (-14 LUFS / -1.5 dBTP)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoudnormMeasurement:
    """Parsed output of the loudnorm measurement pass (values from stderr JSON)."""

    input_i: float
    input_tp: float
    input_lra: float
    input_thresh: float
    output_i: float
    output_tp: float
    output_lra: float
    output_thresh: float
    normalization_type: str
    target_offset: float


def _loudnorm_target(tp: float | None = None) -> str:
    target_tp = tp if tp is not None else TARGET_TRUE_PEAK_DBTP
    return f"loudnorm=I={_fmt(TARGET_LUFS)}:TP={_fmt(target_tp)}:LRA={_fmt(TARGET_LRA)}"


def loudnorm_measure_cmd(input_path: str) -> list[str]:
    """Pass 1: measure loudness. JSON stats land on stderr; output is discarded."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        input_path,
        "-vn",
        "-af",
        f"{_loudnorm_target()}:print_format=json",
        "-f",
        "null",
        "-",
    ]


# Safety margin (dB) below the linear-mode true-peak ceiling; accounts for
# AAC encoding overshoot measured on final renders (~0.2-0.5 dB).
LOUDNORM_LINEAR_HEADROOM_DB = 0.3


def loudnorm_linear_ok(measured: LoudnormMeasurement, target_tp: float) -> bool:
    """Linear mode only when the required gain keeps peaks under the ceiling.

    A single static gain reaches -14 LUFS only if ``input_tp + gain`` stays
    below ``target_tp - headroom``. Otherwise linear mode silently lands under
    the loudness target (observed -15 LUFS on EN voiceovers) and dynamic mode
    must be used instead.
    """
    gain = TARGET_LUFS - measured.input_i
    return measured.input_tp + gain <= target_tp - LOUDNORM_LINEAR_HEADROOM_DB


def loudnorm_apply_cmd(
    input_path: str,
    output_path: str,
    measured: LoudnormMeasurement,
    *,
    target_tp: float = -2.0,
    linear: bool | None = None,
) -> list[str]:
    """Pass 2: apply normalization using pass-1 measurements.

    Video is stream-copied (visual master unchanged); audio re-encoded AAC 48 kHz.
    Target TP is -2.0 dBTP by default to provide headroom against AAC compression overshoot.
    ``linear`` defaults to :func:`loudnorm_linear_ok`: dynamic mode when a static
    gain would clip peaks or miss the loudness target.
    """
    if linear is None:
        linear = loudnorm_linear_ok(measured, target_tp)
    af = (
        f"{_loudnorm_target(tp=target_tp)}"
        f":measured_I={_fmt(measured.input_i)}"
        f":measured_TP={_fmt(measured.input_tp)}"
        f":measured_LRA={_fmt(measured.input_lra)}"
        f":measured_thresh={_fmt(measured.input_thresh)}"
        f":offset={_fmt(measured.target_offset)}"
        f":linear={'true' if linear else 'false'}:print_format=summary"
    )
    return [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-af",
        af,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-b:a",
        AUDIO_BITRATE,
        *_FASTSTART_ARGS,
        output_path,
    ]


_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_loudnorm_json(stderr_text: str) -> LoudnormMeasurement:
    """Extract the loudnorm JSON block that ffmpeg prints on stderr.

    The block is surrounded by log noise (frame counters, filter banners), so we
    scan for the last brace-delimited JSON object containing ``input_i``.
    """
    for block in reversed(_JSON_BLOCK_RE.findall(stderr_text)):
        if '"input_i"' not in block:
            continue
        data = json.loads(block)
        return LoudnormMeasurement(
            input_i=float(data["input_i"]),
            input_tp=float(data["input_tp"]),
            input_lra=float(data["input_lra"]),
            input_thresh=float(data["input_thresh"]),
            output_i=float(data["output_i"]),
            output_tp=float(data["output_tp"]),
            output_lra=float(data["output_lra"]),
            output_thresh=float(data["output_thresh"]),
            normalization_type=str(data.get("normalization_type", "")),
            target_offset=float(data["target_offset"]),
        )
    raise ValueError("no loudnorm JSON block found in ffmpeg stderr")


# ---------------------------------------------------------------------------
# Caption burn-in
# ---------------------------------------------------------------------------


def portrait_crop_image_cmd(
    input_path: str, output_path: str, *, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT
) -> list[str]:
    """Normalize a still keyframe to the vertical master frame (9:16).

    Keyframe sources are arbitrary (HD photo search returns landscape/square
    images). Downstream consumers animate whatever aspect they are given, and
    the clip normalizer would then center-crop away both sides. Scaling to
    cover and cropping HERE keeps the composition decision at the keyframe
    stage, where a portrait candidate can also be preferred at fetch time.
    """
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_path,
        "-vf",
        vf,
        "-frames:v",
        "1",
        output_path,
    ]


def default_force_style() -> str:
    """libass force_style for SRT burn-in: Noto Sans, bottom-center safe zone."""
    return (
        f"FontName={FONT_NAME},Fontsize=54,Alignment=2,"
        f"MarginV={SAFE_BOTTOM_MARGIN},MarginL=60,MarginR=60,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1"
    )


def burn_captions_cmd(
    input_path: str,
    subtitle_path: str,
    output_path: str,
    *,
    fonts_dir: str | None = None,
    force_style: str | None = None,
) -> list[str]:
    """Burn subtitles into the video via the ``subtitles`` filter.

    For ``.srt`` inputs the default Noto Sans safe-zone style is force-applied
    (pass ``force_style=""`` to disable); ``.ass`` files carry their own style
    block. ``fonts_dir`` lets libass find bundled Noto Sans without a system
    install. Paths go through :func:`escape_filter_path` (Windows drive colons
    and backslashes).
    """
    if force_style is None and subtitle_path.lower().endswith(".srt"):
        force_style = default_force_style()

    vf = f"subtitles=filename={escape_filter_path(subtitle_path)}"
    if fonts_dir:
        vf += f":fontsdir={escape_filter_path(fonts_dir)}"
    if force_style:
        vf += f":force_style={quote_filter_value(force_style)}"

    return [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        vf,
        *_H264_ARGS,
        *_ENCODE_QUALITY_ARGS,
        "-c:a",
        "copy",
        *_FASTSTART_ARGS,
        output_path,
    ]


# ---------------------------------------------------------------------------
# Thumbnail and platform derivatives
# ---------------------------------------------------------------------------


def thumbnail_cmd(
    input_path: str,
    output_path: str,
    *,
    at_seconds: float = 1.0,
    overlay_text: str | None = None,
    font_file: str | None = None,
) -> list[str]:
    """Extract one frame as a thumbnail, optionally overlaying centered text.

    ``expansion=none`` disables drawtext ``%`` expansion so overlay text is
    always literal; text goes through :func:`escape_drawtext_text`.
    """
    if at_seconds < 0:
        raise ValueError("at_seconds must be >= 0")
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        _fmt(at_seconds),
        "-i",
        input_path,
        "-frames:v",
        "1",
    ]
    if overlay_text:
        draw = (
            f"drawtext=text={escape_drawtext_text(overlay_text)}"
            ":expansion=none"
            ":fontcolor=white:fontsize=88:borderw=6:bordercolor=black"
            ":x=(w-text_w)/2"
            f":y=h-{SAFE_BOTTOM_MARGIN}-text_h"
        )
        if font_file:
            draw += f":fontfile={escape_filter_path(font_file)}"
        cmd += ["-vf", draw]
    cmd += ["-q:v", "2", output_path]
    return cmd


@dataclass(frozen=True)
class PlatformProfile:
    """Per-platform derivative spec. The default profile is the master itself."""

    name: str = "default"
    width: int = TARGET_WIDTH
    height: int = TARGET_HEIGHT
    fps: int = TARGET_FPS
    video_codec: str = "libx264"
    video_profile: str = "high"
    pix_fmt: str = "yuv420p"
    audio_codec: str = "aac"
    audio_sample_rate: int = AUDIO_SAMPLE_RATE
    audio_bitrate: str = AUDIO_BITRATE
    crf: int = VIDEO_CRF
    # MVP: every platform ships the master unchanged; flip this per platform
    # later (e.g. bitrate caps or different resolutions) without touching callers.
    identical_to_master: bool = True


DEFAULT_PROFILE = PlatformProfile()


def platform_derivative_cmd(
    master_path: str,
    output_path: str,
    profile: PlatformProfile = DEFAULT_PROFILE,
) -> list[str]:
    """Produce a platform derivative from the finished rendition.

    Default profile stream-copies (byte-identical streams, fresh faststart
    moov). Non-identical profiles re-encode with scale/pad to the profile's
    geometry — the hook for future per-platform tweaks.
    """
    if profile.identical_to_master:
        return ["ffmpeg", "-y", "-i", master_path, "-c", "copy", *_FASTSTART_ARGS, output_path]

    vf = (
        f"scale={profile.width}:{profile.height}:force_original_aspect_ratio=decrease,"
        f"pad={profile.width}:{profile.height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={profile.fps},setsar=1"
    )
    return [
        "ffmpeg",
        "-y",
        "-i",
        master_path,
        "-vf",
        vf,
        "-c:v",
        profile.video_codec,
        "-profile:v",
        profile.video_profile,
        "-pix_fmt",
        profile.pix_fmt,
        "-preset",
        VIDEO_PRESET,
        "-crf",
        str(profile.crf),
        "-c:a",
        profile.audio_codec,
        "-ar",
        str(profile.audio_sample_rate),
        "-b:a",
        profile.audio_bitrate,
        *_FASTSTART_ARGS,
        output_path,
    ]
