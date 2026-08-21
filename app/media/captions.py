"""Caption generation: SRT from TTS timepoints plus a styled ASS for burn-in.

TTS (Google Cloud Text-to-Speech) returns SSML mark timepoints as
``mark name -> seconds``; the convention is one mark per scene named
``s0..s4`` placed at the start of each scene's narration. Captions are
segmented into blocks of at most ``MAX_LINES`` lines of ``MAX_LINE_CHARS``
characters and distributed across each scene's time span proportionally to
text length.

Safe zone (PLAN.md): keep text inside 1080x1920 minus 220 px top and 320 px
bottom margins. With ``PlayResY=1920`` and ``Alignment=2`` (bottom-center),
``MarginV=320`` keeps burned-in captions clear of platform UI chrome.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MAX_LINE_CHARS = 42
MAX_LINES = 2

PLAY_RES_X = 1080
PLAY_RES_Y = 1920
SAFE_TOP_MARGIN = 220
SAFE_BOTTOM_MARGIN = 320
FONT_NAME = "Noto Sans"
DEFAULT_FONT_SIZE = 54
DEFAULT_MARGIN_L = 60
DEFAULT_MARGIN_R = 60

DEFAULT_MASTER_DURATION = 38.8
DEFAULT_MARK_PREFIX = "s"

ASS_STYLE_NAME = "Caption"


@dataclass(frozen=True)
class CaptionEvent:
    """One caption block: [start, end) seconds and 1..MAX_LINES display lines."""

    start: float
    end: float
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def wrap_caption_lines(text: str, max_chars: int = MAX_LINE_CHARS) -> list[str]:
    """Greedy word-wrap into lines of at most ``max_chars`` characters.

    Words longer than ``max_chars`` are hard-split so no line ever exceeds the
    limit (URLs, hashtags, agglutinated words).
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    lines: list[str] = []
    current = ""
    for word in text.split():
        if len(word) > max_chars:
            if current:
                lines.append(current)
                current = ""
            while len(word) > max_chars:
                lines.append(word[:max_chars])
                word = word[max_chars:]
            current = word
            continue
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def segment_caption_blocks(
    text: str, max_chars: int = MAX_LINE_CHARS, max_lines: int = MAX_LINES
) -> list[tuple[str, ...]]:
    """Split text into caption blocks of at most ``max_lines`` wrapped lines."""
    if max_lines < 1:
        raise ValueError("max_lines must be >= 1")
    lines = wrap_caption_lines(text, max_chars)
    return [tuple(lines[i : i + max_lines]) for i in range(0, len(lines), max_lines)]


def events_from_timepoints(
    narrations: Sequence[str],
    timepoints: Mapping[str, float],
    *,
    total_duration: float = DEFAULT_MASTER_DURATION,
    mark_prefix: str = DEFAULT_MARK_PREFIX,
    max_chars: int = MAX_LINE_CHARS,
    max_lines: int = MAX_LINES,
) -> list[CaptionEvent]:
    """Build caption events from per-scene narrations and TTS mark timepoints.

    Scene ``i`` spans ``timepoints[f"{prefix}{i}"]`` up to the next scene's mark
    (or ``total_duration`` for the last scene). Blocks inside a scene share the
    span proportionally to their character count, staying contiguous.
    """
    events: list[CaptionEvent] = []
    for i, narration in enumerate(narrations):
        text = " ".join(narration.split())
        if not text:
            continue
        mark = f"{mark_prefix}{i}"
        if mark not in timepoints:
            raise KeyError(f"missing TTS timepoint mark {mark!r}")
        start = float(timepoints[mark])
        end = float(timepoints.get(f"{mark_prefix}{i + 1}", total_duration))
        if end <= start:
            raise ValueError(f"scene {i} has non-positive span: start={start}, end={end}")
        blocks = segment_caption_blocks(text, max_chars, max_lines)
        weights = [sum(len(line) for line in block) for block in blocks]
        total_weight = sum(weights)
        span = end - start
        cursor = start
        cumulative = 0
        for block, weight in zip(blocks, weights, strict=True):
            cumulative += weight
            block_end = start + span * (cumulative / total_weight)
            events.append(
                CaptionEvent(start=round(cursor, 3), end=round(block_end, 3), lines=block)
            )
            cursor = block_end
    return events


def format_srt_timestamp(seconds: float) -> str:
    """``HH:MM:SS,mmm``; negative inputs clamp to zero."""
    ms_total = max(0, round(seconds * 1000))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_ass_timestamp(seconds: float) -> str:
    """``H:MM:SS.cc`` (centiseconds); negative inputs clamp to zero."""
    cs_total = max(0, round(seconds * 100))
    hours, rem = divmod(cs_total, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, cs = divmod(rem, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"


def build_srt(events: Sequence[CaptionEvent]) -> str:
    """Render numbered SRT blocks; empty input yields an empty string."""
    blocks: list[str] = []
    for number, event in enumerate(events, start=1):
        timing = f"{format_srt_timestamp(event.start)} --> {format_srt_timestamp(event.end)}"
        blocks.append(f"{number}\n{timing}\n" + "\n".join(event.lines))
    return "\n\n".join(blocks) + "\n" if blocks else ""


def ass_style_block(
    *,
    font_name: str = FONT_NAME,
    font_size: int = DEFAULT_FONT_SIZE,
    play_res_x: int = PLAY_RES_X,
    play_res_y: int = PLAY_RES_Y,
    margin_l: int = DEFAULT_MARGIN_L,
    margin_r: int = DEFAULT_MARGIN_R,
    margin_v: int = SAFE_BOTTOM_MARGIN,
) -> str:
    """[Script Info] + [V4+ Styles] header keeping captions in the safe zone.

    Alignment=2 is bottom-center; ``margin_v`` (px from the bottom at PlayRes
    scale) enforces the 320 px bottom safe margin from PLAN.md.
    """
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        "ScaledBorderAndShadow: yes\n"
        "WrapStyle: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: {ASS_STYLE_NAME},{font_name},{font_size},"
        "&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        "0,0,0,0,100,100,0,0,1,2,1,2,"
        f"{margin_l},{margin_r},{margin_v},1\n"
    )


def _sanitize_ass_text(text: str) -> str:
    """Neutralize ASS override blocks; braces would otherwise inject style tags."""
    return text.replace("{", "(").replace("}", ")")


def build_ass(
    events: Sequence[CaptionEvent],
    *,
    font_name: str = FONT_NAME,
    font_size: int = DEFAULT_FONT_SIZE,
) -> str:
    """Full ASS document (style block + Dialogue events) for subtitle burn-in."""
    lines = [ass_style_block(font_name=font_name, font_size=font_size)]
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")
    for event in events:
        text = "\\N".join(_sanitize_ass_text(line) for line in event.lines)
        lines.append(
            f"Dialogue: 0,{format_ass_timestamp(event.start)},"
            f"{format_ass_timestamp(event.end)},{ASS_STYLE_NAME},,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"
