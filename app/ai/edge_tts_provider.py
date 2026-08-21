"""Vietnamese free-tier TTS through edge-tts (unofficial Microsoft endpoint).

PLAN.md "Chế độ miễn phí": Groq has no Vietnamese voice, so VI narration uses
``edge-tts`` (default voice ``ModelConfig.edge_tts_voice_vi``, no API key).
The endpoint is unofficial, which is why this stays behind the ``TTSProvider``
protocol — Google Cloud TTS can be swapped back in via configuration.

edge-tts is async-only; the sync ``synthesize`` bridges with ``asyncio.run``
(or a dedicated thread when an event loop is already running). Alongside the
audio chunks the stream yields ``WordBoundary`` events whose ``offset`` /
``duration`` are in 100-nanosecond ticks; they are converted to seconds and
folded into the same per-sentence ``s0..sN`` + ``end`` mark timepoints that
``app.ai.tts.build_ssml`` produces, so VI captions get real timestamps
without a Whisper pass.

The network call is injectable: ``communicate_factory(text, voice)`` returns
any object with an ``async stream()`` generator. Tests replace it with a fake
and never import ``edge_tts`` (the real import happens lazily inside the
default factory only).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from app.ai.base import Timepoint, TTSResult
from app.ai.groq_providers import strip_ssml
from app.ai.tts import END_MARK, split_sentences
from app.config import ModelConfig, get_model_config

#: edge-tts has no configured EN voice in ModelConfig (Groq covers EN); this is
#: the fallback when the operator selects ``tts_provider_en=edge`` anyway.
DEFAULT_VOICE_EN = "en-US-AriaNeural"

TICKS_PER_SECOND = 10_000_000  # WordBoundary offsets/durations are 100 ns ticks
AUDIO_MIME_TYPE = "audio/mpeg"  # edge-tts default output is MP3


def ticks_to_seconds(ticks: float | int) -> float:
    """Convert a 100-nanosecond tick count to seconds."""
    return float(ticks) / TICKS_PER_SECOND


@dataclass(frozen=True)
class WordBoundary:
    """One spoken word with start/end in seconds (converted from ticks)."""

    text: str
    start_seconds: float
    end_seconds: float


def _default_communicate_factory(text: str, voice: str) -> Any:
    import edge_tts  # noqa: PLC0415 — lazy: tests inject a fake factory instead

    return edge_tts.Communicate(text, voice, boundary="WordBoundary")


def sentence_timepoints(
    text: str, boundaries: Sequence[WordBoundary]
) -> tuple[Timepoint, ...]:
    """Per-sentence mark timepoints (``s0..sN`` + ``end``) from word boundaries.

    Mark ``s{i}`` lands on the end of sentence ``i``'s last word — the same
    convention as SSML mark timepointing. Words are attributed to sentences
    proportionally to each sentence's word count, which is exact when edge-tts
    emits one boundary per word and degrades gracefully otherwise.
    """
    if not boundaries:
        return ()
    counts = [len(sentence.split()) for sentence in split_sentences(text)]
    total = sum(counts)
    points: list[Timepoint] = []
    if total > 0:
        n = len(boundaries)
        cumulative = 0
        for i, count in enumerate(counts):
            cumulative += count
            index = max(0, min(n, (n * cumulative) // total) - 1)
            points.append(Timepoint(f"s{i}", round(boundaries[index].end_seconds, 4)))
    points.append(Timepoint(END_MARK, round(boundaries[-1].end_seconds, 4)))
    return tuple(points)


class EdgeTTSProvider:
    """``TTSProvider`` over edge-tts with word-boundary caption timepoints.

    The ``voice`` argument passed by ``TTSService`` (a Google voice name) is
    ignored; the voice configured at construction time is used and reported
    back in the result.
    """

    def __init__(
        self,
        *,
        voice: str | None = None,
        model_config: ModelConfig | None = None,
        communicate_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        cfg = model_config or get_model_config()
        self._voice = voice or cfg.edge_tts_voice_vi
        self._communicate_factory = communicate_factory or _default_communicate_factory

    @property
    def voice(self) -> str:
        return self._voice

    def synthesize(self, *, ssml: str, voice: str, language_code: str) -> TTSResult:
        text = strip_ssml(ssml)
        audio, boundaries = self._run_async(self._collect_stream(text))
        timepoints = sentence_timepoints(text, boundaries)
        duration = round(boundaries[-1].end_seconds, 4) if boundaries else 0.0
        return TTSResult(
            audio_bytes=audio,
            timepoints=timepoints,
            duration_seconds=duration,
            voice=self._voice,
            audio_mime_type=AUDIO_MIME_TYPE,
        )

    async def _collect_stream(self, text: str) -> tuple[bytes, list[WordBoundary]]:
        communicate = self._communicate_factory(text, self._voice)
        audio = bytearray()
        boundaries: list[WordBoundary] = []
        async for chunk in communicate.stream():
            kind = chunk.get("type")
            if kind == "audio":
                audio.extend(chunk.get("data") or b"")
            elif kind == "WordBoundary":
                offset = float(chunk.get("offset", 0))
                duration = float(chunk.get("duration", 0))
                boundaries.append(
                    WordBoundary(
                        text=str(chunk.get("text", "")),
                        start_seconds=ticks_to_seconds(offset),
                        end_seconds=ticks_to_seconds(offset + duration),
                    )
                )
        return bytes(audio), boundaries

    @staticmethod
    def _run_async(coro: Any) -> Any:
        """Bridge async edge-tts into the sync provider protocol.

        ``asyncio.run`` when no loop is running; otherwise the coroutine runs
        on its own loop in a dedicated thread (``asyncio.run`` would raise
        inside a running loop).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
