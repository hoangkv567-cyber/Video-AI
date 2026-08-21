"""EdgeTTSProvider tests using a fake Communicate factory.

No network and no edge_tts import: the real module is poisoned in sys.modules
for the duration of each test, so any accidental use of the default factory
(the only place edge_tts is imported) would raise immediately.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from app.ai.base import TTSProvider
from app.ai.edge_tts_provider import (
    DEFAULT_VOICE_EN,
    TICKS_PER_SECOND,
    EdgeTTSProvider,
    WordBoundary,
    sentence_timepoints,
    ticks_to_seconds,
)
from app.ai.tts import build_ssml
from app.config import ModelConfig

CFG = ModelConfig()

TEXT = "Xin chào bạn. Hẹn gặp lại."
WORDS = ["Xin", "chào", "bạn", "Hẹn", "gặp", "lại"]
AUDIO_PARTS = [b"\x01\x02\x03", b"\x04\x05", b"\x06\x07\x08\x09"]

WORD_TICKS = 5_000_000  # 0.5 s between word starts
WORD_DURATION_TICKS = 4_000_000  # each word lasts 0.4 s


def make_chunks() -> list[dict[str, Any]]:
    """Interleave audio chunks with WordBoundary events (offsets in 100 ns ticks)."""
    chunks: list[dict[str, Any]] = [{"type": "audio", "data": AUDIO_PARTS[0]}]
    for i, word in enumerate(WORDS):
        chunks.append(
            {
                "type": "WordBoundary",
                "offset": i * WORD_TICKS,
                "duration": WORD_DURATION_TICKS,
                "text": word,
            }
        )
    chunks.insert(3, {"type": "audio", "data": AUDIO_PARTS[1]})
    chunks.append({"type": "audio", "data": AUDIO_PARTS[2]})
    return chunks


class FakeCommunicate:
    """Stands in for edge_tts.Communicate: streams canned chunks, no network."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks

    async def stream(self):
        for chunk in self._chunks:
            yield chunk


def make_factory(calls: list[tuple[str, str]], chunks: list[dict[str, Any]] | None = None):
    def factory(text: str, voice: str) -> FakeCommunicate:
        calls.append((text, voice))
        return FakeCommunicate(chunks if chunks is not None else make_chunks())

    return factory


@pytest.fixture(autouse=True)
def _no_real_edge_tts(monkeypatch: pytest.MonkeyPatch):
    """Poison the edge_tts import: the default (network) factory would blow up."""
    monkeypatch.setitem(sys.modules, "edge_tts", None)


def test_tick_conversion() -> None:
    assert TICKS_PER_SECOND == 10_000_000
    assert ticks_to_seconds(10_000_000) == 1.0
    assert ticks_to_seconds(5_000_000) == 0.5
    assert ticks_to_seconds(0) == 0.0


def test_synthesize_assembles_audio_and_word_boundary_timepoints() -> None:
    calls: list[tuple[str, str]] = []
    provider = EdgeTTSProvider(
        voice="vi-VN-HoaiMyNeural", model_config=CFG, communicate_factory=make_factory(calls)
    )
    assert isinstance(provider, TTSProvider)

    result = provider.synthesize(
        ssml=build_ssml(TEXT), voice="vi-VN-Neural2-A", language_code="vi-VN"
    )

    # SSML was stripped to plain text and the configured voice was used
    # (not the Google voice name TTSService passes through).
    assert calls == [(TEXT, "vi-VN-HoaiMyNeural")]
    # Audio chunks are concatenated in stream order.
    assert result.audio_bytes == b"".join(AUDIO_PARTS)
    assert result.audio_mime_type == "audio/mpeg"
    assert result.voice == "vi-VN-HoaiMyNeural"

    # Ticks -> seconds: sentence 1 ends with word 3 (offset 2*0.5s + 0.4s dur),
    # sentence 2 (and the audio) ends with word 6 at 5*0.5s + 0.4s.
    marks = {t.mark_name: t.seconds for t in result.timepoints}
    assert marks["s0"] == pytest.approx(1.4)
    assert marks["s1"] == pytest.approx(2.9)
    assert marks["end"] == pytest.approx(2.9)
    assert result.duration_seconds == pytest.approx(2.9)


def test_default_voice_comes_from_model_config() -> None:
    calls: list[tuple[str, str]] = []
    provider = EdgeTTSProvider(model_config=CFG, communicate_factory=make_factory(calls))
    assert provider.voice == CFG.edge_tts_voice_vi == "vi-VN-HoaiMyNeural"
    provider.synthesize(ssml="<speak>Chào.</speak>", voice="x", language_code="vi-VN")
    assert calls[0][1] == "vi-VN-HoaiMyNeural"
    assert DEFAULT_VOICE_EN.startswith("en-")


def test_no_boundaries_yields_empty_timepoints_and_zero_duration() -> None:
    calls: list[tuple[str, str]] = []
    chunks = [{"type": "audio", "data": b"\xaa\xbb"}]
    provider = EdgeTTSProvider(
        voice="vi-VN-HoaiMyNeural", model_config=CFG,
        communicate_factory=make_factory(calls, chunks),
    )
    result = provider.synthesize(ssml=build_ssml("Chào."), voice="x", language_code="vi-VN")
    assert result.audio_bytes == b"\xaa\xbb"
    assert result.timepoints == ()
    assert result.duration_seconds == 0.0


async def test_synthesize_inside_running_event_loop_uses_dedicated_thread() -> None:
    """asyncio_mode=auto runs this in a loop; the sync bridge must still work."""
    calls: list[tuple[str, str]] = []
    provider = EdgeTTSProvider(
        voice="vi-VN-HoaiMyNeural", model_config=CFG, communicate_factory=make_factory(calls)
    )
    result = provider.synthesize(ssml=build_ssml(TEXT), voice="x", language_code="vi-VN")
    assert result.audio_bytes == b"".join(AUDIO_PARTS)
    assert result.duration_seconds == pytest.approx(2.9)
    assert calls == [(TEXT, "vi-VN-HoaiMyNeural")]


def test_sentence_timepoints_proportional_fallback_on_count_mismatch() -> None:
    # 2 sentences of 3 narration words each, but only 4 transcribed boundaries:
    # the mapping degrades proportionally and stays monotonic within bounds.
    boundaries = [
        WordBoundary(text=f"w{i}", start_seconds=i * 0.5, end_seconds=i * 0.5 + 0.4)
        for i in range(4)
    ]
    points = {t.mark_name: t.seconds for t in sentence_timepoints(TEXT, boundaries)}
    assert set(points) == {"s0", "s1", "end"}
    assert points["s0"] <= points["s1"] == points["end"] == pytest.approx(1.9)


def test_real_edge_tts_never_imported() -> None:
    calls: list[tuple[str, str]] = []
    provider = EdgeTTSProvider(
        voice="vi-VN-HoaiMyNeural", model_config=CFG, communicate_factory=make_factory(calls)
    )
    provider.synthesize(ssml=build_ssml("Chào bạn."), voice="x", language_code="vi-VN")
    # The poisoned sys.modules entry is untouched: nothing imported edge_tts.
    assert sys.modules["edge_tts"] is None
