"""Contract tests for the Groq free-tier adapters (offline via httpx.MockTransport).

Covers: sentence chunking under the 200-char Orpheus limit, WAV concatenation
with the stdlib wave module, the TTS/Whisper/LLM HTTP contracts (success,
429-then-success, persistent 500, non-retryable 400, Vietnamese rejection),
zero-dollar cost events with real units, and the Whisper word-timestamp ->
captions timepoint adapter proven end-to-end against the real SRT builder.
"""

from __future__ import annotations

import io
import json
import wave

import httpx
import pytest
from conftest import make_video_plan
from sqlalchemy.orm import Session

from app.ai.base import ScriptProvider, SourceInfo, TTSProvider
from app.ai.groq_providers import (
    GroqScriptProvider,
    GroqTTSProvider,
    GroqWhisperAligner,
    WordTimestamp,
    chunk_text,
    concat_wav,
    scene_mark_timepoints,
    strip_ssml,
)
from app.ai.tts import build_ssml
from app.config import ModelConfig
from app.costs import CostLedger
from app.errors import UpstreamError, ValidationFailed
from app.media.captions import build_srt, events_from_timepoints
from app.models import CostEvent, Creative
from app.schemas.videoplan import VideoPlan

CFG = ModelConfig()
SAMPLE_RATE = 24000


def make_wav(seconds: float, *, rate: int = SAMPLE_RATE, channels: int = 1, width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(b"\x00" * (width * channels * int(rate * seconds)))
    return buffer.getvalue()


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def tts_provider(handler, **kwargs) -> GroqTTSProvider:
    return GroqTTSProvider(
        api_key="test-key",
        model_config=CFG,
        http_client=make_client(handler),
        sleep=lambda _s: None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Sentence chunking (<= groq_tts_max_chars, never mid-word)
# ---------------------------------------------------------------------------


class TestChunking:
    def test_short_sentences_become_one_chunk_each(self) -> None:
        assert chunk_text("First sentence here. Second one!", 200) == [
            "First sentence here.",
            "Second one!",
        ]

    def test_chunks_respect_limit_without_midword_splits(self) -> None:
        text = " ".join(f"word{i}" for i in range(120)) + "."
        chunks = chunk_text(text, CFG.groq_tts_max_chars)
        assert len(chunks) > 1
        assert all(len(chunk) <= CFG.groq_tts_max_chars for chunk in chunks)
        # Re-joining reproduces every word intact and in order.
        assert " ".join(chunks).split() == text.split()

    def test_long_sentence_splits_on_commas_first(self) -> None:
        clause = "a" * 80
        sentence = f"{clause}, {clause}, {clause}."
        chunks = chunk_text(sentence, 200)
        assert chunks == [f"{clause}, {clause},", f"{clause}."]
        assert all(len(chunk) <= 200 for chunk in chunks)

    def test_long_sentence_without_commas_splits_on_spaces(self) -> None:
        sentence = " ".join(["abcdefghij"] * 50) + "."
        chunks = chunk_text(sentence, 200)
        assert len(chunks) > 1
        assert all(len(chunk) <= 200 for chunk in chunks)
        assert " ".join(chunks).split() == sentence.split()

    def test_single_giant_word_hard_splits_as_last_resort(self) -> None:
        word = "x" * 450
        chunks = chunk_text(word + ".", 200)
        assert all(len(chunk) <= 200 for chunk in chunks)
        assert "".join(chunks) == word + "."

    def test_strip_ssml_recovers_plain_text(self) -> None:
        text = "Hello world. Goodbye now."
        assert strip_ssml(build_ssml(text)) == text


# ---------------------------------------------------------------------------
# WAV concatenation via the stdlib wave module
# ---------------------------------------------------------------------------


class TestWavConcat:
    def test_concat_preserves_frames_rate_and_duration(self) -> None:
        data, duration = concat_wav([make_wav(0.1), make_wav(0.05)])
        with wave.open(io.BytesIO(data), "rb") as reader:
            assert reader.getframerate() == SAMPLE_RATE
            assert reader.getnchannels() == 1
            assert reader.getsampwidth() == 2
            assert reader.getnframes() == int(SAMPLE_RATE * 0.15)
        assert duration == pytest.approx(0.15)

    def test_empty_input_yields_empty_audio(self) -> None:
        assert concat_wav([]) == (b"", 0.0)

    def test_sample_rate_mismatch_raises_upstream_error(self) -> None:
        with pytest.raises(UpstreamError, match="mismatched"):
            concat_wav([make_wav(0.1), make_wav(0.1, rate=22050)])

    def test_sample_width_mismatch_raises_upstream_error(self) -> None:
        with pytest.raises(UpstreamError, match="mismatched"):
            concat_wav([make_wav(0.1), make_wav(0.1, width=1)])

    def test_invalid_wav_bytes_raise_upstream_error(self) -> None:
        with pytest.raises(UpstreamError, match="invalid WAV"):
            concat_wav([b"not a wav at all"])


# ---------------------------------------------------------------------------
# GroqTTSProvider HTTP contract
# ---------------------------------------------------------------------------


class TestGroqTTSProvider:
    def test_success_one_request_per_sentence_chunk(self) -> None:
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(
                {"json": json.loads(request.content), "auth": request.headers.get("Authorization")}
            )
            return httpx.Response(200, content=make_wav(0.5))

        provider = tts_provider(handler)
        assert isinstance(provider, TTSProvider)
        text = "Hello world. Goodbye now."
        result = provider.synthesize(
            ssml=build_ssml(text), voice="en-US-Neural2-F", language_code="en-US"
        )

        assert len(requests) == 2  # one request per sentence chunk
        assert requests[0]["json"]["input"] == "Hello world."
        assert requests[1]["json"]["input"] == "Goodbye now."
        for seen in requests:
            assert seen["json"]["model"] == CFG.groq_tts_model_en
            assert seen["json"]["voice"] == CFG.groq_tts_voice_en
            assert seen["json"]["response_format"] == "wav"
            assert len(seen["json"]["input"]) <= CFG.groq_tts_max_chars
            assert seen["auth"] == "Bearer test-key"

        # Duration measured from WAV frames of the concatenated segments.
        assert result.duration_seconds == pytest.approx(1.0)
        with wave.open(io.BytesIO(result.audio_bytes), "rb") as reader:
            assert reader.getnframes() == SAMPLE_RATE  # 1.0 s at 24 kHz mono
        marks = {t.mark_name: t.seconds for t in result.timepoints}
        assert marks["s0"] == pytest.approx(0.5)
        assert marks["s1"] == pytest.approx(1.0)
        assert marks["end"] == pytest.approx(1.0)
        assert result.voice == CFG.groq_tts_voice_en
        assert result.audio_mime_type == "audio/wav"

    def test_retries_429_then_succeeds(self) -> None:
        calls = {"n": 0}
        sleeps: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, json={"error": {"message": "rate limited"}})
            return httpx.Response(200, content=make_wav(0.2))

        provider = GroqTTSProvider(
            api_key="test-key", model_config=CFG, http_client=make_client(handler),
            sleep=sleeps.append,
        )
        result = provider.synthesize(
            ssml=build_ssml("Hi there."), voice="v", language_code="en-US"
        )
        assert calls["n"] == 2
        assert len(sleeps) == 1  # one backoff sleep between the attempts
        assert result.duration_seconds == pytest.approx(0.2)

    def test_persistent_500_exhausts_retries_and_raises_retryable(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(500, json={"error": "boom"})

        provider = tts_provider(handler)
        with pytest.raises(UpstreamError) as excinfo:
            provider.synthesize(ssml=build_ssml("Hi."), voice="v", language_code="en-US")
        assert excinfo.value.retryable is True
        assert calls["n"] == 3  # default max_attempts

    def test_400_is_not_retried(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json={"error": "bad request"})

        provider = tts_provider(handler)
        with pytest.raises(UpstreamError) as excinfo:
            provider.synthesize(ssml=build_ssml("Hi."), voice="v", language_code="en-US")
        assert excinfo.value.retryable is False
        assert calls["n"] == 1

    def test_vietnamese_rejected_before_any_http_call(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no HTTP request expected for locale vi")

        provider = tts_provider(handler)
        with pytest.raises(ValidationFailed, match="Vietnamese"):
            provider.synthesize(
                ssml=build_ssml("Xin chào."), voice="vi-VN-Neural2-A", language_code="vi-VN"
            )

    def test_records_zero_cost_event_with_char_units(
        self, db_session: Session, creative: Creative
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=make_wav(0.3))

        provider = tts_provider(handler)
        provider.bind_cost_ledger(CostLedger(db_session), creative.id)
        text = "Hello world. Bye."
        provider.synthesize(ssml=build_ssml(text), voice="v", language_code="en-US")

        events = (
            db_session.query(CostEvent)
            .filter(CostEvent.creative_id == creative.id, CostEvent.kind == "groq_tts")
            .all()
        )
        assert len(events) == 1
        assert events[0].amount_usd == 0.0
        assert events[0].unit_price_usd == 0.0
        assert events[0].units == float(len(text))
        assert events[0].model_id == CFG.groq_tts_model_en
        assert "free-tier" in events[0].note


# ---------------------------------------------------------------------------
# GroqWhisperAligner -> captions timepoints -> SRT (end to end)
# ---------------------------------------------------------------------------

NARRATIONS = ["Hello world today.", "Goodbye now."]
WHISPER_PAYLOAD = {
    "text": "Hello world today. Goodbye now.",
    "duration": 3.0,
    "words": [
        {"word": "Hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.4, "end": 0.8},
        {"word": "today", "start": 0.8, "end": 1.2},
        {"word": "Goodbye", "start": 2.0, "end": 2.4},
        {"word": "now", "start": 2.4, "end": 2.8},
    ],
}


def make_aligner(handler, **kwargs) -> GroqWhisperAligner:
    return GroqWhisperAligner(
        api_key="test-key",
        model_config=CFG,
        http_client=make_client(handler),
        sleep=lambda _s: None,
        **kwargs,
    )


class TestGroqWhisperAligner:
    def test_transcribe_words_sends_multipart_contract_and_parses_words(
        self, db_session: Session, creative: Creative
    ) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["content"] = request.content
            return httpx.Response(200, json=WHISPER_PAYLOAD)

        aligner = make_aligner(handler)
        aligner.bind_cost_ledger(CostLedger(db_session), creative.id)
        transcription = aligner.transcribe_words(b"RIFF-fake-audio", filename="mix.wav")

        assert captured["url"].endswith("/audio/transcriptions")
        body = captured["content"]
        assert CFG.groq_whisper_model.encode() in body
        assert b"verbose_json" in body
        assert b"timestamp_granularities" in body and b"word" in body
        assert b"mix.wav" in body and b"RIFF-fake-audio" in body

        assert [w.word for w in transcription.words] == [
            "Hello", "world", "today", "Goodbye", "now",
        ]
        assert transcription.words[3].start == pytest.approx(2.0)
        assert transcription.duration_seconds == pytest.approx(3.0)

        events = (
            db_session.query(CostEvent)
            .filter(CostEvent.creative_id == creative.id, CostEvent.kind == "groq_whisper")
            .all()
        )
        assert len(events) == 1
        assert events[0].amount_usd == 0.0
        assert events[0].units == pytest.approx(3.0)  # seconds of whisper audio
        assert "free-tier" in events[0].note

    def test_scene_mark_timepoints_match_captions_contract(self) -> None:
        words = tuple(
            WordTimestamp(word=w["word"], start=w["start"], end=w["end"])
            for w in WHISPER_PAYLOAD["words"]
        )
        timepoints = scene_mark_timepoints(words, NARRATIONS)
        # Exactly the {"s0": start, ...} mapping events_from_timepoints consumes.
        assert timepoints == {"s0": 0.0, "s1": 2.0}

    def test_caption_timepoints_feed_real_srt_builder_end_to_end(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=WHISPER_PAYLOAD)

        aligner = make_aligner(handler)
        timepoints = aligner.caption_timepoints(b"RIFF-fake-audio", NARRATIONS)
        events = events_from_timepoints(NARRATIONS, timepoints, total_duration=3.0)
        srt = build_srt(events)
        assert "1\n00:00:00,000 --> 00:00:02,000\nHello world today." in srt
        assert "2\n00:00:02,000 --> 00:00:03,000\nGoodbye now." in srt

    def test_alignment_with_empty_inputs_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot align"):
            scene_mark_timepoints((), NARRATIONS)
        words = (WordTimestamp("hi", 0.0, 0.4),)
        with pytest.raises(ValueError, match="cannot align"):
            scene_mark_timepoints(words, ["", "  "])


# ---------------------------------------------------------------------------
# GroqScriptProvider (LLM fallback, ScriptProvider protocol)
# ---------------------------------------------------------------------------


class TestGroqScriptProvider:
    def make_provider(self, handler, **kwargs) -> GroqScriptProvider:
        return GroqScriptProvider(
            api_key="test-key",
            model_config=CFG,
            http_client=make_client(handler),
            sleep=lambda _s: None,
            **kwargs,
        )

    def test_generate_plan_json_object_contract(self) -> None:
        plan = make_video_plan()
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["json"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": json.dumps(plan)}}],
                    "usage": {"total_tokens": 1234},
                },
            )

        provider = self.make_provider(handler)
        assert isinstance(provider, ScriptProvider)  # drop-in for GeminiScriptProvider
        sources = [SourceInfo(url="https://example.com/post", title="Post")]
        result = provider.generate_plan("New AI model launch", sources, "brief")

        assert result == plan
        VideoPlan.model_validate(result)  # the returned dict is a valid VideoPlan v1
        payload = captured["json"]
        assert payload["model"] == CFG.groq_llm_model
        assert payload["response_format"] == {"type": "json_object"}
        assert "New AI model launch" in payload["messages"][0]["content"]
        assert "https://example.com/post" in payload["messages"][0]["content"]

    def test_records_zero_cost_event_with_token_units(
        self, db_session: Session, creative: Creative
    ) -> None:
        plan = make_video_plan()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": json.dumps(plan)}}],
                    "usage": {"total_tokens": 1234},
                },
            )

        provider = self.make_provider(handler)
        provider.bind_cost_ledger(CostLedger(db_session), creative.id)
        provider.generate_plan("Topic", [], "brief")

        events = (
            db_session.query(CostEvent)
            .filter(CostEvent.creative_id == creative.id, CostEvent.kind == "groq_llm")
            .all()
        )
        assert len(events) == 1
        assert events[0].amount_usd == 0.0
        assert events[0].units == 1234.0
        assert events[0].model_id == CFG.groq_llm_model
        assert "free-tier" in events[0].note

    def test_missing_usage_falls_back_to_char_estimate(
        self, db_session: Session, creative: Creative
    ) -> None:
        content = json.dumps({"ok": True})

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        provider = self.make_provider(handler)
        provider.bind_cost_ledger(CostLedger(db_session), creative.id)
        provider.generate_plan("Topic", [], "brief")
        event = (
            db_session.query(CostEvent)
            .filter(CostEvent.creative_id == creative.id, CostEvent.kind == "groq_llm")
            .one()
        )
        assert event.units > 0  # ~4 chars/token estimate
        assert event.amount_usd == 0.0

    def test_shorten_narrations_posts_shorten_prompt(self) -> None:
        plan = make_video_plan()
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["json"] = json.loads(request.content)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(plan)}}]}
            )

        provider = self.make_provider(handler)
        issues = [{"locale": "vi", "scene_index": 2, "message": "too long"}]
        result = provider.shorten_narrations(plan, issues)
        assert result == plan
        prompt = captured["json"]["messages"][0]["content"]
        assert "Shorten" in prompt
        assert "scene=2" in prompt

    def test_empty_choices_raise_retryable_upstream_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        provider = self.make_provider(handler)
        with pytest.raises(UpstreamError) as excinfo:
            provider.generate_plan("Topic", [], "brief")
        assert excinfo.value.retryable is True
