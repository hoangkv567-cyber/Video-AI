"""Groq free-tier adapters (PLAN.md "Chế độ miễn phí").

Three OpenAI-compatible endpoints on ``https://api.groq.com/openai/v1``:

- ``GroqTTSProvider`` — Orpheus English TTS. The ~200-char per-request input
  limit (``ModelConfig.groq_tts_max_chars``) means narration is split into
  per-sentence chunks (never mid-word; a single over-long sentence falls back
  to comma then space splits), one request per chunk; the returned WAV
  segments are validated and concatenated with the stdlib ``wave`` module and
  duration is measured from WAV frames. Because every sentence is synthesized
  separately, the ``s0..sN`` + ``end`` mark timepoints (same convention as
  ``app.ai.tts.build_ssml``) are exact. English only — Groq has no Vietnamese
  voice; ``"vi"`` raises immediately.
- ``GroqWhisperAligner`` — word timestamps via ``whisper-large-v3``
  (``response_format=verbose_json`` + ``timestamp_granularities[]=word``).
  :func:`scene_mark_timepoints` / :meth:`GroqWhisperAligner.caption_timepoints`
  fold the word times into the per-scene mark mapping
  (``{"s0": scene0_start_seconds, ...}``) that
  ``app.media.captions.events_from_timepoints`` consumes for SRT building.
- ``GroqScriptProvider`` — chat-completions ``ScriptProvider`` fallback for
  when the Gemini free-tier quota is exhausted. It reuses the Gemini prompt
  builders and JSON extraction, so it is a drop-in replacement.

All HTTP goes through httpx with an injectable client/transport (the
``GoogleTTSProvider`` pattern); 408/429/5xx are retried with exponential
backoff through an injectable ``sleep``. Successful calls record zero-dollar
CostLedger rows (note prefixed ``free-tier``) with real units — chars for TTS,
audio seconds for Whisper, a token count/estimate for the LLM — once a ledger
is bound via ``bind_cost_ledger`` (quota tracking, per PLAN.md).
"""

from __future__ import annotations

import io
import re
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import unescape

import httpx

from app.ai.base import SourceInfo, Timepoint, TTSResult
from app.ai.gemini import build_plan_prompt, build_shorten_prompt, extract_json
from app.ai.tts import END_MARK, split_sentences
from app.config import ModelConfig, get_model_config, get_settings
from app.costs import CostLedger
from app.errors import UpstreamError, ValidationFailed
from app.media.captions import DEFAULT_MARK_PREFIX

GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_TTS_ENDPOINT = f"{GROQ_API_BASE}/audio/speech"
GROQ_TRANSCRIPTIONS_ENDPOINT = f"{GROQ_API_BASE}/audio/transcriptions"
GROQ_CHAT_COMPLETIONS_ENDPOINT = f"{GROQ_API_BASE}/chat/completions"

FREE_TIER_NOTE = "free-tier"

_TAG_RE = re.compile(r"<[^>]+>")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")


def strip_ssml(ssml: str) -> str:
    """Plain narration text from SSML: tags removed, entities and spaces normalized."""
    return " ".join(unescape(_TAG_RE.sub(" ", ssml)).split())


# ---------------------------------------------------------------------------
# Sentence chunking for the per-request input limit
# ---------------------------------------------------------------------------


def _pack_parts(parts: Sequence[str], max_chars: int) -> list[str]:
    """Greedily join parts with single spaces into chunks of at most ``max_chars``.

    A single part longer than ``max_chars`` becomes its own (oversized) chunk;
    the caller hard-splits those as a last resort.
    """
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current} {part}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks


def chunk_sentence(sentence: str, max_chars: int) -> list[str]:
    """Split one sentence into chunks of at most ``max_chars``, never mid-word.

    A sentence within the limit passes through unchanged. Over-long sentences
    split on clause boundaries (comma/semicolon/colon) first, then on spaces.
    Only a single word longer than ``max_chars`` (no split point exists) is
    hard-split at the limit as a last resort.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    normalized = " ".join(sentence.split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]
    parts: list[str] = []
    for clause in _CLAUSE_SPLIT_RE.split(normalized):
        if len(clause) <= max_chars:
            parts.append(clause)
        else:
            parts.extend(clause.split())
    chunks: list[str] = []
    for chunk in _pack_parts(parts, max_chars):
        if len(chunk) <= max_chars:
            chunks.append(chunk)
        else:  # single word longer than the limit: no word boundary to use
            chunks.extend(chunk[i : i + max_chars] for i in range(0, len(chunk), max_chars))
    return chunks


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Sentence-based chunks of at most ``max_chars`` for the whole narration."""
    chunks: list[str] = []
    for sentence in split_sentences(text):
        chunks.extend(chunk_sentence(sentence, max_chars))
    return chunks


# ---------------------------------------------------------------------------
# WAV validation and concatenation (stdlib wave module)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WavInfo:
    channels: int
    sample_width: int
    frame_rate: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.frame_rate if self.frame_rate else 0.0


def wav_info(data: bytes) -> WavInfo:
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            return WavInfo(
                channels=reader.getnchannels(),
                sample_width=reader.getsampwidth(),
                frame_rate=reader.getframerate(),
                frame_count=reader.getnframes(),
            )
    except (wave.Error, EOFError) as exc:
        raise UpstreamError(f"groq tts returned invalid WAV data: {exc}", retryable=True) from exc


def concat_wav(segments: Sequence[bytes]) -> tuple[bytes, float]:
    """Concatenate WAV segments into one file; returns (wav_bytes, duration_seconds).

    All segments must share sample rate, width and channel count (the API is
    expected to be consistent); a mismatch raises ``UpstreamError``.
    """
    if not segments:
        return b"", 0.0
    infos = [wav_info(segment) for segment in segments]
    first = infos[0]
    for i, info in enumerate(infos[1:], start=1):
        if (info.channels, info.sample_width, info.frame_rate) != (
            first.channels,
            first.sample_width,
            first.frame_rate,
        ):
            raise UpstreamError(
                "groq tts WAV segments have mismatched audio parameters",
                retryable=False,
                details={
                    "segment": i,
                    "expected": [first.channels, first.sample_width, first.frame_rate],
                    "got": [info.channels, info.sample_width, info.frame_rate],
                },
            )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(first.channels)
        writer.setsampwidth(first.sample_width)
        writer.setframerate(first.frame_rate)
        for segment in segments:
            with wave.open(io.BytesIO(segment), "rb") as reader:
                writer.writeframes(reader.readframes(reader.getnframes()))
    total_frames = sum(info.frame_count for info in infos)
    return buffer.getvalue(), total_frames / first.frame_rate


# ---------------------------------------------------------------------------
# Shared HTTP adapter: auth, injectable client, retry with backoff, ledger
# ---------------------------------------------------------------------------


class _GroqAdapter:
    """Base for Groq adapters: httpx client injection, retries and cost rows."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_config: ModelConfig | None = None,
        http_client: httpx.Client | None = None,
        timeout: float = 120.0,
        max_attempts: int = 3,
        backoff_base_seconds: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        ledger: CostLedger | None = None,
        creative_id: str | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else get_settings().groq_api_key
        self._cfg = model_config or get_model_config()
        self._http = http_client
        self._timeout = timeout
        self._max_attempts = max(1, max_attempts)
        self._backoff_base = backoff_base_seconds
        self._sleep = sleep
        self._ledger = ledger
        self._creative_id = creative_id

    def bind_cost_ledger(self, ledger: CostLedger, creative_id: str) -> None:
        """Attach the ledger + creative so calls log zero-cost free-tier usage rows."""
        self._ledger = ledger
        self._creative_id = creative_id

    def _record(self, *, kind: str, model_id: str, units: float, note: str) -> None:
        if self._ledger is None or self._creative_id is None:
            return
        self._ledger.record_actual(
            self._creative_id,
            kind=kind,
            model_id=model_id,
            units=units,
            unit_price_usd=0.0,
            amount_usd=0.0,
            note=f"{FREE_TIER_NOTE} {note}",
        )

    def _post_with_retry(
        self,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """POST with 408/429/5xx retries (exponential backoff, injectable sleep)."""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        client = self._http
        owns_client = client is None
        if client is None:
            client = httpx.Client(timeout=self._timeout)
        try:
            last_status: int | None = None
            for attempt in range(self._max_attempts):
                response = client.post(
                    url, json=json_body, data=data, files=files, headers=headers
                )
                if response.status_code < 400:
                    return response
                last_status = response.status_code
                retryable = response.status_code in {408, 429} or response.status_code >= 500
                if not retryable:
                    raise UpstreamError(
                        f"groq request failed with HTTP {response.status_code}",
                        retryable=False,
                        details={"status_code": response.status_code, "endpoint": url},
                    )
                if attempt + 1 < self._max_attempts:
                    self._sleep(self._backoff_base * (2**attempt))
            raise UpstreamError(
                f"groq request failed with HTTP {last_status} "
                f"after {self._max_attempts} attempts",
                retryable=True,
                details={"status_code": last_status, "endpoint": url},
            )
        finally:
            if owns_client:
                client.close()


# ---------------------------------------------------------------------------
# Orpheus TTS (English only)
# ---------------------------------------------------------------------------


class GroqTTSProvider(_GroqAdapter):
    """English TTS through Groq Orpheus, implementing the ``TTSProvider`` protocol.

    The ``voice`` argument from ``TTSService`` (a Google voice name) is
    ignored; the configured ``ModelConfig.groq_tts_voice_en`` is used and
    reported back in the result.
    """

    def synthesize(self, *, ssml: str, voice: str, language_code: str) -> TTSResult:
        lang = (language_code or "").split("-")[0].lower()
        if lang != "en":
            raise ValidationFailed(
                f"GroqTTSProvider is English-only, got language_code={language_code!r}. "
                "Groq Orpheus has no Vietnamese voice — use edge-tts or Google Cloud "
                "TTS for locale 'vi'.",
                details={"language_code": language_code},
            )
        cfg = self._cfg
        text = strip_ssml(ssml)
        segments: list[bytes] = []
        sentence_durations: list[float] = []
        for sentence in split_sentences(text):
            duration = 0.0
            for fragment in chunk_sentence(sentence, cfg.groq_tts_max_chars):
                segment = self._synthesize_chunk(fragment)
                segments.append(segment)
                duration += wav_info(segment).duration_seconds
            sentence_durations.append(duration)

        audio, total_duration = concat_wav(segments)
        timepoints: list[Timepoint] = []
        cursor = 0.0
        for i, seconds in enumerate(sentence_durations):
            cursor += seconds
            timepoints.append(Timepoint(f"s{i}", round(cursor, 4)))
        timepoints.append(Timepoint(END_MARK, round(total_duration, 4)))

        self._record(
            kind="groq_tts",
            model_id=cfg.groq_tts_model_en,
            units=float(len(text)),
            note=f"tts en {len(segments)} chunk(s)",
        )
        return TTSResult(
            audio_bytes=audio,
            timepoints=tuple(timepoints),
            duration_seconds=round(total_duration, 4),
            voice=cfg.groq_tts_voice_en,
            audio_mime_type="audio/wav",
        )

    def _synthesize_chunk(self, chunk: str) -> bytes:
        response = self._post_with_retry(
            GROQ_TTS_ENDPOINT,
            json_body={
                "model": self._cfg.groq_tts_model_en,
                "voice": self._cfg.groq_tts_voice_en,
                "input": chunk,
                "response_format": "wav",
            },
        )
        return response.content


# ---------------------------------------------------------------------------
# Whisper word-timestamp alignment for captions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass(frozen=True)
class WhisperTranscription:
    text: str
    duration_seconds: float
    words: tuple[WordTimestamp, ...]


def scene_mark_timepoints(
    words: Sequence[WordTimestamp],
    narrations: Sequence[str],
    *,
    mark_prefix: str = DEFAULT_MARK_PREFIX,
) -> dict[str, float]:
    """Fold word timestamps into the per-scene mark mapping captions consume.

    Returns ``{"s0": scene0_start_seconds, "s1": ...}`` — exactly the
    ``timepoints`` argument of ``app.media.captions.events_from_timepoints``
    (one mark per scene at the scene's narration start). Words are assigned to
    scenes proportionally to each narration's word count, which is exact when
    the transcription's word count matches the narrations and degrades
    gracefully when Whisper tokenizes slightly differently.
    """
    counts = [len(narration.split()) for narration in narrations]
    total = sum(counts)
    if not words or total == 0:
        raise ValueError("cannot align captions: no transcribed words or empty narrations")
    timepoints: dict[str, float] = {}
    cumulative = 0
    for i, count in enumerate(counts):
        index = min(len(words) - 1, (len(words) * cumulative) // total)
        timepoints[f"{mark_prefix}{i}"] = words[index].start
        cumulative += count
    return timepoints


class GroqWhisperAligner(_GroqAdapter):
    """Word-level timestamps from Groq Whisper for VI/EN caption timing."""

    def transcribe_words(
        self, audio_bytes: bytes, *, filename: str = "audio.wav", language: str | None = None
    ) -> WhisperTranscription:
        data: dict[str, Any] = {
            "model": self._cfg.groq_whisper_model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        }
        if language:
            data["language"] = language
        response = self._post_with_retry(
            GROQ_TRANSCRIPTIONS_ENDPOINT,
            data=data,
            files={"file": (filename, audio_bytes, "application/octet-stream")},
        )
        payload = response.json()
        words = tuple(
            WordTimestamp(
                word=str(raw.get("word", "")).strip(),
                start=float(raw["start"]),
                end=float(raw["end"]),
            )
            for raw in payload.get("words") or []
            if isinstance(raw, dict) and "start" in raw and "end" in raw
        )
        duration = float(payload.get("duration") or (words[-1].end if words else 0.0))
        self._record(
            kind="groq_whisper",
            model_id=self._cfg.groq_whisper_model,
            units=duration,
            note=f"whisper align {len(words)} words",
        )
        return WhisperTranscription(
            text=str(payload.get("text", "")), duration_seconds=duration, words=words
        )

    def caption_timepoints(
        self,
        audio_bytes: bytes,
        narrations: Sequence[str],
        *,
        filename: str = "audio.wav",
        language: str | None = None,
        mark_prefix: str = DEFAULT_MARK_PREFIX,
    ) -> dict[str, float]:
        """Transcribe and return the captions-ready ``{"s0": seconds, ...}`` mapping."""
        transcription = self.transcribe_words(audio_bytes, filename=filename, language=language)
        return scene_mark_timepoints(transcription.words, narrations, mark_prefix=mark_prefix)


# ---------------------------------------------------------------------------
# LLM script fallback (ScriptProvider protocol)
# ---------------------------------------------------------------------------


class GroqScriptProvider(_GroqAdapter):
    """Drop-in ``ScriptProvider`` fallback for exhausted Gemini free-tier quota.

    Same prompts and JSON extraction as ``GeminiScriptProvider``; the model is
    ``ModelConfig.groq_llm_model`` with ``json_object`` response format.
    """

    _TEMPERATURE = 0.6

    def generate_plan(self, topic: str, sources: list[SourceInfo], brief: str) -> dict:
        return self._chat_json(build_plan_prompt(topic, sources, brief), note="script plan")

    def shorten_narrations(self, plan: dict, issues: list[dict]) -> dict:
        return self._chat_json(build_shorten_prompt(plan, issues), note="script shorten")

    def _chat_json(self, prompt: str, *, note: str) -> dict:
        model_id = self._cfg.groq_llm_model
        response = self._post_with_retry(
            GROQ_CHAT_COMPLETIONS_ENDPOINT,
            json_body={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": self._TEMPERATURE,
            },
        )
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise UpstreamError(
                "groq chat completion had no message content", retryable=True
            ) from exc
        usage = payload.get("usage") or {}
        tokens = usage.get("total_tokens")
        if not isinstance(tokens, int | float) or tokens <= 0:
            tokens = (len(prompt) + len(content)) // 4  # ~4 chars/token estimate
        self._record(kind="groq_llm", model_id=model_id, units=float(tokens), note=note)
        return extract_json(content)
