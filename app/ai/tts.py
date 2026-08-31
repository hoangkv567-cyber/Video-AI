"""TTSService: per-scene synthesis for vi/en with SSML marks and duration gating.

SSML `<mark>` tags between sentences yield timepoints used for captions. When
measured duration exceeds the 7.6 s scene budget by more than 5%, the result is
flagged shorten-needed (never time-stretched outside 0.95–1.05; the allowed
ffmpeg atempo factor is exposed via `allowed_atempo_factor`).

`GoogleTTSProvider` talks to the Cloud Text-to-Speech REST API through httpx
with an injectable client/transport so contract tests run offline with
`httpx.MockTransport`; google-cloud-texttospeech is never imported.
"""

import base64
import re
from collections.abc import Callable
from dataclasses import dataclass
from xml.sax.saxutils import escape

import httpx

from app.ai.base import Timepoint, TTSProvider, TTSResult
from app.config import ModelConfig, get_model_config
from app.costs import CostLedger, tts_price_per_char_usd
from app.errors import UpstreamError
from app.models import Creative
from app.schemas.videoplan import NARRATION_MAX_SECONDS, NARRATION_TOLERANCE

TARGET_SECONDS = NARRATION_MAX_SECONDS  # 7.6
ATEMPO_MIN = 0.95
ATEMPO_MAX = 1.05
END_MARK = "end"

LANGUAGE_CODES: dict[str, str] = {"vi": "vi-VN", "en": "en-US"}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [part for part in _SENTENCE_SPLIT_RE.split(stripped) if part.strip()]


def build_ssml(text: str) -> str:
    """SSML with a mark after every sentence plus a final 'end' mark.

    The end mark's timepoint equals the full audio duration; the sentence marks
    give caption boundaries.
    """
    parts = ["<speak>"]
    for i, sentence in enumerate(split_sentences(text)):
        parts.append(escape(sentence))
        parts.append(f'<mark name="s{i}"/>')
    parts.append(f'<mark name="{END_MARK}"/>')
    parts.append("</speak>")
    return "".join(parts)


def allowed_atempo_factor(
    duration_seconds: float, target_seconds: float = TARGET_SECONDS
) -> float | None:
    """ffmpeg atempo factor fitting `duration` into `target`, or None if outside 0.95–1.05."""
    if duration_seconds <= 0 or target_seconds <= 0:
        return None
    factor = duration_seconds / target_seconds
    if ATEMPO_MIN <= factor <= ATEMPO_MAX:
        return round(factor, 4)
    return None


@dataclass(frozen=True)
class SceneAudio:
    locale: str
    voice: str
    audio_bytes: bytes
    audio_mime_type: str
    timepoints: tuple[Timepoint, ...]
    duration_seconds: float
    target_seconds: float
    shorten_needed: bool
    atempo_factor: float | None
    char_count: int
    cost_usd: float


class TTSService:
    def __init__(
        self,
        provider: TTSProvider,
        ledger: CostLedger,
        model_config: ModelConfig | None = None,
        *,
        model_id: str | None = None,
        unit_price_usd: float | None = None,
    ) -> None:
        self._provider = provider
        self._ledger = ledger
        self._cfg = model_config or get_model_config()
        self._model_id = model_id
        self._unit_price_usd = unit_price_usd

    def voice_for(self, locale: str) -> str:
        if locale == "vi":
            return self._cfg.tts_voice_vi
        if locale == "en":
            return self._cfg.tts_voice_en
        raise ValueError(f"unsupported locale: {locale}")

    def synthesize_scene(
        self,
        creative: Creative,
        text: str,
        locale: str,
        *,
        target_seconds: float = TARGET_SECONDS,
    ) -> SceneAudio:
        voice = self.voice_for(locale)
        language_code = LANGUAGE_CODES[locale]
        ssml = build_ssml(text)
        char_count = len(text)
        price_per_char = (
            tts_price_per_char_usd(self._cfg)
            if self._unit_price_usd is None
            else self._unit_price_usd
        )
        cost = char_count * price_per_char

        self._ledger.check_cap(creative, cost)
        result = self._provider.synthesize(ssml=ssml, voice=voice, language_code=language_code)
        self._ledger.record_actual(
            creative.id,
            kind="tts",
            model_id=self._model_id or result.voice,
            units=float(char_count),
            unit_price_usd=price_per_char,
            note=f"tts {locale} scene narration",
        )

        duration = result.duration_seconds
        shorten_needed = duration > target_seconds * (1 + NARRATION_TOLERANCE)
        atempo = None if duration <= target_seconds else allowed_atempo_factor(
            duration, target_seconds
        )
        return SceneAudio(
            locale=locale,
            voice=result.voice,
            audio_bytes=result.audio_bytes,
            audio_mime_type=result.audio_mime_type,
            timepoints=result.timepoints,
            duration_seconds=duration,
            target_seconds=target_seconds,
            shorten_needed=shorten_needed,
            atempo_factor=atempo,
            char_count=char_count,
            cost_usd=round(cost, 6),
        )


# ---------------------------------------------------------------------------
# Real Google Cloud TTS provider over httpx (injectable client/transport)
# ---------------------------------------------------------------------------

TTS_ENDPOINT = "https://texttospeech.googleapis.com/v1beta1/text:synthesize"


def _duration_from_timepoints(timepoints: tuple[Timepoint, ...]) -> float:
    for point in timepoints:
        if point.mark_name == END_MARK:
            return point.seconds
    if timepoints:
        return max(point.seconds for point in timepoints)
    return 0.0


class GoogleTTSProvider:
    """Cloud Text-to-Speech REST adapter with SSML_MARK timepointing.

    Auth is either an API key or a bearer-token supplier callable; neither is
    ever logged. Pass `http_client=httpx.Client(transport=httpx.MockTransport(...))`
    in contract tests.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        token_provider: Callable[[], str] | None = None,
        http_client: httpx.Client | None = None,
        endpoint: str = TTS_ENDPOINT,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._token_provider = token_provider
        self._http = http_client
        self._endpoint = endpoint
        self._timeout = timeout

    def synthesize(self, *, ssml: str, voice: str, language_code: str) -> TTSResult:
        payload = {
            "input": {"ssml": ssml},
            "voice": {"languageCode": language_code, "name": voice},
            "audioConfig": {"audioEncoding": "MP3"},
            "enableTimePointing": ["SSML_MARK"],
        }
        params: dict[str, str] = {}
        headers: dict[str, str] = {}
        if self._api_key:
            params["key"] = self._api_key
        elif self._token_provider is not None:
            headers["Authorization"] = f"Bearer {self._token_provider()}"

        client = self._http
        owns_client = client is None
        if client is None:
            client = httpx.Client(timeout=self._timeout)
        try:
            response = client.post(self._endpoint, json=payload, params=params, headers=headers)
            if response.status_code >= 400:
                retryable = response.status_code in {408, 429} or response.status_code >= 500
                raise UpstreamError(
                    f"tts synthesis failed with HTTP {response.status_code}",
                    retryable=retryable,
                    details={"status_code": response.status_code},
                )
            data = response.json()
        finally:
            if owns_client:
                client.close()

        audio = base64.b64decode(data.get("audioContent", "") or "")
        timepoints = tuple(
            Timepoint(mark_name=str(tp["markName"]), seconds=float(tp["timeSeconds"]))
            for tp in data.get("timepoints", []) or []
            if "markName" in tp and "timeSeconds" in tp
        )
        return TTSResult(
            audio_bytes=audio,
            timepoints=timepoints,
            duration_seconds=_duration_from_timepoints(timepoints),
            voice=voice,
        )
