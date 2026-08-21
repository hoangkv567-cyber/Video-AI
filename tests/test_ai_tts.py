"""TTSService: SSML marks, duration budget flagging, atempo bounds, httpx contract."""

import base64
import json

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.base import FakeTTSProvider, Timepoint
from app.ai.tts import (
    GoogleTTSProvider,
    TTSService,
    allowed_atempo_factor,
    build_ssml,
    split_sentences,
)
from app.costs import CostLedger
from app.errors import UpstreamError
from app.models import CostEvent, Creative


def make_service(db_session: Session) -> tuple[TTSService, CostLedger, FakeTTSProvider]:
    provider = FakeTTSProvider()
    ledger = CostLedger(db_session, default_cap_usd=6.0)
    return TTSService(provider, ledger), ledger, provider


class TestSSML:
    def test_split_sentences(self) -> None:
        assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
        assert split_sentences("   ") == []

    def test_build_ssml_marks_and_escaping(self) -> None:
        ssml = build_ssml("R&D is huge. AI < ML? Yes!")
        assert ssml.startswith("<speak>")
        assert ssml.endswith("</speak>")
        assert "R&amp;D" in ssml
        assert "AI &lt; ML?" in ssml
        assert '<mark name="s0"/>' in ssml
        assert '<mark name="s1"/>' in ssml
        assert '<mark name="s2"/>' in ssml
        assert '<mark name="end"/>' in ssml


class TestAtempoBounds:
    @pytest.mark.parametrize(
        ("duration", "expected"),
        [
            (7.6, 1.0),
            (7.79, pytest.approx(1.025)),
            (7.98, pytest.approx(1.05)),
            (8.5, None),  # would need > 1.05 — shorten instead
            (7.0, None),  # would need < 0.95 — never stretch that far
            (7.3, pytest.approx(0.9605)),
            (0.0, None),
        ],
    )
    def test_allowed_factor(self, duration: float, expected: float | None) -> None:
        assert allowed_atempo_factor(duration) == expected


class TestSynthesis:
    def test_within_budget_no_flag_and_cost_recorded(
        self, db_session: Session, creative: Creative
    ) -> None:
        service, ledger, provider = make_service(db_session)
        text = "A short line about AI." * 3  # 66 chars -> ~4.7 s at 14 chars/s
        result = service.synthesize_scene(creative, text, "en")

        assert result.voice == "en-US-Neural2-F"
        assert not result.shorten_needed
        assert result.atempo_factor is None  # under budget: no adjustment needed
        assert result.duration_seconds < 7.6
        assert result.audio_bytes.startswith(b"FAKEAUDIO:")
        assert provider.calls[0]["language_code"] == "en-US"

        events = db_session.query(CostEvent).filter_by(creative_id=creative.id).all()
        assert len(events) == 1
        assert events[0].kind == "tts"
        assert not events[0].projected
        assert events[0].units == pytest.approx(len(text))
        assert events[0].amount_usd == pytest.approx(len(text) * 16 / 1_000_000)
        assert ledger.total_actual(creative.id) > 0

    def test_slightly_over_budget_gets_atempo_not_shorten(
        self, db_session: Session, creative: Creative
    ) -> None:
        service, _, _ = make_service(db_session)
        text = "x" * 108  # 108 / 14 = 7.714 s -> within the 5% tolerance
        result = service.synthesize_scene(creative, text, "en")

        assert not result.shorten_needed
        assert result.atempo_factor == pytest.approx(7.7143 / 7.6, abs=1e-3)
        assert 1.0 < result.atempo_factor <= 1.05

    def test_over_budget_by_more_than_5_percent_flags_shorten(
        self, db_session: Session, creative: Creative
    ) -> None:
        service, _, _ = make_service(db_session)
        text = "a" * 200  # vi: 200 / 15 = 13.3 s >> 7.98 s
        result = service.synthesize_scene(creative, text, "vi")

        assert result.voice == "vi-VN-Neural2-A"
        assert result.shorten_needed
        assert result.atempo_factor is None  # no legal time-stretch can fix this

    def test_timepoints_cover_marks_and_end(
        self, db_session: Session, creative: Creative
    ) -> None:
        service, _, _ = make_service(db_session)
        result = service.synthesize_scene(creative, "First part. Second part.", "en")

        names = [t.mark_name for t in result.timepoints]
        assert names == ["s0", "s1", "end"]
        seconds = [t.seconds for t in result.timepoints]
        assert seconds == sorted(seconds)
        assert seconds[-1] == pytest.approx(result.duration_seconds)

    def test_unsupported_locale_raises(self, db_session: Session, creative: Creative) -> None:
        service, _, _ = make_service(db_session)
        with pytest.raises(ValueError, match="unsupported locale"):
            service.synthesize_scene(creative, "hello", "fr")


class TestGoogleTTSContract:
    """Offline contract test through httpx.MockTransport."""

    AUDIO = b"ID3-fake-mp3-bytes"

    def make_provider(self) -> tuple[GoogleTTSProvider, list[httpx.Request]]:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            body = json.loads(request.content.decode())
            assert body["enableTimePointing"] == ["SSML_MARK"]
            assert body["input"]["ssml"].startswith("<speak>")
            assert body["voice"] == {"languageCode": "en-US", "name": "en-US-Neural2-F"}
            return httpx.Response(
                200,
                json={
                    "audioContent": base64.b64encode(self.AUDIO).decode(),
                    "timepoints": [
                        {"markName": "s0", "timeSeconds": 3.2},
                        {"markName": "end", "timeSeconds": 6.4},
                    ],
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        return GoogleTTSProvider(api_key="test-key", http_client=client), seen

    def test_synthesize_parses_audio_and_timepoints(self) -> None:
        provider, seen = self.make_provider()
        result = provider.synthesize(
            ssml=build_ssml("Hello there. General AI."),
            voice="en-US-Neural2-F",
            language_code="en-US",
        )
        assert result.audio_bytes == self.AUDIO
        assert result.timepoints == (Timepoint("s0", 3.2), Timepoint("end", 6.4))
        assert result.duration_seconds == pytest.approx(6.4)  # end-mark timepoint
        assert seen[0].url.params["key"] == "test-key"

    @pytest.mark.parametrize(("status", "retryable"), [(429, True), (500, True), (403, False)])
    def test_http_errors_map_to_upstream_error(self, status: int, retryable: bool) -> None:
        client = httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(status, json={}))
        )
        provider = GoogleTTSProvider(api_key="test-key", http_client=client)
        with pytest.raises(UpstreamError) as excinfo:
            provider.synthesize(ssml="<speak>x</speak>", voice="v", language_code="en-US")
        assert excinfo.value.retryable is retryable

    def test_bearer_token_used_when_no_api_key(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["Authorization"] == "Bearer tok-123"
            return httpx.Response(200, json={"audioContent": "", "timepoints": []})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = GoogleTTSProvider(token_provider=lambda: "tok-123", http_client=client)
        result = provider.synthesize(ssml="<speak>x</speak>", voice="v", language_code="en-US")
        assert result.duration_seconds == 0.0
