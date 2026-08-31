"""Provider factory tests: env-driven selection matrix, the vi+groq config
error, pipeline-kind resolution and the Gemini-primary / Groq-fallback script
providers. Settings are built with ``_env_file=None`` so the local .env never
leaks into the matrix."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.edge_tts_provider import DEFAULT_VOICE_EN, EdgeTTSProvider
from app.ai.factory import (
    PIPELINE_KEYFRAME_MOTION,
    PIPELINE_VEO,
    build_groq_script_provider,
    build_image_provider,
    build_script_provider,
    build_tts_provider,
    build_video_pipeline_kind,
    generation_preflight_issues,
)
from app.ai.gemini import GeminiImageProvider, GeminiScriptProvider
from app.ai.groq_providers import GroqScriptProvider, GroqTTSProvider
from app.ai.procedural import ProceduralImageProvider
from app.ai.tts import GoogleTTSProvider
from app.config import ModelConfig, Settings
from app.errors import ValidationFailed

CFG = ModelConfig()


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


class TestTTSSelectionMatrix:
    @pytest.mark.parametrize(
        ("locale", "choice", "expected_type"),
        [
            ("en", "groq", GroqTTSProvider),
            ("en", "google", GoogleTTSProvider),
            ("en", "edge", EdgeTTSProvider),
            ("vi", "edge", EdgeTTSProvider),
            ("vi", "google", GoogleTTSProvider),
        ],
    )
    def test_selection(self, locale: str, choice: str, expected_type: type) -> None:
        key = "tts_provider_en" if locale == "en" else "tts_provider_vi"
        provider = build_tts_provider(locale, make_settings(**{key: choice}), CFG)
        assert type(provider) is expected_type

    def test_free_tier_defaults_use_edge_for_both_locales(self) -> None:
        settings = make_settings()
        assert type(build_tts_provider("en", settings, CFG)) is EdgeTTSProvider
        assert type(build_tts_provider("vi", settings, CFG)) is EdgeTTSProvider

    def test_vi_plus_groq_raises_clear_config_error(self) -> None:
        with pytest.raises(ValidationFailed, match="Vietnamese"):
            build_tts_provider("vi", make_settings(tts_provider_vi="groq"), CFG)

    def test_unknown_provider_name_raises(self) -> None:
        with pytest.raises(ValidationFailed, match="unknown tts provider"):
            build_tts_provider("en", make_settings(tts_provider_en="polly"), CFG)

    def test_unknown_locale_raises(self) -> None:
        with pytest.raises(ValidationFailed, match="unsupported TTS locale"):
            build_tts_provider("fr", make_settings(), CFG)

    def test_edge_voices_per_locale(self) -> None:
        vi = build_tts_provider("vi", make_settings(tts_provider_vi="edge"), CFG)
        assert isinstance(vi, EdgeTTSProvider)
        assert vi.voice == CFG.edge_tts_voice_vi == "vi-VN-HoaiMyNeural"
        en = build_tts_provider("en", make_settings(tts_provider_en="edge"), CFG)
        assert isinstance(en, EdgeTTSProvider)
        assert en.voice == DEFAULT_VOICE_EN


class TestVideoPipelineKind:
    def test_keyframe_motion_is_the_free_tier_default(self) -> None:
        assert build_video_pipeline_kind(make_settings()) == PIPELINE_KEYFRAME_MOTION

    def test_veo_selectable_for_the_paid_path(self) -> None:
        assert build_video_pipeline_kind(make_settings(video_provider="veo")) == PIPELINE_VEO

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValidationFailed, match="unknown video_provider"):
            build_video_pipeline_kind(make_settings(video_provider="sora"))


class TestGenerationPreflight:
    def test_default_only_needs_gemini_image_key(self) -> None:
        issues = generation_preflight_issues(make_settings())
        assert [issue["code"] for issue in issues] == ["gemini_image_key_missing"]

    def test_groq_terms_are_an_explicit_gate(self) -> None:
        settings = make_settings(
            gemini_api_key="k",
            groq_api_key="k",
            tts_provider_en="groq",
            groq_orpheus_terms_accepted=False,
        )
        assert [issue["code"] for issue in generation_preflight_issues(settings)] == [
            "groq_orpheus_terms_required"
        ]

    def test_ready_free_configuration_has_no_blockers(self) -> None:
        settings = make_settings(gemini_api_key="k")
        assert generation_preflight_issues(settings) == []

    def test_procedural_images_do_not_need_a_gemini_key(self) -> None:
        settings = make_settings(image_provider="procedural")
        assert generation_preflight_issues(settings) == []


class TestImageProvider:
    def test_gemini_is_the_default(self) -> None:
        provider = build_image_provider(make_settings(gemini_api_key="k"))
        assert type(provider) is GeminiImageProvider

    def test_procedural_is_selectable_for_offline_demo(self) -> None:
        provider = build_image_provider(make_settings(image_provider="procedural"))
        assert type(provider) is ProceduralImageProvider


class TestScriptProviders:
    def test_gemini_is_primary(self) -> None:
        provider = build_script_provider(make_settings(gemini_api_key="k"))
        assert type(provider) is GeminiScriptProvider

    def test_groq_fallback_constructor_exposed(self) -> None:
        provider = build_groq_script_provider(make_settings(groq_api_key="k"), CFG)
        assert type(provider) is GroqScriptProvider


def test_groq_and_edge_model_settings_are_runtime_overridable() -> None:
    cfg = make_settings(
        groq_llm_model="replacement-llm",
        groq_llm_max_completion_tokens=2048,
        groq_tts_model_en="replacement-tts",
        groq_tts_voice_en="replacement-voice",
        groq_whisper_model="replacement-whisper",
        edge_tts_voice_vi="replacement-vi-voice",
    ).model_defaults()

    assert cfg.groq_llm_model == "replacement-llm"
    assert cfg.groq_llm_max_completion_tokens == 2048
    assert cfg.groq_tts_model_en == "replacement-tts"
    assert cfg.groq_tts_voice_en == "replacement-voice"
    assert cfg.groq_whisper_model == "replacement-whisper"
    assert cfg.edge_tts_voice_vi == "replacement-vi-voice"


class TestProceduralModelLabel:
    def test_procedural_provider_never_reports_paid_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Free backends must not echo the caller's Gemini model id."""
        import base64

        provider = ProceduralImageProvider()

        def fake_run(*args, **kwargs):
            png = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
                "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            )
            # Provider requires >500 bytes of output; pad after the signature.
            png = png + b"\x00" * 600
            return SimpleNamespace(returncode=0, stdout=png, stderr=b"")

        def no_network(*args, **kwargs):
            raise OSError("network disabled in test")

        import httpx

        monkeypatch.setattr(httpx, "get", no_network)
        monkeypatch.setattr("app.ai.procedural.subprocess.run", fake_run)
        result = provider.generate_image(
            "cinematic AI chip", model_id="gemini-3.1-flash-image"
        )

        assert "gemini" not in result.model_id.lower()
