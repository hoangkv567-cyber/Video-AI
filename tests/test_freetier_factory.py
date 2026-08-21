"""Provider factory tests: env-driven selection matrix, the vi+groq config
error, pipeline-kind resolution and the Gemini-primary / Groq-fallback script
providers. Settings are built with ``_env_file=None`` so the local .env never
leaks into the matrix."""

from __future__ import annotations

import pytest

from app.ai.edge_tts_provider import DEFAULT_VOICE_EN, EdgeTTSProvider
from app.ai.factory import (
    PIPELINE_KEYFRAME_MOTION,
    PIPELINE_VEO,
    build_groq_script_provider,
    build_script_provider,
    build_tts_provider,
    build_video_pipeline_kind,
)
from app.ai.gemini import GeminiScriptProvider
from app.ai.groq_providers import GroqScriptProvider, GroqTTSProvider
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

    def test_free_tier_defaults_are_groq_en_and_edge_vi(self) -> None:
        settings = make_settings()
        assert type(build_tts_provider("en", settings, CFG)) is GroqTTSProvider
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


class TestScriptProviders:
    def test_gemini_is_primary(self) -> None:
        provider = build_script_provider(make_settings(gemini_api_key="k"))
        assert type(provider) is GeminiScriptProvider

    def test_groq_fallback_constructor_exposed(self) -> None:
        provider = build_groq_script_provider(make_settings(groq_api_key="k"), CFG)
        assert type(provider) is GroqScriptProvider
