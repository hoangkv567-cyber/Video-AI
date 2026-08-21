"""Provider selection from Settings: free-tier defaults, paid adapters by config.

PLAN.md "Chế độ miễn phí": the free stack (Groq Orpheus EN TTS, edge-tts VI,
keyframe-motion instead of Veo) is the default; the paid adapters (Google
Cloud TTS, Veo) stay in code and are re-enabled purely through settings.
All provider imports are lazy so unused adapters cost nothing.
"""

from __future__ import annotations

import httpx

from app.ai.base import ScriptProvider, TTSProvider
from app.config import ModelConfig, Settings, get_model_config, get_settings
from app.errors import ValidationFailed

PIPELINE_KEYFRAME_MOTION = "keyframe_motion"
PIPELINE_VEO = "veo"
_PIPELINE_KINDS = frozenset({PIPELINE_KEYFRAME_MOTION, PIPELINE_VEO})

_TTS_CHOICES = frozenset({"groq", "edge", "google"})


def build_tts_provider(
    locale: str,
    settings: Settings | None = None,
    model_config: ModelConfig | None = None,
    http_client: httpx.Client | None = None,
) -> TTSProvider:
    """TTS provider for a locale per ``tts_provider_en`` / ``tts_provider_vi``.

    Raises ``ValidationFailed`` for unsupported combinations — most notably
    ``vi`` + ``groq`` (Groq Orpheus has no Vietnamese voice).
    """
    settings = settings or get_settings()
    cfg = model_config or get_model_config()
    if locale == "en":
        choice = settings.tts_provider_en
    elif locale == "vi":
        choice = settings.tts_provider_vi
    else:
        raise ValidationFailed(f"unsupported TTS locale: {locale!r} (expected 'vi' or 'en')")

    if choice not in _TTS_CHOICES:
        raise ValidationFailed(
            f"unknown tts provider {choice!r} for locale {locale!r} "
            f"(expected one of {sorted(_TTS_CHOICES)})"
        )
    if choice == "groq":
        if locale != "en":
            raise ValidationFailed(
                "tts provider 'groq' supports English only: Groq Orpheus has no "
                "Vietnamese voice — configure 'edge' or 'google' for locale 'vi'",
                details={"locale": locale, "tts_provider": choice},
            )
        from app.ai.groq_providers import GroqTTSProvider  # noqa: PLC0415 — lazy

        return GroqTTSProvider(
            api_key=settings.groq_api_key, model_config=cfg, http_client=http_client
        )
    if choice == "edge":
        from app.ai.edge_tts_provider import (  # noqa: PLC0415 — lazy
            DEFAULT_VOICE_EN,
            EdgeTTSProvider,
        )

        voice = cfg.edge_tts_voice_vi if locale == "vi" else DEFAULT_VOICE_EN
        return EdgeTTSProvider(voice=voice, model_config=cfg)
    # "google": the paid adapter, kept re-enablable via configuration.
    from app.ai.tts import GoogleTTSProvider  # noqa: PLC0415 — lazy

    return GoogleTTSProvider(api_key=settings.gemini_api_key or None, http_client=http_client)


def build_video_pipeline_kind(settings: Settings | None = None) -> str:
    """``"keyframe_motion"`` (free tier, no Veo) or ``"veo"`` (paid) from settings."""
    settings = settings or get_settings()
    kind = settings.video_provider
    if kind not in _PIPELINE_KINDS:
        raise ValidationFailed(
            f"unknown video_provider {kind!r} (expected one of {sorted(_PIPELINE_KINDS)})"
        )
    return kind


def build_script_provider(
    settings: Settings | None = None, model_config: ModelConfig | None = None
) -> ScriptProvider:
    """Primary script provider: Gemini structured output (free tier)."""
    from app.ai.gemini import GeminiScriptProvider  # noqa: PLC0415 — lazy

    settings = settings or get_settings()
    cfg = model_config or get_model_config()
    return GeminiScriptProvider(api_key=settings.gemini_api_key, model_id=cfg.gemini_text_model)


def build_groq_script_provider(
    settings: Settings | None = None,
    model_config: ModelConfig | None = None,
    http_client: httpx.Client | None = None,
) -> ScriptProvider:
    """Fallback script provider for when the Gemini free-tier quota is exhausted."""
    from app.ai.groq_providers import GroqScriptProvider  # noqa: PLC0415 — lazy

    settings = settings or get_settings()
    return GroqScriptProvider(
        api_key=settings.groq_api_key,
        model_config=model_config or get_model_config(),
        http_client=http_client,
    )
