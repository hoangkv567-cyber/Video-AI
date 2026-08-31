"""Provider selection from Settings: free-tier defaults, paid adapters by config.

PLAN.md "Chế độ miễn phí": the free stack (Groq Orpheus EN TTS, edge-tts VI,
keyframe-motion instead of Veo) is the default; the paid adapters (Google
Cloud TTS, Veo) stay in code and are re-enabled purely through settings.
All provider imports are lazy so unused adapters cost nothing.
"""

from __future__ import annotations

import httpx

from app.ai.base import ImageProvider, ScriptProvider, TTSProvider
from app.config import ModelConfig, Settings, get_model_config, get_settings
from app.errors import ValidationFailed

PIPELINE_KEYFRAME_MOTION = "keyframe_motion"
PIPELINE_VEO = "veo"
PIPELINE_WAN = "wan"
_PIPELINE_KINDS = frozenset({PIPELINE_KEYFRAME_MOTION, PIPELINE_VEO, PIPELINE_WAN})

_TTS_CHOICES = frozenset({"groq", "edge", "google"})
_IMAGE_CHOICES = frozenset({"gemini", "procedural"})


def generation_preflight_issues(settings: Settings | None = None) -> list[dict[str, str]]:
    """Return actionable configuration blockers before a generation Job is queued."""
    settings = settings or get_settings()
    issues: list[dict[str, str]] = []
    if settings.image_provider not in _IMAGE_CHOICES:
        issues.append(
            {
                "code": "image_provider_invalid",
                "message": f"Unsupported IMAGE_PROVIDER={settings.image_provider!r}",
            }
        )
    elif settings.image_provider == "gemini" and not settings.gemini_api_key:
        issues.append(
            {
                "code": "gemini_image_key_missing",
                "message": "GEMINI_API_KEY is required for style-board/keyframe generation",
            }
        )
    for locale, choice in (
        ("en", settings.tts_provider_en),
        ("vi", settings.tts_provider_vi),
    ):
        if choice == "groq":
            if not settings.groq_api_key:
                issues.append(
                    {
                        "code": "groq_key_missing",
                        "message": f"GROQ_API_KEY is required for {locale} TTS",
                    }
                )
            if not settings.groq_orpheus_terms_accepted:
                issues.append(
                    {
                        "code": "groq_orpheus_terms_required",
                        "message": (
                            "Accept the Orpheus model terms in Groq Console or configure "
                            f"TTS_PROVIDER_{locale.upper()}=edge"
                        ),
                    }
                )
        elif choice == "google" and not settings.google_tts_api_key:
            issues.append(
                {
                    "code": "google_tts_key_missing",
                    "message": f"GOOGLE_TTS_API_KEY is required for {locale} TTS",
                }
            )
        elif choice not in _TTS_CHOICES:
            issues.append(
                {
                    "code": "tts_provider_invalid",
                    "message": f"Unsupported TTS provider {choice!r} for {locale}",
                }
            )
    if settings.video_provider not in _PIPELINE_KINDS:
        issues.append(
            {
                "code": "video_provider_invalid",
                "message": f"Unsupported VIDEO_PROVIDER={settings.video_provider!r}",
            }
        )
    elif settings.video_provider == PIPELINE_WAN and not settings.wan_api_key:
        issues.append(
            {
                "code": "wan_key_missing",
                "message": "WAN_API_KEY is required for wan image-to-video generation",
            }
        )
    return issues


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

    return GoogleTTSProvider(api_key=settings.google_tts_api_key or None, http_client=http_client)


def build_image_provider(settings: Settings | None = None) -> ImageProvider:
    """Gemini (free quota + existing fallbacks) or fully offline FFmpeg cards."""
    settings = settings or get_settings()
    kind = settings.image_provider
    if kind not in _IMAGE_CHOICES:
        raise ValidationFailed(
            f"unknown image_provider {kind!r} (expected one of {sorted(_IMAGE_CHOICES)})"
        )
    if kind == "procedural":
        from app.ai.procedural import ProceduralImageProvider  # noqa: PLC0415 — lazy

        return ProceduralImageProvider()
    from app.ai.gemini import GeminiImageProvider  # noqa: PLC0415 — lazy

    return GeminiImageProvider(api_key=settings.gemini_api_key)


def build_video_pipeline_kind(settings: Settings | None = None) -> str:
    """``"keyframe_motion"`` (free), ``"veo"`` or ``"wan"`` (paid) from settings."""
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
    return GeminiScriptProvider(
        api_key=settings.gemini_api_key,
        model_id=cfg.gemini_text_model,
        fallback_model_ids=cfg.gemini_text_fallback_models,
    )


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
