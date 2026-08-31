"""Application settings and versioned model/pricing configuration.

Model IDs and prices live here (versioned, overridable via env) because the
Veo/TTS previews and quotas can change; the provider-readiness workflow
validates them before paid capabilities are enabled.
"""

from functools import lru_cache
from urllib.parse import urlsplit

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _looks_like_placeholder(value: str) -> bool:
    normalized = value.strip().lower().replace("-", "_")
    return not normalized or "change_me" in normalized or normalized.startswith("replace_me")


class ModelConfig(BaseModel):
    """Versioned model + pricing table. Probed at startup, never hardcoded elsewhere."""

    config_version: str = "2026-08-31"

    gemini_text_model: str = "gemini-3.7-flash"
    gemini_text_fallback_models: tuple[str, ...] = (
        "gemini-3.6-flash",
        "gemini-3.5-flash",
    )
    gemini_image_model: str = "gemini-3.1-flash-image"
    veo_model_lite: str = "veo-3.1-lite-generate-preview"
    veo_model_fast: str = "veo-3.1-fast-generate-preview"
    tts_voice_vi: str = "vi-VN-Neural2-A"
    tts_voice_en: str = "en-US-Neural2-F"

    # Alibaba Cloud Model Studio (international/Singapore) image-to-video.
    # wan2.7-i2v: 720P/1080P, duration 2-15 s, first-frame accepts base64
    # data URIs (verified live 2026-08-23). List price $0.10/s at 720P with a
    # 50 s free quota (90 days from activation) — enough for one 40 s master.
    wan_i2v_model: str = "wan2.7-i2v"
    wan_usd_per_second_720p: float = 0.10

    # Free-tier stack (see PLAN.md "Chế độ miễn phí"): Groq Orpheus has no
    # Vietnamese, so VI voice comes from edge-tts; caption timing for both
    # locales comes from Groq Whisper word timestamps.
    # llama-3.3 was retired from Groq's catalog; gpt-oss-120b is the strongest
    # free-tier text model on this key (probed 2026-08-21).
    groq_llm_model: str = "openai/gpt-oss-120b"
    groq_llm_max_completion_tokens: int = 4096  # below the free-tier 8K TPM cap
    groq_tts_model_en: str = "canopylabs/orpheus-v1-english"
    groq_tts_voice_en: str = "hannah"
    groq_whisper_model: str = "whisper-large-v3"
    edge_tts_voice_vi: str = "vi-VN-HoaiMyNeural"
    groq_tts_max_chars: int = 200  # per-request input limit; chunk and concat

    # Reference prices in USD; reverify in the provider-readiness workflow.
    veo_lite_usd_per_second: float = 0.05
    veo_fast_usd_per_second: float = 0.10  # Veo 3.1 Fast, 720p
    gemini_text_usd_per_call_estimate: float = 0.01
    gemini_image_usd_per_image: float = 0.067  # 1K output used by GeminiImageProvider
    tts_usd_per_million_chars: float = 16.0

    scene_count: int = 5
    scene_seconds: float = 8.0
    crossfade_seconds: float = 0.3

    @property
    def gemini_text_model_chain(self) -> tuple[str, ...]:
        """Primary and fallback text models, normalized without duplicates."""
        models = (
            self.gemini_text_model,
            *self.gemini_text_fallback_models,
        )
        return tuple(dict.fromkeys(model.strip() for model in models if model.strip()))

    @property
    def master_duration_seconds(self) -> float:
        return (
            self.scene_count * self.scene_seconds - (self.scene_count - 1) * self.crossfade_seconds
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "dev"
    secret_key: str = "dev-secret-change-me"
    token_encryption_key: str = ""

    database_url: str = "sqlite:///./videoai.sqlite3"
    redis_url: str = "redis://localhost:6379/0"

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "videoai"
    minio_secret_key: str = "videoai"
    minio_bucket: str = "videoai"
    minio_secure: bool = False

    gemini_api_key: str = ""
    google_tts_api_key: str = ""
    groq_api_key: str = ""
    groq_orpheus_terms_accepted: bool = False
    wan_api_key: str = ""

    # Provider selection. Free-tier defaults: no Veo (no free tier exists) —
    # the visual master is built from keyframes with FFmpeg motion instead.
    # "wan": Alibaba wan2.7-i2v paid adapter (50 s free quota, then $0.10/s).
    video_provider: str = "keyframe_motion"  # keyframe_motion | veo | wan
    # When the paid video provider's quota is exhausted mid-generation, scenes
    # fall back here (render-stage Ken Burns) so the creative still completes.
    # Set empty to disable and fail the job instead.
    video_fallback_provider: str = "keyframe_motion"
    image_provider: str = "gemini"  # gemini (free quota) | procedural (offline FFmpeg cards)
    tts_provider_en: str = "edge"  # groq | google | edge
    tts_provider_vi: str = "edge"  # edge | google  (Groq has no Vietnamese)
    caption_timing_provider: str = "estimate"  # estimate (free) | groq_whisper | tts_marks

    gemini_text_model: str = ""
    gemini_text_fallback_models: str = ""
    gemini_image_model: str = ""
    veo_model_lite: str = ""
    veo_model_fast: str = ""
    groq_llm_model: str = ""
    groq_llm_max_completion_tokens: int = 0
    groq_tts_model_en: str = ""
    groq_tts_voice_en: str = ""
    groq_whisper_model: str = ""
    edge_tts_voice_vi: str = ""
    wan_i2v_model: str = ""

    cost_hard_cap_usd: float = 6.0

    youtube_client_id: str = ""
    youtube_client_secret: str = ""
    facebook_app_id: str = ""
    facebook_app_secret: str = ""
    tiktok_client_key: str = ""
    tiktok_client_secret: str = ""
    zalo_app_id: str = ""
    zalo_app_secret: str = ""

    public_base_url: str = "http://localhost:8000"
    timezone_display: str = "Asia/Ho_Chi_Minh"

    @model_validator(mode="after")
    def validate_environment_contract(self) -> "Settings":
        if self.app_env not in {"dev", "test", "production"}:
            raise ValueError("APP_ENV must be dev, test, or production")
        if self.app_env != "production":
            return self

        invalid: list[str] = []
        if len(self.secret_key) < 32 or _looks_like_placeholder(self.secret_key):
            invalid.append("SECRET_KEY")
        if len(self.token_encryption_key) < 32 or _looks_like_placeholder(
            self.token_encryption_key
        ):
            invalid.append("TOKEN_ENCRYPTION_KEY(32+ characters required)")

        database = urlsplit(self.database_url)
        if (
            not database.scheme.startswith("postgresql")
            or not database.hostname
            or not database.username
            or database.path in {"", "/"}
        ):
            invalid.append("DATABASE_URL(postgresql required)")
        if (
            not database.password
            or len(database.password) < 16
            or _looks_like_placeholder(database.password)
            or database.password == "videoai"
        ):
            invalid.append("DATABASE_URL(strong password required)")

        redis = urlsplit(self.redis_url)
        if redis.scheme not in {"redis", "rediss"} or not redis.hostname:
            invalid.append("REDIS_URL(redis/rediss required)")
        if (
            not redis.password
            or len(redis.password) < 16
            or _looks_like_placeholder(redis.password)
        ):
            invalid.append("REDIS_URL(strong password required)")

        public_url = urlsplit(self.public_base_url)
        if (
            public_url.scheme != "https"
            or public_url.hostname in {None, "localhost", "127.0.0.1", "::1"}
            or public_url.username is not None
            or public_url.password is not None
        ):
            invalid.append("PUBLIC_BASE_URL(https domain required)")
        if (
            len(self.minio_access_key) < 3
            or self.minio_access_key == "videoai"
            or _looks_like_placeholder(self.minio_access_key)
        ):
            invalid.append("MINIO_ACCESS_KEY")
        if (
            len(self.minio_secret_key) < 16
            or self.minio_secret_key in {"videoai", "videoai-secret"}
            or _looks_like_placeholder(self.minio_secret_key)
        ):
            invalid.append("MINIO_SECRET_KEY")
        if invalid:
            raise ValueError(
                "production settings are unsafe or incomplete: " + ", ".join(invalid)
            )
        return self

    def model_defaults(self) -> ModelConfig:
        """ModelConfig with env overrides applied on top of the versioned defaults."""
        cfg = ModelConfig()
        overrides = {
            "gemini_text_model": self.gemini_text_model,
            "gemini_image_model": self.gemini_image_model,
            "veo_model_lite": self.veo_model_lite,
            "veo_model_fast": self.veo_model_fast,
            "groq_llm_model": self.groq_llm_model,
            "groq_tts_model_en": self.groq_tts_model_en,
            "groq_tts_voice_en": self.groq_tts_voice_en,
            "groq_whisper_model": self.groq_whisper_model,
            "edge_tts_voice_vi": self.edge_tts_voice_vi,
            "wan_i2v_model": self.wan_i2v_model,
        }
        data = cfg.model_dump()
        data.update({k: v for k, v in overrides.items() if v})
        if self.gemini_text_fallback_models.strip():
            data["gemini_text_fallback_models"] = tuple(
                model.strip()
                for model in self.gemini_text_fallback_models.split(",")
                if model.strip()
            )
        if self.groq_llm_max_completion_tokens > 0:
            data["groq_llm_max_completion_tokens"] = self.groq_llm_max_completion_tokens
        return ModelConfig(**data)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_model_config() -> ModelConfig:
    return get_settings().model_defaults()
