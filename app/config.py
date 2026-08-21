"""Application settings and versioned model/pricing configuration.

Model IDs and prices live here (versioned, overridable via env) because the
Veo/TTS previews and quotas can change; a startup probe validates them.
"""

from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Versioned model + pricing table. Probed at startup, never hardcoded elsewhere."""

    config_version: str = "2026-08-21"

    gemini_text_model: str = "gemini-3.7-flash"
    gemini_image_model: str = "gemini-3.1-flash-image"
    veo_model_lite: str = "veo-3.1-lite-generate-preview"
    veo_model_fast: str = "veo-3.1-fast-generate-preview"
    tts_voice_vi: str = "vi-VN-Neural2-A"
    tts_voice_en: str = "en-US-Neural2-F"

    # Free-tier stack (see PLAN.md "Chế độ miễn phí"): Groq Orpheus has no
    # Vietnamese, so VI voice comes from edge-tts; caption timing for both
    # locales comes from Groq Whisper word timestamps.
    groq_llm_model: str = "llama-3.3-70b-versatile"
    groq_tts_model_en: str = "canopylabs/orpheus-v1-english"
    groq_tts_voice_en: str = "hannah"
    groq_whisper_model: str = "whisper-large-v3"
    edge_tts_voice_vi: str = "vi-VN-HoaiMyNeural"
    groq_tts_max_chars: int = 200  # per-request input limit; chunk and concat

    # Reference prices in USD; verify against the live pricing page at startup.
    veo_lite_usd_per_second: float = 0.05
    veo_fast_usd_per_second: float = 0.15
    gemini_text_usd_per_call_estimate: float = 0.01
    gemini_image_usd_per_image: float = 0.039
    tts_usd_per_million_chars: float = 16.0

    scene_count: int = 5
    scene_seconds: float = 8.0
    crossfade_seconds: float = 0.3

    @property
    def master_duration_seconds(self) -> float:
        return (
            self.scene_count * self.scene_seconds
            - (self.scene_count - 1) * self.crossfade_seconds
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
    google_tts_credentials_json: str = ""
    groq_api_key: str = ""

    # Provider selection. Free-tier defaults: no Veo (no free tier exists) —
    # the visual master is built from keyframes with FFmpeg motion instead.
    video_provider: str = "keyframe_motion"  # keyframe_motion | veo
    tts_provider_en: str = "groq"  # groq | google | edge
    tts_provider_vi: str = "edge"  # edge | google  (Groq has no Vietnamese)
    caption_timing_provider: str = "groq_whisper"  # groq_whisper | tts_marks | estimate

    gemini_text_model: str = ""
    gemini_image_model: str = ""
    veo_model_lite: str = ""
    veo_model_fast: str = ""

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

    def model_defaults(self) -> ModelConfig:
        """ModelConfig with env overrides applied on top of the versioned defaults."""
        cfg = ModelConfig()
        overrides = {
            "gemini_text_model": self.gemini_text_model,
            "gemini_image_model": self.gemini_image_model,
            "veo_model_lite": self.veo_model_lite,
            "veo_model_fast": self.veo_model_fast,
        }
        data = cfg.model_dump()
        data.update({k: v for k, v in overrides.items() if v})
        return ModelConfig(**data)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_model_config() -> ModelConfig:
    return get_settings().model_defaults()
