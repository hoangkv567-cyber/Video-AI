"""Production configuration fails closed while local development stays simple."""

import pytest
from cryptography.fernet import Fernet
from pydantic import ValidationError

from app.config import ModelConfig, Settings


def test_default_development_settings_remain_available() -> None:
    settings = Settings(app_env="dev", _env_file=None)
    assert settings.app_env == "dev"


def test_default_gemini_text_model_chain_is_ordered() -> None:
    assert ModelConfig().gemini_text_model_chain == (
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
    )


def test_gemini_text_model_chain_env_override_is_trimmed_and_deduplicated() -> None:
    settings = Settings(
        gemini_text_model="custom-primary",
        gemini_text_fallback_models=(
            "custom-primary, custom-secondary, custom-secondary, custom-tertiary"
        ),
        _env_file=None,
    )

    assert settings.model_defaults().gemini_text_model_chain == (
        "custom-primary",
        "custom-secondary",
        "custom-tertiary",
    )


def test_production_rejects_local_urls_and_placeholder_secrets() -> None:
    with pytest.raises(ValidationError, match="production settings are unsafe"):
        Settings(app_env="production", _env_file=None)


def test_production_accepts_complete_secure_contract() -> None:
    settings = Settings(
        app_env="production",
        secret_key="s" * 48,
        token_encryption_key=Fernet.generate_key().decode("ascii"),
        database_url="postgresql+psycopg://videoai:strong-db-password@postgres:5432/videoai",
        redis_url="redis://:strong-redis-password@redis:6379/0",
        minio_access_key="prod-access",
        minio_secret_key="strong-minio-secret",
        public_base_url="https://video.example.com",
        _env_file=None,
    )
    assert settings.app_env == "production"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("token_encryption_key", "too-short"),
        ("redis_url", "redis://redis:6379/0"),
        ("redis_url", "http://:strong-redis-password@redis:6379/0"),
    ],
)
def test_production_rejects_weak_crypto_and_redis_contract(
    field: str, value: str
) -> None:
    kwargs = {
        "app_env": "production",
        "secret_key": "s" * 48,
        "token_encryption_key": Fernet.generate_key().decode("ascii"),
        "database_url": (
            "postgresql+psycopg://videoai:strong-db-password@postgres:5432/videoai"
        ),
        "redis_url": "redis://:strong-redis-password@redis:6379/0",
        "minio_access_key": "prod-access",
        "minio_secret_key": "strong-minio-secret",
        "public_base_url": "https://video.example.com",
        "_env_file": None,
        field: value,
    }
    with pytest.raises(ValidationError, match="production settings are unsafe"):
        Settings(**kwargs)
