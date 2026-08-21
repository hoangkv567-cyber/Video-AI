"""Token crypto tests: Fernet round-trip, ephemeral dev/test key, no plaintext leaks."""

import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.sqlite3")

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
from cryptography.fernet import Fernet  # noqa: E402

from app.publishing.crypto import (  # noqa: E402
    REDACTED,
    decrypt_credentials,
    derive_fernet_key,
    encrypt_credentials,
    get_fernet,
    redact,
)

CREDS = {
    "access_token": "ya29.super-secret-access-token",
    "refresh_token": "1//refresh-secret-value",
    "page_id": "1234567890",
}


class TestRoundTrip:
    def test_encrypt_decrypt_round_trip(self) -> None:
        token = encrypt_credentials(CREDS)
        assert isinstance(token, str)
        assert decrypt_credentials(token) == CREDS

    def test_two_encryptions_share_the_process_key(self) -> None:
        first = encrypt_credentials(CREDS)
        second = encrypt_credentials(CREDS)
        assert decrypt_credentials(first) == decrypt_credentials(second) == CREDS

    def test_explicit_passphrase_key(self) -> None:
        token = encrypt_credentials(CREDS, key="my-team-passphrase")
        assert decrypt_credentials(token, key="my-team-passphrase") == CREDS

    def test_wrong_key_raises_without_leaking(self) -> None:
        token = encrypt_credentials(CREDS, key="right-key")
        with pytest.raises(ValueError) as excinfo:
            decrypt_credentials(token, key="wrong-key")
        assert "super-secret" not in str(excinfo.value)
        assert "right-key" not in str(excinfo.value)


class TestKeyHandling:
    def test_valid_fernet_key_used_verbatim(self) -> None:
        raw = Fernet.generate_key().decode("ascii")
        assert derive_fernet_key(raw) == raw.encode("ascii")

    def test_passphrase_derivation_is_deterministic(self) -> None:
        assert derive_fernet_key("hello") == derive_fernet_key("hello")
        assert derive_fernet_key("hello") != derive_fernet_key("other")

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(ValueError):
            derive_fernet_key("")

    def test_test_env_gets_ephemeral_key_when_unset(self) -> None:
        settings = SimpleNamespace(app_env="test", token_encryption_key="")
        fernet = get_fernet(settings=settings)
        token = encrypt_credentials(CREDS, settings=settings)
        assert decrypt_credentials(token, settings=settings) == CREDS
        assert fernet is not None

    def test_production_requires_a_configured_key(self) -> None:
        settings = SimpleNamespace(app_env="production", token_encryption_key="")
        with pytest.raises(RuntimeError):
            get_fernet(settings=settings)


class TestNoPlaintextLeaks:
    def test_ciphertext_never_contains_plaintext(self) -> None:
        token = encrypt_credentials(CREDS)
        for secret in CREDS.values():
            assert secret not in token

    def test_repr_of_redacted_credentials_hides_secrets(self) -> None:
        redacted = redact(CREDS)
        text = repr(redacted)
        assert "super-secret" not in text
        assert "refresh-secret" not in text
        assert REDACTED in text
        # Non-sensitive identifiers survive for debugging.
        assert redacted["page_id"] == "1234567890"

    def test_redact_handles_nested_dicts(self) -> None:
        nested = {"oauth": {"client_secret": "shh", "scope": "upload"}, "name": "chan"}
        redacted = redact(nested)
        assert redacted["oauth"]["client_secret"] == REDACTED
        assert redacted["oauth"]["scope"] == "upload"
        assert redacted["name"] == "chan"
        assert "shh" not in repr(redacted)
