"""Fernet encryption for stored OAuth tokens.

The key comes from ``settings.token_encryption_key``. Any non-empty string is
accepted: a valid Fernet key is used verbatim, anything else is derived via
SHA-256 -> urlsafe base64. An empty key is allowed only in dev/test, where an
ephemeral per-process key is generated (tokens do not survive a restart there,
which is fine for local runs). Production must configure a real key.

No plaintext secret may ever reach logs: use :func:`redact` before logging any
credential dict.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Protocol

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings

REDACTED = "***redacted***"
_SENSITIVE_MARKERS = ("token", "secret", "key", "password", "credential", "authorization")

_EPHEMERAL_KEY: bytes | None = None


class SettingsLike(Protocol):
    app_env: str
    token_encryption_key: str


def _ephemeral_key() -> bytes:
    global _EPHEMERAL_KEY
    if _EPHEMERAL_KEY is None:
        _EPHEMERAL_KEY = Fernet.generate_key()
    return _EPHEMERAL_KEY


def derive_fernet_key(raw: str) -> bytes:
    """Return a valid Fernet key for any non-empty passphrase (deterministic)."""
    if not raw:
        raise ValueError("cannot derive a Fernet key from an empty string")
    candidate = raw.encode("utf-8")
    try:
        Fernet(candidate)
    except (ValueError, TypeError):
        return base64.urlsafe_b64encode(hashlib.sha256(candidate).digest())
    return candidate


def get_fernet(key: str | None = None, settings: SettingsLike | None = None) -> Fernet:
    """Build the Fernet instance from an explicit key or the app settings."""
    if key:
        return Fernet(derive_fernet_key(key))
    settings = settings if settings is not None else get_settings()
    if settings.token_encryption_key:
        return Fernet(derive_fernet_key(settings.token_encryption_key))
    if settings.app_env in {"dev", "test"}:
        return Fernet(_ephemeral_key())
    raise RuntimeError(
        "token_encryption_key must be configured outside dev/test environments"
    )


def encrypt_credentials(
    credentials: Mapping[str, Any],
    *,
    key: str | None = None,
    settings: SettingsLike | None = None,
) -> str:
    """Encrypt a credential dict to an opaque Fernet token string."""
    payload = json.dumps(dict(credentials), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return get_fernet(key, settings).encrypt(payload).decode("ascii")


def decrypt_credentials(
    token: str,
    *,
    key: str | None = None,
    settings: SettingsLike | None = None,
) -> dict[str, Any]:
    """Decrypt a stored token string back into the credential dict."""
    try:
        payload = get_fernet(key, settings).decrypt(token.encode("ascii"))
    except InvalidToken as exc:
        # Never include the token or key material in the error message.
        raise ValueError("credential token is invalid or was encrypted with another key") from exc
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("credential payload must be a JSON object")
    return data


def redact(credentials: Mapping[str, Any]) -> dict[str, Any]:
    """Log-safe copy of a credential dict: sensitive values are masked."""
    out: dict[str, Any] = {}
    for k, v in credentials.items():
        lowered = k.lower()
        if any(marker in lowered for marker in _SENSITIVE_MARKERS):
            out[k] = REDACTED
        elif isinstance(v, Mapping):
            out[k] = redact(v)
        else:
            out[k] = v
    return out
