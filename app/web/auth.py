"""Dashboard authentication helpers.

Password hashing uses stdlib ``hashlib.pbkdf2_hmac`` (sha256, per-user random
salt). Sessions are a signed cookie (itsdangerous) carrying only the user id —
no server-side session store needed for this internal tool. CSRF tokens are
signed values bound to the session identity and checked on every mutating POST
handled by the dashboard itself.

No secrets are ever logged; the signing key comes from ``Settings.secret_key``.
"""

import hashlib
import hmac
import secrets
from collections.abc import Callable
from typing import Annotated
from urllib.parse import quote

from fastapi import Depends, HTTPException, Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import User
from app.states import Role

SESSION_COOKIE = "videoai_session"
SESSION_MAX_AGE_SECONDS = 12 * 3600
CSRF_MAX_AGE_SECONDS = 8 * 3600
ANON_CSRF_BIND = "anon"

_PBKDF2_ALGO = "pbkdf2_sha256"
_PBKDF2_ITERATIONS = 210_000
_SALT_BYTES = 16


# --- Password hashing -------------------------------------------------------


def hash_password(password: str, *, iterations: int = _PBKDF2_ITERATIONS) -> str:
    """Hash a password with a fresh per-user salt.

    Format: ``pbkdf2_sha256$<iterations>$<salt_hex>$<digest_hex>``.
    """
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_PBKDF2_ALGO}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verification against a stored ``hash_password`` value."""
    try:
        algo, iterations_s, salt_hex, digest_hex = stored.split("$")
        iterations = int(iterations_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, AttributeError):
        return False
    if algo != _PBKDF2_ALGO or iterations < 1:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, expected)


# --- Signed session cookie --------------------------------------------------


def _serializer(salt: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_settings().secret_key, salt=salt)


def create_session_token(user_id: str) -> str:
    return _serializer("videoai.session").dumps({"uid": user_id})


def read_session_token(token: str, max_age: int = SESSION_MAX_AGE_SECONDS) -> str | None:
    try:
        data = _serializer("videoai.session").loads(token, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return None
    uid = data.get("uid") if isinstance(data, dict) else None
    return uid if isinstance(uid, str) else None


# --- CSRF -------------------------------------------------------------------


def make_csrf_token(bind: str = ANON_CSRF_BIND) -> str:
    """Signed CSRF token bound to a session identity (user id, or anon pre-login)."""
    return _serializer("videoai.csrf").dumps({"bind": bind})


def verify_csrf_token(
    token: str, bind: str = ANON_CSRF_BIND, max_age: int = CSRF_MAX_AGE_SECONDS
) -> bool:
    try:
        data = _serializer("videoai.csrf").loads(token, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return False
    return isinstance(data, dict) and data.get("bind") == bind


# --- FastAPI dependencies ---------------------------------------------------


def get_current_user(
    request: Request, db: Annotated[Session, Depends(get_db)]
) -> User | None:
    """Resolve the logged-in user from the session cookie, or None."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    user_id = read_session_token(token)
    if user_id is None:
        return None
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        return None
    return user


def login_required(
    request: Request, user: Annotated[User | None, Depends(get_current_user)]
) -> User:
    """Page dependency: redirect anonymous visitors to /login."""
    if user is None:
        next_path = quote(request.url.path, safe="/")
        raise HTTPException(status_code=303, headers={"Location": f"/login?next={next_path}"})
    return user


def require_roles(*roles: Role) -> Callable[..., User]:
    """Role gate for mutating pages. Admin always passes."""
    allowed = {r.value for r in roles} | {Role.ADMIN.value}

    def dependency(user: Annotated[User, Depends(login_required)]) -> User:
        if user.role not in allowed:
            raise HTTPException(status_code=403, detail="Không đủ quyền truy cập trang này")
        return user

    return dependency
