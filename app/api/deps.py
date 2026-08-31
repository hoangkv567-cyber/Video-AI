"""Lightweight API auth dependencies.

The dashboard's signed session cookie is honored when present; in dev/test an
``X-User-Id`` header identifies the caller so contract tests need no login
flow. Errors use the unified AppError envelope (401/403).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.errors import AppError
from app.models import User
from app.states import Role
from app.web.auth import get_current_user, verify_csrf_token


class Unauthorized(AppError):
    status_code = 401
    code = "unauthorized"


class Forbidden(AppError):
    status_code = 403
    code = "forbidden"


class CsrfRejected(AppError):
    status_code = 403
    code = "csrf_rejected"


def get_api_user(
    request: Request, db: Annotated[Session, Depends(get_db)]
) -> User | None:
    """Resolve the caller: session cookie first, X-User-Id header in dev/test."""
    user = get_current_user(request, db)
    if user is not None:
        if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
            token = request.headers.get("X-CSRF-Token", "")
            if not verify_csrf_token(token, user.id):
                raise CsrfRejected("missing or invalid CSRF token")
        return user
    if get_settings().app_env in {"dev", "test"}:
        user_id = request.headers.get("X-User-Id")
        if user_id:
            candidate = db.get(User, user_id)
            if candidate is not None and candidate.is_active:
                return candidate
    return None


def require_user(user: Annotated[User | None, Depends(get_api_user)]) -> User:
    if user is None:
        raise Unauthorized("authentication required")
    return user


def require_roles(*roles: Role) -> Callable[..., User]:
    """Role gate; admin always passes."""
    allowed = {r.value for r in roles} | {Role.ADMIN.value}

    def dependency(user: Annotated[User, Depends(require_user)]) -> User:
        if user.role not in allowed:
            raise Forbidden(
                "insufficient role for this operation",
                details={"required": sorted(allowed), "actual": user.role},
            )
        return user

    return dependency
