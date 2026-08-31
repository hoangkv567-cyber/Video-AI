"""OAuth start/callback flows for platform connections (PLAN.md §2/§5).

``/oauth/{platform}/start`` issues a random ``state`` plus a PKCE S256
challenge, stores both in a signed, short-lived cookie and redirects to the
platform's authorize URL. ``/oauth/{platform}/callback`` verifies the state,
exchanges the code through an injectable httpx client, stores the credentials
ENCRYPTED (app.publishing.crypto) on a ConnectedAccount and runs a capability
probe via app.publishing.registry. No token ever reaches a log.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.config import Settings, get_settings
from app.db import get_db
from app.errors import NotFound, UpstreamError, ValidationFailed
from app.models import ConnectedAccount, User
from app.publishing.crypto import encrypt_credentials
from app.publishing.registry import create_publisher, probe_and_record
from app.states import Capability, Platform, Role

router = APIRouter(prefix="/oauth", tags=["oauth"])

STATE_COOKIE = "videoai_oauth"
STATE_MAX_AGE_SECONDS = 600


@dataclass(frozen=True)
class OAuthEndpoints:
    authorize_url: str
    token_url: str
    scopes: tuple[str, ...]


_ENDPOINTS: dict[str, OAuthEndpoints] = {
    Platform.YOUTUBE.value: OAuthEndpoints(
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=("https://www.googleapis.com/auth/youtube.upload",),
    ),
    Platform.FACEBOOK.value: OAuthEndpoints(
        authorize_url="https://www.facebook.com/v21.0/dialog/oauth",
        token_url="https://graph.facebook.com/v21.0/oauth/access_token",
        scopes=("pages_show_list", "pages_read_engagement", "pages_manage_posts"),
    ),
    Platform.TIKTOK.value: OAuthEndpoints(
        authorize_url="https://www.tiktok.com/v2/auth/authorize/",
        token_url="https://open.tiktokapis.com/v2/oauth/token/",
        scopes=("user.info.basic", "video.upload"),
    ),
    Platform.ZALO.value: OAuthEndpoints(
        authorize_url="https://oauth.zaloapp.com/v4/oa/permission",
        token_url="https://oauth.zaloapp.com/v4/oa/access_token",
        scopes=(),
    ),
}


def _client_pair(platform: str, settings: Settings) -> tuple[str, str]:
    pairs = {
        Platform.YOUTUBE.value: (settings.youtube_client_id, settings.youtube_client_secret),
        Platform.FACEBOOK.value: (settings.facebook_app_id, settings.facebook_app_secret),
        Platform.TIKTOK.value: (settings.tiktok_client_key, settings.tiktok_client_secret),
        Platform.ZALO.value: (settings.zalo_app_id, settings.zalo_app_secret),
    }
    return pairs[platform]


def _require_platform(platform: str) -> OAuthEndpoints:
    endpoints = _ENDPOINTS.get(platform)
    if endpoints is None:
        raise NotFound(f"unknown platform {platform!r}")
    return endpoints


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_settings().secret_key, salt="videoai.oauth")


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def get_http_client() -> Iterator[httpx.Client]:
    """Injectable httpx client; tests override with a MockTransport client."""
    client = httpx.Client(timeout=30.0)
    try:
        yield client
    finally:
        client.close()


def _redirect_uri(platform: str, settings: Settings) -> str:
    return f"{settings.public_base_url.rstrip('/')}/oauth/{platform}/callback"


@router.get("/{platform}/start")
def oauth_start(
    platform: str,
    user: Annotated[User, Depends(require_roles(Role.ADMIN))],
) -> RedirectResponse:
    endpoints = _require_platform(platform)
    settings = get_settings()
    client_id, _ = _client_pair(platform, settings)

    state = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(48)  # 64 chars, within the PKCE 43-128 window
    params = {
        "client_id": client_id,
        "redirect_uri": _redirect_uri(platform, settings),
        "response_type": "code",
        "state": state,
        "code_challenge": _pkce_challenge(verifier),
        "code_challenge_method": "S256",
    }
    if endpoints.scopes:
        params["scope"] = " ".join(endpoints.scopes)

    response = RedirectResponse(
        url=f"{endpoints.authorize_url}?{urlencode(params)}", status_code=302
    )
    payload = {
        "platform": platform,
        "state": state,
        "verifier": verifier,
        "user_id": user.id,
    }
    response.set_cookie(
        STATE_COOKIE,
        _serializer().dumps(payload),
        max_age=STATE_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=settings.app_env not in {"dev", "test"},
    )
    return response


def _read_state_cookie(
    request: Request,
    platform: str,
    state: str,
    user_id: str,
) -> dict[str, Any]:
    token = request.cookies.get(STATE_COOKIE)
    if not token:
        raise ValidationFailed("missing oauth state cookie", code="oauth_state_missing")
    try:
        data = _serializer().loads(token, max_age=STATE_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired) as exc:
        raise ValidationFailed(
            "oauth state cookie is invalid or expired", code="oauth_state_invalid"
        ) from exc
    if (
        not isinstance(data, dict)
        or data.get("platform") != platform
        or data.get("state") != state
        or data.get("user_id") != user_id
    ):
        raise ValidationFailed("oauth state mismatch", code="oauth_state_mismatch")
    return data


def _exchange_code(
    client: httpx.Client,
    endpoints: OAuthEndpoints,
    *,
    platform: str,
    code: str,
    verifier: str,
    settings: Settings,
) -> dict[str, Any]:
    client_id, client_secret = _client_pair(platform, settings)
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": _redirect_uri(platform, settings),
        "code_verifier": verifier,
    }
    try:
        response = client.post(endpoints.token_url, data=form)
    except httpx.HTTPError as exc:
        raise UpstreamError(f"{platform} token exchange failed: {type(exc).__name__}") from exc
    if response.status_code >= 400:
        # Never echo the response body: it can contain token material.
        raise UpstreamError(
            f"{platform} token exchange returned HTTP {response.status_code}",
            details={"status_code": response.status_code},
        )
    data = response.json()
    if not isinstance(data, dict) or not data.get("access_token"):
        raise UpstreamError(f"{platform} token exchange returned no access_token")
    return data


@router.get("/{platform}/callback")
def oauth_callback(
    platform: str,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    client: Annotated[httpx.Client, Depends(get_http_client)],
    user: Annotated[User, Depends(require_roles(Role.ADMIN))],
    code: str = "",
    state: str = "",
) -> RedirectResponse:
    endpoints = _require_platform(platform)
    if not code or not state:
        raise ValidationFailed("code and state query parameters are required")
    settings = get_settings()
    cookie = _read_state_cookie(request, platform, state, user.id)

    token_data = _exchange_code(
        client,
        endpoints,
        platform=platform,
        code=code,
        verifier=str(cookie.get("verifier", "")),
        settings=settings,
    )

    scopes_raw = token_data.get("scope", "")
    scopes = [s for s in str(scopes_raw).replace(",", " ").split() if s]

    if platform == "facebook" and "access_token" in token_data:
        try:
            perms_resp = client.get(
                "https://graph.facebook.com/v21.0/me/permissions",
                params={"access_token": token_data["access_token"]},
            )
            if perms_resp.status_code < 400:
                granted = [
                    p["permission"]
                    for p in perms_resp.json().get("data", [])
                    if p.get("status") == "granted"
                ]
                if granted:
                    scopes = granted
            pages_resp = client.get(
                "https://graph.facebook.com/v21.0/me/accounts",
                params={"access_token": token_data["access_token"]},
            )
            if pages_resp.status_code < 400:
                pages = pages_resp.json().get("data", [])
                if pages:
                    page = pages[-1] if len(pages) > 1 else pages[0]
                    token_data["page_id"] = page["id"]
                    token_data["page_access_token"] = page["access_token"]
                    token_data["page_name"] = page.get("name", "")
        except Exception:
            pass

    expires_at: datetime | None = None
    expires_in = token_data.get("expires_in")
    if isinstance(expires_in, int | float):
        expires_at = datetime.now(UTC) + timedelta(seconds=float(expires_in))

    account = (
        db.query(ConnectedAccount)
        .filter(ConnectedAccount.platform == platform)
        .order_by(ConnectedAccount.created_at.desc())
        .first()
    )
    if account is None:
        account = ConnectedAccount(platform=platform)
        db.add(account)
    account.encrypted_credentials = encrypt_credentials(token_data)
    account.scopes = scopes
    account.token_expires_at = expires_at
    account.status = "active"
    remote_id = token_data.get("page_id") or token_data.get("open_id") or token_data.get("oa_id") or ""
    if remote_id:
        account.remote_account_id = str(remote_id)
    if token_data.get("page_name"):
        account.display_name = str(token_data["page_name"])
    db.flush()

    try:
        publisher = create_publisher(
            platform, dict(token_data), client=client, settings=settings
        )
        probe_and_record(account, publisher)
    except Exception:
        # A probe crash must never lose the stored (encrypted) connection.
        account.capability = Capability.MANUAL.value
        account.last_probe_result = {"capability": Capability.MANUAL.value, "error": "probe_crashed"}

    db.commit()
    response = RedirectResponse(url="/connections", status_code=303)
    response.delete_cookie(STATE_COOKIE)
    return response
