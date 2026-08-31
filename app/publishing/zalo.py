"""Zalo OA publisher (PLAN.md §3).

When the connected account's capability is DIRECT/SCHEDULE (a paid OA with
content-creation permission) the adapter drives the OA video upload + article
create/verify APIs. Otherwise every operation degrades to NEEDS_ACTION with a
complete manual bundle (MP4 reference, caption, thumbnail) plus a deep link to
OA Manager, so the operator can post by hand.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.publishing.base import (
    AccountLike,
    PublishContext,
    Publisher,
    PublishError,
    PublishNeedsAction,
    PublishResult,
)
from app.publishing.bundle import OA_MANAGER_DEEP_LINK, build_bundle
from app.publishing.retry import raise_for_publish_status
from app.states import Capability

UPLOAD_VIDEO_ENDPOINT = "https://openapi.zalo.me/v2.0/oa/upload/video"
ARTICLE_CREATE_ENDPOINT = "https://openapi.zalo.me/v2.0/article/create"
ARTICLE_VERIFY_ENDPOINT = "https://openapi.zalo.me/v2.0/article/verify"
TOKEN_ENDPOINT = "https://oauth.zaloapp.com/v4/oa/access_token"

CONTENT_SCOPES = {"article.create", "oa.content"}

_API_CAPABLE = {Capability.DIRECT, Capability.SCHEDULE}


def _check_zalo_payload(payload: dict[str, Any], context: str) -> dict[str, Any]:
    """Zalo returns HTTP 200 with an error code in the body; map nonzero to NEEDS_ACTION."""
    error = payload.get("error", 0)
    if error not in (0, None):
        raise PublishNeedsAction(
            f"zalo api error {error} ({context})",
            details={"error": error, "message": payload.get("message", ""), "context": context},
        )
    return payload


class ZaloPublisher(Publisher):
    platform = "zalo"

    def __init__(
        self,
        credentials: dict[str, Any] | None = None,
        *,
        capability: Capability | str = Capability.MANUAL,
        bundle_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(credentials, **kwargs)
        self._capability = Capability(capability)
        self._bundle_dir = bundle_dir

    # -- contract ------------------------------------------------------------

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.DIRECT, Capability.SCHEDULE, Capability.MANUAL})

    def validate(
        self, target: PublishContext, rendition: Mapping[str, Any] | None = None
    ) -> None:
        problems: list[str] = []
        if not target.title.strip():
            problems.append("title is required")
        if not (target.file_path or target.file_url):
            problems.append("either file_path or file_url is required")
        if problems:
            raise PublishNeedsAction(
                "zalo target validation failed", details={"problems": problems}
            )

    def prepare(self, ctx: PublishContext) -> dict[str, Any]:
        self.check_cost_cap(ctx)
        if self._capability not in _API_CAPABLE:
            self._needs_action_with_bundle(ctx)
        return {"api": "zalo_oa"}

    def recover_prepare(self, ctx: PublishContext) -> dict[str, Any] | None:
        # Zalo's prepare phase is entirely local and therefore safe to rebuild.
        if self._capability not in _API_CAPABLE:
            return None
        return {"api": "zalo_oa"}

    def upload(self, ctx: PublishContext, session: dict[str, Any]) -> dict[str, Any]:
        if self._capability not in _API_CAPABLE:
            self._needs_action_with_bundle(ctx)
        if not ctx.file_path:
            raise PublishNeedsAction("zalo OA video upload requires a local file_path")
        data = Path(ctx.file_path).read_bytes()
        file_name = Path(ctx.file_path).name

        def _upload() -> dict[str, Any]:
            response = self._send(
                lambda: self.client.post(
                    UPLOAD_VIDEO_ENDPOINT,
                    files={"file": (file_name, data, "video/mp4")},
                    headers=self._auth_headers(),
                )
            )
            raise_for_publish_status(response, context="zalo.upload.video")
            return _check_zalo_payload(response.json(), "zalo.upload.video")

        payload = self._run(
            _upload,
            lambda: self._ambiguous_outcome("upload.video"),
        )
        video_token = payload.get("data", {}).get("token")
        if not video_token:
            raise PublishError("zalo video upload returned no token")
        session["video_token"] = video_token
        self.record_api_call("upload.video")
        return session

    def finalize(self, ctx: PublishContext, session: dict[str, Any]) -> PublishResult:
        if self._capability not in _API_CAPABLE:
            self._needs_action_with_bundle(ctx)
        video_token = session.get("video_token")
        if not video_token:
            raise PublishError("zalo session has no video token; run upload first")
        body = {
            "type": "video",
            "title": ctx.title,
            "description": ctx.description,
            "status": "show",
            "video_id": video_token,
        }

        def _create() -> dict[str, Any]:
            response = self._send(
                lambda: self.client.post(
                    ARTICLE_CREATE_ENDPOINT, json=body, headers=self._auth_headers()
                )
            )
            raise_for_publish_status(response, context="zalo.article.create")
            return _check_zalo_payload(response.json(), "zalo.article.create")

        payload = self._run(
            _create,
            lambda: self._ambiguous_outcome("article.create"),
        )
        creation_token = payload.get("data", {}).get("token")
        if not creation_token:
            raise PublishError("zalo article create returned no token")
        self.record_api_call("article.create")
        return PublishResult(
            remote_post_id=str(creation_token),
            remote_status={"state": "created", "video_token": video_token},
            raw=payload,
        )

    def poll_status(self, remote_post_id: str) -> dict[str, Any]:
        def _verify() -> dict[str, Any]:
            response = self._send(
                lambda: self.client.post(
                    ARTICLE_VERIFY_ENDPOINT,
                    json={"token": remote_post_id},
                    headers=self._auth_headers(),
                )
            )
            raise_for_publish_status(response, context="zalo.article.verify")
            return _check_zalo_payload(response.json(), "zalo.article.verify")

        payload = self._run(_verify)
        self.record_api_call("article.verify")
        return payload.get("data", {})

    def refresh_credentials(self) -> dict[str, Any]:
        refresh_token = self._credentials.get("refresh_token")
        if not refresh_token:
            raise PublishNeedsAction(
                "zalo token expired and no refresh_token is stored; reconnect the OA"
            )
        response = self._send(
            lambda: self.client.post(
                TOKEN_ENDPOINT,
                data={
                    "app_id": self._settings.zalo_app_id,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                headers={"secret_key": self._settings.zalo_app_secret},
            )
        )
        if response.status_code >= 400:
            raise PublishNeedsAction(
                "zalo token refresh failed; reconnect the OA",
                details={"status_code": response.status_code},
            )
        token = response.json()
        self._credentials["access_token"] = token["access_token"]
        if "refresh_token" in token:
            self._credentials["refresh_token"] = token["refresh_token"]
        self.record_api_call("oauth.refresh")
        return dict(self._credentials)

    def probe_capability(self, account: AccountLike) -> Capability:
        if account.status != "active":
            return Capability.BLOCKED
        scopes = set(account.scopes or [])
        if scopes & CONTENT_SCOPES:
            # No native scheduling on Zalo OA; the internal scheduler calls at due time.
            return Capability.DIRECT
        return Capability.MANUAL

    # -- helpers -------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return {"access_token": str(self._credentials.get("access_token", ""))}

    def _needs_action_with_bundle(self, ctx: PublishContext) -> None:
        out_dir = ctx.extra.get("bundle_dir") or self._bundle_dir or "bundles"
        manifest = build_bundle(ctx, self.platform, out_dir, deep_link=OA_MANAGER_DEEP_LINK)
        raise PublishNeedsAction(
            "zalo OA lacks direct publish permission; a manual bundle was created",
            details={"bundle": manifest, "deep_link": OA_MANAGER_DEEP_LINK},
        )
