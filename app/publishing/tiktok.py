"""TikTok publisher — MANUAL bundle by default, optional Upload-to-Inbox (PLAN.md §3).

The MVP never posts to TikTok silently. The default capability is MANUAL (the
operator posts from the exported bundle). The Upload-to-Inbox draft flow is
available only behind the DRAFT capability AND an explicit per-publish user
consent token carried in the context; without consent the adapter raises
PublishNeedsAction. Direct Post is intentionally not implemented.
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
    PublishTimeout,
)
from app.publishing.retry import raise_for_publish_status
from app.states import Capability

INBOX_INIT_ENDPOINT = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
STATUS_FETCH_ENDPOINT = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
TOKEN_ENDPOINT = "https://open.tiktokapis.com/v2/oauth/token/"

UPLOAD_SCOPE = "video.upload"

# Statuses proving the upload reached TikTok (probe result: do NOT re-upload).
_UPLOAD_EXISTS_STATUSES = {"PROCESSING_UPLOAD", "SEND_TO_USER_INBOX", "PUBLISH_COMPLETE"}


class TikTokPublisher(Publisher):
    platform = "tiktok"

    # -- contract ------------------------------------------------------------

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.DRAFT, Capability.MANUAL})

    def validate(
        self, target: PublishContext, rendition: Mapping[str, Any] | None = None
    ) -> None:
        problems: list[str] = []
        if not target.file_path:
            problems.append("a local file_path is required")
        if rendition is not None:
            duration = rendition.get("duration_seconds")
            if duration is not None and float(duration) > 600.0:
                problems.append("duration exceeds the 10-minute TikTok limit")
        if problems:
            raise PublishNeedsAction(
                "tiktok target validation failed", details={"problems": problems}
            )

    def prepare(self, ctx: PublishContext) -> dict[str, Any]:
        """Init the Upload-to-Inbox session. Requires explicit user consent."""
        self.check_cost_cap(ctx)
        self._require_consent(ctx)
        if not ctx.file_path:
            raise PublishNeedsAction("tiktok inbox upload requires a local file_path")
        size = Path(ctx.file_path).stat().st_size

        def _init() -> dict[str, Any]:
            response = self._send(
                lambda: self.client.post(
                    INBOX_INIT_ENDPOINT,
                    json={
                        "source_info": {
                            "source": "FILE_UPLOAD",
                            "video_size": size,
                            "chunk_size": size,
                            "total_chunk_count": 1,
                        }
                    },
                    headers=self._auth_headers(),
                )
            )
            raise_for_publish_status(response, context="tiktok.inbox.init")
            data = response.json().get("data", {})
            if not data.get("publish_id") or not data.get("upload_url"):
                raise PublishError("tiktok inbox init returned no publish_id/upload_url")
            return {"publish_id": data["publish_id"], "upload_url": data["upload_url"]}

        session = self._run(
            _init,
            lambda: self._ambiguous_outcome("inbox.init"),
        )
        session["video_size"] = size
        self.record_api_call("inbox.init")
        return session

    def upload(self, ctx: PublishContext, session: dict[str, Any]) -> dict[str, Any]:
        self._require_consent(ctx)
        if not ctx.file_path:
            raise PublishNeedsAction("tiktok inbox upload requires a local file_path")
        data = Path(ctx.file_path).read_bytes()
        upload_url = session["upload_url"]
        publish_id = session["publish_id"]

        def _put() -> dict[str, Any]:
            response = self._send(
                lambda: self.client.put(
                    upload_url,
                    content=data,
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Range": f"bytes 0-{len(data) - 1}/{len(data)}",
                    },
                )
            )
            raise_for_publish_status(response, context="tiktok.inbox.upload")
            return {"uploaded": True}

        def _probe() -> dict[str, Any] | None:
            try:
                status = self.poll_status(publish_id)
            except (PublishTimeout, PublishError):
                return None
            if status.get("status") in _UPLOAD_EXISTS_STATUSES:
                return {"uploaded": True, "recovered_from_status_probe": True}
            return None

        payload = self._run(_put, _probe)
        session["upload_response"] = payload
        self.record_api_call("inbox.upload")
        return session

    def finalize(self, ctx: PublishContext, session: dict[str, Any]) -> PublishResult:
        """Inbox drafts are finished by the user inside the TikTok app."""
        publish_id = session.get("publish_id")
        if not publish_id:
            raise PublishError("tiktok session has no publish_id")
        return PublishResult(
            remote_post_id=str(publish_id),
            remote_status={
                "state": "SEND_TO_USER_INBOX",
                "note": "user must confirm the draft inside the TikTok app",
            },
            raw=session.get("upload_response", {}),
        )

    def recover_upload(
        self, ctx: PublishContext, session: dict[str, Any]
    ) -> dict[str, Any] | None:
        publish_id = session.get("publish_id")
        if not publish_id:
            return None
        try:
            status = self.poll_status(str(publish_id))
        except PublishError:
            return None
        if status.get("status") not in _UPLOAD_EXISTS_STATUSES:
            return None
        recovered = dict(session)
        recovered["upload_response"] = {
            "uploaded": True,
            "recovered_from_status_probe": True,
        }
        return recovered

    def recover_finalize(
        self, ctx: PublishContext, session: dict[str, Any]
    ) -> PublishResult | None:
        # Upload-to-Inbox finalize is a local projection of publish_id; the
        # user performs the actual publish in TikTok.
        if not session.get("publish_id"):
            return None
        return self.finalize(ctx, session)

    def poll_status(self, remote_post_id: str) -> dict[str, Any]:
        def _fetch() -> dict[str, Any]:
            response = self._send(
                lambda: self.client.post(
                    STATUS_FETCH_ENDPOINT,
                    json={"publish_id": remote_post_id},
                    headers=self._auth_headers(),
                )
            )
            raise_for_publish_status(response, context="tiktok.status.fetch")
            return response.json().get("data", {})

        data = self._run(_fetch)
        self.record_api_call("status.fetch")
        return data

    def refresh_credentials(self) -> dict[str, Any]:
        refresh_token = self._credentials.get("refresh_token")
        if not refresh_token:
            raise PublishNeedsAction(
                "tiktok token expired and no refresh_token is stored; reconnect the account"
            )
        response = self._send(
            lambda: self.client.post(
                TOKEN_ENDPOINT,
                data={
                    "client_key": self._settings.tiktok_client_key,
                    "client_secret": self._settings.tiktok_client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )
        )
        if response.status_code >= 400:
            raise PublishNeedsAction(
                "tiktok token refresh failed; reconnect the account",
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
        if UPLOAD_SCOPE in scopes:
            # Draft only: the user still confirms in-app; never auto/direct post.
            return Capability.DRAFT
        return Capability.MANUAL

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _require_consent(ctx: PublishContext) -> None:
        if not ctx.consent_token:
            raise PublishNeedsAction(
                "tiktok upload requires explicit user consent; export the manual "
                "bundle or collect a consent token first",
                details={"reason": "consent_required"},
            )

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._credentials.get('access_token', '')}"}
