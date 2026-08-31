"""YouTube Shorts publisher — videos.insert resumable upload (PLAN.md §3).

Defaults are deliberately conservative: an unaudited API project must never
default to public, so privacy is forced to ``private`` unless the publisher is
constructed with ``audited_project=True``. Scheduling uses ``publishAt`` (which
itself requires private status until the publish time). The AI/synthetic-media
disclosure and madeForKids flags come from the plan's Disclosure block.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
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
    to_utc_z,
)
from app.publishing.retry import raise_for_publish_status
from app.schemas.videoplan import DESCRIPTION_MAX, TITLE_MAX
from app.states import Capability

UPLOAD_ENDPOINT = "https://www.googleapis.com/upload/youtube/v3/videos"
VIDEOS_ENDPOINT = "https://www.googleapis.com/youtube/v3/videos"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
SHORTS_MAX_SECONDS = 180.0
ALLOWED_PRIVACY = {"private", "public", "unlisted"}


class YouTubePublisher(Publisher):
    platform = "youtube"

    def __init__(
        self,
        credentials: dict[str, Any] | None = None,
        *,
        audited_project: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(credentials, **kwargs)
        self._audited_project = audited_project

    # -- contract ------------------------------------------------------------

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.DIRECT, Capability.SCHEDULE, Capability.MANUAL})

    def validate(
        self, target: PublishContext, rendition: Mapping[str, Any] | None = None
    ) -> None:
        problems: list[str] = []
        if not target.title.strip():
            problems.append("title is required")
        elif len(target.title) > TITLE_MAX:
            problems.append(f"title exceeds {TITLE_MAX} characters")
        if len(target.description) > DESCRIPTION_MAX:
            problems.append(f"description exceeds {DESCRIPTION_MAX} characters")
        if not target.file_path:
            problems.append("a local file_path is required for the resumable upload")
        if target.scheduled_at is not None:
            if target.scheduled_at.tzinfo is None:
                problems.append("scheduled_at must be timezone-aware (UTC)")
            elif target.scheduled_at <= datetime.now(UTC):
                problems.append("scheduled_at must be in the future")
        if rendition is not None:
            duration = rendition.get("duration_seconds")
            if duration is not None and float(duration) > SHORTS_MAX_SECONDS:
                problems.append(f"duration exceeds the Shorts limit of {SHORTS_MAX_SECONDS:.0f}s")
        if problems:
            raise PublishNeedsAction(
                "youtube target validation failed", details={"problems": problems}
            )

    def prepare(self, ctx: PublishContext) -> dict[str, Any]:
        """Open the resumable upload session (videos.insert init)."""
        self.check_cost_cap(ctx)
        body = {
            "snippet": {
                "title": ctx.title,
                "description": ctx.description,
                "tags": [h.lstrip("#") for h in ctx.hashtags],
                "categoryId": "28",  # Science & Technology
            },
            "status": self._status_body(ctx),
        }

        def _init() -> dict[str, Any]:
            response = self._send(
                lambda: self.client.post(
                    UPLOAD_ENDPOINT,
                    params={"uploadType": "resumable", "part": "snippet,status"},
                    json=body,
                    headers=self._auth_headers(),
                )
            )
            raise_for_publish_status(response, context="youtube.videos.insert.init")
            upload_url = response.headers.get("Location")
            if not upload_url:
                raise PublishError("resumable init returned no upload Location header")
            return {"upload_url": upload_url}

        session = self._run(
            _init,
            lambda: self._ambiguous_outcome("videos.insert.init"),
        )
        self.record_api_call("videos.insert.init")
        return session

    def upload(self, ctx: PublishContext, session: dict[str, Any]) -> dict[str, Any]:
        if not ctx.file_path:
            raise PublishNeedsAction("youtube upload requires a local file_path")
        data = Path(ctx.file_path).read_bytes()
        upload_url = session["upload_url"]

        def _put() -> dict[str, Any]:
            response = self._send(
                lambda: self.client.put(
                    upload_url,
                    content=data,
                    headers={**self._auth_headers(), "Content-Type": "video/mp4"},
                )
            )
            raise_for_publish_status(response, context="youtube.videos.insert.upload")
            return response.json()

        def _probe() -> dict[str, Any] | None:
            """Resumable status query — never re-sends bytes (no duplicate upload)."""
            try:
                response = self._send(
                    lambda: self.client.put(
                        upload_url,
                        headers={
                            **self._auth_headers(),
                            "Content-Range": f"bytes */{len(data)}",
                        },
                    )
                )
            except PublishTimeout:
                return None
            if response.status_code in (200, 201):
                return response.json()  # upload already completed remotely
            return None  # 308 (incomplete) or error: safe to retry the upload

        payload = self._run(_put, _probe)
        session["video"] = payload
        session["video_id"] = payload.get("id")
        self.record_api_call("videos.insert.upload")
        return session

    def finalize(self, ctx: PublishContext, session: dict[str, Any]) -> PublishResult:
        video = session.get("video") or {}
        video_id = session.get("video_id")
        if not video_id:
            raise PublishError("resumable upload did not return a video id")
        status = video.get("status", {})
        return PublishResult(
            remote_post_id=str(video_id),
            remote_status={
                "upload_status": status.get("uploadStatus"),
                "privacy_status": status.get("privacyStatus"),
                "publish_at": status.get("publishAt"),
            },
            raw=video,
        )

    def recover_upload(
        self, ctx: PublishContext, session: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Query the resumable session; never create a second video resource."""
        if not ctx.file_path or not session.get("upload_url"):
            return None
        data_size = Path(ctx.file_path).stat().st_size
        try:
            response = self._send(
                lambda: self.client.put(
                    str(session["upload_url"]),
                    headers={
                        **self._auth_headers(),
                        "Content-Range": f"bytes */{data_size}",
                    },
                )
            )
        except PublishTimeout:
            return None
        if response.status_code not in (200, 201):
            # 308 proves that the same resumable session is still available,
            # but a blind full replay after a process death is unnecessary.
            return None
        payload = response.json()
        if not payload.get("id"):
            return None
        recovered = dict(session)
        recovered["video"] = payload
        recovered["video_id"] = payload["id"]
        self.record_api_call("videos.insert.recover")
        return recovered

    def recover_finalize(
        self, ctx: PublishContext, session: dict[str, Any]
    ) -> PublishResult | None:
        # YouTube finalization is local: videos.insert already created the
        # resource and returned all fields used by ``finalize``.
        if not session.get("video_id"):
            return None
        return self.finalize(ctx, session)

    def poll_status(self, remote_post_id: str) -> dict[str, Any]:
        """videos.list with status + processingDetails."""

        def _get() -> dict[str, Any]:
            response = self._send(
                lambda: self.client.get(
                    VIDEOS_ENDPOINT,
                    params={"part": "status,processingDetails", "id": remote_post_id},
                    headers=self._auth_headers(),
                )
            )
            raise_for_publish_status(response, context="youtube.videos.list")
            return response.json()

        data = self._run(_get)
        self.record_api_call("videos.list")
        items = data.get("items", [])
        if not items:
            return {"exists": False}
        item = items[0]
        return {
            "exists": True,
            "status": item.get("status", {}),
            "processing": item.get("processingDetails", {}),
        }

    def refresh_credentials(self) -> dict[str, Any]:
        refresh_token = self._credentials.get("refresh_token")
        if not refresh_token:
            raise PublishNeedsAction(
                "youtube token expired and no refresh_token is stored; reconnect the account"
            )
        response = self._send(
            lambda: self.client.post(
                TOKEN_ENDPOINT,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": self._settings.youtube_client_id,
                    "client_secret": self._settings.youtube_client_secret,
                },
            )
        )
        if response.status_code >= 400:
            raise PublishNeedsAction(
                "youtube token refresh failed; reconnect the account",
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
            # publishAt scheduling works with the upload scope alone.
            return Capability.SCHEDULE
        return Capability.MANUAL

    # -- helpers -------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._credentials.get('access_token', '')}"}

    def _effective_privacy(self, ctx: PublishContext) -> str:
        privacy = ctx.privacy if ctx.privacy in ALLOWED_PRIVACY else "private"
        if ctx.scheduled_at is not None:
            return "private"  # publishAt requires private until the publish time
        if privacy == "public" and not self._audited_project:
            return "private"  # unaudited project must not default public
        return privacy

    def _status_body(self, ctx: PublishContext) -> dict[str, Any]:
        status: dict[str, Any] = {
            "privacyStatus": self._effective_privacy(ctx),
            "selfDeclaredMadeForKids": ctx.disclosure.made_for_kids,
            "containsSyntheticMedia": ctx.disclosure.synthetic_media,
        }
        if ctx.scheduled_at is not None:
            status["publishAt"] = to_utc_z(ctx.scheduled_at)
        return status
