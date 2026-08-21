"""Facebook Page Reels publisher — START -> upload -> FINISH (PLAN.md §3).

Uses the Reels Publishing API on a Page: an upload session is opened with
``upload_phase=start``, the binary (or a hosted URL) goes to the returned
rupload URL, and ``upload_phase=finish`` publishes — with
``scheduled_publish_time`` when the requested time falls inside the native
scheduling window (10 minutes to 30 days ahead).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
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

GRAPH_BASE = "https://graph.facebook.com/v21.0"

SCHEDULE_MIN_AHEAD = timedelta(minutes=10)
SCHEDULE_MAX_AHEAD = timedelta(days=30)

REQUIRED_SCOPES = {"pages_show_list", "pages_read_engagement", "pages_manage_posts"}


class FacebookPublisher(Publisher):
    platform = "facebook"

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
            problems.append("either file_path or a hosted file_url is required")
        if target.scheduled_at is not None:
            problem = self._schedule_problem(target.scheduled_at)
            if problem:
                problems.append(problem)
        if rendition is not None:
            duration = rendition.get("duration_seconds")
            if duration is not None and float(duration) > 90.0:
                problems.append("duration exceeds the 90s Reels limit")
        if problems:
            raise PublishNeedsAction(
                "facebook target validation failed", details={"problems": problems}
            )

    def prepare(self, ctx: PublishContext) -> dict[str, Any]:
        """START phase: open the Reels upload session on the Page."""
        self.check_cost_cap(ctx)
        page_id = self._page_id()

        def _start() -> dict[str, Any]:
            response = self._send(
                lambda: self.client.post(
                    f"{GRAPH_BASE}/{page_id}/video_reels",
                    data={"upload_phase": "start", "access_token": self._token()},
                )
            )
            raise_for_publish_status(response, context="facebook.video_reels.start")
            payload = response.json()
            if not payload.get("video_id") or not payload.get("upload_url"):
                raise PublishError("reels start phase returned no video_id/upload_url")
            return {"video_id": payload["video_id"], "upload_url": payload["upload_url"]}

        session = self._run(_start)
        self.record_api_call("video_reels.start")
        return session

    def upload(self, ctx: PublishContext, session: dict[str, Any]) -> dict[str, Any]:
        upload_url = session["upload_url"]
        video_id = session["video_id"]

        if ctx.file_url:
            headers = {"Authorization": f"OAuth {self._token()}", "file_url": ctx.file_url}
            content = b""
        elif ctx.file_path:
            content = Path(ctx.file_path).read_bytes()
            headers = {
                "Authorization": f"OAuth {self._token()}",
                "offset": "0",
                "file_size": str(len(content)),
            }
        else:
            raise PublishNeedsAction("facebook upload requires file_path or file_url")

        def _upload() -> dict[str, Any]:
            response = self._send(
                lambda: self.client.post(upload_url, content=content, headers=headers)
            )
            raise_for_publish_status(response, context="facebook.video_reels.upload")
            return response.json()

        def _probe() -> dict[str, Any] | None:
            """Query the video status; recover without re-sending bytes."""
            try:
                response = self._send(
                    lambda: self.client.get(
                        f"{GRAPH_BASE}/{video_id}",
                        params={"fields": "status", "access_token": self._token()},
                    )
                )
            except PublishTimeout:
                return None
            if response.status_code >= 400:
                return None
            status = response.json().get("status", {})
            if status.get("uploading_phase", {}).get("status") == "complete":
                return {"success": True, "recovered_from_status_probe": True}
            return None

        payload = self._run(_upload, _probe)
        session["upload_response"] = payload
        self.record_api_call("video_reels.upload")
        return session

    def finalize(self, ctx: PublishContext, session: dict[str, Any]) -> PublishResult:
        """FINISH phase: publish immediately or schedule natively."""
        page_id = self._page_id()
        video_id = session["video_id"]
        video_state, scheduled_ts = self._schedule_state(ctx)
        form: dict[str, Any] = {
            "access_token": self._token(),
            "video_id": video_id,
            "upload_phase": "finish",
            "video_state": video_state,
            "description": self._caption(ctx),
        }
        if scheduled_ts is not None:
            form["scheduled_publish_time"] = str(scheduled_ts)

        def _finish() -> dict[str, Any]:
            response = self._send(
                lambda: self.client.post(f"{GRAPH_BASE}/{page_id}/video_reels", data=form)
            )
            raise_for_publish_status(response, context="facebook.video_reels.finish")
            return response.json()

        payload = self._run(_finish)
        self.record_api_call("video_reels.finish")
        return PublishResult(
            remote_post_id=str(payload.get("post_id") or video_id),
            remote_status={
                "video_state": video_state,
                "scheduled_publish_time": scheduled_ts,
                "video_id": video_id,
            },
            raw=payload,
        )

    def poll_status(self, remote_post_id: str) -> dict[str, Any]:
        def _get() -> dict[str, Any]:
            response = self._send(
                lambda: self.client.get(
                    f"{GRAPH_BASE}/{remote_post_id}",
                    params={"fields": "status", "access_token": self._token()},
                )
            )
            raise_for_publish_status(response, context="facebook.video.status")
            return response.json()

        payload = self._run(_get)
        self.record_api_call("video.status")
        return payload.get("status", {})

    def refresh_credentials(self) -> dict[str, Any]:
        current = self._token()
        if not current:
            raise PublishNeedsAction(
                "facebook token missing; reconnect the Page account"
            )
        response = self._send(
            lambda: self.client.get(
                f"{GRAPH_BASE}/oauth/access_token",
                params={
                    "grant_type": "fb_exchange_token",
                    "client_id": self._settings.facebook_app_id,
                    "client_secret": self._settings.facebook_app_secret,
                    "fb_exchange_token": current,
                },
            )
        )
        if response.status_code >= 400:
            raise PublishNeedsAction(
                "facebook token refresh failed; reconnect the Page account",
                details={"status_code": response.status_code},
            )
        token = response.json()
        self._credentials["page_access_token"] = token["access_token"]
        self.record_api_call("oauth.refresh")
        return dict(self._credentials)

    def probe_capability(self, account: AccountLike) -> Capability:
        if account.status != "active":
            return Capability.BLOCKED
        scopes = set(account.scopes or [])
        if REQUIRED_SCOPES.issubset(scopes):
            return Capability.SCHEDULE  # native scheduled_publish_time available
        if "pages_manage_posts" in scopes:
            return Capability.DIRECT
        return Capability.MANUAL

    # -- helpers -------------------------------------------------------------

    def _page_id(self) -> str:
        page_id = self._credentials.get("page_id")
        if not page_id:
            raise PublishNeedsAction("facebook credentials are missing page_id")
        return str(page_id)

    def _token(self) -> str:
        return str(self._credentials.get("page_access_token", ""))

    @staticmethod
    def _caption(ctx: PublishContext) -> str:
        parts = [ctx.title, ctx.description, " ".join(ctx.hashtags)]
        return "\n\n".join(p for p in parts if p.strip())

    @staticmethod
    def _schedule_problem(scheduled_at: datetime) -> str | None:
        if scheduled_at.tzinfo is None:
            return "scheduled_at must be timezone-aware (UTC)"
        delta = scheduled_at - datetime.now(UTC)
        if delta < SCHEDULE_MIN_AHEAD or delta > SCHEDULE_MAX_AHEAD:
            return (
                "scheduled_at is outside the native scheduling window "
                "(10 minutes to 30 days ahead)"
            )
        return None

    def _schedule_state(self, ctx: PublishContext) -> tuple[str, int | None]:
        if ctx.scheduled_at is None:
            return "PUBLISHED", None
        problem = self._schedule_problem(ctx.scheduled_at)
        if problem:
            raise PublishNeedsAction(problem, details={"scheduled_at": str(ctx.scheduled_at)})
        return "SCHEDULED", int(ctx.scheduled_at.timestamp())
