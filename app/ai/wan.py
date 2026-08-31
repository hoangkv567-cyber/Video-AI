"""Alibaba Cloud Model Studio wan2.7-i2v adapter (VideoProvider protocol).

Talks to the international (Singapore) DashScope-compatible endpoint:
submit an async video-synthesis task, poll it, and download the MP4 while
the signed URL is still valid (24 h). The first frame is passed inline as a
base64 data URI so keyframes never need a public URL.

The provider plugs into :class:`app.ai.veo.VeoService` unchanged: the
operation name is the DashScope ``task_id`` and succeeded polls carry the
downloaded bytes. Pricing lives in :class:`app.config.ModelConfig`
($0.10/s at 720P, 50 s free quota) and flows through the shared cost ledger
with the $6 hard cap.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from app.ai.base import OP_FAILED, OP_RUNNING, OP_SUCCEEDED, VideoOperation
from app.ai.veo import VeoService
from app.config import ModelConfig, get_model_config
from app.costs import wan_price_per_second
from app.errors import ProviderQuotaExhausted, UpstreamError

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com"
SUBMIT_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
TASK_PATH = "/api/v1/tasks/{task_id}"

# Task status values DashScope reports.
_PENDING_STATUSES = frozenset({"PENDING", "RUNNING"})
_FAILED_STATUSES = frozenset({"FAILED", "CANCELED", "UNKNOWN"})


def _data_uri(image_bytes: bytes) -> str:
    """PNG/JPEG detection for the inline first frame (no public URL needed)."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif image_bytes.startswith(b"\xff\xd8"):
        mime = "image/jpeg"
    else:
        raise UpstreamError(
            "wan first frame must be PNG or JPEG bytes",
            retryable=False,
        )
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


class WanVideoProvider:
    """``VideoProvider`` over Model Studio video-synthesis (wan2.7-i2v)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model_config: ModelConfig | None = None,
        http_client: httpx.Client | None = None,
        timeout: float = 120.0,
        resolution: str = "720P",
    ) -> None:
        from app.config import get_settings

        self._api_key = api_key if api_key is not None else get_settings().wan_api_key
        self._base = base_url.rstrip("/")
        self._cfg = model_config or get_model_config()
        self._http = http_client
        self._timeout = timeout
        self._resolution = resolution

    # -- HTTP plumbing ------------------------------------------------------

    def _headers(self, *, async_mode: bool) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if async_mode:
            headers["X-DashScope-Async"] = "enable"
        return headers

    def _request(
        self, method: str, url: str, *, json_body: dict[str, Any] | None = None
    ) -> httpx.Response:
        client = self._http
        owns = client is None
        if client is None:
            client = httpx.Client(timeout=self._timeout)
        try:
            response = client.request(
                method, url, json=json_body, headers=self._headers(async_mode=True)
            )
            if response.status_code < 400:
                return response
            retryable = response.status_code in {408, 429} or response.status_code >= 500
            details: dict[str, Any] = {
                "status_code": response.status_code,
                "endpoint": url,
            }
            provider_code: str | None = None
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                error = payload.get("error") or payload.get("output") or {}
                # DashScope puts top-level code/message on quota rejections.
                provider_code = str(
                    (error.get("code") if isinstance(error, dict) else None)
                    or payload.get("code")
                    or ""
                ) or None
                provider_message = str(
                    (error.get("message") if isinstance(error, dict) else None)
                    or payload.get("message")
                    or ""
                )
                if provider_code:
                    details["provider_code"] = provider_code
                if provider_message:
                    details["provider_message"] = provider_message[:300]
            if provider_code and "quota" in provider_code.lower():
                raise ProviderQuotaExhausted(
                    f"wan provider quota exhausted: {provider_code}",
                    details=details,
                )
            raise UpstreamError(
                f"wan request failed with HTTP {response.status_code}",
                retryable=retryable,
                details=details,
            )
        finally:
            if owns:
                client.close()

    # -- VideoProvider protocol ----------------------------------------------

    def submit(
        self,
        *,
        prompt: str,
        model_id: str,
        duration_seconds: float,
        keyframe_bytes: bytes | None = None,
    ) -> str:
        """Create one video-synthesis task; returns the DashScope task id.

        This mutating POST is deliberately attempted only once. A timeout,
        429, or 5xx cannot prove that the provider did not create a paid task;
        VeoService therefore leaves the durable intent ambiguous for explicit
        operator reconciliation instead of risking a duplicate submission.
        """
        if keyframe_bytes is None:
            raise UpstreamError(
                "wan2.7-i2v requires a keyframe (first_frame) image",
                retryable=False,
            )
        body: dict[str, Any] = {
            "model": model_id,
            "input": {
                "prompt": prompt,
                "media": [
                    {"type": "first_frame", "url": _data_uri(keyframe_bytes)}
                ],
            },
            "parameters": {
                "resolution": self._resolution,
                "duration": max(2, min(15, int(round(duration_seconds)))),
                "watermark": False,
                "prompt_extend": False,
            },
        }
        response = self._request("POST", self._base + SUBMIT_PATH, json_body=body)
        task_id = str((response.json().get("output") or {}).get("task_id", ""))
        if not task_id:
            raise UpstreamError(
                "wan submit response had no task_id",
                retryable=True,
                details={"endpoint": SUBMIT_PATH},
            )
        return task_id

    def poll(self, operation_name: str) -> VideoOperation:
        """Poll a task; a succeeded task is downloaded immediately."""
        response = self._request(
            "GET", self._base + TASK_PATH.format(task_id=operation_name)
        )
        output = response.json().get("output") or {}
        status = str(output.get("task_status", ""))
        if status in _PENDING_STATUSES:
            return VideoOperation(operation_name, OP_RUNNING)
        if status == "SUCCEEDED":
            video_url = str(output.get("video_url", ""))
            if not video_url:
                raise UpstreamError(
                    "wan task succeeded without video_url",
                    retryable=True,
                    details={"task_id": operation_name},
                )
            usage = response.json().get("usage") or {}
            duration = float(usage.get("output_video_duration") or 0.0)
            return VideoOperation(
                operation_name,
                OP_SUCCEEDED,
                video_bytes=self._download(video_url),
                duration_seconds=duration,
            )
        message = str(output.get("message") or status or "task failed")
        code = str(output.get("code") or status or "wan_task_failed")
        return VideoOperation(
            operation_name,
            OP_FAILED,
            error=f"{code}: {message}",
        )

    def _download(self, video_url: str) -> bytes:
        """Fetch the MP4 while the signed URL is live (24 h expiry)."""
        client = self._http
        owns = client is None
        if client is None:
            client = httpx.Client(timeout=self._timeout, follow_redirects=True)
        try:
            response = client.get(video_url)
            if response.status_code != 200 or not response.content:
                raise UpstreamError(
                    f"wan video download failed with HTTP {response.status_code}",
                    retryable=True,
                )
            return response.content
        finally:
            if owns:
                client.close()


class WanService(VeoService):
    """VeoService bound to Wan: model resolution, pricing and ledger kind.

    Submission-intent durability, projected→actual cost handoff, restart-safe
    re-polling and hero escalation all come from the Veo implementation.
    """

    cost_kind = "wan"

    def resolve_model(self, *, is_hero: bool = False, previously_failed: bool = False) -> str:
        # Single tier for now; escalation hooks exist if Alibaba ships a
        # faster/higher-quality variant worth the price step.
        return self._cfg.wan_i2v_model

    @staticmethod
    def _cost_note(scene_id: str) -> str:
        return f"wan scene {scene_id}"

    def price_per_second(self, model_id: str) -> float:
        return wan_price_per_second(model_id, self._cfg)
