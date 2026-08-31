"""WanVideoProvider / WanService tests against a mocked DashScope endpoint.

No network: submit/poll/download all run through httpx.MockTransport.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from app.ai.base import OP_FAILED, OP_RUNNING, OP_SUCCEEDED, VideoProvider
from app.ai.veo import InMemoryOperationStore
from app.ai.wan import WanService, WanVideoProvider
from app.config import ModelConfig
from app.costs import CostLedger, wan_price_per_second
from app.errors import CostCapExceeded, UpstreamError

CFG = ModelConfig()
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
TASK_ID = "task-abc-123"
VIDEO_URL = "https://oss.example.com/video.mp4?sig=1"
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"v" * 2048


def make_provider(
    handler, *, sleep=lambda _s: None
) -> WanVideoProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return WanVideoProvider(
        api_key="test-wan-key", model_config=CFG, http_client=client
    )


def submit_ok_handler(calls: list[dict[str, Any]]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            {"url": str(request.url), "headers": dict(request.headers), "json": json.loads(request.content)}
        )
        return httpx.Response(200, json={"output": {"task_id": TASK_ID, "task_status": "PENDING"}})

    return handler


class TestSubmit:
    def test_satisfies_video_provider_protocol(self) -> None:
        provider = make_provider(lambda request: httpx.Response(500))
        assert isinstance(provider, VideoProvider)

    def test_submit_builds_dashscope_async_body(self) -> None:
        calls: list[dict[str, Any]] = []
        provider = make_provider(submit_ok_handler(calls))
        task_id = provider.submit(
            prompt="camera pushes into the glowing chip",
            model_id=CFG.wan_i2v_model,
            duration_seconds=8.0,
            keyframe_bytes=PNG,
        )
        assert task_id == TASK_ID
        call = calls[0]
        assert call["headers"]["x-dashscope-async"] == "enable"
        assert call["headers"]["authorization"] == "Bearer test-wan-key"
        body = call["json"]
        assert body["model"] == CFG.wan_i2v_model
        assert body["input"]["prompt"] == "camera pushes into the glowing chip"
        media = body["input"]["media"]
        assert media[0]["type"] == "first_frame"
        assert media[0]["url"].startswith("data:image/png;base64,")
        assert base64.b64decode(media[0]["url"].split(",", 1)[1]) == PNG
        assert body["parameters"]["duration"] == 8
        assert body["parameters"]["resolution"] == "720P"
        assert body["parameters"]["watermark"] is False

    def test_submit_requires_keyframe(self) -> None:
        provider = make_provider(lambda request: httpx.Response(200))
        with pytest.raises(UpstreamError) as excinfo:
            provider.submit(
                prompt="p", model_id=CFG.wan_i2v_model, duration_seconds=8.0, keyframe_bytes=None
            )
        assert excinfo.value.retryable is False

    def test_submit_429_is_not_retried(self) -> None:
        calls: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(429, json={"error": {"code": "rate_limited"}})

        provider = make_provider(handler)
        with pytest.raises(UpstreamError) as excinfo:
            provider.submit(
                prompt="p",
                model_id=CFG.wan_i2v_model,
                duration_seconds=8.0,
                keyframe_bytes=PNG,
            )
        assert excinfo.value.retryable is True
        assert len(calls) == 1

    def test_submit_5xx_is_not_retried(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(503, json={"error": {"code": "ServiceUnavailable"}})

        provider = make_provider(handler)
        with pytest.raises(UpstreamError) as excinfo:
            provider.submit(
                prompt="p",
                model_id=CFG.wan_i2v_model,
                duration_seconds=8.0,
                keyframe_bytes=PNG,
            )
        assert excinfo.value.retryable is True
        assert len(calls) == 1

    def test_submit_missing_task_id_is_not_retried(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, json={"output": {"task_status": "PENDING"}})

        provider = make_provider(handler)
        with pytest.raises(UpstreamError) as excinfo:
            provider.submit(
                prompt="p",
                model_id=CFG.wan_i2v_model,
                duration_seconds=8.0,
                keyframe_bytes=PNG,
            )
        assert excinfo.value.retryable is True
        assert len(calls) == 1

    def test_submit_400_is_not_retried(self) -> None:
        calls: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(
                400, json={"error": {"code": "InvalidParameter", "message": "bad model"}}
            )

        provider = make_provider(handler)
        with pytest.raises(UpstreamError) as excinfo:
            provider.submit(
                prompt="p", model_id=CFG.wan_i2v_model, duration_seconds=8.0, keyframe_bytes=PNG
            )
        assert excinfo.value.retryable is False
        assert excinfo.value.details["provider_code"] == "InvalidParameter"
        assert len(calls) == 1


class TestPoll:
    def test_pending_and_running_map_to_op_running(self) -> None:
        for status in ("PENDING", "RUNNING"):
            provider = make_provider(
                lambda request, s=status: httpx.Response(
                    200, json={"output": {"task_id": TASK_ID, "task_status": s}}
                )
            )
            op = provider.poll(TASK_ID)
            assert op.status == OP_RUNNING
            assert op.video_bytes is None

    def test_succeeded_downloads_bytes(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "api/v1/tasks" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "output": {
                            "task_id": TASK_ID,
                            "task_status": "SUCCEEDED",
                            "video_url": VIDEO_URL,
                        },
                        "usage": {"output_video_duration": 8},
                    },
                )
            assert str(request.url) == VIDEO_URL
            return httpx.Response(200, content=MP4)

        provider = make_provider(handler)
        op = provider.poll(TASK_ID)
        assert op.status == OP_SUCCEEDED
        assert op.video_bytes == MP4
        assert op.duration_seconds == pytest.approx(8.0)

    def test_failed_task_maps_to_op_failed_with_message(self) -> None:
        provider = make_provider(
            lambda request: httpx.Response(
                200,
                json={
                    "output": {
                        "task_id": TASK_ID,
                        "task_status": "FAILED",
                        "code": "InternalError",
                        "message": "generation exploded",
                    }
                },
            )
        )
        op = provider.poll(TASK_ID)
        assert op.status == OP_FAILED
        assert "InternalError" in (op.error or "")
        assert "generation exploded" in (op.error or "")

    def test_succeeded_without_video_url_is_retryable(self) -> None:
        provider = make_provider(
            lambda request: httpx.Response(
                200, json={"output": {"task_id": TASK_ID, "task_status": "SUCCEEDED"}}
            )
        )
        with pytest.raises(UpstreamError) as excinfo:
            provider.poll(TASK_ID)
        assert excinfo.value.retryable is True


class TestWanService:
    def test_price_and_ledger_kind(self, db_session) -> None:
        provider = make_provider(lambda request: httpx.Response(500))
        service = WanService(
            provider, CostLedger(db_session, default_cap_usd=6.0), InMemoryOperationStore()
        )
        assert service.cost_kind == "wan"
        assert service.resolve_model() == CFG.wan_i2v_model
        assert service.price_per_second(CFG.wan_i2v_model) == pytest.approx(0.10)
        assert wan_price_per_second(CFG.wan_i2v_model) == pytest.approx(0.10)
        with pytest.raises(ValueError):
            wan_price_per_second("unknown-model")

    def test_full_scene_flow_records_wan_costs(
        self, db_session, creative, video_plan_dict
    ) -> None:
        from app.models import CostEvent
        from app.schemas.videoplan import VideoPlan

        plan = VideoPlan.model_validate(video_plan_dict)
        ledger = CostLedger(db_session, default_cap_usd=6.0)

        def handler(request: httpx.Request) -> httpx.Response:
            if "video-synthesis" in str(request.url):
                return httpx.Response(
                    200, json={"output": {"task_id": TASK_ID, "task_status": "PENDING"}}
                )
            if "api/v1/tasks" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "output": {
                            "task_id": TASK_ID,
                            "task_status": "SUCCEEDED",
                            "video_url": VIDEO_URL,
                        },
                        "usage": {"output_video_duration": 8},
                    },
                )
            return httpx.Response(200, content=MP4)

        provider = make_provider(handler)
        service = WanService(
            provider, ledger, InMemoryOperationStore(), model_config=CFG
        )

        submission = service.ensure_submitted(
            creative, "scene-1", plan.scenes[0].visual_prompt_en, keyframe_bytes=PNG
        )
        assert submission.model_id == CFG.wan_i2v_model
        projected = (
            db_session.query(CostEvent)
            .filter_by(creative_id=creative.id, kind="wan", projected=True)
            .one()
        )
        assert projected.units == pytest.approx(8.0)
        assert projected.amount_usd == pytest.approx(0.80)

        result = service.poll(creative, "scene-1")
        assert result.status == OP_SUCCEEDED
        assert result.video_bytes == MP4
        assert result.actual_cost_usd == pytest.approx(0.80)

        service.checkpoint_success(creative, "scene-1", result)
        actual = (
            db_session.query(CostEvent)
            .filter_by(creative_id=creative.id, kind="wan", projected=False)
            .one()
        )
        assert actual.amount_usd == pytest.approx(0.80)
        assert "wan scene" in actual.note

    def test_cap_blocks_wan_projection(self, db_session, creative) -> None:
        creative.cost_cap_usd = 0.5  # below one 8 s scene at $0.10/s
        ledger = CostLedger(db_session, default_cap_usd=0.5)
        service = WanService(
            make_provider(lambda request: httpx.Response(500)), ledger, InMemoryOperationStore()
        )
        with pytest.raises(CostCapExceeded):
            service.ensure_submitted(
                creative, "scene-1", "prompt", keyframe_bytes=PNG, duration_seconds=8.0
            )


class TestQuotaExhaustion:
    def test_free_tier_only_403_raises_provider_quota_exhausted(self) -> None:
        from app.errors import ProviderQuotaExhausted

        provider = make_provider(
            lambda request: httpx.Response(
                403,
                json={
                    "code": "AllocationQuota.FreeTierOnly",
                    "message": "The free quota has been exhausted.",
                    "request_id": "r-1",
                },
            )
        )
        with pytest.raises(ProviderQuotaExhausted) as excinfo:
            provider.submit(
                prompt="p",
                model_id=CFG.wan_i2v_model,
                duration_seconds=8.0,
                keyframe_bytes=PNG,
            )
        assert "FreeTierOnly" in excinfo.value.message
        assert excinfo.value.details["provider_code"] == "AllocationQuota.FreeTierOnly"
