"""Current Gemini model contract tests using an injected offline client."""

from types import SimpleNamespace

import pytest

from app.ai.gemini import (
    GeminiImageProvider,
    GeminiResearchProvider,
    GeminiScriptProvider,
    GenAIVideoProvider,
)
from app.errors import UpstreamError, ValidationFailed


class RecordingModels:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response

    def generate_videos(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(name="operations/smoke")


class RecordingClient:
    def __init__(self, models: RecordingModels) -> None:
        self.models = models


def test_image_generation_requests_vertical_1k_image_only() -> None:
    inline = SimpleNamespace(data=b"image-bytes", mime_type="image/png")
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(inline_data=inline)]))
        ]
    )
    models = RecordingModels(response)
    provider = GeminiImageProvider(client=RecordingClient(models), model_id="image-model")

    result = provider.generate_image("vertical technology scene")

    assert result.image_bytes == b"image-bytes"
    assert models.calls[0]["config"] == {
        "response_modalities": ["IMAGE"],
        "image_config": {"aspect_ratio": "9:16", "image_size": "1K"},
    }


def test_video_submission_passes_requested_duration() -> None:
    models = RecordingModels()
    provider = GenAIVideoProvider(client=RecordingClient(models))

    operation_name = provider.submit(
        prompt="abstract particles",
        model_id="veo-model",
        duration_seconds=4.0,
    )

    assert operation_name == "operations/smoke"
    config = models.calls[0]["config"]
    assert config.aspect_ratio == "9:16"
    assert config.resolution == "720p"
    assert config.duration_seconds == 4


def test_video_submission_rejects_unsupported_duration_before_network() -> None:
    models = RecordingModels()
    provider = GenAIVideoProvider(client=RecordingClient(models))

    with pytest.raises(ValidationFailed, match="4, 6, or 8"):
        provider.submit(
            prompt="abstract particles",
            model_id="veo-model",
            duration_seconds=7.5,
        )

    assert models.calls == []


class QuotaError(Exception):
    status_code = 429


class ServerError(Exception):
    status_code = 500


class PerModelModels:
    def __init__(self, outcomes: dict[str, object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes[str(kwargs["model"])]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, candidates=[])


def test_research_falls_back_in_order_only_after_quota_and_reuses_selected_model() -> None:
    models = PerModelModels(
        {
            "primary": QuotaError("primary quota"),
            "secondary": _text_response('{"topics": []}'),
            "tertiary": _text_response('{"topics": []}'),
        }
    )
    provider = GeminiResearchProvider(
        client=RecordingClient(models),
        model_id="primary",
        fallback_model_ids=("secondary", "tertiary"),
    )

    assert provider.research("AI", "technology", 72) == []
    assert provider.research("AI", "technology", 168) == []

    assert [call["model"] for call in models.calls] == [
        "primary",
        "secondary",
        "secondary",
    ]
    assert all(
        call["config"] == {"tools": [{"google_search": {}}]}
        for call in models.calls
    )


def test_research_does_not_downgrade_model_for_non_quota_failure() -> None:
    models = PerModelModels(
        {
            "primary": ServerError("provider unavailable"),
            "secondary": _text_response('{"topics": []}'),
        }
    )
    provider = GeminiResearchProvider(
        client=RecordingClient(models),
        model_id="primary",
        fallback_model_ids=("secondary",),
    )

    with pytest.raises(UpstreamError) as caught:
        provider.research("AI", "technology", 72)

    assert caught.value.details["status_code"] == 500
    assert [call["model"] for call in models.calls] == ["primary"]


def test_research_stops_when_fallback_has_non_quota_failure() -> None:
    models = PerModelModels(
        {
            "primary": QuotaError("primary quota"),
            "secondary": ServerError("provider unavailable"),
            "tertiary": _text_response('{"topics": []}'),
        }
    )
    provider = GeminiResearchProvider(
        client=RecordingClient(models),
        model_id="primary",
        fallback_model_ids=("secondary", "tertiary"),
    )

    with pytest.raises(UpstreamError) as caught:
        provider.research("AI", "technology", 72)

    assert caught.value.details["status_code"] == 500
    assert [call["model"] for call in models.calls] == ["primary", "secondary"]


def test_research_reaches_third_model_when_first_two_exhaust_quota() -> None:
    models = PerModelModels(
        {
            "gemini-3.7-flash": QuotaError("primary quota"),
            "gemini-3.6-flash": QuotaError("secondary quota"),
            "gemini-3.5-flash": _text_response('{"topics": []}'),
        }
    )
    provider = GeminiResearchProvider(client=RecordingClient(models))

    assert provider.research("AI", "technology", 72) == []
    assert [call["model"] for call in models.calls] == [
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
    ]


def test_script_generation_uses_same_quota_fallback_chain() -> None:
    models = PerModelModels(
        {
            "primary": QuotaError("primary quota"),
            "secondary": _text_response('{"schema_version": "1.0"}'),
        }
    )
    provider = GeminiScriptProvider(
        client=RecordingClient(models),
        model_id="primary",
        fallback_model_ids=("secondary",),
    )

    assert provider.generate_plan("topic", [], "brief") == {"schema_version": "1.0"}
    assert [call["model"] for call in models.calls] == ["primary", "secondary"]
    assert models.calls[1]["config"] == {"response_mime_type": "application/json"}


def test_sdk_error_is_sanitized_and_classified() -> None:
    models = RecordingModels(error=QuotaError("sentinel-provider-payload"))
    provider = GeminiResearchProvider(client=RecordingClient(models))

    with pytest.raises(UpstreamError) as caught:
        provider.research("AI", "technology", 72)

    assert caught.value.retryable
    assert caught.value.details == {
        "provider": "gemini",
        "operation": "research",
        "status_code": 429,
    }
    assert "sentinel-provider-payload" not in caught.value.message


def test_transport_timeout_without_status_is_retryable_and_sanitized() -> None:
    models = RecordingModels(error=TimeoutError("sentinel-network-payload"))
    provider = GeminiResearchProvider(client=RecordingClient(models))

    with pytest.raises(UpstreamError) as caught:
        provider.research("AI", "technology", 72)

    assert caught.value.retryable
    assert "status_code" not in caught.value.details
    assert "sentinel-network-payload" not in caught.value.message
