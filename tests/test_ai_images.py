"""Regression tests for Gemini image fallback behavior."""

from types import SimpleNamespace

import httpx
import pytest

from app.ai.gemini import GeminiImageProvider


def test_gemini_image_failure_reaches_offline_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A total provider/network outage must still produce a keyframe."""

    class FailingModels:
        def generate_content(self, **_kwargs: object) -> object:
            raise TimeoutError("provider unavailable")

    provider = GeminiImageProvider(api_key="test-key")
    monkeypatch.setattr(
        provider,
        "_get_client",
        lambda: SimpleNamespace(models=FailingModels()),
    )

    def no_network(*_args: object, **_kwargs: object) -> object:
        raise httpx.ConnectError("network unavailable")

    monkeypatch.setattr(httpx, "get", no_network)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=b"png" * 200, stderr=b""),
    )

    result = provider.generate_image("cinematic AI chip in a modern laboratory")

    assert result.model_id == "procedural_keyframe"
    assert result.mime_type == "image/png"
    assert len(result.image_bytes) > 500
